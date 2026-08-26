"""Paste this file into one Colab cell after the branch is pushed."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from google.colab import drive


REPOSITORY = "https://github.com/Dabsoysauce/latentrelationsondlm.git"
BRANCH = "optimize/adaptive-pos-cpu-under-3h"
COMMIT = "3da9dede1b43308c0583083c9ec70b6139134d01"
RUN_ID = "paper-restoration-v1-diffullama-pos-token-class-linear-probes"
RUN_DIR = (
    Path("/content/drive/MyDrive/dlmrel-paper-results (1)/diffullama")
    / "exploratory_extensions"
    / "diffullama_7b"
    / "ewt"
    / "pos_token_class_linear_probes"
    / RUN_ID
)
CHECKOUT = Path("/content/latentrelationsondlm-pos12")
LOCAL_CACHE = Path("/content/dlmrel-pos-cache")
TOTAL_BUDGET_SECONDS = 8 * 60 * 60
VALIDATION_RESERVE_SECONDS = 15 * 60


def run(*args: str, cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True, env=os.environ.copy())


os.environ.update(
    {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
)
drive.mount("/content/drive")
if not RUN_DIR.is_dir():
    raise FileNotFoundError(f"Completed-extraction run is missing: {RUN_DIR}")

logical_cpus = os.cpu_count() or 1
worker_counts = [count for count in (6, 8, 10, 12) if count <= logical_cpus]
if not worker_counts:
    worker_counts = [max(1, logical_cpus)]
free_gib = shutil.disk_usage("/content").free / 2**30
print(
    json.dumps(
        {
            "logical_cpus": logical_cpus,
            "worker_counts": worker_counts,
            "local_free_gib": round(free_gib, 2),
            "run_dir": str(RUN_DIR),
            "protocol": "12 candidates/depth; progress .25/.50/.75",
            "total_budget_hours": TOTAL_BUDGET_SECONDS / 3600,
        },
        indent=2,
    )
)

if CHECKOUT.exists():
    shutil.rmtree(CHECKOUT)
run("git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(CHECKOUT))
run("git", "fetch", "origin", BRANCH, cwd=CHECKOUT)
run("git", "checkout", "--detach", COMMIT, cwd=CHECKOUT)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
if head != COMMIT:
    raise RuntimeError(f"Expected {COMMIT}, got {head}")

run("python", "-m", "pip", "install", "-q", "-e", ".[dev]", cwd=CHECKOUT)
run(
    "python",
    "-m",
    "pytest",
    "-q",
    "tests/test_paper_pos_adaptive_12.py",
    cwd=CHECKOUT,
)
run(
    "python",
    "-m",
    "dlmrel.cli",
    "pos-fit-adaptive-12",
    "--run-dir",
    str(RUN_DIR),
    "--local-cache",
    str(LOCAL_CACHE),
    "--budget-seconds",
    str(TOTAL_BUDGET_SECONDS),
    "--validation-reserve-seconds",
    str(VALIDATION_RESERVE_SECONDS),
    "--worker-counts",
    *map(str, worker_counts),
    cwd=CHECKOUT,
)
run(
    "python",
    "-m",
    "dlmrel.cli",
    "validate-pos-adaptive-12",
    "--run-dir",
    str(RUN_DIR),
    cwd=CHECKOUT,
)

output = RUN_DIR / "pos_adaptive_12"
manifest = json.loads((output / "adaptive_manifest.json").read_text(encoding="utf-8"))
if manifest.get("status") != "confirmed":
    raise RuntimeError("Confirmation failed; causal rankings remain unavailable")
print(f"CONFIRMED choices: {output / 'pos_head_choices.csv'}")
print(f"Three-progress trajectories: {output / 'selected_head_progress_trajectories.csv'}")
print(f"Coverage manifest: {output / 'coverage_manifest.csv'}")
print(f"Measured benchmark: {output / 'benchmark.json'}")
