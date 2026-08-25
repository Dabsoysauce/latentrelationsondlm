"""Paste this entire file into one Google Colab cell.

The cell never invokes extraction or constructs a model. It checks out the reviewed
implementation, mounts the existing Drive results, runs the CPU-only adaptive fit,
validates the reduced bundle, and exposes rankings only after the confirmation gate.
"""

import base64
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from google.colab import drive


BASE_COMMIT = "fff83d1f27d69927babd1d86085e220021a581f6"
IMPLEMENTATION_COMMIT = "15e78bda770c17feba79f0d36beea310a8af9abe"
IMPLEMENTATION_TREE = "6cfa96e4e13cc249134ddec694db15202240eb38"
IMPLEMENTATION_PATCH_SHA256 = "d7fa1cf714cbac8c705a8d41a6a4b42306ad93512d247311764b78f08466a039"
OPTIMIZATION_COMMIT = "136e289"
OPTIMIZED_TREE = "ee0269a58b4523dccfd17bd66e17802e413a0e8e"
OPTIMIZATION_PATCH_SHA256 = "f9100e180fd86b99086cb2b8be678c0635a7280940d080ad73451b744095bdb0"
REPOSITORY = "https://github.com/Dabsoysauce/latentrelationsondlm.git"
RUN_ID = "paper-restoration-v1-diffullama-pos-token-class-linear-probes"
RESULTS_ROOT = Path("/content/drive/MyDrive/dlmrel-paper-results (1)/diffullama")
RUN_DIR = (
    RESULTS_ROOT
    / "exploratory_extensions"
    / "diffullama_7b"
    / "ewt"
    / "pos_token_class_linear_probes"
    / RUN_ID
)
CHECKOUT = Path("/content/latentrelationsondlm-pos-adaptive")
LOCAL_CACHE = Path("/content/dlmrel-pos-cache")
FIT_BUDGET_SECONDS = 2 * 60 * 60 + 45 * 60
VALIDATION_RESERVE_SECONDS = 15 * 60

# Generated from `git diff --binary BASE_COMMIT..IMPLEMENTATION_COMMIT`, then gzip
# compressed. Embedding the reviewed patch makes the cell runnable before the local
# implementation branch is pushed; both its SHA-256 and resulting Git tree are checked.
IMPLEMENTATION_PATCH_GZIP_BASE64 = (
    "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"
    "TcdtwC0J6lCQ3QOoB4/wHfBz738DszlFwXMxAQA="
)
OPTIMIZATION_PATCH_GZIP_BASE64 = (
    "H4sIAAAAAAAACqVTTW/bRhC961cMdCIrak2rboIYJWDAbU8FWtRAL4ZBTLhDeZv9IGaHjZRfXyyXimVXSgWUOogazXs77+0b"
    "bfoe1uutEcCryN2Vto7JXtFuIDaOvMSrAQfidghRDXv4eEnXwnhNO6jrH/Tm5oNS+n1dd/odXNf1u5ubxXq9vuy0xWq1uvD"
    "EuztY33y/qTbvYTV9f4C7O9DUQ9sbKXpGR7cwaPUTCv6SflXwXQWRSN+C8VIB03a0yOYLign+FnobUMrbBaTH9GDJF5Gk2J"
    "cl/Aib+Y/0MJpI8CfakX5mDlwsf//tASJZ6hIV9IQyMkXoghc0qfCZGOQZPcjnAJ3FGCkuy0wZO7TE0MCDoNfI+mEqFKVKS"
    "nblYvW1S7c7aGaAEkYf+8Au9UxME7HpzcT2a9iaKKb7g7ZMMZrgixcJ981r+RU43LVGiJtNXdcVMHodXBsFhZrk2mKdcHmm"
    "EwNUsC9zC5OM7Ochq6OZKthVsM9qjnh0m8qzym+Dj/oXkD7TfdPfaEcUKnojQroCYTT+7eULRflWIA4333NwED9ZQvbKGk/I"
    "rQuaLBg3BJYTvp4AOhI2XTxgsOtGxm7fxi4wVdBf57cFZM9O6Z1UJK35ZQ8NZIFHeTgHmY06TzFNnCyZAuUHFQW7T0WqPC"
    "7n/C6flMOh8IPCiMy4L8sjXCLL7RY/kl0+KQmtH92wL+YgDEza5I1ojsZUc/3fKcrzlK8D/3XIc+25+6LDDmyzDod/BTaSl"
    "NyH0QtxMRtVKheitF1wLvjiunysnx7rp4xiv82W5RVRmnocrbTst0VK02Grn8e+T0dORmVa1YXJngOPmpuKl+Y36P/e4yOt"
    "9/9ziyXHpayOhi8vva5XG/0SxHNEb65iHnTCQTN54wM7tEU0X6iZ+VR8xoHeQCjKKcTEfgowp/uMs/8Abp1EVyEHAAA="
)


def run(*args: str, cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True, env=os.environ.copy())


# Set native thread limits before importing NumPy/scikit-learn in subprocesses.
os.environ.update(
    {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
)

drive.mount("/content/drive")
if not RUN_DIR.is_dir():
    raise FileNotFoundError(f"Expected completed-extraction run is missing: {RUN_DIR}")

logical_cpus = os.cpu_count() or 1
worker_counts = [count for count in (6, 8, 10, 12) if count <= logical_cpus]
if not worker_counts:
    worker_counts = [max(1, logical_cpus)]
disk = shutil.disk_usage("/content")
print(
    json.dumps(
        {
            "run_dir": str(RUN_DIR),
            "logical_cpus": logical_cpus,
            "worker_counts_to_benchmark": worker_counts,
            "local_free_gib": round(disk.free / 2**30, 2),
            "fit_budget_seconds": FIT_BUDGET_SECONDS,
            "validation_reserve_seconds": VALIDATION_RESERVE_SECONDS,
        },
        indent=2,
    )
)

if CHECKOUT.exists():
    shutil.rmtree(CHECKOUT)
run("git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(CHECKOUT))
run("git", "checkout", "--detach", BASE_COMMIT, cwd=CHECKOUT)
head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True
).strip()
if head != BASE_COMMIT:
    raise RuntimeError(f"Wrong required base commit: {head}")
patch_bytes = gzip.decompress(base64.b64decode(IMPLEMENTATION_PATCH_GZIP_BASE64))
if hashlib.sha256(patch_bytes).hexdigest() != IMPLEMENTATION_PATCH_SHA256:
    raise RuntimeError("Embedded reviewed implementation patch failed SHA-256 validation")
patch_path = Path("/content/dlmrel-pos-adaptive-reviewed.patch")
patch_path.write_bytes(patch_bytes)
run("git", "apply", "--index", str(patch_path), cwd=CHECKOUT)
tree = subprocess.check_output(
    ["git", "write-tree"], cwd=CHECKOUT, text=True
).strip()
if tree != IMPLEMENTATION_TREE:
    raise RuntimeError(f"Reviewed implementation tree mismatch: {tree}")
optimization_bytes = gzip.decompress(base64.b64decode(OPTIMIZATION_PATCH_GZIP_BASE64))
if hashlib.sha256(optimization_bytes).hexdigest() != OPTIMIZATION_PATCH_SHA256:
    raise RuntimeError("Embedded exact-optimization patch failed SHA-256 validation")
optimization_path = Path("/content/dlmrel-pos-exact-optimization.patch")
optimization_path.write_bytes(optimization_bytes)
run("git", "apply", "--index", str(optimization_path), cwd=CHECKOUT)
optimized_tree = subprocess.check_output(
    ["git", "write-tree"], cwd=CHECKOUT, text=True
).strip()
if optimized_tree != OPTIMIZED_TREE:
    raise RuntimeError(f"Exact-optimized implementation tree mismatch: {optimized_tree}")
print(
    f"Verified implementation {IMPLEMENTATION_COMMIT} plus optimization "
    f"{OPTIMIZATION_COMMIT}, tree {optimized_tree}"
)

run("python", "-m", "pip", "install", "-q", "-e", ".[dev]", cwd=CHECKOUT)

# Short CPU-only safety pilot: equality, resume, local staging, and causal fail-closed tests.
run(
    "python",
    "-m",
    "pytest",
    "-q",
    "tests/test_paper_pos_stages.py",
    "tests/test_paper_pos_adaptive.py",
    "tests/test_paper_optimizations.py",
    cwd=CHECKOUT,
)

# Benchmark time is charged to the 2 h 45 m fitting window. The runner stages only
# active read-only features locally, writes atomic fit checkpoints to Drive, resumes
# every valid fit, prints progress/ETA, and stops before a conservatively unsafe batch.
adaptive_command = [
    "python",
    "-m",
    "dlmrel.cli",
    "pos-fit-adaptive",
    "--run-dir",
    str(RUN_DIR),
    "--local-cache",
    str(LOCAL_CACHE),
    "--budget-seconds",
    str(FIT_BUDGET_SECONDS + VALIDATION_RESERVE_SECONDS),
    "--validation-reserve-seconds",
    str(VALIDATION_RESERVE_SECONDS),
    "--worker-counts",
    *map(str, worker_counts),
]
run(*adaptive_command, cwd=CHECKOUT)

# This command fails unless the confirmation gate passed and all artifact hashes,
# coverage, rankings, and required primary residual fits reproduce exactly.
run(
    "python",
    "-m",
    "dlmrel.cli",
    "validate-pos-adaptive",
    "--run-dir",
    str(RUN_DIR),
    cwd=CHECKOUT,
)

adaptive_dir = RUN_DIR / "pos_adaptive"
status = json.loads((adaptive_dir / "adaptive_manifest.json").read_text(encoding="utf-8"))
if status.get("status") != "confirmed":
    raise RuntimeError("Adaptive POS evidence is not confirmed; rankings remain unavailable")
rankings = adaptive_dir / "pos_head_rankings_adaptive.csv"
if not rankings.is_file():
    raise RuntimeError("Confirmed status exists but the validated ranking artifact is missing")
print(f"CONFIRMED reduced-grid rankings: {rankings}")
print(f"Coverage manifest: {adaptive_dir / 'coverage_manifest.csv'}")
print(f"Measured throughput: {adaptive_dir / 'benchmark.json'}")
print(f"Quantitative decision table: {adaptive_dir / 'decision_table.csv'}")
