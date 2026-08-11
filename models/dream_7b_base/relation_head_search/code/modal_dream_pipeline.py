"""Full dlmrel pipeline for Dream-7B on a Modal A100.

Runs `dlmrel data -> nulls -> search -> analyze` with configs/dream-7b.yaml in
one GPU container. The local (patched) dlmrel source is mounted into the image,
so this exercises exactly the code sitting in this working tree — including the
new `family='dream'` loader and the slow-tokenizer alignment fallback.

Environment note: this container installs transformers==4.51.3 (Dream's
requirement). The DiffuLLaMA families need ==4.44.2, so those configs must NOT
be run through this script.

Persistent state:
  * Volume `dream7b-hf-cache`  — HF weights (already warm from the smoke test)
  * Volume `dlmrel-work`       — data/ud cache (incl. the 3-model common-pool
                                 JSON) and results/dream-7b-ewt

Usage:
    modal run scripts/modal_dream_pipeline.py
    # then fetch results:
    modal volume get dlmrel-work results/dream-7b-ewt <dest>
    modal volume get dlmrel-work data/ud <dest>   # common-pool cache JSON
"""

from pathlib import Path

import modal

# This file lives at models/<model>/<experiment>/code/, so the repository root
# is four levels up. Resolving from __file__ rather than the process working
# directory means `modal run` works from anywhere, not just the repo root.
REPO_ROOT = Path(__file__).resolve().parents[4]
if not (REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(
        f"expected the repository root at {REPO_ROOT}, but there is no "
        "pyproject.toml there — this script has been moved to a different "
        "depth and REPO_ROOT needs updating."
    )

app = modal.App("dream7b-dlmrel-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.51.3",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "pandas",
        "scipy",
        "pyyaml",
        "conllu",
    )
    .add_local_dir(
        REPO_ROOT / "src" / "dlmrel",
        remote_path="/root/pkgs/dlmrel",
    )
    .add_local_dir(
        REPO_ROOT / "configs",
        remote_path="/root/configs",
    )
)

hf_cache = modal.Volume.from_name("dream7b-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("dlmrel-work", create_if_missing=True)


def _run_stages(stages: tuple[str, ...]) -> None:
    import os
    import sys
    import time

    os.environ["HF_HOME"] = "/cache/huggingface"
    sys.path.insert(0, "/root/pkgs")
    os.chdir("/work")  # data/ud cache and results/ both land on the volume

    from dlmrel.cli import main as dlmrel_main

    for stage in stages:
        print(f"\n{'=' * 60}\n=== dlmrel {stage} ===\n{'=' * 60}", flush=True)
        t0 = time.time()
        rc = dlmrel_main([stage, "--config", "/root/configs/dream-7b.yaml"])
        if rc != 0:
            raise RuntimeError(f"dlmrel {stage} exited with {rc}")
        print(f"=== {stage} done in {time.time() - t0:.0f}s ===", flush=True)
        work.commit()
        hf_cache.commit()


@app.function(
    image=image,
    volumes={"/cache": hf_cache, "/work": work},
    timeout=2 * 60 * 60,
    cpu=4,
)
def run_cpu_stages(stages: tuple[str, ...]) -> None:
    """data / nulls / analyze — tokenizer and pandas work, no GPU billing."""
    _run_stages(stages)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/cache": hf_cache, "/work": work},
    timeout=2 * 60 * 60,
)
def run_gpu_stages(stages: tuple[str, ...]) -> None:
    """search (and curve, if ever) — the actual forward passes."""
    _run_stages(stages)


@app.local_entrypoint()
def main():
    run_cpu_stages.remote(("data", "nulls"))
    run_gpu_stages.remote(("search",))
    run_cpu_stages.remote(("analyze",))
