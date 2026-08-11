from pathlib import Path

import modal

CONFIG = "/root/configs/dream-7b.yaml"

if modal.is_local():
    HERE = Path(__file__).resolve().parent
    REPO_ROOT = HERE.parents[3]
    if not (REPO_ROOT / "pyproject.toml").exists():
        raise SystemExit(f"no pyproject.toml at {REPO_ROOT}, check REPO_ROOT depth")
else:
    HERE = Path("/root/configs")
    REPO_ROOT = Path("/root")

app = modal.App("dream7b-relation-head-search")

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
        "scikit-learn",
    )
    .add_local_dir(REPO_ROOT / "src" / "dlmrel", remote_path="/root/pkgs/dlmrel")
    .add_local_dir(HERE, remote_path="/root/configs")
)

hf_cache = modal.Volume.from_name("dream7b-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("dlmrel-work", create_if_missing=True)


def _run(stages):
    import os
    import sys
    import time

    os.environ["HF_HOME"] = "/cache/huggingface"
    sys.path.insert(0, "/root/pkgs")
    os.chdir("/work")

    from dlmrel.cli import main as dlmrel_main

    for stage in stages:
        print(f"\n{'=' * 60}\n=== dlmrel {stage} ===\n{'=' * 60}", flush=True)
        t0 = time.time()
        rc = dlmrel_main([stage, "--config", CONFIG])
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
def run_cpu(stages):
    _run(stages)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/cache": hf_cache, "/work": work},
    timeout=3 * 60 * 60,
    memory=32768,
)
def run_gpu(stages):
    _run(stages)


@app.local_entrypoint()
def main(stages: str = "data,nulls,search,analyze"):
    wanted = [s.strip() for s in stages.split(",") if s.strip()]
    gpu_stages = ("search", "curve", "entropy", "logitlens", "posprobe")

    pending_cpu = []
    for stage in wanted:
        if stage in gpu_stages:
            if pending_cpu:
                run_cpu.remote(tuple(pending_cpu))
                pending_cpu = []
            run_gpu.remote((stage,))
        else:
            pending_cpu.append(stage)
    if pending_cpu:
        run_cpu.remote(tuple(pending_cpu))
