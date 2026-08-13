"""Thin Modal launcher for the shared dlmrel CLI."""

from __future__ import annotations

import subprocess

import modal


app = modal.App("dlmrel-rigorous-pipeline")
volume = modal.Volume.from_name("dlmrel-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .add_local_dir(".", remote_path="/repo", copy=True)
    .run_commands("cd /repo && pip install -e .")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/repo/results": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def launch(arguments: list[str]) -> None:
    subprocess.run(
        ["python", "-m", "dlmrel.cli", *arguments], cwd="/repo", check=True
    )
    volume.commit()


@app.local_entrypoint()
def main(
    model: str,
    dataset: str,
    experiment: str,
    run_id: str = "",
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    arguments = [
        "run",
        "--model",
        model,
        "--dataset",
        dataset,
        "--experiment",
        experiment,
    ]
    if run_id:
        arguments.extend(["--run-id", run_id])
    if resume:
        arguments.append("--resume")
    if dry_run:
        arguments.append("--dry-run")
    launch.remote(arguments)
