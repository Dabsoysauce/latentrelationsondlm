"""Generate the DiffuLLaMA and Dream primary-only POS Colab notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _source(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def build_notebook(
    *,
    title: str,
    branch: str,
    commit: str,
    run_id: str,
    results_root: str,
    run_suffix: str,
    checkout: str,
    cache: str,
) -> dict:
    setup = f"""
import json, os, subprocess
from pathlib import Path
from google.colab import drive

REPOSITORY = 'https://github.com/Dabsoysauce/latentrelationsondlm.git'
BRANCH = {branch!r}
PROTOCOL_COMMIT = {commit!r}
RUN_ID = {run_id!r}
RESULTS_ROOT = Path({results_root!r})
RUN_DIR = RESULTS_ROOT / {run_suffix!r} / RUN_ID
CHECKOUT = Path({checkout!r})
LOCAL_CACHE = Path({cache!r})
TOTAL_BUDGET_SECONDS = 5 * 60 * 60
VALIDATION_RESERVE_SECONDS = 15 * 60

def run(*args, cwd=None):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True, env=os.environ.copy())

os.environ.update({{
    'CUDA_VISIBLE_DEVICES': '',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
}})
drive.mount('/content/drive')
"""
    checkout_cell = """
if not CHECKOUT.exists():
    run('git', 'clone', '--filter=blob:none', '--no-checkout', REPOSITORY, CHECKOUT)
run('git', 'fetch', 'origin', BRANCH, cwd=CHECKOUT)
run('git', 'checkout', '--detach', PROTOCOL_COMMIT, cwd=CHECKOUT)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip()
assert head == PROTOCOL_COMMIT, (head, PROTOCOL_COMMIT)
run('python', '-m', 'pip', 'install', '-q', '-e', '.[dev]', cwd=CHECKOUT)
"""
    gate = """
if not RUN_DIR.is_dir():
    raise FileNotFoundError(f'Completed extraction run is missing: {RUN_DIR}')
feature_parquets = list((RUN_DIR / 'checkpoints').glob('paper-pos-*-features*.parquet'))
if len(feature_parquets) != 180:
    raise RuntimeError(
        f'Expected 180 extracted feature chunks; found {len(feature_parquets)}. '
        'Do not start fitting or load a model.'
    )
logical_cpus = os.cpu_count() or 1
worker_counts = [n for n in (6, 8, 10, 12) if n <= logical_cpus] or [max(1, logical_cpus)]
print(json.dumps({
    'run_dir': str(RUN_DIR),
    'feature_parquets': len(feature_parquets),
    'logical_cpus': logical_cpus,
    'workers_to_benchmark': worker_counts,
}, indent=2))
run(
    'python', '-m', 'pytest', '-q',
    'tests/test_paper_pos_primary_adaptive.py',
    'tests/test_paper_pos_adaptive.py',
    'tests/test_paper_pos_adaptive_12.py',
    'tests/test_paper_optimizations.py',
    cwd=CHECKOUT,
)
"""
    fit = """
run(
    'python', '-m', 'dlmrel.cli', 'pos-fit-primary-adaptive',
    '--run-dir', RUN_DIR,
    '--local-cache', LOCAL_CACHE,
    '--budget-seconds', str(TOTAL_BUDGET_SECONDS),
    '--validation-reserve-seconds', str(VALIDATION_RESERVE_SECONDS),
    '--worker-counts', *map(str, worker_counts),
    cwd=CHECKOUT,
)
"""
    validate = """
run(
    'python', '-m', 'dlmrel.cli', 'validate-pos-primary-adaptive',
    '--run-dir', RUN_DIR,
    cwd=CHECKOUT,
)
OUTPUT = RUN_DIR / 'pos_adaptive_primary'
manifest = json.loads((OUTPUT / 'adaptive_manifest.json').read_text())
assert manifest['status'] == 'confirmed'
print(json.dumps({
    'status': manifest['status'],
    'primary_condition_only': manifest['primary_condition_only'],
    'elapsed_hours': manifest['elapsed_seconds'] / 3600,
    'full_workers': manifest['full_workers'],
    'main_workers': manifest['main_workers'],
    'choices': str(OUTPUT / 'pos_head_choices.csv'),
    'rankings': str(OUTPUT / 'pos_head_rankings_primary.csv'),
    'coverage': str(OUTPUT / 'coverage_manifest.csv'),
}, indent=2))
"""
    return {
        "cells": [
            _markdown(
                f"""
# {title}

CPU-only fitting from the completed extraction. This protocol supports only the
paper's primary POS condition: progress `.50`, middle depth. It screens all 32
heads with seed 42, freezes top/bottom/uncertainty candidates, confirms them on
seeds 43/44, and publishes one matched middle-depth causal pair only after the
held-out stability gate passes.

Expected runtime: **3.5–4.2 hours**; hard cap: **5 hours**, including a 15-minute
validation reserve. Early/late and multi-progress POS claims are explicitly removed.
The GPU is disabled and no model is loaded.
"""
            ),
            _code(setup),
            _code(checkout_cell),
            _code(gate),
            _code(fit),
            _code(validate),
        ],
        "metadata": {
            "accelerator": "",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--branch", default="optimize/primary-pos-under-5h")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebooks = root / "notebooks"
    variants = [
        (
            "DiffuLLaMA POS primary-only adaptive completion",
            "DiffuLLaMA_POS_Primary_Adaptive.ipynb",
            "paper-restoration-v1-diffullama-pos-token-class-linear-probes",
            "/content/drive/MyDrive/dlmrel-paper-results (1)/diffullama",
            "exploratory_extensions/diffullama_7b/ewt/pos_token_class_linear_probes",
            "/content/latentrelationsondlm-pos-primary",
            "/content/dlmrel-pos-primary-cache",
        ),
        (
            "Dream POS primary-only adaptive completion",
            "Dream_POS_Primary_Adaptive.ipynb",
            "paper-restoration-v1-dream-pos-token-class-linear-probes",
            "/content/drive/MyDrive/dlmrel-paper-results/dream",
            "exploratory_extensions/dream_7b/ewt/pos_token_class_linear_probes",
            "/content/latentrelationsondlm-dream-pos-primary",
            "/content/dlmrel-dream-pos-primary-cache",
        ),
    ]
    for title, filename, run_id, results_root, suffix, checkout, cache in variants:
        notebook = build_notebook(
            title=title,
            branch=args.branch,
            commit=args.commit,
            run_id=run_id,
            results_root=results_root,
            run_suffix=suffix,
            checkout=checkout,
            cache=cache,
        )
        (notebooks / filename).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
