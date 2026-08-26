"""Append a fail-closed matched head-ablation follow-on to fixed-12 notebooks."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _source(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code(text: str) -> dict:
    ast.parse(text.strip())
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def _ablation_cells(model: str) -> list[dict]:
    if model == "diffullama":
        model_name = "DiffuLLaMA-7B"
        model_id = "diffullama_7b"
        model_config = "configs/models/diffullama_7b.yaml"
        requirements = "requirements/diffullama.txt"
        run_prefix = "paper-restoration-v1-diffullama"
        batch_arguments = """
        '--timestep-batch-size', '8',
        '--native-batch-size', '8',
        '--intervention-batch-size', '8',
        '--sentence-batch-size', '8',
        '--adaptive-batch-max-size', '32',"""
    elif model == "dream":
        model_name = "Dream-7B"
        model_id = "dream_7b"
        model_config = "configs/models/dream_7b.yaml"
        requirements = "requirements/dream.txt"
        run_prefix = "paper-restoration-v1-dream"
        batch_arguments = """
        '--timestep-batch-size', '8',"""
    else:
        raise ValueError(f"unknown model: {model}")

    markdown = f"""
## {model_name} matched POS-head causal ablation (GPU follow-on)

Run this section only after the preceding POS validation reports `confirmed`.
Change Colab to an **A100 High-RAM** runtime, rerun only the setup/checkout cells
at the top (leave `RUN_EXTRACTION = False` in the Dream notebook), and then run
the two cells below. The first cell fails closed unless the confirmed fixed-12
rankings and the model-specific relation-selection lock are both valid. The
second cell resumes the canonical matched ablation and skips it if it is already
complete. It never reruns POS extraction.
"""
    preflight = f"""
from google.colab import userdata

# Revalidate the exact ranking bundle before allowing model inference.
run('python', '-m', 'dlmrel.cli', 'validate-pos-adaptive-12', '--run-dir', RUN_DIR, cwd=CHECKOUT)
OUTPUT = RUN_DIR / 'pos_adaptive_12'
manifest = json.loads((OUTPUT / 'adaptive_manifest.json').read_text())
if manifest.get('status') != 'confirmed' or manifest.get('matched_causal_ablation_allowed') is not True:
    raise RuntimeError('Fixed-12 POS rankings have not passed held-out confirmation.')
os.environ['DLMREL_POS_HEAD_RANKINGS'] = str(OUTPUT)

# The earlier POS cells deliberately hide the GPU. Re-enable it only here.
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        'Switch Colab to an A100 High-RAM runtime, rerun setup/checkout, '
        'then rerun this cell.'
    )

hf_token = userdata.get('HF_TOKEN')
if not hf_token:
    raise RuntimeError('Add HF_TOKEN to Colab Secrets before running the 7B-model ablation.')
os.environ['HF_TOKEN'] = hf_token
del hf_token

MODEL_ID = {model_id!r}
MODEL_CONFIG = {model_config!r}
RUN_PREFIX = {run_prefix!r}
SELECTION_RUN = (
    RESULTS_ROOT / 'confirmatory_ewt' / MODEL_ID / 'ewt'
    / 'relation_head_receiver_prediction' / f'{{RUN_PREFIX}}-relation-selection'
)
LOCK_DIR = SELECTION_RUN / 'selection-locks'
if not (LOCK_DIR / 'selection_bundle.json').is_file():
    raise FileNotFoundError(
        f'Required model-specific relation-selection lock is missing: {{LOCK_DIR}}. '
        'Complete the relation-selection experiment before head ablation.'
    )

run('python', '-m', 'pip', 'install', '-q', '-r', {requirements!r}, cwd=CHECKOUT)
run(
    'python', '-m', 'dlmrel.cli', 'validate-selection-locks',
    '--model', MODEL_CONFIG,
    '--dataset', 'configs/datasets/ewt.yaml',
    '--experiment', 'configs/experiments/relation_head_receiver_prediction.yaml',
    '--selection-lock', LOCK_DIR,
    cwd=CHECKOUT,
)
print(json.dumps({{
    'gpu': torch.cuda.get_device_name(0),
    'pos_rankings': str(OUTPUT),
    'selection_lock': str(LOCK_DIR),
}}, indent=2))
"""
    run_ablation = f"""
ABLATION_RUN_ID = f'{{RUN_PREFIX}}-matched-relation-head-ablation'
ABLATION_RUN = (
    RESULTS_ROOT / 'exploratory_extensions' / MODEL_ID / 'ewt'
    / 'matched_relation_head_ablation' / ABLATION_RUN_ID
)
summary_path = ABLATION_RUN / 'summary.json'
already_complete = False
if summary_path.is_file():
    already_complete = json.loads(summary_path.read_text()).get('completion_status') == 'complete'

if already_complete:
    print('Already complete; skipping model inference:', ABLATION_RUN)
else:
    run(
        'python', '-m', 'dlmrel.cli', 'run',
        '--model', MODEL_CONFIG,
        '--dataset', 'configs/datasets/ewt.yaml',
        '--experiment', 'configs/experiments/matched_relation_head_ablation.yaml',
        '--results', RESULTS_ROOT,
        '--run-id', ABLATION_RUN_ID,
        '--resume',
        '--selection-lock', LOCK_DIR,{batch_arguments}
        cwd=CHECKOUT,
    )

run('python', '-m', 'dlmrel.cli', 'validate', '--run-dir', ABLATION_RUN, cwd=CHECKOUT)
summary = json.loads((ABLATION_RUN / 'summary.json').read_text())
if summary.get('completion_status') != 'complete':
    raise RuntimeError('Matched head ablation did not reach a validated complete state.')
print(json.dumps({{
    'model': {model_name!r},
    'status': summary['completion_status'],
    'run_dir': str(ABLATION_RUN),
    'resumable': True,
    'pos_protocol': manifest['schema_version'],
}}, indent=2))
"""
    return [_markdown(markdown), _code(preflight), _code(run_ablation)]


def append_cells(source: Path, output: Path, model: str) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    rendered = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    if "matched POS-head causal ablation (GPU follow-on)" in rendered:
        raise ValueError(f"ablation section already exists in {source}")
    notebook["cells"].extend(_ablation_cells(model))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("diffullama", "dream"), required=True)
    args = parser.parse_args()
    append_cells(args.source, args.output, args.model)


if __name__ == "__main__":
    main()
