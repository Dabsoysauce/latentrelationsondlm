# Latent relations in diffusion language models

This repository tests whether diffusion language models encode UD dependency
relations in attention, representations, logits, and causal behavior. The
rigorous pipeline separates historical reproduction, confirmatory EWT testing,
locked external-treebank transfer, and exploratory extensions.

Current claim status: historical evidence supports a DiffuGPT-S
object-to-verb attention motif, including a reported pre-unmask effect. The
archived preliminary larger-model results do **not** yet establish that effect,
locked cross-treebank transfer, or causal necessity. Those claims require the
GPU gates and full runs documented below.

## Install

Core CPU validation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,probe]"
.\.venv\Scripts\python.exe -m pytest -q
```

Install one model-specific requirements file in a separate environment. Dream
and DiffuLLaMA intentionally use incompatible Transformers versions.

## Quickstart

```powershell
dlmrel data audit --dataset configs/datasets/ewt.yaml
dlmrel model smoke-test --model configs/models/diffugpt.yaml --dry-run
dlmrel run --model configs/models/fake.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/head_search.yaml --dry-run
dlmrel status --results results
```

Remove `--dry-run` only in the appropriate environment. Every non-dry run uses
`results/<track>/<model>/<dataset>/<experiment>/<run_id>/` and refuses to
overwrite a completed run. Use `--run-id <id> --resume` for deterministic
resumption.

## Capability matrix

| Model | Logits/hidden | Attention | DLA/ablation | Denoising time | Status |
|---|---:|---:|---:|---:|---|
| Fake CPU | yes | yes | yes | synthetic | CPU test harness |
| DiffuGPT-S | yes | yes | gated | yes | GPU gate not run |
| DiffuLLaMA | yes | yes | gated | yes | GPU gate not run |
| Dream-7B | yes | yes | gated | analysis schedule | GPU gate not run |
| LLaDA-8B | yes | disabled | disabled | yes | attention parity not run |
| GPT-2 | yes | yes | gated | not applicable | final-state baseline |

“Gated” means the adapter does not advertise the capability until a validated
intervention/decomposition is available. GPT-2 is never used for masked-state
or denoising claims.

## Protocol and guides

- [Frozen experiment protocol](docs/experiment_protocol.md)
- [Treebanks and immutable revisions](docs/treebanks.md)
- [Model adapters and environments](docs/model_adapters.md)
- [Result schemas](docs/result_format.md)
- [Legacy reproduction](docs/reproduction.md)
- [Colab guide](docs/colab_guide.md)
- [Modal guide](docs/modal_guide.md)
- [Implementation status](docs/implementation_status.md)

Historical outputs live under
`results/reference_legacy/pre_rigorous_6e5aaec/` with explicit provenance.
They are regression references, not confirmatory results.
