# Linguistic Relations in Diffusion Language Models

This repository studies where and when diffusion language models represent linguistic relations during denoising.

We compare DiffuGPT-S, DiffuLLaMA-7B, Dream-7B, and LLaDA-8B using the same data, experiments, and evaluation methods.

## Experiments

- Relation-head search
- Relation accuracy over diffusion time
- Attention entropy
- Part-of-speech probing
- Logit-lens analysis

## Repository structure

```text
configs/     Model and experiment settings
data/        Shared Universal Dependencies data
src/dlmrel/  Reusable model, experiment, and evaluation code
results/     Outputs organized by model
tests/       Automated checks
docs/        Shared protocol and team documentation
```

`src/dlmrel/models/` contains model-specific code. Each shared experiment is implemented once in `src/dlmrel/experiments/`.

## Setup

```bash
python -m venv .venv
pip install -e ".[dream]"
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

## Run an experiment

Prepare the shared UD data once:

```bash
dlmrel prepare-data
```

Run relation-head search on Dream-7B:

```bash
dlmrel run --model dream_7b --experiment head_search
```

Results are saved under `results/<model>/<experiment>/`.

See `docs/experiment_protocol.md` for the shared scientific procedure.
