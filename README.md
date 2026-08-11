# Linguistic Relations in Diffusion Language Models

This repository studies **where and when diffusion language models represent linguistic relations during denoising**.

We compare DiffuGPT-S, DiffuLLaMA-7B, Dream-7B, and LLaDA-8B using the same data and experimental procedures.

## Experiments

- Relation-head search
- Relation accuracy over denoising time
- Attention entropy
- Part-of-speech probing
- Logit-lens analysis

## Structure

```text
configs/     Model and experiment settings
data/        Shared Universal Dependencies data and fixed splits
src/dlmrel/  Reusable model, experiment, and evaluation code
results/     Outputs organized by model and experiment
tests/       Automated checks
docs/        Shared protocol and team documentation
```

`dlmrel` is the shared Python package. Model-specific behavior belongs in `src/dlmrel/models/`, while each experiment is implemented once in `src/dlmrel/experiments/`.

## Setup

```bash
git clone <repository-url>
cd dlm-generalization
python -m venv .venv
```

Activate the environment:

```bash
# macOS or Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies for the model you want to run:

```bash
pip install -e ".[dream]"
```

## Run an experiment

Prepare the shared UD data once:

```bash
dlmrel prepare-data
```

Run Dream-7B relation-head search:

```bash
dlmrel run --model dream_7b --experiment head_search
```

Results are saved to:

```text
results/<model>/<experiment>/
â”œâ”€â”€ config.yaml
â”œâ”€â”€ metrics.csv
â”œâ”€â”€ summary.json
â””â”€â”€ figures/
```

See `docs/experiment_protocol.md` for the shared scientific procedure.
