# Latent relations in diffusion language models

This repository runs scaled versions of the original DiffuGPT paper experiments
on Dream-7B and DiffuLLaMA-7B. The corrected active protocol uses English EWT
selection/test only, all six predefined dependency relations, seeds
`[42, 43, 44]`, and model-relative early/middle/late depths.

The active experiment names are:

1. Relation-Head Receiver Prediction
2. Relation-Head Receiver Prediction over Diffusion Time
3. Attention Entropy
4. POS/Token-Class Linear Probes
5. Final-Token Prediction by Layer
6. Prediction Before Unmasking: Timing Analysis
7. Direct Logit Attribution
8. Matched Relation-Head Ablation
9. Attention Heatmaps and Trajectories
10. Multilingual Relation-Head Transfer (new extension)

Corrected runs do not read development data, run permutations, or apply Holm
correction. Legacy modules and completed result directories remain available
only for provenance and are not reachable from the corrected configurations.

## Install and verify

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Dream and DiffuLLaMA use separate pinned environments. The ready-to-run
launchers are:

- `notebooks/Dream_Paper_Experiments.ipynb`
- `notebooks/DiffuLLaMA_Paper_Experiments.ipynb`

## Core commands

```bash
dlmrel prepare --dataset configs/datasets/ewt.yaml

dlmrel run \
  --model configs/models/dream_7b.yaml \
  --dataset configs/datasets/ewt.yaml \
  --experiment configs/experiments/relation_head_receiver_prediction.yaml \
  --results /path/to/results/dream \
  --run-id paper-v1-dream-relation-selection \
  --resume

dlmrel validate-selection-locks \
  --model configs/models/dream_7b.yaml \
  --selection-lock /path/to/relation-run/selection-locks

dlmrel run \
  --model configs/models/dream_7b.yaml \
  --dataset configs/datasets/ewt.yaml \
  --experiment configs/experiments/relation_head_receiver_prediction_over_diffusion_time.yaml \
  --selection-lock /path/to/relation-run/selection-locks \
  --results /path/to/results/dream \
  --run-id paper-v1-dream-relation-time \
  --resume
```

Every costly sentence loop retains atomic 300-sentence checkpoints. Native
trajectory, entropy, POS, DLA, ablation, and heatmap evidence have
experiment-specific resumable identities. Completed runs cannot be silently
overwritten.

See [the frozen protocol](docs/PROTOCOL.md), [GPU running
instructions](docs/RUNNING.md), and [implementation status](docs/STATUS.md).
