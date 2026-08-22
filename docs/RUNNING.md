# Running the restored GPU experiments

Use the model-specific Colab notebook:

- `notebooks/Dream_Paper_Experiments.ipynb`
- `notebooks/DiffuLLaMA_Paper_Experiments.ipynb`

They authenticate with Colab Secrets, verify an exact Git commit, mount Drive,
install the correct pinned model environment, run the complete CPU suite,
prepare all three datasets, and keep each costly experiment in its own opt-in
cell. Dream and DiffuLLaMA results use separate roots.

## Recommended eight-day order

1. Day 1: review the pinned commit; run CPU tests and real-model smoke tests.
2. Days 1–2: run Relation-Head Receiver Prediction for both models and validate
   all six selection locks.
3. Days 2–4: run the 64-step relation curves with entropy-cache export, then
   run Attention Entropy by reusing that completed time-run directory.
4. Days 3–5: run POS probes; the pinned Stanford dependency provisions itself.
5. Days 3–6: run native final-token and timing trajectories.
6. Days 5–7: run DLA, then matched causal ablation after DLA shape checks pass.
7. Days 6–7: generate heatmap evidence and PDFs.
8. Days 7–8: run German/Japanese locked transfer, validate every run, package
   summaries/figures, and reserve time for reruns.

Do not start every experiment simultaneously. Relation-time, DLA, ablation,
heatmaps, and transfer depend on the completed model-matching
`selection-locks/` directory.

## Canonical command pattern

```bash
dlmrel run \
  --model configs/models/dream_7b.yaml \
  --dataset configs/datasets/ewt.yaml \
  --experiment configs/experiments/EXPERIMENT_ID.yaml \
  --results /content/drive/MyDrive/dlmrel-paper-results/dream \
  --run-id paper-restoration-v1-dream-EXPERIMENT_ID \
  --resume
```

Add `--selection-lock /absolute/path/to/selection-locks` for:

- `relation_head_receiver_prediction_over_diffusion_time`
- `direct_logit_attribution`
- `matched_relation_head_ablation`
- `attention_heatmaps_and_trajectories`
- `multilingual_relation_head_transfer`

German and Japanese transfer use their matching dataset YAML and the same
model's English locks. Never cross Dream and DiffuLLaMA locks; validation
rejects the model/revision mismatch.

For the English time curve, add `--export-attention-cache`. Then run Attention
Entropy with `--attention-cache /absolute/path/to/completed-time-run`. Both use
`--timestep-batch-size 8` by default. A smaller value such as `4` lowers peak
GPU memory without changing scientific identity; a larger value can be used on
an A100 after a smoke run. Cache reuse is validated against the exact model,
dataset, depths, seeds, and all expected timesteps.

POS downloads Stanford tagger 4.2.0 once into
`DLMREL_STANFORD_POS_CACHE` (or the normal user cache). The notebooks point this
at Drive so runtime restarts reuse it. The 75 MB binary is not committed.

## Restart and review

Rerun an interrupted command with the same result root, run ID, and
`--resume`. The notebook helper skips an already-complete result. Use:

```bash
dlmrel validate --run-dir /absolute/run/path
dlmrel summarize --run-dir /absolute/run/path --rows 5
```

`summarize` does not open large Parquet evidence. Copy or package only after
`validate` succeeds. No local, Modal, or serverless GPU run is required for
repository validation.
