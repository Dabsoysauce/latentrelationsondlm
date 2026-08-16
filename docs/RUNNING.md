# Running the GPU experiments

## Recommended order

Use Google Colab with an A100 or another high-memory NVIDIA GPU. Clone the
repository, select a GPU runtime, and store the Hugging Face token in Colab
Secrets as `HF_TOKEN`. Never paste the token into the notebook or repository.

Run Dream and DiffuLLaMA in separate Colab runtimes:

```bash
pip install -e .
pip install -r requirements/dream.txt       # Dream runtime only
# or
pip install -r requirements/diffullama.txt  # DiffuLLaMA runtime only
```

The notebook at `notebooks/colab_runner.ipynb` exposes the same commands. For
each model, use this order:

All active experiment configurations use exactly three seeds: `42`, `43`, and
`44`, including the POS probe. Do not resume a run that was initialized with
the older one-seed or five-seed protocol; give the three-seed rerun a new run
ID.

```bash
dlmrel prepare --dataset configs/datasets/ewt.yaml
dlmrel smoke-test --model configs/models/dream_7b.yaml
dlmrel run --model configs/models/dream_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/head_search.yaml --run-id dream-ewt-v1
```

If interrupted, rerun the last command with `--resume` and the same run ID.
Sentence-level checkpoints are written atomically every 300 processed
sentences. Resume reuses every validated completed chunk and begins at the
first unfinished range. Runs produced before chunking may also reuse their
complete atomic whole-seed checkpoints; incomplete temporary files are
ignored. A scientific change to the model, data manifests, experiment,
three-seed list, progress points, scoring, or source selection lock is rejected
rather than resumed.
After head search completes, copy the path to its `selection_lock.json`. Use
that lock for the EWT time curve and for German/Japanese transfer:

```bash
dlmrel run --model configs/models/dream_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/time_curve.yaml --selection-lock <lock-path> --run-id dream-ewt-time-v1
dlmrel prepare --dataset configs/datasets/de_gsd.yaml
dlmrel run --model configs/models/dream_7b.yaml --dataset configs/datasets/de_gsd.yaml --experiment configs/experiments/external_transfer.yaml --selection-lock <lock-path> --run-id dream-de-v1
dlmrel prepare --dataset configs/datasets/ja_gsd.yaml
dlmrel run --model configs/models/dream_7b.yaml --dataset configs/datasets/ja_gsd.yaml --experiment configs/experiments/external_transfer.yaml --selection-lock <lock-path> --run-id dream-ja-v1
```

Repeat with `configs/models/diffullama_7b.yaml`. Run attention entropy, logit
lens, and POS probe only after the model smoke test passes. Save complete run
directories to Drive before ending a Colab session, and run `dlmrel validate`
before treating any output as a research result.
