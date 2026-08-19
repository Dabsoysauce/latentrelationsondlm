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
`44`, including the POS probe. Runs created with this protocol can resume with
the same run ID; older one-seed or five-seed runs require a new run ID because
their scientific configuration is genuinely different.

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

### Derive the six locks from the existing Dream run

If `dream-english-head-3seed-v1` already finished its all-head select/dev
scoring, do not rerun the model. In Colab, with the same Python `RESULTS`
variable used for the run, execute:

```bash
!dlmrel derive-relation-locks \
  --source-run "{RESULTS}/confirmatory_ewt/dream_7b/ewt/confirmatory_head_search/dream-english-head-3seed-v1" \
  --output "{RESULTS}/derived/dream-english-head-3seed-v1-relation-selection"
```

The output must be a new directory outside the completed source run. The
command is CPU-only: it loads no model, performs no inference or rescoring,
and reads only the saved select/dev evidence plus configuration, manifests,
and provenance. It never reads `instances.parquet`, test metrics, or the
source summary, and it never modifies the source run.

After head search completes, use the canonical `relation-selection/` directory
(or its `relation_selection_bundle.json`) for the EWT time curve and for
German/Japanese transfer. That source contains all six locks and lets each
downstream relation resolve its own head. Passing the legacy
`selection_lock.json` is supported only for an explicitly object-only run; it
cannot produce the other five relation results.

```bash
dlmrel run --model configs/models/dream_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/time_curve.yaml --selection-lock <relation-selection-directory> --run-id dream-ewt-time-v1
dlmrel prepare --dataset configs/datasets/de_gsd.yaml
dlmrel run --model configs/models/dream_7b.yaml --dataset configs/datasets/de_gsd.yaml --experiment configs/experiments/external_transfer.yaml --selection-lock <relation-selection-directory> --run-id dream-de-v1
dlmrel prepare --dataset configs/datasets/ja_gsd.yaml
dlmrel run --model configs/models/dream_7b.yaml --dataset configs/datasets/ja_gsd.yaml --experiment configs/experiments/external_transfer.yaml --selection-lock <relation-selection-directory> --run-id dream-ja-v1
```

The existing Dream select/dev all-head files remain reusable for CPU-only
six-lock derivation. The saved object-selected test rows do not contain the
other five locked heads, and the corrected selection-aware permutation needs
all-head test predictions because a null permutation can select any eligible
head. Those targeted test computations still require a new GPU run; deriving
locks alone does not create or imply them.

For the interrupted `dream-english-head-3seed-v1` run, do not restart the full
head search. Follow the copy-first two-phase procedure in
`docs/HEAD_SEARCH_RECOVERY.md`: run `recover-head-search-test-grid` once on GPU,
then run the model-free `finalize-head-search` command on CPU. Both commands
resume compatible partial work, and the original Drive run remains untouched.

Repeat with `configs/models/diffullama_7b.yaml`. Run attention entropy, logit
lens, and POS probe only after the model smoke test passes. Save complete run
directories to Drive before ending a Colab session, and run `dlmrel validate`
before treating any output as a research result.

For optional serverless execution, the thin Modal wrapper invokes these same
commands without changing their science. Start with the credential-free local
dry run and supervised workflow in `docs/MODAL.md`; no paid job or result
promotion is automatic.
