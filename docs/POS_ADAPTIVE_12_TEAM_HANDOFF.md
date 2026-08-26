# POS fixed-12 protocol: team handoff

## Scientific scope

This protocol preserves early, middle, and late depth claims. It uses the
already-extracted progress checkpoints `.25`, `.50`, and `.75`; exact `.20` and
`.80` would require new GPU extraction and are intentionally not used.

Supported after the confirmation gate passes:

- all-head seed-42 POS screens at progress `.50` for all three depths;
- held-out-seed confirmation within a preregistered 12-head set at each depth;
- three-seed `.25/.50/.75` trajectories for the confirmed high- and
  lower-decoding head at each depth;
- canonical shuffled-label and random-feature controls for those final causal
  heads at progress `.50` across all three seeds;
- matched causal ablation at early, middle, and late depths, explicitly labeled
  `primary_p050_fixed12`.

Not supported: an exhaustive all-head trajectory, a four-progress average, or a
claim that the selected heads are globally extreme under every held-out seed.
The claim is that the symmetric fixed-12 candidates were selected by the
unbiased seed-42 screen and passed the saved three-seed stability gate.

## Frozen allocation

Per depth, the seed-42 `.50` ranking contributes:

- ranks 1-4: high core;
- ranks 5-6: high boundary;
- ranks 27-28: low boundary;
- ranks 29-32: low core.

This rule always gives exactly 12 candidates and is saved before seeds 43/44 are
read. No favorable-result selection is permitted.

Work units:

| Phase | Work |
|---|---:|
| Worker benchmark | 49 main-classifier fits |
| All-head `.50` screen | 96 main-classifier fits |
| Held-out confirmation | 72 main-classifier fits |
| Selected-head `.25/.75` extension | 36 main-classifier fits |
| Final selected-head controls at `.50` | 18 canonical logical fits |
| Primary midpoint residual | 3 canonical logical fits |

The 204 scientific main-only fits use a separate atomic namespace under
`pos_adaptive_12/main_fit_checkpoints`. The 18 selected-head control fits and
three residual fits use ordinary canonical checkpoints and are reusable by a
future full-grid run.

## Runtime

At the saved baseline, one full logical probe took 226.2 seconds and trains three
classifiers. Treating a main-only fit as one classifier gives approximately 6.6
hours of classifier-equivalent work including the worker benchmark. Local
feature staging and matrix reuse should reduce repeated overhead, but the live
benchmark is authoritative.

Operating estimate: **6.5-7.5 hours**. The cell enforces an **8-hour total cap**
with 15 minutes reserved for validation. A conservative batch guard stops before
starting work estimated to cross the deadline. If ranking stability fails, no
causal ranking is published.

## Branch and review

- Branch: `optimize/adaptive-pos-cpu-under-3h`
- Protocol implementation commit:
  `3da9dede1b43308c0583083c9ec70b6139134d01`
- Extraction is never invoked and the 7B model is never loaded.
- Existing canonical extraction and fit checkpoints are read/reused, never
  rewritten merely for benchmarking.

Before running in Colab, push the branch:

```bash
git push -u origin optimize/adaptive-pos-cpu-under-3h
```

The branch has been pushed to `origin` as of 2026-08-25.

Then paste `scripts/colab_pos_adaptive_12_cell.py` into one Colab cell. It mounts
Drive, verifies the exact commit, disables CUDA, installs the CPU dependencies,
runs the focused pilot, benchmarks 6/8/10/12 workers, resumes atomically, validates
the bundle, and prints the artifact paths.

The equivalent notebook is `notebooks/DiffuLLaMA_POS_Adaptive_12.ipynb`.

## Dream-7B replica

`notebooks/Dream_POS_Adaptive_12.ipynb` runs the same frozen allocation and
validation under a separate Dream identity:

- model: `configs/models/dream_7b.yaml`;
- result root: `/content/drive/MyDrive/dlmrel-paper-results/dream`;
- run ID: `paper-restoration-v1-dream-pos-token-class-linear-probes`;
- output: the Dream run's own `pos_adaptive_12/` directory.

The CPU protocol itself is model-generic and does not load Dream. Unlike the
completed DiffuLLaMA extraction, Dream extraction has not been inventoried here.
The Dream notebook therefore checks for exactly 180 validated feature chunks and
fails closed if they are absent. It contains a separate `RUN_EXTRACTION` gate,
disabled by default, for completing Dream extraction on a GPU. After extraction,
switch to the 12-logical-CPU runtime before fitting. Never point the Dream
notebook at the DiffuLLaMA result root or reuse rankings across models.

Direct command after checkout:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m dlmrel.cli pos-fit-adaptive-12 \
  --run-dir "/content/drive/MyDrive/dlmrel-paper-results (1)/diffullama/exploratory_extensions/diffullama_7b/ewt/pos_token_class_linear_probes/paper-restoration-v1-diffullama-pos-token-class-linear-probes" \
  --local-cache /content/dlmrel-pos-cache \
  --budget-seconds 28800 \
  --validation-reserve-seconds 900 \
  --worker-counts 6 8 10 12
```

Validation/resume command:

```bash
python -m dlmrel.cli validate-pos-adaptive-12 \
  --run-dir "/content/drive/MyDrive/dlmrel-paper-results (1)/diffullama/exploratory_extensions/diffullama_7b/ewt/pos_token_class_linear_probes/paper-restoration-v1-diffullama-pos-token-class-linear-probes"
```

Re-running `pos-fit-adaptive-12` is the resume operation. Valid atomic main and
canonical checkpoints are skipped.

## Artifacts

All reduced artifacts are under `pos_adaptive_12/`:

- `adaptive_manifest.json`: authoritative status and hashes;
- `preregistration.json`: frozen design;
- `candidate_selection.json`: exact 12 heads and reasons per depth;
- `benchmark.json`: live CPU/RAM/worker measurements;
- `pos_head_choices.csv`: confirmed high/low causal inputs;
- `pos_head_rankings_adaptive_12.csv`: midpoint rankings;
- `selected_head_progress_trajectories.csv`: `.25/.50/.75` results;
- `selected_head_control_metrics.csv`: final-head controls;
- `primary_residual_metrics.csv`: midpoint residual results;
- `coverage_manifest.csv`: all 1,152 canonical head cells and evidence tier.

Matched causal ablation must point `DLMREL_POS_HEAD_RANKINGS` to this directory.
It rejects missing, provisional, incomplete, or hash-mismatched evidence.

## Stop conditions

Do not publish rankings if any of these occur:

- fewer than 32 screen heads at any depth;
- candidate set not exactly 12 per depth;
- incomplete seed 43/44 confirmation;
- selected high or low falls outside the corresponding extreme three in a seed;
- median pairwise seed Spearman below `.5`;
- missing `.25/.50/.75` three-seed trajectory;
- selected control result differs from the main-only result;
- artifact hash or coverage validation fails;
- the eight-hour guard stops the run.

If stopped only for time, rerun the same command to resume. If stopped for rank
instability, report that the fixed-12 protocol was insufficient; do not expand or
change candidates after inspecting held-out results without registering a new
protocol.
