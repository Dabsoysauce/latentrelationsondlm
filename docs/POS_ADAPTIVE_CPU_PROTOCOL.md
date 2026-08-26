# DiffuLLaMA POS CPU fitting: measured state and adaptive decision gate

Scope: run `paper-restoration-v1-diffullama-pos-token-class-linear-probes`,
starting from commit `fff83d1f27d69927babd1d86085e220021a581f6`.
Extraction is complete and is never invoked by this protocol.

## Exact consumers and claims

The canonical POS runner produces `instances.parquet`, `per_seed_metrics.csv`,
`metrics.csv`, and `pos_head_rankings.csv`. The repository does not generate a
POS-specific figure. The notebook review package copies the two metric CSVs.

The canonical protocol claim is exhaustive: four progress/mask conditions,
three relative depths, three seeds, residual features, all 32 head-output
features, accuracy, macro-F1, majority, shuffled-label, and random-feature
controls. The head-only grid is `3 * 4 * 3 * 32 = 1,152` logical fits. The code
also fits 36 residual probes; every logical probe trains the main classifier and
two additional logistic-regression controls.

Matched causal ablation is the only explicit downstream code consumer. The
canonical consumer averages accuracy over all available progress conditions for
each relative depth, then selects the highest- and lowest-accuracy head. A
reduced primary-progress ranking therefore changes that estimand. Adaptive
causal rows are explicitly named `*_primary_p050_*`; they cannot be presented
as the canonical four-progress-average comparison.

The repository's separate single-condition `masked_pos_probe` fixes normalized
progress at `.5` and takes the midpoint hidden state. That is the only explicit
primary POS condition in the source. The adaptive minimum therefore evaluates
the middle residual at progress `.5` for all three seeds before head screening.

## Drive inventory measured 2026-08-25

- Extracted feature checkpoints: 180 Parquet plus 180 metadata files.
- Compressed extracted-feature size: 24,097,468,152 bytes (24.10 GB decimal;
  22.44 GiB).
- Coverage: select and test, seeds 42/43/44, progress 0/.25/.5/.75, all three
  depths and all heads.
- Completed atomic head fits: 32/1,152 (2.78%).
- Completed cell: seed 42, progress 0, early depth, heads 0 through 31.
- Completed fit files: 32 Parquet/metadata pairs. No file needs migration.

The last recorded command requested 12 Python workers. Across the 30 within-run
checkpoint gaps (excluding the restart gap), the observed cadence was:

| Statistic | Seconds per completed fit |
|---|---:|
| Minimum | 177.7 |
| P10 | 193.6 |
| Median | 225.3 |
| Mean | 226.2 |
| P90 | 256.0 |
| Maximum | 273.4 |

Mean observed throughput was 15.9 fits/hour. Fit-checkpoint serialization was
not the bottleneck: Parquet writes averaged 21.9 ms and metadata writes 5.3 ms
from Drive timestamps, or 27.2 ms per completed pair.

The one-seed auxiliary screen is not stable enough to choose causal heads. Its
accuracy range is only 3.79 percentage points: head 5 was highest at 0.3077 and
head 6 lowest at 0.2698. The preregistered uncertainty rule retains 11 heads at
the high boundary and 9 at the low boundary (20 unique candidates). This is a
planning proxy only; seed-42 progress-.5 screens determine the real candidate
sets before seeds 43/44 are read.

## Baseline-only decision table

These times use the measured pre-optimization mean and P10/P90 cadence. They do
not claim optimized Colab performance.

| Option | Head fits (% of 1,152) | Reused | New | Baseline wall time (fast–slow) | Seeds | Heads | Progress/depth | Claims | Causal-selection risk |
|---|---:|---:|---:|---:|---|---|---|---|---|
| Full original | 1,152 (100%) | 32 | 1,120 | 70.4 h (60.2–79.6) | 3 everywhere | 32 everywhere | 4 x 3 | All canonical claims | Lowest |
| Primary-only full-head | 96 (8.33%) | 0 | 96 | 6.03 h (5.16–6.83) | 3 | 32 | `.5`, middle | Primary midpoint only | Early/late unsupported |
| Screen + top/bottom/ambiguous confirmation | 72 proxy (6.25%) | 0 | 72 | 4.52 h (3.87–5.12) | seed 42 all; 43/44 candidates | 32 screen; 20 proxy confirmed | `.5`, middle | Primary head ranking only | Low at middle only |
| Stratified 50% | 576 (50%) | 32 | 544 | 34.2 h (29.3–38.7) | Balanced, incomplete cells | Predetermined stratification | 4 x 3 | Coarse trends | Moderate; omitted head can win |
| Stratified 25% | 288 (25%) | 32 | 256 | 16.1 h (13.8–18.2) | Balanced, sparse | Predetermined stratification | 4 x 3 | Descriptive trends only | High |
| Adaptive proxy | 248 (21.53%) | 32 | 216 | 13.6 h (11.6–15.4) | seed 42 all; 43/44 candidates | 32 screen at each depth; 20 proxy confirmed per depth | `.5` x 3, plus existing p0/early | Primary residual plus primary-progress high/low causal inputs | Fail-closed; full held-out expansion if unstable |

The adaptive proxy adds three residual fits outside the head-only counts. Its
held-out confirmation allocation is 120/216 new head fits (55.6%), above the
15% minimum. Depending on the actual progress-.5 screens, the head allocation
ranges from 176/1,152 including existing fits (15.28%) to 320/1,152 (27.78%).

## Runtime decision gate

No optimized time is scientifically reportable until the current Colab runtime
runs the real-data benchmark. The automatic runner measures the active
condition's staging/read time, a full representative fit, worker batches at
6/8/10/12 workers, CPU use, peak RSS, local and Drive checkpoint writes, and
real-data sequential/parallel identity. It includes benchmark time in the
2-hour-45-minute fitting window and reserves the final 15 minutes for
validation/artifacts.

Zero-overhead throughput thresholds are useful fail-fast checks:

- Full remaining 1,120 head fits: more than 407 fits/hour, plus 36 residuals
  and feature loading. Anything lower rules out the canonical grid.
- Adaptive proxy 216 new head fits: more than 78.5 fits/hour, plus three
  residuals, benchmarking, and feature loading. In practice the selected rate
  must be higher.

The runner stops before a batch whose conservative estimate crosses the fitting
deadline. If held-out rankings fail the stability rule, it expands that depth to
all 32 heads for seeds 43/44. If the expansion cannot fit, no final ranking file
is created and matched causal ablation remains blocked.

## Exact optimization changes

- Loads one seed/progress select/test table pair once, uses it for all requested
  probes, then releases it; it no longer retains all selection tables.
- Optionally stages only active read-only feature chunks on local Colab disk.
- Keeps all finished fit checkpoints and final artifacts on Drive with the
  existing atomic v1 identity.
- Limits native BLAS/OpenMP pools to one thread while parallel Python fit tasks
  run, preventing `workers * logical CPUs` oversubscription.
- Benchmarks 6/8/10/12 workers and selects the fastest memory-safe count.
- Checks every existing atomic checkpoint before opening its feature table.
- Writes a 1,152-row coverage manifest with a reason for every included or
  omitted head fit.
- Publishes reduced artifacts only under `pos_adaptive/`; the canonical summary
  and canonical `pos_head_rankings.csv` are not written.
- Requires all 32 heads in seed-42 screens at progress `.5` for middle, early,
  and late depths; candidate selection is frozen before seeds 43/44 are read.
- Accepts adaptive causal inputs only after three-seed extreme-rank stability,
  median pairwise Spearman >= .5, nonzero mean margins, and bundle hash checks.

## Claims after a successful adaptive gate

Retained:

- Primary midpoint residual POS decoding across all three seeds.
- At progress `.5`, unbiased all-head screens and held-out-seed-confirmed high
  versus lower POS-decoding heads at early/middle/late depths.
- Accuracy, macro-F1, and controls for every included fit.
- A matched causal comparison explicitly labeled as primary-progress `.5`.

Narrowed:

- The causal POS ranking changes from a four-progress average to progress `.5`.
- The exhaustive multi-progress grid becomes primary-progress inference with
  seed-42 p0/early evidence retained only as auxiliary coverage.

Removed unless the canonical full command is completed later:

- Exhaustive head-by-progress trajectories.
- Three-seed uncertainty for omitted heads and progress conditions.
- A claim that all four mask ratios were evaluated for every head/depth/seed.

