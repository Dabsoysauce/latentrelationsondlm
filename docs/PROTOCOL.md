# Frozen old-paper protocol

## Global invariants

The corrected protocol targets Dream-7B and DiffuLLaMA-7B with their pinned
model, tokenizer, and remote-code revisions. It uses EWT plus locked transfer
to German GSD and Japanese GSD. Every stochastic experiment uses exactly seeds
`42`, `43`, and `44`. Fully masked and forced fully visible observations
are stored once rather than presented as three independent observations.

Generic depth comparisons map `early=.20`, `middle=.50`, and `late=.90`
to `round(fraction * (number_of_layers - 1))`. Every output records the
label, fraction, actual block, and total block count.

Corrected runners use selection and held-out test only. They never open, hash,
checkpoint, or tune on a development split. The official development files
remain only as dataset and legacy provenance. Corrected runners also never
execute permutation inference or Holm correction.

## Relation-Head Receiver Prediction

Every head is scored once on fully visible EWT selection sentences at timestep
63. For each of the six relations, the direct selection winner becomes the
immutable lock. Ties use accuracy descending, valid denominator descending,
layer ascending, then head ascending. All six locks are published before the
test manifest is opened.

The executable old-code receiver rule is authoritative: use the final attender
subtoken row, remove BOS and the complete attender span, and choose one source
subtoken by argmax. A prediction is correct when that one source subtoken lies
inside the gold receiver span. Receiver word spans are never summed before
selection. This differs from paper prose that describes averaging attender
rows; the code/paper disagreement is recorded in lock metadata.

Held-out EWT test runs score only the matching frozen relation head and report
raw counts, accuracy, old receiver controls, and lock provenance.

## Relation-Head Receiver Prediction over Diffusion Time

The fully visible locks are never reselected. A single RNG reset produces one
nested 64-state teacher-forced trajectory per sentence/seed. Still-masked
tokens reveal with probability `1 / remaining_steps`. Reports cover every
timestep and the old two-way visibility split:

- `both_masked`: no subtoken in either endpoint is visible;
- `at_least_one_revealed`: at least one endpoint subtoken is visible.

One revealed piece immediately leaves `both_masked`. The configured minimum
masked-instance cutoff is reported alongside raw denominators.

Execution microbatches consecutive states from one already-materialized nested
trajectory. The state IDs, seed-reset schedule, scoring order, and outputs are
unchanged. For seeds 43 and 44, deterministic endpoints are removed before the
GPU call rather than computed and discarded afterward.

## Attention Entropy

Attention Entropy uses gold teacher-forced trajectories, all 64 steps, all
heads in the relative early/middle/late layers, and the old normalized entropy.
Only the BOS query row is excluded; the BOS source column remains in each
distribution. Outputs include per-head trajectories, per-layer summaries,
early and late entropy, delta, slope, direction, percentages, and seed
mean/standard deviation.

The supplied source did not preserve unambiguous early/late window boundaries.
The active config freezes inclusive windows `0..15` and `48..63` and labels
that choice as a provenance limitation rather than presenting it as recovered
fact.

When requested, the English relation-time runner computes these exact entropy
rows from the attentions already in memory and stores them under a separate
validated checkpoint identity. Attention Entropy may reuse only a completed
cache with the same model revision, tokenizer revision, test manifest, relative
depth mapping, seeds, and timestep coverage. Otherwise it fails closed.

## POS/Token-Class Linear Probes

Gold teacher-forced selection trajectories fit fixed probes and held-out test
trajectories evaluate them. The protocol covers four old mask ratios and all
three relative depths, plus head-level attention-output features. It reports
accuracy, macro-F1, class counts, majority, shuffled-label, and random-feature
controls.

The label inventory is `NOUN, VERB, ADJ, ADV, PREP, DET, PRON, CONJ`.
Stanford's log-linear POS tagger supplies PTB tags, which are mapped into this
inventory and inherited by every subtoken of a word. Special/out-of-inventory
tokens are excluded. The original Stanford model and JAR were not supplied.
The runner downloads Stanford tagger 4.2.0 and its
`english-left3words-distsim` model into a cache, verifies the archive, JAR, and
model with frozen SHA-256 values, and sets the two required paths automatically.
Explicit `STANFORD_POS_TAGGER_JAR` and `STANFORD_POS_TAGGER_MODEL` paths still
override the cache. It never silently uses UD UPOS. The exact historical
Stanford release remains unrecovered, so 4.2.0 is recorded as a frozen
reproducible dependency, not claimed as the original release.

## Native eventual-token and timing experiments

Both native experiments use the preserved 24-prompt manifest: 12 reasoning and
12 creative prompts. They store 64 pre-forward sequences, aligned final-layer
argmax sequences, and eventual generated token IDs under temperature `.95`,
top-p `.9`, length `96`, and random `1/(t+1)` reveal. Adapter alignment is
explicit: Dream is unshifted (`0`) and DiffuLLaMA is shifted (`-1`).

Final-Token Prediction by Layer evaluates still-masked positions against their
eventual generated token at early/middle/late layers. Top-1 is primary;
top-5, rank, and MRR are additional. The optional trained probe uses fixed
settings and a prompt-held-out split with no tuning role.

Prediction Before Unmasking reports refinement curves, masked fraction,
argmax-at-unmask match, `found_time`, `unmask_time`, their difference,
lead steps, before/exact-at-unmask percentages, task strata, and
punctuation/function/content/number/whitespace classes.

## DLA and causal ablation

Direct Logit Attribution captures the actual concatenated value-weighted head
tensor entering a Llama-style attention output projection. It selects only the
requested head slice of `o_proj.weight`, obtains its additive residual
contribution, then applies the model final norm and unembedding. It records
target logit, rank, vocabulary percentile, target POS, relation selectivity,
tensor shapes, and module paths. Unsupported architectures fail rather than
approximate.

Matched Relation-Head Ablation uses a forward pre-hook to zero exactly one
requested input slice to `o_proj`, leaving every other head slice unchanged.
It compares frozen selected heads with low-relation controls under matched
sentence, target, timestep, visibility, and seed. POS-decodable versus lower
POS-decoding controls run when an exact completed POS ranking directory is
provided through `DLMREL_POS_HEAD_RANKINGS`; otherwise that source dependency
is reported as blocked.

## Heatmaps and multilingual extension

Heatmap scoring first saves evidence, then plotting reads only the saved files.
The preserved five qualitative sentences produce fully visible all-head grids
and surface diagnostics. The first manifest-order example for each relation
(not performance-selected) produces selected-head 64-step trajectories for
all three seeds, masked/visible labels, direction, entropy titles, annotated
spans, and PDF figures.

Multilingual Relation-Head Transfer applies the matching model's English locks
to German and Japanese without reselection. It provides fully visible and
all-64-step evidence with the same visibility rule and finite-value checks.
This is an extension, not an original-paper experiment.

## Resume and artifacts

Long sentence loops commit validated Parquet chunks every 300 input sentences.
The shared relation/entropy trajectory uses 20-sentence chunks to bound the
much larger entropy table in memory.
Checkpoint identity includes the scientific config, relevant manifests, seed,
time coordinate, stage, heads, and sentence range. Corrected scientific hashes
exclude development metadata. Native histories and other non-sentence evidence
have dedicated resumable stages. Incompatible legacy locks and checkpoints are
rejected.

Legacy dev/permutation code remains importable only to validate or recover
historical results. Canonical configs dispatch exclusively to modules named
`paper_*.py`.
