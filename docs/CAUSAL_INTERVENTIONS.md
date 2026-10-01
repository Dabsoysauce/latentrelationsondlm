# Causal head intervention implementation note

Repository inspection (before implementation, base `2e3293e`):

* `pipeline.load_adapter` loads pinned Dream and DiffuLLaMA wrappers. Dream has
  prediction offset 0; DiffuLLaMA has offset -1. `models.native.aligned_logits`
  is the authoritative conversion to a token's prediction position.
* `diffusion.teacher_forced_trajectory` reveals **gold** tokens; it cannot measure
  an intervention's effect on subsequent generation. `models.native` implements
  the preserved random-reveal sampler, with probability 1 / remaining steps,
  top-p sampling, and per-trajectory RNG replay for batches.
* `models.decomposition.projection_module` resolves the exact input to
  `self_attn.o_proj`: [batch, position, query_heads * head_dim]. Each contiguous
  head slice is the attention-weighted value output, before output projection
  and residual addition. Query heads, not KV heads, index these slices.
* Existing `capture_or_ablate_projection`, `ablate_projection_batch`, and
  `paper_causal` implement DLA and single-forward knockout. They remain intact.
  Their receiver-token score at the dependent query is a diagnostic, not the
  governor's own aligned generation probability.
* `relations.build_example` / `extract_relations` define eligible UD pairs;
  `alignment` provides BOS-shifted character-overlap subtoken spans. Adjective
  and determiner relations are split by subject/object noun role (six labels).
  `data.load_manifest_examples` and `shared.instance_metadata` preserve splits,
  exclusions, POS, distance/direction, clause and punctuation, and BPE metadata.
* `paper_protocol.load_selection_bundle` validates model-specific frozen
  selection-only head locks; `selection_all_head_scores.csv` provides relation
  scores. Older local result exports use different selection protocols; the
  causal CLI reads them through `relation_selection.load_relation_locks` and
  records their original provenance. It does not regenerate them or silently
  convert them into selection-only evidence. Existing low-relation controls are not
  entropy/magnitude matched; such matching requires additional metadata.
* `evaluation.statistics.sentence_clustered_bootstrap` preserves dependence
  among repeated relations/seeds from one sentence and is reused.

Extension design: reusable scoped projection scaling plus explicit per-forward
batch gates; a reconstruction entry point to the existing native sampler;
paired, free-running reconstruction of the same gold-aligned sentence length.
All non-BOS tokens start masked, so both dependency endpoints are eligible at
step zero. This is an explicitly new reconstruction task, not a replacement for
observational attention accuracy. Gold tokens score outcomes but never enter
the generated suffix. Reveal draws are exogenous and shared across conditions.
An optional fixed gold context can be supplied through the sampler API.

Pre means every subtoken of both endpoints is unrevealed; post means all are
revealed. Partial endpoint revelation is a distinct state. Early/late split the
exogenous complete pre-window, not a model-outcome-selected window. Absolute
half-open step bounds additionally restrict every window. Suppression applies
to all sequence positions of the selected head for an active example.

Post-revelation cannot change frozen endpoint tokens in this monotonic sampler.
It is a temporal negative control with that structural limitation, not by itself
evidence for H1. Non-target recovery and other-relation outcomes are essential.
Exact lexical endpoint recovery is not parsed relation preservation; the latter
is unavailable without a validated language-specific generated-text parser.

## Fixed targets, not discovery

No target discovery, ranking, top-k selection, or model head search runs here.
`--selection-lock` is mandatory. Both validated existing six-relation bundles
and the existing legacy object-only lock are accepted. Missing relations fail;
one object lock is never broadcast to other relations. Explicit `--layer` and
`--head` overrides are recorded in the resolved config, including joint sets.

The inspected existing `dream-ewt-head-v2/relation-selection` bundle fixes:

| Relation | Layer | Head |
|---|---:|---:|
| object_to_verb | 2 | 3 |
| subject_to_verb | 1 | 13 |
| object_adj_to_noun | 1 | 13 |
| subject_adj_to_noun | 1 | 13 |
| object_det_to_noun | 0 | 2 |
| subject_det_to_noun | 5 | 18 |

These are zero-based indices. Several relations share a head, so their labels
do not imply distinct mechanisms. Other-relation controls exclude the target
head and record the relation labels associated with the chosen head. No new
ranking was performed to obtain this table.

## Running

Run from the repository root, with Python >=3.10. Use the repository's pinned
model requirements in separate environments for Dream and DiffuLLaMA. Install
the appropriate CUDA PyTorch build on the GPU host before the editable install.

```bash
python -m pip install -e '.[dev]' -r requirements/dream.txt
python -m pytest tests/test_causal_trajectory.py -q
# Optional: tiny random-weight model using the actual pinned Dream source.
# Downloads model code/config only, never 7B checkpoint weights.
DLMREL_TEST_PINNED_DREAM=1 python -m pytest tests/test_causal_architectures.py -q
dlmrel prepare --dataset configs/datasets/ewt.yaml
```

Set `LOCKS` to your **existing** result bundle; do not run head selection again.
For the inspected Dream run, its path within the existing result archive is:

```bash
LOCKS=dlmrel-results/confirmatory_ewt/dream_7b/ewt/confirmatory_head_search/dream-ewt-head-v2/relation-selection
dlmrel causal-run --model configs/models/dream_7b.yaml --selection-lock "$LOCKS" --output results/causal/smoke --smoke-test
dlmrel causal-run --model configs/models/dream_7b.yaml --selection-lock "$LOCKS" --output results/causal/pilot --causal-config configs/causal/pilot.yaml
```

`--dry-run` checks config, existing target locks and prepared test manifests
without loading a model. `--smoke-test` caps the run at two sentences and one
seed, preserving 64 steps and requested conditions. Every eligible pair in
each admitted sentence receives its own intervention trajectory. All pairs in
that sentence are evaluated for each intervention. The same sentences, spans,
context, and random schedules appear in every condition, providing exact
within-example matching on all recorded structural properties.

After inspecting pilot baselines for floor effects, matched control contrasts,
global degradation and sanity results, expand explicitly:

```bash
# All six existing relations, all temporal windows, full alpha sweep.
dlmrel causal-run --model configs/models/dream_7b.yaml --selection-lock "$LOCKS" --causal-config configs/causal/full.yaml --output results/causal/full
# Final recovery effect of knockout restricted to successive 8-step intervals.
dlmrel causal-run --model configs/models/dream_7b.yaml --selection-lock "$LOCKS" --causal-config configs/causal/temporal.yaml --output results/causal/temporal
# Re-render from saved data without model inference.
python -m dlmrel.experiments.causal_plots results/causal/pilot
```

CLI overrides: repeat `--relation`, `--seed`, `--alpha`, `--window`, `--control`,
or paired `--layer`/`--head`; also `--examples`, `--batch-size`, `--start`,
`--stop`. Repeated scalar options replace the configured list. `--start` or
`--stop` replaces configured interval sweeps. Language comes from the dataset
config. For German/Japanese, prepare `de_gsd.yaml`/`ja_gsd.yaml`, then pass that
dataset to the same command while retaining the model's existing locks. For
DiffuLLaMA use `requirements/diffullama.txt`, `diffullama_7b.yaml`, and its own
existing locks. Cross-model lock substitution fails validation.

The 64-step schedule remains the repository standard. Window fractions and
absolute ranges are configurable. Early/late partition complete pre-revelation
at `ceil((first_endpoint_reveal_step+1)*split)`. Tiny windows can have no active
steps; these cases remain in the denominator and are auditable in raw output.
`pre` requires all subtokens of both endpoints to remain masked. The sampler's
explicit boolean scheduler state is authoritative even if it samples a mask ID.

## Controls and interpretation

Random heads exclude all locked targets. Same-layer controls preserve the
number of requested heads in each layer. Low-relation controls prefer the
target's layer, using existing selection scores. Attention matching requires
`attention_entropy` and/or `attention_magnitude` columns in that same selection
CSV; it minimizes standardized feature distance among the lowest quartile of
relation scores. Missing metadata is recorded as unavailable, not replaced by
an unmatched head. Joint interventions support random/same-layer/low-relation
controls of equal cardinality; automatic attention matching is single-head only.
The default is one control draw per kind; repeat with prespecified
`control_seed` values to assess sensitivity to head choice.

Full reconstruction intentionally starts with no lexical context except BOS.
This can cause low exact recovery, especially on long sentences. Report those
floor effects; do not select successful baselines after seeing outcomes. The
sampler API also permits fixed-context reconstruction via `reconstruction_mask`,
but the CLI's primary task is explicitly all-non-BOS reconstruction. This is a
new behavioral endpoint; it cannot reproduce historical attention accuracy.

`governor_exact` requires every aligned governor subtoken to match the gold IDs;
`dependent_exact` is analogous. `endpoint_exact` is lexical recovery of both
endpoints, **not syntactic relation preservation**. Probabilities and logits are
measured at each token's aligned prediction position on its commitment step.
The governor probability is the mean subtoken probability; its log probability
is the mean conditional log probability, not a joint span likelihood. Raw traces
also retain these values at every step. Non-target accuracy excludes the pair's
subtokens; overall accuracy excludes BOS. Specificity degradation is target
accuracy loss minus non-target accuracy loss. Positive values mean greater
target damage, but must also exceed matched-head effects to support specificity.

Other relation rows include `shares_anchor_tokens`. Shared endpoints cannot
serve as independent unrelated-token controls. Disjoint other-relation summaries
are emitted when such pairs exist. Parsed relation preservation stays JSON null
with an explicit reason; no parse is invented for altered generated text.

## Artifacts and inference

* `config.resolved.json`: model/data revisions, fixed heads, original lock kind
  and hash, selection-score hash, manifest hashes, seeds and all settings.
* `examples/`: atomically saved baseline/intervention traces and paired shards.
  Partial runs retain finished evidence; output directories must be empty.
* `paired.jsonl`: all endpoint, structural, control, generation and behavioral
  fields, seed, active steps, reveal steps, and trace paths.
* `aggregate.csv`, `per_seed.csv`: means, differences, 95% sentence-clustered
  bootstrap intervals, two-sided paired sentence sign-flip tests, Holm-adjusted
  p-values, standardized sentence-level paired effect sizes, seed mean/SD.
* `matched_control_contrasts.csv`: paired target-head effect minus each control
  head effect, with the same clustering. Relative contrasts should not be
  interpreted as relative changes in raw accuracy.
* Exact McNemar tests are reported per seed on the prespecified first pair per
  sentence for each head relation, with `mcnemar_n`, harmed/helped counts. Other
  pairs remain in the full clustered analysis. Seeds are repeated measurements,
  never independent extra sentences; zero-variance effect sizes are undefined.
* `control_availability.json`, `exclusions.csv`, `sanity.json`, `status.json`:
  missing controls, unchanged alignment exclusions, numerical checks, completion.
* `figures/`: PDF, 300 dpi PNG and CSV for head controls, relation matrix,
  temporal windows/interval curve, dose response, and global degradation.

An interval curve measures final recovery following suppression in that interval;
it is not a relabeled attention curve. Dose-response plots report raw recovery;
paired effect uncertainty is available in aggregate tables. Matrix cohorts may
differ by anchor relation and retain overlapping pairs, so compare columns
within a fixed row/cohort and consult disjoint-token controls.

## Validation limits

Automated CPU tests cover multi-head/layer indexing, all requested scales,
unselected heads, masking and interval gates, partial spans, patching and hook
cleanup on exceptions, native default equivalence, BOS/subtoken alignment,
prediction offsets, paired RNG replay, alpha=1, batch/single equivalence,
complete artifact/statistics/figure generation and control availability.
The opt-in test exercises the pinned Dream grouped-query attention architecture
with tiny random weights and full 64-step identity trajectories.

This development host has CPU-only PyTorch and no CUDA device. No pretrained
7B causal pilot or hypothesis test result is claimed. A GPU pilot with the
commands above remains necessary before scientific interpretation or expanding
to expensive grids. Tiny-model effects only verify that the intervention changes
downstream logits; they are not evidence for linguistic mechanisms.

Validation performed on 2026-09-30: **354 tests passed** with the opt-in pinned
Dream test enabled (Python 3.11.9, CPU PyTorch 2.14.1, transformers 4.51.3).
Ruff on every changed Python module and `git diff --check` passed. The suite
retains one existing scalar-conversion warning in `paper_causal.py`. On this
Windows host, tests needed a short writable `--basetemp` path because historical
checkpoint names otherwise exceeded the host's path limit. The real-model smoke
attempt correctly stopped at CUDA preflight; the existing Dream lock dry run
and EWT manifest preparation passed.
