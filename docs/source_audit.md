# Source and repository audit

Audit date: 2026-08-12
Implementation base: `6e5aaec772f942573add5d211ab10edf683d69bd`

## Evidence hierarchy

| Source | Role | Disposition |
|---|---|---|
| Algoverse original paper/review package | Scientific definitions and reviewer requests | Protocol authority where the code is ambiguous |
| Git history and source at the base commit | Migratable implementation logic | Retained where compatible with the repaired protocol |
| Previously committed CSVs/figures | Historical regression evidence | Archived; never treated as confirmatory |
| New resolved configs, manifests, locks, and raw rows | Reproducible evidence | Required for every rigorous run |

## Verified contradictions at the base commit

1. `load_treebanks` concatenated UD train/dev/test and `split_sentences`
   created new select/dev/test partitions. This violates official-boundary
   confirmatory evaluation and invalidates a confirmatory interpretation of
   existing metrics.
2. Head search scored every head on test and emitted test rankings. The repaired
   design permits only a select/dev-derived locked head to enter test.
3. The shared pool was sentence-text based and omitted LLaDA. The primary
   cross-model analysis instead needs a stable instance-level intersection.
4. The alignment fallback used decoded, stripped fragments and substring
   search. This is not safe for byte-level tokens, Unicode, repeated text, or
   no-space tokenization.
5. Temporal aggregation omitted timestep from grouping keys, collapsing the
   curve. The masked denominator was sentence-level rather than eligible
   relation-instance-level.
6. Entropy ran only at the final frame; POS probing was fully visible and did
   not tune on dev; logit lens reported only top-1 at a few steps.
7. Direct logit attribution and causal head ablation were absent.
8. Runtime cloning followed an unpinned DiffuLLaMA repository branch.
9. Several YAML fields were ignored, core imports were undeclared, CLI examples
   had drifted, and documentation contained placeholders.
10. Existing outputs lacked sufficient model/dataset revision, split manifest,
    exclusion, raw-instance, and environment provenance.

## Historical scientific result disposition

The historical DiffuGPT object-to-verb pre-unmask result is the main legacy
regression target. Preliminary final-state receiver heads exist for DiffuGPT,
DiffuLLaMA, and Dream, but those artifacts do not establish a larger-model
pre-unmask effect. Mixed noun-modifier results are often compatible with fixed
offset behavior. No current artifact establishes locked cross-treebank or
cross-lingual transfer. These are hypotheses for the repaired pipeline, not
accepted conclusions.

## Reviewer-driven additions

The implementation protocol therefore requires multiple DLMs, a GPT-2
final-state baseline, official EWT select/dev/test boundaries, EWT-locked
transfer to GUM/LinES/ParTUT, structural challenge slices, stronger automatic
controls, sentence-clustered uncertainty, multi-seed trajectories, explicit
causal tests, pinned revisions, and complete run provenance.
