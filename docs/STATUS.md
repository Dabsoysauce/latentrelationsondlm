# Verified status and provenance

## What is implemented

- Dream-7B, DiffuLLaMA-7B, and a fake CPU test adapter.
- English EWT, German GSD, and Japanese GSD at pinned UD 2.15 revisions.
- Official-boundary manifests, cross-role deduplication, strict alignment, and
  common-instance cross-model comparison.
- Locked head search and transfer, time curves, attention entropy, logit lens,
  masked POS probing, controls, clustered uncertainty, resume, and validation.
- Independent select-top-five/dev-choice locks for all six canonical
  relations, with enforced denominators, per-seed evidence, selection-aware
  permutation tests, Holm correction for the five predefined secondaries, and
  a CPU-only derivation command for completed all-head runs.
- Central strict JSON normalization for NumPy/Parquet values, deterministic
  checkpointed within-instance permutation inference, and relation-aware
  downstream lock resolution.

CPU tests verify the protocol plumbing. Real-model GPU smoke tests and full
experiments have not yet been run in this implementation; “implemented” does
not mean “scientifically validated.”

Existing Dream select/dev all-head inference remains reusable for six-lock
derivation, but its object-head-only test rows do not provide five secondary
test results or the all-head test evidence required by the corrected
permutation null.

## Why earlier results are preliminary

The earlier repository results are Dream and DiffuLLaMA experiments produced
before this protocol. The previous loader could combine UD splits and resplit
them, head discovery and final reporting were not protected by the current
selection lock, cross-model rows were not guaranteed to be identical, and
controls, uncertainty, versions, commands, and exclusions were less complete.
Those outputs are being rerun, not reused as final evidence. DiffuGPT is older
work and is outside this rerun.

## Archive location

The preliminary files were removed from the active branch but remain
recoverable from Git commit `a809be6` under
`results/reference_legacy/pre_rigorous_6e5aaec/`. A standalone local copy was
created at:

```text
C:\Users\haisa\Downloads\Algoverse\preliminary-results-pre-rigorous-6e5aaec.zip
SHA-256: 236E8A9558F4BD1B41ADF32F1E4BD635B86A6929F909410D14CAFB4E8AC334D2
```

Upload that ZIP as a GitHub Release asset if the team wants convenient access
without keeping generated outputs on the main branch.
