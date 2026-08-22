# Implementation status and source limitations

## Corrected active implementation

The ten canonical experiment configs dispatch to selection/test-only runners.
Fully visible selection, single-source receiver argmax, all-64-step curves,
two-way visibility, relative depths, native eventual-token targets, timing,
exact Llama-style DLA, exact single-head intervention, saved-evidence plotting,
and locked multilingual transfer are implemented.

All corrected configs freeze seeds `[42, 43, 44]`. Corrected code contains no
development loader, permutation execution, or Holm correction. Historical
modules remain for backward-compatible inspection and recovery of already
completed old-ID results only.

CPU fake-model and pure tensor tests verify scientific definitions and artifact
plumbing. Real Dream/DiffuLLaMA runs are intentionally not executed as part of
implementation validation.

## Explicit source ambiguities and blockers

- Paper prose averages attender rows, but the preserved executable relation
  notebook selects the final attender subtoken. The active code follows the
  executable source, as requested.
- Exact entropy early/late boundaries were not recoverable from the supplied
  notebook/CSVs. Config freezes `0..15` and `48..63` and labels the
  limitation in every summary.
- The old Stanford log-linear tagger model/JAR were not included. Exact POS
  runs require external paths in `STANFORD_POS_TAGGER_JAR` and
  `STANFORD_POS_TAGGER_MODEL`; no UD-UPOS substitution is made.
- POS-head causal comparison requires a completed exact POS run supplied
  through `DLMREL_POS_HEAD_RANKINGS`.
- The local research-context index named several `context/*.md` files, but
  those files were absent. Paper, Drive notebooks, source notebook, and
  archived CSVs are the available provenance basis.

Existing result directories, notebooks, Drive data, Modal infrastructure, and
historical IDs are not overwritten by corrected runs.
