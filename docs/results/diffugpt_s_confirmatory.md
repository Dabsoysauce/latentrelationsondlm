# DiffuGPT-S Confirmatory Results

This directory documents the completed rigorous DiffuGPT-S experiments on English EWT: head search, time curve, attention entropy, logit lens, and POS probe.

## Experiment

DiffuGPT-S had no adapter in the rigorous framework before this work. `src/dlmrel/models/diffugpt.py` and the generalized wrapper-checkpoint remap in `src/dlmrel/models/_backbone.py` add one, reusing DiffuLLaMA's `DiscreteDiffusionModel` wrapper and third_party revision rather than duplicating the loading logic.

Head search selects and locks one head per relation on the EWT select/dev split, then scores the frozen selection once on held-out test. The other four experiments consume that same frozen lock.

## Runs

Run IDs:

- `diffugpt_s-head_search-v1`
- `diffugpt_s-time_curve-v1`
- `diffugpt_s-attention_entropy-v1`
- `diffugpt_s-logit_lens-v1`
- `diffugpt_s-pos_probe-v1`

Model:

`DiffuGPT-S` (`diffusionfamily/diffugpt-s`, 124M params, 12 layers x 12 heads), run on CPU in float32.

Dataset:

`UD English EWT`

## Validation

All five runs passed `dlmrel validate` with no validation errors.

## Reading the numbers

The confirmatory head-search test scores the frozen lock exclusively at `normalized_progress=0.0`, `visibility=both_masked` — the hardest point in the trajectory, before any token is revealed. At that point every masked position is embedded identically, so position is the only signal available to any head. For five of six relations (all noun-modifier relations, which have a single dominant word-order offset), the locked head's accuracy is identical to the fixed-offset baseline — that is the expected outcome at t=0, not a bug. `object_to_verb` is the one relation with a genuinely spread offset distribution, and it is the one relation where the locked head separates from the fixed-offset baseline (0.361 vs 0.347 accuracy; gold receiver attention mass 0.298 vs 0.018 for a matched control, p=0.000999). `object_to_verb`'s `holm_adjusted_p_value` is null by design: it is the pre-registered primary relation, tested at its own raw p-value rather than folded into the five-relation Holm-corrected secondary family.

`timecurve_metrics.csv` shows `object_to_verb` accuracy rising with denoising progress on the locked head — 0.361 at `normalized_progress=0.0` up to 0.683 by `0.875` — consistent with structure strengthening as the sequence is revealed, the same qualitative pattern seen in DiffuLLaMA-7B and Dream-7B.

Included files:

- `selection_lock.json` — frozen head selection produced by head search
- `headsearch_summary.json` / `headsearch_validation.json` / `headsearch_metrics.csv`
- `timecurve_summary.json` / `timecurve_validation.json` / `timecurve_metrics.csv`
- `attention_entropy_summary.json` / `attention_entropy_validation.json` / `attention_entropy_metrics.csv`
- `logitlens_summary.json` / `logitlens_validation.json` / `logitlens_metrics.csv`
- `posprobe_summary.json` / `posprobe_validation.json` / `posprobe_metrics.csv`

## Attribution

Model port, experiment runs, and results prepared by ihateSAS.
