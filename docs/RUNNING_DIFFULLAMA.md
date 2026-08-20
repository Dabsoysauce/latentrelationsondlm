# Finishing the DiffuLLaMA-7B experiments

Everything DiffuLLaMA still needs, in the order to run it. Each command was
validated with `--dry-run` against prepared manifests; none of them has been run
against the real model.

## Already done

| Experiment | Status |
| --- | --- |
| `head_search` (EWT) | run; produced the selection lock used below |
| `time_curve` (EWT) | run — PR #18, `docs/results/diffullama_confirmatory/` |

## Still to run

| # | Experiment | Dataset | Lock | Forward passes |
| --- | --- | --- | --- | ---: |
| 1 | `pos_probe` | ewt | no | 18,000 |
| 2 | `logit_lens` | ewt | no | 15,000 |
| 3 | `attention_entropy` | ewt | no | 27,000 |
| 4 | `external_transfer` | de_gsd | yes | 2,766 |
| 5 | `external_transfer` | ja_gsd | yes | 1,611 |

Ordered cheapest first so a failure surfaces early.

`dlmrel run --dry-run` prints a much larger `estimated_forward_passes` because
it multiplies every trajectory point by *all* manifest sentences. Only
`pos_probe` actually reads select+dev+test; entropy and logit lens read the test
split alone, and a locked transfer reads the test split at the single frozen
progress point recorded in the lock. The table above is the real cost.

## Prerequisite: the selection lock

Runs 4 and 5 need the EWT lock, which currently lives in the open PR #18. Merge
it first, then use:

```text
docs/results/diffullama_confirmatory/selection_lock.json
```

It pins `object_to_verb` at layer 17, head 15. Its `select_manifest_hash` and
`dev_manifest_hash` match what `dlmrel prepare` produces from
`configs/datasets/ewt.yaml`, so it is valid against the current manifests.

## Setup

Requires an A100. DiffuLLaMA-7B is bf16 with eager attention, and eager is not
optional: sdpa accepts `output_attentions=True` and returns nothing, which turns
every result into zeros.

```bash
pip install -e .
pip install -r requirements/diffullama.txt
```

The pin in that requirements file is load-bearing. DiffuLLaMA's
`attention_patch.py` replaces `LlamaModel.forward` with the 4.44-era
implementation and patches `LlamaFlashAttention2`, which later releases removed.

Then prepare the three treebanks and smoke-test the adapter:

```bash
dlmrel prepare --dataset configs/datasets/ewt.yaml
dlmrel prepare --dataset configs/datasets/de_gsd.yaml
dlmrel prepare --dataset configs/datasets/ja_gsd.yaml
dlmrel smoke-test --model configs/models/diffullama_7b.yaml
```

Do not skip the smoke test. The checkpoint is stored under a `denoise_model.*`
wrapper namespace, and loading it the obvious way leaves the backbone randomly
initialised with only a warning.

## The runs

```bash
dlmrel run --model configs/models/diffullama_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/pos_probe.yaml --run-id diffullama-ewt-posprobe-v1
```

```bash
dlmrel run --model configs/models/diffullama_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/logit_lens.yaml --run-id diffullama-ewt-logitlens-v1
```

```bash
dlmrel run --model configs/models/diffullama_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/attention_entropy.yaml --run-id diffullama-ewt-entropy-v1
```

```bash
dlmrel run --model configs/models/diffullama_7b.yaml --dataset configs/datasets/de_gsd.yaml --experiment configs/experiments/external_transfer.yaml --selection-lock docs/results/diffullama_confirmatory/selection_lock.json --run-id diffullama-de-transfer-v1
```

```bash
dlmrel run --model configs/models/diffullama_7b.yaml --dataset configs/datasets/ja_gsd.yaml --experiment configs/experiments/external_transfer.yaml --selection-lock docs/results/diffullama_confirmatory/selection_lock.json --run-id diffullama-ja-transfer-v1
```

If Colab disconnects, rerun the same command with `--resume` and the same
`--run-id`. Checkpoints are written every 300 sentences.

## After each run

```bash
dlmrel validate --run-dir results/<track>/diffullama_7b/<dataset>/<experiment>/<run-id>
```

`dlmrel run` already validates and exits non-zero on failure, so this is for
re-checking a directory later. Copy run directories to Drive before the session
ends.

## Known risk: Japanese

Run 5 puts Japanese text through DiffuLLaMA's LLaMA SentencePiece tokenizer.
Alignment goes through character offsets, and Japanese has no whitespace word
boundaries, so a large share of sentences may fail full alignment and be
excluded. Check `exclusions` in that run's output before reading anything into
the result — a transfer number computed over a small surviving subset is not
comparable to the German one.
