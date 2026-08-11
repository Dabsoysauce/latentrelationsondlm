# dlmrel

Do attention heads in diffusion language models track grammatical relations, or
do they just point a fixed number of positions away?

Diffusion LMs denoise a masked sequence over many steps, so you can stop them
partway and look at what the model has figured out while most of the sentence
is still `[MASK]`. We use that to ask whether a head has locked onto a
dependency before either of its endpoints has been revealed.

## The one thing to understand before reading any number

A head that always attends, say, one position to the right will get
adjective-to-noun right almost every time in English, because those two words
are nearly always next to each other. It has learned nothing about syntax. So
comparing head accuracy to random chance is meaningless.

Every number here is compared against a **fixed-offset baseline**: predict the
receiver as `attender + k`, with `k` chosen on the same split used to pick the
head, and scored on held-out data. The baseline gets exactly the same tuning
the head gets. A head must clear the *upper end* of that baseline's confidence
interval, not its point estimate.

## Status

| model | heads | object→verb | status |
|---|---|---|---|
| DiffuGPT-S | 144 | **0.802** (null 0.329, margin +0.425) | done |
| Dream-7B | 784 | **0.800** (null 0.326, margin +0.422) | done |
| DiffuLLaMA-7B | 1024 | — | **in progress** |
| LLaDA-8B | — | — | not started |

Object→verb replicates across two unrelated architectures (GPT-2-derived and
Qwen2-derived) at nearly the same accuracy and margin. All runs are on
**verified-identical splits** — same 8,783-sentence pool, same select/dev/test
membership and order, same 21,530 relation instances.

## Install

Two mutually exclusive environments. `transformers` versions do not overlap:

```bash
pip install -e ".[dev]"           # CPU: data, baselines, analysis
pip install -e ".[gpu]"           # DiffuGPT-S / DiffuLLaMA -> transformers==4.44.2
pip install -e ".[dream]"         # Dream-7B               -> transformers==4.51.3
```

Running a Dream config in a 4.44.2 environment (or vice versa) fails at load
rather than producing wrong numbers, but do not try to share one runtime.

## Run

```bash
dlmrel data     --config configs/<model>.yaml   # CPU, ~1 min
dlmrel nulls    --config configs/<model>.yaml   # CPU, seconds
dlmrel search   --config configs/<model>.yaml   # GPU
dlmrel analyze  --config configs/<model>.yaml   # CPU, seconds
```

Configs: `diffugpt-s.yaml` (smallest, run first), `default.yaml`
(DiffuLLaMA-7B), `dream-7b.yaml`.

On Modal, `models/dream_7b_base/relation_head_search/code/` has a worked
example that runs the CPU stages on a CPU container and only `search` on an
A100. Dream's full pipeline cost about 13 minutes of A100 time.

## Three ways to get wrong numbers with no error message

These have all happened. Each one produces plausible-looking output.

1. **Non-eager attention.** `sdpa` and `flash_attention_2` accept
   `output_attentions=True` and quietly return nothing, so every accuracy
   becomes zero. Keep `attn_implementation: eager`.

2. **The DiffuGPT-S checkpoint.** It is saved under different parameter names
   than a plain GPT-2, so loading it the obvious way matches zero weights and
   gives you a randomly initialised model with only a warning. The loader
   remaps the names and refuses anything that fills less than 90% of the model.

3. **Split drift between models.** Splits are carved by index from a shuffled
   pool, so two models admitting slightly different sentences diverge far more
   than the pool difference implies — a ~1% difference once produced only 72.9%
   test-split overlap. `common_pool_models` in each config pins every model to
   the same pool. If you add a model, every model's pool key changes; check
   that the pool is still the same *set* before comparing across runs.

If a search comes back near chance, run `scripts/diagnose_heads.py` before
assuming the result is real.

## Layout

The repository was reorganised into `models/<model>/<experiment>/{code,results}`
(PR #6). That reorg did not carry the pipeline code across, so this branch
restores it. Where the new structure was unambiguous, artifacts are filed into
it:

- `models/*/relation_head_search/results/` — headline outputs per model.
  `head_scores_merged.csv` is included deliberately: it is the expensive
  product of the GPU stage, and keeping it means `dlmrel analyze` can be
  re-run without another A100.
- `cross_model_analysis/figures/` — publication figures (PDF + PNG).

The shared pieces are left at the root because they are not per-model and the
package has to stay importable:

- `src/dlmrel/` — the package. `nulls.py` is the baseline, `stats.py` the
  head-vs-baseline comparison, `splits.py` the data splits, `model.py` the
  per-family loaders.
- `configs/` — one YAML per model. Everything that changes a reported number.
- `scripts/` — diagnostics and figure generation.

**Open question for the team:** if the package should live somewhere else under
the new structure (`shared/dlmrel/`?), say so and it can move — that is a
rename plus a one-line change to `pyproject.toml`, not a rewrite. It was left
alone here rather than guessed at.

## Adding a model

`model.py` dispatches on `ModelConfig.family`. Dream is the worked example of a
non-DiffuLLaMA family: it is natively bidirectional, so `DreamAdapter` sets
`mask_free = True` and the diffusion code calls it with `attention_mask=None`,
skipping the DiffuLLaMA anneal mask entirely. That also keeps DiffuLLaMA's
`model.py` — which rewrites transformers-4.44 internals on import — off Dream's
code path, which is what makes the two version pins coexist in one repository.

Dream also has no fast tokenizer, so `alignment.py` falls back to a manual
character-offset scan. It fails closed: a token that cannot be placed gets a
zero-width span and drops the word (and, under `require_full_alignment`, the
sentence) rather than aligning to the wrong position.
