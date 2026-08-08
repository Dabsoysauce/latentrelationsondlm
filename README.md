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
the head gets.

Under that baseline, object-to-verb is the interesting relation. Its endpoints
sit 1 to 4 words apart with no dominant distance, so no fixed offset can solve
it. The noun-modifier relations are mostly solvable by position alone, and we
say so rather than claiming them.

## Install

```bash
pip install -e ".[dev]"    # CPU: data, baselines, analysis
pip install -e ".[gpu]"    # adds torch and the pinned transformers
```

## Run

```bash
dlmrel data     --config configs/default.yaml   # CPU, ~2 min, downloads UD-EWT
dlmrel nulls    --config configs/default.yaml   # CPU, seconds
dlmrel search   --config configs/default.yaml   # GPU, ~15-30 min for 7B
dlmrel curve    --config configs/default.yaml   # GPU, long - see note below
dlmrel analyze  --config configs/default.yaml   # CPU
```

`configs/diffugpt-s.yaml` runs the same thing on the small model. Do that one
first: it takes minutes and the answer is already known (head L4 H0 should get
about 0.80 on object-to-verb). If it doesn't, something is broken in the setup
rather than in the science.

Results land in `results/<run>/`. On Colab that disk is wiped when the session
ends, so copy them to Drive.

## Three ways to get wrong numbers with no error message

These have all happened. Each one produces plausible-looking output.

1. **Non-eager attention.** `sdpa` and `flash_attention_2` accept
   `output_attentions=True` and quietly return nothing, so every accuracy
   becomes zero. Keep `attn_implementation: eager`.

2. **The DiffuGPT-S checkpoint.** It is saved under different parameter names
   than a plain GPT-2, so loading it the obvious way matches zero weights and
   gives you a randomly initialised model with only a warning. The loader now
   remaps the names and refuses anything that fills less than 90% of the model.

3. **transformers version.** Must be `4.44.2`. The DiffuLLaMA attention patch
   rewrites internals that later versions removed. There is no workaround.

If a search comes back near chance, run `scripts/diagnose_heads.py` before
assuming the result is real. It checks whether the model has ordinary
previous-token and next-token heads at all, which a working GPT-2 always does.

## Layout

- `src/dlmrel/` — the package. `nulls.py` is the baseline, `stats.py` is the
  head-vs-baseline comparison, `splits.py` builds the data splits.
- `configs/` — one YAML per model. Everything that changes a reported number
  lives here.
- `scripts/diagnose_heads.py` — run when a result looks wrong.
- `scripts/verify_legacy.py` — reproduces the published paper's numbers from
  the old pipeline's output files.
- `proposal/` — the NeurIPS DiffuLM submission plan.

## Data splits

Sentences come from UD English-EWT, all splits pooled and resampled at random,
then divided into `select` (search heads), `dev`, and `test` (report). Two
details that matter: both models are restricted to the sentences *both*
tokenizers accept, so they are scored on identical data; and repeated sentences
are removed, because EWT contains duplicate boilerplate that would otherwise
land in two splits at once.

## Note on `dlmrel curve`

As configured this is 5 seeds x 64 timesteps x 1000 sentences, which is about
13 hours. Do not start it casually. It needs stride and sentence-count options
added first.
