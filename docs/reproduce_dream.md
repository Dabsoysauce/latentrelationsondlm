# Reproducing the Dream-7B results

How to run every Dream-7B experiment and check your numbers against the
recorded run.

> **Which branch to use:** the full working pipeline is on the `dream-7b`
> branch. The shared-structure port isn't finished yet, so run from `dream-7b`
> until it lands on `main`. The commands are the same either way; only the
> file paths change.

## What you should get

| experiment | headline number to verify |
|---|---|
| relation head search | object→verb head **L2 H11**, test acc **0.800**, null 0.326, margin **+0.422**, verdict *survives* |
| data prep | pool **8,783** sentences, **21,530** relation instances |
| POS probe | peak **0.9405** at layer 4, lexical baseline **0.7735** |
| attention entropy | mean sink mass never exceeds **~0.15** |
| logit lens | masked-token top-1 peaks **~0.097** at layer 24 |
| accuracy over time | masked-state object→verb **0.263** vs null 0.346 (below null — expected) |

If your object→verb comes out near 0.08 (chance) instead of 0.80, the model
didn't load correctly — see Gotcha 1.

## 1. Environment

Dream needs `transformers==4.51.3`, which conflicts with the DiffuLLaMA
families' 4.44.2. Use a **dedicated** environment — do not install both.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dream]" ".[probe]"
```

`.[dream]` pins transformers 4.51.3 + accelerate + sentencepiece; `.[probe]`
adds scikit-learn for the POS probe. Python 3.10–3.12 (3.13 has no wheels for
the pinned stack).

## 2. Get the code

```bash
git clone https://github.com/Dabsoysauce/latentrelationsondlm.git
cd latentrelationsondlm
git checkout dream-7b     # until the shared-structure port lands on main
```

## 3. Run it — pick one

### Option A: Modal (no local GPU needed)

The Dream weights (~14 GB) download once into a persistent volume, then the
GPU stage runs on an A100. ~13 min for the head search.

```bash
pip install modal && modal setup
modal run models/dream_7b_base/relation_head_search/code/run_pipeline.py
# other experiments:
modal run models/dream_7b_base/relation_head_search/code/run_pipeline.py --stages curve,analyze
modal run models/dream_7b_base/relation_head_search/code/run_pipeline.py --stages entropy,logitlens,posprobe
```

### Option B: Colab (Colab Pro, A100)

1. Runtime → Change runtime type → A100.
2. Clone the repo (private, so use a token), then:
   ```
   !pip install -q -e ".[dream]" ".[probe]"
   ```
3. **Runtime → Restart session** (the transformers pin only takes effect after
   a restart — this is the most common mistake).
4. Run the stages, then copy results to Drive before the session disconnects
   (Colab wipes local disk):
   ```
   !dlmrel data    --config models/dream_7b_base/relation_head_search/code/dream-7b.yaml
   !dlmrel search  --config models/dream_7b_base/relation_head_search/code/dream-7b.yaml
   !dlmrel analyze --config models/dream_7b_base/relation_head_search/code/dream-7b.yaml
   ```

### Option C: local CUDA GPU

```bash
dlmrel data    --config models/dream_7b_base/relation_head_search/code/dream-7b.yaml
dlmrel search  --config models/dream_7b_base/relation_head_search/code/dream-7b.yaml
dlmrel analyze --config models/dream_7b_base/relation_head_search/code/dream-7b.yaml
```

## 4. Where results land

`results/dream-7b-ewt/` — the headline table is `head_vs_null.csv`. Open it and
check the object→verb row against the table above.

## Gotchas that produce wrong numbers silently

1. **Non-eager attention.** Dream must load with `attn_implementation="eager"`.
   Under sdpa it returns no attention weights and every accuracy is zero. The
   loader raises at load time if this happens, so a clean load means you're OK.
2. **Wrong transformers version.** If you didn't restart after installing (Colab)
   or installed the 4.44.2 stack, Dream's remote code misbehaves. Check
   `python -c "import transformers; print(transformers.__version__)"` → 4.51.3.
3. **Adding a model changes the splits' cache key.** The config lists all models
   in `common_pool_models` so every model scores the same sentences. If you edit
   that list, the pool is rebuilt; confirm it's still 8,783 sentences before
   comparing across models.

## First thing to run to confirm the setup works

```bash
python -m pytest -q
```

CPU-only, ~2 seconds. If the tests pass, the pipeline is intact before you
spend GPU time.
