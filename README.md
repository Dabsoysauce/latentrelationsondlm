# Positional or relational? Attention heads in diffusion language models

Diffusion language models denoise a masked sequence over many steps, so unlike
autoregressive models they expose intermediate states. That makes it possible to
ask whether an attention head tracks a grammatical dependency, and whether it
does so *before* the tokens at either end of that dependency have been revealed.

The trap is that a head which always points a fixed number of positions away
solves any dependency whose endpoints sit at a near-constant distance. In
English, adjective→noun and determiner→noun are adjacent almost by definition,
so a `+1` head scores above 0.75 on them without representing anything
syntactic. **Every number in this repository is reported against a fixed-offset
null model**, fit exactly the way a head is selected.

This is a rebuild. It supersedes a pipeline whose headline was measured against
uniform chance (~6%) — a baseline that four of six relations only cleared
because the null was wrong.

**Picking this up cold? Read [HANDOFF.md](HANDOFF.md) first** — full project
history, every design decision and why, what has and has not been run, the
failure modes that produce wrong numbers silently, and the work queue.

## What the pilot data already establishes

Recomputed from the previous runs' raw score files, best head per relation on
held-out sentences, against a `receiver = attender + k` null with *k* fit on
train:

| relation | k\* | null | DiffuGPT-S | Δ | heads > null | DiffuLLaMA-7B | Δ | heads > null |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| object→verb | −2 | 0.314 | **0.755** | +0.441 | 29/144 | **0.877** | +0.600 | 151/1024 |
| subject→verb | +1 | 0.429 | 0.521 | +0.092 | 6/144 | **0.660** | +0.227 | 46/1024 |
| object adj→noun | +1 | 0.750 | 0.827 | +0.077 | 1/144 | **0.942** | +0.192 | 13/1024 |
| subject adj→noun | +1 | 0.682 | 0.773 | +0.091 | 2/144 | **0.886** | +0.205 | 19/1024 |
| object det→noun | +1 | 0.553 | 0.600 | +0.047 | 3/144 | **0.895** | +0.337 | 42/1024 |
| subject det→noun | +1 | 0.600 | 0.589 | −0.011 | 3/144 | **0.856** | +0.256 | 30/1024 |

Two things follow.

**Scale rescues the finding rather than threatening it.** DiffuGPT-S clears the
null on one relation; DiffuLLaMA-7B clears it on all six. The four relations
that look like positional artifacts in the small model recruit 13–42 heads each
in the large one.

**The `heads > null` column is the answer to multiple comparisons.** Searching
144 or 1024 heads and reporting the winner needs a defence. For object→verb the
signal is carried by a broad population spanning every layer, and select→test
rank correlation is ρ ≈ 0.97. For the small model's weak relations, 1–3 heads
out of 144 clear the bar — indistinguishable from what searching that many
hypotheses buys for free.

### A double dissociation

Two heads in DiffuLLaMA-7B, profiled across all six relations:

| head | object→verb | the four adjacent relations | |
|---|--:|--:|---|
| L18 H10 | 0.141 | 0.856 – 0.942 | positional |
| L3 H11 | 0.877 | 0.000 – 0.023 | relational |

L18 H10 solves everything adjacent and nothing else, which is what a `+1`
heuristic looks like. L3 H11 scores essentially zero on the adjacent relations,
so no fixed offset can account for it. `dlmrel analyze` reports both profiles
(`selectivity` and `adjacency_bias`) for every head.

### Where scale costs something

Object→verb accuracy while *both* endpoints are still masked is 0.434 in
DiffuGPT-S and 0.108 in DiffuLLaMA-7B — below the null. Final-state relation
heads strengthen with scale; anticipation during denoising weakens. These are
separable phenomena and should not be reported as one. Note the two numbers are
not yet protocol-matched (5 seeds with a ≥25-masked gate versus a single run),
which `dlmrel curve` fixes by retaining per-instance correctness.

## Install

```bash
pip install -e ".[dev]"          # CPU: data, nulls, analysis
pip install -e ".[gpu]"          # adds torch/transformers for the head search
```

The GPU stages clone [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA)
into `third_party/` at runtime for its `DiscreteDiffusionModel` wrapper and
`get_anneal_attn_mask`.

## Run

```bash
dlmrel data     --config configs/default.yaml   # CPU  ~2 min, downloads UD-EWT
dlmrel nulls    --config configs/default.yaml   # CPU  seconds
dlmrel search   --config configs/default.yaml   # GPU  the long pole
dlmrel curve    --config configs/default.yaml   # GPU
dlmrel analyze  --config configs/default.yaml   # CPU
```

`data`, `nulls` and `analyze` need no GPU, so the null model and every
diagnostic can be validated before spending A100 hours.

## Design decisions worth knowing about

**Three splits, not two.** `select` searches all heads, `dev` tunes anything
else, `test` reports. The previous pipeline held out sentences but selected and
reported the head on the same data.

**Random sampling.** Sentences are drawn from a shuffled pool under a fixed
seed. The previous pipeline walked EWT from the top until a quota filled; EWT is
ordered by genre, so its "1000 training sentences" were a contiguous block of
one source.

**`max_seq_len` is 128, not 64.** The old cap preferentially deleted exactly the
long-distance dependencies that distinguish a relation head from an offset head.

**BOS and self are masked before the argmax.** Diffusion LMs park enormous
attention mass on BOS; without excluding it every head "predicts" position 0.

**Eager attention is mandatory.** `sdpa` and `flash_attention_2` accept
`output_attentions=True` and silently return nothing, which turns every accuracy
into zero without raising. `load_model` asserts against this at startup.

**Word distance is stored at extraction time.** It is what the null consumes and
what the distance-stratified analysis bins over — the experiment that turns the
double dissociation from an anecdote about two heads into a curve.

**The templated corpus is retired.** On the fixed-word-order synthetic sentences
used early in this project, a single head reaches accuracy 1.000 on three
separate relations, because every relation there is adjacent. Results on it
measure adjacency, not syntax.

## Layout

```
src/dlmrel/
  config.py      all knobs that change a reported number
  treebank.py    UD download, pooling, disjoint splits
  alignment.py   UD words -> model sub-tokens, via character offsets
  relations.py   gold relation extraction, with word distance
  nulls.py       the fixed-offset null model  <- the point of the rebuild
  stats.py       selection diagnostics, CIs, head profiling
  model.py       model loading, with the eager-attention guard
  diffusion.py   masking schedule and attention extraction
  scoring.py     head search and masked-state curve
  splits.py      deterministic split reconstruction for GPU stages
  cli.py         dlmrel data|nulls|search|curve|analyze
scripts/
  verify_legacy.py   reproduce the table above from the old result files
```

## Open questions this repo is built to answer

1. Does the double dissociation hold as a *curve* over dependency distance,
   rather than as two hand-picked heads?
2. Does masked-state anticipation really vanish at scale, under a
   protocol-matched comparison?
3. Is the masked-state metric well posed at all? It scores against the gold
   parse of the original sentence, which penalises a model that denoises to a
   different but valid sentence.
