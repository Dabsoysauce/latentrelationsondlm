# HANDOFF — `dlmrel`: positional or relational attention heads in diffusion LMs

**Written 2026-08-02.** This document is the complete context for the project.
It assumes you know nothing about it. If you are a person or an agent picking
this folder up cold, read this file top to bottom before touching anything —
particularly §3, which is the one idea the whole rebuild exists to enforce.

`README.md` is the short public-facing version. This file is the long one: what
happened, why every design decision is what it is, what has actually been run,
what has *not*, and where the bodies are buried.

---

## Contents

1. [What the project is asking](#1-what-the-project-is-asking)
2. [History — how we got here](#2-history--how-we-got-here)
3. [The central correction: the fixed-offset null](#3-the-central-correction-the-fixed-offset-null)
4. [What the data already establishes](#4-what-the-data-already-establishes)
5. [State of play — done vs. not done](#5-state-of-play--done-vs-not-done)
6. [The codebase, module by module](#6-the-codebase-module-by-module)
7. [Running it](#7-running-it)
8. [Output files and what each column means](#8-output-files-and-what-each-column-means)
9. [Gotchas that will silently ruin a run](#9-gotchas-that-will-silently-ruin-a-run)
10. [External data and where it lives](#10-external-data-and-where-it-lives)
11. [Open questions and the work queue](#11-open-questions-and-the-work-queue)
12. [Publication context](#12-publication-context)
13. [Glossary](#13-glossary)

---

## 1. What the project is asking

Diffusion language models (DLMs) generate by iteratively denoising a fully
masked sequence over many steps. Unlike autoregressive models, they expose
**intermediate states** — you can freeze the model partway through generation
and inspect it while much of the sentence is still `[MASK]`.

That makes a question available that you cannot ask of GPT-style models:

> Does an attention head track a **grammatical dependency** — and does it do so
> *before* the tokens at either end of that dependency have been revealed?

If yes, the model is committing to syntactic structure ahead of lexical
content, which is a substantive claim about how DLMs generate.

We study six dependency relations, always in the direction **attender →
receiver** (the attention row is read from the attender, and we ask which
position it points at):

| relation name | attender | receiver | typical distance |
|---|---|---|---|
| `object_to_verb` | object noun | its governing verb | spread, −1 to −4 |
| `subject_to_verb` | subject noun | its governing verb | usually +1 |
| `object_adj_to_noun` | adjective (`amod`) | object noun it modifies | +1 |
| `subject_adj_to_noun` | adjective (`amod`) | subject noun it modifies | +1 |
| `object_det_to_noun` | determiner (`det`) | object noun | +1, sometimes +2 |
| `subject_det_to_noun` | determiner (`det`) | subject noun | +1, sometimes +2 |

The metric is **receiver-prediction accuracy**: take the attention row from the
attender's last sub-token, mask out BOS and the attender's own positions, take
the argmax, and ask whether it lands anywhere in the receiver's token span.

---

## 2. History — how we got here

### 2.1 The submitted paper

A paper titled *"Latent Linguistic Relations During Diffusion Language Model
Denoising"* was submitted to **Sci-FM @ COLM 2026** and **rejected**.

- **Reviewer A:** 4 / Weak Accept, confidence 2.
- **Reviewer B:** 2 / Reject, confidence 4. Objections: a single small model;
  narrow linguistic scope; weak causal evidence; an ablation that contradicted
  the paper's own claim.

The paper's headline numbers, all from **DiffuGPT-S** (12 layers × 12 heads =
144 heads):

- object→verb receiver-prediction accuracy **75.5%** on held-out sentences;
- **43.4% ± 3.3%** while both endpoints were still masked, "versus 6.1% random";
- **61.7% ± 1.9%** once endpoints were revealed.

### 2.2 The audit (2026-08-01)

Re-derivation from the raw result files found a problem that neither reviewer
named, and that is worse than anything they did name: **the baseline was
wrong.** See §3. Under the correct baseline, four of six relations have
confidence intervals containing the null and one is *beaten* by it.

Two further defects were found in the old pipeline:

1. **Sequential sampling.** `collect_ud_examples` walked the treebank from the
   top until a quota filled. UD-EWT is ordered by genre, so "1000 training
   sentences" was a contiguous block of one source.
2. **`MAX_SEQ_LEN = 64`.** This preferentially deleted long sentences, i.e.
   exactly the long-distance dependencies that distinguish a relation head from
   a fixed-offset head.

And a **misfiling** was corrected. A DiffuLLaMA-7B run from June 2026 had been
written off internally as a failed replication ("45% vs 75.5%, 0% masked
state"). That judgment came from the **templated synthetic corpus**, not from
UD gold. On UD gold the same 7B run is the *strongest result in the project*
(§4). Scaling up is therefore not a gamble — it is buildout of an existing
result.

### 2.3 The templated corpus is retired

Early work used synthetic template-generated sentences
(`SAE4DLM-CE/dlm_order/data/train_data_large.txt`). On that corpus a single
head (L0 H3) reaches accuracy **1.000** on three separate relations
simultaneously. That is not syntax — every relation in a fixed-word-order
template is adjacent, so the corpus measures adjacency and nothing else.
**Do not use it. Do not cite numbers from it.**

### 2.4 The rebuild

On 2026-08-01 the project was rebuilt as this package. It briefly lived in a
private GitHub repo, `Dabsoysauce/dlm-relation-heads`; on 2026-08-02 the user
deleted both the GitHub repo and the local `.git` directory. **This folder is
currently not under version control.** If you want history, `git init` fresh.

---

## 3. The central correction: the fixed-offset null

**Read this section twice. Everything else is downstream of it.**

A head that always points a fixed number of positions away from itself —
"attend to the token immediately after me" — will solve any dependency whose
endpoints sit at a near-constant distance. It represents no syntax whatsoever.

In English, adjective→noun and determiner→noun are adjacent almost by
definition. So the correct baseline is not uniform chance. It is:

```
receiver = attender + k
```

with **k chosen to maximise accuracy on the same split used to select the
head**, and accuracy reported on held-out data. That is protocol-matched: the
null gets the same freedom to be tuned that the head does.

Measured this way, on the pilot data:

| relation | best k | null accuracy |
|---|--:|--:|
| object→verb | −2 | 0.314 |
| subject→verb | +1 | 0.429 |
| object adj→noun | +1 | 0.750 |
| subject adj→noun | +1 | 0.682 |
| object det→noun | +1 | 0.553 |
| subject det→noun | +1 | 0.600 |

The paper compared against **6.1%**. A `+1` head clears 0.75 on adjective→noun
for free. That is the whole problem.

**Why object→verb survives and the others do not:** its true offset is *spread*
across roughly −1 to −4, so no single k captures it. A relation with a
concentrated distance distribution is trivially solvable by position; a relation
with a spread one is not. `dlmrel nulls` prints the distance distribution per
relation for exactly this reason — it tells you in advance which relations can
even in principle produce an interesting result.

### 3.1 The multiple-comparisons answer

Searching 144 heads (DiffuGPT-S) or 1024 (DiffuLLaMA-7B) and reporting the
winner needs a defence. Holding out *sentences* does not provide one, because
the same head is both selected and reported. Two diagnostics settle it without
inventing a correction factor:

- **`n_heads_above_null`** — how many of the searched heads clear the null at
  all. A genuinely represented relation recruits a broad population (29/144,
  151/1024 for object→verb). Noise recruits 1–3 out of 144, which is roughly
  what searching that many hypotheses buys you for free.
- **`selection_rho`** — Spearman correlation between select-split and
  test-split accuracy across all heads. ρ ≈ 0.97 throughout, meaning the whole
  ranking transfers and the winner was not a lucky draw.

The rebuild also uses **three splits** (`select` / `dev` / `test`) rather than
two, so head selection is itself held out.

### 3.2 The verdict rule

`build_head_vs_null_table` marks a relation `"survives"` **only if the head's
Wilson lower bound exceeds the null point estimate.** A head whose interval
contains the null is reported as `"not distinguishable"`. This is deliberately
conservative and should not be loosened to make a table look better.

---

## 4. What the data already establishes

All numbers below were **recomputed from the previous runs' raw score files
using this package's code**, not copied from the old notebooks.
`scripts/verify_legacy.py` reproduces them on demand.

### 4.1 Best head per relation vs the fixed-offset null

| relation | k\* | null | **DiffuGPT-S** | Δ | heads>null | **DiffuLLaMA-7B** | Δ | heads>null |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| object→verb | −2 | 0.314 / 0.277 | **0.755** | +0.441 | 29/144 | **0.877** | +0.600 | 151/1024 |
| subject→verb | +1 | 0.429 / 0.433 | 0.521 | +0.092 | 6/144 | **0.660** | +0.227 | 46/1024 |
| object adj→noun | +1 | 0.750 | 0.827 | +0.077 | 1/144 | **0.942** | +0.192 | 13/1024 |
| subject adj→noun | +1 | 0.682 | 0.773 | +0.091 | 2/144 | **0.886** | +0.205 | 19/1024 |
| object det→noun | +1 | 0.553 / 0.558 | 0.600 | +0.047 | 3/144 | **0.895** | +0.337 | 42/1024 |
| subject det→noun | +1 | 0.600 | 0.589 | **−0.011** | 3/144 | **0.856** | +0.256 | 30/1024 |

(Where two null values are shown, they are the DiffuGPT-S and DiffuLLaMA-7B
splits respectively — the two runs used different sentence samples, so the null
differs slightly.)

**Read this table as: scale rescues the finding rather than threatening it.**
DiffuGPT-S clears the null on one relation out of six. DiffuLLaMA-7B clears it
on all six, and the four relations that look like positional artifacts in the
small model recruit 13–42 heads each in the large one.

For DiffuGPT-S, Wilson intervals confirm the weakness — four relations'
intervals contain the null:

- object adj→noun 0.827 [0.697, 0.918] vs null 0.750
- subject adj→noun 0.773 [0.622, 0.885] vs null 0.682
- object det→noun 0.600 [0.488, 0.705] vs null 0.553
- subject det→noun 0.589 [0.480, 0.692] vs null 0.600

### 4.2 The double dissociation (DiffuLLaMA-7B)

This is the single most publishable object in the project. Two heads, profiled
across all six relations:

| head | object→verb | the four adjacent relations | interpretation |
|---|--:|--:|---|
| **L18 H10** | 0.141 | 0.856 – 0.942 | **positional** — a `+1` heuristic |
| **L3 H11** | 0.877 | 0.000 – 0.023 | **relational** |

L18 H10 solves everything adjacent and nothing else. L3 H11 scores essentially
*zero* on the adjacent relations while solving object→verb — and no fixed
offset can produce that profile. The two heads are exact inverses of each
other.

`dlmrel analyze` computes `selectivity` (max relation minus second-best) and
`adjacency_bias` (adjacent mean minus distant mean) for every head, which is
how you find these two automatically rather than by hand.

### 4.3 Where scale costs something

Object→verb accuracy while **both endpoints are still masked**:

- DiffuGPT-S: **0.434**
- DiffuLLaMA-7B: **0.108** — *below* the null.

So final-state relation heads strengthen with scale while anticipation during
denoising appears to weaken. These are separable phenomena and must not be
reported as one finding.

**Caveat, and it is a large one:** these two numbers are **not
protocol-matched.** DiffuGPT-S used 5 seeds with a ≥25-masked-positions gate;
the 7B number is a single run. `dlmrel curve` fixes this by retaining
per-instance rows. Do not quote the 0.108 until the comparison is rerun. See
also open question 3 in §11 — the metric itself may be ill-posed.

---

## 5. State of play — done vs. not done

### Done

- [x] Full audit of the previous pipeline; three defects identified (baseline,
      sampling, sequence cap).
- [x] Misfiled 7B result recovered and re-scored.
- [x] Complete rewrite as an installable package (`dlmrel`, ~2,300 lines).
- [x] Fixed-offset null implemented and unit-tested.
- [x] Selection diagnostics (`n_heads_above_null`, rank correlation, Wilson and
      bootstrap CIs).
- [x] Three-way disjoint splits with deterministic reconstruction.
- [x] `scripts/verify_legacy.py` reproduces the §4.1 table for both models
      exactly from the old CSVs.
- [x] 21 unit tests passing; `ruff check` clean.

### Not done — nothing has been run against real data in this package

- [ ] **`dlmrel data` has never been run.** `results/` is empty. There are no
      new numbers yet — every number in §4 comes from the *old* runs, re-scored.
- [ ] The GPU stages have never executed. DiffuLLaMA-7B has not been loaded by
      this code.
- [ ] **Causal ablation is not implemented.** Reviewer B's "weak causal
      evidence" is not yet addressed anywhere in this package. Ablating the
      object→verb head across all held-out sentences, with matched-offset
      control heads, is the largest missing piece.
- [ ] **The masked-state metric is unfixed.** It still scores against the gold
      parse of the *original* sentence (§11, Q3).
- [ ] No decision on how the corrected tables fold back into the manuscript.

---

## 6. The codebase, module by module

```
src/dlmrel/
  config.py      132 lines  every knob that changes a reported number
  treebank.py    131        UD download, pooling, three disjoint splits
  alignment.py    80        UD words -> model sub-tokens via character offsets
  relations.py   228        gold relation extraction, with word distance
  nulls.py       168        the fixed-offset null model   <- the point
  stats.py       162        selection diagnostics, CIs, head profiling
  model.py       133        model loading, with the eager-attention guard
  diffusion.py   175        masking schedule and attention extraction
  scoring.py     238        head search and masked-state curve
  splits.py       54        deterministic split reconstruction for GPU stages
  cli.py         247        dlmrel data|nulls|search|curve|analyze
scripts/
  verify_legacy.py  101     reproduce the §4.1 table from the old result files
tests/
  test_nulls.py     120     12 tests
  test_relations.py 120      9 tests
configs/default.yaml        the run definition
```

### `config.py`

Dataclasses `TreebankConfig`, `ModelConfig`, `DiffusionConfig`,
`AnalysisConfig`, wrapped in `Config` with `.load()` / `.save()` over YAML.
`RELATION_NAMES` is the canonical tuple of six relations and the iteration
order everywhere. `SUBJECT_DEPS = {nsubj, csubj}`, `OBJECT_DEPS = {obj, iobj}`,
`NOUN_UPOS = {NOUN, PROPN}`, `VERB_UPOS = {VERB, AUX}`.

Every default here encodes an audit finding. Changing `max_seq_len`, `shuffle`,
`seed`, or `attender_token` changes reported numbers.

### `treebank.py`

Downloads CoNLL-U from `raw.githubusercontent.com/UniversalDependencies/<repo>`
and caches under `data/ud/`. **Deliberately merges UD's own train/dev/test into
one pool before splitting** — our splits are over *sentences sampled for
probing*, not over the original parser-training partition, and merging first
means all three probing splits are drawn from one homogeneous pool.

`split_sentences` carves three disjoint splits from a single shuffled pool
under a fixed seed, so a sentence can never appear in two splits. A `None`
budget means "everything left over".

A missing treebank split is a 404 and is expected (not every treebank ships
every split); it is caught and skipped rather than aborting the pool.

### `alignment.py`

Maps UD word indices onto model sub-token indices **through character
offsets**, not through tokenizer-specific word markers. This matters: the
previous DiffuLLaMA port special-cased SentencePiece's `▁` and therefore
diverged from the GPT-2 path it was supposed to replicate. Character overlap
serves both.

`find_char_spans` **fails closed** — returns `None` if any token form can't be
located in order, and the sentence is dropped rather than guessed at.
Multi-word tokens (`can't` → range id `2-3` plus two pieces) are detected and
those sentences skipped by default.

### `relations.py`

`RelationInstance` carries both token spans and, crucially,
`word_distance = receiver_word_idx - attender_word_idx`. **This is stored at
extraction time, not recovered later**, because it is what the null consumes
and what the distance-stratified analysis bins over.

`extract_relations` walks the sentence, resolves each token's head, and
classifies. Determiners and adjectives are assigned subject/object *role* by
looking at the governing noun's own deprel (`_noun_role`). UD subtypes are
stripped (`nsubj:pass` → `nsubj`) by `_base_dep`.

`build_example` returns `None` on any failure — multi-word tokens, missing
text metadata, character-span failure, sequence length outside
`[min_seq_len, max_seq_len]`, incomplete alignment, or no relations found.
`build_examples` prints how many sentences dropped and the instance counts per
relation, which is your first sanity check on any run.

### `nulls.py` — the centerpiece

- `fit_offset_null(df, relation, offset_range=(-15,15))` — grid-searches k,
  **excluding k=0** (self is masked before the argmax, so the head can never
  predict itself; the null must not be allowed to either).
- `offset_accuracy` / `offset_correctness` — aggregate and per-instance.
- `build_null_table(select_df, test_df, ...)` — fit on `select`, report on
  `test`, with a percentile bootstrap CI.
- `offset_distribution` — the diagnostic that explains *why* a relation does or
  does not survive.
- `_as_list` — span columns survive a CSV round trip as strings; this accepts
  either form. If you add a new code path that reads spans from disk, route it
  through here.

The anchor used by the null is **the same token the attention row is read
from** (`attender_span[-1]` when `attender_token="last"`). If those ever
diverge, the null stops measuring the same thing as the head and the whole
comparison is void.

### `stats.py`

`n_heads_above_null`, `selection_rank_correlation`, `wilson_ci`,
`build_head_vs_null_table` (§3.2 verdict rule), `positional_selectivity` (§4.2
head profiling), `stratify_by_distance` (accuracy as a function of dependency
distance — the experiment that turns the double dissociation from an anecdote
about two heads into a curve).

### `model.py`

Clones `HKUNLP/DiffuLLaMA` into `third_party/` at runtime (it is not on PyPI)
for its `DiscreteDiffusionModel` wrapper and `get_anneal_attn_mask`. Loads a
`LlamaForCausalLM` or `GPT2LMHeadModel` backbone, wraps it, and then runs
`_assert_attentions_returned` — a single forward pass that **raises at load
time** if attentions come back `None`. See §9.

### `diffusion.py`

`state_at_time` rebuilds x_t at a chosen timestep. The schedule is
**teacher-forced**: rather than letting the model generate, the true sentence
is progressively revealed with per-step probability `1 / (steps - progress)`.
That reproduces the marginal masking rate seen in training while keeping the
gold parse valid for visible tokens — which is what makes the masked-state
measurement well defined at all. **The schedule is reproduced verbatim from the
original notebooks so numbers stay comparable; do not "clean it up".**

`t=0` is fully masked; `t = steps-1` is forced fully unmasked so the head
search is scored on a complete sentence.

`receiver_predictions` sets BOS and every attender-span column to `-inf`
**before** the argmax. Order matters — masking after the argmax is a different
and wrong measurement.

### `scoring.py`

`score_split` — accuracy of every (layer, head) on every relation at the final
frame. `merge_splits` — joins per-split tables into the wide form analysis
expects (`accuracy_select` / `accuracy_test` / `accuracy_dev`).

`masked_state_curve` returns **one row per (relation, seed, timestep,
instance)** — deliberately *not* a pre-aggregated curve. Keeping raw rows is
what lets the offset null be recomputed over exactly the instances that
contributed to the masked-state average, which is the mismatch that made the
old 43.4%-vs-31.4% comparison approximate.

`aggregate_curve` applies the `min_masked` gate. That gate exists because late
in denoising almost nothing is masked, so "both endpoints masked" selects a
vanishing, unrepresentative set. It is applied to the masked statistic only.

### `splits.py`

Example objects hold tokenizer-dependent spans, so they are **rebuilt rather
than serialised** between the CPU and GPU stages. `examples_for_split` rebuilds
deterministically and then asserts the result against the
`sentences_<split>.csv` manifest written by `dlmrel data`. If the config
drifted, it raises instead of silently scoring different sentences.

---

## 7. Running it

### Install

```bash
cd <this folder>
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # CPU: data, nulls, analysis, tests
pip install -e ".[gpu]"     # adds torch/transformers for the head search
```

`transformers` is **pinned to 4.44.2** under `[gpu]` — later versions changed
the attention-implementation plumbing that `output_attentions` depends on.

### Pipeline

```bash
dlmrel data     --config configs/default.yaml   # CPU  ~2 min, downloads UD-EWT
dlmrel nulls    --config configs/default.yaml   # CPU  seconds
dlmrel search   --config configs/default.yaml   # GPU  the long pole
dlmrel curve    --config configs/default.yaml   # GPU
dlmrel analyze  --config configs/default.yaml   # CPU
```

`--out` overrides `out_dir` if you want to keep runs side by side.

**Run `data` and `nulls` first and actually read the output.** They are free,
and they tell you whether the experiment is worth GPU time: if the nulls under
random full-treebank sampling come out very different from §4.1, every
downstream claim moves with them.

### Tests

```bash
pytest -q          # 21 tests
ruff check .
```

### Verifying against the old runs

```bash
python scripts/verify_legacy.py \
    --diffugpt   ~/Downloads/_udseed \
    --diffullama ~/Downloads/relation_head_search_ud_gold_diffullama
```

This must keep reproducing §4.1. If it drifts, a definition changed somewhere
and the comparison to prior results is no longer valid.

### Current config (`configs/default.yaml`)

DiffuLLaMA-7B (`diffusionfamily/diffullama`), UD_English-EWT, splits
4000/1000/1000, `max_seq_len 128`, `steps 64`, seeds `[42..46]`,
`min_masked_positions 25`, offset range `[-15, 15]`, 10,000 bootstrap
resamples, `out_dir: results/diffullama-ewt`.

---

## 8. Output files and what each column means

All under `out_dir`.

| file | written by | contents |
|---|---|---|
| `config.yaml` | `data` | the exact config that produced this run |
| `sentences_{select,dev,test}.csv` | `data` | split manifests; `splits.py` asserts against these |
| `relation_instances.csv` | `data` | every gold relation instance — the audit trail |
| `offset_null.csv` | `nulls` | k, fit accuracy, test accuracy, bootstrap CI per relation |
| `head_scores_{select,dev,test}.csv` | `search` | accuracy of every (layer, head) × relation |
| `head_scores_merged.csv` | `search` | the wide join everything downstream reads |
| `model_meta.json` | `search` | layers, heads, hidden size, token ids |
| `curve_raw.csv` | `curve` | **per-instance** correctness across diffusion time |
| `curve_aggregate.csv` | `curve` | masked/unmasked means ± std across seeds |
| `head_vs_null.csv` | `analyze` | **the headline table** |
| `head_profiles.csv` | `analyze` | per-head selectivity and adjacency bias |
| `masked_state_vs_null.csv` | `analyze` | masked-state accuracy vs null on matched instances |

Key columns:

- `relation_instances.csv`: `split, sentence_idx, sentence, source, relation,
  attender_text, receiver_text, dep, attender_span, receiver_span,
  attender_word_idx, receiver_word_idx, word_distance`. Spans are lists that
  round-trip as strings — read them through `nulls._as_list`.
- `head_vs_null.csv`: `head_test_acc`, `head_ci_lo/hi` (Wilson),
  `null_test_acc`, `delta`, `n_heads_above_null`, `selection_rho`, `verdict`.
- `curve_raw.csv`: `correct`, `both_endpoints_masked`, `n_masked`,
  `word_distance`, plus spans — everything needed to recompute any baseline on
  matched instances.

---

## 9. Gotchas that will silently ruin a run

Each of these produces **wrong numbers without raising an error**. They are
listed in order of how much damage they do.

1. **Non-eager attention.** `sdpa` and `flash_attention_2` accept
   `output_attentions=True` and return nothing. Every accuracy becomes zero and
   nothing complains. `model.py` asserts against this at load time — do not
   remove that check, and do not "optimise" the attention implementation.
2. **Comparing against uniform chance.** ~6% is meaningless here. See §3.
3. **BOS.** Diffusion LMs park enormous attention mass on position 0 (the
   attention sink). Without excluding it, every head "predicts" BOS. Exclude
   before the argmax, never after.
4. **`k = 0` in the null.** Self is masked for the head, so the null must not be
   allowed to predict self either. `fit_offset_null` excludes it.
5. **`df.head`.** In pandas, `df.head` is the *method*, not the `head` column.
   `m[(m.layer==18) & (m.head==10)]` silently returns an empty frame. Always
   `m['head'] == 10`. This nearly caused the double dissociation to be missed.
6. **The `min_masked_positions` gate.** Without it, "both endpoints masked" late
   in denoising selects a tiny unrepresentative set. Apply to the masked
   statistic only, never to the unmasked one.
7. **Anchor mismatch.** The null's anchor must be the same sub-token the
   attention row is read from. Changing `attender_token` changes both, together
   — do not change one in isolation.
8. **Config drift between stages.** The GPU stages rebuild splits from config.
   Change any treebank setting after `dlmrel data` and `splits.py` will raise —
   that is the guard working. Rerun `data`.
9. **Determiner→noun is not always adjacent.** "The old man" puts it at
   distance 2. A unit test pins this, because it explains why the det→noun null
   sits near 0.55–0.60 rather than saturating.
10. **Memory.** Eager attention with `output_attentions=True` materialises a
    full `[batch, heads, seq, seq]` tensor per layer. For 32 layers at
    `seq_len=128` this is fine, but it is far heavier than sdpa — check the
    batch size that fits before starting a long run.

---

## 10. External data and where it lives

These paths are **outside this folder** and are not copied in. If you are
relocating the project, copy them alongside it or `verify_legacy.py` will stop
working.

| what | path | notes |
|---|---|---|
| DiffuGPT-S UD results | `~/Downloads/_udseed/` | `ud_gold_relation_instances.csv` (4859 instances, 1187 sentences, 987 train / 200 eval), `ud_top_relation_heads.csv`, `ud_relation_accuracy_masked_vs_unmasked_seeds.csv` |
| **DiffuLLaMA-7B UD results** | `~/Downloads/relation_head_search_ud_gold_diffullama/` | **the misfiled run — the strongest result in the project.** Treat as precious until reproduced. |
| submitted manuscript | `~/Downloads/_jr10/blackboxnlp-2026/manuscript.tex` | 539 lines |
| old notebooks | `~/SAE4DLM-CE/dlm_order/` | superseded; sequential-sampling defect is in the `collect_ud_examples` cell |
| templated corpus | `~/SAE4DLM-CE/dlm_order/data/` | **retired — see §2.3** |

`verify_legacy.py` expects each results directory to contain
`ud_gold_relation_instances.csv` and `ud_relation_head_scores_merged.csv`.

UD-EWT itself is downloaded on demand into `data/ud/` — nothing to preserve.

---

## 11. Open questions and the work queue

### The three open scientific questions

1. **Does the double dissociation hold as a *curve* over dependency distance,**
   rather than as two hand-picked heads? A fixed-offset head's accuracy should
   collapse the moment distance leaves its offset; a relational head's should
   not. `stratify_by_distance` is implemented and unused. This is the cheapest
   way to turn §4.2 from an anecdote into a result.

2. **Does masked-state anticipation really vanish at scale,** under a
   protocol-matched comparison? Currently 0.434 (small) vs 0.108 (7B), but the
   protocols differ. `dlmrel curve` + `masked_state_vs_null.csv` answer this.

3. **Is the masked-state metric well posed at all?** It scores against the gold
   parse of the *original* sentence. A model that denoises to a different but
   perfectly grammatical sentence is penalised for being right about its own
   output. The fix — parse the model's actual completion and score against
   that — is **not implemented**, and until it is, the 0.108 may be measuring
   the wrong thing entirely rather than a real failure of scale.

### Work queue, in the order I would do it

1. `dlmrel data` + `dlmrel nulls` on full random-split EWT. **Free, and it
   gates everything.** Specifically: do the thin relations grow? Subject
   adj→noun had 44 eval instances in the pilot; at ~6,000 sentences instead of
   ~1,200 it should reach several hundred, and the CIs in §4.1 that currently
   swallow the null may separate.
2. `dlmrel search` on DiffuLLaMA-7B. The long pole.
3. `dlmrel analyze` → confirm the double dissociation reproduces on data the
   pilot never saw.
4. Distance-stratified curves (Q1).
5. `dlmrel curve` → protocol-matched masked-state comparison (Q2).
6. **Implement the causal ablation.** Not in this package. Ablate the
   object→verb head across all held-out sentences and measure the effect on
   the dependent token's probability, with **matched-offset control heads** —
   i.e. ablate a positional head with the same average attention displacement,
   so the effect is attributable to the relation and not to removing attention
   mass generally. This is the direct answer to Reviewer B.
7. Fix the masked-state metric (Q3).
8. Consider a second treebank (GUM is the obvious next one — different genres,
   already supported by `treebank.py`'s `_STEMS`) to show the finding is not an
   EWT artifact.

---

## 12. Publication context

- Rejected from **Sci-FM @ COLM 2026** (§2.1).
- Target under discussion: a **NeurIPS 2026 workshop**, with **LP4FM**
  recommended; deadline **29 August 2026, AoE**. Confirm this date
  independently before planning around it.
- Note that a **public** repo conflicts with double-blind review until
  camera-ready. The repo was private for this reason before being deleted.
- The manuscript at `~/Downloads/_jr10/blackboxnlp-2026/manuscript.tex` still
  contains the uniform-chance framing throughout — abstract, results, and
  discussion. **Any reuse of it requires re-scoring against the offset null
  first.** Numbers citing a "6%" or "6.1%" chance baseline should not be
  carried over in any form.

The honest framing for a resubmission, given §4: the story is no longer "a
small DLM tracks syntax." It is "most apparent relation-tracking in a small DLM
is positional, and only object→verb survives a proper null — but at 7B scale
the effect is real across all six relations, and positional and relational
heads doubly dissociate." That is a stronger paper *and* it absorbs both
reviewers' objections (single small model; weak evidence) rather than arguing
with them.

---

## 13. Glossary

| term | meaning |
|---|---|
| **attender → receiver** | direction of the studied relation; the attention row is read *from* the attender and should land *on* the receiver |
| **fixed-offset null** | `receiver = attender + k`, k fit on the selection split, reported held-out. §3 |
| **positional head** | high on adjacent relations, near zero on distant ones — implements an offset heuristic |
| **relational head** | high on one relation and near zero elsewhere; no fixed offset can produce this profile |
| **double dissociation** | one head positional, another the exact inverse. §4.2 |
| **masked state** | a denoising frame where *both* endpoints of a relation are still `[MASK]` |
| **`min_masked_positions`** | gate (default 25) excluding late frames where almost nothing is masked |
| **select / dev / test** | disjoint sentence splits: search heads / tune / report |
| **`n_heads_above_null`** | how many searched heads clear the null — the multiple-comparisons diagnostic |
| **teacher-forced reveal** | the true sentence is progressively unmasked rather than generated, so the gold parse stays valid |
| **attention sink** | the enormous attention mass DLMs put on BOS; must be excluded before the argmax |
| **DiffuGPT-S** | 12 layers × 12 heads = 144 heads; the pilot model |
| **DiffuLLaMA-7B** | 32 layers × 32 heads = 1024 heads; `diffusionfamily/diffullama` |
