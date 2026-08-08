# Camera-ready revision notes — COLM 2026 NonAR-LM

Source: copied from `~/Downloads/_jr10/colm-2026/`, which is the tree that
produced the accepted PDF. Working copy is `camera-ready/colm-2026/`.

Accepted with **6 / 7 / 6** (confidence 3 / 3 / 2). Reviewer dZ2J (7) raised no
weaknesses. Everything below responds to hwdx (6) and kRnT (6).

---

## Reviewer response map

| # | Reviewer point | Status | What changed |
|---|---|---|---|
| hwdx-1 | Single small model, single English treebank; generalization untested | **Addressed in limitations** | Limitations rewritten to name the constraint explicitly and to say what would resolve it. The DiffuLLaMA-7B replication was deliberately **not** added — see "Deliberately not done". |
| hwdx-2 | No comparison to GPT-2 heads or from-scratch masked diffusion; behavior may be inherited from AR pretraining | **Addressed in limitations** | New sentence stating that because DiffuGPT-S is initialized from GPT-2 we cannot separate diffusion-trained structure from inherited structure, and naming the two comparisons that would. |
| hwdx-3 | Oracle nearest-POS beats the heads in **every row** of Table 2; isolate what heads add beyond POS+locality | **Fixed — the main change** | See below. |
| hwdx-4 | Thin statistics — no CIs or multi-seed aggregates | **Partly fixed** | Wilson score intervals added to every row of Table 2. Multi-seed reporting for the timestep figure is **unresolved** — see "Blockers". |
| hwdx-5 | The specific UD treebank is never named | **Fixed** | Named as UD\_English-EWT in the abstract, introduction, §4.3 and §5.2, with a note that sentences were taken in corpus order from the train and dev files. |
| hwdx-6 | Broken citation `?` in two places | **Already fixed** | The Dai et al. (2026) bib entry is present in this tree; the June PDF predates it. Verified against the Aug 5 rebuild. |
| kRnT | Claims stronger than the evidence, especially causal | **Fixed** | Every causal claim now states that it rests on three head–sentence cases. Abstract, introduction, §5.2 and the appendix all softened; the appendix now says explicitly that three hand-inspected cases cannot establish the effect in general. |

---

## The main change: Table 2

**The problem.** The old `Near tok.` column scored **0.000** on object
determiner→noun. That is not a coincidence — "closest valid token" breaks ties
toward the *preceding* token, and a determiner's noun is always the *following*
word. So the locality control was guaranteed to fail on exactly the relations
where locality is the whole story, which made the heads look far better than
they are. A referee who spotted the 0.000 would have drawn the same conclusion.

**The fix.** `Near tok.` is replaced by `Offset*`: the best fixed-offset
predictor `r̂ = a + k`, with a single `k` fit on the training split and reported
held-out, so the baseline gets exactly the tuning the head selection gets
(`k = −2` for object→verb, `k = +1` for all others). Wilson score intervals are
reported on every head. A relation counts as evidenced only when its interval
lies entirely above `Offset*`.

| relation | head | 95% CI | Random | Offset* | Oracle POS | verdict |
|---|--:|--:|--:|--:|--:|---|
| object→verb | 0.755 | [.694, .807] | 0.061 | 0.314 | 0.845 | **clears** |
| subject→verb | 0.521 | [.465, .577] | 0.059 | 0.429 | 0.581 | **clears** |
| obj adj→noun | 0.827 | [.703, .906] | 0.041 | 0.750 | 0.923 | not distinguishable |
| subj adj→noun | 0.682 | [.534, .800] | 0.047 | 0.682 | 0.841 | not distinguishable |
| obj det→noun | 0.600 | [.494, .698] | 0.057 | 0.553 | 0.765 | not distinguishable |
| subj det→noun | 0.589 | [.486, .685] | 0.052 | 0.600 | 0.789 | not distinguishable |

**Consequence.** Four of six relations are demoted in print. The abstract,
introduction and discussion no longer claim adjective→noun or determiner→noun
as relation heads.

**Why this is a better paper, not a weaker one.** hwdx's objection was that
POS+locality might explain the whole result. The camera-ready now concedes that
at the final step it largely does for four relations — and then makes the
argument that survives it: the **oracle POS baseline cannot run during
denoising**, because it has to read parts of speech that masked endpoints do not
have. The 41.6% masked-state result is therefore not subject to the criticism at
all. Object→verb is the one relation no fixed offset and no POS-locality rule
solves, and it is now unambiguously the paper's central case. This is also where
the text was already heading ("emphasizing object→verb as the strongest
nontrivial case") — the revision commits to it.

A formal definition of `Offset*` and the reason `k = 0` is excluded were added
to §4.3.

---

## Blockers before submission

1. **`AFFILIATION-TODO` / `EMAIL-TODO`** in the author block. Names, order and
   the equal-contribution footnote are final; affiliations are not known here.
   If the four authors are not all at one institution the block needs splitting
   with `\And`.
2. **Acknowledgments** — the `\subsubsection*{Acknowledgments}` stub at the end
   of the file is still empty (this TODO predates the camera-ready work).
3. **The 41.6% vs 43.4% discrepancy.** The timestep numbers in the paper
   (41.6% masked, 63.9% unmasked) do not match the five-seed sweep in
   `ud_relation_accuracy_masked_vs_unmasked_seeds.csv` (43.4% ± 3.3%,
   61.7% ± 1.9%). Two different protocols are in circulation and only one can
   ship. This is flagged in a `TODO(authors)` comment above the figure in the
   `.tex`. It is also the remaining half of hwdx-4: reporting mean ± std over
   seeds would close that weakness properly.

---

## Deliberately not done

- **DiffuLLaMA-7B results are not included.** They would answer hwdx-1
  directly, but they came from the pipeline with the sequential-sampling defect
  (sentences taken in genre order) and have not been re-run. Publishing them
  would put numbers into a proceedings that we already know need re-scoring.
  Scale is cited as future work instead.
- **No new experiments.** Everything in this revision is derived from data
  already collected; the Wilson intervals and the offset baseline are computed
  from the existing instance and score files.

---

## Build

The tree keeps the original layout, so `\bibliography{colm-2026/references}`
resolves — compile from `camera-ready/`, not from inside `colm-2026/`.

Verified: no LaTeX errors from these edits, and the Table 2 overfull box
introduced by the CI column was fixed by compacting the interval notation and
reducing `\tabcolsep` to 3pt. `tectonic` cannot embed most of the matplotlib
figure PDFs — the unmodified original fails on the same 60 figures, so this is a
local toolchain limitation, not a source problem. Build on Overleaf as before.
