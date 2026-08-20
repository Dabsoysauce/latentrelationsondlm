# DiffuLLaMA-7B: POS probe, logit lens, attention entropy, and external transfer

The five experiments DiffuLLaMA still needed after head search and the EWT time
curve. All five completed and passed `dlmrel validate` with no errors.

Curated outputs are under `docs/results/diffullama_remaining/`. Full run
directories are not committed: `results/*/*/*/*/*/` is gitignored by design, and
these runs carry 9.6GB of resume checkpoints plus a 461MB per-position record
for attention entropy. `metrics.csv` and `per_seed_metrics.csv` are the
reportable aggregates; rerun an experiment to regenerate the rest.

## Runs

| Experiment | Dataset | Run ID | Test sentences | Excluded |
| --- | --- | --- | ---: | ---: |
| `masked_pos_probe` | ewt | `diffullama-ewt-posprobe-v1` | 727 | 268 |
| `rank_logit_lens` | ewt | `diffullama-ewt-logitlens-v1` | 732 | 268 |
| `attention_entropy_over_time` | ewt | `diffullama-ewt-entropy-v1` | 732 | 268 |
| `ewt_locked_transfer` | de_gsd | `diffullama-de-transfer-v1` | 586 | 336 |
| `ewt_locked_transfer` | ja_gsd | `diffullama-ja-transfer-v1` | 359 | 178 |

Exclusions are all `tokenization_alignment_or_relation_filter`, which merges
alignment failure with "sentence contains no target relation" and so cannot
separate the two. Japanese lost 33% against German's 36%, so SentencePiece
alignment on unsegmented Japanese was not the problem it was expected to be.

Model load was verified first: attention row sums were within 2.87e-3 of one
against a bfloat16 tolerance of 7.8e-3, and two identical forward passes agreed
exactly. See `smoke_test.json`.

## External transfer is the substantive result

Both transfer runs replay the EWT-locked head, layer 17 head 15, frozen at
`selection_progress` 0.0 with a fixed-offset null of -1.

German, `object_to_verb`, 415 instances across 3 seeds:

| predictor | accuracy |
| --- | ---: |
| locked head | 0.183 [0.146, 0.224] |
| next token | 0.219 |
| nearest | 0.125 |
| fixed offset | 0.118 |
| uniform | 0.067 |
| oracle POS | 0.877 |

Japanese, `object_to_verb`, 301 instances across 3 seeds:

| predictor | accuracy |
| --- | ---: |
| locked head | 0.010 [0.000, 0.021] |
| uniform | 0.048 |
| fixed offset / next / previous / nearest | 0.000 |
| oracle POS | 0.821 |

The head clears the fixed-offset null on German but loses to a next-token
predictor, and on `subject_to_verb` it loses to next-token outside its interval
(0.153 [0.128, 0.179] against 0.217). On Japanese it scores below uniform.

In both languages `p_gold_mass_greater` is about 0.50: the gold receiver carries
more attention mass than a matched control on half of instances. German's
`mean_gold_mass` of 0.051 against a matched 0.012 is carried by a minority of
instances and does not survive the per-instance comparison.

### Reading

This is the pattern of a word-order heuristic rather than a relation. German
shares broad constituent order with English and transfers partially, though not
past a next-token baseline. Japanese is head-final, which puts the verb far from
the object; there the head collapses below chance while every adjacency baseline
goes to exactly 0.000 for the same structural reason. Oracle POS stays above 0.82
in both, so the relation is recoverable from these sentences and this head is
simply not recovering it.

This is consistent with how the head was chosen. The lock freezes
`selection_progress` at 0.0, where every token is `[MASK]` and attention can
only be a function of position, so the selected head is a fixed-offset predictor
by construction and transfers as far as English word order does.

That reading is inferred from the lock file and the metrics tables, not from the
selection code, and should be confirmed against `relation_selection.py` before
it is treated as settled.

## Masked POS probe

| quantity | value |
| --- | ---: |
| probe accuracy | 0.552 (seed sd 0.017) |
| macro F1 | 0.435 |
| lexical baseline | 0.864 |
| majority baseline | 0.161 |
| shuffled-label control | 0.104 |
| random-feature control | 0.096 |

Well clear of both controls, so the hidden states carry real part-of-speech
information. The lexical baseline is higher, but it reads word forms that the
model cannot see at a masked position, so the two are not measuring the same
thing. How many probed positions were actually masked at `normalized_progress`
0.5 is not recorded in the metrics, and that determines whether the comparison
is meaningful at all.

## Logit lens

Top-1 recovery of masked tokens peaks at 0.048, at depth 32 and
`normalized_progress` 0.25; top-5 reaches at most 0.245 anywhere in the sweep.
Depths 0 through 29 stay at or below 0.043, so most of the recovery appears in
the final three layers. At `normalized_progress` 0.0, where the whole sequence
is masked, depth 31 still reaches 0.030, which is the positional and unigram
prior rather than content.

## Attention entropy

`bos_sink_mass` runs from 0.679 at both ends of the trajectory down to 0.540 at
the middle: between 54% and 68% of all attention mass sits on BOS. Normalized
entropy is close to flat at about 0.40 throughout.

Head search excludes the BOS column before taking its argmax, so the analysis
operates on the remaining third of the distribution. Any writeup of the head
results should say so explicitly.
