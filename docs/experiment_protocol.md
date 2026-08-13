# Frozen experiment protocol

## Tracks and hypotheses

`legacy_reproduction` preserves versioned DiffuGPT quirks but is not a default.
`confirmatory_ewt` selects on official EWT train, chooses among select top-five
heads on official dev, writes an immutable lock, and evaluates that one head on
official test. The primary hypothesis is object-to-verb receiver prediction
while both whole-word endpoints are masked. `external_treebank_transfer`
applies that identical EWT lock to GUM, LinES, and ParTUT test without
reselection. `exploratory_extensions` contains local reselection, structural
slices, generated trajectories, and German/Japanese stress tests.

## Data and selection freeze

UD release 2.15 files are pinned by repository commit and SHA-256. Stable IDs
contain treebank, original split, UD sent ID, and normalized-text hash. Select,
dev, and test are sampled independently from official train, dev, and test;
normalized text is deduplicated across roles. Tokenizer rejection never causes
replacement in the base manifest. Model-specific eligibility and exclusions
are joined by stable instance ID. Primary cross-model summaries use the common
valid instance intersection; full eligible results are secondary.

All heads are scored on select. The top five by receiver-span top-1 accuracy
enter dev; dev accuracy, denominator, layer, then head form the deterministic
decision. The lock stores model/config/manifest/candidate hashes. Test code
loads only that head. All-head test ranking is prohibited.

## Relations and candidates

Direction is dependent query/attender to syntactic head receiver. Relations are
object/subject to verb and adjective/determiner to subject/object noun. UD
subtypes are preserved; passive subjects are labeled structurally. Candidate
receivers are word spans. Confirmatory scoring averages attender subtoken rows
and sums receiver-span mass after excluding special, padding, EOS, and attender
positions. Legacy uses the last attender subtoken. Structural records include
signed distance, direction, punctuation, clause depth/embedding, coordination,
relative clause, passive voice, intervening nouns/verbs, word/BPE lengths, and
alignment diagnostics.

## Trajectories and visibility

Teacher-forced gold trajectories and native generated trajectories are never
mixed. The shared reveal schedule is an analysis intervention. Five seeds
`[42,43,44,45,46]` and normalized progress
`[0,.125,.25,.375,.5,.625,.75,.875,1]` are defaults. Every raw row retains
native step, normalized progress, seed, sentence/instance/treebank/head, and one
of `both_masked`, `attender_visible_only`, `receiver_visible_only`, or
`both_visible`. A masked denominator counts eligible relation instances at the
timestep, not masked tokens in a sentence.

## Controls, statistics, and claim language

Controls frozen without test outcomes are uniform valid receiver, nearest,
previous/next, select-fit fixed offset, oracle receiver POS, wrong same-POS,
selection-aware permuted labels, and a matched alternative. Matching relaxes
punctuation/BPE/length, then distance, using deterministic levels; unmatched
rows remain unmatched. Primary matched evidence is
`P(mass_gold > mass_alternative)`, paired mass difference, and ties.

Sentences are bootstrap clusters and seed repeats are hierarchical, not extra
sentences. Report per-seed mean/SD, 95% clustered intervals, raw denominators,
exclusions, paired effects, and a selection-aware permutation p-value.
Predefined secondary families use Holm correction (BH is a documented
sensitivity). Attention, decoding, and logit lens are correlational. Causal
necessity requires matched head ablation. Cross-model/treebank/language claims
require their corresponding locked-transfer controls to pass.
