# Frozen research protocol

## Scope

The confirmatory English dataset is UD English EWT 2.15. German GSD and
Japanese GSD are locked multilingual transfers. The research models are
Dream-7B and DiffuLLaMA-7B; the fake model exists only for CPU tests.

The primary analysis is object-to-verb receiver prediction while both words
are masked. Five predefined relation-selection secondaries are
subject-to-verb, object-adjective-to-noun, subject-adjective-to-noun,
object-determiner-to-noun, and subject-determiner-to-noun. Other secondary
analyses are the locked-head time curve, attention entropy, logit lens, and
masked POS probe. DLA, head ablation, native-generation
timing, GPT-2, LLaDA, DiffuGPT, and the extra English treebanks are deferred and
are not represented as supported results.

## Data freeze

Each dataset YAML pins the UD repository commit and the SHA-256 of train, dev,
and test files. EWT train supplies selection sentences, EWT dev chooses among
the select-set top five heads, and EWT test evaluates the one locked head.
German and Japanese keep their official boundaries and receive the exact EWT
lock without reselection. Normalized duplicate text is removed across roles,
and tokenizer failures are recorded rather than replaced.

## Scoring and selection

Dependencies are directed from the dependent query word to its syntactic head.
Attention rows are averaged over query subtokens and receiver mass is summed
over receiver subtokens. Special tokens, self positions, and invalid alignments
cannot become receiver candidates.

Head selection is performed independently for each of the six canonical
relations. A head must have at least 25 relation rows to be eligible. Within a
relation, select ranking is accuracy descending, denominator descending, layer
ascending, then head ascending. Only the top five eligible select heads enter
development evaluation, where the same ordering chooses the lock; a dev head
outside that gate cannot win. Insufficient relations receive a documented
status and no forced lock. Three-seed evidence and select/dev rank stability
remain in each relation's candidate tables and lock.

Only the primary `object_to_verb` lock is exposed to EWT test or the existing
time-curve and transfer paths. Test outcomes cannot affect any relation lock.
Selection-aware permutation p-values are computed per relation; the primary is
reported separately and Holm correction covers the fixed family of five
secondaries.

## Controls and uncertainty

The locked head is compared with a select-fit fixed offset, uniform receiver,
nearest receiver, previous and next word, oracle receiver POS, wrong same-POS
word, and a deterministic matched alternative. The primary interval is a
sentence-clustered bootstrap, so repeated relation and seed rows from one
sentence are never treated as independent sentences. Seed summaries remain
separate, selection-aware permutations repeat select-plus-dev selection, and
Holm correction is the default if a reported family contains multiple tests.

Every active experiment uses exactly three stochastic seeds: `42`, `43`, and
`44`. The POS probe is fit, tuned on development data, and evaluated
independently for each seed before its seed-level metrics are summarized.

Scientific configuration identity includes the pinned model and revisions,
dataset, manifest hashes, experiment, seeds, progress points, scoring, and the
scientific contents of any source selection lock. Runtime paths, run IDs,
`resume`, and `dry_run` are operational metadata and do not alter that
identity. Long GPU loops write an atomic checkpoint after every 300 input
sentences; a checkpoint is reusable only when its scientific configuration,
manifests, stage, seed, time point, head selection, and sentence range match.

## Outputs and claims

Each head-search run contains an immutable `relation-selection/` bundle with
six relation records, per-relation candidate tables, successful locks,
permutation results, source hashes, copied config, metadata, summary, and
validation. Its primary alias is byte-equivalent to the legacy
`selection_lock.json`, so downstream code remains compatible. Each run also
contains `config.resolved.yaml`, `command.txt`, `environment.json`,
`manifest_refs.json`, `selection_lock.json` when applicable,
`instances.parquet`, `exclusions.parquet`, `per_seed_metrics.csv`,
`metrics.csv`, `summary.json`, `validation.json`, and resumable checkpoints.
Attention, logit-lens, and probe findings are correlational. No causal,
cross-model, or multilingual claim is made until its corresponding real GPU
run is complete and validates successfully.
