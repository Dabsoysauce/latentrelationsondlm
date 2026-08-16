# Frozen research protocol

## Scope

The confirmatory English dataset is UD English EWT 2.15. German GSD and
Japanese GSD are locked multilingual transfers. The research models are
Dream-7B and DiffuLLaMA-7B; the fake model exists only for CPU tests.

The primary analysis is object-to-verb receiver prediction while both words
are masked. Secondary analyses are the locked-head time curve, attention
entropy, logit lens, and masked POS probe. DLA, head ablation, native-generation
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

All heads are ranked on EWT selection data. Only the top five enter development
evaluation; development accuracy, denominator, layer, then head provide the
fixed tie-break. The resulting lock stores model, configuration, manifest, and
candidate hashes. Test and transfer code load only the locked layer and head.

## Controls and uncertainty

The locked head is compared with a select-fit fixed offset, uniform receiver,
nearest receiver, previous and next word, oracle receiver POS, wrong same-POS
word, and a deterministic matched alternative. The primary interval is a
sentence-clustered bootstrap, so repeated relation and seed rows from one
sentence are never treated as independent sentences. Seed summaries remain
separate, selection-aware permutations repeat select-plus-dev selection, and
Holm correction is the default if a reported family contains multiple tests.

Scientific configuration identity includes the pinned model and revisions,
dataset, manifest hashes, experiment, seeds, progress points, scoring, and the
scientific contents of any source selection lock. Runtime paths, run IDs,
`resume`, and `dry_run` are operational metadata and do not alter that
identity. Long GPU loops write an atomic checkpoint after every 300 input
sentences; a checkpoint is reusable only when its scientific configuration,
manifests, stage, seed, time point, head selection, and sentence range match.

## Outputs and claims

Each run contains `config.resolved.yaml`, `command.txt`, `environment.json`,
`manifest_refs.json`, `selection_lock.json` when applicable,
`instances.parquet`, `exclusions.parquet`, `per_seed_metrics.csv`,
`metrics.csv`, `summary.json`, `validation.json`, and resumable checkpoints.
Attention, logit-lens, and probe findings are correlational. No causal,
cross-model, or multilingual claim is made until its corresponding real GPU
run is complete and validates successfully.
