# Latent relations in diffusion language models

This repository reruns the Dream-7B and DiffuLLaMA-7B experiments with one
auditable protocol. The active datasets are English EWT, German GSD, and
Japanese GSD. The active analyses are head search, locked transfer, time
curves, attention entropy, logit lens, and masked POS probing.

The earlier Dream and DiffuLLaMA outputs are preliminary because they were
produced before official split preservation, select/dev/test head locking, and
the current controls and provenance records. They are preserved outside the
active branch and will be rerun; they are not used as final evidence. DiffuGPT
belongs to older work and is not part of this rerun.

## Install and verify

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Dream and DiffuLLaMA require separate GPU environments because they use
different pinned Transformers versions. See [RUNNING.md](docs/RUNNING.md).

## Commands

```powershell
dlmrel prepare --dataset configs/datasets/ewt.yaml
dlmrel smoke-test --model configs/models/dream_7b.yaml
dlmrel run --model configs/models/dream_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/head_search.yaml --run-id dream-ewt-v1
dlmrel derive-relation-locks --source-run <completed-head-search-run> --output <new-derived-directory>
dlmrel validate --run-dir results/confirmatory_ewt/dream_7b/ewt/confirmatory_head_search/dream-ewt-v1
dlmrel compare --runs <dream-run> <diffullama-run> --output results/comparison.csv
```

Every real run records its exact model and dataset revisions, configuration,
command, environment, manifests, exclusions, raw rows, seed summaries, and
validation result. Completed runs cannot be silently overwritten.

Head search creates independent select/dev locks for the six canonical
relations. `object_to_verb` remains the primary confirmatory relation; the
other five are predefined secondaries. New test and downstream runs resolve
each relation to its own lock, while legacy `selection_lock.json` files remain
object-only. A completed legacy all-head select/dev run can be converted with
`derive-relation-locks`; that command performs no model inference and does not
read or modify locked-test artifacts. Existing object-head-only test rows are
not six-relation results and cannot supply the corrected permutation test.

## Repository map

```text
configs/      Three datasets, two research models, six analyses
notebooks/    Colab GPU launcher
src/dlmrel/   Data, model, experiment, statistics, and artifact code
tests/        Methodological and software checks
docs/         Protocol, running instructions, and verified status
```

- [Frozen protocol](docs/PROTOCOL.md)
- [How to run the GPU experiments](docs/RUNNING.md)
- [Guarded Modal execution and repair workflow](docs/MODAL.md)
- [Implementation and preliminary-result status](docs/STATUS.md)
