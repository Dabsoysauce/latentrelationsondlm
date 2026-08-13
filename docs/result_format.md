# Result format

Run schema is `dlmrel-run-v1`; config schema is `dlmrel-config-v2`; manifest
schema is `dlmrel-manifest-v1`; selection lock schema is
`dlmrel-selection-lock-v1`.

Each unique, non-overwriting run directory contains:

| File | Required content |
|---|---|
| `config.resolved.yaml` | every scientific and runtime field consumed |
| `command.txt` | exact launch command |
| `run_metadata.json` | schema/config hash, Git state, time and completion |
| `environment.json` | Python/platform/Git and available accelerator record |
| `manifest_refs.json` | ordered select/dev/test manifest SHA-256 hashes |
| `selection_lock.json` | required for locked test/transfer runs |
| `exclusions.parquet` | sentence/instance/reason rows |
| `instances.parquet` | raw numerators, predictions, masses, grouping keys |
| `per_seed_metrics.csv` | seed-level aggregates |
| `metrics.csv` | frozen summary metrics and counts |
| `summary.json` | completion, capabilities, headline counts |
| `validation.json` | validator result and error list |
| `checkpoints/` | atomic deterministic shards |
| `figures/` | derived visualizations only |

Instance primary key is `(model, dataset, sentence_id, instance_id, seed,
timestep, layer, head, metric)`. Attention mass is unitless probability mass;
distance is signed words; normalized progress lies in `[0,1]`; visibility uses
the four allowed whole-word states. Validation fails missing metadata, schema or
hash mismatch, corrupt/duplicate shards, incomplete status, or unsupported
capability claims.
