# Treebanks

All corpora use Universal Dependencies 2.15 (`CC BY-SA 4.0` distribution;
individual upstream sources may add attribution terms). Config YAML stores the
exact repository commit and per-file SHA-256.

| Config | Language/domain | Role | 2.15 commit | Eligible base select/dev/test |
|---|---|---|---|---:|
| `ewt.yaml` | English web | confirmatory discovery/test | `4dc8e10` | 4000/1000/1000 |
| `gum.yaml` | English multi-genre | primary external replication | `34d01cb` | 8448/1192/1156 |
| `lines.yaml` | English translations | secondary external replication | `eced5e1` | 3116/1009/1008 |
| `partut.yaml` | English parallel text | secondary external replication | `5fe6cf2` | 1773/155/153 |
| `de_gsd.yaml` | German web/news/reviews | freer-order exploratory | `d47d567` | 13758/787/922 |
| `ja_gsd.yaml` | Japanese web/news | head-final exploratory | `8e5794f` | 7011/504/537 |

Counts were produced by the CPU audit with minimum four syntactic words,
cross-role normalized-text deduplication, and no model tokenization. ParTUT’s
small dev/test sets are reported as-is; no significance-driven resampling is
allowed. German/Japanese null results require language-competence and
tokenization sanity checks and cannot by themselves imply absent structure.

`dlmrel data audit --dataset <yaml>` verifies checksum, official boundary,
deterministic order/hash, counts, and zero normalized-text overlap.
