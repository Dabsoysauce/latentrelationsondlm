# Recovering `dream-english-head-3seed-v1`

This recovery preserves the frozen protocol. It does not recompute select or
development evidence, does not change relation locks, and does not substitute
the existing locked-head test rows for the required all-head permutation
evidence.

## Two phases

1. `recover-head-search-test-grid` loads Dream and the tokenizer, verifies the
   saved select/dev evidence and six-lock bundle, loads only the EWT test
   examples, and scores all model heads at progress/timestep 0 for seeds
   42/43/44. Its stage name is
   `test-all-heads-selection-permutation`, distinct from the legacy
   `test-locked-head` stage. It writes atomic 300-sentence checkpoints and the
   narrow consolidated file `test_all_head_permutation_evidence.parquet`.
2. `finalize-head-search` does not load Dream or a tokenizer. It reads only
   permutation-required columns from select/dev/all-head-test Parquet files,
   handles one relation at a time, resumes the six 1,000-permutation
   checkpoints, and uses the existing locked `instances.parquet` for controls,
   bootstrap intervals, ordinary metrics, seed metrics, and structural slices.

The GPU phase requires 732 × 3 = **2,196 test-sentence forward passes**. Each
forward exposes all attention heads. It creates nine sentence checkpoints:
three ranges (0–300, 300–600, and 600–732) for each of three seeds. The expected
all-head evidence size is 7,233 × the detected Dream head count; if the pinned
model exposes 1,024 layer/head pairs, that is 7,406,592 narrow rows. The CPU
phase performs 1,000 selection-aware permutations for each of six relations,
plus the frozen 2,000-draw clustered bootstraps; it performs no model inference.

## Exact copy-first Colab commands

Use the corrected repository revision after reviewing, committing, and pushing
these local changes. Mount Drive and define the same results root as before:

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import shutil

RESULTS = Path("/content/drive/MyDrive/dlmrel-results")
SOURCE_RUN = (
    RESULTS
    / "confirmatory_ewt/dream_7b/ewt/confirmatory_head_search"
    / "dream-english-head-3seed-v1"
)
RECOVERY_RUN = (
    RESULTS
    / "recovery-work"
    / "dream-english-head-3seed-v1"
)

assert SOURCE_RUN.is_dir(), SOURCE_RUN
if not RECOVERY_RUN.exists():
    RECOVERY_RUN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_RUN, RECOVERY_RUN)

print("Source remains untouched:", SOURCE_RUN)
print("Recovery copy:", RECOVERY_RUN)
```

In the GPU runtime, install the pinned Dream environment and run only the
missing grid:

```python
%cd /content/latentrelationsondlm

import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

!pip install -q -e .
!pip install -q -r requirements/dream.txt
!dlmrel recover-head-search-test-grid --run-dir "{RECOVERY_RUN}"
```

If Colab disconnects, rerun the last command with the same `RECOVERY_RUN`. It
validates and reuses completed 300-sentence all-head chunks. It never interprets
the differently named legacy locked-head chunks as all-head evidence. If the
consolidated evidence is already complete, it exits without loading Dream.

The second phase may run in a fresh CPU runtime. Mount Drive, clone/check out
the same repository revision, install the package, recreate the `RECOVERY_RUN`
Python variable above, and execute:

```python
%cd /content/latentrelationsondlm
!pip install -q -e .
!dlmrel finalize-head-search --run-dir "{RECOVERY_RUN}"
!dlmrel validate --run-dir "{RECOVERY_RUN}"
```

If CPU finalization is interrupted, rerun `finalize-head-search`. Completed
permutation indices are reused only after their scientific and evidence hashes
match. A completed valid finalization is idempotent and simply validates again.

Do not point either recovery command at `SOURCE_RUN`; all development and
recovery work should remain in `RECOVERY_RUN` until its final validation has
passed and the original is intentionally archived unchanged.
