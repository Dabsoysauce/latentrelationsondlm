# Research-grade pipeline verification audit

Audit branch: `agent/research-grade-verification-audit`  
Baseline commit: `e0f80baf4db57881849f926a0dae631260ffe558`  
Audit date: 2026-08-18

This audit verifies the active pipeline within the tested scope. It does not
claim that the code is mathematically bug-free. No result directory was read,
deleted, or modified, and no frozen scientific setting was changed.

## Pre-edit baseline

The worktree was clean and detached at the baseline commit; the audit branch
was created before editing. No `AGENTS.md` was present.

| Check | Version/result |
|---|---|
| Python | 3.12.13 |
| pip | 25.0.1 |
| pytest | 8.4.2; 90 passed in 64.55 s; no failures, warnings, or skips |
| Ruff | 0.16.3; all checks passed |
| `compileall` | passed, no output |
| `pip check` | no broken requirements found |

Commands were run exactly as follows from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
```

## Final CPU and fake-model verification

After all source and test changes, the same four checks produced:

| Check | Final result |
|---|---|
| `python -m pytest -q` | 157 passed in 56.25 s after the recovery addendum; no failures, warnings, or skips |
| `ruff check .` | all checks passed |
| `python -m compileall -q src tests` | passed, no output |
| `python -m pip check` | no broken requirements found |

The focused final artifact/CLI regression run also passed 22 tests in 18.71 s.
The local GPU preflight reported `nvidia-smi`, `HF_TOKEN`, and the Modal CLI as
unavailable. Consequently, no model was downloaded and no GPU job was started.

## Scientific test-traceability matrix

“CPU” means a hand-calculated or invariant unit test. “Fake” means a complete
deterministic CLI/artifact workflow, not evidence about a real model. “GPU” is
required to verify the pinned remote implementation itself.

| Scientific rule | Main implementation | Verification | Tier / remaining gap |
|---|---|---|---|
| Strict frozen model, data, experiment, seed, step, progress, and scoring identity | `config.py`; `artifacts.scientific_configuration` | `test_config_strict.py`; `test_checkpoints.py` | CPU; real remote revisions still need GPU loading |
| Official train/dev/test boundaries, deterministic sampling, post-dedup budgets, min words | `splits.py` | `test_splits.py`; `test_data_integrity.py` | CPU |
| Pinned raw checksums and deterministic manifest identity | `treebank.py`; `data.load_audit` | coordinated CSV/audit tamper and raw-file tamper tests | CPU |
| BOS-inclusive 128-subtoken limit, fail-closed alignment, MWT/empty-node removal | `relations.build_example`; `alignment.py` | `test_relations.py`; `test_alignment.py` | CPU; tokenizer-specific offsets need both real tokenizers |
| Six relation rules and structural metadata | `relations.extract_relations` | positive, subtype, wrong-POS/head, passive, relative, coordination, punctuation, and stable-ID tests | CPU; embedded-clause definition remains a reported caveat |
| Frozen denoising schedule and per-sentence RNG reset | `diffusion.state_at_time` | endpoint, exact timestep, monotonicity, determinism, different-seed, invalid-time tests | CPU; real device RNG still needs GPU |
| Attention normalization and word-span scoring | `diffusion.receiver_span_scores`; `experiments.shared.score_attention_heads` | hand-computed aggregation, tie, punctuation-candidate, self exclusion, invalid-gold tests | CPU; Dream/DiffuLLaMA tensor convention needs GPU |
| Six independent select-top-5/dev locks, denominator 25, no test leakage | `relation_selection.py`; `selection.py` | `test_relation_selection.py`; `test_selection_lock.py` | CPU/fake |
| Full selection-aware permutation, B=1000/seed 42, finite-sample p, five-secondary Holm | `permutation.py`; `head_search._run_permutations` | slow independent reference, resume identity, split streams, corruption, Holm tests | CPU/fake |
| Fixed offset, receiver controls, matched alternatives, structural slices, clustered bootstrap | `controls.py`; `evaluation/statistics.py`; `experiments/shared.py` | hand calculations and independent 2,000-draw bootstrap reference | CPU |
| Nine-point locked time curve without reselection | `experiments/time_curve.py` | union/filter runner test, exact 27 seed-time calls, aggregation test, fake CLI | CPU/fake |
| Natural-log entropy, BOS removal/sink, equal sentence weighting, degenerate BOS | `experiments/attention_entropy.py` | hand-computed matrices, all-BOS test, all seed/time/layer/head orchestration, fake CLI | CPU/fake; real attentions need GPU |
| Logit-lens depths, normalization, ranks, visibility, final parity | `experiments/logit_lens.py` | known hidden states/ranks/logits, orchestration, fake CLI | CPU/fake; measured real parity needs GPU |
| POS select scaling, dev-only C, select-only final fit, test isolation, controls | `experiments/pos_probe.py` | call-order, exact-tie C, reproducibility, invalid-split, aggregation, checkpoint tests | CPU/fake; real hidden states need GPU |
| Exact model-matched EWT-lock transfer, target test only | `relation_selection.load_relation_locks`; `head_search.run_locked_transfer` | source tamper/revision tests and target-only transfer runner test | CPU/fake; multilingual model inference needs GPU |
| Atomic 300-sentence checkpoint/resume and legacy whole-seed reuse | `checkpoints.py` | interruption, equality, corrupt/temp/hash/range, legacy seed 42/43 tests | CPU |
| Required, internally consistent, finite run artifacts | `artifacts.validate_run`; `shared.write_frames` | missing/corrupt/nonfinite/duplicate/shard mismatch and CLI validation tests | CPU/fake |
| Compatible cross-model settings and common-instance comparison | `evaluation/compare_models.py` | seed/progress incompatibility, common set, duplicates, deterministic sort tests | CPU |
| CLI prepare/smoke/run/resume/validate/compare and all active artifact contracts | `cli.py`; `fake_run.py` | command dispatch plus all six experiment configurations | Fake; fake execution is not model correctness |

## Bugs found and corrected

| Severity | Root cause | Potential effect on earlier output | Correction |
|---|---|---|---|
| High | Dataclass defaults allowed omitted YAML fields despite the “strict” loader | An incomplete custom config could silently use defaults | Every field is required; active pins and frozen settings are validated explicitly |
| High | Prepared CSVs and `audit.json` were trusted without rebuilding from checksum-verified raw UD data | Coordinated or unnoticed manifest changes could alter the sample while appearing internally consistent | Raw files are rechecked and all manifests are deterministically rebuilt and compared |
| High | Cross-model compatibility checked only dataset ID, experiment type, and scoring | Runs with different seeds, progress points, revisions, budgets, or other science could be combined | All non-model resolved scientific configuration plus manifest hashes must match |
| High | Final validation checked file presence and a few hashes but did not parse tables or compare Parquet with shards | Modified, corrupt, duplicate, infinite, or wrong-shaped artifacts could validate | Validation now parses artifacts, checks schemas/science/provenance/counts, and compares shards |
| High | JSON shards received pandas `NaN` for legitimate unavailable controls | A scientifically valid run could fail during final JSON serialization | Table-only missing values are explicitly serialized as JSON `null`; infinities remain errors |
| Medium | Checkpoint metadata did not verify that output rows belonged to the declared sentence range | A misplaced/corrupt chunk or legacy file could introduce foreign rows on resume | Computed, loaded, and legacy chunks reject out-of-range sentence IDs |
| Medium | Permutation resume accepted overlong, non-finite, invalid-head, or status-inconsistent progress | Corrupt resumed null distributions could be used | Progress values, bounds, heads, status, and temporary files are validated |
| Medium | POS test features for all seeds were computed before C was frozen | No test label entered C selection, so metrics were not mathematically leaked, but the execution order violated the frozen discipline | All seed-specific select/dev fits and C choices finish before the test manifest is opened |
| Medium | POS evidence omitted token form and word index | Predictions within the same sentence were difficult to audit and duplicate-check | Final POS rows now retain both fields |
| Medium | An invalid gold receiver was silently skipped during attention scoring | Corrupt alignment could silently reduce denominators | Invalid gold-candidate state now fails loudly |
| Medium | Fake CLI execution always generated head-search artifacts regardless of requested experiment | CPU integration could falsely appear to cover other runners | Fake workflows and validation are now experiment-specific |
| Low | Normalized entropy divided by `ln(1)` for an all-BOS sequence | Degenerate validation input produced `NaN`; official min-word sentences are not affected | The uniquely defined degenerate normalized entropy is reported as zero |
| Low | Real smoke scalar conversion retained gradients | Dream emitted a PyTorch conversion warning and needless graph retention | Smoke validation runs under `no_grad` and reports detached measured errors |

No repository evidence was found that any existing official result was damaged
by these bugs. The repository status says real-model outputs under this pipeline
have not yet been declared complete. Previously produced Dream/DiffuLLaMA
results remain preliminary and should still be rerun under the frozen pipeline.

## Frozen behaviors not changed

- Timestep-0 masking contains no stochastic draws, so equal input rows are
  identical across seeds 42/43/44. The denominator of 25 is nevertheless the
  frozen pooled-row denominator. This duplicates evidence for selection at
  time zero; it is a scientific design choice, not an implementation accident.
- `embedded_clause` currently includes dependency paths deeper than one as
  well as explicit clausal dependencies. That can be broader than the ordinary
  linguistic meaning of “embedded clause.” It affects a descriptive structural
  slice, not the primary receiver-accuracy calculation. Changing it requires
  approval and a versioned rerun of that slice.
- Punctuation remains an eligible aligned receiver word under the current
  protocol; BOS, special positions, and the query itself do not.

## Changed and added files

Production code:

- `src/dlmrel/artifacts.py`: strict serialization, run-artifact parsing,
  consistency checks, and stable hashes of finalized scientific outputs.
- `src/dlmrel/checkpoints.py`: sentence-range validation for new chunks and
  compatible whole-seed checkpoints, with streaming chunk iteration for the
  recovery grid.
- `src/dlmrel/cli.py`: explicit GPU-only missing-grid and CPU-only head-search
  finalization commands.
- `src/dlmrel/config.py`: fail-closed fields, immutable revisions/checksums,
  and frozen seeds, steps, progress points, and scoring validation.
- `src/dlmrel/data.py`: fail-closed tokenization exclusions and deterministic
  manifest reconstruction from checksum-verified raw UD files.
- `src/dlmrel/evaluation/compare_models.py`: complete non-model scientific
  compatibility, duplicate rejection, common-instance grouping, and sorting.
- `src/dlmrel/experiments/attention_entropy.py`: finite all-BOS normalization.
- `src/dlmrel/experiments/head_search.py`: routes fresh runs through the same
  persisted all-head evidence and relation-wise finalizer used by recovery.
- `src/dlmrel/experiments/pos_probe.py`: select/dev fitting before any test
  access, explicit validation, and auditable word identifiers.
- `src/dlmrel/experiments/shared.py`: invalid-gold failure and strict JSON
  serialization of nullable control values.
- `src/dlmrel/fake_run.py`: experiment-specific end-to-end fake workflows and
  the same recovery finalizer used by real head search.
- `src/dlmrel/head_search_recovery.py` (new): phased evidence validation,
  missing test-grid scoring, and model-free relation-wise finalization.
- `src/dlmrel/permutation.py`: strict resumed-progress and temporary-file
  validation.
- `src/dlmrel/pipeline.py`: gradient-free measured smoke checks and final
  artifact identity recording.

Tests:

- `tests/test_artifacts.py`: artifact corruption, non-finite value, shard, and
  finalized-output tampering tests.
- `tests/test_checkpoints.py`: ranges, duplicates, corruption, and legacy
  checkpoint compatibility.
- `tests/test_cli.py`: all active fake experiment CLI workflows and validation.
- `tests/test_compare.py`: incompatible science, common instances, duplicate
  observations, and deterministic ordering.
- `tests/test_config_strict.py`: required fields, frozen settings, immutable
  checksum/revision rules, and snapshots of active model/dataset YAMLs.
- `tests/test_metrics.py`: controls, pooling, bootstrap, and safe empty output.
- `tests/test_permutation.py`: corrupt, inconsistent, overlong, and temporary
  permutation progress.
- `tests/test_pos_probe.py`: split isolation, tie-breaking, feature validation,
  controls, aggregation, and trace fields.
- `tests/test_relations.py`: all six relations plus structural and invalid
  counterexamples.
- `tests/test_splits.py`: official boundaries, deduplication, budgets, and
  deterministic sampling.
- `tests/test_active_runners.py` (new): time-curve, entropy, logit-lens, and
  external-transfer orchestration invariants.
- `tests/test_attention_entropy.py` (new): independent entropy calculations
  and the degenerate sequence case.
- `tests/test_data_integrity.py` (new): raw/manifest/audit tamper detection and
  recorded tokenizer failures.
- `tests/test_denoising_attention.py` (new): frozen schedule and hand-computed
  attention aggregation/candidate behavior.
- `tests/test_logit_lens.py` (new): known ranks, depths, visibility, and exact
  final-depth parity.
- `tests/test_head_search_recovery.py` (new): copy-only recovery, test-stage
  isolation, interruption/resume, model-free finalization, corruption, and
  uninterrupted-equivalence tests.

Documentation:

- `docs/VERIFICATION_AUDIT.md` (new): baseline, traceability matrix, findings,
  caveats, exact checks, GPU boundary, and proposed commit message.
- `docs/HEAD_SEARCH_RECOVERY.md` (new): exact copy-first Colab GPU and CPU
  recovery commands for `dream-english-head-3seed-v1`.
- `docs/RUNNING.md`: links the interrupted Dream run to the phased procedure.

## Final diff inspection

The final local branch is `agent/research-grade-verification-audit` at baseline
commit `e0f80baf4db57881849f926a0dae631260ffe558`. Inspection found 21 modified
tracked files and six intentional untracked additions listed above. No active
config, result directory, notebook, or dependency pin changed. `git diff
--check` passed; Git only printed the repository's existing LF-to-CRLF working
copy notices on Windows. No change is staged, committed, pushed, or merged.

## Real-model validation status

The CPU audit alone could not make a GPU claim. The Dream smoke measurement is
now recorded below; Dream's tiny non-reportable runner validation and both
DiffuLLaMA GPU checks remain. Record the exact attention row-sum error,
repeat-forward maximum absolute error, attention/logit/hidden-state shapes,
final-depth parity error, and one interruption/resume comparison. Do not relax
a tolerance based only on a failing value.

The audit must be updated with those measured results before marking GPU
validation complete. Full official experiments are separate from this
software-validation step.

The Dream smoke rerun measured a maximum bfloat16 attention row-sum error of
`0.0029296875`, with the expected attention shapes, no nonfinite values, and an
unpadded input. The smoke-only normalization tolerance is therefore `1e-2`;
this accommodates the observed bfloat16 accumulation error while continuing to
reject materially malformed attention rows. This does not change any model,
dataset, experiment, seed, scoring, relation, or head-selection setting.

### Exact Modal smoke commands (not executed)

These commands were checked against the current Modal CLI documentation on
2026-08-18. They intentionally stop at the two measured model smoke tests;
starting a standard `dlmrel run` with an active config would launch an official
workload, not a tiny software-validation run.

From PowerShell in this repository, with `HF_TOKEN` already set locally:

```powershell
py -m pip install --upgrade modal
modal setup
modal secret create dlmrel-huggingface HF_TOKEN="$env:HF_TOKEN"
modal volume create dlmrel-validation-cache
modal shell --gpu A100-80GB --memory 65536 --add-python 3.12 --add-local . --volume dlmrel-validation-cache --secret dlmrel-huggingface
```

Inside that first Modal shell, run Dream and then exit:

```bash
cp -R /mnt/latentrelationsondlm-worktree /tmp/dlmrel-validation
cd /tmp/dlmrel-validation
export HF_HOME=/mnt/dlmrel-validation-cache/huggingface
python -m pip install -e .
python -m pip install -r requirements/dream.txt
python -m pip freeze > /mnt/dlmrel-validation-cache/dream-pip-freeze.txt
dlmrel smoke-test --model configs/models/dream_7b.yaml --output /mnt/dlmrel-validation-cache/dream-smoke.json
```

Open a fresh shell with the same `modal shell` command, then run DiffuLLaMA
and exit:

```bash
cp -R /mnt/latentrelationsondlm-worktree /tmp/dlmrel-validation
cd /tmp/dlmrel-validation
export HF_HOME=/mnt/dlmrel-validation-cache/huggingface
python -m pip install -e .
python -m pip install -r requirements/diffullama.txt
python -m pip freeze > /mnt/dlmrel-validation-cache/diffullama-pip-freeze.txt
dlmrel smoke-test --model configs/models/diffullama_7b.yaml --output /mnt/dlmrel-validation-cache/diffullama-smoke.json
```

The repository does not currently expose a safe isolated CLI for the requested
tiny real-model runs through all six experiment runners. The standard CLI
would use the frozen official manifests and could incur substantial cost.
Accordingly, runner-level real-model execution and the real-model interruption/
resume comparison remain unexecuted and require approval for a dedicated
non-reportable validation harness or for the full official runs.

## Proposed commit message

```text
audit and harden frozen DLM research pipeline
```
