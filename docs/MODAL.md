# Guarded Modal execution

## What this adds

Modal supplies a temporary CPU or GPU machine, three persistent Volumes, logs, and a run-identity
lock. It does not decide what the research does. The remote worker checks out one exact Git commit
and invokes the existing `dlmrel` CLI with an argument list and `shell=False`. The normal `dlmrel`
configuration, manifests, relation locks, atomic 300-sentence checkpoints, statistics, summaries,
and validation remain the only scientific implementation.

The repair controller is separate. When an implementation failure is reproducible, it gives Codex
only a redacted log tail and a fixed repair policy. Codex leaves a patch in a checkout with no
Modal or GitHub write credential. A later secret-free job rejects protected changes and runs the
full CPU checks. Only after those checks does a trusted job commit, push, and open or update a draft
PR. It never merges.

Humans still control all scientific changes, secrets, resource/cost choices, promotion of scratch
results, PR review, and merge. Passing tests shows that the specified software contracts held; it
does not prove that the scientific interpretation is correct.

## Accounts and secret names

You need a Modal account, a GitHub repository where Actions is enabled, and an OpenAI API project
for optional repairs. Add these repository Actions secrets under **Settings → Secrets and
variables → Actions**:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `OPENAI_API_KEY`

GitHub supplies the short-lived `GITHUB_TOKEN`; do not create a replacement repository secret for
it. The workflow does not use an `HF_TOKEN`. Dream and DiffuLLaMA are accessed publicly or from the
model cache. If model access ever requires a private Hugging Face token, stop and use a supervised,
human-approved procedure rather than exposing a long-lived token to automatically repaired code.

Never paste a secret into a workflow input, log, config, prompt, notebook, or repository file.

## Local Modal setup

From a clean checkout at the commit you intend to run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/modal.txt
.\.venv\Scripts\modal.exe setup
.\.venv\Scripts\modal.exe token info
```

The wrapper uses ephemeral `modal run`; it does not deploy a persistent app. It lazily creates:

- `dlmrel-model-cache` for public Hugging Face/model cache files.
- `dlmrel-scratch-results` for isolated attempt artifacts and existing checkpoint formats.
- `dlmrel-attempt-logs` for complete sanitized process logs.
- `dlmrel-run-identity-locks`, a Modal Dict whose atomic `skip_if_exists` write prevents two
  supported submissions from owning the same run identity.

## Credential-free local dry run

This is the first command to run. It uses recorded fixtures and does not contact Modal, OpenAI,
GitHub, Hugging Face, or a GPU; it does not write to git.

```powershell
.\.venv\Scripts\python.exe -m scripts.modal_agent_loop dry-run
```

It demonstrates success, infrastructure retry, deterministic repair, repeated-failure stop,
protected-change stop, and exhausted-attempt stop.

## Trigger a supervised workflow

1. Push the reviewed wrapper commit to GitHub. Do not deploy it.
2. Open **Actions → Guarded Modal DLM run → Run workflow**.
3. Paste the full 40-character commit SHA, not a branch name.
4. Choose one operation, active model/dataset/experiment, matching track, and matching resource
   profile.
5. Use a safe immutable run ID. For resume/recovery, also supply the expected scientific hash,
   exact manifest-hash JSON, and path to a staged copy.
6. Leave `mode=supervised` and `dry_run=true`, and choose `cpu-small`, for the first workflow check.
7. After the dry run passes, rerun with `dry_run=false` only when you intend to spend compute.

Supervised mode submits Modal and, on a repairable failure, prepares a tested draft PR. It stops
before running the repaired source on Modal. A human reviews the diff and chooses whether to rerun.

`guarded-auto` is optional. Each workflow run makes at most one Codex repair. A trusted job queues
the next workflow run only after the patch passed policy and all CPU checks. The same run-identity
concurrency key serializes the chain. The controller allows at most two infrastructure retries and
three Codex repair attempts, never auto-merges, and stops on the conditions below.

## RunSpec paths and routing

Every user value is validated before compute. Config paths are exact members of the active config
allowlists; run IDs and relative Volume paths allow only letters, numbers, `.`, `_`, and `-` path
segments. Absolute paths, `..`, backslashes, shell metacharacters, arbitrary commands, unknown
fields, mutable Git refs, and official result namespaces are rejected.

Routes are fixed:

| Operation/model | Route | CLI responsibility |
|---|---|---|
| CPU tests, dry runs, validation | CPU | Existing pytest or `dlmrel validate` |
| Dream smoke/run | Dream image | Existing Dream-compatible `dlmrel` command |
| DiffuLLaMA smoke/run | DiffuLLaMA image | Existing DiffuLLaMA-compatible `dlmrel` command |

Dream installs `requirements/dream.txt`; DiffuLLaMA installs
`requirements/diffullama.txt`. They never share one Transformers installation. Images do not contain
model weights. The worker reports Python, selected package versions, exact repository commit, the
saved model/tokenizer/code revisions, CUDA/PyTorch information emitted by the runtime, and the GPU
information emitted by the job where available.

## Persistence, logs, and resume

Attempts write only below:

```text
scratch/<run-id>/<operation>/attempt-<n>/<track>/<model>/<dataset>/<experiment>/<run-id>/
```

The worker explicitly reloads result/log Volumes before use and commits them before and after the
CLI call. It does not invent checkpoints. A resume reuses the exact `dlmrel` whole-seed and atomic
300-sentence checkpoints after `dlmrel` validates their scientific and manifest identities.
For a real experiment in a fresh checkout, the wrapper first invokes the existing `dlmrel prepare`
command for the selected revision-pinned dataset. Preparation verifies the configured upstream
revision and checksums and stops on mismatch; it does not alter experiment configuration.

Inspect evidence in three places:

- The GitHub Actions summary and the 14-day sanitized attempt artifact.
- The Modal dashboard/call ID recorded in `RunResult`.
- `dlmrel-attempt-logs`, using `modal volume ls` or `modal volume get`.

The structured result includes the sanitized argv, commit, hashes, timing, failure category and
signature, bounded output tails, complete log location, last checkpoint, validation, scratch path,
and recommended action. It never includes a process environment or secret value.

The workflow at this pinned commit supports ordinary Dream smoke tests, experiment runs/resume, and
validation. It intentionally does not expose newer recovery/finalization commands that are absent
from the pinned scientific source. Promotion from scratch to an official path is not automated.

## Resource profiles and cost controls

Rates are an operational snapshot from Modal's public pricing page on 2026-08-18; verify the page
before a paid run. The RunSpec rejects a timeout whose request-cost estimate exceeds the supplied
ceiling. Modal billing can still reflect actual bursting, storage, downloads, taxes, or future rate
changes, so also configure a Modal Environment/Workspace budget.

| Profile | Resources | Timeout ceiling | Approximate ceiling at full timeout |
|---|---|---:|---:|
| `cpu-small` | 4 physical CPU, 16 GiB | 2 h | $0.64 |
| `cpu-large` | 8 physical CPU, 64 GiB | 12 h | $10.67 |
| `dream-a100-80gb` | A100 80 GB, 4 CPU, 32 GiB | 6 h | $17.66 |
| `diffullama-a100-80gb` | A100 80 GB, 4 CPU, 32 GiB | 6 h | $17.66 |

No automatic response may change the GPU type, precision, model, attention implementation, or a
scientific setting. Resource exhaustion stops for operator review.

## Stop and cancellation

Cancel the GitHub workflow from its Actions page to stop the controller. Then use the Modal
dashboard/call reference in the job summary to cancel a still-running function call. `modal app
list` and `modal app logs` help locate an ephemeral app. Use `modal app stop` only after confirming
the exact app ID; it stops that app and is not a checkpoint deletion command.

Never delete a result/checkpoint directory to make resume work. An incomplete `.tmp` checkpoint is
handled by the existing `dlmrel` checkpoint logic.

## Automatic stop conditions

The controller stops rather than improvising when:

- three Codex repair attempts or two infrastructure retries are exhausted;
- the same normalized implementation failure recurs after a repair;
- a repair is empty, changes a protected/out-of-allowlist path, or touches sensitive scientific
  source semantics;
- active config bytes or scientific identity change;
- the regression, focused, full pytest, Ruff, compileall, or `pip check` fails;
- artifacts/checkpoints would be deleted, an official/completed run would be overwritten, or a
  staged copy/hash is incompatible;
- a failure needs a scientific decision, private remote secret, unapproved dependency change, or
  higher cost/resource ceiling;
- the failing commit is stale or cannot be reproduced.

## Security limitations

The supported worker clones only the public repository URL and strips OpenAI, GitHub, Modal, and
Hugging Face credentials from the `dlmrel` subprocess environment. Modal client credentials stay in
the Actions submit step; the remote function never receives them as user secrets. Codex receives
only the OpenAI key through the pinned official action and never receives Modal or push credentials.

GitHub concurrency and the atomic Modal Dict lock protect supported submissions. Directly modifying
Volumes, calling worker functions outside `submit_job`, changing workflow permissions, or running a
different unreviewed wrapper is outside this boundary. Logs and repository text remain untrusted
prompt-injection surfaces even after redaction. Human review is mandatory before promotion or merge.
