# Implementation status

Allowed status words are: `implemented`, `CPU-tested`, `GPU-smoke-tested`,
`full-run-complete`, `blocked`, `unsupported`, and `not-run`.

| Area | Status | Note |
|---|---|---|
| Audit and historical archive | implemented | Base SHA and protocol contradictions recorded |
| Strict config and schemas | not-run | In progress on this branch |
| Official-boundary data manifests | not-run | In progress on this branch |
| Alignment and structural relation metadata | not-run | In progress on this branch |
| Artifacts, locks, resume, validator | not-run | In progress on this branch |
| Fake CPU adapter/end-to-end path | not-run | In progress on this branch |
| Legacy DiffuGPT reproduction | not-run | Requires its pinned model environment |
| Confirmatory EWT full run | not-run | Requires GPU execution after smoke gates |
| External-treebank full transfer | not-run | Requires an EWT selection lock and GPU execution |
| Exploratory multilingual runs | not-run | No scientific result claimed |

| Model | Adapter status | GPU gate | Full run |
|---|---|---|---|
| Fake deterministic CPU | not-run | unsupported | not-run |
| DiffuGPT-S | implemented | not-run | not-run |
| DiffuLLaMA | implemented | not-run | not-run |
| Dream-7B | implemented | not-run | not-run |
| LLaDA-8B | unsupported | not-run | not-run |
| GPT-2 | not-run | not-run | not-run |

`implemented` means code exists; it does not imply numerical validation.
Unavailable GPU work remains `not-run`, never silently passed.
