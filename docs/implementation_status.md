# Implementation status

Allowed status words: `implemented`, `CPU-tested`, `GPU-smoke-tested`,
`full-run-complete`, `blocked`, `unsupported`, and `not-run`.

| Milestone | Status | Evidence/limitation |
|---|---|---|
| Audit and historical archive | implemented | Commit `a809be6`; base SHA/provenance retained |
| Strict config and schemas | CPU-tested | Unknown and capability-incompatible fields reject |
| Official-boundary manifests | CPU-tested | UD 2.15 EWT/GUM/LinES/ParTUT/de/ja audits |
| Alignment/relation metadata | CPU-tested | Offset path plus structural relation fixtures |
| Artifacts, locks, resume, validator | CPU-tested | Atomic/idempotent shard and no-overwrite tests |
| Fake CPU end-to-end | CPU-tested | Deterministic tensors, DLA, ablation, CLI artifacts |
| Legacy DiffuGPT reproduction | implemented | GPU full run not-run |
| Confirmatory EWT protocol | implemented | GPU smoke and full run not-run |
| EWT-locked external transfer | implemented | Requires completed EWT lock; full run not-run |
| Entropy/POS/logit lens | implemented | CPU helpers tested; real GPU runs not-run |
| DLA/causal ablation | implemented | Fake isolation tested; real adapters gated |
| Colab/Modal runners | implemented | Shared CLI; cloud execution not-run |
| Exploratory multilingual | implemented | Config/audit only; competence/full runs not-run |
| Native-generation timing schema | implemented | Adapter execution unsupported/not-run |

| Model | Adapter | CPU | GPU smoke | Full run |
|---|---|---|---|---|
| Fake deterministic CPU | implemented | CPU-tested | unsupported | full-run-complete |
| DiffuGPT-S | implemented | CPU-tested | not-run | not-run |
| DiffuLLaMA | implemented | CPU-tested | not-run | not-run |
| Dream-7B | implemented | CPU-tested | not-run | not-run |
| LLaDA-8B logits/hidden | implemented | CPU-tested | not-run | not-run |
| LLaDA attention/head analyses | unsupported | not-run | not-run | not-run |
| GPT-2 final state | implemented | CPU-tested | not-run | not-run |

“Implemented” means the path and honest capability gate exist; it never implies
numerical validation. GPU and expensive run status remains `not-run` until an
artifact produced on appropriate hardware validates it.

GPU probe on 2026-08-13: `nvidia-smi` was unavailable in the implementation
environment. Consequently every real-model GPU smoke/full run remains
`not-run`; no unavailable gate is marked passed.
