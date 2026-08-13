# Experiment ownership and gate checklist

Assign a human owner before launching any paid or long-running job. Status must
use the vocabulary in `implementation_status.md`.

| Area | Owner | Status | Required artifact |
|---|---|---|---|
| DiffuGPT legacy reproduction | unassigned | not-run | validated run + tolerance table |
| DiffuGPT confirmatory EWT | unassigned | not-run | GPU smoke + EWT selection lock |
| DiffuLLaMA confirmatory EWT | unassigned | not-run | GPU smoke + EWT selection lock |
| Dream confirmatory EWT | unassigned | not-run | GPU smoke + EWT selection lock |
| LLaDA parity | unassigned | not-run | official/instrumented error report |
| GPT-2 final-state comparison | unassigned | not-run | common-instance comparison |
| GUM/LinES/ParTUT transfer | unassigned | not-run | unchanged EWT lock hashes |
| German/Japanese exploratory | unassigned | not-run | competence/tokenization audit |

The owner records hardware, exact command, run ID, cost/time, validation output,
and unexpected exclusions. A status change to `GPU-smoke-tested` or
`full-run-complete` requires a committed or externally checksummed artifact;
verbal confirmation is insufficient.
