# PR title

Rigorous multi-model, multi-treebank DLM interpretability pipeline

# PR body

## Scientific purpose

This PR repairs the DLM relation-analysis pipeline so future results can support
clearly separated legacy, confirmatory EWT, locked external-treebank transfer,
and exploratory claims. It does not present unavailable GPU runs as completed
evidence.

## What changed

- archived pre-rigorous results with explicit provenance at base SHA
  `6e5aaec772f942573add5d211ab10edf683d69bd`;
- pinned UD 2.15 EWT, GUM, LinES, ParTUT, German GSD, and Japanese GSD by commit
  and per-file SHA-256 while preserving official train/dev/test boundaries;
- added strict typed configs, stable manifests, instance intersections,
  structured alignment exclusions, relation metadata, and challenge slices;
- implemented select top-five/dev choice/immutable lock/test-one-head flow and
  unchanged EWT-lock transfer;
- added receiver controls, matched alternatives, sentence-clustered intervals,
  multi-seed aggregation, correction helpers, and selection-aware permutation;
- implemented versioned non-overwriting artifacts, deterministic resume
  checkpoints, validator, and a complete fake CPU path;
- repaired timestep/visibility aggregation, entropy-over-time, masked POS
  probing, rank-based logit lens, and explicit DLA/ablation semantics;
- pinned DiffuGPT, DiffuLLaMA, Dream, LLaDA, GPT-2, and remote-code revisions;
  LLaDA attention and all unvalidated real-model DLA/ablation capabilities stay
  disabled;
- added shared CLI, GPT-2 final-state path, Colab notebook, Modal launcher,
  isolated model requirements, and complete protocol/reproduction docs.

## Validation performed

- clean editable core install succeeded;
- `python -m pytest -q`: **61 passed**;
- `python -m ruff check src tests`: **passed**;
- `python -m compileall -q src tests runners`: **passed**;
- `git diff --check 6e5aaec..HEAD`: **passed**;
- fake confirmatory end-to-end run and `validate-run`: **passed**;
- all five model configs smoke-test dry-run successfully;
- all six pinned treebank audits passed official-boundary and zero-overlap
  checks. Counts are documented in `docs/treebanks.md`.

## GPU gates and limitations

`nvidia-smi` was unavailable in the implementation environment. Real-model GPU
smoke tests and full DiffuGPT/DiffuLLaMA/Dream/LLaDA/GPT-2 experiments are
therefore `not-run`, not passed. LLaDA attention remains unsupported until
official-versus-instrumented numerical parity is recorded. Native-generation
timing has a strict schema but no adapter advertises execution yet. DLA and
causal ablation are CPU-tested on the fake adapter and capability-gated for real
models.

## Reproduction entrypoints

Use `dlmrel data audit`, `dlmrel model smoke-test`, `dlmrel run`,
`dlmrel validate-run`, `dlmrel compare`, and `dlmrel status`. Exact local,
legacy, confirmatory, transfer, GPT-2, Colab, and Modal commands are in the
README and `docs/` guides. Large weights, downloaded corpora, caches, secrets,
and full run outputs are excluded from Git.
