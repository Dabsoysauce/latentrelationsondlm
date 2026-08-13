# Model adapters

Model YAML pins checkpoint, tokenizer, and remote-code revisions. Environments
are isolated under `requirements/`; CPU imports stay lazy.

DiffuGPT-S and DiffuLLaMA use checkpoint revisions `391d985…` and `31b2285…`
plus DiffuLLaMA code `c17e897…`; runtime acquisition checks out that detached
commit. Dream pins `6572adb…` and eager attention. GPT-2 pins `607a30d…` and is
static/final-state only. LLaDA pins `0f2787f…`; official logits/hidden states
are exposed, but attention, entropy, DLA, and ablation remain disabled until an
instrumented eager path matches official logits/hidden states under recorded
absolute/relative tolerances.

The GPU smoke command must record exact revisions, missing/unexpected weights,
mask/special tokens, offset availability, tensor shapes, row sums, padding,
determinism, final-depth logit parity, and intervention isolation when enabled:

```powershell
dlmrel model smoke-test --model configs/models/dream_7b.yaml --dataset configs/datasets/ewt.yaml
```

No real-model GPU smoke gate has been run merely because CPU code imports.
