# Modal guide

Install Modal locally, authenticate with Modal’s CLI, and store Hugging Face
credentials as a Modal secret named `huggingface`. Do not hard-code tokens.

```powershell
modal run runners/modal_app.py --model configs/models/dream_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/head_search.yaml --dry-run
modal run runners/modal_app.py --model configs/models/dream_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/head_search.yaml --run-id ewt-dream-v1
```

The wrapper calls `python -m dlmrel.cli`; it contains no scientific fork.
Results and checkpoint shards persist in the mounted volume. Repeat with
`--run-id ewt-dream-v1 --resume` after preemption. Check provider pricing and
terminate idle containers; 7B/8B multi-seed analyses are expensive.
