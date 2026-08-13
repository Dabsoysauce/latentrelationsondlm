# Reproducing Dream-7B analyses

Dream uses a dedicated Transformers 4.51.3 environment; do not combine it with
DiffuGPT/DiffuLLaMA’s 4.44.2 environment.

```powershell
py -3.11 -m venv .venv-dream
.\.venv-dream\Scripts\python.exe -m pip install -e .
.\.venv-dream\Scripts\python.exe -m pip install -r requirements/dream.txt
dlmrel data audit --dataset configs/datasets/ewt.yaml
dlmrel model smoke-test --model configs/models/dream_7b.yaml --dataset configs/datasets/ewt.yaml --output dream-smoke.json
dlmrel run --model configs/models/dream_7b.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/head_search.yaml --run-id dream-ewt-v1
```

The smoke artifact must show checkpoint revision `6572adb…`, eager attentions,
normalized attention rows, deterministic eval output, and final logit-lens
parity. The confirmatory run writes select/dev scores, `selection_lock.json`,
and only the locked head’s EWT test rows. Resume a preempted run with the exact
same command plus `--resume`; completed directories cannot be overwritten.

The historical Dream result—object-to-verb L2H11 at approximately 0.80 final
state—is archived under `results/reference_legacy/pre_rigorous_6e5aaec/` as a
regression reference. It used merged/resplit UD data and is not an expected
confirmatory number. Historical masked object-to-verb performance was below
its offset null, so it does not establish the primary pre-unmask effect.

For Colab and Modal, use [colab_guide.md](colab_guide.md) and
[modal_guide.md](modal_guide.md); both invoke the same CLI.
