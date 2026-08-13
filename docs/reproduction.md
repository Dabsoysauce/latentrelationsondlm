# Reproduction

Historical preliminary results and provenance are at
`results/reference_legacy/pre_rigorous_6e5aaec/`. Reproduce the source-derived
DiffuGPT protocol with its isolated environment and explicitly legacy config:

```powershell
py -3.11 -m venv .venv-diffugpt
.\.venv-diffugpt\Scripts\python.exe -m pip install -e .
.\.venv-diffugpt\Scripts\python.exe -m pip install -r requirements/diffugpt.txt
dlmrel run --model configs/models/diffugpt.yaml --dataset configs/datasets/ewt.yaml --experiment configs/experiments/legacy_diffugpt.yaml --track legacy_reproduction
```

This path preserves last-subtoken selection and the historical final-state
objective. Numerical comparison must report checkpoint/version, head ranking,
instance count, and absolute difference from source artifacts; no tolerance is
declared passed until the GPU reproduction actually runs. Legacy output cannot
be interpreted as a confirmatory official-boundary result.
