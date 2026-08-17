# DiffuLLaMA-7B Confirmatory Time-Curve Results

This directory documents the completed rigorous DiffuLLaMA-7B time-curve experiment on English EWT.

## Experiment

The time-curve run measures receiver-prediction accuracy across diffusion progress using a frozen attention-head selection.

The selected head was produced by the confirmatory head-search stage and then locked before running the trajectory experiment.

Earlier preliminary DiffuLLaMA head-search results already exist in the repository history, so this directory focuses on the new confirmatory time-curve output.

## Runs

Run ID:

`diffullama-ewt-time-optimized-v2`

Model:

`DiffuLLaMA-7B`

Dataset:

`UD English EWT`

## Validation

The completed time-curve run passed `dlmrel validate` with no validation errors.

Included files:

- `selection_lock.json` — frozen head selection used by the time-curve experiment
- `timecurve_summary.json` — summary of the completed run
- `timecurve_validation.json` — validation result
- `timecurve_metrics.csv` — trajectory metrics across diffusion progress

## Attribution

Experiment run and results prepared by Jennifer Seok.