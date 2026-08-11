# Experiment protocol

All models must use the same:

- Universal Dependencies sentences and select/dev/test splits
- Relation definitions and token-span scoring rules
- Masking states, timesteps, and random seeds
- Head-selection procedure and held-out test evaluation
- Positional controls, confidence intervals, and result schema

Do not tune a model using the test split. Record every final run configuration with its results.
