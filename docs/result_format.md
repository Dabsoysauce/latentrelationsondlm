# Result format

Each completed run must write:

```text
results/<model>/<experiment>/
â”œâ”€â”€ config.yaml
â”œâ”€â”€ metrics.csv
â”œâ”€â”€ summary.json
â””â”€â”€ figures/
```

Use the same filenames and metric columns for every model so cross-model analysis can load them automatically.
