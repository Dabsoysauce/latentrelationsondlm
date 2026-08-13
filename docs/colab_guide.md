# Colab guide

Open `notebooks/colab_runner.ipynb`, select an A100/high-memory runtime for 7B/8B
models, and mount Drive only when persistence is needed. The notebook installs
one model environment, prints GPU diagnostics, runs `model smoke-test
--dry-run`, and then invokes the same `dlmrel` CLI. Keep Hugging Face tokens in
Colab Secrets (`HF_TOKEN`); never paste them into the notebook. Copy the entire
run directory, including shards and checksums, to Drive before disconnecting.
Resume with the same model/dataset/experiment plus `--run-id <id> --resume`.
Large five-seed curves can consume many GPU-hours; validate a tiny smoke shard
before starting a full run.
