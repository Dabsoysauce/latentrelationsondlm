#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 RUN_ID SHARED_RESULTS_ROOT LOCAL_RESULTS_ROOT [WORKERS=48]" >&2
  exit 2
fi

run_id=$1
shared_results_root=$2
local_results_root=$3
worker_count=${4:-48}
python_bin=${PYTHON_BIN:-python3}
model_config=${MODEL_CONFIG:-configs/models/dream_7b.yaml}
dataset_config=${DATASET_CONFIG:-configs/datasets/ewt.yaml}
experiment_config=${EXPERIMENT_CONFIG:-configs/experiments/pos_token_class_linear_probes.yaml}
relative_run="exploratory_extensions/dream_7b/ewt/pos_token_class_linear_probes/${run_id}"
shared_run_dir="${shared_results_root}/${relative_run}"
local_run_dir="${local_results_root}/${relative_run}"
mirror_dir="${shared_run_dir}/fit_checkpoints"
log_dir="${local_run_dir}/logs/pos_fit_shards"

if [[ "${shared_run_dir}" == "${local_run_dir}" ]]; then
  echo "shared and local run directories must differ" >&2
  exit 2
fi
if [[ ! -f "${shared_run_dir}/pos_extract_status.json" ]]; then
  echo "missing completed extraction status: ${shared_run_dir}/pos_extract_status.json" >&2
  exit 1
fi

mkdir -p "${local_run_dir}" "${mirror_dir}" "${log_dir}"
echo "Staging validated extraction artifacts from shared storage to local SSD."
if ! rsync -a "${shared_run_dir}/" "${local_run_dir}/"; then
  echo "local SSD staging failed" >&2
  exit 1
fi

sync_small_artifacts() {
  rsync -a --exclude 'checkpoints/' "${local_run_dir}/" "${shared_run_dir}/"
}

echo "Launching ${worker_count} one-thread POS fit workers."
echo "Monitor: find '${mirror_dir}' -name '*.parquet' | wc -l"
pids=()
for ((index = 0; index < worker_count; index++)); do
  log_path="${log_dir}/shard-${index}-of-${worker_count}.log"
  (
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    "${python_bin}" -m dlmrel.cli run \
      --model "${model_config}" \
      --dataset "${dataset_config}" \
      --experiment "${experiment_config}" \
      --results "${local_results_root}" \
      --run-id "${run_id}" \
      --resume \
      --pos-stage fit \
      --pos-fit-shard-count "${worker_count}" \
      --pos-fit-shard-index "${index}" \
      --pos-fit-checkpoint-mirror "${mirror_dir}"
  ) >"${log_path}" 2>&1 &
  pid=$!
  pids+=("${pid}")
  echo "worker ${index}: pid=${pid} log=${log_path}"
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    echo "worker ${index} failed; inspect ${log_dir}/shard-${index}-of-${worker_count}.log" >&2
    failed=1
  fi
done
sync_small_artifacts
if ((failed)); then
  echo "At least one worker failed. Rerun this command to resume missing checkpoints." >&2
  exit 1
fi

echo "All workers finished. Verifying every expected key and reducing final artifacts."
if ! "${python_bin}" -m dlmrel.cli run \
  --model "${model_config}" \
  --dataset "${dataset_config}" \
  --experiment "${experiment_config}" \
  --results "${local_results_root}" \
  --run-id "${run_id}" \
  --resume \
  --pos-stage fit \
  --pos-fit-aggregate-only \
  --pos-fit-checkpoint-mirror "${mirror_dir}"; then
  sync_small_artifacts
  echo "reducer failed closed; no complete summary was published" >&2
  exit 1
fi
sync_small_artifacts
echo "POS fit and reducer complete; final artifacts were synchronized to ${shared_run_dir}."
