#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-celeba_ovr}
shard_spec=${2:-0-26}
dataset=${3:-datasets/celebA/datasetfile_ovr}

EPOCHS=${EPOCHS:-6}
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-0.0005}
OPTIMIZER=${OPTIMIZER:-adam}
CHKPT_INTERVAL=${CHKPT_INTERVAL:--1}
LOSS_MODE=${LOSS_MODE:-auto}
FOCAL_TASKS=${FOCAL_TASKS:-mustache,goatee,sideburns,double_chin,bags_under_eyes}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
FOCAL_ALPHA=${FOCAL_ALPHA:--1}
DROPOUT_RATE=${DROPOUT_RATE:-0.3}

echo "======================================================================"
echo "TRAIN CELEBA OVR"
echo "======================================================================"
echo "Container  : ${container}"
echo "Shards     : ${shard_spec}"
echo "Dataset    : ${dataset}"
echo "Epochs     : ${EPOCHS}"
echo "Batch size : ${BATCH_SIZE}"
echo "LR         : ${LEARNING_RATE}"
echo "Optimizer  : ${OPTIMIZER}"
echo "Loss mode  : ${LOSS_MODE}"
echo "Focal tasks: ${FOCAL_TASKS}"
echo "Unlearn    : drop-model (khong retrain shard bi bo)"
echo "======================================================================"

declare -A shard_seen=()
selected_shards=()

add_shard() {
  local s="$1"
  if (( s < 0 || s > 26 )); then
    echo "Shard out of bounds: ${s}. Valid range is 0-26"
    exit 1
  fi
  if [[ -z "${shard_seen[$s]+x}" ]]; then
    shard_seen[$s]=1
    selected_shards+=("$s")
  fi
}

parse_shard_spec() {
  local spec="$1"
  IFS=',' read -ra parts <<< "$spec"
  for part in "${parts[@]}"; do
    part="${part// /}"
    if [[ -z "$part" ]]; then
      continue
    fi

    if [[ "$part" =~ ^[0-9]+-[0-9]+$ ]]; then
      local start="${part%-*}"
      local end="${part#*-}"
      if (( start > end )); then
        echo "Invalid range '${part}': start > end"
        exit 1
      fi
      for s in $(seq "$start" "$end"); do
        add_shard "$s"
      done
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      add_shard "$part"
    else
      echo "Invalid shard spec token: '${part}'"
      echo "Use format: 0-2 or 0,3,5 or 0-2,8,10-12"
      exit 1
    fi
  done
}

parse_shard_spec "${shard_spec}"

if (( ${#selected_shards[@]} == 0 )); then
  echo "No shard selected from spec: ${shard_spec}"
  exit 1
fi

echo "Resolved  : ${selected_shards[*]}"

for shard in "${selected_shards[@]}"; do
  echo ""
  echo "--------------------------------------------------------------------"
  echo "Training shard=${shard}"
  echo "--------------------------------------------------------------------"

  python sisa_celeba_ovr.py \
    --container "${container}" \
    --dataset "${dataset}" \
    --shard "${shard}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --optimizer "${OPTIMIZER}" \
    --chkpt_interval "${CHKPT_INTERVAL}" \
    --loss_mode "${LOSS_MODE}" \
    --focal_tasks "${FOCAL_TASKS}" \
    --focal_gamma "${FOCAL_GAMMA}" \
    --focal_alpha "${FOCAL_ALPHA}" \
    --dropout_rate "${DROPOUT_RATE}"
done

echo ""
echo "Done: train OVR shards ${shard_spec}."
echo "Next: bash example-scripts/celeba-sharding/data_ovr.sh ${container} ${dataset}"
