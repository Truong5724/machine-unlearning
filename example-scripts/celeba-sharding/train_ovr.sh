#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-celeba_ovr}
shard_spec=${2:-0-26}
dataset=${3:-datasets/celebA/datasetfile_ovr}

EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
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
echo "======================================================================"

IFS='-' read -r start_shard end_shard <<< "${shard_spec}"
if [[ -z "${start_shard}" || -z "${end_shard}" ]]; then
  echo "Invalid shard range: ${shard_spec}. Use start-end, e.g. 0-26"
  exit 1
fi

if (( start_shard < 0 || end_shard > 26 || start_shard > end_shard )); then
  echo "Shard range out of bounds: ${shard_spec}. Valid range is 0-26"
  exit 1
fi

for shard in $(seq "${start_shard}" "${end_shard}"); do
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
