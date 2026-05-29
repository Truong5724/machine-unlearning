#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface_ovr}
label=${2:-0}
shard_spec=${3:-0-9}

EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LEARNING_RATE:-0.0005}
OPTIMIZER=${OPTIMIZER:-adamw}
DROPOUT_RATE=${DROPOUT_RATE:-0.2}
CHKPT=${CHKPT_INTERVAL:-5}
LOSS_MODE=${LOSS_MODE:-auto}
FOCAL_TASKS=${FOCAL_TASKS:-race_others,age_bin2}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
FOCAL_ALPHA=${FOCAL_ALPHA:--1}
USE_SCHEDULER=${USE_SCHEDULER:-1}
SCHEDULER_FACTOR=${SCHEDULER_FACTOR:-0.5}
SCHEDULER_PATIENCE=${SCHEDULER_PATIENCE:-2}
SCHEDULER_MIN_LR=${SCHEDULER_MIN_LR:-0.00001}

echo "======================================================================"
echo "TRAIN UTKFACE OVR"
echo "======================================================================"
echo "Container : ${container}"
echo "Label     : ${label}"
echo "Shards    : ${shard_spec}"
echo "Epochs    : ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Optimizer : ${OPTIMIZER}"
echo "LR        : ${LR}"
echo "Dropout   : ${DROPOUT_RATE}"
echo "Loss mode : ${LOSS_MODE}"
echo "Focal task: ${FOCAL_TASKS}"
echo "Scheduler : ${USE_SCHEDULER}"
echo "======================================================================"

IFS='-' read -r start_shard end_shard <<< "${shard_spec}"

if [[ -z "${start_shard}" || -z "${end_shard}" ]]; then
  echo "Invalid shard range: ${shard_spec}. Use format start-end, e.g. 0-2"
  exit 1
fi

if (( start_shard < 0 || end_shard > 9 || start_shard > end_shard )); then
  echo "Shard range out of bounds: ${shard_spec}. Valid range is 0-9"
  exit 1
fi

for shard in $(seq "${start_shard}" "${end_shard}"); do
  echo ""
  echo "--------------------------------------------------------------------"
  echo "Training shard=${shard} label=${label}"
  echo "--------------------------------------------------------------------"

  python sisa_utkface_ovr.py \
    --train \
    --container "${container}" \
    --dataset datasets/UTKFace/datasetfile_ovr \
    --shard "${shard}" \
    --label "${label}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --optimizer "${OPTIMIZER}" \
    --dropout_rate "${DROPOUT_RATE}" \
    --chkpt_interval "${CHKPT}" \
    --use_scheduler \
    --loss_mode "${LOSS_MODE}" \
    --focal_tasks "${FOCAL_TASKS}" \
    --focal_gamma "${FOCAL_GAMMA}" \
    --focal_alpha "${FOCAL_ALPHA}" \
    --scheduler_factor "${SCHEDULER_FACTOR}" \
    --scheduler_patience "${SCHEDULER_PATIENCE}" \
    --scheduler_min_lr "${SCHEDULER_MIN_LR}"
done

echo ""
echo "✅ TRAIN OVR DONE (shards ${shard_spec})"
echo "======================================================================"

