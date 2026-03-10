#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface_ovr}
label=${2:-0}

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LEARNING_RATE:-0.001}
OPTIMIZER=${OPTIMIZER:-adam}
CHKPT=${CHKPT_INTERVAL:-5}

echo "======================================================================"
echo "TRAIN UTKFACE OVR"
echo "======================================================================"
echo "Container : ${container}"
echo "Label     : ${label}"
echo "Epochs    : ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "======================================================================"

for shard in $(seq 0 9); do
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
    --chkpt_interval "${CHKPT}"
done

echo ""
echo "✅ TRAIN OVR DONE (10 shards)"
echo "======================================================================"

