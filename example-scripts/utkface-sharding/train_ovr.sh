#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface_ovr}
label=${2:-0}
shard_spec=${3:-0-9}

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
echo "Shards    : ${shard_spec}"
echo "Epochs    : ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
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
    --chkpt_interval "${CHKPT}"
done

echo ""
echo "✅ TRAIN OVR DONE (shards ${shard_spec})"
echo "======================================================================"

