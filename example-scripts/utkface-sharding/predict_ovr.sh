#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface_ovr}
label=${2:-0}

BATCH_SIZE=${BATCH_SIZE:-128}

echo "======================================================================"
echo "PREDICT UTKFACE OVR"
echo "======================================================================"
echo "Container : ${container}"
echo "Label     : ${label}"
echo "Batch size: ${BATCH_SIZE}"
echo "======================================================================"

for shard in $(seq 0 9); do
  echo ""
  echo "Predict shard=${shard}"
  python sisa_utkface_ovr.py \
    --test \
    --container "${container}" \
    --dataset datasets/UTKFace/datasetfile_ovr \
    --shard "${shard}" \
    --label "${label}" \
    --batch_size "${BATCH_SIZE}"
done

echo ""
echo "✅ PREDICT OVR DONE"
echo "======================================================================"

