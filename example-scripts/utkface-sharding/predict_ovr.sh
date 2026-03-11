#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface_ovr}
label=${2:-0}
shard_spec=${3:-0-9}
eval_split=${4:-test}

BATCH_SIZE=${BATCH_SIZE:-128}

echo "======================================================================"
echo "PREDICT UTKFACE OVR"
echo "======================================================================"
echo "Container : ${container}"
echo "Label     : ${label}"
echo "Shards    : ${shard_spec}"
echo "Split     : ${eval_split}"
echo "Batch size: ${BATCH_SIZE}"
echo "======================================================================"

if [[ "${eval_split}" != "val" && "${eval_split}" != "test" ]]; then
  echo "Invalid eval split: ${eval_split}. Use val or test"
  exit 1
fi

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
  echo "Predict shard=${shard}"
  python sisa_utkface_ovr.py \
    --test \
    --container "${container}" \
    --dataset datasets/UTKFace/datasetfile_ovr \
    --shard "${shard}" \
    --label "${label}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_split "${eval_split}"
done

echo ""
  echo "✅ PREDICT OVR DONE (shards ${shard_spec}, split ${eval_split})"
echo "======================================================================"

