#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface}
label=${2:-0}
output_type=${3:-argmax}

for shard in 0 1 2; do
  echo "Predict shard=${shard} label=${label}"
  python sisa_utkface_multitask.py \
    --test \
    --container ${container} \
    --dataset datasets/UTKFace/datasetfile_ver2 \
    --shard ${shard} \
    --label ${label} \
    --output_type ${output_type}
done

echo "Prediction done for all multitask shards"
