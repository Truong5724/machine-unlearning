#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface}
label=${2:-0}

epochs=${EPOCHS:-40}
batch_size=${BATCH_SIZE:-64}
lr=${LEARNING_RATE:-0.001}
optimizer=${OPTIMIZER:-adam}
chkpt=${CHKPT_INTERVAL:-5}

for shard in 0 1 2; do
  echo "Training shard=${shard} label=${label}"
  python sisa_utkface_multitask.py \
    --train \
    --container ${container} \
    --dataset datasets/UTKFace/datasetfile_ver2 \
    --shard ${shard} \
    --label ${label} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --learning_rate ${lr} \
    --optimizer ${optimizer} \
    --chkpt_interval ${chkpt}
done

echo "Training done for all multitask shards"
