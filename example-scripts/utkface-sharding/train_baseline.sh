#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

BATCH_SIZE=64
EPOCHS=60
LR=0.001

for i in $(seq 0 $((${shards}-1))); do

    checkpoint="containers/utkface/cache/shard-${i}:0.pt"
    [[ -f "${checkpoint}" ]] && continue

    python sisa_utkface.py \
        --model utkface \
        --train \
        --slices 1 \
        --dataset datasets/UTKFace/datasetfile \
        --label 0 \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LR} \
        --optimizer adam \
        --container utkface \
        --shard "${i}"
done