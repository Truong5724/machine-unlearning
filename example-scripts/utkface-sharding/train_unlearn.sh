#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

BATCH_SIZE=64
EPOCHS=60
LR=0.001

python sisa.py \
    --model utkface \
    --train \
    --slices 1 \
    --dataset datasets/UTKFace/datasetfile \
    --label 100 \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --optimizer adam \
    --container utkface \
    --shard 0