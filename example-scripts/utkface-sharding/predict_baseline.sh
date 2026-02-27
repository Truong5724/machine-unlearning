#!/bin/bash

shards=$1

for i in $(seq 0 $((${shards}-1))); do
    python sisa_utkface.py \
        --model utkface \
        --test \
        --dataset datasets/UTKFace/datasetfile \
        --label 0 \
        --container utkface \
        --shard "${i}"
done