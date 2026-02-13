#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

for i in $(seq 0 "$((${shards}-1))"); do
    for j in {0..15}; do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/16"
        r=$((${j}*${shards}/5))
        python sisa.py --model celeba --train --slices 1 --dataset datasets/celebA/datasetfile --label "${r}" --epochs 30 --batch_size 32 --learning_rate 0.0001 --optimizer adam --chkpt_interval 5 --container "celeba" --shard "${i}"
    done
done
