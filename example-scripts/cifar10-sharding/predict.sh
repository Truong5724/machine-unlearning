#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

for i in $(seq 0 "$((${shards}-1))"); do
    for j in 50 100 500; do
        echo "shard: $((${i}+1))/${shards}, number of unlearning requests: ${j}"
        python sisa.py --model cifar10 --test --dataset datasets/CIFAR-10/datasetfile --label "${j}" --batch_size 16 --container "cifar10" --shard "${i}"
    done
done
