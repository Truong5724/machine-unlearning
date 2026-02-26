#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

for i in $(seq 0 "$((${shards}-1))"); do
    for j in 100; do
        echo "shard: $((${i}+1))/${shards}, number of unlearning requests: ${j}"
        python sisa_cifar10.py --model cifar10 --train --slices 1 --dataset datasets/CIFAR-10/datasetfile --label "${j}" --epochs 100 --batch_size 64 --learning_rate 0.1 --dropout_rate 0.2 --optimizer sgd --chkpt_interval 1 --container "cifar10" --shard "${i}"
    done
done