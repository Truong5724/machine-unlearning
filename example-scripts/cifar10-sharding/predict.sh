#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

for i in $(seq 0 "$((${shards}-1))"); do
    for j in 0; do
        echo "shard: $((${i}+1))/${shards}, number of unlearning requests: ${j}"
        python sisa_cifar10.py --model cifar10 --test --dataset datasets/CIFAR-10/datasetfile --label "${j}" --batch_size 16 --container "cifar10" --shard "${i}"
    done
done

# Uncomment to run unlearning scenario: unlearn classes 0, 1, and 2
classes="0 1 2"
label="class_$(echo ${classes} | tr ' ' ',')"

for i in $(seq 0 "$((${shards}-1))"); do
    echo "shard: $((${i}+1))/${shards}, unlearning classes: ${label}"
    python sisa_cifar10.py --model cifar10 --test --dataset datasets/CIFAR-10/datasetfile --label "${label}" --batch_size 16 --container "cifar10" --shard "${i}"
done