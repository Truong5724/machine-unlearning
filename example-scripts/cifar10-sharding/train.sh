#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

for i in $(seq 0 "$((${shards}-1))"); do
    for j in 0; do
        echo "shard: $((${i}+1))/${shards}, number of unlearning requests: ${j}"
        python sisa_cifar10.py --model cifar10 --train --slices 1 --dataset datasets/CIFAR-10/datasetfile --label "${j}" --epochs 100 --batch_size 64 --learning_rate 0.05 --dropout_rate 0.2 --optimizer sgd --chkpt_interval 5 --container "cifar10" --shard "${i}"
    done
done

# Uncomment to run unlearning scenario: unlearn classes 0, 1, and 2
classes="0 1 2"
label="class_$(echo ${classes} | tr ' ' ',')"

for i in $(seq 0 "$((${shards}-1))"); do
    echo "shard: $((${i}+1))/${shards}, unlearning classes: ${label}"
    python sisa_cifar10.py --model cifar10 --train --slices 1 --dataset datasets/CIFAR-10/datasetfile --label "${label}" --epochs 100 --batch_size 64 --learning_rate 0.05 --dropout_rate 0.2 --optimizer sgd --chkpt_interval 5 --container "cifar10" --shard "${i}"
done