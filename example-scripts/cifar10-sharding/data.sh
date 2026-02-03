#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

if [[ ! -f general-report.csv ]]; then
    echo "nb_shards,nb_requests,accuracy,retraining_time" > general-report.csv
fi

for j in {0..15}; do
    r=$((${j}*${shards}/5))
    acc=$(python aggregation.py --strategy uniform --container "cifar10" --shards "${shards}" --dataset datasets/CIFAR-10/datasetfile --label "${r}")
    cat containers/cifar10/times/shard-*:"${r}".time > "containers/cifar10/times/times.tmp"
    time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
    echo "${shards},${r},${acc},${time}" >> general-report.csv
done
