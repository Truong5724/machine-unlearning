#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

if [[ ! -f cifar10-general-report.csv ]]; then
    echo "nb_shards,nb_requests,accuracy,retraining_time" > cifar10-general-report.csv
fi

for j in 100; do
    acc=$(python aggregation.py --strategy uniform --container "cifar10" --shards "${shards}" --dataset datasets/CIFAR-10/datasetfile --label "${j}")
    cat containers/cifar10/times/shard-*:"${j}".time > "containers/cifar10/times/times.tmp"
    time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
    echo "${shards},${j},${acc},${time}" >> cifar10-general-report.csv
done
