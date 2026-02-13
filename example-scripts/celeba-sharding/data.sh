#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

if [[ ! -f celeba-report.csv ]]; then
    echo "nb_shards,nb_requests,accuracy,retraining_time" > celeba-report.csv
fi

for j in {0..15}; do
    r=$((${j}*${shards}/5))
    acc=$(python aggregation.py --strategy uniform --container "celeba" --shards "${shards}" --dataset datasets/celebA/datasetfile --label "${r}")
    cat containers/celeba/times/shard-*:"${r}".time > "containers/celeba/times/times.tmp"
    time=$(python time_stats.py --container "celeba" | awk -F ',' '{print $1}')
    echo "${shards},${r},${acc},${time}" >> celeba-report.csv
done

echo "Results saved to celeba-report.csv"
