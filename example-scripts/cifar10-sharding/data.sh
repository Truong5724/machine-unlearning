#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

if [[ ! -f cifar10-general-report.csv ]]; then
    echo "nb_shards, nb_requests, forgot_shard_id, retained_accuracy, unlearning_accuracy, retained_precision_macro, retained_recall_macro, retained_f1_macro, training_time" > cifar10-general-report.csv
fi

for j in 0; do
    acc=$(python aggregation.py --strategy proportional --container "cifar10" --shards "${shards}" --dataset datasets/CIFAR-10/datasetfile --label "${j}" | tail -n 1)
    cat containers/cifar10/times/shard-*:"${j}".time > "containers/cifar10/times/times.tmp"
    time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
    echo "${shards},${j},None,${acc},${time}" >> cifar10-general-report.csv
done

# Uncomment to run unlearning scenario: unlearn classes 0, 1, and 2
classes=(0 1 2)
label="class_$(IFS=,; echo "${classes[*]}")"

acc=$(python aggregation.py --strategy proportional --container "cifar10" --shards "${shards}" --dataset datasets/CIFAR-10/datasetfile --label "${label}" --unlearn_shards "${classes[@]}" | tail -n 1)
cat containers/cifar10/times/shard-*:"${label}".time > "containers/cifar10/times/times.tmp"
time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
echo "${shards},None,${label},${acc},${time}" >> cifar10-general-report.csv
