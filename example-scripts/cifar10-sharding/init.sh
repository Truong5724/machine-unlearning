#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
    
if [[ ! -d "containers/cifar10" ]] ; then
    mkdir "containers/cifar10"
    mkdir "containers/cifar10/cache"
    mkdir "containers/cifar10/times"
    mkdir "containers/cifar10/outputs"
    echo 0 > "containers/cifar10/times/null.time"
fi

# python distribution.py --shards "${shards}" --distribution uniform --container "cifar10" --dataset datasets/CIFAR-10/datasetfile --label 0

# for j in 0; do
#     python distribution.py --requests "${j}" --distribution uniform --container "cifar10" --dataset datasets/CIFAR-10/datasetfile --label "${j}"
# done

# Unlearn class scenario: unlearn classes 0, 1, and 2
classes=(0 1 2)
label="class_$(IFS=,; echo "${classes[*]}")"

python distribution.py \
  --requests 1 \
  --unlearn_class ${classes[@]} \
  --container "cifar10" \
  --dataset datasets/CIFAR-10/datasetfile \
  --label "${label}"
