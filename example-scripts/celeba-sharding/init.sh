#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
    
if [[ ! -d "containers/celeba" ]] ; then
    mkdir -p "containers/celeba"
    mkdir -p "containers/celeba/cache"
    mkdir -p "containers/celeba/times"
    mkdir -p "containers/celeba/outputs"
    echo 0 > "containers/celeba/times/null.time"
fi

python distribution.py --shards "${shards}" --distribution uniform --container "celeba" --dataset datasets/celebA/datasetfile --label 0

for j in {1..15}; do
    r=$((${j}*${shards}/5))
    python distribution.py --requests "${r}" --distribution uniform --container "celeba" --dataset datasets/celebA/datasetfile --label "${r}"
done
