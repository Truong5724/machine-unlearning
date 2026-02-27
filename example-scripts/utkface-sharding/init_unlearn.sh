#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

[[ -f datasets/UTKFace/datasetfile ]] || exit 1
[[ -f datasets/UTKFace/utkface_train.h5 ]] || exit 1

mkdir -p containers/utkface/{cache,times,outputs,shards}
echo 0 > containers/utkface/times/null.time

python distribution.py \
    --shards "${shards}" \
    --distribution uniform \
    --container utkface \
    --dataset datasets/UTKFace/datasetfile \
    --label 0