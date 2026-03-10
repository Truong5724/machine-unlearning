#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

label=${1:-0}
container=${2:-utkface_ovr}
dataset=${3:-datasets/UTKFace/datasetfile_ovr}

if [[ -z "${label}" ]]; then
  echo "Usage: bash example-scripts/utkface-sharding/data_ovr.sh <label> [container] [datasetfile_ovr]"
  exit 1
fi

python aggregation_ovr.py \
  --container "${container}" \
  --label "${label}" \
  --dataset "${dataset}"

