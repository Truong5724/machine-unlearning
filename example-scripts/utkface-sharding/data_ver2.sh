#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

label=${1:-}
container=${2:-utkface}
dataset=${3:-datasets/UTKFace/datasetfile_ver2}

if [[ -z "${label}" ]]; then
    echo "Usage: bash example-scripts/utkface-sharding/data_ver2.sh <label> [container] [datasetfile_ver2]"
    exit 1
fi

python aggregation_ver2.py \
  --container "${container}" \
  --label "${label}" \
  --dataset "${dataset}"
