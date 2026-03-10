#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface}
label=${2:-0}

echo "Init UTKFace multitask SISA for container=${container}, label=${label}"

mkdir -p containers/${container}/{cache,times,outputs}
echo 0 > containers/${container}/times/null.time

python utkface_multitask_partition.py \
  --container ${container} \
  --dataset datasets/UTKFace/datasetfile_ver2 \
  --label ${label}

echo "Done. 3 shards created: 0=gender(2 slices), 1=age(3 slices), 2=race(5 slices)."
