#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface_ovr}
label=${2:-0}

echo "Init UTKFace OVR SISA for container=${container}, label=${label}"

mkdir -p containers/${container}/{cache,times,outputs}
echo 0 > containers/${container}/times/null.time

python utkface_ovr_partition.py \
  --container ${container} \
  --dataset datasets/UTKFace/datasetfile_ovr \
  --label ${label}

echo "✅ Done. 10 shards created (OVR heads)."

