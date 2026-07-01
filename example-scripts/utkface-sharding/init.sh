#!/bin/bash
# init_simple.sh - Init với 3 scenarios đơn giản

set -eou pipefail
IFS=$'\n\t'

shards=$1

echo "Init SISA: ${shards} shards + 3 unlearning scenarios"

# Check
[[ -f datasets/UTKFace/datasetfile ]] || { echo "❌ Dataset not found"; exit 1; }

# Setup
mkdir -p containers/utkface/{cache,times,outputs,shards}
echo 0 > containers/utkface/times/null.time

# Create shards
python distribution_safe.py --shards "${shards}" --distribution uniform \
    --container utkface --dataset datasets/UTKFace/datasetfile --label 0

# Create requests
for requests in 0 100 500; do
    python distribution_safe.py --requests "${requests}" --distribution uniform \
        --container utkface --dataset datasets/UTKFace/datasetfile --label "${requests}"
    echo "✅ Created requestfile:${requests}.npy"
done

echo "✅ Done! Use --label 0/100/500 for training"