#!/bin/bash
# init.sh - Init SISA với 3 scenarios (0, 100, 500)

set -eou pipefail
IFS=$'\n\t'

if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_shards>"
    echo "Example: $0 5"
    exit 1
fi

shards=$1

echo "================================================================="
echo "🚀 Init SISA UTKFace - ${shards} shards + 3 scenarios"
echo "================================================================="

# Kiểm tra dataset
[[ -f datasets/UTKFace/datasetfile_ver2 ]] || { echo "❌ Datasetfile not found!"; exit 1; }

# Tạo thư mục
mkdir -p containers/utkface/{cache,times,outputs}

echo "📦 Creating shards and request files..."

# Tạo shards (label 0 = no unlearning)
python distribution_safe.py \
    --shards "${shards}" \
    --distribution uniform \
    --container utkface \
    --dataset datasets/UTKFace/datasetfile_ver2 \
    --label 0

# Tạo request files cho các scenario
for req in 0 100 500; do
    python distribution_safe.py \
        --requests "${req}" \
        --distribution uniform \
        --container utkface \
        --dataset datasets/UTKFace/datasetfile_ver2 \
        --label "${req}"
    echo "✅ Created requestfile for ${req} samples"
done

echo "================================================================="
echo "✅ Init completed successfully!"
echo "Next step: ./train.sh ${shards}"
echo "================================================================="