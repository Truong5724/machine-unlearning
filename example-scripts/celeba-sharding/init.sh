#!/usr/bin/env bash
# init.sh - Initialize CelebA multitask SISA container

set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <number_of_shards> [scenario_label ...]"
    echo "Example: $0 5 0 100 500"
    exit 1
fi

shards="$1"
shift || true

labels=("$@")
if [[ ${#labels[@]} -eq 0 ]]; then
    labels=(0 100 500)
fi

echo "======================================================================"
echo "KHỞI TẠO CELEBA MULTITASK CONTAINER"
echo "======================================================================"
echo "Shards: ${shards}"
echo "Scenarios: ${labels[*]}"
echo ""

if [[ ! -f "datasets/celebA/datasetfile_multitask" ]]; then
    echo "❌ Missing datasetfile_multitask"
    echo "   Run: python datasets/celebA/prepare_data_multitask.py"
    exit 1
fi

if [[ ! -f "datasets/celebA/celeba_train.h5" || ! -f "datasets/celebA/celeba_test.h5" ]]; then
    echo "❌ Missing CelebA HDF5 files"
    echo "   Run: python datasets/celebA/prepare_data_multitask.py"
    exit 1
fi

mkdir -p "containers/celeba/cache" "containers/celeba/times" "containers/celeba/outputs"

echo "🔄 Creating shard partition..."
python distribution.py \
    --shards "${shards}" \
    --distribution uniform \
    --container "celeba" \
    --dataset "datasets/celebA/datasetfile_multitask" \
    --label 0

echo "✅ Created ${shards} shards"

echo ""
echo "🔄 Creating request files..."
for req in "${labels[@]}"; do
    "${PYTHON_BIN}" distribution.py \
        --requests "${req}" \
        --distribution uniform \
        --container "celeba" \
        --dataset "datasets/celebA/datasetfile_multitask" \
        --label "${req}"
    echo "   requestfile:${req}.npy"
done

echo ""
echo "✅ Init completed successfully!"
echo "Next step: bash example-scripts/celeba-sharding/train.sh ${shards}"
echo "======================================================================"