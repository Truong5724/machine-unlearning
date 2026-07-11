#!/bin/bash
# init_celeba.sh

set -eou pipefail
IFS=$'\n\t'

shards=$1
samples=${2:-30000}

echo "================================================================="
echo "🚀 Init SISA CelebA - ${shards} shards | ${samples} samples"
echo "================================================================="

DATASET_FILE="datasets/celebA/datasetfile_celeba"

if [[ ! -f "$DATASET_FILE" ]]; then
    echo "❌ Không tìm thấy $DATASET_FILE"
    ls -l datasets/celebA/
    exit 1
fi

echo "📄 Sử dụng: $DATASET_FILE"

mkdir -p containers/celeba/{cache,times,outputs}

# Partition
echo "📊 Running partition ${samples} samples..."
python partition_celebA.py \
    --container celeba \
    --dataset "$DATASET_FILE" \
    --slices_per_shard "${shards}" \
    --samples "${samples}"

# Request files
for req in 0 100 500; do
    if [ "$req" -eq 0 ]; then
        python - <<EOF
import numpy as np
shards_arr = np.load("containers/celeba/splitfile.npy", allow_pickle=True)
requests = [np.array([], dtype=np.int64) for _ in range(len(shards_arr))]
np.save("containers/celeba/requestfile:0.npy", np.array(requests, dtype=object))
print("✅ Created requestfile:0")
EOF
    else
        python distribution_safe.py \
            --requests "${req}" \
            --distribution uniform \
            --container celeba \
            --dataset "$DATASET_FILE" \
            --label "${req}"
    fi
done

echo "================================================================="
echo "✅ Init CelebA completed!"
echo "Next: ./train_celeba.sh ${shards}"
echo "================================================================="