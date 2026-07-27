#!/bin/bash
# predict.sh - Predict trên test set cho từng shard

set -eou pipefail
IFS=$'\n\t'

if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_shards> [label]"
    echo ""
    echo "Examples:"
    echo "  Random unlearning:"
    echo "    $0 5 100"
    echo ""
    echo "  Forget Female:"
    echo "    $0 5 forget_gender_0"
    echo ""
    echo "  Forget Male:"
    echo "    $0 5 forget_gender_1"
    exit 1
fi

shards=$1
label=${2:-0}
BATCH_SIZE=128

echo "======================================================================"
echo "PREDICTION - UTKFACE MULTITASK"
echo "======================================================================"
echo "Shards     : ${shards}"
echo "Label      : ${label}"
echo "Batch size : ${BATCH_SIZE}"
echo "======================================================================"

LOG_FILE="containers/utkface/prediction.log"
echo "Prediction started at $(date)" > "${LOG_FILE}"

start_time=$(date +%s)

for i in $(seq 0 $((shards-1))); do
    echo ""
    echo "→ Shard $((i+1))/${shards} (label=${label})"
    
    checkpoint="containers/utkface/cache/shard-${i}:${label}.pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "❌ Checkpoint missing: ${checkpoint}"
        echo "   Please run train first!"
        exit 1
    fi

    python sisa_utkface_multitask.py \
        --test \
        --container utkface \
        --shard "${i}" \
        --label "${label}" \
        --dataset datasets/UTKFace/datasetfile_ver2 \
        --batch_size ${BATCH_SIZE} \
        2>&1 | tee -a "${LOG_FILE}"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "❌ Error predicting shard ${i}"
        exit 1
    fi

    echo "✅ Shard ${i} prediction done"
done

end_time=$(date +%s)
total