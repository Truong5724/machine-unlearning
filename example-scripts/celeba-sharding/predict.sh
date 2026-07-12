#!/bin/bash
# predict_celeba.sh - Predict CelebA Multitask

set -euo pipefail
IFS=$'\n\t'

shards=$1
label=${2:-0}
container=${3:-celeba}
dataset=${4:-datasets/celebA/datasetfile_celeba}
batch_size=${5:-128}

echo "======================================================================"
echo "PREDICT CELEBA MULTITASK"
echo "======================================================================"
echo "Shards     : ${shards}"
echo "Label      : ${label}"
echo "Container  : ${container}"
echo "Dataset    : ${dataset}"
echo "Batch size : ${batch_size}"
echo "======================================================================"

LOG_FILE="containers/${container}/prediction.log"
echo "Prediction started at $(date)" > "${LOG_FILE}"

start_time=$(date +%s)

for ((i=0;i<shards;i++)); do

    checkpoint="containers/${container}/cache/shard-${i}:${label}.pt"
    output_file="containers/${container}/outputs/shard-${i}:${label}.npy"

    if [[ ! -f "${checkpoint}" ]]; then
        echo "❌ Missing checkpoint: ${checkpoint}"
        exit 1
    fi

    if [[ -f "${output_file}" ]]; then
        echo "✅ Skip existing output: ${output_file}"
        continue
    fi

    echo ""
    echo "=============================================================="
    echo "Predicting shard ${i}/${shards}"
    echo "=============================================================="

    python sisa_celebA_multitask.py \
        --test \
        --container "${container}" \
        --dataset "${dataset}" \
        --shard "${i}" \
        --label "${label}" \
        --batch_size "${batch_size}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "❌ Prediction failed on shard ${i}"
        exit 1
    fi
done

end_time=$(date +%s)
elapsed=$((end_time-start_time))

echo ""
echo "======================================================================"
echo "✅ ALL PREDICTIONS COMPLETED"
echo "======================================================================"
echo "Elapsed : $((elapsed/60))m $((elapsed%60))s"
echo "Outputs : containers/${container}/outputs/"
echo "Log     : ${LOG_FILE}"
echo "======================================================================"