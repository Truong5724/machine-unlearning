#!/usr/bin/env bash
# predict.sh - Predict CelebA multitask test set per shard

set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <number_of_shards> [label]"
    echo "Example: $0 5 0"
    exit 1
fi

shards="$1"
label="${2:-0}"
BATCH_SIZE=128

LOG_FILE="containers/celeba/prediction.log"
echo "Prediction started at $(date)" > "${LOG_FILE}"

echo "======================================================================"
echo "CELEBA MULTITASK PREDICTION"
echo "======================================================================"
echo "Shards     : ${shards}"
echo "Label      : ${label}"
echo "Batch size : ${BATCH_SIZE}"
echo "======================================================================"

start_time=$(date +%s)

for i in $(seq 0 $((shards - 1))); do
    checkpoint="containers/celeba/cache/shard-${i}:${label}.pt"
    output_file="containers/celeba/outputs/shard-${i}:${label}.npy"

    if [[ -f "${output_file}" ]]; then
        echo "✅ Skip existing output: ${output_file}"
        continue
    fi

    if [[ ! -f "${checkpoint}" ]]; then
        echo "❌ Missing checkpoint: ${checkpoint}"
        echo "   Run train.sh first."
        exit 1
    fi

    echo ""
    echo "→ Predicting shard ${i}/${shards}"

    "${PYTHON_BIN}" sisa_celebA_multitask.py \
        --test \
        --container "celeba" \
        --dataset "datasets/celebA/datasetfile_multitask" \
        --shard "${i}" \
        --label "${label}" \
        --batch_size "${BATCH_SIZE}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "❌ Error predicting shard ${i}"
        exit 1
    fi
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "======================================================================"
echo "✅ Prediction completed"
echo "Elapsed: $((elapsed / 60))m $((elapsed % 60))s"
echo "Outputs: containers/celeba/outputs/"
echo "Log file: ${LOG_FILE}"
echo "======================================================================"
