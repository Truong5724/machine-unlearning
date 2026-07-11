#!/usr/bin/env bash
# train.sh - Train CelebA multitask SISA shards

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

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
OPTIMIZER="${OPTIMIZER:-adam}"
CHKPT_INTERVAL="${CHKPT_INTERVAL:-5}"
SLICES="${SLICES:-3}"

LOG_FILE="containers/celeba/training.log"
echo "Training started at $(date)" > "${LOG_FILE}"

mkdir -p "containers/celeba/cache" "containers/celeba/times" "containers/celeba/outputs"

echo "======================================================================"
echo "CELEBA MULTITASK TRAINING"
echo "======================================================================"
echo "Shards     : ${shards}"
echo "Label      : ${label}"
echo "Slices     : ${SLICES}"
echo "Epochs     : ${EPOCHS}"
echo "Batch size : ${BATCH_SIZE}"
echo "Learning   : ${LEARNING_RATE}"
echo "Optimizer  : ${OPTIMIZER}"
echo "======================================================================"

start_time=$(date +%s)
for i in $(seq 0 $((shards - 1))); do
    checkpoint="containers/celeba/cache/shard-${i}:${label}.pt"
    if [[ -f "${checkpoint}" ]]; then
        echo "✅ Skip existing checkpoint: ${checkpoint}"
        continue
    fi

    echo ""
    echo "→ Training shard ${i}/${shards} (label=${label})"

    "${PYTHON_BIN}" sisa_celebA_multitask.py \
        --train \
        --container "celeba" \
        --dataset "datasets/celebA/datasetfile_multitask" \
        --shard "${i}" \
        --label "${label}" \
        --slices "${SLICES}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --optimizer "${OPTIMIZER}" \
        --chkpt_interval "${CHKPT_INTERVAL}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "❌ Error while training shard ${i}"
        exit 1
    fi
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "======================================================================"
echo "✅ Training completed"
echo "Elapsed: $((elapsed / 60))m $((elapsed % 60))s"
echo "Checkpoints: containers/celeba/cache/"
echo "Log file: ${LOG_FILE}"
echo "======================================================================"
