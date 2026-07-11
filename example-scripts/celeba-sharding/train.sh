#!/bin/bash
# train_celeba.sh - Train CelebA SISA

set -eou pipefail
IFS=$'\n\t'

shards=$1

BATCH_SIZE=64
EPOCHS=30
LEARNING_RATE=0.001
OPTIMIZER=adam
CHKPT_INTERVAL=5
SLICES=1

scenarios=(0 100 500)

echo "================================================================="
echo "🚀 TRAINING CELEBA MULTITASK SISA"
echo "Shards     : ${shards}"
echo "Scenarios  : ${scenarios[*]}"
echo "Epochs     : ${EPOCHS}"
echo "Batch size : ${BATCH_SIZE}"
echo "================================================================="

LOG_FILE="containers/celeba/training.log"
echo "Training started at $(date)" > "${LOG_FILE}"

for label in "${scenarios[@]}"; do
    echo ""
    echo "=============================================================="
    echo "SCENARIO label=${label}"
    echo "=============================================================="

    for i in $(seq 0 $((shards-1))); do
        echo "→ Training Shard $i (label=${label})"

        checkpoint="containers/celeba/cache/shard-${i}:${label}.pt"
        if [[ -f "${checkpoint}" ]]; then
            echo "   ✅ Already exists, skipping"
            continue
        fi

        python sisa_celebA_multitask.py \
            --train \
            --container celeba \
            --shard "${i}" \
            --label "${label}" \
            --dataset datasetfile_celeba \
            --slices ${SLICES} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --optimizer ${OPTIMIZER} \
            --chkpt_interval ${CHKPT_INTERVAL} \
            2>&1 | tee -a "${LOG_FILE}"

        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            echo "❌ Error on shard ${i}"
            exit 1
        fi
    done

    echo "✅ Scenario ${label} finished"
done

echo ""
echo "================================================================="
echo "✅ ALL CELEBA TRAINING COMPLETED!"
echo "================================================================="