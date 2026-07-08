#!/bin/bash
# train.sh - Train SISA cho nhiều scenarios

set -eou pipefail
IFS=$'\n\t'

if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_shards>"
    exit 1
fi

shards=$1

# ===============================
# Cấu hình
# ===============================
BATCH_SIZE=64
EPOCHS=30
LEARNING_RATE=0.001
OPTIMIZER=adam
CHKPT_INTERVAL=5
SLICES=1

scenarios=(0 100 500)

echo "================================================================="
echo "🚀 TRAINING SISA UTKFace MULTITASK"
echo "Shards     : ${shards}"
echo "Scenarios  : ${scenarios[*]}"
echo "Epochs     : ${EPOCHS}"
echo "Batch size : ${BATCH_SIZE}"
echo "Slices     : ${SLICES}"
echo "================================================================="

LOG_FILE="containers/utkface/training.log"
echo "Training started at $(date)" > "${LOG_FILE}"

global_start=$(date +%s)

for label in "${scenarios[@]}"; do
    echo ""
    echo "=============================================================="
    echo "🔄 SCENARIO label=${label}"
    echo "=============================================================="

    scenario_start=$(date +%s)

    for i in $(seq 0 $((shards-1))); do
        echo "→ Training Shard $i / ${shards} (label=${label})"

        checkpoint="containers/utkface/cache/shard-${i}:${label}.pt"
        if [[ -f "${checkpoint}" ]]; then
            echo "   ✅ Already exists, skipping"
            continue
        fi

        python sisa_utkface_multitask.py \
            --train \
            --container utkface \
            --shard "${i}" \
            --label "${label}" \
            --dataset datasets/UTKFace/datasetfile_ver2 \
            --slices ${SLICES} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --optimizer ${OPTIMIZER} \
            --chkpt_interval ${CHKPT_INTERVAL} \
            2>&1 | tee -a "${LOG_FILE}"

        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            echo "❌ Error on shard ${i}, label ${label}"
            exit 1
        fi
    done

    scenario_end=$(date +%s)
    echo "✅ Scenario ${label} finished in $(( (scenario_end - scenario_start)/60 )) minutes"
done

global_end=$(date +%s)
total_time=$((global_end - global_start))

echo ""
echo "================================================================="
echo "🎉 ALL TRAINING COMPLETED!"
echo "Total time : $((total_time/3600))h $(( (total_time%3600)/60 ))m"
echo "================================================================="
echo "Next: Evaluate with"
echo "   python evaluate.py --container utkface --label 0"
echo "   python evaluate.py --container utkface --label 100"
echo "   python evaluate.py --container utkface --label 500"
echo "================================================================="