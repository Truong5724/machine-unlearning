#!/bin/bash
# train.sh - Train với nhiều unlearning scenarios

set -eou pipefail
IFS=$'\n\t'

shards=$1

# ===============================
# ⚙️ Cấu hình train
# ===============================
BATCH_SIZE=64
EPOCHS=30
LEARNING_RATE=0.001
OPTIMIZER=adam
CHKPT_INTERVAL=5

# Các scenarios cần train
scenarios=(0 100 500)

echo "======================================================================"
echo "TRAINING UTKFACE - MULTIPLE SCENARIOS"
echo "======================================================================"
echo "Shards: ${shards}"
echo "Scenarios: ${scenarios[@]}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "======================================================================"
echo ""

LOG_FILE="containers/utkface/training.log"
echo "Training started at $(date)" > "${LOG_FILE}"

global_start=$(date +%s)

# ===============================
# 🔄 Train từng scenario
# ===============================
for label in "${scenarios[@]}"; do
    
    echo ""
    echo "======================================================================"
    echo "SCENARIO: label=${label}"
    echo "======================================================================"
    echo ""
    
    scenario_start=$(date +%s)
    
    # Train tất cả shards với label này
    for i in $(seq 0 "$((${shards}-1))"); do
        
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Scenario ${label} - Shard $((${i}+1))/${shards}"
        echo "----------------------------------------------------------------------"
        
        checkpoint="containers/utkface/cache/shard-${i}:${label}.pt"
        
        if [[ -f "${checkpoint}" ]]; then
            echo "✅ Checkpoint exists, skip!"
            continue
        fi
        
        echo "🔄 Training shard ${i} with label ${label}..."
        
        python sisa_utkface.py \
            --model utkface \
            --train \
            --slices 1 \
            --dataset datasets/UTKFace/datasetfile \
            --label ${label} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --optimizer ${OPTIMIZER} \
            --chkpt_interval ${CHKPT_INTERVAL} \
            --container "utkface" \
            --shard "${i}" \
            2>&1 | tee -a "${LOG_FILE}"
        
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            echo "❌ Error training shard ${i}, label ${label}"
            exit 1
        fi
        
        echo "✅ Done"
    done
    
    scenario_end=$(date +%s)
    scenario_time=$((${scenario_end} - ${scenario_start}))
    
    echo ""
    echo "✅ Scenario ${label} complete: $((${scenario_time} / 60))m"
    echo ""
done

global_end=$(date +%s)
total_time=$((${global_end} - ${global_start}))

echo ""
echo "======================================================================"
echo "✅ ALL TRAINING COMPLETE!"
echo "======================================================================"
echo "Total time: $((${total_time} / 3600))h $((${total_time} % 3600 / 60))m"
echo "Trained scenarios: ${scenarios[@]}"
echo ""
echo "Next steps:"
echo "  ./predict.sh ${shards} 0"
echo "  ./predict.sh ${shards} 100"
echo "  ./predict.sh ${shards} 500"
echo "======================================================================"