#!/bin/bash
# predict_fast.sh - Prediction cho FAST MODE

set -eou pipefail
IFS=$'\n\t'

shards=$1
BATCH_SIZE=16
scenarios=(0 5 10 15)

echo "======================================================================"
echo "PREDICTION - FAST MODE"
echo "======================================================================"

LOG_FILE="containers/utkface/prediction_fast.log"
echo "Prediction started at $(date)" > "${LOG_FILE}"

total_tasks=$((${shards} * 4))
current_task=0

for i in $(seq 0 "$((${shards}-1))"); do
    for j in "${scenarios[@]}"; do
        current_task=$((${current_task} + 1))
        r=$((${j}*${shards}/5))
        
        echo ""
        echo "Task ${current_task}/${total_tasks}: Shard ${i}, Scenario j=${j}"
        
        output_file="containers/utkface/outputs/shard-${i}:${r}.npy"
        if [[ -f "${output_file}" ]]; then
            echo "✅ Output exists, skip"
            continue
        fi
        
        checkpoint="containers/utkface/cache/shard-${i}:${r}.pt"
        if [[ ! -f "${checkpoint}" ]]; then
            echo "❌ Checkpoint missing: ${checkpoint}"
            exit 1
        fi
        
        python sisa.py \
            --model utkface \
            --test \
            --dataset datasets/UTKFace/datasetfile \
            --label "${r}" \
            --batch_size ${BATCH_SIZE} \
            --container "utkface" \
            --shard "${i}" \
            2>&1 | tee -a "${LOG_FILE}"
        
        echo "✅ Done"
    done
done

echo ""
echo "✅ PREDICTION HOÀN TẤT!"
echo "Bước tiếp theo: ./data_fast.sh ${shards}"