#!/bin/bash
# predict.sh - Predict trên test set

set -eou pipefail
IFS=$'\n\t'

shards=$1
label=${2:-0}  # Default label=0 nếu không truyền

BATCH_SIZE=128  # Predict có thể dùng batch lớn hơn

echo "======================================================================"
echo "PREDICTION - UTKFACE"
echo "======================================================================"
echo "Shards: ${shards}"
echo "Label: ${label}"
echo "Batch size: ${BATCH_SIZE}"
echo "======================================================================"
echo ""

LOG_FILE="containers/utkface/prediction.log"
echo "Prediction started at $(date)" > "${LOG_FILE}"

total_tasks=${shards}
current_task=0
start_time=$(date +%s)

for i in $(seq 0 "$((${shards}-1))"); do
    current_task=$((${current_task} + 1))
    
    echo ""
    echo "Shard $((${i}+1))/${shards}"
    
    output_file="containers/utkface/outputs/shard-${i}:${label}.npy"
    
    if [[ -f "${output_file}" ]]; then
        echo "✅ Output exists, skip"
        continue
    fi
    
    checkpoint="containers/utkface/cache/shard-${i}:${label}.pt"
    
    if [[ ! -f "${checkpoint}" ]]; then
        echo "❌ Checkpoint missing: ${checkpoint}"
        echo "   Run train first!"
        exit 1
    fi
    
    echo "🔄 Predicting..."
    
    python sisa_utkface.py \
        --model utkface \
        --test \
        --dataset datasets/UTKFace/datasetfile \
        --label "${label}" \
        --batch_size ${BATCH_SIZE} \
        --container "utkface" \
        --shard "${i}" \
        2>&1 | tee -a "${LOG_FILE}"
    
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "❌ Error"
        exit 1
    fi
    
    echo "✅ Done"
    
    # ETA
    current_time=$(date +%s)
    elapsed=$((${current_time} - ${start_time}))
    avg_time=$((${elapsed} / ${current_task}))
    remaining=$((${total_tasks} - ${current_task}))
    eta=$((${remaining} * ${avg_time}))
    
    echo "⏱️  ETA: $((${eta} / 60))m $((${eta} % 60))s"
done

end_time=$(date +%s)
total_time=$((${end_time} - ${start_time}))

echo ""
echo "======================================================================"
echo "✅ PREDICTION COMPLETE!"
echo "======================================================================"
echo "Time: $((${total_time} / 60))m $((${total_time} % 60))s"
echo "Outputs: containers/utkface/outputs/"
echo ""
echo "Next: ./aggregate.sh ${shards} ${label}"
echo "======================================================================"