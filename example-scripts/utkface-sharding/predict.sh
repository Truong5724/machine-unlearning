# #!/bin/bash
# # predict_fast.sh - Prediction cho FAST MODE

# set -eou pipefail
# IFS=$'\n\t'

# shards=$1
# BATCH_SIZE=16
# scenarios=(0 5 10 15)

# echo "======================================================================"
# echo "PREDICTION - FAST MODE"
# echo "======================================================================"

# LOG_FILE="containers/utkface/prediction_fast.log"
# echo "Prediction started at $(date)" > "${LOG_FILE}"

# total_tasks=$((${shards} * 4))
# current_task=0

# for i in $(seq 0 "$((${shards}-1))"); do
#     for j in "${scenarios[@]}"; do
#         current_task=$((${current_task} + 1))
#         r=$((${j}*${shards}/5))
        
#         echo ""
#         echo "Task ${current_task}/${total_tasks}: Shard ${i}, Scenario j=${j}"
        
#         output_file="containers/utkface/outputs/shard-${i}:${r}.npy"
#         if [[ -f "${output_file}" ]]; then
#             echo "✅ Output exists, skip"
#             continue
#         fi
        
#         checkpoint="containers/utkface/cache/shard-${i}:${r}.pt"
#         if [[ ! -f "${checkpoint}" ]]; then
#             echo "❌ Checkpoint missing: ${checkpoint}"
#             exit 1
#         fi
        
#         python sisa.py \
#             --model utkface \
#             --test \
#             --dataset datasets/UTKFace/datasetfile \
#             --label "${r}" \
#             --batch_size ${BATCH_SIZE} \
#             --container "utkface" \
#             --shard "${i}" \
#             2>&1 | tee -a "${LOG_FILE}"
        
#         echo "✅ Done"
#     done
# done

# echo ""
# echo "✅ PREDICTION HOÀN TẤT!"
# echo "Bước tiếp theo: ./data_fast.sh ${shards}"
#!/bin/bash
# predict_utkface_simple.sh - Prediction UTKFace (NO UNLEARNING)

set -eou pipefail
IFS=$'\n\t'

shards=$1
BATCH_SIZE=32

echo "======================================================================"
echo "PREDICTION UTKFACE - SIMPLE MODE"
echo "======================================================================"
echo "Shards: ${shards}"
echo "Batch size: ${BATCH_SIZE}"
echo "======================================================================"

LOG_FILE="containers/utkface/prediction.log"
echo "Prediction started at $(date)" > "${LOG_FILE}"

total_tasks=${shards}
current_task=0

for i in $(seq 0 "$((${shards}-1))"); do

    current_task=$((${current_task} + 1))

    echo ""
    echo "======================================================================"
    echo "Shard $((${i}+1))/${shards}"
    echo "======================================================================"

    output_file="containers/utkface/outputs/shard-${i}:0.npy"
    checkpoint="containers/utkface/cache/shard-${i}:0.pt"

    if [[ -f "${output_file}" ]]; then
        echo "✅ Output đã tồn tại, skip"
        continue
    fi

    if [[ ! -f "${checkpoint}" ]]; then
        echo "❌ Checkpoint thiếu: ${checkpoint}"
        exit 1
    fi

    echo "🔄 Predicting..."

    python sisa.py \
        --model utkface \
        --test \
        --dataset datasets/UTKFace/datasetfile \
        --label 0 \
        --batch_size ${BATCH_SIZE} \
        --container "utkface" \
        --shard "${i}" \
        2>&1 | tee -a "${LOG_FILE}"

    echo "✅ Done shard ${i}"

done

echo ""
echo "======================================================================"
echo "✅ PREDICTION HOÀN TẤT!"
echo "======================================================================"