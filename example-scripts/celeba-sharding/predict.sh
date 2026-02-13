# #!/bin/bash
# # predict_optimized.sh - Dự đoán trên test set với batch_size tối ưu

# set -eou pipefail
# IFS=$'\n\t'

# shards=$1

# # Cấu hình
# BATCH_SIZE=16  # Predict có thể dùng batch_size lớn hơn (không cần gradients)

# echo "======================================================================"
# echo "PREDICTION - SISA SHARDS"
# echo "======================================================================"
# echo "Số shards: ${shards}"
# echo "Batch size: ${BATCH_SIZE}"
# echo ""

# # Log file
# LOG_FILE="containers/celeba/prediction.log"
# echo "Prediction started at $(date)" > "${LOG_FILE}"

# # Tổng số tasks
# total_tasks=$((${shards} * 16))
# current_task=0

# start_time=$(date +%s)

# for i in $(seq 0 "$((${shards}-1))"); do
#     for j in {0..15}; do
#         current_task=$((${current_task} + 1))
#         r=$((${j}*${shards}/5))
        
#         echo ""
#         echo "======================================================================"
#         echo "Task ${current_task}/${total_tasks}"
#         echo "Shard: $((${i}+1))/${shards} | Requests: ${r} (scenario $((${j}+1))/16)"
#         echo "======================================================================"
        
#         # Kiểm tra output đã tồn tại chưa
#         output_file="containers/celeba/outputs/shard-${i}:${r}.npy"
#         if [[ -f "${output_file}" ]]; then
#             echo "✅ Output đã tồn tại, skip!"
#             continue
#         fi
        
#         # Kiểm tra checkpoint tồn tại
#         checkpoint="containers/celeba/cache/shard-${i}:${r}.pt"
#         if [[ ! -f "${checkpoint}" ]]; then
#             echo "❌ Checkpoint không tồn tại: ${checkpoint}"
#             echo "   Hãy chạy train_optimized.sh trước!"
#             exit 1
#         fi
        
#         echo "🔄 Predicting..."
#         python sisa.py \
#             --model celeba \
#             --test \
#             --dataset datasets/celebA/datasetfile \
#             --label "${r}" \
#             --batch_size ${BATCH_SIZE} \
#             --container "celeba" \
#             --shard "${i}" \
#             2>&1 | tee -a "${LOG_FILE}"
        
#         if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
#             echo "❌ Lỗi khi predict shard ${i}, scenario ${j}"
#             exit 1
#         fi
        
#         echo "✅ Hoàn thành shard ${i}, scenario ${j}"
        
#         # Tính ETA
#         current_time=$(date +%s)
#         elapsed=$((${current_time} - ${start_time}))
#         avg_time_per_task=$((${elapsed} / ${current_task}))
#         remaining_tasks=$((${total_tasks} - ${current_task}))
#         eta=$((${remaining_tasks} * ${avg_time_per_task}))
        
#         echo "⏱️  ETA: $((${eta} / 60))m $((${eta} % 60))s"
#     done
# done

# end_time=$(date +%s)
# total_time=$((${end_time} - ${start_time}))

# echo ""
# echo "======================================================================"
# echo "✅ PREDICTION HOÀN TẤT!"
# echo "======================================================================"
# echo "Tổng thời gian: $((${total_time} / 60))m $((${total_time} % 60))s"
# echo "Outputs: containers/celeba/outputs/"
# echo "Log file: ${LOG_FILE}"
# echo ""
# echo "Bước tiếp theo:"
# echo "  ./data_optimized.sh ${shards}"
# echo "======================================================================"
#!/bin/bash
# predict_optimized.sh - Predict test set (train-only mode)

set -eou pipefail
IFS=$'\n\t'

shards=$1

BATCH_SIZE=16

echo "======================================================================"
echo "PREDICTION - SISA SHARDS (TRAIN-ONLY MODE)"
echo "======================================================================"
echo "Số shards: ${shards}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

LOG_FILE="containers/celeba/prediction.log"
echo "Prediction started at $(date)" > "${LOG_FILE}"

total_tasks=${shards}
current_task=0

start_time=$(date +%s)

for i in $(seq 0 "$((${shards}-1))"); do
    current_task=$((${current_task} + 1))
    r=0

    echo ""
    echo "======================================================================"
    echo "Task ${current_task}/${total_tasks}"
    echo "Shard: $((${i}+1))/${shards} | Requests: ${r}"
    echo "======================================================================"

    output_file="containers/celeba/outputs/shard-${i}:${r}.npy"
    if [[ -f "${output_file}" ]]; then
        echo "✅ Output đã tồn tại, skip!"
        continue
    fi

    checkpoint="containers/celeba/cache/shard-${i}:${r}.pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "❌ Checkpoint không tồn tại: ${checkpoint}"
        echo "   Hãy chạy train_optimized.sh trước!"
        exit 1
    fi

    echo "🔄 Predicting..."
    python sisa.py \
        --model celeba \
        --test \
        --dataset datasets/celebA/datasetfile \
        --label "${r}" \
        --batch_size ${BATCH_SIZE} \
        --container "celeba" \
        --shard "${i}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "❌ Lỗi khi predict shard ${i}"
        exit 1
    fi

    echo "✅ Hoàn thành shard ${i}"

    current_time=$(date +%s)
    elapsed=$((${current_time} - ${start_time}))
    avg_time_per_task=$((${elapsed} / ${current_task}))
    remaining_tasks=$((${total_tasks} - ${current_task}))
    eta=$((${remaining_tasks} * ${avg_time_per_task}))

    echo "⏱️  ETA: $((${eta} / 60))m $((${eta} % 60))s"
done

end_time=$(date +%s)
total_time=$((${end_time} - ${start_time}))

echo ""
echo "======================================================================"
echo "✅ PREDICTION HOÀN TẤT!"
echo "======================================================================"
echo "Tổng thời gian: $((${total_time} / 60))m $((${total_time} % 60))s"
echo "Outputs: containers/celeba/outputs/"
echo "Log file: ${LOG_FILE}"
echo "======================================================================"
