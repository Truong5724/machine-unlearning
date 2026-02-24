# #!/bin/bash
# # train_fast.sh - Train FAST MODE (4 scenarios, 15 epochs)

# set -eou pipefail
# IFS=$'\n\t'

# shards=$1

# # Cấu hình FAST MODE
# BATCH_SIZE=8
# EPOCHS=15           # Giảm từ 30 xuống 15
# LEARNING_RATE=0.0005  # Tăng LR để converge nhanh hơn
# OPTIMIZER=adam
# CHKPT_INTERVAL=5

# # Chỉ train 4 scenarios (thay vì 16)
# scenarios=(0 5 10 15)

# echo "======================================================================"
# echo "TRAINING FAST MODE - TỐI ƯU CHO COLAB FREE"
# echo "======================================================================"
# echo "Shards: ${shards}"
# echo "Scenarios: 4 (thay vì 16)"
# echo "Epochs: ${EPOCHS} (thay vì 30)"
# echo "Batch size: ${BATCH_SIZE}"
# echo "Learning rate: ${LEARNING_RATE} (cao hơn để train nhanh)"
# echo ""
# echo "⏱️  Thời gian ước tính: $((${shards} * 4)) giờ"
# echo "   (Thay vì $((${shards} * 16 * 2)) giờ với config gốc!)"
# echo "======================================================================"
# echo ""

# LOG_FILE="containers/utkface/training_fast.log"
# echo "Training FAST MODE started at $(date)" > "${LOG_FILE}"

# total_tasks=$((${shards} * 4))
# current_task=0
# start_time=$(date +%s)

# for i in $(seq 0 "$((${shards}-1))"); do
#     for j in "${scenarios[@]}"; do
#         current_task=$((${current_task} + 1))
#         r=$((${j}*${shards}/5))
        
#         echo ""
#         echo "======================================================================"
#         echo "Task ${current_task}/${total_tasks}"
#         echo "Shard: $((${i}+1))/${shards} | Scenario: j=${j} (${r} requests)"
#         echo "======================================================================"
        
#         checkpoint="containers/utkface/cache/shard-${i}:${r}.pt"
#         if [[ -f "${checkpoint}" ]]; then
#             echo "✅ Checkpoint đã tồn tại, skip!"
#             continue
#         fi
        
#         echo "🔄 Training with FAST settings..."
#         echo "   Epochs: ${EPOCHS} | Batch: ${BATCH_SIZE} | LR: ${LEARNING_RATE}"
        
#         python sisa.py \
#             --model utkface \
#             --train \
#             --slices 1 \
#             --dataset datasets/UTKFace/datasetfile \
#             --label "${r}" \
#             --epochs ${EPOCHS} \
#             --batch_size ${BATCH_SIZE} \
#             --learning_rate ${LEARNING_RATE} \
#             --optimizer ${OPTIMIZER} \
#             --chkpt_interval ${CHKPT_INTERVAL} \
#             --container "utkface" \
#             --shard "${i}" \
#             2>&1 | tee -a "${LOG_FILE}"
        
#         if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
#             echo "❌ Lỗi khi train shard ${i}, scenario ${j}"
#             exit 1
#         fi
        
#         echo "✅ Hoàn thành"
        
#         # ETA
#         current_time=$(date +%s)
#         elapsed=$((${current_time} - ${start_time}))
#         avg_time=$((${elapsed} / ${current_task}))
#         remaining=$((${total_tasks} - ${current_task}))
#         eta=$((${remaining} * ${avg_time}))
        
#         echo "⏱️  Avg time per task: $((${avg_time} / 60))m"
#         echo "⏱️  ETA: $((${eta} / 3600))h $((${eta} % 3600 / 60))m"
        
#         # Backup checkpoint to Google Drive (if mounted)
#         if [[ -d "/content/drive/MyDrive" ]]; then
#             echo "💾 Backing up to Google Drive..."
#             mkdir -p "/content/drive/MyDrive/utkface_checkpoints"
#             cp "${checkpoint}" "/content/drive/MyDrive/utkface_checkpoints/"
#         fi
#     done
# done

# end_time=$(date +%s)
# total_time=$((${end_time} - ${start_time}))

# echo ""
# echo "======================================================================"
# echo "✅ TRAINING FAST MODE HOÀN TẤT!"
# echo "======================================================================"
# echo "Tổng thời gian: $((${total_time} / 3600))h $((${total_time} % 3600 / 60))m"
# echo "Avg per task: $((${total_time} / ${total_tasks} / 60))m"
# echo ""
# echo "📊 So sánh:"
# echo "   FAST mode: ${total_tasks} tasks × $((${total_time} / ${total_tasks} / 60))m = $((${total_time} / 3600))h"
# echo "   FULL mode: $((${shards} * 16)) tasks × 2h = $((${shards} * 32))h"
# echo "   Tiết kiệm: $((${shards} * 32 - ${total_time} / 3600))h!"
# echo ""
# echo "Bước tiếp theo:"
# echo "  ./predict_fast.sh ${shards}"
# echo "======================================================================"
#!/bin/bash
# train_utkface_simple.sh - Train UTKFace KHÔNG UNLEARNING

set -eou pipefail
IFS=$'\n\t'

shards=$1

# ===============================
# ⚙️ Cấu hình train
# ===============================
BATCH_SIZE=64
EPOCHS=60
LEARNING_RATE=0.001
OPTIMIZER=adam
CHKPT_INTERVAL=5

echo "======================================================================"
echo "TRAINING UTKFACE - SIMPLE MODE (NO UNLEARNING)"
echo "======================================================================"
echo "Shards: ${shards}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "======================================================================"
echo ""

LOG_FILE="containers/utkface/training.log"
echo "Training started at $(date)" > "${LOG_FILE}"

total_tasks=${shards}
current_task=0
start_time=$(date +%s)

# ===============================
# 🔄 Train từng shard
# ===============================
for i in $(seq 0 "$((${shards}-1))"); do

    current_task=$((${current_task} + 1))

    echo ""
    echo "======================================================================"
    echo "Shard $((${i}+1))/${shards}"
    echo "======================================================================"

    checkpoint="containers/utkface/cache/shard-${i}:0.pt"

    if [[ -f "${checkpoint}" ]]; then
        echo "✅ Checkpoint đã tồn tại, skip!"
        continue
    fi

    echo "🔄 Training shard ${i}..."

    python sisa.py \
        --model utkface \
        --train \
        --slices 1 \
        --dataset datasets/UTKFace/datasetfile \
        --label 0 \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --optimizer ${OPTIMIZER} \
        --chkpt_interval ${CHKPT_INTERVAL} \
        --container "utkface" \
        --shard "${i}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "❌ Lỗi khi train shard ${i}"
        exit 1
    fi

    echo "✅ Hoàn thành shard ${i}"

    # ETA
    current_time=$(date +%s)
    elapsed=$((${current_time} - ${start_time}))
    avg_time=$((${elapsed} / ${current_task}))
    remaining=$((${total_tasks} - ${current_task}))
    eta=$((${remaining} * ${avg_time}))

    echo "⏱️  Avg time per shard: $((${avg_time} / 60))m"
    echo "⏱️  ETA: $((${eta} / 3600))h $((${eta} % 3600 / 60))m"

done

end_time=$(date +%s)
total_time=$((${end_time} - ${start_time}))

echo ""
echo "======================================================================"
echo "✅ TRAINING HOÀN TẤT!"
echo "======================================================================"
echo "Tổng thời gian: $((${total_time} / 3600))h $((${total_time} % 3600 / 60))m"
echo "======================================================================"