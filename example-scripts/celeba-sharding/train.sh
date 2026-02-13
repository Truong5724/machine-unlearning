#!/bin/bash
# train_optimized.sh - Train SISA shards với batch_size tối ưu cho Colab

set -eou pipefail
IFS=$'\n\t'

shards=$1

# Cấu hình tối ưu cho Colab
BATCH_SIZE=8        # Giảm từ 32 xuống 8 để tránh OOM
EPOCHS=30
LEARNING_RATE=0.0001
OPTIMIZER=adam
CHKPT_INTERVAL=5

echo "======================================================================"
echo "TRAINING SISA SHARDS - TỐI ƯU CHO COLAB"
echo "======================================================================"
echo "Số shards: ${shards}"
echo "Batch size: ${BATCH_SIZE} (tối ưu cho Colab Free)"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Optimizer: ${OPTIMIZER}"
echo ""
echo "⚠️  QUAN TRỌNG:"
echo "   - Training sẽ mất 4-8 giờ với ${shards} shards"
echo "   - Nếu bị OOM, giảm BATCH_SIZE xuống 4 trong script này"
echo "   - Colab Free timeout sau 12h, nên chạy với <=5 shards"
echo "======================================================================"
echo ""

# Log file
LOG_FILE="containers/celeba/training.log"
echo "Training started at $(date)" > "${LOG_FILE}"

# Tổng số tasks
total_tasks=$((${shards} * 16))
current_task=0

start_time=$(date +%s)

for i in $(seq 0 "$((${shards}-1))"); do
    for j in 0; do
        current_task=$((${current_task} + 1))
        r=0
        
        echo ""
        echo "======================================================================"
        echo "Task ${current_task}/${total_tasks}"
        echo "Shard: $((${i}+1))/${shards} | Requests: ${r} (scenario $((${j}+1))/16)"
        echo "======================================================================"
        
        # Kiểm tra checkpoint đã tồn tại chưa
        checkpoint="containers/celeba/cache/shard-${i}:${r}.pt"
        if [[ -f "${checkpoint}" ]]; then
            echo "✅ Checkpoint đã tồn tại, skip!"
            continue
        fi
        
        # Kiểm tra RAM trước khi train
        if command -v free &> /dev/null; then
            free -h | grep Mem
        fi
        
        # Train command
        echo "🔄 Training..."
        python sisa.py \
            --model celeba \
            --train \
            --slices 1 \
            --dataset datasets/celebA/datasetfile \
            --label "${r}" \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --optimizer ${OPTIMIZER} \
            --chkpt_interval ${CHKPT_INTERVAL} \
            --container "celeba" \
            --shard "${i}" \
            2>&1 | tee -a "${LOG_FILE}"
        
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            echo "❌ Lỗi khi train shard ${i}, scenario ${j}"
            echo "   Xem log: ${LOG_FILE}"
            exit 1
        fi
        
        echo "✅ Hoàn thành shard ${i}, scenario ${j}"
        
        # Tính thời gian còn lại
        current_time=$(date +%s)
        elapsed=$((${current_time} - ${start_time}))
        avg_time_per_task=$((${elapsed} / ${current_task}))
        remaining_tasks=$((${total_tasks} - ${current_task}))
        eta=$((${remaining_tasks} * ${avg_time_per_task}))
        
        echo "⏱️  ETA: $((${eta} / 3600))h $((${eta} % 3600 / 60))m"
        
        # Clear cache để tránh memory leak
        if command -v sync &> /dev/null; then
            sync
            echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        fi
        
    done
done

end_time=$(date +%s)
total_time=$((${end_time} - ${start_time}))

echo ""
echo "======================================================================"
echo "✅ TRAINING HOÀN TẤT!"
echo "======================================================================"
echo "Tổng thời gian: $((${total_time} / 3600))h $((${total_time} % 3600 / 60))m"
echo "Checkpoints: containers/celeba/cache/"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Bước tiếp theo:"
echo "  ./predict_optimized.sh ${shards}"
echo "======================================================================"