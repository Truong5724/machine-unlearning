# #!/bin/bash
# # data_optimized.sh - Tổng hợp kết quả và tạo report

# set -eou pipefail
# IFS=$'\n\t'

# shards=$1

# echo "======================================================================"
# echo "DATA AGGREGATION - SISA RESULTS"
# echo "======================================================================"
# echo "Số shards: ${shards}"
# echo ""

# # Tạo CSV file
# REPORT_FILE="celeba-report.csv"

# if [[ ! -f ${REPORT_FILE} ]]; then
#     echo "📄 Tạo file report..."
#     echo "nb_shards,nb_requests,accuracy,retraining_time" > ${REPORT_FILE}
#     echo "✅ Đã tạo ${REPORT_FILE}"
# else
#     echo "⚠️  File ${REPORT_FILE} đã tồn tại"
#     echo "   Backup file cũ..."
#     cp ${REPORT_FILE} "${REPORT_FILE}.backup.$(date +%s)"
#     echo "nb_shards,nb_requests,accuracy,retraining_time" > ${REPORT_FILE}
# fi

# echo ""
# echo "🔄 Aggregating predictions và tính metrics..."
# echo ""

# for j in {0..15}; do
#     r=$((${j}*${shards}/5))
    
#     echo "--------------------------------------------------------------------"
#     echo "Scenario $((${j}+1))/16: ${r} requests"
#     echo "--------------------------------------------------------------------"
    
#     # Kiểm tra outputs tồn tại
#     missing=0
#     for i in $(seq 0 "$((${shards}-1))"); do
#         output_file="containers/celeba/outputs/shard-${i}:${r}.npy"
#         if [[ ! -f "${output_file}" ]]; then
#             echo "❌ Thiếu output: ${output_file}"
#             missing=1
#         fi
#     done
    
#     if [[ ${missing} -eq 1 ]]; then
#         echo "⚠️  Thiếu outputs, skip scenario này"
#         continue
#     fi
    
#     # Aggregate predictions
#     echo "🔄 Aggregating ${shards} shards..."
#     acc=$(python aggregation.py \
#         --strategy uniform \
#         --container "celeba" \
#         --shards "${shards}" \
#         --dataset datasets/celebA/datasetfile \
#         --label "${r}")
    
#     echo "✅ Accuracy: ${acc}"
    
#     # Tính retraining time
#     echo "🔄 Tính retraining time..."
#     cat containers/celeba/times/shard-*:"${r}".time > "containers/celeba/times/times.tmp"
#     time=$(python time_stats.py --container "celeba" | awk -F ',' '{print $1}')
    
#     echo "✅ Retraining time: ${time}s"
    
#     # Lưu vào CSV
#     echo "${shards},${r},${acc},${time}" >> ${REPORT_FILE}
    
#     echo ""
# done

# echo "======================================================================"
# echo "✅ DATA AGGREGATION HOÀN TẤT!"
# echo "======================================================================"
# echo ""
# echo "📊 KẾT QUẢ:"
# echo "--------------------------------------------------------------------"
# cat ${REPORT_FILE}
# echo "--------------------------------------------------------------------"
# echo ""
# echo "File đã lưu: ${REPORT_FILE}"
# echo ""
# echo "📈 PHÂN TÍCH KẾT QUẢ:"
# echo ""

# # Hiển thị summary statistics
# echo "Summary Statistics:"
# echo "-------------------"

# # Accuracy range
# acc_values=$(tail -n +2 ${REPORT_FILE} | awk -F ',' '{print $3}')
# if [[ ! -z "${acc_values}" ]]; then
#     min_acc=$(echo "${acc_values}" | sort -n | head -1)
#     max_acc=$(echo "${acc_values}" | sort -n | tail -1)
#     echo "Accuracy range: ${min_acc} - ${max_acc}"
# fi

# # Time range
# time_values=$(tail -n +2 ${REPORT_FILE} | awk -F ',' '{print $4}')
# if [[ ! -z "${time_values}" ]]; then
#     min_time=$(echo "${time_values}" | sort -n | head -1)
#     max_time=$(echo "${time_values}" | sort -n | tail -1)
#     echo "Retraining time range: ${min_time}s - ${max_time}s"
# fi

# echo ""
# echo "💡 Gợi ý cho khóa luận:"
# echo "  1. Vẽ biểu đồ: accuracy vs nb_requests"
# echo "  2. Vẽ biểu đồ: retraining_time vs nb_requests"
# echo "  3. So sánh với baseline (retrain from scratch)"
# echo "  4. Phân tích trade-off giữa accuracy và unlearning speed"
# echo ""
# echo "Để visualize kết quả:"
# echo "  python plot_results.py --input ${REPORT_FILE}"
# echo ""
# echo "======================================================================"
#!/bin/bash
# data_optimized.sh - Aggregation (train-only mode)

set -eou pipefail
IFS=$'\n\t'

shards=$1

echo "======================================================================"
echo "DATA AGGREGATION - TRAIN ONLY MODE"
echo "======================================================================"
echo "Số shards: ${shards}"
echo ""

REPORT_FILE="celeba-report.csv"

echo "nb_shards,nb_requests,accuracy,retraining_time" > ${REPORT_FILE}

r=0

echo "--------------------------------------------------------------------"
echo "Aggregating for r=${r}"
echo "--------------------------------------------------------------------"

# Check outputs
missing=0
for i in $(seq 0 "$((${shards}-1))"); do
    output_file="containers/celeba/outputs/shard-${i}:${r}.npy"
    if [[ ! -f "${output_file}" ]]; then
        echo "❌ Thiếu output: ${output_file}"
        missing=1
    fi
done

if [[ ${missing} -eq 1 ]]; then
    echo "⚠️  Thiếu outputs, hãy chạy predict trước!"
    exit 1
fi

echo "🔄 Aggregating ${shards} shards..."

acc=$(python aggregation.py \
    --strategy uniform \
    --container "celeba" \
    --shards "${shards}" \
    --dataset datasets/celebA/datasetfile \
    --label "${r}" | tail -n 1)

echo "✅ Accuracy: ${acc}"

echo "🔄 Tính training time..."
cat containers/celeba/times/shard-*:"${r}".time > \
    "containers/celeba/times/times.tmp"

time=$(python time_stats.py --container "celeba" | awk -F ',' '{print $1}')

echo "✅ Training time: ${time}s"

echo "${shards},${r},${acc},${time}" >> ${REPORT_FILE}

echo ""
echo "======================================================================"
echo "✅ DATA AGGREGATION HOÀN TẤT!"
echo "======================================================================"
echo ""
cat ${REPORT_FILE}
echo ""
echo "File đã lưu: ${REPORT_FILE}"
echo "======================================================================"
