# #!/bin/bash
# # data_fast.sh - Aggregate kết quả FAST MODE

# set -eou pipefail
# IFS=$'\n\t'

# shards=$1
# scenarios=(0 5 10 15)

# echo "======================================================================"
# echo "DATA AGGREGATION - FAST MODE"
# echo "======================================================================"

# REPORT_FILE="utkface-report-fast.csv"

# if [[ -f ${REPORT_FILE} ]]; then
#     echo "⚠️  Backup file cũ..."
#     cp ${REPORT_FILE} "${REPORT_FILE}.backup.$(date +%s)"
# fi

# echo "nb_shards,nb_requests,accuracy,retraining_time" > ${REPORT_FILE}

# for j in "${scenarios[@]}"; do
#     r=$((${j}*${shards}/5))
    
#     echo ""
#     echo "Scenario j=${j}: ${r} requests"
#     echo "------------------------------------------------------------"
    
#     # Check outputs exist
#     missing=0
#     for i in $(seq 0 "$((${shards}-1))"); do
#         output_file="containers/utkface/outputs/shard-${i}:${r}.npy"
#         if [[ ! -f "${output_file}" ]]; then
#             echo "❌ Missing: ${output_file}"
#             missing=1
#         fi
#     done
    
#     if [[ ${missing} -eq 1 ]]; then
#         echo "⚠️  Skip scenario này"
#         continue
#     fi
    
#     echo "🔄 Aggregating..."
#     acc=$(python aggregation.py \
#         --strategy uniform \
#         --container "utkface" \
#         --shards "${shards}" \
#         --dataset datasets/UTKFace/datasetfile \
#         --label "${r}")
    
#     echo "✅ Accuracy: ${acc}"
    
#     cat containers/utkface/times/shard-*:"${r}".time > "containers/utkface/times/times.tmp"
#     time=$(python time_stats.py --container "utkface" | awk -F ',' '{print $1}')
    
#     echo "✅ Time: ${time}s"
#     echo "${shards},${r},${acc},${time}" >> ${REPORT_FILE}
# done

# echo ""
# echo "======================================================================"
# echo "✅ AGGREGATION HOÀN TẤT!"
# echo "======================================================================"
# echo ""
# cat ${REPORT_FILE}
# echo ""
# echo "File: ${REPORT_FILE}"
# echo ""
# echo "Visualize: python plot_results.py --input ${REPORT_FILE} --output utkface-fast"
# echo "======================================================================"
#!/bin/bash
# data_utkface_simple.sh - Aggregate UTKFace (NO UNLEARNING)

set -eou pipefail
IFS=$'\n\t'

shards=$1

echo "======================================================================"
echo "DATA AGGREGATION - SIMPLE MODE"
echo "======================================================================"
echo "Shards: ${shards}"
echo "======================================================================"

REPORT_FILE="utkface-report-simple.csv"

if [[ -f ${REPORT_FILE} ]]; then
    echo "⚠️  Backup file cũ..."
    cp ${REPORT_FILE} "${REPORT_FILE}.backup.$(date +%s)"
fi

echo "nb_shards,accuracy,retraining_time" > ${REPORT_FILE}

echo ""
echo "🔎 Checking output files..."

missing=0
for i in $(seq 0 "$((${shards}-1))"); do
    output_file="containers/utkface/outputs/shard-${i}:0.npy"
    if [[ ! -f "${output_file}" ]]; then
        echo "❌ Missing: ${output_file}"
        missing=1
    fi
done

if [[ ${missing} -eq 1 ]]; then
    echo "❌ Một số output bị thiếu. Hãy chạy predict trước."
    exit 1
fi

echo ""
echo "🔄 Aggregating predictions..."

acc=$(python aggregation.py \
    --strategy uniform \
    --container "utkface" \
    --shards "${shards}" \
    --dataset datasets/UTKFace/datasetfile \
    --label 0)

echo "✅ Accuracy: ${acc}"

echo ""
echo "⏱  Collecting training time..."

cat containers/utkface/times/shard-*:"0".time > "containers/utkface/times/times.tmp"

time=$(python time_stats.py --container "utkface" | awk -F ',' '{print $1}')

echo "✅ Time: ${time}s"

echo "${shards},${acc},${time}" >> ${REPORT_FILE}

echo ""
echo "======================================================================"
echo "✅ AGGREGATION HOÀN TẤT!"
echo "======================================================================"
echo ""
cat ${REPORT_FILE}
echo ""
echo "File: ${REPORT_FILE}"
echo ""
echo "Visualize:"
echo "python plot_results.py --input ${REPORT_FILE} --output utkface-simple"
echo "======================================================================"