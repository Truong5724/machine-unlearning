#!/bin/bash
# data.sh - Tính metrics tổng hợp sau khi train + unlearn

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: ./data.sh <num_shards> <label>"
  echo "Example: ./data.sh 5 100"
  exit 1
fi

shards=$1
label=$2

echo "=============================================================="
echo "📊 DATA SUMMARY - UTKFace SISA"
echo "Shards : ${shards}"
echo "Label  : ${label}"
echo "=============================================================="

# Kiểm tra outputs tồn tại
echo "🔍 Checking outputs..."
for i in $(seq 0 $((shards-1))); do
  file="containers/utkface/outputs/shard-${i}:${label}-gender.npy"  # kiểm tra ít nhất 1 task
  if [[ ! -f "$file" ]]; then
    echo "❌ Missing output: $file"
    echo "   Chạy predict.sh trước!"
    exit 1
  fi
done
echo "✅ All outputs exist."

# Test set metrics (dùng aggregation script)
echo "📈 Computing test set metrics..."
test_metrics=$(python aggregation_ver2.py \
  --strategy uniform \
  --container utkface \
  --shards ${shards} \
  --dataset datasets/UTKFace/datasetfile_ver2 \
  --label ${label} 2>&1)

echo "Test metrics:"
echo "${test_metrics}"

# Unlearn accuracy trên forgot set
unlearn_acc="N/A"
if [[ "${label}" != "0" ]]; then
  echo "📉 Computing accuracy on forgotten set..."
  unlearn_output=$(python example-scripts/utkface-sharding/evaluate_forgot.py \
    --container utkface \
    --label "${label}" \
    --shards ${shards} \
    --dataset datasets/UTKFace/datasetfile_ver2)

  echo "${unlearn_output}"
  
  # Trích xuất Mean forgot acc nếu có
  mean_forgot=$(echo "${unlearn_output}" | grep -o "Mean forgot acc: [0-9.]*%" | grep -o "[0-9.]*" | tail -n 1)
  if [[ -n "${mean_forgot}" ]]; then
    unlearn_acc="${mean_forgot}%"
  fi
else
  echo "Baseline (label=0) - No unlearning"
fi

# Training time (tự tính, không dùng time_stats.py)
echo "⏱️  Calculating training time..."
TIME_DIR="containers/utkface/times"
total_time=0.0
count=0

for timefile in ${TIME_DIR}/shard-*:${label}.time; do
  if [[ -f "$timefile" ]]; then
    t=$(cat "$timefile" 2>/dev/null || echo "0")
    total_time=$(awk "BEGIN {print $total_time + $t}")
    count=$((count + 1))
  fi
done

if [[ $count -gt 0 ]]; then
  avg_time=$(awk "BEGIN {print $total_time / $count}")
  echo "Total training time : ${total_time} seconds"
  echo "Average per shard   : ${avg_time} seconds"
else
  echo "No time files found"
fi

echo "=============================================================="
echo "✅ SUMMARY DONE for label=${label}"
echo "Test metrics      : OK"
echo "Unlearn accuracy  : ${unlearn_acc}"
echo "=============================================================="