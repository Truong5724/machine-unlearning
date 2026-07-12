#!/bin/bash
# data.sh - Tính metrics tổng hợp CelebA multitask sau khi train + unlearn

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: ./data.sh <num_shards> <label>"
  echo "Example: ./data.sh 5 100"
  exit 1
fi


shards=$1
label=$2


CONTAINER="celeba"
DATASET="datasets/celebA/datasetfile_multitask"


echo "=============================================================="
echo "📊 DATA SUMMARY - CelebA Multitask SISA"
echo "Shards : ${shards}"
echo "Label  : ${label}"
echo "=============================================================="



# ---------------------------------------------------------------
# Check outputs
# ---------------------------------------------------------------

echo "🔍 Checking outputs..."

for i in $(seq 0 $((shards-1))); do

  file="containers/${CONTAINER}/outputs/shard-${i}:${label}.npy"

  if [[ ! -f "$file" ]]; then
    echo "❌ Missing output: $file"
    echo "   Chạy predict.sh trước!"
    exit 1
  fi

done


echo "✅ All outputs exist."



# ---------------------------------------------------------------
# Aggregate + metrics
# ---------------------------------------------------------------

echo "📈 Computing test set metrics..."


test_metrics=$(python aggregation_celebA_multitask.py \
  --strategy uniform \
  --container ${CONTAINER} \
  --shards ${shards} \
  --dataset ${DATASET} \
  --label ${label} 2>&1)


echo ""
echo "Test metrics:"
echo "${test_metrics}"



# ---------------------------------------------------------------
# Forgot set evaluation
# ---------------------------------------------------------------

unlearn_acc="N/A"


if [[ "${label}" != "0" ]]; then

  echo ""
  echo "📉 Computing accuracy on forgotten set..."


  if [[ -f "evaluate_forgot.py" ]]; then

    unlearn_output=$(python evaluate_forgot.py \
      --container ${CONTAINER} \
      --label "${label}" \
      --shards ${shards} \
      --dataset ${DATASET})


    echo "${unlearn_output}"


    mean_forgot=$(echo "${unlearn_output}" \
      | grep -o "Mean forgot acc: [0-9.]*%" \
      | grep -o "[0-9.]*" \
      | tail -n 1)


    if [[ -n "${mean_forgot}" ]]; then
      unlearn_acc="${mean_forgot}%"
    fi


  else

    echo "⚠️ evaluate_forgot.py not found, skip."

  fi


else

  echo "Baseline (label=0) - No unlearning"

fi



# ---------------------------------------------------------------
# Training time
# ---------------------------------------------------------------

echo ""
echo "⏱️  Calculating training time..."


TIME_DIR="containers/${CONTAINER}/times"


total_time=0.0
count=0


for timefile in ${TIME_DIR}/shard-*:${label}.time; do

  if [[ -f "$timefile" ]]; then

    t=$(cat "$timefile" 2>/dev/null || echo "0")

    total_time=$(awk "BEGIN {print $total_time + $t}")

    count=$((count+1))

  fi

done



if [[ $count -gt 0 ]]; then

  avg_time=$(awk "BEGIN {print $total_time / $count}")

  echo "Total training time : ${total_time} seconds"
  echo "Average per shard   : ${avg_time} seconds"

else

  echo "No time files found"

fi



echo ""
echo "=============================================================="
echo "✅ SUMMARY DONE for label=${label}"
echo "Test metrics      : OK"
echo "Unlearn accuracy  : ${unlearn_acc}"
echo "=============================================================="