#!/bin/bash
set -euo pipefail

shards=$1
label=$2

if [[ -z "$shards" || -z "$label" ]]; then
  echo "Usage: ./data_simple.sh <num_shards> <label>"
  exit 1
fi

# Check outputs exist
for i in $(seq 0 $((${shards}-1))); do
  file="containers/utkface/outputs/shard-${i}:${label}.npy"
  [[ -f "$file" ]] || { echo "Missing $file"; exit 1; }
done

# Compute test-set metrics
test_metrics=$(python aggregation.py \
  --strategy uniform \
  --container utkface \
  --shards ${shards} \
  --dataset datasets/UTKFace/datasetfile \
  --label ${label})

# Compute unlearn accuracy on the forgotten set (skip baseline label=0)
unlearn_acc="N/A"
if [[ "${label}" != "0" ]]; then
  unlearn_acc=$(python example-scripts/utkface-sharding/evaluate_forgot.py \
    --container utkface \
    --label ${label} \
    --shards ${shards} \
    --dataset datasets/UTKFace/datasetfile \
    | awk -F': ' '/Average accuracy on forgot set/ {print $2}' | tail -n 1)

  if [[ -z "${unlearn_acc}" ]]; then
    unlearn_acc="N/A"
  fi
fi

# Compute training time
TIME_DIR="containers/utkface/times"
rm -f ${TIME_DIR}/times.tmp
cat ${TIME_DIR}/shard-*:${label}.time > ${TIME_DIR}/times.tmp

time=$(python time_stats.py --container utkface | awk -F ',' '{print $1}')

echo "Test metrics: ${test_metrics}"
echo "Unlearn accuracy: ${unlearn_acc}"
echo "Training time: ${time}s"