#!/bin/bash
set -e

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

# Compute accuracy
acc=$(python aggregation.py \
  --strategy uniform \
  --container utkface \
  --shards ${shards} \
  --dataset datasets/UTKFace/datasetfile \
  --label ${label})

# Compute training time
TIME_DIR="containers/utkface/times"
rm -f ${TIME_DIR}/times.tmp
cat ${TIME_DIR}/shard-*:${label}.time > ${TIME_DIR}/times.tmp

time=$(python time_stats.py --container utkface | awk -F ',' '{print $1}')

echo "Accuracy: ${acc}"
echo "Training time: ${time}s"