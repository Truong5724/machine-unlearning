#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-celeba_ovr}
dataset=${2:-datasets/celebA/datasetfile_ovr}
slices_per_shard=${3:-2}
seed=${4:-42}

echo "======================================================================"
echo "INIT CELEBA OVR SISA"
echo "======================================================================"
echo "Container        : ${container}"
echo "Datasetfile      : ${dataset}"
echo "Slices per shard : ${slices_per_shard}"
echo "Seed             : ${seed}"
echo "======================================================================"

if [[ ! -f "${dataset}" ]]; then
  echo "Missing datasetfile: ${dataset}"
  echo "Run prepare_data_ovr.py first."
  exit 1
fi

if [[ ! -f "datasets/celebA/celeba_ovr_train.h5" ]]; then
  echo "Missing datasets/celebA/celeba_ovr_train.h5"
  exit 1
fi

if [[ ! -f "datasets/celebA/celeba_ovr_val.h5" ]]; then
  echo "Missing datasets/celebA/celeba_ovr_val.h5"
  exit 1
fi

if [[ ! -f "datasets/celebA/celeba_ovr_test.h5" ]]; then
  echo "Missing datasets/celebA/celeba_ovr_test.h5"
  exit 1
fi

mkdir -p "containers/${container}/cache"
mkdir -p "containers/${container}/times"
mkdir -p "containers/${container}/outputs"
echo 0 > "containers/${container}/times/null.time"

python celeba_ovr_partition.py \
  --container "${container}" \
  --dataset "${dataset}" \
  --slices_per_shard "${slices_per_shard}" \
  --seed "${seed}"

echo ""
echo "Done: CelebA OVR partition initialized (27 shards)."
echo "Next: bash example-scripts/celeba-sharding/train_ovr.sh ${container} 0-26"
