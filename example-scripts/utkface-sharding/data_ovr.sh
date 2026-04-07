#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

label=${1:-0}
container=${2:-utkface_ovr}
dataset=${3:-datasets/UTKFace/datasetfile_ovr}
objective=${4:-f1}
tune_split=${5:-val}
eval_split=${6:-test}
exclude_eval_task=${7:-}
exclude_tune_too=${8:-false}

if [[ -z "${label}" ]]; then
  echo "Usage: bash example-scripts/utkface-sharding/data_ovr.sh <label> [container] [datasetfile_ovr] [objective] [tune_split] [eval_split] [exclude_eval_task] [exclude_tune_too]"
  exit 1
fi

if [[ "${objective}" != "f1" && "${objective}" != "bacc" ]]; then
  echo "Invalid objective: ${objective}. Use f1 or bacc"
  exit 1
fi

if [[ "${tune_split}" != "val" && "${tune_split}" != "test" ]]; then
  echo "Invalid tune split: ${tune_split}. Use val or test"
  exit 1
fi

if [[ "${eval_split}" != "val" && "${eval_split}" != "test" ]]; then
  echo "Invalid eval split: ${eval_split}. Use val or test"
  exit 1
fi

if [[ "${exclude_tune_too}" != "true" && "${exclude_tune_too}" != "false" ]]; then
  echo "Invalid exclude_tune_too: ${exclude_tune_too}. Use true or false"
  exit 1
fi

cmd=(python aggregation_ovr.py \
  --container "${container}" \
  --label "${label}" \
  --dataset "${dataset}" \
  --tune_thresholds \
  --tune_objective "${objective}" \
  --tune_split "${tune_split}" \
  --eval_split "${eval_split}" \
  --save_thresholds)

if [[ -n "${exclude_eval_task}" ]]; then
  cmd+=(--exclude_eval_task "${exclude_eval_task}")
fi

if [[ "${exclude_tune_too}" == "true" ]]; then
  cmd+=(--exclude_tune_too)
fi

"${cmd[@]}"

