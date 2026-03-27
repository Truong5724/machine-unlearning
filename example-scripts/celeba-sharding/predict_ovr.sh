#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-celeba_ovr}
dataset=${2:-datasets/celebA/datasetfile_ovr}
split=${3:-test}
threshold=${4:-0.5}
include_tasks=${5:-}
exclude_tasks=${6:-}
save_json=${7:-}

if [[ "${split}" != "train" && "${split}" != "val" && "${split}" != "test" ]]; then
  echo "Invalid split: ${split}. Use train|val|test"
  exit 1
fi

echo "======================================================================"
echo "PREDICT/EVAL CELEBA OVR (FIXED THRESHOLD)"
echo "======================================================================"
echo "Container   : ${container}"
echo "Dataset     : ${dataset}"
echo "Split       : ${split}"
echo "Threshold   : ${threshold}"
echo "Include     : ${include_tasks:-<all>}"
echo "Exclude     : ${exclude_tasks:-<none>}"
if [[ -n "${save_json}" ]]; then
  echo "Save JSON   : ${save_json}"
fi
echo "======================================================================"

cmd=(python aggregation_celebA.py
  --container "${container}"
  --dataset "${dataset}"
  --split "${split}"
  --threshold "${threshold}")

if [[ -n "${include_tasks}" ]]; then
  cmd+=(--include_tasks "${include_tasks}")
fi

if [[ -n "${exclude_tasks}" ]]; then
  cmd+=(--exclude_tasks "${exclude_tasks}")
fi

if [[ -n "${save_json}" ]]; then
  cmd+=(--save_json "${save_json}")
fi

"${cmd[@]}"
