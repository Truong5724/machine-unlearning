#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-celeba_ovr}
dataset=${2:-datasets/celebA/datasetfile_ovr}
objective=${3:-bacc}
tune_split=${4:-val}
eval_split=${5:-test}
include_tasks=${6:-}
exclude_tasks=${7:-}
save_json=${8:-}
MIN_THRESHOLD=${MIN_THRESHOLD:-0.05}
MAX_THRESHOLD=${MAX_THRESHOLD:-1.0}

if [[ "${objective}" != "f1" && "${objective}" != "bacc" ]]; then
  echo "Invalid objective: ${objective}. Use f1|bacc"
  exit 1
fi

if [[ "${tune_split}" != "train" && "${tune_split}" != "val" && "${tune_split}" != "test" ]]; then
  echo "Invalid tune split: ${tune_split}. Use train|val|test"
  exit 1
fi

if [[ "${eval_split}" != "train" && "${eval_split}" != "val" && "${eval_split}" != "test" ]]; then
  echo "Invalid eval split: ${eval_split}. Use train|val|test"
  exit 1
fi

echo "======================================================================"
echo "PREDICT CELEBA OVR (VAL->TEST)"
echo "======================================================================"
echo "Container   : ${container}"
echo "Dataset     : ${dataset}"
echo "Objective   : ${objective}"
echo "Tune split  : ${tune_split}"
echo "Eval split  : ${eval_split}"
echo "Thr range   : [${MIN_THRESHOLD}, ${MAX_THRESHOLD}]"
echo "Include     : ${include_tasks:-<all>}"
echo "Exclude     : ${exclude_tasks:-<none>}"
if [[ -n "${save_json}" ]]; then
  echo "Save JSON   : ${save_json}"
fi
echo "======================================================================"

cmd=(python aggregation_celebA.py
  --container "${container}"
  --dataset "${dataset}"
  --split "${eval_split}"
  --tune_thresholds
  --tune_split "${tune_split}"
  --tune_objective "${objective}"
  --min_threshold "${MIN_THRESHOLD}"
  --max_threshold "${MAX_THRESHOLD}"
  --save_thresholds)

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
