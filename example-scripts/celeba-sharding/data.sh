#!/usr/bin/env bash
# data.sh - Aggregate CelebA multitask predictions into a report

set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <number_of_shards> [label]"
    echo "Example: $0 5 0"
    exit 1
fi

shards="$1"
label="${2:-0}"
REPORT_FILE="celeba_multitask_report.csv"

if [[ ! -f "datasets/celebA/datasetfile_multitask" ]]; then
    echo "❌ Missing datasetfile_multitask"
    exit 1
fi

for i in $(seq 0 $((shards - 1))); do
    output_file="containers/celeba/outputs/shard-${i}:${label}.npy"
    if [[ ! -f "${output_file}" ]]; then
        echo "❌ Missing prediction output: ${output_file}"
        echo "   Run predict.sh first."
        exit 1
    fi
done

NUM_SHARDS="$shards" LABEL="$label" "${PYTHON_BIN}" - <<'PY'
import csv
import importlib
import json
import os
from pathlib import Path

import numpy as np

from aggregation_celebA import binary_metrics

shards = int(os.environ["NUM_SHARDS"])
label = os.environ["LABEL"]

with open("datasets/celebA/datasetfile_multitask") as f:
    datasetfile = json.load(f)

dataloader = importlib.import_module("datasets.celebA." + datasetfile["dataloader"])

indices = np.arange(int(datasetfile["nb_test"]), dtype=np.int64)
_, labels = dataloader.load(indices, category="test")

stack = None
for shard in range(shards):
    pred_path = Path(f"containers/celeba/outputs/shard-{shard}:{label}.npy")
    arr = np.load(pred_path)
    if stack is None:
        stack = np.zeros_like(arr, dtype=np.float64)
    stack += arr

stack /= max(shards, 1)
np.save(f"containers/celeba/outputs/aggregated:{label}.npy", stack)

metrics = []
with open("celeba_multitask_report.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["attr_idx", "acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"])
    for attr_idx in range(stack.shape[1]):
        y_true = np.asarray(labels[:, attr_idx], dtype=np.int64)
        y_score = stack[:, attr_idx].astype(np.float64)
        metric = binary_metrics(y_true, y_score, threshold=0.5)
        metrics.append(metric)
        writer.writerow([
            attr_idx,
            f"{metric['acc']:.6f}",
            f"{metric['bacc']:.6f}",
            f"{metric['f1']:.6f}",
            f"{metric['precision']:.6f}",
            f"{metric['recall']:.6f}",
            f"{metric['roc_auc']:.6f}",
            f"{metric['pr_auc']:.6f}",
        ])

summary = {
    "macro_acc": float(np.mean([m["acc"] for m in metrics])),
    "macro_bacc": float(np.mean([m["bacc"] for m in metrics])),
    "macro_f1": float(np.mean([m["f1"] for m in metrics])),
    "macro_precision": float(np.mean([m["precision"] for m in metrics])),
    "macro_recall": float(np.mean([m["recall"] for m in metrics])),
    "macro_roc_auc": float(np.mean([m["roc_auc"] for m in metrics])),
    "macro_pr_auc": float(np.mean([m["pr_auc"] for m in metrics])),
}

print("=== CELEBA MULTITASK AGGREGATION SUMMARY ===")
for key, value in summary.items():
    print(f"{key}={value:.6f}")
PY

echo ""
echo "======================================================================"
echo "✅ Data aggregation completed"
echo "Report: ${REPORT_FILE}"
echo "======================================================================"
