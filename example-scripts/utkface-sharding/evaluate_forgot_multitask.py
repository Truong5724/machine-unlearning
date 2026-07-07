"""
Evaluate joint multitask model on the FORGOT set (unlearned training samples).
Aggregates predictions across data shards via uniform voting.
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
from torch.nn.functional import softmax

TASKS = ("gender", "age", "race")
NUM_CLASSES = {"gender": 2, "age": 3, "race": 5}


def load_dataset(dataset_path):
    with open(dataset_path) as f:
        datasetfile = json.loads(f.read())
    module_name = ".".join(
        dataset_path.replace("\\", "/").split("/")[:-1] + [datasetfile["dataloader"]]
    )
    dataloader = importlib.import_module(module_name)
    return datasetfile, dataloader


def compute_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0

    recalls = []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    return {"acc": acc, "bacc": float(np.mean(recalls)) if recalls else 0.0}


def aggregate_shard_predictions(all_shard_outputs, strategy="uniform", weights=None):
    outputs = np.array(all_shard_outputs)
    if weights is None:
        w = np.ones(outputs.shape[0]) / outputs.shape[0]
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
    return np.argmax(
        np.tensordot(w.reshape(1, -1), outputs, axes=1), axis=2
    ).reshape(outputs.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="utkface")
    parser.add_argument("--label", required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", default="datasets/UTKFace/datasetfile_ver2")
    parser.add_argument("--strategy", default="uniform", choices=["uniform", "proportional"])
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    sys.path.insert(0, repo_root)

    from architectures.utkface_multitask import MultiTaskModel

    datasetfile, dataloader = load_dataset(args.dataset)
    input_shape = tuple(datasetfile["input_shape"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    requestfile = np.load(
        f"containers/{args.container}/requestfile:{args.label}.npy", allow_pickle=True
    )
    all_forgot = np.unique(
        np.concatenate([np.asarray(r, dtype=np.int64) for r in requestfile if len(r) > 0])
        if any(len(r) > 0 for r in requestfile)
        else np.array([], dtype=np.int64)
    )

    print("=" * 70)
    print("EVALUATE FORGOT SET (JOINT MULTITASK)")
    print("=" * 70)
    print(f"Container: {args.container}")
    print(f"Label: {args.label}")
    print(f"Forgot samples: {len(all_forgot)}")
    print()

    if all_forgot.size == 0:
        print("No forgot data (baseline label=0)")
        return

    if args.strategy == "proportional":
        split = np.load(f"containers/{args.container}/splitfile.npy", allow_pickle=True)
        weights = np.array([len(s) for s in split], dtype=float)
    else:
        weights = None

    _, forgot_labels = dataloader.load_multitask(all_forgot, category="train")

    task_metrics = {}
    for task in TASKS:
        shard_outputs = []
        for shard_idx in range(args.shards):
            ckpt = f"containers/{args.container}/cache/shard-{shard_idx}:{args.label}.pt"
            if not os.path.exists(ckpt):
                continue

            model = MultiTaskModel(input_shape=input_shape).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            probs = np.empty((0, NUM_CLASSES[task]), dtype=np.float32)
            with torch.no_grad():
                for i in range(0, len(all_forgot), args.batch_size):
                    batch_ids = all_forgot[i : i + args.batch_size]
                    images, _ = dataloader.load_multitask(batch_ids, category="train")
                    x = torch.from_numpy(images).to(device)
                    logits = model(x)[task]
                    probs = np.concatenate(
                        (probs, softmax(logits, dim=1).cpu().numpy()), axis=0
                    )
            shard_outputs.append(probs)

        if not shard_outputs:
            print(f"No models found for task {task}")
            continue

        preds = aggregate_shard_predictions(shard_outputs, args.strategy, weights)
        y_true = np.asarray(forgot_labels[task], dtype=np.int64)
        task_metrics[task] = compute_metrics(y_true, preds, NUM_CLASSES[task])
        print(
            f"{task:6s}: acc={task_metrics[task]['acc'] * 100:.2f}% "
            f"bacc={task_metrics[task]['bacc'] * 100:.2f}%"
        )

    if task_metrics:
        mean_acc = float(np.mean([m["acc"] for m in task_metrics.values()]))
        print()
        print(f"Mean forgot acc: {mean_acc * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
