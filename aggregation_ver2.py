import argparse
import importlib
import json
import os

import numpy as np

TASKS = ("gender", "age", "race")
NUM_CLASSES = {"gender": 2, "age": 3, "race": 5}


def load_dataloader(dataset_path):
    with open(dataset_path, "r") as f:
        datasetfile = json.loads(f.read())

    module_name = ".".join(
        dataset_path.replace("\\", "/").split("/")[:-1] + [datasetfile["dataloader"]]
    )
    dataloader = importlib.import_module(module_name)
    return datasetfile, dataloader


def compute_multiclass_metrics(output_matrix, labels):
    preds = np.argmax(output_matrix, axis=1)
    labels = np.asarray(labels, dtype=np.int64)

    acc = float(np.mean(preds == labels)) if labels.size else 0.0

    num_classes = int(output_matrix.shape[1]) if output_matrix.ndim == 2 else 0
    precisions, recalls, f1s = [], [], []

    for cls in range(num_classes):
        tp = float(np.sum((preds == cls) & (labels == cls)))
        fp = float(np.sum((preds == cls) & (labels != cls)))
        fn = float(np.sum((preds != cls) & (labels == cls)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": acc,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "bacc": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def aggregate_task_outputs(container, label, task, shards, strategy="uniform"):
    outputs = []
    for shard in range(shards):
        output_path = f"containers/{container}/outputs/shard-{shard}:{label}-{task}.npy"
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Missing output: {output_path}")
        outputs.append(np.load(output_path, allow_pickle=True))

    outputs = np.array(outputs)

    if strategy == "uniform":
        weights = np.ones(outputs.shape[0]) / outputs.shape[0]
    elif strategy == "proportional":
        split = np.load(f"containers/{container}/splitfile.npy", allow_pickle=True)
        shard_sizes = np.array([len(s) for s in split], dtype=float)
        weights = shard_sizes / shard_sizes.sum()
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    votes = np.argmax(
        np.tensordot(weights.reshape(1, -1), outputs, axes=1), axis=2
    ).reshape(outputs.shape[1])
    return votes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="utkface")
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", default="datasets/UTKFace/datasetfile_ver2")
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--strategy", default="uniform", choices=["uniform", "proportional"])
    args = parser.parse_args()

    datasetfile, dataloader = load_dataloader(args.dataset)
    test_indices = np.arange(datasetfile["nb_test"])
    _, labels = dataloader.load_multitask(test_indices, category="test")

    print("=" * 70)
    print("UTKFACE JOINT MULTITASK EVALUATION (TEST SET)")
    print("=" * 70)
    print(f"Container: {args.container}")
    print(f"Label: {args.label}")
    print(f"Shards: {args.shards}")
    print(f"Strategy: {args.strategy}")
    print()

    all_metrics = {}
    for task in TASKS:
        preds = aggregate_task_outputs(
            args.container, args.label, task, args.shards, strategy=args.strategy
        )
        task_labels = np.asarray(labels[task], dtype=np.int64)
        metrics = compute_multiclass_metrics(
            np.eye(NUM_CLASSES[task])[preds], task_labels
        )
        all_metrics[task] = metrics

        print(
            f"{task:6s}: acc={metrics['acc'] * 100:.2f}% "
            f"prec={metrics['precision'] * 100:.2f}% "
            f"bacc={metrics['bacc'] * 100:.2f}% "
            f"f1={metrics['f1'] * 100:.2f}%"
        )

    mean_acc = float(np.mean([m["acc"] for m in all_metrics.values()]))
    mean_precision = float(np.mean([m["precision"] for m in all_metrics.values()]))
    mean_bacc = float(np.mean([m["bacc"] for m in all_metrics.values()]))
    mean_f1 = float(np.mean([m["f1"] for m in all_metrics.values()]))
    print()
    print(f"Mean multitask acc    : {mean_acc * 100:.2f}%")
    print(f"Mean multitask prec   : {mean_precision * 100:.2f}%")
    print(f"Mean multitask bacc   : {mean_bacc * 100:.2f}%")
    print(f"Mean multitask f1     : {mean_f1 * 100:.2f}%")

    times = []
    for shard in range(args.shards):
        time_path = f"containers/{args.container}/times/shard-{shard}:{args.label}.time"
        if os.path.exists(time_path):
            with open(time_path, "r") as f:
                try:
                    times.append(float(f.read().strip()))
                except ValueError:
                    pass

    if times:
        print(f"Total training time (sum shards): {np.sum(times):.2f}s")

    print("=" * 70)


if __name__ == "__main__":
    main()
