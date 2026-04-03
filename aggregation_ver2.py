import argparse
import importlib
import json
import os

import numpy as np

TASK_BY_SHARD = {0: "gender", 1: "age", 2: "race"}


def load_dataloader(dataset_path):
    with open(dataset_path, "r") as f:
        datasetfile = json.loads(f.read())

    module_name = ".".join(
        dataset_path.replace("\\", "/").split("/")[:-1] + [datasetfile["dataloader"]]
    )
    dataloader = importlib.import_module(module_name)
    return datasetfile, dataloader


def compute_acc(output_matrix, labels):
    preds = np.argmax(output_matrix, axis=1)
    return float(np.mean(preds == labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="utkface")
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", default="datasets/UTKFace/datasetfile_ver2")
    args = parser.parse_args()

    datasetfile, dataloader = load_dataloader(args.dataset)
    test_indices = np.arange(datasetfile["nb_test"])
    _, labels = dataloader.load_multitask(test_indices, category="test")

    print("=" * 70)
    print("UTKFACE MULTITASK EVALUATION")
    print("=" * 70)
    print(f"Container: {args.container}")
    print(f"Label: {args.label}")
    print(f"Dataset: {args.dataset}")
    print()

    all_acc = {}
    for shard, task in TASK_BY_SHARD.items():
        output_path = f"containers/{args.container}/outputs/shard-{shard}:{args.label}.npy"
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Missing output: {output_path}")

        outputs = np.load(output_path, allow_pickle=True)
        task_labels = np.asarray(labels[task], dtype=np.int64)
        acc = compute_acc(outputs, task_labels)
        all_acc[task] = acc

        print(f"Shard {shard} ({task}) accuracy: {acc * 100:.2f}%")

    mean_acc = float(np.mean(list(all_acc.values())))
    print()
    print(f"Mean multitask accuracy: {mean_acc * 100:.2f}%")

    times = []
    for shard in TASK_BY_SHARD:
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
