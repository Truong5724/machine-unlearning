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
CLASS_NAMES = {
    "gender": {
        0: "Female",
        1: "Male",
    },

    "age": {
        0: "Young (0-17)",
        1: "Adult (18-59)",
        2: "Senior (60+)",
    },

    "race": {
        0: "White",
        1: "Black",
        2: "Asian",
        3: "Indian",
        4: "Others",
    }
}


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

    precisions, recalls, f1s = [], [], []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": acc,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "bacc": float(np.mean(recalls))
    }


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


def get_training_time(container, label):
    """Tính tổng training time từ các file .time"""
    total = 0.0
    import glob
    time_files = glob.glob(
        f"containers/{container}/times/shard-*:{label}.time"
    )
    for f in time_files:
        try:
            with open(f, 'r') as file:
                total += float(file.read().strip())
        except:
            pass
    return total


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

    # Load forgot samples
    requestfile = np.load(f"containers/{args.container}/requestfile:{args.label}.npy", allow_pickle=True)
    all_forgot = np.unique(
        np.concatenate([np.asarray(r, dtype=np.int64) for r in requestfile if len(r) > 0])
    ) if any(len(r) > 0 for r in requestfile) else np.array([], dtype=np.int64)

    print(f"✅ Đã kết nối dataset: Train={datasetfile.get('nb_train', 'N/A')}, Test={datasetfile.get('nb_test', 'N/A')}")
    print(f"📊 Forgot samples: {len(all_forgot)}")
    # Decode class unlearning information
    forget_task = None
    forget_classes = []

    if args.label.startswith("forget_"):

        parts = args.label.split("_")

        # format:
        # forget_gender_0
        # forget_age_0_2
        # forget_race_1_2

        if len(parts) >= 3:
            forget_task = parts[1]
            forget_classes = [
                int(x) for x in parts[2:]
            ]

            print("\n================ FORGET INFO ================")
            print(
                f"Forgot task : {forget_task}"
            )
            print(
                "Forgot class:"
            )
            for cls in forget_classes:

                name = CLASS_NAMES.get(
                    forget_task,
                    {}
                ).get(
                    cls,
                    f"Unknown({cls})"
                )
                print(
                    f"  - {cls}: {name}"
                )
            print(
                "=============================================="
            )
    if args.label.startswith("forget_"):

        parts = args.label.split("_")

        forget_task = parts[1]

        forget_classes = [
          int(x) for x in parts[2:]
        ]

        print("\n================ FORGET INFO ================")
        print(f"Forgot task   : {forget_task}")
        print(f"Forgot class  : {forget_classes}")
        print("==============================================")

    # Training time
    train_time = get_training_time(args.container, args.label)
    print(f"Training time: {train_time}s")

    if len(all_forgot) == 0:
        print("0.0000, 0.0000, 0.0000, 0.0000, -1.0000")
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
                    probs = np.concatenate((probs, softmax(logits, dim=1).cpu().numpy()), axis=0)
            shard_outputs.append(probs)

        if not shard_outputs:
            continue

        preds = aggregate_shard_predictions(shard_outputs, args.strategy, weights)
        y_true = np.asarray(forgot_labels[task], dtype=np.int64)
        task_metrics[task] = compute_metrics(y_true, preds, NUM_CLASSES[task])

    if task_metrics:

        print("\n================ FORGOT SET METRICS ================")
        if forget_task is not None:
            print(
                f"Forgot task: {forget_task}"
            )
            print(
                "Forgot classes:"
            )
            for cls in forget_classes:
                print(
                    f"  {CLASS_NAMES[forget_task][cls]}"
                )
        print()

        for task, metrics in task_metrics.items():
            print(
                f"{task.upper():8s} | "
                f"ACC={metrics['acc']*100:.2f}% "
                f"BACC={metrics['bacc']*100:.2f}% "
                f"F1={metrics['f1']*100:.2f}% "
                f"RECALL={metrics['recall']*100:.2f}% "
                f"PREC={metrics['precision']*100:.2f}%"
            )
        print("====================================================")
        # Mean multitask metric
        mean_acc = np.mean(
            [m["acc"] for m in task_metrics.values()]
        )

        mean_bacc = np.mean(
            [m["bacc"] for m in task_metrics.values()]
        )

        mean_f1 = np.mean(
            [m["f1"] for m in task_metrics.values()]
        )

        mean_recall = np.mean(
            [m["recall"] for m in task_metrics.values()]
        )

        mean_precision = np.mean(
            [m["precision"] for m in task_metrics.values()]
        )
        print("\n================ SUMMARY ================")
        print(
            f"Mean ACC       : {mean_acc*100:.2f}%"
        )
        print(
            f"Mean BACC      : {mean_bacc*100:.2f}%"
        )
        print(
            f"Mean F1        : {mean_f1*100:.2f}%"
        )
        print(
            f"Mean Recall    : {mean_recall*100:.2f}%"
        )
        print(
            f"Mean Precision : {mean_precision*100:.2f}%"
        )

        print("==========================================")
    else:
        print(
            "0.0000, 0.0000, 0.0000, 0.0000, -1.0000"
        )


if __name__ == "__main__":
    main()