import argparse
import importlib
import json
import os

import numpy as np

TASKS = ("gender", "age", "race")
NUM_CLASSES = {"gender": 2, "age": 3, "race": 5}

CLASS_NAMES = {
    "gender": [
        "Male",
        "Female"
    ],

    "age": [
        "Young",
        "Middle",
        "Old"
    ],

    "race": [
        "White",
        "Black",
        "Asian",
        "Indian",
        "Others"
    ]
}


def load_dataloader(dataset_path):
    with open(dataset_path, "r") as f:
        datasetfile = json.loads(f.read())

    module_name = ".".join(
        dataset_path.replace("\\", "/").split("/")[:-1] + [datasetfile["dataloader"]]
    )
    dataloader = importlib.import_module(module_name)
    return datasetfile, dataloader



def compute_multiclass_metrics(output_matrix, labels, class_names=None):
    preds = np.argmax(output_matrix, axis=1)
    labels = np.asarray(labels, dtype=np.int64)

    acc = float(np.mean(preds == labels)) if labels.size else 0.0

    num_classes = int(output_matrix.shape[1]) if output_matrix.ndim == 2 else 0

    precisions = []
    recalls = []
    f1s = []

    class_metrics = {}

    for cls in range(num_classes):

        tp = float(np.sum((preds == cls) & (labels == cls)))
        fp = float(np.sum((preds == cls) & (labels != cls)))
        fn = float(np.sum((preds != cls) & (labels == cls)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)


        name = class_names[cls] if class_names else f"class_{cls}"

        class_metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(np.sum(labels == cls))
        }


    return {
        "acc": acc,

        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,

        "class_metrics": class_metrics
    }



def aggregate_task_outputs(container, label, task, shards, strategy="uniform"):
    """Simple majority vote như SISA gốc"""

    outputs = []

    for shard in range(shards):

        output_path = f"containers/{container}/outputs/shard-{shard}:{label}-{task}.npy"

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"Missing output for shard {shard}: {output_path}"
            )

        outputs.append(np.load(output_path))


    outputs = np.array(outputs)


    if strategy == "uniform":

        votes = np.argmax(
            np.sum(outputs, axis=0),
            axis=1
        )


    elif strategy == "proportional":

        split = np.load(
            f"containers/{container}/splitfile.npy",
            allow_pickle=True
        )

        shard_sizes = np.array(
            [len(s) for s in split],
            dtype=float
        )

        weights = shard_sizes / shard_sizes.sum()

        weighted = np.tensordot(
            weights.reshape(-1, 1, 1),
            outputs,
            axes=(0, 0)
        )

        votes = np.argmax(
            weighted.squeeze(0),
            axis=1
        )


    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}"
        )


    return votes



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--container",
        default="utkface"
    )

    parser.add_argument(
        "--label",
        required=True
    )

    parser.add_argument(
        "--dataset",
        default="datasets/UTKFace/datasetfile_ver2"
    )

    parser.add_argument(
        "--shards",
        type=int,
        default=3
    )

    parser.add_argument(
        "--strategy",
        default="uniform",
        choices=["uniform", "proportional"]
    )

    args = parser.parse_args()


    datasetfile, dataloader = load_dataloader(args.dataset)

    test_indices = np.arange(datasetfile["nb_test"])

    _, labels = dataloader.load_multitask(
        test_indices,
        category="test"
    )


    print("=" * 70)
    print("UTKFACE MULTITASK EVALUATION (TEST SET - SISA)")
    print("=" * 70)

    print(f"Container : {args.container}")
    print(f"Label     : {args.label}")
    print(f"Shards    : {args.shards}")
    print(f"Strategy  : {args.strategy} (majority vote)")

    print("-" * 70)



    all_metrics = {}


    for task in TASKS:

        preds = aggregate_task_outputs(
            args.container,
            args.label,
            task,
            args.shards,
            strategy=args.strategy
        )


        task_labels = np.asarray(
            labels[task],
            dtype=np.int64
        )

        print(NUM_CLASSES)
        print(CLASS_NAMES)
        metrics = compute_multiclass_metrics(
            np.eye(NUM_CLASSES[task])[preds],
            task_labels,
            CLASS_NAMES[task]
        )


        all_metrics[task] = metrics



        print(
            f"{task:6s}: "
            f"acc={metrics['acc']*100:6.2f}% | "
            f"macro-prec={metrics['macro_precision']*100:6.2f}% | "
            f"macro-recall={metrics['macro_recall']*100:6.2f}% | "
            f"macro-f1={metrics['macro_f1']*100:6.2f}%"
        )


        print("  Class metrics:")

        print(metrics["class_metrics"].keys())
        for cls_name, cls_metric in metrics["class_metrics"].items():

            print(
                f"    {cls_name:10s}: "
                f"prec={cls_metric['precision']*100:6.2f}% | "
                f"recall={cls_metric['recall']*100:6.2f}% | "
                f"f1={cls_metric['f1']*100:6.2f}% | "
                f"support={cls_metric['support']}"
            )

        print()



    mean_acc = np.mean(
        [m["acc"] for m in all_metrics.values()]
    )

    mean_prec = np.mean(
        [m["macro_precision"] for m in all_metrics.values()]
    )

    mean_bacc = np.mean(
        [m["macro_recall"] for m in all_metrics.values()]
    )

    mean_f1 = np.mean(
        [m["macro_f1"] for m in all_metrics.values()]
    )


    print("-" * 70)

    print(
        f"Mean Acc       : {mean_acc*100:6.2f}%"
    )

    print(
        f"Mean Macro Prec: {mean_prec*100:6.2f}%"
    )

    print(
        f"Mean Macro Rec : {mean_bacc*100:6.2f}%"
    )

    print(
        f"Mean Macro F1  : {mean_f1*100:6.2f}%"
    )

    print("=" * 70)



    # Training time summary

    times = []

    for shard in range(args.shards):

        time_path = (
            f"containers/{args.container}/times/"
            f"shard-{shard}:{args.label}.time"
        )


        if os.path.exists(time_path):

            with open(time_path, "r") as f:

                try:
                    times.append(
                        float(f.read().strip())
                    )

                except:
                    pass


    if times:

        print(
            f"Total training time: {np.sum(times)/60:.1f} minutes"
        )



if __name__ == "__main__":
    main()