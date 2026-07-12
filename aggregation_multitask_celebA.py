"""
CelebA Multitask Aggregation (27 attributes)

Aggregate SISA shard predictions:
    shard-{id}:{label}.npy

Each output shape:
    (num_test_samples, 27)

Metrics:
    ACC
    Balanced ACC
    Precision
    Recall
    F1
    ROC-AUC
    PR-AUC
"""

import argparse
import importlib
import json
import os

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


NUM_ATTRIBUTES = 27

TASKS = [
    f"attr_{i}" for i in range(NUM_ATTRIBUTES)
]


def load_dataloader(dataset_path):

    with open(dataset_path, "r") as f:
        datasetfile = json.loads(f.read())

    module_name = ".".join(
        dataset_path.replace("\\", "/").split("/")[:-1]
        + [datasetfile["dataloader"]]
    )

    dataloader = importlib.import_module(module_name)

    return datasetfile, dataloader



def binary_metrics(y_true, y_score):

    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float32
    )


    y_pred = (
        y_score >= 0.5
    ).astype(np.int64)


    result = {}


    result["acc"] = float(
        accuracy_score(
            y_true,
            y_pred
        )
    )


    result["bacc"] = float(
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )


    result["precision"] = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    result["recall"] = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    result["f1"] = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    try:
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_score
            )
        )

    except:
        result["roc_auc"] = 0.0



    try:
        result["pr_auc"] = float(
            average_precision_score(
                y_true,
                y_score
            )
        )

    except:
        result["pr_auc"] = 0.0



    return result



def aggregate_outputs(
    container,
    label,
    shards,
    strategy="uniform"
):

    outputs = []


    for shard in range(shards):

        path = (
            f"containers/{container}/outputs/"
            f"shard-{shard}:{label}.npy"
        )


        if not os.path.exists(path):

            print(
                f"⚠️ Skip missing shard output: {path}"
            )

            continue


        outputs.append(
            np.load(path)
        )


    if len(outputs) == 0:

        raise FileNotFoundError(
            "No shard prediction files found"
        )


    outputs = np.asarray(outputs)


    print(
        "Loaded prediction shape:",
        outputs.shape
    )


    # ---------------------------------
    # Uniform aggregation
    # ---------------------------------

    if strategy == "uniform":

        final_scores = np.mean(
            outputs,
            axis=0
        )


    # ---------------------------------
    # Weighted aggregation
    # ---------------------------------

    elif strategy == "proportional":

        split = np.load(
            f"containers/{container}/splitfile.npy",
            allow_pickle=True
        )


        valid_sizes = []

        for i in range(shards):

            path = (
                f"containers/{container}/outputs/"
                f"shard-{i}:{label}.npy"
            )

            if os.path.exists(path):

                valid_sizes.append(
                    len(split[i])
                )


        weights = (
            np.asarray(valid_sizes)
            /
            np.sum(valid_sizes)
        )


        final_scores = np.tensordot(
            weights,
            outputs,
            axes=(0,0)
        )


    else:

        raise ValueError(
            f"Unknown strategy {strategy}"
        )


    return final_scores



def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--container",
        default="celeba_multitask"
    )


    parser.add_argument(
        "--label",
        required=True
    )


    parser.add_argument(
        "--dataset",
        default="datasets/celebA/datasetfile_celeba"
    )


    parser.add_argument(
        "--shards",
        type=int,
        default=3
    )


    parser.add_argument(
        "--strategy",
        default="uniform",
        choices=[
            "uniform",
            "proportional"
        ]
    )


    args = parser.parse_args()



    datasetfile, dataloader = load_dataloader(
        args.dataset
    )


    test_indices = np.arange(
        datasetfile["nb_test"]
    )


    _, test_labels = dataloader.load(
        test_indices,
        category="test"
    )



    print("="*80)

    print(
        "CELEBA MULTITASK AGGREGATION"
    )

    print("="*80)


    print(
        f"Container : {args.container}"
    )

    print(
        f"Label     : {args.label}"
    )

    print(
        f"Shards    : {args.shards}"
    )

    print(
        f"Strategy  : {args.strategy}"
    )

    print("-"*80)



    scores = aggregate_outputs(
        args.container,
        args.label,
        args.shards,
        args.strategy
    )


    print(
        "Aggregated score shape:",
        scores.shape
    )


    all_metrics = []



    for i in range(NUM_ATTRIBUTES):


        metric = binary_metrics(
            test_labels[:,i],
            scores[:,i]
        )


        all_metrics.append(
            metric
        )


        print(
            f"{TASKS[i]:8s} | "
            f"ACC={metric['acc']*100:6.2f}% "
            f"BACC={metric['bacc']*100:6.2f}% "
            f"F1={metric['f1']*100:6.2f}% "
            f"ROC={metric['roc_auc']*100:6.2f}% "
            f"PR={metric['pr_auc']*100:6.2f}%"
        )



    print("-"*80)



    print(
        "OVERALL (27 attributes)"
    )


    for key in [
        "acc",
        "bacc",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ]:


        mean_value = np.mean(
            [
                m[key]
                for m in all_metrics
            ]
        )


        print(
            f"Mean {key.upper():10s}: "
            f"{mean_value*100:6.2f}%"
        )


    print("="*80)



if __name__ == "__main__":

    main()