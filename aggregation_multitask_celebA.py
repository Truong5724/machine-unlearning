"""
CelebA Multitask Aggregation (27 attributes)

Pipeline:
    Validation:
        Aggregate shard scores
        Tune threshold per attribute

    Test:
        Apply saved thresholds
        Compute metrics

Metrics:
    ACC
    BACC
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
    f"attr_{i}"
    for i in range(NUM_ATTRIBUTES)
]



# ============================================================
# LOAD DATASET
# ============================================================

def load_dataloader(dataset_path):

    with open(dataset_path,"r") as f:
        datasetfile = json.load(f)


    module_name = ".".join(
        dataset_path.replace("\\","/")
        .split("/")[:-1]
        +
        [
            datasetfile["dataloader"]
        ]
    )


    dataloader = importlib.import_module(
        module_name
    )


    return datasetfile, dataloader





# ============================================================
# METRICS
# ============================================================

def binary_metrics(
        y_true,
        y_score,
        threshold=0.5
):


    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )


    y_score = np.asarray(
        y_score,
        dtype=np.float32
    )


    y_pred = (
        y_score >= threshold
    ).astype(np.int64)



    result = {}


    result["threshold"] = float(
        threshold
    )


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
# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(
        y_true,
        y_score,
        objective="bacc"
):

    """
    Tìm threshold tốt nhất trên validation set

    objective:
        bacc
        f1
    """


    best_threshold = 0.5
    best_value = -1.0



    thresholds = np.linspace(
        0.05,
        0.95,
        91
    )


    for threshold in thresholds:


        metric = binary_metrics(
            y_true,
            y_score,
            threshold
        )


        value = metric[objective]


        if value > best_value:

            best_value = value

            best_threshold = threshold



    return float(best_threshold)




# ============================================================
# AGGREGATE SHARD OUTPUTS
# ============================================================

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
                f"⚠️ Missing: {path}"
            )

            continue



        outputs.append(
            np.load(path)
        )



    if len(outputs) == 0:

        raise FileNotFoundError(
            "No shard outputs found"
        )



    outputs = np.asarray(outputs)



    print(
        "Loaded prediction shape:",
        outputs.shape
    )



    # ========================================================
    # Uniform averaging
    # ========================================================

    if strategy == "uniform":


        scores = np.mean(
            outputs,
            axis=0
        )



    # ========================================================
    # Weighted averaging
    # ========================================================

    elif strategy == "proportional":


        split = np.load(
            f"containers/{container}/splitfile.npy",
            allow_pickle=True
        )


        sizes = []


        for shard in range(shards):

            path = (
                f"containers/{container}/outputs/"
                f"shard-{shard}:{label}.npy"
            )


            if os.path.exists(path):

                sizes.append(
                    len(split[shard])
                )



        weights = (
            np.asarray(sizes)
            /
            np.sum(sizes)
        )


        scores = np.tensordot(
            weights,
            outputs,
            axes=(0,0)
        )



    else:

        raise ValueError(
            f"Unknown strategy {strategy}"
        )



    print(
        "Aggregated score shape:",
        scores.shape
    )


    return scores





# ============================================================
# LOAD SPLIT LABELS
# ============================================================

def load_split_labels(
        datasetfile,
        dataloader,
        split
):


    if split == "val":

        n = datasetfile["nb_val"]


    elif split == "test":

        n = datasetfile["nb_test"]


    else:

        raise ValueError(
            "Only val/test supported"
        )



    indices = np.arange(
        n,
        dtype=np.int64
    )


    _, labels = dataloader.load(
        indices,
        category=split
    )


    return labels
# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--container",
        default="celeba"
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


    parser.add_argument(
        "--objective",
        default="bacc",
        choices=[
            "bacc",
            "f1"
        ]
    )


    parser.add_argument(
        "--tune_split",
        default="val"
    )


    parser.add_argument(
        "--eval_split",
        default="test"
    )


    args = parser.parse_args()



    datasetfile, dataloader = load_dataloader(
        args.dataset
    )



    print("="*80)

    print(
        "CELEBA MULTITASK SISA AGGREGATION"
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

    print(
        f"Objective : {args.objective}"
    )


    print("-"*80)



    # ========================================================
    # Aggregate all shard predictions
    # ========================================================

    scores = aggregate_outputs(
        args.container,
        args.label,
        args.shards,
        args.strategy
    )



    # ========================================================
    # Tune threshold on validation
    # ========================================================

    val_labels = load_split_labels(
        datasetfile,
        dataloader,
        args.tune_split
    )


    thresholds = {}



    print("\nThreshold tuning:")


    for i in range(NUM_ATTRIBUTES):


        threshold = tune_threshold(
            val_labels[:,i],
            scores[:len(val_labels),i],
            args.objective
        )


        thresholds[
            TASKS[i]
        ] = threshold



        print(
            f"{TASKS[i]:8s} "
            f"threshold={threshold:.3f}"
        )



    # save threshold

    threshold_dir = (
        f"containers/{args.container}/outputs/thresholds"
    )

    os.makedirs(
        threshold_dir,
        exist_ok=True
    )


    threshold_file = (
        f"{threshold_dir}/{args.label}.json"
    )


    with open(
        threshold_file,
        "w"
    ) as f:

        json.dump(
            thresholds,
            f,
            indent=4
        )


    print(
        "\nSaved:",
        threshold_file
    )



    # ========================================================
    # Evaluate test
    # ========================================================

    test_labels = load_split_labels(
        datasetfile,
        dataloader,
        args.eval_split
    )



    print("\nTEST RESULTS")

    print("-"*80)



    all_metrics = []



    for i in range(NUM_ATTRIBUTES):


        metric = binary_metrics(
            test_labels[:,i],
            scores[:len(test_labels),i],
            thresholds[TASKS[i]]
        )


        all_metrics.append(
            metric
        )


        print(
            f"{TASKS[i]:8s} | "
            f"ACC={metric['acc']*100:6.2f}% "
            f"BACC={metric['bacc']*100:6.2f}% "
            f"F1={metric['f1']*100:6.2f}% "
            f"PREC={metric['precision']*100:6.2f}% "
            f"REC={metric['recall']*100:6.2f}% "
            f"ROC={metric['roc_auc']*100:6.2f}% "
            f"PR={metric['pr_auc']*100:6.2f}%"
        )



    print("-"*80)

    print(
        "OVERALL (27 ATTRIBUTES)"
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


        value = np.mean(
            [
                m[key]
                for m in all_metrics
            ]
        )


        print(
            f"Mean {key.upper():10s}: "
            f"{value*100:6.2f}%"
        )



    print("="*80)




if __name__ == "__main__":

    main()