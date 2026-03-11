import argparse
import importlib
import json
import os

import numpy as np

OVR_TASKS = [
    "gender_female",
    "gender_male",
    "age_bin0",
    "age_bin1",
    "age_bin2",
    "race_white",
    "race_black",
    "race_asian",
    "race_indian",
    "race_others",
]


def binary_confusion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def safe_div(num, den):
    if den == 0:
        return 0.0
    return float(num) / float(den)


def binary_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)

    tp, tn, fp, fn = binary_confusion(y_true, y_pred)

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    tpr = safe_div(tp, tp + fn)  # recall positive
    tnr = safe_div(tn, tn + fp)  # recall negative
    bacc = 0.5 * (tpr + tnr)

    precision = safe_div(tp, tp + fp)
    recall = tpr
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "acc": acc,
        "bacc": bacc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def average_precision_score(y_true, y_score):
    """Compute Average Precision (PR-AUC for step-wise PR curve)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    pos_total = int(np.sum(y_true == 1))
    if pos_total == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted == 1)
    pred_count = np.arange(1, len(y_sorted) + 1)
    precision = tp_cum / pred_count

    ap = float(np.sum(precision[y_sorted == 1]) / pos_total)
    return ap


def tune_threshold(y_true, y_score, objective="f1"):
    """Find best threshold for a binary task according to objective metric."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if objective not in {"f1", "bacc"}:
        raise ValueError("objective must be 'f1' or 'bacc'")

    # Add edges so search can represent predict-all-positive / predict-all-negative.
    candidates = np.unique(
        np.concatenate(
            [
                y_score,
                np.array([0.0, 1.0, np.nextafter(0.0, -1.0), np.nextafter(1.0, 2.0)]),
            ]
        )
    )

    best_thr = 0.5
    best_metrics = binary_metrics(y_true, y_score, best_thr)
    best_value = best_metrics[objective]

    for thr in candidates:
        m = binary_metrics(y_true, y_score, float(thr))
        value = m[objective]
        if (value > best_value) or (
            np.isclose(value, best_value) and abs(float(thr) - 0.5) < abs(best_thr - 0.5)
        ):
            best_value = value
            best_thr = float(thr)
            best_metrics = m

    return best_thr, best_metrics


def load_dataloader(dataset_path):
    with open(dataset_path, "r") as f:
        datasetfile = json.loads(f.read())
    module_name = ".".join(
        dataset_path.replace("\\", "/").split("/")[:-1]
        + [datasetfile["dataloader"]]
    )
    dl = importlib.import_module(module_name)
    return datasetfile, dl


def get_output_path(container, label, shard, split):
    if split == "test":
        legacy = f"containers/{container}/outputs/shard-{shard}:{label}.npy"
        if os.path.exists(legacy):
            return legacy
    return f"containers/{container}/outputs/shard-{shard}:{label}:{split}.npy"


def load_preds(container, label, split):
    preds = {}
    for shard, task in enumerate(OVR_TASKS):
        path = get_output_path(container, label, shard, split)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing output ({split}): {path}")
        out = np.load(path, allow_pickle=True)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        preds[task] = out[:, 0]
    return preds


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá UTKFace OVR (10 head binary) trên test set."
    )
    parser.add_argument("--container", default="utkface_ovr")
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", default="datasets/UTKFace/datasetfile_ovr")
    parser.add_argument(
        "--tune_thresholds",
        action="store_true",
        help="Tune threshold riêng cho từng head.",
    )
    parser.add_argument(
        "--tune_objective",
        default="f1",
        choices=["f1", "bacc"],
        help="Metric mục tiêu để tune threshold từng head.",
    )
    parser.add_argument(
        "--save_thresholds",
        action="store_true",
        help="Lưu threshold từng head ra JSON trong outputs.",
    )
    parser.add_argument(
        "--tune_split",
        default="val",
        choices=["val", "test"],
        help="Split dùng để tune threshold (khuyến nghị val).",
    )
    parser.add_argument(
        "--eval_split",
        default="test",
        choices=["val", "test"],
        help="Split dùng để báo cáo metric cuối.",
    )
    args = parser.parse_args()

    ds, dl = load_dataloader(args.dataset)
    eval_idx = np.arange(ds[f"nb_{args.eval_split}"])
    _, y_eval_dict = dl.load_ovr(eval_idx, category=args.eval_split)

    y_tune_dict = None
    preds_tune = None
    if args.tune_thresholds:
        tune_key = f"nb_{args.tune_split}"
        if tune_key not in ds:
            raise KeyError(
                f"{tune_key} không có trong datasetfile. "
                "Hãy prepare lại để có split tương ứng."
            )
        tune_idx = np.arange(ds[tune_key])
        _, y_tune_dict = dl.load_ovr(tune_idx, category=args.tune_split)
        preds_tune = load_preds(args.container, args.label, args.tune_split)

    print("=" * 70)
    print("UTKFACE OVR EVALUATION")
    print("=" * 70)
    print(f"Container: {args.container}")
    print(f"Label    : {args.label}")
    print(f"Dataset  : {args.dataset}")
    print(f"Eval split: {args.eval_split}")
    if args.tune_thresholds:
        print(f"Tune split: {args.tune_split}")
    print()

    preds_eval = load_preds(args.container, args.label, args.eval_split)

    # Tính metrics từng head
    accs = {}
    baccs = {}
    f1s = {}
    pras = {}
    thresholds = {}

    print("Per-head metrics:")
    print(
        f"{'task':<15} {'thr':>7} {'acc':>8} {'bacc':>8} "
        f"{'f1':>8} {'pr_auc':>8}"
    )
    print("-" * 70)

    for task in OVR_TASKS:
        y_true = np.asarray(y_eval_dict[task], dtype=np.int64)
        y_score = np.asarray(preds_eval[task], dtype=np.float64)

        if args.tune_thresholds:
            y_tune = np.asarray(y_tune_dict[task], dtype=np.int64)
            score_tune = np.asarray(preds_tune[task], dtype=np.float64)
            thr, _ = tune_threshold(y_tune, score_tune, objective=args.tune_objective)
            m = binary_metrics(y_true, y_score, thr)
        else:
            thr = 0.5
            m = binary_metrics(y_true, y_score, thr)

        pr_auc = average_precision_score(y_true, y_score)

        thresholds[task] = float(thr)
        accs[task] = m["acc"]
        baccs[task] = m["bacc"]
        f1s[task] = m["f1"]
        pras[task] = pr_auc

        print(
            f"{task:<15} {thr:7.4f} {m['acc']*100:8.2f} {m['bacc']*100:8.2f} "
            f"{m['f1']*100:8.2f} {pr_auc*100:8.2f}"
        )

    # Group-level accuracy (gender / age / race)
    print("\nGroup-level accuracy (argmax trong mỗi group):")

    # Gender (2 head)
    g0 = preds_eval["gender_female"]
    g1 = preds_eval["gender_male"]
    gender_hat = (g1 > g0).astype(np.int64)  # 0=female,1=male
    y_gender = np.where(y_eval_dict["gender_male"] == 1, 1, 0)
    gender_acc = float((gender_hat == y_gender).mean())
    print(f"Gender: {gender_acc*100:6.2f}%")

    # Age (3 bins)
    a0 = preds_eval["age_bin0"]
    a1 = preds_eval["age_bin1"]
    a2 = preds_eval["age_bin2"]
    age_stack = np.stack([a0, a1, a2], axis=1)  # (N,3)
    age_hat = np.argmax(age_stack, axis=1)
    # ground truth: 0 if age_bin0==1, etc.
    y_age = np.zeros_like(age_hat)
    y_age[y_eval_dict["age_bin1"] == 1] = 1
    y_age[y_eval_dict["age_bin2"] == 1] = 2
    age_acc = float((age_hat == y_age).mean())
    print(f"Age bins: {age_acc*100:6.2f}%")

    # Race (5 heads)
    r_stack = np.stack(
        [
            preds_eval["race_white"],
            preds_eval["race_black"],
            preds_eval["race_asian"],
            preds_eval["race_indian"],
            preds_eval["race_others"],
        ],
        axis=1,
    )  # (N,5)
    race_hat = np.argmax(r_stack, axis=1)  # 0..4
    # ground truth từ y_dict
    y_race = np.zeros_like(race_hat)
    for i, name in enumerate(
        ["race_white", "race_black", "race_asian", "race_indian", "race_others"]
    ):
        y_race[y_eval_dict[name] == 1] = i
    race_acc = float((race_hat == y_race).mean())
    print(f"Race   : {race_acc*100:6.2f}%")

    if args.tune_thresholds:
        print(
            f"\nThreshold tuning: ON (objective={args.tune_objective}, "
            f"tune_split={args.tune_split})"
        )
    else:
        print("\nThreshold tuning: OFF (fixed threshold=0.5)")

    print("Mean head accuracy :", float(np.mean(list(accs.values()))) * 100, "%")
    print("Mean head bal. acc :", float(np.mean(list(baccs.values()))) * 100, "%")
    print("Mean head F1       :", float(np.mean(list(f1s.values()))) * 100, "%")
    print("Mean head PR-AUC   :", float(np.mean(list(pras.values()))) * 100, "%")

    if args.save_thresholds:
        path = f"containers/{args.container}/outputs/thresholds:{args.label}.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "label": args.label,
                    "tuned": bool(args.tune_thresholds),
                    "objective": args.tune_objective if args.tune_thresholds else None,
                    "thresholds": thresholds,
                },
                f,
                indent=2,
            )
        print(f"Saved thresholds: {path}")

    print("=" * 70)


if __name__ == "__main__":
    main()

