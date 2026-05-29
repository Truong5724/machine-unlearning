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

ATTRIBUTE_GROUPS = [
    ["gender_female", "gender_male"],
    ["age_bin0", "age_bin1", "age_bin2"],
    ["race_white", "race_black", "race_asian", "race_indian", "race_others"],
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


def parse_exclude_tasks(text):
    if text is None or str(text).strip() == "":
        return []
    raw = [x.strip() for x in str(text).split(",") if x.strip()]
    invalid = [x for x in raw if x not in OVR_TASKS]
    if invalid:
        raise ValueError(
            f"exclude task không hợp lệ: {invalid}. Hợp lệ: {OVR_TASKS}"
        )
    seen = set()
    out = []
    for t in raw:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def load_preds(container, label, split, expected_len, skip_tasks=None):
    skip_tasks = set(skip_tasks or [])
    preds = {}
    missing = []
    for shard, task in enumerate(OVR_TASKS):
        if task in skip_tasks:
            continue
        path = get_output_path(container, label, shard, split)
        if not os.path.exists(path):
            missing.append((shard, task, path))
            preds[task] = np.zeros(expected_len, dtype=np.float32)
            continue
        out = np.load(path, allow_pickle=True)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        preds[task] = out[:, 0]
    return preds, missing


def apply_exclusion_filter(y_dict, preds, exclude_tasks):
    """Drop samples where any excluded task is positive (y==1)."""
    keep_mask = np.ones(len(np.asarray(y_dict[OVR_TASKS[0]])), dtype=bool)
    for t in exclude_tasks:
        keep_mask &= (np.asarray(y_dict[t], dtype=np.int64) == 0)

    filtered_y = {}
    filtered_preds = {}
    for task in OVR_TASKS:
        filtered_y[task] = np.asarray(y_dict[task])[keep_mask]
    for task in preds.keys():
        filtered_preds[task] = np.asarray(preds[task])[keep_mask]

    removed = int((~keep_mask).sum())
    kept = int(keep_mask.sum())
    return filtered_y, filtered_preds, removed, kept


def group_accuracy(task_list, y_dict, preds, exclude_tasks=None, missing_tasks=None):
    """Compute group accuracy by argmax across a group's heads.

    If all heads for the group are missing/unlearned, return (0.0, 0)
    so the report shows 0% for that group.
    """
    exclude_set = set(exclude_tasks or [])
    missing_set = set(missing_tasks or [])
    tasks = [t for t in task_list if t in OVR_TASKS]
    tasks = [t for t in tasks if t not in exclude_set]
    # Exclude tasks that are missing/unlearned (they were zero-filled earlier)
    tasks = [t for t in tasks if t in preds and t not in missing_set]

    # If no available heads remain for this group, return 0.0 (user expectation)
    if len(tasks) == 0:
        return 0.0, 0

    stack = np.stack([preds[t] for t in tasks], axis=1)
    y_hat = np.argmax(stack, axis=1)

    y_true = np.zeros_like(y_hat)
    for i, t in enumerate(tasks):
        y_true[np.asarray(y_dict[t], dtype=np.int64) == 1] = i

    acc = float((y_hat == y_true).mean())
    return acc, len(tasks)


def active_tasks_for_mean(attribute_groups, missing_tasks=None, exclude_tasks=None):
    """Return tasks that should count toward overall mean metrics.

    A whole attribute group is dropped only when every task in that group is
    missing/unlearned. If only some tasks are missing, the group still counts
    normally and the missing tasks remain in the mean with their zero metrics.
    """
    missing_set = set(missing_tasks or [])
    exclude_set = set(exclude_tasks or [])
    active = []
    for group in attribute_groups:
        group_tasks = [t for t in group if t in OVR_TASKS and t not in exclude_set]
        if not group_tasks:
            continue
        if all(t in missing_set for t in group_tasks):
            continue
        active.extend(group_tasks)
    return active


def mean_over_tasks(metric_map, tasks):
    values = [float(metric_map[t]) for t in tasks if t in metric_map]
    if not values:
        return 0.0
    return float(np.mean(values))


def save_thresholds_combined(container, label, tune_enabled, objective, thresholds):
    """Save thresholds as one combined file for the whole label."""
    out_dir = f"containers/{container}/outputs/thresholds"
    os.makedirs(out_dir, exist_ok=True)

    combined_path = f"{out_dir}/thresholds:{label}.json"
    with open(combined_path, "w") as f:
        json.dump(
            {
                "label": label,
                "tuned": bool(tune_enabled),
                "objective": objective if tune_enabled else None,
                "thresholds": {k: float(v) for k, v in thresholds.items()},
            },
            f,
            indent=2,
        )

    return out_dir, combined_path


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
    parser.add_argument(
        "--exclude_eval_tasks",
        default="",
        help="Danh sách task cần loại khỏi eval, ngăn cách bởi dấu phẩy (vd: race_asian,age_bin1).",
    )
    parser.add_argument(
        "--exclude_tune_too",
        action="store_true",
        help="Nếu bật, áp dụng cùng lọc loại class cho tune split.",
    )
    args = parser.parse_args()

    ds, dl = load_dataloader(args.dataset)
    eval_idx = np.arange(ds[f"nb_{args.eval_split}"])
    _, y_eval_dict = dl.load_ovr(eval_idx, category=args.eval_split)

    exclude_tasks = parse_exclude_tasks(args.exclude_eval_tasks)
    exclude_task_set = set(exclude_tasks)

    y_tune_dict = None
    preds_tune = None
    missing_tune_tasks = set()
    if args.tune_thresholds:
        tune_key = f"nb_{args.tune_split}"
        if tune_key not in ds:
            raise KeyError(
                f"{tune_key} không có trong datasetfile. "
                "Hãy prepare lại để có split tương ứng."
            )
        tune_idx = np.arange(ds[tune_key])
        _, y_tune_dict = dl.load_ovr(tune_idx, category=args.tune_split)
        preds_tune, missing_tune = load_preds(
            args.container,
            args.label,
            args.tune_split,
            len(np.asarray(y_tune_dict[OVR_TASKS[0]])),
            skip_tasks=exclude_task_set,
        )
        missing_tune_tasks = {task for _, task, _ in missing_tune}
    else:
        missing_tune = []
    missing_eval_tasks = set()

    if exclude_tasks:
        preds_eval, missing_eval = load_preds(
            args.container,
            args.label,
            args.eval_split,
            len(np.asarray(y_eval_dict[OVR_TASKS[0]])),
            skip_tasks=exclude_task_set,
        )
        missing_eval_tasks = {task for _, task, _ in missing_eval}
        y_eval_dict, preds_eval, removed_eval, kept_eval = apply_exclusion_filter(
            y_eval_dict,
            preds_eval,
            exclude_tasks,
        )

        if kept_eval == 0:
            raise ValueError(
                f"Sau khi loại class {exclude_tasks}, eval split không còn mẫu nào."
            )

        if args.tune_thresholds and args.exclude_tune_too:
            y_tune_dict, preds_tune, removed_tune, kept_tune = apply_exclusion_filter(
                y_tune_dict,
                preds_tune,
                exclude_tasks,
            )
            if kept_tune == 0:
                raise ValueError(
                    f"Sau khi loại class {exclude_tasks}, tune split không còn mẫu nào."
                )
    else:
        preds_eval, missing_eval = load_preds(
            args.container,
            args.label,
            args.eval_split,
            len(np.asarray(y_eval_dict[OVR_TASKS[0]])),
        )
        missing_eval_tasks = {task for _, task, _ in missing_eval}

    print("=" * 70)
    print("UTKFACE OVR EVALUATION")
    print("=" * 70)
    print(f"Container: {args.container}")
    print(f"Label    : {args.label}")
    print(f"Dataset  : {args.dataset}")
    print(f"Eval split: {args.eval_split}")
    if args.tune_thresholds:
        print(f"Tune split: {args.tune_split}")
    if exclude_tasks:
        print(f"Exclude tasks from eval positives: {', '.join(exclude_tasks)}")
        print(f"Eval kept/removed: {kept_eval}/{removed_eval}")
        if args.tune_thresholds and args.exclude_tune_too:
            print(f"Tune kept/removed: {kept_tune}/{removed_tune}")
    if missing_eval:
        miss_str = ", ".join(f"{task}@shard{shard}" for shard, task, _ in missing_eval)
        print(f"Missing eval outputs skipped: {miss_str}")
    if args.tune_thresholds and missing_tune:
        miss_str = ", ".join(f"{task}@shard{shard}" for shard, task, _ in missing_tune)
        print(f"Missing tune outputs skipped: {miss_str}")
    print()

    # Tính metrics từng head
    accs = {}
    baccs = {}
    f1s = {}
    pras = {}
    head_supports = {}
    thresholds = {}

    print("Per-head metrics:")
    print(
        f"{'task':<15} {'thr':>7} {'acc':>8} {'bacc':>8} "
        f"{'f1':>8} {'pr_auc':>8}"
    )
    print("-" * 70)

    for task in OVR_TASKS:
        if task in exclude_task_set:
            if args.tune_thresholds and task in preds_tune:
                y_tune = np.asarray(y_tune_dict[task], dtype=np.int64)
                score_tune = np.asarray(preds_tune[task], dtype=np.float64)
                thr, _ = tune_threshold(y_tune, score_tune, objective=args.tune_objective)
            else:
                thr = 0.5

            thresholds[task] = float(thr)
            print(
                f"{task:<15} {thr:7.4f} {'-':>8} {'-':>8} {'-':>8} {'-':>8}  "
                "(skipped)"
            )
            continue

        if task in missing_eval_tasks:
            thresholds[task] = 0.0
            accs[task] = 0.0
            baccs[task] = 0.0
            f1s[task] = 0.0
            pras[task] = 0.0
            head_supports[task] = int(np.sum(np.asarray(y_eval_dict[task], dtype=np.int64) == 1))
            print(
                f"{task:<15} {0.0:7.4f} {0.0:8.2f} {0.0:8.2f} {0.0:8.2f} {0.0:8.2f}  "
                "(missing/unlearned)"
            )
            continue

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
        head_supports[task] = int(np.sum(y_true == 1))

        print(
            f"{task:<15} {thr:7.4f} {m['acc']*100:8.2f} {m['bacc']*100:8.2f} "
            f"{m['f1']*100:8.2f} {pr_auc*100:8.2f}"
        )

    # Group-level accuracy (gender / age / race)
    print("\nGroup-level accuracy (argmax trong mỗi group):")

    gender_acc, g_heads = group_accuracy(
        ["gender_female", "gender_male"],
        y_eval_dict,
        preds_eval,
        exclude_tasks=exclude_tasks,
        missing_tasks=missing_eval_tasks,
    )
    age_acc, a_heads = group_accuracy(
        ["age_bin0", "age_bin1", "age_bin2"],
        y_eval_dict,
        preds_eval,
        exclude_tasks=exclude_tasks,
        missing_tasks=missing_eval_tasks,
    )
    race_acc, r_heads = group_accuracy(
        ["race_white", "race_black", "race_asian", "race_indian", "race_others"],
        y_eval_dict,
        preds_eval,
        exclude_tasks=exclude_tasks,
        missing_tasks=missing_eval_tasks,
    )

    if gender_acc is None:
        print("Gender: N/A (no head after exclusion)")
    else:
        suffix = "" if g_heads == 2 else f" (partial heads {g_heads}/2)"
        print(f"Gender: {gender_acc*100:6.2f}%{suffix}")

    if age_acc is None:
        print("Age bins: N/A (no head after exclusion)")
    else:
        suffix = "" if a_heads == 3 else f" (partial heads {a_heads}/3)"
        print(f"Age bins: {age_acc*100:6.2f}%{suffix}")

    if race_acc is None:
        print("Race   : N/A (no head after exclusion)")
    else:
        suffix = "" if r_heads == 5 else f" (partial heads {r_heads}/5)"
        print(f"Race   : {race_acc*100:6.2f}%{suffix}")

    if args.tune_thresholds:
        print(
            f"\nThreshold tuning: ON (objective={args.tune_objective}, "
            f"tune_split={args.tune_split})"
        )
    else:
        print("\nThreshold tuning: OFF (fixed threshold=0.5)")

    mean_tasks = active_tasks_for_mean(
        ATTRIBUTE_GROUPS,
        missing_tasks=missing_eval_tasks,
        exclude_tasks=exclude_tasks,
    )

    print(
        "Mean head accuracy :",
        mean_over_tasks(accs, mean_tasks) * 100,
        "% (skip fully missing attrs)",
    )
    support_sum = float(np.sum([float(head_supports[t]) for t in mean_tasks if t in head_supports]))
    if support_sum > 0:
        weighted_mean_acc = float(
            np.sum([
                float(accs[t]) * float(head_supports[t])
                for t in mean_tasks
                if t in accs and t in head_supports
            ]) / support_sum
        )
    else:
        weighted_mean_acc = 0.0
    print("Weighted mean acc  :", weighted_mean_acc * 100, "% (pos-support weighted)")
    print("Mean head bal. acc :", mean_over_tasks(baccs, mean_tasks) * 100, "%")
    print("Mean head F1       :", mean_over_tasks(f1s, mean_tasks) * 100, "%")
    print("Mean head PR-AUC   :", mean_over_tasks(pras, mean_tasks) * 100, "%")

    if args.save_thresholds:
        out_dir, legacy_path = save_thresholds_combined(
            args.container,
            args.label,
            args.tune_thresholds,
            args.tune_objective,
            thresholds,
        )
        print(f"Saved thresholds: {legacy_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()

