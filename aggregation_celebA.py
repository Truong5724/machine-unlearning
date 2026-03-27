import argparse
import importlib
import importlib.util
import json
import os

import numpy as np
import torch

from architectures.celeba_ovr import OVRModel, OVR_TASKS


def safe_div(num, den):
    if den == 0:
        return 0.0
    return float(num) / float(den)


def binary_confusion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def binary_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)

    tp, tn, fp, fn = binary_confusion(y_true, y_pred)

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
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


def tune_threshold(y_true, y_score, objective="f1"):
    """Find best threshold for a binary task according to objective metric."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if objective not in {"f1", "bacc"}:
        raise ValueError("objective must be 'f1' or 'bacc'")

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


def load_dataset_config(datasetfile_path):
    with open(datasetfile_path, "r") as f:
        datasetfile = json.load(f)

    datasetfile_abs = os.path.abspath(datasetfile_path)
    dataset_dir = os.path.dirname(datasetfile_abs)
    dataloader_name = datasetfile["dataloader"]

    module_name = ".".join(
        datasetfile_path.replace("\\", "/").split("/")[:-1]
        + [dataloader_name]
    )

    try:
        dataloader = importlib.import_module(module_name)
    except Exception:
        py_path = os.path.join(dataset_dir, f"{dataloader_name}.py")
        if not os.path.exists(py_path):
            raise
        spec = importlib.util.spec_from_file_location(dataloader_name, py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load dataloader module from {py_path}")
        dataloader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataloader)

    return datasetfile, dataloader


def load_thresholds_file(path):
    with open(path, "r") as f:
        payload = json.load(f)

    th = payload.get("thresholds", payload)
    out = {}
    for task in OVR_TASKS:
        if task in th:
            out[task] = float(th[task])
    return out


def parse_task_list(csv_text):
    if not csv_text.strip():
        return []
    return [x.strip() for x in csv_text.split(",") if x.strip()]


def resolve_eval_tasks(include_tasks, exclude_tasks):
    include_set = set(include_tasks) if include_tasks else set(OVR_TASKS)
    exclude_set = set(exclude_tasks)
    tasks = [t for t in OVR_TASKS if t in include_set and t not in exclude_set]
    return tasks


def load_shard_model(container, shard, input_shape, dropout_rate, device):
    ckpt_link = f"containers/{container}/cache/shard-{shard}.pt"
    if not os.path.exists(ckpt_link):
        return None

    model = OVRModel(input_shape=input_shape, dropout_rate=dropout_rate).to(device)
    state = torch.load(ckpt_link, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_task(model, task, x_tensor, batch_size=256):
    probs = []
    with torch.no_grad():
        for i in range(0, x_tensor.size(0), batch_size):
            xb = x_tensor[i : i + batch_size]
            logits = model.forward_task(xb, task)
            pb = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(pb)
    if not probs:
        return np.array([], dtype=np.float32)
    return np.concatenate(probs, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate/evaluate CelebA OVR models. Bo qua shard khong ton tai."
    )
    parser.add_argument("--container", required=True)
    parser.add_argument("--dataset", default="datasets/celebA/datasetfile_ovr")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--include_tasks", default="", help="CSV task names to keep")
    parser.add_argument("--exclude_tasks", default="", help="CSV task names to drop")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--thresholds_file",
        default="",
        help="Path JSON thresholds per-task (vd: outputs/thresholds:celebA.json)",
    )
    parser.add_argument(
        "--tune_thresholds",
        action="store_true",
        help="Tune threshold riêng cho từng task bằng split --tune_split",
    )
    parser.add_argument(
        "--tune_split",
        default="val",
        choices=["train", "val", "test"],
        help="Split dùng để tune threshold (khuyến nghị val)",
    )
    parser.add_argument(
        "--tune_objective",
        default="f1",
        choices=["f1", "bacc"],
        help="Metric mục tiêu khi tune threshold",
    )
    parser.add_argument(
        "--save_thresholds",
        action="store_true",
        help="Lưu threshold từng task vào outputs/thresholds:celebA.json",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="", help="cuda:0, cpu, ...")
    parser.add_argument("--save_json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    if args.tune_thresholds and args.thresholds_file:
        raise ValueError("Không dùng đồng thời --tune_thresholds và --thresholds_file")

    unknown = [
        x for x in parse_task_list(args.include_tasks) + parse_task_list(args.exclude_tasks)
        if x not in OVR_TASKS
    ]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")

    eval_tasks = resolve_eval_tasks(
        parse_task_list(args.include_tasks), parse_task_list(args.exclude_tasks)
    )
    if not eval_tasks:
        raise ValueError("No task left to evaluate after include/exclude filtering")

    ds, dl = load_dataset_config(args.dataset)
    n = int(ds[f"nb_{args.split}"])
    idx = np.arange(n, dtype=np.int64)
    x_np, y_dict = dl.load_ovr(idx, category=args.split)

    x_tune = None
    y_tune_dict = None
    thresholds_from_file = {}
    if args.thresholds_file:
        if not os.path.exists(args.thresholds_file):
            raise FileNotFoundError(f"Missing thresholds file: {args.thresholds_file}")
        thresholds_from_file = load_thresholds_file(args.thresholds_file)
    if args.tune_thresholds:
        tune_key = f"nb_{args.tune_split}"
        if tune_key not in ds:
            raise KeyError(
                f"{tune_key} không có trong datasetfile. Hãy prepare lại để có split tương ứng."
            )
        tune_n = int(ds[tune_key])
        tune_idx = np.arange(tune_n, dtype=np.int64)
        x_tune_np, y_tune_dict = dl.load_ovr(tune_idx, category=args.tune_split)
        x_tune = torch.from_numpy(x_tune_np)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    x = torch.from_numpy(x_np).to(device)
    if x_tune is not None:
        x_tune = x_tune.to(device)

    metrics_by_task = {}
    thresholds_by_task = {}
    available = []
    missing = []

    for shard, task in enumerate(OVR_TASKS):
        if task not in eval_tasks:
            continue

        model = load_shard_model(args.container, shard, tuple(ds["input_shape"]), args.dropout_rate, device)
        if model is None:
            missing.append(task)
            continue

        y_true = np.asarray(y_dict[task], dtype=np.int64)
        y_score = predict_task(model, task, x, batch_size=args.batch_size)

        thr = float(thresholds_from_file.get(task, args.threshold))
        if args.tune_thresholds:
            y_tune = np.asarray(y_tune_dict[task], dtype=np.int64)
            y_score_tune = predict_task(model, task, x_tune, batch_size=args.batch_size)
            thr, _ = tune_threshold(y_tune, y_score_tune, objective=args.tune_objective)

        m = binary_metrics(y_true, y_score, threshold=thr)
        m["n"] = int(len(y_true))
        m["pos_ratio"] = float(np.mean(y_true)) if len(y_true) else 0.0
        m["threshold"] = thr
        metrics_by_task[task] = m
        thresholds_by_task[task] = thr
        available.append(task)

    if not available:
        raise FileNotFoundError(
            f"No shard checkpoint found in containers/{args.container}/cache for selected tasks"
        )

    macro_acc = float(np.mean([metrics_by_task[t]["acc"] for t in available]))
    macro_f1 = float(np.mean([metrics_by_task[t]["f1"] for t in available]))

    print("=" * 72)
    print("CELEBA OVR AGGREGATION")
    print("=" * 72)
    print(f"Container : {args.container}")
    print(f"Dataset   : {args.dataset}")
    print(f"Split     : {args.split}")
    print(f"Device    : {device}")
    if args.tune_thresholds:
        print(f"Threshold : tuned on {args.tune_split} ({args.tune_objective})")
    elif args.thresholds_file:
        print(f"Threshold : loaded from {args.thresholds_file}")
    else:
        print(f"Threshold : fixed={args.threshold}")
    print(f"Selected  : {len(eval_tasks)} tasks")
    print(f"Available : {len(available)} tasks")
    if missing:
        print(f"Missing   : {', '.join(missing)}")
    print()
    print(f"{'task':<22} {'thr':>7} {'acc':>8} {'bacc':>8} {'f1':>8} {'prec':>8} {'rec':>8} {'pos%':>8}")
    print("-" * 72)
    for task in available:
        m = metrics_by_task[task]
        print(
            f"{task:<22} {m['threshold']:7.4f} {m['acc']:8.4f} {m['bacc']:8.4f} {m['f1']:8.4f} {m['precision']:8.4f} "
            f"{m['recall']:8.4f} {100.0*m['pos_ratio']:8.2f}"
        )

    print("-" * 72)
    print(f"Macro-ACC : {macro_acc:.4f}")
    print(f"Macro-F1  : {macro_f1:.4f}")
    print("=" * 72)

    report = {
        "container": args.container,
        "dataset": args.dataset,
        "split": args.split,
        "threshold": args.threshold,
        "thresholds_file": args.thresholds_file,
        "tune_thresholds": bool(args.tune_thresholds),
        "tune_split": args.tune_split,
        "tune_objective": args.tune_objective,
        "selected_tasks": eval_tasks,
        "available_tasks": available,
        "missing_tasks": missing,
        "macro_acc": macro_acc,
        "macro_f1": macro_f1,
        "thresholds": thresholds_by_task,
        "by_task": metrics_by_task,
    }

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report: {args.save_json}")

    if args.save_thresholds:
        thr_path = f"containers/{args.container}/outputs/thresholds:celebA.json"
        payload = {
            "container": args.container,
            "dataset": args.dataset,
            "tune_split": args.tune_split if args.tune_thresholds else None,
            "objective": args.tune_objective if args.tune_thresholds else None,
            "thresholds": thresholds_by_task,
        }
        with open(thr_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved thresholds: {thr_path}")


if __name__ == "__main__":
    main()
