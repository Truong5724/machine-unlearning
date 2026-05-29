import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from architectures.celeba_ovr import OVRModel, OVR_TASKS


def load_dataset_input_shape(dataset_path: str) -> Tuple[int, int, int]:
    if not os.path.exists(dataset_path):
        return (3, 64, 64)

    with open(dataset_path, "r") as f:
        ds = json.load(f)

    shape = ds.get("input_shape", [3, 64, 64])
    if len(shape) != 3:
        return (3, 64, 64)
    return tuple(int(x) for x in shape)


def preprocess_image(image_path: str, input_shape: Tuple[int, int, int]) -> torch.Tensor:
    _, h, w = input_shape
    img = Image.open(image_path).convert("RGB")
    img = img.resize((w, h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr)


def load_thresholds(container: str, thresholds_file: str) -> Dict[str, float]:
    thresholds = {task: 0.5 for task in OVR_TASKS}

    if not thresholds_file:
        return thresholds

    if os.path.isdir(thresholds_file):
        for task in OVR_TASKS:
            task_path = os.path.join(thresholds_file, f"thresholds:{task}.json")
            if not os.path.exists(task_path):
                continue
            with open(task_path, "r") as f:
                payload = json.load(f)
            if "threshold" in payload:
                thresholds[task] = float(payload["threshold"])
            elif task in payload.get("thresholds", {}):
                thresholds[task] = float(payload["thresholds"][task])
        return thresholds

    with open(thresholds_file, "r") as f:
        payload = json.load(f)

    saved = payload.get("thresholds", payload)
    for task in OVR_TASKS:
        if task in saved:
            thresholds[task] = float(saved[task])
    return thresholds


def infer_single_image(args):
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    input_shape = load_dataset_input_shape(args.dataset)
    x = preprocess_image(args.image, input_shape).to(device)
    thresholds = load_thresholds(args.container, args.thresholds_file)

    probs: Dict[str, float] = {}
    preds_bin: Dict[str, int] = {}
    missing_tasks = []

    for shard, task in enumerate(OVR_TASKS):
        ckpt = f"containers/{args.container}/cache/shard-{shard}.pt"
        if not os.path.exists(ckpt):
            missing_tasks.append(task)
            continue

        model = OVRModel(input_shape=input_shape, dropout_rate=args.dropout_rate).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            logits = model.forward_task(x, task)
            prob = torch.sigmoid(logits)[0].item()

        thr = thresholds.get(task, 0.5)
        pred = int(prob >= thr)
        probs[task] = float(prob)
        preds_bin[task] = pred

    if not probs:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint shard nào trong containers/{args.container}/cache."
        )

    print("=" * 72)
    print("CELEBA OVR SINGLE-IMAGE INFERENCE")
    print("=" * 72)
    print(f"Image      : {args.image}")
    print(f"Container  : {args.container}")
    print(f"Device     : {device}")
    print(f"Loaded head: {len(probs)}/{len(OVR_TASKS)}")
    if missing_tasks:
        print(f"Missing    : {', '.join(missing_tasks)}")

    print("\nPer-head outputs:")
    print(f"{'task':<22} {'prob':>8} {'thr':>8} {'pred':>8} {'label':>10}")
    print("-" * 72)
    for task in OVR_TASKS:
        if task not in probs:
            continue
        thr = thresholds.get(task, 0.5)
        label = "yes" if preds_bin[task] == 1 else "no"
        print(f"{task:<22} {probs[task]:8.4f} {thr:8.4f} {preds_bin[task]:8d} {label:>10}")

    positive_tasks = [task for task, pred in preds_bin.items() if pred == 1]
    print("\nSummary:")
    print(f"Positive attributes: {len(positive_tasks)}")
    if positive_tasks:
        print(f"Predicted yes      : {', '.join(positive_tasks)}")
    else:
        print("Predicted yes      : <none>")
    print("=" * 72)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Infer 1 ảnh CelebA với các checkpoint OVR hiện có."
    )
    parser.add_argument("--image", required=True, help="Path tới ảnh CelebA")
    parser.add_argument("--container", default="celeba_ovr", help="Tên container")
    parser.add_argument(
        "--dataset",
        default="datasets/celebA/datasetfile_ovr",
        help="Path datasetfile_ovr để lấy input_shape",
    )
    parser.add_argument(
        "--thresholds_file",
        default="containers/celeba_ovr/outputs/thresholds",
        help="Path JSON threshold hoặc thư mục thresholds",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--device", default="", help="cuda:0, cpu, ...")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    infer_single_image(args)