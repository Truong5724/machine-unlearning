import argparse
import json
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from architectures.utkface_ovr import OVRModel, OVR_TASKS


RACE_LABELS = ["white", "black", "asian", "indian", "others"]


def parse_utkface_filename(image_path: str) -> Optional[Tuple[int, int, int]]:
    """Parse age/gender/race from UTKFace filename: age_gender_race_*.jpg."""
    name = os.path.basename(image_path)
    match = re.match(r"^(\d+)_(\d)_(\d)_", name)
    if not match:
        return None

    age = int(match.group(1))
    gender = int(match.group(2))
    race = int(match.group(3))
    if gender not in (0, 1) or race < 0 or race > 4:
        return None
    return age, gender, race


def gender_text_repo(gender_code: int) -> str:
    # User-selected convention for filename parsing.
    return "male" if gender_code == 0 else "female"


def age_to_bin(age: int) -> int:
    if age <= 18:
        return 0
    if age <= 60:
        return 1
    return 2


def load_dataset_input_shape(dataset_path: str) -> Tuple[int, int, int]:
    if not os.path.exists(dataset_path):
        return (3, 64, 64)

    with open(dataset_path, "r") as f:
        ds = json.load(f)

    shape = ds.get("input_shape", [3, 64, 64])
    if len(shape) != 3:
        return (3, 64, 64)
    return tuple(int(x) for x in shape)


def load_thresholds(container: str, label: str) -> Dict[str, float]:
    thresholds = {task: 0.5 for task in OVR_TASKS}

    # Preferred format: one threshold file per task/model.
    has_per_task = False
    for shard, task in enumerate(OVR_TASKS):
        task_candidates = [
            f"containers/{container}/outputs/thresholds/thresholds:{task}:{label}.json",
            f"containers/{container}/outputs/thresholds/thresholds:{task}.json",
        ]
        for path in task_candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                thresholds[task] = float(data.get("threshold", 0.5))
                has_per_task = True
                break
            except Exception:
                pass

    if has_per_task:
        return thresholds

    # Older format: one threshold file per shard/model.
    has_per_shard = False
    for shard, task in enumerate(OVR_TASKS):
        path = f"containers/{container}/outputs/thresholds/shard-{shard}:{label}.json"
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            thresholds[task] = float(data.get("threshold", 0.5))
            has_per_shard = True
        except Exception:
            pass

    if has_per_shard:
        return thresholds

    # Combined file format in thresholds directory.
    combined_candidates = [
        f"containers/{container}/outputs/thresholds/thresholds:{label}.json",
        # Backward compatibility with older location.
        f"containers/{container}/outputs/thresholds:{label}.json",
    ]

    data = None
    for path in combined_candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            break

    if data is None:
        return thresholds

    saved = data.get("thresholds", {})
    for task in OVR_TASKS:
        if task in saved:
            thresholds[task] = float(saved[task])
    return thresholds


def preprocess_image(image_path: str, input_shape: Tuple[int, int, int]) -> torch.Tensor:
    _, h, w = input_shape
    img = Image.open(image_path).convert("RGB")
    img = img.resize((w, h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr)


def infer_single_image(args):
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    input_shape = load_dataset_input_shape(args.dataset)
    x = preprocess_image(args.image, input_shape).to(device)

    thresholds = load_thresholds(args.container, args.label)

    probs: Dict[str, float] = {}
    preds_bin: Dict[str, int] = {}
    loaded_tasks = []
    missing_tasks = []
    broken_link_tasks = []

    for shard, task in enumerate(OVR_TASKS):
        ckpt = f"containers/{args.container}/cache/shard-{shard}:{args.label}.pt"
        if os.path.islink(ckpt) and not os.path.exists(ckpt):
            broken_link_tasks.append(task)
            missing_tasks.append(task)
            continue
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
        loaded_tasks.append(task)

    if len(loaded_tasks) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint shard nào trong containers/{args.container}/cache cho label={args.label}."
        )

    print("=" * 72)
    print("UTKFACE OVR SINGLE-IMAGE INFERENCE")
    print("=" * 72)
    print(f"Image      : {args.image}")
    print(f"Container  : {args.container}")
    print(f"Label      : {args.label}")
    print(f"Device     : {device}")
    print(f"Loaded head: {len(loaded_tasks)}/{len(OVR_TASKS)}")
    if missing_tasks:
        print(f"Missing    : {', '.join(missing_tasks)}")
    if broken_link_tasks:
        print(f"Broken link: {', '.join(broken_link_tasks)}")

    print("\nPer-head outputs:")
    print(f"{'task':<15} {'prob':>8} {'thr':>8} {'pred':>8}")
    print("-" * 72)
    for task in OVR_TASKS:
        if task not in probs:
            continue
        thr = thresholds.get(task, 0.5)
        print(f"{task:<15} {probs[task]:8.4f} {thr:8.4f} {preds_bin[task]:8d}")

    print("\nGroup predictions:")
    gender_tasks = ["gender_female", "gender_male"]
    age_tasks = ["age_bin0", "age_bin1", "age_bin2"]
    race_tasks = ["race_white", "race_black", "race_asian", "race_indian", "race_others"]

    available_gender = [t for t in gender_tasks if t in probs]
    available_age = [t for t in age_tasks if t in probs]
    available_race = [t for t in race_tasks if t in probs]
    missing_race = [t for t in race_tasks if t not in probs]

    if len(available_gender) > 0:
        gender_idx = int(np.argmax([probs[t] for t in available_gender]))
        chosen = available_gender[gender_idx]
        gender_pred = "female" if chosen == "gender_female" else "male"
        suffix = ""
        if len(available_gender) < len(gender_tasks):
            suffix = f" (partial heads {len(available_gender)}/{len(gender_tasks)})"
        print(f"Gender     : {gender_pred}{suffix}")
    else:
        print("Gender     : N/A (không có head)")

    if len(available_age) > 0:
        age_idx = int(np.argmax([probs[t] for t in available_age]))
        chosen = available_age[age_idx]
        age_map = {
            "age_bin0": "0-18",
            "age_bin1": "19-60",
            "age_bin2": "61+",
        }
        age_pred = age_map[chosen]
        suffix = ""
        if len(available_age) < len(age_tasks):
            suffix = f" (partial heads {len(available_age)}/{len(age_tasks)})"
        print(f"Age bin    : {age_pred}{suffix}")
    else:
        print("Age bin    : N/A (không có head)")

    if len(available_race) > 0:
        race_idx = int(np.argmax([probs[t] for t in available_race]))
        chosen = available_race[race_idx]
        race_map = {
            "race_white": "white",
            "race_black": "black",
            "race_asian": "asian",
            "race_indian": "indian",
            "race_others": "others",
        }
        race_pred = race_map[chosen]
        suffix = ""
        if len(available_race) < len(race_tasks):
            suffix = f" (partial heads {len(available_race)}/{len(race_tasks)})"
        print(f"Race       : {race_pred}{suffix}")
        if missing_race:
            print(f"Race heads missing: {', '.join(missing_race)}")
    else:
        print("Race       : N/A (không có head)")
        print(f"Race heads missing: {', '.join(missing_race)}")

    gt = parse_utkface_filename(args.image)
    if gt is not None:
        age_gt, gender_gt, race_gt = gt
        print("\nGround truth from filename:")
        print(f"Age        : {age_gt} (bin {age_to_bin(age_gt)})")
        print(f"Gender code: {gender_gt}")
        print(f"Gender     : {gender_text_repo(gender_gt)} (0=male, 1=female)")
        print(f"Race       : {RACE_LABELS[race_gt]}")

    print("=" * 72)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Infer 1 ảnh UTKFace với các checkpoint OVR shard hiện có."
    )
    parser.add_argument("--image", required=True, help="Path tới ảnh UTKFace .jpg")
    parser.add_argument("--container", default="utkface_ovr", help="Tên container")
    parser.add_argument("--label", default="0", help="Nhãn checkpoint (shard-*:label.pt)")
    parser.add_argument(
        "--dataset",
        default="datasets/UTKFace/datasetfile_ovr",
        help="Path datasetfile_ovr để lấy input_shape",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--device", default="", help="cuda:0, cpu, ...")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    infer_single_image(args)
