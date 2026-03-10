import argparse
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

TASK_TO_SHARD = {name: i for i, name in enumerate(OVR_TASKS)}


def load_slices(container, task):
    path = f"containers/{container}/ovr_slices.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run utkface_ovr_partition.py first.")
    data = np.load(path, allow_pickle=True)
    if task not in data:
        raise KeyError(f"Task '{task}' not in {path}. keys={list(data.keys())}")
    return [np.asarray(x, dtype=np.int64) for x in data[task]]


def main():
    parser = argparse.ArgumentParser(
        description="Tạo requestfile cho UTKFace OVR để unlearn 1 slice (một phần dữ liệu của 1 head OVR)."
    )
    parser.add_argument("--container", default="utkface_ovr")
    parser.add_argument("--label", required=True, help="Tên request label (vd: forget-age-bin1-slice0)")
    parser.add_argument(
        "--task",
        choices=OVR_TASKS,
        required=True,
        help="Tên head OVR cần unlearn (vd: age_bin1, race_white, ...)",
    )
    parser.add_argument(
        "--slice",
        type=int,
        required=True,
        help="ID slice (0 .. slices_per_shard-1) theo ovr_slices.npz",
    )
    parser.add_argument(
        "--mode",
        default="overwrite",
        choices=["overwrite", "merge"],
        help="overwrite: tạo requestfile mới; merge: union với requestfile đang có nếu tồn tại.",
    )
    args = parser.parse_args()

    shard = TASK_TO_SHARD[args.task]
    slices = load_slices(args.container, args.task)
    if args.slice < 0 or args.slice >= len(slices):
        raise ValueError(f"slice phải trong [0..{len(slices)-1}]")

    forget_idx = np.asarray(slices[args.slice], dtype=np.int64)
    request_path = f"containers/{args.container}/requestfile:{args.label}.npy"

    if args.mode == "merge" and os.path.exists(request_path):
        loaded = np.load(request_path, allow_pickle=True)
        loaded = np.asarray(loaded, dtype=object).ravel()
        req = np.empty(len(OVR_TASKS), dtype=object)
        for i in range(len(OVR_TASKS)):
            if i < len(loaded) and loaded[i] is not None:
                req[i] = np.asarray(loaded[i], dtype=np.int64)
            else:
                req[i] = np.array([], dtype=np.int64)
        current = req[shard]
        req[shard] = np.union1d(current, forget_idx)
    else:
        req = np.empty(len(OVR_TASKS), dtype=object)
        for i in range(len(OVR_TASKS)):
            req[i] = np.array([], dtype=np.int64)
        req[shard] = forget_idx

    os.makedirs(f"containers/{args.container}", exist_ok=True)
    np.save(request_path, req)

    info = {
        "container": args.container,
        "label": args.label,
        "task": args.task,
        "shard": shard,
        "slice": int(args.slice),
        "forget_count": int(len(forget_idx)),
        "mode": args.mode,
    }
    with open(f"containers/{args.container}/requestfile:{args.label}.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Created OVR requestfile:")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

