import argparse
import json
import os

import numpy as np


TASK_BY_SHARD = {0: "gender", 1: "age", 2: "race"}
SHARD_BY_TASK = {v: k for k, v in TASK_BY_SHARD.items()}


def load_slices(container, task):
    path = f"containers/{container}/multitask_slices.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run init_multitask.sh first.")
    data = np.load(path, allow_pickle=True)
    if task not in data:
        raise KeyError(f"Task '{task}' not found in {path}. Keys={list(data.keys())}")
    return [np.asarray(x, dtype=np.int64) for x in data[task]]


def main():
    parser = argparse.ArgumentParser(
        description="Create UTKFace multitask requestfile to forget a whole slice (bin)."
    )
    parser.add_argument("--container", default="utkface")
    parser.add_argument(
        "--label",
        required=True,
        help="Request label name (e.g. forget-age-slice2). This becomes requestfile:<label>.npy",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=["gender", "age", "race"], help="Task to forget slice from")
    group.add_argument("--shard", type=int, choices=[0, 1, 2], help="Shard to forget slice from")
    parser.add_argument(
        "--slice",
        type=int,
        required=True,
        help="Slice id to forget. gender: 0=female,1=male. age: 0..4. race: 0..4",
    )
    parser.add_argument(
        "--mode",
        default="overwrite",
        choices=["overwrite", "merge"],
        help="overwrite: create fresh requestfile. merge: union with existing requestfile if present.",
    )
    args = parser.parse_args()

    shard = args.shard if args.shard is not None else SHARD_BY_TASK[args.task]
    task = TASK_BY_SHARD[shard]

    slices = load_slices(args.container, task)
    if args.slice < 0 or args.slice >= len(slices):
        raise ValueError(f"slice must be in [0..{len(slices)-1}] for task {task}")

    forget_indices = np.asarray(slices[args.slice], dtype=np.int64)
    request_path = f"containers/{args.container}/requestfile:{args.label}.npy"

    if args.mode == "merge" and os.path.exists(request_path):
        req = np.load(request_path, allow_pickle=True)
        req = np.asarray(req, dtype=object)
        while len(req) < 3:
            req = np.append(req, np.array([np.array([], dtype=np.int64)], dtype=object))
        current = np.asarray(req[shard], dtype=np.int64)
        req[shard] = np.union1d(current, forget_indices)
    else:
        req = np.array([np.array([], dtype=np.int64) for _ in range(3)], dtype=object)
        req[shard] = forget_indices

    os.makedirs(f"containers/{args.container}", exist_ok=True)
    np.save(request_path, req)

    info = {
        "container": args.container,
        "label": args.label,
        "task": task,
        "shard": shard,
        "slice": int(args.slice),
        "forget_count": int(len(forget_indices)),
        "mode": args.mode,
    }
    with open(f"containers/{args.container}/requestfile:{args.label}.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Created requestfile for multitask slice unlearning")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

