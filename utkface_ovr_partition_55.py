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


def build_balanced_ovr_indices(y, all_indices, seed):
    """Select all positive samples and match them with negatives.

    If negatives are fewer than positives, sample negatives with replacement so
    the shard still reaches the same number of negatives as positives.
    """
    y = np.asarray(y, dtype=np.int64)
    all_indices = np.asarray(all_indices, dtype=np.int64)

    pos_idx = all_indices[y == 1]
    neg_idx = all_indices[y == 0]

    pos_total = int(len(pos_idx))
    neg_total = int(len(neg_idx))

    if pos_total == 0:
        return np.array([], dtype=np.int64), {
            "pos_total": 0,
            "neg_total": neg_total,
            "selected_pos": 0,
            "selected_neg": 0,
            "augmented_neg": 0,
            "reduced": False,
        }

    rng = np.random.default_rng(seed)
    selected_pos = pos_idx.copy()

    if neg_total >= pos_total:
        selected_neg = rng.choice(neg_idx, size=pos_total, replace=False)
        augmented_neg = 0
    else:
        selected_neg = rng.choice(neg_idx, size=pos_total, replace=True)
        augmented_neg = pos_total - neg_total

    selected = np.concatenate([selected_pos, selected_neg])
    rng.shuffle(selected)

    return selected.astype(np.int64), {
        "pos_total": pos_total,
        "neg_total": neg_total,
        "selected_pos": pos_total,
        "selected_neg": int(len(selected_neg)),
        "augmented_neg": int(augmented_neg),
        "reduced": bool(neg_total < pos_total),
    }


def load_ovr_dataloader(datasetfile_path):
    """Load dataloader module from datasetfile metadata if available."""
    with open(datasetfile_path, "r") as f:
        datasetfile = json.load(f)

    module_name = ".".join(
        datasetfile_path.replace("\\", "/").split("/")[:-1]
        + [datasetfile["dataloader"]]
    )
    module = importlib.import_module(module_name)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Khởi tạo partition cho UTKFace OVR (10 shards, mỗi shard 1 head OVR)."
    )
    parser.add_argument("--container", required=True, help="Tên container (vd: utkface_ovr)")
    parser.add_argument(
        "--dataset",
        default="datasets/UTKFace/datasetfile_ovr",
        help="Path tới datasetfile_ovr",
    )
    parser.add_argument(
        "--label",
        default="0",
        help="Nhãn request (requestfile:<label>.npy), mặc định 0 cho baseline",
    )
    parser.add_argument(
        "--slices_per_shard",
        type=int,
        default=2,
        help="Số slice SISA trong mỗi shard (mặc định 2, random indices)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed để tạo slices (cố định để reproducible)",
    )
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        ds = json.load(f)

    nb_train = ds["nb_train"]
    all_indices = np.arange(nb_train, dtype=np.int64)

    dataloader = load_ovr_dataloader(args.dataset)
    labels_by_task = dataloader.load_ovr_labels(all_indices, category="train")

    container_dir = f"containers/{args.container}"
    os.makedirs(f"{container_dir}/cache", exist_ok=True)
    os.makedirs(f"{container_dir}/times", exist_ok=True)
    os.makedirs(f"{container_dir}/outputs", exist_ok=True)

    print("Created UTKFace OVR partitions:")
    print("  rule: keep all positives, sample negatives to match positives")

    requests = np.array(
        [np.array([], dtype=np.int64) for _ in range(len(OVR_TASKS))], dtype=object
    )
    np.save(f"{container_dir}/requestfile:{args.label}.npy", requests)

    slices_dict = {}
    splitfile = np.empty(len(OVR_TASKS), dtype=object)

    for shard, name in enumerate(OVR_TASKS):
        selected_idx, stats = build_balanced_ovr_indices(
            labels_by_task[name], all_indices, seed=args.seed + shard
        )

        if len(selected_idx) > 0:
            rng = np.random.default_rng(args.seed + shard)
            perm = rng.permutation(selected_idx)
            slices = np.array_split(perm, args.slices_per_shard)
        else:
            slices = [np.array([], dtype=np.int64) for _ in range(args.slices_per_shard)]

        slices_dict[name] = np.array([np.asarray(s, dtype=np.int64) for s in slices], dtype=object)
        splitfile[shard] = selected_idx.astype(np.int64)

        if stats["selected_pos"] == 0:
            status = "empty"
        elif stats["reduced"]:
            status = "augmented_neg"
        else:
            status = "balanced"

        print(
            f"  {name}: status={status}, sizes={[len(x) for x in slices_dict[name]]}, "
            f"pos_total={stats['pos_total']}, neg_total={stats['neg_total']}, "
            f"selected_pos={stats['selected_pos']}, selected_neg={stats['selected_neg']}, "
            f"augmented_neg={stats['augmented_neg']}"
        )

    np.savez(f"{container_dir}/ovr_slices.npz", **slices_dict)
    np.save(f"{container_dir}/splitfile.npy", splitfile)

    meta = {
        "tasks": OVR_TASKS,
        "task_by_shard": {str(i): name for i, name in enumerate(OVR_TASKS)},
        "slices_per_shard": args.slices_per_shard,
        "partition_mode": "all_pos_and_matched_neg",
    }
    with open(f"{container_dir}/ovr_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

