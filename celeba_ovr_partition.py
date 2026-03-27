import argparse
import importlib
import json
import os

import numpy as np

OVR_TASKS = [
    "male",
    "young",
    "smiling",
    "mouth_slightly_open",
    "big_lips",
    "big_nose",
    "pointy_nose",
    "high_cheekbones",
    "oval_face",
    "wavy_hair",
    "straight_hair",
    "bangs",
    "receding_hairline",
    "black_hair",
    "blond_hair",
    "brown_hair",
    "eyeglasses",
    "bushy_eyebrows",
    "arched_eyebrows",
    "bags_under_eyes",
    "chubby",
    "double_chin",
    "wearing_earrings",
    "wearing_necklace",
    "mustache",
    "goatee",
    "sideburns",
]


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
        description="Khởi tạo partition cho CelebA OVR (27 shards, mỗi shard 1 head OVR)."
    )
    parser.add_argument("--container", required=True, help="Tên container (vd: celeba_ovr)")
    parser.add_argument(
        "--dataset",
        default="datasets/celebA/datasetfile_ovr",
        help="Path tới datasetfile_ovr",
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

    container_dir = f"containers/{args.container}"
    os.makedirs(f"{container_dir}/cache", exist_ok=True)
    os.makedirs(f"{container_dir}/times", exist_ok=True)
    os.makedirs(f"{container_dir}/outputs", exist_ok=True)

    # splitfile: 27 shard, mỗi shard ban đầu nhìn thấy toàn bộ train indices.
    splitfile = np.array(
        [all_indices.copy() for _ in range(len(OVR_TASKS))], dtype=object
    )
    np.save(f"{container_dir}/splitfile.npy", splitfile)

    # SISA slices: chia random indices thành N slice (default 2) cho mỗi shard
    np.random.seed(args.seed)
    slices_dict = {}
    for shard, name in enumerate(OVR_TASKS):
        idx = np.random.permutation(all_indices)
        slices = np.array_split(idx, args.slices_per_shard)
        slices_dict[name] = np.array(slices, dtype=object)

    np.savez(f"{container_dir}/ovr_slices.npz", **slices_dict)

    meta = {
        "tasks": OVR_TASKS,
        "task_by_shard": {str(i): name for i, name in enumerate(OVR_TASKS)},
        "slices_per_shard": args.slices_per_shard,
    }
    with open(f"{container_dir}/ovr_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Created CelebA OVR partitions:")
    labels_by_task = None
    try:
        dataloader = load_ovr_dataloader(args.dataset)
        labels_by_task = dataloader.load_ovr_labels(all_indices, category="train")
    except Exception as e:
        print(f"  (Không load được nhãn OVR để thống kê class balance: {e})")

    for name in OVR_TASKS:
        lengths = [len(x) for x in slices_dict[name]]
        if labels_by_task is None:
            print(f"  {name}: sizes={lengths}")
            continue

        y = np.asarray(labels_by_task[name], dtype=np.int64)
        pos_total = int(y.sum())
        neg_total = int(len(y) - pos_total)
        pos_per_slice = [
            int(y[np.asarray(s, dtype=np.int64)].sum()) for s in slices_dict[name]
        ]
        print(
            f"  {name}: sizes={lengths}, pos_total={pos_total}, "
            f"neg_total={neg_total}, pos_per_slice={pos_per_slice}"
        )


if __name__ == "__main__":
    main()
