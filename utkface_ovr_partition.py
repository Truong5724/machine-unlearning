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
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        ds = json.load(f)

    nb_train = ds["nb_train"]
    all_indices = np.arange(nb_train, dtype=np.int64)

    container_dir = f"containers/{args.container}"
    os.makedirs(f"{container_dir}/cache", exist_ok=True)
    os.makedirs(f"{container_dir}/times", exist_ok=True)
    os.makedirs(f"{container_dir}/outputs", exist_ok=True)

    # splitfile: 10 shard, mỗi shard ban đầu nhìn thấy toàn bộ train indices.
    # Ta chỉ cần biết số phần tử cho mỗi shard; bản thân SISA OVR sẽ dùng ovr_slices.npz.
    splitfile = np.array(
        [all_indices.copy() for _ in range(len(OVR_TASKS))], dtype=object
    )
    np.save(f"{container_dir}/splitfile.npy", splitfile)

    requests = np.array(
        [np.array([], dtype=np.int64) for _ in range(len(OVR_TASKS))], dtype=object
    )
    np.save(f"{container_dir}/requestfile:{args.label}.npy", requests)

    # SISA slices: chia random indices thành N slice (default 2) cho mỗi shard
    np.random.seed(42)
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

    print("Created UTKFace OVR partitions:")
    for name in OVR_TASKS:
        lengths = [len(x) for x in slices_dict[name]]
        print(f"  {name}: {lengths}")


if __name__ == "__main__":
    main()

