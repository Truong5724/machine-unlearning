import argparse
import importlib
import importlib.util
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
    """Load dataloader module from datasetfile metadata.

    Tries package import first (datasets.celebA.dataloader_ovr), then falls back
    to loading module by file path if package import is unavailable.
    """
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
        return importlib.import_module(module_name)
    except Exception:
        # Fallback: load directly from file path
        py_path = os.path.join(dataset_dir, f"{dataloader_name}.py")
        if not os.path.exists(py_path):
            raise
        spec = importlib.util.spec_from_file_location(dataloader_name, py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load dataloader module from {py_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def make_balanced_indices(y, all_indices, target_pos_samples, seed):
    """Build per-task indices with balanced positive/negative counts.

    Returns:
        selected_indices: ndarray
        stats: dict with detailed counts
    """
    y = np.asarray(y, dtype=np.int64)
    all_indices = np.asarray(all_indices, dtype=np.int64)

    pos_idx = all_indices[y == 1]
    neg_idx = all_indices[y == 0]

    pos_total = len(pos_idx)
    neg_total = len(neg_idx)
    total = pos_total + neg_total
    if total == 0:
        return np.array([], dtype=np.int64), {
            "pos_total": int(pos_total),
            "neg_total": int(neg_total),
            "target_pos": int(target_pos_samples),
            "selected_total": 0,
            "selected_pos": 0,
            "selected_neg": 0,
            "reduced": True,
        }

    selected_pos = min(int(target_pos_samples), pos_total)
    selected_neg = min(selected_pos, neg_total)
    selected_pos = min(selected_pos, selected_neg)
    selected_neg = selected_pos

    selected_total = int(selected_pos + selected_neg)

    rng = np.random.default_rng(seed)
    chosen_pos = (
        rng.choice(pos_idx, size=selected_pos, replace=False)
        if selected_pos > 0
        else np.array([], dtype=np.int64)
    )
    chosen_neg = (
        rng.choice(neg_idx, size=selected_neg, replace=False)
        if selected_neg > 0
        else np.array([], dtype=np.int64)
    )

    selected = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(selected)

    return selected.astype(np.int64), {
        "pos_total": int(pos_total),
        "neg_total": int(neg_total),
        "target_pos": int(target_pos_samples),
        "selected_total": int(selected_total),
        "selected_pos": int(selected_pos),
        "selected_neg": int(selected_neg),
        "reduced": bool(selected_pos < target_pos_samples),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Khởi tạo partition cho CelebA OVR (27 shards, mỗi shard 1 head OVR, giữ tỉ lệ gốc theo task)."
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
    parser.add_argument(
        "--per_task_samples",
        type=int,
        default=30000,
        help="Tổng số mẫu mục tiêu mỗi shard (mặc định 30000 = 15k pos + 15k neg)",
    )
    parser.add_argument(
        "--target_pos_samples",
        type=int,
        default=15000,
        help="Số mẫu dương mục tiêu mỗi shard (mặc định 15000)",
    )
    parser.add_argument(
        "--skip_rare",
        action="store_true",
        help="Bỏ qua task nếu không đủ mẫu dương để đạt target_pos_samples (không lặp mẫu).",
    )
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        ds = json.load(f)

    container_dir = f"containers/{args.container}"
    os.makedirs(f"{container_dir}/cache", exist_ok=True)
    os.makedirs(f"{container_dir}/times", exist_ok=True)
    os.makedirs(f"{container_dir}/outputs", exist_ok=True)

    dataloader = load_ovr_dataloader(args.dataset)
    nb_train_ds = int(ds.get("nb_train", 0))
    nb_train_real = int(getattr(dataloader, "train_size", nb_train_ds))

    if nb_train_real <= 0:
        raise RuntimeError(
            "Train size = 0 trong HDF5. Hay chay lai prepare_data_ovr.py va kiem tra input_dir/attr_file."
        )

    if nb_train_ds != nb_train_real:
        print(
            f"[WARN] nb_train(datasetfile)={nb_train_ds} != train_size(HDF5)={nb_train_real}. "
            "Su dung train_size(HDF5)."
        )

    all_indices = np.arange(nb_train_real, dtype=np.int64)
    labels_by_task = dataloader.load_ovr_labels(all_indices, category="train")

    rng = np.random.default_rng(args.seed)
    slices_dict = {}
    splitfile = np.empty(len(OVR_TASKS), dtype=object)

    print("Created CelebA OVR partitions (balanced pos/neg per shard):")
    print(
        f"  per_task_samples={args.per_task_samples}, target_pos_samples={args.target_pos_samples}, "
        f"slices_per_shard={args.slices_per_shard}"
    )

    for shard, name in enumerate(OVR_TASKS):
        y = np.asarray(labels_by_task[name], dtype=np.int64)

        selected_idx, st = make_balanced_indices(
            y,
            all_indices,
            target_pos_samples=args.target_pos_samples,
            seed=args.seed + shard,
        )

        if args.skip_rare and st["selected_pos"] < args.target_pos_samples:
            selected_idx = np.array([], dtype=np.int64)
            st["selected_total"] = 0
            st["selected_pos"] = 0
            st["selected_neg"] = 0

        if len(selected_idx) > 0:
            perm = rng.permutation(selected_idx)
            slices = np.array_split(perm, args.slices_per_shard)
        else:
            slices = [np.array([], dtype=np.int64) for _ in range(args.slices_per_shard)]

        slices_obj = np.array([np.asarray(s, dtype=np.int64) for s in slices], dtype=object)
        slices_dict[name] = slices_obj
        splitfile[shard] = selected_idx.astype(np.int64)

        pos_per_slice = []
        neg_per_slice = []
        for s in slices_obj:
            ys = y[np.asarray(s, dtype=np.int64)] if len(s) > 0 else np.array([], dtype=np.int64)
            p = int(ys.sum()) if len(ys) > 0 else 0
            n = int(len(ys) - p)
            pos_per_slice.append(p)
            neg_per_slice.append(n)

        status = "reduced" if st["reduced"] else "full"
        if st["selected_total"] == 0:
            status = "skipped"

        print(
            f"  {name}: status={status}, selected={st['selected_total']} "
            f"(pos={st['selected_pos']}, neg={st['selected_neg']}), "
            f"pool(pos={st['pos_total']}, neg={st['neg_total']}), "
            f"slice_sizes={[len(s) for s in slices_obj]}, "
            f"slice_pos={pos_per_slice}, slice_neg={neg_per_slice}"
        )

    np.save(f"{container_dir}/splitfile.npy", splitfile)
    np.savez(f"{container_dir}/ovr_slices.npz", **slices_dict)

    meta = {
        "tasks": OVR_TASKS,
        "task_by_shard": {str(i): name for i, name in enumerate(OVR_TASKS)},
        "slices_per_shard": args.slices_per_shard,
        "partition_mode": "balanced_pos_neg_cap",
        "per_task_samples": args.per_task_samples,
        "target_pos_samples": args.target_pos_samples,
        "skip_rare": bool(args.skip_rare),
    }
    with open(f"{container_dir}/ovr_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
