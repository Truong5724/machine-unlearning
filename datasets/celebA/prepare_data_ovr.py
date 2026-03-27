"""
Chuẩn bị CelebA dataset cho SISA OVR training.

- Chọn lọc 27 attributes từ list 40 attributes
- Stratified multi-label sampling cho train/val/test
- Lưu vào HDF5 format (tương tự UTKFace)

Yêu cầu: pip install h5py tqdm Pillow
"""

import os
import json
import argparse
import csv
import numpy as np
from PIL import Image
import h5py
from tqdm import tqdm


def _repo_root_from_this_file():
    # .../datasets/celebA/prepare_data_ovr.py -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_existing_path(path, expect="file"):
    """Resolve a path that must already exist, trying cwd then repo-root relative."""
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(path)
        candidates.append(os.path.join(_repo_root_from_this_file(), path))

    for p in candidates:
        if expect == "file" and os.path.isfile(p):
            return os.path.abspath(p)
        if expect == "dir" and os.path.isdir(p):
            return os.path.abspath(p)

    raise FileNotFoundError(f"Không tìm thấy {expect}: {path}")


def _resolve_output_dir(path):
    """Resolve output dir without creating it; supports cwd/repo-root relative paths."""
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(path)
        candidates.append(os.path.join(_repo_root_from_this_file(), path))

    for p in candidates:
        if os.path.isdir(p):
            return os.path.abspath(p)

    raise FileNotFoundError(
        f"Output dir không tồn tại: {path}. "
        "Hãy truyền --output_dir là thư mục đã có sẵn."
    )

# Danh sách 27 attributes được chọn cho OVR
OVR_ATTRIBUTES = [
    "Male",
    "Young",
    "Smiling",
    "Mouth_Slightly_Open",
    "Big_Lips",
    "Big_Nose",
    "Pointy_Nose",
    "High_Cheekbones",
    "Oval_Face",
    "Wavy_Hair",
    "Straight_Hair",
    "Bangs",
    "Receding_Hairline",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Eyeglasses",
    "Bushy_Eyebrows",
    "Arched_Eyebrows",
    "Bags_Under_Eyes",
    "Chubby",
    "Double_Chin",
    "Wearing_Earrings",
    "Wearing_Necklace",
    "Mustache",
    "Goatee",
    "Sideburns",
]

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


def _to_binary_01(values):
    """Normalize attribute values to 0/1 from {-1,1} or {0,1}."""
    arr = np.asarray(values, dtype=np.int32)
    uniq = set(np.unique(arr).tolist())
    if uniq.issubset({-1, 1}):
        return ((arr + 1) // 2).astype(np.int64)
    if uniq.issubset({0, 1}):
        return arr.astype(np.int64)
    raise ValueError(
        f"Unsupported attribute encoding: {sorted(uniq)}. Expected -1/1 or 0/1."
    )


def load_attributes(attr_file):
    """Load attributes from CelebA TXT format or CSV format (image_id,...)."""
    with open(attr_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    # CSV format, e.g. header: image_id,5_o_Clock_Shadow,...
    if "," in first_line and first_line.lower().startswith("image_id"):
        with open(attr_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV attribute file is missing header")

            all_attr_names = [x for x in reader.fieldnames if x != "image_id"]
            select_names = []
            for attr in OVR_ATTRIBUTES:
                if attr in all_attr_names:
                    select_names.append(attr)
                else:
                    raise ValueError(f"Attribute {attr} không tìm thấy trong CelebA attributes!")

            filenames = []
            attributes = []
            for row in reader:
                filenames.append(row["image_id"])
                raw_vals = [int(row[name]) for name in select_names]
                attributes.append(_to_binary_01(raw_vals))

        return np.array(filenames), np.array(attributes, dtype=np.int64), OVR_ATTRIBUTES

    # Original CelebA TXT format
    with open(attr_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError("Invalid CelebA TXT attribute file format")

    _ = int(lines[0].strip())
    all_attr_names = lines[1].strip().split()

    select_indices = []
    for attr in OVR_ATTRIBUTES:
        if attr in all_attr_names:
            select_indices.append(all_attr_names.index(attr))
        else:
            raise ValueError(f"Attribute {attr} không tìm thấy trong CelebA attributes!")

    filenames = []
    attributes = []
    for line in lines[2:]:
        parts = line.strip().split()
        if not parts:
            continue
        filenames.append(parts[0])
        all_attrs = np.array([int(x) for x in parts[1:]], dtype=np.int32)
        selected_attrs = all_attrs[select_indices]
        attributes.append(_to_binary_01(selected_attrs))

    return np.array(filenames), np.array(attributes, dtype=np.int64), OVR_ATTRIBUTES


def stratified_multilabel_sampling(attributes, pool_indices, target_samples, seed=42):
    """
    Stratified multi-label sampling với số lượng mẫu chính xác (nếu có thể).

    Args:
        attributes: ndarray (N, num_attrs), giá trị 0/1
        pool_indices: indices ứng viên để sample từ đó
        target_samples: số lượng mẫu cần lấy
        seed: random seed

    Returns:
        ndarray indices đã chọn (global indices theo dataset gốc)
    """
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    n_pool = len(pool_indices)
    if target_samples <= 0 or n_pool == 0:
        return np.array([], dtype=np.int64)
    if target_samples >= n_pool:
        return pool_indices.copy()

    rng = np.random.default_rng(seed)

    label_strings = np.array([
        ''.join(map(str, row)) for row in attributes[pool_indices]
    ])
    unique_labels, inverse_indices = np.unique(label_strings, return_inverse=True)
    stratum_counts = np.bincount(inverse_indices)

    # Quota theo tỷ lệ + phân bổ phần dư để đạt đúng target_samples
    raw_quota = stratum_counts.astype(np.float64) * float(target_samples) / float(n_pool)
    base_quota = np.floor(raw_quota).astype(np.int64)
    base_quota = np.minimum(base_quota, stratum_counts)
    remaining = int(target_samples - int(base_quota.sum()))

    if remaining > 0:
        frac = raw_quota - base_quota
        candidates = np.where(stratum_counts > base_quota)[0]
        if len(candidates) > 0:
            order = candidates[np.argsort(-frac[candidates])]
            # Break tie ngẫu nhiên nhẹ để tránh bias cố định
            if len(order) > 1:
                jitter = rng.random(len(order)) * 1e-9
                order = order[np.argsort(-(frac[order] + jitter))]
            for sid in order:
                if remaining <= 0:
                    break
                can_add = int(stratum_counts[sid] - base_quota[sid])
                add = min(1, can_add)
                if add > 0:
                    base_quota[sid] += add
                    remaining -= add

    selected = []
    for sid in range(len(unique_labels)):
        k = int(base_quota[sid])
        if k <= 0:
            continue
        local = np.where(inverse_indices == sid)[0]
        chosen_local = rng.choice(local, size=k, replace=False)
        selected.append(pool_indices[chosen_local])

    if not selected:
        return np.array([], dtype=np.int64)
    return np.concatenate(selected).astype(np.int64)


def load_images_batch(img_dir, filenames, target_size=(64, 64), batch_size=1000):
    """Generator: Tải ảnh theo batch"""
    batch_images = []
    batch_indices = []
    failed = []

    for idx, fname in enumerate(filenames):
        img_path = os.path.join(img_dir, fname)
        try:
            if not os.path.exists(img_path):
                failed.append(fname)
                continue

            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).transpose(2, 0, 1)

            batch_images.append(img_array)
            batch_indices.append(idx)

            if len(batch_images) == batch_size or idx == len(filenames) - 1:
                yield np.array(batch_images, dtype=np.uint8), batch_indices, failed
                batch_images = []
                batch_indices = []
                failed = []

        except Exception as e:
            print(f"\n⚠️  Lỗi khi tải {fname}: {e}")
            failed.append(fname)
            continue


def save_to_hdf5_ovr(h5_file, filenames, labels, img_dir, selected_indices, 
                      target_size=(64, 64), batch_size=1000):
    """Lưu ảnh và labels OVR vào HDF5"""
    
    selected_filenames = filenames[selected_indices]
    selected_labels = labels[selected_indices]
    
    n_samples = len(selected_indices)
    
    # Tạo datasets
    images_ds = h5_file.create_dataset(
        'images',
        shape=(n_samples, 3, target_size[0], target_size[1]),
        maxshape=(n_samples, 3, target_size[0], target_size[1]),
        dtype='uint8',
        chunks=(1, 3, target_size[0], target_size[1]),
    )
    
    labels_ds = h5_file.create_dataset(
        'labels',
        shape=(n_samples, len(OVR_ATTRIBUTES)),
        maxshape=(n_samples, len(OVR_ATTRIBUTES)),
        dtype='int64',
        chunks=(1, len(OVR_ATTRIBUTES)),
    )
    
    # Load ảnh theo batch và lưu vào HDF5
    global_idx = 0
    for batch_images, batch_indices, failed in load_images_batch(
        img_dir, selected_filenames, target_size, batch_size
    ):
        # batch_indices là indices trong selected_filenames
        # Cần convert sang global indices trong h5 file
        batch_size_actual = len(batch_indices)
        
        images_ds[global_idx:global_idx + batch_size_actual] = batch_images
        labels_ds[global_idx:global_idx + batch_size_actual] = selected_labels[batch_indices]
        
        global_idx += batch_size_actual
        
    if global_idx < n_samples:
        print(f"⚠️  Chỉ tải được {global_idx}/{n_samples} ảnh")
        images_ds.resize((global_idx, 3, target_size[0], target_size[1]))
        labels_ds.resize((global_idx, len(OVR_ATTRIBUTES)))
    
    # Lưu metadata
    h5_file.attrs['num_samples'] = global_idx
    h5_file.attrs['num_attributes'] = len(OVR_ATTRIBUTES)
    h5_file.attrs['input_shape'] = [3, target_size[0], target_size[1]]
    h5_file.attrs['attributes'] = OVR_ATTRIBUTES
    h5_file.attrs['tasks'] = OVR_TASKS
    return int(global_idx)


def create_datasetfile(output_dir, train_samples, val_samples, test_samples):
    """Tạo file metadata datasetfile_ovr"""
    metadata = {
        "type": "celeba_ovr",
        "dataloader": "dataloader_ovr",
        "input_shape": [3, 64, 64],
        "nb_train": train_samples,
        "nb_val": val_samples,
        "nb_test": test_samples,
        "attributes": OVR_ATTRIBUTES,
        "tasks": OVR_TASKS,
    }
    
    path = os.path.join(output_dir, "datasetfile_ovr")
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Tạo datasetfile: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị CelebA OVR dataset với stratified sampling"
    )
    parser.add_argument(
        "--input_dir",
        default="datasets/celebA/img_align_celeba",
        help="Đường dẫn tới thư mục ảnh CelebA"
    )
    parser.add_argument(
        "--attr_file",
        default="datasets/celebA/list_attr_celeba.txt",
        help="Đường dẫn tới file attribute"
    )
    parser.add_argument(
        "--output_dir",
        default="datasets/celebA",
        help="Thư mục output"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=50000,
        help="Số samples cho training set"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=10000,
        help="Số samples cho test set"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=10000,
        help="Số samples cho validation set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=64,
        help="Kích thước ảnh (64x64)"
    )
    args = parser.parse_args()

    # Resolve paths robustly (không tự tạo output_dir).
    args.input_dir = _resolve_existing_path(args.input_dir, expect="dir")
    args.attr_file = _resolve_existing_path(args.attr_file, expect="file")
    args.output_dir = _resolve_output_dir(args.output_dir)

    print("=" * 80)
    print("CHUẨN BỊ CELEBA OVR DATASET")
    print("=" * 80)
    print(f"Input dir : {os.path.abspath(args.input_dir)}")
    print(f"Attr file : {os.path.abspath(args.attr_file)}")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")

    # Load attributes
    print(f"\n📂 Đang load attributes từ {args.attr_file}...")
    filenames, attributes, attr_names = load_attributes(args.attr_file)
    print(f"✅ Đã load {len(filenames)} ảnh, {len(attr_names)} attributes")
    print(f"   OVR tasks: {', '.join(OVR_TASKS)}")

    print(f"\n📊 Đang stratified sampling train/val/test...")
    all_indices = np.arange(len(filenames), dtype=np.int64)

    train_indices = stratified_multilabel_sampling(
        attributes, all_indices, args.train_samples, seed=args.seed
    )

    remain_after_train = np.setdiff1d(all_indices, train_indices)
    val_indices = stratified_multilabel_sampling(
        attributes,
        remain_after_train,
        min(args.val_samples, len(remain_after_train)),
        seed=args.seed + 1,
    )

    remain_after_val = np.setdiff1d(remain_after_train, val_indices)
    test_indices = stratified_multilabel_sampling(
        attributes,
        remain_after_val,
        min(args.test_samples, len(remain_after_val)),
        seed=args.seed + 2,
    )

    print(f"✅ Train: {len(train_indices)} samples")
    print(f"✅ Val  : {len(val_indices)} samples")
    print(f"✅ Test : {len(test_indices)} samples")

    # Tạo HDF5 files
    print(f"\n💾 Đang lưu training set...")
    train_path = os.path.join(args.output_dir, "celeba_ovr_train.h5")
    with h5py.File(train_path, 'w') as f:
        train_saved = save_to_hdf5_ovr(
            f, filenames, attributes, args.input_dir, 
            train_indices, (args.target_size, args.target_size)
        )
    print(f"✅ Lưu xong: {train_path}")
    print(f"   Saved train samples: {train_saved}")

    print(f"\n💾 Đang lưu val set...")
    val_path = os.path.join(args.output_dir, "celeba_ovr_val.h5")
    with h5py.File(val_path, 'w') as f:
        val_saved = save_to_hdf5_ovr(
            f, filenames, attributes, args.input_dir,
            val_indices, (args.target_size, args.target_size)
        )
    print(f"✅ Lưu xong: {val_path}")
    print(f"   Saved val samples: {val_saved}")

    print(f"\n💾 Đang lưu test set...")
    test_path = os.path.join(args.output_dir, "celeba_ovr_test.h5")
    with h5py.File(test_path, 'w') as f:
        test_saved = save_to_hdf5_ovr(
            f, filenames, attributes, args.input_dir,
            test_indices, (args.target_size, args.target_size)
        )
    print(f"✅ Lưu xong: {test_path}")
    print(f"   Saved test samples: {test_saved}")

    # Tạo datasetfile
    print(f"\n📄 Đang tạo datasetfile...")
    create_datasetfile(
        args.output_dir, 
        train_saved,
        val_saved,
        test_saved,
    )

    if train_saved == 0:
        raise RuntimeError(
            "Train set sau khi save = 0. Kiem tra lai --input_dir co dung va co anh hop le."
        )

    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH!")
    print("=" * 80)


if __name__ == "__main__":
    main()
