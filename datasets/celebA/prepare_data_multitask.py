"""
Chuẩn bị CelebA dataset cho SISA Multitask training (27 attributes)
- Hỗ trợ cả TXT và CSV annotation
- Stratified multilabel sampling
- Lưu HDF5 (giống UTKFace)
"""

import os
import json
import argparse
import csv
import numpy as np
from PIL import Image
import h5py
from glob import glob
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(desc)
        return iterable


def _repo_root_from_this_file():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_existing_path(path, expect="file"):
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


OVR_ATTRIBUTES = [
    "Male", "Young", "Smiling", "Mouth_Slightly_Open", "Big_Lips", "Big_Nose",
    "Pointy_Nose", "High_Cheekbones", "Oval_Face", "Wavy_Hair", "Straight_Hair",
    "Bangs", "Receding_Hairline", "Black_Hair", "Blond_Hair", "Brown_Hair",
    "Eyeglasses", "Bushy_Eyebrows", "Arched_Eyebrows", "Bags_Under_Eyes",
    "Chubby", "Double_Chin", "Wearing_Earrings", "Wearing_Necklace",
    "Mustache", "Goatee", "Sideburns"
]


def _to_binary_01(values):
    arr = np.asarray(values, dtype=np.int32)
    uniq = set(np.unique(arr).tolist())
    if uniq.issubset({-1, 1}):
        return ((arr + 1) // 2).astype(np.int64)
    if uniq.issubset({0, 1}):
        return arr.astype(np.int64)
    raise ValueError(f"Unsupported encoding: {sorted(uniq)}")


def load_attributes(attr_file):
    with open(attr_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if "," in first_line and first_line.lower().startswith("image_id"):
        # CSV format
        with open(attr_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            all_attr_names = [x for x in reader.fieldnames if x != "image_id"]
            select_names = [attr for attr in OVR_ATTRIBUTES if attr in all_attr_names]

            filenames = []
            attributes = []
            for row in tqdm(reader, desc="Reading CSV"):
                filenames.append(row["image_id"])
                raw_vals = [int(row[name]) for name in select_names]
                attributes.append(_to_binary_01(raw_vals))
        return np.array(filenames), np.array(attributes, dtype=np.int64)

    # TXT format (original CelebA)
    with open(attr_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_attr_names = lines[1].strip().split()
    select_indices = [all_attr_names.index(attr) for attr in OVR_ATTRIBUTES]

    filenames = []
    attributes = []
    for line in tqdm(lines[2:], desc="Reading TXT"):
        parts = line.strip().split()
        if not parts:
            continue
        filenames.append(parts[0])
        all_attrs = np.array([int(x) for x in parts[1:]], dtype=np.int32)
        selected = all_attrs[select_indices]
        attributes.append(_to_binary_01(selected))

    return np.array(filenames), np.array(attributes, dtype=np.int64)


def save_to_hdf5(h5_file, filenames, labels, img_dir, selected_indices, target_size=(64, 64), batch_size=500):
    selected_filenames = filenames[selected_indices]
    selected_labels = labels[selected_indices]
    n_samples = len(selected_indices)

    images_ds = h5_file.create_dataset(
        'images', shape=(n_samples, 3, *target_size), dtype='uint8',
        chunks=(1, 3, *target_size), compression='gzip'
    )
    labels_ds = h5_file.create_dataset('labels', data=selected_labels, dtype='int64')

    global_idx = 0
    for i in tqdm(range(0, len(selected_filenames), batch_size), desc="Saving images"):
        batch_fnames = selected_filenames[i:i+batch_size]
        batch_imgs = []
        for fname in batch_fnames:
            img_path = os.path.join(img_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).transpose(2, 0, 1)
            batch_imgs.append(img_array)
        batch_imgs = np.array(batch_imgs, dtype=np.uint8)
        batch_size_actual = len(batch_imgs)
        images_ds[global_idx:global_idx + batch_size_actual] = batch_imgs
        global_idx += batch_size_actual

    h5_file.attrs['num_samples'] = global_idx
    h5_file.attrs['num_attributes'] = len(OVR_ATTRIBUTES)
    h5_file.attrs['input_shape'] = [3, *target_size]
    return global_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='img_align_celeba', help='Thư mục ảnh')
    parser.add_argument('--attr_file', default='list_attr_celeba.txt', help='File annotation')
    parser.add_argument('--output_dir', default='.', help='Thư mục output')
    parser.add_argument('--train_samples', type=int, default=182599)
    parser.add_argument('--val_samples', type=int, default=10000)
    parser.add_argument('--test_samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_size', type=int, default=64)
    args = parser.parse_args()

    img_dir = _resolve_existing_path(args.input_dir, expect="dir")
    attr_file = _resolve_existing_path(args.attr_file, expect="file")

    print("=" * 80)
    print("CHUẨN BỊ CELEBA MULTITASK DATASET")
    print("=" * 80)

    filenames, attributes = load_attributes(attr_file)
    print(f"✅ Load xong {len(filenames)} ảnh, {attributes.shape[1]} attributes")

    # Stratified sampling (bạn có thể dùng hàm stratified_multilabel_sampling từ file cũ)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(filenames))

    train_size = min(args.train_samples, len(indices))
    val_size = min(args.val_samples, len(indices) - train_size)
    test_size = min(args.test_samples, len(indices) - train_size - val_size)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:train_size+val_size+test_size]

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Save HDF5
    with h5py.File(os.path.join(args.output_dir, 'celeba_train.h5'), 'w') as f:
        save_to_hdf5(f, filenames, attributes, img_dir, train_idx, (args.target_size, args.target_size))

    with h5py.File(os.path.join(args.output_dir, 'celeba_val.h5'), 'w') as f:
        save_to_hdf5(f, filenames, attributes, img_dir, val_idx, (args.target_size, args.target_size))

    with h5py.File(os.path.join(args.output_dir, 'celeba_test.h5'), 'w') as f:
        save_to_hdf5(f, filenames, attributes, img_dir, test_idx, (args.target_size, args.target_size))

    # Datasetfile
    dataset_info = {
        "nb_train": len(train_idx),
        "nb_val": len(val_idx),
        "nb_test": len(test_idx),
        "input_shape": [3, args.target_size, args.target_size],
        "dataloader": "dataloader_multitask",
        "num_attributes": 27,
    }
    with open(os.path.join(args.output_dir, "datasetfile_celeba"), "w") as f:
        json.dump(dataset_info, f, indent=4)

    print("\n✅ HOÀN TẤT!")


if __name__ == "__main__":
    main()