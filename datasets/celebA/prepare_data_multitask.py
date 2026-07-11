"""
Chuẩn bị CelebA dataset cho SISA training (Multitask - 27 attributes)
- Parse annotation file
- Tạo train/test split
- Lưu dưới dạng HDF5
"""

import os
import json
import argparse
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


def load_celeba_annotations(txt_path):
    """Đọc file annotation của CelebA"""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Bỏ 2 dòng header
    lines = lines[2:]
    data = {}
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_name = parts[0]
        labels = [int(x) for x in parts[1:]]
        # Convert -1 → 0 (CelebA convention)
        labels = [1 if x == 1 else 0 for x in labels]
        data[img_name] = np.array(labels, dtype=np.int64)
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Chuẩn bị CelebA dataset')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--img_dir', default=os.path.join(script_dir, 'img_align_celeba'), help='Thư mục chứa ảnh CelebA')
    parser.add_argument('--anno_file', default=os.path.join(script_dir, 'list_attr_celeba.txt'), help='File annotation')
    parser.add_argument('--target_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()

    target_size = (args.target_size, args.target_size)

    print("=" * 70)
    print("CHUẨN BỊ CELEBA MULTITASK DATASET (27 attributes)")
    print("=" * 70)

    # Load annotations
    annotations = load_celeba_annotations(args.anno_file)
    print(f"✅ Đọc được {len(annotations)} annotations")

    # Tìm ảnh
    img_files = []
    labels_list = []
    for img_name, label in annotations.items():
        img_path = os.path.join(args.img_dir, img_name)
        if os.path.exists(img_path):
            img_files.append(img_path)
            labels_list.append(label)

    print(f"✅ Tìm thấy {len(img_files)} ảnh hợp lệ")

    # Shuffle
    indices = np.random.RandomState(42).permutation(len(img_files))
    img_files = [img_files[i] for i in indices]
    labels_list = [labels_list[i] for i in indices]

    # Split
    train_size = int(len(img_files) * args.train_ratio)
    train_files = img_files[:train_size]
    train_labels = np.array(labels_list[:train_size])
    test_files = img_files[train_size:]
    test_labels = np.array(labels_list[train_size:])

    print(f"Train: {len(train_files)} | Test: {len(test_files)}")

    # Save HDF5
    def save_hdf5(files, labels, filename):
        with h5py.File(filename, 'w') as h5f:
            h5f.attrs['num_samples'] = len(files)
            h5f.attrs['num_attributes'] = 27
            h5f.attrs['has_multitask_labels'] = True

            # Images dataset
            images_ds = h5f.create_dataset('images', shape=(len(files), 3, *target_size), 
                                         dtype='uint8', chunks=(1, 3, *target_size), compression='gzip')
            labels_ds = h5f.create_dataset('labels', data=labels, dtype='int64')

            for i, img_path in enumerate(tqdm(files, desc=f"Saving {filename}")):
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size, Image.LANCZOS)
                img_array = np.array(img).transpose(2, 0, 1)
                images_ds[i] = img_array

    save_hdf5(train_files, train_labels, os.path.join(script_dir, 'celeba_train.h5'))
    save_hdf5(test_files, test_labels, os.path.join(script_dir, 'celeba_test.h5'))

    # Datasetfile
    dataset_info = {
        "nb_train": len(train_files),
        "nb_test": len(test_files),
        "input_shape": [3, args.target_size, args.target_size],
        "dataloader": "dataloader_multitask",
        "num_attributes": 27,
    }
    with open(os.path.join(script_dir, 'datasetfile_multitask'), "w") as f:
        json.dump(dataset_info, f, indent=4)

    print("\n✅ HOÀN TẤT PREPARE CELEBA!")


if __name__ == "__main__":
    main()