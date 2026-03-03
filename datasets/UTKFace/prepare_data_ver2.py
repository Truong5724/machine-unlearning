"""
Chuẩn bị UTKFace dataset cho SISA training
- Parse filename để lấy age, gender, race
- Tạo train/test split (80/20)
- Lưu dưới dạng HDF5 (memory-efficient)

Task mặc định: Binary Gender Classification (0=Female, 1=Male)
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


def parse_filename(filename):
    """
    Parse UTKFace filename: [age]_[gender]_[race]_[date&time].jpg
    
    Returns:
        age (int): 0-116
        gender (int): 0=Female, 1=Male
        race (int): 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
        None if parsing fails
    """
    try:
        basename = os.path.basename(filename)
        parts = basename.split('_')
        
        if len(parts) < 3:
            return None
        
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
        
        # Validate
        if not (0 <= age <= 116 and gender in [0, 1] and 0 <= race <= 4):
            return None
        
        return age, gender, race
    except (ValueError, IndexError):
        return None


def load_images_batch(img_files, labels, target_size=(64, 64), batch_size=1000):
    """
    Generator: Load ảnh theo batch
    
    Yields:
        batch_images: numpy array (batch_size, 3, H, W)
        batch_labels: numpy array (batch_size,)
        batch_indices: list các index thành công
    """
    batch_images = []
    batch_labels = []
    batch_indices = []
    failed = []
    
    for idx, (img_file, label) in enumerate(zip(img_files, labels)):
        try:
            img = Image.open(img_file).convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
            
            batch_images.append(img_array)
            batch_labels.append(label)
            batch_indices.append(idx)
            
            if len(batch_images) == batch_size or idx == len(img_files) - 1:
                yield (np.array(batch_images, dtype=np.uint8),
                       np.array(batch_labels, dtype=np.int64),
                       batch_indices)
                batch_images = []
                batch_labels = []
                batch_indices = []
        except Exception as e:
            failed.append(img_file)
            continue
    
    if failed:
        print(f"\n⚠️  Không tải được {len(failed)} ảnh")


def save_to_hdf5(h5_file, img_files, labels, ages, genders, races, target_size=(64, 64), batch_size=1000):
    """Lưu ảnh vào HDF5"""
    total_samples = len(img_files)
    
    images_dataset = h5_file.create_dataset(
        'images',
        shape=(total_samples, 3, target_size[0], target_size[1]),
        dtype='uint8',
        chunks=(1, 3, target_size[0], target_size[1]),
        compression='gzip',
        compression_opts=4
    )
    
    labels_dataset = h5_file.create_dataset(
        'labels',
        shape=(total_samples,),
        dtype='int64'
    )

    age_dataset = h5_file.create_dataset(
        'age',
        shape=(total_samples,),
        dtype='int64'
    )
    gender_dataset = h5_file.create_dataset(
        'gender',
        shape=(total_samples,),
        dtype='int64'
    )
    race_dataset = h5_file.create_dataset(
        'race',
        shape=(total_samples,),
        dtype='int64'
    )
    
    print(f"Đang lưu {total_samples} ảnh vào HDF5...")
    total_saved = 0
    
    for batch_images, batch_labels, batch_indices in tqdm(
        load_images_batch(img_files, labels, target_size, batch_size),
        total=total_samples // batch_size + 1,
        desc="Processing"
    ):
        for i, idx in enumerate(batch_indices):
            images_dataset[idx] = batch_images[i]
            labels_dataset[idx] = batch_labels[i]
            age_dataset[idx] = ages[idx]
            gender_dataset[idx] = genders[idx]
            race_dataset[idx] = races[idx]
        total_saved += len(batch_indices)
    
    print(f"✅ Đã lưu {total_saved}/{total_samples} ảnh")
    return total_saved


def analyze_distribution(labels, label_names):
    """Phân tích phân bố labels"""
    print("\n" + "=" * 70)
    print("PHÂN BỐ LABELS")
    print("=" * 70)
    
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    for label, count in zip(unique, counts):
        ratio = count / total * 100
        label_name = label_names.get(label, f"Class {label}")
        print(f"  {label_name:<15} : {count:>6} samples ({ratio:>5.1f}%)")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Chuẩn bị UTKFace dataset')
    parser.add_argument('--img_dir', default='UTKFace', help='Thư mục chứa ảnh UTKFace')
    parser.add_argument('--task', default='gender', choices=['gender', 'age_group', 'race'],
                       help='Task: gender (binary), age_group, race')
    parser.add_argument('--target_size', type=int, default=64, help='Resize ảnh (default: 64x64)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size khi prepare')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    args = parser.parse_args()
    
    target_size = (args.target_size, args.target_size)
    
    print("=" * 70)
    print("CHUẨN BỊ UTKFACE DATASET")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Target size: {target_size}")
    print(f"Train ratio: {args.train_ratio}")
    print()
    
    # Tìm tất cả ảnh
    print("🔍 Đang tìm ảnh...")
    img_files = glob(os.path.join(args.img_dir, '*.jpg'))
    
    if len(img_files) == 0:
        img_files = glob(os.path.join(args.img_dir, '**', '*.jpg'), recursive=True)
    
    if len(img_files) == 0:
        print(f"❌ Không tìm thấy ảnh trong {args.img_dir}")
        print("   Hãy chạy download script hoặc unzip dataset trước!")
        return
    
    print(f"✅ Tìm thấy {len(img_files)} ảnh")
    
    # Parse filenames
    print("\n🔍 Đang parse filenames...")
    valid_files = []
    ages = []
    genders = []
    races = []
    
    for img_file in tqdm(img_files, desc="Parsing"):
        result = parse_filename(img_file)
        if result is not None:
            age, gender, race = result
            valid_files.append(img_file)
            ages.append(age)
            genders.append(gender)
            races.append(race)
    
    print(f"✅ Parse thành công {len(valid_files)}/{len(img_files)} ảnh")
    
    if len(valid_files) == 0:
        print("❌ Không có ảnh hợp lệ!")
        return
    
    # Chọn labels theo task
    if args.task == 'gender':
        labels = np.array(genders, dtype=np.int64)
        nb_classes = 2
        label_names = {0: 'Female', 1: 'Male'}
        task_name = 'Gender Classification'
    elif args.task == 'race':
        labels = np.array(races, dtype=np.int64)
        nb_classes = 5
        label_names = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
        task_name = 'Race Classification'
    else:  # age_group
        # Binary: Young (0-30) vs Old (31+)
        labels = np.array([0 if age <= 30 else 1 for age in ages], dtype=np.int64)
        nb_classes = 2
        label_names = {0: 'Young (≤30)', 1: 'Old (>30)'}
        task_name = 'Age Group Classification'
    
    # Phân tích distribution
    analyze_distribution(labels, label_names)
    
    # Shuffle data
    print("\n🔀 Shuffling data...")
    indices = np.arange(len(valid_files))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    valid_files = [valid_files[i] for i in indices]
    labels = labels[indices]
    ages = np.array(ages, dtype=np.int64)[indices]
    genders = np.array(genders, dtype=np.int64)[indices]
    races = np.array(races, dtype=np.int64)[indices]
    
    # Train/test split
    train_size = int(len(valid_files) * args.train_ratio)
    
    train_files = valid_files[:train_size]
    train_labels = labels[:train_size]
    train_ages = ages[:train_size]
    train_genders = genders[:train_size]
    train_races = races[:train_size]
    
    test_files = valid_files[train_size:]
    test_labels = labels[train_size:]
    test_ages = ages[train_size:]
    test_genders = genders[train_size:]
    test_races = races[train_size:]
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_files)} samples")
    print(f"   Test: {len(test_files)} samples")
    
    # Save train set
    print("\n📦 Tạo utkface_train.h5...")
    if not os.path.exists('utkface_train.h5'):
        with h5py.File('utkface_train.h5', 'w') as h5f:
            h5f.attrs['task'] = args.task
            h5f.attrs['target_size'] = target_size
            h5f.attrs['num_samples'] = len(train_files)
            h5f.attrs['nb_classes'] = nb_classes
            h5f.attrs['has_multitask_labels'] = True
            
            save_to_hdf5(
                h5f,
                train_files,
                train_labels,
                train_ages,
                train_genders,
                train_races,
                target_size,
                args.batch_size,
            )
        print("✅ Đã lưu utkface_train.h5")
    else:
        print("✅ utkface_train.h5 đã tồn tại")
    
    # Save test set
    print("\n📦 Tạo utkface_test.h5...")
    if not os.path.exists('utkface_test.h5'):
        with h5py.File('utkface_test.h5', 'w') as h5f:
            h5f.attrs['task'] = args.task
            h5f.attrs['target_size'] = target_size
            h5f.attrs['num_samples'] = len(test_files)
            h5f.attrs['nb_classes'] = nb_classes
            h5f.attrs['has_multitask_labels'] = True
            
            save_to_hdf5(
                h5f,
                test_files,
                test_labels,
                test_ages,
                test_genders,
                test_races,
                target_size,
                args.batch_size,
            )
        print("✅ Đã lưu utkface_test.h5")
    else:
        print("✅ utkface_test.h5 đã tồn tại")
    
    # Tạo datasetfile
    print("\n📄 Tạo datasetfile...")
    dataset_info = {
        "nb_train": len(train_files),
        "nb_test": len(test_files),
        "input_shape": [3, target_size[0], target_size[1]],
        "nb_classes": nb_classes,
        "dataloader": "dataloader_ver2",
        "task": args.task,
        "task_name": task_name,
        "storage_format": "hdf5",
        "label_names": label_names
    }
    
    with open("datasetfile_ver2", "w") as f:
        json.dump(dataset_info, f, indent=4)
    print("✅ Đã tạo datasetfile_ver2")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT!")
    print("=" * 70)
    print(f"📊 Dataset: UTKFace")
    print(f"🎯 Task: {task_name}")
    print(f"   Train: {len(train_files):,} samples")
    print(f"   Test: {len(test_files):,} samples")
    print(f"   Classes: {nb_classes}")
    print(f"   Image size: {target_size}")
    print(f"\n💾 Files:")
    print(f"   utkface_train.h5 (~{os.path.getsize('utkface_train.h5')/1024**2:.1f}MB)")
    print(f"   utkface_test.h5 (~{os.path.getsize('utkface_test.h5')/1024**2:.1f}MB)")
    print(f"   datasetfile_ver2")
    print(f"\n💡 Bước tiếp theo:")
    print(f"   1. Copy dataloader_optimized.py → dataloader.py")
    print(f"   2. Test: python test_dataloader.py")
    print(f"   3. Init: ./init_fast.sh 3")
    print(f"   4. Train: ./train_fast.sh 3")
    print("=" * 70)


if __name__ == "__main__":
    main()