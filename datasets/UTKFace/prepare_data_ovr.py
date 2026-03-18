"""
Chuẩn bị UTKFace dataset cho kiến trúc One-vs-Rest (OVR).

- Parse filename để lấy age, gender, race
- Tạo train/val/test split (mặc định 80/10/10)
- Lưu dưới dạng HDF5 (memory-efficient)
- Giữ đầy đủ:
    * images
    * age  (0..116)
    * gender (0=male, 1=female)
    * race (0=White,1=Black,2=Asian,3=Indian,4=Others)

Các head OVR sẽ được định nghĩa trong dataloader_ovr.py:
- 2 head gender: female / male
- 3 head age bin: 0-18, 19-60, 61-116
- 5 head race: white / black / asian / indian / others
"""

import os
import json
import argparse
import shutil
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
        gender (int): 0=Male, 1=Female
        race (int): 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
        None if parsing fails
    """
    try:
        basename = os.path.basename(filename)
        parts = basename.split("_")
        if len(parts) < 3:
            return None
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
        if not (0 <= age <= 116 and gender in [0, 1] and 0 <= race <= 4):
            return None
        return age, gender, race
    except (ValueError, IndexError):
        return None


def load_images_batch(img_files, target_size=(64, 64), batch_size=1000):
    """
    Generator: Load ảnh theo batch.

    Yields:
        batch_images: numpy array (B, 3, H, W)
        batch_indices: list index global của các ảnh trong batch
    """
    batch_images = []
    batch_indices = []
    failed = []

    for idx, img_file in enumerate(img_files):
        try:
            img = Image.open(img_file).convert("RGB")
            img = img.resize(target_size, Image.LANCZOS)
            arr = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
            batch_images.append(arr)
            batch_indices.append(idx)

            if len(batch_images) == batch_size or idx == len(img_files) - 1:
                yield (
                    np.array(batch_images, dtype=np.uint8),
                    batch_indices,
                )
                batch_images = []
                batch_indices = []
        except Exception:
            failed.append(img_file)
            continue

    if failed:
        print(f"\n⚠️  Không tải được {len(failed)} ảnh")


def save_to_hdf5(h5_file, img_files, ages, genders, races,
                 target_size=(64, 64), batch_size=1000):
    """Lưu ảnh + age/gender/race vào HDF5."""
    total = len(img_files)

    images_ds = h5_file.create_dataset(
        "images",
        shape=(total, 3, target_size[0], target_size[1]),
        dtype="uint8",
        chunks=(1, 3, target_size[0], target_size[1]),
        compression="gzip",
        compression_opts=4,
    )
    age_ds = h5_file.create_dataset("age", shape=(total,), dtype="int64")
    gender_ds = h5_file.create_dataset("gender", shape=(total,), dtype="int64")
    race_ds = h5_file.create_dataset("race", shape=(total,), dtype="int64")

    print(f"Đang lưu {total} ảnh vào HDF5 (OVR)...")
    saved = 0

    for batch_images, batch_idxs in tqdm(
        load_images_batch(img_files, target_size, batch_size),
        total=total // batch_size + 1,
        desc="Processing images",
    ):
        for i, idx in enumerate(batch_idxs):
            images_ds[idx] = batch_images[i]
            age_ds[idx] = ages[idx]
            gender_ds[idx] = genders[idx]
            race_ds[idx] = races[idx]
        saved += len(batch_idxs)

    print(f"✅ Đã lưu {saved}/{total} ảnh")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị UTKFace OVR dataset")
    parser.add_argument(
        "--img_dir", default="UTKFace", help="Thư mục chứa ảnh UTKFace"
    )
    parser.add_argument(
        "--target_size", type=int, default=64, help="Resize ảnh (default: 64x64)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size khi prepare"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train split ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè lại toàn bộ file *_ovr.h5 hiện có để đồng bộ split.",
    )
    args = parser.parse_args()

    target_size = (args.target_size, args.target_size)

    print("=" * 70)
    print("CHUẨN BỊ UTKFACE DATASET - OVR")
    print("=" * 70)
    print(f"Target size: {target_size}")
    print(
        "Split ratio: "
        f"train={args.train_ratio}, val={args.val_ratio}, "
        f"test={1.0 - args.train_ratio - args.val_ratio}"
    )
    print()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if args.train_ratio <= 0 or args.val_ratio <= 0 or test_ratio <= 0:
        raise ValueError(
            "Tỉ lệ không hợp lệ. Cần train_ratio>0, val_ratio>0 và train+val<1."
        )

    # Tìm ảnh
    print("🔍 Đang tìm ảnh...")
    img_files = glob(os.path.join(args.img_dir, "*.jpg"))
    if len(img_files) == 0:
        img_files = glob(os.path.join(args.img_dir, "**", "*.jpg"), recursive=True)

    if len(img_files) == 0:
        print(f"❌ Không tìm thấy ảnh trong {args.img_dir}")
        return

    print(f"✅ Tìm thấy {len(img_files)} ảnh")

    # Parse filename
    print("\n🔍 Đang parse filenames...")
    valid_files = []
    ages = []
    genders = []
    races = []

    for fpath in tqdm(img_files, desc="Parsing"):
        parsed = parse_filename(fpath)
        if parsed is None:
            continue
        age, gender, race = parsed
        valid_files.append(fpath)
        ages.append(age)
        genders.append(gender)
        races.append(race)

    print(f"✅ Parse thành công {len(valid_files)}/{len(img_files)} ảnh")
    if len(valid_files) == 0:
        print("❌ Không có ảnh hợp lệ!")
        return

    ages = np.asarray(ages, dtype=np.int64)
    genders = np.asarray(genders, dtype=np.int64)
    races = np.asarray(races, dtype=np.int64)

    # Shuffle
    print("\n🔀 Shuffling data...")
    idx_all = np.arange(len(valid_files))
    np.random.seed(42)
    np.random.shuffle(idx_all)

    valid_files = [valid_files[i] for i in idx_all]
    ages = ages[idx_all]
    genders = genders[idx_all]
    races = races[idx_all]

    # Train/val/test split
    n_total = len(valid_files)
    train_size = int(n_total * args.train_ratio)
    val_size = int(n_total * args.val_ratio)
    test_size = n_total - train_size - val_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            "Kích thước split không hợp lệ. Hãy điều chỉnh train_ratio/val_ratio."
        )

    train_end = train_size
    val_end = train_size + val_size

    train_files = valid_files[:train_end]
    val_files = valid_files[train_end:val_end]
    test_files = valid_files[val_end:]

    train_ages = ages[:train_end]
    val_ages = ages[train_end:val_end]
    test_ages = ages[val_end:]
    train_genders = genders[:train_end]
    val_genders = genders[train_end:val_end]
    test_genders = genders[val_end:]
    train_races = races[:train_end]
    val_races = races[train_end:val_end]
    test_races = races[val_end:]

    print(f"\n📊 Split:")
    print(f"   Train: {len(train_files)} samples")
    print(f"   Val  : {len(val_files)} samples")
    print(f"   Test:  {len(test_files)} samples")

    train_h5 = "utkface_train_ovr.h5"
    val_h5 = "utkface_val_ovr.h5"
    test_h5 = "utkface_test_ovr.h5"
    existing_h5 = [p for p in [train_h5, val_h5, test_h5] if os.path.exists(p)]

    if existing_h5 and not args.overwrite:
        raise FileExistsError(
            "Đã tồn tại file OVR HDF5. Để tránh lệch split/metadata, "
            "hãy chạy lại với --overwrite hoặc xóa các file: "
            f"{', '.join(existing_h5)}"
        )

    if args.overwrite:
        for p in existing_h5:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)

    # Lưu HDF5 train
    print("\n📦 Tạo utkface_train_ovr.h5...")
    with h5py.File(train_h5, "w") as h5f:
        h5f.attrs["target_size"] = target_size
        h5f.attrs["num_samples"] = len(train_files)
        h5f.attrs["has_ovr_labels"] = True
        save_to_hdf5(
            h5f,
            train_files,
            train_ages,
            train_genders,
            train_races,
            target_size,
            args.batch_size,
        )
    print("✅ Đã lưu utkface_train_ovr.h5")

    # Lưu HDF5 val
    print("\n📦 Tạo utkface_val_ovr.h5...")
    with h5py.File(val_h5, "w") as h5f:
        h5f.attrs["target_size"] = target_size
        h5f.attrs["num_samples"] = len(val_files)
        h5f.attrs["has_ovr_labels"] = True
        save_to_hdf5(
            h5f,
            val_files,
            val_ages,
            val_genders,
            val_races,
            target_size,
            args.batch_size,
        )
    print("✅ Đã lưu utkface_val_ovr.h5")

    # Lưu HDF5 test
    print("\n📦 Tạo utkface_test_ovr.h5...")
    with h5py.File(test_h5, "w") as h5f:
        h5f.attrs["target_size"] = target_size
        h5f.attrs["num_samples"] = len(test_files)
        h5f.attrs["has_ovr_labels"] = True
        save_to_hdf5(
            h5f,
            test_files,
            test_ages,
            test_genders,
            test_races,
            target_size,
            args.batch_size,
        )
    print("✅ Đã lưu utkface_test_ovr.h5")

    # Tạo datasetfile_ovr
    print("\n📄 Tạo datasetfile_ovr...")
    dataset_info = {
        "nb_train": len(train_files),
        "nb_val": len(val_files),
        "nb_test": len(test_files),
        "input_shape": [3, target_size[0], target_size[1]],
        "nb_classes": 2,  # Mỗi head OVR là binary
        "dataloader": "dataloader_ovr",
        "storage_format": "hdf5",
    }

    with open("datasetfile_ovr", "w") as f:
        json.dump(dataset_info, f, indent=4)
    print("✅ Đã tạo datasetfile_ovr")

    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT UTKFACE OVR!")
    print("=" * 70)
    print(f"Train: {len(train_files):,} samples")
    print(f"Val  : {len(val_files):,} samples")
    print(f"Test:  {len(test_files):,} samples")
    print(f"Image size: {target_size}")
    print("Files:")
    print("  utkface_train_ovr.h5")
    print("  utkface_val_ovr.h5")
    print("  utkface_test_ovr.h5")
    print("  datasetfile_ovr")
    print("=" * 70)


if __name__ == "__main__":
    main()

