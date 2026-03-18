"""
Dataloader cho UTKFace OVR dataset - lazy loading với HDF5.

Files:
- utkface_train_ovr.h5
    * images: (N, 3, 64, 64) uint8
    * age   : (N,) int64
    * gender: (N,) int64
    * race  : (N,) int64
- utkface_val_ovr.h5
- utkface_test_ovr.h5

Các head OVR (10 binary tasks):
- gender_female: gender == 1
- gender_male  : gender == 0
- age_bin0     : age in [0,18]
- age_bin1     : age in [19,60]
- age_bin2     : age in [61,116]
- race_white   : race == 0
- race_black   : race == 1
- race_asian   : race == 2
- race_indian  : race == 3
- race_others  : race == 4
"""

import os
import numpy as np
import h5py

pwd = os.path.dirname(os.path.realpath(__file__))

train_path = os.path.join(pwd, "utkface_train_ovr.h5")
val_path = os.path.join(pwd, "utkface_val_ovr.h5")
test_path = os.path.join(pwd, "utkface_test_ovr.h5")

if not os.path.exists(train_path):
    raise FileNotFoundError(
        f"Không tìm thấy {train_path}. Hãy chạy prepare_data_ovr.py trước!"
    )
if not os.path.exists(val_path):
    raise FileNotFoundError(
        f"Không tìm thấy {val_path}. Hãy chạy prepare_data_ovr.py trước!"
    )
if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"Không tìm thấy {test_path}. Hãy chạy prepare_data_ovr.py trước!"
    )

train_file = h5py.File(train_path, "r")
val_file = h5py.File(val_path, "r")
test_file = h5py.File(test_path, "r")

try:
    train_size = train_file.attrs["num_samples"]
    val_size = val_file.attrs["num_samples"]
    test_size = test_file.attrs["num_samples"]
    print("✅ Đã kết nối UTKFace OVR HDF5:")
    print(f"   Train: {train_size} samples")
    print(f"   Val  : {val_size} samples")
    print(f"   Test : {test_size} samples")
except Exception as e:
    print(f"⚠️  Không đọc được metadata OVR: {e}")
    train_size = len(train_file["images"])
    val_size = len(val_file["images"])
    test_size = len(test_file["images"])
    print(f"   Train: {train_size} samples")
    print(f"   Val  : {val_size} samples")
    print(f"   Test : {test_size} samples")


def _select_file(category):
    if category == "train":
        return train_file
    if category == "val":
        return val_file
    if category == "test":
        return test_file
    raise ValueError(f"category phải là 'train', 'val' hoặc 'test', nhận: {category}")


def _normalize_indices(indices):
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    if indices.ndim == 0:
        indices = np.array([indices])
    return indices


def load(indices, category="train"):
    """
    Load ảnh theo indices, normalize về [0,1].

    Returns:
        X: (N, 3, 64, 64) float32
        y: dummy labels (all zeros) để tương thích, không dùng cho OVR.
    """
    h5_file = _select_file(category)
    indices = _normalize_indices(indices)

    if len(indices) == 0:
        X = np.empty((0, 3, 64, 64), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        return X, y

    sorted_idx = np.argsort(indices)
    sorted_indices = indices[sorted_idx]

    X = h5_file["images"][sorted_indices]
    # tạo dummy label 0 cho tương thích
    y = np.zeros(len(sorted_indices), dtype=np.int64)

    unsort_idx = np.argsort(sorted_idx)
    X = X[unsort_idx]
    y = y[unsort_idx]

    X = X.astype(np.float32) / 255.0
    return X, y


def load_ovr_labels(indices, category="train"):
    """
    Trả về dictionary 10 head binary cho các indices cho sẵn.
    """
    h5_file = _select_file(category)
    indices = _normalize_indices(indices)

    if len(indices) == 0:
        empty = np.empty((0,), dtype=np.int64)
        return {
            "gender_female": empty,
            "gender_male": empty,
            "age_bin0": empty,
            "age_bin1": empty,
            "age_bin2": empty,
            "race_white": empty,
            "race_black": empty,
            "race_asian": empty,
            "race_indian": empty,
            "race_others": empty,
        }

    sorted_idx = np.argsort(indices)
    sorted_indices = indices[sorted_idx]

    ages = np.asarray(h5_file["age"][sorted_indices], dtype=np.int64)
    genders = np.asarray(h5_file["gender"][sorted_indices], dtype=np.int64)
    races = np.asarray(h5_file["race"][sorted_indices], dtype=np.int64)

    # Unsort để trả về đúng thứ tự yêu cầu
    unsort_idx = np.argsort(sorted_idx)
    ages = ages[unsort_idx]
    genders = genders[unsort_idx]
    races = races[unsort_idx]

    # Age bins: 0-18, 19-60, 61-116
    ages = np.clip(ages, 0, 116)
    age_bin0 = (ages <= 18).astype(np.int64)
    age_bin1 = ((ages >= 19) & (ages <= 60)).astype(np.int64)
    age_bin2 = (ages >= 61).astype(np.int64)

    labels = {
        # UTKFace convention used here: 0=male, 1=female
        "gender_female": (genders == 1).astype(np.int64),
        "gender_male": (genders == 0).astype(np.int64),
        "age_bin0": age_bin0,
        "age_bin1": age_bin1,
        "age_bin2": age_bin2,
        "race_white": (races == 0).astype(np.int64),
        "race_black": (races == 1).astype(np.int64),
        "race_asian": (races == 2).astype(np.int64),
        "race_indian": (races == 3).astype(np.int64),
        "race_others": (races == 4).astype(np.int64),
    }
    return labels


def load_ovr(indices, category="train"):
    """
    Trả về:
        X: ảnh (N,3,64,64)
        y_dict: dict 10 head binary
    """
    X, _ = load(indices, category=category)
    y = load_ovr_labels(indices, category=category)
    return X, y


def get_dataset_size(category="train"):
    if category == "train":
        return len(train_file["images"])
    if category == "val":
        return len(val_file["images"])
    if category == "test":
        return len(test_file["images"])
    raise ValueError("category phải là 'train', 'val' hoặc 'test'")


def close():
    train_file.close()
    val_file.close()
    test_file.close()
    print("✅ Đã đóng HDF5 OVR")

