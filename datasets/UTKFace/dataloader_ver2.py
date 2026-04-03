"""
UTKFace HDF5 dataloader (lazy loading + augmentation)
Tối ưu cho Google Colab
"""

import numpy as np
import h5py
import os
import torch
from torchvision import transforms

# ─────────────────────────────────────────────
# AGE BINNING
# [0..17] → Young
# [18..59] → Adult
# [60..116] → Senior
# ─────────────────────────────────────────────

AGE_EDGES = np.array([0, 18, 60, 117], dtype=np.int64)
AGE_NB_CLASSES = 3

pwd = os.path.dirname(os.path.realpath(__file__))

train_path = os.path.join(pwd, "utkface_train.h5")
test_path = os.path.join(pwd, "utkface_test.h5")

train_file = h5py.File(train_path, "r")
test_file = h5py.File(test_path, "r")

train_size = train_file.attrs["num_samples"]
test_size = test_file.attrs["num_samples"]

print("✅ UTKFace HDF5 loaded")
print("Train:", train_size)
print("Test :", test_size)


# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _normalize_indices(indices):

    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)

    if indices.ndim == 0:
        indices = np.array([indices])

    return indices.astype(np.int64)


def _fetch_sorted(h5_file, key, indices):

    if len(indices) == 0:
        return np.array([], dtype=np.int64)

    sorted_idx = np.argsort(indices)
    sorted_indices = indices[sorted_idx]

    values = h5_file[key][sorted_indices]

    unsort_idx = np.argsort(sorted_idx)

    return np.asarray(values[unsort_idx], dtype=np.int64)


def _age_to_3bins(age_values):

    age_values = np.asarray(age_values, dtype=np.int64)

    age_values = np.clip(age_values, AGE_EDGES[0], AGE_EDGES[-1]-1)

    return np.digitize(age_values, AGE_EDGES[1:-1], right=False).astype(np.int64)


# ─────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────

def _apply_transform(images, category):

    processed = []

    for img in images:

        img = img.transpose(1,2,0)  # CHW → HWC
        img = img.astype(np.uint8)

        if category == "train":
            img = train_transform(img)
        else:
            img = test_transform(img)

        processed.append(img)

    return torch.stack(processed).numpy()


# ─────────────────────────────────────────────
# MAIN API
# ─────────────────────────────────────────────

def load(indices, category="train"):

    h5_file = train_file if category=="train" else test_file

    indices = _normalize_indices(indices)

    if len(indices) > 0:

        sorted_idx = np.argsort(indices)
        sorted_indices = indices[sorted_idx]

        X = h5_file["images"][sorted_indices]
        y = h5_file["labels"][sorted_indices]

        unsort_idx = np.argsort(sorted_idx)

        X = X[unsort_idx]
        y = y[unsort_idx]

    else:

        H = train_file["images"].shape[2]
        W = train_file["images"].shape[3]

        X = np.zeros((0,3,H,W), dtype=np.uint8)
        y = np.array([], dtype=np.int64)

    X = _apply_transform(X, category)

    return X, y


def load_attributes(indices, category="train"):

    h5_file = train_file if category=="train" else test_file

    indices = _normalize_indices(indices)

    ages = _fetch_sorted(h5_file,"age",indices)
    genders = _fetch_sorted(h5_file,"gender",indices)
    races = _fetch_sorted(h5_file,"race",indices)

    if "age_bin" in h5_file:

        age_bin = _fetch_sorted(h5_file,"age_bin",indices)

    else:

        age_bin = _age_to_3bins(ages)

    return {

        "age": ages,
        "age_bin": age_bin,
        "gender": genders,
        "race": races
    }


def load_multitask(indices, category="train"):

    X,_ = load(indices, category)

    attrs = load_attributes(indices, category)

    return X, {

        "gender": attrs["gender"],
        "age": attrs["age_bin"],
        "race": attrs["race"]
    }


def get_dataset_size(category="train"):

    if category=="train":
        return len(train_file["images"])

    if category=="test":
        return len(test_file["images"])

    raise ValueError


def close():

    train_file.close()
    test_file.close()

    print("HDF5 closed")