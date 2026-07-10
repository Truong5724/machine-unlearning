"""
CelebA HDF5 dataloader (multitask - 27 attributes)
Tương tự phong cách UTKFace
"""

import numpy as np
import h5py
import os
import torch
from torchvision import transforms

# ─────────────────────────────────────────────
pwd = os.path.dirname(os.path.realpath(__file__))

train_path = os.path.join(pwd, "celeba_train.h5")
test_path = os.path.join(pwd, "celeba_test.h5")

train_file = h5py.File(train_path, "r")
test_file = h5py.File(test_path, "r")

train_size = train_file.attrs["num_samples"]
test_size = test_file.attrs["num_samples"]

print("✅ CelebA HDF5 loaded")
print("Train:", train_size)
print("Test :", test_size)


# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
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


# ─────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────

def _apply_transform(images, category):
    processed = []
    for img in images:
        img = img.transpose(1, 2, 0)  # CHW → HWC
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
    h5_file = train_file if category == "train" else test_file
    indices = _normalize_indices(indices)

    if len(indices) > 0:
        sorted_idx = np.argsort(indices)
        sorted_indices = indices[sorted_idx]

        X = h5_file["images"][sorted_indices]
        y = h5_file["labels"][sorted_indices]   # shape (N, 27)

        unsort_idx = np.argsort(sorted_idx)
        X = X[unsort_idx]
        y = y[unsort_idx]
    else:
        H = train_file["images"].shape[2]
        W = train_file["images"].shape[3]
        X = np.zeros((0, 3, H, W), dtype=np.uint8)
        y = np.zeros((0, 27), dtype=np.int64)

    X = _apply_transform(X, category)
    return X, y


def get_dataset_size(category="train"):
    if category == "train":
        return len(train_file["images"])
    if category == "test":
        return len(test_file["images"])
    raise ValueError("Invalid category")


def close():
    train_file.close()
    test_file.close()
    print("HDF5 closed")