"""
CelebA HDF5 dataloader (multitask - 27 attributes)
Support:
    train
    val
    test
"""

import numpy as np
import h5py
import os
import torch
from torchvision import transforms


# ============================================================
# PATH
# ============================================================

pwd = os.path.dirname(os.path.realpath(__file__))

train_path = os.path.join(pwd, "celeba_train.h5")
val_path   = os.path.join(pwd, "celeba_val.h5")
test_path  = os.path.join(pwd, "celeba_test.h5")


train_file = h5py.File(train_path, "r")
val_file   = h5py.File(val_path, "r")
test_file  = h5py.File(test_path, "r")


train_size = train_file.attrs["num_samples"]
val_size   = val_file.attrs["num_samples"]
test_size  = test_file.attrs["num_samples"]


print("✅ CelebA HDF5 loaded")
print("Train:", train_size)
print("Val  :", val_size)
print("Test :", test_size)



# ============================================================
# TRANSFORM
# ============================================================

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


eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])



# ============================================================
# HELPERS
# ============================================================

def _normalize_indices(indices):

    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)

    if indices.ndim == 0:
        indices = np.array([indices])

    return indices.astype(np.int64)



# ============================================================
# IMAGE PROCESS
# ============================================================

def _apply_transform(images, category):

    processed = []

    for img in images:

        # CHW -> HWC
        img = img.transpose(1,2,0)

        img = img.astype(np.uint8)


        if category == "train":
            img = train_transform(img)

        else:
            img = eval_transform(img)


        processed.append(img)


    return torch.stack(processed).numpy()



# ============================================================
# MAIN API
# ============================================================

def load(indices, category="train"):


    if category == "train":
        h5_file = train_file

    elif category == "val":
        h5_file = val_file

    elif category == "test":
        h5_file = test_file

    else:
        raise ValueError(
            f"Unknown category: {category}"
        )


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


        H = h5_file["images"].shape[2]

        W = h5_file["images"].shape[3]


        X = np.zeros(
            (0,3,H,W),
            dtype=np.uint8
        )

        y = np.zeros(
            (0,27),
            dtype=np.int64
        )



    X = _apply_transform(
        X,
        category
    )


    return X,y



# ============================================================
# SIZE
# ============================================================

def get_dataset_size(category="train"):


    if category == "train":
        return len(train_file["images"])


    if category == "val":
        return len(val_file["images"])


    if category == "test":
        return len(test_file["images"])


    raise ValueError(
        f"Invalid category {category}"
    )



# ============================================================
# CLOSE
# ============================================================

def close():

    train_file.close()

    val_file.close()

    test_file.close()

    print("HDF5 closed")