import os
import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Hàm đọc 1 batch, trả về (data, labels)
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data'] # Mảng numpy với kích thước (10000, 3072)
        labels = batch[b'labels'] # Mảng 1 chiều kích thước 10000

        # Chuyển về dạng (C, H, W) để phù hợp với đầu vào của PyTorch
        images = images.reshape((10000, 3, 32, 32))
        return images, labels

all_images = []
all_labels = []

for i in range(1, 6):
    batch_images, batch_labels = load_cifar_batch(f"cifar-10-batches-py/data_batch_{i}")
    all_images.append(batch_images)
    all_labels.append(batch_labels)

all_images = np.concatenate(all_images, axis=0) # (50000, 3, 32, 32)
all_labels = np.concatenate(all_labels, axis=0) # (50000, )

# Đếm số lượng lớp 
num_class = np.unique(all_labels).shape[0] # 10 lớp

# Lưu vào file
if (not os.path.exists(f'cifar{num_class}_train.npy') 
    or not os.path.exists(f'cifar{num_class}_test.npy')):
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2)
    np.save(f'cifar{num_class}_train.npy', {'X': X_train, 'y': y_train})
    np.save(f'cifar{num_class}_test.npy', {'X': X_test, 'y': y_test})

    # Cập nhật file dataset
    if not os.path.exists("datasetfile"):
        dataset_info = {
            "nb_train": len(X_train),
            "nb_test": len(X_test),
            "input_shape": X_train.shape[1:],
            "nb_classes": num_class,
            "dataloader": "dataloader"
        }

        with open("datasetfile", "w") as f:
            json.dump(dataset_info, f, indent=4)