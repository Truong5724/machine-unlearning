import os
import json
import numpy as np
import pickle

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

# Số lượng lớp 
num_class = 10 # np.unique(all_labels).shape[0] (nếu muốn chắc chắn)

# Tải dữ liệu test
test_images, test_labels = load_cifar_batch(f"cifar-10-batches-py/test_batch")

from sklearn.model_selection import train_test_split

if not os.path.exists(f'cifar{num_class}_train.npy'):
    X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.1, random_state=42, stratify=all_labels)
    np.save(f'cifar{num_class}_train.npy', {'X': X_train, 'y': y_train})
    np.save(f'cifar{num_class}_val.npy', {'X': X_val, 'y': y_val})

# Lưu vào file
if not os.path.exists(f'cifar{num_class}_test.npy'):
    np.save(f'cifar{num_class}_test.npy', {'X': test_images, 'y': np.array(test_labels)})

# Cập nhật file dataset
if not os.path.exists("datasetfile"):
    dataset_info = {
        "nb_train": len(X_train),
        "nb_val": len(X_val),
        "nb_test": len(test_images),
        "input_shape": all_images.shape[1:],
        "nb_classes": num_class,
        "dataloader": "dataloader"
    }

    with open("datasetfile", "w") as f:
        json.dump(dataset_info, f, indent=4)