"""
Dataloader cho CelebA dataset
Load dữ liệu từ file .npy đã được chuẩn bị sẵn
"""

import numpy as np
import os

pwd = os.path.dirname(os.path.realpath(__file__))

# Đường dẫn đến các file dữ liệu
train_path = os.path.join(pwd, 'celeba_train.npy')
test_path = os.path.join(pwd, 'celeba_test.npy')

# Kiểm tra file tồn tại
if not os.path.exists(train_path):
    raise FileNotFoundError(
        f"Không tìm thấy {train_path}. "
        "Hãy chạy prepare_data.py trước!"
    )

if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"Không tìm thấy {test_path}. "
        "Hãy chạy prepare_data.py trước!"
    )

# Load dữ liệu
try:
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    
    # Xử lý format của numpy array
    if train_data.ndim == 0:
        train_data = train_data.item()
    elif train_data.shape == (1,):
        train_data = train_data.reshape((1,))[0]
    
    if test_data.ndim == 0:
        test_data = test_data.item()
    elif test_data.shape == (1,):
        test_data = test_data.reshape((1,))[0]
    
    # Trích xuất và normalize dữ liệu
    X_train = train_data['X'].astype(np.float32) / 255.0  # Normalize về [0, 1]
    X_test = test_data['X'].astype(np.float32) / 255.0
    y_train = train_data['y'].astype(np.int64)
    y_test = test_data['y'].astype(np.int64)
    
    print(f"✅ Đã load CelebA dataset:")
    print(f"   Train: {X_train.shape}, labels: {y_train.shape}")
    print(f"   Test: {X_test.shape}, labels: {y_test.shape}")
    
except Exception as e:
    raise RuntimeError(f"Lỗi khi load dữ liệu: {e}")

def load(indices, category='train'):
    """
    Load dữ liệu theo indices
    
    Args:
        indices: array hoặc list các chỉ số
        category: 'train' hoặc 'test'
    
    Returns:
        X: ảnh (numpy array, shape: [N, 3, 64, 64])
        y: labels (numpy array, shape: [N])
    """
    if category == 'train':
        return X_train[indices], y_train[indices]
    elif category == 'test':
        return X_test[indices], y_test[indices]
    else:
        raise ValueError(f"category phải là 'train' hoặc 'test', nhận được: {category}")