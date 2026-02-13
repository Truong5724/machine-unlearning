"""
Dataloader cho CelebA dataset - TỐI ƯU CHO COLAB
Sử dụng HDF5 với lazy loading - chỉ load ảnh khi cần

QUAN TRỌNG:
- Không load toàn bộ dataset vào RAM
- Chỉ load ảnh khi được yêu cầu qua hàm load()
- Tiết kiệm RAM tối đa cho Colab

Yêu cầu: pip install h5py
"""

import numpy as np
import h5py
import os

pwd = os.path.dirname(os.path.realpath(__file__))

# Đường dẫn đến HDF5 files
train_path = os.path.join(pwd, 'celeba_train.h5')
test_path = os.path.join(pwd, 'celeba_test.h5')

# Kiểm tra files tồn tại
if not os.path.exists(train_path):
    raise FileNotFoundError(
        f"Không tìm thấy {train_path}. "
        "Hãy chạy prepare_data_optimized.py trước!"
    )

if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"Không tìm thấy {test_path}. "
        "Hãy chạy prepare_data_optimized.py trước!"
    )

# Mở HDF5 files trong chế độ read-only
# 'r' mode cho phép nhiều process đọc cùng lúc
train_file = h5py.File(train_path, 'r')
test_file = h5py.File(test_path, 'r')

# Lấy thông tin metadata
try:
    train_size = train_file.attrs['num_samples']
    test_size = test_file.attrs['num_samples']
    attribute = train_file.attrs['attribute']
    
    print(f"✅ Đã kết nối CelebA HDF5 dataset:")
    print(f"   Train: {train_size} samples")
    print(f"   Test: {test_size} samples")
    print(f"   Attribute: {attribute}")
    print(f"   💾 Memory-efficient: Lazy loading enabled")
    
except Exception as e:
    print(f"⚠️  Cảnh báo: Không đọc được metadata: {e}")
    train_size = len(train_file['images'])
    test_size = len(test_file['images'])
    print(f"✅ Đã kết nối dataset: Train={train_size}, Test={test_size}")


def load(indices, category='train'):
    """
    Load dữ liệu theo indices - LAZY LOADING
    
    Chỉ load ảnh được yêu cầu, không load toàn bộ dataset vào RAM
    
    Args:
        indices: array hoặc list các chỉ số cần load
        category: 'train' hoặc 'test'
    
    Returns:
        X: ảnh (numpy array, shape: [N, 3, 64, 64]), đã normalize [0, 1]
        y: labels (numpy array, shape: [N])
    """
    if category == 'train':
        h5_file = train_file
    elif category == 'test':
        h5_file = test_file
    else:
        raise ValueError(f"category phải là 'train' hoặc 'test', nhận: {category}")
    
    # Convert indices sang numpy array nếu cần
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    
    # Đảm bảo indices là 1D array
    if indices.ndim == 0:
        indices = np.array([indices])
    
    # QUAN TRỌNG: HDF5 yêu cầu indices phải sorted!
    # Lưu thứ tự gốc để trả về đúng thứ tự
    if len(indices) > 0:
        sorted_idx = np.argsort(indices)
        sorted_indices = indices[sorted_idx]
        
        # Load với sorted indices
        X = h5_file['images'][sorted_indices]  # Shape: (N, 3, 64, 64), dtype: uint8
        y = h5_file['labels'][sorted_indices]   # Shape: (N,), dtype: int64
        
        # Trả về theo thứ tự gốc
        unsort_idx = np.argsort(sorted_idx)
        X = X[unsort_idx]
        y = y[unsort_idx]
    else:
        # Empty indices
        X = np.array([], dtype=np.uint8).reshape(0, 3, 64, 64)
        y = np.array([], dtype=np.int64)
    
    # Normalize về [0, 1] để training
    X = X.astype(np.float32) / 255.0
    
    return X, y


def get_dataset_size(category='train'):
    """
    Lấy kích thước dataset mà không cần load data
    
    Args:
        category: 'train' hoặc 'test'
    
    Returns:
        int: số lượng samples
    """
    if category == 'train':
        return len(train_file['images'])
    elif category == 'test':
        return len(test_file['images'])
    else:
        raise ValueError(f"category phải là 'train' hoặc 'test'")


def close():
    """
    Đóng HDF5 files khi không cần nữa
    Gọi hàm này khi kết thúc training để giải phóng resources
    """
    train_file.close()
    test_file.close()
    print("✅ Đã đóng HDF5 files")


# Giữ lại các biến này để tương thích với code cũ
# Nhưng không load data vào RAM
class LazyArray:
    """
    Wrapper class để giả lập numpy array nhưng không load data
    """
    def __init__(self, h5_dataset):
        self.h5_dataset = h5_dataset
        self.shape = h5_dataset.shape
        self.dtype = h5_dataset.dtype
    
    def __getitem__(self, key):
        """Lazy loading khi truy cập qua index"""
        # Handle sorting for HDF5
        if isinstance(key, (list, np.ndarray)):
            key = np.array(key)
            if len(key) > 0:
                sorted_idx = np.argsort(key)
                sorted_key = key[sorted_idx]
                data = self.h5_dataset[sorted_key]
                unsort_idx = np.argsort(sorted_idx)
                data = data[unsort_idx]
            else:
                data = np.array([])
        else:
            data = self.h5_dataset[key]
        
        if self.h5_dataset.name.endswith('images'):
            return data.astype(np.float32) / 255.0
        return data
    
    def __len__(self):
        return len(self.h5_dataset)


# Tạo lazy arrays để tương thích với code cũ
X_train = LazyArray(train_file['images'])
X_test = LazyArray(test_file['images'])
y_train = LazyArray(train_file['labels'])
y_test = LazyArray(test_file['labels'])

print(f"   📊 X_train shape: {X_train.shape}")
print(f"   📊 X_test shape: {X_test.shape}")