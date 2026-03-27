"""
Dataloader cho CelebA OVR dataset
Sử dụng HDF5 với lazy loading - chỉ load ảnh khi cần

Yêu cầu: pip install h5py
"""

import numpy as np
import h5py
import os

pwd = os.path.dirname(os.path.realpath(__file__))

# Đường dẫn đến HDF5 files
train_path = os.path.join(pwd, 'celeba_ovr_train.h5')
val_path = os.path.join(pwd, 'celeba_ovr_val.h5')
test_path = os.path.join(pwd, 'celeba_ovr_test.h5')

# Kiểm tra files tồn tại
if not os.path.exists(train_path):
    raise FileNotFoundError(
        f"Không tìm thấy {train_path}. "
        "Hãy chạy prepare_data_ovr.py trước!"
    )

if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"Không tìm thấy {test_path}. "
        "Hãy chạy prepare_data_ovr.py trước!"
    )

if not os.path.exists(val_path):
    raise FileNotFoundError(
        f"Không tìm thấy {val_path}. "
        "Hãy chạy prepare_data_ovr.py trước!"
    )

# Mở HDF5 files trong chế độ read-only
train_file = h5py.File(train_path, 'r')
val_file = h5py.File(val_path, 'r')
test_file = h5py.File(test_path, 'r')

# Lấy thông tin metadata
try:
    train_size = train_file.attrs['num_samples']
    val_size = val_file.attrs['num_samples']
    test_size = test_file.attrs['num_samples']
    attributes = list(train_file.attrs['attributes'])
    tasks = list(train_file.attrs['tasks'])
    
    print(f"✅ Đã kết nối CelebA OVR HDF5 dataset:")
    print(f"   Train: {train_size} samples")
    print(f"   Val  : {val_size} samples")
    print(f"   Test: {test_size} samples")
    print(f"   Attributes: {len(attributes)}")
    print(f"   💾 Memory-efficient: Lazy loading enabled")
    
except Exception as e:
    print(f"⚠️  Cảnh báo: Không đọc được metadata: {e}")
    train_size = len(train_file['images'])
    val_size = len(val_file['images'])
    test_size = len(test_file['images'])
    attributes = []
    tasks = []
    print(f"✅ Đã kết nối dataset: Train={train_size}, Val={val_size}, Test={test_size}")


def load(indices, category='train'):
    """
    Load dữ liệu theo indices - LAZY LOADING
    
    Args:
        indices: array hoặc list các chỉ số cần load
        category: 'train', 'val' hoặc 'test'
    
    Returns:
        X: ảnh (numpy array, shape: [N, 3, 64, 64]), dtype uint8
        y: labels (numpy array, shape: [N, num_attributes]), dtype int64
    """
    if category == 'train':
        h5_file = train_file
    elif category == 'val':
        h5_file = val_file
    elif category == 'test':
        h5_file = test_file
    else:
        raise ValueError(f"category phải là 'train', 'val' hoặc 'test', nhận: {category}")
    
    # Convert indices sang numpy array nếu cần
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    
    # Đảm bảo indices là 1D array
    if indices.ndim == 0:
        indices = np.array([indices])
    
    # HDF5 yêu cầu indices phải sorted!
    # Lưu thứ tự gốc để trả về đúng thứ tự
    if len(indices) > 0:
        sorted_idx = np.argsort(indices)
        sorted_indices = indices[sorted_idx]
        
        # Load với sorted indices
        X = h5_file['images'][sorted_indices]  # Shape: (N, 3, 64, 64), dtype: uint8
        y = h5_file['labels'][sorted_indices]   # Shape: (N, num_attributes), dtype: int64
        
        # Trả về theo thứ tự gốc
        unsort_idx = np.argsort(sorted_idx)
        X = X[unsort_idx]
        y = y[unsort_idx]
    else:
        X = np.zeros((0, 3, 64, 64), dtype=np.uint8)
        y = np.zeros((0, len(attributes)), dtype=np.int64)
    
    # Normalize ảnh từ uint8 [0, 255] sang float32 [0, 1]
    X = X.astype(np.float32) / 255.0
    
    return X, y


def load_ovr(indices, category='train'):
    """
    Load dữ liệu OVR - trả về dict {task_name: labels}
    
    Args:
        indices: array hoặc list các chỉ số cần load
        category: 'train', 'val' hoặc 'test'
    
    Returns:
        X: ảnh (numpy array, shape: [N, 3, 64, 64])
        y_dict: {task_name: labels array [N]}
    """
    X, y_full = load(indices, category)
    
    # Tạo dict với task names
    y_dict = {}
    for i, task in enumerate(tasks):
        y_dict[task] = y_full[:, i]
    
    return X, y_dict


def load_ovr_labels(indices, category='train'):
    """
    Load chỉ labels OVR mà không load ảnh
    
    Returns:
        dict: {task_name: labels array [N]}
    """
    if category == 'train':
        h5_file = train_file
    elif category == 'val':
        h5_file = val_file
    elif category == 'test':
        h5_file = test_file
    else:
        raise ValueError(f"category phải là 'train', 'val' hoặc 'test', nhận: {category}")
    
    # Convert indices sang numpy array nếu cần
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    
    if indices.ndim == 0:
        indices = np.array([indices])
    
    if len(indices) > 0:
        sorted_idx = np.argsort(indices)
        sorted_indices = indices[sorted_idx]
        
        y_full = h5_file['labels'][sorted_indices]
        
        # Trả về theo thứ tự gốc
        unsort_idx = np.argsort(sorted_idx)
        y_full = y_full[unsort_idx]
    else:
        y_full = np.zeros((0, len(attributes)), dtype=np.int64)
    
    # Tạo dict với task names
    y_dict = {}
    for i, task in enumerate(tasks):
        y_dict[task] = y_full[:, i]
    
    return y_dict


if __name__ == "__main__":
    # Test
    print("\n" + "=" * 80)
    print("TEST DATALOADER")
    print("=" * 80)
    
    # Load 10 samples từ train
    print("\nLoading 10 train samples...")
    X, y_dict = load_ovr(np.arange(10), category='train')
    print(f"X shape: {X.shape}, dtype: {X.dtype}, min: {X.min():.3f}, max: {X.max():.3f}")
    print(f"Tasks: {list(y_dict.keys())}")
    print(f"Sample labels (first 3 samples):")
    for i in range(min(3, len(X))):
        print(f"  Sample {i}: {y_dict}")
    
    # Load chỉ labels
    print("\nLoading labels only...")
    y_labels = load_ovr_labels(np.arange(10), category='train')
    print(f"Label keys: {list(y_labels.keys())}")
    print(f"First task ({list(y_labels.keys())[0]}) shape: {y_labels[list(y_labels.keys())[0]].shape}")
