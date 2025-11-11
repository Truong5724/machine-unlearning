# import torchvision
# import torchvision.transforms as transforms

# # Tự động tải + lưu vào thư mục ./data
# trainset = torchvision.datasets.CIFAR10(
#     root='./datasets/CIFAR-10/',      # Thư mục lưu (tạo nếu chưa có)
#     train=True,         # True = train set (50k ảnh)
#     download=True,      # TỰ ĐỘNG TẢI
#     transform=transforms.ToTensor()
# )

# testset = torchvision.datasets.CIFAR10(
#     root='./datasets/CIFAR-10/',
#     train=False,        # False = test set (10k ảnh)
#     download=True,
#     transform=transforms.ToTensor()
# )
# mở 1 vài ảnh để kiểm tra trong data_batch_1
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ===== Hàm đọc 1 batch =====
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']       # dạng (10000, 3072)
        labels = batch[b'labels']   # danh sách nhãn

        # reshape (10000, 3, 32, 32) → (10000, 32, 32, 3)
        data = data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
        return data, labels

# ===== Đọc batch 1 =====
images, labels = load_cifar_batch('datasets/CIFAR-10/cifar-10-batches-py/data_batch_1')

print("Kích thước ảnh:", images.shape)
print("Số nhãn:", len(labels))

# ===== Hiển thị 1 ảnh ngẫu nhiên =====
idx = 10
plt.imshow(images[idx])
plt.title(f"Label: {labels[idx]}")
plt.show()


