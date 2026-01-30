### TẢI TẬP DỮ LIỆU CIFAR-10

import torchvision
import torchvision.transforms as transforms

trainset = torchvision.datasets.CIFAR10(
    root='./datasets/CIFAR-10/', # Thư mục lưu (tạo nếu chưa có)
    train=True, # Train set (50k ảnh)
    download=True,      
    transform=transforms.ToTensor()
)

testset = torchvision.datasets.CIFAR10(
    root='./datasets/CIFAR-10/',
    train=False, # Test set (10k ảnh)
    download=True,
    transform=transforms.ToTensor()
)

