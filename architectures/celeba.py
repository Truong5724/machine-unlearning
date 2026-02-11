import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Model(nn.Module):
    """Kiến trúc CNN cho CelebA (ảnh 64x64x3, phân loại nhị phân)
    
    Kiến trúc ResNet-style với skip connections, tối ưu cho CelebA.
    Đây là kiến trúc phổ biến và hiệu quả cho bài toán phân loại thuộc tính khuôn mặt.
    
    Input shape mong đợi: (batch, 3, 64, 64)
    Output: (batch, nb_classes) - logits cho binary classification
    """
    def __init__(self, input_shape, nb_classes, dropout_rate=0.3, *args, **kwargs):
        super(Model, self).__init__()
        
        # Convolution ban đầu: 64x64 -> 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Các lớp residual blocks
        # 16x16 -> 16x16 (layer1)
        # 16x16 -> 8x8 (layer2)
        # 8x8 -> 4x4 (layer3)
        # 4x4 -> 2x2 (layer4)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling và classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, nb_classes)
        
        # Khởi tạo trọng số
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """Tạo một lớp gồm nhiều residual blocks"""
        layers = []
        # Block đầu tiên có thể có stride để downsampling
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Các block còn lại
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Khởi tạo trọng số theo He initialization (tốt cho ReLU)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv + pooling: 64x64 -> 16x16
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Residual layers
        x = self.layer1(x)  # 16x16
        x = self.layer2(x)  # 8x8
        x = self.layer3(x)  # 4x4
        x = self.layer4(x)  # 2x2
        
        # Global pooling và classification
        x = self.avgpool(x)  # 1x1
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Residual block cơ bản với skip connection
    
    Sử dụng pre-activation pattern: BN -> ReLU -> Conv
    Giúp gradient flow tốt hơn trong quá trình training.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (identity hoặc projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Pre-activation: BN -> ReLU -> Conv
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out
