import torch
import torch.nn as nn

class CelebAMultiTaskModel(nn.Module):
    """
    Multitask Model cho CelebA - 27 attributes (binary classification)
    1 backbone + 27 heads riêng biệt
    """
    def __init__(self, input_shape=(3, 64, 64), dropout_rate=0.3, num_attributes=27):
        super().__init__()
        
        channels = input_shape[0]
        
        # Backbone feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flat_size = 256

        # Shared layers
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # 27 independent heads (binary classification)
        self.heads = nn.ModuleList([nn.Linear(256, 2) for _ in range(num_attributes)])

    def _encode(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)   # flatten
        x = self.shared_fc(x)
        return x

    def forward(self, x):
        h = self._encode(x)
        outputs = {}
        for i in range(len(self.heads)):
            outputs[f"attr_{i}"] = self.heads[i](h)
        return outputs

    def forward_single(self, x, attr_idx):
        """Forward cho 1 attribute cụ thể (dùng khi cần)"""
        h = self._encode(x)
        return self.heads[attr_idx](h)


# Test model
if __name__ == "__main__":
    model = CelebAMultiTaskModel(input_shape=(3, 64, 64), num_attributes=27)
    dummy_input = torch.randn(4, 3, 64, 64)
    outputs = model(dummy_input)
    print("Output keys:", list(outputs.keys()))
    print("One head output shape:", outputs["attr_0"].shape)  # torch.Size([4, 2])