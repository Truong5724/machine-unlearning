import torch
import torch.nn as nn
import torch.nn.functional as F


OVR_TASKS = [
    "male",
    "young",
    "smiling",
    "mouth_slightly_open",
    "big_lips",
    "big_nose",
    "pointy_nose",
    "high_cheekbones",
    "oval_face",
    "wavy_hair",
    "straight_hair",
    "bangs",
    "receding_hairline",
    "black_hair",
    "blond_hair",
    "brown_hair",
    "eyeglasses",
    "bushy_eyebrows",
    "arched_eyebrows",
    "bags_under_eyes",
    "chubby",
    "double_chin",
    "wearing_earrings",
    "wearing_necklace",
    "mustache",
    "goatee",
    "sideburns",
]


class BasicBlock(nn.Module):
    """Basic residual block dùng cho backbone ResNet-style."""

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate and dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class OVRModel(nn.Module):
    """
    Backbone ResNet-style + 27 binary heads (one-vs-rest) cho CelebA.

    - Dùng chung backbone, sau đó MLP nhúng -> 128-dim embedding
    - Mỗi head: Linear(128 -> 1), output là logit (chưa sigmoid)
    """

    def __init__(self, input_shape=(3, 64, 64), dropout_rate=0.3):
        super().__init__()

        c = input_shape[0]
        self.dropout_rate = float(dropout_rate)

        # 64x64 -> 32x32
        self.conv1 = nn.Conv2d(c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 32x32 -> 16x16
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 16x16 -> 16x16 -> 8x8 -> 4x4
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 256 -> 128 embedding
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )

        self.heads = nn.ModuleDict({name: nn.Linear(128, 1) for name in OVR_TASKS})

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(
            BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                dropout_rate=self.dropout_rate * 0.25,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    dropout_rate=self.dropout_rate * 0.25,
                )
            )
        return nn.Sequential(*layers)

    def _encode(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.embed(x)
        return x

    def forward(self, x):
        """Trả về dict: {task_name: logits (B,)}"""
        h = self._encode(x)
        return {name: head(h).squeeze(1) for name, head in self.heads.items()}

    def forward_task(self, x, task):
        """
        Trả về logit (B,) cho một task cụ thể.
        """
        if task not in self.heads:
            raise ValueError(f"Unsupported OVR task: {task}")
        h = self._encode(x)
        return self.heads[task](h).squeeze(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = OVRModel(input_shape=(3, 64, 64))
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    print("CelebA OVRModel test:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
    print(f"\nTotal tasks: {len(out)}")
