import torch
import torch.nn as nn
import torch.nn.functional as F


OVR_TASKS = [
    "gender_female",
    "gender_male",
    "age_bin0",
    "age_bin1",
    "age_bin2",
    "race_white",
    "race_black",
    "race_asian",
    "race_indian",
    "race_others",
]


class OVRModel(nn.Module):
    """
    Backbone CNN + 10 binary heads (one-vs-rest) cho UTKFace.

    - Dùng chung feature extractor + MLP ẩn
    - Mỗi head: Linear(128 -> 1), output là logit (chưa sigmoid)
    """

    def __init__(self, input_shape=(3, 64, 64), dropout_rate=0.3):
        super().__init__()

        c = input_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )

        self.flat_size = 128 * 8 * 8
        self.shared = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.heads = nn.ModuleDict(
            {name: nn.Linear(128, 1) for name in OVR_TASKS}
        )

    def _encode(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared(x)
        return x

    def forward(self, x):
        """
        Trả về dict: {task_name: logits (B,)}
        """
        h = self._encode(x)
        out = {}
        for name, head in self.heads.items():
            out[name] = head(h).squeeze(1)
        return out

    def forward_task(self, x, task):
        """
        Trả về logit (B,) cho một task cụ thể.
        """
        if task not in self.heads:
            raise ValueError(f"Unsupported OVR task: {task}")
        h = self._encode(x)
        return self.heads[task](h).squeeze(1)


if __name__ == "__main__":
    model = OVRModel(input_shape=(3, 64, 64))
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    print("OVRModel test:")
    for k, v in out.items():
        print(k, v.shape)

