import torch.nn as nn

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取部分（卷积块）
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 输入3通道RGB → 32个特征图，3×3卷积
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 32×32 → 16×16
            nn.Dropout(0.25)                   # 随机丢弃25%
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # 32 → 64通道
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 16×16 → 8×8
            nn.Dropout(0.25)
        )
        # 分类头（全连接）
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 展平 64×8×8 = 4096
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x
