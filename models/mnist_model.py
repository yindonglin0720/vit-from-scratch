import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)   # 输入784 → 隐藏层128
        self.fc2 = nn.Linear(128, 10)       # 隐藏层128 → 输出10（0~9共10类）
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)           # 把图片展平：[B,1,28,28] → [B,784]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x