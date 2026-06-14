import torch
from torch.utils.data import Dataset


class MyMNIST(Dataset):
    """把 torchvision 的 MNIST 包装成自己写的 Dataset——理解 Dataset 的骨架"""

    def __init__(self, data, labels):
        self.data = data  # 所有图片
        self.labels = labels  # 所有标签

    def __len__(self):
        """必须实现：告诉 DataLoader 一共有多少样本"""
        return len(self.data)

    def __getitem__(self, idx):
        """必须实现：给定索引 idx，返回 (图片, 标签)"""
        return self.data[idx], self.labels[idx]


# 用刚才的 train_data 来测试
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(
    root=r"C:\Users\text\Desktop\vit-from-scratch\data",
    train=True,
    download=True,
    transform=transform
)

# 用我们自己写的 Dataset 包装
my_dataset = MyMNIST(train_data.data, train_data.targets)
print(f"自己写的 Dataset — 长度: {len(my_dataset)}")
print(f"第0个样本: shape={my_dataset[0][0].shape}, label={my_dataset[0][1]}")

# 放进 DataLoader
from torch.utils.data import DataLoader

loader = DataLoader(my_dataset, batch_size=32, shuffle=True)
imgs, lbls = next(iter(loader))
print(f"batch shape: {imgs.shape}")
print(f"labels: {lbls[:10]}")