import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root = r"C:\Users\text\Desktop\vit-from-scratch\data",train=True,download=True,transform=transform)

print(f"训练样本数: {len(train_data)}")
print(f"单张图片 shape: {train_data[0][0].shape}")
print(f"标签: {train_data[0][1]}")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
images, labels = next(iter(train_loader))

print("=== 第一个 batch ===")
print(f"images shape: {images.shape}")
print(f"labels shape: {labels.shape}")
print(f"labels（前10个）: {labels[:10]}")