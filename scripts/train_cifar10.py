import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pickle
import numpy as np
import sys
sys.path.append(r"C:\Users\text\Desktop\vit-from-scratch")
from models.cifar10_model import CIFAR10CNN

# ====== 自定义 CIFAR10 Dataset ======
class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        path = root + r"\cifar-10-batches-py"
        if train:
            files = [f"{path}\data_batch_{i}" for i in range(1, 6)]
        else:
            files = [f"{path}\\test_batch"]
        self.data, self.labels = [], []
        for f in files:
            d = pickle.load(open(f, "rb"))
            self.data.append(d["data"])
            self.labels.append(d["labels"])
        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32)
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(img)
        return img, self.labels[idx]

# ====== 数据加载 ======
train_data = CIFAR10Dataset(r"C:\Users\text\Desktop\vit-from-scratch\data", train=True)
test_data = CIFAR10Dataset(r"C:\Users\text\Desktop\vit-from-scratch\data", train=False)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
print(f"训练: {len(train_data)}, 测试: {len(test_data)}")

imgs, lbls = next(iter(train_loader))
print(f"batch shape: {imgs.shape}, labels: {lbls[:5]}")


# ====== 初始化模型 ======
model = CIFAR10CNN()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ====== 训练 ======
epochs = 10
best_acc = 0.0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), r"C:\Users\text\Desktop\vit-from-scratch\models\best_cifar10.pth")
        print(f"  -> 保存最佳模型 (Acc: {acc:.2f}%)")

print(f"训练完成！最佳准确率: {best_acc:.2f}%")