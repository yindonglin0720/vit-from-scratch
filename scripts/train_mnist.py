import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.append(r"C:\Users\text\Desktop\vit-from-scratch")
from models.mnist_model import MNISTModel

# ====== 1. 准备数据 ======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(
    root=r"C:\Users\text\Desktop\vit-from-scratch\data",
    train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root=r"C:\Users\text\Desktop\vit-from-scratch\data",
    train=False, download=True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ====== 2. 初始化模型 ======
model = MNISTModel()
print(model)

# ====== 3. loss 和 optimizer ======
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ====== 4. 训练 + 验证 ======
epochs = 5
best_acc = 0.0

for epoch in range(epochs):
    # —— 训练 ——
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

    # —— 验证 ——
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    # —— 保存最佳 ——
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), r"C:\Users\text\Desktop\vit-from-scratch\models\best_mnist.pth")
        print(f"  -> 保存最佳模型 (Acc: {acc:.2f}%)")

print("训练完成！")