# vit-from-scratch

PyTorch 深度学习工程实践第 1 周——从环境配置到 CNN 图像分类完整训练。

## 环境

- Python 3.10.20，PyTorch 2.11.0+cu128，CUDA 12.8
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU

## 目录结构
├── data/          # 数据集（MNIST / CIFAR10）
├── models/        # 模型定义 + 训练好的权重
├── scripts/       # 训练、验证、Dataset 脚本
├── utils/         # 训练曲线、工具函数
├── hello_torch.py # 环境检测入口
└── env_report.txt # 环境验证记录

## Week 1 实验结果

### MNIST (MLP)

| 模型 | 准确率 |
|------|--------|
| 2层全连接 (784→128→10) | 96.91% |

### CIFAR10 (CNN)

| 实验 | 配置 | 准确率 |
|------|------|--------|
| baseline | CNN, 无增强, 无 BN, 10 epochs | 79.07% |
| 升级版 | +数据增强 +BatchNorm +GPU, 25 epochs | **82.55%** |

![训练曲线](utils/training_curve.png)

## 运行方式

```bash
conda activate pytorch
python hello_torch.py
python scripts/train_mnist.py
python scripts/train_cifar10.py
```