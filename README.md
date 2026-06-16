# vit-from-scratch

> 第 1 周 PyTorch 工程基础闭环项目 —— 从环境配置、数据集加载、模型搭建到完整训练、验证、调参与结果可视化的全链路实践。

PyTorch 深度学习工程实践第 1 周——从环境配置到 CNN 图像分类完整训练。

## 项目背景

本项目是我多模态大模型学习路线第 1 周的工程基础成果。目标不是追求 SOTA 准确率，而是在 MNIST 和 CIFAR10 两个数据集上独立完成数据加载、模型搭建、训练循环、验证评估、调参对比和结果整理的全流程，建立「拿到一个新任务能从零写到交付」的工程能力。

## 技术栈

`PyTorch 2.11` `CUDA 12.8` `Python 3.10` `CNN` `MLP` `BatchNorm` `Data Augmentation` `Git`

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

## 简历 Bullet 初稿

- 独立搭建 PyTorch 工程环境，完成 MNIST/CIFAR10 两个数据集的网络设计、训练调试与模型部署，MNIST 准确率 96.91%，CIFAR10 准确率 82.55%
- 掌握 CNN/MLP 模型搭建、数据增强（RandomFlip/RandomCrop）、BatchNorm 正则化、GPU 加速训练（RTX 5070）等深度学习工程实践
- 建立规范化 Git 工作流（12+ commits），编写具备项目背景、实验对比表、训练曲线和可复现运行命令的 README 文档