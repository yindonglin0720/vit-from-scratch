项目名：vit-from-scratch
简洁：PyTorch 深度学习工程实践——从环境配置到图像分类模型完整训练，包含 MNIST/CIFAR10 实验。


环境：Python 3.10.20
PyTorch 2.11.0+cu128
CUDA 12.8
GPU: NVIDIA GeForce RTX 5070 Laptop GPU



目录结构：
vit-from-scratch/
├── data/        # 数据集文件
├── models/      # 模型定义
├── scripts/     # 训练、评估脚本
├── utils/       # 工具函数
├── env_report.txt  # 环境验证记录
└── hello_torch.py  # 环境检测入口


## 运行方式

# 1. 激活环境
conda activate pytorch

# 2. 验证环境
python hello_torch.py