# CLIP 图文检索 baseline

> 第 2 周多模态入门项目——CLIP 中文模型本地推理 + 图文检索 + Recall@K 评估

## 任务

用中文 CLIP 模型实现文本搜图和图搜文，在 120 组合成图文数据上评估检索效果。

## 环境

- Python 3.10.20 / PyTorch 2.11.0+cu128 / CUDA 12.8
- GPU: RTX 5070 Laptop
- 模型：ModelScope 中文 CLIP (ViT-B/16 + BERT-base)

## 目录结构
scripts/
├── clip_demo.py         # D3: CLIP 单次推理验证
├── gen_clip_dataset.py  # D4: 生成120组合成图文
├── clip_dataset.py      # D4: Dataset 类
├── text_to_image.py     # D5: 文本搜图
└── image_to_text.py     # D6: 图搜文 + Recall@K
data/clip_images/         # 120张图片 + metadata.csv
notes/clip_principle.txt  # D2: CLIP 原理笔记
## 实验结果

### 文本搜图 (D5)

| 查询 | Top-1 结果 | 分数 |
|------|-----------|------|
| 一个蓝色的方块 | 蓝色的方块 | 44.37 |
| 一个红色的圆形 | 红色的圆形 | 46.02 |
| 一个绿色的三角形 | 绿色的三角形 | 50.86 |

### 图搜文 + Recall@K (D6)

| 指标 | 结果 |
|------|------|
| Recall@1 | 93.3% |
| Recall@5 | 100.0% |
| Recall@10 | 100.0% |

## 运行方式

```bash
conda activate pytorch
python scripts/clip_demo.py        # 单次推理测试
python scripts/gen_clip_dataset.py # 生成数据集
python scripts/text_to_image.py    # 文本搜图
python scripts/image_to_text.py    # 图搜文 + Recall@K
```

## 关键收获

1. CLIP = 图片塔(ViT) + 文本塔(BERT) + 共享投影空间
2. 中文模型需注意词表/维度与标准CLIP的差异
3. 图文检索的评估指标用 Recall@K 而非准确率