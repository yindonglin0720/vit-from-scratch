"""CLIP 图文数据集 —— 加载图片+中文描述"""
import os
import csv
from PIL import Image
from torch.utils.data import Dataset


class CLIPImageTextDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: 图片文件夹路径（里面有 metadata.csv 和图片）
        transform: 图片预处理（CLIP 的 preprocess）
        """
        self.data_dir = data_dir
        self.transform = transform

        # 读 metadata.csv
        csv_path = os.path.join(data_dir, "metadata.csv")
        self.samples = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(data_dir, row["filename"])
                caption = row["caption"]
                self.samples.append((img_path, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, caption
