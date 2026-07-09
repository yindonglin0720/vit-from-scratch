"""VQA 数据 → LLaMA-Factory SFT 格式 + train/val/test 拆分 —— D5"""
import json
import os
import random

random.seed(42)

# 读取
vqa_path = r"C:\Users\text\Desktop\vit-from-scratch\data\vqa_test.json"
with open(vqa_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

# 转 SFT 格式
sft_data = []
for s in samples:
    item = {
        "messages": [
            {"role": "user", "content": f"<image>{s['question']}"},
            {"role": "assistant", "content": s["reference_answer"]},
        ],
        "images": [s["image"]]
    }
    sft_data.append(item)

# 70/15/15 拆分
random.shuffle(sft_data)
n = len(sft_data)
train = sft_data[:int(n * 0.7)]
val = sft_data[int(n * 0.7):int(n * 0.85)]
test = sft_data[int(n * 0.85):]

# 保存
out_dir = r"C:\Users\text\Desktop\vit-from-scratch\data\sft"
os.makedirs(out_dir, exist_ok=True)

for name, data in [("train", train), ("val", val), ("test", test)]:
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print(f"train: {len(train)} 条 → data/sft/train.json")
print(f"val:   {len(val)} 条 → data/sft/val.json")
print(f"test:  {len(test)} 条 → data/sft/test.json")