"""same test set 上 zero-shot vs LoRA 对比 —— D2"""
import json

# 读取
with open(r"C:\Users\text\Desktop\vit-from-scratch\outputs\vqa_results.json", "r", encoding="utf-8") as f:
    zero = json.load(f)
with open(r"C:\Users\text\Desktop\vit-from-scratch\outputs\lora_results.json", "r", encoding="utf-8") as f:
    lora = json.load(f)

# 用 (image, question) 做 key 匹配
zero_dict = {(z["image"], z["question"]): z for z in zero}

print("| # | 图片 | 问题 | 参考答案 | zero-shot | LoRA |")
print("|---|---|---|---|---|---|")
for i, l in enumerate(lora):
    key = (l["image"], l["question"])
    z = zero_dict.get(key)
    z_ans = z["model_answer"][:80].replace("\n", " ") if z else "未找到"
    l_ans = l["model_answer"][:80].replace("\n", " ")
    ref = l["reference_answer"]
    img = l["image"].replace("data/clip_images/", "").replace("data/vqa_images/", "")
    print(f"| {i+1} | {img} | {l['question']} | {ref} | {z_ans}... | {l_ans}... |")

# 简单匹配对比
z_correct = sum(1 for l in lora if (k := (l["image"], l["question"])) in zero_dict and zero_dict[k]["reference_answer"] in zero_dict[k]["model_answer"])
l_correct = sum(1 for l in lora if l["reference_answer"] in l["model_answer"])
print(f"\nzero-shot exact_match: {z_correct}/{len(lora)}")
print(f"LoRA     exact_match: {l_correct}/{len(lora)}")