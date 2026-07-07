"""图搜文 + Recall@K 评估：输入图片 → 找最匹配的描述"""
import torch
import torch.nn.functional as F
import open_clip
from transformers import BertConfig, BertModel, BertTokenizer
from PIL import Image
import os, sys
sys.path.append(r"C:\Users\text\Desktop\vit-from-scratch")
from scripts.clip_dataset import CLIPImageTextDataset

# ========== 1. 加载模型（同 D3/D5）==========
model_dir = r"C:\Users\text\.cache\modelscope\hub\models\iic\multi-modal_clip-vit-base-patch16_zh"
raw = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location='cpu')
state = {}
for k, v in raw['state_dict'].items():
    new_k = k.replace('module.', '') if k.startswith('module.') else k
    state[new_k] = v.clone()

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=None)
model.load_state_dict({k: v for k, v in state.items() if k.startswith('visual.')}, strict=False)
model.eval()

bert_tokenizer = BertTokenizer(os.path.join(model_dir, "vocab.txt"))
bert = BertModel(BertConfig(vocab_size=21128, hidden_size=768, num_hidden_layers=12,
                              num_attention_heads=12, intermediate_size=3072,
                              max_position_embeddings=512))
bert.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith('bert.')}, strict=False)
bert.eval()

text_proj = state['text_projection']
logit_scale = state['logit_scale']

# ========== 2. 加载数据集 + 预计算文本向量 ==========
ds = CLIPImageTextDataset(r"C:\Users\text\Desktop\vit-from-scratch\data\clip_images", transform=preprocess)
print(f"数据集: {len(ds)} 张图片")

all_captions = [ds.samples[i][1] for i in range(len(ds))]
unique_captions = list(set(all_captions))  # 去重（120张图但只有30种描述）
print(f"唯一描述数: {len(unique_captions)}")

txt_feats = []
with torch.no_grad():
    for cap in unique_captions:
        tokens = bert_tokenizer([cap], padding=True, return_tensors='pt')
        out = bert(**tokens).last_hidden_state[:, 0, :]
        out = F.normalize(out @ text_proj, dim=-1)
        txt_feats.append(out)
txt_feats = torch.cat(txt_feats, dim=0)   # [30, 512]

# ========== 3. 图搜文 ==========
def image_to_text(img_path, top_k=5):
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        img_feat = F.normalize(model.encode_image(img_tensor), dim=-1)
    similarity = (img_feat @ txt_feats.T) * logit_scale.exp()
    scores, indices = similarity[0].topk(top_k)

    print(f"\n图片: {os.path.basename(img_path)}（正确答案: {ds.samples[0][1]}）" if False else f"\n图片: {os.path.basename(img_path)}")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"  #{rank} [{score:.2f}] {unique_captions[idx]}")
    return indices, scores

# ========== 4. Recall@K 评估 ==========
print("\n=== Recall@K 评估 ===")
recall1, recall5, recall10 = 0, 0, 0
total = len(ds)

for i in range(total):
    img_tensor, true_cap = ds[i]
    with torch.no_grad():
        img_feat = F.normalize(model.encode_image(img_tensor.unsqueeze(0)), dim=-1)
    similarity = (img_feat @ txt_feats.T) * logit_scale.exp()
    _, indices = similarity[0].topk(10)
    top_caps = [unique_captions[idx] for idx in indices]

    if true_cap in top_caps[:1]:  recall1 += 1
    if true_cap in top_caps[:5]:  recall5 += 1
    if true_cap in top_caps[:10]: recall10 += 1

print(f"Recall@1:  {recall1/total*100:.1f}%")
print(f"Recall@5:  {recall5/total*100:.1f}%")
print(f"Recall@10: {recall10/total*100:.1f}%")

# 展示 3 个样例
for i in [0, 40, 80]:
    image_to_text(ds.samples[i][0], top_k=5)
