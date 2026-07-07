"""CLIP 图文相似度推理 — 中文模型：图片塔 open_clip + 文本塔 BERT（正确分词器）"""
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import open_clip
from transformers import BertConfig, BertModel, BertTokenizer
import os

model_dir = r"C:\Users\text\.cache\modelscope\hub\models\iic\multi-modal_clip-vit-base-patch16_zh"

# ========== 1. 加载所有权重 ==========
raw = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location='cpu')
state = {}
for k, v in raw['state_dict'].items():
    new_k = k.replace('module.', '') if k.startswith('module.') else k
    state[new_k] = v.clone()

# ========== 2. 图片塔 ==========
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=None)
vis_state = {k: v for k, v in state.items() if k.startswith('visual.')}
model.load_state_dict(vis_state, strict=False)
model.eval()

# ========== 3. 文本塔：BERT + 中文分词器 ==========
bert_tokenizer = BertTokenizer(os.path.join(model_dir, "vocab.txt"))
bert_config = BertConfig(
    vocab_size=21128, hidden_size=768, num_hidden_layers=12,
    num_attention_heads=12, intermediate_size=3072,
    max_position_embeddings=512,
)
bert = BertModel(bert_config)
bert_state = {}
for k, v in state.items():
    if k.startswith('bert.'):
        bert_state[k[5:]] = v
bert.load_state_dict(bert_state, strict=False)
bert.eval()

text_proj = state['text_projection']   # [768, 512]
logit_scale = state['logit_scale']

# ========== 4. 生成测试图（有纹理的橘色区域）==========
img = Image.new('RGB', (224, 224), color=(255, 100, 50))
draw = ImageDraw.Draw(img)
for i in range(0, 224, 20):
    draw.rectangle([i, 0, i+10, 224], fill=(255, 140, 70))
img.save(r"C:\Users\text\Desktop\vit-from-scratch\test_img.jpg")

# ========== 5. 推理 ==========
img_tensor = preprocess(img).unsqueeze(0)
texts = ["一个桔色的方块", "蓝色海洋波浪", "绿色森林树木", "一只黑白色的猫"]

with torch.no_grad():
    img_feat = model.encode_image(img_tensor)

    # 用 BERT 中文分词器编码文本
    txt_inputs = bert_tokenizer(texts, padding=True, return_tensors='pt')
    txt_out = bert(**txt_inputs).last_hidden_state[:, 0, :]  # CLS token [B, 768]
    txt_feat = txt_out @ text_proj                           # [B, 512]

img_feat = F.normalize(img_feat, dim=-1)
txt_feat = F.normalize(txt_feat, dim=-1)
similarity = img_feat @ txt_feat.T * logit_scale.exp()

print("\n=== 相似度结果 ===")
for i, s in enumerate(similarity[0]):
    print(f"  {s.item():.4f} | {texts[i]}")
