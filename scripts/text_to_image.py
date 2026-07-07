"""文本搜图：输入描述 → 返回 Top-K 最匹配的图片"""
import torch
import torch.nn.functional as F
import open_clip
from transformers import BertConfig, BertModel, BertTokenizer
from PIL import Image
import os

# ========== 1. 加载模型（和 D3 一样）==========
model_dir = r"C:\Users\text\.cache\modelscope\hub\models\iic\multi-modal_clip-vit-base-patch16_zh"
raw = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location='cpu')
state = {}
for k, v in raw['state_dict'].items():
    new_k = k.replace('module.', '') if k.startswith('module.') else k
    state[new_k] = v.clone()

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=None)
vis_state = {k: v for k, v in state.items() if k.startswith('visual.')}
model.load_state_dict(vis_state, strict=False)
model.eval()

bert_tokenizer = BertTokenizer(os.path.join(model_dir, "vocab.txt"))
bert = BertModel(BertConfig(vocab_size=21128, hidden_size=768, num_hidden_layers=12,
                              num_attention_heads=12, intermediate_size=3072,
                              max_position_embeddings=512))
bert_state = {k[5:]: v for k, v in state.items() if k.startswith('bert.')}
bert.load_state_dict(bert_state, strict=False)
bert.eval()

text_proj = state['text_projection']
logit_scale = state['logit_scale']

# ========== 2. 加载数据集 ==========
import sys
sys.path.append(r"C:\Users\text\Desktop\vit-from-scratch")
from scripts.clip_dataset import CLIPImageTextDataset

data_dir = r"C:\Users\text\Desktop\vit-from-scratch\data\clip_images"
ds = CLIPImageTextDataset(data_dir, transform=preprocess)
print(f"数据集: {len(ds)} 张图片")

# ========== 3. 预计算所有图片的向量 ==========
print("正在计算图片向量...")
all_img_feats = []
all_paths = []
with torch.no_grad():
    for i in range(len(ds)):
        img, _ = ds[i]
        feat = model.encode_image(img.unsqueeze(0))
        feat = F.normalize(feat, dim=-1)
        all_img_feats.append(feat)
        all_paths.append(ds.samples[i][0])

all_img_feats = torch.cat(all_img_feats, dim=0)  # [120, 512]
print(f"图片特征矩阵: {all_img_feats.shape}")

# ========== 4. 文本搜图 ==========
def search(query_text, top_k=5):
    tokens = bert_tokenizer([query_text], padding=True, return_tensors='pt')
    with torch.no_grad():
        txt_feat = bert(**tokens).last_hidden_state[:, 0, :]
        txt_feat = (txt_feat @ text_proj)
        txt_feat = F.normalize(txt_feat, dim=-1)
    similarity = (txt_feat @ all_img_feats.T) * logit_scale.exp()  # [1, 120]
    scores, indices = similarity[0].topk(top_k)

    print(f"\n查询: {query_text}")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        path = all_paths[idx]
        cap = ds.samples[idx][1]
        print(f"  #{rank} [{score:.2f}] {cap}  ({os.path.basename(path)})")
    return indices, scores

# 测试三条查询
search("一个蓝色的方块")
search("一个红色的圆形")
search("一个绿色的三角形")
