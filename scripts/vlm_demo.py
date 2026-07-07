"""Qwen3-VL 单图问答 Demo —— D2"""
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# 🔴 必须手敲 第1段：加载模型和处理器
model_dir = r"C:\Users\text\.cache\modelscope\hub\models\Qwen\Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

processor = AutoProcessor.from_pretrained(model_dir)

# ============================================
# 🔴 必须手敲 第2段：准备测试数据
# ============================================
# 复用 clip_images 的合成图 + 再加真实图更好
test_images = [
    r"C:\Users\text\Desktop\vit-from-scratch\data\clip_images\0001.jpg",  # 红色方块
    r"C:\Users\text\Desktop\vit-from-scratch\data\clip_images\0011.jpg",  # 蓝色方块
]

test_questions = [
    "这是什么形状？什么颜色？",
    "描述一下这张图片的内容。",
    "图片里有几个物体？",
]

# ============================================
# 🔴 必须手敲 第3段：单图问答函数
# ============================================
def ask(image_path, question):
    """给一张图和一个问题，返回模型的文字回答"""
    img = Image.open(image_path).convert("RGB")

    # messages 格式是 Qwen3-VL 的规范
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ]
        }
    ]

    # processor.apply_chat_template 是核心：图文 → token 序列
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # 移到 GPU（Qwen3-VL 不会自动搬，必须手动）
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 自回归生成
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # 只取新生成的部分（刨掉输入的 token）
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return answer


# ============================================
# 第4段：跑测试（不需要手敲，理解即可）
# ============================================
if __name__ == "__main__":
    for img_path in test_images:
        print(f"\n{'='*50}")
        print(f"图片: {img_path}")
        for q in test_questions:
            ans = ask(img_path, q)
            print(f"  问题: {q}")
            print(f"  回答: {ans}")