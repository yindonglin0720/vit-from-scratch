"""加载 LoRA adapter 跑测试集推理 —— D7"""
import json, os, torch
from PIL import Image
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_dir = r"C:\Users\text\.cache\modelscope\hub\models\Qwen\Qwen3-VL-4B-Instruct"
adapter_dir = r"C:\Users\text\Desktop\vit-from-scratch\outputs\lora-adapter"
img_root = r"C:\Users\text\Desktop\vit-from-scratch"

# 加载基础模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map={"": 0}
)
# 挂上 LoRA adapter
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()
processor = AutoProcessor.from_pretrained(model_dir)

# 读取测试集
test_path = r"C:\Users\text\Desktop\vit-from-scratch\data\sft\test.json"
with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

results = []
for i, item in enumerate(test_data):
    img = Image.open(os.path.join(img_root, item["images"][0])).convert("RGB")
    question = item["messages"][0]["content"].replace("<image>", "").strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    results.append({
        "image": item["images"][0],
        "question": question,
        "reference_answer": item["messages"][1]["content"],
        "model_answer": answer,
    })

    if (i + 1) % 3 == 0:
        print(f"进度: {i+1}/{len(test_data)}")

# 保存
out_path = r"C:\Users\text\Desktop\vit-from-scratch\outputs\lora_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 简单匹配
correct = sum(1 for r in results if r["reference_answer"] in r["model_answer"])
print(f"\nLoRA 模型 exact_match: {correct}/{len(results)}")
print(f"结果保存到: {out_path}")