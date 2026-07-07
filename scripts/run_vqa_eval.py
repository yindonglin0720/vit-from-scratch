"""批量 VQA 推理 + 结果保存 —— D5"""
import json
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os

# ============================================
# 🔴 必须手敲 第1段：加载模型（同 D2）
# ============================================
model_dir = r"C:\Users\text\.cache\modelscope\hub\models\Qwen\Qwen3-VL-4B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(model_dir)


# ============================================
# 🔴 必须手敲 第2段：批量推理函数
# ============================================
def run_vqa(json_path):
    """读取 vqa_test.json，逐条推理，返回结果列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    results = []
    for i, sample in enumerate(samples):
        img_path = os.path.join(r"C:\Users\text\Desktop\vit-from-scratch", sample["image"])
        question = sample["question"]

        try:
            img = Image.open(img_path).convert("RGB")
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

        except Exception as e:
            answer = f"[ERROR] {e}"

        results.append({
            "index": i,
            "image": sample["image"],
            "question": question,
            "reference_answer": sample["reference_answer"],
            "model_answer": answer,
        })

        if (i + 1) % 5 == 0:
            print(f"进度: {i+1}/{len(samples)}")

    return results


# ============================================
# 第3段：保存结果（理解即可，不需要手敲）
# ============================================
if __name__ == "__main__":
    json_path = r"C:\Users\text\Desktop\vit-from-scratch\data\vqa_test.json"
    results = run_vqa(json_path)

    # 保存 JSON
    out_json = r"C:\Users\text\Desktop\vit-from-scratch\outputs\vqa_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存 CSV（方便 Excel 打开看）
    out_csv = r"C:\Users\text\Desktop\vit-from-scratch\outputs\vqa_results.csv"
    with open(out_csv, "w", encoding="utf-8-sig") as f:
        f.write("index,image,question,reference_answer,model_answer\n")
        for r in results:
            ans = r["model_answer"].replace('"', '""').replace("\n", " ")
            f.write(f'{r["index"]},{r["image"]},"{r["question"]}",*{r["reference_answer"]}*,"{ans}"\n')

    correct = sum(1 for r in results if r["reference_answer"] in r["model_answer"])
    print(f"\n简单匹配准确率: {correct}/{len(results)}")