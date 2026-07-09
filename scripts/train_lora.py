"""Qwen3-VL-4B LoRA 微调 —— D6"""
import json, os, torch
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ============================================
# 🔴 必须手敲 第1段：数据集类
# ============================================
class VQADataset(Dataset):
    def __init__(self, json_path, processor, img_root):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor
        self.img_root = img_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(os.path.join(self.img_root, item["images"][0])).convert("RGB")
        question = item["messages"][0]["content"].replace("<image>", "").strip()
        answer = item["messages"][1]["content"]

        # 用 PIL Image 而非 "<image>" 字符串，processor 才能识别
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ]
            },
            {
                "role": "assistant",
                "content": answer,
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


# ============================================
# 🔴 必须手敲 第2段：LoRA 配置
# ============================================
model_dir = r"C:\Users\text\.cache\modelscope\hub\models\Qwen\Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map={"": 0}
)
model.gradient_checkpointing_enable()  # 省显存
model.config.use_cache = False         # 训练时必须关

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,               # rank，越大越强越费显存
    lora_alpha=16,     # 缩放因子
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "o_proj"],  # 只对 Q/V/O 加 LoRA
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(model_dir)


# ============================================
# 第3段：加载数据并训练（理解即可）
# ============================================
data_root = r"C:\Users\text\Desktop\vit-from-scratch"
train_ds = VQADataset(r"C:\Users\text\Desktop\vit-from-scratch\data\sft\train.json", processor, data_root)
val_ds = VQADataset(r"C:\Users\text\Desktop\vit-from-scratch\data\sft\val.json", processor, data_root)

training_args = TrainingArguments(
    output_dir=r"C:\Users\text\Desktop\vit-from-scratch\outputs\lora-checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=1,       # 8GB 只能 batch=1
    gradient_accumulation_steps=4,       # 等效 batch=4
    per_device_eval_batch_size=1,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=5,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=1,
    remove_unused_columns=False,
    report_to="none",
)

# ============================================
# 第3段：加载数据并训练
# ============================================
data_root = r"C:\Users\text\Desktop\vit-from-scratch"
train_ds = VQADataset(r"C:\Users\text\Desktop\vit-from-scratch\data\sft\train.json", processor, data_root)
val_ds = VQADataset(r"C:\Users\text\Desktop\vit-from-scratch\data\sft\val.json", processor, data_root)

def collate_fn(batch):
    result = {}
    keys_to_pad = {"input_ids", "labels", "attention_mask"}
    keys_to_stack = {"image_grid_thw", "pixel_values"}

    for key in batch[0].keys():
        values = [b[key] for b in batch]
        if key in keys_to_pad:
            result[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
        elif key in keys_to_stack:
            result[key] = torch.stack(values, dim=0) if values[0].dim() > 0 else torch.stack(values)
        else:
            try:
                result[key] = torch.stack(values, dim=0) if values[0].dim() > 0 else torch.stack(values)
            except:
                pass
    return result

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collate_fn,
)

trainer.train()

adapter_dir = r"C:\Users\text\Desktop\vit-from-scratch\outputs\lora-adapter"
model.save_pretrained(adapter_dir)
print(f"Adapter 保存到: {adapter_dir}")