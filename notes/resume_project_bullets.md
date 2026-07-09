# 简历项目描述 — 多模态图像问答系统

## 项目名
基于 Qwen3-VL 的图像问答系统构建、评测与 LoRA 微调

---

## 简洁版（海投用，3 条 bullet）

- 基于 Qwen3-VL-4B-Instruct 构建多模态 VQA 系统，设计覆盖颜色形状、数量判断、OCR、空间关系、物体边界 6 类共 84 条测试样本，实现批量推理与三层评分体系（exact_match/keyword_match/human_score），零样本 human_score 92.3%
- 使用 PEFT 框架对 Qwen3-VL 进行 LoRA 微调，rank=8 仅训练 0.1% 参数，完成数据格式转换、训练/验证拆分、adapter 保存与加载推理全链路，train_loss 从 21 降至 15
- 独立完成 CLIP 双塔图文检索（Recall@5 100%）与 VLM 问答系统从数据处理到模型部署的全流程，在 RTX 5070 8GB 笔记本上完成 4B 模型推理与微调

---

## 详细版（对口岗位用，3 条 bullet + 补充）

- 基于 Qwen3-VL-4B-Instruct 构建多模态 VQA 评测系统：设计 6 类 84 条测试样本（颜色形状/数量/OCR/空间关系/物体边界），实现三层评分（exact_match 61.9% → keyword_match 77.4% → human_score 92.3%），按类型分项统计并完成错误归因分析
- 使用 PEFT + Transformers Trainer 实现 Qwen3-VL 的 LoRA 微调：LoraConfig(r=8, alpha=16) 仅训练 486 万参数（0.1%），完成 58 条 SFT 数据格式转换、train/val/test 拆分、adapter 保存与加载推理，loss 从 21.35 收敛至 15.55
- 复现 CLIP 双塔图文检索（中文 ViT-B/16 + BERT），构造 120 张合成数据集，实现文本搜图和图搜文双向检索，Recall@5 达 100%

**补充描述**（面试展开用）：
- 模型加载突破：绕过 ModelScope pipeline 依赖爆炸，直接 torch.load + open_clip + transformers 手动拆解 state_dict；修复 CLS token/pooler_output 导致相似度值全为 0.02 的 bug
- 工程限制与应对：8GB 显存使用 FP16 推理 4B 模型，LoRA 训练使用 gradient_checkpointing_enable 和 per_device_batch_size=1；LLaMA-Factory 因网络问题未安装后改用 PEFT 直接实现训练循环
- 诚实复盘：LoRA 后测试集 exact_match 与 zero-shot 持平（8/13 vs 8/13），因为 58 条训练数据不足。定位瓶颈在数据量而非方法，下一步方向清晰