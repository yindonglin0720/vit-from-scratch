1. LoRA 不是重新训练整个模型，而是在 Attn 里挂一段额外小矩阵，只改这部分。省掉 99% 的可训练参数，所以能在消费级显卡上跑。
2. rank 越小训练的参数量越少也更省显存但也越弱，alpha 用来放大 LoRA 的输出（常设 rank 的 2 倍），target_modules 决定在哪些矩阵旁边挂小分支（一般只做 Q 和 V）。
3. QLoRA + LLaMA-Factory。因为 8GB 显存跑不动全量，QLoRA 把原始权重压到 4bit 再配合 LoRA，刚好塞进你的显卡。LLaMA-Factory 提供一键启动脚本和中文文档，不用自己写训练循环。