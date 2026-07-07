选了什么模型：Qwen3-VL-4B-Instruct
选择理由：4B 参数，FP16 约 8GB，正好吃满 RTX 5070 Laptop 8GB 显存，不用量化
备选方案：如果跑不通换 Qwen3-VL-2B，或切 InternVL 小模型
下载方式：ModelScope snapshot_download
环境信息：PyTorch 2.11, CUDA 12.8, transformers 版本