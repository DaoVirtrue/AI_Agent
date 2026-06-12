# B2-02: LLaMA-Factory 全流程指南

## 概述

LLaMA-Factory 是目前最流行的LLM微调框架之一，提供WebUI和命令行两种使用方式，支持100+模型的LoRA/QLoRA/全参数微调。

**版本**: LLaMA-Factory v0.9+

## 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 创建虚拟环境
conda create -n llama-factory python=3.10
conda activate llama-factory

# 安装依赖
pip install -e ".[torch,metrics]"

# 如需Flash Attention 2 (可选，加速训练)
pip install flash-attn --no-build-isolation
```

## 2. dataset_info.json 注册数据集

编辑 `data/dataset_info.json`，在文件末尾添加：

```json
{
  "rag_citation_train": {
    "file_name": "rag_citation_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  },
  "rag_refusal_train": {
    "file_name": "rag_refusal_train.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "rag_multidoc_train": {
    "file_name": "rag_multidoc_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

## 3. LoRA配置

创建 `configs/lora_rag.yaml`:

```yaml
### 模型配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
# 或使用: meta-llama/Llama-3-8B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

### 微调方法
finetuning_type: lora

### LoRA超参数
lora_rank: 16          # LoRA秩。8-64之间。rank越大模型容量越大但训练越慢
lora_alpha: 32         # LoRA缩放因子。通常alpha = 2*rank
lora_dropout: 0.05     # Dropout防止过拟合
lora_target: all       # 目标模块。all=所有线性层, 或指定["q_proj","v_proj"]

# QLoRA配置 (量化LoRA，节省显存)
# 取消注释以下行启用QLoRA:
# quantization_bit: 4
# quantization_method: bitsandbytes  # 或: hqq, eetq
# double_quantization: true

### 数据集
dataset: rag_citation_train,rag_refusal_train,rag_multidoc_train
template: qwen           # Qwen系列用qwen, Llama系列用llama3
cutoff_len: 2048         # 最大序列长度
overwrite_cache: true
preprocessing_num_workers: 4

### 训练超参数
output_dir: saves/qwen2.5-7b-lora-rag
logging_steps: 10
save_steps: 500
plot_loss: true          # 绘制损失曲线

per_device_train_batch_size: 2
gradient_accumulation_steps: 8  # 有效batch_size = 2*8 = 16
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true               # bfloat16混合精度

### 验证
val_size: 0.1            # 10%数据作为验证集
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 500
```

## 4. 启动训练

### WebUI方式

```bash
# 启动WebUI
llamafactory-cli webui

# 浏览器访问 http://localhost:7860
# 1. 选择模型和数据集
# 2. 配置LoRA参数
# 3. 点击"开始训练"
```

### 命令行方式

```bash
# 使用YAML配置
llamafactory-cli train configs/lora_rag.yaml

# 或直接传参
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --dataset rag_citation_train \
    --template qwen \
    --output_dir saves/qwen-7b-lora-rag \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --bf16
```

## 5. 训练监控

### WandB (推荐)

```bash
# 安装
pip install wandb

# 登录
wandb login

# 训练时自动记录（在config中添加）:
# report_to: wandb
# run_name: rag-finetune-v1
```

**关键监控指标**:
| 指标 | 含义 | 正常范围 |
|------|------|---------|
| train/loss | 训练损失 | 持续下降 |
| eval/loss | 验证损失 | 低于train/loss |
| train/learning_rate | 学习率 | 按cosine衰减 |
| train/epoch | 当前epoch | - |
| train/global_step | 全局步数 | - |

### TensorBoard

```bash
# LLaMA-Factory自动记录到output_dir/runs
tensorboard --logdir saves/qwen2.5-7b-lora-rag/runs
```

### 训练过程解读

```
阶段1 (前10%步数): 损失快速下降，warmup阶段
阶段2 (10%-60%):   损失平稳下降，有效学习阶段
阶段3 (60%-80%):   损失下降趋缓，开始收敛
阶段4 (80%-100%):  损失接近平台，可能开始过拟合
                   → 如果eval/loss上升，应早停
```

## 6. LoRA权重导出与合并

```bash
# 导出LoRA权重到HuggingFace格式
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path saves/qwen2.5-7b-lora-rag/checkpoint-xxx \
    --template qwen \
    --finetuning_type lora \
    --export_dir models/qwen-7b-rag-merged \
    --export_size 2 \
    --export_device cpu
```

## 7. 推理测试

```bash
# 交互式对话测试
llamafactory-cli chat \
    --model_name_or_path models/qwen-7b-rag-merged \
    --template qwen

# API服务
llamafactory-cli api \
    --model_name_or_path models/qwen-7b-rag-merged \
    --template qwen \
    --port 8000
```

## 8. 常见问题

### Q: OOM (Out of Memory) 怎么办？

```
方案优先级:
1. 减小batch_size (per_device_train_batch_size: 1)
2. 增大gradient_accumulation_steps (保持有效batch_size不变)
3. 启用QLoRA (quantization_bit: 4)
4. 减小cutoff_len (从2048减到1024)
5. 使用更小的模型 (7B → 3B, 3B → 1.5B)
```

### Q: 训练损失不下降？

```
检查清单:
- [ ] 数据集是否正确加载？（检查第一个batch的输出）
- [ ] learning_rate是否合适？（5e-5是安全默认值）
- [ ] warmup_ratio是否过大？（默认0.1）
- [ ] 数据格式是否与template匹配？
```

### Q: 生成内容重复或质量差？

```
- 检查数据集是否包含多样化样本
- 增加epoch数（但如果损失已收敛则无效）
- 调整temperature和top_p推理参数
- 降低lora_rank（减少过拟合）
```

## 9. 推荐配置速查表

| 模型规模 | GPU | 方法 | lora_rank | batch_size | grad_accum |
|---------|-----|------|-----------|------------|------------|
| 7B | 16GB | LoRA | 16 | 2 | 8 |
| 7B | 12GB | QLoRA-4bit | 16 | 1 | 8 |
| 14B | 24GB | QLoRA-4bit | 16 | 1 | 8 |
| 70B | 48GB | QLoRA-4bit | 8 | 1 | 16 |
| 70B | 80GB | QLoRA-4bit | 16 | 2 | 8 |
