# B2-03: Axolotl 替代方案

## 概述

Axolotl 是一个功能强大的LLM微调框架，相比LLaMA-Factory提供了更灵活的训练配置和更多高级特性。

**仓库**: https://github.com/axolotl-ai-cloud/axolotl

## 1. 快速开始

```bash
# 安装
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl
pip install -e .

# 或使用Docker
docker run --gpus all -it \
    -v $(pwd)/data:/workspace/data \
    winglian/axolotl:latest
```

## 2. Axolotl YAML配置

```yaml
# axolotl_rag_config.yml
base_model: Qwen/Qwen2.5-7B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

# 模型加载
load_in_8bit: false
load_in_4bit: false
strict: false

# 数据集 (支持多种格式)
datasets:
  - path: data/rag_citation_train.jsonl
    ds_type: json
    type: sharegpt
    conversation: chatml  # conversation template

  - path: data/rag_refusal_train.jsonl
    ds_type: json
    type: alpaca

# 数据集预处理
dataset_prepared_path: last_run_prepared
val_set_size: 0.1
sequence_len: 2048
sample_packing: true       # ★样本打包（提高GPU利用率）
eval_sample_packing: false

# LoRA配置
adapter: lora
lora_model_dir:
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# QLoRA配置
# load_in_4bit: true
# lora_r: 16
# lora_alpha: 32

# 训练超参数
output_dir: ./outputs/axolotl-rag
num_epochs: 3
micro_batch_size: 2
gradient_accumulation_steps: 8
optimizer: adamw_bnb_8bit  # bitsandbytes优化的AdamW
lr_scheduler: cosine
learning_rate: 5.0e-5

# ★梯度检查点（用时间换显存）
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

# ★多GPU (DeepSpeed ZeRO)
# deepspeed: configs/zero2.json

# 混合精度
bf16: true
tf32: false

# 日志
logging_steps: 10
eval_steps: 500
save_steps: 500
wandb_project: rag-finetune
wandb_name: axolotl-rag-v1

# 早停
early_stopping_patience: 3
```

## 3. 核心特性对比: Axolotl vs LLaMA-Factory

| 特性 | LLaMA-Factory | Axolotl | 说明 |
|------|---------------|---------|------|
| **WebUI** | 完整WebUI | 无 | LLaMA-Factory更易上手 |
| **样本打包** | 不支持 | 支持 | ★ Axolotl能提高GPU利用率40%+ |
| **DeepSpeed集成** | 有限支持 | 完整支持 | ZeRO-1/2/3 |
| **FSDP** | 不支持 | 支持 | 替代DDP的分布式策略 |
| **梯度检查点** | 基础支持 | 完整支持 | 节省显存的关键特性 |
| **Flash Attention** | 可选 | 内置支持 | Axolotl更稳定 |
| **数据格式** | Alpaca, ShareGPT | Alpaca, ShareGPT, ChatML, 自定义 | Axolotl更灵活 |
| **模型支持** | 100+ (通过template) | 广泛 (通过transformers) | 两者都够用 |
| **QLoRA** | bitsandbytes | bitsandbytes + HQQ + EETQ | Axolotl选择更多 |
| **多模态** | 有限 | Llava, Qwen-VL 支持 | ★ Axolotl适合多模态 |
| **校验点重训** | 支持 | 支持 | |
| **TPU支持** | 不支持 | 支持 | Axolotl可跑TPU |

## 4. 样本打包 (Sample Packing)

**Axolotl独占特性**，是它的最大优势之一。

### 原理

```
不用打包 (LLaMA-Factory默认):
Batch中每个样本填充到max_length
  [样本1: 500 tokens | PAD x 1500]
  [样本2: 300 tokens | PAD x 1700]
  [样本3: 800 tokens | PAD x 1200]
  GPU利用率: 约26%

使用打包 (Axolotl):
多个短样本拼接成接近max_length的序列
  [样本1|EOS|样本2|EOS|样本3|EOS|样本4|PAD x 50]
  [样本5|EOS|样本6|EOS|样本7|EOS|EOS|PAD x 300]
  GPU利用率: 约90%
```

### 效果

- GPU利用率从~25% → ~85%
- 训练吞吐量提升 2-3x
- 适用于短对话数据特别有效

### 注意事项

- 大样本（接近max_length）几乎无收益
- 需要正确处理attention mask防止跨样本注意力
- 对EOS token位置敏感

## 5. 梯度检查点 (Gradient Checkpointing)

**两种实现方式**:

| 方式 | 显存节省 | 速度损失 | 适用 |
|------|---------|---------|------|
| 无检查点 | 0% | 0% | 显存充足 |
| 普通检查点 | ~30% | ~20% | 标准配置 |
| reentrant=True | ~40% | ~25% | 极致省显存 |
| reentrant=False | ~30% | ~15% | 推荐配置 |

Axolotl中推荐的配置:
```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false  # PyTorch 2.0+推荐
```

## 6. 多GPU加速配置

### DeepSpeed ZeRO (Axolotl推荐)

```json
// configs/zero2.json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": true
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto"
}
```

### GPU数量与batch size参考

| GPU数量 | micro_batch | grad_accum | 有效batch | 适用模型规模 |
|---------|-------------|------------|-----------|-------------|
| 1x A10 (24GB) | 1 | 16 | 16 | 7B QLoRA |
| 1x A100 (80GB) | 4 | 4 | 16 | 7B LoRA |
| 2x A100 | 4 | 2 | 16 | 14B QLoRA |
| 4x A100 | 2 | 2 | 16 | 70B QLoRA |
| 8x A100 | 4 | 1 | 32 | 70B LoRA |

## 7. 选择建议

```
你适合 LLaMA-Factory 如果:
├── 初次接触LLM微调
├── 希望有WebUI操作界面
├── 数据集较简单（Alpaca/ShareGPT格式）
├── 使用单个GPU训练
└── 训练7B-14B模型

你适合 Axolotl 如果:
├── 需要样本打包提高吞吐量
├── 使用多GPU训练（DeepSpeed/FSDP）
├── 需要灵活的分布式策略
├── 训练多模态模型（Llava等）
├── 数据集格式多样或需要自定义
└── 对训练效率有更高要求
```

## 8. 训练命令

```bash
# Axolotl训练
accelerate launch -m axolotl.cli.train axolotl_rag_config.yml

# 或使用DeepSpeed
accelerate launch --config_file configs/deepspeed_zero2.yaml \
    -m axolotl.cli.train axolotl_rag_config.yml

# 合并LoRA权重
python -m axolotl.cli.merge_lora axolotl_rag_config.yml \
    --lora_model_dir ./outputs/axolotl-rag
```
