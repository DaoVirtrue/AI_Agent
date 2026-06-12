# Track B: 模型工程化微调 (Model Fine-tuning)

## 课程概述

本轨道覆盖 RAG 系统中两个核心模型的微调：**Embedding 模型**（提升检索召回率）与**LLM 指令模型**（提升生成质量与引用准确性）。

**先修要求**：已完成 Phase 01-05（基础RAG全栈）

**同步阶段**：
- B1 Embedding 微调 → 对应 Phase 06 检索质量提升
- B2 LLM 指令微调 → 对应 Phase 07-08 生成质量提升与成本优化

## 微调金字塔：决策框架

```
        ┌──────────────┐
        │   Fine-tune   │  ← 当Prompt+RAG不足时
        ├──────────────┤
        │ RAG + Prompt  │  ← 工程优化首选
        ├──────────────┤
        │ Prompt Only   │  ← 成本最低
        └──────────────┘
```

**何时微调？**
- 检索命中率持续低于 70%（微调 Embedding）
- 引用准确率低于 80%（微调 LLM）
- 特定领域术语反复识别失败
- 负样本（不相关结果）反复出现但无法通过 prompt 排除

## 学习目标

1. **B1 Embedding 微调**
   - 从用户日志构建三元组数据集（query, positive, negative）
   - 使用 BM25 和跨批共享策略挖掘难负样本
   - 使用 sentence-transformers 训练 BGE-M3（MultipleNegativesRankingLoss + MatryoshkaLoss）
   - 通过 MTEB 中文基准和检索命中率评估微调效果
   - 导出 ONNX / TEI 部署，执行 INT8 量化

2. **B2 LLM 指令微调**
   - 构建 RAG 专用指令数据集（引用追踪 + 拒绝回答训练）
   - 掌握 LLaMA-Factory 全流程（LoRA 配置、dataset_info.json、训练监控）
   - 了解 Axolotl 替代方案及对比
   - 合并 LoRA 权重 → GGUF 导出 → Ollama 部署
   - 计算微调成本收益（GPU vs API 盈亏平衡分析）

## 目录结构

```
track-B-模型工程化微调/
├── README.md                      # 本文件
├── b1-embedding微调/
│   ├── pitfalls/                  # 常见坑点
│   ├── b1-01-三元组数据集构建.py  # 三元组数据集
│   ├── b1-02-难负样本挖掘.py      # 难负样本挖掘
│   ├── b1-03-bge-m3微调管道.py    # BGE-M3微调管道
│   ├── b1-04-微调后评估.py        # 微调后评估
│   └── b1-05-导出与部署.py        # 导出与部署
└── b2-llm指令微调/
    ├── pitfalls/
    ├── b2-01-rag专用指令数据集.py  # RAG专用指令数据集
    ├── b2-02-llama-factory全流程.md # LLaMA-Factory全流程
    ├── b2-03-axolotl替代方案.md    # Axolotl替代方案
    ├── b2-04-合并量化部署.py      # 合并量化部署
    └── b2-05-微调成本收益分析.py  # 微调成本收益分析
```

## 技术栈

| 领域 | 核心技术 |
|------|---------|
| Embedding训练 | sentence-transformers, BGE-M3, MatryoshkaLoss |
| 难负样本 | BM25, FAISS, Cross-batch negatives |
| 评估 | MTEB, PCA/t-SNE, Hit Rate |
| 推理部署 | ONNX, TEI, INT8 Quantization |
| LLM微调 | LLaMA-Factory, Axolotl, LoRA/QLoRA |
| 量化合并 | bitsandbytes, llama.cpp GGUF, Ollama |

## 硬件要求

| 任务 | 最小GPU | 推荐GPU |
|------|---------|---------|
| Embedding微调 (BGE-M3) | 16GB VRAM | 24GB+ (A10, A100) |
| LLM 7B LoRA | 12GB VRAM | 24GB (RTX 4090, A10) |
| LLM 70B QLoRA | 48GB VRAM | 80GB (A100) |
| 推理 (GGUF q4) | CPU 16GB RAM | CPU 32GB+ RAM |
