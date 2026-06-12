# Phase 04: RAG 深度优化 ⭐

> **最关键阶段** | 预计时间：3-4 周 | 目标：准确率从 60-70% 提升至 90%+

---

## 概述

Phase 04 是 RAG 课程体系中最关键、最核心的阶段。本阶段从数据清洗到检索增强，系统性地优化 RAG 管线的每一个环节。完成本阶段后，您将具备构建工业级 RAG 系统的能力。

## 模块概览

| # | 模块 | 文件 | 核心内容 |
|---|------|------|----------|
| 1 | 数据清洗管道 | `01-data-cleaning-pipeline.py` | 水印去除、页眉页脚剥离、空行清理、MD5去重、语义去重 |
| 2 | 语义切分 | `02-semantic-chunking-production.py` | 句子嵌入+相似度低谷检测、三级层次（父→中→子） |
| 3 | 特殊内容切分 | `03-special-content-chunking.py` | 表格检测、公式提取、代码块隔离、图表占位 |
| 4 | 元数据增强 | `04-metadata-enrichment.py` | 自动生成摘要、实体标签、领域词典链接 |
| 5 | 混合检索 RRF | `05-hybrid-retrieval-rrf.py` | 稠密+稀疏检索、RRF融合、动态权重 |
| 6 | 查询优化管线 | `06-query-optimization.py` | 7步查询预处理：清洗→意图分类→消歧→重写→扩展→补全→分解 |
| 7 | 查询重写 | `07-query-rewriting.py` | LLM驱动的查询重写、少样本示例、语义保真度检查 |
| 8 | 多轮对话补全 | `08-multi-turn-completion.py` | 指代消解、上下文拼接、长期记忆池 |
| 9 | 复杂查询分解 | `09-complex-query-decomposition.py` | 多条件/多步推理检测、子问题拆分、并行检索融合 |
| 10 | 动态Top-K | `10-dynamic-topk.py` | 查询复杂度评分、自适应K值、悬崖检测 |
| 11 | 相似度阈值 | `11-similarity-threshold.py` | 集合级阈值配置、网格搜索调优、召回率vs精确率 |
| 12 | 重排序管道 | `12-reranking-pipeline.py` | BGE-Reranker、Cohere Rerank、Cross-Encoder、批量重排 |
| 13 | 中间信息丢失排序 | `13-lost-in-middle-ranker.py` | 位置感知重排序、LongContextReorder |
| 14 | 父子切片检索 | `14-parent-child-retrieval.py` | 子块精确匹配+父块上下文扩展、两端检索 |

---

## 统一参数配置表

### 通用场景（客服FAQ、百科问答）

| 参数 | 值 | 说明 |
|------|-----|------|
| 语义切分相似度阈值 | 0.85 - 0.90 | 句子间余弦相似度低谷 |
| 切分重叠率 | 20% | 相邻块重叠比例 |
| 最小块大小 | 128 tokens | 低于此值合并到相邻块 |
| 最大块大小 | 1024 tokens | 超过此值强制切分 |
| 稠密:稀疏检索权重 | 7:3 | 偏向语义理解 |
| RRF 融合 k 值 | 60 | RRF 平滑因子 |
| 相似度阈值 | 0.75 | 低于此值的结果丢弃 |
| 动态Top-K范围 | 3 - 10 | 简单查询3，复杂查询10 |
| 意图分类类型 | fact/process/comparison/chat | 4类主要意图 |

### 专业场景（法律、医学、金融）

| 参数 | 值 | 说明 |
|------|-----|------|
| 语义切分相似度阈值 | 0.90 - 0.95 | 更严格的语义边界 |
| 切分重叠率 | 30% | 更大重叠防止信息断裂 |
| 最小块大小 | 256 tokens | 保持术语完整性 |
| 最大块大小 | 2048 tokens | 保留更多上下文 |
| 稠密:稀疏检索权重 | 4:6 | 偏向关键词精确匹配 |
| RRF 融合 k 值 | 60 | 与通用场景相同 |
| 相似度阈值 | 0.85 | 更高的相关性要求 |
| 动态Top-K范围 | 3 - 12 | 复杂查询允许更多候选 |
| 意图分类类型 | fact/process/comparison/troubleshoot/reasoning/chat | 6类全意图 |

---

## 学习路径

```
Week 1: 数据准备
  Day 1-2: 数据清洗管道 (Module 1)
  Day 3-4: 语义切分 (Module 2)
  Day 5: 特殊内容切分 (Module 3)
  Week 1 检查点: 清洗后的文档库搭建

Week 2: 检索优化
  Day 1-2: 元数据增强 (Module 4)
  Day 3-4: 混合检索 RRF (Module 5) ⭐
  Day 5: 查询优化管线 (Module 6)
  Week 2 检查点: 检索准确率评估

Week 3: 查询与重排序
  Day 1: 查询重写 (Module 7)
  Day 2: 多轮补全 (Module 8)
  Day 3: 复杂查询分解 (Module 9)
  Day 4: 动态Top-K + 相似度阈值 (Modules 10-11)
  Day 5: 重排序管道 (Module 12)
  Week 3 检查点: 端到端准确率 > 85%

Week 4: 高级技术
  Day 1-2: Lost-in-Middle 排序 (Module 13)
  Day 3-4: 父子切片检索 (Module 14)
  Day 5: 最终评估与调优
  Week 4 检查点: 端到端准确率 > 90%
```

---

## 关键指标

| 指标 | 优化前 | 优化后目标 |
|------|--------|-----------|
| 检索准确率 (Precision@5) | 60-70% | 90%+ |
| 检索召回率 (Recall@10) | 55-65% | 85%+ |
| MRR (Mean Reciprocal Rank) | 0.5-0.6 | 0.85+ |
| NDCG@10 | 0.5-0.6 | 0.88+ |
| 平均检索延迟 | < 200ms | < 500ms (含重排序) |
| 查询处理延迟 | N/A | < 100ms |

---

## 常见陷阱 (pitfalls/)

| 陷阱 | 文件 | 严重程度 |
|------|------|----------|
| BM25 静默返回空结果 | `01-bm25-integration-failure.md` | 高 |
| RRF 融合权重配置错误 | `02-rrf-weight-misconfiguration.md` | 高 |
| 查询重写导致语义漂移 | `03-query-rewriting-drift.md` | 中 |
| 意图分类错误 | `04-intent-classification-errors.md` | 中 |
| 重排序延迟瓶颈 | `05-reranker-latency-bottleneck.md` | 中 |

---

## 检查点练习 (checkpoint/)

10 个综合练习，涵盖混合检索器实现、RRF权重调优、查询优化器构建、重排序评估、父子检索实现等。

详见 `checkpoint/exercises.md` 和 `checkpoint/solutions.md`。

---

## 前置要求

- Python 3.10+
- CUDA 兼容 GPU（推荐，非必需）
- 已完成 Phase 01-03 或具备同等基础
- 依赖：sentence-transformers, rank-bm25, FlagEmbedding, torch, scikit-learn

## 安装依赖

```bash
pip install sentence-transformers rank-bm25 FlagEmbedding torch scikit-learn \
            numpy tqdm matplotlib seaborn opencv-python-headless Pillow \
            camelot-py[cv] tabula-py PyPDF2 python-docx openpyxl
```
