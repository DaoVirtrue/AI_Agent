# Phase 03: RAG核心概念（RAG Core Concepts）

**时长：2-3周** | **难度：中级** | **前置：Phase 02（向量与嵌入基础）**

---

## 学习目标

完成本阶段后，你将能够：

1. **从零构建Naive RAG系统**：理解检索增强生成的完整四阶段流水线（加载→分块→嵌入→检索→生成）
2. **掌握5种文本分块策略**：能根据文档类型选择最优分块方法
3. **对比8种主流嵌入模型**：在中文+英文场景下做出有数据支撑的选型决策
4. **评估5种向量数据库**：从原型到生产的迁移路径
5. **诊断Naive RAG的5大失败模式**：为Phase 04的高级检索技术建立动机

---

## 文件清单

| 文件 | 类型 | 核心内容 | 建议时长 |
|------|------|----------|----------|
| `01-naive-rag-from-scratch.ipynb` | Notebook | MinimalRAG类，四阶段流水线，失败演示 | 3-4小时 |
| `02-document-loading.ipynb` | Notebook | 6种文档加载器，统一接口，错误处理 | 2-3小时 |
| `03-text-splitting-deep-dive.ipynb` | Notebook | 5种分块策略实现与可视化对比 | 3-4小时 |
| `04-embeddings-comparison.py` | Python脚本 | 8种嵌入模型基准测试，Matryoshka演示 | 2-3小时 |
| `05-vector-store-selection.py` | Python脚本 | 5种向量数据库对比，迁移脚本，基准测试 | 2-3小时 |
| `06-basic-retrieval.ipynb` | Notebook | Dense检索，Top-K效应，距离度量，查询扩展 | 2-3小时 |
| `07-end-to-end-rag-api.py` | FastAPI应用 | 完整RAG API服务，文档上传→检索→生成 | 3-4小时 |
| `pitfalls/` | 参考文档 | 5个常见陷阱的诊断与修复 | 1-2小时 |
| `checkpoint/` | 练习 | 8道练习题+详解答案 | 3-4小时 |

**总计：约21-31小时**

---

## 关键概念

### Naive RAG的四阶段流水线

```
文档加载 → 文本分块 → 向量嵌入 → 相似度检索 → LLM生成
   ↑                                              |
   └──────────── 反馈循环（Phase 04引入）←──────────┘
```

### 五种文本分块策略

| 策略 | 原理 | 适用场景 | 典型参数 |
|------|------|----------|----------|
| **Fixed-Length** | 固定token数切割 | 均匀文本 | chunk_size=500, overlap=50 |
| **Recursive Character** | 按分隔符层级递归切割 | 通用文档 | ["\n\n", "\n", "。", ";", " ", ""] |
| **Sentence-Aware** | 句子边界检测 | 自然语言 | spaCy / 正则表达式 |
| **Semantic** | 嵌入相似度低谷检测 | 话题分散的文档 | 相似度阈值=0.6 |
| **Adaptive** | 根据文档类型自适应 | 混合文档集 | FAQ(256/30), 技术(512/50), 报告(1024/100) |

### 八种嵌入模型速览

| 模型 | 维度 | 中文支持 | 最佳场景 | 近似成本/1M tokens |
|------|------|----------|----------|-------------------|
| OpenAI text-embedding-3-small | 512/1536 | 中等 | 通用英文 | $0.020 |
| OpenAI text-embedding-3-large | 3072/256/1024 | 良好 | 高精度需求 | $0.130 |
| BGE-M3 | 1024 | **最佳中文** | 多语言/中文 | 免费(本地) |
| BGE-large-zh-v1.5 | 1024 | **最佳中文** | 中文专用 | 免费(本地) |
| m3e-base | 768 | 优秀中文 | 中文轻量 | 免费(本地) |
| Cohere embed-v3 | 1024 | 良好 | 多语言企业 | $0.100 |
| Jina embeddings-v3 | 1024/256/768 | 良好 | 长文档 | $0.020 |
| Voyage-large-2 | 1536 | 中等 | 英文高精度 | $0.060 |

### 五种向量数据库对比

| 数据库 | 部署方式 | 索引算法 | 适用阶段 | 扩展能力 |
|--------|----------|----------|----------|----------|
| **Chroma** | 嵌入式/Python | HNSW | 原型/开发 | 百万级 |
| **Qdrant** | Docker/Cloud | HNSW | 中小生产 | 千万级 |
| **Milvus** | Docker Compose/K8s | 多种(HNSW/IVF/DiskANN) | 企业生产 | 百亿级 |
| **Pinecone** | SaaS | 托管 | 快速上线 | 十亿级 |
| **FAISS** | 嵌入式/C++ | 多种 | 研究/离线 | 无上限 |

---

## 常见陷阱预览

| 陷阱 | 症状 | 出现文件 |
|------|------|----------|
| Naive RAG幻觉 | 回答与文档无关，编造事实 | pitfalls/01 |
| 错误的分块大小 | 检索结果不完整或噪声过多 | pitfalls/02 |
| 嵌入模型不匹配 | 中文检索效果极差 | pitfalls/03 |
| 向量数据库瓶颈 | 查询延迟飙升，内存不足 | pitfalls/04 |
| 冷启动缓慢 | 服务启动需要数分钟 | pitfalls/05 |

---

## 环境准备

```bash
# 核心依赖
pip install openai numpy scipy tiktoken spacy jupyter fastapi uvicorn
pip install pypdf2 pdfplumber python-docx markdown beautifulsoup4
pip install pytesseract pillow chromadb qdrant-client pymilvus pinecone-client
pip install sentence-transformers FlagEmbedding cohere voyageai

# spaCy中文模型
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm

# OCR引擎（Windows需手动安装Tesseract）
# https://github.com/UB-Mannheim/tesseract/wiki

# Milvus（需要Docker）
# docker compose up -d  # 使用05-vector-store-selection.py中提供的配置
```

---

## 学习路径建议

1. **Day 1-3**：完成 `01-naive-rag-from-scratch.ipynb`，亲手构建第一个RAG系统
2. **Day 4-5**：完成 `02-document-loading.ipynb`，学会处理各种格式的文档
3. **Day 6-8**：完成 `03-text-splitting-deep-dive.ipynb`，理解分块策略对检索质量的深远影响
4. **Day 9-10**：运行 `04-embeddings-comparison.py`，获得嵌入模型选型的实战经验
5. **Day 11-12**：运行 `05-vector-store-selection.py`，了解不同向量数据库的优劣
6. **Day 13-14**：完成 `06-basic-retrieval.ipynb`，深入理解检索机制
7. **Day 15-17**：构建 `07-end-to-end-rag-api.py`，将知识整合为可部署的服务
8. **Day 18-19**：阅读 `pitfalls/` 文档，完成 `checkpoint/` 练习
9. **Day 20-21**：复习、自我评估，准备进入 Phase 04

---

## 与下一阶段的衔接

Phase 03 构建的是 **Naive RAG**——它的失败模式直接构成了 Phase 04 的课程目录：
- 检索失败 → Phase 04 的查询重写、HyDE、多路召回
- 分块不优 → Phase 04 的父子文档、句子窗口检索
- 相关性差 → Phase 04 的重排序、Self-RAG、CRAG

**记住：理解Naive RAG为什么失败，比学会它如何工作更重要。**
