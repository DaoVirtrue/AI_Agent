# 📚 全术语词典

> 按字母/拼音排序，每个术语标注首次出现阶段和核心定义。

---

## A

### Agent（智能体）
- **Phase 06** | 能够自主感知环境、制定计划、调用工具、执行行动的 AI 系统
- 核心循环：Perception → Planning → Action → Reflection

### Agentic RAG
- **Phase 09** | LLM Agent 主动控制检索决策（是否检索、检索哪个库、检索几次）
- 与 Naive RAG 的区别：Agent 可以多轮检索、自我修正

### A/B Testing（A/B 测试）
- **Phase 02** | 将流量分为对照组（A）和实验组（B），通过统计检验比较两组指标差异
- 关键指标：p 值（<0.05 显著）、Cohen's d（效应量）

### Adaptive RAG
- **Phase 09** | 根据查询复杂度自动选择检索策略（简单→直接检索、中等→迭代检索、复杂→多步推理）

---

## B

### BGE-M3（BAAI General Embedding M3）
- **Phase 03** | 智源研究院开源的多语言 Embedding 模型，中文场景最佳选择
- 支持 Dense（1024维）+ Sparse（词汇权重）双模表示，8192 token 上下文

### BM25（Best Match 25）
- **Phase 04** | 基于词频和逆文档频率的经典稀疏检索算法
- 在 RAG 中与向量检索互补，对专业术语和精确关键词效果更好

### Blackboard Pattern（黑板模式）
- **Phase 07** | 多 Agent 共享同一数据结构，各自独立读写，适合信息聚合场景

---

## C

### Chain-of-Thought (CoT，思维链)
- **Phase 02** | 引导 LLM 分步推理的 Prompt 技术
- 典型触发词："让我们一步步思考"（Let's think step by step）

### Chunk（文本块/切片）
- **Phase 03** | 将长文档切分为小块以便向量化和检索
- 关键参数：chunk_size（块大小）、chunk_overlap（重叠量）

### Circuit Breaker（熔断器）
- **Phase 09** | 当外部依赖连续失败达到阈值时，自动切断请求，防止级联故障
- 三态：CLOSED（正常）→ OPEN（熔断）→ HALF_OPEN（探测恢复）

### Corrective RAG (CRAG)
- **Phase 09** | 检索后对结果质量打分，低质量时自动回退到 Web 搜索

### Checkpointer（检查点）
- **Phase 08** | LangGraph 的状态持久化机制，将每次节点执行的 State 自动存储
- 支持崩溃恢复、中断续传、时光旅行

---

## D

### DSPy
- **Phase 02** | 斯坦福开源的 Prompt 自动优化框架
- 核心理念：将 Prompt 工程从"手工调参"转变为"数据驱动编译"

### Drift Detection（漂移检测）
- **Phase 10** | 监控数据分布和模型性能随时间的变化
- 方法：MMD（最大均值差异）、KS 检验、余弦距离

---

## E

### Embedding（向量嵌入）
- **Phase 03** | 将文本转换为固定维度向量的技术
- 语义相近的文本在向量空间中距离更近

### Evaluation（评估）
- **Phase 10** | 系统化衡量 AI 系统质量的方法
- 六维框架：准确性、相关性、忠实度、流畅度、延迟、成本

---

## F

### Faithfulness（忠实度）
- **Phase 10** | RAG 生成答案是否基于检索到的上下文（而非幻觉）
- RAGAS 指标之一：将答案分解为陈述，逐一检查是否被上下文支持

### Few-shot（少样本提示）
- **Phase 02** | 在 Prompt 中提供 2-5 个示例，引导 LLM 按特定模式输出

### Fine-tuning（微调）
- **Track B** | 在预训练模型上用特定领域数据继续训练
- LoRA/QLoRA 是最常用的参数高效微调方法

### Function Calling（函数调用）
- **Phase 06** | LLM 选择并调用预定义的函数/工具来获取外部信息或执行操作
- 关键：JSON Schema 定义工具的参数和描述

---

## G

### Golden Dataset（金标数据集）
- **Phase 10** | 人工标注的高质量测试集，用于评估和回归测试
- 推荐：50-500 条，含 20% 对抗样本

### Graph RAG
- **Phase 09** | 结合知识图谱的 RAG，通过实体关系推理增强检索

---

## H

### Hallucination（幻觉）
- **Phase 02 起贯穿全部** | LLM 生成看似合理但事实错误的内容
- 在 RAG 中通过强制溯源 Prompt 抑制，在 Agent 中通过工具校验抑制

### HITL（Human-in-the-Loop，人机协作）
- **Phase 08** | 在 AI 流程关键节点引入人工审批
- LangGraph 通过 `interrupt()` 实现

### HNSW（Hierarchical Navigable Small World）
- **Phase 03** | 高性能近似最近邻搜索算法，向量数据库常用索引
- 速度快（对数复杂度），召回率 95-99%

### HyDE（Hypothetical Document Embeddings）
- **Phase 04** | LLM 先生成"假设答案"，用假设答案的向量检索，而非直接用问题检索
- 效果：用"文档语言空间"检索比"问题语言空间"检索更准

### Hybrid Search（混合检索）
- **Phase 04** | 同时使用向量检索（语义）+ BM25（关键词），结果通过 RRF 融合
- 通用场景权重 7:3，专业场景 4:6

---

## I

### Intent Classification（意图分类）
- **Phase 04** | 将用户查询分为 6 类：事实/流程/对比/故障/推理/闲聊
- 不同意图路由到不同检索策略

### Interrupt（中断）
- **Phase 08** | LangGraph 的暂停执行机制，等待外部输入后继续
- 用于：大额交易审批、敏感内容审核、缺少参数补全

---

## K

### Knowledge Graph（知识图谱）
- **Phase 09** | 实体-关系-实体的结构化知识网络
- Graph RAG 从文档中提取实体→建图→图遍历+向量检索

---

## L

### LangChain
- **Phase 05** | 最流行的 LLM 应用开发框架，核心抽象是 Chain
- LCEL（LangChain Expression Language）是声明式组合语法

### LangGraph
- **Phase 08** | LangChain 团队开发的 Agent 工作流编排框架
- 核心概念：StateGraph、Node、Edge、Conditional Edge、Checkpointer

### LCEL（LangChain Expression Language）
- **Phase 05** | LangChain 的声明式管道语法
- 使用 `|` 管道操作符组合组件：`prompt | llm | parser`

### LlamaIndex
- **Track C** | 数据索引与检索框架，以"索引"为核心抽象
- 强项：高级检索策略（递归检索、子问题引擎、路由引擎）

### LoRA（Low-Rank Adaptation）
- **Track B** | 参数高效微调技术，只训练低秩矩阵而非全量参数
- 推荐配置：rank=16, alpha=32, dropout=0.05

### Lost in the Middle
- **Phase 04** | LLM 对长文本中间部分关注度天然偏低
- 解决：LongContextReorder 将重要内容移到开头和结尾

---

## M

### MCP（Model Context Protocol）
- **Track C / Phase 07** | Anthropic 发布的工具/资源调用标准协议
- 基于 JSON-RPC 2.0，支持 stdio/HTTP/SSE 传输

### MMR（Maximal Marginal Relevance）
- **Phase 02** | 多样化示例选择算法，平衡与查询的相关性和示例间的差异性

### MRR（Mean Reciprocal Rank）
- **Phase 10** | 检索评估指标，第一个相关文档排名倒数的均值

### Multi-Agent System（多智能体系统）
- **Phase 07** | 多个 Agent 协作完成复杂任务
- 模式：顺序流水线、层级管理、辩论、黑板、拍卖

### Multi-Tenant（多租户）
- **Phase 09** | 同一 AI 系统服务多个客户，数据和资源严格隔离
- 三级隔离：Collection 级 + 元数据过滤 + API Key

---

## N

### Naive RAG
- **Phase 03** | 最简 RAG 实现：Index → Retrieve → Augment → Generate
- 缺陷：无查询优化、无重排序、无混合检索 → 准确率通常仅 60-70%

### NDCG（Normalized Discounted Cumulative Gain）
- **Phase 10** | 归一化折损累计增益，考虑排序位置的检索评估指标

---

## P

### Parent-Child Retrieval（父子检索）
- **Phase 04** | 两阶段检索：先用小粒度 child chunk（256 token）精准检索，再展开到 parent chunk（1024 token）提供上下文

### PII（Personally Identifiable Information）
- **Phase 11** | 个人身份信息（手机号、身份证、银行卡等），需自动检测并脱敏

### Plan-and-Execute
- **Phase 06** | Agent 模式：先制定完整计划，再按步执行
- 优势：全局规划更合理；劣势：灵活性不如 ReAct

### Prompt Caching
- **Phase 02** | 将不变的内容（System Prompt、Few-shot 示例）缓存，大幅降低 API 成本
- Anthropic：缓存命中后输入价格降低 90%

### Prompt Injection（提示注入）
- **Phase 02/11** | 恶意用户通过精心构造的输入覆盖或绕过 System Prompt
- 防御：分隔符包裹 + 18 种正则检测 + 纵深防御

---

## Q

### QLoRA
- **Track B** | 量化 + LoRA，将模型量化为 4-bit 再微调，大幅降低显存需求

### Query Rewriting（查询改写）
- **Phase 04** | 将用户的非正式/不完整查询改写为标准专业表述
- 方法：LLM 改写 + 多维度扩展 + 指代消解

---

## R

### RAG（Retrieval-Augmented Generation，检索增强生成）
- **Phase 03** | 检索 + 生成的 AI 应用范式
- 四阶段：Index → Retrieve → Augment → Generate

### RAGAS
- **Phase 10** | RAG 专用评估框架
- 指标：Faithfulness、Answer Relevancy、Context Precision/Recall/Relevancy

### ReAct（Reasoning + Acting）
- **Phase 06** | 最主流的 Agent 推理框架
- 循环：Thought → Action → Observation → Thought...

### Rerank（重排序）
- **Phase 04** | 检索后对 Top-K 结果二次精排，提升精准度
- 模型：BGE-Reranker-v2-m3（中文）、Cohere Rerank v3（多语言）

### ReWOO（Reasoning Without Observation）
- **Phase 06** | 变体 Agent 模式：先计划（含占位符）→ 批量执行 → 最终推理
- 优势：仅 3 次 LLM 调用，Token 成本最低

### RRF（Reciprocal Rank Fusion，倒数排名融合）
- **Phase 04** | 融合多个检索结果列表的算法
- 公式：`RRFscore(d) = Σ 1/(k + rank_i(d))`，k 通常取 60

---

## S

### Self-RAG
- **Phase 09** | 生成时通过特殊 Token 自我反思检索质量
- Token：`<RETRIEVE>`, `<ISREL>`, `<ISSUP>`, `<ISUSE>`

### Semantic Chunking（语义切分）
- **Phase 03** | 基于句子嵌入相似度检测语义边界进行切分
- 阈值：通用 85-90%，专业 90-95%

### SSE（Server-Sent Events，服务端推送事件）
- **Track A** | HTTP 单向流式传输协议，用于 LLM token-by-token 输出

### StateGraph
- **Phase 08** | LangGraph 核心概念：用有向图定义 Agent 工作流
- State 是全局唯一数据容器（TypedDict），Node 是执行单元

---

## T

### Time Travel（时光旅行）
- **Phase 08** | LangGraph 基于 Checkpointer 的状态回退能力
- 用途：回溯到任意历史快照→换策略重试

### Token
- **Phase 01** | LLM 处理的最小文本单元（约 0.75 个英文单词或 1.5 个中文字符）

### Tree-of-Thought (ToT，思维树)
- **Phase 02/06** | BFS 搜索思维空间：每步生成多分支→评估→剪枝→保留最佳
- 成本约是 CoT 的 18 倍

---

## V

### Vector Database（向量数据库）
- **Phase 03** | 专门存储和检索向量的数据库
- 推荐演进：Chroma（原型）→ Qdrant（增长）→ Milvus（企业）→ Pinecone（免运维）

### Vibe Coding
- **Phase 12** | Andrej Karpathy 提出的 AI 辅助编程范式
- 核心理念：描述意图 → AI 生成 → 人工审核 → 迭代修改

---

## Z

### Zero-shot（零样本提示）
- **Phase 02** | 不给任何示例，直接描述任务要求 LLM 执行

---

> 💡 **使用建议**：遇到不认识的术语时，Ctrl+F 搜索此页面。每个术语在对应 Phase 中有详细讲解。
