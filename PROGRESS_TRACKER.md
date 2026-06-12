# ✅ 进度追踪器

> 追踪你的学习进度。每完成一个模块，将 `[ ]` 改为 `[x]`。

---

## 📊 总览

| 阶段 | 名称 | 状态 | 开始日期 | 完成日期 |
|------|------|------|---------|---------|
| 01 | 环境与基础 | ⬜ | | |
| 02 | Prompt 工程 | ⬜ | | |
| 03 | RAG 核心概念 | ⬜ | | |
| 04 | RAG 深度优化 | ⬜ | | |
| 05 | LangChain LCEL | ⬜ | | |
| 06 | Agent 基础 | ⬜ | | |
| 07 | 多智能体系统 | ⬜ | | |
| 08 | LangGraph 企业级 | ⬜ | | |
| 09 | 生产级 RAG | ⬜ | | |
| 10 | 评估与监控 | ⬜ | | |
| 11 | 企业安全与运营 | ⬜ | | |
| 12 | 毕业项目 | ⬜ | | |

---

## Phase 01：环境与基础（预计 1 周）

### 学习目标
- [ ] Python 异步编程（async/await）
- [ ] TypedDict、Pydantic、dataclasses 实战
- [ ] OpenAI API 调用、Token 计数、流式输出
- [ ] Anthropic API 调用、错误重试
- [ ] Ollama 本地模型部署、GGUF 基础
- [ ] 环境配置管理（pydantic-settings）

### 代码文件
- [ ] `01-python-essentials.ipynb` — Python AI 开发基础
- [ ] `02-llm-api-basics.ipynb` — LLM API 调用基础
- [ ] `03-local-model-setup.ipynb` — 本地模型部署
- [ ] `04-environment-config.py` — 环境配置管理

### 坑点理解
- [ ] API Key 泄露（`.gitignore` + pre-commit hook）
- [ ] Token 超限错误（token 计数 + 截断策略）
- [ ] 429 限流（指数退避 + jitter）
- [ ] 流式 JSON 不完整（缓冲区 + 完整性校验）

### 检验点
- [ ] 完成 10 道练习题
- [ ] 能正确调用 OpenAI 和 Anthropic API
- [ ] 能在本地 Ollama 运行模型

---

## Phase 02：Prompt 工程（预计 1-2 周）

### 学习目标
- [ ] Zero-shot / Few-shot / CoT / ReAct / Self-Ask 设计
- [ ] Jinja2 模板引擎 + 结构化输出
- [ ] Token 预算管理 + 滑动窗口
- [ ] A/B 测试框架 + 统计检验
- [ ] DSPy 自动化优化

### 代码文件
- [ ] `01-prompt-design-patterns.ipynb`
- [ ] `02-template-engines.ipynb`
- [ ] `03-output-structuring.ipynb`
- [ ] `04-context-window-management.ipynb`
- [ ] `05-few-shot-selection.py`
- [ ] `06-prompt-optimization-pipeline.py`
- [ ] `07-dspy-automated-optimization.py`

### 坑点理解
- [ ] Prompt 注入（18 种正则防御）
- [ ] JSON 输出格式不稳定
- [ ] 位置偏见（Lost in the Middle）
- [ ] 过度工程（回报递减）
- [ ] 成本爆炸（缓存策略）

---

## Phase 03：RAG 核心概念（预计 2-3 周）

### 学习目标
- [ ] 从零手写 Naive RAG
- [ ] 多格式文档加载（PDF/Word/MD/Web/OCR）
- [ ] 5 种文本切分策略对比
- [ ] 8 种 Embedding 模型基准测试
- [ ] 向量库选型与迁移
- [ ] FastAPI 端到端 RAG API

### 代码文件
- [ ] `01-naive-rag-from-scratch.ipynb`
- [ ] `02-document-loading.ipynb`
- [ ] `03-text-splitting-deep-dive.ipynb`
- [ ] `04-embeddings-comparison.py`
- [ ] `05-vector-store-selection.py`
- [ ] `06-basic-retrieval.ipynb`
- [ ] `07-end-to-end-rag-api.py`

### 坑点理解
- [ ] Naive RAG 幻觉
- [ ] Chunk 大小不当
- [ ] Embedding 模型语言不匹配
- [ ] 向量库超规模
- [ ] 冷启动延迟

---

## Phase 04：RAG 深度优化（预计 3-4 周）⭐

### 学习目标
- [ ] 数据清洗管道
- [ ] 语义自适应切分
- [ ] 特殊内容独立切分
- [ ] 混合检索（Dense + BM25 + RRF）
- [ ] 查询优化七步法
- [ ] 动态 Top-K + 相似度阈值
- [ ] 双重排序（BGE-Reranker + LLM 精筛）
- [ ] HyDE、PRF、父子切片

### 代码文件（14 个）
- [ ] `01-data-cleaning-pipeline.py`
- [ ] `02-semantic-chunking-production.py`
- [ ] `03-special-content-chunking.py`
- [ ] `04-metadata-enrichment.py`
- [ ] `05-hybrid-retrieval-rrf.py`
- [ ] `06-query-optimization.py`
- [ ] `07-query-rewriting.py`
- [ ] `08-multi-turn-completion.py`
- [ ] `09-complex-query-decomposition.py`
- [ ] `10-dynamic-topk.py`
- [ ] `11-similarity-threshold.py`
- [ ] `12-reranking-pipeline.py`
- [ ] `13-lost-in-middle-ranker.py`
- [ ] `14-parent-child-retrieval.py`

### 坑点理解
- [ ] BM25 静默失败
- [ ] RRF 权重误配
- [ ] 查询改写语义漂移
- [ ] 意图分类错误
- [ ] 重排延迟瓶颈

---

## Phase 05：LangChain LCEL（预计 1-2 周）

### 学习目标
- [ ] LCEL 管道操作符
- [ ] RunnablePassthrough/Parallel/Branch
- [ ] ChatPromptTemplate + 输出解析器
- [ ] 完整 LCEL RAG 链
- [ ] 工具调用与流式

### 代码文件（9 个）
- [ ] `01-lcel-fundamentals.ipynb`
- [ ] `02-prompt-templates.ipynb`
- [ ] `03-output-parsers.ipynb`
- [ ] `04-building-rag-chain.ipynb`
- [ ] `05-document-loaders.ipynb`
- [ ] `06-vector-stores-langchain.ipynb`
- [ ] `07-retrievers-and-routing.ipynb`
- [ ] `08-tool-calling-langchain.ipynb`
- [ ] `09-streaming-and-callbacks.ipynb`

### 坑点理解
- [ ] LCEL 调试困难
- [ ] 链组合错误
- [ ] 序列化问题
- [ ] 版本兼容性

---

## Phase 06：Agent 基础（预计 2 周）

### 学习目标
- [ ] 从零实现 ReAct Agent
- [ ] Function Calling 深度
- [ ] 工具执行框架
- [ ] 记忆系统（短期/长期/情景）
- [ ] LangChain Agent 对比

### 代码文件（7 个）
- [ ] `01-agent-control-loop.ipynb`
- [ ] `02-react-agent-from-scratch.py`
- [ ] `03-function-calling-deep-dive.ipynb`
- [ ] `04-tool-execution-framework.py`
- [ ] `05-agent-memory-systems.py`
- [ ] `06-memory-manager.py`
- [ ] `07-langchain-agent-builder.ipynb`

### 坑点理解
- [ ] 无限循环
- [ ] 工具调用失败
- [ ] 工具选错
- [ ] 记忆溢出
- [ ] 幻觉工具名
- [ ] JSON 参数解析失败

---

## Phase 07-12（进度占位）

<details>
<summary>Phase 07：多智能体系统（点击展开）</summary>

- [ ] `01-sequential-pipeline.py`
- [ ] `02-hierarchical-manager-worker.py`
- [ ] `03-debate-ensemble.py`
- [ ] `04-escalation-pattern.py`
- [ ] `05-blackboard-pattern.py`
- [ ] `06-agent-communication-protocol.py`
- [ ] `07-role-playing-agents.py`
</details>

<details>
<summary>Phase 08：LangGraph 企业级（点击展开）</summary>

- [ ] `01-stategraph-fundamentals.ipynb`
- [ ] `02-conditional-edges.ipynb`
- [ ] `03-rag-agent-graph.py`
- [ ] `04-checkpointer-postgres.py`
- [ ] `05-interrupt-human-in-loop.py`
- [ ] `06-time-travel.py`
- [ ] `07-master-sub-architecture.py`
- [ ] `08-redis-distributed-lock.py`
- [ ] `09-message-queue-integration.py`
- [ ] `10-enterprise-rag-agent-complete.py`
</details>

<details>
<summary>Phase 09：生产级 RAG 系统（点击展开）</summary>

- [ ] `01-data-pipeline-production.py`
- [ ] `02-multi-tenant-isolation.py`
- [ ] `03-four-level-cache.py`
- [ ] `04-prompt-caching.py`
- [ ] `05-circuit-breaker.py`
- [ ] `06-rate-limiter.py`
- [ ] `07-disaster-recovery.py`
- [ ] `08-incremental-update.py`
- [ ] `09-index-rebuild-cron.py`
- [ ] `10-self-rag-pattern.py`
- [ ] `11-corrective-rag.py`
- [ ] `12-adaptive-rag-routing.py`
- [ ] `13-knowledge-graph-rag.py`
</details>

<details>
<summary>Phase 10-12 + 轨道/补充（点击展开）</summary>

详见各目录的 README.md 和 checkpoint/ 文件。
</details>

---

## 🏆 毕业标准

- [ ] 通用事实准确率 ≥ 90%
- [ ] 专业领域准确率 ≥ 94%
- [ ] 幻觉率 < 5%
- [ ] 关键信息可追溯到源 Chunk
- [ ] 全流程自动化
- [ ] 50 题综合考试 ≥ 85%
- [ ] 3 个系统设计任务通过

---

> 💡 **使用方式**：每个 Phase 完成后回来打勾。建议每天/每周回顾进度。
