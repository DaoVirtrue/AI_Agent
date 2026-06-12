# 📇 源文档索引

> 每份课程文件对应哪份源文档的哪些章节，方便深度学习时回溯原文。

---

## 源文档 → 课程映射

| 源文档 | 文件大小 | 主要映射阶段 |
|--------|---------|------------|
| `01-RAG检索增强生成.md` | 118 KB | Phase 03, 04 |
| `02-RAG知识库构建实战.md` | 114 KB | Phase 03, 04, 09 |
| `03-RAG框架对比与选型.md` | 67 KB | Phase 05, Track C |
| `01-Agent智能体架构.md` | 86 KB | Phase 06 |
| `02-多智能体系统.md` | 75 KB | Phase 07 |
| `03-Agent工具与插件开发.md` | 95 KB | Phase 06, 11 |
| `02-Prompt驱动开发.md` | 30 KB | Phase 02 |
| `03-Prompt优化系统化方法.md` | 97 KB | Phase 02 |
| `03-Function Calling.md` | 29 KB | Phase 06 |
| `04-AI应用性能评估.md` | 119 KB | Phase 10 |
| `01-模型调优方法论.md` | 46 KB | Phase 09, Track B |
| `02-Fine-tuning实战.md` | 50 KB | Track B |
| `01-AI应用开发范式总览.md` | 36 KB | Phase 01, 全体系 |
| `LangGraph企业级AI Agent落地设计文档.md` | 10 KB | Phase 08 |
| `RAG准确率全方位增强全栈落地手册.docx` | 12 KB | Phase 04, 09 |
| `RAG工程落地全流程详细步骤.docx` | 11 KB | Phase 09 |
| `生产级RAG全链路落地验收清单.docx` | 20 KB | Phase 09, 10, 12 |
| `04-Vibe Coding.md` | 20 KB | Phase 12 |
| `05-Skills与插件化架构.md` | 32 KB | Phase 11 |
| `AGENT_CREATION_GUIDE(1).md` | 37 KB | Track C (Dify) |
| `Agent_design_paradigm_exam(1).md` | 15 KB | Phase 06, 07 |

---

## 按阶段查找源文档

### Phase 01：环境与基础
- `01-AI应用开发范式总览.md` → 第 1 节：全栈 AI 架构师能力模型
- `01-AI应用开发范式总览.md` → 第 5 节：完整技术栈

### Phase 02：Prompt 工程
- `02-Prompt驱动开发.md` → 全篇（Prompt 设计模式、模板引擎、安全、版本管理、A/B 测试）
- `03-Prompt优化系统化方法.md` → 全篇（优化流水线、DSPy、元提示、遗传算法、Few-shot 选择）
- `01-模型调优方法论.md` → 第 1-2 节：调优金字塔、决策框架

### Phase 03：RAG 核心概念
- `01-RAG检索增强生成.md` → 全篇（Naive RAG、检索策略、Chunking、Embedding、向量库）
- `02-RAG知识库构建实战.md` → 第 1-3 节（文档加载、OCR、表格提取、索引管理）
- `03-RAG框架对比与选型.md` → 第 7 节：框架选择决策矩阵、学习路径

### Phase 04：RAG 深度优化
- `01-RAG检索增强生成.md` → 检索优化、查询重写、HyDE、PRF、重排序
- `02-RAG知识库构建实战.md` → 第 4-6 节（检索优化、缓存、评估、监控、成本）
- `RAG准确率全方位增强全栈落地手册.docx` → 全篇（8 大优化模块、统一参数表）

### Phase 05：LangChain LCEL
- `03-RAG框架对比与选型.md` → 第 2 节：LangChain 详解（LCEL、Chain 类型、LangSmith）

### Phase 06：Agent 基础
- `01-Agent智能体架构.md` → 全篇（ReAct、Plan-Execute、ReWOO、ToT、Reflexion、记忆、安全）
- `03-Agent工具与插件开发.md` → 全篇（工具定义、沙箱、权限、MCP）
- `03-Function Calling.md` → 全篇（Schema 设计、路由、并行、错误处理）

### Phase 07：多智能体系统
- `02-多智能体系统.md` → 全篇（顺序、层级、辩论、黑板、通信协议、A2A、降级）
- `Agent_design_paradigm_exam(1).md` → 第 3 节：范式对比与组合推荐

### Phase 08：LangGraph 企业级
- `LangGraph企业级AI Agent落地设计文档.md` → 全篇（StateGraph、Checkpointer、interrupt、Time Travel、Master-Sub、企业部署 7 规则）

### Phase 09：生产级 RAG 系统
- `02-RAG知识库构建实战.md` → 第 7-9 节（多租户、容灾、增量更新、成本优化）
- `RAG工程落地全流程详细步骤.docx` → 全篇（8 阶段管道、6 周时间线）
- `生产级RAG全链路落地验收清单.docx` → 全篇（9 大验收部分、15 个高频问题）
- `01-模型调优方法论.md` → 第 3-8 节（SFT 流程、数据增强、评估、迭代）

### Phase 10：评估与监控
- `04-AI应用性能评估.md` → 全篇（六维评估、RAGAS、Golden Dataset、A/B 测试、漂移检测）
- `生产级RAG全链路落地验收清单.docx` → 第 8 节：评估与 DevOps

### Phase 11：企业安全与运营
- `01-AI应用开发范式总览.md` → 第 9-10 节（生产部署检查清单、四层安全架构、成本优化金字塔）
- `05-Skills与插件化架构.md` → 全篇（Skills 架构、多租户、金丝雀部署、熔断降级、AIAppRouter）

### Phase 12：毕业项目
- `04-Vibe Coding.md` → 全篇（AI 辅助开发、代码质量保障、技术债务管理）
- `生产级RAG全链路落地验收清单.docx` → 第 9 节：生产上线红线

### Track A：多模态与流式
- `01-RAG检索增强生成.md` → 多模态 RAG、Cross-Lingual Retrieval
- `03-Agent工具与插件开发.md` → 流式工具调用

### Track B：模型微调
- `02-Fine-tuning实战.md` → 全篇（数据工程、LLaMA-Factory、LoRA 超参、训练监控）
- `01-模型调优方法论.md` → 第 5-7 节（数据准备、样本量指导）

### Track C：多框架实战
- `03-RAG框架对比与选型.md` → 第 2-7 节（各框架详解、对比、决策树）
- `AGENT_CREATION_GUIDE(1).md` → 全篇（Dify Agent 实战）
- `03-Agent工具与插件开发.md` → MCP 协议详解

### Track D：合规与商业
- `05-Skills与插件化架构.md` → 第 8 节：多租户深度隔离
- `01-AI应用开发范式总览.md` → 第 11 节：成本优化金字塔

### 补充模块 01-03
- `04-AI应用性能评估.md` → 测试策略、CI/CD 集成
- `生产级RAG全链路落地验收清单.docx` → 15 个高频问题 → Runbook 素材
- `04-Vibe Coding.md` → 团队协作规范、代码审查清单

---

> 💡 **使用方式**：学完某节课后，回到这里查看对应源文档章节进行深度学习。
