# 第05阶段：LangChain 表达式语言（LCEL）与生产级 RAG 链

> **学习周期**：1-2 周  
> **核心主题**：掌握 LangChain Expression Language (LCEL)，构建生产级 RAG 链条  
> **前置阶段**：[第03阶段：从零实现RAG](../phase-03-zero-to-rag/) | [第04阶段：向量数据库](../phase-04-vector-databases/)

---

## 📋 阶段概述

本阶段聚焦于 **LangChain Expression Language (LCEL)**，这是 LangChain 生态中构建可组合、可观察、可流式传输 LLM 应用程序的统一接口。你将学习如何使用 `|` 管道操作符声明式地组合组件，将自定义实现与 LCEL 实现进行对比，并最终构建包含对话历史、源引用和流式输出的生产级 RAG 链。

LCEL 的核心优势：
- **声明式语法**：通过 `|` 操作符直观地表达数据流
- **自动并行化**：`RunnableParallel` 自动处理并发执行
- **内置可观测性**：所有 LCEL 链自动集成 LangSmith 追踪
- **原生流式支持**：链中的每个组件都能流式传输数据
- **异步/同步统一**：同一套代码同时支持同步和异步调用

---

## 🎯 学习目标

完成本阶段后，你应当能够：

1. **理解 LCEL 核心概念**：掌握 `|` 管道操作符、`RunnablePassthrough`、`RunnableLambda`、`RunnableParallel`、`RunnableBranch` 和 `RunnableSequence`
2. **构建复杂提示模板**：使用 `ChatPromptTemplate`、`MessagesPlaceholder`、少样本示例选择器和语义相似度选择器
3. **构建生产级 RAG 链**：用 LCEL 构建完整的检索增强生成链，包括上下文格式化、检索和答案生成
4. **实现对话式 RAG**：集成聊天历史记录，支持多轮对话
5. **对比 LCEL 与自定义实现**：量化代码减少量和可读性提升
6. **理解工具调用**：掌握 Agent 工具定义和 Tool 装饰器的使用
7. **实现流式输出**：掌握 `stream()`、`astream()` 和 `astream_events()` 的用法
8. **使用回调系统**：实现自定义回调处理器，用于日志记录、监控和调试
9. **构建端到端 RAG 应用**：将检索、生成、来源引用和错误处理整合为一个完整的可运行应用

---

## 📚 文件清单（9个文件）

| 序号 | 文件名 | 内容说明 | 类型 |
|------|--------|----------|------|
| 01 | `01-lcel-fundamentals.ipynb` | LCEL 核心基础：管道操作符、RunnablePassthrough、RunnableLambda、RunnableParallel、RunnableBranch、RunnableSequence | Notebook |
| 02 | `02-prompt-templates.ipynb` | 提示模板系统：ChatPromptTemplate、MessagesPlaceholder、FewShotChatMessagePromptTemplate、语义相似度选择器 | Notebook |
| 03 | `03-tool-calling.ipynb` | 工具调用与 Agent：Tool 装饰器、StructuredTool、AgentExecutor、ReAct Agent、自定义工具开发 | Notebook |
| 04 | `04-building-rag-chain.ipynb` | 构建 RAG 链：完整 LCEL RAG 链、与自定义实现对比、对话式 RAG、源引用、代码量对比 | Notebook |
| 05 | `05-streaming-callbacks.ipynb` | 流式传输与回调：stream/astream、astream_events、自定义回调处理器、BaseCallbackHandler、AsyncCallbackHandler | Notebook |
| 06 | `06-advanced-chains.ipynb` | 高级链构建：多步推理链、条件分支链、路由链、fallback 机制、with_fallbacks() | Notebook |
| 07 | `07-conversational-rag.ipynb` | 对话式 RAG 应用：完整聊天历史管理、上下文压缩、会话摘要、token 预算管理 | Notebook |
| 08 | `08-end-to-end-app.py` | 端到端 RAG 应用脚本：完整的命令行 RAG 工具，包含数据加载、索引、检索、生成和导出 | Python 脚本 |
| 09 | `09-unit-tests.py` | 单元测试套件：使用 pytest 对所有核心链组件进行测试，包括 mock LLM、测试夹具和参数化测试 | Python 脚本 |

---

## 🔧 前置条件

### 必需环境

| 要求 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | LCEL 依赖的类型注解和模式匹配特性 |
| pip | ≥ 23.0 | 包管理工具 |

### 核心依赖

```bash
pip install langchain>=0.3.0
pip install langchain-core>=0.3.0
pip install langchain-openai>=0.2.0
pip install langchain-community>=0.3.0
```

### 辅助依赖

```bash
pip install chromadb>=0.5.0       # 向量存储
pip install tiktoken>=0.7.0       # Token 计数
pip install python-dotenv>=1.0.0  # 环境变量管理
pip install numpy>=1.26.0         # 数值计算
pip install pytest>=8.0.0         # 测试框架（文件09）
pip install pytest-asyncio>=0.23  # 异步测试（文件09）
```

### 一键安装

```bash
pip install langchain langchain-core langchain-openai langchain-community chromadb tiktoken python-dotenv numpy pytest pytest-asyncio
```

### 环境配置

在项目根目录创建 `.env` 文件：

```bash
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_TRACING_V2=false  # 设为 true 可启用 LangSmith 追踪
LANGCHAIN_API_KEY=           # LangSmith API 密钥（可选）
```

---

## 🗂️ 推荐学习路径

按以下顺序逐步学习，每个步骤建立在前一步的基础上：

```
01-lcel-fundamentals.ipynb
        │
        ▼
02-prompt-templates.ipynb
        │
        ▼
03-tool-calling.ipynb
        │
        ▼
04-building-rag-chain.ipynb  ←── 核心：与第03阶段自定义实现对比
        │
        ▼
05-streaming-callbacks.ipynb
        │
        ▼
06-advanced-chains.ipynb
        │
        ▼
07-conversational-rag.ipynb
        │
        ▼
08-end-to-end-app.py         ←── 最终产出：完整的命令行 RAG 工具
        │
        ▼
09-unit-tests.py              ←── 质量保证：测试套件
```

### 每日建议计划

| 天数 | 内容 | 预计时长 |
|------|------|----------|
| 第1天 | 环境搭建 + 01-lcel-fundamentals.ipynb（前半） | 2-3 小时 |
| 第2天 | 01-lcel-fundamentals.ipynb（后半）+ 02-prompt-templates.ipynb | 2-3 小时 |
| 第3天 | 03-tool-calling.ipynb | 2-3 小时 |
| 第4天 | 04-building-rag-chain.ipynb（重点：与自定义实现的对比） | 3-4 小时 |
| 第5天 | 05-streaming-callbacks.ipynb + 06-advanced-chains.ipynb | 2-3 小时 |
| 第6天 | 07-conversational-rag.ipynb | 2-3 小时 |
| 第7天 | 08-end-to-end-app.py + 09-unit-tests.py | 3-4 小时 |
| 可选 | 复习、优化、扩展练习 | 灵活 |

---

## ✅ 完成评估标准

### 基础要求（必须全部满足）

- [ ] 能够解释 LCEL `|` 操作符的工作原理
- [ ] 能够用 `RunnablePassthrough` 和 `RunnableLambda` 构建简单链
- [ ] 能够用 `RunnableParallel` 实现并发执行
- [ ] 能够使用 `RunnableBranch` 实现条件路由
- [ ] 能够用 `ChatPromptTemplate` 构建包含系统消息和历史的提示模板
- [ ] 能够用 `FewShotChatMessagePromptTemplate` 创建少样本示例提示
- [ ] 能够用 LCEL 构建完整的 RAG 链（检索 + 生成）
- [ ] 能够对比 LCEL 实现与自定义实现，量化代码减少量
- [ ] 能够实现对话式 RAG（带聊天历史管理）
- [ ] 能够使用 `stream()` 方法实现流式输出

### 进阶要求（建议满足）

- [ ] 能够定义自定义 Tool 并集成到 Agent 中
- [ ] 能够使用 `astream_events()` 实现细粒度事件流
- [ ] 能够实现自定义 `BaseCallbackHandler` / `AsyncCallbackHandler`
- [ ] 能够构建带 fallback 机制的多模型链（`with_fallbacks()`）
- [ ] 能够实现上下文压缩和会话摘要
- [ ] 能够运行完整的单元测试套件（文件09）并通过所有测试
- [ ] 能够运行端到端应用（文件08）并处理实际文档

### 自我检测问题

1. `RunnablePassthrough.assign()` 与直接使用 `RunnablePassthrough()` 有什么区别？
2. 为什么 LCEL 链中的 `RunnableParallel` 能自动并行执行？
3. `MessagesPlaceholder` 在对话式 RAG 中起什么作用？
4. 自定义 RAG 实现（第03阶段）与 LCEL 实现的主要区别是什么？
5. `stream()` 和 `invoke()` 在内部实现上有何不同？
6. `with_fallbacks()` 如何提高系统的健壮性？

---

## 📖 参考资源

- [LangChain Expression Language (LCEL) 官方文档](https://python.langchain.com/docs/expression_language/)
- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain 工具调用指南](https://python.langchain.com/docs/how_to/tool_calling/)
- [LangChain 流式传输指南](https://python.langchain.com/docs/how_to/streaming/)
- [LangChain 回调文档](https://python.langchain.com/docs/how_to/callbacks/)
- [第03阶段：从零实现RAG](../phase-03-zero-to-rag/) - 用于对比学习
- [第04阶段：向量数据库](../phase-04-vector-databases/) - 向量存储基础

---

> **提示**：学习 LCEL 时，请将重点放在理解"数据如何流经管道"这一核心思想上。每个 `|` 操作符的左侧输出会自动成为右侧的输入。一旦掌握了这个思维模型，构建复杂链将变得直观而自然。
