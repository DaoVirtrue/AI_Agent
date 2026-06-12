# C1-05：LlamaIndex vs LangChain 终极对比指南

> **适用读者**：需要在两个框架之间做技术选型的架构师、技术负责人
> **前置阅读**：Track B（LangChain 全栈实战）与本 Track C1-01 至 C1-04

---

## 一、定位差异（一句话总结）

| 框架 | 核心定位 | 设计哲学 |
|------|----------|----------|
| **LlamaIndex** | 数据索引与检索引擎 | "先索引，后查询" —— 以数据为中心 |
| **LangChain** | LLM 应用编排框架 | "链式编排" —— 以流程为中心 |

> LlamaIndex 回答的是"如何高效地从海量数据中检索到与问题最相关的信息"。
> LangChain 回答的是"如何把 LLM、工具、记忆、检索串联成一个完整的应用"。

---

## 二、架构对比

### LlamaIndex 核心架构

```
[数据源] → Data Connector → [Document]
                              ↓
                   IngestionPipeline (Transformations)
                              ↓
                         [Nodes] → Index
                              ↓
                    Query → Retriever → [Nodes]
                              ↓
                    Response Synthesizer → Answer
```

关键抽象：**Document → Node → Index → Retriever → Query Engine**

### LangChain 核心架构

```
[用户输入] → Prompt Template → [LLM] → Output Parser → [响应]
                 ↑
           Chain/LCEL (串联逻辑)
                 ↑
    [工具/Tool] [检索/Retriever] [记忆/Memory]
```

关键抽象：**Chain → Tool → Agent → Memory → Callback**

### 重叠区分析

```
              LangChain 强势区
              (Agent/Chain/Workflow)
                  ┌──────────────┐
     ┌────────────┼──────────────┼────────────┐
     │            │   重叠区     │            │
     │ LlamaIndex │  • 文档加载  │            │
     │  强势区    │  • 文档分割  │            │
     │            │  • 嵌入生成  │            │
     │ • 数据连接 │  • 向量存储  │            │
     │ • 索引结构 │  • 检索器    │            │
     │ • 查询引擎 │  • RAG       │            │
     └────────────┴──────────────┴────────────┘
```

---

## 三、能力矩阵对比

| 能力维度 | LlamaIndex | LangChain | 胜出者 |
|----------|------------|-----------|--------|
| **数据加载** | 160+ 原生连接器 | 需要社区 loader | LlamaIndex |
| **文档解析** | IngestionPipeline 体系 | Text Splitters | 持平 |
| **索引结构** | 6种索引（Vector/Tree/KW/...） | 依赖向量库 | LlamaIndex |
| **检索能力** | 递归/融合/HyDE/窗口 | 基础检索器 | LlamaIndex |
| **查询引擎** | Router/SubQ/SQL/多模态 | Chain/LCEL 组合 | 持平 |
| **Agent 框架** | AgentRunner/Worker | LangGraph/CrewAI | LangChain |
| **工作流编排** | Workflow（较新） | LangGraph（成熟） | LangChain |
| **多模态** | 原生图片/音频索引 | 需要额外组件 | LlamaIndex |
| **结构化数据** | SQL/DataFrame 引擎 | SQL Agent | LlamaIndex |
| **评估框架** | 内置 Faithfulness/Relevancy | 依赖 LangSmith | 持平 |
| **可观测性** | Callback + Arize/Phoenix | LangSmith（商业） | 持平 |
| **生态丰富度** | 中等偏上 | 非常庞大 | LangChain |
| **学习曲线** | 较陡（概念多） | 平缓（链式直观） | LangChain |

---

## 四、代码风格对比：同一 RAG 任务

### 基础 RAG：检索一个文档集合并回答问题

**LlamaIndex 写法：**

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载 → 索引 → 查询，三行搞定
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query("什么是RAG？")
print(response)
```

**LangChain 写法：**

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 加载 → 分割 → 嵌入 → 存储 → 链，五步
loader = DirectoryLoader("./data", glob="**/*.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever()
)
response = qa.invoke("什么是RAG？")
```

**结论**：基础 RAG 任务 LlamaIndex 更简洁。但 LangChain 的显式步骤更便于定制。

---

### 高级 RAG：路由+子问题分解+重排序

**LlamaIndex** 提供了开箱即用的 RouterQueryEngine、SubQuestionQueryEngine、SentenceTransformerRerank，组合使用代码量约 50 行。

**LangChain** 需要用 LCEL / LangGraph 手动编排这些逻辑，代码量约 150 行，但灵活性更高。

**结论**：高级 RAG 场景下，LlamaIndex 的"乐高式"组合更快速，LangChain 的自定义更灵活。

---

## 五、选型决策框架

### 决策树

```
你的需求是什么？
  │
  ├── "我要构建一个以数据检索为核心的 RAG 系统"
  │   └── → 选 LlamaIndex
  │       ├── 数据源种类多（>5种）
  │       ├── 需要多种索引策略
  │       ├── 需要高级检索（递归/融合/重排序）
  │       └── 需要查询结构化数据（SQL/DF）
  │
  ├── "我要构建一个以 Agent/工作流为核心的 AI 应用"
  │   └── → 选 LangChain + LangGraph
  │       ├── 需要复杂的多步推理工作流
  │       ├── 需要人机协同（Human-in-the-Loop）
  │       ├── 需要并行/条件分支的 Agent 编排
  │       └── 需要与大量外部 API 工具集成
  │
  ├── "我两个都要"
  │   └── → LlamaIndex（检索层）+ LangChain/LangGraph（编排层）
  │       ├── LangChain 集成：from llama_index.core.langchain_helpers... 
  │       ├── LlamaIndex 作为 LangChain Retriever
  │       └── 或直接用 LlamaIndex Workflow（新特性，趋近 LangGraph）
  │
  └── "我是初学者，想快速上手"
      └── → 先学 LangChain（生态大、教程多），再用 LlamaIndex 补检索
```

### 典型场景推荐

| 项目类型 | 推荐方案 | 原因 |
|----------|----------|------|
| 企业内部知识库 Q&A | LlamaIndex | 数据连接+索引+检索能力强 |
| 客服 Agent | LangChain + LangGraph | Agent 编排+人机协同强 |
| 多模态文档分析 | LlamaIndex | 原生支持图片/表格索引 |
| 自动化工作流 | LangChain + LangGraph | 工作流编排最成熟 |
| 金融合规审查 | LlamaIndex | SQL+文本混合检索优势 |
| 代码助手 Agent | LangChain + LangGraph | 工具调用+状态管理强 |
| RAG 即服务平台 | LlamaIndex | 索引管理+查询引擎开箱即用 |

---

## 六、迁移指南

### 从 LangChain 迁移到 LlamaIndex

**何时考虑迁移：**
- 检索需求变得复杂（开始手写递归检索、融合检索）
- 数据源种类激增（LangChain loader 不够用）
- 需要对索引做细粒度管理（增量更新、版本管理）

**如何渐进式迁移：**
1. 保留 LangChain 的 Agent/Chain 层
2. 用 LlamaIndex 替换检索层（作为 LangChain 的 Retriever）
3. 逐步将数据加载和索引构建迁移到 LlamaIndex
4. 最后（可选）用 LlamaIndex Workflow 替换 LangGraph

### 从 LlamaIndex 迁移到 LangChain

**何时考虑迁移：**
- Agent 逻辑变得复杂（需要条件分支、并行执行、人机协同）
- 需要丰富的工具生态（LangChain 社区工具更多）
- 团队更熟悉 LangChain 生态

**如何渐进式迁移：**
1. 保留 LlamaIndex 的 Index/Retriever
2. 将 LlamaIndex QueryEngine 包装为 LangChain Tool
3. 用 LangGraph 构建上层编排逻辑
4. 逐步将简单查询替换为 Chain

### 混合使用代码示例

```python
# 混合方案：LlamaIndex 负责检索，LangChain 负责 Agent 编排

from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from langchain.agents import create_openai_tools_agent
from langchain_openai import ChatOpenAI

# LlamaIndex 层：构建索引和查询引擎
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 将 LlamaIndex 查询引擎包装为 LangChain Tool
llama_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="knowledge_base",
    description="查询企业内部知识库",
)

# LangChain 层：用 Agent 编排
llm = ChatOpenAI(model="gpt-4o")
agent = create_openai_tools_agent(llm, [llama_tool.to_langchain_tool()])
```

---

## 七、社区与生态

| 维度 | LlamaIndex | LangChain |
|------|------------|-----------|
| GitHub Stars | ~38k | ~100k |
| 贡献者数量 | ~800+ | ~3000+ |
| 企业采用 | 中型企业为主 | 广泛采用 |
| 商业产品 | LlamaCloud（托管） | LangSmith（可观测）+ LangGraph Cloud |
| 文档质量 | 较好（有 Cookbook） | 一般（更新频繁） |
| 中文社区 | 较少 | 活跃 |
| 更新频率 | 周更 | 日更（API 频繁变动） |

---

## 八、总结：我该选哪个？

**一句话总结：数据和检索选 LlamaIndex，流程和 Agent 选 LangChain，两者结合是最优解。**

| 场景 | 推荐 |
|------|------|
| 快速原型 RAG 系统 | LlamaIndex |
| 复杂 Agent 工作流 | LangChain + LangGraph |
| 企业级 RAG 平台 | LlamaIndex（检索） + LangGraph（编排） |
| 学习 RAG 原理 | 先 LangChain 入门，再用 LlamaIndex 深入 |
| 多模态 RAG | LlamaIndex |
| 工具调用型 Agent | LangChain + LangGraph |

**最终建议**：不要二选一。把 LlamaIndex 看作"世界上最好的 RAG 检索库"，把 LangChain 看作"最灵活的 LLM 应用编排框架"，各取所长。
