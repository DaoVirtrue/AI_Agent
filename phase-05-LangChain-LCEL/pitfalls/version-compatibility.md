# LangChain 版本兼容性问题：包拆分导致的导入错误与迁移

## 症状

升级 LangChain 相关包（或不小心在已安装旧版本的环境中运行新代码）后，代码突然无法运行，抛出以下典型异常：

```python
# 症状 1: 导入错误
ImportError: cannot import name 'Chroma' from 'langchain.vectorstores'

# 症状 2: 属性错误
AttributeError: module 'langchain.chat_models' has no attribute 'ChatOpenAI'

# 症状 3: 模块不存在
ModuleNotFoundError: No module named 'langchain.schema'

# 症状 4: 弃用警告（运行时不终止但行为可能改变）
DeprecationWarning: Importing from langchain.llms is deprecated.
Please use langchain_openai.OpenAI instead.

# 症状 5: 运行时类型错误（API 参数名变更）
TypeError: ChatOpenAI.__init__() got an unexpected keyword argument 'model_name'
Did you mean 'model'?

# 症状 6: Pydantic 验证错误（升级后 schema 不兼容）
ValidationError: 1 validation error for ChatOpenAI
model_name -> Field required [type=missing, ...]
```

## 根因分析

LangChain 从 0.1.x 到 0.2.x 再到 0.3.x 经历了一次重大的架构重构——**单体包到多包生态的拆分**。这不是简单的版本号递增，而是底层模块组织方式的根本性改变。

### 版本演变时间线

```
2023 Q2-Q4: LangChain 0.0.x 时代
  ├─ 所有功能集中在一个 langchain 包中
  ├─ import langchain.vectorstores / langchain.chat_models 等
  └─ 问题：依赖膨胀、安装缓慢、版本冲突频繁

2024 Q1: LangChain 0.1.x 时代（过渡期）
  ├─ langchain-core 首次发布（核心抽象）
  ├─ langchain-community 首次发布（社区集成）
  ├─ 旧导入路径仍可用，但输出 DeprecationWarning
  └─ 建议用户开始迁移导入路径

2024 Q2-Q3: LangChain 0.2.x 时代（正式拆分）
  ├─ langchain-openai、langchain-text-splitters 等独立包发布
  ├─ 许多旧导入路径变为硬错误（不再只是警告）
  ├─ Pydantic v2 迁移开始
  └─ RetrievalQA、LLMChain 等旧链被标记为弃用

2024 Q4+: LangChain 0.3.x 时代（稳定拆分）
  ├─ 包拆分完成，旧路径完全移除
  ├─ Pydantic v2 强制要求
  ├─ Python ≥ 3.10 为最低要求
  └─ LCEL 成为唯一推荐的链构建方式
```

### 拆分的根本原因

1. **依赖膨胀**：旧的 `langchain` 单体包安装了所有第三方集成（OpenAI、Chroma、Pinecone 等），即使你只使用其中一个。这导致安装包体积巨大、依赖冲突频繁。
2. **版本锁死**：所有集成共享同一个 `langchain` 版本号，某个集成需要更新时，整个包都要发版。
3. **测试爆炸**：修改一行核心代码需要测试上百个集成。
4. **安全漏洞范围**：某个集成包的依赖有安全漏洞时，所有用户都受影响。

### 拆分后包职责

| 包名 | 职责 | pip 安装 |
|------|------|----------|
| `langchain-core` | 核心抽象：Runnable、BaseMessage、BaseRetriever、BaseTool、Callbacks、OutputParsers | `pip install langchain-core` |
| `langchain-community` | 社区维护的第三方集成：向量存储、文档加载器、LLM 包装器 | `pip install langchain-community` |
| `langchain-openai` | OpenAI 专用集成：ChatOpenAI、OpenAIEmbeddings | `pip install langchain-openai` |
| `langchain-text-splitters` | 文本分割器：RecursiveCharacterTextSplitter 等 | `pip install langchain-text-splitters` |
| `langchain` | 顶层统一入口（薄封装），聚合上述包 | `pip install langchain` |
| `langgraph` | 有状态多步骤 Agent 编排（独立于 langchain） | `pip install langgraph` |
| `langsmith` | 可观测性与追踪（独立于 langchain） | `pip install langsmith` |

## 真实场景

### 场景 1：向量存储导入失败

这是最常遇到的版本兼容性问题。

```python
# === 错误代码（LangChain 0.1.x 及更早） ===
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

升级到 0.3.x 后的错误：

```
Traceback (most recent call last):
  File "app.py", line 3, in <module>
    from langchain.vectorstores import Chroma
ImportError: cannot import name 'Chroma' from 'langchain.vectorstores'
```

**修复方法**：向量存储已迁移到集成专用包或 `langchain-community`。

```python
# === 正确代码（LangChain 0.3.x） ===
# 方案 A: 使用 langchain-community（推荐，多数社区集成在此）
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 方案 B: 使用 langchain-chroma（Chroma 专用包，更轻量）
# pip install langchain-chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
```

### 场景 2：ChatOpenAI 导入和参数均变更

```python
# === 错误代码（LangChain 0.1.x） ===
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-4",          # ← 旧参数名
    temperature=0,
    max_tokens=1024,             # ← 旧参数名
    request_timeout=60           # ← 已移除
)
```

升级后的错误：

```
ImportError: cannot import name 'ChatOpenAI' from 'langchain.chat_models'
```

即使修复了导入，还有参数错误：

```
TypeError: ChatOpenAI.__init__() got an unexpected keyword argument 'model_name'
```

**修复方法**：

```python
# === 正确代码（LangChain 0.3.x） ===
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",               # model_name → model
    temperature=0,
    max_completion_tokens=1024,  # max_tokens → max_completion_tokens
    timeout=60                   # request_timeout → timeout
)
```

### 场景 3：RetrievalQA 被废弃

```python
# === 错误代码（LangChain 0.1.x） ===
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
result = qa_chain.invoke({"query": "你的问题"})
```

升级到 0.3.x 后，`RetrievalQA` 仍然存在但标记为废弃并有运行时警告。官方强烈建议迁移到 LCEL。

**修复方法**：

```python
# === 正确代码（LangChain 0.3.x，使用 LCEL） ===
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template(
    "基于以下上下文回答问题：\n\n{context}\n\n问题：{question}\n\n回答："
)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("你的问题")
```

### 场景 4：文档加载器导入失败

```python
# === 错误代码 ===
from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

# 错误: ModuleNotFoundError 或 ImportError
```

**修复方法**：

```python
# === 正确代码 ===
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
```

## 常见破坏性变更速查表（20 个常见案例）

### 向量存储类

| # | 旧导入路径 (0.1.x) | 新导入路径 (0.3.x) | 所需 pip 包 |
|---|---|---|---|
| 1 | `from langchain.vectorstores import Chroma` | `from langchain_community.vectorstores import Chroma` | langchain-community |
| 2 | `from langchain.vectorstores import FAISS` | `from langchain_community.vectorstores import FAISS` | langchain-community |
| 3 | `from langchain.vectorstores import Qdrant` | `from langchain_community.vectorstores import Qdrant` | langchain-community |
| 4 | `from langchain.vectorstores import Pinecone` | `from langchain_pinecone import PineconeVectorStore` | langchain-pinecone |
| 5 | `from langchain.vectorstores import Weaviate` | `from langchain_weaviate import WeaviateVectorStore` | langchain-weaviate |

### 模型与嵌入类

| # | 旧导入路径 (0.1.x) | 新导入路径 (0.3.x) | 所需 pip 包 |
|---|---|---|---|
| 6 | `from langchain.chat_models import ChatOpenAI` | `from langchain_openai import ChatOpenAI` | langchain-openai |
| 7 | `from langchain.llms import OpenAI` | `from langchain_openai import OpenAI` | langchain-openai |
| 8 | `from langchain.embeddings import OpenAIEmbeddings` | `from langchain_openai import OpenAIEmbeddings` | langchain-openai |
| 9 | `from langchain.embeddings import HuggingFaceEmbeddings` | `from langchain_huggingface import HuggingFaceEmbeddings` | langchain-huggingface |
| 10 | `from langchain.embeddings import HuggingFaceBgeEmbeddings` | `from langchain_community.embeddings import HuggingFaceBgeEmbeddings` | langchain-community |

### 文档处理类

| # | 旧导入路径 (0.1.x) | 新导入路径 (0.3.x) | 所需 pip 包 |
|---|---|---|---|
| 11 | `from langchain.document_loaders import TextLoader` | `from langchain_community.document_loaders import TextLoader` | langchain-community |
| 12 | `from langchain.document_loaders import PyPDFLoader` | `from langchain_community.document_loaders import PyPDFLoader` | langchain-community |
| 13 | `from langchain.document_loaders import WebBaseLoader` | `from langchain_community.document_loaders import WebBaseLoader` | langchain-community |
| 14 | `from langchain.text_splitter import RecursiveCharacterTextSplitter` | `from langchain_text_splitters import RecursiveCharacterTextSplitter` | langchain-text-splitters |
| 15 | `from langchain.text_splitter import CharacterTextSplitter` | `from langchain_text_splitters import CharacterTextSplitter` | langchain-text-splitters |

### 核心类型与工具类

| # | 旧导入路径 (0.1.x) | 新导入路径 (0.3.x) | 说明 |
|---|---|---|---|
| 16 | `from langchain.schema import HumanMessage, AIMessage, SystemMessage` | `from langchain_core.messages import HumanMessage, AIMessage, SystemMessage` | 消息类型 |
| 17 | `from langchain.schema import BaseRetriever, Document` | `from langchain_core.retrievers import BaseRetriever` + `from langchain_core.documents import Document` | 检索器和文档 |
| 18 | `from langchain.callbacks import BaseCallbackHandler, StdOutCallbackHandler` | `from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler` | 回调处理器 |
| 19 | `from langchain.tools import tool, StructuredTool` | `from langchain_core.tools import tool, StructuredTool` | 工具装饰器 |
| 20 | `from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder` | `from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder` | 提示模板 |

### 参数名称变更对照表

| 类 | 旧参数名 (0.1.x) | 新参数名 (0.3.x) | 备注 |
|---|---|---|---|
| `ChatOpenAI` | `model_name` | `model` | 统一命名 |
| `ChatOpenAI` | `max_tokens` | `max_completion_tokens` | 更精确的语义 |
| `ChatOpenAI` | `request_timeout` | `timeout` | 简化 |
| `ChatOpenAI` | `headers` | `default_headers` | 明确默认值 |
| `OpenAIEmbeddings` | `chunk_size` | `dimensions`（含义不同） | API 变化 |
| `OpenAIEmbeddings` | `deployment` | `model` | Azure 相关 |
| `Chroma.from_documents()` | `embedding` | `embedding_function` | 部分版本变更 |

### 已完全移除的概念和类

| 移除项 | 替代方案 | 迁移难度 |
|--------|---------|----------|
| `langchain.chains.LLMChain` | LCEL: `prompt \| llm \| StrOutputParser()` | 低 |
| `langchain.chains.RetrievalQA` | LCEL RAG 链 | 低 |
| `langchain.chains.ConversationalRetrievalChain` | LCEL + `RunnableWithMessageHistory` 或 LangGraph | 中 |
| `langchain.chains.SimpleSequentialChain` | LCEL: `chain1 \| chain2` | 低 |
| `langchain.memory.ConversationBufferMemory` | `langchain_community.chat_message_histories` + `RunnableWithMessageHistory` | 中 |
| `langchain.agents.initialize_agent()` | `langgraph.prebuilt.create_react_agent()` 或手动构建 AgentExecutor | 高 |
| `langchain.chains.ConversationChain` | LCEL + `RunnableWithMessageHistory` | 中 |
| `langchain.indexes.VectorstoreIndexCreator` | 手动创建 vectorstore | 低 |

## 解决方案

### 方案 1：使用 langchain-cli 自动迁移（最推荐）

这是官方提供的一键迁移工具，能自动修复大部分导入路径变更。

```bash
# 安装迁移 CLI 工具
pip install langchain-cli

# 预览所有需要变更的内容（不修改文件，安全查看）
langchain-cli migrate /path/to/your/code --diff

# 输出示例:
# File: app.py
#   Line 3: from langchain.vectorstores import Chroma
#         -> from langchain_community.vectorstores import Chroma
#   Line 5: from langchain.chat_models import ChatOpenAI
#         -> from langchain_openai import ChatOpenAI

# 自动执行迁移（修改原文件）
langchain-cli migrate /path/to/your/code --yes

# 仅迁移特定文件
langchain-cli migrate /path/to/your/code --file my_code.py

# 递归迁移整个目录树（排除 .venv, node_modules 等）
langchain-cli migrate /path/to/your/code --recursive --yes
```

`langchain-cli migrate` 的覆盖范围：
- 包路径迁移（`langchain.vectorstores` → `langchain_community.vectorstores`）
- 类名迁移（已知的类名变更）
- 弃用标记（对于无法自动迁移的项，标记 `# TODO: Manual migration needed`）

### 方案 2：渐进式代码兼容策略

如果你的项目需要同时支持多个 LangChain 版本（例如在过渡期），可以使用兼容导入：

```python
"""
兼容导入策略：尝试新路径，失败则回退旧路径
"""

# --- 向量存储 ---
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
        import warnings
        warnings.warn(
            "正在使用过时的 langchain.vectorstores.Chroma。"
            "请安装 langchain-community 并更新导入语句。",
            FutureWarning,
            stacklevel=2
        )
    except ImportError:
        raise ImportError(
            "无法导入 Chroma。请执行: pip install langchain-community"
        )

# --- LLM ---
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        import warnings
        warnings.warn(
            "正在使用过时的 langchain.chat_models.ChatOpenAI。"
            "请安装 langchain-openai 并更新导入语句。",
            FutureWarning,
            stacklevel=2
        )
    except ImportError:
        raise ImportError(
            "无法导入 ChatOpenAI。请执行: pip install langchain-openai"
        )

# --- 核心类型（langchain-core 在 0.2+ 总是可用） ---
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

# --- 文本分割器 ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### 方案 3：查阅 CHANGELOG 和官方迁移指南

每次大版本升级前必须查阅的资源：

```bash
# 官方迁移指南（必读）
https://python.langchain.com/docs/versions/v0_2/    # 0.1.x → 0.2.x
https://python.langchain.com/docs/versions/v0_3/    # 0.2.x → 0.3.x

# GitHub Releases（含完整 CHANGELOG）
https://github.com/langchain-ai/langchain/releases
https://github.com/langchain-ai/langchain-core/releases

# 各子包的迁移指南
https://python.langchain.com/docs/versions/migrating/
```

**查阅 CHANGELOG 的要点**：
1. 关注 "Breaking Changes" / "Deprecated" / "Removed" 部分
2. 检查你正在使用的具体类/函数是否被列出
3. 注意最小依赖版本要求的变化（Python, Pydantic, openai）
4. 查看迁移脚本或修复建议

### 方案 4：完整 RAG 链迁移示例（Before/After）

以下是一个完整 RAG 应用从 0.1.x 迁移到 0.3.x 的完整对比：

**迁移前（LangChain 0.1.x）**：

```python
# ========== 迁移前: LangChain 0.1.x ==========
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import StdOutCallbackHandler

# 文档加载
loader = TextLoader("knowledge.txt")
documents = loader.load()

# 分割
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 嵌入与存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./db"
)

# 构建 QA 链（旧方式）
llm = ChatOpenAI(model_name="gpt-4", max_tokens=1024)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

result = qa.invoke({"query": "什么是 RAG?"})
print(result["result"])
```

**迁移后（LangChain 0.3.x）**：

```python
# ========== 迁移后: LangChain 0.3.x ==========
import os
from langchain_community.vectorstores import Chroma            # 变更 1
from langchain_openai import OpenAIEmbeddings                 # 变更 2
from langchain_openai import ChatOpenAI                       # 变更 3
from langchain_community.document_loaders import TextLoader   # 变更 4
from langchain_community.document_loaders import PyPDFLoader  # 变更 5
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 变更 6
from langchain_core.prompts import ChatPromptTemplate         # 变更 7
from langchain_core.output_parsers import StrOutputParser     # 变更 8
from langchain_core.runnables import RunnablePassthrough      # 变更 9
from langchain_core.runnables import RunnableLambda           # 新增
from langchain_core.callbacks import StdOutCallbackHandler    # 变更 10
from langchain_core.documents import Document                 # 新增

# 文档加载（API 不变）
loader = TextLoader("knowledge.txt")
documents = loader.load()

# 分割（API 不变，仅导入路径变了）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 嵌入与存储（API 不变）
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,             # 或 embedding_function=embeddings
    persist_directory="./db"
)

# 构建 RAG 链（新方式: LCEL 替代 RetrievalQA）
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs: list) -> str:
    """将文档列表格式化为 context 字符串，包含来源信息"""
    return "\n\n".join(
        f"[来源: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

prompt = ChatPromptTemplate.from_template(
    "基于以下上下文回答用户问题。\n\n"
    "上下文：\n{context}\n\n"
    "问题：{question}\n\n"
    "回答："
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # model_name → model

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 调用
result = rag_chain.invoke("什么是 RAG?")
print(result)

# 流式调用
for chunk in rag_chain.stream("什么是 RAG?"):
    print(chunk, end="", flush=True)
```

### 方案 5：依赖版本锁定

为防止意外的版本升级导致代码崩溃，使用精确的依赖锁定：

```bash
# 导出当前所有依赖的精确版本
pip freeze > requirements-lock.txt

# 或使用 pip-tools 进行更精确的依赖管理
pip install pip-tools
pip-compile requirements.in -o requirements-lock.txt

# requirements-lock.txt 示例:
# langchain==0.3.15
# langchain-core==0.3.28
# langchain-community==0.3.14
# langchain-openai==0.2.12
# langchain-text-splitters==0.3.5
# langsmith==0.2.10
# openai==1.60.0
# pydantic==2.10.0
# chromadb==0.6.3
```

## 版本兼容矩阵

### LangChain 核心包版本兼容矩阵

| langchain | langchain-core | langchain-community | langchain-openai | langchain-text-splitters | langsmith | Python | Pydantic |
|-----------|---------------|--------------------|--------------------|--------------------------|-----------|--------|----------|
| 0.0.x (2023) | (内嵌) | (内嵌) | (内嵌) | (内嵌) | (内嵌) | ≥3.8 | v1.x |
| 0.1.0 - 0.1.20 | (内嵌) | ≥0.0.1 | ≥0.0.1 | (内嵌) | ≥0.0.1 | ≥3.8 | v1.x |
| 0.2.0 - 0.2.17 | ≥0.2.0, <0.3.0 | ≥0.2.0, <0.3.0 | ≥0.1.0, <0.3.0 | ≥0.1.0, <0.3.0 | ≥0.1.0 | ≥3.8, <3.13 | v1.x / v2.x |
| 0.3.0 - 0.3.15 | ≥0.3.0, <0.4.0 | ≥0.3.0, <0.4.0 | ≥0.2.0, <0.4.0 | ≥0.3.0, <0.4.0 | ≥0.1.0 | ≥3.10, <3.14 | v2.x |

### 独立子包间的最小版本约束

如果你不使用 `langchain` 主包（推荐），而是直接依赖各子包：

```bash
# 场景 A: 仅需 LCEL（无第三方集成）
pip install langchain-core>=0.3.0

# 场景 B: LCEL + OpenAI
pip install langchain-core>=0.3.0 langchain-openai>=0.2.0

# 场景 C: LCEL + OpenAI + 社区集成
pip install langchain-core>=0.3.0 \
            langchain-openai>=0.2.0 \
            langchain-community>=0.3.0

# 场景 D: 完整栈
pip install langchain-core>=0.3.0 \
            langchain-openai>=0.2.0 \
            langchain-community>=0.3.0 \
            langchain-text-splitters>=0.3.0
```

### 第三方 SDK 版本兼容

| LangChain 版本 | openai (Python SDK) | chromadb | pydantic | tiktoken | httpx |
|---------------|--------------------|----------|----------|----------|-------|
| 0.1.x | ≥1.0.0, <2.0.0 | ≥0.4.0 | ≥1.0, <3.0 | ≥0.4.0 | ≥0.22.0 |
| 0.2.x | ≥1.12.0, <2.0.0 | ≥0.4.22 | ≥1.0, <3.0 | ≥0.5.0 | ≥0.24.0 |
| 0.3.x | ≥1.40.0, <2.0.0 | ≥0.5.0 | ≥2.0, <3.0 | ≥0.7.0 | ≥0.27.0 |

### 常用的锁定安装命令

```bash
# 锁定 0.3.x 最新稳定版（2025 年推荐）
pip install langchain==0.3.15 \
            langchain-core==0.3.28 \
            langchain-community==0.3.14 \
            langchain-openai==0.2.12 \
            langchain-text-splitters==0.3.5

# 锁定 0.2.x 稳定版（兼容 Python 3.8-3.9）
pip install langchain==0.2.17 \
            langchain-core==0.2.38 \
            langchain-community==0.2.17 \
            langchain-openai==0.1.25 \
            langchain-text-splitters==0.2.4

# 开发最新版（不推荐生产使用）
pip install langchain-core --pre -U
```

## 迁移检查清单

在升级 LangChain 版本前，逐项确认以下 5 点：

1. **[ ] 备份当前环境**
   ```bash
   # 导出精确版本的依赖文件
   pip freeze > requirements-lock-$(date +%Y%m%d).txt

   # 如果有 git 仓库，创建迁移分支
   git checkout -b migrate/langchain-0.3.x
   ```

2. **[ ] 在全新隔离环境中测试升级**
   ```bash
   # 创建临时虚拟环境进行升级验证
   python -m venv /tmp/lc-upgrade-test
   source /tmp/lc-upgrade-test/bin/activate  # Windows: \tmp\lc-upgrade-test\Scripts\activate

   # 安装目标版本的包
   pip install langchain==0.3.15 \
               langchain-openai==0.2.12 \
               langchain-community==0.3.14

   # 复制项目代码到临时环境
   cp -r /path/to/your/project/* /tmp/lc-upgrade-test/src/

   # 运行测试
   cd /tmp/lc-upgrade-test/src/
   python -m pytest tests/ -v
   ```

3. **[ ] 修复所有导入语句**
   ```bash
   # 使用 langchain-cli 自动迁移
   pip install langchain-cli
   langchain-cli migrate ./ --yes

   # 手动验证关键导入
   grep -rn "langchain\." --include="*.py" ./ | grep -v langchain_core | grep -v langchain_community | grep -v langchain_openai | grep -v langchain_text
   ```

   重点检查以下模式：
   - `from langchain.vectorstores` → `from langchain_community.vectorstores`
   - `from langchain.chat_models` → `from langchain_openai`
   - `from langchain.embeddings` → `from langchain_openai` 或 `langchain_community.embeddings`
   - `from langchain.schema` → `from langchain_core.messages` 等
   - `from langchain.text_splitter` → `from langchain_text_splitters`
   - `from langchain.callbacks` → `from langchain_core.callbacks`
   - `from langchain.tools` → `from langchain_core.tools`
   - `from langchain.prompts` → `from langchain_core.prompts`

4. **[ ] 替换已废弃的 API**
   - `LLMChain(...)` → `prompt | llm | StrOutputParser()`
   - `ConversationChain(...)` → `RunnableWithMessageHistory`
   - `RetrievalQA.from_chain_type(...)` → LCEL RAG 链
   - `initialize_agent(...)` → `create_react_agent()` (LangGraph) 或手动 AgentExecutor
   - `ConversationBufferMemory(...)` → `ChatMessageHistory` + `RunnableWithMessageHistory`
   - `VectorstoreIndexCreator(...)` → 手动创建 vectorstore
   - `SimpleSequentialChain(...)` → `chain1 | chain2`

5. **[ ] 运行完整测试套件**
   ```bash
   # 运行单元测试
   python -m pytest tests/unit/ -v --tb=short

   # 运行集成测试
   python -m pytest tests/integration/ -v --tb=short

   # 运行端到端测试（如果有）
   python -m pytest tests/e2e/ -v --tb=short

   # 运行带弃用警告的测试（提前发现未来的问题）
   python -W error::DeprecationWarning -m pytest tests/ -v
   ```

   额外验证项：
   - 向量数据库是否正常连接和查询
   - LLM API 调用是否成功（包括流式调用）
   - 回调处理器是否正常触发
   - Token 使用量统计是否正确
   - 错误处理逻辑是否仍然有效

## 版本回滚方案

如果升级后发现无法在合理时间内修复所有问题：

```bash
# === 回滚到 0.1.x（兼容旧代码） ===
pip install langchain==0.1.20 \
            langchain-core==0.1.52 \
            langchain-community==0.0.38 \
            langchain-openai==0.1.23

# === 回滚到 0.2.x（过渡版本） ===
pip install langchain==0.2.17 \
            langchain-core==0.2.38 \
            langchain-community==0.2.17 \
            langchain-openai==0.1.25 \
            langchain-text-splitters==0.2.4

# === 使用之前备份的锁定文件精确回滚 ===
pip install -r requirements-lock-20240601.txt

# === 使用 git 回滚代码 ===
git checkout main  # 回到 main 分支
git branch -D migrate/langchain-0.3.x  # 删除失败的迁移分支
```

## 诊断工具

```python
"""
LangChain 版本诊断工具
运行此脚本快速诊断当前环境的 LangChain 生态版本状态
"""

import sys
import importlib.metadata as metadata
from typing import Optional


def get_version(package_name: str) -> Optional[str]:
    """安全获取包版本"""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def diagnose_langchain_environment():
    """诊断当前 LangChain 生态的版本状态"""

    print("=" * 70)
    print("LangChain 生态版本诊断报告")
    print(f"Python 版本: {sys.version}")
    print(f"日期: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # LangChain 核心包
    core_packages = [
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-openai",
        "langchain-text-splitters",
        "langchain-experimental",
    ]

    print("\n[核心包版本]")
    print(f"{'包名':<35} {'版本':<15} {'状态'}")
    print("-" * 60)

    for pkg in core_packages:
        version = get_version(pkg)
        if version:
            major = int(version.split(".")[0])
            minor = int(version.split(".")[1]) if len(version.split(".")) > 1 else 0
            if major >= 0 and minor >= 3:
                status = "最新"
            elif major >= 0 and minor >= 2:
                status = "过渡版本"
            else:
                status = "旧版本 - 建议升级"
            print(f"  {pkg:<33} {version:<15} {status}")
        else:
            print(f"  {pkg:<33} {'未安装':<15} --")

    # 扩展集成包
    extra_packages = [
        "langchain-chroma",
        "langchain-pinecone",
        "langchain-qdrant",
        "langchain-huggingface",
        "langgraph",
        "langsmith",
    ]

    print("\n[扩展集成包]")
    print(f"{'包名':<35} {'版本':<15} {'备注'}")
    print("-" * 60)

    for pkg in extra_packages:
        version = get_version(pkg)
        if version:
            print(f"  {pkg:<33} {version:<15} 已安装")
        else:
            print(f"  {pkg:<33} {'未安装':<15} (按需安装)")

    # 关键依赖
    dependencies = [
        "openai",
        "pydantic",
        "chromadb",
        "tiktoken",
        "httpx",
        "numpy",
        "sqlalchemy",
    ]

    print("\n[关键依赖版本]")
    print(f"{'依赖':<35} {'版本':<15}")
    print("-" * 60)

    for dep in dependencies:
        version = get_version(dep)
        if version:
            print(f"  {dep:<33} {version:<15}")
        else:
            print(f"  {dep:<33} {'未安装':<15}")

    # 兼容性检查
    print("\n[兼容性检查]")

    langchain_ver = get_version("langchain")
    core_ver = get_version("langchain-core")

    if langchain_ver and core_ver:
        lc_major = int(langchain_ver.split(".")[0])
        lc_minor = int(langchain_ver.split(".")[1])
        core_major = int(core_ver.split(".")[0])
        core_minor = int(core_ver.split(".")[1])

        if lc_major != core_major:
            print(f"  警告: langchain ({langchain_ver}) 和 langchain-core ({core_ver}) 主版本不匹配！")
        elif abs(lc_minor - core_minor) > 1:
            print(f"  警告: langchain ({langchain_ver}) 和 langchain-core ({core_ver}) 次版本差距较大")
        else:
            print(f"  langchain 和 langchain-core 版本兼容")

    # Python 版本检查
    py_version = sys.version_info
    if py_version < (3, 10) and langchain_ver:
        lc_minor = int(langchain_ver.split(".")[1]) if langchain_ver else 0
        if lc_minor >= 3:
            print(f"  错误: LangChain 0.3.x 需要 Python ≥ 3.10，当前: {py_version.major}.{py_version.minor}")
    else:
        print(f"  Python 版本满足要求 ({py_version.major}.{py_version.minor})")

    print("=" * 70)


if __name__ == "__main__":
    diagnose_langchain_environment()
```

## 总结

> **核心原则**：
> 1. **永远在隔离环境中测试升级** -- 不要在生产环境直接升级大版本
> 2. **使用 `langchain-cli migrate`** -- 自动处理 80% 的导入路径变更
> 3. **关注弃用警告** -- 运行时添加 `-W error::DeprecationWarning` 将警告转为错误
> 4. **锁定精确版本** -- 使用 `==X.Y.Z` 而非 `>=X.Y`，防止意外升级
> 5. **逐包升级** -- 先升级 `langchain-core`，确认无问题后再升级 `langchain-openai` 等集成包
>
> LangChain 的包拆分是工程化的必然选择，虽然一次性迁移成本不低，但长期收益（更快的安装速度、更少的依赖冲突、更清晰的 API 边界）是显著的。
