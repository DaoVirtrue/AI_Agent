# 序列化问题：链的保存和加载故障

## 症状

链在开发环境中正常运行，但在以下情况下失败：

1. 从磁盘加载已保存的链后
2. 升级 LangChain 库版本后加载旧链
3. 将链从一台机器迁移到另一台
4. CI/CD 流水线中重建环境后

典型错误：

```python
# 从磁盘加载链
from langchain_core.load import load

chain = load("saved_chain.json")

# TypeError: __init__() got an unexpected keyword argument 'model_name'
# PicklingError: Can't pickle <class 'openai.Embedding'>: ...
# ModuleNotFoundError: No module named 'langchain.chains'
```

## 根因分析

LangChain 链的序列化涉及多个层面的问题：

1. **版本不匹配**：序列化的链包含特定版本的类引用，库升级后类签名可能变化
2. **Pickle 依赖**：部分组件使用 pickle 序列化，pickle 文件与 Python 版本和库版本耦合
3. **动态对象**：API Key、连接池等运行时对象无法被可靠序列化
4. **导入路径迁移**：LangChain 0.1.x 到 0.2.x+ 进行了大量包重组

## 真实场景

```python
# 场景: 用 LangChain 0.1.16 保存链
# langchain==0.1.16
# langchain-openai==0.0.5
# langchain-chroma==0.0.7

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# 构建并保存
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=Chroma(...).as_retriever(),
)
chain.save("qa_chain.json")

# --- 几个星期后，升级了库 ---
# langchain==0.2.1          (升级)
# langchain-openai==0.1.0   (升级)
# langchain-chroma==0.1.0   (升级)

# 尝试加载
from langchain_core.load import load

chain = load("qa_chain.json")
# 错误! ChatOpenAI 的 __init__ 参数从 'model_name' 变成了 'model'
# 或者 Chroma 的序列化格式变了
```

## 解决方案 1：使用 LangGraph 的 JSON 序列化状态（推荐）

不序列化链本身，而是序列化链的**配置**（JSON 安全），在加载时重建链。

```python
import json
from typing import Dict, Any
from langgraph.graph import StateGraph, MessagesState

# === 保存时：只保存配置，不保存链 ===
def save_chain_config(config: Dict[str, Any], filepath: str):
    """保存链的配置（JSON 安全），而非链对象本身"""
    config_to_save = {
        "version": "1.0.0",
        "langchain_version": "0.2.0",
        "components": {
            "llm": {
                "provider": "openai",
                "model": config.get("model", "gpt-4o"),
                "temperature": config.get("temperature", 0.0),
            },
            "embeddings": {
                "provider": "openai",
                "model": "text-embedding-3-small",
            },
            "vector_store": {
                "provider": "chroma",
                "collection_name": config.get("collection_name", "default"),
                "persist_directory": config.get("persist_directory", "./chroma_db"),
            },
            "retriever": {
                "search_type": config.get("search_type", "similarity"),
                "k": config.get("k", 4),
            },
        },
        "chains": {
            "rag_prompt_template": (
                "基于以下上下文回答问题:\n\n"
                "{context}\n\n"
                "问题: {question}\n\n"
                "回答:"
            ),
        },
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)


# === 加载时：从配置重建链 ===
def load_chain_from_config(filepath: str):
    """从 JSON 配置文件重建链"""
    with open(filepath, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    components = config["components"]
    
    # 重建 LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=components["llm"]["model"],
        temperature=components["llm"]["temperature"],
    )
    
    # 重建嵌入
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model=components["embeddings"]["model"])
    
    # 重建向量存储
    from langchain_chroma import Chroma
    vectorstore = Chroma(
        collection_name=components["vector_store"]["collection_name"],
        embedding_function=embeddings,
        persist_directory=components["vector_store"]["persist_directory"],
    )
    
    # 重建检索器
    retriever = vectorstore.as_retriever(
        search_type=components["retriever"]["search_type"],
        search_kwargs={"k": components["retriever"]["k"]},
    )
    
    # 重建链
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    prompt = ChatPromptTemplate.from_template(
        config["chains"]["rag_prompt_template"]
    )
    
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain
```

## 解决方案 2：使用 LangGraph Checkpoint 持久化状态

```python
# LangGraph 的内置 checkpointing 使用 JSON 序列化状态
# 不序列化链的代码，只序列化链在运行时的状态

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

# 创建带 checkpoint 的图
checkpointer = MemorySaver()

# 定义图结构
workflow = StateGraph(MessagesState)
# ... 添加节点和边 ...

# 编译时将 checkpointer 传入
app = workflow.compile(checkpointer=checkpointer)

# 执行并保存 checkpoint
config = {"configurable": {"thread_id": "conversation_001"}}
result = app.invoke({"messages": [("user", "你好")]}, config=config)

# --- 跨会话恢复 ---
# 使用相同的 thread_id 恢复状态
config2 = {"configurable": {"thread_id": "conversation_001"}}
# state 中包含之前的消息历史
state = app.get_state(config2)
print(f"历史消息数: {len(state.values.get('messages', []))}")

# 继续对话
result2 = app.invoke(
    {"messages": [("user", "再详细解释一下")]},
    config=config2
)
```

## 解决方案 3：固定库版本 (requirements.txt)

```txt
# requirements.txt - 固定所有相关库的版本
langchain==0.2.14
langchain-openai==0.1.20
langchain-chroma==0.1.3
langchain-community==0.2.12
langchain-core==0.2.36
langgraph==0.2.5
langsmith==0.1.85
chromadb==0.5.5
pydantic==2.8.2
pydantic-core==2.20.1
```

配合使用 `pip freeze`：

```bash
# 保存当前所有库的精确版本
pip freeze > requirements-lock.txt

# 在新环境中安装精确版本
pip install -r requirements-lock.txt
```

## 从 LangChain 0.1.x 到 0.2.x 的序列化迁移路径

```python
# 旧方式 (0.1.x) - 使用 chain.save() / load_chain()
# 新方式 (0.2.x) - 使用 langchain_core.load

# 如果你有旧的序列化文件，使用迁移工具
# langchain-cli 提供迁移命令

# 手动迁移步骤：
# 1. 读取旧的序列化文件
# 2. 提取配置信息（model, prompt template, retriever params）
# 3. 手动重建链（参考解决方案1）
# 4. 保存为新的 JSON 配置格式
# 5. 测试重建后的链是否与原始行为一致
```

## 最佳实践：使用 model.register() 注册自定义组件

```python
# 如果你有自定义的 Runnable 子类需要被序列化

from langchain_core.runnables import RunnableSerializable
from langchain_core.load import load, dumpd

# 在保存前注册你的自定义类
# 这样加载时 langchain_core 知道如何反序列化

# 注意: 大多数情况下，使用配置驱动的方法（方案1）比序列化更可靠
```

## 检查清单

1. **[ ] 库版本一致性**：开发环境和生产环境使用相同的 `requirements.txt`（或 `poetry.lock`/`Pipfile.lock`）
2. **[ ] 优先使用配置重建**：保存链的配置参数（JSON 格式），在运行时重建链，而非序列化链对象本身
3. **[ ] API Key 外部化**：永远不要将 API Key 序列化到文件中。使用环境变量或密钥管理服务
4. **[ ] 版本标记**：在配置文件中包含 `langchain_version` 字段，加载时可以检查兼容性
5. **[ ] 测试加载流程**：定期测试 "从零重建环境 -> 加载配置 -> 运行链" 流程，确保序列化方案始终有效

```python
# 集成测试: 验证链的保存和加载
def test_chain_reproducibility():
    """验证从配置重建的链与原始链行为一致"""
    
    # 1. 构建原始链
    original_chain = build_chain()
    original_result = original_chain.invoke("测试查询")
    
    # 2. 保存配置
    save_chain_config(config_data, "/tmp/test_chain_config.json")
    
    # 3. 从配置重建
    rebuilt_chain = load_chain_from_config("/tmp/test_chain_config.json")
    rebuilt_result = rebuilt_chain.invoke("测试查询")
    
    # 4. 验证结果一致
    assert original_result == rebuilt_result, (
        f"链重建后结果不一致!\n"
        f"原始: {original_result}\n"
        f"重建: {rebuilt_result}"
    )
    
    print("✓ 链序列化/反序列化验证通过")
```
