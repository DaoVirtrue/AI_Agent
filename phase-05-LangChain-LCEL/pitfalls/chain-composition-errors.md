# 链组合错误：输入/输出模式不兼容

## 症状

尝试组合两个 LCEL 组件时，抛出以下类型错误：

```python
TypeError: Expected str, got <class 'langchain_core.documents.Document'>
```

或

```python
KeyError: 'answer'  # 期望的键在输入中不存在
```

## 根因分析

LCEL 链通过 `|` 运算符连接，前一个组件的**输出类型**必须与后一个组件的**输入类型**兼容。当类型不匹配时，LangChain 在运行时抛出错误。

常见的不兼容模式：

| 前一个组件输出 | 后一个组件期望 | 结果 |
|---|---|---|
| `dict` with key `"text"` | `dict` with key `"answer"` | KeyError: 'answer' |
| `List[Document]` | `str` | TypeError |
| `str` | `dict` | AttributeError |
| `AIMessage` 对象 | `str` | 静默失败（输出格式错） |

## 真实场景

### 场景1：将检索器直接接到 Prompt 模板

```python
# 错误示例
prompt = ChatPromptTemplate.from_messages([
    ("human", "基于以下内容回答: {context}\n\n问题: {question}")
])

retriever = vectorstore.as_retriever()

# 期望: retriever 的输出自动变成 prompt 的 context 变量
# 实际: retriever 返回 List[Document], 而 prompt 需要 dict with str
chain = retriever | prompt  # TypeError!
```

错误信息类似：
```
TypeError: Expected a dict input but got List[Document]
```

### 场景2：自定义 Retriever 输出格式不匹配

```python
class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query):
        # 错误: 返回了 dict 而非 List[Document]
        return {"documents": [...], "query": query}

# 当尝试将其连接到 Prompt 中的 context 变量时
retriever = MyRetriever()
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt
# 期望 retriever 返回的 List[Document] 能在 prompt 中格式化
# 但 retriever 返回的是 dict，导致 context 是 {"documents": [...], "query": ...}
```

## 解决方案 1：使用 RunnableLambda 进行类型转换

```python
from langchain_core.runnables import RunnableLambda

# 将 List[Document] 转换为 str
def format_docs(docs: list) -> str:
    """将文档列表格式化为单个字符串"""
    return "\n\n".join(
        f"[来源: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

# 在链中插入转换器
chain = (
    retriever                    # 输出: List[Document]
    | RunnableLambda(format_docs) # 转换为: str
    | prompt                     # 将 str 作为 context 变量
    | llm
)

# 或者更完整的版本，构建 dict 输入
from langchain_core.runnables import RunnablePassthrough

def build_prompt_input(query: str, docs: list) -> dict:
    return {
        "context": "\n\n".join(d.page_content for d in docs),
        "question": query,
    }

chain_with_context = (
    RunnablePassthrough.assign(
        context=retriever | RunnableLambda(format_docs)
    )
    | prompt
    | llm
)
```

## 解决方案 2：使用 itemgetter 和字典解构

```python
from operator import itemgetter

# 问题: Chain A 输出 {"result": "text", "confidence": 0.95}
# Chain B 需要输入 "text" (str)

# 方案: 使用 itemgetter 提取特定键
chain_a = some_chain  # 输出: {"result": "text", "confidence": 0.95}

chain_b = (
    {"text": itemgetter("result")}  # 从 Chain A 输出中提取 result
    | second_prompt
    | llm
)

full_chain = chain_a | chain_b
```

实际示例 - 完整的 RAG 链数据流：

```python
# RAG 链的标准模式
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        # RunnablePassthrough.assign() 可以在保留原始输入的同时添加新字段
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

## 解决方案 3：使用 .with_types() 验证输入输出类型

```python
from langchain_core.runnables import RunnableLambda

# 添加类型断言
def assert_output_is_str(value):
    """断言输出是字符串，否则抛出明确的错误"""
    if not isinstance(value, str):
        raise TypeError(
            f"Expected str output, got {type(value).__name__}: {str(value)[:100]}"
        )
    return value

def assert_output_is_list_of_documents(value):
    """断言输出是 Document 列表"""
    from langchain_core.documents import Document
    if not isinstance(value, list):
        raise TypeError(f"Expected List[Document], got {type(value).__name__}")
    if value and not isinstance(value[0], Document):
        raise TypeError(
            f"Expected List[Document], got List[{type(value[0]).__name__}]"
        )
    return value

# 在链中插入类型检查（仅在开发时）
safe_chain = (
    retriever
    | RunnableLambda(assert_output_is_list_of_documents)
    | RunnableLambda(format_docs)
    | RunnableLambda(assert_output_is_str)
    | prompt
    | llm
)
```

## 常见组合模式速查

### RAG 三明治 (The RAG Sandwich)

```python
# 最常见的 RAG 链组合模式
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

数据流：
```
{"question": "..."} 
  -> {"context": "formatted docs", "question": "..."}
  -> ChatPromptValue (messages)
  -> AIMessage (response)
  -> str (answer)
```

### 检索-重排序-生成链

```python
# 更复杂的链: 检索 -> 重排序 -> 生成
reranker = CohereRerank()  # 或其他重排序器

complex_rag = (
    {
        "context": (
            itemgetter("question")
            | retriever                    # List[Document] (k=20)
            | RunnableLambda(lambda docs: reranker.compress_documents(
                documents=docs,
                query=itemgetter("question")  # 注意这里需要 query
            ))                             # List[Document] (top k)
            | format_docs                   # str
        ),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
)
```

### 多步检索链

```python
# 第一步: 用 LLM 生成搜索查询
# 第二步: 检索
# 第三步: 用检索结果生成回答

query_generation_prompt = ChatPromptTemplate.from_template(
    "为以下问题生成一个适合搜索的关键词查询: {question}\n\n关键词查询:"
)

multistep_chain = (
    {
        "question": RunnablePassthrough(),
        "search_query": (
            query_generation_prompt | llm | StrOutputParser()
        ),
    }
    | RunnablePassthrough.assign(
        context=lambda d: format_docs(
            retriever.invoke(d["search_query"])
        )
    )
    | prompt
    | llm
    | StrOutputParser()
)
```

## 检查清单

1. **[ ] 确认每个组件的输出类型**：阅读组件文档，确认其 `invoke()` 方法的返回类型
2. **[ ] 确认每个组件的输入类型**：阅读下一组件的文档，确认其期望的输入格式
3. **[ ] 中间类型转换**：当 Retriever (List[Document]) 连接到 Prompt (dict) 时，使用 RunnableLambda 或 format 函数做转换
4. **[ ] 字典键匹配**：确保前一个组件输出的字典键与下一个组件期望的输入键完全一致（包括大小写）
5. **[ ] 测试每个子链**：在组合前单独测试每个组件，确认输入输出类型

```python
# 快速类型检查工具
def check_chain_io(step_name, input_data, output_data, next_step_name, expected_input_type):
    """检查链步骤的输入输出类型兼容性"""
    print(f"检查: {step_name} -> {next_step_name}")
    print(f"  输出类型: {type(output_data).__name__}")
    print(f"  期望输入类型: {expected_input_type.__name__}")
    
    if isinstance(output_data, expected_input_type):
        print("  ✓ 类型匹配")
    else:
        print(f"  ✗ 类型不匹配! {step_name} 输出 {type(output_data).__name__}")
        print(f"    但 {next_step_name} 期望 {expected_input_type.__name__}")
```
