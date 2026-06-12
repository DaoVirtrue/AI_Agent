# LCEL 链调试指南：应对中间步骤输出错误

## 症状

链成功运行（无异常抛出），但最终输出不符合预期。例如：

- RAG 链的 LLM 回答与上下文无关，看似在"编造"
- 答案中缺少关键信息，明明文档库中有相关内容
- 输出格式正确但内容错误

## 根因分析

LCEL 链通过 `|` (管道) 运算符组合，中间数据流是不透明的。不像传统编程可以 `print()` 或设置断点，LCEL 链的中间值很难被观察到。

**核心问题**：当 `chain.invoke()` 返回错误结果时，无法知道是哪一步出了问题。

## 真实场景

一个 RAG 链配置如下：

```python
# 有问题的链
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("什么是向量数据库？")
# 输出: "抱歉，我无法回答这个问题..."
# 预期: 应该从文档库中检索到相关内容并回答

# 问题出在哪？可能是:
# 1. retriever 检索到了空结果（配置错误）
# 2. prompt 模板格式不对
# 3. llm 参数设置有问题
```

根本原因是 `retriever` 配置错误（`search_type="similarity_score_threshold"` 且 `score_threshold=0.9` 太高），导致检索返回空文档列表，模型没有上下文只能回复"不知道"。

## 解决方案 1：使用 RunnablePassthrough.assign() 捕获中间值

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 调试辅助函数
def debug_print(name: str):
    """创建一个打印中间值的 Runnable"""
    def _print_and_pass(data):
        print(f"\n=== {name} ===")
        if isinstance(data, dict):
            for k, v in data.items():
                preview = str(v)[:200] if v else "(空)"
                print(f"  {k}: {preview}")
        else:
            print(f"  {str(data)[:200]}")
        return data
    return RunnableLambda(_print_and_pass)

# 在链中插入调试节点
debuggable_chain = (
    RunnablePassthrough()                          # 步骤0: 输入
    | debug_print("步骤0: 用户输入")
    | RunnablePassthrough.assign(
        context=retriever                          # 步骤1: 检索
    )
    | debug_print("步骤1: 检索结果")
    | RunnablePassthrough.assign(
        formatted_context=lambda d: "\n\n".join(
            doc.page_content for doc in d["context"]
        )
    )
    | debug_print("步骤2: 格式化后的上下文")
    | prompt
    | debug_print("步骤3: 完整提示词")
    | llm
    | debug_print("步骤4: LLM 原始输出")
    | StrOutputParser()
    | debug_print("步骤5: 最终输出")
)

# 现在可以清楚看到每一步的数据
result = debuggable_chain.invoke("什么是向量数据库？")
```

## 解决方案 2：附加调试回调记录所有步骤

```python
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List
import json

class StepByStepDebugCallback(BaseCallbackHandler):
    """逐步调试回调：记录链中每个组件的输入输出"""
    
    def __init__(self):
        self.step_counter = 0
        self.history = []
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.step_counter += 1
        step = {
            "step": self.step_counter,
            "name": serialized.get("name", serialized.get("id", ["?"])[-1]),
            "event": "start",
            "inputs": self._safe_serialize(inputs),
        }
        self.history.append(step)
        print(f"[步骤{self.step_counter}] {step['name']} 开始")
        print(f"  输入: {json.dumps(step['inputs'], ensure_ascii=False, default=str)[:200]}")
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        last_step = self.history[-1] if self.history else {}
        name = last_step.get("name", "?")
        serialized_output = self._safe_serialize(outputs)
        print(f"[步骤{self.step_counter}] {name} 完成")
        print(f"  输出: {json.dumps(serialized_output, ensure_ascii=False, default=str)[:200]}")
        print("-" * 60)
    
    def _safe_serialize(self, obj):
        """安全序列化（处理 Document 等复杂对象）"""
        if isinstance(obj, dict):
            return {
                k: self._safe_serialize(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            if len(obj) > 3:
                return [self._safe_serialize(obj[0])] + [
                    f"... ({len(obj) - 2} more items) ..."
                ] + [self._safe_serialize(obj[-1])]
            return [self._safe_serialize(item) for item in obj]
        elif hasattr(obj, 'page_content'):
            return f"Document(content='{obj.page_content[:100]}...', metadata={obj.metadata})"
        elif hasattr(obj, 'content'):
            return f"Message(content='{obj.content[:100]}...')"
        else:
            return str(obj)[:200]

# 使用调试回调
debug_handler = StepByStepDebugCallback()
result = chain.invoke(input, config={"callbacks": [debug_handler]})

# 如果结果不对，查看记录的历史
for step in debug_handler.history:
    print(f"步骤{step['step']}: {step['name']} - {step['event']}")
```

## 解决方案 3：拆分子链并单独测试

```python
# 将 RAG 链分解为独立可测试的子链

# 子链1: 检索
retrieval_chain = retriever

# 子链2: 格式化
format_chain = RunnableLambda(
    lambda docs: "\n\n".join(d.page_content for d in docs)
)

# 子链3: 提示词构造
prompt_chain = prompt

# 子链4: LLM 生成
generation_chain = llm | StrOutputParser()

# 逐一测试
print("=== 测试检索 ===")
docs = retrieval_chain.invoke("什么是向量数据库？")
print(f"检索到 {len(docs)} 个文档")
assert len(docs) > 0, "检索返回空！检查检索器配置"

print("\n=== 测试格式化 ===")
formatted = format_chain.invoke(docs)
print(f"格式化后长度: {len(formatted)} 字符")
assert len(formatted) > 50, "格式化后上下文太短"

print("\n=== 测试提示词 ===")
filled_prompt = prompt_chain.invoke({
    "context": formatted,
    "question": "什么是向量数据库？"
})
print(f"提示词总长度: {len(str(filled_prompt))} 字符")

print("\n=== 测试生成 ===")
answer = generation_chain.invoke(filled_prompt)
print(f"回答: {answer[:200]}...")

# 全部通过后，再组合
```

## 解决方案 4：使用 .with_fallbacks() 优雅降级

```python
from langchain_core.runnables import RunnableLambda

# 备用检索方案
def fallback_retrieval(query: str):
    """当主检索器返回空时，使用关键词匹配兜底"""
    # 简单的关键词匹配
    keywords = query.split()
    results = []
    for doc in all_documents:
        score = sum(1 for kw in keywords if kw in doc.page_content)
        if score > 0:
            results.append(doc)
    return sorted(results, key=lambda d: sum(
        1 for kw in keywords if kw in d.page_content
    ), reverse=True)[:3]

fallback_retriever = RunnableLambda(fallback_retrieval)

# 带降级的检索器
robust_retriever = retriever.with_fallbacks([
    fallback_retriever
])

# 现在即使主检索器返回空，也会使用备用方案
```

## 调试检查清单

1. **[ ] 检查检索器输出**：`retriever.invoke(query)` 是否返回非空文档列表？文档内容是否相关？
2. **[ ] 检查提示词格式**：打印 `prompt.invoke({"context": ..., "question": ...})` 的结果，确认占位符被正确替换
3. **[ ] 检查 LLM 输入**：提示词中是否有截断、编码问题或特殊字符？
4. **[ ] 检查 LLM 输出格式**：`llm.invoke(messages)` 的原始输出是否符合预期？是否被 `StrOutputParser` 意外截断？
5. **[ ] 检查类型兼容性**：链的每一步的输出类型是否与下一步的输入类型匹配？（见 chain-composition-errors.md）
6. **[ ] 检查数值参数**：`temperature`, `top_p`, `k` 等参数是否合理？过高/过低可能影响输出质量

## 完整的内联调试代码示例

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def create_self_debuggable_chain(retriever, llm, prompt):
    """
    创建自带调试功能的 RAG 链。
    
    在开发阶段使用此链替代正式链，可以清楚看到每个中间步骤。
    注意：切换到生产环境时应移除此包装。
    """
    
    def log_context(d):
        """记录检索到的上下文"""
        docs = d.get("context", [])
        print(f"\n{'='*60}")
        print(f"检索到 {len(docs)} 个文档:")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc.page_content[:100]}...")
        if len(docs) == 0:
            print("  ⚠️  警告: 未检索到任何文档！")
        print(f"{'='*60}\n")
        return d
    
    def log_prompt(messages):
        """记录完整的提示词"""
        print(f"\n{'='*60}")
        print(f"完整提示词 ({sum(len(str(m.content)) for m in messages)} 字符):")
        for i, msg in enumerate(messages):
            role = getattr(msg, 'type', 'unknown')
            print(f"  [{role}]: {str(msg.content)[:200]}...")
        print(f"{'='*60}\n")
        return messages
    
    def log_answer(text):
        """记录最终答案"""
        print(f"\n{'='*60}")
        print(f"最终答案 ({len(text)} 字符):")
        print(f"  {text[:200]}...")
        print(f"{'='*60}\n")
        return text
    
    return (
        RunnablePassthrough.assign(context=retriever)
        | RunnableLambda(log_context)
        | RunnablePassthrough.assign(
            formatted_context=lambda d: "\n\n".join(
                doc.page_content for doc in d["context"]
            )
        )
        | prompt
        | RunnableLambda(log_prompt)
        | llm
        | RunnablePassthrough.assign(
            raw_answer=lambda d: d.content
        )
        | RunnableLambda(lambda d: d["content"])
        | RunnableLambda(log_answer)
    )

# 使用
debug_chain = create_self_debuggable_chain(retriever, llm, prompt)
answer = debug_chain.invoke("你的问题")
```
