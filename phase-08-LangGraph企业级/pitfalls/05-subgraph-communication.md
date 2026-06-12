# 坑点 #5：子图通信断裂（Subgraph Communication Breakdown）

> "Master图期望子图返回'rag_answer'字段，但子图的State定义里叫'answer'。两个图编译都通过了，但子图的输出从未出现在Master的State中。"

---

## 症状

1. **子图输出丢失**：子图执行了但结果没有传递回Master图
2. **字段名不一致**：Master叫 `rag_answer`，子图叫 `answer`
3. **State映射错误**：input/output key配置错误，数据传到了错误的字段
4. **子图内部状态污染Master状态**：子图的内部字段污染了Master State

---

## 根因

LangGraph子图的State独立于Master State。如果State字段名不匹配，子图的输出字段会被LangGraph忽略，只将匹配的字段合并回Master State。

```
Master State: {query, rag_answer, tool_result, ...}
RAG Sub State: {query, answer, docs, ...}

rag_answer ← 存在于Master，但子图输出的是 answer
answer    ← 子图输出这个，但Master State中没有这个字段
→ result: rag_answer 永远为空！
```

---

## 解法

### 1. 显式字段映射

```python
def subgraph_entry_with_mapping(master_state: MasterState) -> dict:
    """
    子图入口：显式地将Master State映射到子图State，
    子图执行完成后显式地将结果映射回Master State。
    """
    # === 输入映射：Master → Sub ===
    sub_input = {
        "query": master_state['user_query'],       # 字段名转换
        "docs": master_state.get('retrieved_docs', []),
        "answer": "",                               # 子图内部字段
        "validation": "",
    }

    # 执行子图
    sub_graph = build_rag_subgraph()
    sub_result = sub_graph.invoke(sub_input)

    # === 输出映射：Sub → Master ===
    return {
        "rag_answer": sub_result.get('answer', ''),  # 关键：字段名转换！
        "retrieved_docs": sub_result.get('docs', []),
        "rag_validation": sub_result.get('validation', 'fail'),
    }
```

### 2. 共享State定义

```python
# 最佳实践：Master和Sub共享一个基础State

class BaseAgentState(TypedDict):
    """所有子图的基础State字段。"""
    user_query: str
    messages: Annotated[list, add_messages]
    final_response: str


class RAGSubState(BaseAgentState):
    """RAG子图扩展字段。"""
    retrieved_docs: list
    rag_answer: str  # 与Master使用相同的字段名！
    rag_validation: str


class MasterState(BaseAgentState):
    """Master图扩展字段。"""
    intent: str
    # RAG字段（与RAGSubState中相同）
    retrieved_docs: list
    rag_answer: str
    rag_validation: str
    # Tool字段
    tool_result: str
    # Approval字段
    approval_status: str


# 现在Master和RAG子图共享相同的字段名：
# Master.rag_answer ← 直接匹配 → RAGSubState.rag_answer ✅
```

### 3. 字段对齐验证

```python
class SubgraphValidator:
    """验证Master和子图之间的字段对齐。"""

    @staticmethod
    def check_field_alignment(
        master_state_class,
        sub_state_class,
        subgraph_name: str,
    ) -> list[str]:
        """
        检查Master和子图之间应该共享的字段是否一致。
        """
        master_fields = set(master_state_class.__annotations__.keys())
        sub_fields = set(sub_state_class.__annotations__.keys())

        issues = []

        # 子图输出但Master没有的字段 → 数据丢失
        output_only_in_sub = sub_fields - master_fields
        if output_only_in_sub:
            issues.append(
                f"[{subgraph_name}] 子图字段不在Master中 → 数据将丢失: "
                f"{output_only_in_sub}"
            )

        # Master有但子图没有的字段 → 子图无法使用这些数据
        input_only_in_master = master_fields - sub_fields
        needed_by_sub = {
            'user_query', 'messages', 'retrieved_docs', 'tool_name',
            'approval_status', 'rag_answer'
        }
        missing_needed = input_only_in_master & needed_by_sub
        if missing_needed:
            issues.append(
                f"[{subgraph_name}] 子图需要但缺少的字段: {missing_needed}"
            )

        return issues

    @staticmethod
    def validate_all_subgraphs(master_class, subs: dict[str, type]) -> list[str]:
        """验证所有子图。"""
        all_issues = []
        for name, sub_class in subs.items():
            issues = SubgraphValidator.check_field_alignment(
                master_class, sub_class, name
            )
            all_issues.extend(issues)
        return all_issues
```

---

## 检查清单

- [ ] Master和所有子图共享相同的字段名（通过继承基类State）
- [ ] 子图入口有显式的输入/输出映射注释
- [ ] 运行了字段对齐验证（CI中自动检查）
- [ ] 端到端测试验证了子图输出能正确传递到Master的响应中
- [ ] 子图内部字段不污染Master State（使用独立前缀区分）

---

**一句话总结**：Master和Sub的State字段名不一致，就像两个说不同语言的人在对话——他们都在说话，但没人理解对方。要么统一语言，要么配一个翻译。
