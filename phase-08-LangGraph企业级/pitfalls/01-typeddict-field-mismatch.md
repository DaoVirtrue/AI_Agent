# 坑点 #1：TypedDict 字段不匹配

> "定义了6个字段的State，节点A更新了'confidence'，节点B期望'confidance'。运行时没有类型检查，KeyError在6个节点后爆发——但错误信息指向了节点B，让你花了2小时调试节点A。"

---

## 症状

1. **运行时KeyError**：节点尝试读取 `state['field_name']` 时报 KeyError
2. **字段拼写错误**：`confidence` vs `confidance`，`retrieved_docs` vs `retrievedDocs`
3. **字段被静默忽略**：节点返回了State中不存在的字段，无警告
4. **Reducers不工作**：`Annotated[list, add_messages]` 被写成 `list[Message]`，失去自动合并行为

---

## 根因

TypedDict 在 Python 中是**运行时不强制验证**的类型注解。错误的字段名在静态分析中可以被捕获（mypy/pyright），但在运行时静默通过。

---

## 解法

### 1. State定义集中管理 + 运行前验证

```python
import inspect
from typing import get_type_hints

class StateValidator:
    """在编译Graph之前验证State定义和节点返回值的一致性。"""

    @staticmethod
    def validate_state_fields(state_class, node_functions: dict[str, callable]) -> list[str]:
        """检查所有节点的返回字段是否在State中定义。"""
        state_fields = set(get_type_hints(state_class).keys())
        issues = []

        for node_name, func in node_functions.items():
            # 检查节点函数的返回类型注解
            return_hints = get_type_hints(func)
            if 'return' in return_hints:
                return_type = return_hints['return']
                # 这里可以进一步检查 dict 的 key 类型
                # ...

            # 运行时模拟：检查节点是否会返回未定义的字段
            # （这是一个启发式检查，不能100%覆盖）
            source = inspect.getsource(func)
            # 检查源代码中可能返回的字段
            for field_pattern in ['"', "'"]:
                # 简单检查：查找 return {"field_name": ...} 模式
                pass

        return issues

    @staticmethod
    def create_state_accessor(state_class):
        """
        创建类型安全的State访问器。
        提供带自动补全和拼写检查的State访问。
        """
        fields = list(get_type_hints(state_class).keys())

        class StateAccessor:
            def __init__(self, state):
                self._state = state

            def __getattr__(self, name):
                if name not in fields and not name.startswith('_'):
                    # 检查是否有相似的字段名（拼写错误检测）
                    import difflib
                    close = difflib.get_close_matches(name, fields, n=1, cutoff=0.7)
                    suggestion = f"。您是否想用 '{close[0]}'?" if close else ""
                    raise AttributeError(
                        f"State 中没有字段 '{name}'{suggestion}。"
                        f"可用字段: {fields}"
                    )
                return self._state.get(name)

        return StateAccessor
```

### 2. 使用 Pydantic BaseModel 替代 TypedDict（带运行时验证）

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class RAGState(BaseModel):
    """
    使用 Pydantic BaseModel 替代 TypedDict。
    获得运行时类型验证、默认值、字段描述。
    """
    question: str = Field(..., min_length=1, description="用户问题")
    retrieved_docs: list = Field(default_factory=list)
    answer: str = Field(default="")
    validation_result: str = Field(default="pass")
    retry_count: int = Field(default=0, ge=0, le=10)

    @field_validator("retry_count")
    @classmethod
    def validate_retry(cls, v):
        if v < 0:
            raise ValueError("retry_count 不能为负数")
        return v

# 使用方式：
# state = RAGState(question="什么是LangGraph?")
# state.answer = "..."  # 类型安全
# state.retry_count = -1  # ❌ ValidationError 立即抛出
```

---

## 检查清单

- [ ] State 定义集中在一个地方，不在多处重复
- [ ] 使用 mypy/pyright 进行静态类型检查（CI中强制执行）
- [ ] 在Graph编译前运行State字段验证
- [ ] 考虑使用Pydantic BaseModel获得运行时验证
- [ ] 所有节点函数的返回类型都有注解
- [ ] 添加了集成测试：验证所有节点返回的字段都在State中

---

**一句话总结**：TypedDict 只是编译时的"善意提醒"，运行时不会帮你检查任何东西。花30分钟写State验证，可以省下未来30小时的调试时间。
