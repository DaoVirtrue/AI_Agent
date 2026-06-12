# 坑点 #2：Checkpointer 序列化失败

> "一切测试正常——直到你重启了服务。PostgreSQL的checkpoint表里确实有数据，但LangGraph无法反序列化，因为你的State里有lambda函数和threading.Lock对象。"

---

## 症状

1. **pickle序列化错误**：`TypeError: cannot pickle '_thread.lock' object`
2. **JSON序列化错误**：`TypeError: Object of type datetime is not JSON serializable`
3. **重启后状态丢失**：Checkpointer显示有checkpoint但无法恢复
4. **静默降级**：序列化失败时不报错，而是从空State重新开始

---

## 根因

Checkpointer 需要将 State 序列化存入数据库。如果 State 中包含不可序列化的对象（lambda、函数、线程锁、文件句柄、数据库连接等），序列化会失败。

常见的不可序列化对象：
- `lambda` 表达式
- `threading.Lock` / `RLock`
- 文件句柄 (`open()`)
- 数据库连接 (`psycopg2.connection`)
- 生成器 (`yield`)
- C扩展对象（如 `numpy.ndarray` 的某些视图）

---

## 解法

### 1. State字段序列化前清理

```python
import json
from typing import Any
from datetime import datetime

class SerializableState:
    """确保State可以被安全序列化的工具类。"""

    @staticmethod
    def make_serializable(obj: Any) -> Any:
        """递归地将对象转换为JSON可序列化形式。"""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (list, tuple)):
            return [SerializableState.make_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {k: SerializableState.make_serializable(v) for k, v in obj.items()}
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return SerializableState.make_serializable(obj.__dict__)
        # 最后手段：转字符串
        return str(obj)

    @staticmethod
    def is_serializable(obj: Any) -> bool:
        """检查对象是否可以JSON序列化。"""
        try:
            json.dumps(SerializableState.make_serializable(obj))
            return True
        except (TypeError, OverflowError):
            return False

# 节点中使用
def safe_node(state: RAGState) -> dict:
    result = some_computation()

    # 确保返回值可序列化
    return_value = {
        "answer": result.answer,
        "metadata": SerializableState.make_serializable(result.metadata),
        # 不要放入不可序列化的对象！
        # "connection": db_conn,  # ❌
    }

    # 验证可序列化
    assert SerializableState.is_serializable(return_value), "返回了不可序列化的对象！"

    return return_value
```

### 2. 在Graph编译前验证State可序列化

```python
def validate_state_serializability(state_class, sample_state: dict) -> list[str]:
    """
    在将Graph投入生产前，验证State可以被正确序列化/反序列化。
    """
    import pickle
    import json

    issues = []

    # 测试 Pickle 序列化（PostgreSQL使用）
    try:
        pickled = pickle.dumps(sample_state)
        restored = pickle.loads(pickled)
    except Exception as e:
        issues.append(f"Pickle序列化失败: {e}")

    # 测试 JSON 序列化
    try:
        json_str = json.dumps(sample_state, default=str)
        json.loads(json_str)
    except Exception as e:
        issues.append(f"JSON序列化失败: {e}")

    return issues

# 使用
sample = {"question": "test", "answer": "", "retry_count": 0}
issues = validate_state_serializability(RAGState, sample)
if issues:
    raise ValueError(f"State序列化验证失败: {issues}")
```

---

## 检查清单

- [ ] State中不包含lambda、线程锁、文件句柄、数据库连接
- [ ] 使用了 `make_serializable()` 处理日期时间等特殊类型
- [ ] 在CI中运行序列化往返测试（pickle + JSON）
- [ ] 重启服务后验证checkpoint可以正确恢复
- [ ] 添加了序列化失败的告警

---

**一句话总结**：如果你的State放不进数据库，那Checkpointer就是一个昂贵的空盒子。让State保持简单——只放数据，不放对象。
