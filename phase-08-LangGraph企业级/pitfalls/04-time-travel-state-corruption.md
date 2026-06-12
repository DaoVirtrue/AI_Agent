# 坑点 #4：时间旅行状态损坏（Time Travel State Corruption）

> "你回退到3步前的checkpoint，重新执行，但新路径产生了旧State中没有的新字段。下游节点尝试读取这些新字段时——KeyError。你的'时间旅行'变成了'时间灾难'。"

---

## 症状

1. **KeyError in restored state**：回退到旧checkpoint后，新路径添加了旧State中没有的字段，下游节点找不到
2. **数据不一致**：回退后某些字段保留了旧值，导致计算错误
3. **部分回退**：只有部分字段回退，其他字段保留了"未来"的值
4. **loop无限循环**：回退点恰好是循环的入口，重新执行后又走同一条失败路径

---

## 根因

时间旅行时，State恢复到特定checkpoint的快照。但如果当前版本的节点函数期望State中有新字段，而这些字段在旧checkpoint中不存在，就会出现字段缺失错误。

另外，Reducers可能不会完全"重置"——`add_messages` 会追加而非覆盖，可能导致messages列表越来越长。

---

## 解法

### 1. State版本兼容层

```python
class StateVersionAdapter:
    """
    State版本适配器：确保从旧checkpoint恢复的State
    包含当前节点所需的所有字段。
    """

    def __init__(self, state_class, default_values: dict):
        self.state_class = state_class
        self.default_values = default_values
        self.expected_fields = set(
            list(state_class.__annotations__.keys())
            if hasattr(state_class, '__annotations__')
            else []
        )

    def adapt(self, old_state: dict) -> dict:
        """
        将旧State适配到当前版本。
        缺失的字段用默认值填充。
        """
        adapted = dict(old_state)

        for field in self.expected_fields:
            if field not in adapted:
                adapted[field] = self.default_values.get(field)
                # 记录字段填充（用于调试）
                adapted.setdefault('_migration_log', []).append(
                    f"字段 '{field}' 从默认值填充"
                )

        # 移除当前版本不需要的废弃字段
        for field in list(adapted.keys()):
            if field not in self.expected_fields and not field.startswith('_'):
                adapted.pop(field)

        return adapted
```

### 2. 安全的 Rewind 函数

```python
class SafeTimeTravel:
    """
    安全的时间旅行：在回退前验证目标checkpoint。
    """

    @staticmethod
    def create_rewind_point(graph, config: dict, label: str = None) -> dict:
        """
        在关键节点之后创建"回退点"。
        记录此时的thread_id和checkpoint_id，用于后续安全回退。
        """
        history = list(graph.get_state_history(config))
        if not history:
            raise ValueError("没有可用的检查点")

        # 记录最新的检查点
        latest = history[0]
        checkpoint_id = latest.config.get("configurable", {}).get("checkpoint_id")

        return {
            "label": label or f"rewind_point_{len(history)}",
            "config": config,
            "checkpoint_id": checkpoint_id,
            "state_summary": {
                k: str(v)[:100] for k, v in latest.values.items()
            },
            "created_at": time.time(),
        }

    @staticmethod
    def verify_rewind_safety(graph, target_config: dict, current_config: dict) -> list[str]:
        """
        验证回退到目标checkpoint是否安全。
        返回潜在问题列表。
        """
        warnings = []

        # 获取目标和当前状态
        target_history = list(graph.get_state_history(target_config))
        current_history = list(graph.get_state_history(current_config))

        if not target_history or not current_history:
            return ["无法获取状态历史"]

        target_state = target_history[0].values
        current_state = current_history[0].values

        # 检查字段差异
        target_fields = set(target_state.keys())
        current_fields = set(current_state.keys())

        missing_in_target = current_fields - target_fields
        if missing_in_target:
            warnings.append(
                f"目标checkpoint缺少字段: {missing_in_target}。"
                f"回退后这些字段将使用默认值。"
            )

        new_in_target = target_fields - current_fields
        if new_in_target:
            warnings.append(
                f"目标checkpoint有额外字段: {new_in_target}。"
                f"这些字段可能来自未来的节点。"
            )

        return warnings

    @staticmethod
    def safe_rewind(graph, target_config: dict, current_config: dict, **override_values) -> dict:
        """
        安全回退：先验证，再回退，最后用override填充缺失字段。
        """
        warnings = SafeTimeTravel.verify_rewind_safety(graph, target_config, current_config)
        if warnings:
            print("[时间旅行] 警告:")
            for w in warnings:
                print(f"  ⚠️ {w}")

        # 执行回退
        result = graph.invoke(override_values or None, target_config)

        return result
```

---

## 检查清单

- [ ] 所有State字段都有合理的默认值
- [ ] 在回退前检查目标checkpoint是否包含当前路径所需的所有字段
- [ ] 对于add_messages等追加型Reducer，回退时有清理策略
- [ ] 创建了回退点记录（用于审计和调试）
- [ ] 在CI中测试了回退-重新执行的往返流程
- [ ] 回退后有验证步骤（确保恢复的State合法）

---

**一句话总结**：时间旅行不是真正的时光机——你只能回到过去，但不能假设过去的State适应现在的代码。在出发前检查你的"时间胶囊"是否兼容。

---

*注意：`time` 和 `typing` 需要在实际文件中通过 import 引入。以上代码为示意片段。*
