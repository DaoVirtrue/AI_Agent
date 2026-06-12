# Phase 07 检验点：多智能体系统

> **完成标准**：独立完成以下7道练习题，能解释每道题的设计思路和代码逻辑。
> **建议时间**：2-3小时
> **难度标记**：⭐ 基础 | ⭐⭐ 中等 | ⭐⭐⭐ 挑战

---

## 练习1 ⭐：实现简化的SequentialPipeline

**任务**：基于 `01-sequential-pipeline.py` 的概念，实现一个更精简的管道类。

**要求**：
1. 创建一个 `SimplePipeline` 类，支持添加处理函数（而非Agent对象）
2. 每个处理函数接收一个字符串，返回一个字符串
3. 管道应记录每个处理函数的名称、耗时、输入/输出长度
4. 提供 `run()` 方法依次执行所有处理函数
5. 提供 `to_report()` 方法生成Markdown格式的执行报告

**输入示例**：
```python
pipeline = SimplePipeline(name="Demo")
pipeline.add_step("uppercase", lambda s: s.upper())
pipeline.add_step("reverse", lambda s: s[::-1])
result = pipeline.run("Hello World")
print(pipeline.to_report())
```

**预期输出**：执行报告包含每个步骤的名称、耗时、输入输出信息。

---

## 练习2 ⭐⭐：实现依赖感知的任务调度器

**任务**：基于 `02-hierarchical-manager-worker.py`，实现一个简化版的任务依赖调度器。

**要求**：
1. 创建 `TaskScheduler` 类
2. 支持添加任务：`add_task(name, depends_on=[], work_fn=...)`
3. 任务有状态（pending → running → completed/failed）
4. 实现死锁检测（循环依赖检测）
5. `execute_all()` 方法：按依赖顺序执行所有任务

**测试用例**：
```python
scheduler = TaskScheduler()

results_log = []
scheduler.add_task("A", depends_on=[], work_fn=lambda: results_log.append("A"))
scheduler.add_task("B", depends_on=["A"], work_fn=lambda: results_log.append("B"))
scheduler.add_task("C", depends_on=["A"], work_fn=lambda: results_log.append("C"))
scheduler.add_task("D", depends_on=["B", "C"], work_fn=lambda: results_log.append("D"))

scheduler.execute_all()
# 预期：results_log == ["A", "B", "C", "D"] 或 ["A", "C", "B", "D"]
#       但 B 和 C 都在 D 之前，且都在 A 之后
```

---

## 练习3 ⭐⭐：实现多Agent投票系统

**任务**：基于 `03-debate-ensemble.py`，实现一个独立的多Agent投票系统。

**要求**：
1. 创建 `VotingEnsemble` 类
2. 注册多个"虚拟Agent"（每个Agent是一个函数：接收问题，返回答案+置信度）
3. `ask(question)` 方法：
   - 所有Agent独立回答
   - 按置信度加权投票
   - 返回最高票答案和共识度评分
4. 返回格式：`{"answer": "...", "confidence": 0.85, "consensus": 0.75, "votes": {...}}`

**测试用例**：
```python
ensemble = VotingEnsemble()
ensemble.add_agent("A", lambda q: ("支持", 0.9))
ensemble.add_agent("B", lambda q: ("反对", 0.6))
ensemble.add_agent("C", lambda q: ("支持", 0.7))

result = ensemble.ask("该方案是否可行？")
# 预期：最终答案为"支持"（因为0.9+0.7 > 0.6）
```

---

## 练习4 ⭐⭐：实现升级规则引擎

**任务**：基于 `04-escalation-pattern.py`，实现一个可配置的升级规则引擎。

**要求**：
1. 创建 `EscalationRuleEngine` 类
2. 支持注册升级规则：`add_rule(name, condition_fn, target_tier)`
3. 每条规则是一个函数：`(query, current_result, attempt_count) -> bool`
4. `evaluate(query, current_result, attempt_count)` 返回最高优先级的升级目标
5. 预定义3条规则：
   - 置信度低于0.6 → 升级到L2
   - 用户要求人工 → 升级到L3
   - 重试超过3次 → 升级到L2

**测试用例**：
```python
engine = EscalationRuleEngine()
engine.add_rule("low_confidence", lambda q, r, n: r["confidence"] < 0.6, "L2")
engine.add_rule("human_request", lambda q, r, n: "人工" in q, "L3")
engine.add_rule("max_retries", lambda q, r, n: n > 3, "L2")

result = engine.evaluate("这个问题需要人工处理", {"confidence": 0.8}, 1)
# 预期：result == "L3"（human_request规则触发）
```

---

## 练习5 ⭐⭐：实现线程安全的共享字典

**任务**：基于 `05-blackboard-pattern.py`，实现一个线程安全的共享字典。

**要求**：
1. 创建 `ThreadSafeDict` 类
2. 所有读写操作都必须获取锁
3. 支持 `get(key)`、`set(key, value, agent_id)`、`delete(key, agent_id)`
4. 记录每次写入的 agent_id 和时间戳
5. 检测冲突：当 agent_a 和 agent_b（不同agent）都 set 同一个 key 时，记录冲突
6. `get_conflicts()` 返回所有未解决的冲突

**测试用例**：
```python
import threading
import time

tsd = ThreadSafeDict()

def agent_a():
    tsd.set("status", "A的版本", "agent_a")
    time.sleep(0.1)
    tsd.set("data", "A的数据", "agent_a")

def agent_b():
    time.sleep(0.05)
    tsd.set("status", "B的版本", "agent_b")  # 冲突！
    tsd.set("data", "B的数据", "agent_b")     # 冲突！

t1 = threading.Thread(target=agent_a)
t2 = threading.Thread(target=agent_b)
t1.start(); t2.start()
t1.join(); t2.join()

conflicts = tsd.get_conflicts()
assert len(conflicts) >= 2  # status 和 data 都应该有冲突记录
```

---

## 练习6 ⭐⭐：消息序列化往返测试

**任务**：基于 `06-agent-communication-protocol.py`，实现完整的消息序列化往返测试。

**要求**：
1. 创建 `AgentMessage` 数据类（包含所有标准字段）
2. 实现 `to_json()` 和 `from_json()` 方法
3. 实现 `to_bytes()` 和 `from_bytes()` 方法
4. 编写往返测试：原始消息 → JSON → 还原 → 比对所有字段
5. 编写字段验证：缺少必填字段时应抛出异常

**测试用例**：
```python
original = AgentMessage(
    sender_id="agent_1",
    receiver_id="agent_2",
    message_type="task_request",
    content={"task": "analyze", "params": {"depth": 5}},
    priority=1,
)

# 往返测试
json_str = original.to_json()
restored = AgentMessage.from_json(json_str)
assert original.sender_id == restored.sender_id
assert original.content == restored.content
assert original.priority == restored.priority

# 字节往返测试
bytes_data = original.to_bytes()
restored_bytes = AgentMessage.from_bytes(bytes_data)
assert original.sender_id == restored_bytes.sender_id

# 验证测试
try:
    AgentMessage.from_json('{"invalid": "data"}')
    assert False, "应该抛出异常"
except (ValueError, KeyError, TypeError):
    pass  # 预期的异常
```

---

## 练习7 ⭐⭐⭐：设计一个多Agent内容创作系统

**任务**：综合运用本阶段学到的所有模式，设计并实现一个完整的多Agent内容创作系统。

**要求**：
1. 包含至少4个不同角色的Agent（研究员、撰稿人、编辑、设计顾问）
2. 使用Sequential模式串联核心流程
3. 在关键节点使用Voting/Ensemble机制进行质量检查
4. 使用Blackboard模式共享参考资料和中间产出
5. 包含升级机制：如果质量不达标（置信度 < 0.7），升级到人工审核
6. 使用标准化的AgentMessage进行Agent间通信
7. 记录完整的执行时间线和各Agent的Token使用估算

**系统架构提示**：
```
用户请求
  │
  ▼
Blackboard（共享空间）
  ├── 研究员A + 研究员B（Ensemble：并行研究，投票选出最佳信息）
  │     ↓
  ├── 撰稿人（Sequential：基于研究资料撰写）
  │     ↓
  ├── 编辑A + 编辑B（Debate：对稿件进行质量控制讨论）
  │     ↓
  ├── 设计顾问（Sequential：提供排版和视觉建议）
  │     ↓
  ├── 质量评估（Escalation：置信度 < 0.7 → 人工审核）
  │     ↓
  ▼
最终输出
```

**评价标准**：
- 系统结构清晰，各模式使用得当（40%）
- 错误处理和边界条件考虑充分（25%）
- 提供了可运行的 `__main__` 演示块（25%）
- 代码注释完整，英文代码 + 中文注释（10%）

---

## 自检清单

完成以上练习后，确认以下能力：

- [ ] 我能实现基本的Sequential Pipeline并分析各阶段耗时
- [ ] 我能设计依赖感知的任务调度器并检测死锁
- [ ] 我能实现加权投票系统并计算共识度
- [ ] 我能设计可配置的升级规则引擎
- [ ] 我能实现线程安全的数据结构并检测冲突
- [ ] 我能实现消息的序列化/反序列化并编写往返测试
- [ ] 我能综合多种模式设计完整的多Agent系统
