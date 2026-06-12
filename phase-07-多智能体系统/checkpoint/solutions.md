# Phase 07 检验点：参考解答

> 注意：以下解答为参考实现。你的实现可能有不同的设计选择，只要满足题目要求即可。关键是理解每种模式的核心思想。

---

## 练习1：实现简化的SequentialPipeline

```python
#!/usr/bin/env python3
"""练习1 参考解答：简化的SequentialPipeline"""

import time
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class StepResult:
    name: str
    input_length: int
    output_length: int
    elapsed_ms: float
    output_preview: str = ""

class SimplePipeline:
    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.steps: list[tuple[str, Callable[[str], str]]] = []
        self.results: list[StepResult] = []

    def add_step(self, name: str, fn: Callable[[str], str]) -> "SimplePipeline":
        self.steps.append((name, fn))
        return self

    def run(self, initial_input: str) -> str:
        self.results = []
        current = initial_input

        for name, fn in self.steps:
            start = time.time()
            output = fn(current)
            elapsed = (time.time() - start) * 1000  # 转毫秒

            result = StepResult(
                name=name,
                input_length=len(current),
                output_length=len(output),
                elapsed_ms=round(elapsed, 2),
                output_preview=output[:80],
            )
            self.results.append(result)
            current = output

        return current

    def to_report(self) -> str:
        lines = [f"# Pipeline Report: {self.name}\n"]
        lines.append("| Step | Input Len | Output Len | Time (ms) | Preview |")
        lines.append("|------|-----------|------------|-----------|---------|")
        for r in self.results:
            lines.append(
                f"| {r.name} | {r.input_length} | {r.output_length} "
                f"| {r.elapsed_ms} | {r.output_preview[:50]}... |"
            )
        lines.append(f"\n**Total steps**: {len(self.results)}")
        lines.append(f"**Total time**: {sum(r.elapsed_ms for r in self.results):.2f}ms")
        return "\n".join(lines)


if __name__ == "__main__":
    pipeline = SimplePipeline(name="Demo")
    pipeline.add_step("uppercase", lambda s: s.upper())
    pipeline.add_step("reverse", lambda s: s[::-1])
    pipeline.add_step("add_prefix", lambda s: f"Result: {s}")

    final = pipeline.run("Hello World")
    print(final)
    print()
    print(pipeline.to_report())
```

---

## 练习2：实现依赖感知的任务调度器

```python
#!/usr/bin/env python3
"""练习2 参考解答：依赖感知的任务调度器"""

from collections import deque
from enum import Enum
from typing import Callable

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Task:
    def __init__(self, name: str, depends_on: list[str], work_fn: Callable):
        self.name = name
        self.depends_on = list(depends_on)
        self.work_fn = work_fn
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None

class TaskScheduler:
    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def add_task(self, name: str, depends_on: list[str] = None,
                 work_fn: Callable = None) -> "TaskScheduler":
        task = Task(name, depends_on or [], work_fn or (lambda: None))
        self.tasks[name] = task
        return self

    def detect_cycle(self) -> list[str] | None:
        """使用DFS检测循环依赖。返回环中节点列表，无环返回None。"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self.tasks}
        path = []

        def dfs(node: str) -> bool:
            color[node] = GRAY
            path.append(node)
            for dep in self.tasks[node].depends_on:
                if dep not in color:
                    continue  # 依赖不存在，跳过
                if color[dep] == GRAY:
                    # 找到环
                    cycle_start = path.index(dep)
                    return True  # path[cycle_start:] 即为环
                if color[dep] == WHITE:
                    if dfs(dep):
                        return True
            path.pop()
            color[node] = BLACK
            return False

        for name in self.tasks:
            if color[name] == WHITE:
                if dfs(name):
                    # 提取环
                    # 找到path中第一个出现在环中的节点
                    return path
        return None

    def execute_all(self) -> dict[str, any]:
        """按依赖顺序执行所有任务。"""
        # 死锁检测
        cycle = self.detect_cycle()
        if cycle:
            raise ValueError(f"检测到循环依赖: {' → '.join(cycle)}")

        completed: set[str] = set()
        results = {}

        while len(completed) < len(self.tasks):
            progress_made = False

            for name, task in self.tasks.items():
                if name in completed:
                    continue
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    continue
                if task.status == TaskStatus.RUNNING:
                    continue

                # 检查依赖是否全部完成
                deps_ready = all(d in completed for d in task.depends_on)
                if not deps_ready:
                    continue

                # 执行任务
                task.status = TaskStatus.RUNNING
                try:
                    task.result = task.work_fn()
                    task.status = TaskStatus.COMPLETED
                    completed.add(name)
                    results[name] = task.result
                    progress_made = True
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    results[name] = None
                    completed.add(name)
                    progress_made = True

            if not progress_made:
                pending = [n for n, t in self.tasks.items()
                          if t.status == TaskStatus.PENDING]
                raise RuntimeError(
                    f"无法继续执行。待执行任务: {pending}。"
                    f"可能存在不可满足的依赖。"
                )

        return results


if __name__ == "__main__":
    scheduler = TaskScheduler()
    results_log = []

    scheduler.add_task("A", depends_on=[], work_fn=lambda: results_log.append("A"))
    scheduler.add_task("B", depends_on=["A"], work_fn=lambda: results_log.append("B"))
    scheduler.add_task("C", depends_on=["A"], work_fn=lambda: results_log.append("C"))
    scheduler.add_task("D", depends_on=["B", "C"], work_fn=lambda: results_log.append("D"))

    scheduler.execute_all()
    print(f"执行顺序: {results_log}")

    # 验证：A必须在B和C之前，B和C必须在D之前
    a_pos = results_log.index("A")
    b_pos = results_log.index("B")
    c_pos = results_log.index("C")
    d_pos = results_log.index("D")
    assert a_pos < b_pos and a_pos < c_pos, "A必须在B和C之前"
    assert b_pos < d_pos and c_pos < d_pos, "B和C必须在D之前"
    print("✅ 所有依赖关系满足")

    # 测试死锁检测
    bad_scheduler = TaskScheduler()
    bad_scheduler.add_task("X", depends_on=["Y"], work_fn=lambda: None)
    bad_scheduler.add_task("Y", depends_on=["X"], work_fn=lambda: None)
    cycle = bad_scheduler.detect_cycle()
    print(f"死锁检测: 发现循环依赖 {'✅' if cycle else '❌'}")
    print(f"循环路径: {cycle}")
```

---

## 练习3：实现多Agent投票系统

```python
#!/usr/bin/env python3
"""练习3 参考解答：多Agent投票系统"""

from collections import Counter

class VotingEnsemble:
    def __init__(self):
        self.agents: dict[str, callable] = {}

    def add_agent(self, agent_id: str, response_fn: callable) -> "VotingEnsemble":
        """
        response_fn: 接收问题字符串，返回 (答案, 置信度) 元组。
        置信度应在 0-1 之间。
        """
        self.agents[agent_id] = response_fn
        return self

    def ask(self, question: str) -> dict:
        """
        所有Agent独立回答，加权投票，返回最终结果。
        """
        if not self.agents:
            raise ValueError("没有注册的Agent，请先调用 add_agent()")

        # 收集所有Agent的回答
        answers = []
        total_weight = 0
        answer_weights: dict[str, float] = {}

        for agent_id, fn in self.agents.items():
            answer, confidence = fn(question)
            confidence = max(0.0, min(1.0, confidence))  # clamp到[0,1]

            answers.append({
                "agent_id": agent_id,
                "answer": answer,
                "confidence": confidence,
            })

            # 加权累积
            answer_weights[answer] = answer_weights.get(answer, 0) + confidence
            total_weight += confidence

        # 找出最高票答案
        if answer_weights:
            best_answer = max(answer_weights, key=answer_weights.get)
            best_weight = answer_weights[best_answer]
            # 共识度 = 最高票权重 / 总权重
            consensus = best_weight / total_weight if total_weight > 0 else 0
            # 最终置信度 = 最高票权重 / Agent数（归一化）
            final_confidence = best_weight / len(self.agents)
        else:
            best_answer = None
            best_weight = 0
            consensus = 0
            final_confidence = 0

        return {
            "question": question,
            "answer": best_answer,
            "confidence": round(final_confidence, 4),
            "consensus": round(consensus, 4),
            "total_votes": len(answers),
            "vote_details": answers,
            "weights": answer_weights,
        }


if __name__ == "__main__":
    ensemble = VotingEnsemble()

    # 三个Agent，两个支持、一个反对
    ensemble.add_agent("A", lambda q: ("支持", 0.9))
    ensemble.add_agent("B", lambda q: ("反对", 0.6))
    ensemble.add_agent("C", lambda q: ("支持", 0.7))

    result = ensemble.ask("该方案是否可行？")
    print(f"最终答案: {result['answer']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"共识度: {result['consensus']:.2f}")
    print(f"投票详情: {result['vote_details']}")

    assert result["answer"] == "支持", "两个支持票应该获胜"
    print("✅ 投票结果正确")

    # 均匀分布情况
    ensemble2 = VotingEnsemble()
    ensemble2.add_agent("X", lambda q: ("选项A", 0.8))
    ensemble2.add_agent("Y", lambda q: ("选项B", 0.8))
    ensemble2.add_agent("Z", lambda q: ("选项A", 0.9))
    result2 = ensemble2.ask("选哪个？")
    print(f"\n均匀分布测试: {result2['answer']} (共识度: {result2['consensus']:.2f})")
```

---

## 练习4：实现升级规则引擎

```python
#!/usr/bin/env python3
"""练习4 参考解答：升级规则引擎"""

from typing import Callable

RuleCondition = Callable[[str, dict, int], bool]

class EscalationRuleEngine:
    def __init__(self):
        self.rules: list[tuple[str, RuleCondition, str]] = []

    def add_rule(self, name: str, condition: RuleCondition,
                 target_tier: str) -> "EscalationRuleEngine":
        """
        Args:
            name: 规则名称
            condition: 判断函数 (query, current_result, attempt_count) -> bool
            target_tier: 升级目标层级 (如 "L2", "L3")
        """
        self.rules.append((name, condition, target_tier))
        return self

    def evaluate(self, query: str, current_result: dict,
                 attempt_count: int) -> str | None:
        """
        评估所有规则，返回触发的最优先升级目标。

        Returns:
            升级目标层级，如果不需升级返回None
        """
        # L3优先级高于L2（先检查L3规则）
        l3_match = None
        l2_match = None

        for name, condition, target in self.rules:
            try:
                triggered = condition(query, current_result, attempt_count)
            except Exception:
                continue  # 规则执行异常 → 跳过

            if triggered:
                if target == "L3":
                    l3_match = name
                elif target == "L2" and l2_match is None:
                    l2_match = name

        if l3_match is not None:
            return "L3"
        if l2_match is not None:
            return "L2"
        return None


if __name__ == "__main__":
    engine = EscalationRuleEngine()

    # 注册规则
    engine.add_rule(
        "low_confidence",
        lambda q, r, n: r.get("confidence", 1.0) < 0.6,
        "L2",
    )
    engine.add_rule(
        "human_request",
        lambda q, r, n: "人工" in q or "客服" in q,
        "L3",
    )
    engine.add_rule(
        "max_retries",
        lambda q, r, n: n > 3,
        "L2",
    )

    # 测试1：正常情况，不应升级
    result1 = engine.evaluate(
        "正常问题",
        {"confidence": 0.85},
        attempt_count=1,
    )
    print(f"测试1 (正常): 升级到 {result1} (预期: None)")

    # 测试2：低置信度 → L2
    result2 = engine.evaluate(
        "复杂问题",
        {"confidence": 0.45},
        attempt_count=1,
    )
    print(f"测试2 (低置信度): 升级到 {result2} (预期: L2)")

    # 测试3：用户要求人工 → L3
    result3 = engine.evaluate(
        "我要人工客服",
        {"confidence": 0.9},
        attempt_count=1,
    )
    print(f"测试3 (人工要求): 升级到 {result3} (预期: L3)")

    # 测试4：重试过多 → L2
    result4 = engine.evaluate(
        "问题",
        {"confidence": 0.7},
        attempt_count=5,
    )
    print(f"测试4 (重试过多): 升级到 {result4} (预期: L2)")

    # 验证
    assert result1 is None
    assert result2 == "L2"
    assert result3 == "L3"
    assert result4 == "L2"
    print("\n✅ 所有测试通过！")
```

---

## 练习5：实现线程安全的共享字典

```python
#!/usr/bin/env python3
"""练习5 参考解答：线程安全的共享字典"""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class WriteRecord:
    agent_id: str
    value: any
    timestamp: float = field(default_factory=time.time)


class ThreadSafeDict:
    def __init__(self):
        self._data: dict[str, any] = {}
        self._records: dict[str, list[WriteRecord]] = {}
        self._conflicts: list[dict] = []
        self._lock = threading.RLock()

    def get(self, key: str) -> any:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: any, agent_id: str) -> None:
        with self._lock:
            # 检查冲突：不同agent对同一key的写入
            if key in self._records:
                existing_agents = {r.agent_id for r in self._records[key]}
                if agent_id not in existing_agents:
                    # 冲突！
                    self._conflicts.append({
                        "key": key,
                        "agents": list(existing_agents) + [agent_id],
                        "existing_value": self._data.get(key),
                        "new_value": value,
                        "timestamp": time.time(),
                    })

            # 写入数据
            self._data[key] = value

            # 记录写入
            record = WriteRecord(agent_id=agent_id, value=value)
            if key not in self._records:
                self._records[key] = []
            self._records[key].append(record)

    def delete(self, key: str, agent_id: str) -> bool:
        with self._lock:
            # 只允许最近的写入者删除
            if key in self._records and self._records[key]:
                last_writer = self._records[key][-1].agent_id
                if last_writer == agent_id:
                    del self._data[key]
                    return True
            return False

    def get_conflicts(self) -> list[dict]:
        with self._lock:
            return list(self._conflicts)

    def get_write_history(self, key: str) -> list[WriteRecord]:
        with self._lock:
            return list(self._records.get(key, []))

    def to_dict(self) -> dict:
        with self._lock:
            return dict(self._data)


if __name__ == "__main__":
    tsd = ThreadSafeDict()

    def agent_a():
        tsd.set("status", "A的版本", "agent_a")
        time.sleep(0.1)
        tsd.set("data", "A的数据", "agent_a")

    def agent_b():
        time.sleep(0.05)
        tsd.set("status", "B的版本", "agent_b")
        tsd.set("data", "B的数据", "agent_b")

    t1 = threading.Thread(target=agent_a)
    t2 = threading.Thread(target=agent_b)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"当前数据: {tsd.to_dict()}")
    print(f"冲突数: {len(tsd.get_conflicts())}")
    for c in tsd.get_conflicts():
        print(f"  冲突: key='{c['key']}', agents={c['agents']}")

    assert len(tsd.get_conflicts()) >= 2, "status 和 data 都应该有冲突记录"
    print("✅ 冲突检测正确")
```

---

## 练习6：消息序列化往返测试

```python
#!/usr/bin/env python3
"""练习6 参考解答：消息序列化往返测试"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    QUERY = "query"
    ANSWER = "answer"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    ERROR = "error"


@dataclass
class AgentMessage:
    message_id: str = field(default_factory=lambda: f"msg-{int(time.time()*1000)}")
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.QUERY
    content: Any = None
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = ""
    priority: int = 1  # 0=低, 1=普通, 2=高, 3=紧急
    ttl: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMessage":
        # 验证必填字段
        required = ["message_id", "sender_id", "message_type", "content"]
        missing = [f for f in required if f not in data or not data[f]]
        if missing:
            raise ValueError(f"缺少必填字段: {missing}")

        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            receiver_id=data.get("receiver_id", ""),
            message_type=MessageType(data.get("message_type", "query")),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            correlation_id=data.get("correlation_id", ""),
            priority=data.get("priority", 1),
            ttl=data.get("ttl", 0.0),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentMessage":
        return cls.from_dict(json.loads(json_str))

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "AgentMessage":
        return cls.from_json(data.decode("utf-8"))


# ---- 测试 ----

if __name__ == "__main__":
    # 创建原始消息
    original = AgentMessage(
        sender_id="agent_1",
        receiver_id="agent_2",
        message_type=MessageType.TASK_REQUEST,
        content={"task": "analyze", "params": {"depth": 5}},
        priority=2,
        metadata={"trace_id": "abc-123"},
    )

    # JSON 往返测试
    json_str = original.to_json()
    restored = AgentMessage.from_json(json_str)
    assert original.sender_id == restored.sender_id, "sender_id 不匹配"
    assert original.content == restored.content, "content 不匹配"
    assert original.priority == restored.priority, "priority 不匹配"
    assert original.metadata == restored.metadata, "metadata 不匹配"
    print("✅ JSON 往返测试通过")

    # Bytes 往返测试
    bytes_data = original.to_bytes()
    restored_bytes = AgentMessage.from_bytes(bytes_data)
    assert original.sender_id == restored_bytes.sender_id, "sender_id 不匹配"
    assert original.content == restored_bytes.content, "content 不匹配"
    print("✅ Bytes 往返测试通过")

    # 序列化大小报告
    print(f"📏 JSON大小: {len(json_str)} bytes")
    print(f"📏 Bytes大小: {len(bytes_data)} bytes")

    # 必填字段验证测试
    try:
        AgentMessage.from_json('{"invalid": "data"}')
        assert False, "应该抛出异常"
    except (ValueError, KeyError, TypeError) as e:
        print(f"✅ 验证测试通过: {type(e).__name__}: {e}")

    # 美观打印
    print(f"\n📝 序列化后的消息:")
    print(original.to_json(indent=2))
```

---

## 练习7：设计一个多Agent内容创作系统

```python
#!/usr/bin/env python3
"""练习7 参考解答：综合多Agent内容创作系统"""

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any

# ---- 基础组件 ----

class MessageType(Enum):
    QUERY = "query"
    ANSWER = "answer"
    DATA = "data"
    VOTE = "vote"

@dataclass
class Message:
    msg_id: str
    sender: str
    receiver: str  # "" = broadcast
    msg_type: MessageType
    content: Any

# Voting Ensemble (from Exercise 3)
class VotingEnsemble:
    def __init__(self):
        self.agents = {}

    def add_agent(self, name, fn):
        self.agents[name] = fn
        return self

    def ask(self, question):
        weights = {}
        total = 0
        details = []
        for name, fn in self.agents.items():
            ans, conf = fn(question)
            conf = max(0.0, min(1.0, conf))
            weights[ans] = weights.get(ans, 0) + conf
            total += conf
            details.append({"agent": name, "answer": ans, "confidence": conf})
        best = max(weights, key=weights.get) if weights else None
        consensus = weights[best] / total if total > 0 and best else 0
        return {"answer": best, "confidence": weights.get(best, 0) / len(self.agents) if self.agents else 0,
                "consensus": consensus, "details": details}

# Escalation Engine (from Exercise 4)
class EscalationEngine:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def check(self, confidence):
        if confidence < self.threshold:
            return "L3_HUMAN_REVIEW"
        return None

# Thread-Safe Blackboard (from Exercise 5)
class Blackboard:
    def __init__(self):
        self._data = {}
        self._lock = threading.RLock()

    def write(self, key, value, agent):
        with self._lock:
            self._data[key] = {"value": value, "agent": agent, "time": time.time()}

    def read(self, key):
        with self._lock:
            entry = self._data.get(key)
            return entry["value"] if entry else None

    def read_all(self):
        with self._lock:
            return {k: v["value"] for k, v in self._data.items()}

# ---- Agent 实现 ----

class ContentAgent:
    def __init__(self, name, role, system_prompt=""):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.tokens_used = 0
        self.time_spent = 0.0

    def process(self, input_text, context=None):
        start = time.time()
        result = self._simulate_work(input_text, context)
        elapsed = time.time() - start
        self.time_spent += elapsed
        self.tokens_used += len(input_text) // 3 + len(result) // 3
        return result

    def _simulate_work(self, input_text, context):
        # 模拟各角色的产出
        if self.role == "researcher":
            return f"[研究员 {self.name}]\n关键发现:\n1. 市场规模500亿\n2. 增长率15%\n3. 三大趋势: AI、云、安全\n\n来源: McKinsey 2025, Gartner 2025"
        elif self.role == "writer":
            research = context.get("research", "") if context else ""
            return f"[撰稿人 {self.name}]\n# 深度分析报告\n\n基于研究资料，本文全面分析...\n\n{research[:100]}...\n\n## 结论\n综合来看，行业前景乐观。"
        elif self.role == "editor":
            draft = input_text if input_text else ""
            return f"[编辑 {self.name}]\n审校意见:\n1. 结构评分: 8/10\n2. 需要改进: 增加案例\n3. 优点: 数据充分\n\n{draft[:100]}..."
        elif self.role == "designer":
            return f"[设计顾问 {self.name}]\n排版建议:\n1. 使用信息图展示数据\n2. 配色方案: 蓝色系\n3. 建议增加3-5张图表"
        else:
            return f"[{self.role} {self.name}] 已处理输入: {input_text[:50]}..."


# ---- 编排器 ----

class ContentCreationOrchestrator:
    def __init__(self):
        self.blackboard = Blackboard()
        self.research_ensemble = VotingEnsemble()
        self.editing_ensemble = VotingEnsemble()
        self.escalation = EscalationEngine(threshold=0.7)
        self.execution_log = []

    def run(self, topic):
        print(f"\n{'='*60}")
        print(f"  内容创作系统启动: {topic}")
        print(f"{'='*60}")

        total_start = time.time()
        self.blackboard.write("topic", topic, "orchestrator")

        # Phase 1: 并行研究 (Ensemble模式)
        print("\n[Phase 1] 研究阶段 (Ensemble: 2研究员并行)")
        researcher_a = ContentAgent("研究员-张三", "researcher")
        researcher_b = ContentAgent("研究员-李四", "researcher")
        self.research_ensemble.add_agent("Zhang", lambda q: (researcher_a.process(q), 0.85))
        self.research_ensemble.add_agent("Li", lambda q: (researcher_b.process(q), 0.80))

        research_result = self.research_ensemble.ask(topic)
        best_research = research_result["answer"]
        self.blackboard.write("research", best_research, "research_ensemble")
        self.blackboard.write("research_quality", research_result["confidence"], "ensemble")
        print(f"  研究共识度: {research_result['consensus']:.2%}")

        # Phase 2: 撰写 (Sequential模式)
        print("\n[Phase 2] 撰写阶段")
        writer = ContentAgent("撰稿人-王五", "writer")
        draft = writer.process(topic, {"research": best_research})
        self.blackboard.write("draft", draft, writer.name)
        print(f"  初稿: {len(draft)} 字符")

        # Phase 3: 质量审查 (Debate/Ensemble模式)
        print("\n[Phase 3] 质量审查阶段 (Ensemble: 2编辑独立评估)")
        editor_a = ContentAgent("编辑-赵六", "editor")
        editor_b = ContentAgent("编辑-钱七", "editor")
        self.editing_ensemble.add_agent("Zhao", lambda q: (editor_a.process(draft), 0.75))
        self.editing_ensemble.add_agent("Qian", lambda q: (editor_b.process(draft), 0.82))

        editing_result = self.editing_ensemble.ask("请审校此稿件")
        quality_score = editing_result["confidence"]
        self.blackboard.write("quality_score", quality_score, "editing_ensemble")
        print(f"  质量评分: {quality_score:.2%}")

        # Phase 4: 设计建议 (Sequential模式)
        print("\n[Phase 4] 设计建议阶段")
        designer = ContentAgent("设计顾问-孙八", "designer")
        design_advice = designer.process(draft)
        self.blackboard.write("design_advice", design_advice, designer.name)

        # Phase 5: 升级检查
        print("\n[Phase 5] 升级检查")
        escalation_result = self.escalation.check(quality_score)
        if escalation_result:
            print(f"  ⚠️ 质量不足 (评分: {quality_score:.2%}) → 升级到 {escalation_result}")
            self.blackboard.write("escalation", escalation_result, "escalation_engine")
            final_output = f"[需要人工审核] 自动生成的质量评分{quality_score:.0%}低于阈值，请人工审阅:\n\n{draft}"
        else:
            print(f"  ✅ 质量达标 (评分: {quality_score:.2%})")
            # Phase 6: 最终整合
            print("\n[Phase 6] 最终整合")
            bb_data = self.blackboard.read_all()
            final_output = f"""# 最终交付物

## 研究摘要
{bb_data.get('research', 'N/A')[:200]}...

## 正文
{draft}

## 设计建议
{design_advice}

## 质量报告
- 质量评分: {quality_score:.1%}
- 研究共识度: {research_result.get('consensus', 0):.1%}
- 需要人工审核: {escalation_result is not None}

---
*本文由多Agent内容创作系统自动生成*
*生成时间: {time.time() - total_start:.1f}s*
"""

        # Token 和 时间统计
        all_agents = [researcher_a, researcher_b, writer, editor_a, editor_b, designer]
        total_tokens = sum(a.tokens_used for a in all_agents)
        total_time = time.time() - total_start

        print(f"\n{'='*60}")
        print(f"  执行统计")
        print(f"{'='*60}")
        print(f"  总耗时: {total_time:.1f}s")
        print(f"  总Token估算: ~{total_tokens}")
        print(f"  最终输出: {len(final_output)} 字符")
        print(f"  是否升级: {'是 → ' + escalation_result if escalation_result else '否'}")

        for a in all_agents:
            print(f"  {a.name} ({a.role}): {a.time_spent:.2f}s, ~{a.tokens_used} tokens")

        return {
            "final_output": final_output,
            "quality_score": quality_score,
            "escalation": escalation_result,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "blackboard": bb_data,
        }


if __name__ == "__main__":
    orchestrator = ContentCreationOrchestrator()
    result = orchestrator.run("2025年企业AI应用趋势分析")
    print(f"\n{'='*60}")
    print("  最终输出预览（前400字符）:")
    print(f"{'='*60}")
    print(result["final_output"][:400])
    print(f"\n... (共 {len(result['final_output'])} 字符)")
```

---

以上7道练习的解答完整涵盖了多智能体系统的核心模式。完成这些练习后，你已具备设计和实现企业级多Agent协作系统的能力。接下来的Phase 08将进入LangGraph的世界，用更工程化的方式实现这些模式。
