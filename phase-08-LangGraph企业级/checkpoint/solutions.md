# Phase 08 检验点：参考解答

> 注意：以下解答为参考实现。LangGraph 允许多种实现方式，只要满足题目要求即可。

---

## 练习1：构建一个带条件路由的StateGraph

```python
#!/usr/bin/env python3
"""练习1 参考解答：带条件路由的StateGraph"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class QueryState(TypedDict):
    query: str
    intent: str
    answer: str
    confidence: float

def node_classify(state: QueryState) -> dict:
    query = state['query']
    query_len = len(query)
    # 根据查询长度估算置信度（简化的启发式）
    if query_len < 50:
        confidence = 0.9
        intent = "simple"
    elif query_len < 200:
        confidence = 0.65
        intent = "medium"
    else:
        confidence = 0.35
        intent = "complex"
    print(f"  [classify] intent={intent}, confidence={confidence:.2f}")
    return {"intent": intent, "confidence": confidence}

def node_respond(state: QueryState) -> dict:
    print(f"  [respond] 生成答案")
    return {
        "answer": f"关于'{state['query'][:50]}'，以下是详细回答...",
        "confidence": state.get('confidence', 0.8),
    }

def node_fallback(state: QueryState) -> dict:
    print(f"  [fallback] 置信度不足，使用保守回答")
    return {
        "answer": f"抱歉，'{state['query'][:50]}'这个问题比较复杂，建议简化问题或咨询专家。",
        "confidence": 0.5,
    }

def route_by_confidence(state: QueryState) -> Literal["respond", "fallback"]:
    if state.get('confidence', 0) >= 0.5:
        return "respond"
    return "fallback"

# 构建图
builder = StateGraph(QueryState)
builder.add_node("classify", node_classify)
builder.add_node("respond", node_respond)
builder.add_node("fallback", node_fallback)

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", route_by_confidence, {
    "respond": "respond",
    "fallback": "fallback",
})
builder.add_edge("respond", END)
builder.add_edge("fallback", END)

graph = builder.compile()

# 测试
for query in ["你好", "请帮我分析一下这个技术问题" * 5, "非常复杂的问题" * 20]:
    print(f"\n查询: {query[:40]}...")
    result = graph.invoke({"query": query, "intent": "", "answer": "", "confidence": 0.0})
    print(f"  意图: {result['intent']}, 置信度: {result['confidence']:.2f}")
    print(f"  答案: {result['answer'][:80]}...")
    assert result['answer'], "答案不应为空"

print("\n✅ 练习1完成")

# 可视化
print(f"\n图结构:\n{graph.get_graph().draw_mermaid()}")
```

---

## 练习2：实现带Checkpointer的状态持久化

```python
#!/usr/bin/env python3
"""练习2 参考解答：Checkpointer状态持久化"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    turn_count: int

def node_greet(state: ChatState) -> dict:
    turn = state.get('turn_count', 0) + 1
    name = state.get('user_name', '用户')
    return {
        "messages": [AIMessage(content=f"欢迎{name}！第{turn}轮对话")],
        "turn_count": turn,
    }

def node_reply(state: ChatState) -> dict:
    last_human = None
    for msg in reversed(state.get('messages', [])):
        if hasattr(msg, 'type') and msg.type == 'human':
            last_human = msg.content
            break
    return {
        "messages": [AIMessage(content=f"回复: {last_human[:50] if last_human else '...'}...")],
    }

# 构建图
builder = StateGraph(ChatState)
builder.add_node("greet", node_greet)
builder.add_node("reply", node_reply)
builder.set_entry_point("greet")
builder.add_edge("greet", "reply")
builder.add_edge("reply", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 会话A：两次调用
config_a = {"configurable": {"thread_id": "user_A"}}
result_a1 = graph.invoke(
    {"messages": [], "user_name": "Alice", "turn_count": 0},
    config_a,
)
result_a2 = graph.invoke(
    {"messages": [HumanMessage(content="你好啊")]},
    config_a,
)

print(f"Alice 第1次: turn_count={result_a1['turn_count']}")
print(f"Alice 第2次: turn_count={result_a2['turn_count']}")

# 验证状态累积：messages应该比第一次多
assert len(result_a2['messages']) > len(result_a1['messages']), "状态应累积"
print("✅ 状态累积验证通过")

# 会话B：独立
config_b = {"configurable": {"thread_id": "user_B"}}
result_b1 = graph.invoke(
    {"messages": [], "user_name": "Bob", "turn_count": 0},
    config_b,
)
assert result_b1['turn_count'] == 1, "Bob应该从1开始（不受Alice影响）"
print("✅ Thread隔离验证通过")

# 检查checkpoints
history_a = list(graph.get_state_history(config_a))
print(f"\nAlice 的 checkpoints: {len(history_a)}")
for i, h in enumerate(history_a):
    print(f"  [{i}] turn_count={h.values.get('turn_count')}, messages={len(h.values.get('messages', []))}")

print("\n✅ 练习2完成")
```

---

## 练习3：实现Human-in-the-Loop中断

```python
#!/usr/bin/env python3
"""练习3 参考解答：Human-in-the-Loop"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

class ApprovalState(TypedDict):
    transaction: dict
    risk_score: float
    approval_status: str
    audit_log: Annotated[list, add_messages]

def node_validate(state: ApprovalState) -> dict:
    txn = state['transaction']
    risk = 50 if txn.get('amount', 0) > 10000 else 20
    print(f"  [validate] 风险评分: {risk}")
    return {
        "risk_score": risk,
        "audit_log": [f"[审计] 验证通过 风险={risk}"],
    }

def node_approve(state: ApprovalState) -> dict:
    risk = state.get('risk_score', 0)
    txn = state.get('transaction', {})

    print(f"  [approve] 风险={risk}, 需要人工审批")

    # 高风险时触发interrupt
    if risk > 30:
        human_input = interrupt({
            "message": "需要人工审批",
            "transaction": txn,
            "risk_score": risk,
        })
        decision = human_input.get("decision", "rejected")
        reason = human_input.get("reason", "")
        print(f"  [approve] 人工决定: {decision}")
    else:
        decision = "approved"
        reason = "低风险自动批准"

    return {
        "approval_status": decision,
        "audit_log": [f"[审计] 审批结果: {decision} | 理由: {reason}"],
    }

def node_execute(state: ApprovalState) -> dict:
    if state.get('approval_status') == 'approved':
        print(f"  [execute] 执行交易")
        return {"audit_log": ["[审计] 交易已执行"]}
    return {"audit_log": ["[审计] 交易已拒绝"]}

def route_after_approval(state: ApprovalState) -> Literal["execute", "reject"]:
    return "execute" if state.get('approval_status') == 'approved' else "reject"

def node_reject(state: ApprovalState) -> dict:
    print(f"  [reject] 交易被拒绝")
    return {}

# 构建图
builder = StateGraph(ApprovalState)
builder.add_node("validate", node_validate)
builder.add_node("approve", node_approve)
builder.add_node("execute", node_execute)
builder.add_node("reject", node_reject)

builder.set_entry_point("validate")
builder.add_edge("validate", "approve")
builder.add_conditional_edges("approve", route_after_approval, {
    "execute": "execute", "reject": "reject",
})
builder.add_edge("execute", END)
builder.add_edge("reject", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 测试低风险（自动批准）
print("\n--- 低风险交易 ---")
config_low = {"configurable": {"thread_id": "low_risk"}}
result_low = graph.invoke({
    "transaction": {"amount": 500, "type": "payment"},
    "risk_score": 0, "approval_status": "", "audit_log": [],
}, config_low)
print(f"审批状态: {result_low['approval_status']}")
assert result_low['approval_status'] == 'approved'
print("✅ 低风险自动批准通过")

# 测试高风险（触发interrupt）
print("\n--- 高风险交易 ---")
config_high = {"configurable": {"thread_id": "high_risk"}}
try:
    graph.invoke({
        "transaction": {"amount": 50000, "type": "transfer"},
        "risk_score": 0, "approval_status": "", "audit_log": [],
    }, config_high)
    print("⚠️ 高风险但没有触发interrupt（在没有真实interrupt的环境中）")
except Exception as e:
    print(f"触发interrupt: {type(e).__name__}")
    # 模拟人工审批恢复
    resume_cmd = Command(resume={"decision": "approved", "reason": "合规审核通过"})
    try:
        result_high = graph.invoke(resume_cmd, config_high)
        print(f"审批状态: {result_high['approval_status']}")
    except Exception:
        print("恢复执行（需要真实的LangGraph interrupt支持）")

print("\n✅ 练习3完成")
```

---

## 练习4：实现时间旅行

```python
#!/usr/bin/env python3
"""练习4 参考解答：时间旅行"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class CounterState(TypedDict):
    step: int
    value: str
    accumulated: list

def node_a(state: CounterState) -> dict:
    return {"step": 1, "value": "A", "accumulated": ["A"]}

def node_b(state: CounterState) -> dict:
    prev = state.get('accumulated', [])
    return {"step": 2, "value": "B", "accumulated": prev + ["B"]}

def node_c(state: CounterState) -> dict:
    prev = state.get('accumulated', [])
    return {"step": 3, "value": "C", "accumulated": prev + ["C"]}

builder = StateGraph(CounterState)
builder.add_node("A", node_a)
builder.add_node("B", node_b)
builder.add_node("C", node_c)
builder.set_entry_point("A")
builder.add_edge("A", "B")
builder.add_edge("B", "C")
builder.add_edge("C", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "time_travel_test"}}

# 执行
result = graph.invoke({"step": 0, "value": "", "accumulated": []}, config)
print(f"最终状态: step={result['step']}, value={result['value']}")
print(f"累积路径: {' → '.join(result['accumulated'])}")

# 查询历史
history = list(graph.get_state_history(config))
print(f"\n历史检查点: {len(history)}")
for i, h in enumerate(history):
    print(f"  [{i}] step={h.values.get('step')}, value={h.values.get('value')}")

# 从中间检查点回退
if len(history) >= 2:
    target = history[2]  # 回退到step 1
    print(f"\n从检查点回退: step={target.values.get('step')}")
    # 使用该checkpoint的config重新执行
    result_rewind = graph.invoke(None, target.config)
    print(f"回退后重新执行: step={result_rewind.get('step')}, "
          f"value={result_rewind.get('value')}")

print("\n✅ 练习4完成")
```

---

## 练习5：设计Master-Sub架构

> 此练习的完整答案请参考 `07-master-sub-architecture.py`。核心设计要点：

```python
# 关键1：共享State基类，确保字段名一致
class BaseState(TypedDict):
    user_query: str
    final_response: str

class RAGSubState(BaseState):
    retrieved_docs: list
    rag_answer: str  # 与Master中的字段名一致！

class MasterState(BaseState):
    intent: str
    retrieved_docs: list
    rag_answer: str   # 同一个字段名！
    tool_result: str

# 关键2：子图入口显式映射
def rag_entry(master_state: MasterState) -> dict:
    sub_input = {
        "user_query": master_state['user_query'],
        "retrieved_docs": [],
        "rag_answer": "",
    }
    sub_graph = build_rag_subgraph()
    sub_result = sub_graph.invoke(sub_input)
    # 返回时使用Master的字段名
    return {
        "rag_answer": sub_result['rag_answer'],
        "retrieved_docs": sub_result['retrieved_docs'],
    }
```

---

## 练习6：实现Redis分布式锁

> 完整答案请参考 `08-redis-distributed-lock.py`。核心要点：

```python
# 上下文管理器模式
lock = DistributedLock("task-001", ttl_ms=5000)

with lock:  # 自动获取锁
    execute_critical_section()
    # 锁在退出with块时自动释放

# 多实例测试
def test_concurrent_access():
    lock_a = DistributedLock("shared-task", acquire_timeout_ms=5000)
    lock_b = DistributedLock("shared-task", acquire_timeout_ms=500)

    with lock_a:  # 实例A获取锁
        try:
            with lock_b:  # 实例B尝试获取 → 应超时
                assert False
        except TimeoutError:
            pass  # 预期行为
```

---

## 练习7：集成消息队列

> 完整答案请参考 `09-message-queue-integration.py`。核心流程：

```python
# 发布
publisher = Publisher(queue)
msg_id = publisher.publish("task_type", {"state": {...}})
# → 返回 202 Accepted

# 消费
consumer = Consumer(queue, graph)
consumer.start()  # 启动轮询

# 内部处理：
# while running:
#     msg = queue.consume()
#     try:
#         result = graph.invoke(msg.payload)
#         queue.ack(msg.id)
#     except Exception:
#         queue.nack(msg, error)  # 自动重试或进DLQ
```

---

## 练习8：构建企业级RAG Agent

> 完整答案请参考 `10-enterprise-rag-agent-complete.py`（约500行）。

架构摘要：

```text
EnterpriseRAGAgent
├── AppSettings (pydantic-settings环境配置)
├── MasterGraph (意图分类 + 路由)
│   ├── RAG SubGraph (检索→生成→验证循环)
│   ├── Tool SubGraph (工具调用→返回)
│   └── Human SubGraph (审批流程)
├── Checkpointer (状态持久化)
├── DistributedLock (防重复执行)
├── MessageQueue (异步任务)
└── API (同步/异步/状态查询/健康检查)
```

核心特性：
- `ask()`: 同步接口
- `ask_async()`: 异步接口（返回202 + 轮询）
- `get_async_result()`: 查询异步任务结果
- `get_stats()`: 运行统计
- `health_check()`: 健康检查
- 完整的错误处理和日志

---

以上8道练习的解答覆盖了LangGraph企业级开发的核心技能。完成这些练习后，你已具备设计和生产化LangGraph Agent系统的能力。
