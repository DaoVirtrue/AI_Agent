# Phase 08 检验点：LangGraph 企业级

> **完成标准**：独立完成以下8道练习题，能解释每道题的设计思路和代码逻辑。
> **建议时间**：3-4小时
> **难度标记**：⭐ 基础 | ⭐⭐ 中等 | ⭐⭐⭐ 挑战

---

## 练习1 ⭐⭐：构建一个带条件路由的StateGraph

**任务**：从头构建一个包含以下功能的StateGraph。

**要求**：
1. 定义State（至少包含：query, intent, answer, confidence）
2. 实现至少3个节点（classify, respond, fallback）
3. 实现条件路由：如果confidence < 0.5，走fallback节点
4. 编译图并可视化（draw_mermaid）
5. 用至少3种不同的query测试路由是否正确

**测试用例**：
```python
# 简单查询 → 高置信度
result = graph.invoke({"query": "1+1等于多少?", ...})

# 复杂问题 → 低置信度 → fallback
result = graph.invoke({"query": "请详细分析..." * 10, ...})
```

---

## 练习2 ⭐⭐：实现带Checkpointer的状态持久化

**任务**：为练习1的图添加Checkpointer，支持状态持久化和恢复。

**要求**：
1. 使用MemorySaver作为Checkpointer
2. 用不同的thread_id隔离两个独立会话
3. 验证：同一个thread_id多次invoke时状态是累积的
4. 验证：不同thread_id的状态完全独立
5. 检查checkpoint数量是否与预期一致

**测试用例**：
```python
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 会话A：两次调用
config_a = {"configurable": {"thread_id": "user_A"}}
result_a1 = graph.invoke({"query": "你好"}, config_a)
result_a2 = graph.invoke({"query": "再见"}, config_a)

# 验证状态累积
assert result_a2的某些字段 >= result_a1的对应字段

# 会话B：独立
config_b = {"configurable": {"thread_id": "user_B"}}
result_b1 = graph.invoke({"query": "Hello"}, config_b)
# 验证B不受A影响
```

---

## 练习3 ⭐⭐⭐：实现Human-in-the-Loop中断

**任务**：构建一个需要人工审批的图，实现interrupt()和恢复逻辑。

**要求**：
1. 图至少包含：validate（验证节点）、approve（审批节点，含interrupt）、execute（执行节点）
2. 在approve节点中调用interrupt()暂停执行
3. 模拟人工审批通过Command(resume=...)恢复
4. 如果审批被拒绝，走reject节点
5. 记录每次interrupt和恢复的审计日志

**测试用例**：
```python
# 第一次invoke → 在approve节点暂停
try:
    result = graph.invoke(state)
except GraphInterrupt as e:
    interrupt_data = e.args[0]
    # 模拟人工批准
    resume_cmd = Command(resume={"decision": "approved", "reason": "合规"})
    result = graph.invoke(resume_cmd, config)
```

---

## 练习4 ⭐⭐：实现时间旅行

**任务**：使用get_state_history进行状迭代历史查询和回退。

**要求**：
1. 创建一个至少3个节点的图
2. 执行图并记录每个checkpoint
3. 使用get_state_history查询历史
4. 从特定checkpoint重新invoke（回退执行）
5. 对比原始执行和回退执行的差异

**测试用例**：
```python
history = list(graph.get_state_history(config))
print(f"共有 {len(history)} 个检查点")

# 找到第二个检查点
checkpoint = history[1]
# 从该点重新执行
result = graph.invoke(None, checkpoint.config)
```

---

## 练习5 ⭐⭐⭐：设计Master-Sub架构

**任务**：设计一个包含至少2个子图的Master-Sub架构。

**要求**：
1. Master图包含intent分类和路由
2. 至少2个子图：RAG子图（检索→生成）和Tool子图（工具调用→返回）
3. Master和Sub的State字段使用相同的命名（通过基类继承）
4. 实现子图入口的显式输入/输出映射
5. 端到端测试验证：rag查询 → rag子图，tool查询 → tool子图

**测试用例**：
```python
master = build_master_graph()
result_rag = master.invoke({"user_query": "搜索RAG最佳实践"})
assert result_rag["intent"] == "rag"
assert len(result_rag["rag_answer"]) > 0

result_tool = master.invoke({"user_query": "计算3.14*2.71"})
assert result_tool["intent"] == "tool"
assert len(result_tool["tool_result"]) > 0
```

---

## 练习6 ⭐⭐：实现Redis分布式锁

**任务**：实现一个保护Graph执行的分布式锁。

**要求**：
1. 实现锁的acquire()和release()方法
2. 锁支持上下文管理器（with语句）
3. 锁有超时机制（acquire_timeout）
4. 两个"实例"同时尝试获取同一锁时，只有一个成功
5. 模拟长时间任务时需要锁续约

**测试用例**：
```python
lock = DistributedLock("task-001", ttl_ms=5000)

# 实例A获取锁
with lock:
    # 实例B尝试获取（应该失败）
    lock_b = DistributedLock("task-001", acquire_timeout_ms=500)
    try:
        with lock_b:
            assert False, "不应该获取到锁"
    except TimeoutError:
        pass  # 预期行为
```

---

## 练习7 ⭐⭐：集成消息队列

**任务**：实现异步任务队列的发布-消费流程。

**要求**：
1. 实现简化版的Publisher（发布任务）
2. 实现简化版的Consumer（消费任务，带ack/nack）
3. 失败消息自动重试（最多3次）
4. 重试耗尽后消息进入DLQ
5. 提供DLQ重新入队功能

**测试用例**：
```python
queue = SimpleMQ()
publisher = Publisher(queue)
consumer = Consumer(queue)

# 发布任务
msg_id = publisher.publish("分析任务", {"query": "..."})

# 消费并处理
msg = queue.consume()
try:
    result = process(msg)
    queue.ack(msg.id)
except Exception:
    queue.nack(msg, "处理失败")

# 验证
stats = queue.get_stats()
assert stats["dlq_length"] == 0  # 正常任务不应进DLQ
```

---

## 练习8 ⭐⭐⭐：构建企业级RAG Agent

**任务**：综合运用本阶段所有技术，从零构建一个企业级RAG Agent。

**要求**：
1. 包含Master-Sub架构（至少rag和tool两个子图）
2. RAG子图包含检索→生成→验证→重试循环
3. 使用Checkpointer（MemorySaver）支持状态持久化
4. 包含分布式锁保护（防止重复执行）
5. 支持HITL：敏感查询触发人工审批
6. 支持异步模式：通过消息队列异步处理
7. 提供完整的API接口（同步/异步/状态查询）
8. 包含健康检查和统计接口
9. 提供可运行的__main__演示块
10. 英文代码 + 中文注释

**评价标准**：
- 架构设计清晰，各组件职责分明（30%）
- 所有功能正常可运行（25%）
- 错误处理和边界条件完善（20%）
- 代码结构和注释质量（15%）
- 统计/监控/健康检查完善（10%）

---

## 自检清单

完成以上练习后，确认以下能力：

- [ ] 我能从零构建带条件路由和循环的StateGraph
- [ ] 我能使用Checkpointer实现跨会话的状态持久化
- [ ] 我能实现interrupt()和Command(resume=...)的人机协作
- [ ] 我能使用get_state_history进行时间旅行调试
- [ ] 我能设计Master-Sub架构并正确映射State字段
- [ ] 我能实现分布式锁保护关键代码段
- [ ] 我能集成消息队列实现异步任务处理
- [ ] 我能从零构建包含全部特性的企业级RAG Agent

---

**恭喜！完成Phase 08全部练习后，你已经具备了设计和生产化LangGraph企业级Agent的能力。**
