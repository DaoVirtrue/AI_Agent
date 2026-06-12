# Supplement 04 检验点：练习题

---

## 练习 1：设计客服对话 FSM
设计一个技术支持客服的状态机，包含至少 6 个状态和 12 条迁移规则。要求：
- 状态：GREETING → COLLECTING_ISSUE → TROUBLESHOOTING → CONFIRMING_RESOLVED → COLLECTING_FEEDBACK → FAREWELL
- 超时：TROUBLESHOOTING 状态 120s 无进展自动转人工
- 逃生：任意状态下"转人工"直接跳转到 HANDOFF 状态

---

## 练习 2：航班订票槽位填充
实现 SlotFillingFSM 收集以下 8 个槽位：
- 出发城市（required）、目的城市（required）、出发日期（required，格式 YYYY-MM-DD）
- 返程日期（optional）、乘客人数（required，1-9）、舱位（optional，经济/商务/头等）
- 手机号（required，11位）、邮箱（optional，含@）

要求：每槽位最多重试 3 次，失败用默认值。

---

## 练习 3：递归摘要压缩
实现 `recursive_summarize(messages, max_tokens=500)` 函数：
- 如果消息总 token 数 ≤ max_tokens，直接拼接返回
- 否则分成 N 段，每段独立摘要，再递归摘要各段的摘要
- 最终摘要需标注压缩比（原始 token / 最终 token）

---

## 练习 4：艾宾浩斯淘汰策略
实现 `EbbinghausMemoryStore`，在现有 MemoryStore 基础上增加：
- 每条记忆存储时记录 `strength`（0-1，重要性越高 strength 越大）
- `apply_forgetting(elapsed_days)` 方法：计算每条记忆的保留率 R = e^(-t/(s*7))，淘汰 R < 0.3 的
- 可视化：打印每天淘汰数量

---

## 练习 5：200K 上下文窗口分配
为一个 RAG Agent 设计 StructuredContextWindow：
- 总预算 200K tokens
- 5 个区域：INSTRUCTION(5K) / TOOL_SCHEMAS(8K) / MEMORY(50K) / RETRIEVED_DOCS(120K) / OUTPUT(17K)
- 实现 `add_to_zone()` 和溢出处理：当 RETRIEVED_DOCS 区域满时，自动压缩最旧的文档

---

## 练习 6：4 Agent 共享黑板
实现 SharedMemoryBlackboard，4 个 Agent 协作规划旅行：
- travel_agent（查航班/酒店）、budget_agent（控制预算）、weather_agent（查天气）、summary_agent（汇总方案）
- travel_agent 写入 flight_info 和 hotel_info
- budget_agent 只读 travel_agent 的数据，写入 budget_analysis
- summary_agent 读所有数据后写 final_plan
- 使用 pub-sub：summary_agent 订阅 "budget:#" 和 "travel:#"

---

## 练习 7：SQLite 持久化记忆
实现 SQLiteMemoryStore：
- 表结构：id, user_id, content, embedding(BLOB), importance, created_at, memory_type, tags
- 方法：add/get/update/delete/query/search_embedding/get_stats
- 版本管理：每次 update 记录 version 历史
- 增量同步：get_changes_since(timestamp) → apply_incoming_changes(changes)

---

## 练习 8：记忆质量评估
对 100 条记忆运行评估：
- retrieval_test：20 个查询，计算 precision@3, recall@3, MRR
- compression_test：10 条原文 vs 摘要，计算事实保留率
- hallucination_test：检测摘要中是否有原文不存在的事实
- freshness_test：评估不同年龄的记忆在当前上下文下的相关性

---

## 练习 9：Agent 记忆自省
为 Agent 添加 `reflect_on_memory()` 方法：
- 评估当前记忆对任务的完整度（0-1）
- 检测矛盾记忆（如两条记忆给出不同答案）
- 如有矛盾，标记低置信度记忆并请求 LLM 重巩固
- 输出 MemoryHealthReport

---

## 练习 10：流式记忆累积
实现 StreamingMemoryOrchestrator：
- 模拟 5 轮对话的流式 token 输入（每 100ms 一个 chunk）
- 每轮结束后 commit_turn() 将完整内容写入记忆
- 在第 3 轮实现 Early-Exit 检索（在用户说到 50% 时就预检索）
- 背压处理：当写入队列 > 10 时触发 backpressure，暂停流式输入
