# 陷阱 1：Agent 无限循环 (Infinite Loop)

## 概述

Agent 无限循环是最常见的陷阱之一。Agent 在 Thought→Action→Observation 循环中不断调用工具，却永远不输出 Final Answer。这会导致 API 费用飙升、响应超时、用户体验极差。

---

## 症状 (Symptoms)

1. **永远不停止**：Agent 在循环中反复调用工具，无论执行多少步都不输出 Final Answer
2. **重复调用**：Agent 用相同或微调的参数反复调用同一个工具
3. **过度收集**：Agent 认为"还需要更多信息"，不断搜索无关数据
4. **日志膨胀**：Scratchpad 中堆积大量冗余的 Action-Observation 对
5. **API 费用异常**：Token 消耗远超预期

---

## 根本原因 (Root Cause)

### 1. 缺少终止信号

LLM 没有明确的"我已经有足够信息"的判断标准。在 ReAct 格式中，LLM 可以无限续写 `Thought → Action` 而不产生 `Final Answer`。

### 2. Prompt 设计缺陷

Prompt 中没有明确告诉模型"何时应该停止"。模型倾向于"做更多事情"而不是"回答问题"。

### 3. 工具返回值模糊

工具的返回信息不够明确或缺少关键信息，导致 Agent 认为"信息不足"。

### 4. 模型输出格式错误

LLM 没有严格遵循 ReAct 格式，输出的 Action 无法被正则解析，导致 Observation 始终是错误提示。

---

## 真实场景 (Real Scenario)

### 场景描述

用户询问：**"北京今天天气怎么样？"**

Agent 的行为记录：

```
Step 1:
  Thought: 用户想知道北京天气，让我查询
  Action: get_weather[city=北京, date=today]
  Observation: 北京 的天气 (today): 晴天，22°C，湿度45%，北风3级

Step 2:
  Thought: 天气信息还不够完整，还需要知道明天的天气来做对比
  Action: get_weather[city=北京, date=tomorrow]
  Observation: 北京 的天气 (tomorrow): 多云，18°C，湿度60%

Step 3:
  Thought: 还需要知道昨天的天气来看趋势
  Action: get_weather[city=北京, date=yesterday]
  Observation: 北京 的天气 (yesterday): 小雨，15°C

Step 4:
  Thought: 也许还需要风的详细信息，再查一次
  Action: get_weather[city=北京, date=today]
  Observation: 北京 的天气 (today): 晴天，22°C，湿度45%，北风3级

Step 5:
  Thought: 我感觉还需要更多数据...
  Action: web_search[query=北京气候特征]
  Observation: (大量北京气候信息)...

... (继续 10+ 步) ...
```

**问题分析**：Agent 在 Step 1 获得天气信息后就足以回答用户问题，但模型陷入了"持续探索"的模式。用户只问了今天的天气，Agent 却查了今天、明天、昨天，还去搜索了气候特征。

---

## 解决方案

### 方案 1：max_steps 硬限制 + 信息丰富的强制停止消息

最简单的防御：设置最大步数，到达上限时强制要求 LLM 输出 Final Answer。

```python
class ReActAgentWithLimit:
    def __init__(self, llm, tools, max_steps=15):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        conversation = [self._build_prompt(task)]

        for step in range(1, self.max_steps + 1):
            response = self.llm.generate("\n".join(conversation))

            # 检查 Final Answer
            final = self._extract_final_answer(response)
            if final:
                return final

            # 执行工具
            action = self._parse_action(response)
            if action:
                observation = self._execute_tool(*action)
                conversation.append(response)
                conversation.append(f"Observation: {observation}")
            else:
                conversation.append(response)
                conversation.append(
                    "Observation: 未检测到有效的 Action。"
                    "如果你已经可以回答问题，请输出 'Final Answer: ...'。"
                )

        # 【关键】到达 max_steps 后的强制终止
        # 不只是抛出一个错误，而是给 LLM 最后一次机会来总结
        conversation.append(
            f"你已经执行了 {self.max_steps} 步操作，达到了系统设定的最大限制。\n\n"
            f"以下是你在对话中收集到的所有 Observation 汇总：\n"
            f"{self._summarize_observations(conversation)}\n\n"
            f"请基于以上汇总信息，直接输出 Final Answer。\n"
            f"Final Answer:"
        )

        final_response = self.llm.generate("\n".join(conversation))
        final = self._extract_final_answer(final_response)
        return final or "任务未能在规定步数内完成。"

    def _summarize_observations(self, conversation: list) -> str:
        """汇总所有 Observation，帮助 LLM 在强制终止时做出回答。"""
        observations = []
        for msg in conversation:
            if msg.startswith("Observation:"):
                observations.append(msg.replace("Observation:", "").strip())
        if not observations:
            return "（没有收集到任何数据）"
        return "\n".join(f"{i+1}. {obs}" for i, obs in enumerate(observations))
```

### 方案 2：Prompt 工程 —— 引导终止行为

在 System Prompt 中加入明确的终止条件：

```python
def _build_prompt_with_stop_conditions(self, task: str) -> str:
    return f"""你是一个高效的 AI Agent。

## 可用工具
{self._build_tools_description()}

## ReAct 输出格式
Thought: <你的推理>
Action: tool_name[param=value]
Observation: <工具返回>

## 【重要】何时必须输出 Final Answer（不要继续调用工具）
1. 当你已经获得了能直接回答用户问题的信息时 → 立即输出 Final Answer
2. 当同一个工具连续调用 2 次返回相同结果时 → 停止，输出 Final Answer
3. 当工具返回信息已经覆盖了用户问题的所有方面时 → 立即输出 Final Answer
4. 当你不确定是否还需要更多信息时 → **宁可回答不完整，也不要继续调用工具**
5. 如果你计划调用的工具和已经调用过的工具功能重叠 → 不要重复调用

## 自检问题
在每次 Action 之前，先问自己：
- "我现在掌握的信息是否已经足够回答用户？"
- 如果答案是"是" → 输出 Final Answer
- 如果答案是"不够" → 明确写出"缺少什么信息"，再执行 Action

现在请解决问题：
Question: {task}
"""
```

### 方案 3：反思检查 (Reflection Check)

在每个 Observation 之后加入反思逻辑，判断是否需要继续：

```python
class ReActAgentWithReflection(ReActAgent):
    """带反思检查的 ReAct Agent。"""

    def run(self, task: str) -> str:
        prompt = self._build_prompt(task)
        context = [prompt]

        for step in range(self.max_steps):
            response = self.llm.generate("\n".join(context))

            final = self._extract_final_answer(response)
            if final:
                return final

            action = self._parse_action(response)
            if action is None:
                # 格式错误，尝试修正
                context.append(response)
                context.append("Observation: 请输出 Action 或 Final Answer。")
                continue

            tool_name, params = action
            observation = self._execute_tool(tool_name, params)
            context.append(response)
            context.append(f"Observation: {observation}")

            # 【反思检查】让 LLM 评估是否需要继续
            reflection_prompt = (
                f"基于以上对话，你已执行 {step + 1} 步。\n"
                f"请用一句话回答：你是否有足够的信息来回答用户的原始问题？"
                f"如果可以，请立即输出 Final Answer。否则，说明还需要什么信息。"
            )
            context.append(reflection_prompt)
            reflection_response = self.llm.generate("\n".join(context))
            context.pop()  # 移除反思提示，保持主干对话清晰

            if self._has_markers_of_sufficiency(reflection_response):
                # LLM 评估认为信息已足够
                context.append(
                    "好的，请基于所有已获取的信息，输出 Final Answer。"
                )
                final_response = self.llm.generate("\n".join(context))
                final = self._extract_final_answer(final_response)
                if final:
                    return final

        return "未能完成。"

    def _has_markers_of_sufficiency(self, text: str) -> bool:
        """检查 LLM 是否认为信息已足够。"""
        sufficiency_markers = [
            "足够", "可以回答", "信息充足", "已经获取",
            "sufficient", "enough", "可以了", "没问题",
        ]
        insufficiency_markers = [
            "不够", "还需要", "需要更多", "不足",
            "not enough", "insufficient", "missing",
        ]
        text_lower = text.lower()
        # 先检查"不足"标记
        for marker in insufficiency_markers:
            if marker in text_lower:
                return False
        # 再检查"足够"标记
        for marker in sufficiency_markers:
            if marker in text_lower:
                return True
        return False
```

### 方案 4：步骤预算 (Step Budget)

为每个子任务分配步骤预算，在预算耗尽前发出警告：

```python
class StepBudgetManager:
    """管理 Agent 的步骤预算，防止单步子任务消耗过多步数。"""

    def __init__(self, total_budget: int = 15):
        self.total_budget = total_budget
        self.budgets: dict[str, int] = {}  # 子任务 → 预算
        self.used: dict[str, int] = {}     # 子任务 → 已用步数

    def allocate(self, subtask: str, budget: int):
        """为子任务分配步骤预算。"""
        self.budgets[subtask] = budget
        self.used[subtask] = 0

    def consume(self, subtask: str) -> str:
        """消耗一步，返回状态信息。"""
        self.used[subtask] = self.used.get(subtask, 0) + 1
        budget = self.budgets.get(subtask, 5)
        used = self.used[subtask]
        remaining = budget - used

        if remaining <= 0:
            return f"[预算耗尽] 子任务 '{subtask}' 已用完 {budget} 步预算，请立即总结。"
        elif remaining <= budget * 0.3:
            return f"[预算警告] 子任务 '{subtask}' 还剩 {remaining}/{budget} 步 (已用 {used} 步)，请尽快得出结论。"
        elif remaining <= budget * 0.5:
            return f"[预算提示] 子任务 '{subtask}' 还剩 {remaining}/{budget} 步。"
        return ""

    def warn_at_50_percent(self, subtask: str) -> str:
        """在剩余 50% 预算时发出警告。"""
        budget = self.budgets.get(subtask, 5)
        used = self.used.get(subtask, 0)
        if used >= budget * 0.5:
            remaining = budget - used
            return (
                f"[预算警告] 你已使用 {used}/{budget} 步处理 '{subtask}'，"
                f"还剩 {remaining} 步。如果还需要更多信息，请考虑能否基于已有信息给出答案。"
            )
        return ""


# 使用示例
def run_with_budget(agent, task: str) -> str:
    budget_manager = StepBudgetManager(total_budget=15)

    # 分析任务，分配预算
    subtasks = agent._analyze_subtasks(task)  # 假设有此方法
    for i, subtask in enumerate(subtasks):
        # 愈靠后的子任务预算愈少（推动尽快结束）
        budget_manager.allocate(subtask, max(5 - i, 2))

    prompt = agent._build_prompt(task)
    context = [prompt]
    current_subtask = subtasks[0] if subtasks else "main"

    for step in range(agent.max_steps):
        # 预算警告注入到上下文
        warning = budget_manager.warn_at_50_percent(current_subtask)
        if warning:
            context.append(f"System: {warning}")

        response = agent.llm.generate("\n".join(context))

        # ...（正常的 ReAct 循环逻辑）...
        budget_manager.consume(current_subtask)

    return "任务完成或达到预算上限。"
```

---

## 预防清单 (Checklist)

在生产环境中部署 Agent 前，请确保以下各项都已处理：

- [ ] 1. 设置了合理的 `max_steps`（默认 10-15，复杂任务可到 20-30）
- [ ] 2. System Prompt 中明确包含了"何时停止"的规则
- [ ] 3. 达到了 `max_steps` 时有优雅的降级处理（汇总 Observation 后强制回答）
- [ ] 4. 每个工具的描述中包含了"何时使用 / 何时不使用"的说明
- [ ] 5. 实现了步骤计数器和日志，便于事后排查
- [ ] 6. 对重复调用同一工具进行了检测和警告
- [ ] 7. 部署了 API 调用次数和 Token 消耗的监控告警
- [ ] 8. 在 Prompt 中加入"宁可回答不完整，也不要过度探索"的指令

---

## 总结

无限循环是 Agent 开发中最常见也最危险的问题。以下是核心要点：

| 方案 | 实施难度 | 效果 | 副作用 |
|------|---------|------|--------|
| max_steps 硬限制 | 低 | 兜底 | 可能截断有效推理 |
| Prompt 工程 | 低 | 中 | 增加 Prompt 长度 |
| 反思检查 | 中 | 高 | 额外 Token 消耗 |
| 步骤预算 | 中 | 高 | 需要预分析任务 |

推荐组合策略：**max_steps（兜底）+ Prompt 工程（预防）+ 反思检查（优化）**。

---

> **下一陷阱**: [02-tool-call-failure.md](./02-tool-call-failure.md) —— 工具调用失败如何优雅恢复
