# 坑点 #3：辩论死锁（Debate Deadlock）

> "我们让3个Agent辩论'最佳技术方案'。Agent A坚决支持微服务，Agent B坚定拥护单体架构，Agent C从中调和。45分钟和$23的API费用之后，他们还在争论同一个问题——没有收敛，没有结论。"

---

## 症状

1. **无限循环**：辩论Agent陷入无限循环的讨论，始终无法达成共识
2. **观点极化**：随着辩论轮次增加，Agent的立场不仅没有靠拢，反而越来越极端
3. **穷举式争论**：Agent不断引入新的论点和反驳，辩论范围持续扩大而非收窄
4. **"平局死锁"**：正反双方得分相同，Judge无法做出倾向性裁决
5. **成本失控**：每次辩论轮次调用所有Agent的LLM，成本 = 轮次数 * Agent数 * 单次调用费用

---

## 根因

**多Agent辩论系统缺乏"收敛机制"。**

```
理论上：辩论通过正反交锋，逐步接近真理
实际上：没有收敛压力 → Agent为了"赢"而辩论 → 立场极化 → 永不结束
```

具体原因：
1. **没有最大轮次限制**（或限制过于宽松）
2. **没有"观点必须趋近"的机制**（Agent只负责反驳，不负责求同）
3. **Judge的评分标准不够明确**，导致双方得分始终接近
4. **辩论主题本身不适合对抗性讨论**（如个人偏好问题）

---

## 解法

### 1. 硬性收敛策略

```python
class ConvergentDebateEngine:
    """
    带收敛保证的辩论引擎。
    在达到最大轮次后强制进行最终投票，不再开启新轮次。
    """

    def __init__(self, max_rounds: int = 3, convergence_threshold: float = 0.10):
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.previous_round_scores: list[float] = []

    def run_debate(self, topic: str, agents: list) -> DebateResult:
        for round_num in range(1, self.max_rounds + 1):
            # 执行一轮辩论
            round_statements = self._execute_round(round_num, topic, agents)

            # 每轮结束后立即评分
            round_scores = self._score_round(round_statements, agents)
            self.previous_round_scores.append(round_scores)

            # 检查是否已收敛
            if self._has_converged(round_scores):
                # 提前结束
                break

        # 强制最终投票
        final_scores = self._final_vote(agents)
        winner = self._determine_winner(final_scores)

        return DebateResult(winner=winner, ...)

    def _has_converged(self, current_scores: dict) -> bool:
        """检查辩论是否已收敛（观点不再显著变化）。"""
        if len(self.previous_round_scores) < 2:
            return False

        prev = self.previous_round_scores[-2]
        curr = current_scores

        # 计算分数变化幅度
        max_change = max(
            abs(curr.get(aid, 0) - prev.get(aid, 0))
            for aid in set(list(prev.keys()) + list(curr.keys()))
        )

        # 变化小于阈值 → 已收敛
        return max_change < self.convergence_threshold

    def _final_vote(self, agents: list) -> dict:
        """强制最终投票：每个Agent必须给出明确的赞成/反对/弃权。"""
        votes = {}
        for agent in agents:
            # 要求Agent做出最终选择（不能继续"一方面...另一方面..."）
            prompt = (
                f"辩论已结束。你必须做出最终决定：\n"
                f"请从以下选项中选择一个：\n"
                f"A. 支持（SUPPORT）\n"
                f"B. 反对（OPPOSE）\n"
                f"C. 弃权（ABSTAIN）\n\n"
                f"必须选择一个选项，不能给出模棱两可的回答。"
            )
            # LLM调用获取最终投票
            vote = self._get_final_vote(agent, prompt)
            votes[agent.agent_id] = vote

        return votes
```

### 2. 加权投票与Tie-Break机制

```python
class TieBreakJudge:
    """
    带平局打破机制的Judge。
    当双方得分差距小于阈值时，自动触发Tie-Break规则。
    """

    TIE_THRESHOLD = 0.05  # 得分差距小于5%视为平局

    TIE_BREAK_RULES = [
        # 规则1：引用更多可靠来源的一方获胜
        lambda pro, con: "pro" if pro.citation_count > con.citation_count else
                         "con" if con.citation_count > pro.citation_count else "tie",
        # 规则2：逻辑一致性更高的一方获胜
        lambda pro, con: "pro" if pro.logic_score > con.logic_score else
                         "con" if con.logic_score > pro.logic_score else "tie",
        # 规则3：风险更低的一方获胜（企业中常用）
        lambda pro, con: "con",  # 默认保守：有疑虑时选择不行动
    ]

    def judge(self, pro_side, con_side) -> str:
        pro_score = self.calculate_score(pro_side)
        con_score = self.calculate_score(con_side)

        if abs(pro_score - con_score) > self.TIE_THRESHOLD:
            # 有明显差距，直接裁决
            return "pro" if pro_score > con_score else "con"

        # 平局：逐条应用Tie-Break规则
        for rule in self.TIE_BREAK_RULES:
            result = rule(pro_side, con_side)
            if result != "tie":
                return result

        # 所有规则都是平局 → 选择保守方案
        return "con"  # 默认保守
```

### 3. 时间和Token硬上限

```python
class DebateGuard:
    """
    辩论守卫：强制执行辩论的硬性限制。
    超时或超Token时强制终止。
    """

    def __init__(
        self,
        max_duration_seconds: float = 120.0,  # 最多2分钟
        max_total_tokens: int = 50000,         # 最多5万Token
        max_cost_usd: float = 2.00,            # 最多$2
    ):
        self.max_duration = max_duration_seconds
        self.max_tokens = max_total_tokens
        self.max_cost = max_cost_usd
        self.start_time = time.time()
        self.total_tokens = 0
        self.total_cost = 0.0

    def check_and_record(
        self,
        tokens_used: int,
        cost: float,
    ) -> bool:
        """
        检查是否超出限制，并记录消耗。
        返回 True 表示可以继续，False 表示必须终止。
        """
        self.total_tokens += tokens_used
        self.total_cost += cost
        elapsed = time.time() - self.start_time

        if elapsed > self.max_duration:
            return False
        if self.total_tokens > self.max_tokens:
            return False
        if self.total_cost > self.max_cost:
            return False

        return True

    def force_terminate(self) -> str:
        """强制终止时返回此时的最佳结果。"""
        return (
            f"辩论已达到预设限制，强制终止。\n"
            f"耗时: {time.time() - self.start_time:.1f}s\n"
            f"Token: {self.total_tokens}\n"
            f"费用: ${self.total_cost:.4f}\n"
            f"建议：基于当前已进行的讨论，由人工做出最终决策。"
        )
```

---

## 检查清单

- [ ] 设置了最大辩论轮次（不超过5轮）
- [ ] 实现了收敛检测（观点变化小于阈值时提前结束）
- [ ] 有明确的Tie-Break机制（不止一个裁决维度）
- [ ] 设置了时间/Token/费用的硬上限
- [ ] 超限时有优雅的降级策略（而非崩溃）
- [ ] 辩论主题适合对抗性讨论（有明确的正反可论证立场）
- [ ] 记录了每次辩论的收敛曲线，用于优化轮次设置

---

**一句话总结**：辩论是手段，结论是目的。如果辩论永远不结束，它就从"求真工具"变成了"Token焚化炉"。设置硬性限制，确保辩论收敛。
