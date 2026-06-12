# 坑点 #2：幻觉级联放大（Hallucination Cascade）

> "Agent A说'2024年市场增长23%'，Agent B基于这个数字推算出'总投资回报率可达180%'，Agent C以此为基础撰写了投资建议书。后来发现Agent A的23%是凭空编造的——但此时错误已经经历了2次放大，最终结论完全不可信。"

---

## 症状

1. **错误逐级放大**：上游Agent（如ResearchAgent）产生一个细微偏差，下游Agent（如AnalysisAgent）将其作为"事实"进一步推理，偏差被指数级放大
2. **"引用链断裂"**：最终输出引用了看似有来源的数据，但追溯3步后发现最原始的来源是AI捏造的
3. **过度自信**：每个Agent都以高置信度输出结果，但整体的错误率远超单个Agent——系统性幻觉比例可达15-30%
4. **难以溯源**：错误发生后，很难定位是哪个Agent在哪个环节首次引入错误

---

## 根因

**多Agent流水线中的"信任传染"效应**：每个Agent默认信任上游输入，不进行独立的事实核查。

```
Agent A: "据McKinsey报告，市场增长23%"
          ↓ (23%是A的幻觉，但B不知道)
Agent B: "基于23%的增长率，预测ROI为180%"
          ↓ (180%是基于幻觉的二次推理)
Agent C: "本投资建议基于严谨的市场分析：回报率180%"
          ↓ (彻底不可信)
```

Agent之间缺乏"质疑机制"——在单Agent系统中，用户可以质疑输出；但在多Agent流水线中，Agent之间默认是"信任链"关系。

---

## 解法

### 1. 交叉验证节点（Cross-Validation Node）

在关键Agent之间插入验证环节，要求每个Agent验证上游数据：

```python
class FactCheckerAgent:
    """在任务流水线中插入的事实检查Agent，负责验证上游Agent的关键声明。"""

    def verify_claims(self, upstream_output: str) -> dict:
        """
        提取上游输出中的关键声明并逐一验证。
        """
        claims = self.extract_claims(upstream_output)

        verified = {}
        for claim in claims:
            # 检查：这个声明有引用来源吗？
            has_citation = any(
                marker in claim.lower()
                for marker in ["来源", "参考", "根据", "据", "source", "ref", "citation"]
            )

            # 检查：数据是否在合理范围内？
            numbers = self.extract_numbers(claim)
            plausibility = all(self.is_plausible(n) for n in numbers)

            # 检查：声明是否与已知事实一致？
            consistency = self.check_consistency(claim)

            verified[claim] = {
                "has_citation": has_citation,
                "plausibility": plausibility,
                "consistency": consistency,
                "trustworthy": has_citation and plausibility and consistency,
            }

        return verified

    def extract_claims(self, text: str) -> list[str]:
        """从文本中提取关键声明。"""
        import re
        # 提取包含数字、百分比的句子（这些最容易产生幻觉）
        claims = []
        patterns = [
            r'([^。.]*?\d+%[^。.]*[。.])',       # 含百分比的句子
            r'([^。.]*?\d+亿美元[^。.]*[。.])',   # 含金额的句子
            r'([^。.]*?(?:增长|下降|提升|降低)[^。.]*[。.])',  # 含趋势的句子
        ]
        for pattern in patterns:
            claims.extend(re.findall(pattern, text))
        return claims[:10]  # 限制数量

    def is_plausible(self, number: float) -> bool:
        """判断数字是否在合理范围内。"""
        # 增长率通常在 -50% ~ +200% 之间
        if -0.5 <= number <= 2.0:
            return True
        return False

    def check_consistency(self, claim: str) -> bool:
        """检查声明与已知事实的一致性（知识库查询）。"""
        # 生产环境中：查询知识库、调用搜索引擎、检查数据库
        # 此处简化
        return True  # 需要外部验证才能返回True

# 在流水线中插入验证节点
pipeline = SequentialPipeline(name="带事实核查的流水线")
pipeline.add_agent(name="ResearchAgent", ...)
pipeline.add_agent(name="FactChecker", ...)  # ← 插入验证
pipeline.add_agent(name="AnalysisAgent", ...)  # 只有在验证通过后才进入下游
```

### 2. 置信度标注与传播

要求每个Agent标注每个数据点的置信度，下游Agent必须根据置信度调整推理：

```python
@dataclass
class VerifiedClaim:
    """带置信度标注的数据声明。"""
    statement: str
    confidence: float  # 0-1
    source: str        # 数据来源
    verification_status: str  # "verified" | "unverified" | "disputed"

class ConfidenceAwareAgent:
    def process(self, upstream_output: list[VerifiedClaim]) -> str:
        # 只对高置信度的数据做高确定性推理
        high_conf = [c for c in upstream_output if c.confidence > 0.85]
        medium_conf = [c for c in upstream_output if 0.6 < c.confidence <= 0.85]
        low_conf = [c for c in upstream_output if c.confidence <= 0.6]

        response = "基于高置信度数据，结论是：..."
        if low_conf:
            response += "\n\n请注意：以下数据置信度较低，相关结论仅供参考：..."
        return response
```

### 3. 端到端幻觉率监控

```python
class HallucinationMonitor:
    """持续监控多Agent流水线的幻觉率。"""

    def __init__(self):
        self.total_claims = 0
        self.verified_claims = 0
        self.hallucinated_claims = 0

    def record(self, claim: VerifiedClaim):
        self.total_claims += 1
        if claim.verification_status == "verified":
            self.verified_claims += 1
        elif claim.verification_status == "disputed":
            self.hallucinated_claims += 1

    @property
    def hallucination_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.hallucinated_claims / self.total_claims

    def alert_if_needed(self, threshold: float = 0.05):
        if self.hallucination_rate > threshold:
            # 触发告警：通知运维、暂停流水线、切换到保守策略
            print(f"[ALERT] 幻觉率 {self.hallucination_rate:.1%} 超过阈值 {threshold:.1%}")
            return True
        return False
```

---

## 检查清单

- [ ] 在关键Agent之间插入了事实验证节点
- [ ] 每个数据声明都标注了置信度和来源
- [ ] 下游Agent根据置信度调整推理的确定性
- [ ] 建立了端到端的幻觉率监控
- [ ] 定期回溯：抽查最终输出的原始数据来源是否真实
- [ ] 对于高风险领域（医疗、法律、金融），所有Agent输出都需可追溯到源文档

---

**一句话总结**：在多Agent系统中，幻觉不是加法问题（每个Agent各自的幻觉概率相加），而是乘法问题（每个Agent基于上游幻觉继续推理）。必须建立"质疑链条"而非"信任链条"。
