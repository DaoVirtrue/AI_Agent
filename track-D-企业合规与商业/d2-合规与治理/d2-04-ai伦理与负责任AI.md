# D2-04：AI 伦理与负责任 AI

> **核心问题**：AI 系统可能产生幻觉、偏见、隐私泄露、环境影响。如何负起责任？
> **核心理念**：负责任 AI = 透明度 + 公平性 + 安全 + 问责 + 可持续性

---

## 一、幻觉风险评估

### 1.1 幻觉严重度分级

```
RAG 幻觉严重度分级：

Level 0: 无幻觉
  回答完全基于检索到的文档，引用准确。
  示例: "根据《员工手册》第3.2条，年假为15天。"

Level 1: 轻微不准确
  主信息正确，次要细节有偏差。
  示例: 把"15天年假"说成"约2周年假"。(可接受)

Level 2: 中度误导
  信息部分正确但关键细节错误。
  示例: 把"提前30天申请"说成"提前15天申请"。(可能造成误事)

Level 3: 严重事实错误
  回答包含明显的事实性错误，可能导致用户做出错误决策。
  示例: 把"危险化学品应远离火源"说成"可以正常加热"。(危险)

Level 4: 完全虚构
  检索失败后 LLM 凭"记忆"编造回答，完全不可信。
  示例: 编造不存在的政策、产品、人物。(严重)
```

### 1.2 不同行业的红线

```yaml
医疗领域（绝不容忍幻觉）:
  红线内:
    ❌ 任何形式的药物剂量建议
    ❌ 诊断意见（即使是"可能"）
    ❌ 治疗效果保证
  安全区:
    ✅ 医学文献检索和摘要
    ✅ 药物说明书内容复述（精确引用）
    ✅ 通用健康知识科普（标注"仅供参考"）
  缓解措施:
    - 所有回答必须精确引用来源（到章节/页码）
    - 置信度 < 0.95 自动转人工
    - 强制添加免责声明
    - 定期用标准医学考试题目验证

法律领域（零容忍）:
  红线内:
    ❌ 法律建议（"你应该..."）
    ❌ 案件结果预测
    ❌ 法条解读（LLM 不是律师）
  安全区:
    ✅ 案例检索（"找到相关的判例"）
    ✅ 法条定位（"相关条款在XX法第X条"）
    ✅ 文书模板生成
  缓解措施:
    - 只检索不解读（检索结果原样呈现）
    - 法条引用必须精确到条款号
    - 强制标注"非法律建议，请咨询律师"

金融领域（高敏感）:
  红线内:
    ❌ 投资建议（"建议买入/卖出"）
    ❌ 市场预测（"明天会涨"）
    ❌ 信用评估建议
  安全区:
    ✅ 合规文档检索
    ✅ 研究报告摘要
    ✅ 公开市场数据的整理
  缓解措施:
    - 所有回答添加"不构成投资建议"
    - 敏感操作需要二次确认
    - 完整的审计追踪

客服领域（中等容忍）:
  红线内:
    ❌ 承诺退款金额和时间
    ❌ 擅自修改用户协议
    ❌ 泄露其他客户信息
  安全区:
    ✅ 标准政策问答
    ✅ 工单创建和状态查询
    ✅ 常见问题解答
```

### 1.3 幻觉检测与缓解

```python
class HallucinationGuard:
    """
    幻觉防护层：在 LLM 回答返回用户前进行幻觉检测。
    """

    def __init__(self, llm):
        self.llm = llm

    def check_faithfulness(self, answer: str, sources: list) -> dict:
        """
        检查回答是否忠实于来源文档。

        方法：用 LLM 逐句检测回答中的每个陈述是否有来源支撑。
        """
        # 将回答拆分为独立陈述
        statements = self._split_into_statements(answer)

        results = []
        for stmt in statements:
            # 对每个陈述，检查是否有来源支撑
            check_prompt = f"""
            来源文档：
            {chr(10).join(s.get('content', '') for s in sources)}

            请判断以下陈述是否完全基于上述来源文档：
            陈述: {stmt}

            回答格式：
            - 如果陈述完全基于来源 → VERIFIED
            - 如果部分基于来源但有推断 → PARTIAL
            - 如果完全无法从来源中找到 → HALLUCINATION
            """
            # result = self.llm.complete(check_prompt)
            results.append({
                "statement": stmt,
                "verdict": "VERIFIED",  # 简化
            })

        hallucinated = [r for r in results if r["verdict"] == "HALLUCINATION"]
        return {
            "total_statements": len(results),
            "hallucinated_count": len(hallucinated),
            "hallucination_rate": len(hallucinated) / len(results) if results else 0,
            "details": results,
        }

    def _split_into_statements(self, text: str) -> list:
        """将文本拆分为独立陈述（按句号/分号）。"""
        import re
        return re.split(r'[。；\n]', text)


# 幻觉防护策略总览
HALLUCINATION_MITIGATION = """
多层级幻觉防护：

第1层：检索质量保证
  - 相似度阈值过滤（score < 0.5 不进入上下文）
  - 重排序精筛
  - 检索结果多样性检查

第2层：Prompt 工程
  - 明确要求"只基于提供的文档回答"
  - 要求"如果文档中没有信息，明确说明"
  - 要求"引用具体来源"

第3层：生成后检查
  - HallucinationGuard 逐句验证
  - 事实一致性评分
  - 引用准确性验证

第4层：运行时监控
  - 实时检测异常模式（过长回答、过于自信的措辞）
  - 自动标记高风险回答
  - 触发人工审核
"""
```

---

## 二、透明度义务

### 2.1 透明度层次

```
透明度阶梯：

Level 1: 基础告知
  ✅ "你正在与 AI 助手对话"
  ✅ "AI 可能产生不准确的信息"

Level 2: 来源透明
  ✅ 每个回答标注信息来源
  ✅ 显示检索到的文档列表
  ✅ 用户可以点击查看原文

Level 3: 过程透明
  ✅ 解释 AI 是如何得出这个答案的（推理链）
  ✅ 显示置信度评分
  ✅ 标注哪些是确定的信息，哪些是推断

Level 4: 完全透明
  ✅ 用户可以查看完整的处理日志
  ✅ 开源模型架构和训练数据（如适用）
  ✅ 独立的第三方审计报告
```

### 2.2 透明度实现

```python
class TransparencyLayer:
    """
    为 AI 回答添加透明度标注。

    在返回给用户前，自动添加：
    - AI 身份标识
    - 来源引用
    - 置信度标注
    - 风险提示
    """

    def annotate_answer(self, answer: str, sources: list,
                        confidence: float, domain: str) -> str:
        """为回答添加透明度标注。"""

        # 1. 添加来源引用
        source_text = "\n\n📚 **信息来源**：\n"
        for i, s in enumerate(sources[:5], 1):
            source_text += f"{i}. [{s['title']}] "
            source_text += f"(相关度: {s['score']:.0%})\n"

        # 2. 添加置信度
        if confidence >= 0.9:
            conf_text = "\n🟢 **置信度：高** — 信息可靠"
        elif confidence >= 0.7:
            conf_text = "\n🟡 **置信度：中** — 建议核实"
        else:
            conf_text = "\n🔴 **置信度：低** — 仅供参考，请务必核实"

        # 3. 添加域特定风险提示
        domain_warnings = {
            "medical": "\n⚠️ **免责声明**：本回答不构成医疗建议。"
                       "如有健康问题，请咨询专业医生。",
            "legal": "\n⚠️ **免责声明**：本回答不构成法律建议。"
                     "如需法律帮助，请咨询执业律师。",
            "financial": "\n⚠️ **免责声明**：本回答不构成投资建议。"
                         "投资有风险，决策需谨慎。",
        }
        warning = domain_warnings.get(domain, "")

        # 4. 添加 AI 标识
        ai_label = "\n🤖 *本回答由 AI 生成*"

        return answer + source_text + conf_text + warning + ai_label
```

---

## 三、环境碳核算

### 3.1 AI 系统的碳足迹

```yaml
RAG 系统碳足迹组成：

1. LLM 推理:
   - 每次查询的 GPU 能耗
   - GPT-4o 约 0.3 Wh/查询
   - DeepSeek-V3 约 0.1 Wh/查询
   - 5000次/天 × 365天 × 0.3Wh = 547 kWh/年

2. 嵌入生成:
   - 每次文档嵌入的能耗
   - BGE-M3 约 0.001 Wh/文档
   - 10万文档全量嵌入 ≈ 1 kWh

3. 基础设施:
   - 服务器持续运行能耗
   - 2台 GPU 服务器 × 500W × 8760h = 8760 kWh/年
   - 存储、网络设备 ≈ 2000 kWh/年

4. 数据中心PUE:
   - 总能耗 × PUE (假设 1.3)
   - 约 15000 kWh/年

碳转换:
  中国电网平均碳排放因子: ~0.57 kg CO2/kWh
  年碳排放: 15000 × 0.57 = 8.55 吨 CO2

碳抵消成本:
  碳价: ~60 CNY/吨 (中国碳市场)
  年抵消成本: ~513 CNY
```

### 3.2 绿色 AI 实践

```python
class CarbonTracker:
    """
    AI 系统碳排放追踪器。

    估算每次 API 调用的碳足迹，并在达到阈值时告警。
    """

    # 各模型估计能耗（Wh/查询）
    MODEL_ENERGY = {
        "gpt-4o": 0.30,
        "gpt-4o-mini": 0.05,
        "claude-sonnet": 0.25,
        "claude-haiku": 0.04,
        "gemini-pro": 0.20,
        "deepseek-v3": 0.10,
        "qwen-max": 0.12,
    }

    CARBON_FACTOR = 0.57  # kg CO2 per kWh (中国平均)

    def __init__(self, monthly_carbon_budget_kg: float = 100.0):
        self.budget = monthly_carbon_budget_kg
        self.current_monthly_kg = 0.0
        self.query_count = 0

    def track_query(self, model: str) -> float:
        """记录一次查询并返回碳排放量（克 CO2）。"""
        energy_wh = self.MODEL_ENERGY.get(model, 0.15)
        carbon_g = energy_wh * self.CARBON_FACTOR / 1000
        self.current_monthly_kg += carbon_g / 1000
        self.query_count += 1
        return carbon_g

    def get_report(self) -> dict:
        """获取碳排放报告。"""
        return {
            "total_queries": self.query_count,
            "monthly_carbon_kg": round(self.current_monthly_kg, 2),
            "budget_remaining_pct": round(
                max(0, (1 - self.current_monthly_kg / self.budget)) * 100, 1
            ),
            "equivalent_trees": round(
                self.current_monthly_kg / 21.0, 1  # 一棵树年吸收21kg CO2
            ),
        }

    def should_alert(self) -> bool:
        """检查是否超过碳预算的80%。"""
        return self.current_monthly_kg > self.budget * 0.8
```

---

## 四、负责任 AI 原则总结

```
负责任 AI 五大原则：

1. 透明度 (Transparency)
   - 告知用户 AI 的存在
   - 标注信息来源
   - 公开模型能力和局限

2. 公平性 (Fairness)
   - 消除偏见
   - 平等对待所有用户
   - 定期偏见审计

3. 安全 (Safety)
   - 防止有害输出
   - 幻觉检测和缓解
   - 安全回退机制

4. 问责 (Accountability)
   - 明确的负责人
   - 完整的审计轨迹
   - 用户申诉渠道

5. 可持续性 (Sustainability)
   - 碳足迹追踪
   - 选择高效模型
   - 优化不必要的计算
```

---

## 五、年度 AI 伦理审查清单

```
□ 是否更新了 Model Card？
□ 是否完成了年度偏见测试？结果是什么？
□ 幻觉率是否在可接受范围内？
□ 是否有用户投诉未处理？
□ 是否有安全事故未复盘？
□ 透明度措施是否充分？
□ 碳排放报告是否完成？
□ 是否进行了红队测试？
□ AI 治理委员会是否召开年度会议？
□ 是否有新的法规需要合规？
□ 是否提供了充分的用户申诉渠道？
□ 员工是否接受了 AI 伦理培训？
```
