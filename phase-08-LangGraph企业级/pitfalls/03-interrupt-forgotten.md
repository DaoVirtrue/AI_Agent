# 坑点 #3：interrupt() 被遗忘

> "审批流程跑通了，测试也过了。但上线后发现'大额转账需要审批'这个规则从来没有被触发过。检查代码发现：风险评分节点计算正确（85分），但approval节点里忘记调用interrupt()了。42笔大额转账全部自动通过，没有一笔经过人工审批。"

---

## 症状

1. **HITL形同虚设**：定义了审批节点，但interrupt()从未被调用
2. **审批条件判断错误**：风险阈值 >= 80 写成了 > 80（边界问题）
3. **interrupt()被条件跳过**：在if分支中，但条件永远不满足
4. **Command(resume=...)格式错误**：恢复时传入的数据格式与interrupt期望不一致

---

## 根因

`interrupt()` 的调用是**手动的**——你需要显式地在节点函数中调用它。很容易被以下情况跳过：
- 条件分支覆盖不全
- 阈值比较用错符号（> vs >=）
- 异常处理过早捕获了本应触发interrupt的路径
- 重构时不小心删除了interrupt()调用

---

## 解法

### 1. interrupt() 强制调用模式

```python
from langgraph.types import interrupt

def node_approval_with_guarantee(state: ApprovalState) -> dict:
    """
    确保 interrupt() 必定被调用的审批节点。
    使用'先中断、再判断'的模式，而不是'判断后可选中断'。
    """
    risk_score = state.get('risk_score', 0)

    # 总是先调用 interrupt()（即使风险低）
    # interrupt 内部的逻辑可以决定是否真正需要人工审批
    human_input = interrupt({
        "type": "approval_required",
        "risk_score": risk_score,
        "transaction": state.get('transaction', {}),
        "auto_approve": risk_score < 30,  # 低风险自动批准标志
    })

    # 人工输入决定最终结果
    return {
        "approval_status": human_input.get("decision", "auto_approved"),
        "approval_reason": human_input.get("reason", "low_risk_auto"),
        "audit_log": state.get('audit_log', []) + [
            f"[审计] 审批: {human_input.get('decision')} "
            f"(风险={risk_score})"
        ],
    }
```

### 2. HITL覆盖测试

```python
import pytest

class TestHITLCoverage:
    """验证所有需要interrupt的路径都已覆盖。"""

    @pytest.mark.parametrize("risk_score,should_interrupt", [
        (0, True),      # 零风险也应该触发interrupt（以审计为目的）
        (29, True),     # 边界值
        (30, True),     # 阈值边界
        (80, True),     # 高风险
        (100, True),    # 最高风险
    ])
    def test_interrupt_always_called(self, risk_score, should_interrupt):
        """验证在所有风险级别下，interrupt() 都会被调用。"""
        graph = build_approval_graph()
        state = {
            "risk_score": risk_score,
            "transaction": {"amount": 1000},
        }
        try:
            graph.invoke(state)
            assert not should_interrupt, f"风险{risk_score}时应该触发interrupt！"
        except GraphInterrupt as e:
            assert should_interrupt, f"风险{risk_score}时不应该触发interrupt！"
            # 验证interrupt传入了正确的数据
            interrupt_data = e.args[0] if e.args else {}
            assert "risk_score" in interrupt_data
```

### 3. 前端集成模板

```python
class HITLResponseHandler:
    """
    前端如何处理 interrupt 的标准模板。
    """

    @staticmethod
    def handle_graph_interrupt(exception, thread_id: str) -> dict:
        """将 GraphInterrupt 转为前端可用的格式。"""
        interrupt_data = exception.args[0] if exception.args else {}

        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "approval_prompt": {
                "title": "需要人工审批",
                "details": interrupt_data.get("transaction", {}),
                "risk_score": interrupt_data.get("risk_score", 0),
                "auto_approve": interrupt_data.get("auto_approve", False),
                "actions": ["approve", "reject"],
            },
            "resume_endpoint": f"/api/approve/{thread_id}",
        }

    @staticmethod
    def build_resume_command(decision: str, reason: str):
        """构建恢复执行的Command对象。"""
        from langgraph.types import Command
        return Command(resume={"decision": decision, "reason": reason})
```

---

## 检查清单

- [ ] 审批节点中的`interrupt()`在所有代码路径中都能到达
- [ ] 阈值比较使用了正确的运算符（>=而非>，或相反取决于需求）
- [ ] 编写了HITL覆盖测试（包括边界值）
- [ ] 前端有明确的审批界面处理`awaiting_approval`状态
- [ ] 恢复执行时`Command(resume=...)`的数据格式与`interrupt()`中定义的一致
- [ ] 审计日志记录了每次interrupt和恢复

---

**一句话总结**：interrupt() 不是自动的——你需要显式调用它。如果你不确定它在所有路径上都会被调用，那就让它成为节点的第一行代码。
