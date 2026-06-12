#!/usr/bin/env python3
"""
Phase 08 - LangGraph 企业级：05 - Human-in-the-Loop（HITL）中断

interrupt() 函数在敏感节点暂停图执行，等待人工审批/输入。
恢复执行使用 Command(resume=human_input)。

核心概念：
  ┌─────────────────────────────────────────────────┐
  │              Graph 执行流程                       │
  │                                                  │
  │  node_1 → node_2 → [node_sensitive] → node_3     │
  │                        │                         │
  │                   interrupt()                     │
  │                        │                         │
  │                   ═══ 暂停 ═══                    │
  │                        │                         │
  │              ← 人工审批/输入 →                    │
  │                        │                         │
  │              Command(resume=...)                  │
  │                        │                         │
  │                   → node_3 (继续)                 │
  └─────────────────────────────────────────────────┘

使用场景：
  1. 大额交易审批
  2. 敏感内容审查
  3. 高风险操作确认
  4. 人工标注/审核节点
  5. 需要外部系统提供数据的节点
"""

import time
import json
from typing import TypedDict, Annotated, Literal, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# ============================================================================
# State 定义
# ============================================================================

class ApprovalState(TypedDict):
    """
    审批流程的 State。

    Attributes:
        messages: 对话消息
        transaction: 交易详情
        risk_score: 风险评分（0-100）
        approval_status: 审批状态 "pending" | "approved" | "rejected"
        human_decision: 人工决策内容
        audit_log: 审计日志
    """
    messages: Annotated[list, add_messages]
    transaction: dict  # {"amount": 50000, "type": "transfer", "to": "..."}
    risk_score: float  # 0-100
    approval_status: str  # "pending", "approved", "rejected"
    human_decision: str  # 人工输入的决策理由
    audit_log: Annotated[list, add_messages]  # 使用 add_messages 作为 reducer


# ============================================================================
# 节点函数
# ============================================================================

def node_validate_transaction(state: ApprovalState) -> dict:
    """
    验证交易节点：检查交易格式和基本合法性。
    此节点不需要人工干预。
    """
    txn = state.get('transaction', {})
    amount = txn.get('amount', 0)
    txn_type = txn.get('type', 'unknown')

    print(f"  [validate] 交易验证: 金额={amount}, 类型={txn_type}")

    # 基本验证
    errors = []
    if amount <= 0:
        errors.append("金额必须大于0")
    if txn_type not in ('transfer', 'payment', 'withdrawal'):
        errors.append(f"未知交易类型: {txn_type}")

    validation_passed = len(errors) == 0

    return {
        "messages": [
            AIMessage(
                content=f"交易验证{'通过' if validation_passed else '失败'}: "
                        f"{' | '.join(errors) if errors else '所有检查通过'}"
            )
        ],
        "audit_log": [
            AIMessage(content=f"[审计] 交易验证: {'通过' if validation_passed else '失败'}")
        ],
    }


def node_risk_assessment(state: ApprovalState) -> dict:
    """
    风险评估节点：计算交易风险分数。
    此节点不需要人工干预。
    """
    txn = state.get('transaction', {})
    amount = txn.get('amount', 0)

    # 风险评分算法（简化版）
    risk = 0

    # 金额风险
    if amount > 100000:
        risk += 50
    elif amount > 50000:
        risk += 35
    elif amount > 10000:
        risk += 20
    elif amount > 1000:
        risk += 10

    # 类型风险
    txn_type = txn.get('type', '')
    if txn_type == 'withdrawal':
        risk += 15
    elif txn_type == 'transfer':
        risk += 10

    # 时间风险（夜间交易风险更高）
    current_hour = time.localtime().tm_hour
    if current_hour < 6 or current_hour > 22:
        risk += 10

    risk = min(100, risk)
    print(f"  [risk] 风险评估: {risk}/100")

    return {
        "risk_score": risk,
        "messages": [
            AIMessage(
                content=f"风险评估完成: {risk}/100 "
                        f"({'高风险' if risk > 60 else '中等风险' if risk > 30 else '低风险'})"
            )
        ],
        "audit_log": [
            AIMessage(content=f"[审计] 风险评估: {risk}/100 "
                             f"(金额={amount}, 类型={txn_type})")
        ],
    }


def node_sensitive_approval(state: ApprovalState) -> dict:
    """
    敏感审批节点：需要人工审查！

    当风险评分 > 30 时，此节点会调用 interrupt() 暂停执行，
    等待人工审批。审批结果通过 Command(resume=...) 传入。

    在真实场景中，interrupt() 会向审批人员的终端/手机发送通知，
    审批人员批准后，流程继续。
    """
    risk = state.get('risk_score', 0)
    txn = state.get('transaction', {})

    print(f"  [approval] 需要人工审批: 风险={risk}/100")

    if risk > 30:
        # 高风险 → 必须人工审批
        print(f"  [approval] ⏸️ 暂停等待人工审批...")

        # interrupt() 会在这里暂停图执行
        # 返回值是人工审批者传入的信息
        human_review = interrupt({
            "message": "需要人工审批",
            "transaction": txn,
            "risk_score": risk,
            "prompt": "请审批此交易 (输入 'approved' 或 'rejected' 及理由):",
        })

        # human_review 是审批者通过 Command(resume=...) 传入的数据
        decision = human_review.get("decision", "rejected")
        reason = human_review.get("reason", "无理由")

        print(f"  [approval] ▶️ 收到人工决定: {decision} - {reason}")

        return {
            "approval_status": decision,
            "human_decision": f"{decision}: {reason}",
            "messages": [
                AIMessage(content=f"人工审批结果: {decision} (理由: {reason})")
            ],
            "audit_log": [
                AIMessage(content=f"[审计] 人工审批: {decision} | 风险={risk} | 理由={reason}")
            ],
        }
    else:
        # 低风险 → 自动批准
        print(f"  [approval] 风险较低，自动批准")
        return {
            "approval_status": "approved",
            "human_decision": "auto_approved: low_risk",
            "messages": [
                AIMessage(content=f"自动批准（风险评分 {risk}/100 低于阈值）")
            ],
            "audit_log": [
                AIMessage(content=f"[审计] 自动批准: 风险={risk} < 阈值")
            ],
        }


def node_execute_transaction(state: ApprovalState) -> dict:
    """
    执行交易节点：只有在批准后才执行。
    """
    status = state.get('approval_status', 'pending')

    if status == 'approved':
        txn = state.get('transaction', {})
        print(f"  [execute] ✅ 执行交易: {txn.get('amount')}元")

        return {
            "messages": [
                AIMessage(
                    content=f"✅ 交易已执行: {txn.get('amount')}元 "
                            f"(类型: {txn.get('type')})"
                )
            ],
            "audit_log": [
                AIMessage(content=f"[审计] 交易执行成功: {json.dumps(txn, ensure_ascii=False)}")
            ],
        }
    else:
        print(f"  [execute] ❌ 交易被拒绝: {status}")

        return {
            "messages": [
                AIMessage(content=f"❌ 交易未执行: 审批状态为 '{status}'")
            ],
            "audit_log": [
                AIMessage(content=f"[审计] 交易已拒绝: 状态={status}")
            ],
        }


def node_notify(state: ApprovalState) -> dict:
    """
    通知节点：发送通知给用户。
    """
    status = state.get('approval_status', 'pending')

    if status == 'approved':
        notification = "📱 通知: 您的交易已批准并执行。"
    else:
        notification = "📱 通知: 您的交易已被拒绝。如有疑问请联系客服。"

    print(f"  [notify] {notification}")

    return {
        "messages": [AIMessage(content=notification)],
        "audit_log": [AIMessage(content=f"[审计] 通知已发送: {status}")],
    }


# ============================================================================
# 路由函数
# ============================================================================

def route_after_approval(state: ApprovalState) -> Literal["execute", "notify"]:
    """根据审批结果路由。"""
    status = state.get('approval_status', 'pending')
    if status == 'approved':
        return "execute"
    return "notify"


# ============================================================================
# 构建图
# ============================================================================

def build_approval_graph() -> StateGraph:
    """
    构建带 HITL 的审批图。

    流程:
      validate → risk_assess → sensitive_approval [interrupt!]
        → approved → execute → notify → END
        → rejected → notify → END
    """
    builder = StateGraph(ApprovalState)

    builder.add_node("validate", node_validate_transaction)
    builder.add_node("risk_assess", node_risk_assessment)
    builder.add_node("approve", node_sensitive_approval)  # ← interrupt() 在这里
    builder.add_node("execute", node_execute_transaction)
    builder.add_node("notify", node_notify)

    builder.set_entry_point("validate")
    builder.add_edge("validate", "risk_assess")
    builder.add_edge("risk_assess", "approve")

    builder.add_conditional_edges(
        "approve",
        route_after_approval,
        {
            "execute": "execute",
            "notify": "notify",
        }
    )

    builder.add_edge("execute", "notify")
    builder.add_edge("notify", END)

    return builder


# ============================================================================
# 前端集成模式
# ============================================================================

class ApprovalAPI:
    """
    前端集成模式：API 端点包装器。

    REST API 工作流:
    1. POST /api/transactions       → 创建交易，返回交易ID
    2. POST /api/transactions/{id}/submit → 提交到图执行
       - 如果返回 status=202 → 需要审批（前端显示审批界面）
       - 如果返回 status=200 → 自动批准
    3. POST /api/transactions/{id}/approve → 提交审批决定
       - 后端调用 graph.invoke(Command(resume=...), config)
    4. GET /api/transactions/{id}  → 查询最终结果
    """

    def __init__(self, graph, checkpointer):
        self.graph = graph
        self.checkpointer = checkpointer
        self._pending_approvals: dict[str, dict] = {}
        self._thread_results: dict[str, dict] = {}

    def submit_transaction(self, transaction: dict) -> dict:
        """
        提交交易到图执行。
        如果遇到 interrupt()，返回 202 和审批信息。
        """
        import uuid

        thread_id = f"txn_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages": [],
            "transaction": transaction,
            "risk_score": 0,
            "approval_status": "pending",
            "human_decision": "",
            "audit_log": [],
        }

        try:
            # 尝试执行到 completion
            result = self.graph.invoke(initial_state, config)
            self._thread_results[thread_id] = result

            return {
                "status": "completed",
                "transaction_id": thread_id,
                "approval_status": result.get("approval_status"),
                "message": "交易已自动处理（无需人工审批）",
            }

        except Exception as e:
            # 图在 interrupt() 处暂停
            # 在真实 LangGraph 中，interrupt 会抛出 GraphInterrupt
            # 这里我们检测到 interrupt 后返回 202
            if "interrupt" in str(e).lower() or hasattr(e, '__cause__'):
                # 保存等待审批的状态
                self._pending_approvals[thread_id] = {
                    "transaction": transaction,
                    "config": config,
                    "status": "awaiting_approval",
                }

                return {
                    "status": "awaiting_approval",
                    "transaction_id": thread_id,
                    "message": "需要人工审批",
                    "prompt": f"交易金额 {transaction.get('amount')}元，请审批",
                }

            raise  # 其他错误向上传播

    def approve_transaction(self, transaction_id: str, decision: str, reason: str) -> dict:
        """
        提交审批决定，恢复图执行。

        Args:
            transaction_id: 交易ID
            decision: "approved" 或 "rejected"
            reason: 审批理由
        """
        pending = self._pending_approvals.get(transaction_id)
        if not pending:
            raise ValueError(f"未找到待审批的交易: {transaction_id}")

        config = pending["config"]

        # 恢复执行
        resume_command = Command(resume={"decision": decision, "reason": reason})
        result = self.graph.invoke(resume_command, config)

        # 清理
        del self._pending_approvals[transaction_id]
        self._thread_results[transaction_id] = result

        return {
            "status": "completed",
            "transaction_id": transaction_id,
            "approval_status": result.get("approval_status"),
            "decision": decision,
        }

    def get_transaction_status(self, transaction_id: str) -> dict:
        """查询交易状态。"""
        if transaction_id in self._pending_approvals:
            return {"status": "awaiting_approval", "transaction_id": transaction_id}

        result = self._thread_results.get(transaction_id)
        if result:
            return {
                "status": "completed",
                "transaction_id": transaction_id,
                "approval_status": result.get("approval_status"),
                "audit_log": [
                    m.content for m in result.get("audit_log", [])
                ],
            }

        raise ValueError(f"未找到交易: {transaction_id}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  LangGraph 企业级 - Human-in-the-Loop 中断演示")
    print("=" * 72)

    # ---- Demo 1: 低风险自动审批 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 低风险交易自动审批（无 interrupt）")
    print("=" * 72)

    # 构建图
    builder = build_approval_graph()
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    result = graph.invoke({
        "messages": [],
        "transaction": {"amount": 500, "type": "payment", "to": "account_123"},
        "risk_score": 0,
        "approval_status": "pending",
        "human_decision": "",
        "audit_log": [],
    })

    print(f"\n  📊 结果:")
    print(f"  审批状态: {result.get('approval_status')}")
    print(f"  审计日志: {len(result.get('audit_log', []))} 条")

    # ---- Demo 2: 高风险需要人工审批 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 高风险交易需要人工审批")
    print("=" * 72)

    config = {"configurable": {"thread_id": "high_risk_demo"}}

    # 第一次 invoke → 会在 interrupt() 处暂停
    print("\n  📤 提交高风险交易...")
    print("  交易: 金额=50000元, 类型=transfer, 夜间交易")

    try:
        # 在真实场景中，interrupt() 会抛出异常
        # 这里演示概念性流程
        result = graph.invoke({
            "messages": [],
            "transaction": {"amount": 50000, "type": "transfer", "to": "account_456"},
            "risk_score": 0,
            "approval_status": "pending",
            "human_decision": "",
            "audit_log": [],
        }, config)
    except Exception as e:
        print(f"\n  ⏸️ 图在 interrupt() 处暂停")
        print(f"  前端应展示审批界面，等待人工输入审批决定")

    # ---- Demo 3: API 模式演示 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 前端集成 API 模式")
    print("=" * 72)

    api = ApprovalAPI(graph, None)

    # 模拟前端调用流程
    print("\n  1. 前端 POST /api/transactions")
    response1 = api.submit_transaction({"amount": 100000, "type": "withdrawal"})
    print(f"     响应: {json.dumps(response1, ensure_ascii=False, indent=2)}")

    print("\n  2. 后端检测到需要审批 → 前端展示审批界面")

    print("\n  3. 审批者点击 '批准' → 前端 POST /api/transactions/{id}/approve")
    if response1.get("status") == "awaiting_approval":
        txn_id = response1["transaction_id"]
        response2 = api.approve_transaction(txn_id, "approved", "经核实，交易合规")
        print(f"     响应: {json.dumps(response2, ensure_ascii=False, indent=2)}")

    print("\n  4. 前端 GET /api/transactions/{id}")
    if response1.get("status") == "awaiting_approval":
        txn_id = response1["transaction_id"]
        response3 = api.get_transaction_status(txn_id)
        print(f"     响应: {json.dumps(response3, ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 72)
    print("  Human-in-the-Loop 中断演示完成！")
    print("=" * 72)
