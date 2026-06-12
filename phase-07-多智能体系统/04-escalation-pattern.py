#!/usr/bin/env python3
"""
Phase 07 - 多智能体系统：04 - 升级模式（Escalation Pattern）

三级升级架构：
  L1 (Tier 1): 简单问题 → 无工具 → 低成本快速响应
  L2 (Tier 2): 复杂问题 → 完整Agent → 工具调用 + 推理
  L3 (Tier 3): 需人工介入 → 加入等待队列 → 人工处理

升级条件（Escalation Criteria）：
  1. 置信度 < 阈值（默认 < 0.70）
  2. 用户明确要求人工服务
  3. 检测到敏感话题（法律、医疗、金融建议）
  4. 重试次数超限（同一问题尝试 > max_retries）

架构图：
   用户请求
      │
      ▼
  ┌─────────────────┐
  │  L1: SimpleAgent │  无工具，纯LLM回答
  │  (gpt-4o-mini)  │  置信度检查 → 低于阈值 → L2
  └────────┬────────┘
           │ 升级条件触发
           ▼
  ┌─────────────────┐
  │ L2: FullAgent    │  完整Agent + 工具
  │ (gpt-4o)         │  置信度检查 → 仍不足 → L3
  └────────┬────────┘
           │ 升级条件触发
           ▼
  ┌─────────────────┐
  │ L3: HumanQueue   │  加入人工处理队列
  │ (人工客服/专家)   │  通知 + 上下文传递
  └─────────────────┘
"""

import time
import json
import uuid
import enum
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, NamedTuple, Optional


# ============================================================================
# 数据模型
# ============================================================================

class Tier(enum.Enum):
    """处理层级。"""
    L1 = 1   # 简单Agent，无工具
    L2 = 2   # 完整Agent，有工具
    L3 = 3   # 人工队列


class EscalationReason(enum.Enum):
    """升级原因枚举。"""
    LOW_CONFIDENCE = "low_confidence"            # 置信度低于阈值
    HUMAN_REQUEST = "human_request"              # 用户明确要求人工
    SENSITIVE_TOPIC = "sensitive_topic"          # 检测到敏感话题
    MAX_RETRIES = "max_retries"                  # 重试次数超限
    TOOL_ERROR = "tool_error"                    # 工具调用失败
    POLICY_VIOLATION = "policy_violation"        # 违反安全策略


@dataclass
class TierResult:
    """
    某一层的处理结果。

    Attributes:
        tier: 处理层级
        answer: 生成的回答（如果有）
        confidence: 置信度（0-1）
        time_elapsed: 耗时（秒）
        escalation_reason: 如果需要升级，升级原因
        should_escalate: 是否应该升级
        metadata: 附加信息（工具调用记录等）
    """
    tier: Tier
    answer: str = ""
    confidence: float = 0.0
    time_elapsed: float = 0.0
    escalation_reason: Optional[EscalationReason] = None
    should_escalate: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class HumanTicket:
    """
    人工处理工单。

    Attributes:
        ticket_id: 唯一工单ID
        user_query: 用户原始问题
        context: 上下文信息（前面层级的处理结果）
        tier_path: 升级路径（L1→L2→L3）
        priority: 优先级（1=最高, 5=最低）
        created_at: 创建时间
        status: 状态
        assigned_to: 指派给谁
        resolution: 解决方案（人工填写）
    """
    ticket_id: str
    user_query: str
    context: dict
    tier_path: list[Tier]
    priority: int = 3
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending | in_progress | resolved | closed
    assigned_to: str = ""
    resolution: str = ""


# ============================================================================
# 升级策略配置
# ============================================================================

@dataclass
class EscalationConfig:
    """
    升级策略配置。

    Attributes:
        confidence_threshold: 置信度阈值（低于此值触发升级）
        max_retries: 同一层级最大重试次数
        sensitive_keywords: 敏感话题关键词列表
        require_human_tiers: 哪些层级需要人工审批
        tier_timeout_seconds: 每层的超时时间
    """
    confidence_threshold: float = 0.70
    max_retries: int = 2
    sensitive_keywords: list[str] = field(default_factory=lambda: [
        # 法律相关
        "起诉", "诉讼", "律师", "法律建议", "赔偿", "合同纠纷",
        # 医疗相关
        "诊断", "处方", "药物", "治疗建议", "手术", "病情",
        # 金融相关
        "投资建议", "股票推荐", "贷款", "担保", "保险理赔",
        # 安全相关
        "自杀", "自残", "暴力", "恐怖",
    ])
    require_human_tiers: list[Tier] = field(default_factory=lambda: [Tier.L3])
    tier_timeout_seconds: dict = field(default_factory=lambda: {
        Tier.L1: 15,
        Tier.L2: 60,
        Tier.L3: 3600,  # 1小时等待人工
    })


# ============================================================================
# Agent 层级实现
# ============================================================================

class TierAgent(ABC):
    """Agent层级基类。"""

    def __init__(self, tier: Tier, model: str = "gpt-4o-mini"):
        self.tier = tier
        self.model = model

    @abstractmethod
    def process(self, query: str, context: dict = None) -> TierResult:
        """
        处理用户请求。

        Args:
            query: 用户问题
            context: 来自前一层级的上下文

        Returns:
            处理结果（含是否升级的判断）
        """
        ...

    @abstractmethod
    def evaluate_confidence(self, answer: str, query: str) -> float:
        """
        评估回答的置信度。

        Args:
            answer: 生成的回答
            query: 原始问题

        Returns:
            置信度（0-1）
        """
        ...

    @abstractmethod
    def should_escalate(
        self,
        query: str,
        result: TierResult,
        config: EscalationConfig,
        retry_count: int,
    ) -> Optional[EscalationReason]:
        """
        判断是否需要升级。

        Returns:
            升级原因，如果不需要升级则返回None
        """
        ...


class L1SimpleAgent(TierAgent):
    """
    L1 简易Agent：无工具访问，快速回答简单问题。
    使用 gpt-4o-mini，低成本高速度。
    """

    def __init__(self):
        super().__init__(tier=Tier.L1, model="gpt-4o-mini")

    def process(self, query: str, context: dict = None) -> TierResult:
        """使用基础LLM快速回答。"""
        start = time.time()

        # 模拟LLM调用
        answer = self._simulate_l1_response(query)
        confidence = self.evaluate_confidence(answer, query)
        time_elapsed = time.time() - start

        return TierResult(
            tier=self.tier,
            answer=answer,
            confidence=confidence,
            time_elapsed=time_elapsed,
            metadata={"model": self.model, "tokens_estimated": len(answer) // 3},
        )

    def _simulate_l1_response(self, query: str) -> str:
        """模拟L1回答（纯LLM，无工具）。"""
        # 对于已知简单问题给出明确答案
        if "你好" in query or "hello" in query.lower():
            return "您好！我是AI助手，很高兴为您服务。请问有什么可以帮助您的？"
        elif "天气" in query:
            return "抱歉，我目前无法获取实时天气数据。建议您查看天气预报应用获取准确信息。"
        elif "价格" in query or "多少钱" in query:
            return "关于具体价格，我需要确认您指的是哪个产品或服务。您能提供更多细节吗？"
        elif "退款" in query:
            return "关于退款，我们通常支持7天内无理由退款。具体政策取决于您的订单类型。如果您需要，我可以帮您转接人工客服处理退款事宜。"
        elif "投诉" in query:
            return "非常抱歉给您带来不好的体验。我理解您的不满。为了更好地解决您的问题，我建议将您转接给高级客服专员处理。"
        else:
            # 通用回答
            return (
                f"关于'{query[:80]}'的问题，以下是我能提供的信息：\n\n"
                f"根据我的知识库，这个问题可以从以下角度分析：\n"
                f"1. 首先，需要明确具体的需求场景\n"
                f"2. 其次，建议查阅最新的官方文档获取准确信息\n"
                f"3. 如果涉及专业领域（法律、医疗、金融），建议咨询专业人士\n\n"
                f"如果您需要更详细的分析，我可以为您转接高级AI助手或人工专家。"
            )

    def evaluate_confidence(self, answer: str, query: str) -> float:
        """L1置信度评估：基于回答质量和问题复杂度。"""
        confidence = 0.85

        # 降分因素
        if len(answer) < 50:
            confidence -= 0.15  # 回答太短
        if "抱歉" in answer or "无法" in answer or "不能" in answer:
            confidence -= 0.20  # 包含了否定/无能为力的表述
        if "转接" in answer or "人工" in answer:
            confidence -= 0.15  # 建议转人工

        # 问题复杂度影响
        query_len = len(query)
        if query_len > 200:
            confidence -= 0.10  # 长问题 → 更复杂
        if "?" in query or "？" in query:
            if query.count("?") + query.count("？") > 2:
                confidence -= 0.10  # 多问句问题

        return max(0.0, min(1.0, confidence))

    def should_escalate(
        self,
        query: str,
        result: TierResult,
        config: EscalationConfig,
        retry_count: int,
    ) -> Optional[EscalationReason]:
        """L1升级判断。"""
        # 1. 置信度不足
        if result.confidence < config.confidence_threshold:
            return EscalationReason.LOW_CONFIDENCE

        # 2. 用户明确要求人工
        human_signals = ["人工", "真人", "客服", "转人工", "human", "person", "agent"]
        if any(signal in query.lower() for signal in human_signals):
            return EscalationReason.HUMAN_REQUEST

        # 3. 敏感话题检测
        if any(kw in query for kw in config.sensitive_keywords):
            return EscalationReason.SENSITIVE_TOPIC

        # 4. 重试超限
        if retry_count >= config.max_retries:
            return EscalationReason.MAX_RETRIES

        return None


class L2FullAgent(TierAgent):
    """
    L2 完整Agent：拥有工具访问、推理能力。
    使用 gpt-4o，能处理更复杂的问题。
    """

    def __init__(self):
        super().__init__(tier=Tier.L2, model="gpt-4o")
        # L2 Agent拥有工具列表
        self.available_tools = [
            "knowledge_search",    # 知识库搜索
            "data_analysis",       # 数据分析
            "document_query",      # 文档查询
            "calculator",          # 计算器
            "web_search",          # 网络搜索
        ]

    def process(self, query: str, context: dict = None) -> TierResult:
        """使用完整Agent能力回答问题。"""
        start = time.time()

        # 获取L1的上下文（如果有）
        l1_answer = context.get("l1_answer", "") if context else ""

        # 模拟工具使用和推理
        tools_used = self._decide_tools(query)
        answer = self._simulate_l2_response(query, l1_answer, tools_used)
        confidence = self.evaluate_confidence(answer, query)
        time_elapsed = time.time() - start

        return TierResult(
            tier=self.tier,
            answer=answer,
            confidence=confidence,
            time_elapsed=time_elapsed,
            metadata={
                "model": self.model,
                "tools_used": tools_used,
                "tools_count": len(tools_used),
                "reasoning_steps": self._count_reasoning_steps(answer),
            },
        )

    def _decide_tools(self, query: str) -> list[str]:
        """根据问题决定使用哪些工具。"""
        tools = []
        query_lower = query.lower()

        if any(w in query_lower for w in ["数据", "统计", "分析", "data", "analytics"]):
            tools.append("data_analysis")
        if any(w in query_lower for w in ["搜索", "查找", "查询", "search", "find"]):
            tools.append("knowledge_search")
        if any(w in query_lower for w in ["计算", "算", "calculate"]):
            tools.append("calculator")
        if any(w in query_lower for w in ["最新", "实时", "新闻", "current", "latest"]):
            tools.append("web_search")

        # 至少使用一个工具
        if not tools:
            tools.append("knowledge_search")

        return tools

    def _simulate_l2_response(
        self,
        query: str,
        l1_answer: str,
        tools_used: list[str],
    ) -> str:
        """模拟L2完整Agent的回答。"""
        return (
            f"## 综合分析（L2 Agent · {self.model}）\n\n"
            f"### 原始问题\n{query}\n\n"
            f"### 处理流程\n"
            f"1. **问题分析**：识别到核心需求 → 需要 {(len(tools_used))} 个工具协同\n"
            f"2. **工具调用**：{', '.join(tools_used)}（全部成功）\n"
            f"3. **信息整合**：综合多源信息\n"
            f"4. **答案生成**：基于证据的推理\n\n"
            f"### 详细回答\n"
            f"针对您的问题，经过深入分析后，以下是详细回答：\n\n"
            f"根据知识库中的{10 + len(query) % 20}条相关文档和数据分析工具的交叉验证，"
            f"我得出以下结论：\n\n"
            f"> 核心发现：该问题的关键因素包括信息完整性、时效性和上下文相关性。"
            f"通过综合分析，得出置信度较高的答案。\n\n"
            f"### 数据来源\n"
            f"- 内部知识库：检索到相关文档\n"
            f"- 数据分析：已验证数据一致性\n"
            f"{'- 实时搜索：已核实最新信息' if 'web_search' in tools_used else ''}\n\n"
            f"### 置信度评估\n此回答基于多重证据源的信息整合，具有较高级别的可靠性。\n"
            f"*生成时间：{datetime.now().isoformat()}*\n"
        )

    def evaluate_confidence(self, answer: str, query: str) -> float:
        """L2置信度评估：考虑工具使用和推理深度。"""
        confidence = 0.88  # L2基准更高

        # 加分因素
        if "工具调用" in answer or "tools_used" in answer:
            confidence += 0.05
        if "数据来源" in answer or "引用" in answer:
            confidence += 0.05

        # 降分因素
        if "不确定" in answer or "可能" in answer or "推测" in answer:
            confidence -= 0.10
        if len(query) > 500:
            confidence -= 0.05  # 超长问题，可能有歧义

        return max(0.0, min(1.0, confidence))

    def _count_reasoning_steps(self, answer: str) -> int:
        """估算推理步骤数。"""
        steps = 0
        for marker in ["1.", "2.", "3.", "4.", "5.", "步骤", "Step"]:
            steps += answer.count(marker)
        return max(1, steps)

    def should_escalate(
        self,
        query: str,
        result: TierResult,
        config: EscalationConfig,
        retry_count: int,
    ) -> Optional[EscalationReason]:
        """L2升级判断。"""
        # 1. 置信度仍不足
        if result.confidence < config.confidence_threshold:
            return EscalationReason.LOW_CONFIDENCE

        # 2. 工具调用失败
        if result.metadata.get("tools_count", 0) == 0:
            return EscalationReason.TOOL_ERROR

        # 3. 敏感话题
        if any(kw in query for kw in config.sensitive_keywords):
            return EscalationReason.SENSITIVE_TOPIC

        # 4. 重试超限
        if retry_count >= config.max_retries:
            return EscalationReason.MAX_RETRIES

        return None


class L3HumanQueue(TierAgent):
    """
    L3 人工队列：需要人工介入的请求。
    生成工单，加入队列，通知人工客服。
    """

    def __init__(self, queue_size: int = 100):
        super().__init__(tier=Tier.L3, model="human")
        self.ticket_queue: deque[HumanTicket] = deque(maxlen=queue_size)
        self._handlers: list[Callable] = []

    def process(self, query: str, context: dict = None) -> TierResult:
        """
        创建人工工单并返回等待通知。

        Args:
            query: 用户原始问题
            context: 所有前序层级的信息（L1回答、L2回答、升级原因等）

        Returns:
            处理结果（表示已加入队列等待）
        """
        start = time.time()

        # 生成唯一工单ID
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

        # 构建上下文
        full_context = context or {}
        full_context["escalation_path"] = full_context.get("escalation_path", [])

        # 计算优先级
        priority = self._calculate_priority(query, full_context)

        # 创建工单
        ticket = HumanTicket(
            ticket_id=ticket_id,
            user_query=query,
            context=full_context,
            tier_path=full_context.get("escalation_path", [Tier.L1, Tier.L2, Tier.L3]),
            priority=priority,
        )

        # 加入队列
        self.ticket_queue.append(ticket)

        # 通知处理器
        for handler in self._handlers:
            handler(ticket)

        time_elapsed = time.time() - start

        # 生成用户侧的通知
        wait_time = self._estimate_wait_time()
        answer = (
            f"## 已转接人工处理\n\n"
            f"**工单编号**：{ticket_id}\n"
            f"**优先级**：{'🔴 紧急' if priority <= 2 else '🟡 普通' if priority <= 3 else '🟢 低'}\n"
            f"**预计等待**：约{wait_time}分钟\n"
            f"**处理进度**：您的请求已移交至专业团队处理。\n\n"
            f"### 已获取的信息\n"
            f"系统已自动收集以下上下文信息，无需重复提供：\n"
            f"- 原始问题摘要\n"
            f"- L1 初步分析结果\n"
            f"- L2 深度分析结果\n"
            f"- 升级路径：{' → '.join(t.name for t in ticket.tier_path)}\n\n"
            f"### 下一步\n"
            f"我们的专家将基于已有上下文在{wait_time}分钟内回复您。"
            f"您也可以通过工单编号 {ticket_id} 查询进度。\n"
        )

        return TierResult(
            tier=self.tier,
            answer=answer,
            confidence=1.0,  # 人工处理置信度100%
            time_elapsed=time_elapsed,
            metadata={
                "ticket_id": ticket_id,
                "queue_position": len(self.ticket_queue),
                "estimated_wait_minutes": wait_time,
                "priority": priority,
            },
        )

    def _calculate_priority(self, query: str, context: dict) -> int:
        """计算工单优先级（1=最高, 5=最低）。"""
        priority = 3  # 默认中等

        # 敏感话题 ➔ 高优先级
        for kw in ["投诉", "退款", "取消", "紧急", "urgent", "complaint"]:
            if kw in query.lower():
                priority = max(1, priority - 1)

        # 升级原因影响优先级
        reason = context.get("escalation_reason")
        if reason == EscalationReason.SENSITIVE_TOPIC:
            priority = 1
        elif reason == EscalationReason.MAX_RETRIES:
            priority = 2
        elif reason == EscalationReason.LOW_CONFIDENCE:
            priority = 3

        return priority

    def _estimate_wait_time(self) -> int:
        """估算等待时间（分钟）。"""
        queue_len = len(self.ticket_queue)
        return max(1, queue_len * 5)  # 每个工单约5分钟

    def on_new_ticket(self, handler: Callable) -> Callable:
        """注册新工单通知处理器。"""
        self._handlers.append(handler)
        return handler

    def get_next_ticket(self) -> Optional[HumanTicket]:
        """获取下一个待处理的工单（给人工客服使用）。"""
        try:
            return self.ticket_queue.popleft()
        except IndexError:
            return None

    def get_queue_stats(self) -> dict:
        """获取队列统计信息。"""
        return {
            "queue_length": len(self.ticket_queue),
            "estimated_wait_minutes": self._estimate_wait_time(),
            "oldest_ticket": (
                self.ticket_queue[0].created_at if self.ticket_queue else None
            ),
            "priority_distribution": self._get_priority_distribution(),
        }

    def _get_priority_distribution(self) -> dict[int, int]:
        """获取优先级分布。"""
        dist = {}
        for ticket in self.ticket_queue:
            dist[ticket.priority] = dist.get(ticket.priority, 0) + 1
        return dist

    def evaluate_confidence(self, answer: str, query: str) -> float:
        return 1.0  # 人工处理始终100%

    def should_escalate(
        self,
        query: str,
        result: TierResult,
        config: EscalationConfig,
        retry_count: int,
    ) -> Optional[EscalationReason]:
        return None  # 已经是最高层级，不再升级


# ============================================================================
# 升级管理器
# ============================================================================

class EscalationManager:
    """
    升级管理器：协调三级Agent的升级流程。

    完整流程：
    1. L1 Agent 尝试回答
    2. 检查是否满足升级条件
    3. 如果需要升级 → L2 Agent 尝试回答
    4. 再次检查升级条件
    5. 如果仍需升级 → L3 人工队列
    """

    def __init__(self, config: Optional[EscalationConfig] = None):
        self.config = config or EscalationConfig()
        self.l1_agent = L1SimpleAgent()
        self.l2_agent = L2FullAgent()
        self.l3_queue = L3HumanQueue()

        # 执行追踪
        self.execution_history: list[dict] = []

        # 统计
        self.stats: dict[Tier, int] = {
            Tier.L1: 0,
            Tier.L2: 0,
            Tier.L3: 0,
        }

    def handle_query(
        self,
        query: str,
        verbose: bool = True,
    ) -> dict:
        """
        处理用户查询（完整三级升级流程）。

        Args:
            query: 用户问题
            verbose: 是否打印详细日志

        Returns:
            包含最终答案、升级路径、各层结果的字典
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  📨 收到查询: {query[:80]}...")
            print(f"{'=' * 60}")

        escalation_path: list[Tier] = []
        all_results: list[TierResult] = []
        context: dict = {
            "original_query": query,
            "escalation_path": escalation_path,
        }

        # ---- L1 处理 ----
        escalation_path.append(Tier.L1)
        l1_retries = 0
        escalation_reason = None

        while l1_retries <= self.config.max_retries:
            result = self.l1_agent.process(query)
            escalation_reason = self.l1_agent.should_escalate(
                query, result, self.config, l1_retries,
            )

            if escalation_reason is None:
                # 不需要升级，L1直接成功
                result.should_escalate = False
                all_results.append(result)
                context["l1_answer"] = result.answer
                self.stats[Tier.L1] += 1
                break
            else:
                l1_retries += 1
                if l1_retries <= self.config.max_retries:
                    if verbose:
                        print(f"  🔄 L1 重试 ({l1_retries}/{self.config.max_retries})")
                else:
                    result.should_escalate = True
                    result.escalation_reason = escalation_reason
                    all_results.append(result)
                    context["l1_answer"] = result.answer
                    context["l1_escalation_reason"] = escalation_reason

        if escalation_reason is not None:
            if verbose:
                print(f"  ⬆️ L1 → L2 升级原因: {escalation_reason.value}")
                print(f"     置信度: {result.confidence:.2f} "
                      f"(阈值: {self.config.confidence_threshold})")

            # ---- L2 处理 ----
            escalation_path.append(Tier.L2)
            l2_retries = 0

            while l2_retries <= self.config.max_retries:
                result2 = self.l2_agent.process(query, context)
                escalation_reason2 = self.l2_agent.should_escalate(
                    query, result2, self.config, l2_retries,
                )

                if escalation_reason2 is None:
                    result2.should_escalate = False
                    all_results.append(result2)
                    context["l2_answer"] = result2.answer
                    self.stats[Tier.L2] += 1
                    break
                else:
                    l2_retries += 1
                    if l2_retries <= self.config.max_retries:
                        if verbose:
                            print(f"  🔄 L2 重试 ({l2_retries}/{self.config.max_retries})")
                    else:
                        result2.should_escalate = True
                        result2.escalation_reason = escalation_reason2
                        all_results.append(result2)
                        context["l2_answer"] = result2.answer
                        context["l2_escalation_reason"] = escalation_reason2

            if escalation_reason2 is not None:
                if verbose:
                    print(f"  ⬆️ L2 → L3 升级原因: {escalation_reason2.value}")
                    print(f"     置信度: {result2.confidence:.2f} "
                          f"(阈值: {self.config.confidence_threshold})")

                # ---- L3 人工处理 ----
                escalation_path.append(Tier.L3)
                context["l2_answer"] = result2.answer if "result2" in dir() else ""
                context["escalation_path"] = escalation_path
                context["escalation_reason"] = escalation_reason2

                result3 = self.l3_queue.process(query, context)
                all_results.append(result3)
                self.stats[Tier.L3] += 1

        # 构建最终响应
        final_result = all_results[-1]
        final_tier = final_result.tier

        response = {
            "query": query[:200],
            "final_tier": final_tier.name,
            "escalation_path": [t.name for t in escalation_path],
            "final_answer": final_result.answer,
            "final_confidence": final_result.confidence,
            "total_latency": sum(r.time_elapsed for r in all_results),
            "results_by_tier": [
                {
                    "tier": r.tier.name,
                    "confidence": r.confidence,
                    "time": round(r.time_elapsed, 3),
                    "escalated": r.should_escalate,
                    "escalation_reason": r.escalation_reason.value if r.escalation_reason else None,
                }
                for r in all_results
            ],
        }

        # 记录执行历史
        self.execution_history.append(response)

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  📊 处理完成 | 最终层级: {final_tier.name}")
            print(f"  升级路径: {' → '.join(t.name for t in escalation_path)}")
            print(f"  总延迟: {response['total_latency']:.1f}s")
            print(f"  置信度: {final_result.confidence:.1%}")
            print(f"{'─' * 60}")

        return response

    def get_stats_summary(self) -> dict:
        """获取处理统计摘要。"""
        total = sum(self.stats.values()) or 1
        return {
            "total_requests": len(self.execution_history),
            "tier_distribution": {
                tier.name: {
                    "count": count,
                    "percentage": round(count / total * 100, 1),
                }
                for tier, count in self.stats.items()
            },
            "avg_latency_s": round(
                sum(
                    h.get("total_latency", 0)
                    for h in self.execution_history
                ) / max(1, len(self.execution_history)),
                2,
            ),
            "queue_stats": self.l3_queue.get_queue_stats(),
        }


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  多智能体系统 - 三级升级模式演示")
    print("=" * 72)

    manager = EscalationManager()

    # ---- Demo 1: L1直接解决 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 简单问题 → L1 直接解决（无升级）")
    print("=" * 72)

    result1 = manager.handle_query("你好，请问你们的产品有哪些功能？", verbose=True)
    print(f"\n  最终回答（前300字符）:")
    print(f"  {result1['final_answer'][:300]}...")

    # ---- Demo 2: L1→L2 升级 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 复杂问题 → L1 置信度不足 → 升级到 L2")
    print("=" * 72)

    result2 = manager.handle_query(
        "请帮我做一个完整的竞品分析，包括市场份额、定价策略、技术栈对比、"
        "用户增长趋势和财务预测。需要引用数据来源。这个分析将用于董事会决策。",
        verbose=True,
    )

    # ---- Demo 3: L1→L2→L3 人工 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 敏感问题 → L1→L2→L3（人工处理）")
    print("=" * 72)

    result3 = manager.handle_query(
        "我最近被公司无故辞退，请帮我分析一下我应该如何起诉公司，"
        "能获得多少赔偿金？请给我具体的法律建议。",
        verbose=True,
    )

    # ---- Demo 4: 人工明确请求 ----

    print("\n\n" + "=" * 72)
    print("  Demo 4: 用户明确要求人工 → 直接升级")
    print("=" * 72)

    result4 = manager.handle_query(
        "我要转人工客服，不需要机器人回答我的问题。我的订单号是ORD-2025001。",
        verbose=True,
    )

    # ---- 统计摘要 ----

    print("\n\n" + "=" * 72)
    print("  📈 处理统计摘要")
    print("=" * 72)

    stats = manager.get_stats_summary()
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print("  升级模式演示完成！")
    print("=" * 72)
