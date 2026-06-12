#!/usr/bin/env python3
"""
上下文窗口编排器 (Context Window Orchestrator)
================================================
管理多组件对上下文窗口资源的竞争，实现预算协商、溢出降级和版本管理。

核心组件：
  - TaskComplexityEstimator: 任务复杂度评估器
  - ContextWindowOrchestrator: 上下文窗口编排器（预算协商）
  - OverflowDegradationHandler: 溢出降级策略处理器
  - ContextVersionManager: 上下文版本管理器（快照/差异/回滚）

应用场景：
  当多个组件（系统提示词、工具输出、检索文档、Agent思考记录等）
  竞争有限的上下文窗口时，编排器负责公平且高效地分配Token预算。
"""

from __future__ import annotations

import time
import json
import copy
import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque


# ============================================================================
# 复杂度等级
# ============================================================================

class ComplexityLevel(Enum):
    """任务复杂度等级"""
    TRIVIAL = 1        # 简单（1步完成）
    SIMPLE = 2         # 简单（2-3步）
    MODERATE = 3       # 中等（多步推理）
    COMPLEX = 4        # 复杂（多Agent协作）
    VERY_COMPLEX = 5   # 非常复杂（需要大量上下文）


# ============================================================================
# 降级策略
# ============================================================================

class DegradationStrategy(Enum):
    """降级处理策略"""
    TRUNCATE_OLDEST = auto()    # 截断最旧内容
    SUMMARIZE = auto()           # 摘要压缩旧内容
    DROP_LOWEST_PRIORITY = auto()  # 丢弃最低优先级
    REQUEST_MORE_BUDGET = auto()   # 请求更多预算（如换更大的模型）
    SPLIT_ACROSS_TURNS = auto()    # 跨轮次分割处理


@dataclass
class DegradationPlan:
    """降级处理计划"""
    strategy: DegradationStrategy
    affected_zones: List[str]        # 受影响的区域
    tokens_to_free: int              # 需要释放的Token数
    estimated_quality_impact: float  # 预估质量影响 [0, 1]
    actions: List[str]               # 具体操作步骤列表
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 任务复杂度评估器
# ============================================================================

class TaskComplexityEstimator:
    """
    任务复杂度评估器。

    根据任务描述和历史上下文评估任务需要的上下文窗口大小。
    """

    # 复杂度特征关键词
    COMPLEXITY_MARKERS = {
        ComplexityLevel.TRIVIAL: [
            "hello", "hi", "what", "who", "when", "你好", "什么是"
        ],
        ComplexityLevel.SIMPLE: [
            "explain", "describe", "how to", "为什么", "如何", "解释"
        ],
        ComplexityLevel.MODERATE: [
            "compare", "analyze", "summarize", "对比", "分析", "总结",
            "multi-step", "多步"
        ],
        ComplexityLevel.COMPLEX: [
            "design", "implement", "architect", "设计", "实现", "架构",
            "multi-agent", "orchestrate", "协调"
        ],
        ComplexityLevel.VERY_COMPLEX: [
            "end-to-end", "pipeline", "system", "端到端", "系统设计",
            "full-stack", "comprehensive", "全面"
        ],
    }

    def estimate(self, task: str,
                  history: Optional[List[Dict]] = None) -> ComplexityLevel:
        """
        评估任务复杂度。

        Args:
            task: 任务描述文本
            history: 历史上下文（可选）

        Returns:
            复杂度等级
        """
        task_lower = task.lower()
        scores = {level: 0 for level in ComplexityLevel}

        # 关键词匹配评分
        for level, markers in self.COMPLEXITY_MARKERS.items():
            for marker in markers:
                if marker.lower() in task_lower:
                    scores[level] += 1

        # 任务长度因子
        task_len = len(task)
        if task_len > 500:
            scores[ComplexityLevel.COMPLEX] += 2
        elif task_len > 200:
            scores[ComplexityLevel.MODERATE] += 1
        elif task_len > 100:
            scores[ComplexityLevel.SIMPLE] += 1

        # 历史上下文因子
        if history:
            history_len = len(history)
            if history_len > 20:
                scores[ComplexityLevel.VERY_COMPLEX] += 2
            elif history_len > 10:
                scores[ComplexityLevel.COMPLEX] += 2
            elif history_len > 5:
                scores[ComplexityLevel.MODERATE] += 1

        # 选择最高分
        best_level = ComplexityLevel.TRIVIAL
        best_score = -1
        for level in ComplexityLevel:
            if scores[level] > best_score:
                best_score = scores[level]
                best_level = level

        return best_level

    def estimate_token_needs(self, task: str,
                              history: Optional[List[Dict]] = None) -> int:
        """
        估算任务需要的Token预算。

        Returns:
            预估所需的Token数量
        """
        level = self.estimate(task, history)
        token_estimates = {
            ComplexityLevel.TRIVIAL: 2000,
            ComplexityLevel.SIMPLE: 8000,
            ComplexityLevel.MODERATE: 32000,
            ComplexityLevel.COMPLEX: 64000,
            ComplexityLevel.VERY_COMPLEX: 128000,
        }
        return token_estimates.get(level, 32000)


# ============================================================================
# 上下文版本管理器
# ============================================================================

@dataclass
class ContextSnapshot:
    """上下文快照"""
    snapshot_id: str
    timestamp: float
    zones_snapshot: Dict[str, str]   # zone_name -> serialized content
    total_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextVersionManager:
    """
    上下文版本管理器。

    功能：
    - 创建上下文快照
    - 计算两个版本间的差异
    - 回滚到历史版本
    """

    def __init__(self, max_snapshots: int = 50):
        self.snapshots: Dict[str, ContextSnapshot] = {}
        self._snapshot_order: List[str] = []  # 有序快照ID列表
        self.max_snapshots = max_snapshots
        self._counter = 0

    def snapshot(self, zones: Dict[str, str],
                  metadata: Optional[Dict] = None) -> ContextSnapshot:
        """
        创建当前上下文快照。

        Args:
            zones: {zone_name: content_string}
            metadata: 附加元数据

        Returns:
            创建的快照对象
        """
        self._counter += 1
        snapshot_id = f"snap-{self._counter:04d}-{int(time.time())}"

        # 序列化并存储
        zones_copy = {k: v for k, v in zones.items()}
        total_tokens = sum(len(v) for v in zones.values())

        snap = ContextSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            zones_snapshot=zones_copy,
            total_tokens=total_tokens,
            metadata=metadata or {},
        )

        self.snapshots[snapshot_id] = snap
        self._snapshot_order.append(snapshot_id)

        # 限制快照数量
        while len(self._snapshot_order) > self.max_snapshots:
            oldest_id = self._snapshot_order.pop(0)
            del self.snapshots[oldest_id]

        return snap

    def diff(self, snap_id_a: str, snap_id_b: str) -> Dict[str, Any]:
        """
        计算两个快照之间的差异。

        Returns:
            {
                "zone_diffs": {zone_name: {"added": int, "removed": int, "changed": int}},
                "new_zones": [str],
                "removed_zones": [str],
                "total_token_diff": int,
            }
        """
        snap_a = self.snapshots.get(snap_id_a)
        snap_b = self.snapshots.get(snap_id_b)

        if not snap_a or not snap_b:
            return {"error": "Snapshot not found"}

        all_zones = set(snap_a.zones_snapshot.keys()) | set(snap_b.zones_snapshot.keys())
        zone_diffs = {}
        new_zones = []
        removed_zones = []

        for zone in all_zones:
            content_a = snap_a.zones_snapshot.get(zone, "")
            content_b = snap_b.zones_snapshot.get(zone, "")

            if zone not in snap_a.zones_snapshot:
                new_zones.append(zone)
                zone_diffs[zone] = {"added": len(content_b), "removed": 0, "changed": len(content_b)}
            elif zone not in snap_b.zones_snapshot:
                removed_zones.append(zone)
                zone_diffs[zone] = {"added": 0, "removed": len(content_a), "changed": len(content_a)}
            else:
                zone_diffs[zone] = {
                    "added": max(0, len(content_b) - len(content_a)),
                    "removed": max(0, len(content_a) - len(content_b)),
                    "changed": self._hamming_distance(content_a, content_b),
                }

        return {
            "zone_diffs": zone_diffs,
            "new_zones": new_zones,
            "removed_zones": removed_zones,
            "total_token_diff": snap_b.total_tokens - snap_a.total_tokens,
        }

    def rollback(self, snapshot_id: str) -> Optional[Dict[str, str]]:
        """
        回滚到指定快照。

        Returns:
            恢复后的 zones 字典，如果快照不存在返回 None
        """
        snap = self.snapshots.get(snapshot_id)
        if not snap:
            return None
        return dict(snap.zones_snapshot)

    def _hamming_distance(self, a: str, b: str) -> int:
        """计算两字符串的简单差异（不同字符数）"""
        min_len = min(len(a), len(b))
        diff = abs(len(a) - len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                diff += 1
        return diff

    def get_latest_snapshot(self) -> Optional[ContextSnapshot]:
        """获取最新快照"""
        if self._snapshot_order:
            return self.snapshots[self._snapshot_order[-1]]
        return None

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """列出所有快照摘要"""
        return [
            {
                "id": sid,
                "timestamp": self.snapshots[sid].timestamp,
                "tokens": self.snapshots[sid].total_tokens,
            }
            for sid in self._snapshot_order
        ]


# ============================================================================
# 上下文窗口编排器
# ============================================================================

class ContextWindowOrchestrator:
    """
    上下文窗口编排器。

    核心功能：
    - 多组件预算协商
    - 动态配额调整
    - 溢出降级处理
    - 上下文版本管理集成
    """

    def __init__(self, total_budget: int = 128000,
                  reserved_for_future: float = 0.15):
        self.total_budget = total_budget
        self.reserved_for_future = reserved_for_future
        self.available_budget = int(total_budget * (1 - reserved_for_future))

        # 组件预算分配 {component_name: allocated_tokens}
        self.component_budgets: Dict[str, int] = {}
        self.component_priorities: Dict[str, float] = {}

        # 版本管理
        self.version_manager = ContextVersionManager()

        # 协商历史
        self.negotiation_history: List[Dict[str, Any]] = []

        # 降级处理器
        self.degradation_handler = OverflowDegradationHandler()

    def register_component(self, name: str, priority: float = 0.5,
                            min_budget: int = 0, max_budget: int = 32000):
        """
        注册一个上下文消费组件。

        Args:
            name: 组件名称
            priority: 优先级 [0, 1]
            min_budget: 最小Token预算
            max_budget: 最大Token预算
        """
        self.component_priorities[name] = priority

    def negotiate_budgets(self,
                           component_requests: Dict[str, int],
                           strategy: str = "proportional") -> Dict[str, int]:
        """
        多组件预算协商。

        协商策略：
        - "proportional": 按优先级比例分配
        - "equal": 均等分配
        - "greedy": 高优先级优先满足
        - "water_filling": 注水法（从低到高递进分配）

        Args:
            component_requests: {component_name: requested_tokens}
            strategy: 协商策略名称

        Returns:
            {component_name: allocated_tokens}
        """
        self.negotiation_history.append({
            "strategy": strategy,
            "requests": dict(component_requests),
            "timestamp": time.time(),
        })

        if strategy == "proportional":
            return self._budget_negotiation_proportional(component_requests)
        elif strategy == "equal":
            return self._budget_negotiation_equal(component_requests)
        elif strategy == "greedy":
            return self._budget_negotiation_greedy(component_requests)
        elif strategy == "water_filling":
            return self._budget_negotiation_water_filling(component_requests)
        else:
            return self._budget_negotiation_proportional(component_requests)

    def _budget_negotiation_proportional(self,
                                          requests: Dict[str, int]) -> Dict[str, int]:
        """按优先级比例分配"""
        total_priority = sum(
            self.component_priorities.get(name, 0.5)
            for name in requests
        )
        if total_priority == 0:
            return {name: 0 for name in requests}

        allocations = {}
        remaining = self.available_budget

        for name, requested in sorted(
            requests.items(),
            key=lambda x: self.component_priorities.get(x[0], 0.5),
            reverse=True,
        ):
            comp_priority = self.component_priorities.get(name, 0.5)
            proportional_share = int(
                self.available_budget * comp_priority / total_priority
            )
            allocated = min(requested, proportional_share, remaining)
            allocations[name] = allocated
            remaining -= allocated

        self.component_budgets = allocations
        return allocations

    def _budget_negotiation_equal(self,
                                   requests: Dict[str, int]) -> Dict[str, int]:
        """均等分配"""
        n = len(requests)
        if n == 0:
            return {}
        share = self.available_budget // n
        allocations = {}
        for name, requested in requests.items():
            allocations[name] = min(requested, share)
        self.component_budgets = allocations
        return allocations

    def _budget_negotiation_greedy(self,
                                    requests: Dict[str, int]) -> Dict[str, int]:
        """贪心：高优先级先满足"""
        sorted_components = sorted(
            requests.items(),
            key=lambda x: self.component_priorities.get(x[0], 0.5),
            reverse=True,
        )
        allocations = {}
        remaining = self.available_budget
        for name, requested in sorted_components:
            allocated = min(requested, remaining)
            allocations[name] = allocated
            remaining -= allocated
        self.component_budgets = allocations
        return allocations

    def _budget_negotiation_water_filling(self,
                                           requests: Dict[str, int]) -> Dict[str, int]:
        """
        注水法分配：从低水位（低预算请求）开始满足，
        逐步提升水位直至预算耗尽或所有请求满足。
        """
        sorted_requests = sorted(requests.items(), key=lambda x: x[1])
        n = len(sorted_requests)
        if n == 0:
            return {}

        allocations = {}
        remaining = self.available_budget

        # 按请求量从小到大满足
        for i, (name, requested) in enumerate(sorted_requests):
            remaining_components = n - i
            fair_share = remaining // remaining_components
            allocated = min(requested, fair_share, remaining)
            allocations[name] = allocated
            remaining -= allocated

        self.component_budgets = allocations
        return allocations

    def detect_overflow(self, current_usage: Dict[str, int]) -> bool:
        """检测是否即将溢出"""
        total_usage = sum(current_usage.values())
        return total_usage > self.available_budget * 0.9  # 90%阈值

    def get_stats(self) -> Dict[str, Any]:
        """获取编排器统计"""
        return {
            "total_budget": self.total_budget,
            "available_budget": self.available_budget,
            "allocated_budget": sum(self.component_budgets.values()),
            "unallocated": self.available_budget - sum(self.component_budgets.values()),
            "component_budgets": dict(self.component_budgets),
            "component_priorities": dict(self.component_priorities),
            "negotiation_count": len(self.negotiation_history),
        }


# ============================================================================
# 溢出降级处理器
# ============================================================================

class OverflowDegradationHandler:
    """
    溢出降级处理器。

    当上下文窗口即将溢出时，选择合适的降级策略。
    """

    def __init__(self):
        self.degradation_log: List[Dict[str, Any]] = []

    def degrade(self,
                current_usage: Dict[str, int],
                budget_limits: Dict[str, int],
                overflow_amount: int) -> DegradationPlan:
        """
        生成降级处理计划。

        Args:
            current_usage: 当前各组件使用量
            budget_limits: 各组件预算上限
            overflow_amount: 溢出量（Token数）

        Returns:
            DegradationPlan 降级计划
        """
        # 找出超预算的组件
        over_budget = {
            name: usage - budget_limits.get(name, usage)
            for name, usage in current_usage.items()
            if usage > budget_limits.get(name, usage)
        }

        if not over_budget:
            # 没有超预算，使用最温和的策略
            plan = DegradationPlan(
                strategy=DegradationStrategy.SUMMARIZE,
                affected_zones=list(current_usage.keys()),
                tokens_to_free=overflow_amount,
                estimated_quality_impact=0.1,
                actions=[
                    f"对 {list(current_usage.keys())} 应用摘要压缩",
                    f"释放 {overflow_amount} tokens",
                ],
            )
        else:
            # 有组件超预算
            most_over = max(over_budget.items(), key=lambda x: x[1])

            plan = DegradationPlan(
                strategy=DegradationStrategy.DROP_LOWEST_PRIORITY,
                affected_zones=[most_over[0]],
                tokens_to_free=overflow_amount,
                estimated_quality_impact=0.3,
                actions=[
                    f"从 '{most_over[0]}' 中移除低优先级内容",
                    f"目标释放: {overflow_amount} tokens",
                    f"当前超出: {most_over[1]} tokens",
                ],
            )

        self.degradation_log.append({
            "timestamp": time.time(),
            "strategy": plan.strategy.name,
            "tokens_freed": plan.tokens_to_free,
            "quality_impact": plan.estimated_quality_impact,
        })

        return plan

    def get_quality_report(self) -> Dict[str, Any]:
        """获取降级质量报告"""
        if not self.degradation_log:
            return {"total_degradations": 0, "avg_quality_impact": 0.0}

        impacts = [d["quality_impact"] for d in self.degradation_log]
        return {
            "total_degradations": len(self.degradation_log),
            "avg_quality_impact": round(sum(impacts) / len(impacts), 4),
            "max_quality_impact": round(max(impacts), 4),
            "total_tokens_freed": sum(d["tokens_freed"] for d in self.degradation_log),
        }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示"""
    print("=" * 70)
    print("  上下文窗口编排器 (Context Window Orchestrator) 演示")
    print("=" * 70)

    # ---- 1. 任务复杂度评估 ----
    print("\n【1. 任务复杂度评估】")
    print("-" * 50)

    estimator = TaskComplexityEstimator()
    test_tasks = [
        "什么是Python?",
        "请解释装饰器的工作原理",
        "对比分析微服务和单体架构的优缺点，并给出技术选型建议",
        "设计一个端到端的RAG系统，包括文档处理、向量检索、LLM生成和评估",
    ]

    for task in test_tasks:
        level = estimator.estimate(task)
        tokens = estimator.estimate_token_needs(task)
        print(f"  任务: {task[:40]}...")
        print(f"    复杂度: {level.name} (Level {level.value})")
        print(f"    预估Token需求: {tokens:,}")

    # ---- 2. 预算协商（5个组件） ----
    print("\n【2. 预算协商（5个组件竞争128K上下文）】")
    print("-" * 50)

    orchestrator = ContextWindowOrchestrator(total_budget=128000)

    # 注册组件
    components = {
        "system_prompt": 1.0,
        "conversation_history": 0.8,
        "retrieved_documents": 0.9,
        "tool_outputs": 0.7,
        "agent_scratchpad": 0.6,
    }

    for name, priority in components.items():
        orchestrator.register_component(name, priority)

    # 组件请求
    requests = {
        "system_prompt": 5000,
        "conversation_history": 30000,
        "retrieved_documents": 50000,
        "tool_outputs": 25000,
        "agent_scratchpad": 20000,
    }

    total_requested = sum(requests.values())
    print(f"  可用预算: {orchestrator.available_budget:,} tokens")
    print(f"  请求总量: {total_requested:,} tokens")
    print(f"  超出: {total_requested - orchestrator.available_budget:,} tokens")
    print()

    # 测试不同策略
    strategies = ["proportional", "equal", "greedy", "water_filling"]
    for strategy in strategies:
        allocations = orchestrator.negotiate_budgets(requests, strategy=strategy)
        total_allocated = sum(allocations.values())
        print(f"  [{strategy:15s}]")
        print(f"    分配结果:")
        for comp, allocated in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
            requested = requests[comp]
            pct = allocated / requested * 100 if requested > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"      {comp:25s}: {allocated:>6,}/{requested:>6,} ({pct:5.1f}%) {bar}")
        print(f"    总计分配: {total_allocated:,} / {orchestrator.available_budget:,}")
        print()

    # ---- 3. 溢出降级处理 ----
    print("【3. 溢出降级处理】")
    print("-" * 50)

    handler = OverflowDegradationHandler()

    # 模拟溢出场景
    current_usage = {
        "system_prompt": 5000,
        "conversation_history": 35000,   # 超预算
        "retrieved_documents": 60000,     # 严重超预算
        "tool_outputs": 15000,
        "agent_scratchpad": 15000,
    }

    budget_limits = {
        "system_prompt": 5000,
        "conversation_history": 25000,
        "retrieved_documents": 40000,
        "tool_outputs": 25000,
        "agent_scratchpad": 10000,
    }

    total_usage = sum(current_usage.values())
    overflow = total_usage - orchestrator.available_budget

    if overflow > 0:
        print(f"  检测到溢出! 溢出量: {overflow:,} tokens")
        plan = handler.degrade(current_usage, budget_limits, overflow)
        print(f"  降级策略: {plan.strategy.name}")
        print(f"  受影响区域: {plan.affected_zones}")
        print(f"  预估质量影响: {plan.estimated_quality_impact:.2%}")
        print(f"  操作步骤:")
        for action in plan.actions:
            print(f"    - {action}")

    print(f"\n  降级质量报告: {json.dumps(handler.get_quality_report(), indent=2)}")

    # ---- 4. 上下文版本管理 ----
    print("\n【4. 上下文版本管理】")
    print("-" * 50)

    version_mgr = ContextVersionManager()

    # 创建初始快照
    zones_v1 = {
        "system": "You are a helpful assistant. v1",
        "history": "User: Hello\nBot: Hi!",
        "docs": "Document about Python...",
    }
    snap1 = version_mgr.snapshot(zones_v1, metadata={"version": "v1"})
    print(f"  创建快照: {snap1.snapshot_id} ({snap1.total_tokens} tokens)")

    # 修改并创建第二个快照
    zones_v2 = {
        "system": "You are a helpful assistant. v2 (updated rules)",
        "history": "User: Hello\nBot: Hi!\nUser: Help with code\nBot: Sure!",
        "docs": "Document about Python and JavaScript...",
    }
    snap2 = version_mgr.snapshot(zones_v2, metadata={"version": "v2", "reason": "added history"})
    print(f"  创建快照: {snap2.snapshot_id} ({snap2.total_tokens} tokens)")

    # 计算差异
    diff = version_mgr.diff(snap1.snapshot_id, snap2.snapshot_id)
    print(f"\n  差异分析:")
    for zone, zone_diff in diff.get("zone_diffs", {}).items():
        print(f"    {zone}:")
        print(f"      +{zone_diff['added']} chars added")
        print(f"      -{zone_diff['removed']} chars removed")
        print(f"      ~{zone_diff['changed']} chars changed")

    # 回滚测试
    rolled_back = version_mgr.rollback(snap1.snapshot_id)
    if rolled_back:
        print(f"\n  回滚到 {snap1.snapshot_id}:")
        for zone, content in rolled_back.items():
            print(f"    {zone}: {content[:50]}...")

    # 快照列表
    print(f"\n  快照历史:")
    for s in version_mgr.list_snapshots():
        print(f"    {s['id']}: {s['tokens']} tokens (t={s['timestamp']:.0f})")

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
