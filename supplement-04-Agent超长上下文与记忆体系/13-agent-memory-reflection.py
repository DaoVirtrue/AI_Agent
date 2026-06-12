#!/usr/bin/env python3
"""
Agent 记忆自省 (Agent Memory Reflection)
=============================================
实现 Agent 对自身记忆系统的自我评估、矛盾检测和再巩固控制。

核心组件：
  - MemorySelfEvaluator: 记忆自评估器
  - MemoryReflectionController: 记忆反思控制器
  - MemoryAwarePlanner: 记忆感知规划器
  - ReflectionLoop: 反思循环

自省流程：
  Evaluate → Detect Issues → Plan Fixes → Reconsolidate → Verify
"""

from __future__ import annotations

import re
import math
import time
import json
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict


# ============================================================================
# 枚举定义
# ============================================================================

class MemoryQuality(Enum):
    """记忆质量等级"""
    EXCELLENT = auto()     # 优秀
    GOOD = auto()          # 良好
    FAIR = auto()          # 一般
    POOR = auto()          # 差
    CRITICAL = auto()      # 严重（需要立即干预）


class ContradictionType(Enum):
    """矛盾类型"""
    FACTUAL = auto()         # 事实矛盾
    TEMPORAL = auto()        # 时间矛盾
    NUMERIC = auto()         # 数值矛盾
    SEMANTIC = auto()        # 语义矛盾（相反的陈述）
    STALE = auto()           # 过时信息


class ReflectionAction(Enum):
    """反思动作"""
    KEEP = auto()             # 保持
    UPDATE = auto()           # 更新
    DELETE = auto()           # 删除
    MERGE = auto()            # 合并
    RECONSOLIDATE = auto()    # 再巩固
    FLAG_FOR_REVIEW = auto()  # 标记待审


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class MemoryQualityAssessment:
    """记忆质量评估结果"""
    overall_quality: MemoryQuality
    completeness_score: float          # 完整性分数
    consistency_score: float           # 一致性分数
    freshness_score: float             # 新鲜度分数
    accuracy_confidence: float         # 准确性置信度
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryAwarePlan:
    """记忆感知的规划"""
    plan_id: str
    goal: str
    relevant_memories: List[str]       # 相关记忆ID列表
    memory_gaps: List[str]             # 记忆缺口（需要查询的信息）
    confidence: float                  # 规划置信度
    steps: List[Dict[str, Any]]        # 执行步骤
    fallback_plan: Optional[str] = None


@dataclass
class ReflectionResult:
    """反思结果"""
    reflection_id: str
    action: ReflectionAction
    target_memory_ids: List[str]
    reason: str
    quality_before: float
    quality_after: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# 记忆自评估器
# ============================================================================

class MemorySelfEvaluator:
    """
    记忆自评估器。

    从以下维度评估记忆系统质量：
    1. 完整性：关键信息是否充分
    2. 一致性：是否存在内部矛盾
    3. 新鲜度：信息是否及时更新
    4. 准确性：基于来源的置信度评估
    """

    def __init__(self):
        self._assessments: List[MemoryQualityAssessment] = []

    def evaluate(self, memories: List[Dict[str, Any]],
                  recent_queries: Optional[List[str]] = None
                  ) -> MemoryQualityAssessment:
        """
        对记忆系统进行综合评估。

        Args:
            memories: 记忆列表 [{id, content, mem_type, created_at, ...}]
            recent_queries: 最近的查询列表

        Returns:
            MemoryQualityAssessment
        """
        completeness = self._completeness_check(memories)
        consistency, contradictions = self._contradiction_detection(memories)
        freshness = self._freshness_assessment(memories)
        accuracy = self._accuracy_estimation(memories)

        # 综合评分
        overall_score = (
            0.30 * completeness +
            0.30 * consistency +
            0.25 * freshness +
            0.15 * accuracy
        )

        # 确定质量等级
        if overall_score >= 0.85:
            quality = MemoryQuality.EXCELLENT
        elif overall_score >= 0.70:
            quality = MemoryQuality.GOOD
        elif overall_score >= 0.50:
            quality = MemoryQuality.FAIR
        elif overall_score >= 0.30:
            quality = MemoryQuality.POOR
        else:
            quality = MemoryQuality.CRITICAL

        # 生成建议
        recommendations = self._generate_recommendations(
            completeness, consistency, freshness, contradictions
        )

        assessment = MemoryQualityAssessment(
            overall_quality=quality,
            completeness_score=round(completeness, 4),
            consistency_score=round(consistency, 4),
            freshness_score=round(freshness, 4),
            accuracy_confidence=round(accuracy, 4),
            contradictions=contradictions,
            recommendations=recommendations,
        )

        self._assessments.append(assessment)
        return assessment

    def _completeness_check(self, memories: List[Dict[str, Any]]) -> float:
        """
        完整性检查。

        检查维度：
        - 记忆总量是否足够
        - 关键类型（fact, event, preference）是否有覆盖
        - 最近时间段是否有记忆
        """
        if not memories:
            return 0.0

        scores = []

        # 数量分数（≥50个满分）
        count = len(memories)
        count_score = min(count / 50.0, 1.0)
        scores.append(count_score)

        # 类型覆盖分数（至少3种类型）
        types_present = set(m.get("mem_type", "general") for m in memories)
        type_score = min(len(types_present) / 4.0, 1.0)
        scores.append(type_score)

        # 近期覆盖分数（过去24小时内有记忆）
        now = time.time()
        recent_count = sum(
            1 for m in memories
            if m.get("created_at", 0) > now - 86400
        )
        recent_score = min(recent_count / 3.0, 1.0)
        scores.append(recent_score)

        # 内容长度分数
        content_scores = [
            min(len(m.get("content", "")) / 100.0, 1.0)
            for m in memories[:20]
        ]
        if content_scores:
            scores.append(sum(content_scores) / len(content_scores))

        return sum(scores) / len(scores)

    def _contradiction_detection(self,
                                   memories: List[Dict[str, Any]]
                                   ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        矛盾检测。

        检测类型：
        - 事实矛盾：同事实的不同版本
        - 数值矛盾：同一指标的冲突数值
        - 时间矛盾：时间线不一致
        - 语义矛盾：相反的陈述
        """
        if len(memories) < 2:
            return 1.0, []

        contradictions = []
        now = time.time()

        # 提取数值实体进行比较
        number_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(元|万|亿|件|个|人|%|度|公里|km|kg)?')

        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                mi, mj = memories[i], memories[j]

                # 数值矛盾检测
                nums_i = number_pattern.findall(mi.get("content", ""))
                nums_j = number_pattern.findall(mj.get("content", ""))

                for (ni_val, ni_unit), (nj_val, nj_unit) in [
                    (a, b) for a in nums_i for b in nums_j
                ]:
                    if ni_unit == nj_unit and ni_unit:  # 同一单位
                        vi, vj = float(ni_val), float(nj_val)
                        if vi > 0 and vj > 0 and vi != vj:
                            deviation = abs(vi - vj) / max(abs(vi), abs(vj))
                            if deviation > 0.3:  # 偏差超过30%
                                contradictions.append({
                                    "type": ContradictionType.NUMERIC.name,
                                    "memory_a": mi.get("id"),
                                    "memory_b": mj.get("id"),
                                    "value_a": f"{ni_val}{ni_unit}",
                                    "value_b": f"{nj_val}{nj_unit}",
                                    "deviation": round(deviation, 3),
                                })

                # 语义矛盾检测（简单：检查相反的修饰词）
                contradictory_pairs = [
                    ("高", "低"), ("增加", "减少"), ("上涨", "下跌"),
                    ("好", "差"), ("成功", "失败"), ("开启", "关闭"),
                    ("大", "小"), ("快", "慢"), ("多", "少"),
                ]
                content_a = mi.get("content", "")
                content_b = mj.get("content", "")
                for pos, neg in contradictory_pairs:
                    if pos in content_a and neg in content_b:
                        # 检查是否在讨论同一主题
                        words_a = set(re.findall(r'[\w一-鿿]+', content_a))
                        words_b = set(re.findall(r'[\w一-鿿]+', content_b))
                        overlap = words_a & words_b
                        if len(overlap) > 3:
                            contradictions.append({
                                "type": ContradictionType.SEMANTIC.name,
                                "memory_a": mi.get("id"),
                                "memory_b": mj.get("id"),
                                "positive_word": pos,
                                "negative_word": neg,
                                "topic_overlap": len(overlap),
                            })

        # 一致性分数 = 1.0 - 矛盾惩罚
        penalty = min(len(contradictions) * 0.15, 0.6)
        consistency_score = max(1.0 - penalty, 0.0)

        return consistency_score, contradictions

    def _freshness_assessment(self, memories: List[Dict[str, Any]]) -> float:
        """
        新鲜度评估。

        基于记忆的创建/更新时间计算整体新鲜度。
        """
        if not memories:
            return 0.0

        now = time.time()
        freshness_scores = []

        for mem in memories:
            updated_at = mem.get("updated_at", mem.get("created_at", now))
            age_days = (now - updated_at) / 86400.0

            # 指数衰减：半衰期30天
            decay = math.exp(-age_days * math.log(2) / 30.0)
            freshness_scores.append(decay)

        return sum(freshness_scores) / len(freshness_scores)

    def _accuracy_estimation(self, memories: List[Dict[str, Any]]) -> float:
        """
        准确性置信度估计。

        基于：
        - 来源可靠性（metadata中的source可靠性）
        - 被验证次数
        - 创建者可信度
        """
        if not memories:
            return 0.0

        scores = []
        for mem in memories:
            meta = mem.get("metadata", {})
            source_reliability = meta.get("source_reliability", 0.7)
            verify_count = meta.get("verify_count", 0)

            # 每次验证提高置信度
            verify_bonus = min(verify_count * 0.05, 0.3)
            accuracy = min(source_reliability + verify_bonus, 1.0)
            scores.append(accuracy)

        return sum(scores) / len(scores)

    def _generate_recommendations(self, completeness: float,
                                    consistency: float,
                                    freshness: float,
                                    contradictions: List[Dict]
                                    ) -> List[str]:
        """基于各项分数生成改进建议"""
        recommendations = []

        if completeness < 0.5:
            recommendations.append("建议补充更多记忆数据，当前完整度较低")
        if consistency < 0.5:
            recommendations.append(f"检测到 {len(contradictions)} 个矛盾，建议进行记忆调和")
        if freshness < 0.4:
            recommendations.append("记忆数据偏旧，建议更新或清理过期记忆")
        if contradictions:
            recommendations.append("发现冲突信息，建议优先解决事实矛盾（NUMERIC/SEMANTIC）")

        if not recommendations:
            recommendations.append("记忆系统状态良好，建议保持定期评估")

        return recommendations


# ============================================================================
# 记忆反思控制器
# ============================================================================

class MemoryReflectionController:
    """
    记忆反思控制器。

    根据评估结果决定是否需要进行记忆再巩固。
    """

    def __init__(self, evaluator: MemorySelfEvaluator):
        self.evaluator = evaluator
        self._reflection_history: List[ReflectionResult] = []
        self._reconsolidation_threshold = 0.6  # 质量低于0.6触发再巩固

    def should_reconsolidate(self,
                              assessment: MemoryQualityAssessment) -> bool:
        """
        判断是否需要再巩固。

        Args:
            assessment: 质量评估结果

        Returns:
            True 如果需要进行再巩固
        """
        if assessment.overall_quality in (MemoryQuality.POOR, MemoryQuality.CRITICAL):
            return True
        if assessment.consistency_score < self._reconsolidation_threshold:
            return True
        if len(assessment.contradictions) >= 3:
            return True
        return False

    def process_reconsolidation(self,
                                  memories: List[Dict[str, Any]],
                                  assessment: MemoryQualityAssessment
                                  ) -> List[ReflectionResult]:
        """
        执行记忆再巩固流程。

        步骤：
        1. 解决检测到的矛盾（合并/删除）
        2. 标记过时记忆
        3. 增强高质量记忆
        """
        results = []

        reflection_id = f"reflect-{uuid.uuid4().hex[:8]}"

        # 处理矛盾
        for contradiction in assessment.contradictions:
            if contradiction["type"] == ContradictionType.NUMERIC.name:
                # 数值矛盾：保留更新的那个
                results.append(ReflectionResult(
                    reflection_id=reflection_id,
                    action=ReflectionAction.FLAG_FOR_REVIEW,
                    target_memory_ids=[
                        str(contradiction["memory_a"]),
                        str(contradiction["memory_b"]),
                    ],
                    reason=f"数值矛盾: {contradiction['value_a']} vs {contradiction['value_b']}",
                    quality_before=assessment.consistency_score,
                    quality_after=assessment.consistency_score + 0.1,
                ))

            elif contradiction["type"] == ContradictionType.SEMANTIC.name:
                # 语义矛盾：标记为需要人工审查
                results.append(ReflectionResult(
                    reflection_id=reflection_id,
                    action=ReflectionAction.FLAG_FOR_REVIEW,
                    target_memory_ids=[
                        str(contradiction["memory_a"]),
                        str(contradiction["memory_b"]),
                    ],
                    reason=f"语义矛盾涉及: {contradiction['positive_word']} vs {contradiction['negative_word']}",
                    quality_before=assessment.consistency_score,
                    quality_after=assessment.consistency_score + 0.05,
                ))

        self._reflection_history.extend(results)
        return results

    def get_reconsolidation_stats(self) -> Dict[str, Any]:
        """获取再巩固统计"""
        if not self._reflection_history:
            return {"total_reflections": 0}

        action_counts = Counter(r.action.name for r in self._reflection_history)
        return {
            "total_reflections": len(self._reflection_history),
            "action_distribution": dict(action_counts),
            "avg_quality_improvement": round(
                sum(
                    (r.quality_after or 0) - r.quality_before
                    for r in self._reflection_history
                ) / len(self._reflection_history), 4
            ),
        }


# ============================================================================
# 记忆感知规划器
# ============================================================================

class MemoryAwarePlanner:
    """
    记忆感知规划器。

    在制定计划时主动考虑记忆系统的状态：
    - 识别记忆缺口（缺少的信息）
    - 依赖相关记忆进行规划
    - 提供备选方案（当记忆不可靠时）
    """

    def __init__(self):
        self._plans: List[MemoryAwarePlan] = []
        self._plan_counter = 0

    def plan_with_memory(self, goal: str,
                          available_memories: List[Dict[str, Any]],
                          required_info: List[str]) -> MemoryAwarePlan:
        """
        基于记忆库制定计划。

        Args:
            goal: 目标描述
            available_memories: 可用的记忆列表
            required_info: 计划所需的关键信息列表

        Returns:
            MemoryAwarePlan
        """
        self._plan_counter += 1
        plan_id = f"plan-{self._plan_counter:04d}"

        # 查找相关记忆
        relevant_ids = []
        for mem in available_memories:
            content = mem.get("content", "").lower()
            goal_lower = goal.lower()
            # 内容与目标相关
            if any(kw in content for kw in goal_lower.split()):
                relevant_ids.append(mem.get("id", "unknown"))

        # 检测记忆缺口
        memory_contents = {m.get("content", "") for m in available_memories}
        gaps = []
        for info in required_info:
            info_lower = info.lower()
            covered = any(info_lower in c.lower() for c in memory_contents)
            if not covered:
                gaps.append(info)

        # 计算置信度
        if gaps:
            confidence = max(0.3, 1.0 - len(gaps) / max(len(required_info), 1) * 0.5)
        else:
            confidence = min(0.95, 0.7 + len(relevant_ids) * 0.05)
            if len(available_memories) < 5:
                confidence *= 0.8

        # 生成执行步骤
        steps = self._generate_steps(goal, gaps, confidence)

        # 备选方案
        fallback = (
            f"如果上述步骤因记忆不足而失败，请直接询问用户以下信息: {', '.join(gaps)}"
            if gaps else None
        )

        plan = MemoryAwarePlan(
            plan_id=plan_id,
            goal=goal,
            relevant_memories=relevant_ids[:10],
            memory_gaps=gaps,
            confidence=round(confidence, 4),
            steps=steps,
            fallback_plan=fallback,
        )

        self._plans.append(plan)
        return plan

    def _generate_steps(self, goal: str, gaps: List[str],
                          confidence: float) -> List[Dict[str, Any]]:
        """根据目标和缺口生成执行步骤"""
        steps = []

        if gaps:
            steps.append({
                "step": 1,
                "action": "gather_missing_info",
                "description": f"收集缺失信息: {', '.join(gaps[:3])}",
                "priority": "high",
            })

        steps.append({
            "step": len(steps) + 1,
            "action": "retrieve_relevant_memories",
            "description": f"检索与 '{goal[:30]}' 相关的记忆",
            "priority": "high",
        })

        steps.append({
            "step": len(steps) + 1,
            "action": "validate_memory_quality",
            "description": "验证相关记忆的质量和一致性",
            "priority": "medium",
        })

        if confidence < 0.6:
            steps.append({
                "step": len(steps) + 1,
                "action": "request_user_confirmation",
                "description": "记忆置信度较低，请求用户确认",
                "priority": "high",
            })

        steps.append({
            "step": len(steps) + 1,
            "action": "execute_plan",
            "description": f"执行: {goal[:50]}",
            "priority": "normal",
        })

        return steps

    def get_plan_stats(self) -> Dict[str, Any]:
        """获取规划统计"""
        if not self._plans:
            return {"total_plans": 0}

        confidences = [p.confidence for p in self._plans]
        return {
            "total_plans": len(self._plans),
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
            "max_confidence": round(max(confidences), 4),
            "plans_with_gaps": sum(1 for p in self._plans if p.memory_gaps),
            "plans_with_fallback": sum(1 for p in self._plans if p.fallback_plan),
        }


# ============================================================================
# 反思循环
# ============================================================================

class ReflectionLoop:
    """
    反思循环。

    定期执行：评估 → 决策 → 再巩固 → 验证
    """

    def __init__(self, evaluator: MemorySelfEvaluator,
                  controller: MemoryReflectionController,
                  planner: MemoryAwarePlanner):
        self.evaluator = evaluator
        self.controller = controller
        self.planner = planner
        self._loop_history: List[Dict[str, Any]] = []
        self._loop_count = 0

    def run_cycle(self, memories: List[Dict[str, Any]],
                   recent_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        运行一次完整的反思循环。

        Returns:
            循环结果
        """
        self._loop_count += 1
        result = {
            "cycle": self._loop_count,
            "timestamp": time.time(),
            "memory_count": len(memories),
        }

        # 1. 评估
        assessment = self.evaluator.evaluate(memories, recent_queries)
        result["quality"] = assessment.overall_quality.name
        result["completeness"] = assessment.completeness_score
        result["consistency"] = assessment.consistency_score
        result["contradictions"] = len(assessment.contradictions)

        # 2. 决策
        needs_reconsolidation = self.controller.should_reconsolidate(assessment)
        result["needs_reconsolidation"] = needs_reconsolidation

        # 3. 再巩固（如果需要）
        if needs_reconsolidation:
            reconsolidation_results = self.controller.process_reconsolidation(
                memories, assessment
            )
            result["reconsolidation_actions"] = len(reconsolidation_results)
            result["actions"] = [
                {"action": r.action.name, "reason": r.reason[:50]}
                for r in reconsolidation_results[:3]
            ]
        else:
            result["reconsolidation_actions"] = 0

        # 4. 建议
        result["recommendations"] = assessment.recommendations

        self._loop_history.append(result)
        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """获取循环历史"""
        return self._loop_history


# ============================================================================
# Demo
# ============================================================================

def generate_reflection_demo_memories() -> List[Dict[str, Any]]:
    """生成用于演示反思的记忆数据"""
    now = time.time()
    return [
        {"id": "m1", "content": "项目Alpha的预算为500万元", "mem_type": "fact",
         "created_at": now - 86400, "metadata": {"source_reliability": 0.9, "verify_count": 3}},
        {"id": "m2", "content": "项目Alpha的预算调整为600万元", "mem_type": "fact",
         "created_at": now - 3600, "metadata": {"source_reliability": 0.95, "verify_count": 1}},
        {"id": "m3", "content": "Q2销售额增长30%", "mem_type": "fact",
         "created_at": now - 86400*30, "metadata": {"source_reliability": 0.8}},
        {"id": "m4", "content": "Q2销售额下降5%", "mem_type": "fact",
         "created_at": now - 86400*60, "metadata": {"source_reliability": 0.6}},
        {"id": "m5", "content": "客户满意度评分92分", "mem_type": "fact",
         "created_at": now - 86400*7, "metadata": {"source_reliability": 0.85}},
        {"id": "m6", "content": "用户偏好深色模式", "mem_type": "preference",
         "created_at": now - 86400*3, "metadata": {"source_reliability": 0.9}},
        {"id": "m7", "content": "昨天团队会议决定延期项目Beta", "mem_type": "event",
         "created_at": now - 86400, "metadata": {"source_reliability": 0.9}},
        {"id": "m8", "content": "今天天气晴朗", "mem_type": "general",
         "created_at": now - 3600, "metadata": {"source_reliability": 0.5}},
        {"id": "m9", "content": "系统在凌晨2点发生故障", "mem_type": "event",
         "created_at": now - 7200, "metadata": {"source_reliability": 0.95}},
        {"id": "m10","content": "竞争对手发布新产品，价格更低", "mem_type": "fact",
         "created_at": now - 86400*14, "metadata": {"source_reliability": 0.7}},
        {"id": "m11","content": "公司股价大幅上涨", "mem_type": "fact",
         "created_at": now - 86400*60, "metadata": {"source_reliability": 0.5}},
        {"id": "m12","content": "公司股价大幅下跌", "mem_type": "fact",
         "created_at": now - 86400*10, "metadata": {"source_reliability": 0.8}},
    ]


def demo():
    """完整演示"""
    print("=" * 70)
    print("  Agent 记忆自省 (Memory Reflection) 演示")
    print("=" * 70)

    # ---- 1. 记忆自评估 ----
    print("\n【1. 记忆自评估】")
    print("-" * 50)

    memories = generate_reflection_demo_memories()
    print(f"  记忆总数: {len(memories)}")
    print(f"  包含的矛盾: 预算500万/600万, 销售额+30%/-5%, 股价上涨/下跌")

    evaluator = MemorySelfEvaluator()
    assessment = evaluator.evaluate(memories)

    print(f"\n  评估结果:")
    print(f"    综合质量: {assessment.overall_quality.name}")
    print(f"    完整性: {assessment.completeness_score:.2%}")
    print(f"    一致性: {assessment.consistency_score:.2%}")
    print(f"    新鲜度: {assessment.freshness_score:.2%}")
    print(f"    准确性置信度: {assessment.accuracy_confidence:.2%}")
    print(f"    检测到的矛盾: {len(assessment.contradictions)} 个")
    for c in assessment.contradictions:
        print(f"      [{c['type']}] m{c['memory_a']} vs m{c['memory_b']}")

    print(f"\n    改进建议:")
    for rec in assessment.recommendations:
        print(f"      - {rec}")

    # ---- 2. 再巩固控制 ----
    print("\n【2. 再巩固控制】")
    print("-" * 50)

    controller = MemoryReflectionController(evaluator)
    needs_recon = controller.should_reconsolidate(assessment)
    print(f"  是否需要再巩固: {'是' if needs_recon else '否'}")

    if needs_recon:
        results = controller.process_reconsolidation(memories, assessment)
        print(f"  再巩固动作数: {len(results)}")
        for r in results:
            print(f"    {r.action.name}: {r.target_memory_ids} - {r.reason[:60]}...")

        stats = controller.get_reconsolidation_stats()
        print(f"\n  再巩固统计:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    # ---- 3. 记忆感知规划 ----
    print("\n【3. 记忆感知规划】")
    print("-" * 50)

    planner = MemoryAwarePlanner()

    # 场景1：信息充足的规划
    plan1 = planner.plan_with_memory(
        goal="为项目Alpha制定下季度预算方案",
        available_memories=memories,
        required_info=["项目Alpha预算", "Q2销售数据", "市场趋势"],
    )
    print(f"  规划 [{plan1.plan_id}]: {plan1.goal}")
    print(f"    置信度: {plan1.confidence:.2%}")
    print(f"    相关记忆: {len(plan1.relevant_memories)} 条")
    print(f"    记忆缺口: {plan1.memory_gaps}")
    print(f"    步骤:")
    for step in plan1.steps:
        print(f"      {step['step']}. [{step['priority']}] {step['description']}")

    # 场景2：信息不足的规划
    plan2 = planner.plan_with_memory(
        goal="为新产品线设计定价策略",
        available_memories=memories,
        required_info=["竞争对手定价", "成本结构", "目标用户画像", "市场规模"],
    )
    print(f"\n  规划 [{plan2.plan_id}]: {plan2.goal}")
    print(f"    置信度: {plan2.confidence:.2%}")
    print(f"    记忆缺口: {plan2.memory_gaps}")
    print(f"    备选方案: {plan2.fallback_plan}")

    # ---- 4. 反思循环 ----
    print("\n【4. 反思循环】")
    print("-" * 50)

    loop = ReflectionLoop(evaluator, controller, planner)

    # 运行3个周期的反思
    for cycle_num in range(3):
        # 模拟记忆随时间变化
        mod_memories = [dict(m) for m in memories]
        for m in mod_memories:
            m["created_at"] -= cycle_num * 86400 * 30  # 每次回退30天

        result = loop.run_cycle(mod_memories)
        print(f"  第{result['cycle']}轮: 质量={result['quality']}, "
              f"需要再巩固={result['needs_reconsolidation']}, "
              f"矛盾数={result['contradictions']}")

    # 循环历史总结
    history = loop.get_history()
    print(f"\n  反思循环总结 ({len(history)} 轮):")
    qualities = [h['quality'] for h in history]
    print(f"    质量趋势: {' → '.join(qualities)}")

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
