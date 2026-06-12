#!/usr/bin/env python3
"""
智能模型路由器 (Model Router)
===============================
根据查询特征自动选择最优模型:

路由规则:
  1. 查询长度 (短→便宜, 长→复杂)
  2. 意图复杂度 (简单事实→便宜, 推理分析→昂贵)
  3. 领域特异性 (通用→便宜, 专业→昂贵)

模型池:
  - gpt-4o-mini: $0.15/1M input, $0.60/1M output (便宜, 快速)
  - gpt-4o: $2.50/1M input, $10/1M output (标准, 平衡)
  - claude-sonnet-4-6: $3.00/1M input, $15/1M output (强大, 推理)
  - deepseek-v3: $0.27/1M input, $1.10/1M output (性价比)

决策日志 + A/B测试覆盖 + 成本对比表
"""

import json
import hashlib
import time
import math
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque


# ============================================================================
# 核心数据结构
# ============================================================================

class ModelTier(Enum):
    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass
class ModelSpec:
    """模型规格"""
    name: str
    tier: ModelTier
    input_price_per_1m: float   # 每百万token价格
    output_price_per_1m: float
    context_window: int
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """路由决策记录"""
    decision_id: str
    query: str
    selected_model: str
    reason: str
    confidence: float  # 0-1
    alternatives: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    ab_test_group: str = ""  # 用于A/B测试
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 模型路由器
# ============================================================================

class ModelRouter:
    """智能模型路由器"""

    # 模型池定义
    MODEL_POOL: Dict[str, ModelSpec] = {
        'gpt-4o-mini': ModelSpec(
            name='gpt-4o-mini',
            tier=ModelTier.CHEAP,
            input_price_per_1m=0.15,
            output_price_per_1m=0.60,
            context_window=128000,
            strengths=['速度快', '成本低', '适合简单任务'],
            weaknesses=['复杂推理弱', '长文本生成一般'],
        ),
        'gpt-4o': ModelSpec(
            name='gpt-4o',
            tier=ModelTier.STANDARD,
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
            context_window=128000,
            strengths=['综合能力强', '多模态', '指令遵循好'],
            weaknesses=['成本高', '速度中等'],
        ),
        'claude-sonnet-4-6': ModelSpec(
            name='claude-sonnet-4-6',
            tier=ModelTier.PREMIUM,
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            context_window=200000,
            strengths=['推理能力强', '长上下文', '安全性好'],
            weaknesses=['成本最高', '没有视觉输入'],
        ),
        'deepseek-v3': ModelSpec(
            name='deepseek-v3',
            tier=ModelTier.CHEAP,
            input_price_per_1m=0.27,
            output_price_per_1m=1.10,
            context_window=65536,
            strengths=['性价比极高', '代码能力强', '中文优秀'],
            weaknesses=['生态系统较新', '合规性考量'],
        ),
    }

    def __init__(
        self,
        ab_test_enabled: bool = False,
        ab_test_split: float = 0.5,
        decision_history_size: int = 1000,
    ):
        self.ab_test_enabled = ab_test_enabled
        self.ab_test_split = ab_test_split

        # 决策日志
        self.decision_log: deque = deque(maxlen=decision_history_size)

        # 统计
        self.stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'count': 0, 'total_cost': 0.0, 'avg_confidence': 0.0}
        )

        # 成本对比基准 (如果全部使用gpt-4o)
        self.total_actual_cost = 0.0
        self.total_baseline_cost = 0.0  # 全部用gpt-4o的成本

        self.decision_counter = 0

    # ========================================================================
    # 路由规则
    # ========================================================================

    def route(
        self,
        query: str,
        user_id: str = "",
        force_model: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> RoutingDecision:
        """为核心路由逻辑

        综合评估:
          1. 查询复杂度
          2. 意图类型
          3. 领域特异性
          4. A/B测试覆盖

        Args:
            query: 用户查询
            user_id: 用户ID (用于A/B分桶)
            force_model: 强制指定模型 (覆盖所有规则)
            intent: 预分类的意图 (可选)

        Returns:
            RoutingDecision
        """
        self.decision_counter += 1
        decision_id = f"RD_{int(time.time())}_{self.decision_counter:06d}"

        # 强制覆盖
        if force_model and force_model in self.MODEL_POOL:
            decision = RoutingDecision(
                decision_id=decision_id,
                query=query,
                selected_model=force_model,
                reason="强制指定",
                confidence=1.0,
            )
            self._log_decision(decision)
            return decision

        # 1. 分析查询特性
        features = self._extract_features(query)

        # 2. 评分模型匹配度
        scores = self._score_models(query, features, intent)

        # 3. 选择最佳模型
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]

        # 4. A/B测试覆盖
        if self.ab_test_enabled and user_id:
            bucket = self._ab_test_bucket(user_id)
            if bucket == 'B' and best_score > 0.7:
                # 实验组: 使用备选模型
                alternatives = sorted(scores, key=scores.get, reverse=True)
                if len(alternatives) > 1:
                    best_model = alternatives[1]
                    reason = f"A/B测试组B - 路由到 {best_model}"
                else:
                    reason = f"路由到 {best_model} (A/B测试组B无备选)"
            else:
                reason = f"路由到 {best_model}"
        else:
            reason = f"路由到 {best_model}"

        # 5. 估算成本
        estimated_cost = self._estimate_cost(query, best_model)
        estimated_latency = self._estimate_latency(best_model)

        # 创建决策
        decision = RoutingDecision(
            decision_id=decision_id,
            query=query,
            selected_model=best_model,
            reason=reason,
            confidence=best_score,
            alternatives=[m for m in scores if m != best_model and scores[m] > 0.3],
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            ab_test_group=bucket if self.ab_test_enabled and user_id else "",
            metadata={
                'query_length': features['length'],
                'complexity': features['complexity'],
                'intent': intent or features['intent_guess'],
                'domain': features['domain'],
                'all_scores': {m: round(s, 3) for m, s in scores.items()},
            },
        )

        self._log_decision(decision)
        return decision

    def _extract_features(self, query: str) -> Dict[str, Any]:
        """提取查询特征"""
        query_upper = query.upper()

        # 长度
        length = len(query)
        word_count = len(query.split())

        # 复杂度
        complex_signals = [
            'ANALYZE', 'COMPARE', 'EVALUATE', 'SYNTHESIZE', 'EXPLAIN',
            'RELATIONSHIP', 'IMPACT', 'IMPLICATION', 'STRATEGY',
            'COMPREHENSIVE', 'DETAILED', 'THOROUGH', 'COMPLEX',
        ]
        simple_signals = [
            'WHAT', 'WHO', 'WHEN', 'WHERE', 'LIST', 'DEFINE', 'SIMPLE',
            'BASIC', 'QUICK',
        ]

        complex_count = sum(1 for kw in complex_signals if kw in query_upper)
        simple_count = sum(1 for kw in simple_signals if kw in query_upper)
        question_marks = query.count('?') + query.count('？')

        complexity_score = min(10, (
            complex_count * 3
            + (length / 100) * 2
            + question_marks * 2
            - simple_count * 1
        ))

        # 意图猜测
        intent_guess = self._guess_intent(query)

        # 领域检测
        domain = self._guess_domain(query)

        return {
            'length': length,
            'word_count': word_count,
            'complexity': complexity_score,
            'complexity_level': 'high' if complexity_score > 6 else ('medium' if complexity_score > 3 else 'low'),
            'question_marks': question_marks,
            'intent_guess': intent_guess,
            'domain': domain,
        }

    def _guess_intent(self, query: str) -> str:
        """猜测用户意图"""
        query_upper = query.upper()

        intents = {
            'fact_lookup': ['WHAT', 'WHO', 'WHEN', 'WHERE', 'WHICH'],
            'explanation': ['HOW', 'WHY', 'EXPLAIN', 'DESCRIBE', 'ELABORATE'],
            'comparison': ['COMPARE', 'VERSUS', 'VS', 'DIFFERENCE', 'SIMILAR'],
            'analysis': ['ANALYZE', 'EVALUATE', 'ASSESS', 'CRITIQUE'],
            'generation': ['CREATE', 'GENERATE', 'WRITE', 'COMPOSE', 'BUILD'],
            'code': ['CODE', 'PROGRAM', 'FUNCTION', 'DEBUG', 'IMPLEMENT', 'SCRIPT'],
            'translation': ['TRANSLATE', 'TRANSLATION', 'CONVERT TO'],
        }

        for intent_name, keywords in intents.items():
            if any(kw in query_upper for kw in keywords):
                return intent_name

        return 'general'

    def _guess_domain(self, query: str) -> str:
        """猜测领域"""
        query_upper = query.upper()

        domains = {
            'healthcare': ['PATIENT', 'DIAGNOSIS', 'TREATMENT', 'MEDICAL', 'CLINICAL', 'DRUG'],
            'finance': ['INVESTMENT', 'STOCK', 'BOND', 'PORTFOLIO', 'REVENUE', 'TAX'],
            'legal': ['LAW', 'CONTRACT', 'LIABILITY', 'COMPLIANCE', 'REGULATION', 'STATUTE'],
            'education': ['STUDENT', 'COURSE', 'LEARNING', 'TEACHING', 'CURRICULUM', 'EXAM'],
            'engineering': ['ARCHITECTURE', 'SYSTEM', 'DESIGN', 'PERFORMANCE', 'SCALABILITY'],
        }

        for domain, keywords in domains.items():
            if any(kw in query_upper for kw in keywords):
                return domain

        return 'general'

    def _score_models(
        self,
        query: str,
        features: Dict[str, Any],
        intent: Optional[str] = None,
    ) -> Dict[str, float]:
        """评分所有模型的匹配度"""
        scores = {}

        effective_intent = intent or features['intent_guess']
        complexity = features['complexity']
        length = features['length']
        domain = features['domain']

        for model_name, spec in self.MODEL_POOL.items():
            score = 0.5  # 基础分

            # 复杂度因素
            if complexity <= 3:
                # 简单查询 → 廉价模型
                if spec.tier == ModelTier.CHEAP:
                    score += 0.3
                elif spec.tier == ModelTier.PREMIUM:
                    score -= 0.2
            elif complexity >= 7:
                # 复杂查询 → 高级模型
                if spec.tier == ModelTier.PREMIUM:
                    score += 0.3
                elif spec.tier == ModelTier.CHEAP:
                    score -= 0.2

            # 意图因素
            intent_bonus = {
                'code': {'deepseek-v3': 0.2, 'claude-sonnet-4-6': 0.1},
                'analysis': {'claude-sonnet-4-6': 0.2, 'gpt-4o': 0.1},
                'generation': {'gpt-4o': 0.15, 'deepseek-v3': 0.1},
                'translation': {'deepseek-v3': 0.15},
                'fact_lookup': {'gpt-4o-mini': 0.3},
            }.get(effective_intent, {})

            score += intent_bonus.get(model_name, 0.0)

            # 领域因素 (专业领域需要更强的模型)
            if domain in ('healthcare', 'finance', 'legal'):
                if spec.tier in (ModelTier.PREMIUM, ModelTier.STANDARD):
                    score += 0.1
                else:
                    score -= 0.1

            # 长度因素
            if length > 500 and spec.context_window < 100000:
                score -= 0.1

            scores[model_name] = round(max(0.0, min(1.0, score)), 4)

        return scores

    def _estimate_cost(self, query: str, model_name: str) -> float:
        """估算单次查询成本"""
        spec = self.MODEL_POOL.get(model_name)
        if not spec:
            return 0.0

        est_input_tokens = len(query) * 1.5  # 粗略估计
        est_output_tokens = 200  # 假设平均输出200 tokens

        input_cost = (est_input_tokens / 1_000_000) * spec.input_price_per_1m
        output_cost = (est_output_tokens / 1_000_000) * spec.output_price_per_1m

        return input_cost + output_cost

    def _estimate_latency(self, model_name: str) -> float:
        """估算延迟 (毫秒)"""
        latency_map = {
            'gpt-4o-mini': 500,
            'gpt-4o': 1500,
            'claude-sonnet-4-6': 2000,
            'deepseek-v3': 800,
        }
        return latency_map.get(model_name, 1000)

    def _ab_test_bucket(self, user_id: str) -> str:
        """MD5哈希A/B分桶"""
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = hash_val % 100
        return 'A' if bucket < self.ab_test_split * 100 else 'B'

    def _log_decision(self, decision: RoutingDecision):
        """记录路由决策"""
        self.decision_log.append(decision)

        # 更新统计
        model_stats = self.stats[decision.selected_model]
        model_stats['count'] += 1
        model_stats['total_cost'] += decision.estimated_cost
        model_stats['avg_confidence'] = (
            (model_stats['avg_confidence'] * (model_stats['count'] - 1) + decision.confidence)
            / model_stats['count']
        )

        # 成本对比
        self.total_actual_cost += decision.estimated_cost
        # 基线: 全部用gpt-4o
        baseline_cost = self._estimate_cost(decision.query, 'gpt-4o')
        self.total_baseline_cost += baseline_cost

    # ========================================================================
    # 成本对比表
    # ========================================================================

    def get_cost_comparison_table(self) -> str:
        """生成成本对比表"""
        lines = [
            f"{'=' * 80}",
            f"  模型成本对比表 (Cost Comparison Table)",
            f"{'=' * 80}",
            f"  {'Model':<25s} {'Tier':<10s} {'Input $/1M':>10s} {'Output $/1M':>11s} {'1K Query Est':>13s}",
            f"  {'-' * 69}",
        ]

        total_cost = 0.0

        for name, spec in self.MODEL_POOL.items():
            if spec.tier == ModelTier.CHEAP:
                tier_display = "🟢 经济"
            elif spec.tier == ModelTier.STANDARD:
                tier_display = "🟡 标准"
            else:
                tier_display = "🔴 高端"

            # 估算1000次查询的总成本 (假设: 1000 input tokens + 300 output tokens)
            per_query_cost = (
                spec.input_price_per_1m * 1000 / 1_000_000
                + spec.output_price_per_1m * 300 / 1_000_000
            )
            k_queries_cost = per_query_cost * 1000

            lines.append(
                f"  {name:<25s} {tier_display:<10s} "
                f"${spec.input_price_per_1m:>8.2f}  ${spec.output_price_per_1m:>9.2f}  ${k_queries_cost:>11.4f}"
            )

        lines.append(f"\n  {'=' * 80}")
        lines.append(f"  实际总成本:  ${self.total_actual_cost:.6f}")
        lines.append(f"  基线成本 (全gpt-4o): ${self.total_baseline_cost:.6f}")
        lines.append(f"  节省:  ${self.total_baseline_cost - self.total_actual_cost:.6f} "
                     f"({(1 - self.total_actual_cost / max(0.000001, self.total_baseline_cost)) * 100:.1f}%)")
        lines.append(f"{'=' * 80}")

        return "\n".join(lines)

    def get_routing_report(self) -> Dict[str, Any]:
        """获取路由报告"""
        return {
            'total_decisions': self.decision_counter,
            'model_distribution': {
                model: stats['count'] for model, stats in self.stats.items()
            },
            'avg_confidence': {
                model: round(stats['avg_confidence'], 3) for model, stats in self.stats.items()
            },
            'total_actual_cost': round(self.total_actual_cost, 6),
            'total_baseline_cost': round(self.total_baseline_cost, 6),
            'cost_savings_pct': round(
                (1 - self.total_actual_cost / max(0.000001, self.total_baseline_cost)) * 100, 1
            ),
        }


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  智能模型路由器 - 完整演示")
    print("=" * 60)

    router = ModelRouter(ab_test_enabled=True, ab_test_split=0.3)

    # 测试查询
    test_queries = [
        ("简单事实", "What is the capital of France?"),
        ("代码生成", "Write a Python function to implement a binary search tree"),
        ("复杂分析", "Analyze the tradeoffs between microservices and monolithic architecture for enterprise applications, considering scalability, maintenance, cost, and team structure"),
        ("翻译任务", "Translate this medical report into Chinese: Patient presents with acute abdominal pain"),
        ("定义查询", "Define machine learning"),
        ("比较分析", "Compare transformer architecture with LSTM for sequence modeling tasks, including attention mechanisms and computational complexity"),
        ("推理问题", "A train leaves Station A at 60 mph. Another train leaves Station B at 45 mph. If the stations are 300 miles apart, when and where do they meet?"),
        ("创意写作", "Write a comprehensive business plan for an AI startup"),
        ("简短查询", "hi"),
        ("法律问题", "What are the GDPR requirements for AI systems that process personal data?"),
    ]

    print()
    for label, query in test_queries:
        user_id = f"user_{hash(query) % 100}"

        decision = router.route(query, user_id=user_id)
        spec = router.MODEL_POOL.get(decision.selected_model)

        ab_tag = f" [AB:{decision.ab_test_group}]" if decision.ab_test_group else ""
        print(f"  [{label:8s}] → {decision.selected_model:20s} "
              f"(tier={spec.tier.value:10s}, cost=${decision.estimated_cost:.6f})"
              f"{ab_tag}")
        print(f"           理由: {decision.reason[:60]}")

    # 成本对比表
    print()
    print(router.get_cost_comparison_table())

    # 路由报告
    print()
    report = router.get_routing_report()
    print(f"路由统计报告:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    print()
    print("=" * 60)
    print("  模型路由器演示完成")
    print("=" * 60)
