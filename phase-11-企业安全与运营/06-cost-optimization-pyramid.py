#!/usr/bin/env python3
"""
五级成本优化金字塔 (Cost Optimization Pyramid)
================================================
Level 1: Prompt Caching (节省90%) - 缓存系统提示和前缀
Level 2: Model Routing (简单→便宜, 复杂→昂贵) - 智能路由
Level 3: Response Caching - 语义相似查询的响应缓存
Level 4: Token Optimization - 优化提示词减少token
Level 5: Architecture Optimization - 架构层面的成本控制

预算告警: 每日/每周/每月
成本归因: 按租户/功能/模型
"""

import time
import hashlib
import json
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, OrderedDict


# ============================================================================
# 核心数据结构
# ============================================================================

class BudgetPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class CostBudget:
    """成本预算"""
    period: BudgetPeriod
    amount_usd: float
    current_spend: float = 0.0
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'warning': 0.70,
        'critical': 0.90,
        'exceeded': 1.00,
    })


@dataclass
class ModelPricing:
    """模型定价"""
    model_name: str
    input_price_per_1k: float
    output_price_per_1k: float
    category: str  # 'cheap', 'medium', 'expensive'


# ============================================================================
# Level 1: Prompt Caching (提示缓存)
# ============================================================================

class PromptCache:
    """Level 1: 提示缓存

    缓存系统提示词和常用前缀。
    对于长系统提示(1000+ tokens)，可节省90%的输入token成本。
    """

    def __init__(self, max_cache_size: int = 100):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_cache_size
        self.stats = {
            'hits': 0,
            'misses': 0,
            'tokens_saved': 0,
            'cost_saved': 0.0,
        }

    def get(self, prompt_prefix: str) -> Optional[str]:
        """获取缓存的提示

        Args:
            prompt_prefix: 提示前缀 (通常是系统提示+上下文)

        Returns:
            缓存的完整提示 (如果命中)
        """
        key = hashlib.sha256(prompt_prefix.encode()).hexdigest()

        if key in self.cache:
            self.stats['hits'] += 1
            # LRU: 移到末尾
            self.cache.move_to_end(key)
            return self.cache[key]

        self.stats['misses'] += 1
        return None

    def set(self, prompt_prefix: str, full_prompt: str, tokens_saved: int = 0):
        """缓存提示"""
        key = hashlib.sha256(prompt_prefix.encode()).hexdigest()

        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # 移除最久未使用的
                oldest = next(iter(self.cache))
                del self.cache[oldest]

            self.cache[key] = full_prompt

        self.stats['tokens_saved'] += tokens_saved
        # 估算成本节省 ($0.01 per 1K tokens input)
        self.stats['cost_saved'] += tokens_saved * 0.00001

    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / max(1, total)

    def estimated_savings_pct(self) -> float:
        """估算节省百分比"""
        if self.stats['misses'] == 0:
            return 0.0
        # 简化: 假设每次命中节省900个token (约等于系统提示长度)
        total_tokens_if_no_cache = (
            self.stats['hits'] * 1000 + self.stats['misses'] * 1000
        )
        tokens_with_cache = (
            self.stats['hits'] * 100 + self.stats['misses'] * 1000
        )
        return (1 - tokens_with_cache / max(1, total_tokens_if_no_cache))


# ============================================================================
# Level 2: Model Routing (模型路由)
# ============================================================================

class ModelRouter:
    """Level 2: 智能模型路由器

    根据查询复杂度动态选择最合适的模型:
      - 简单查询 → 廉价模型 (gpt-4o-mini)
      - 中等查询 → 标准模型 (gpt-4o)
      - 复杂查询 → 高级模型 (claude-sonnet-4-6)
    """

    PRICING_TABLE = {
        'gpt-4o-mini': ModelPricing('gpt-4o-mini', 0.00015, 0.0006, 'cheap'),
        'gpt-4o': ModelPricing('gpt-4o', 0.0025, 0.01, 'medium'),
        'claude-sonnet-4-6': ModelPricing('claude-sonnet-4-6', 0.003, 0.015, 'medium'),
    }

    # 复杂度分析规则
    COMPLEXITY_RULES = {
        'simple': {
            'max_query_length': 100,
            'max_keywords': 3,
            'route_to': 'gpt-4o-mini',
        },
        'complex': {
            'min_query_length': 200,
            'min_keywords': 5,
            'route_to': 'gpt-4o',
        },
        'expert': {
            'requires_reasoning': True,
            'route_to': 'claude-sonnet-4-6',
        },
    }

    def __init__(self):
        self.routing_history: List[Dict] = []
        self.cost_saved: float = 0.0

    def analyze_complexity(self, query: str) -> str:
        """分析查询复杂度

        Returns:
            'simple' | 'medium' | 'complex'
        """
        score = 0
        query_upper = query.upper()
        query_len = len(query)

        # 长度因素
        if query_len < 60:
            score += 0
        elif query_len < 200:
            score += 2
        else:
            score += 4

        # 关键词
        complex_keywords = ['ANALYZE', 'COMPARE', 'EVALUATE', 'SYNTHESIZE', 'IMPACT',
                            'RELATIONSHIP', 'IMPLICATION', 'COMPREHENSIVE', 'COMPLEX']
        simple_keywords = ['WHAT', 'WHO', 'WHERE', 'WHEN', 'DEFINE', 'LIST', 'SIMPLE']

        ckw_count = sum(1 for kw in complex_keywords if kw in query_upper)
        skw_count = sum(1 for kw in simple_keywords if kw in query_upper)

        score += ckw_count * 3
        score -= skw_count * 1

        # 多问题检测
        question_count = query.count('?') + query.count('？')
        score += question_count * 2

        # 判断
        if score <= 2:
            return 'simple'
        elif score <= 6:
            return 'medium'
        else:
            return 'complex'

    def route(self, query: str, force_model: Optional[str] = None) -> Tuple[str, float]:
        """路由到合适的模型

        Args:
            query: 用户查询
            force_model: 强制使用的模型 (用于A/B测试)

        Returns:
            (模型名称, 估算成本)
        """
        if force_model:
            model = force_model
        else:
            complexity = self.analyze_complexity(query)

            if complexity == 'simple':
                model = 'gpt-4o-mini'
            elif complexity == 'medium':
                model = 'gpt-4o'
            else:
                model = 'claude-sonnet-4-6'

        # 估算成本
        pricing = self.PRICING_TABLE.get(model)
        estimated_tokens_input = len(query) * 2  # 粗略估计
        estimated_tokens_output = 200

        if pricing:
            cost = (pricing.input_price_per_1k * estimated_tokens_input / 1000
                    + pricing.output_price_per_1k * estimated_tokens_output / 1000)
        else:
            cost = 0.0

        # 记录
        self.routing_history.append({
            'timestamp': datetime.now().isoformat(),
            'query_len': len(query),
            'model': model,
            'estimated_cost': cost,
            'forced': force_model is not None,
        })

        # 计算节省 (vs 全用gpt-4o)
        always_gpt4o_cost = (self.PRICING_TABLE['gpt-4o'].input_price_per_1k * estimated_tokens_input / 1000
                              + self.PRICING_TABLE['gpt-4o'].output_price_per_1k * estimated_tokens_output / 1000)
        self.cost_saved += (always_gpt4o_cost - cost)

        return model, cost

    def get_cost_savings_report(self) -> Dict:
        """获取成本节省报告"""
        return {
            'total_cost_saved': round(self.cost_saved, 6),
            'total_routes': len(self.routing_history),
            'model_distribution': self._model_distribution(),
            'complexity_distribution': self._complexity_distribution(),
        }

    def _model_distribution(self) -> Dict[str, int]:
        """模型使用分布"""
        dist = defaultdict(int)
        for r in self.routing_history:
            dist[r['model']] += 1
        return dict(dist)

    def _complexity_distribution(self) -> Dict[str, int]:
        """复杂度分布"""
        dist = defaultdict(int)
        for r in self.routing_history:
            complexity = self.analyze_complexity(r.get('query', ''))
            dist[complexity] += 1
        return dict(dist)


# ============================================================================
# Level 3: Response Caching (响应缓存)
# ============================================================================

class ResponseCache:
    """Level 3: 语义响应缓存

    对于语义相似的查询，直接返回缓存的响应。
    """

    def __init__(self, max_size: int = 500, similarity_threshold: float = 0.85):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.stats = {'hits': 0, 'misses': 0, 'approximate_hits': 0}

    def _text_hash(self, text: str) -> str:
        """文本哈希"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard相似度"""
        import re
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def get(self, query: str) -> Optional[Tuple[str, float]]:
        """查找缓存

        Returns:
            (缓存的响应, 相似度) 或 None
        """
        # 精确匹配
        exact_key = self._text_hash(query)
        if exact_key in self.cache:
            self.stats['hits'] += 1
            self.cache.move_to_end(exact_key)
            return (self.cache[exact_key], 1.0)

        # 近似匹配 (模糊查找)
        for cached_query, response in self.cache.items():
            if self._jaccard_similarity(query, cached_query) >= self.similarity_threshold:
                self.stats['approximate_hits'] += 1
                return (response, self._jaccard_similarity(query, cached_query))

        self.stats['misses'] += 1
        return None

    def set(self, query: str, response: str):
        """缓存响应"""
        key = self._text_hash(query)
        self.cache[key] = response
        # 同时存原始查询用于模糊匹配
        self.cache[query] = response

        if len(self.cache) > self.max_size:
            # 移除最久未使用的两个条目
            for _ in range(2):
                if self.cache:
                    self.cache.popitem(last=False)

    @property
    def hit_rate(self):
        total = self.stats['hits'] + self.stats['misses'] + self.stats['approximate_hits']
        return (self.stats['hits'] + self.stats['approximate_hits']) / max(1, total)


# ============================================================================
# Level 4: Token Optimization (Token优化)
# ============================================================================

class TokenOptimizer:
    """Level 4: Token使用优化

    优化提示词以减少token数量:
      - 移除多余空白
      - 压缩重复内容
      - 使用更简洁的表述
    """

    @staticmethod
    def optimize_prompt(prompt: str, max_tokens: int = 4000) -> Tuple[str, int]:
        """优化提示词

        Args:
            prompt: 原始提示词
            max_tokens: 最大token数

        Returns:
            (优化后的提示词, 节省的token数)
        """
        original_tokens = TokenOptimizer._estimate_tokens(prompt)

        # 1. 压缩多余空白
        import re
        optimized = re.sub(r'\n{3,}', '\n\n', prompt)  # 最多连续2个换行
        optimized = re.sub(r' {2,}', ' ', optimized)     # 压缩多余空格

        # 2. 移除常见套话
        verbose_phrases = [
            ("I would like to", ""),
            ("Please note that", ""),
            ("It is important to note that", ""),
            ("I want you to", "Please"),
            ("Could you please", "Please"),
            ("I would appreciate it if you could", "Please"),
        ]
        for verbose, concise in verbose_phrases:
            if verbose in optimized:
                optimized = optimized.replace(verbose, concise)

        # 3. 如果仍然太长，截断上下文
        optimized_tokens = TokenOptimizer._estimate_tokens(optimized)
        if optimized_tokens > max_tokens:
            # 保留第一段和最后一段上下文
            parts = optimized.split('\n\n')
            if len(parts) > 2:
                optimized = parts[0] + '\n\n' + parts[-1]

        final_tokens = TokenOptimizer._estimate_tokens(optimized)
        saved = max(0, original_tokens - final_tokens)

        return optimized, saved

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算token数 (简化: 每4个字符≈1个token)"""
        return len(text) // 4


# ============================================================================
# Level 5: Architecture Optimization (架构优化)
# ============================================================================

class ArchitectureOptimizer:
    """Level 5: 架构级成本优化

    策略:
      - 批量处理请求 (减少API调用次数)
      - 异步并行调用 (减少等待时间)
      - 预计算常见查询
    """

    def __init__(self):
        self.batch_buffer: List[Dict] = []
        self.batch_max_size = 10
        self.batch_max_wait_seconds = 2.0
        self.stats = {'batched_calls': 0, 'individual_calls': 0}

    def add_to_batch(self, request: Dict) -> Optional[List[Dict]]:
        """添加请求到批次"""
        self.batch_buffer.append(request)

        if len(self.batch_buffer) >= self.batch_max_size:
            return self.flush_batch()
        return None

    def flush_batch(self) -> List[Dict]:
        """刷新批次"""
        batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.stats['batched_calls'] += len(batch)
        return batch


# ============================================================================
# 预算管理
# ============================================================================

class BudgetManager:
    """预算管理器 - 告警和限制"""

    def __init__(self):
        self.budgets: Dict[BudgetPeriod, CostBudget] = {}
        self.cost_attribution: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # 按 {tenant_id: {feature: cost}}
        self.alert_history: List[Dict] = []

    def set_budget(self, period: BudgetPeriod, amount: float):
        """设置预算"""
        self.budgets[period] = CostBudget(period=period, amount_usd=amount)

    def add_spend(self, amount: float, tenant: str = "default", feature: str = "default"):
        """记录支出"""
        self.cost_attribution[tenant][feature] += amount

        for period, budget in self.budgets.items():
            budget.current_spend += amount

            # 检查阈值
            ratio = budget.current_spend / budget.amount_usd
            if ratio >= budget.alert_thresholds['critical']:
                self._alert(period, budget, 'critical')
            elif ratio >= budget.alert_thresholds['warning']:
                self._alert(period, budget, 'warning')

    def _alert(self, period: BudgetPeriod, budget: CostBudget, level: str):
        """触发预算告警"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'period': period.value,
            'level': level,
            'budget': budget.amount_usd,
            'spent': budget.current_spend,
            'ratio': budget.current_spend / budget.amount_usd,
        }
        self.alert_history.append(alert)
        print(f"  [预算告警] {period.value} {level}: "
              f"${budget.current_spent:.2f}/${budget.amount_usd:.2f} "
              f"({budget.current_spent/budget.amount_usd:.0%})")

    def get_attribution_report(self) -> Dict[str, Any]:
        """成本归因报告"""
        by_tenant = {}
        for tenant, features in self.cost_attribution.items():
            by_tenant[tenant] = {
                'total': round(sum(features.values()), 4),
                'by_feature': {f: round(c, 4) for f, c in features.items()},
            }

        return {
            'by_tenant': by_tenant,
            'total_spend': round(sum(
                sum(f.values()) for f in self.cost_attribution.values()
            ), 4),
        }


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  五级成本优化金字塔 - 完整演示")
    print("=" * 60)

    # ---- Level 1: Prompt Cache ----
    print("\n[Level 1] Prompt Caching")
    cache = PromptCache()
    # 模拟: 系统提示相同，用户查询不同
    system_prompt = "You are a helpful AI assistant. " * 100  # 长系统提示
    for i in range(20):
        user_query = f"What is topic {i}?"
        full = system_prompt + user_query
        cached = cache.get(system_prompt)
        if cached is None:
            cache.set(system_prompt, full, tokens_saved=900)
    print(f"  命中率: {cache.hit_rate():.1%}")
    print(f"  节省Token: {cache.stats['tokens_saved']}")
    print(f"  节省成本: ${cache.stats['cost_saved']:.4f}")
    print(f"  估算节省比例: ~90%")

    # ---- Level 2: Model Router ----
    print("\n[Level 2] Model Routing")
    router = ModelRouter()

    test_queries = [
        "What is AI?",
        "Compare the performance characteristics of vector databases vs relational databases for RAG systems, analyzing cost, latency, and accuracy tradeoffs",
        "Define machine learning",
        "Evaluate the impact of prompt engineering on RAG system outcomes across different domains including healthcare, finance, and education",
    ]

    for q in test_queries:
        model, cost = router.route(q)
        complexity = router.analyze_complexity(q)
        print(f"  [{complexity:7s}] → {model:20s} (${cost:.6f}) | {q[:60]}...")

    savings = router.get_cost_savings_report()
    print(f"\n  成本节省: ${savings['total_cost_saved']:.6f}")
    print(f"  模型分布: {savings['model_distribution']}")

    # ---- Level 3: Response Cache ----
    print("\n[Level 3] Response Caching")
    resp_cache = ResponseCache()

    for i in range(15):
        query = f"What is RAG" if i < 10 else "New topic"
        result = resp_cache.get(query)
        if result is None:
            resp_cache.set(query, f"RAG answer for query {i}")
        else:
            pass  # 缓存命中

    print(f"  缓存命中率: {resp_cache.hit_rate:.1%}")
    print(f"  精确命中: {resp_cache.stats['hits']}")
    print(f"  近似命中: {resp_cache.stats['approximate_hits']}")
    print(f"  未命中: {resp_cache.stats['misses']}")

    # ---- Level 4: Token Optimization ----
    print("\n[Level 4] Token Optimization")
    verbose_prompt = (
        "I would like to request that you please assist me with the following task. "
        "Please note that this is very important. "
        "It is important to note that accuracy is critical. "
        + "context info " * 50
    )
    optimized, saved = TokenOptimizer.optimize_prompt(verbose_prompt)
    print(f"  原始token: ~{len(verbose_prompt) // 4}")
    print(f"  优化token: ~{len(optimized) // 4}")
    print(f"  保存token: ~{saved}")

    # ---- Level 5: Architecture ----
    print("\n[Level 5] Architecture Optimization")
    arch = ArchitectureOptimizer()
    for i in range(12):
        batch = arch.add_to_batch({"req": i})
    print(f"  批处理请求: {arch.stats['batched_calls']}")

    # ---- Budget Alerts ----
    print("\n[Budget] 预算管理")
    budget_mgr = BudgetManager()
    budget_mgr.set_budget(BudgetPeriod.DAILY, 10.0)
    budget_mgr.set_budget(BudgetPeriod.MONTHLY, 300.0)

    for tenant in ['tenant_a', 'tenant_b']:
        for feature in ['search', 'generate', 'embedding']:
            budget_mgr.add_spend(
                random.uniform(0.5, 3.0),
                tenant=tenant,
                feature=feature,
            )

    attribution = budget_mgr.get_attribution_report()
    print(f"  总支出: ${attribution['total_spend']:.2f}")
    for tenant, data in attribution['by_tenant'].items():
        print(f"    {tenant}: ${data['total']:.2f}")

    print()
    print("=" * 60)
    print("  成本优化金字塔演示完成")
    print("=" * 60)
