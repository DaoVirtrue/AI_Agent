#!/usr/bin/env python3
"""
服务降级管理 (Service Degradation Manager)
===========================================
6级降级策略:

L0 - Full (全功能): 所有功能正常
L1 - Light (轻度降级): 成本速率超预算 → 关闭高价模型路由
L2 - Moderate (中度降级): 错误率>0.5 → 关闭非核心功能
L3 - Heavy (重度降级): 延迟>120s → 使用缓存/简化回答
L4 - Severe (严重降级): API错误率>0.3 → 仅使用本地能力
L5 - Graceful (优雅降级): 完全故障 → 返回预设友好消息

每级定义:
  - 触发条件
  - 降级内容
  - 用户影响
  - 恢复条件
"""

import time
import json
import threading
from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from collections import deque, defaultdict


# ============================================================================
# 核心数据结构
# ============================================================================

class DegradationLevel(IntEnum):
    """降级级别 (数字越大越严重)"""
    L0_FULL = 0        # 全功能
    L1_LIGHT = 1       # 轻度降级
    L2_MODERATE = 2    # 中度降级
    L3_HEAVY = 3       # 重度降级
    L4_SEVERE = 4      # 严重降级
    L5_GRACEFUL = 5    # 优雅降级 (最小化服务)


@dataclass
class DegradationRule:
    """降级规则"""
    target_level: DegradationLevel
    condition_fn: Callable[['ServiceMetrics'], bool]
    description: str

    def evaluate(self, metrics: 'ServiceMetrics') -> bool:
        """评估是否触发此降级规则"""
        try:
            return self.condition_fn(metrics)
        except Exception:
            return False


@dataclass
class LevelConfig:
    """级别配置"""
    level: DegradationLevel
    name: str
    description: str
    user_impact: str
    recovery_conditions: str
    disabled_features: List[str] = field(default_factory=list)
    enabled_fallback: List[str] = field(default_factory=list)


@dataclass
class ServiceMetrics:
    """服务指标 (用于降级决策)"""
    # 成本和资源
    current_cost_rate_per_hour: float = 0.0  # 每小时成本
    cost_budget_per_hour: float = 10.0       # 成本预算

    # 错误率
    manager_error_rate: float = 0.0          # 模型/管理器错误率
    api_error_rate: float = 0.0              # 外部API错误率

    # 延迟
    avg_latency_seconds: float = 0.0         # 平均延迟
    p95_latency_seconds: float = 0.0         # P95延迟

    # 可用性
    llm_api_available: bool = True           # LLM API是否可用
    vector_db_available: bool = True         # 向量数据库是否可用
    cache_available: bool = True             # 缓存是否可用

    # 时间窗口
    error_rate_window_minutes: int = 5
    latency_window_minutes: int = 5

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# 降级管理器
# ============================================================================

class DegradationManager:
    """服务降级管理器

    6级降级 + 自动恢复
    """

    # 各级别配置
    LEVEL_CONFIGS: Dict[DegradationLevel, LevelConfig] = {
        DegradationLevel.L0_FULL: LevelConfig(
            level=DegradationLevel.L0_FULL,
            name="全功能",
            description="所有功能正常",
            user_impact="无",
            recovery_conditions="N/A (基准状态)",
            disabled_features=[],
            enabled_fallback=[],
        ),
        DegradationLevel.L1_LIGHT: LevelConfig(
            level=DegradationLevel.L1_LIGHT,
            name="轻度降级",
            description="成本超过预算限制",
            user_impact="复杂查询可能使用较便宜的模型，回答质量轻微下降",
            recovery_conditions="成本速率回到预算的80%以下持续10分钟",
            disabled_features=['premium_model_routing', 'costly_tools'],
            enabled_fallback=['cheap_model_only', 'response_cache_aggressive'],
        ),
        DegradationLevel.L2_MODERATE: LevelConfig(
            level=DegradationLevel.L2_MODERATE,
            name="中度降级",
            description="管理器错误率异常升高",
            user_impact="非核心功能暂时不可用 (如代码审查、详细分析)。核心问答仍正常",
            recovery_conditions="错误率<0.3持续15分钟",
            disabled_features=['code_review', 'detailed_analysis', 'report_generation'],
            enabled_fallback=['cache_first', 'simplified_answers'],
        ),
        DegradationLevel.L3_HEAVY: LevelConfig(
            level=DegradationLevel.L3_HEAVY,
            name="重度降级",
            description="响应延迟严重过高",
            user_impact="回答可能被截断或简化。仅返回关键信息。交互式对话不可用",
            recovery_conditions="P95延迟<60s持续20分钟",
            disabled_features=['interactive_conversation', 'follow_up_questions'],
            enabled_fallback=['truncated_answers', 'static_responses', 'offline_cache'],
        ),
        DegradationLevel.L4_SEVERE: LevelConfig(
            level=DegradationLevel.L4_SEVERE,
            name="严重降级",
            description="外部API大量失败",
            user_impact="无法生成新回答。只能使用预缓存的响应和本地规则",
            recovery_conditions="API错误率<0.1持续30分钟或手动恢复",
            disabled_features=['llm_generation', 'external_api', 'real_time_search'],
            enabled_fallback=['pre_cached_only', 'rule_based_responses', 'offline_mode'],
        ),
        DegradationLevel.L5_GRACEFUL: LevelConfig(
            level=DegradationLevel.L5_GRACEFUL,
            name="优雅降级",
            description="系统完全不可用",
            user_impact="返回友好提示消息。告知用户系统正在恢复中。提供联系方式和预计恢复时间",
            recovery_conditions="手动恢复或所有指标正常持续30分钟",
            disabled_features=['all'],
            enabled_fallback=['graceful_message', 'contact_support', 'status_page_link'],
        ),
    }

    def __init__(self):
        self.current_level: DegradationLevel = DegradationLevel.L0_FULL
        self.previous_level: DegradationLevel = DegradationLevel.L0_FULL
        self.level_history: deque = deque(maxlen=100)

        # 降级规则
        self.degradation_rules: List[DegradationRule] = []
        self._init_default_rules()

        # 恢复检查
        self._stable_since: Optional[datetime] = None
        self._recovery_threshold_minutes = 10

        # 统计
        self.stats: Dict[DegradationLevel, int] = defaultdict(int)
        self.last_updated: datetime = datetime.now()

        # 回调
        self.on_degradation: Optional[Callable] = None
        self.on_recovery: Optional[Callable] = None
        self.status_message: str = ""

        # 指标窗口
        self._metric_history: deque = deque(maxlen=100)

    def _init_default_rules(self):
        """初始化默认降级规则"""
        # L1: 成本超预算
        self.degradation_rules.append(DegradationRule(
            target_level=DegradationLevel.L1_LIGHT,
            condition_fn=lambda m: m.current_cost_rate_per_hour > m.cost_budget_per_hour,
            description=f"成本速率超预算",
        ))

        # L2: 管理器错误率过高
        self.degradation_rules.append(DegradationRule(
            target_level=DegradationLevel.L2_MODERATE,
            condition_fn=lambda m: m.manager_error_rate > 0.5,
            description="管理器错误率超过50%",
        ))

        # L3: 延迟过高
        self.degradation_rules.append(DegradationRule(
            target_level=DegradationLevel.L3_HEAVY,
            condition_fn=lambda m: m.avg_latency_seconds > 120,
            description="平均延迟超过120秒",
        ))

        # L4: API错误率过高
        self.degradation_rules.append(DegradationRule(
            target_level=DegradationLevel.L4_SEVERE,
            condition_fn=lambda m: m.api_error_rate > 0.3,
            description="外部API错误率超过30%",
        ))

        # L5: LLM完全不可用
        self.degradation_rules.append(DegradationRule(
            target_level=DegradationLevel.L5_GRACEFUL,
            condition_fn=lambda m: not m.llm_api_available and not m.cache_available,
            description="LLM API和缓存都不可用",
        ))

    # ========================================================================
    # 降级评估
    # ========================================================================

    def evaluate(self, metrics: ServiceMetrics) -> Tuple[DegradationLevel, str]:
        """评估并应用降级级别

        Args:
            metrics: 当前服务指标

        Returns:
            (新级别, 变更说明)
        """
        self._metric_history.append(metrics)

        # 检查降级规则 (从高到低)
        new_level = DegradationLevel.L0_FULL
        reasons = []

        for rule in self.degradation_rules:
            if rule.evaluate(metrics):
                if rule.target_level > new_level:
                    new_level = rule.target_level
                    reasons.append(rule.description)

        # 如果需要降级
        if new_level > self.current_level:
            old_level = self.current_level
            self.current_level = new_level
            self.previous_level = old_level
            self.last_updated = datetime.now()
            self.stats[new_level] += 1

            config = self.LEVEL_CONFIGS[new_level]
            self.status_message = config.description

            if self.on_degradation:
                self.on_degradation(old_level, new_level, config)

            change = f"降级: {old_level.name} → {new_level.name} ({', '.join(reasons)})"

        # 检查是否可以恢复
        elif self._can_recover(metrics):
            old_level = self.current_level
            self.current_level = DegradationLevel.L0_FULL
            self.previous_level = old_level
            self.last_updated = datetime.now()
            self.status_message = "服务已恢复正常"

            if self.on_recovery:
                self.on_recovery(old_level, DegradationLevel.L0_FULL)

            change = f"恢复: {old_level.name} → L0_FULL"
        else:
            change = f"保持: {self.current_level.name}"

        # 记录历史
        self.level_history.append({
            'timestamp': datetime.now().isoformat(),
            'from': self.previous_level.value,
            'to': self.current_level.value,
            'reasons': reasons,
        })

        return self.current_level, change

    def _can_recover(self, metrics: ServiceMetrics) -> bool:
        """检查是否满足恢复条件"""
        if self.current_level == DegradationLevel.L0_FULL:
            return False

        # 获取当前级别的恢复条件
        config = self.LEVEL_CONFIGS[self.current_level]

        # 检查所有降级规则是否已经不再满足
        for rule in self.degradation_rules:
            if rule.target_level <= self.current_level:
                if rule.evaluate(metrics):
                    return False

        # 需要稳定一段时间
        if self._stable_since is None:
            self._stable_since = datetime.now()
            return False

        return (datetime.now() - self._stable_since).total_seconds() >= self._recovery_threshold_minutes * 60

    # ========================================================================
    # 查询
    # ========================================================================

    def get_config(self) -> LevelConfig:
        """获取当前级别的配置"""
        return self.LEVEL_CONFIGS[self.current_level]

    def is_feature_disabled(self, feature_name: str) -> bool:
        """检查某个功能是否被禁用"""
        config = self.get_config()
        return feature_name in config.disabled_features

    def get_degraded_response(self, original_response: str = "") -> str:
        """根据当前级别生成降级响应

        Returns:
            适合当前级别的响应
        """
        if self.current_level == DegradationLevel.L5_GRACEFUL:
            return (
                "我们目前遇到技术困难，正在努力恢复服务。\n"
                "请稍后再试或通过 support@example.com 联系我们。\n"
                f"状态页面: https://status.example.com\n"
                f"参考ID: 技术支持热线 400-XXX-XXXX"
            )

        if self.current_level == DegradationLevel.L4_SEVERE:
            return (
                f"[服务降级通知] 由于外部服务临时不可用，我们当前只能提供有限服务。\n"
                f"以下是我们缓存中最相关的信息:\n\n"
                f"{original_response[:300] if original_response else '正在查找本地资源...'}\n\n"
                f"完整服务预计在未来30分钟内恢复。"
            )

        if self.current_level >= DegradationLevel.L1_LIGHT:
            return original_response  # L1-L3: 仍然返回回答但可能简化

        return original_response

    def get_status_page(self) -> Dict[str, Any]:
        """获取状态页面数据"""
        config = self.get_config()
        return {
            'current_level': self.current_level.value,
            'level_name': config.name,
            'description': config.description,
            'user_impact': config.user_impact,
            'last_updated': self.last_updated.isoformat(),
            'degradation_count': {lvl.value: count for lvl, count in self.stats.items()},
            'history': list(self.level_history)[-10:],
        }


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  服务降级管理 - 完整演示")
    print("=" * 60)

    manager = DegradationManager()

    # 配置回调
    def on_degrade(old, new, config):
        print(f"  🚨 警报: {old.name} → {new.name}")
        print(f"     影响: {config.user_impact}")

    def on_recover(old, new):
        print(f"  ✅ 恢复: {old.name} → {new.name}")

    manager.on_degradation = on_degrade
    manager.on_recovery = on_recover

    # ---- 测试场景 ----
    scenarios = [
        ("正常状态", ServiceMetrics(
            current_cost_rate_per_hour=5.0,
            manager_error_rate=0.05,
            avg_latency_seconds=2.0,
            api_error_rate=0.02,
        )),
        ("成本超预算", ServiceMetrics(
            current_cost_rate_per_hour=15.0,  # 超过10/h预算
            cost_budget_per_hour=10.0,
            manager_error_rate=0.05,
            avg_latency_seconds=2.0,
            api_error_rate=0.02,
        )),
        ("高错误率", ServiceMetrics(
            current_cost_rate_per_hour=8.0,
            cost_budget_per_hour=10.0,
            manager_error_rate=0.65,  # 超过50%
            avg_latency_seconds=3.0,
            api_error_rate=0.05,
        )),
        ("超高延迟", ServiceMetrics(
            current_cost_rate_per_hour=6.0,
            cost_budget_per_hour=10.0,
            manager_error_rate=0.1,
            avg_latency_seconds=150.0,  # 超过120s
            api_error_rate=0.03,
        )),
        ("API大量失败", ServiceMetrics(
            current_cost_rate_per_hour=4.0,
            cost_budget_per_hour=10.0,
            manager_error_rate=0.1,
            avg_latency_seconds=2.0,
            api_error_rate=0.45,  # 超过30%
        )),
        ("完全故障", ServiceMetrics(
            current_cost_rate_per_hour=0,
            manager_error_rate=0,
            avg_latency_seconds=0,
            api_error_rate=1.0,
            llm_api_available=False,
            cache_available=False,
        )),
    ]

    print()
    for label, metrics in scenarios:
        print(f"{'─' * 60}")
        print(f"场景: {label}")

        new_level, change = manager.evaluate(metrics)
        config = manager.get_config()

        print(f"  {change}")
        print(f"  级别: [{new_level.value}] {config.name}")
        print(f"  禁用: {config.disabled_features}")
        print(f"  用户影响: {config.user_impact}")

        # 模拟降级响应
        degraded_resp = manager.get_degraded_response("这是正常的AI回答内容")
        if "通知" in degraded_resp:
            print(f"  降级响应: {degraded_resp[:150]}...")

    # 状态页面
    print(f"\n{'─' * 60}")
    print("系统状态页面")
    status = manager.get_status_page()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 恢复测试
    print(f"\n{'─' * 60}")
    print("恢复测试")

    # 先降级到L3
    manager.evaluate(ServiceMetrics(
        avg_latency_seconds=200,
        manager_error_rate=0.1,
        api_error_rate=0.02,
    ))
    print(f"  当前级别: {manager.current_level.name}")

    # 恢复正常指标 (需要连续多次才能恢复 - 模拟多轮检查)
    for i in range(3):
        new_level, _ = manager.evaluate(ServiceMetrics(
            avg_latency_seconds=2.0,
            manager_error_rate=0.02,
            api_error_rate=0.01,
        ))

    print(f"  恢复级别: {manager.current_level.name}")

    print()
    print("=" * 60)
    print("  服务降级管理演示完成")
    print("=" * 60)
