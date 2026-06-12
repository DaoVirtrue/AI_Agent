#!/usr/bin/env python3
"""
A/B测试框架 (A/B Testing Framework)
=====================================
生产环境中的A/B测试框架，包含:
  - MD5哈希分桶 (确定性、平衡性)
  - Welch's t-检验 (p<0.05)
  - Cohen's d 效应量 (effect size)
  - 样本量计算 (power analysis)
  - 流量分配 (traffic allocation)
  - 序贯测试与提前停止 (sequential testing with early stopping)
  - Bonferroni校正 (多重比较)
  - 报告生成 (report generation)
"""

import hashlib
import math
import json
import time
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

try:
    import numpy as np
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# 核心数据结构
# ============================================================================

class ExperimentStatus(Enum):
    """实验状态"""
    DRAFT = "draft"
    RUNNING = "running"
    STOPPED = "stopped"
    CONCLUDED = "concluded"
    ROLLED_BACK = "rolled_back"


class TestResult(Enum):
    """测试结论"""
    SIGNIFICANT_WINNER_A = "A wins (significant)"
    SIGNIFICANT_WINNER_B = "B wins (significant)"
    INCONCLUSIVE = "inconclusive"
    EARLY_STOP_NULL = "early stop (futility)"
    EARLY_STOP_WINNER = "early stop (efficacy)"


@dataclass
class ABTestConfig:
    """A/B测试配置"""
    experiment_id: str
    experiment_name: str
    description: str = ""

    # 实验组
    variant_a_name: str = "Control (A)"
    variant_b_name: str = "Treatment (B)"

    # 流量分配
    traffic_allocation_a: float = 0.50   # A组流量比例
    traffic_allocation_b: float = 0.50   # B组流量比例

    # 统计参数
    significance_level: float = 0.05     # α
    power_level: float = 0.80            # 1-β
    minimum_detectable_effect: float = 0.02  # MDE (相对)

    # 测试参数
    min_sample_size_per_variant: int = 1000
    max_test_duration_days: int = 14
    early_stopping_enabled: bool = True
    early_stopping_looks: int = 5  # 序贯测试的检查次数

    # 指标
    primary_metric: str = "weighted_score"  # 主指标名称
    metrics: List[str] = field(default_factory=lambda: [
        "weighted_score", "accuracy", "relevance", "faithfulness"
    ])

    # Bonferroni校正
    apply_bonferroni: bool = False

    # 其他
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)


@dataclass
class VariantMetrics:
    """实验变体的指标数据"""
    variant_name: str
    sample_size: int
    metric_values: Dict[str, List[float]] = field(default_factory=dict)
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """A/B测试结果"""
    experiment_id: str
    status: TestResult
    primary_metric: str

    # A组统计
    mean_a: float = 0.0
    std_a: float = 0.0
    n_a: int = 0

    # B组统计
    mean_b: float = 0.0
    std_b: float = 0.0
    n_b: int = 0

    # 检验结果
    statistic: float = 0.0
    p_value: float = 1.0
    degrees_of_freedom: float = 0.0
    effect_size_cohens_d: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    relative_lift: float = 0.0

    # 序贯测试
    sequential_result: Optional[Dict[str, Any]] = None

    # 多重比较
    bonferroni_threshold: Optional[float] = None
    per_metric_results: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    stopped_early: bool = False
    stopping_reason: str = ""
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_significant(self, alpha: Optional[float] = None) -> bool:
        """判断结果是否显著"""
        threshold = alpha or 0.05
        if self.bonferroni_threshold:
            threshold = self.bonferroni_threshold
        return self.p_value < threshold

    def effect_size_category(self) -> str:
        """Cohen's d效应量解释"""
        d = abs(self.effect_size_cohens_d)
        if d < 0.2:
            return "negligible (可忽略)"
        elif d < 0.5:
            return "small (小)"
        elif d < 0.8:
            return "medium (中)"
        else:
            return "large (大)"

    def summary(self) -> str:
        """生成结果摘要字符串"""
        lines = [
            f"A/B Test Result: {self.experiment_id}",
            f"  Status: {self.status.value}",
            f"  Metric: {self.primary_metric}",
            f"  A ({self.n_a} samples): mean={self.mean_a:.4f}, std={self.std_a:.4f}",
            f"  B ({self.n_b} samples): mean={self.mean_b:.4f}, std={self.std_b:.4f}",
            f"  Relative Lift: {self.relative_lift:+.2%}",
            f"  Cohen's d: {self.effect_size_cohens_d:.4f} ({self.effect_size_category()})",
            f"  p-value: {self.p_value:.6f} (α={self.bonferroni_threshold or 0.05})",
            f"  Significant: {self.is_significant()}",
            f"  95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]",
        ]
        if self.stopped_early:
            lines.append(f"  ⚠ Stopped Early: {self.stopping_reason}")
        return "\n".join(lines)


# ============================================================================
# A/B测试框架
# ============================================================================

class ABTestFramework:
    """A/B测试框架

    支持:
      - MD5哈希分桶
      - Welch's t-test
      - Cohen's d
      - 样本量计算
      - 序贯测试
      - Bonferroni校正
      - 报告生成
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.active_experiments: Dict[str, ABTestConfig] = {}
        self.experiment_data: Dict[str, Dict[str, List[float]]] = {}
        self.experiment_results: Dict[str, ABTestResult] = {}
        self._data_lock = {}  # 简化: 模拟线程安全

    # ========================================================================
    # 分桶 (MD5哈希)
    # ========================================================================

    @staticmethod
    def hash_bucket(
        identifier: str,
        num_buckets: int = 10000,
        salt: str = ""
    ) -> int:
        """MD5哈希分桶 - 确定性、均匀分布

        Args:
            identifier: 用户/会话标识符
            num_buckets: 总桶数 (默认10000, 精度0.01%)
            salt: 盐值 (用于不同实验隔离)

        Returns:
            桶编号 (0 到 num_buckets-1)
        """
        hash_input = f"{identifier}:{salt}"
        hash_hex = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        hash_int = int(hash_hex, 16)
        return hash_int % num_buckets

    @staticmethod
    def assign_variant(
        user_id: str,
        experiment_id: str,
        traffic_a: float = 0.5,
        traffic_b: float = 0.5,
    ) -> str:
        """为用户分配实验变体

        Args:
            user_id: 用户ID
            experiment_id: 实验ID
            traffic_a: A组流量比例
            traffic_b: B组流量比例

        Returns:
            'A' 或 'B'
        """
        bucket = ABTestFramework.hash_bucket(
            user_id, salt=experiment_id, num_buckets=10000
        )
        # A组的桶范围: 0 到 traffic_a*10000 - 1
        a_threshold = int(traffic_a * 10000)

        if bucket < a_threshold:
            return 'A'
        else:
            return 'B'

    @staticmethod
    def validate_bucket_balance(
        user_ids: List[str],
        experiment_id: str,
        traffic_a: float = 0.5,
    ) -> Dict[str, Any]:
        """验证分桶的平衡性

        Args:
            user_ids: 用户ID列表
            experiment_id: 实验ID
            traffic_a: A组流量比例

        Returns:
            平衡性验证结果
        """
        counts = {'A': 0, 'B': 0}
        for uid in user_ids:
            variant = ABTestFramework.assign_variant(uid, experiment_id, traffic_a, 1.0 - traffic_a)
            counts[variant] += 1

        total = len(user_ids)
        actual_ratio_a = counts['A'] / total if total > 0 else 0

        return {
            'total_users': total,
            'a_count': counts['A'],
            'b_count': counts['B'],
            'expected_ratio_a': traffic_a,
            'actual_ratio_a': round(actual_ratio_a, 4),
            'deviation': round(abs(actual_ratio_a - traffic_a), 4),
            'balanced': abs(actual_ratio_a - traffic_a) < 0.01,
        }

    # ========================================================================
    # Welch's t-检验
    # ========================================================================

    @classmethod
    def welch_t_test(
        cls,
        group_a: List[float],
        group_b: List[float],
    ) -> Tuple[float, float, float, float, float]:
        """Welch's t-检验 (不假设等方差)

        Args:
            group_a: A组数据
            group_b: B组数据

        Returns:
            (t统计量, p值, 自由度, A组均值, B组均值)
        """
        if SCIPY_AVAILABLE:
            # 使用scipy实现
            t_stat, p_value = scipy_stats.ttest_ind(
                group_a, group_b, equal_var=False, alternative='two-sided'
            )

            # 计算Welch-Satterthwaite自由度
            n1, n2 = len(group_a), len(group_b)
            s1, s2 = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

            if s1 == 0 and s2 == 0:
                df = n1 + n2 - 2
            else:
                num = (s1/n1 + s2/n2) ** 2
                den = ((s1/n1)**2)/(n1-1) + ((s2/n2)**2)/(n2-1)
                df = num / den if den > 0 else n1 + n2 - 2

            mean_a = np.mean(group_a)
            mean_b = np.mean(group_b)

            return t_stat, p_value, df, mean_a, mean_b
        else:
            # 从零实现 Welch's t-test
            n1, n2 = len(group_a), len(group_b)
            if n1 < 2 or n2 < 2:
                return 0.0, 1.0, 0.0, 0.0, 0.0

            # 均值和方差
            mean_a = sum(group_a) / n1
            mean_b = sum(group_b) / n2

            var_a = sum((x - mean_a) ** 2 for x in group_a) / (n1 - 1) if n1 > 1 else 0
            var_b = sum((x - mean_b) ** 2 for x in group_b) / (n2 - 1) if n2 > 1 else 0

            # t统计量
            se = math.sqrt(var_a / n1 + var_b / n2)
            if se == 0:
                return 0.0, 1.0, 0.0, mean_a, mean_b

            t_stat = (mean_b - mean_a) / se

            # Welch-Satterthwaite自由度
            if var_a == 0 and var_b == 0:
                df = n1 + n2 - 2
            else:
                num = (var_a/n1 + var_b/n2) ** 2
                den = ((var_a/n1)**2)/(n1-1) + ((var_b/n2)**2)/(n2-1)
                df = num / den if den > 0 else n1 + n2 - 2

            # p值 (使用简化的t分布近似)
            p_value = cls._approx_p_value(abs(t_stat), df)

            return t_stat, p_value, df, mean_a, mean_b

    @staticmethod
    def _approx_p_value(t: float, df: float) -> float:
        """t分布的近似p值 (无scipy时的回退方案)"""
        # 使用正态分布近似 (对于大自由度足够)
        # 标准正态CDF近似
        x = abs(t)
        # Abramowitz and Stegun approximation
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p_approx = 0.2316419
        t_recip = 1.0 / (1.0 + p_approx * x)

        normal_cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-x * x / 2.0) * (
            b1 * t_recip + b2 * t_recip**2 + b3 * t_recip**3
            + b4 * t_recip**4 + b5 * t_recip**5
        )

        p = 2.0 * (1.0 - normal_cdf)
        return max(0.0, min(1.0, p))

    # ========================================================================
    # Cohen's d 效应量
    # ========================================================================

    @classmethod
    def cohens_d(
        cls,
        group_a: List[float],
        group_b: List[float],
    ) -> float:
        """计算Cohen's d 效应量

        d = (M2 - M1) / s_pooled
        其中 s_pooled 是合并标准差

        Args:
            group_a: A组数据
            group_b: B组数据

        Returns:
            Cohen's d值
        """
        n1, n2 = len(group_a), len(group_b)
        if n1 < 2 or n2 < 2:
            return 0.0

        mean_a = sum(group_a) / n1
        mean_b = sum(group_b) / n2

        if SCIPY_AVAILABLE:
            var_a = np.var(group_a, ddof=1)
            var_b = np.var(group_b, ddof=1)
        else:
            var_a = sum((x - mean_a) ** 2 for x in group_a) / (n1 - 1)
            var_b = sum((x - mean_b) ** 2 for x in group_b) / (n2 - 1)

        pooled_sd = math.sqrt(((n1 - 1) * var_a + (n2 - 1) * var_b) / (n1 + n2 - 2))

        if pooled_sd == 0:
            return 0.0

        return (mean_b - mean_a) / pooled_sd

    @classmethod
    def confidence_interval(
        cls,
        group_a: List[float],
        group_b: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """计算均值差的置信区间

        Args:
            group_a: A组数据
            group_b: B组数据
            confidence: 置信水平 (默认0.95)

        Returns:
            (下界, 上界)
        """
        n1, n2 = len(group_a), len(group_b)
        if n1 < 2 or n2 < 2:
            return (0.0, 0.0)

        mean_a = sum(group_a) / n1
        mean_b = sum(group_b) / n2
        mean_diff = mean_b - mean_a

        if SCIPY_AVAILABLE:
            var_a = np.var(group_a, ddof=1)
            var_b = np.var(group_b, ddof=1)
        else:
            var_a = sum((x - mean_a) ** 2 for x in group_a) / (n1 - 1)
            var_b = sum((x - mean_b) ** 2 for x in group_b) / (n2 - 1)

        se = math.sqrt(var_a / n1 + var_b / n2)

        # 自由度
        if var_a == 0 and var_b == 0:
            df = n1 + n2 - 2
        else:
            num = (var_a/n1 + var_b/n2) ** 2
            den = ((var_a/n1)**2)/(n1-1) + ((var_b/n2)**2)/(n2-1)
            df = num / den if den > 0 else n1 + n2 - 2

        # t临界值 (简化: 使用1.96作为95%置信度)
        if confidence == 0.95:
            if df > 120:
                t_crit = 1.96
            elif df > 60:
                t_crit = 2.00
            elif df > 30:
                t_crit = 2.04
            elif df > 10:
                t_crit = 2.23
            else:
                t_crit = 2.57
        else:
            t_crit = 1.96  # 简单回退

        margin = t_crit * se
        return (mean_diff - margin, mean_diff + margin)

    # ========================================================================
    # 样本量计算 (power analysis)
    # ========================================================================

    @classmethod
    def calculate_sample_size(
        cls,
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        significance_level: float = 0.05,
        power: float = 0.80,
        is_relative: bool = True,
    ) -> int:
        """计算每个变体所需的最小样本量

        使用公式: n = 2 * (z_alpha/2 + z_beta)^2 * sigma^2 / delta^2

        Args:
            baseline_mean: 基线均值
            baseline_std: 基线标准差
            minimum_detectable_effect: 最小可检测效应
            significance_level: 显著性水平
            power: 统计功效
            is_relative: MDE是否为相对值

        Returns:
            每个变体的最小样本量
        """
        # z值 (标准正态分布的分位数)
        z_alpha_2 = cls._norm_ppf(1.0 - significance_level / 2.0)
        z_beta = cls._norm_ppf(power)

        # 效应量 (绝对)
        delta = minimum_detectable_effect
        if is_relative:
            delta = baseline_mean * minimum_detectable_effect

        if delta == 0:
            return float('inf')

        sigma = baseline_std if baseline_std > 0 else 0.01

        n = 2 * ((z_alpha_2 + z_beta) ** 2) * (sigma ** 2) / (delta ** 2)
        return max(1, math.ceil(n))

    @staticmethod
    def _norm_ppf(p: float) -> float:
        """正态分布分位数函数的近似值"""
        if p <= 0 or p >= 1:
            return 0.0

        # 使用有理函数近似
        if p < 0.5:
            return -ABTestFramework._norm_ppf(1.0 - p)

        # Abramowitz and Stegun approximation for inverse normal
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)

    # ========================================================================
    # 序贯测试 (Sequential Testing with Early Stopping)
    # ========================================================================

    @classmethod
    def sequential_bounds(
        cls,
        num_looks: int,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """计算序贯测试的边界值 (Pocock/O'Brien-Fleming方法)

        Args:
            num_looks: 中间检查次数
            alpha: 总体显著性水平

        Returns:
            {'upper': 上界z值, 'lower': 下界z值}
        """
        # 使用Pocock方法: 每次检查使用相同的临界值
        # 调整后的alpha: alpha / (2 * num_looks)
        adjusted_alpha = alpha / (2 * num_looks)

        # 计算z临界值
        if SCIPY_AVAILABLE:
            z_crit = scipy_stats.norm.ppf(1.0 - adjusted_alpha)
        else:
            z_crit = cls._norm_ppf(1.0 - adjusted_alpha)

        return {
            'upper': z_crit,
            'lower': -z_crit,
            'adjusted_alpha_per_look': adjusted_alpha,
            'num_looks': num_looks,
        }

    @classmethod
    def sequential_test(
        cls,
        data_a_by_look: List[List[float]],
        data_b_by_look: List[List[float]],
        alpha: float = 0.05,
        early_stopping_type: str = 'pocock',
    ) -> Dict[str, Any]:
        """执行序贯测试

        Args:
            data_a_by_look: 每次检查的A组数据累积
            data_b_by_look: 每次检查的B组数据累积
            alpha: 显著性水平
            early_stopping_type: 'pocock' 或 'obrien_fleming'

        Returns:
            序贯测试结果
        """
        num_looks = len(data_a_by_look)
        boundaries = cls.sequential_bounds(num_looks, alpha)

        results = []
        stopped_early = False
        stopping_reason = ""
        stopped_at_look = -1

        for i in range(num_looks):
            a_cumulative = []
            b_cumulative = []
            for j in range(i + 1):
                a_cumulative.extend(data_a_by_look[j])
                b_cumulative.extend(data_b_by_look[j])

            if len(a_cumulative) < 2 or len(b_cumulative) < 2:
                results.append({'look': i+1, 'z_score': 0, 'action': 'continue'})
                continue

            # 计算z-score
            t_stat, p_value, df, mean_a, mean_b = cls.welch_t_test(
                a_cumulative, b_cumulative
            )

            # 将t统计量近似转换为z-score
            mean_diff = mean_b - mean_a
            n_a, n_b = len(a_cumulative), len(b_cumulative)
            var_a = sum((x - mean_a) ** 2 for x in a_cumulative) / (n_a - 1) if n_a > 1 else 0
            var_b = sum((x - mean_b) ** 2 for x in b_cumulative) / (n_b - 1) if n_b > 1 else 0

            se = math.sqrt(var_a / n_a + var_b / n_b)
            z_score = mean_diff / se if se > 0 else 0

            # 判断
            if z_score > boundaries['upper']:
                action = 'stop_winner_b'
                stopped_early = True
                stopped_at_look = i + 1
                stopping_reason = f"B显著优于A (z={z_score:.2f} > {boundaries['upper']:.2f})"
            elif z_score < boundaries['lower']:
                action = 'stop_winner_a'
                stopped_early = True
                stopped_at_look = i + 1
                stopping_reason = f"A显著优于B (z={z_score:.2f} < {boundaries['lower']:.2f})"
            else:
                action = 'continue'

            results.append({
                'look': i + 1,
                'z_score': round(z_score, 4),
                'p_value': round(p_value, 6),
                'n_a': n_a,
                'n_b': n_b,
                'mean_a': round(mean_a, 4),
                'mean_b': round(mean_b, 4),
                'action': action,
            })

            if stopped_early:
                break

        return {
            'looks_performed': len(results),
            'boundaries': boundaries,
            'results': results,
            'stopped_early': stopped_early,
            'stopped_at_look': stopped_at_look,
            'stopping_reason': stopping_reason,
            'early_stopping_type': early_stopping_type,
        }

    # ========================================================================
    # Bonferroni校正
    # ========================================================================

    @classmethod
    def bonferroni_correction(
        cls,
        p_values: List[float],
        alpha: float = 0.05,
    ) -> Tuple[List[float], float]:
        """Bonferroni校正

        alpha_adjusted = alpha / n  (其中n是比较次数)

        Args:
            p_values: 各指标的p值列表
            alpha: 原始显著性水平

        Returns:
            (调整后的p值列表, 调整后的alpha)
        """
        n = len(p_values)
        if n == 0:
            return [], alpha

        adjusted_alpha = alpha / n
        adjusted_p = [min(p * n, 1.0) for p in p_values]

        return adjusted_p, adjusted_alpha

    # ========================================================================
    # 实验管理
    # ========================================================================

    def create_experiment(self, config: ABTestConfig) -> str:
        """创建新实验

        Args:
            config: 实验配置

        Returns:
            实验ID
        """
        self.active_experiments[config.experiment_id] = config
        self.experiment_data[config.experiment_id] = {
            'A': [],
            'B': [],
        }
        return config.experiment_id

    def record_observation(
        self,
        experiment_id: str,
        user_id: str,
        metric_value: float,
        metric_name: str = "weighted_score",
    ) -> Optional[str]:
        """记录一次观测 (用户被自动分配到A/B组)

        Args:
            experiment_id: 实验ID
            user_id: 用户ID
            metric_value: 指标值
            metric_name: 指标名称

        Returns:
            分配的变体 ('A' or 'B') 如果实验不存在则返回None
        """
        if experiment_id not in self.active_experiments:
            print(f"[警告] 实验 {experiment_id} 不存在")
            return None

        config = self.active_experiments[experiment_id]
        variant = self.assign_variant(
            user_id, experiment_id,
            config.traffic_allocation_a, config.traffic_allocation_b,
        )

        if experiment_id not in self.experiment_data:
            self.experiment_data[experiment_id] = {'A': [], 'B': []}
        if metric_name not in self.experiment_data[experiment_id]:
            self.experiment_data[experiment_id][metric_name] = []

        self.experiment_data[experiment_id].setdefault(variant, [])
        self.experiment_data[experiment_id][variant].append(metric_value)

        return variant

    def analyze_experiment(
        self,
        experiment_id: str,
        primary_metric: str = "weighted_score",
        metrics: Optional[List[str]] = None,
        apply_bonferroni: bool = False,
    ) -> ABTestResult:
        """分析实验结果

        Args:
            experiment_id: 实验ID
            primary_metric: 主指标
            metrics: 要评估的所有指标
            apply_bonferroni: 是否应用Bonferroni校正

        Returns:
            ABTestResult
        """
        if experiment_id not in self.experiment_data:
            raise ValueError(f"实验 {experiment_id} 数据不存在")

        config = self.active_experiments.get(experiment_id)
        data = self.experiment_data[experiment_id]

        group_a = data.get('A', [])
        group_b = data.get('B', [])

        if not group_a or not group_b:
            return ABTestResult(
                experiment_id=experiment_id,
                status=TestResult.INCONCLUSIVE,
                primary_metric=primary_metric,
                stopping_reason="数据不足",
            )

        # 计算所有指标
        all_metrics = metrics or [primary_metric]
        per_metric = {}

        for metric in all_metrics:
            metric_data = data.get(metric, data)
            if isinstance(metric_data, dict):
                a_values = metric_data.get('A', group_a)
                b_values = metric_data.get('B', group_b)
            else:
                a_values = group_a
                b_values = group_b

            if not a_values or not b_values:
                per_metric[metric] = {
                    'p_value': 1.0,
                    'effect_size': 0.0,
                    'significant': False,
                }
                continue

            t_stat, p_val, df, mean_a, mean_b = self.welch_t_test(a_values, b_values)
            effect_size = self.cohens_d(a_values, b_values)
            ci = self.confidence_interval(a_values, b_values)

            per_metric[metric] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'degrees_of_freedom': df,
                'effect_size': effect_size,
                'confidence_interval': ci,
                'mean_a': mean_a,
                'mean_b': mean_b,
                'n_a': len(a_values),
                'n_b': len(b_values),
            }

        # 主指标结果
        primary_result = per_metric.get(primary_metric, {})

        # Bonferroni校正
        bonferroni_threshold = None
        if apply_bonferroni and len(per_metric) > 1:
            p_values = [v['p_value'] for v in per_metric.values()]
            adjusted_p, alpha_adj = self.bonferroni_correction(p_values)
            bonferroni_threshold = alpha_adj

            # 更新每个指标的显著性判断
            for (metric, result), adj_p in zip(per_metric.items(), adjusted_p):
                result['adjusted_p_value'] = adj_p
                result['significant'] = adj_p < alpha_adj

        # 判断结果
        p_val = primary_result.get('p_value', 1.0)
        threshold = bonferroni_threshold or 0.05

        if p_val < threshold:
            if primary_result.get('mean_b', 0) > primary_result.get('mean_a', 0):
                status = TestResult.SIGNIFICANT_WINNER_B
            else:
                status = TestResult.SIGNIFICANT_WINNER_A
        else:
            status = TestResult.INCONCLUSIVE

        # 相对提升
        mean_a = primary_result.get('mean_a', 0)
        mean_b = primary_result.get('mean_b', 0)
        relative_lift = (mean_b - mean_a) / abs(mean_a) if mean_a != 0 else 0

        result = ABTestResult(
            experiment_id=experiment_id,
            status=status,
            primary_metric=primary_metric,
            mean_a=round(mean_a, 4),
            std_a=round(math.sqrt(sum((x - mean_a)**2 for x in group_a) / max(1, len(group_a)-1)), 4),
            n_a=len(group_a),
            mean_b=round(mean_b, 4),
            std_b=round(math.sqrt(sum((x - mean_b)**2 for x in group_b) / max(1, len(group_b)-1)), 4),
            n_b=len(group_b),
            statistic=round(primary_result.get('t_statistic', 0), 4),
            p_value=round(p_val, 6),
            degrees_of_freedom=round(primary_result.get('degrees_of_freedom', 0), 2),
            effect_size_cohens_d=round(primary_result.get('effect_size', 0), 4),
            confidence_interval=primary_result.get('confidence_interval', (0, 0)),
            relative_lift=round(relative_lift, 4),
            bonferroni_threshold=bonferroni_threshold,
            per_metric_results=per_metric,
        )

        self.experiment_results[experiment_id] = result
        return result

    # ========================================================================
    # 报告生成
    # ========================================================================

    def generate_report(
        self,
        experiment_id: str,
        output_format: str = 'text',
    ) -> str:
        """生成实验报告

        Args:
            experiment_id: 实验ID
            output_format: 'text', 'json', 'markdown'

        Returns:
            报告字符串
        """
        result = self.experiment_results.get(experiment_id)
        if not result:
            return f"实验 {experiment_id} 尚未分析"

        config = self.active_experiments.get(experiment_id)

        if output_format == 'text':
            return self._generate_text_report(result, config)
        elif output_format == 'markdown':
            return self._generate_markdown_report(result, config)
        elif output_format == 'json':
            return self._generate_json_report(result, config)
        else:
            return self._generate_text_report(result, config)

    def _generate_text_report(
        self,
        result: ABTestResult,
        config: Optional[ABTestConfig] = None,
    ) -> str:
        """生成文本报告"""
        lines = [
            "=" * 60,
            f"  A/B Test Report: {result.experiment_id}",
            "=" * 60,
            "",
        ]

        if config:
            lines.extend([
                f"  Experiment: {config.experiment_name}",
                f"  Description: {config.description}",
                f"  Variant A: {config.variant_a_name}",
                f"  Variant B: {config.variant_b_name}",
                "",
            ])

        lines.extend([
            f"  Status: {result.status.value}",
            f"  Primary Metric: {result.primary_metric}",
            "",
            "  --- Descriptive Statistics ---",
            f"  Variant A: n={result.n_a}, mean={result.mean_a:.4f}, std={result.std_a:.4f}",
            f"  Variant B: n={result.n_b}, mean={result.mean_b:.4f}, std={result.std_b:.4f}",
            "",
            "  --- Inferential Statistics ---",
            f"  Relative Lift (B vs A): {result.relative_lift:+.2%}",
            f"  Cohen's d: {result.effect_size_cohens_d:.4f} ({result.effect_size_category()})",
            f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]",
            f"  t-statistic: {result.statistic:.4f}",
            f"  df: {result.degrees_of_freedom:.2f}",
            f"  p-value: {result.p_value:.6f}",
        ])

        if result.bonferroni_threshold:
            lines.append(f"  Bonferroni Adjusted α: {result.bonferroni_threshold:.6f}")

        lines.append(f"  Significant (α={'bonferroni' if result.bonferroni_threshold else 0.05}): {result.is_significant()}")

        if result.stopped_early:
            lines.append(f"  ⚠ Stopped Early: {result.stopping_reason}")

        # 各指标结果
        if result.per_metric_results:
            lines.extend([
                "",
                "  --- Per-Metric Results ---",
            ])
            for metric, metric_result in result.per_metric_results.items():
                sig_mark = "✓" if metric_result.get('significant', False) else "✗"
                lines.append(
                    f"  {metric}: p={metric_result.get('p_value', 1):.6f}, "
                    f"d={metric_result.get('effect_size', 0):.4f} [{sig_mark}]"
                )

        lines.extend([
            "",
            "  --- Recommendation ---",
        ])

        if result.is_significant():
            if result.relative_lift > 0:
                if result.effect_size_cohens_d > 0.5:
                    lines.append("  ✅ STRONG: 建议部署B变体")
                else:
                    lines.append("  ✅ 建议部署B变体 (效应量小，注意实际影响)")
            else:
                lines.append("  ❌ B变体效果不如A，建议保持A变体")
        elif result.relative_lift > 0 and result.p_value < 0.1:
            lines.append("  🤔 有正向趋势但未达显著水平，建议继续收集数据")
        else:
            lines.append("  ⏸ 无显著差异，建议继续实验或终止")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _generate_markdown_report(
        self,
        result: ABTestResult,
        config: Optional[ABTestConfig] = None,
    ) -> str:
        """生成Markdown报告"""
        # 使用同样的文本输出，因为已经是可读格式
        return self._generate_text_report(result, config)

    def _generate_json_report(
        self,
        result: ABTestResult,
        config: Optional[ABTestConfig] = None,
    ) -> str:
        """生成JSON报告"""
        report_data = {
            'experiment_id': result.experiment_id,
            'status': result.status.value,
            'primary_metric': result.primary_metric,
            'statistics': {
                'a': {
                    'n': result.n_a,
                    'mean': result.mean_a,
                    'std': result.std_a,
                },
                'b': {
                    'n': result.n_b,
                    'mean': result.mean_b,
                    'std': result.std_b,
                },
            },
            'test_results': {
                't_statistic': result.statistic,
                'p_value': result.p_value,
                'degrees_of_freedom': result.degrees_of_freedom,
                'effect_size_cohens_d': result.effect_size_cohens_d,
                'confidence_interval_95': list(result.confidence_interval),
                'relative_lift': result.relative_lift,
                'significant': result.is_significant(),
            },
            'bonferroni': {
                'applied': result.bonferroni_threshold is not None,
                'threshold': result.bonferroni_threshold,
            } if result.bonferroni_threshold else None,
            'per_metric': result.per_metric_results,
            'early_stopping': {
                'stopped_early': result.stopped_early,
                'reason': result.stopping_reason,
            },
        }
        return json.dumps(report_data, indent=2, ensure_ascii=False)


# ============================================================================
# 便捷入口函数
# ============================================================================

def run_ab_test(
    group_a_values: List[float],
    group_b_values: List[float],
    experiment_name: str = "quick_test",
    metrics: Optional[List[str]] = None,
) -> ABTestResult:
    """快速执行A/B测试

    Args:
        group_a_values: A组数据
        group_b_values: B组数据
        experiment_name: 实验名称
        metrics: 指标列表

    Returns:
        ABTestResult
    """
    framework = ABTestFramework()
    experiment_id = f"quick_{experiment_name}"

    config = ABTestConfig(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )
    framework.create_experiment(config)

    # 直接注入数据
    framework.experiment_data[experiment_id] = {
        'A': group_a_values,
        'B': group_b_values,
    }

    return framework.analyze_experiment(
        experiment_id=experiment_id,
        primary_metric="weighted_score",
        metrics=metrics,
    )


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  A/B测试框架 - 完整演示")
    print("=" * 60)

    # ---- 1. 分桶演示 ----
    print("\n[1] MD5哈希分桶演示")
    print("-" * 40)

    test_users = [f"user_{i}" for i in range(1000)]
    balance_check = ABTestFramework.validate_bucket_balance(
        test_users,
        experiment_id="demo_exp_001",
        traffic_a=0.5,
    )
    print(f"  总用户数: {balance_check['total_users']}")
    print(f"A/B分布: A={balance_check['a_count']}, B={balance_check['b_count']}")
    print(f"  实际A比例: {balance_check['actual_ratio_a']:.3f} (期望: {balance_check['expected_ratio_a']:.3f})")
    print(f"  平衡: {balance_check['balanced']}")

    # 单用户分桶
    for uid in ['alice', 'bob', 'charlie']:
        variant = ABTestFramework.assign_variant(uid, "demo_exp_001")
        print(f"  {uid} → 分桶 {variant}")

    # ---- 2. 统计检验演示 ----
    print("\n[2] 统计检验演示 (Welch's t-test + 效应量)")
    print("-" * 40)

    # 生成模拟数据: B组比A组好5%
    import random
    random.seed(42)

    n_samples = 500
    baseline_mean = 0.75
    treatment_effect = 0.04  # 4%提升

    group_a = [baseline_mean + random.gauss(0, 0.08) for _ in range(n_samples)]
    group_b = [baseline_mean + treatment_effect + random.gauss(0, 0.08) for _ in range(n_samples)]

    # Welch's t-test
    t_stat, p_val, df, mean_a, mean_b = ABTestFramework.welch_t_test(group_a, group_b)
    print(f"  A组: mean={mean_a:.4f}, std={math.sqrt(sum((x-mean_a)**2 for x in group_a)/(n_samples-1)):.4f}")
    print(f"  B组: mean={mean_b:.4f}, std={math.sqrt(sum((x-mean_b)**2 for x in group_b)/(n_samples-1)):.4f}")
    print(f"  t统计量: {t_stat:.4f}")
    print(f"  p值: {p_val:.6f}")
    print(f"  显著 (α=0.05): {p_val < 0.05}")

    # Cohen's d
    d = ABTestFramework.cohens_d(group_a, group_b)
    print(f"  Cohen's d: {d:.4f}")

    # 置信区间
    ci = ABTestFramework.confidence_interval(group_a, group_b)
    print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # ---- 3. 样本量计算 ----
    print("\n[3] 样本量计算 (Power Analysis)")
    print("-" * 40)

    required_n = ABTestFramework.calculate_sample_size(
        baseline_mean=0.75,
        baseline_std=0.08,
        minimum_detectable_effect=0.03,  # 3%相对提升
        significance_level=0.05,
        power=0.80,
        is_relative=True,
    )
    print(f"  基线均值: 0.75, 基线标准差: 0.08")
    print(f"  MDE: 3% (相对), α: 0.05, Power: 0.80")
    print(f"  需要最小样本量 (每变体): {required_n}")

    # ---- 4. 序贯测试 ----
    print("\n[4] 序贯测试 (Sequential Testing)")
    print("-" * 40)

    # 模拟5次中间检查
    looks = 5
    data_a_by_look = []
    data_b_by_look = []

    for i in range(looks):
        batch_size = 200
        batch_a = [baseline_mean + random.gauss(0, 0.08) for _ in range(batch_size)]
        batch_b = [baseline_mean + treatment_effect + random.gauss(0, 0.08) for _ in range(batch_size)]
        data_a_by_look.append(batch_a)
        data_b_by_look.append(batch_b)

    seq_result = ABTestFramework.sequential_test(
        data_a_by_look, data_b_by_look,
        alpha=0.05,
        early_stopping_type='pocock',
    )

    print(f"  边界z值: ±{seq_result['boundaries']['upper']:.2f}")
    print(f"  提前停止: {seq_result['stopped_early']}")
    for r in seq_result['results']:
        print(f"    Look {r['look']}: z={r['z_score']:.3f}, p={r['p_value']:.6f}, "
              f"n_a={r['n_a']}, n_b={r['n_b']}, action={r['action']}")

    # ---- 5. A/B测试完整流程 ----
    print("\n[5] A/B测试完整流程")
    print("-" * 40)

    framework = ABTestFramework()

    config = ABTestConfig(
        experiment_id="EXP_2024_001",
        experiment_name="新型检索算法 vs 基线",
        description="测试新的混合检索算法是否能提升回答质量",
        variant_a_name="BM25基线",
        variant_b_name="混合检索 (BM25 + 向量)",
        traffic_allocation_a=0.5,
        traffic_allocation_b=0.5,
        min_sample_size_per_variant=500,
        max_test_duration_days=7,
        early_stopping_enabled=True,
        early_stopping_looks=3,
        apply_bonferroni=True,
        metrics=["weighted_score", "accuracy", "relevance", "faithfulness"],
    )

    framework.create_experiment(config)
    print(f"  实验创建: {config.experiment_id}")

    # 模拟用户流量
    print(f"  模拟 {n_samples * 2} 用户流量...")
    for i in range(n_samples * 2):
        uid = f"user_{i:04d}"
        # 根据分配的变体生成不同的数据
        variant = framework.record_observation("EXP_2024_001", uid,
                                              baseline_mean + (treatment_effect if random.random() < 0.5 else 0) + random.gauss(0, 0.08))

    # 分析
    print(f"\n  分析实验结果...")
    result = framework.analyze_experiment(
        "EXP_2024_001",
        primary_metric="weighted_score",
        apply_bonferroni=True,
    )

    # 打印结果
    print(result.summary())
    print()

    # 生成报告
    text_report = framework.generate_report("EXP_2024_001", output_format='text')
    print(text_report)

    # JSON报告
    json_report = framework.generate_report("EXP_2024_001", output_format='json')
    print(f"\n  JSON Report (前500字符):")
    print(json_report[:500])

    # ---- 6. 快速测试 ----
    print("\n[6] 快速A/B测试")
    print("-" * 40)

    quick_result = run_ab_test(
        group_a_values=[0.70, 0.72, 0.75, 0.71, 0.73, 0.74, 0.72, 0.76],
        group_b_values=[0.78, 0.80, 0.77, 0.82, 0.79, 0.81, 0.80, 0.83],
        experiment_name="demo_quick_test",
    )
    print(quick_result.summary())

    print()
    print("=" * 60)
    print("  A/B测试框架演示完成")
    print("=" * 60)
