#!/usr/bin/env python3
"""
漂移检测器 (Drift Detector)
============================
监控RAG系统中的数据漂移和概念漂移:

数据漂移 (Data Drift):
  - MMD (Maximum Mean Discrepancy) - RBF核
  - 余弦距离 (分布中心)
  - KS检验 (逐维度)

概念漂移 (Concept Drift):
  - z-score: 近期窗口 vs 历史均值

健康状态:
  - CRITICAL (>5 数据漂移 或 性能退化)
  - WARNING (>2)
  - HEALTHY

中国注释，英文代码。
"""

import math
import json
import time
from typing import List, Dict, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# 数据结构
# ============================================================================

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DriftType(Enum):
    """漂移类型"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class DriftAlert:
    """漂移告警"""
    alert_id: str
    drift_type: DriftType
    dimension: str
    severity: HealthStatus
    score: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class DriftReport:
    """漂移检测报告"""
    timestamp: str
    overall_status: HealthStatus
    data_drifts: List[DriftAlert]
    concept_drifts: List[DriftAlert]
    performance_alerts: List[DriftAlert]
    summary: str = ""

    def active_alert_count(self) -> int:
        """活跃告警数量"""
        return len(self.data_drifts) + len(self.concept_drifts) + len(self.performance_alerts)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'overall_status': self.overall_status.value,
            'active_alerts': self.active_alert_count(),
            'data_drifts': [
                {'dimension': a.dimension, 'score': a.score, 'message': a.message}
                for a in self.data_drifts
            ],
            'concept_drifts': [
                {'dimension': a.dimension, 'score': a.score, 'message': a.message}
                for a in self.concept_drifts
            ],
            'performance_alerts': [
                {'dimension': a.dimension, 'score': a.score, 'message': a.message}
                for a in self.performance_alerts
            ],
        }


# ============================================================================
# 漂移检测器核心
# ============================================================================

class DriftDetector:
    """漂移检测器

    监控数据分布和模型性能的变化，及时发现漂移。
    """

    # 阈值配置
    MMD_THRESHOLD = 0.05           # MMD漂移阈值
    COSINE_THRESHOLD = 0.15        # 余弦距离阈值
    KS_PVALUE_THRESHOLD = 0.01     # KS检验p值阈值
    ZSCORE_THRESHOLD = 2.576       # z-score阈值 (99%置信)
    PERFORMANCE_DEGRADE_THRESHOLD = 0.05  # 性能退化阈值

    def __init__(
        self,
        window_size: int = 100,
        reference_size: int = 500,
        drift_history_size: int = 1000,
        alert_callback: Optional[Callable] = None,
    ):
        """初始化漂移检测器

        Args:
            window_size: 滑动窗口大小
            reference_size: 参考数据大小
            drift_history_size: 漂移历史记录大小
            alert_callback: 告警回调函数
        """
        self.window_size = window_size
        self.reference_size = reference_size
        self.alert_callback = alert_callback

        # 数据存储
        self.reference_data: Dict[str, List[float]] = defaultdict(list)
        self.current_window: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.performance_history: deque = deque(maxlen=drift_history_size)

        # 漂移历史
        self.drift_history: deque = deque(maxlen=drift_history_size)
        self.alerts_history: deque = deque(maxlen=drift_history_size)

        # 健康状态
        self.current_status = HealthStatus.HEALTHY

    # ========================================================================
    # 1. MMD (Maximum Mean Discrepancy)
    # ========================================================================

    def compute_mmd(
        self,
        reference_samples: List[float],
        current_samples: List[float],
        kernel: str = 'rbf',
        sigma: Optional[float] = None,
    ) -> float:
        """计算最大均值差异 (Maximum Mean Discrepancy)

        MMD² = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        使用RBF (高斯) 核

        Args:
            reference_samples: 参考分布样本
            current_samples: 当前分布样本
            kernel: 核函数类型 ('rbf' 或 'linear')
            sigma: RBF核带宽

        Returns:
            MMD值 (正值，越大差异越大)
        """
        if not reference_samples or not current_samples:
            return 0.0

        X = np.array(reference_samples, dtype=np.float64)
        Y = np.array(current_samples, dtype=np.float64)

        # 数据标准化到[0,1]范围
        all_data = np.concatenate([X, Y])
        data_range = np.ptp(all_data)
        if data_range > 0:
            X = X / data_range
            Y = Y / data_range

        # 如果是一维数据，重塑为二维
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if sigma is None:
            # 使用中位数启发式选择sigma
            sigma = self._median_pairwise_distance(np.concatenate([X, Y]))
            if sigma == 0:
                sigma = 1.0

        # RBF核
        gamma = 1.0 / (2.0 * sigma ** 2)

        # K_XX
        K_XX = self._rbf_kernel(X, X, gamma)
        # K_YY
        K_YY = self._rbf_kernel(Y, Y, gamma)
        # K_XY
        K_XY = self._rbf_kernel(X, Y, gamma)

        mmd2 = np.mean(K_XX) + np.mean(K_YY) - 2.0 * np.mean(K_XY)
        return max(0.0, float(mmd2))

    def _rbf_kernel(
        self,
        X: 'np.ndarray',
        Y: 'np.ndarray',
        gamma: float,
    ) -> 'np.ndarray':
        """计算RBF(高斯)核矩阵

        K(x,y) = exp(-gamma * ||x-y||²)
        """
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)  # (n, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)  # (1, m)

        # ||X-Y||² = ||X||² + ||Y||² - 2*X*Y^T
        distances = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
        distances = np.maximum(distances, 0.0)  # 避免负值

        return np.exp(-gamma * distances)

    def _median_pairwise_distance(self, X: 'np.ndarray') -> float:
        """计算样本对距离的中位数

        用于RBF核的带宽选择
        """
        if X.shape[0] < 2:
            return 1.0

        # 采样以减少计算量
        n_samples = min(X.shape[0], 200)
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sample = X[indices]

        # 计算成对距离
        X_norm = np.sum(X_sample ** 2, axis=1).reshape(-1, 1)
        Y_norm = X_norm.reshape(1, -1)
        distances = X_norm + Y_norm - 2.0 * np.dot(X_sample, X_sample.T)

        # 取上三角的非零元素
        triu_idx = np.triu_indices(n_samples, k=1)
        valid_distances = np.maximum(distances[triu_idx], 0.0)
        if len(valid_distances) == 0:
            return 1.0

        return float(np.sqrt(np.median(valid_distances)))

    # ========================================================================
    # 2. 余弦距离 (分布中心)
    # ========================================================================

    @staticmethod
    def compute_cosine_distance(
        reference_center: List[float],
        current_center: List[float],
    ) -> float:
        """计算两个分布中心的余弦距离

        cosine_distance = 1 - cosine_similarity

        Args:
            reference_center: 参考分布中心
            current_center: 当前分布中心

        Returns:
            余弦距离 (0-2, 0表示完全相同)
        """
        if NUMPY_AVAILABLE:
            ref = np.array(reference_center, dtype=np.float64)
            cur = np.array(current_center, dtype=np.float64)
        else:
            ref = list(reference_center)
            cur = list(current_center)

        # 处理零向量
        norm_ref = np.linalg.norm(ref) if NUMPY_AVAILABLE else math.sqrt(sum(x**2 for x in ref))
        norm_cur = np.linalg.norm(cur) if NUMPY_AVAILABLE else math.sqrt(sum(x**2 for x in cur))

        if norm_ref == 0 and norm_cur == 0:
            return 0.0
        if norm_ref == 0 or norm_cur == 0:
            return 1.0

        dot_product = np.dot(ref, cur) if NUMPY_AVAILABLE else sum(a*b for a,b in zip(ref, cur))
        cos_sim = dot_product / (norm_ref * norm_cur)

        # 限制在[-1, 1]范围内
        cos_sim = max(-1.0, min(1.0, cos_sim))

        return 1.0 - cos_sim

    # ========================================================================
    # 3. KS检验 (Kolmogorov-Smirnov)
    # ========================================================================

    def compute_ks_test(
        self,
        reference_samples: List[float],
        current_samples: List[float],
    ) -> Tuple[float, float]:
        """Kolmogorov-Smirnov检验 (每维度)

        Args:
            reference_samples: 参考分布样本
            current_samples: 当前分布样本

        Returns:
            (KS统计量, p值)
        """
        if SCIPY_AVAILABLE:
            statistic, p_value = scipy_stats.ks_2samp(reference_samples, current_samples)
            return float(statistic), float(p_value)
        else:
            # 从零实现两样本KS检验
            return self._ks_2samp_basic(reference_samples, current_samples)

    def _ks_2samp_basic(
        self,
        data1: List[float],
        data2: List[float],
    ) -> Tuple[float, float]:
        """从零实现两样本KS检验 (简化版)"""
        n1, n2 = len(data1), len(data2)
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0

        # 合并排序
        combined = []
        for x in data1:
            combined.append((x, 1))
        for x in data2:
            combined.append((x, 2))
        combined.sort(key=lambda x: x[0])

        # 计算经验CDF差异
        cdf1 = 0
        cdf2 = 0
        max_diff = 0.0

        for val, group in combined:
            if group == 1:
                cdf1 += 1
            else:
                cdf2 += 1

            diff = abs(cdf1 / n1 - cdf2 / n2)
            max_diff = max(max_diff, diff)

        # 简化p值计算 (使用KS分布的近似)
        n_eff = math.sqrt(n1 * n2 / (n1 + n2))
        z = max_diff * n_eff

        # 近似p值 (Kolmogorov分布)
        p_value = self._ks_pvalue_approx(z)

        return max_diff, p_value

    @staticmethod
    def _ks_pvalue_approx(z: float) -> float:
        """KS检验p值的近似计算"""
        if z < 0.01:
            return 1.0

        # Kolmogorov-Smirnov分布的渐近公式
        p = 0.0
        for k in range(1, 100):
            term = 2.0 * (-1.0)**(k-1) * math.exp(-2.0 * k**2 * z**2)
            p += term
            if abs(term) < 1e-15:
                break

        return min(1.0, max(0.0, p))

    # ========================================================================
    # 4. 概念漂移检测 (z-score)
    # ========================================================================

    def detect_concept_drift(
        self,
        metric_name: str,
        recent_window: List[float],
        historical_mean: float,
        historical_std: float,
    ) -> Tuple[bool, float, DriftAlert]:
        """检测概念漂移

        使用z-score方法:
          z = (current_mean - historical_mean) / (historical_std / sqrt(n))

        Args:
            metric_name: 指标名称
            recent_window: 最近窗口的数据
            historical_mean: 历史均值
            historical_std: 历史标准差

        Returns:
            (是否漂移, z-score, DriftAlert)
        """
        if not recent_window or historical_std == 0:
            return False, 0.0, DriftAlert(
                alert_id=f"concept_{metric_name}_{int(time.time())}",
                drift_type=DriftType.CONCEPT_DRIFT,
                dimension=metric_name,
                severity=HealthStatus.HEALTHY,
                score=0.0,
                threshold=self.ZSCORE_THRESHOLD,
                message="数据不足，无法检测",
            )

        current_mean = sum(recent_window) / len(recent_window)
        n = len(recent_window)

        z_score = (current_mean - historical_mean) / (historical_std / math.sqrt(n))

        is_drift = abs(z_score) > self.ZSCORE_THRESHOLD

        severity = HealthStatus.CRITICAL if is_drift else HealthStatus.HEALTHY

        message = (
            f"概念漂移检测: {metric_name}, "
            f"当前均值={current_mean:.4f}, 历史均值={historical_mean:.4f}, "
            f"z-score={z_score:.2f}"
        )

        alert = DriftAlert(
            alert_id=f"concept_{metric_name}_{int(time.time())}",
            drift_type=DriftType.CONCEPT_DRIFT,
            dimension=metric_name,
            severity=severity,
            score=abs(z_score),
            threshold=self.ZSCORE_THRESHOLD,
            details={
                'current_mean': current_mean,
                'historical_mean': historical_mean,
                'historical_std': historical_std,
                'z_score': z_score,
                'n': n,
            },
            message=message,
        )

        return is_drift, z_score, alert

    # ========================================================================
    # 综合漂移检测
    # ========================================================================

    def check_data_drift(
        self,
        dimension_name: str,
        current_samples: List[float],
    ) -> List[DriftAlert]:
        """检查单个维度的数据漂移

        使用三种方法:
          1. MMD (RBF核)
          2. 余弦距离 (分布中心)
          3. KS检验

        Args:
            dimension_name: 维度名称
            current_samples: 当前窗口的样本

        Returns:
            DriftAlert列表
        """
        alerts = []
        reference = self.reference_data.get(dimension_name, [])

        if not reference or not current_samples:
            return alerts

        if len(reference) < 10 or len(current_samples) < 10:
            return alerts

        # 1. MMD检验
        mmd_score = self.compute_mmd(reference, current_samples)
        if mmd_score > self.MMD_THRESHOLD:
            alerts.append(DriftAlert(
                alert_id=f"mmd_{dimension_name}_{int(time.time())}",
                drift_type=DriftType.DATA_DRIFT,
                dimension=dimension_name,
                severity=HealthStatus.WARNING,
                score=mmd_score,
                threshold=self.MMD_THRESHOLD,
                details={'method': 'MMD', 'kernel': 'rbf'},
                message=f"MMD漂移: {dimension_name} (score={mmd_score:.4f})",
            ))

        # 2. 余弦距离 (分布中心)
        ref_center = [sum(reference) / len(reference)]
        cur_center = [sum(current_samples) / len(current_samples)]
        cos_dist = self.compute_cosine_distance(ref_center, cur_center)
        if cos_dist > self.COSINE_THRESHOLD:
            alerts.append(DriftAlert(
                alert_id=f"cosine_{dimension_name}_{int(time.time())}",
                drift_type=DriftType.DATA_DRIFT,
                dimension=dimension_name,
                severity=HealthStatus.WARNING,
                score=cos_dist,
                threshold=self.COSINE_THRESHOLD,
                details={'method': 'cosine_distance'},
                message=f"分布中心偏移: {dimension_name} (cos_dist={cos_dist:.4f})",
            ))

        # 3. KS检验
        ks_stat, p_value = self.compute_ks_test(reference, current_samples)
        if p_value < self.KS_PVALUE_THRESHOLD:
            alerts.append(DriftAlert(
                alert_id=f"ks_{dimension_name}_{int(time.time())}",
                drift_type=DriftType.DATA_DRIFT,
                dimension=dimension_name,
                severity=HealthStatus.CRITICAL,
                score=ks_stat,
                threshold=self.KS_PVALUE_THRESHOLD,
                details={'method': 'KS', 'statistic': ks_stat, 'p_value': p_value},
                message=f"KS检验: {dimension_name} (D={ks_stat:.4f}, p={p_value:.6f})",
            ))

        return alerts

    def update_and_check(
        self,
        dimension_values: Dict[str, float],
        performance_score: Optional[float] = None,
    ) -> DriftReport:
        """更新观察值并检查漂移

        主入口: 一次性完成数据更新和漂移检查

        Args:
            dimension_values: {维度名: 值} 的字典
            performance_score: 性能分数 (可选)

        Returns:
            DriftReport
        """
        all_alerts: List[DriftAlert] = []
        data_alerts: List[DriftAlert] = []
        concept_alerts: List[DriftAlert] = []
        perf_alerts: List[DriftAlert] = []

        # 1. 更新当前窗口
        for dim, value in dimension_values.items():
            self.current_window[dim].append(value)

        # 2. 数据漂移检查
        for dim in dimension_values.keys():
            window_data = list(self.current_window[dim])
            if len(window_data) >= self.window_size:
                dim_alerts = self.check_data_drift(dim, window_data)
                data_alerts.extend(dim_alerts)
                all_alerts.extend(dim_alerts)

        # 3. 概念漂移检查
        for dim in dimension_values.keys():
            window_data = list(self.current_window[dim])
            reference_data = self.reference_data.get(dim, [])

            if len(reference_data) >= 30 and len(window_data) >= 10:
                if NUMPY_AVAILABLE:
                    hist_mean = float(np.mean(reference_data))
                    hist_std = float(np.std(reference_data, ddof=1))
                else:
                    hist_mean = sum(reference_data) / len(reference_data)
                    hist_std = math.sqrt(
                        sum((x - hist_mean)**2 for x in reference_data) / (len(reference_data) - 1)
                    ) if len(reference_data) > 1 else 0.0

                if hist_std > 0:
                    is_drift, z_score, alert = self.detect_concept_drift(
                        dim, window_data, hist_mean, hist_std
                    )
                    if is_drift:
                        concept_alerts.append(alert)
                        all_alerts.append(alert)

        # 4. 性能退化检查
        if performance_score is not None:
            self.performance_history.append(performance_score)
            if len(self.performance_history) >= 10:
                perf_alerts = self._check_performance_degradation()
                all_alerts.extend(perf_alerts)

        # 5. 确定整体健康状态
        data_drift_count = len(data_alerts)
        perf_degrade = any(
            a.drift_type == DriftType.PERFORMANCE_DEGRADATION for a in perf_alerts
        )

        if data_drift_count > 5 or perf_degrade:
            self.current_status = HealthStatus.CRITICAL
        elif data_drift_count > 2:
            self.current_status = HealthStatus.WARNING
        else:
            self.current_status = HealthStatus.HEALTHY

        # 6. 记录历史
        for alert in data_alerts:
            self.drift_history.append({
                'type': 'data_drift',
                'dimension': alert.dimension,
                'score': alert.score,
                'timestamp': alert.timestamp,
            })

        for alert in concept_alerts:
            self.drift_history.append({
                'type': 'concept_drift',
                'dimension': alert.dimension,
                'score': alert.score,
                'timestamp': alert.timestamp,
            })

        # 7. 触发告警回调
        if self.alert_callback and all_alerts:
            for alert in all_alerts:
                if alert.severity in (HealthStatus.WARNING, HealthStatus.CRITICAL):
                    try:
                        self.alert_callback(alert)
                    except Exception as e:
                        print(f"[警告] 告警回调失败: {e}")

        # 8. 生成报告
        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            overall_status=self.current_status,
            data_drifts=data_alerts,
            concept_drifts=concept_alerts,
            performance_alerts=perf_alerts,
            summary=self._generate_summary(data_alerts, concept_alerts, perf_alerts),
        )

        return report

    def _check_performance_degradation(self) -> List[DriftAlert]:
        """检查性能退化"""
        alerts = []
        recent = list(self.performance_history)[-20:]  # 最近20个观测

        if len(recent) < 10:
            return alerts

        # 比较前10和后10
        first_half = recent[:10]
        second_half = recent[10:]

        mean_start = sum(first_half) / len(first_half)
        mean_end = sum(second_half) / len(second_half)

        if mean_start > 0:
            degradation = (mean_start - mean_end) / mean_start
            if degradation > self.PERFORMANCE_DEGRADE_THRESHOLD:
                alerts.append(DriftAlert(
                    alert_id=f"perf_degrade_{int(time.time())}",
                    drift_type=DriftType.PERFORMANCE_DEGRADATION,
                    dimension="overall_performance",
                    severity=HealthStatus.CRITICAL,
                    score=degradation,
                    threshold=self.PERFORMANCE_DEGRADE_THRESHOLD,
                    details={
                        'mean_before': mean_start,
                        'mean_after': mean_end,
                        'degradation_pct': degradation,
                    },
                    message=f"性能退化: {degradation:.1%} (从{mean_start:.4f}到{mean_end:.4f})",
                ))

        return alerts

    def set_reference(
        self,
        dimension_name: str,
        reference_samples: List[float],
    ) -> None:
        """设置参考基线数据

        Args:
            dimension_name: 维度名称
            reference_samples: 参考样本
        """
        self.reference_data[dimension_name] = reference_samples[:self.reference_size]
        print(f"[参考基线] {dimension_name}: {len(self.reference_data[dimension_name])} 样本")

    def _generate_summary(
        self,
        data_alerts: List[DriftAlert],
        concept_alerts: List[DriftAlert],
        perf_alerts: List[DriftAlert],
    ) -> str:
        """生成漂移报告摘要"""
        parts = []
        if data_alerts:
            parts.append(f"{len(data_alerts)} 数据漂移告警")
        if concept_alerts:
            parts.append(f"{len(concept_alerts)} 概念漂移告警")
        if perf_alerts:
            parts.append(f"{len(perf_alerts)} 性能退化告警")

        if not parts:
            return "无漂移检测到，系统健康"

        summary = "；".join(parts)
        summary += f" | 整体状态: {self.current_status.value}"
        return summary

    def get_status(self) -> HealthStatus:
        """获取当前健康状态"""
        return self.current_status


# ============================================================================
# 便捷工具
# ============================================================================

def simulate_drift_data(
    n_samples: int = 200,
    drift_point: int = 100,  # 漂移开始的位置
    drift_magnitude: float = 0.05,
    baseline_mean: float = 0.75,
    noise_std: float = 0.05,
    seed: int = 42,
) -> List[float]:
    """模拟带漂移的数据

    Args:
        n_samples: 总样本数
        drift_point: 漂移开始点
        drift_magnitude: 漂移幅度
        baseline_mean: 基线均值
        noise_std: 噪声标准差
        seed: 随机种子

    Returns:
        带漂移的时序数据
    """
    import random
    random.seed(seed)

    data = []
    for i in range(n_samples):
        if i < drift_point:
            value = baseline_mean + random.gauss(0, noise_std)
        else:
            # 线性漂移
            drift = drift_magnitude * min(1.0, (i - drift_point) / 50)
            value = baseline_mean - drift + random.gauss(0, noise_std)
        data.append(value)

    return data


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  漂移检测器 - 完整演示")
    print("=" * 60)

    # ---- 初始化检测器 ----
    detector = DriftDetector(
        window_size=50,
        reference_size=200,
        alert_callback=lambda alert: print(f"  🔔 告警: {alert.message}"),
    )

    # ---- 设置参考基线 ----
    print("\n[1] 设置参考基线数据")
    print("-" * 40)

    # 模拟历史基线数据 (无漂移)
    import random
    random.seed(123)

    for dim in ['accuracy', 'relevance', 'faithfulness', 'fluency', 'latency', 'cost']:
        if dim == 'latency':
            reference = [max(0.5, random.gauss(2.0, 0.5)) for _ in range(200)]
        elif dim == 'cost':
            reference = [max(0.001, random.gauss(0.02, 0.005)) for _ in range(200)]
        else:
            reference = [max(0.0, min(1.0, random.gauss(0.75, 0.08))) for _ in range(200)]
        detector.set_reference(dim, reference)
        print(f"  设置 {dim}: {len(reference)} 样本, mean={sum(reference)/len(reference):.4f}")

    # ---- 模拟正常数据 ----
    print("\n[2] 注入正常数据 (50个样本)")
    print("-" * 40)

    for i in range(50):
        dims = {
            'accuracy': max(0.0, min(1.0, random.gauss(0.75, 0.08))),
            'relevance': max(0.0, min(1.0, random.gauss(0.80, 0.06))),
            'faithfulness': max(0.0, min(1.0, random.gauss(0.72, 0.10))),
            'fluency': max(0.0, min(1.0, random.gauss(0.85, 0.05))),
            'latency': max(0.5, random.gauss(2.0, 0.5)),
            'cost': max(0.001, random.gauss(0.02, 0.005)),
        }
        perf = 0.78 + random.gauss(0, 0.02)

        report = detector.update_and_check(dims, perf)

    print(f"  窗口填充完成")
    print(f"  健康状态: {detector.get_status().value}")

    # ---- 模拟数据漂移 ----
    print("\n[3] 注入漂移数据 (50个样本)")
    print("-" * 40)

    for i in range(50):
        # accuracy 和 faithfulness 发生漂移
        dims = {
            'accuracy': max(0.0, min(1.0, random.gauss(0.60, 0.10))),  # 从0.75降到0.60
            'relevance': max(0.0, min(1.0, random.gauss(0.80, 0.06))),  # 正常
            'faithfulness': max(0.0, min(1.0, random.gauss(0.55, 0.12))),  # 从0.72降到0.55
            'fluency': max(0.0, min(1.0, random.gauss(0.85, 0.05))),  # 正常
            'latency': max(0.5, random.gauss(4.0, 1.5)),  # 延迟增加
            'cost': max(0.001, random.gauss(0.02, 0.005)),  # 正常
        }
        perf = 0.72 + random.gauss(0, 0.03)  # 性能下降

        report = detector.update_and_check(dims, perf)

    print(f"  漂移数据注入完成")
    print(f"  健康状态: {detector.get_current_status.value}")

    # ---- 打印活跃漂移 ----
    print("\n[4] 当前漂移告警总结")
    print("-" * 40)

    print(f"  整体状态: {report.overall_status.value}")
    print(f"  活跃告警数: {report.active_alert_count()}")
    print(f"  数据漂移: {len(report.data_drifts)}")
    for alert in report.data_drifts:
        print(f"    - {alert.message}")

    if report.concept_drifts:
        print(f"  概念漂移: {len(report.concept_drifts)}")
        for alert in report.concept_drifts:
            print(f"    - {alert.message}")

    if report.performance_alerts:
        print(f"  性能退化: {len(report.performance_alerts)}")
        for alert in report.performance_alerts:
            print(f"    - {alert.message}")

    print(f"\n  摘要: {report.summary}")

    # ---- MMD独立演示 ----
    print("\n[5] MMD独立演示")
    print("-" * 40)

    ref_samples = [random.gauss(0.75, 0.08) for _ in range(100)]
    same_dist_samples = [random.gauss(0.75, 0.08) for _ in range(100)]
    drifted_samples = [random.gauss(0.65, 0.10) for _ in range(100)]

    mmd_same = detector.compute_mmd(ref_samples, same_dist_samples)
    mmd_drift = detector.compute_mmd(ref_samples, drifted_samples)

    print(f"  同分布 MMD: {mmd_same:.6f} (期望接近0)")
    print(f"  不同分布 MMD: {mmd_drift:.6f} (期望>阈值)")

    # ---- KS检验独立演示 ----
    print("\n[6] KS检验独立演示")
    print("-" * 40)

    ks_stat, p_value = detector.compute_ks_test(ref_samples, drifted_samples)
    print(f"  KS统计量: {ks_stat:.4f}")
    print(f"  p值: {p_value:.6f}")
    print(f"  漂移检测: {p_value < 0.01}")

    # ---- 概念漂移演示 ----
    print("\n[7] 概念漂移演示")
    print("-" * 40)

    historical_perf = [0.78, 0.79, 0.77, 0.80, 0.78, 0.79, 0.77, 0.78, 0.79, 0.80]
    recent_perf = [0.72, 0.71, 0.73, 0.70, 0.72]  # 明显下降

    hist_mean = sum(historical_perf) / len(historical_perf)
    hist_std = math.sqrt(
        sum((x - hist_mean)**2 for x in historical_perf) / (len(historical_perf) - 1)
    )

    is_drift, z_score, alert = detector.detect_concept_drift(
        "overall_performance", recent_perf, hist_mean, hist_std
    )

    print(f"  历史均值: {hist_mean:.4f}, 历史std: {hist_std:.4f}")
    print(f"  最近均值: {sum(recent_perf)/len(recent_perf):.4f}")
    print(f"  z-score: {z_score:.2f}")
    print(f"  漂移检测: {is_drift}")

    # ---- 健康状态总结 ----
    print("\n" + "=" * 50)
    print(f"  最终健康状态: {detector.get_current_status.value}")
    print("=" * 50)

    # ---- 导出探测报告 ----
    report_dict = report.to_dict()
    print(f"\n  报告JSON (简化):")
    print(f"  {json.dumps(report_dict, ensure_ascii=False, indent=2)[:600]}")

    print()
    print("=" * 60)
    print("  漂移检测器演示完成")
    print("=" * 60)
