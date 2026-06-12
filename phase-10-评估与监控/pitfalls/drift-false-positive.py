#!/usr/bin/env python3
"""
陷阱4: 漂移误报 (Drift False Positive)
=========================================
症状: 漂移检测频繁触发告警，但系统实际运转正常
根因:
  1. 阈值设置过严 (过于敏感)
  2. 正常的周期性波动被误判为漂移
  3. 样本量波动导致的检测不稳定
  4. 未考虑多指标比较的多重性问题

解决方案:
  - 使用自适应阈值 (基于历史波动范围)
  - 区分周期性和结构性变化
  - 使用EWMA平滑检测
  - 漂移确认机制 (连续N次检测到才告警)
"""

import math
import random
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta


@dataclass
class AdaptiveDriftDetector:
    """自适应漂移检测器 - 减少误报"""

    # 自适应参数
    base_threshold: float = 2.0  # 基础z-score阈值
    sensitivity: float = 0.2     # 学习率 (用于阈值自适应)

    # 历史数据
    historical_means: deque = field(default_factory=lambda: deque(maxlen=100))
    historical_stds: deque = field(default_factory=lambda: deque(maxlen=100))
    seasonal_patterns: Dict[str, List[float]] = field(default_factory=dict)

    # 确认窗口
    confirmation_window: int = 3  # 连续N次才确认漂移
    _pending_alerts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def detect(
        self,
        metric_name: str,
        recent_values: List[float],
        historical_values: List[float],
    ) -> Dict[str, Any]:
        """自适应漂移检测

        Steps:
        1. 计算自适应阈值
        2. 考虑周期性模式
        3. 应用确认窗口
        4. 使用EWMA平滑

        Args:
            metric_name: 指标名称
            recent_values: 近期值
            historical_values: 历史值

        Returns:
            检测结果
        """
        if not recent_values or len(historical_values) < 10:
            return {'drift_detected': False, 'reason': '数据不足'}

        # 1. 计算自适应阈值
        hist_mean = sum(historical_values) / len(historical_values)
        hist_std = self._calculate_std(historical_values)

        # 更新历史统计
        self.historical_means.append(hist_mean)
        self.historical_stds.append(hist_std)

        # 自适应阈值: 基于历史波动范围
        if len(self.historical_stds) >= 5:
            median_std = sorted(self.historical_stds)[len(self.historical_stds) // 2]
            adaptive_threshold = self.base_threshold * (1 + self.sensitivity * median_std)
        else:
            adaptive_threshold = self.base_threshold

        # 2. 使用EWMA平滑的当前均值
        recent_ewma = self._ewma(recent_values, alpha=0.3)
        recent_std = self._calculate_std(recent_values)

        # 3. 计算z-score (使用平滑后的值)
        if hist_std > 0:
            z_score = abs(recent_ewma - hist_mean) / (hist_std / math.sqrt(len(recent_values)))
        else:
            z_score = 0.0

        # 4. 检查周期性 (检查是否是正常的周期性波动)
        is_seasonal = self._check_seasonality(metric_name, recent_ewma, hist_mean, hist_std)

        # 5. 决定是否告警
        if z_score > adaptive_threshold and not is_seasonal:
            self._pending_alerts[metric_name] += 1

            # 确认窗口
            if self._pending_alerts[metric_name] >= self.confirmation_window:
                self._pending_alerts[metric_name] = 0
                return {
                    'drift_detected': True,
                    'z_score': round(z_score, 2),
                    'threshold': round(adaptive_threshold, 2),
                    'is_seasonal': is_seasonal,
                    'ewma_current': round(recent_ewma, 4),
                    'historical_mean': round(hist_mean, 4),
                    'severity': 'CRITICAL' if z_score > adaptive_threshold * 1.5 else 'WARNING',
                    'confirmation': 'confirmed',
                }
            else:
                return {
                    'drift_detected': False,
                    'z_score': round(z_score, 2),
                    'pending_count': self._pending_alerts[metric_name],
                    'status': f'pending ({self._pending_alerts[metric_name]}/{self.confirmation_window})',
                }
        else:
            # 重置待确认计数
            self._pending_alerts[metric_name] = max(0, self._pending_alerts[metric_name] - 1)

            return {
                'drift_detected': False,
                'z_score': round(z_score, 2),
                'threshold': round(adaptive_threshold, 2),
                'is_seasonal': is_seasonal,
                'status': 'normal',
                'reason': 'seasonal pattern' if is_seasonal else 'within normal range',
            }

    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _ewma(self, values: List[float], alpha: float = 0.3) -> float:
        """指数加权移动平均"""
        if not values:
            return 0.0
        ewma = values[0]
        for v in values[1:]:
            ewma = alpha * v + (1 - alpha) * ewma
        return ewma

    def _check_seasonality(
        self,
        metric_name: str,
        current_value: float,
        hist_mean: float,
        hist_std: float,
    ) -> bool:
        """检查周期性模式

        简化的周期检测: 检查是否在历史波动范围内
        """
        if hist_std == 0:
            return False

        # 使用3-sigma规则: 如果当前值在3个标准差内，可能是正常波动
        deviation = abs(current_value - hist_mean)
        return deviation <= 3 * hist_std


class FalsePositiveCalibrator:
    """误报率校准器"""

    @staticmethod
    def calibrate_threshold(
        historical_normal_data: List[List[float]],
        target_fpr: float = 0.01,  # 目标误报率 1%
    ) -> float:
        """基于历史正常数据校准阈值

        Args:
            historical_normal_data: 历史正常时期的数据 (多个维度)
            target_fpr: 目标误报率

        Returns:
            校准后的阈值
        """
        # 在不同阈值下测试误报率
        z_scores = []

        for dimension_data in historical_normal_data:
            if len(dimension_data) < 20:
                continue

            # 使用前一半作为参考，后一半作为测试
            split = len(dimension_data) // 2
            reference = dimension_data[:split]
            test = dimension_data[split:]

            ref_mean = sum(reference) / len(reference)
            ref_std = math.sqrt(
                sum((x - ref_mean) ** 2 for x in reference) / (len(reference) - 1)
            ) if len(reference) > 1 else 0.0

            if ref_std == 0:
                continue

            # 计算所有测试点的z-score
            for val in test:
                z = abs(val - ref_mean) / (ref_std / math.sqrt(len(reference)))
                z_scores.append(z)

        if not z_scores:
            return 2.0  # 默认阈值

        # 找到满足目标误报率的阈值
        z_scores.sort()
        threshold_idx = int(len(z_scores) * (1 - target_fpr))
        calibrated_threshold = z_scores[min(threshold_idx, len(z_scores) - 1)]

        return max(2.0, calibrated_threshold)  # 不低于2.0


SOLUTION_CHECKLIST = """
漂移误报减少检查清单:

□ 1. 使用自适应阈值而非固定阈值
□ 2. 考虑周期性波动 (周末效应、流量高峰)
□ 3. 使用确认窗口 (连续N次检测到才告警)
□ 4. 应用EWMA平滑减少噪声
□ 5. 定期校准阈值 (基于历史误报率)
□ 6. 区分不同严重级别的告警
□ 7. 记录误报率并持续优化
"""


if __name__ == "__main__":
    print("=" * 60)
    print("  漂移误报减少 - 演示")
    print("=" * 60)

    detector = AdaptiveDriftDetector()

    # 模拟正常波动
    print("\n[1] 正常波动检测 (不应漂移)")
    normal_history = [0.75 + random.gauss(0, 0.05) for _ in range(100)]

    for i in range(10):
        recent = [0.75 + random.gauss(0, 0.06) for _ in range(20)]
        result = detector.detect("accuracy", recent, normal_history)
        drift_status = '🚨 漂移!' if result['drift_detected'] else '✓ 正常'
        print(f"  检查{i+1}: {drift_status} z={result.get('z_score', 0):.2f} "
              f"阈值={result.get('threshold', 2.0):.2f}")

    # 模拟真实漂移
    print("\n[2] 真实漂移检测 (应该漂移)")
    for i in range(10):
        recent = [0.55 + random.gauss(0, 0.06) for _ in range(20)]  # 明显下降
        result = detector.detect("accuracy", recent, normal_history)
        drift_status = '🚨 漂移!' if result['drift_detected'] else '✓ 正常'
        confirm = result.get('pending_count', 0)
        print(f"  检查{i+1}: {drift_status} z={result.get('z_score', 0):.2f} "
              f"确认={confirm}/{detector.confirmation_window}")

    # 阈值校准
    print("\n[3] 阈值校准")
    calibrator = FalsePositiveCalibrator()

    # 生成多维度的正常历史数据
    normal_data = [
        [0.75 + random.gauss(0, 0.05) for _ in range(200)]  # accuracy
        for _ in range(5)
    ]

    calibrated = calibrator.calibrate_threshold(normal_data, target_fpr=0.01)
    print(f"  校准后阈值: {calibrated:.2f} (目标误报率: 1%)")
    print(f"  默认阈值: 2.0")
    print(f"  相比默认阈值: {'更严格' if calibrated > 2.0 else '更宽松'}")

    print(f"\n{SOLUTION_CHECKLIST}")
