#!/usr/bin/env python3
"""
陷阱2: 小样本偏差 (Small Sample Bias)
=======================================
症状: A/B测试结果不稳定，相同的实验重复得到不同结论
根因:
  1. 样本量不足导致统计功效低
  2. 小样本下的异常值影响大
  3. 置信区间宽但被忽视
  4. 过早停止实验 (peeking)

解决方案:
  - 提前计算所需样本量
  - 使用bootstrap重采样评估稳定性
  - 报告置信区间而不仅是点估计
  - 使用序贯测试控制多次检查的错误率
"""

import math
import random
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SampleSizeAnalyzer:
    """样本量分析器"""

    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    mde: float = 0.02  # 最小可检测效应
    alpha: float = 0.05
    power: float = 0.80

    def calculate_required_n(self) -> int:
        """计算所需的最小样本量

        n = 2 * (z_alpha/2 + z_beta)^2 * sigma^2 / delta^2
        """
        z_alpha_2 = self._norm_ppf(1 - self.alpha / 2)
        z_beta = self._norm_ppf(self.power)

        delta = self.mde  # 绝对效应量

        if delta == 0:
            return float('inf')

        sigma = max(self.baseline_std, 0.01)
        n = 2 * ((z_alpha_2 + z_beta) ** 2) * (sigma ** 2) / (delta ** 2)
        return math.ceil(n)

    @staticmethod
    def _norm_ppf(p: float) -> float:
        """正态分布分位数 (近似)"""
        if p <= 0 or p >= 1:
            return 0.0
        if p < 0.5:
            return -SampleSizeAnalyzer._norm_ppf(1 - p)

        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)

    def power_analysis_table(self) -> List[Dict[str, Any]]:
        """生成功效分析表: 不同样本量下的统计功效"""
        required_n = self.calculate_required_n()
        table = []

        sample_sizes = [
            required_n // 4,
            required_n // 2,
            required_n,
            required_n * 2,
        ]

        for n in sample_sizes:
            if n <= 0:
                continue

            delta = self.mde
            sigma = max(self.baseline_std, 0.01)

            # 计算给定样本量下的功效
            z_alpha_2 = self._norm_ppf(1 - self.alpha / 2)
            ncp = delta / (sigma * math.sqrt(2 / n))  # 非中心参数

            # 简化的功效计算
            power_approx = self._norm_cdf(ncp - z_alpha_2)

            table.append({
                'n_per_variant': n,
                'total_n': n * 2,
                'power': round(max(0, min(1, power_approx)), 4),
                'sufficient': n >= required_n,
            })

        return table

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """正态分布CDF (近似)"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class BootstrapAnalyzer:
    """Bootstrap重采样分析器 - 评估小样本下的估计稳定性"""

    @staticmethod
    def bootstrap_ci(
        data: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Bootstrap置信区间

        Args:
            data: 样本数据
            n_bootstrap: 重采样次数
            confidence: 置信水平

        Returns:
            {
                'original_mean': 原始均值,
                'bootstrap_mean': bootstrap均值,
                'bootstrap_std': bootstrap标准差,
                'ci_lower': 置信区间下界,
                'ci_upper': 置信区间上界,
                'ci_width': 区间宽度,
                'stable': 估计是否稳定,
            }
        """
        n = len(data)
        if n < 3:
            return {'stable': False, 'reason': f'样本量太小 (n={n})'}

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = [random.choice(data) for _ in range(n)]
            bootstrap_means.append(sum(sample) / n)

        bootstrap_means.sort()

        original_mean = sum(data) / n
        bootstrap_mean_of_means = sum(bootstrap_means) / len(bootstrap_means)

        # 计算bootstrap标准差
        bootstrap_std = math.sqrt(
            sum((x - bootstrap_mean_of_means) ** 2 for x in bootstrap_means)
            / (len(bootstrap_means) - 1)
        )

        # 百分位置信区间
        alpha = (1 - confidence) / 2
        lower_idx = int(alpha * n_bootstrap)
        upper_idx = int((1 - alpha) * n_bootstrap)

        ci_lower = bootstrap_means[lower_idx]
        ci_upper = bootstrap_means[upper_idx]
        ci_width = ci_upper - ci_lower

        # 稳定性判断: CI宽度小于均值的20%
        stable = ci_width < abs(original_mean) * 0.2 if original_mean != 0 else ci_width < 0.1

        return {
            'original_mean': round(original_mean, 4),
            'bootstrap_mean': round(bootstrap_mean_of_means, 4),
            'bootstrap_std': round(bootstrap_std, 4),
            'ci_lower': round(ci_lower, 4),
            'ci_upper': round(ci_upper, 4),
            'ci_width': round(ci_width, 4),
            'confidence': confidence,
            'stable': stable,
            'n_samples': n,
            'n_bootstrap': n_bootstrap,
        }

    @staticmethod
    def check_stability(data: List[float]) -> Dict[str, Any]:
        """检查小样本下估计的稳定性

        方法: 比较不同子样本大小的估计值
        """
        n = len(data)
        if n < 30:
            return {
                'stable': False,
                'warning': f'样本量不足 (n={n}<30)，估计可能不稳定',
                'recommendation': f'建议至少收集30个样本，当前仅{n}个',
            }

        # 用不同大小的子样本测试稳定性
        sample_sizes = [10, 20, 30, min(50, n), n]
        estimates = []

        for size in sample_sizes:
            if size <= n:
                subset = random.sample(data, size)
                mean_est = sum(subset) / len(subset)
                estimates.append({'size': size, 'mean': round(mean_est, 4)})

        # 检查收敛性
        if len(estimates) >= 2:
            last_two = estimates[-2:]
            convergence = abs(last_two[1]['mean'] - last_two[0]['mean'])
        else:
            convergence = float('inf')

        stable = convergence < 0.02  # 最后两个估计差异小于2%

        return {
            'stable': stable,
            'convergence': round(convergence, 4),
            'subsample_estimates': estimates,
            'recommendation': (
                '估计已稳定' if stable
                else f'估计尚未稳定，建议增加到50+样本'
            ),
        }


def check_early_stopping_risk(
    peek_times: int,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """检查频繁"偷看"实验结果的假阳性风险

    Args:
        peek_times: 偷看（中间检查）次数
        alpha: 名义显著性水平

    Returns:
        实际错误率等
    """
    # 多次偷看的实际假阳性率
    actual_alpha = 1 - (1 - alpha) ** peek_times

    # Bonferroni校正
    bonferroni_alpha = alpha / peek_times

    return {
        'nominal_alpha': alpha,
        'peek_times': peek_times,
        'actual_false_positive_rate': round(actual_alpha, 6),
        'bonferroni_corrected_alpha': round(bonferroni_alpha, 6),
        'warning': (
            '安全' if peek_times <= 1
            else f'偷看{peek_times}次使假阳性率从{alpha*100:.1f}%增加到{actual_alpha*100:.1f}%!'
        ),
    }


# 解决方案检查清单
SOLUTION_CHECKLIST = """
小样本偏差避免清单:

□ 1. 实验前计算所需样本量 (power analysis)
□ 2. 使用Bootstrap评估估计稳定性
□ 3. 总是报告置信区间而非仅点估计
□ 4. 不使用"偷看"结果做出最终决策
□ 5. 对于多次中间检查，使用Bonferroni或序贯测试
□ 6. 小样本实验只作探索性分析
□ 7. 关键决策要求 n >= 100 每变体
"""


if __name__ == "__main__":
    print("=" * 60)
    print("  小样本偏差检测 - 演示")
    print("=" * 60)

    # 1. 样本量计算
    print("\n[1] 功效分析 - 所需样本量")
    analyzer = SampleSizeAnalyzer(
        baseline_mean=0.75,
        baseline_std=0.08,
        mde=0.03,
        alpha=0.05,
        power=0.80,
    )

    required_n = analyzer.calculate_required_n()
    print(f"  检测3%提升 (MDE=0.03):")
    print(f"  需要每变体 {required_n} 个样本")

    print(f"\n  功效分析表:")
    for row in analyzer.power_analysis_table():
        status = '✓ 足够' if row['sufficient'] else '✗ 不足'
        print(f"    n={row['n_per_variant']:>6d} → 功效={row['power']:.3f} {status}")

    # 2. Bootstrap分析
    print(f"\n[2] Bootstrap稳定性分析")

    # 小样本
    small_sample = [random.gauss(0.75, 0.08) for _ in range(15)]
    bootstrap_result = BootstrapAnalyzer.bootstrap_ci(small_sample)
    print(f"  小样本 (n=15):")
    print(f"    均值: {bootstrap_result['original_mean']:.4f}")
    print(f"    95% CI: [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
    print(f"    CI宽度: {bootstrap_result['ci_width']:.4f}")
    print(f"    稳定: {bootstrap_result['stable']}")

    # 大样本
    large_sample = [random.gauss(0.75, 0.08) for _ in range(200)]
    bootstrap_large = BootstrapAnalyzer.bootstrap_ci(large_sample)
    print(f"\n  大样本 (n=200):")
    print(f"    均值: {bootstrap_large['original_mean']:.4f}")
    print(f"    95% CI: [{bootstrap_large['ci_lower']:.4f}, {bootstrap_large['ci_upper']:.4f}]")
    print(f"    CI宽度: {bootstrap_large['ci_width']:.4f}")
    print(f"    稳定: {bootstrap_large['stable']}")

    # 3. 偷看风险
    print(f"\n[3] 偷看实验风险")
    for peeks in [1, 3, 5, 10]:
        risk = check_early_stopping_risk(peeks)
        print(f"  偷看{peeks}次: 实际假阳性率 {risk['actual_false_positive_rate']:.4f} "
              f"(名义{risk['nominal_alpha']})")

    # 4. 收敛性检查
    print(f"\n[4] 收敛性检查")
    stability = BootstrapAnalyzer.check_stability(small_sample)
    print(f"  小样本: {stability['recommendation']}")

    stability_large = BootstrapAnalyzer.check_stability(large_sample)
    print(f"  大样本: {stability_large['recommendation']}")

    print(f"\n{SOLUTION_CHECKLIST}")
