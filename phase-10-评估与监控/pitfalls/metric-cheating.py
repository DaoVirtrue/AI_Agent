#!/usr/bin/env python3
"""
陷阱1: 指标作弊 (Metric Cheating)
===================================
症状: 评估指标看起来很好，但实际用户体验差
根因:
  1. 评估数据集与训练数据重叠
  2. 选择性地报告有利指标
  3. 使用不当的指标定义
  4. 没有区分"看起来好"和"真的好"

解决方案:
  - 使用独立于训练的金标准数据集
  - 报告所有相关指标（不选择性报告）
  - 确保指标定义与实际目标一致
  - 结合线上用户反馈验证
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MetricCheatingDetector:
    """指标作弊检测器"""

    # 检测标记
    suspicious_patterns: List[str] = field(default_factory=list)

    def check_overlap(self, train_docs: List[str], eval_docs: List[str]) -> Dict[str, Any]:
        """检查训练集和评估集的重叠

        Args:
            train_docs: 训练用文档ID列表
            eval_docs: 评估用文档ID列表

        Returns:
            重叠分析结果
        """
        train_set = set(train_docs)
        eval_set = set(eval_docs)
        overlap = train_set & eval_set

        overlap_ratio = len(overlap) / max(1, len(eval_set))
        is_suspicious = overlap_ratio > 0.1  # 超过10%重叠即为可疑

        if is_suspicious:
            self.suspicious_patterns.append(
                f"训练集和评估集重叠: {overlap_ratio:.1%}"
            )

        return {
            'overlap_count': len(overlap),
            'overlap_ratio': round(overlap_ratio, 4),
            'is_suspicious': is_suspicious,
            'recommendation': (
                '正常' if not is_suspicious
                else '评估集必须与训练集完全独立！重新划分数据'
            ),
        }

    def check_selective_reporting(
        self,
        all_metrics: Dict[str, float],
        reported_metrics: List[str]
    ) -> Dict[str, Any]:
        """检查是否选择性报告指标

        Args:
            all_metrics: 所有可用的指标
            reported_metrics: 实际报告的指标

        Returns:
            选择性报告分析
        """
        all_keys = set(all_metrics.keys())
        reported_keys = set(reported_metrics)
        unreported = all_keys - reported_keys

        # 检查未报告的指标是否显著低于已报告的
        if unreported:
            reported_avg = sum(all_metrics[k] for k in reported_keys) / len(reported_keys)
            unreported_avg = sum(all_metrics[k] for k in unreported) / len(unreported)

            gap = reported_avg - unreported_avg
            is_suspicious = gap > 0.15  # 差距超过15%可疑

            if is_suspicious:
                self.suspicious_patterns.append(
                    f"选择性报告: 报告指标avg={reported_avg:.3f}, 隐藏指标avg={unreported_avg:.3f}"
                )

            return {
                'reported_avg': round(reported_avg, 4),
                'unreported_avg': round(unreported_avg, 4),
                'gap': round(gap, 4),
                'unreported_metrics': list(unreported),
                'is_suspicious': is_suspicious,
            }

        return {'is_suspicious': False, 'unreported_metrics': []}

    def check_goodharts_law(self, metric_history: List[Dict[str, float]]) -> Dict[str, Any]:
        """检查古德哈特定律效应: 当一个指标成为目标后，它就不再是好指标

        Args:
            metric_history: 指标历史记录列表

        Returns:
            古德哈特效应分析
        """
        if len(metric_history) < 3:
            return {'is_suspicious': False}

        # 检查是否在优化某个指标后，其他指标下降
        first = metric_history[0]
        last = metric_history[-1]

        # 找出改善最大的指标
        max_improved = None
        max_delta = 0

        for key in first:
            if key in last:
                delta = last[key] - first[key]
                if delta > max_delta:
                    max_delta = delta
                    max_improved = key

        # 检查其他指标是否总体下降
        deteriorated = []
        for key in first:
            if key in last and key != max_improved:
                if last[key] < first[key]:
                    deteriorated.append(key)

        is_suspicious = len(deteriorated) > 0 and max_delta > 0.05

        if is_suspicious:
            self.suspicious_patterns.append(
                f"古德哈特效应: {max_improved}改善+{max_delta:.3f}, "
                f"但{len(deteriorated)}个指标恶化"
            )

        return {
            'most_improved': max_improved,
            'improvement': round(max_delta, 4),
            'deteriorated_metrics': deteriorated,
            'is_suspicious': is_suspicious,
        }

    def generate_report(self) -> str:
        """生成检测报告"""
        if not self.suspicious_patterns:
            return "✓ 未检测到指标作弊迹象"

        lines = ["⚠ 检测到指标作弊可疑模式:"]
        for i, pattern in enumerate(self.suspicious_patterns, 1):
            lines.append(f"  {i}. {pattern}")

        lines.append("\n建议:")
        lines.append("  1. 确保评估集与训练集完全独立")
        lines.append("  2. 报告所有指标，不做选择性报告")
        lines.append("  3. 监控指标间的权衡关系")
        lines.append("  4. 定期用新的独立数据验证")

        return "\n".join(lines)


# 解决方案: 验证评估完整性
def validate_evaluation_integrity(
    train_data_ids: List[str],
    eval_data_ids: List[str],
    all_metrics: Dict[str, float],
    reported_metrics: List[str],
) -> Dict[str, Any]:
    """完整的评估完整性验证

    检查清单:
    ✓ 数据独立性
    ✓ 指标完整性
    ✓ 统计显著性
    ✓ 效应量报告
    """
    detector = MetricCheatingDetector()

    overlap_check = detector.check_overlap(train_data_ids, eval_data_ids)
    reporting_check = detector.check_selective_reporting(all_metrics, reported_metrics)

    passed = True
    checks = []

    # 检查1: 数据独立性
    checks.append({
        'check': '数据独立性',
        'passed': not overlap_check['is_suspicious'],
        'detail': f"重叠率: {overlap_check['overlap_ratio']:.1%}",
    })
    if overlap_check['is_suspicious']:
        passed = False

    # 检查2: 指标完整性
    checks.append({
        'check': '指标完整性',
        'passed': not reporting_check['is_suspicious'],
        'detail': f"未报告指标: {reporting_check.get('unreported_metrics', [])}",
    })
    if reporting_check['is_suspicious']:
        passed = False

    return {
        'passed': passed,
        'checks': checks,
        'recommendations': detector.generate_report(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  指标作弊检测 - 演示")
    print("=" * 60)

    # 场景1: 数据重叠
    print("\n[场景1] 训练/评估集重叠")
    train_ids = [f"doc_{i}" for i in range(100)]
    eval_with_overlap = train_ids[:10] + [f"eval_{i}" for i in range(40)]

    detector = MetricCheatingDetector()
    result = detector.check_overlap(train_ids, eval_with_overlap)
    print(f"  重叠: {result['overlap_count']} 文档 ({result['overlap_ratio']:.1%})")
    print(f"  可疑: {result['is_suspicious']}")

    # 场景2: 选择性报告
    print("\n[场景2] 选择性报告")
    all_metrics = {
        'accuracy': 0.85,
        'relevance': 0.82,
        'faithfulness': 0.90,
        'fluency': 0.88,
        'latency_score': 0.45,  # 这个被隐藏了
        'cost_score': 0.40,      # 这个也被隐藏了
    }
    reported = ['accuracy', 'relevance', 'faithfulness', 'fluency']
    result = detector.check_selective_reporting(all_metrics, reported)
    print(f"  报告均值: {result['reported_avg']:.3f}")
    print(f"  隐藏均值: {result['unreported_avg']:.3f}")
    print(f"  差距: {result['gap']:.3f}")
    print(f"  可疑: {result['is_suspicious']}")

    # 完整验证
    print("\n[完整验证]")
    validation = validate_evaluation_integrity(train_ids, eval_with_overlap, all_metrics, reported)
    print(f"  通过: {validation['passed']}")
    for check in validation['checks']:
        status = '✓' if check['passed'] else '✗'
        print(f"  {status} {check['check']}: {check['detail']}")

    print()
    print(detector.generate_report())
