#!/usr/bin/env python3
"""
持续改进闭环 (Continuous Improvement Closed-Loop Pipeline)
==========================================================
完整的评估-分析-决策-记录闭环:

步骤:
  1. evaluate - 执行评估
  2. analyze  - 分析结果 (对比阈值, 回归检测)
  3. gate     - 门控决策 (PASS/WARN/FAIL)
  4. record   - 记录结果和元数据

包含:
  - 基线管理和实验比较
  - 95%置信规则: 不超过基线95%置信区间上限
  - OptimizationCycle跟踪
  - CI/CD集成接口
"""

import json
import time
import math
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# 数据结构
# ============================================================================

class GateDecision(Enum):
    """门控决策"""
    PASS = "PASS"           # 通过，可以部署
    WARN = "WARN"           # 警告，需要关注但可以部署
    FAIL = "FAIL"           # 失败，阻止部署
    PENDING = "PENDING"     # 待评估


@dataclass
class MetricThreshold:
    """指标阈值"""
    metric_name: str
    min_acceptable: float
    target: float
    max_acceptable: float
    direction: str = "higher_is_better"  # 'higher_is_better' | 'lower_is_better'


@dataclass
class BaselineSnapshot:
    """基线快照"""
    baseline_id: str
    metrics: Dict[str, float]  # 指标名 → 均值
    std_devs: Dict[str, float]  # 指标名 → 标准差
    sample_size: int
    timestamp: str
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """评估结果"""
    run_id: str
    metrics: Dict[str, float]  # 指标名 → 得分
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OptimizationCycle:
    """优化周期记录"""
    cycle_id: str
    cycle_number: int
    decision: GateDecision
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    changes_made: List[str]
    timestamp: str
    notes: str = ""
    duration_seconds: float = 0.0


# ============================================================================
# 持续改进管道
# ============================================================================

class ContinuousImprovementPipeline:
    """持续改进闭环管道

    完整的 评估→分析→门控→记录 循环。
    """

    def __init__(
        self,
        project_name: str = "rag-pipeline",
        history_size: int = 100,
    ):
        """初始化管道

        Args:
            project_name: 项目名称
            history_size: 历史记录保留数量
        """
        self.project_name = project_name
        self.history_size = history_size

        # 阈值配置
        self.thresholds: Dict[str, MetricThreshold] = {}

        # 基线
        self.baseline: Optional[BaselineSnapshot] = None
        self.baseline_history: deque = deque(maxlen=history_size)

        # 评估历史
        self.evaluation_history: deque = deque(maxlen=history_size)

        # 优化周期
        self.optimization_cycles: deque = deque(maxlen=history_size)

        # 决策日志
        self.gate_decisions: deque = deque(maxlen=history_size)

        # 回归检测配置
        self.regression_confidence = 0.95  # 95%置信

        # 统计
        self.current_cycle_number = 0

        # CI/CD集成接口
        self.ci_cd_webhook_url: Optional[str] = None
        self.on_gate_decision: Optional[Callable] = None

    # ========================================================================
    # 阈值配置
    # ========================================================================

    def set_threshold(
        self,
        metric_name: str,
        min_acceptable: float,
        target: float,
        max_acceptable: float = 1.0,
        direction: str = "higher_is_better",
    ) -> None:
        """设置指标阈值

        Args:
            metric_name: 指标名称
            min_acceptable: 最低可接受值
            target: 目标值
            max_acceptable: 最高可接受值
            direction: 'higher_is_better' 或 'lower_is_better'
        """
        self.thresholds[metric_name] = MetricThreshold(
            metric_name=metric_name,
            min_acceptable=min_acceptable,
            target=target,
            max_acceptable=max_acceptable,
            direction=direction,
        )

    def set_default_thresholds(self) -> None:
        """设置默认的评估阈值"""
        defaults = [
            ("accuracy", 0.65, 0.85, 1.0, "higher_is_better"),
            ("relevance", 0.70, 0.85, 1.0, "higher_is_better"),
            ("faithfulness", 0.70, 0.90, 1.0, "higher_is_better"),
            ("fluency", 0.75, 0.90, 1.0, "higher_is_better"),
            ("latency_score", 0.50, 0.80, 1.0, "higher_is_better"),
            ("cost_score", 0.60, 0.90, 1.0, "higher_is_better"),
            ("weighted_score", 0.70, 0.85, 1.0, "higher_is_better"),
            ("hit_rate@5", 0.70, 0.90, 1.0, "higher_is_better"),
            ("mrr", 0.65, 0.85, 1.0, "higher_is_better"),
            ("avg_latency_seconds", None, 2.0, 5.0, "lower_is_better"),
            ("error_rate", None, 0.01, 0.05, "lower_is_better"),
        ]

        for name, min_val, target, max_val, direction in defaults:
            self.set_threshold(name, min_val, target, max_val, direction)

    # ========================================================================
    # 评估 (Evaluate)
    # ========================================================================

    def evaluate(
        self,
        evaluator_fn: Callable[..., Dict[str, float]],
        **evaluator_kwargs,
    ) -> EvaluationResult:
        """执行评估

        Args:
            evaluator_fn: 评估函数，返回 {metric_name: score} 字典
            **evaluator_kwargs: 传递给评估函数的参数

        Returns:
            EvaluationResult
        """
        run_id = self._generate_run_id()
        start_time = time.perf_counter()

        # 执行评估
        raw_metrics = evaluator_fn(**evaluator_kwargs)

        elapsed = time.perf_counter() - start_time

        # 创建结果
        result = EvaluationResult(
            run_id=run_id,
            metrics=raw_metrics,
            details={
                'evaluation_time_seconds': elapsed,
                'evaluator_kwargs': {k: str(v)[:50] for k, v in evaluator_kwargs.items()},
            },
        )

        # 记录到历史
        self.evaluation_history.append(result)

        return result

    def _generate_run_id(self) -> str:
        """生成唯一的运行ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_val = hashlib.md5(f"{self.project_name}_{timestamp}_{self.current_cycle_number}".encode()).hexdigest()[:8]
        return f"eval_{timestamp}_{hash_val}"

    # ========================================================================
    # 分析 (Analyze)
    # ========================================================================

    def analyze(
        self,
        eval_result: EvaluationResult,
        check_regression: bool = True,
        check_thresholds: bool = True,
    ) -> Dict[str, Any]:
        """分析评估结果

        Args:
            eval_result: 评估结果
            check_regression: 是否进行回归检测
            check_thresholds: 是否检查阈值

        Returns:
            分析报告字典
        """
        analysis = {
            'run_id': eval_result.run_id,
            'timestamp': eval_result.timestamp,
            'threshold_checks': {},
            'regression_analysis': {},
            'warnings': [],
            'errors': [],
            'passed': True,
        }

        # 1. 阈值检查
        if check_thresholds:
            for metric_name, value in eval_result.metrics.items():
                threshold = self.thresholds.get(metric_name)
                if threshold is None:
                    continue

                check_result = self._check_threshold(metric_name, value, threshold)
                analysis['threshold_checks'][metric_name] = check_result

                if check_result['status'] == 'FAIL':
                    analysis['errors'].append(
                        f"{metric_name}={value:.4f} 低于最低要求 {threshold.min_acceptable}"
                    )
                    analysis['passed'] = False
                elif check_result['status'] == 'WARN':
                    analysis['warnings'].append(
                        f"{metric_name}={value:.4f} 低于目标值 {threshold.target}"
                    )

        # 2. 回归检测 (与基线比较)
        if check_regression and self.baseline:
            for metric_name, value in eval_result.metrics.items():
                if metric_name not in self.baseline.metrics:
                    continue

                reg_result = self._detect_regression(
                    metric_name,
                    value,
                    self.baseline.metrics[metric_name],
                    self.baseline.std_devs.get(metric_name, 0.0),
                    self.baseline.sample_size,
                    confidence=self.regression_confidence,
                )
                analysis['regression_analysis'][metric_name] = reg_result

                if reg_result['is_regression']:
                    analysis['errors'].append(
                        f"{metric_name} 显著退化: 基线={self.baseline.metrics[metric_name]:.4f}, "
                        f"当前={value:.4f}, p={reg_result.get('p_value', 1):.4f}"
                    )
                    analysis['passed'] = False

        return analysis

    def _check_threshold(
        self,
        metric_name: str,
        value: float,
        threshold: MetricThreshold,
    ) -> Dict[str, Any]:
        """检查单个指标是否满足阈值

        Args:
            metric_name: 指标名称
            value: 当前值
            threshold: 阈值定义

        Returns:
            检查结果
        """
        result = {
            'metric': metric_name,
            'value': value,
            'target': threshold.target,
            'status': 'PASS',
        }

        if threshold.direction == "higher_is_better":
            if threshold.min_acceptable is not None and value < threshold.min_acceptable:
                result['status'] = 'FAIL'
            elif value < threshold.target:
                result['status'] = 'WARN'
            else:
                result['status'] = 'PASS'
        else:  # lower_is_better
            if threshold.max_acceptable is not None and value > threshold.max_acceptable:
                result['status'] = 'FAIL'
            elif value > threshold.target:
                result['status'] = 'WARN'
            else:
                result['status'] = 'PASS'

        result['direction'] = threshold.direction
        return result

    def _detect_regression(
        self,
        metric_name: str,
        current_value: float,
        baseline_mean: float,
        baseline_std: float,
        baseline_n: int,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """检测是否发生回归 (95%置信规则)

        使用单样本z-test: 当前值是否显著低于基线?
        95%置信区间: 不超过基线均值 - z_crit * std/sqrt(n)

        Args:
            metric_name: 指标名称
            current_value: 当前值
            baseline_mean: 基线均值
            baseline_std: 基线标准差
            baseline_n: 基线样本数
            confidence: 置信水平

        Returns:
            回归检测结果
        """
        if baseline_n < 2 or baseline_std == 0:
            return {
                'metric': metric_name,
                'is_regression': False,
                'current_value': current_value,
                'baseline_mean': baseline_mean,
                'message': '基线数据不足 (n<2 或 std=0)',
            }

        # 标准误差
        se = baseline_std / math.sqrt(baseline_n) if baseline_n > 0 else baseline_std

        # z临界值 (95% → 1.645 单侧)
        z_crit = self._z_value_for_confidence(confidence, one_tailed=True)

        # 计算置信区间下界 (对于单侧测试: 不超过这个值)
        lower_bound = baseline_mean - z_crit * se

        # 判断: 当前值是否低于95%置信区间下界
        threshold = self.thresholds.get(metric_name)

        if threshold and threshold.direction == "lower_is_better":
            # 对于"越低越好"的指标: 当前值是否显著高于基线
            upper_bound = baseline_mean + z_crit * se
            is_regression = current_value > upper_bound
            message = f"当前值({current_value:.4f}) > 上界({upper_bound:.4f})"
        else:
            # 对于"越高越好"的指标: 当前值是否显著低于基线
            is_regression = current_value < lower_bound
            message = f"当前值({current_value:.4f}) < 下界({lower_bound:.4f})"

        return {
            'metric': metric_name,
            'is_regression': is_regression,
            'current_value': current_value,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'baseline_n': baseline_n,
            'confidence_level': confidence,
            'confidence_interval_lower': lower_bound,
            'z_critical': z_crit,
            'message': message,
        }

    @staticmethod
    def _z_value_for_confidence(confidence: float, one_tailed: bool = False) -> float:
        """获取置信水平对应的z值"""
        if one_tailed:
            mapping = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        else:
            mapping = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}

        # 最接近的key
        closest = min(mapping.keys(), key=lambda k: abs(k - confidence))
        return mapping[closest]

    # ========================================================================
    # 门控决策 (Gate)
    # ========================================================================

    def gate(
        self,
        analysis: Dict[str, Any],
        changes_summary: Optional[List[str]] = None,
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        """门控决策: PASS / WARN / FAIL

        决策规则:
          - 所有阈值检查通过 且 无回归 → PASS
          - 有WARN但无FAIL → WARN
          - 有FAIL → FAIL

        Args:
            analysis: 分析报告 (来自analyze方法)
            changes_summary: 变更摘要列表

        Returns:
            (门控决策, 决策详情)
        """
        decision_details = {
            'analysis_id': analysis.get('run_id'),
            'timestamp': datetime.now().isoformat(),
            'changes_summary': changes_summary or [],
            'reason': '',
        }

        if not analysis.get('passed', True):
            # 有FAIL
            decision = GateDecision.FAIL
            decision_details['reason'] = f"有 {len(analysis['errors'])} 个检查失败"
            decision_details['errors'] = analysis['errors']
        elif analysis.get('warnings', []):
            # 有WARN但无FAIL
            decision = GateDecision.WARN
            decision_details['reason'] = f"有 {len(analysis['warnings'])} 个警告"
            decision_details['warnings'] = analysis['warnings']
        else:
            # 一切正常
            decision = GateDecision.PASS
            decision_details['reason'] = "所有检查通过"

        decision_details['decision'] = decision.value

        # 记录决策
        self.gate_decisions.append({
            'decision': decision,
            'details': decision_details,
            'timestamp': decision_details['timestamp'],
        })

        # 触发CI/CD回调
        if self.on_gate_decision:
            try:
                self.on_gate_decision(decision, decision_details)
            except Exception as e:
                print(f"[警告] CI/CD回调失败: {e}")

        return decision, decision_details

    # ========================================================================
    # 记录 (Record)
    # ========================================================================

    def record(
        self,
        eval_result: EvaluationResult,
        analysis: Dict[str, Any],
        decision: GateDecision,
        changes_made: List[str],
        notes: str = "",
    ) -> OptimizationCycle:
        """记录优化周期

        Args:
            eval_result: 评估结果
            analysis: 分析报告
            decision: 门控决策
            changes_made: 本周期所做的变更
            notes: 备注

        Returns:
            OptimizationCycle
        """
        self.current_cycle_number += 1

        # 获取变更前的指标 (从基线获取)
        metrics_before = {}
        if self.baseline:
            metrics_before = self.baseline.metrics
        elif self.optimization_cycles:
            last_cycle = self.optimization_cycles[-1]
            metrics_before = last_cycle.metrics_after

        cycle = OptimizationCycle(
            cycle_id=f"{self.project_name}_cycle_{self.current_cycle_number:04d}",
            cycle_number=self.current_cycle_number,
            decision=decision,
            metrics_before=metrics_before,
            metrics_after=eval_result.metrics,
            changes_made=changes_made,
            timestamp=datetime.now().isoformat(),
            notes=notes,
        )

        self.optimization_cycles.append(cycle)

        # 如果通过，更新基线
        if decision == GateDecision.PASS and self.current_cycle_number > 0:
            new_baseline = BaselineSnapshot(
                baseline_id=f"baseline_{datetime.now().strftime('%Y%m%d')}_{self.current_cycle_number}",
                metrics=eval_result.metrics,
                std_devs={k: 0.01 for k in eval_result.metrics},  # 简化：使用固定标准差
                sample_size=self._count_evaluations(),
                timestamp=datetime.now().isoformat(),
                version=f"v{self.current_cycle_number}",
                metadata={
                    'changes': changes_made,
                    'cycle_number': self.current_cycle_number,
                },
            )

            if self.baseline:
                self.baseline_history.append(self.baseline)

            self.baseline = new_baseline

        return cycle

    def _count_evaluations(self) -> int:
        """统计评估次数"""
        return len(self.evaluation_history)

    # ========================================================================
    # 完整闭环执行
    # ========================================================================

    def run_cycle(
        self,
        evaluator_fn: Callable[..., Dict[str, float]],
        changes_made: List[str],
        notes: str = "",
        **evaluator_kwargs,
    ) -> Dict[str, Any]:
        """执行完整的评估→分析→门控→记录循环

        这是主入口方法。

        Args:
            evaluator_fn: 评估函数
            changes_made: 本周期所做的变更
            notes: 备注
            **evaluator_kwargs: 评估参数

        Returns:
            完整循环结果
        """
        cycle_start = time.perf_counter()

        # Step 1: Evaluate
        eval_result = self.evaluate(evaluator_fn, **evaluator_kwargs)

        # Step 2: Analyze
        analysis = self.analyze(eval_result)

        # Step 3: Gate
        decision, decision_details = self.gate(analysis, changes_made)

        # Step 4: Record
        cycle = self.record(eval_result, analysis, decision, changes_made, notes)

        cycle_duration = time.perf_counter() - cycle_start

        return {
            'eval_result': eval_result,
            'analysis': analysis,
            'decision': decision.value,
            'decision_details': decision_details,
            'cycle': cycle,
            'duration_seconds': cycle_duration,
            'improvement': self._calculate_improvement(cycle),
        }

    def _calculate_improvement(self, cycle: OptimizationCycle) -> Dict[str, float]:
        """计算改进幅度"""
        improvements = {}
        for metric_name, current_value in cycle.metrics_after.items():
            if metric_name in cycle.metrics_before:
                before = cycle.metrics_before[metric_name]
                if before != 0:
                    improvements[metric_name] = (current_value - before) / abs(before)
                else:
                    improvements[metric_name] = current_value
        return improvements

    # ========================================================================
    # CI/CD集成
    # ========================================================================

    def set_ci_cd_webhook(self, url: str) -> None:
        """设置CI/CD Webhook URL"""
        self.ci_cd_webhook_url = url

    def set_ci_cd_callback(self, callback: Callable) -> None:
        """设置CI/CD决策回调

        回调签名: callback(decision: GateDecision, details: Dict) -> None
        """
        self.on_gate_decision = callback

    # ========================================================================
    # 报告生成
    # ========================================================================

    def generate_improvement_report(self) -> str:
        """生成改进报告"""
        lines = [
            f"{'=' * 60}",
            f"  持续改进报告: {self.project_name}",
            f"{'=' * 60}",
            f"  总周期数: {self.current_cycle_number}",
            f"  基线版本: {self.baseline.version if self.baseline else 'N/A'}",
            f"",
            f"  {'─' * 50}",
            f"  周期历史:",
            f"  {'─' * 50}",
        ]

        for cycle in self.optimization_cycles:
            icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'PENDING': '⏸️'}.get(cycle.decision.value, '❓')
            changes = ', '.join(cycle.changes_made[:3])
            lines.append(
                f"  {icon} 周期{cycle.cycle_number:04d}: {cycle.decision.value:5s} "
                f"| 变更: {changes}"
            )

        # 当前基线指标
        if self.baseline:
            lines.append(f"\n  {'─' * 50}")
            lines.append(f"  当前基线: {self.baseline.baseline_id}")
            lines.append(f"  {'─' * 50}")
            for metric, value in sorted(self.baseline.metrics.items()):
                threshold = self.thresholds.get(metric)
                if threshold:
                    target = threshold.target
                    status = "✓" if value >= target else "✗"
                else:
                    status = "-"
                lines.append(f"    {metric:25s}: {value:.4f}  [{status}]")

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)

    def export_history(self) -> Dict[str, Any]:
        """导出完整历史记录"""
        return {
            'project': self.project_name,
            'total_cycles': self.current_cycle_number,
            'current_baseline': {
                'id': self.baseline.baseline_id if self.baseline else None,
                'metrics': self.baseline.metrics if self.baseline else {},
                'version': self.baseline.version if self.baseline else None,
            } if self.baseline else None,
            'cycles': [
                {
                    'cycle_id': c.cycle_id,
                    'number': c.cycle_number,
                    'decision': c.decision.value,
                    'metrics_before': c.metrics_before,
                    'metrics_after': c.metrics_after,
                    'changes': c.changes_made,
                    'timestamp': c.timestamp,
                }
                for c in self.optimization_cycles
            ],
            'recent_evaluations': [
                {
                    'run_id': e.run_id,
                    'metrics': e.metrics,
                    'timestamp': e.timestamp,
                }
                for e in list(self.evaluation_history)[-10:]
            ],
        }


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  持续改进闭环 - 完整演示")
    print("=" * 60)

    # ---- 初始化管道 ----
    pipeline = ContinuousImprovementPipeline(project_name="RAG_v2_improvement")
    pipeline.set_default_thresholds()

    print("\n已配置阈值:")
    for name, t in sorted(pipeline.thresholds.items()):
        print(f"  {name:25s}: min={str(t.min_acceptable):8s}, target={str(t.target):8s}, "
              f"dir={t.direction}")

    # ---- 模拟评估函数 ----
    import random
    random.seed(42)

    def simulate_evaluation(iteration: int) -> Dict[str, float]:
        """模拟评估函数 - 每次迭代略有改进"""
        base_scores = {
            'weighted_score': 0.72,
            'accuracy': 0.68,
            'relevance': 0.73,
            'faithfulness': 0.75,
            'fluency': 0.82,
            'latency_score': 0.70,
            'cost_score': 0.78,
            'error_rate': 0.04,
        }

        # 每次迭代有小幅改进
        improvement = iteration * 0.03
        noise = 0.02

        scores = {}
        for k, v in base_scores.items():
            if k == 'error_rate':
                scores[k] = max(0.0, v - improvement + random.gauss(0, noise))
            else:
                scores[k] = min(1.0, v + improvement + random.gauss(0, noise))

        return scores

    # ---- 执行多个优化周期 ----
    print("\n" + "=" * 50)
    print("  执行优化周期")
    print("=" * 50)

    for i in range(4):
        print(f"\n{'─' * 50}")
        print(f"  周期 {i + 1}")
        print(f"{'─' * 50}")

        # 模拟的变更
        changes = {
            1: ["升级嵌入模型 (text-embedding-3-large)", "增加重排序步骤"],
            2: ["优化提示词模板", "增加缓存机制"],
            3: ["添加更激进的幻觉检测", "调整检索K值"],
            4: ["部署新版本RAG管道", "添加查询重写"],
        }.get(i + 1, ["迭代改进"])

        result = pipeline.run_cycle(
            evaluator_fn=lambda: simulate_evaluation(i + 1),
            changes_made=changes,
            notes=f"第{i+1}轮优化",
        )

        print(f"  决策: {result['decision']}")
        print(f"  耗时: {result['duration_seconds']:.2f}s")

        for metric, value in result['eval_result'].metrics.items():
            before = result['cycle'].metrics_before.get(metric, 0)
            delta = value - before
            print(f"    {metric:20s}: {before:.4f} → {value:.4f} ({delta:+.4f})")

        if result['analysis']['warnings']:
            for w in result['analysis']['warnings'][:3]:
                print(f"    ⚠️ {w}")

        if result['analysis']['errors']:
            for e in result['analysis']['errors'][:3]:
                print(f"    ❌ {e}")

    # ---- 门控决策演示 ----
    print("\n" + "=" * 50)
    print("  门控决策测试")
    print("=" * 50)

    test_evaluations = [
        ("优秀结果", {'weighted_score': 0.88, 'accuracy': 0.85}, [GateDecision.PASS]),
        ("及格结果", {'weighted_score': 0.72, 'accuracy': 0.66}, [GateDecision.WARN]),
        ("失败结果", {'weighted_score': 0.55, 'accuracy': 0.50}, [GateDecision.FAIL]),
    ]

    for name, metrics, expected in test_evaluations:
        eval_result = EvaluationResult(
            run_id=f"test_{name}",
            metrics=metrics,
        )
        analysis = pipeline.analyze(eval_result)
        decision, _ = pipeline.gate(analysis)
        expected_icon = "✅" if decision in expected else "❌"
        print(f"  {expected_icon} {name}: {decision.value} (期望: {[e.value for e in expected]})")

    # ---- 生成报告 ----
    print("\n" + "=" * 50)
    print("  改进报告")
    print("=" * 50)

    report = pipeline.generate_improvement_report()
    print(report)

    # ---- 导出历史 ----
    print("\n" + "=" * 50)
    print("  历史导出 (JSON)")
    print("=" * 50)

    history = pipeline.export_history()
    print(f"  总周期: {history['total_cycles']}")
    print(f"  当前基线: {history['current_baseline']['id']}")
    print(f"  基线指标: {list(history['current_baseline']['metrics'].keys())}")

    # 保存到文件
    history_file = "improvement_history.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n  历史已保存到: {history_file}")

    print()
    print("=" * 60)
    print("  持续改进闭环演示完成")
    print("=" * 60)
