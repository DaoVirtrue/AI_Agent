#!/usr/bin/env python3
"""
在线监控系统 (Online Monitoring)
=================================
基于Prometheus + Grafana的生产监控体系:

Prometheus指标:
  - skill_calls_total (Counter) - 技能调用计数
  - skill_execution_duration_seconds (Histogram) - 执行时间
  - skill_tokens_total (Counter) - token消耗
  - skill_error_rate (Gauge) - 错误率 (5分钟窗口)
  - skill_in_flight (Gauge) - 正在处理的请求数

Grafana Dashboard JSON模板 + 告警规则:
  - error_rate > 5% for 5min → CRITICAL
  - P99 > 10s for 5min → WARNING
"""

import time
import json
import threading
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict, Counter

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class MetricLabels:
    """指标标签"""
    labels: Dict[str, str] = field(default_factory=dict)

    def to_prometheus_str(self) -> str:
        """转为Prometheus标签格式: {key="value", ...}"""
        if not self.labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(parts) + "}"

    def __hash__(self):
        return hash(tuple(sorted(self.labels.items())))

    def __eq__(self, other):
        return self.labels == other.labels


@dataclass
class AlertRule:
    """告警规则"""
    alert_name: str
    severity: str  # 'critical', 'warning', 'info'
    expression: str
    description: str
    for_duration: str = "5m"  # Prometheus语法
    runbook_url: str = ""


# ============================================================================
# Prometheus指标模拟器
# ============================================================================

class Counter:
    """Prometheus Counter - 只增不减"""

    def __init__(self, name: str, help_text: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.help = help_text
        self.label_names = label_names or []
        self._values: Dict[Tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels) -> None:
        """增加计数"""
        with self._lock:
            key = self._make_key(labels)
            self._values[key] += value

    def get(self, **labels) -> float:
        """获取当前值"""
        key = self._make_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _make_key(self, labels: Dict[str, str]) -> Tuple:
        """生成键"""
        return tuple(labels.get(ln, "") for ln in self.label_names)

    def export(self) -> str:
        """导出为Prometheus文本格式"""
        lines = [f"# HELP {self.name} {self.help}",
                 f"# TYPE {self.name} counter"]
        with self._lock:
            for key, value in self._values.items():
                if key:
                    label_str = "{"
                    parts = [f'{self.label_names[i]}="{key[i]}"' for i in range(len(self.label_names))]
                    label_str += ",".join(parts) + "}"
                else:
                    label_str = ""
                lines.append(f"{self.name}{label_str} {value}")
        return "\n".join(lines)


class Gauge:
    """Prometheus Gauge - 可增可减"""

    def __init__(self, name: str, help_text: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.help = help_text
        self.label_names = label_names or []
        self._values: Dict[Tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **labels) -> None:
        """设置值"""
        with self._lock:
            key = self._make_key(labels)
            self._values[key] = value

    def inc(self, value: float = 1.0, **labels) -> None:
        """增加"""
        with self._lock:
            key = self._make_key(labels)
            self._values[key] += value

    def dec(self, value: float = 1.0, **labels) -> None:
        """减少"""
        with self._lock:
            key = self._make_key(labels)
            self._values[key] -= value

    def get(self, **labels) -> float:
        """获取当前值"""
        key = self._make_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _make_key(self, labels: Dict[str, str]) -> Tuple:
        return tuple(labels.get(ln, "") for ln in self.label_names)

    def export(self) -> str:
        """导出为Prometheus文本格式"""
        lines = [f"# HELP {self.name} {self.help}",
                 f"# TYPE {self.name} gauge"]
        with self._lock:
            for key, value in self._values.items():
                if key:
                    label_str = "{"
                    parts = [f'{self.label_names[i]}="{key[i]}"' for i in range(len(self.label_names))]
                    label_str += ",".join(parts) + "}"
                else:
                    label_str = ""
                lines.append(f"{self.name}{label_str} {value}")
        return "\n".join(lines)


class Histogram:
    """Prometheus Histogram"""

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Optional[List[float]] = None,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.help = help_text
        self.buckets = sorted(buckets or [0.1, 0.5, 1, 2, 5, 10, 30, 60])
        self.label_names = label_names or []

        # 存储: key → {bucket: count, '+Inf': count, 'count': int, 'sum': float}
        self._values: Dict[Tuple, Dict[str, Any]] = defaultdict(
            lambda: {'count': 0, 'sum': 0.0, 'bucket_counts': {b: 0 for b in self.buckets}}
        )
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        """记录观测值"""
        with self._lock:
            key = self._make_key(labels)
            entry = self._values[key]
            entry['count'] += 1
            entry['sum'] += value

            # 落入哪个桶
            for bucket in self.buckets:
                if value <= bucket:
                    entry['bucket_counts'][bucket] += 1

    def _make_key(self, labels: Dict[str, str]) -> Tuple:
        return tuple(labels.get(ln, "") for ln in self.label_names)

    def export(self) -> str:
        """导出为Prometheus文本格式"""
        lines = [f"# HELP {self.name} {self.help}",
                 f"# TYPE {self.name} histogram"]

        with self._lock:
            for key, entry in self._values.items():
                if key:
                    label_str = "{"
                    parts = [f'{self.label_names[i]}="{key[i]}"' for i in range(len(self.label_names))]
                    label_str += ",".join(parts) + "}"
                else:
                    label_str = ""

                # 桶计数
                for bucket in self.buckets:
                    b_label = "{le=\"" + str(bucket) + "\""
                    if label_str:
                        b_label += "," + label_str[1:]
                    else:
                        b_label += "}"
                    lines.append(
                        f"{self.name}_bucket{b_label} {entry['bucket_counts'][bucket]}"
                    )

                # +Inf桶
                inf_label = "{le=\"+Inf\""
                if label_str:
                    inf_label += "," + label_str[1:]
                else:
                    inf_label += "}"
                lines.append(f"{self.name}_bucket{inf_label} {entry['count']}")

                # 总和和计数
                lines.append(f"{self.name}_sum{label_str} {entry['sum']}")
                lines.append(f"{self.name}_count{label_str} {entry['count']}")

        return "\n".join(lines)


# ============================================================================
# 指标注册表
# ============================================================================

class MetricsRegistry:
    """Prometheus指标注册表"""

    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def create_counter(
        self,
        name: str,
        help_text: str,
        label_names: Optional[List[str]] = None,
    ) -> Counter:
        """创建Counter"""
        with self._lock:
            if name in self.counters:
                return self.counters[name]
            counter = Counter(name, help_text, label_names)
            self.counters[name] = counter
            return counter

    def create_gauge(
        self,
        name: str,
        help_text: str,
        label_names: Optional[List[str]] = None,
    ) -> Gauge:
        """创建Gauge"""
        with self._lock:
            if name in self.gauges:
                return self.gauges[name]
            gauge = Gauge(name, help_text, label_names)
            self.gauges[name] = gauge
            return gauge

    def create_histogram(
        self,
        name: str,
        help_text: str,
        buckets: Optional[List[float]] = None,
        label_names: Optional[List[str]] = None,
    ) -> Histogram:
        """创建Histogram"""
        with self._lock:
            if name in self.histograms:
                return self.histograms[name]
            histogram = Histogram(name, help_text, buckets, label_names)
            self.histograms[name] = histogram
            return histogram

    def export_all(self) -> str:
        """导出所有指标为Prometheus文本格式"""
        lines = []

        for counter in self.counters.values():
            lines.append(counter.export())
            lines.append("")

        for gauge in self.gauges.values():
            lines.append(gauge.export())
            lines.append("")

        for histogram in self.histograms.values():
            lines.append(histogram.export())
            lines.append("")

        return "\n".join(lines).rstrip()


# ============================================================================
# Skill监控指标管理
# ============================================================================

class SkillMetricsManager:
    """技能监控指标管理器

    管理以下指标:
      - skill_calls_total: 技能调用总次数
      - skill_execution_duration_seconds: 执行时间直方图
      - skill_tokens_total: Token消耗总量
      - skill_error_rate: 错误率 (5分钟滑动窗口)
      - skill_in_flight: 正在处理的请求数
    """

    # 时间窗口配置 (秒)
    ERROR_RATE_WINDOW_SECONDS = 300  # 5分钟

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or MetricsRegistry()

        # 创建标准指标
        self.skill_calls = self.registry.create_counter(
            "skill_calls_total",
            "技能调用总次数",
            label_names=["skill_name", "status"],
        )

        self.execution_duration = self.registry.create_histogram(
            "skill_execution_duration_seconds",
            "技能执行持续时间",
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
            label_names=["skill_name"],
        )

        self.tokens_total = self.registry.create_counter(
            "skill_tokens_total",
            "Token消耗总量",
            label_names=["skill_name", "token_type"],
        )

        self.error_rate = self.registry.create_gauge(
            "skill_error_rate",
            "技能错误率 (5分钟窗口)",
            label_names=["skill_name"],
        )

        self.in_flight = self.registry.create_gauge(
            "skill_in_flight",
            "正在处理中的技能请求数",
            label_names=["skill_name"],
        )

        # 错误时间窗口 (用于计算滑动窗口错误率)
        self._error_timestamps: Dict[str, deque] = defaultdict(deque)
        self._success_timestamps: Dict[str, deque] = defaultdict(deque)
        self._error_lock = threading.Lock()

    def record_call(
        self,
        skill_name: str,
        duration_seconds: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        status: str = "success",
    ) -> None:
        """记录一次技能调用

        Args:
            skill_name: 技能名称
            duration_seconds: 执行时长
            tokens_input: 输入Token数
            tokens_output: 输出Token数
            status: 'success' 或 'error'
        """
        # 调用计数
        self.skill_calls.inc(skill_name=skill_name, status=status)

        # 执行时间
        self.execution_duration.observe(duration_seconds, skill_name=skill_name)

        # Token计数
        if tokens_input > 0:
            self.tokens_total.inc(tokens_input, skill_name=skill_name, token_type="input")
        if tokens_output > 0:
            self.tokens_total.inc(tokens_output, skill_name=skill_name, token_type="output")

        # 更新错误率窗口
        now = time.time()
        with self._error_lock:
            if status == "error":
                self._error_timestamps[skill_name].append(now)
            else:
                self._success_timestamps[skill_name].append(now)

            # 清理过期记录
            cutoff = now - self.ERROR_RATE_WINDOW_SECONDS
            self._cleanup_expired(skill_name, cutoff)

            # 重新计算错误率
            self._recalculate_error_rate(skill_name)

    def record_in_flight(self, skill_name: str, delta: int = 1) -> None:
        """更新正在处理的请求数

        Args:
            skill_name: 技能名称
            delta: 变化量 (+1开始请求, -1完成请求)
        """
        self.in_flight.inc(delta, skill_name=skill_name)

    def _cleanup_expired(self, skill_name: str, cutoff: float) -> None:
        """清理过期的错误/成功时间戳"""
        while self._error_timestamps[skill_name] and self._error_timestamps[skill_name][0] < cutoff:
            self._error_timestamps[skill_name].popleft()
        while self._success_timestamps[skill_name] and self._success_timestamps[skill_name][0] < cutoff:
            self._success_timestamps[skill_name].popleft()

    def _recalculate_error_rate(self, skill_name: str) -> None:
        """重新计算错误率"""
        error_count = len(self._error_timestamps[skill_name])
        success_count = len(self._success_timestamps[skill_name])
        total = error_count + success_count

        if total > 0:
            rate = error_count / total
        else:
            rate = 0.0

        self.error_rate.set(rate, skill_name=skill_name)

    def get_all_metrics(self) -> str:
        """导出所有指标"""
        return self.registry.export_all()

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {
            'current_time': datetime.now().isoformat(),
            'skills': {},
        }

        for counter_name, counter in self.registry.counters.items():
            summary.setdefault('counters', {})[counter_name] = dict(counter._values)

        for gauge_name, gauge in self.registry.gauges.items():
            summary.setdefault('gauges', {})[gauge_name] = dict(gauge._values)

        return summary


# ============================================================================
# Grafana Dashboard JSON模板
# ============================================================================

class GrafanaDashboardBuilder:
    """Grafana仪表板构建器"""

    @staticmethod
    def build_rag_dashboard_json() -> str:
        """构建RAG监控仪表板的JSON定义

        包含面板:
          1. 技能调用速率 (QPS)
          2. 执行延迟 (P50/P95/P99)
          3. Token消耗趋势
          4. 错误率时间线
          5. 并发请求数
          6. 缓存命中率
          7. 成本累积
          8. 各技能性能对比
        """
        dashboard = {
            "title": "RAG System Monitoring Dashboard",
            "uid": "rag-system-monitoring-v1",
            "version": 1,
            "refresh": "10s",
            "time": {"from": "now-1h", "to": "now"},
            "timezone": "browser",
            "panels": [
                # 行1: 概览统计
                {
                    "id": 1,
                    "title": "技能调用速率 (QPS)",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": 'rate(skill_calls_total[1m])',
                            "legendFormat": "{{skill_name}} - {{status}}",
                        }
                    ],
                    "yaxes": [{"format": "short"}],
                },
                {
                    "id": 2,
                    "title": "并发请求数",
                    "type": "graph",
                    "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "skill_in_flight",
                            "legendFormat": "{{skill_name}}",
                        }
                    ],
                },
                # 行2: 延迟和错误
                {
                    "id": 3,
                    "title": "执行延迟 (P50/P95/P99)",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": 'histogram_quantile(0.50, rate(skill_execution_duration_seconds_bucket[5m]))',
                            "legendFormat": "P50 - {{skill_name}}",
                        },
                        {
                            "expr": 'histogram_quantile(0.95, rate(skill_execution_duration_seconds_bucket[5m]))',
                            "legendFormat": "P95 - {{skill_name}}",
                        },
                        {
                            "expr": 'histogram_quantile(0.99, rate(skill_execution_duration_seconds_bucket[5m]))',
                            "legendFormat": "P99 - {{skill_name}}",
                        },
                    ],
                    "thresholds": [
                        {"value": 5, "color": "yellow"},
                        {"value": 10, "color": "red"},
                    ],
                },
                {
                    "id": 4,
                    "title": "错误率 (5分钟窗口)",
                    "type": "graph",
                    "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "skill_error_rate",
                            "legendFormat": "{{skill_name}}",
                        }
                    ],
                    "thresholds": [
                        {"value": 0.02, "color": "yellow"},
                        {"value": 0.05, "color": "red"},
                    ],
                },
                # 行3: Token和成本
                {
                    "id": 5,
                    "title": "Token消耗速率",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 16, "w": 8, "h": 8},
                    "targets": [
                        {
                            "expr": 'rate(skill_tokens_total[5m])',
                            "legendFormat": "{{skill_name}} - {{token_type}}",
                        }
                    ],
                },
                {
                    "id": 6,
                    "title": "累积Token消耗",
                    "type": "stat",
                    "gridPos": {"x": 8, "y": 16, "w": 4, "h": 8},
                    "targets": [
                        {
                            "expr": "sum(skill_tokens_total)",
                        }
                    ],
                },
                {
                    "id": 7,
                    "title": "各技能性能对比",
                    "type": "table",
                    "gridPos": {"x": 0, "y": 24, "w": 24, "h": 10},
                    "targets": [
                        {
                            "expr": 'skill_calls_total',
                            "format": "table",
                        }
                    ],
                },
            ],
            "templating": {
                "list": [
                    {
                        "name": "skill_name",
                        "type": "query",
                        "query": "label_values(skill_calls_total, skill_name)",
                        "multi": True,
                    },
                ]
            },
        }

        return json.dumps(dashboard, indent=2, ensure_ascii=False)

    @staticmethod
    def build_alert_rules() -> List[AlertRule]:
        """构建告警规则列表"""
        return [
            AlertRule(
                alert_name="HighErrorRate",
                severity="critical",
                expression="skill_error_rate > 0.05",
                description="技能错误率超过5%",
                for_duration="5m",
                runbook_url="https://wiki.company.com/runbooks/high-error-rate",
            ),
            AlertRule(
                alert_name="HighLatency",
                severity="warning",
                expression="histogram_quantile(0.99, rate(skill_execution_duration_seconds_bucket[5m])) > 10",
                description="P99延迟超过10秒",
                for_duration="5m",
                runbook_url="https://wiki.company.com/runbooks/high-latency",
            ),
            AlertRule(
                alert_name="HighConcurrency",
                severity="warning",
                expression="skill_in_flight > 100",
                description="并发请求数超过100",
                for_duration="1m",
            ),
            AlertRule(
                alert_name="TokenRateAnomaly",
                severity="warning",
                expression="rate(skill_tokens_total[15m]) > 100000",
                description="Token消耗速率异常",
                for_duration="5m",
            ),
            AlertRule(
                alert_name="ServiceDown",
                severity="critical",
                expression="up == 0",
                description="服务不可用",
                for_duration="1m",
                runbook_url="https://wiki.company.com/runbooks/service-down",
            ),
        ]


# ============================================================================
# 模拟数据生成器
# ============================================================================

class SimulationRunner:
    """模拟器: 生成模拟流量用于演示"""

    def __init__(self, metrics_manager: SkillMetricsManager):
        self.metrics = metrics_manager
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start_simulation(self, duration_seconds: int = 60, interval: float = 0.5):
        """启动模拟

        Args:
            duration_seconds: 模拟持续时间
            interval: 请求间隔
        """
        self.running = True
        self._thread = threading.Thread(
            target=self._simulate,
            args=(duration_seconds, interval),
            daemon=True,
        )
        self._thread.start()
        print(f"[模拟器] 启动，持续 {duration_seconds}s，间隔 {interval}s")

    def stop(self):
        """停止模拟"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _simulate(self, duration_seconds: int, interval: float):
        """执行模拟"""
        import random
        random.seed(42)

        skills = ["translation", "summarization", "code_review", "qa", "customer_support"]

        start_time = time.time()
        iterations = 0

        while self.running and (time.time() - start_time) < duration_seconds:
            for skill in skills:
                if random.random() < 0.4:  # 40%概率触发
                    # 模拟调用开始
                    self.metrics.record_in_flight(skill, 1)

                    # 模拟执行时间 (对数正态分布)
                    base_latency = {
                        'translation': 2.0,
                        'summarization': 3.0,
                        'code_review': 5.0,
                        'qa': 1.5,
                        'customer_support': 1.0,
                    }.get(skill, 2.0)

                    duration = max(0.1, random.lognormvariate(
                        math.log(base_latency * 0.7), 0.3
                    ))

                    # 模拟Token消耗
                    tokens_in = random.randint(100, 2000)
                    tokens_out = random.randint(50, 1500)

                    # 模拟偶尔的错误 (5%概率)
                    status = "error" if random.random() < 0.05 else "success"

                    # 记录指标
                    self.metrics.record_call(
                        skill_name=skill,
                        duration_seconds=duration,
                        tokens_input=tokens_in,
                        tokens_output=tokens_out,
                        status=status,
                    )

                    # 调用完成
                    self.metrics.record_in_flight(skill, -1)

            iterations += 1
            time.sleep(interval)

        print(f"[模拟器] 完成 {iterations} 个周期，生成 ~{iterations * len(skills) * 0.4:.0f} 次调用")


# ============================================================================
# Prometheus端点模拟
# ============================================================================

class PrometheusEndpoint:
    """模拟Prometheus /metrics端点"""

    def __init__(self, metrics_manager: SkillMetricsManager):
        self.metrics_manager = metrics_manager

    def handler(self) -> str:
        """HTTP处理器的响应"""
        return self.metrics_manager.get_all_metrics()


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  在线监控系统 - 完整演示")
    print("=" * 60)

    # ---- 初始化 ----
    print("\n[1] 初始化指标系统")
    print("-" * 40)

    registry = MetricsRegistry()
    metrics_manager = SkillMetricsManager(registry)
    print("  指标注册表初始化完成")
    print(f"  已注册 {len(registry.counters)} 个Counter, "
          f"{len(registry.gauges)} 个Gauge, "
          f"{len(registry.histograms)} 个Histogram")

    # ---- 手动记录一些调用 ----
    print("\n[2] 手动记录技能调用")
    print("-" * 40)

    metrics_manager.record_call("qa", 1.2, 200, 500, "success")
    metrics_manager.record_call("qa", 3.5, 300, 800, "success")
    metrics_manager.record_call("qa", 15.0, 500, 1200, "error")
    metrics_manager.record_call("translation", 2.1, 100, 300, "success")
    metrics_manager.record_call("summarization", 4.0, 800, 400, "success")
    metrics_manager.record_call("code_review", 8.5, 1500, 800, "success")

    print("  记录了6次手动调用")

    # ---- 导出Prometheus指标 ----
    print("\n[3] Prometheus指标导出 (部分)")
    print("-" * 40)

    prom_data = PrometheusEndpoint(metrics_manager).handler()
    # 只打印前20行
    lines = prom_data.split('\n')[:25]
    for line in lines:
        print(f"  {line}")
    print(f"  ... (共 {len(prom_data.split(chr(10)))} 行)")

    # ---- 运行模拟 ----
    print("\n[4] 运行流量模拟 (30秒)")
    print("-" * 40)

    simulator = SimulationRunner(metrics_manager)
    simulator.start_simulation(duration_seconds=30, interval=0.3)
    simulator._thread.join(timeout=35)

    # ---- 指标摘要 ----
    print("\n[5] 指标摘要")
    print("-" * 40)

    summary = metrics_manager.get_summary()
    print(f"  当前时间: {summary['current_time']}")
    print()

    if 'counters' in summary:
        print("  Counters:")
        for name, values in summary['counters'].items():
            total = sum(v for v in values.values())
            print(f"    {name}: {total:.0f} total")

    if 'gauges' in summary:
        print("\n  Gauges:")
        for name, values in summary['gauges'].items():
            for key, val in values.items():
                if isinstance(key, tuple):
                    label_str = ", ".join(str(k) for k in key)
                    print(f"    {name}{{{label_str}}}: {val:.4f}")
                else:
                    print(f"    {name}: {val:.4f}")

    # ---- Grafana Dashboard JSON ----
    print("\n[6] Grafana Dashboard JSON (简化)")
    print("-" * 40)

    dashboard_json = GrafanaDashboardBuilder.build_rag_dashboard_json()
    dashboard_obj = json.loads(dashboard_json)
    print(f"  仪表板标题: {dashboard_obj['title']}")
    print(f"  面板数量: {len(dashboard_obj['panels'])}")
    for panel in dashboard_obj['panels']:
        print(f"    - [{panel['id']}] {panel['title']}")

    # ---- 告警规则 ----
    print("\n[7] 告警规则")
    print("-" * 40)

    alert_rules = GrafanaDashboardBuilder.build_alert_rules()
    for rule in alert_rules:
        severity_icon = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}.get(rule.severity, '⚪')
        print(f"  {severity_icon} {rule.alert_name}")
        print(f"      规则: {rule.expression}")
        print(f"      持续时间: {rule.for_duration}")
        if rule.runbook_url:
            print(f"      Runbook: {rule.runbook_url}")
        print()

    # ---- 完整Prometheus导出 ----
    print("\n[8] 完整Prometheus指标导出")
    print("-" * 40)

    full_metrics = prom_data
    # 统计各类型指标
    for keyword in ['# HELP', '# TYPE']:
        count = full_metrics.count(keyword)
        print(f"  {keyword}行数: {count}")

    # 保存到文件
    metrics_file = "prometheus_metrics.txt"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(full_metrics)
    print(f"\n  指标已导出到: {metrics_file}")

    # 保存Grafana Dashboard
    dashboard_file = "grafana_dashboard.json"
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(dashboard_json)
    print(f"  Dashboard已导出到: {dashboard_file}")

    print()
    print("=" * 60)
    print("  在线监控系统演示完成")
    print("=" * 60)
