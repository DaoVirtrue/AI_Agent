#!/usr/bin/env python3
"""
索引重建Cron任务 (Weekly HNSW Rebuild with Blue-Green Deployment)

特性:
  - 每周Cron定时重建HNSW索引
  - 蓝绿部署: 构建新索引(staging) → 验证 → 切换
  - 指标对比: 新索引 vs 当前索引
  - 改进则切换，退化则保留并告警
  - 漂移检测: 重建前检测索引质量漂移
"""

import os
import json
import time
import random
import hashlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Callable
from collections import defaultdict

import numpy as np


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class IndexStatus(Enum):
    """索引状态"""
    ACTIVE = "active"           # 活跃（当前服务）
    BUILDING = "building"       # 构建中
    STAGING = "staging"         # 待验证
    VALIDATED = "validated"     # 已验证
    SWAPPED = "swapped"         # 已切换
    DEPRECATED = "deprecated"   # 已废弃


class RebuildDecision(Enum):
    """重建决策"""
    SWAP = "swap"                 # 切换到新索引
    KEEP_CURRENT = "keep_current" # 保留当前索引
    ALERT = "alert"               # 告警（指标退化严重）


@dataclass
class IndexMetrics:
    """索引指标"""
    index_id: str
    total_vectors: int
    index_size_mb: float
    build_time_seconds: float

    # 查询性能指标
    avg_query_latency_ms: float
    p99_query_latency_ms: float
    queries_per_second: float

    # 质量指标
    avg_recall_at_10: float       # Recall@10
    avg_mrr: float                # 平均倒数排名 (MRR)
    avg_ndcg_at_10: float         # NDCG@10
    empty_result_rate: float      # 空结果率

    # 结构指标
    avg_degree: float             # 平均节点度
    max_degree: int               # 最大节点度
    connectivity: float           # 图连通性
    dead_ends: int                # 死端节点数

    @staticmethod
    def from_dict(data: dict) -> "IndexMetrics":
        return IndexMetrics(**data)

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class DriftReport:
    """漂移检测报告"""
    detected: bool
    drift_score: float            # 0.0-1.0 (越高越漂移)
    metrics_affected: list[str]   # 受影响的指标
    recommendations: list[str]
    timestamp: float


@dataclass
class ComparisonResult:
    """指标对比结果"""
    metric_name: str
    current_value: float
    new_value: float
    delta_pct: float
    improved: bool
    significant: bool  # 是否显著变化 (>5%)


@dataclass
class RebuildReport:
    """重建报告"""
    rebuild_id: str
    started_at: float
    completed_at: float
    decision: RebuildDecision
    current_metrics: IndexMetrics
    new_metrics: IndexMetrics
    comparisons: list[ComparisonResult]
    swapped: bool
    alerts_fired: list[str]


# ============================================================
# HNSW索引构建器 (Staging Index Builder)
# ============================================================

class HNSWRebuilder:
    """HNSW索引重建器"""

    def __init__(self, dimension: int = 384, M: int = 16, ef_construction: int = 200):
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self._index = None
        self._id_map: dict[int, str] = {}

    def build(self, vectors: list[list[float]], ids: list[str],
              progress_callback: Optional[Callable] = None) -> IndexMetrics:
        """构建新HNSW索引

        Args:
            vectors: 向量数据
            ids: 对应的ID列表
            progress_callback: 进度回调(percent, message)

        Returns:
            IndexMetrics: 构建完成的索引指标
        """
        t_start = time.time()
        rebuild_id = f"hnsw_{int(t_start)}_{hashlib.md5(str(t_start).encode()).hexdigest()[:6]}"

        print(f"[Rebuild] 开始构建HNSW索引: {rebuild_id}")
        print(f"[Rebuild] 向量数: {len(vectors)}, 维度: {self.dimension}")

        if progress_callback:
            progress_callback(0.0, "初始化索引结构")

        try:
            import hnswlib

            self._index = hnswlib.Index(space="cosine", dim=self.dimension)
            self._index.init_index(
                max_elements=len(vectors),
                ef_construction=self.ef_construction,
                M=self.M,
            )

            if progress_callback:
                progress_callback(0.2, "添加向量到索引")

            # 批量添加
            batch_size = 1000
            vectors_np = np.array(vectors)
            indices = np.arange(len(vectors))

            for start in range(0, len(vectors), batch_size):
                end = min(start + batch_size, len(vectors))
                self._index.add_items(vectors_np[start:end], indices[start:end])

                for i in range(start, end):
                    self._id_map[int(indices[i])] = ids[i]

                if progress_callback:
                    progress_callback(
                        0.2 + 0.6 * (end / len(vectors)),
                        f"添加向量 {end}/{len(vectors)}"
                    )

            self._index.set_ef(50)

        except ImportError:
            # 降级: 模拟构建
            print("[Rebuild] hnswlib未安装，使用模拟索引")
            if progress_callback:
                for p in [0.3, 0.6, 0.9]:
                    progress_callback(p, f"模拟构建 {p:.0%}")
                    time.sleep(0.05)
            self._index = _MockHNSWIndex(vectors, self.dimension)

        build_time = time.time() - t_start

        if progress_callback:
            progress_callback(1.0, "索引构建完成")

        # 计算指标
        metrics = self._compute_build_metrics(rebuild_id, len(vectors), build_time)

        print(f"[Rebuild] 构建完成: {build_time:.1f}s")
        print(f"[Rebuild] 索引大小: {metrics.index_size_mb:.1f}MB")

        return metrics

    def search(self, query_vector: list[float], k: int = 10) -> list[tuple[str, float]]:
        """在新索引上搜索"""
        if self._index is None:
            return []

        query_np = np.array(query_vector)

        try:
            import hnswlib
            if isinstance(self._index, hnswlib.Index):
                labels, distances = self._index.knn_query(query_np.reshape(1, -1), k=k)
                return [(self._id_map.get(int(l), "unknown"), float(d))
                        for l, d in zip(labels[0], distances[0])]
        except ImportError:
            pass

        # 模拟搜索
        results = []
        for i in range(min(k, 10)):
            results.append((f"doc_{i:04d}", random.uniform(0.1, 0.9)))
        return results

    def _compute_build_metrics(self, index_id: str, total_vectors: int,
                               build_time: float) -> IndexMetrics:
        """计算构建指标"""
        # 估算索引大小
        index_size_mb = total_vectors * self.dimension * 4 / (1024 * 1024) * 1.2  # 1.2x overhead

        # 模拟查询测试
        simulated_latencies = [random.uniform(2.0, 15.0) for _ in range(100)]
        simulated_latencies.sort()

        # 模拟质量指标
        return IndexMetrics(
            index_id=index_id,
            total_vectors=total_vectors,
            index_size_mb=index_size_mb,
            build_time_seconds=build_time,
            avg_query_latency_ms=np.mean(simulated_latencies),
            p99_query_latency_ms=simulated_latencies[int(99 * len(simulated_latencies) / 100)],
            queries_per_second=1000.0 / max(np.mean(simulated_latencies), 1),
            avg_recall_at_10=random.uniform(0.85, 0.98),
            avg_mrr=random.uniform(0.70, 0.95),
            avg_ndcg_at_10=random.uniform(0.75, 0.96),
            empty_result_rate=random.uniform(0.0, 0.05),
            avg_degree=random.uniform(10, 20),
            max_degree=random.randint(30, 60),
            connectivity=random.uniform(0.90, 1.0),
            dead_ends=random.randint(0, 10),
        )


class _MockHNSWIndex:
    """模拟HNSW索引"""

    def __init__(self, vectors: list, dimension: int):
        self.vectors = vectors
        self.dimension = dimension

    def knn_query(self, query, k):
        # 模拟最近邻
        scores = []
        for i, vec in enumerate(self.vectors):
            score = np.dot(np.array(vec), np.array(query).flatten())
            scores.append((i, 1.0 - score))
        scores.sort(key=lambda x: x[1])
        top = scores[:k]
        return (
            np.array([[t[0] for t in top]]),
            np.array([[t[1] for t in top]])
        )


# ============================================================
# 指标对比器 (Metrics Comparator)
# ============================================================

class MetricsComparator:
    """指标对比器 - 比较新旧索引指标"""

    IMPROVEMENT_THRESHOLD = 0.05  # 5% 显著变化阈值

    @staticmethod
    def compare(current: IndexMetrics, new: IndexMetrics) -> list[ComparisonResult]:
        """对比两个索引的所有指标"""
        comparisons = []

        metric_pairs = [
            ("avg_query_latency_ms", "平均查询延迟", False),  # False = 越低越好
            ("p99_query_latency_ms", "P99查询延迟", False),
            ("queries_per_second", "每秒查询数", True),       # True = 越高越好
            ("avg_recall_at_10", "Recall@10", True),
            ("avg_mrr", "MRR", True),
            ("avg_ndcg_at_10", "NDCG@10", True),
            ("empty_result_rate", "空结果率", False),
            ("avg_degree", "平均度", True),
            ("connectivity", "连通性", True),
            ("dead_ends", "死端节点数", False),
        ]

        for metric_key, metric_name, higher_is_better in metric_pairs:
            current_val = getattr(current, metric_key, 0)
            new_val = getattr(new, metric_key, 0)

            if current_val == 0:
                continue

            delta_pct = (new_val - current_val) / abs(current_val) * 100

            if higher_is_better:
                improved = new_val > current_val
            else:
                improved = new_val < current_val

            significant = abs(delta_pct) > MetricsComparator.IMPROVEMENT_THRESHOLD * 100

            comparisons.append(ComparisonResult(
                metric_name=metric_name,
                current_value=current_val,
                new_value=new_val,
                delta_pct=delta_pct,
                improved=improved,
                significant=significant,
            ))

        return comparisons

    @staticmethod
    def decide(comparisons: list[ComparisonResult]) -> tuple[RebuildDecision, list[str]]:
        """根据对比结果做出决策

        Returns:
            (决策, 告警消息列表)
        """
        alerts = []

        improved_count = sum(1 for c in comparisons if c.improved and c.significant)
        degraded_count = sum(1 for c in comparisons if not c.improved and c.significant)

        # 加权评分
        score = 0
        critical_metrics = {"Recall@10", "MRR", "NDCG@10", "平均查询延迟"}
        for c in comparisons:
            weight = 2.0 if c.metric_name in critical_metrics else 1.0
            if c.improved:
                score += weight * (abs(c.delta_pct) if c.significant else 0)
            else:
                score -= weight * (abs(c.delta_pct) if c.significant else 0)

        print(f"[Compare] 改进指标: {improved_count}, 退化指标: {degraded_count}, 评分: {score:.1f}")

        if score > 5:
            return RebuildDecision.SWAP, alerts
        elif score < -10:
            alerts.append(f"严重退化! 评分={score:.1f}, 改进={improved_count}, 退化={degraded_count}")
            return RebuildDecision.ALERT, alerts
        else:
            alerts.append(f"变化不显著，评分={score:.1f}")
            return RebuildDecision.KEEP_CURRENT, alerts


# ============================================================
# 漂移检测器 (Drift Detector)
# ============================================================

class DriftDetector:
    """索引质量漂移检测器"""

    def __init__(self, window_size: int = 10, drift_threshold: float = 0.1):
        """
        Args:
            window_size: 漂移检测窗口大小
            drift_threshold: 漂移阈值
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self._metric_history: list[dict] = []

    def record(self, metrics: IndexMetrics):
        """记录当前指标"""
        self._metric_history.append({
            "timestamp": time.time(),
            "recall": metrics.avg_recall_at_10,
            "mrr": metrics.avg_mrr,
            "ndcg": metrics.avg_ndcg_at_10,
            "latency": metrics.avg_query_latency_ms,
            "empty_rate": metrics.empty_result_rate,
        })

        # 保持窗口大小
        while len(self._metric_history) > self.window_size:
            self._metric_history.pop(0)

    def detect(self) -> DriftReport:
        """检测漂移"""
        if len(self._metric_history) < 3:
            return DriftReport(
                detected=False,
                drift_score=0.0,
                metrics_affected=[],
                recommendations=["数据点不足，无法检测漂移"],
                timestamp=time.time(),
            )

        # 取前一半和后一半的均值对比
        mid = len(self._metric_history) // 2
        first_half = self._metric_history[:mid]
        second_half = self._metric_history[mid:]

        drift_scores = {}
        for metric_key in ["recall", "mrr", "ndcg", "latency", "empty_rate"]:
            first_vals = [h[metric_key] for h in first_half]
            second_vals = [h[metric_key] for h in second_half]

            first_mean = np.mean(first_vals)
            second_mean = np.mean(second_vals)

            if first_mean > 0:
                drift = abs(second_mean - first_mean) / first_mean
                drift_scores[metric_key] = drift

        overall_drift = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        detected = overall_drift > self.drift_threshold

        affected = [
            m for m, score in drift_scores.items()
            if score > self.drift_threshold
        ]

        recommendations = []
        if detected:
            if "recall" in affected or "mrr" in affected:
                recommendations.append("检索质量下降，建议重建索引")
            if "latency" in affected:
                recommendations.append("查询延迟增加，检查索引参数")
            if "empty_rate" in affected:
                recommendations.append("空结果率上升，检查数据覆盖")

        return DriftReport(
            detected=detected,
            drift_score=overall_drift,
            metrics_affected=affected,
            recommendations=recommendations if recommendations else ["指标稳定，无需操作"],
            timestamp=time.time(),
        )


# ============================================================
# 蓝绿部署管理器 (Blue-Green Deployment Manager)
# ============================================================

class BlueGreenDeployer:
    """蓝绿部署管理器"""

    def __init__(self):
        self._active_index: Any = None       # 绿色（当前活跃）
        self._staging_index: Any = None      # 蓝色（新建/待切换）
        self._active_id: str = ""
        self._staging_id: str = ""
        self._lock = threading.RLock()

    def deploy_staging(self, rebuilder: HNSWRebuilder, staging_id: str):
        """部署到staging"""
        with self._lock:
            self._staging_index = rebuilder
            self._staging_id = staging_id
        print(f"[BlueGreen] Staging就绪: {staging_id}")

    def swap(self) -> bool:
        """蓝绿切换"""
        with self._lock:
            if self._staging_index is None:
                print("[BlueGreen] 无Staging索引，无法切换")
                return False

            old_active_id = self._active_id
            old_active_index = self._active_index

            # 切换
            self._active_index = self._staging_index
            self._active_id = self._staging_id

            # 旧索引变为staging（用于回滚）
            self._staging_index = old_active_index
            self._staging_id = old_active_id

            print(f"[BlueGreen] 蓝绿切换完成:")
            print(f"  新活跃 (绿): {self._active_id}")
            print(f"  旧活跃 (蓝): {self._staging_id}")
            return True

    def rollback(self) -> bool:
        """回滚到之前的索引"""
        return self.swap()  # 再次交换即为回滚

    def get_active(self) -> tuple[Any, str]:
        """获取当前活跃索引"""
        return self._active_index, self._active_id


# ============================================================
# 验证查询集 (Validation Query Set)
# ============================================================

class ValidationQuerySet:
    """验证查询集 - 用于对比新旧索引质量"""

    def __init__(self):
        self._queries: list[dict] = []

    def load_standard_queries(self):
        """加载标准验证查询集"""
        self._queries = [
            {
                "query": "什么是RAG系统？",
                "expected_chunks": ["rag_basics_001", "rag_overview_002"],
                "category": "basic",
            },
            {
                "query": "向量数据库的选型标准有哪些？",
                "expected_chunks": ["vector_db_001", "db_selection_003"],
                "category": "technical",
            },
            {
                "query": "如何优化检索速度？",
                "expected_chunks": ["perf_optim_001", "cache_strategy_002"],
                "category": "technical",
            },
            {
                "query": "熔断器模式的工作原理？",
                "expected_chunks": ["circuit_breaker_001"],
                "category": "architecture",
            },
            {
                "query": "多租户隔离的最佳实践？",
                "expected_chunks": ["multitenant_001", "isolation_002"],
                "category": "architecture",
            },
        ]
        return self._queries

    def add_query(self, query: str, expected_chunks: list[str], category: str = "custom"):
        """添加自定义验证查询"""
        self._queries.append({
            "query": query,
            "expected_chunks": expected_chunks,
            "category": category,
        })

    def evaluate_index(self, search_fn: Callable, k: int = 10) -> dict:
        """评估索引在验证集上的表现

        Args:
            search_fn: 搜索函数 (query, k) -> [chunk_id, ...]
            k: 检索数量

        Returns:
            {"recall@k", "mrr", "ndcg@k"} 指标
        """
        recalls = []
        mrrs = []
        ndcgs = []

        for q in self._queries:
            results = search_fn(q["query"], k)
            result_ids = [r[0] if isinstance(r, tuple) else r for r in results]
            expected = q["expected_chunks"]

            # Recall@k
            hits = sum(1 for eid in expected if eid in result_ids)
            recall = hits / len(expected) if expected else 0.0
            recalls.append(recall)

            # MRR
            for rank, rid in enumerate(result_ids, 1):
                if rid in expected:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)

            # NDCG@k (简化)
            dcg = sum(
                (1.0 if rid in expected else 0.0) / np.log2(rank + 1)
                for rank, rid in enumerate(result_ids[:k], 1)
            )
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        return {
            "avg_recall": np.mean(recalls) if recalls else 0.0,
            "avg_mrr": np.mean(mrrs) if mrrs else 0.0,
            "avg_ndcg": np.mean(ndcgs) if ndcgs else 0.0,
            "total_queries": len(self._queries),
        }


# ============================================================
# 重建编排器 (Rebuild Orchestrator)
# ============================================================

class RebuildOrchestrator:
    """索引重建编排器 - 整合所有组件"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.rebuilder = HNSWRebuilder(dimension=dimension)
        self.comparator = MetricsComparator()
        self.drift_detector = DriftDetector()
        self.deployer = BlueGreenDeployer()
        self.validator = ValidationQuerySet()
        self.validator.load_standard_queries()

        self._current_metrics: Optional[IndexMetrics] = None
        self._rebuild_history: list[RebuildReport] = []

    def weekly_rebuild(self, vectors: list[list[float]], ids: list[str],
                       progress_callback: Optional[Callable] = None) -> RebuildReport:
        """执行每周重建流程

        流程:
        1. 漂移检测
        2. 构建Staging索引
        3. 验证查询
        4. 指标对比
        5. 决策（交换/保留/告警）
        6. 执行蓝绿部署
        """
        rebuild_id = f"rebuild_{int(time.time())}_{random.randint(1000, 9999)}"
        t_start = time.time()

        print(f"\n{'='*60}")
        print(f"每周索引重建: {rebuild_id}")
        print(f"{'='*60}")

        # 步骤0: 漂移检测
        print("\n[Step 0] 漂移检测")
        drift_report = self.drift_detector.detect()
        print(f"  漂移: {'检测到' if drift_report.detected else '未检测到'} (分数: {drift_report.drift_score:.3f})")
        if drift_report.recommendations:
            for rec in drift_report.recommendations:
                print(f"  建议: {rec}")

        # 步骤1: 构建Staging索引
        print("\n[Step 1] 构建Staging索引")
        new_metrics = self.rebuilder.build(
            vectors, ids,
            progress_callback=progress_callback,
        )

        # 步骤2: 验证查询
        print("\n[Step 2] 验证查询集评估")
        def search_fn(query, k):
            # 使用新索引搜索
            try:
                import numpy as np
                query_vec = np.random.randn(self.dimension).tolist()
                return self.rebuilder.search(query_vec, k)
            except Exception:
                return [(f"sim_{i}", random.uniform(0.1, 0.9)) for i in range(k)]

        validation_scores = self.validator.evaluate_index(search_fn)
        print(f"  Recall@10: {validation_scores['avg_recall']:.3f}")
        print(f"  MRR:       {validation_scores['avg_mrr']:.3f}")
        print(f"  NDCG@10:   {validation_scores['avg_ndcg']:.3f}")

        # 用验证集分数更新new_metrics
        new_metrics.avg_recall_at_10 = validation_scores["avg_recall"]
        new_metrics.avg_mrr = validation_scores["avg_mrr"]
        new_metrics.avg_ndcg_at_10 = validation_scores["avg_ndcg"]

        # 步骤3: 指标对比
        print("\n[Step 3] 指标对比")
        if self._current_metrics:
            comparisons = self.comparator.compare(self._current_metrics, new_metrics)
            for comp in comparisons:
                icon = "↑" if comp.improved else "↓"
                sig = " *显著*" if comp.significant else ""
                print(f"  {icon} {comp.metric_name}: {comp.current_value:.3f} → {comp.new_value:.3f} ({comp.delta_pct:+.1f}%){sig}")
        else:
            # 首次构建
            comparisons = []
            print("  首次构建，无对比基准")
            # 设置基准
            self._current_metrics = new_metrics

        # 步骤4: 决策
        print("\n[Step 4] 决策")
        if comparisons:
            decision, alerts = self.comparator.decide(comparisons)
        else:
            decision = RebuildDecision.SWAP  # 首次必然交换
            alerts = []

        print(f"  决策: {decision.value}")
        for alert in alerts:
            print(f"  告警: {alert}")

        # 步骤5: 执行部署
        swapped = False
        if decision == RebuildDecision.SWAP:
            print("\n[Step 5] 蓝绿交换")
            self.deployer.deploy_staging(self.rebuilder, new_metrics.index_id)
            swapped = self.deployer.swap()
            if swapped:
                self._current_metrics = new_metrics
                self.drift_detector.record(new_metrics)
        elif decision == RebuildDecision.KEEP_CURRENT:
            print("\n[Step 5] 保留当前索引")
        else:  # ALERT
            print("\n[Step 5] 告警 - 保留当前索引并通知运维")
            alarms_fired = alerts
            # 生产环境: 发送PagerDuty/钉钉/邮件告警

        # 生成报告
        report = RebuildReport(
            rebuild_id=rebuild_id,
            started_at=t_start,
            completed_at=time.time(),
            decision=decision,
            current_metrics=self._current_metrics or new_metrics,
            new_metrics=new_metrics,
            comparisons=comparisons,
            swapped=swapped,
            alerts_fired=alerts,
        )

        self._rebuild_history.append(report)
        return report

    def get_rebuild_history(self) -> list[dict]:
        """获取重建历史"""
        return [
            {
                "rebuild_id": r.rebuild_id,
                "decision": r.decision.value,
                "swapped": r.swapped,
                "duration": f"{r.completed_at - r.started_at:.0f}s",
                "comparisons": len(r.comparisons),
                "alerts": len(r.alerts_fired),
            }
            for r in self._rebuild_history
        ]


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("索引重建Cron任务 - 演示运行")
    print("=" * 60)

    # 1. 初始化编排器
    orchestrator = RebuildOrchestrator(dimension=384)

    # 2. 生成模拟向量数据
    print("\n--- 准备向量数据 ---")
    num_vectors = 1000
    test_vectors = np.random.randn(num_vectors, 384).tolist()
    test_ids = [f"chunk_{i:05d}" for i in range(num_vectors)]
    print(f"向量数: {num_vectors}, 维度: 384")

    # 3. 注册进度回调
    def progress(percent: float, message: str):
        bar = "█" * int(percent * 20) + "░" * (20 - int(percent * 20))
        print(f"\r  [{bar}] {percent:.0%} {message}", end="", flush=True)

    # 4. 执行首次重建（无基准对比）
    print("\n\n--- 第1次重建（初始） ---")
    report1 = orchestrator.weekly_rebuild(test_vectors, test_ids, progress_callback=progress)
    print(f"\n  重建ID: {report1.rebuild_id}")
    print(f"  决策: {report1.decision.value}")
    print(f"  已交换: {report1.swapped}")

    # 5. 记录指标（模拟使用后指标变化）
    print("\n--- 模拟指标漂移 ---")
    for i in range(6):
        mock_metrics = IndexMetrics(
            index_id=f"mock_{i}",
            total_vectors=1000,
            index_size_mb=12.5,
            build_time_seconds=3.0,
            avg_query_latency_ms=8.0 + i * 0.5,
            p99_query_latency_ms=25.0 + i,
            queries_per_second=120.0 - i * 5,
            avg_recall_at_10=0.95 - i * 0.02,
            avg_mrr=0.88 - i * 0.02,
            avg_ndcg_at_10=0.90 - i * 0.02,
            empty_result_rate=0.01 + i * 0.01,
            avg_degree=15.0,
            max_degree=40,
            connectivity=0.97 - i * 0.01,
            dead_ends=i * 2,
        )
        orchestrator.drift_detector.record(mock_metrics)

    drift = orchestrator.drift_detector.detect()
    print(f"漂移检测: {drift.detected} (分数: {drift.drift_score:.3f})")
    print(f"受影响指标: {drift.metrics_affected}")
    for rec in drift.recommendations:
        print(f"  建议: {rec}")

    # 6. 模拟添加更多向量后的重建（模拟索引增长）
    print("\n\n--- 第2次重建（新增数据后） ---")
    num_vectors2 = 1100  # 增加了100个向量
    test_vectors2 = np.random.randn(num_vectors2, 384).tolist()
    test_ids2 = [f"chunk_{i:05d}" for i in range(num_vectors2)]

    report2 = orchestrator.weekly_rebuild(test_vectors2, test_ids2)
    print(f"\n  重建ID: {report2.rebuild_id}")
    print(f"  决策: {report2.decision.value}")
    print(f"  已交换: {report2.swapped}")
    print(f"  对比项: {len(report2.comparisons)}")
    print(f"  告警: {len(report2.alerts_fired)}")

    # 7. 重建历史
    print("\n--- 重建历史 ---")
    for h in orchestrator.get_rebuild_history():
        print(f"  {h['rebuild_id']} | {h['decision']} | 交换={h['swapped']} | {h['duration']}")

    # 8. 验证查询集
    print("\n--- 验证查询集 ---")
    queries = orchestrator.validator._queries
    for q in queries:
        print(f"  [{q['category']}] {q['query'][:40]}... (期望: {len(q['expected_chunks'])} 块)")

    # 9. 蓝绿部署状态
    print("\n--- 蓝绿部署状态 ---")
    active_idx, active_id = orchestrator.deployer.get_active()
    print(f"  活跃索引: {active_id}")
    staging_idx, staging_id = orchestrator.deployer._staging_index, orchestrator.deployer._staging_id
    print(f"  Staging:  {staging_id or '无'}")

    print("\n" + "=" * 60)
    print("索引重建Cron演示完成！")
    print("=" * 60)
