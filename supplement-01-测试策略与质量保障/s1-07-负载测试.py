#!/usr/bin/env python3
"""
S1-07：负载测试 —— Locust + 并发模拟 + 容量规划
====================================================
RAG 系统的性能特征与普通 Web 应用不同：
- LLM 调用是高延迟（1-5秒）的 I/O 密集型操作
- 向量检索是带宽密集型操作
- 并发控制不当会导致 LLM API 过载或限流

本文件演示：
1. Locust 负载测试脚本
2. 并发用户行为模拟
3. P50/P95/P99 延迟基线
4. 容量规划公式
5. 自动扩容触发条件

依赖：pip install locust
"""

import time
import random
import statistics
from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import math


# ============================================================================
# 第一部分：Locust 负载测试脚本
# ============================================================================

# 以下是可以直接运行的 Locust 脚本
# 运行方式: locust -f s1-07-负载测试.py --host=http://localhost:8000

LOCUST_SCRIPT = '''
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import json
import random
import time

# 模拟真实用户问题
USER_QUERIES = [
    "公司的年假政策是什么？",
    "如何申请报销？",
    "产品支持哪些部署方式？",
    "系统的技术架构是怎样的？",
    "如何重置我的密码？",
    "最近的订单状态是什么？",
    "退款需要什么条件？",
    "如何联系技术支持？",
    "公司的培训政策是什么？",
    "如何查看我的考勤记录？",
]

class RAGUser(HttpUser):
    """
    模拟 RAG 系统的真实用户行为。

    用户行为模式：
    - 80% 的请求是知识库问答（快速响应）
    - 15% 的请求是复杂分析（慢速响应）
    - 5% 的请求是文档上传（极度慢速）
    """

    wait_time = between(2, 8)  # 真实用户思考时间 2-8 秒

    def on_start(self):
        """用户登录。"""
        self.client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "testpass",
        })

    @task(8)  # 权重 8/10 = 80%
    def simple_qa(self):
        """简单问答（最常见）。"""
        query = random.choice(USER_QUERIES)
        start = time.time()

        with self.client.post(
            "/api/chat",
            json={"query": query, "stream": False},
            catch_response=True,
            name="QA - 简单问答",
        ) as response:
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                response.success()
            elif response.status_code == 429:
                response.failure("Rate Limited")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1.5)  # 权重 1.5/10 = 15%
    def complex_analysis(self):
        """复杂分析（较慢）。"""
        complex_queries = [
            "分析AI技术对公司业务的影响和机遇",
            "对比不同产品线的市场表现和用户反馈",
            "评估最近三个月的客户满意度变化趋势",
        ]
        query = random.choice(complex_queries)

        with self.client.post(
            "/api/chat",
            json={"query": query, "detailed": True},
            catch_response=True,
            name="QA - 复杂分析",
        ) as response:
            if response.elapsed.total_seconds() > 10:
                response.failure("Response too slow (>10s)")
            elif response.status_code == 200:
                response.success()

    @task(0.5)  # 权重 0.5/10 = 5%
    def document_upload(self):
        """文档上传（很慢）。"""
        files = {"file": ("test.txt", b"测试文档内容")}

        with self.client.post(
            "/api/documents/upload",
            files=files,
            catch_response=True,
            name="Upload - 文档上传",
        ) as response:
            if response.elapsed.total_seconds() > 30:
                response.failure("Upload too slow (>30s)")
            elif response.status_code == 200:
                response.success()


# ---- 自定义事件：收集详细指标 ----

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始时初始化指标收集。"""
    print("负载测试开始...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束时打印汇总报告。"""
    print("\\n========== 负载测试汇总 ==========")

    stats = environment.stats
    for name, entry in stats.entries.items():
        print(f"\\n{name}:")
        print(f"  请求数: {entry.num_requests}")
        print(f"  失败数: {entry.num_failures}")
        print(f"  平均响应时间: {entry.avg_response_time:.0f}ms")
        print(f"  P50: {entry.get_response_time_percentile(0.5):.0f}ms")
        print(f"  P95: {entry.get_response_time_percentile(0.95):.0f}ms")
        print(f"  P99: {entry.get_response_time_percentile(0.99):.0f}ms")
        print(f"  RPS: {entry.total_rps:.1f}")
'''

print("=" * 60)
print("Locust 负载测试脚本")
print("=" * 60)
print(LOCUST_SCRIPT)


# ============================================================================
# 第二部分：延迟基线分析
# ============================================================================

class LatencyBaseline:
    """
    延迟基线分析器。

    从负载测试结果中提取 P50/P95/P99 基线，
    用于容量规划和自动扩容决策。
    """

    def __init__(self):
        self.samples: List[float] = []

    def add_sample(self, latency_ms: float):
        """添加延迟样本。"""
        self.samples.append(latency_ms)

    def percentile(self, p: float) -> float:
        """计算百分位数。"""
        if not self.samples:
            return 0
        sorted_samples = sorted(self.samples)
        idx = int(math.ceil(p * len(sorted_samples))) - 1
        return sorted_samples[max(0, min(idx, len(sorted_samples) - 1))]

    def get_baseline(self) -> dict:
        """获取延迟基线。"""
        if not self.samples:
            return {"error": "no samples"}

        return {
            "count": len(self.samples),
            "min": min(self.samples),
            "max": max(self.samples),
            "mean": statistics.mean(self.samples),
            "p50": self.percentile(0.50),
            "p90": self.percentile(0.90),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "std_dev": statistics.stdev(self.samples) if len(self.samples) > 1 else 0,
        }

    def check_degradation(self, new_baseline: dict,
                          max_p99_increase_pct: float = 0.20) -> bool:
        """
        检查新版本是否有性能退化。
        P99 延迟增加超过 20% → 性能退化告警。
        """
        current_p99 = self.percentile(0.99)
        if current_p99 == 0:
            return False
        increase = (new_baseline["p99"] - current_p99) / current_p99
        return increase > max_p99_increase_pct


# ============================================================================
# 第三部分：容量规划
# ============================================================================

class CapacityPlanner:
    """
    RAG 系统容量规划器。

    核心公式：
    1. 所需并发数 = 峰值 QPS × (平均响应时间 / 1000)
    2. 所需 GPU 数量 = 并发数 / 单 GPU 支持并发数
    3. LLM API 成本 = QPS × 单次调用 token 数 × 单价
    """

    def __init__(self):
        pass

    def concurrent_users_required(
        self,
        peak_qps: float,
        avg_response_time_ms: float,
    ) -> float:
        """
        计算所需并发用户数。

        公式: C = Q × R
        C = 并发数, Q = QPS, R = 响应时间(秒)

        例: 100 QPS × 2秒响应 = 需要 200 并发
        """
        return peak_qps * (avg_response_time_ms / 1000)

    def gpu_instances_required(
        self,
        concurrent_users: float,
        queries_per_gpu_per_second: float,
    ) -> int:
        """
        计算所需 GPU 实例数。

        GPU 预热策略：
        - 冷启动(0→100%): 需要提前 5-10 分钟波次扩容
        - 温启动(50%→100%): 可即时响应
        """
        return math.ceil(concurrent_users / queries_per_gpu_per_second)

    def llm_daily_cost(
        self,
        daily_queries: int,
        tokens_per_query: int,
        price_per_1m_tokens: float,
    ) -> float:
        """计算 LLM 日成本。"""
        total_tokens = daily_queries * tokens_per_query
        return (total_tokens / 1_000_000) * price_per_1m_tokens

    def storage_required(
        self,
        num_documents: int,
        avg_tokens_per_doc: int,
        vector_dim: int = 1536,
        overhead_factor: float = 1.3,
    ) -> float:
        """
        计算向量存储所需空间。

        每个向量: vector_dim × 4 bytes (float32)
        每个文档: N chunks × vector_dim × 4 bytes
        """
        avg_chunks_per_doc = max(1, avg_tokens_per_doc / 500)
        bytes_per_vector = vector_dim * 4
        raw_gb = (num_documents * avg_chunks_per_doc * bytes_per_vector) / (1024**3)
        return raw_gb * overhead_factor

    def plan(self, params: dict) -> dict:
        """
        执行完整容量规划。

        params = {
            "peak_qps": 500,
            "avg_response_time_ms": 2000,
            "daily_queries": 100_000,
            "avg_tokens_per_query": 800,
            "llm_price_per_1m_tokens": 2.50,
            "num_documents": 100_000,
            "avg_tokens_per_doc": 1000,
            "queries_per_gpu_per_second": 20,
        }
        """
        concurrent = self.concurrent_users_required(
            params["peak_qps"],
            params["avg_response_time_ms"],
        )
        gpus = self.gpu_instances_required(
            concurrent,
            params["queries_per_gpu_per_second"],
        )
        daily_llm_cost = self.llm_daily_cost(
            params["daily_queries"],
            params["avg_tokens_per_query"],
            params["llm_price_per_1m_tokens"],
        )
        storage_gb = self.storage_required(
            params["num_documents"],
            params["avg_tokens_per_doc"],
        )

        return {
            "required_concurrent_users": round(concurrent),
            "recommended_gpu_instances": gpus,
            "estimated_daily_llm_cost": round(daily_llm_cost, 2),
            "required_storage_gb": round(storage_gb, 1),
            "recommended_total_ram_gb": round(storage_gb * 2.5, 1),
        }


# ============================================================================
# 第四部分：HPA 自动扩容策略
# ============================================================================

def hpa_config_template():
    """Kubernetes HPA 自动扩容配置。"""
    print("\n" + "=" * 60)
    print("【Kubernetes HPA 自动扩容配置】")
    print("=" * 60)

    config = '''
# Horizontal Pod Autoscaler 配置

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api

  minReplicas: 2      # 最小副本数（保证高可用）
  maxReplicas: 20     # 最大副本数（成本上限）

  metrics:
    # 指标1：CPU 使用率
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70

    # 指标2：内存使用率
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80

    # 指标3：自定义指标 - LLM API 请求队列深度
    - type: Pods
      pods:
        metric:
          name: llm_request_queue_depth
        target:
          type: AverageValue
          averageValue: "50"   # 队列超过50 → 扩容

    # 指标4：自定义指标 - P99 响应时间
    - type: Pods
      pods:
        metric:
          name: rag_response_time_p99_ms
        target:
          type: AverageValue
          averageValue: "3000"  # P99超过3秒 → 扩容

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60  # 扩容前观察60秒
      policies:
        - type: Percent
          value: 50                    # 每次扩容最多50%
          periodSeconds: 60
        - type: Pods
          value: 4                     # 每次扩容最多4个Pod
          periodSeconds: 60
      selectPolicy: Min                # 选变化较小的策略

    scaleDown:
      stabilizationWindowSeconds: 300  # 缩容前观察5分钟
      policies:
        - type: Percent
          value: 10                    # 每次缩容最多10%
          periodSeconds: 60
'''
    print(config)


# ============================================================================
# 第五部分：演示
# ============================================================================

def demo():
    """负载测试和容量规划演示。"""
    print("=" * 60)
    print("容量规划演示")
    print("=" * 60)

    # 延迟基线
    baseline = LatencyBaseline()
    # 模拟1000次调用的延迟数据
    random.seed(42)
    for _ in range(1000):
        # 模拟正态分布的延迟：均值2000ms，标准差500ms
        latency = max(100, random.gauss(2000, 500))
        baseline.add_sample(latency)

    bl = baseline.get_baseline()
    print("\n  [模拟延迟基线] (1000次调用)")
    print(f"    P50: {bl['p50']:.0f}ms")
    print(f"    P95: {bl['p95']:.0f}ms")
    print(f"    P99: {bl['p99']:.0f}ms")
    print(f"    均值: {bl['mean']:.0f}ms")
    print(f"    标准差: {bl['std_dev']:.0f}ms")

    # 容量规划
    planner = CapacityPlanner()
    plan = planner.plan({
        "peak_qps": 500,
        "avg_response_time_ms": 2000,
        "daily_queries": 100_000,
        "avg_tokens_per_query": 800,
        "llm_price_per_1m_tokens": 2.50,
        "num_documents": 100_000,
        "avg_tokens_per_doc": 1000,
        "queries_per_gpu_per_second": 20,
    })

    print("\n  [容量规划结果]")
    print(f"    所需并发: {plan['required_concurrent_users']}")
    print(f"    推荐 GPU 实例: {plan['recommended_gpu_instances']}")
    print(f"    预估 LLM 日成本: ${plan['estimated_daily_llm_cost']}")
    print(f"    所需存储: {plan['required_storage_gb']} GB")
    print(f"    推荐总内存: {plan['recommended_total_ram_gb']} GB")

    # 大促场景（10倍流量）
    plan_promo = planner.plan({
        "peak_qps": 5000,
        "avg_response_time_ms": 2000,
        "daily_queries": 1_000_000,
        "avg_tokens_per_query": 800,
        "llm_price_per_1m_tokens": 2.50,
        "num_documents": 100_000,
        "avg_tokens_per_doc": 1000,
        "queries_per_gpu_per_second": 20,
    })

    print("\n  [大促场景 ×10流量]")
    print(f"    所需并发: {plan_promo['required_concurrent_users']}")
    print(f"    推荐 GPU 实例: {plan_promo['recommended_gpu_instances']}")
    print(f"    预估 LLM 日成本: ${plan_promo['estimated_daily_llm_cost']}")

    hpa_config_template()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("负载测试与容量规划")
    print("=" * 60)
    print("Locust 脚本已在上面输出。运行方式：")
    print("  locust -f s1-07-负载测试.py --host=http://localhost:8000\n")

    demo()

    print("\n[完成] 负载测试演示结束。")
    print("  [提示] 生产环境建议:")
    print("    1. 每次大版本发布前运行负载测试")
    print("    2. 与基线对比 P99 延迟变化")
    print("    3. 大促前按 10× 容量规划")
    print("    4. 自动扩容配置应保守（宁可多扩容，不可响应超时）")
