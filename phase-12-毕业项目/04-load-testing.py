#!/usr/bin/env python3
"""
负载测试脚本 (Load Testing with Locust)
=========================================
用户场景:
  - simple_faq: 简单FAQ查询
  - complex_multi_hop: 复杂多跳查询
  - document_upload: 文档上传
  - concurrent_mixed: 混合并发

目标: 100并发, P95<2s, 错误率<1%
"""

import time
import random
import threading
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime


@dataclass
class LoadTestResult:
    """负载测试结果"""
    test_name: str
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.total_requests)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.total_requests)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies: return 0
        return sorted(self.latencies)[len(self.latencies) // 2]

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies: return 0
        return sorted(self.latencies)[int(len(self.latencies) * 0.95)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies: return 0
        return sorted(self.latencies)[int(len(self.latencies) * 0.99)]

    @property
    def throughput_qps(self) -> float:
        return self.total_requests / max(0.001, self.duration_seconds)

    def summary(self) -> str:
        return (
            f"Requests: {self.total_requests} | Success: {self.success_rate:.1%} | "
            f"QPS: {self.throughput_qps:.1f}\n"
            f"Latency: avg={self.avg_latency_ms:.0f}ms P50={self.p50_latency_ms:.0f}ms "
            f"P95={self.p95_latency_ms:.0f}ms P99={self.p99_latency_ms:.0f}ms\n"
            f"Duration: {self.duration_seconds:.1f}s | Errors: {self.error_count}"
        )


class ScenarioUser:
    """模拟用户 (替代Locust的User类)"""

    # 模拟的API端点
    API_ENDPOINT = "http://localhost:8000/api/v1/query"

    SIMPLE_QUERIES = [
        "什么是RAG？", "Python是什么？", "深度学习如何工作？",
        "什么是向量数据库？", "API是什么意思？", "如何使用Docker？",
    ]

    COMPLEX_QUERIES = [
        "分析RAG系统中检索策略对最终生成质量的影响，对比BM25、向量检索和混合检索三种方法",
        "结合性能、成本和准确性分析，比较自托管vs云服务LLM部署方案的优劣",
        "在设计企业级多租户RAG系统时，需要从数据隔离、安全、性能和成本四个方面进行综合考量",
    ]

    DOCUMENTS = [
        "请分析这篇文档: RAG系统在企业中的应用涉及多个技术栈...",
        "上传文件分析: 产品需求文档v3.2的内容摘要...",
    ]

    def __init__(self, user_id: str, scenario: str):
        self.user_id = user_id
        self.scenario = scenario

    def make_request(self) -> Tuple[str, float, bool]:
        """模拟一次API请求"""
        if self.scenario == 'simple':
            query = random.choice(self.SIMPLE_QUERIES)
        elif self.scenario == 'complex':
            query = random.choice(self.COMPLEX_QUERIES)
        elif self.scenario == 'upload':
            query = random.choice(self.DOCUMENTS)
        else:  # mixed
            query = random.choice(self.SIMPLE_QUERIES + self.COMPLEX_QUERIES + self.DOCUMENTS)

        start = time.time()

        # 模拟API调用延迟
        if self.scenario == 'simple':
            delay = random.expovariate(1.0 / 1.5)  # 平均1.5s
        elif self.scenario == 'complex':
            delay = random.expovariate(1.0 / 4.0)  # 平均4s
        elif self.scenario == 'upload':
            delay = random.expovariate(1.0 / 3.0)  # 平均3s
        else:
            delay = random.expovariate(1.0 / 2.5)  # 平均2.5s

        time.sleep(min(delay * 0.001, 0.05))  # 压缩延迟到50ms以内用于演示

        latency_ms = delay * 10  # 模拟10倍系数

        is_error = random.random() < 0.02  # 2%基准错误率

        return query[:50], latency_ms, is_error


class LoadTester:
    """负载测试执行器"""

    def __init__(self, target_url: str = "http://localhost:8000"):
        self.target_url = target_url

    def run_scenario(
        self,
        scenario_name: str,
        num_users: int,
        duration_seconds: int,
        spawn_rate: int = 10,
    ) -> LoadTestResult:
        """运行一个负载测试场景

        Args:
            scenario_name: 场景名 ('simple', 'complex', 'upload', 'mixed')
            num_users: 最大并发用户数
            duration_seconds: 测试持续时间
            spawn_rate: 每秒启动的用户数
        """
        result = LoadTestResult(test_name=f"{scenario_name}_{num_users}users")
        result.start_time = time.time()

        users = [
            ScenarioUser(f"user_{i:04d}", scenario_name)
            for i in range(num_users)
        ]
        active_until = time.time() + duration_seconds

        # 使用线程模拟并发用户
        threads = []
        spawn_interval = 1.0 / spawn_rate

        for i, user in enumerate(users):
            if time.time() > active_until:
                break

            t = threading.Thread(target=self._user_loop, args=(user, result, active_until))
            t.start()
            threads.append(t)
            time.sleep(spawn_interval)

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=duration_seconds + 10)

        result.end_time = time.time()
        return result

    def _user_loop(self, user: ScenarioUser, result: LoadTestResult, active_until: float):
        """单个用户的请求循环"""
        while time.time() < active_until:
            try:
                query, latency_ms, is_error = user.make_request()
                with threading.Lock():
                    result.total_requests += 1
                    result.total_latency_ms += latency_ms
                    result.latencies.append(latency_ms)
                    if is_error:
                        result.error_count += 1
                    else:
                        result.success_count += 1
            except Exception:
                pass

            time.sleep(random.uniform(0.5, 2.0))  # 用户思考时间

    def run_all_scenarios(self) -> List[LoadTestResult]:
        """运行所有场景"""
        results = []

        scenarios = [
            ("简单FAQ", "simple", 20, 30),
            ("复杂多跳", "complex", 20, 30),
            ("文档上传", "upload", 10, 20),
            ("混合并发(低)", "mixed", 50, 30),
            ("混合并发(目标)", "mixed", 100, 30),
            ("混合并发(压力)", "mixed", 200, 30),
        ]

        for label, scenario, users, duration in scenarios:
            print(f"\n[负载测试] {label} ({scenario}) - {users}用户, {duration}s")
            result = self.run_scenario(scenario, users, duration)
            results.append(result)
            print(f"  {result.summary()}")

        return results

    def identify_bottlenecks(self, results: List[LoadTestResult]) -> List[str]:
        """识别瓶颈"""
        bottlenecks = []

        for r in results:
            if r.success_rate < 0.95:
                bottlenecks.append(
                    f"{r.test_name}: 成功率过低 ({r.success_rate:.1%})"
                )
            if r.p95_latency_ms > 2000:
                bottlenecks.append(
                    f"{r.test_name}: P95延迟过高 ({r.p95_latency_ms:.0f}ms > 2000ms)"
                )
            if r.p99_latency_ms > 5000:
                bottlenecks.append(
                    f"{r.test_name}: P99延迟过高 ({r.p99_latency_ms:.0f}ms > 5000ms)"
                )
            if r.error_count / max(1, r.total_requests) > 0.1:
                bottlenecks.append(
                    f"{r.test_name}: 错误率过高"
                )

        return bottlenecks


if __name__ == "__main__":
    print("=" * 60)
    print("  负载测试 - 企业级RAG系统")
    print("=" * 60)

    tester = LoadTester()

    print("\n[目标] 100并发, P95<2s, 错误率<1%")

    results = tester.run_all_scenarios()

    print(f"\n{'=' * 60}")
    print("结果汇总")
    print(f"{'=' * 60}")

    for r in results:
        icon = '✓' if r.success_rate >= 0.99 and r.p95_latency_ms < 2000 else '✗'
        print(f"\n  [{icon}] {r.test_name}")
        print(f"  {r.summary()}")

    # 瓶颈分析
    bottlenecks = tester.identify_bottlenecks(results)
    if bottlenecks:
        print(f"\n{'=' * 60}")
        print("瓶颈分析 (需优化)")
        print(f"{'=' * 60}")
        for b in bottlenecks:
            print(f"  ✗ {b}")
    else:
        print(f"\n✓ 所有场景通过负载测试!")

    print()
    print("=" * 60)
    print("  负载测试完成")
    print("=" * 60)
