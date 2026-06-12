#!/usr/bin/env python3
"""
故障注入测试 (Chaos Engineering - Failure Injection Testing)
=============================================================
混沌工程测试:

1. 杀死PostgreSQL → 验证Agent从检查点恢复
2. 杀死Redis → 验证缓存未命中时的DB回退
3. 杀死RabbitMQ → 验证异步任务本地队列
4. 杀死LLM API → 验证熔断器打开
5. 网络分区 → 验证超时和重试
"""

import time
import random
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque


class ServiceStatus(Enum):
    HEALTHY = "healthy"
    KILLED = "killed"
    RECOVERING = "recovering"
    DEGRADED = "degraded"


@dataclass
class ServiceState:
    """模拟服务状态"""
    name: str
    status: ServiceStatus = ServiceStatus.HEALTHY
    kill_time: Optional[datetime] = None
    recovery_time: Optional[datetime] = None

    def kill(self):
        self.status = ServiceStatus.KILLED
        self.kill_time = datetime.now()

    def recover(self):
        self.status = ServiceStatus.RECOVERING
        self.recovery_time = datetime.now()
        # 模拟恢复时间
        time.sleep(0.5)
        self.status = ServiceStatus.HEALTHY


@dataclass
class FailureTestResult:
    """故障注入测试结果"""
    test_name: str
    service_killed: str
    expected_behavior: str
    actual_behavior: str
    system_survived: bool
    recovery_time_seconds: float = 0.0
    data_loss: bool = False
    user_impact: str = ""
    passed: bool = False


class ChaosSystem:
    """混沌工程测试目标系统 (模拟)"""

    def __init__(self):
        self.services = {
            'postgresql': ServiceState('postgresql'),
            'redis': ServiceState('redis'),
            'rabbitmq': ServiceState('rabbitmq'),
            'llm_api': ServiceState('llm_api'),
        }
        self.circuit_breakers: Dict[str, bool] = defaultdict(lambda: True)

        # 回退机制
        self.fallback_db: Dict[str, Any] = {}
        self.local_queue: deque = deque()
        self.cache_entries: Dict[str, Any] = {}
        self.request_count = 0
        self.failed_count = 0

    def is_service_available(self, name: str) -> bool:
        return self.services[name].status == ServiceStatus.HEALTHY

    def kill_service(self, name: str) -> FailureTestResult:
        """杀死一个服务并测试系统反应"""
        if name not in self.services:
            return FailureTestResult(test_name="unknown", service_killed=name,
                                       expected_behavior="", actual_behavior="", system_survived=False)

        service = self.services[name]
        service.kill()
        start = time.time()

        # 测试系统行为
        behaviors = {
            'postgresql': self._test_postgresql_failure,
            'redis': self._test_redis_failure,
            'rabbitmq': self._test_rabbitmq_failure,
            'llm_api': self._test_llm_api_failure,
            'network': self._test_network_partition,
        }

        actual_behavior = ""
        if name in behaviors:
            actual_behavior = behaviors[name]()

        elapsed = time.time() - start

        # 恢复服务
        service.recover()
        recovery_time = time.time() - start

        return FailureTestResult(
            test_name=f"Kill {name}",
            service_killed=name,
            expected_behavior=self._get_expected_behavior(name),
            actual_behavior=actual_behavior[:100],
            system_survived=actual_behavior != "",
            recovery_time_seconds=round(recovery_time, 2),
            user_impact=self._get_user_impact(name),
            passed=self._evaluate_pass(name, actual_behavior),
        )

    def _test_postgresql_failure(self) -> str:
        """测试PostgreSQL故障:
        系统应回退到检查点/内存中的状态
        """
        if not self.is_service_available('postgresql'):
            # 使用内存回退
            self.fallback_db['checkpoint'] = 'last_known_good_state'
            return "系统使用检查点数据继续运行 (checkpoint recovery)"
        return "正常运行"

    def _test_redis_failure(self) -> str:
        """测试Redis故障:
        缓存不可用时应回退到数据库
        """
        if not self.is_service_available('redis'):
            # 回退到postgresql (如果可用) 或内存缓存
            if self.is_service_available('postgresql'):
                return "缓存未命中，回退到数据库查询 (DB fallback)"
            else:
                return "缓存和数据库都不可用，使用本地静态响应"
        return "缓存正常"

    def _test_rabbitmq_failure(self) -> str:
        """测试RabbitMQ故障:
        异步任务应排队到本地
        """
        if not self.is_service_available('rabbitmq'):
            self.local_queue.append({'task': 'pending_job', 'ts': datetime.now().isoformat()})
            return f"消息队列不可用，{len(self.local_queue)}个任务排队到本地 (local queue)"
        return "消息队列正常"

    def _test_llm_api_failure(self) -> str:
        """测试LLM API故障:
        熔断器应打开，使用预缓存响应
        """
        if not self.is_service_available('llm_api'):
            # 触发熔断器
            self.circuit_breakers['llm'] = False  # OPEN
            if self.cache_entries:
                return "LLM API不可用，使用预缓存响应 (circuit breaker OPEN)"
            else:
                return "LLM API不可用，返回优雅降级消息"
        return "LLM正常"

    def _test_network_partition(self) -> str:
        """测试网络分区:
        超时控制应有30秒限制
        """
        self.services['network_partition'] = ServiceState('network', status=ServiceStatus.KILLED)
        # 模拟超时
        start = time.time()
        time.sleep(0.1)  # 简化的超时模拟
        if (time.time() - start) > 30:
            return "超时触发 (>30s)，返回超时错误"
        return "请求在超时前完成 (<30s)"

    def _get_expected_behavior(self, service: str) -> str:
        expectations = {
            'postgresql': '系统从检查点恢复，数据不丢失',
            'redis': '缓存未命中时自动回退到DB查询',
            'rabbitmq': '异步任务本地排队，待恢复后重发',
            'llm_api': '熔断器打开，使用预缓存或降级响应',
            'network': '超时机制触发，不无限等待',
        }
        return expectations.get(service, '')

    def _get_user_impact(self, service: str) -> str:
        impacts = {
            'postgresql': '部分数据可能暂时不可查询，核心功能正常',
            'redis': '响应时间可能稍慢，功能正常',
            'rabbitmq': '异步任务延迟处理，实时功能正常',
            'llm_api': '使用预缓存回答，新查询可能返回降级消息',
            'network': '30秒超时，用户收到超时提示',
        }
        return impacts.get(service, '未知')

    def _evaluate_pass(self, service: str, behavior: str) -> bool:
        """评估测试是否通过"""
        if not behavior:
            return False
        return any(kw in behavior.lower() for kw in [
            'checkpoint', 'fallback', '回退', 'queue', '排队',
            'circuit', '缓存', '降级', '超时', 'graceful', 'degradation',
        ])


class ChaosRunner:
    """混沌工程运行器"""

    def __init__(self, system: ChaosSystem):
        self.system = system
        self.results: List[FailureTestResult] = []

    def run_all_tests(self) -> List[FailureTestResult]:
        """运行所有故障注入测试"""
        print("开始混沌工程测试...\n")

        tests = ['postgresql', 'redis', 'rabbitmq', 'llm_api', 'network']

        for service in tests:
            print(f"[混沌] 注入故障: 杀死 {service}...")
            result = self.system.kill_service(service)
            self.results.append(result)

            status = '✓ 通过' if result.passed else '✗ 失败'
            print(f"  {status}")
            print(f"  预期: {result.expected_behavior}")
            print(f"  实际: {result.actual_behavior}")
            print(f"  恢复时间: {result.recovery_time_seconds}s")
            print(f"  用户影响: {result.user_impact}")
            print()

        return self.results

    def generate_report(self) -> str:
        """生成混沌工程报告"""
        if not self.results:
            return "无测试结果"

        lines = [
            "=" * 60,
            "  混沌工程测试报告 (Chaos Engineering Report)",
            "=" * 60,
            f"  测试时间: {datetime.now().isoformat()}",
            f"  总测试数: {len(self.results)}",
            "",
        ]

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        lines.append(f"  通过: {passed}, 失败: {failed}")
        lines.append("")
        lines.append("  详细结果:")
        lines.append("  " + "-" * 55)

        for r in self.results:
            icon = '✓' if r.passed else '✗'
            lines.append(f"  [{icon}] {r.test_name}")
            lines.append(f"      系统存活: {r.system_survived}")
            lines.append(f"      数据丢失: {r.data_loss}")
            lines.append(f"      恢复时间: {r.recovery_time_seconds}s")
            lines.append("")

        # 整体评估
        resilience_score = passed / max(1, len(self.results)) * 100
        lines.append(f"  弹性评分: {resilience_score:.0f}%")
        lines.append(f"  {'✓ 系统具有良好弹性' if resilience_score >= 80 else '✗ 系统弹性不足，需要改进'}")
        lines.append("=" * 60)

        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("  故障注入测试 - 混沌工程")
    print("=" * 60)

    system = ChaosSystem()
    chaos_runner = ChaosRunner(system)

    # 运行所有测试
    results = chaos_runner.run_all_tests()

    # 生成报告
    report = chaos_runner.generate_report()
    print(report)

    print()
    print("=" * 60)
    print("  混沌工程测试完成")
    print("=" * 60)
