#!/usr/bin/env python3
"""
熔断器模式 (Circuit Breaker Pattern)
三态熔断器: CLOSED → OPEN → HALF_OPEN

特性:
  - 三态模型: CLOSED(正常) / OPEN(熔断) / HALF_OPEN(探测)
  - 失败阈值: 60秒内5次失败 → OPEN
  - 重置超时: 30秒 → HALF_OPEN
  - 每依赖独立熔断: LLM API, Vector DB, Reranker
  - 监控集成: Prometheus metrics兼容
  - 指数退避: 每次重新进入OPEN时增加冷却时间
"""

import time
import threading
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "CLOSED"           # 正常: 请求正常通过
    OPEN = "OPEN"               # 熔断: 直接拒绝请求
    HALF_OPEN = "HALF_OPEN"     # 半开: 允许探测请求


class DependencyType(Enum):
    """依赖类型"""
    LLM_API = "llm_api"               # LLM API调用
    VECTOR_DB = "vector_db"            # 向量数据库
    RERANKER = "reranker"              # 重排序服务
    EMBEDDING_SERVICE = "embedding"    # 嵌入服务
    WEB_SEARCH = "web_search"          # Web搜索
    REDIS = "redis"                    # 缓存服务


@dataclass
class CircuitConfig:
    """熔断器配置"""
    failure_threshold: int = 5         # 失败阈值
    failure_window: float = 60.0       # 失败计数窗口（秒）
    timeout: float = 30.0              # OPEN→HALF_OPEN冷却时间（秒）
    half_open_max_requests: int = 1    # HALF_OPEN最大探测请求
    max_timeout: float = 300.0         # 最大冷却时间（指数退避上限）
    success_threshold_half_open: int = 2  # HALF_OPEN下需要连续成功才能CLOSE

    @staticmethod
    def for_dependency(dep_type: DependencyType) -> "CircuitConfig":
        """根据依赖类型推荐配置"""
        configs = {
            DependencyType.LLM_API: CircuitConfig(
                failure_threshold=5,
                failure_window=60.0,
                timeout=30.0,
                half_open_max_requests=2,
            ),
            DependencyType.VECTOR_DB: CircuitConfig(
                failure_threshold=3,
                failure_window=30.0,
                timeout=15.0,
                half_open_max_requests=1,
            ),
            DependencyType.RERANKER: CircuitConfig(
                failure_threshold=5,
                failure_window=60.0,
                timeout=30.0,
                half_open_max_requests=1,
            ),
            DependencyType.EMBEDDING_SERVICE: CircuitConfig(
                failure_threshold=3,
                failure_window=30.0,
                timeout=15.0,
                half_open_max_requests=1,
            ),
            DependencyType.WEB_SEARCH: CircuitConfig(
                failure_threshold=3,
                failure_window=30.0,
                timeout=20.0,
                half_open_max_requests=1,
            ),
            DependencyType.REDIS: CircuitConfig(
                failure_threshold=2,
                failure_window=10.0,
                timeout=5.0,
                half_open_max_requests=1,
            ),
        }
        return configs.get(dep_type, CircuitConfig())


@dataclass
class CircuitMetrics:
    """熔断器监控指标"""
    name: str
    state: CircuitState = CircuitState.CLOSED
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_rejections: int = 0        # 被熔断拒绝的请求
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    opened_count: int = 0            # 进入OPEN次数
    current_timeout: float = 0.0     # 当前冷却时间

    def to_prometheus(self) -> str:
        """转换为Prometheus metrics格式"""
        state_value = {
            CircuitState.CLOSED: 0,
            CircuitState.OPEN: 1,
            CircuitState.HALF_OPEN: 2,
        }
        return f"""# HELP circuit_breaker_state State of circuit breaker (0=CLOSED,1=OPEN,2=HALF_OPEN)
# TYPE circuit_breaker_state gauge
circuit_breaker_state{{name="{self.name}"}} {state_value.get(self.state, -1)}
# HELP circuit_breaker_requests_total Total requests through circuit breaker
# TYPE circuit_breaker_requests_total counter
circuit_breaker_requests_total{{name="{self.name}"}} {self.total_requests}
# HELP circuit_breaker_failures_total Total failures in circuit breaker
# TYPE circuit_breaker_failures_total counter
circuit_breaker_failures_total{{name="{self.name}"}} {self.total_failures}
# HELP circuit_breaker_rejections_total Total rejected requests (open circuit)
# TYPE circuit_breaker_rejections_total counter
circuit_breaker_rejections_total{{name="{self.name}"}} {self.total_rejections}
"""


# ============================================================
# 滑动窗口失败计数器
# ============================================================

class SlidingWindowCounter:
    """滑动窗口失败计数器"""

    def __init__(self, window_seconds: float = 60.0):
        self.window = window_seconds
        self._failures: list[float] = []  # 失败时间戳列表
        self._lock = threading.RLock()

    def record_failure(self):
        """记录一次失败"""
        with self._lock:
            now = time.time()
            self._failures.append(now)
            self._cleanup(now)

    def count(self) -> int:
        """获取窗口内的失败次数"""
        with self._lock:
            self._cleanup(time.time())
            return len(self._failures)

    def clear(self):
        """清空计数（进入HALF_OPEN时）"""
        with self._lock:
            self._failures.clear()

    def _cleanup(self, now: float):
        """清理窗口外的旧记录"""
        cutoff = now - self.window
        self._failures = [t for t in self._failures if t > cutoff]


# ============================================================
# 熔断器核心实现
# ============================================================

class CircuitBreaker:
    """三态熔断器"""

    def __init__(self, name: str, config: Optional[CircuitConfig] = None,
                 dependency: Optional[DependencyType] = None):
        """
        Args:
            name: 熔断器名称
            config: 配置
            dependency: 依赖类型（自动应用推荐配置）
        """
        self.name = name
        self.config = config or (
            CircuitConfig.for_dependency(dependency) if dependency else CircuitConfig()
        )
        self.dependency = dependency

        self._state = CircuitState.CLOSED
        self._state_lock = threading.RLock()
        self._failure_counter = SlidingWindowCounter(self.config.failure_window)
        self._last_state_change = time.time()
        self._half_open_successes = 0
        self._half_open_requests = 0
        self._current_timeout = self.config.timeout
        self._open_count = 0  # 进入OPEN的次数（用于指数退避）

        # 监控指标
        self.metrics = CircuitMetrics(name=name)

        # 状态变化回调
        self._state_callbacks: list[Callable] = []

    # ========== 状态管理 ==========

    @property
    def state(self) -> CircuitState:
        with self._state_lock:
            self._transition_if_needed()
        return self._state

    def on_state_change(self, callback: Callable):
        """注册状态变化回调"""
        self._state_callbacks.append(callback)

    def _change_state(self, new_state: CircuitState):
        """切换状态并通知回调"""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        self.metrics.state = new_state

        for cb in self._state_callbacks:
            try:
                cb(self.name, old_state, new_state)
            except Exception:
                pass

    def _transition_if_needed(self):
        """检查是否需要状态转换"""
        now = time.time()

        if self._state == CircuitState.OPEN:
            # 检查是否应该进入HALF_OPEN
            if now - self._last_state_change >= self._current_timeout:
                self._change_state(CircuitState.HALF_OPEN)
                self._half_open_successes = 0
                self._half_open_requests = 0
                self._failure_counter.clear()

        elif self._state == CircuitState.CLOSED:
            # 检查是否应该进入OPEN
            failure_count = self._failure_counter.count()
            if failure_count >= self.config.failure_threshold:
                self._change_state(CircuitState.OPEN)
                self._open_count += 1
                self.metrics.opened_count = self._open_count
                # 指数退避
                self._current_timeout = min(
                    self.config.timeout * (2 ** (self._open_count - 1)),
                    self.config.max_timeout,
                )
                self.metrics.current_timeout = self._current_timeout

    # ========== 请求执行 ==========

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """执行受熔断器保护的调用

        Args:
            func: 被保护的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            CircuitBreakerOpenError: 熔断器打开时
            原始异常: 函数执行失败时
        """
        with self._state_lock:
            self._transition_if_needed()

            # OPEN状态: 直接拒绝
            if self._state == CircuitState.OPEN:
                self.metrics.total_requests += 1
                self.metrics.total_rejections += 1
                raise CircuitBreakerOpenError(
                    f"熔断器 [{self.name}] 处于OPEN状态, 拒绝请求"
                )

            # HALF_OPEN状态: 限制探测请求数
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    self.metrics.total_requests += 1
                    self.metrics.total_rejections += 1
                    raise CircuitBreakerOpenError(
                        f"熔断器 [{self.name}] HALF_OPEN探测请求已满"
                    )
                self._half_open_requests += 1

        # 执行实际调用
        self.metrics.total_requests += 1
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    async def execute_async(self, func, *args, **kwargs) -> Any:
        """异步版本的execute"""
        with self._state_lock:
            self._transition_if_needed()

            if self._state == CircuitState.OPEN:
                self.metrics.total_requests += 1
                self.metrics.total_rejections += 1
                raise CircuitBreakerOpenError(
                    f"熔断器 [{self.name}] 处于OPEN状态"
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    self.metrics.total_requests += 1
                    self.metrics.total_rejections += 1
                    raise CircuitBreakerOpenError(
                        f"熔断器 [{self.name}] HALF_OPEN探测请求已满"
                    )
                self._half_open_requests += 1

        self.metrics.total_requests += 1
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self):
        """成功回调"""
        with self._state_lock:
            self.metrics.total_successes += 1
            self.metrics.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold_half_open:
                    self._change_state(CircuitState.CLOSED)
                    self._open_count = 0  # 重置退避

    def _on_failure(self, error: Exception):
        """失败回调"""
        with self._state_lock:
            self.metrics.total_failures += 1
            self.metrics.last_failure_time = time.time()
            self._failure_counter.record_failure()

            if self._state == CircuitState.HALF_OPEN:
                # HALF_OPEN探测失败 → 立即回到OPEN
                self._change_state(CircuitState.OPEN)
                self._open_count += 1
                self._current_timeout = min(
                    self.config.timeout * (2 ** (self._open_count - 1)),
                    self.config.max_timeout,
                )
                self.metrics.current_timeout = self._current_timeout
            elif self._state == CircuitState.CLOSED:
                # 触发阈值检查
                self._transition_if_needed()

    # ========== 重置 ==========

    def reset(self):
        """手动重置熔断器"""
        with self._state_lock:
            self._change_state(CircuitState.CLOSED)
            self._failure_counter.clear()
            self._half_open_successes = 0
            self._half_open_requests = 0
            self._open_count = 0
            self._current_timeout = self.config.timeout

    def force_open(self):
        """手动强制打开熔断器（运维操作）"""
        with self._state_lock:
            self._change_state(CircuitState.OPEN)

    def force_close(self):
        """手动强制关闭熔断器（运维操作）"""
        self.reset()


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常"""
    pass


# ============================================================
# 装饰器 (Decorator)
# ============================================================

def circuit_breaker(name: str, config: Optional[CircuitConfig] = None,
                    dependency: Optional[DependencyType] = None):
    """熔断器装饰器"""
    cb = CircuitBreaker(name, config, dependency)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cb.execute(func, *args, **kwargs)
        wrapper.circuit_breaker = cb
        return wrapper

    return decorator


# ============================================================
# 熔断器管理器 (管理多个依赖的熔断器)
# ============================================================

class CircuitBreakerManager:
    """熔断器管理器 - 管理所有依赖的熔断器"""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get_or_create(self, name: str, dependency: Optional[DependencyType] = None,
                      config: Optional[CircuitConfig] = None) -> CircuitBreaker:
        """获取或创建熔断器"""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    config=config or CircuitConfig.for_dependency(dependency)
                    if dependency else CircuitConfig(),
                    dependency=dependency,
                )
            return self._breakers[name]

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """获取熔断器"""
        return self._breakers.get(name)

    def execute(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """通过名称执行受保护的调用"""
        breaker = self.get_or_create(name)
        return breaker.execute(func, *args, **kwargs)

    def reset_all(self):
        """重置所有熔断器"""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_all_metrics(self) -> dict[str, CircuitMetrics]:
        """获取所有熔断器指标"""
        return {name: cb.metrics for name, cb in self._breakers.items()}

    def get_prometheus_metrics(self) -> str:
        """获取Prometheus格式的所有指标"""
        lines = []
        for cb in self._breakers.values():
            lines.append(cb.metrics.to_prometheus())
        return "\n".join(lines)

    def health_report(self) -> dict:
        """健康报告"""
        report = {}
        for name, cb in self._breakers.items():
            report[name] = {
                "state": cb.state.value,
                "dependency": cb.dependency.value if cb.dependency else "unknown",
                "open_count": cb.metrics.opened_count,
                "current_timeout": f"{cb._current_timeout:.1f}s",
                "failure_rate": (
                    cb.metrics.total_failures / max(cb.metrics.total_requests, 1)
                ),
                "rejection_rate": (
                    cb.metrics.total_rejections / max(cb.metrics.total_requests, 1)
                ),
            }
        return report


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("熔断器系统 - 演示运行")
    print("=" * 60)

    # 1. 创建熔断器管理器
    manager = CircuitBreakerManager()

    # 为各个依赖创建熔断器
    manager.get_or_create("llm_api", DependencyType.LLM_API)
    manager.get_or_create("vector_db", DependencyType.VECTOR_DB)
    manager.get_or_create("reranker", DependencyType.RERANKER)

    print("\n--- 初始状态 ---")
    for name, breaker in manager._breakers.items():
        print(f"  [{name}] 状态={breaker.state.value}, 配置={breaker.config}")

    # 2. 模拟正常调用
    print("\n--- 模拟正常调用 ---")

    def successful_call():
        return "调用成功!"

    try:
        result = manager.execute("llm_api", successful_call)
        print(f"LLM API调用: {result}")
    except Exception as e:
        print(f"LLM API调用失败: {e}")

    # 3. 模拟连续失败触发熔断
    print("\n--- 模拟连续失败触发熔断 ---")

    def failing_call():
        raise ConnectionError("连接超时!")

    for i in range(6):
        try:
            result = manager.execute("vector_db", failing_call)
            print(f"  第{i+1}次: {result}")
        except CircuitBreakerOpenError as e:
            print(f"  第{i+1}次: 熔断器打开! {e}")
        except Exception as e:
            print(f"  第{i+1}次: 失败 {e}")

    # 4. 查看熔断后状态
    print("\n--- 熔断后状态 ---")
    vb = manager.get_breaker("vector_db")
    print(f"  状态: {vb.state.value}")
    print(f"  打开次数: {vb.metrics.opened_count}")
    print(f"  当前冷却: {vb._current_timeout:.1f}s")
    print(f"  总失败: {vb.metrics.total_failures}")
    print(f"  被拒绝: {vb.metrics.total_rejections}")

    # 5. 快速快进时间模拟HALF_OPEN
    print("\n--- 模拟冷却后进入HALF_OPEN ---")
    vb._last_state_change = time.time() - 31  # 快进31秒

    try:
        result = manager.execute("vector_db", successful_call)
        print(f"HALF_OPEN探测: {result}")
        print(f"探测后状态: {vb.state.value}")
        print(f"HALF_OPEN成功计数: {vb._half_open_successes}")

        # 再次成功（达到阈值）
        result2 = manager.execute("vector_db", successful_call)
        print(f"第二次探测: {result2}")
        print(f"状态: {vb.state.value}")  # 应为CLOSED
    except Exception as e:
        print(f"探测失败: {e}")

    # 6. 演示指数退避
    print("\n--- 指数退避演示 ---")
    demo_cb = CircuitBreaker("demo_backoff", CircuitConfig(timeout=5.0, max_timeout=300.0))

    for i in range(4):
        # 手动触发OPEN
        for _ in range(5):
            demo_cb._failure_counter.record_failure()
        demo_cb._transition_if_needed()
        print(f"  第{i+1}次进入OPEN, 冷却={demo_cb._current_timeout:.1f}s")
        # 手动恢复到CLOSED以准备下一次
        demo_cb.reset()

    # 7. 监控指标
    print("\n--- Prometheus指标 ---")
    metrics = manager.get_prometheus_metrics()
    print(metrics)

    # 8. 健康报告
    print("--- 健康报告 ---")
    health = manager.health_report()
    for name, info in health.items():
        print(f"  [{name}] 状态={info['state']}, "
              f"失败率={info['failure_rate']:.1%}, "
              f"拒绝率={info['rejection_rate']:.1%}")

    # 9. 装饰器用法演示
    print("\n--- 装饰器用法 ---")

    @circuit_breaker("my_llm_service", dependency=DependencyType.LLM_API)
    def call_llm(prompt: str) -> str:
        # 模拟有时成功有时失败
        import random
        if random.random() < 0.3:
            raise TimeoutError("LLM Timeout")
        return f"LLM回复: {prompt[:20]}..."

    # 多次调用观察行为
    results = {"success": 0, "failure": 0, "rejected": 0}
    for i in range(10):
        try:
            resp = call_llm(f"测试提示词 {i}")
            results["success"] += 1
        except CircuitBreakerOpenError:
            results["rejected"] += 1
        except Exception:
            results["failure"] += 1

    print(f"10次调用结果: 成功={results['success']}, 失败={results['failure']}, 拒绝={results['rejected']}")

    print("\n" + "=" * 60)
    print("熔断器系统演示完成！")
    print("=" * 60)
