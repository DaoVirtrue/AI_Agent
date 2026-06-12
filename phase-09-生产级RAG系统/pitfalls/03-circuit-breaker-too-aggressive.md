# Pitfall 03: 熔断器过于激进 (False-Positive Tripping)

## 症状 (Symptoms)

- 熔断器频繁进入OPEN状态，即使是间歇性网络抖动
- LLM API偶尔返回429（速率限制）就触发熔断，导致所有后续请求被拒绝
- 向量数据库短暂慢查询（GC暂停）被误判为服务不可用
- 熔断器恢复后立即因之前积压的请求再次熔断
- HALF_OPEN探测请求太少导致恢复过慢

## 根本原因 (Root Cause)

熔断器配置不考虑依赖特性：

1. **所有依赖使用相同配置**: LLM API的速率限制和向量数据库的GC暂停需要不同的阈值和窗口
2. **故障分类不区分**: 将可重试错误（429、超时）和不可重试错误（400、认证失败）同等对待
3. **冷却时间固定**: 不使用指数退避，导致服务刚恢复就被打垮
4. **缺少降级策略**: 熔断后没有fallback，直接返回错误

## 真实场景 (Real Scenario)

某RAG服务在黑色星期五流量高峰期：
- LLM API偶尔返回429（很正常的高峰限流）
- 熔断器在30秒内收到5次429后进入OPEN
- 所有用户请求被拒绝30秒
- 30秒后HALF_OPEN仅允许1个探测请求
- 该探测请求也遇到429（峰值未过）
- 熔断器重新进入OPEN
- 如此循环，服务中断超过10分钟

正确做法：429不应计入熔断计数，应使用速率限制器控制。

## 完整解决方案 (Complete Solution)

```python
from enum import Enum

class ErrorCategory(Enum):
    """错误分类"""
    RETRYABLE_TIMEOUT = "retryable_timeout"      # 可重试超时
    RETRYABLE_RATE_LIMIT = "retryable_rate_limit" # 可重试速率限制
    RETRYABLE_SERVER_ERROR = "retryable_server_error"  # 5xx
    NON_RETRYABLE = "non_retryable"              # 不可重试（4xx认证/参数错误）
    CIRCUIT_BREAKER_WORTHY = "circuit_breaker_worthy"  # 应触发熔断


def classify_error(exception: Exception, dependency: str) -> ErrorCategory:
    """
    对错误进行分类，只有特定类型才触发熔断
    """
    error_str = str(exception).lower()
    
    # 速率限制 → 不应触发熔断
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return ErrorCategory.RETRYABLE_RATE_LIMIT
    
    # 连接超时 → 可能触发熔断（取决于频率）
    if "timeout" in error_str or "timed out" in error_str:
        return ErrorCategory.RETRYABLE_TIMEOUT
    
    # 服务端错误 → 应触发熔断
    if "500" in error_str or "503" in error_str or "connection refused" in error_str:
        return ErrorCategory.CIRCUIT_BREAKER_WORTHY
    
    # 认证/参数错误 → 不应触发熔断（这是客户端问题）
    if "401" in error_str or "403" in error_str or "400" in error_str:
        return ErrorCategory.NON_RETRYABLE
    
    # 默认
    return ErrorCategory.CIRCUIT_BREAKER_WORTHY


class PerDependencyCircuitBreaker:
    """每依赖定制的熔断器"""
    
    # 不同依赖的推荐配置
    DEPENDENCY_CONFIGS = {
        "llm_api": {
            "failure_threshold": 10,     # 更高阈值（LLM偶尔超时正常）
            "failure_window": 120,       # 更长窗口
            "timeout": 15,               # 较短冷却
            "ignore_errors": ["429", "rate_limit", "too_many_requests"],
            "required_failure_types": ["timeout", "500", "503", "connection_error"],
        },
        "vector_db": {
            "failure_threshold": 3,      # 较低阈值（向量DB应该很稳定）
            "failure_window": 30,
            "timeout": 10,
            "ignore_errors": [],
            "required_failure_types": ["connection_refused", "timeout"],
        },
        "reranker": {
            "failure_threshold": 5,
            "failure_window": 60,
            "timeout": 30,
            "ignore_errors": [],
        },
    }
    
    def __init__(self, name: str, dependency_type: str):
        self.name = name
        config = self.DEPENDENCY_CONFIGS.get(dependency_type, {})
        self.failure_threshold = config.get("failure_threshold", 5)
        self.failure_window = config.get("failure_window", 60)
        self.timeout = config.get("timeout", 30)
        self.ignore_errors = config.get("ignore_errors", [])
        self.required_failure_types = config.get("required_failure_types", [])
        
        self._failures = []
        self._state = "CLOSED"
        self._open_count = 0
    
    def should_count_as_failure(self, error: Exception) -> bool:
        """判断该错误是否应计入熔断计数"""
        error_str = str(error).lower()
        
        # 忽略特定错误类型
        for ignore_pattern in self.ignore_errors:
            if ignore_pattern.lower() in error_str:
                return False
        
        # 检查是否为应触发熔断的错误类型
        if self.required_failure_types:
            for required_type in self.required_failure_types:
                if required_type.lower() in error_str:
                    return True
            return False  # 不在白名单内的错误不触发
        
        return True
    
    def calculate_backoff_timeout(self) -> float:
        """指数退避冷却时间"""
        base_timeout = self.timeout
        backoff = base_timeout * (2 ** min(self._open_count, 4))  # 最大16倍
        return min(backoff, 300)  # 最长5分钟


def correct_circuit_breaker_usage():
    """正确的熔断器使用模式"""
    
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
    
    # 模式1: 熔断器 + 重试 + 降级 组合
    @circuit_breaker("llm_api", dependency="llm_api")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    )
    def call_llm_with_fallback(prompt: str):
        """带降级的LLM调用"""
        try:
            return primary_llm_call(prompt)
        except Exception:
            # 降级到备用模型
            return fallback_llm_call(prompt)
    
    # 模式2: 监控指标不区分各种失败
    # Prometheus指标:
    # - circuit_breaker_failures_total{type="retryable"}
    # - circuit_breaker_failures_total{type="rate_limit"}
    # - circuit_breaker_failures_total{type="non_retryable"}
    
    # 模式3: HALF_OPEN允许多个探测（>1）
    config = CircuitConfig(
        half_open_max_requests=3,  # 允许3个探测请求
        success_threshold_half_open=2,  # 2个成功即关闭
    )
```

## 检查清单 (Checklist)

- [ ] 每个依赖类型使用不同熔断器配置
- [ ] 429/速率限制错误不计入熔断计数
- [ ] 客户端错误（400/401/403）不计入熔断计数
- [ ] 实现指数退避冷却时间（避免雷鸣群效应）
- [ ] HALF_OPEN探测请求数 > 1（至少2-3个）
- [ ] 熔断后提供降级方案（非直接返回错误）
- [ ] 按错误类型分别记录熔断指标
- [ ] 定期Review各依赖的熔断器配置
- [ ] 压测验证熔断器行为（不依赖生产事故验证）
