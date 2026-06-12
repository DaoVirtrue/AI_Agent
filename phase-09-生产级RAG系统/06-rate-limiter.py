#!/usr/bin/env python3
"""
速率限制器 (Rate Limiter)
Token Bucket Algorithm Implementation

特性:
  - 令牌桶算法 (Token Bucket)
  - 每租户 + 每端点速率限制
  - 可配置: requests/second, tokens/minute
  - 突发允许 (Burst allowance)
  - 优雅降级: 接近限制时排队, 超限返回429
  - Retry-After 头部
"""

import time
import threading
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any
from collections import defaultdict


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class RateLimitResult(Enum):
    """速率限制结果"""
    ALLOWED = "allowed"           # 允许通过
    QUEUED = "queued"             # 进入队列（接近限制）
    REJECTED = "rejected"         # 拒绝（超过限制）


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_second: float = 10.0       # 每秒请求数
    tokens_per_minute: int = 100000          # 每分钟token数
    burst_size: int = 20                     # 突发允许大小
    queue_enabled: bool = True               # 是否启用队列
    queue_max_size: int = 100                # 最大队列长度
    queue_timeout: float = 30.0              # 队列超时（秒）

    @staticmethod
    def for_tier(tier: str) -> "RateLimitConfig":
        """根据租户等级获取限流配置"""
        configs = {
            "free": RateLimitConfig(
                requests_per_second=5.0,
                tokens_per_minute=50000,
                burst_size=10,
                queue_max_size=50,
            ),
            "pro": RateLimitConfig(
                requests_per_second=50.0,
                tokens_per_minute=500000,
                burst_size=100,
                queue_max_size=500,
            ),
            "enterprise": RateLimitConfig(
                requests_per_second=500.0,
                tokens_per_minute=5000000,
                burst_size=1000,
                queue_max_size=5000,
            ),
        }
        return configs.get(tier, configs["free"])


@dataclass
class RateLimitInfo:
    """速率限制信息（响应头部用）"""
    limit: int               # 时间窗口内的配额
    remaining: int           # 剩余配额
    reset_at: float          # 重置时间戳
    retry_after: float = 0.0  # 建议重试等待时间（秒）


# ============================================================
# 令牌桶实现 (Token Bucket)
# ============================================================

class TokenBucket:
    """令牌桶"""

    def __init__(self, rate: float, burst: int = 10, name: str = ""):
        """
        Args:
            rate: 令牌生成速率（token/秒）
            burst: 桶容量（最大突发）
            name: 桶名称
        """
        self.rate = float(rate)
        self.burst = burst
        self.name = name

        self._tokens = float(burst)  # 初始满桶
        self._last_refill = time.time()
        self._lock = threading.RLock()

        # 统计
        self._total_consumed = 0
        self._total_rejected = 0

    def consume(self, tokens: int = 1) -> bool:
        """尝试消费令牌

        Args:
            tokens: 需要消费的令牌数

        Returns:
            True: 消费成功，False: 令牌不足
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_consumed += tokens
                return True

            self._total_rejected += 1
            return False

    def available_tokens(self) -> float:
        """获取当前可用令牌数"""
        with self._lock:
            self._refill()
            return self._tokens

    def wait_time_for(self, tokens: int = 1) -> float:
        """计算获取指定令牌需要的等待时间"""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            needed = tokens - self._tokens
            return needed / self.rate if self.rate > 0 else float("inf")

    def reset(self):
        """重置令牌桶"""
        with self._lock:
            self._tokens = float(self.burst)
            self._last_refill = time.time()

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.rate
        self._tokens = min(self._tokens + new_tokens, float(self.burst))
        self._last_refill = now

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "available": self.available_tokens(),
            "total_consumed": self._total_consumed,
            "total_rejected": self._total_rejected,
            "rate": self.rate,
            "burst": self.burst,
        }


# ============================================================
# 请求队列 (Request Queue)
# ============================================================

class RequestQueue:
    """请求队列 - 用于优雅降级"""

    def __init__(self, max_size: int = 100, timeout: float = 30.0):
        self.max_size = max_size
        self.timeout = timeout
        self._queue: list[tuple] = []  # (arrival_time, callback)
        self._lock = threading.RLock()
        self._total_queued = 0
        self._total_timeouts = 0

    def enqueue(self, callback: Callable) -> bool:
        """入队"""
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False
            self._queue.append((time.time(), callback))
            self._total_queued += 1
            return True

    def dequeue_valid(self) -> Optional[Callable]:
        """取出一个未超时的请求"""
        with self._lock:
            now = time.time()
            # 清理超时请求
            valid = []
            for arrival, cb in self._queue:
                if now - arrival <= self.timeout:
                    valid.append((arrival, cb))
                else:
                    self._total_timeouts += 1
            self._queue = valid

            if self._queue:
                return self._queue.pop(0)[1]
            return None

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> dict:
        return {
            "queue_size": self.size,
            "total_queued": self._total_queued,
            "total_timeouts": self._total_timeouts,
            "max_size": self.max_size,
        }


# ============================================================
# 速率限制器 (Rate Limiter)
# ============================================================

class RateLimiter:
    """每租户+每端点的速率限制器"""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}       # key -> bucket
        self._queues: dict[str, RequestQueue] = {}       # key -> queue
        self._configs: dict[str, RateLimitConfig] = {}   # key -> config
        self._global_lock = threading.RLock()

    # ========== 配置管理 ==========

    def configure(self, tenant_id: str, endpoint: str, config: RateLimitConfig):
        """配置特定租户+端点的限流"""
        key = self._make_key(tenant_id, endpoint)
        with self._global_lock:
            self._configs[key] = config

            # 创建令牌桶
            self._buckets[key] = TokenBucket(
                rate=config.requests_per_second,
                burst=config.burst_size,
                name=key,
            )

            # 创建请求队列
            if config.queue_enabled:
                self._queues[key] = RequestQueue(
                    max_size=config.queue_max_size,
                    timeout=config.queue_timeout,
                )

    def configure_tenant(self, tenant_id: str, config: RateLimitConfig,
                         endpoints: Optional[list[str]] = None):
        """为租户的所有端点配置限流（或指定端点列表）"""
        if endpoints is None:
            endpoints = ["query", "ingest", "admin"]

        for ep in endpoints:
            self.configure(tenant_id, ep, config)

    def get_config(self, tenant_id: str, endpoint: str) -> RateLimitConfig:
        """获取配置"""
        key = self._make_key(tenant_id, endpoint)
        return self._configs.get(key, RateLimitConfig())

    # ========== 速率检查 ==========

    def check(self, tenant_id: str, endpoint: str, tokens: int = 1) -> tuple[RateLimitResult, RateLimitInfo]:
        """检查请求是否可以通过速率限制

        Args:
            tenant_id: 租户ID
            endpoint: 端点名称（query/ingest/admin等）
            tokens: 所需令牌数

        Returns:
            (结果, 限流信息)
        """
        key = self._make_key(tenant_id, endpoint)

        with self._global_lock:
            bucket = self._buckets.get(key)
            config = self._configs.get(key, RateLimitConfig())

            if bucket is None:
                # 未配置，允许通过
                return RateLimitResult.ALLOWED, RateLimitInfo(
                    limit=999999, remaining=999999, reset_at=time.time() + 60
                )

            available = bucket.available_tokens()
            wait_time = bucket.wait_time_for(tokens)

            info = RateLimitInfo(
                limit=config.burst_size,
                remaining=max(0, int(available)),
                reset_at=time.time() + wait_time,
            )

            # 尝试直接消费
            if bucket.consume(tokens):
                return RateLimitResult.ALLOWED, info

            # 令牌不足，检查是否可以排队
            if config.queue_enabled:
                queue = self._queues.get(key)
                if queue and queue.size < config.queue_max_size:
                    info.retry_after = wait_time
                    return RateLimitResult.QUEUED, info

            # 拒绝
            info.retry_after = wait_time
            return RateLimitResult.REJECTED, info

    def check_or_wait(self, tenant_id: str, endpoint: str, tokens: int = 1,
                      block: bool = True, timeout: float = 30.0) -> tuple[RateLimitResult, RateLimitInfo]:
        """检查或在队列中等待

        Args:
            block: 是否阻塞等待
            timeout: 最大等待时间
        """
        result, info = self.check(tenant_id, endpoint, tokens)

        if result == RateLimitResult.QUEUED and block:
            # 等待令牌补充
            wait = min(info.retry_after, timeout)
            if wait > 0:
                time.sleep(wait)
                # 重新尝试
                return self.check(tenant_id, endpoint, tokens)

        return result, info

    # ========== 响应头生成 ==========

    def generate_headers(self, info: RateLimitInfo) -> dict:
        """生成HTTP响应头"""
        headers = {
            "X-RateLimit-Limit": str(info.limit),
            "X-RateLimit-Remaining": str(info.remaining),
            "X-RateLimit-Reset": str(int(info.reset_at)),
        }
        if info.retry_after > 0:
            headers["Retry-After"] = str(int(info.retry_after))
        return headers

    # ========== 统计与监控 ==========

    def get_stats(self, tenant_id: str, endpoint: str) -> dict:
        """获取特定租户+端点的统计"""
        key = self._make_key(tenant_id, endpoint)
        bucket = self._buckets.get(key)
        queue = self._queues.get(key)

        stats = {
            "key": key,
            "bucket": bucket.stats if bucket else None,
            "queue": queue.stats if queue else None,
        }
        return stats

    def get_all_stats(self) -> dict[str, dict]:
        """获取所有统计"""
        return {key: self.get_stats(*self._parse_key(key)) for key in self._buckets}

    def reset(self, tenant_id: str, endpoint: str):
        """重置特定租户+端点的限流器"""
        key = self._make_key(tenant_id, endpoint)
        bucket = self._buckets.get(key)
        if bucket:
            bucket.reset()
        queue = self._queues.get(key)
        if queue:
            queue._queue.clear()

    def reset_all(self):
        """重置所有限流器"""
        for bucket in self._buckets.values():
            bucket.reset()
        for queue in self._queues.values():
            queue._queue.clear()

    # ========== 辅助方法 ==========

    @staticmethod
    def _make_key(tenant_id: str, endpoint: str) -> str:
        return f"{tenant_id}:{endpoint}"

    @staticmethod
    def _parse_key(key: str) -> tuple[str, str]:
        parts = key.split(":", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""


# ============================================================
# 装饰器 (Decorator)
# ============================================================

class RateLimited:
    """速率限制装饰器/上下文管理器"""

    def __init__(self, limiter: RateLimiter, tenant_id: str, endpoint: str,
                 tokens: int = 1, auto_queue: bool = True):
        self.limiter = limiter
        self.tenant_id = tenant_id
        self.endpoint = endpoint
        self.tokens = tokens
        self.auto_queue = auto_queue
        self._result: Optional[RateLimitResult] = None
        self._info: Optional[RateLimitInfo] = None

    def __enter__(self):
        self._result, self._info = self.limiter.check_or_wait(
            self.tenant_id, self.endpoint, self.tokens, block=self.auto_queue
        )
        if self._result == RateLimitResult.REJECTED:
            raise RateLimitExceededError(
                f"速率限制超出: {self.tenant_id}/{self.endpoint}",
                retry_after=self._info.retry_after if self._info else 0,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # 不抑制异常

    @property
    def result(self) -> Optional[RateLimitResult]:
        return self._result

    @property
    def headers(self) -> dict:
        return self.limiter.generate_headers(self._info) if self._info else {}


class RateLimitExceededError(Exception):
    """速率限制超出异常"""

    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


# ============================================================
# 令牌计数估算器 (Token Counter for Rate Limiting)
# ============================================================

class TokenEstimator:
    """Token估算器 - 用于基于token的速率限制"""

    @staticmethod
    def estimate_request_tokens(query: str, context_chunks: int = 0,
                                avg_chunk_size: int = 500) -> int:
        """估算请求所需token数

        Args:
            query: 查询文本
            context_chunks: 上下文块数
            avg_chunk_size: 每块平均字符数
        """
        import re

        # 查询token（中文~1.5字符/token，英文~4字符/token）
        query_cn = len(re.findall(r"[一-鿿]", query))
        query_en = len(query) - query_cn
        query_tokens = int(query_cn / 1.5 + query_en / 4)

        # 上下文token
        context_chars = context_chunks * avg_chunk_size
        context_tokens = int(context_chars / 3)  # 折衷估算

        # 系统提示token（固定约100-200）
        system_tokens = 150

        # 输出token（预估输入token的1/3）
        output_tokens = int((query_tokens + context_tokens) * 0.33)

        return query_tokens + context_tokens + system_tokens + output_tokens


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("速率限制器 - 演示运行")
    print("=" * 60)

    # 1. 初始化速率限制器
    limiter = RateLimiter()

    # 配置不同等级的租户
    limiter.configure_tenant("free_tenant", RateLimitConfig.for_tier("free"))
    limiter.configure_tenant("pro_tenant", RateLimitConfig.for_tier("pro"))
    limiter.configure_tenant("enterprise_tenant", RateLimitConfig.for_tier("enterprise"))

    # 2. 令牌桶基础演示
    print("\n--- 令牌桶基础 ---")

    # 创建一个测试桶: 5 token/秒, 突发10
    test_bucket = TokenBucket(rate=5.0, burst=10, name="test")

    print(f"初始令牌: {test_bucket.available_tokens():.1f}")

    # 消费10个令牌（突发全部用尽）
    for i in range(10):
        success = test_bucket.consume(1)
        print(f"  消费{i+1}: {'成功' if success else '不足'}", end="")
        if i == 9:
            print()

    print(f"剩余令牌: {test_bucket.available_tokens():.1f}")
    print(f"需要等待: {test_bucket.wait_time_for(1):.2f}s 补充1个令牌")

    # 3. 速率检查演示
    print("\n--- 速率限制检查 ---")

    # 模拟快速连续请求
    endpoints = ["query", "ingest", "admin"]
    results_count = {"allowed": 0, "queued": 0, "rejected": 0}

    for i in range(30):
        ep = endpoints[i % 3]
        result, info = limiter.check("free_tenant", ep, tokens=1)

        results_count[result.value] += 1

        if i < 5 or result != RateLimitResult.ALLOWED:
            status = {"allowed": "✅", "queued": "⏳", "rejected": "❌"}
            print(f"  请求{i+1}: {status[result.value]} {ep} "
                  f"(剩余:{info.remaining}, 限:{info.limit}, "
                  f"Retry-After:{info.retry_after:.1f}s)")

    print(f"\n 结果: 允许={results_count['allowed']}, 排队={results_count['queued']}, 拒绝={results_count['rejected']}")

    # 4. 响应头生成
    print("\n--- HTTP响应头 ---")
    result, info = limiter.check("pro_tenant", "query", tokens=1)
    headers = limiter.generate_headers(info)
    for header, value in headers.items():
        print(f"  {header}: {value}")

    # 5. 不同等级对比
    print("\n--- 不同等级对比 ---")
    for tier in ["free", "pro", "enterprise"]:
        # 先重置
        limiter.reset(f"{tier}_tenant", "query")

        # 快速连续请求直到被限制
        success_count = 0
        for _ in range(1000):
            result, _ = limiter.check(f"{tier}_tenant", "query", tokens=1)
            if result == RateLimitResult.ALLOWED:
                success_count += 1
            else:
                break

        config = limiter.get_config(f"{tier}_tenant", "query")
        print(f"  {tier:12s}: 突发支持 {success_count} 请求 "
              f"(配置: {config.requests_per_second}rps, burst={config.burst_size})")

    # 6. 基于token的速率限制
    print("\n--- Token-based速率限制 ---")

    # 重置
    limiter.reset("free_tenant", "query")

    test_queries = [
        ("短查询", "什么是RAG？"),
        ("中等查询", "请详细解释RAG系统中检索增强生成的工作原理和各个组件如何协同工作？"),
        ("长查询", "请从架构、性能、安全、成本和可扩展性等多个维度详细分析生产级RAG系统的设计原则和最佳实践？"),
    ]

    for name, query in test_queries:
        tokens = TokenEstimator.estimate_request_tokens(query, context_chunks=5)
        result, info = limiter.check("free_tenant", "query", tokens=max(1, tokens // 100))

        status = {"allowed": "✅", "queued": "⏳", "rejected": "❌"}
        print(f"  {name}: ~{tokens} token, 结果={status[result.value]} (剩余={info.remaining})")

    # 7. 统计报告
    print("\n--- 统计报告 ---")
    all_stats = limiter.get_all_stats()
    for key, stats in all_stats.items():
        if stats["bucket"]:
            b = stats["bucket"]
            print(f"  {key}: 可用={b['available']:.1f}, 消耗={b['total_consumed']}, 拒绝={b['total_rejected']}")

    # 8. 装饰器/上下文管理器演示
    print("\n--- 上下文管理器演示 ---")
    try:
        with RateLimited(limiter, "free_tenant", "query", tokens=1) as rl:
            print(f"  请求通过 (结果: {rl.result.value})")
            print(f"  响应头: {rl.headers}")
    except RateLimitExceededError as e:
        print(f"  请求被拒绝: {e}, Retry-After={e.retry_after:.0f}s")

    # 9. Token估算
    print("\n--- Token估算 ---")
    sample_query = "什么是生产级RAG系统的最佳架构设计？"
    for chunks in [0, 3, 5, 10]:
        tokens = TokenEstimator.estimate_request_tokens(sample_query, context_chunks=chunks)
        print(f"  chunks={chunks}: ~{tokens} tokens")

    print("\n" + "=" * 60)
    print("速率限制器演示完成！")
    print("=" * 60)
