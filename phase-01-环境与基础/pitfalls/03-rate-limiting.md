# 坑点 #3：API 速率限制（429 错误）

> "前 20 条文档处理得飞快，第 21 条开始全炸了——这不是代码的 bug，是 API 的流量控制。"

---

## 症状

你正在写一个批量文档处理脚本，要调用 LLM 处理 500 篇文档。前几十个请求顺利返回，然后突然：

1. **HTTP 429 错误**：`"Rate limit exceeded. Please try again in 12 seconds."`
2. **间歇性失败**：有时候成功有时候失败，让你以为有"玄学 bug"
3. **静默丢数据**：没有正确处理 429 重试的代码，请求失败后那条文档就被悄悄跳过了
4. **大批量任务跑了一夜然后全失败**：你睡觉前启动了脚本，醒来发现才处理了 47/500 篇，剩下全是 429
5. **并发请求看起来更快，但崩得也更彻底**：你用 `asyncio.gather` 同时发 50 个请求，第一轮就触发了硬限制，API 直接封了你的 Key 几分钟

---

## 根因

所有 LLM API 都有速率限制（Rate Limiting），分为两个维度：

1. **RPM（Requests Per Minute）**：每分钟允许的请求数
2. **TPM（Tokens Per Minute）**：每分钟允许处理的 token 总数

超出任何一个维度，API 都会返回 HTTP 429。更关键的是，**限制是针对整个 API Key 的**——如果你有多个并发脚本共用一个 Key，它们会互相撞墙。

---

## 各模型速率限制速查表

| 模型 / Tier            | RPM 限制    | TPM 限制        | 备注                          |
|------------------------|------------|-----------------|-------------------------------|
| GPT-4o-mini (Tier 1)   | 500        | 200,000         | 新用户起步档                   |
| GPT-4o-mini (Tier 3)   | 5,000      | 2,000,000       | 消费 $50+ 后自动升级           |
| GPT-4o-mini (Tier 5)   | 30,000     | 150,000,000     | 消费 $1,000+ 后自动升级        |
| GPT-4o (Tier 1)        | 500        | 30,000          | 新用户                         |
| GPT-4o (Tier 5)        | 10,000     | 30,000,000      | 高消费用户                     |
| Claude Sonnet (Tier 1) | 50         | 40,000          | Anthropic 默认限制较保守       |
| Claude Sonnet (Tier 3) | 1,000      | 400,000         | 消费升级后                     |
| Claude Sonnet (Tier 4) | 2,000      | 800,000         | 高消费用户                     |
| Claude Haiku (Tier 1)  | 50         | 50,000          |                                |
| Claude Haiku (Tier 4)  | 2,000      | 1,000,000       |                                |
| Gemini 1.5 Flash       | 1,500      | 4,000,000       | 免费额度内                     |
| Gemini 1.5 Pro          | 360        | 2,000,000       | 免费额度内                     |

**注意**：
- 以上限制可能随时调整，请以官方文档为准
- Anthropic 的限制通过 `Organization → Limits` 查看
- OpenAI 的限制通过 `Platform → Limits` 查看
- 本地模型（Ollama）没有速率限制（但受限于本地显存和算力）

---

## 修复方案

### 方案 1：指数退避重试 + 随机 Jitter

遇到 429 时不要立即重试，而是等待一段时间。等待时间指数增长，同时加入随机因子避免"惊群效应"（多个请求同时重试再次撞墙）。

### 方案 2：Token Bucket 限流器

在客户端侧主动限流，确保每秒/每分钟发出的请求数不超过预设值。

### 方案 3：异步并发控制（Semaphore）

限制同时进行的并发请求数，避免突发流量。

---

## 代码示例

### 指数退避重试（不依赖 tenacity）

```python
# retry_with_backoff.py
"""指数退避重试实现（不依赖第三方库 tenacity）。

核心逻辑：
  1. 遇到 429 时，读取 Retry-After 响应头（如果有）
  2. 否则使用指数退避：1s → 2s → 4s → 8s → 16s（最多 60s）
  3. 每次退避加入随机 jitter（±25%）避免惊群效应
  4. 区分可重试错误（429, 5xx）和不可重试错误（401, 403）
"""

import asyncio
import logging
import random
import time
from typing import TypeVar, Callable, Awaitable, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 默认配置
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0    # 秒
DEFAULT_MAX_DELAY = 60.0    # 秒
DEFAULT_JITTER = 0.25       # ±25%


def _is_retryable(status_code: int) -> bool:
    """判断 HTTP 状态码是否可重试。"""
    # 429 (Rate Limit) 和 5xx (Server Error) 可重试
    return status_code == 429 or 500 <= status_code < 600


def _calculate_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    retry_after: float | None = None,
) -> float:
    """计算本次重试的等待时间。

    Args:
        attempt: 当前是第几次重试（从 1 开始）
        base_delay: 基础等待时间
        max_delay: 最大等待时间
        jitter: 随机抖动比例（0.25 = ±25%）
        retry_after: 服务器建议的等待秒数（从 Retry-After 头获取）

    Returns:
        等待秒数
    """
    if retry_after is not None:
        # 优先使用服务器建议的等待时间
        delay = min(retry_after, max_delay)
    else:
        # 指数退避：base * 2^(attempt-1)
        delay = base_delay * (2 ** (attempt - 1))
        delay = min(delay, max_delay)

    # 添加随机 jitter
    jitter_amount = delay * jitter
    actual_delay = delay + random.uniform(-jitter_amount, jitter_amount)
    return max(0.1, actual_delay)  # 最少等待 0.1 秒


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    retryable_check: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[int, float, Exception], None] | None = None,
    **kwargs: Any,
) -> T:
    """异步指数退避重试包装器。

    Args:
        func: 要重试的异步函数
        *args: 传递给 func 的位置参数
        max_retries: 最大重试次数
        base_delay: 基础退避时间（秒）
        max_delay: 最大退避时间（秒）
        jitter: 随机抖动比例
        retryable_check: 自定义可重试判断函数。返回 True 表示可重试
        on_retry: 重试时的回调函数，参数为 (attempt, delay, exception)
        **kwargs: 传递给 func 的关键字参数

    Returns:
        func 的返回值

    Raises:
        最后一次重试仍然失败时，抛出原始异常
    """
    last_exception = None

    for attempt in range(1, max_retries + 2):  # 1 initial + N retries
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            # 判断是否可重试
            is_retryable = False
            retry_after = None

            # 检查是否有 HTTP 状态码
            status_code = getattr(e, 'status_code', None) or getattr(
                getattr(e, 'response', None), 'status_code', None
            )
            if status_code is not None:
                is_retryable = _is_retryable(status_code)
                # 尝试从响应头获取 Retry-After
                headers = getattr(getattr(e, 'response', None), 'headers', None)
                if headers:
                    retry_after_str = headers.get('Retry-After') or headers.get(
                        'retry-after'
                    )
                    if retry_after_str:
                        try:
                            retry_after = float(retry_after_str)
                        except ValueError:
                            pass

            # 如果自定义判断函数存在，优先使用
            if retryable_check is not None:
                is_retryable = retryable_check(e)

            # 不可重试，或已达最大重试次数
            if not is_retryable or attempt > max_retries:
                logger.error(
                    f"重试 {attempt-1}/{max_retries} 次后仍然失败: {e}"
                )
                raise

            # 计算等待时间
            delay = _calculate_delay(attempt, base_delay, max_delay, jitter, retry_after)

            logger.warning(
                f"请求失败 ({e})，"
                f"{delay:.1f}s 后进行第 {attempt}/{max_retries} 次重试..."
            )

            if on_retry:
                on_retry(attempt, delay, e)

            await asyncio.sleep(delay)

    # 这行理论上不会执行到
    raise last_exception  # type: ignore


# ============================================================
# 同步版本
# ============================================================

def retry_with_backoff_sync(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    **kwargs: Any,
) -> T:
    """同步版指数退避重试。"""
    last_exception = None

    for attempt in range(1, max_retries + 2):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            status_code = getattr(e, 'status_code', None) or getattr(
                getattr(e, 'response', None), 'status_code', None
            )
            if status_code and not _is_retryable(status_code):
                raise
            if attempt > max_retries:
                raise

            delay = _calculate_delay(attempt, base_delay, max_delay, jitter)
            logger.warning(
                f"请求失败，{delay:.1f}s 后重试 ({attempt}/{max_retries})..."
            )
            time.sleep(delay)

    raise last_exception


# ============================================================
# Token Bucket 限流器
# ============================================================

class TokenBucket:
    """基于 Token Bucket 算法的限流器。

    原理：
      - 桶的容量为 max_tokens
      - 每秒自动补充 refill_rate 个 token
      - 每次请求需要消耗 tokens_per_request 个 token
      - 如果桶中 token 不足，请求被延迟或拒绝

    用法：
        limiter = TokenBucket(rate=100, period=60)  # 60秒最多100个请求
        async with limiter:
            await call_api()
    """

    def __init__(
        self,
        rate: int = 100,        # 在 period 时间内允许的请求数
        period: float = 60.0,   # 时间窗口（秒）
        burst: int | None = None,  # 突发容量（默认等于 rate）
    ):
        """
        Args:
            rate: 每个时间窗口允许的请求数
            period: 时间窗口（秒）
            burst: 桶的最大容量（允许的突发请求数）
        """
        self.rate = rate
        self.period = period
        self.refill_rate = rate / period  # 每秒补充的 token 数
        self.max_tokens = burst if burst is not None else rate
        self._tokens = float(self.max_tokens)  # 当前 token 数
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """根据经过的时间补充 token。"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.max_tokens,
            self._tokens + elapsed * self.refill_rate,
        )
        self._last_refill = now

    async def acquire(self) -> float:
        """获取一个请求许可。如果需要等待，返回等待秒数。"""
        async with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            # 计算需要等待的时间
            wait_time = (1.0 - self._tokens) / self.refill_rate
            self._tokens = 0.0
            return wait_time

    async def __aenter__(self):
        wait = await self.acquire()
        if wait > 0:
            logger.debug(f"TokenBucket: 等待 {wait:.2f}s 获取许可")
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def available_tokens(self) -> float:
        """当前可用的 token 数（近似值，非线程安全）。"""
        return self._tokens


class TokenBucketSync:
    """Token Bucket 的同步版本。"""

    def __init__(self, rate: int = 100, period: float = 60.0, burst: int | None = None):
        self.rate = rate
        self.period = period
        self.refill_rate = rate / period
        self.max_tokens = burst if burst is not None else rate
        self._tokens = float(self.max_tokens)
        self._last_refill = time.monotonic()
        import threading
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def acquire(self) -> float:
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            wait_time = (1.0 - self._tokens) / self.refill_rate
            self._tokens = 0.0
            return wait_time

    def __enter__(self):
        wait = self.acquire()
        if wait > 0:
            time.sleep(wait)
        return self

    def __exit__(self, *args):
        pass


# ============================================================
# 异步并发控制器（Semaphore + TokenBucket）
# ============================================================

class AsyncLimiter:
    """组合限流器：同时控制并发数和请求速率。

    用法：
        limiter = AsyncLimiter(
            max_concurrent=10,    # 最多 10 个并发请求
            max_rpm=500,          # 每分钟最多 500 个请求
        )

        sem = asyncio.Semaphore(10)
        async with limiter:
            result = await call_llm_api(prompt)
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        max_rpm: int | None = None,
        max_tpm: int | None = None,
    ):
        """
        Args:
            max_concurrent: 最大并发请求数
            max_rpm: 每分钟最大请求数（None 则不限制）
            max_tpm: 每分钟最大 token 数（None 则不限制）
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._bucket = (
            TokenBucket(rate=max_rpm, period=60.0)
            if max_rpm is not None
            else None
        )
        self._token_bucket = (
            TokenBucket(rate=max_tpm, period=60.0)
            if max_tpm is not None
            else None
        )
        self.max_concurrent = max_concurrent
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm

    async def __aenter__(self):
        """进入限流区：等待并发槽位 + 速率许可。"""
        await self._semaphore.acquire()
        if self._bucket:
            await self._bucket.acquire()
        return self

    async def __aexit__(self, *args):
        """离开限流区：释放并发槽位。"""
        self._semaphore.release()

    async def acquire_with_tokens(self, token_count: int = 0) -> None:
        """获取许可的同时报告将使用的 token 数。

        Args:
            token_count: 本次请求预计消耗的 token 数
        """
        await self._semaphore.acquire()
        if self._bucket:
            await self._bucket.acquire()
        if self._token_bucket and token_count > 0:
            # TPM 限制：按比例消耗 token
            wait = await self._token_bucket.acquire()
            if wait > 0:
                await asyncio.sleep(wait)

    def release(self) -> None:
        """释放并发槽位。"""
        self._semaphore.release()


# ============================================================
# 完整示例：批量文档处理
# ============================================================

async def demo_batch_processing():
    """演示如何使用限流器批量处理文档。"""

    # 模拟的 LLM API 调用
    async def mock_call_api(doc_id: int, prompt: str) -> dict:
        """模拟 API 调用（实际使用中替换为真实的 API 调用）。"""
        # 模拟不同的处理时间
        delay = 0.1 + random.random() * 0.3
        await asyncio.sleep(delay)

        # 模拟随机失败（5% 概率返回 429）
        if random.random() < 0.05:
            class Fake429Error(Exception):
                status_code = 429
            raise Fake429Error("Rate limit exceeded")

        return {
            "doc_id": doc_id,
            "result": f"Processed: {prompt[:30]}...",
            "tokens_used": len(prompt) // 4,
        }

    # 创建限流器
    limiter = AsyncLimiter(max_concurrent=5, max_rpm=30)

    # 要处理的文档
    documents = [
        f"这是第 {i} 篇文档的内容，包含大量需要 LLM 处理的文本。" * 10
        for i in range(50)
    ]

    results = []
    errors = []

    async def process_one(doc_id: int, content: str):
        """处理单篇文档（带重试和限流）。"""
        prompt = f"请总结以下文档：\n\n{content}"

        async def api_call():
            async with limiter:
                return await mock_call_api(doc_id, prompt)

        try:
            result = await retry_with_backoff(
                api_call,
                max_retries=3,
                base_delay=1.0,
            )
            results.append(result)
            return result
        except Exception as e:
            errors.append((doc_id, str(e)))
            return None

    # 并发处理所有文档（并发数由 limiter 控制）
    tasks = [process_one(i, doc) for i, doc in enumerate(documents)]
    await asyncio.gather(*tasks)

    # 打印汇总
    print(f"处理完成: {len(results)} 成功, {len(errors)} 失败")
    print(f"总耗时: 由 limiter 控制不超过 RPM 限制")
    if errors:
        print(f"失败列表: {errors[:5]}...")


# ============================================================
# 装饰器版本：快速为重试添加退避
# ============================================================

import functools


def with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
):
    """为任意 async 函数添加指数退避重试的装饰器。

    用法:
        @with_backoff(max_retries=5)
        async def call_api(prompt: str) -> dict:
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                **kwargs,
            )
        return wrapper
    return decorator


# 使用示例
@with_backoff(max_retries=5, base_delay=2.0)
async def robust_api_call(prompt: str) -> dict:
    """自动带退避重试的 API 调用。"""
    # 实际的 API 调用...
    ...


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_batch_processing())
```

---

## 检查清单

在批量调用 LLM API 之前，确认以下 3 项：

- [ ] **已知当前的 RPM/TPM 限制**: 不是凭记忆，是去 OpenAI/Anthropic 控制台的 Limits 页面查看实时的限制值（Tier 等级会随消费额度自动变化）
- [ ] **已实现指数退避重试**: 429 错误不应该让程序直接崩溃或跳过数据——每次 429 都应该被捕获并自动重试
- [ ] **批量任务有并发控制**: `asyncio.gather` 传入 100 个任务不等于同时跑 100 个——用 Semaphore 控制并发数，用 TokenBucket 控制速率

三项全空 = 批量任务必然失败。

---

## 延伸阅读

- [OpenAI: Rate Limits Guide](https://platform.openai.com/docs/guides/rate-limits)
- [Anthropic: Rate Limits](https://docs.anthropic.com/en/api/rate-limits)
- [Google AI: Quotas and Limits](https://ai.google.dev/pricing)
- [AWS: Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)

---

**一句话总结**：429 不是 bug，是 API 在跟你说"慢一点"。不听劝的代价是数据丢失和账单翻倍。
