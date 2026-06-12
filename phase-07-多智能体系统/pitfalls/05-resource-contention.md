# 坑点 #5：资源争抢（Resource Contention）

> "5个Agent同时运行在8GB内存的服务器上。Agent A正在加载一个2GB的模型，Agent B也在做同样的事。结果：OOM Killer杀掉了3个Agent进程，整个任务链断裂，用户只看到了一个500错误。"

---

## 症状

1. **OOM（内存溢出）**：多Agent同时加载大模型或处理大数据时，内存耗尽
2. **API限流（429 Too Many Requests）**：多Agent同时调用同一个LLM API，触发速率限制
3. **数据库连接池耗尽**：每个Agent都打开数据库连接，连接池被占满
4. **GPU显存争抢**：多个Agent推理任务同时提交，GPU OOM
5. **文件锁冲突**：多Agent尝试读写同一文件，导致死锁或数据损坏
6. **非确定性失败**：同一个任务有时成功有时失败，取决于Agent执行的时序

---

## 根因

**多Agent系统本质上是并发系统，但开发者往往忽视了并发控制。**

两个层面的问题：

1. **基础设施层面**：CPU、内存、GPU、网络带宽、磁盘IO是有限资源，多Agent并发访问必然争抢
2. **外部服务层面**：LLM API有速率限制、数据库有连接数上限、向量数据库有查询并发限制

在单Agent系统中这些问题不容易暴露，但在多Agent系统中并发度被放大，资源瓶颈迅速浮现。

---

## 解法

### 1. 信号量控制并发度

```python
import threading
import time
from contextlib import contextmanager

class ResourceManager:
    """
    资源管理器：使用信号量（Semaphore）控制对不同资源的并发访问。
    """

    def __init__(self):
        # 各类资源的并发限制
        self._semaphores = {
            "llm_api": threading.Semaphore(3),        # LLM API最多3个并发
            "database": threading.Semaphore(10),       # 数据库连接池10个
            "file_io": threading.Semaphore(5),         # 文件IO最多5个并发
            "gpu_inference": threading.Semaphore(2),   # GPU推理最多2个并发
        }

        # 资源使用统计
        self._usage_stats: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self, resource_type: str, timeout: float = 30.0):
        """
        获取资源访问权限（上下文管理器）。

        Usage:
            with resource_mgr.acquire("llm_api", timeout=10):
                response = call_llm_api(...)
        """
        semaphore = self._semaphores.get(resource_type)
        if semaphore is None:
            raise ValueError(f"Unknown resource type: {resource_type}")

        # 尝试获取信号量
        acquired = semaphore.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(
                f"Failed to acquire '{resource_type}' within {timeout}s. "
                f"Current usage: {self.get_usage(resource_type)}"
            )

        start_time = time.time()
        try:
            yield
        finally:
            semaphore.release()
            elapsed = time.time() - start_time
            with self._lock:
                self._usage_stats.setdefault(resource_type, []).append(elapsed)

    def get_usage(self, resource_type: str) -> dict:
        """获取资源使用情况。"""
        semaphore = self._semaphores.get(resource_type)
        if semaphore is None:
            return {}

        max_workers = semaphore._value  # 这是一个内部属性，生产环境需要更优雅的包装

        recent_times = self._usage_stats.get(resource_type, [])[-100:]
        avg_time = sum(recent_times) / len(recent_times) if recent_times else 0

        return {
            "resource": resource_type,
            "max_concurrent": max_workers + semaphore._initial_value - semaphore._value,  # 估算
            "recent_avg_time_s": round(avg_time, 3),
            "total_uses": len(self._usage_stats.get(resource_type, [])),
        }

# 使用示例
resource_mgr = ResourceManager()

class ResourceAwareAgent:
    def __init__(self, agent_id: str, resource_mgr: ResourceManager):
        self.agent_id = agent_id
        self.resource_mgr = resource_mgr

    def call_llm(self, prompt: str) -> str:
        """
        带资源控制的LLM调用。
        如果无法在10秒内获取到API访问许可，抛出超时异常。
        """
        with self.resource_mgr.acquire("llm_api", timeout=10):
            # 此时保证并发数在限制内
            response = self._actual_llm_call(prompt)
        return response
```

### 2. 退避重试（处理429限流）

```python
class RateLimitHandler:
    """
    API限流处理器。
    检测到429错误时自动退避重试，而非立即失败。
    """

    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def call_with_retry(self, api_call_fn, *args, **kwargs):
        """
        带退避重试的API调用。

        指数退避序列：1s → 2s → 4s → 8s → 16s（最多5次）
        加上随机抖动（jitter）避免雷群效应（thundering herd）。
        """
        import random

        for attempt in range(self.max_retries):
            try:
                return api_call_fn(*args, **kwargs)
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise  # 最后一次尝试失败，向上抛出

                # 指数退避 + 随机抖动
                delay = self.base_delay * (2 ** attempt)
                jitter = random.uniform(0, delay * 0.5)
                total_delay = delay + jitter

                print(
                    f"[429] Rate limited. Retrying in {total_delay:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(total_delay)

    def adaptive_throttle(self, recent_429_count: int, window_seconds: float = 60):
        """根据最近的429错误数量，自适应降低并发度。"""
        if recent_429_count == 0:
            return 5  # 正常并发度

        if recent_429_count <= 2:
            return 3  # 轻度限流

        if recent_429_count <= 5:
            return 2  # 中度限流

        return 1  # 重度限流，单线程执行
```

### 3. Agent优先级与资源调度

```python
from dataclasses import dataclass
from enum import Enum
import heapq

class AgentPriority(Enum):
    CRITICAL = 0   # 关键任务：立即执行
    HIGH = 1       # 高优先级
    NORMAL = 2     # 普通
    LOW = 3        # 低优先级：可以等待

@dataclass(order=True)
class PrioritizedTask:
    """带优先级的任务，用于优先队列调度。"""
    priority: int
    agent_id: str = field(compare=False)
    task_fn: callable = field(compare=False)
    submitted_at: float = field(default_factory=time.time)

class PriorityScheduler:
    """
    优先级调度器：确保关键Agent的任务优先获取资源。
    """

    def __init__(self, resource_mgr: ResourceManager):
        self.resource_mgr = resource_mgr
        self._queue: list[PrioritizedTask] = []
        self._lock = threading.Lock()

    def submit(self, agent_id: str, priority: AgentPriority, task_fn) -> None:
        """提交任务到调度队列。"""
        task = PrioritizedTask(
            priority=priority.value,
            agent_id=agent_id,
            task_fn=task_fn,
        )
        with self._lock:
            heapq.heappush(self._queue, task)

    def run_next(self, resource_type: str, timeout: float = 30.0) -> Any:
        """执行下一个最高优先级的任务。"""
        with self._lock:
            if not self._queue:
                return None
            task = heapq.heappop(self._queue)

        with self.resource_mgr.acquire(resource_type, timeout=timeout):
            return task.task_fn()
```

---

## 检查清单

- [ ] 为每种受限资源（API、DB、GPU、文件）设置了并发上限
- [ ] 使用信号量（Semaphore）或连接池限制并发访问
- [ ] 实现了指数退避 + 随机抖动的429重试策略
- [ ] 监控了资源使用率（内存、CPU、GPU、连接数）并设置告警
- [ ] 关键Agent有更高的资源获取优先级
- [ ] 在压力测试中验证了多Agent并发场景下的资源使用
- [ ] 有降级策略：当资源不足时，非关键Agent自动降速或暂停

---

**一句话总结**：多Agent系统在资源层面是一个并发系统——用并发控制的思维来管理，而不是假设"Agent会自动协调好"。信号量、连接池、退避重试是标配基础设施。
