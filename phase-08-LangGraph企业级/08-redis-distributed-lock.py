#!/usr/bin/env python3
"""
Phase 08 - LangGraph 企业级：08 - Redis 分布式锁

防止多个服务实例同时执行同一个Graph任务。
使用 Redis 锁 + 锁续约（Renewal）+ 自动释放。

核心概念：
  ┌──────────────────────────────────────────────────────────┐
  │                  Redis 分布式锁                           │
  │                                                          │
  │  Instance A              Instance B                      │
  │  ┌─────────────┐        ┌─────────────┐                  │
  │  │ 获取锁       │        │ 获取锁       │                  │
  │  │ lock("task-1")│       │ lock("task-1")│ ← 失败！       │
  │  │   ↓         │        │   ↓         │  锁已被A持有     │
  │  │ 执行Graph    │        │ 返回"忙碌"   │                  │
  │  │   ↓         │        │  或等待重试   │                  │
  │  │ 续约锁       │        └─────────────┘                  │
  │  │   ↓         │                                          │
  │  │ 释放锁       │                                          │
  │  └─────────────┘                                          │
  └──────────────────────────────────────────────────────────┘

锁Key模式：graph-lock:{thread_id}:{node_name}
锁超时：30s（默认），通过续约线程每10s续约一次
释放：执行完成或异常时自动释放
"""

import os
import time
import uuid
import json
import threading
import functools
from typing import TypedDict, Annotated, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================================
# Redis 锁实现（支持真实Redis和模拟模式）
# ============================================================================

# 尝试导入 redis，如果不可用则使用模拟模式
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisDistributedLock:
    """
    Redis 分布式锁。

    特性：
    1. 基于 Redis SET NX PX 的原子操作
    2. 使用 Lua 脚本保证释放锁的安全性（只释放自己持有的锁）
    3. 自动续约（Renewal）：后台线程定期延长锁的TTL
    4. 优雅释放：finally块确保锁一定被释放
    5. 支持超时等待：获取锁失败时等待重试

    使用方式：
        lock = RedisDistributedLock("task-123")
        with lock:
            # 执行Graph
            result = graph.invoke(state)
    """

    # Lua脚本: 安全释放锁（仅释放自己持有的锁）
    RELEASE_LOCK_SCRIPT = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """

    # Lua脚本: 续约锁
    RENEW_LOCK_SCRIPT = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("PEXPIRE", KEYS[1], ARGV[2])
    else
        return 0
    end
    """

    def __init__(
        self,
        lock_key: str,
        ttl_ms: int = 30000,        # 锁的TTL（毫秒）
        renewal_interval_ms: int = 10000,  # 续约间隔（毫秒）
        acquire_timeout_ms: int = 10_000,  # 获取锁超时（毫秒）
        redis_client=None,
        redis_url: str = None,
    ):
        self.lock_key = f"graph-lock:{lock_key}"
        self.ttl_ms = ttl_ms
        self.renewal_interval_ms = renewal_interval_ms
        self.acquire_timeout_ms = acquire_timeout_ms
        self.lock_value = str(uuid.uuid4())  # 锁的唯一标识
        self._acquired = False
        self._renewal_thread: Optional[threading.Thread] = None
        self._renewal_stop = threading.Event()

        # 初始化 Redis 客户端
        if redis_client:
            self._redis = redis_client
        elif redis_url:
            if not REDIS_AVAILABLE:
                raise ImportError("需要安装 redis: pip install redis")
            self._redis = redis.Redis.from_url(redis_url)
        else:
            # 默认连接
            if not REDIS_AVAILABLE:
                raise ImportError("需要安装 redis: pip install redis")
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD", None),
                db=int(os.getenv("REDIS_DB", "0")),
                decode_responses=True,
            )

    # ---- 锁操作 ----

    def acquire(self) -> bool:
        """
        获取锁。

        Returns:
            True 如果成功获取，False 如果超时
        """
        start = time.time()

        while True:
            # 尝试 SET key value NX PX ttl（原子操作）
            acquired = self._redis.set(
                self.lock_key,
                self.lock_value,
                nx=True,
                px=self.ttl_ms,
            )

            if acquired:
                self._acquired = True
                # 启动续约线程
                self._start_renewal()
                return True

            # 检查是否超时
            elapsed_ms = (time.time() - start) * 1000
            if elapsed_ms >= self.acquire_timeout_ms:
                return False

            # 等待后重试
            time.sleep(min(0.1, self.acquire_timeout_ms / 10000))

    def release(self) -> bool:
        """
        释放锁。

        Returns:
            True 如果成功释放，False 如果锁已被其他人持有或已过期
        """
        if not self._acquired:
            return False

        # 停止续约
        self._stop_renewal()

        # 安全释放（Lua脚本保证原子性）
        try:
            lua_release = self._redis.register_script(self.RELEASE_LOCK_SCRIPT)
            result = lua_release(keys=[self.lock_key], args=[self.lock_value])
            self._acquired = False
            return result == 1
        except Exception:
            self._acquired = False
            return False

    def is_locked(self) -> bool:
        """检查锁是否被持有。"""
        return self._redis.exists(self.lock_key) > 0

    def get_lock_owner(self) -> Optional[str]:
        """获取当前锁的持有者。"""
        return self._redis.get(self.lock_key)

    # ---- 续约机制 ----

    def _start_renewal(self):
        """启动锁续约后台线程。"""
        self._renewal_stop.clear()
        self._renewal_thread = threading.Thread(
            target=self._renewal_loop,
            daemon=True,
            name=f"lock-renewal-{self.lock_key}",
        )
        self._renewal_thread.start()

    def _stop_renewal(self):
        """停止续约线程。"""
        self._renewal_stop.set()
        if self._renewal_thread and self._renewal_thread.is_alive():
            self._renewal_thread.join(timeout=2.0)

    def _renewal_loop(self):
        """续约循环：定期延长锁的TTL。"""
        lua_renew = self._redis.register_script(self.RENEW_LOCK_SCRIPT)

        while not self._renewal_stop.wait(self.renewal_interval_ms / 1000):
            try:
                result = lua_renew(
                    keys=[self.lock_key],
                    args=[self.lock_value, self.ttl_ms],
                )
                if result != 1:
                    # 锁已丢失（可能是被动过期）
                    self._acquired = False
                    break
            except Exception:
                # Redis连接问题，停止续约
                break

    # ---- 上下文管理器 ----

    def __enter__(self) -> "RedisDistributedLock":
        acquired = self.acquire()
        if not acquired:
            raise TimeoutError(
                f"无法在 {self.acquire_timeout_ms}ms 内获取锁: {self.lock_key}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False  # 不抑制异常

    def __del__(self):
        """析构时确保释放锁。"""
        try:
            self.release()
        except Exception:
            pass


# ============================================================================
# 模拟 Redis 锁（用于无Redis环境的演示）
# ============================================================================

class MockRedisDistributedLock:
    """
    模拟 Redis 锁（使用内存字典 + threading.Lock）。

    用于演示和测试，不适用于真正的分布式环境（同一进程内有效）。
    """

    # 全局锁存储（跨实例共享）
    _global_locks: dict[str, str] = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        lock_key: str,
        ttl_ms: int = 30000,
        renewal_interval_ms: int = 10000,
        acquire_timeout_ms: int = 10_000,
        **kwargs,
    ):
        self.lock_key = f"graph-lock:{lock_key}"
        self.ttl_ms = ttl_ms
        self.renewal_interval_ms = renewal_interval_ms
        self.acquire_timeout_ms = acquire_timeout_ms
        self.lock_value = str(uuid.uuid4())
        self._acquired = False
        self._renewal_thread: Optional[threading.Thread] = None
        self._renewal_stop = threading.Event()
        self._expires_at: float = 0

    def acquire(self) -> bool:
        start = time.time()

        while True:
            with self._global_lock:
                current_owner = self._global_locks.get(self.lock_key)

                # 检查锁是否可用（不存在或已过期）
                if current_owner is None or (self._expires_at > 0 and time.time() > self._expires_at):
                    self._global_locks[self.lock_key] = self.lock_value
                    self._expires_at = time.time() + self.ttl_ms / 1000
                    self._acquired = True
                    self._start_renewal()
                    return True

            elapsed_ms = (time.time() - start) * 1000
            if elapsed_ms >= self.acquire_timeout_ms:
                return False

            time.sleep(0.1)

    def release(self) -> bool:
        if not self._acquired:
            return False

        self._stop_renewal()

        with self._global_lock:
            if self._global_locks.get(self.lock_key) == self.lock_value:
                del self._global_locks[self.lock_key]
                self._acquired = False
                return True
            self._acquired = False
            return False

    def is_locked(self) -> bool:
        with self._global_lock:
            owner = self._global_locks.get(self.lock_key)
            if owner is None:
                return False
            if time.time() > self._expires_at:
                return False
            return True

    def _start_renewal(self):
        self._renewal_stop.clear()
        self._renewal_thread = threading.Thread(
            target=self._renewal_loop,
            daemon=True,
            name=f"mock-lock-renewal-{self.lock_key}",
        )
        self._renewal_thread.start()

    def _stop_renewal(self):
        self._renewal_stop.set()
        if self._renewal_thread and self._renewal_thread.is_alive():
            self._renewal_thread.join(timeout=1.0)

    def _renewal_loop(self):
        while not self._renewal_stop.wait(self.renewal_interval_ms / 1000):
            with self._global_lock:
                if self._global_locks.get(self.lock_key) == self.lock_value:
                    self._expires_at = time.time() + self.ttl_ms / 1000
                else:
                    self._acquired = False
                    break

    def __enter__(self):
        acquired = self.acquire()
        if not acquired:
            raise TimeoutError(
                f"无法在 {self.acquire_timeout_ms}ms 内获取锁: {self.lock_key}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# 根据环境选择锁实现
if REDIS_AVAILABLE:
    DistributedLock = RedisDistributedLock
else:
    DistributedLock = MockRedisDistributedLock


# ============================================================================
# 锁装饰器
# ============================================================================

def with_distributed_lock(
    lock_key: str,
    ttl_ms: int = 30000,
    acquire_timeout_ms: int = 10_000,
    on_failure: str = "raise",
):
    """
    装饰器：在执行函数前获取分布式锁。

    Args:
        lock_key: 锁的键名
        ttl_ms: 锁TTL
        acquire_timeout_ms: 获取锁超时
        on_failure: "raise" | "return_none" | "return_busy"

    Usage:
        @with_distributed_lock("task-{task_id}")
        def process_task(task_id):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 支持锁键中的占位符
            key = lock_key.format(
                args=args,
                kwargs=kwargs,
                **kwargs,
            )

            lock = DistributedLock(
                lock_key=key,
                ttl_ms=ttl_ms,
                acquire_timeout_ms=acquire_timeout_ms,
            )

            try:
                with lock:
                    return func(*args, **kwargs)
            except TimeoutError as e:
                if on_failure == "return_none":
                    return None
                elif on_failure == "return_busy":
                    return {"status": "busy", "message": "正在被其他实例处理"}
                else:
                    raise

        return wrapper
    return decorator


# ============================================================================
# 带锁保护的 Graph 执行
# ============================================================================

class GraphTask(TypedDict):
    """简化的任务State。"""
    task_id: str
    task_data: str
    result: str
    status: str  # "pending" | "processing" | "completed" | "error"


def node_process(state: GraphTask) -> dict:
    """处理节点：模拟耗时操作。"""
    print(f"  [process] 处理任务 {state['task_id']}: {state['task_data'][:50]}...")
    time.sleep(0.2)  # 模拟处理
    return {
        "result": f"任务 {state['task_id']} 已处理完成",
        "status": "completed",
    }


def build_task_graph() -> StateGraph:
    """构建简单的任务处理图。"""
    builder = StateGraph(GraphTask)
    builder.add_node("process", node_process)
    builder.set_entry_point("process")
    builder.add_edge("process", END)
    return builder.compile()


class LockedGraphExecutor:
    """
    带锁保护的Graph执行器。

    确保同一个 task_id 不会被多个实例同时处理。
    """

    def __init__(self, graph, lock_ttl_ms: int = 30000):
        self.graph = graph
        self.lock_ttl_ms = lock_ttl_ms
        self.execution_count = 0
        self.lock_failures = 0

    def execute_with_lock(
        self,
        task_id: str,
        state: dict,
        acquire_timeout_ms: int = 5000,
    ) -> Optional[dict]:
        """
        带锁执行Graph。

        Args:
            task_id: 任务ID（也用作锁键）
            state: 图的初始状态
            acquire_timeout_ms: 锁获取超时

        Returns:
            执行结果，如果无法获取锁则返回{"status": "busy"}
        """
        lock = DistributedLock(
            lock_key=f"graph-task:{task_id}",
            ttl_ms=self.lock_ttl_ms,
            acquire_timeout_ms=acquire_timeout_ms,
        )

        try:
            with lock:
                print(f"  🔒 获取锁成功: graph-task:{task_id}")
                self.execution_count += 1
                result = self.graph.invoke(state)
                print(f"  🔓 释放锁: graph-task:{task_id}")
                return result
        except TimeoutError:
            self.lock_failures += 1
            print(f"  ⚠️ 无法获取锁（任务可能正在被其他实例处理）: graph-task:{task_id}")
            return {
                "status": "busy",
                "task_id": task_id,
                "message": "任务正在被其他实例处理，请稍后重试",
            }

    def get_stats(self) -> dict:
        return {
            "executions": self.execution_count,
            "lock_failures": self.lock_failures,
            "failure_rate": round(
                self.lock_failures / max(1, self.execution_count + self.lock_failures), 3
            ),
        }


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  LangGraph 企业级 - Redis 分布式锁演示")
    print("=" * 72)

    if not REDIS_AVAILABLE:
        print("\n  ⚠️ 未安装 redis 库，使用模拟锁模式")
        print("  安装: pip install redis")
        print("  启动: docker run -d --name redis -p 6379:6379 redis:7\n")
    else:
        print("\n  ✅ Redis 已连接")

    # ---- Demo 1: 基本锁操作 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 基本锁获取/释放")
    print("=" * 72)

    lock1 = DistributedLock("demo-1", ttl_ms=5000)

    print(f"  尝试获取锁: {lock1.lock_key}")
    acquired = lock1.acquire()
    print(f"  获取结果: {'✅ 成功' if acquired else '❌ 失败'}")
    print(f"  锁状态: {'🔒 已锁定' if lock1.is_locked() else '🔓 未锁定'}")

    # 尝试再次获取同一个锁（应该失败）
    lock2 = DistributedLock("demo-1", acquire_timeout_ms=500)
    print(f"\n  另一个实例尝试获取同一锁...")
    try:
        with lock2:
            print(f"  居然成功了（不应该！）")
    except TimeoutError:
        print(f"  ❌ 无法获取（预期行为：锁已被lock1持有）")

    # 释放锁1
    released = lock1.release()
    print(f"\n  释放锁1: {'✅' if released else '❌'}")

    # 现在lock2应该能获取
    try:
        with lock2:
            print(f"  释放后，lock2获取锁: ✅ 成功（预期）")
    except TimeoutError:
        print(f"  ❌ 释放后仍无法获取（不应该）")

    # ---- Demo 2: 上下文管理器 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 上下文管理器模式")
    print("=" * 72)

    task_lock = DistributedLock("task-process-001", ttl_ms=10_000)

    try:
        with task_lock:
            print(f"  🔒 已获取锁，开始处理任务...")
            time.sleep(0.2)
            print(f"  处理完成")
            # 锁会在退出 with 块时自动释放
    except TimeoutError:
        print(f"  ❌ 获取锁超时")

    print(f"  锁已自动释放: {'🔓' if not task_lock.is_locked() else '🔒 异常！'}")

    # ---- Demo 3: Graph 执行保护 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 保护 Graph 执行（防重复）")
    print("=" * 72)

    graph = build_task_graph()
    executor = LockedGraphExecutor(graph, lock_ttl_ms=5000)

    # 正常执行
    result1 = executor.execute_with_lock(
        task_id="task-001",
        state={
            "task_id": "task-001",
            "task_data": "处理用户上传的文件",
            "result": "",
            "status": "pending",
        },
    )
    print(f"  执行结果: {result1.get('status') if result1 else 'N/A'}")

    # 模拟并发：用线程模拟另一个实例尝试执行同一任务
    def concurrent_executor():
        return executor.execute_with_lock(
            task_id="task-001",
            state={
                "task_id": "task-001",
                "task_data": "另一个实例的请求",
                "result": "",
                "status": "pending",
            },
            acquire_timeout_ms=500,
        )

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # 先启动一个执行（获取锁）
        f1 = pool.submit(
            executor.execute_with_lock,
            "task-002",
            {"task_id": "task-002", "task_data": "实例A", "result": "", "status": "pending"},
        )
        time.sleep(0.05)
        # 尝试另一个实例（应该被锁拒绝）
        f2 = pool.submit(
            executor.execute_with_lock,
            "task-002",
            {"task_id": "task-002", "task_data": "实例B", "result": "", "status": "pending"},
            acquire_timeout_ms=500,
        )

        r1 = f1.result()
        r2 = f2.result()
        print(f"\n  实例A: {r1.get('status') if r1 else 'N/A'}")
        print(f"  实例B: {r2.get('status') if r2 else 'N/A'} (期望: busy)")

    # ---- 统计 ----

    print(f"\n\n📊 执行统计:")
    print(json.dumps(executor.get_stats(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print("  Redis 分布式锁演示完成！")
    print("=" * 72)
