# 坑点 #6：Redis 锁过期（Redis Lock Expiry）

> "Graph执行了58秒，Redis锁的TTL是30秒。第31秒时锁过期了，另一个实例获得了锁，开始执行同一个任务。结果是：同一笔交易被执行了两次！"

---

## 症状

1. **任务重复执行**：同一个任务被多个实例同时处理
2. **数据不一致**：两个实例写入冲突的数据
3. **锁续约失败但不报错**：续约线程静默失败
4. **Redis连接断开**：续约线程因网络问题无法续约
5. **时间不一致**：多台服务器时钟不同步导致锁判断异常

---

## 根因

Redis锁的TTL（Time To Live）是有限制的。如果Graph任务执行时间超过锁的TTL，锁会自动过期释放。此时其他等待中的实例会获取锁并开始执行同一任务。

常见场景：
- 大文档RAG检索耗时 > 锁TTL
- 批处理任务执行时间长
- 网络延迟导致续约失败
- Redlock算法的时钟漂移问题

---

## 解法

### 1. 超时保护 + 锁续约双保险

```python
class RobustRedisLock:
    """
    鲁棒的Redis分布式锁实现。

    双重保护：
    1. 锁TTL设置得足够长（任务预估时间的2-3倍）
    2. 续约线程每TTL/3时间续约一次
    3. 如果续约失败，强制任务进入安全关闭流程
    """

    def __init__(
        self,
        lock_key: str,
        ttl_ms: int = 120_000,        # 2分钟，比之前的30秒更长
        renewal_interval_ms: int = 40_000,  # 每40秒续约（TTL/3）
        max_task_duration_ms: int = 100_000,  # 任务最长时间限制
    ):
        self.lock_key = lock_key
        self.ttl_ms = ttl_ms
        self.renewal_interval_ms = renewal_interval_ms
        self.max_task_duration_ms = max_task_duration_ms
        self._acquired = False
        self._renewal_failed = threading.Event()
        self._task_start: float = 0

    def execute_with_lock(self, task_fn: callable, *args, **kwargs) -> Any:
        """
        带锁执行任务。如果锁丢失，立即停止任务。

        Args:
            task_fn: 要执行的任务函数
            *args, **kwargs: 任务参数

        Returns:
            任务结果

        Raises:
            LockLostError: 如果在执行过程中锁丢失
        """
        if not self.acquire():
            raise TimeoutError(f"无法获取锁: {self.lock_key}")

        self._task_start = time.time()

        try:
            # 在后台线程中执行任务
            result_container = []
            exception_container = []

            def task_wrapper():
                try:
                    result_container.append(task_fn(*args, **kwargs))
                except Exception as e:
                    exception_container.append(e)

            task_thread = threading.Thread(target=task_wrapper, daemon=True)
            task_thread.start()

            # 监控：检查锁是否仍然持有，任务是否超时
            while task_thread.is_alive():
                elapsed = (time.time() - self._task_start) * 1000

                # 检查锁是否丢失
                if self._renewal_failed.is_set():
                    raise LockLostError(f"锁已丢失: {self.lock_key}")

                # 检查任务是否超时
                if elapsed > self.max_task_duration_ms:
                    raise TimeoutError(
                        f"任务超时 ({elapsed:.0f}ms > {self.max_task_duration_ms}ms)"
                    )

                task_thread.join(timeout=0.1)

            # 任务完成
            if exception_container:
                raise exception_container[0]
            return result_container[0] if result_container else None

        finally:
            self.release()

    def _renewal_loop(self):
        """续约循环 + 失败检测。"""
        while not self._renewal_stop.wait(self.renewal_interval_ms / 1000):
            success = self._renew()
            if not success:
                self._renewal_failed.set()  # 通知主线程锁已丢失
                break


class LockLostError(Exception):
    """锁在执行过程中丢失的异常。"""
    pass
```

### 2. 任务幂等性设计

```python
class IdempotentTask:
    """
    幂等任务设计：即使任务被执行多次，结果也是一致的。

    关键：使用唯一ID检测重复执行。
    """

    def __init__(self, task_id: str, db_connection):
        self.task_id = task_id
        self.db = db_connection

    def check_and_mark_processing(self) -> bool:
        """
        原子操作：检查并标记任务为'处理中'。
        如果已经被标记，返回False（避免重复执行）。
        """
        result = self.db.execute(
            "UPDATE tasks SET status = 'processing' "
            "WHERE id = ? AND status = 'pending' "
            "RETURNING id",
            (self.task_id,)
        )
        return result is not None

    def execute(self, task_fn: callable) -> Any:
        """
        幂等执行：先标记再执行。
        """
        if not self.check_and_mark_processing():
            return {"status": "skipped", "reason": "already_processed"}

        try:
            result = task_fn()
            self.db.execute(
                "UPDATE tasks SET status = 'completed', result = ? WHERE id = ?",
                (json.dumps(result), self.task_id)
            )
            return result
        except Exception as e:
            self.db.execute(
                "UPDATE tasks SET status = 'failed', error = ? WHERE id = ?",
                (str(e), self.task_id)
            )
            raise
```

---

## 检查清单

- [ ] 锁TTL设置为任务预估最长执行时间的2-3倍
- [ ] 续约间隔 = TTL / 3（留有足够余量）
- [ ] 续约失败时触发告警并强制停止任务
- [ ] 任务设计了幂等性（重复执行不产生副作用）
- [ ] 对关键操作使用了数据库层面的幂等检查
- [ ] Redis连接有重试和健康检查机制
- [ ] 多机部署时考虑了时钟同步问题

---

**一句话总结**：Redis锁不是银弹——TTL到期后它会自动消失。对于长时间运行的任务，锁续约不是可选项，而是必需品。加上幂等设计作为最后防线。
