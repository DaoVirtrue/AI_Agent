# 坑点 #7：死信队列溢出（DLQ Overflow）

> "凌晨3点，监控告警响了：'磁盘使用率95%'。检查发现死信队列里有17万条消息——每条消息包含一个50MB的base64图片。你的服务器还有5GB磁盘空间，按这个速度，30分钟后就会彻底宕机。"

---

## 症状

1. **磁盘空间耗尽**：DLQ无限增长，占满所有可用磁盘
2. **告警风暴**：每条进入DLQ的消息都触发告警（N条消息=N条告警）
3. **消息膨胀**：每条消息携带大量冗余数据（base64编码的文件）
4. **无法恢复**：DLQ中消息太多，手动处理不切实际
5. **监控盲区**：DLQ增长没有监控，直到出事才发现

---

## 根因

死信队列的设计初衷是"不要丢弃失败的消息"——但这意味着它天然有无限增长的趋势。如果：
1. 消费者持续失败（上游bug、API变更）
2. 消息体过大（携带文件、图片）
3. 没有DLQ消息的自动清理策略
4. 没有DLQ大小的监控告警

DLQ会从"安全网"变成"定时炸弹"。

---

## 解法

### 1. DLQ容量管理和自动清理

```python
class ManagedDLQ:
    """
    受管理的死信队列：自动限制大小和TTL。
    """

    def __init__(
        self,
        max_size: int = 10_000,         # 最大消息数
        max_age_days: int = 7,          # 最大保留天数
        max_total_size_bytes: int = 1024 * 1024 * 500,  # 500MB总大小上限
        cleanup_batch_size: int = 100,   # 每次清理的批次大小
    ):
        self.max_size = max_size
        self.max_age_days = max_age_days
        self.max_total_size_bytes = max_total_size_bytes
        self.cleanup_batch_size = cleanup_batch_size

        self._messages: deque = deque()
        self._total_bytes: int = 0
        self._lock = threading.Lock()

    def add(self, message: QueueMessage) -> bool:
        """
        添加消息到DLQ。如果超出容量限制，自动清理最旧的消息。

        Returns:
            True 如果成功添加，False 如果消息被立即丢弃
        """
        with self._lock:
            message_size = len(message.to_json().encode('utf-8'))

            # 检查单条消息是否过大
            if message_size > self.max_total_size_bytes * 0.1:
                # 消息过大（超过总容量的10%），记录但不存储完整内容
                truncated = QueueMessage(
                    message_id=message.message_id,
                    task_type=message.task_type,
                    payload={"error": "message_too_large", "original_size": message_size},
                    retry_count=message.retry_count,
                    status=TaskStatus.DEAD_LETTER,
                )
                self._messages.append(truncated)
                # 仍然触发告警
                return False

            # 检查总容量
            while (len(self._messages) >= self.max_size
                   or self._total_bytes + message_size > self.max_total_size_bytes):
                self._evict_oldest()

            self._messages.append(message)
            self._total_bytes += message_size

            return True

    def _evict_oldest(self):
        """淘汰最旧的消息。"""
        if not self._messages:
            return

        oldest = self._messages.popleft()
        evicted_size = len(oldest.to_json().encode('utf-8'))
        self._total_bytes = max(0, self._total_bytes - evicted_size)

        # 记录被淘汰的消息（用于审计）
        logger.warning(
            f"DLQ已满，淘汰最旧消息: {oldest.message_id} "
            f"(创建于 {oldest.created_at})"
        )

    def cleanup_expired(self) -> int:
        """清理超过保留期的消息。返回清理数量。"""
        with self._lock:
            cutoff = time.time() - self.max_age_days * 86400
            old_count = sum(
                1 for m in self._messages if m.created_at < cutoff
            )

            # 保留最近的消息
            self._messages = deque(
                m for m in self._messages if m.created_at >= cutoff
            )

            # 重新计算总大小
            self._total_bytes = sum(
                len(m.to_json().encode('utf-8')) for m in self._messages
            )

            return old_count

    def get_stats(self) -> dict:
        """获取DLQ统计。"""
        with self._lock:
            return {
                "message_count": len(self._messages),
                "total_bytes": self._total_bytes,
                "total_mb": round(self._total_bytes / 1024 / 1024, 2),
                "usage_pct": round(
                    self._total_bytes / self.max_total_size_bytes * 100, 1
                ),
                "oldest_message_age_days": round(
                    (time.time() - self._messages[0].created_at) / 86400, 1
                ) if self._messages else 0,
            }

    def alert_if_needed(self, warning_threshold_pct: float = 70):
        """检查是否接近容量上限。"""
        stats = self.get_stats()
        if stats["usage_pct"] > warning_threshold_pct:
            # 触发告警
            print(
                f"[ALERT] DLQ使用率: {stats['usage_pct']:.0f}% "
                f"({stats['total_mb']:.0f}MB / {self.max_total_size_bytes // 1024 // 1024}MB)"
            )
            return True
        return False
```

### 2. 消息大小限制

```python
class MessageSizeLimiter:
    """
    消息大小限制：在发布到队列之前就检查消息大小。
    """

    MAX_MESSAGE_BYTES = 1024 * 1024  # 1MB

    @classmethod
    def validate(cls, message: QueueMessage) -> list[str]:
        """验证消息是否过大。"""
        issues = []
        size = len(message.to_json().encode('utf-8'))

        if size > cls.MAX_MESSAGE_BYTES:
            issues.append(
                f"消息过大: {size / 1024:.0f}KB > "
                f"{cls.MAX_MESSAGE_BYTES / 1024:.0f}KB 限制"
            )

        # 特别检查：payload中是否有base64编码的数据
        payload = message.payload
        for key, value in payload.items():
            if isinstance(value, str) and len(value) > 100_000:
                if value.startswith(('data:', 'iVBOR')):  # Base64标志
                    issues.append(
                        f"payload['{key}'] 包含base64数据 "
                        f"({len(value) / 1024:.0f}KB)，建议使用对象存储URL替代"
                    )

        return issues

    @classmethod
    def strip_large_payload(cls, message: QueueMessage) -> QueueMessage:
        """
        替换大payload为引用。
        将base64文件替换为对象存储URL。
        """
        payload = dict(message.payload)
        for key, value in list(payload.items()):
            if isinstance(value, str) and len(value) > 100_000:
                # 存储到对象存储，替换为URL
                storage_url = f"s3://rag-files/{message.message_id}/{key}"
                payload[key] = {"type": "s3_reference", "url": storage_url}
                # 实际存储逻辑...
                # s3_client.put_object(Bucket='rag-files', Key=f'{message.message_id}/{key}', Body=value)

        return QueueMessage(
            message_id=message.message_id,
            task_type=message.task_type,
            payload=payload,
            retry_count=message.retry_count,
        )
```

### 3. 定时清理任务

```python
class DLQMaintenanceScheduler:
    """
    DLQ定期维护调度器。
    """

    def __init__(self, dlq: ManagedDLQ, interval_hours: int = 6):
        self.dlq = dlq
        self.interval_hours = interval_hours
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """启动定期清理。"""
        self._running = True
        self._thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="dlq-maintenance",
        )
        self._thread.start()

    def _maintenance_loop(self):
        while self._running:
            time.sleep(self.interval_hours * 3600)

            # 清理过期消息
            try:
                cleaned = self.dlq.cleanup_expired()
                if cleaned > 0:
                    logger.info(f"DLQ清理: 移除了 {cleaned} 条过期消息")

                # 检查容量
                self.dlq.alert_if_needed()
            except Exception as e:
                logger.error(f"DLQ清理失败: {e}")
```

---

## 检查清单

- [ ] 设置了DLQ的最大消息数限制
- [ ] 设置了DLQ的最大总大小限制（字节）
- [ ] 有消息过期自动清理机制（按天数）
- [ ] 限制了单条消息的最大大小（发布前验证）
- [ ] 大文件使用对象存储URL替代base64嵌入
- [ ] 监控了DLQ的大小变化趋势
- [ ] 设置了DLQ使用率告警（如70%时告警）
- [ ] 有手动从DLQ重新入队的工具

---

**一句话总结**：DLQ是一个"安全网"，但不是"无底洞"。不给它设置边界，它就会成为你的存储灾难。定时清理 + 容量限制 + 大小监控 → 让安全网不变成陷阱。
