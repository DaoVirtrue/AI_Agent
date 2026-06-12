#!/usr/bin/env python3
"""
Phase 08 - LangGraph 企业级：09 - 消息队列集成（RabbitMQ）

RabbitMQ 发布者（Publisher）用于异步任务提交。
消费者（Consumer）带 ack/nack 确认。
指数退避重试（3 次）+ 死信队列（DLQ）。

架构图：
  ┌──────────┐     ┌──────────────┐     ┌──────────────┐
  │  API     │────→│  RabbitMQ    │────→│  Worker      │
  │  Server  │     │              │     │  (Consumer)  │
  │          │     │  ┌────────┐  │     │              │
  │ POST /   │     │  │ Queue  │  │     │ graph.invoke │
  │ analyze  │     │  │        │  │     │ (state)      │
  │          │     │  └────────┘  │     │              │
  │ 202      │     │              │     │  失败 → 重试  │
  │ Accepted │     │  ┌────────┐  │     │  重试耗尽→DLQ│
  └──────────┘     │  │  DLQ   │←─┼─────┤              │
                   │  └────────┘  │     └──────────────┘
                   └──────────────┘

模式：
  API 返回 202 Accepted → MQ 异步处理 → 结果存储 → 轮询/回调获取
"""

import os
import json
import time
import uuid
import threading
import functools
from typing import TypedDict, Annotated, Callable, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================================
# 消息数据模型
# ============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class QueueMessage:
    """
    消息队列中的消息。

    Attributes:
        message_id: 唯一消息ID
        task_type: 任务类型
        payload: 任务载荷
        retry_count: 当前重试次数
        max_retries: 最大重试次数
        status: 任务状态
        created_at: 创建时间
        last_error: 最近一次错误
    """
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_type: str = "rag_query"
    payload: dict = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    last_error: str = ""

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_error": self.last_error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "QueueMessage":
        data = json.loads(json_str)
        return cls(
            message_id=data.get("message_id", ""),
            task_type=data.get("task_type", "rag_query"),
            payload=data.get("payload", {}),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            status=TaskStatus(data.get("status", "pending")),
            created_at=data.get("created_at", time.time()),
            last_error=data.get("last_error", ""),
        )


# ============================================================================
# 模拟 RabbitMQ（无需安装 pika）
# ============================================================================

class MockRabbitMQ:
    """
    模拟 RabbitMQ（用于演示和测试）。

    功能与真实 RabbitMQ 一致：
    - 发布（publish）
    - 消费（consume，带 ack/nack）
    - 死信队列（DLQ）
    - 重试逻辑
    """

    def __init__(self):
        self._queue: deque[QueueMessage] = deque()
        self._dlq: deque[QueueMessage] = deque()  # 死信队列
        self._results: dict[str, QueueMessage] = {}  # message_id → result
        self._lock = threading.Lock()
        self._processing: set[str] = set()

    # ---- 发布 ----

    def publish(self, message: QueueMessage) -> str:
        """
        发布消息到队列。

        Args:
            message: 要发布的消息

        Returns:
            消息ID
        """
        with self._lock:
            message.status = TaskStatus.PENDING
            self._queue.append(message)
            print(f"  📤 发布消息: {message.message_id[:8]}... "
                  f"(类型: {message.task_type})")
        return message.message_id

    def publish_batch(self, messages: list[QueueMessage]) -> list[str]:
        """批量发布消息。"""
        return [self.publish(msg) for msg in messages]

    # ---- 消费 ----

    def consume(self) -> Optional[QueueMessage]:
        """获取下一个待处理的消息。"""
        with self._lock:
            if not self._queue:
                return None

            msg = self._queue.popleft()
            msg.status = TaskStatus.PROCESSING
            self._processing.add(msg.message_id)
            return msg

    def ack(self, message_id: str) -> bool:
        """
        确认消息处理成功。

        Args:
            message_id: 消息ID

        Returns:
            是否成功确认
        """
        with self._lock:
            self._processing.discard(message_id)
            return True

    def nack(self, message: QueueMessage, error: str = "") -> bool:
        """
        否定确认：消息处理失败，需要决定重试还是移至DLQ。

        Args:
            message: 失败的消息
            error: 错误信息

        Returns:
            True 如果消息被重新入队或移至DLQ
        """
        with self._lock:
            self._processing.discard(message.message_id)
            message.last_error = error
            message.retry_count += 1

            if message.retry_count < message.max_retries:
                # 重试：重新入队
                message.status = TaskStatus.RETRYING
                self._queue.append(message)
                print(f"  🔄 消息 {message.message_id[:8]} 重试 "
                      f"({message.retry_count}/{message.max_retries}): {error}")
                return True
            else:
                # 超过重试次数：移至死信队列
                message.status = TaskStatus.DEAD_LETTER
                self._dlq.append(message)
                print(f"  💀 消息 {message.message_id[:8]} 已移至 DLQ "
                      f"(重试耗尽): {error}")
                return True

    # ---- 查询 ----

    def get_queue_length(self) -> int:
        """获取队列长度。"""
        with self._lock:
            return len(self._queue)

    def get_dlq_length(self) -> int:
        """获取死信队列长度。"""
        with self._lock:
            return len(self._dlq)

    def get_processing_count(self) -> int:
        """获取正在处理的消息数。"""
        with self._lock:
            return len(self._processing)

    def get_stats(self) -> dict:
        """获取队列统计。"""
        with self._lock:
            return {
                "queue_length": len(self._queue),
                "dlq_length": len(self._dlq),
                "processing": len(self._processing),
                "total_dlq_messages": [
                    {
                        "id": m.message_id[:8],
                        "task_type": m.task_type,
                        "retries": m.retry_count,
                        "error": m.last_error[:80],
                    }
                    for m in self._dlq
                ],
            }

    def requeue_from_dlq(self, message_id: str) -> bool:
        """
        从死信队列中重新入队一条消息。

        Args:
            message_id: 消息ID

        Returns:
            是否成功重新入队
        """
        with self._lock:
            for i, msg in enumerate(self._dlq):
                if msg.message_id == message_id:
                    msg.retry_count = 0  # 重置重试计数
                    msg.status = TaskStatus.PENDING
                    msg.last_error = ""
                    self._queue.append(msg)
                    del self._dlq[i]
                    return True
            return False


# ============================================================================
# Consumer（消费者）
# ============================================================================

class QueueConsumer:
    """
    消息队列消费者：从队列获取消息，执行 Graph 任务。

    功能：
    1. 拉取消息（consume）
    2. 执行 Graph 任务
    3. ack 成功 / nack 失败（带重试）
    4. 指数退避：1s → 2s → 4s
    """

    def __init__(
        self,
        queue: MockRabbitMQ,
        graph,
        consumer_id: str = None,
        max_retries: int = 3,
        base_backoff_s: float = 1.0,
    ):
        self.queue = queue
        self.graph = graph
        self.consumer_id = consumer_id or f"consumer_{uuid.uuid4().hex[:6]}"
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.processed_count = 0
        self.failed_count = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, poll_interval_s: float = 0.5):
        """启动消费者轮询线程。"""
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(poll_interval_s,),
            daemon=True,
            name=f"consumer-{self.consumer_id}",
        )
        self._thread.start()
        print(f"  ▶️ 消费者 {self.consumer_id} 已启动")

    def stop(self):
        """停止消费者。"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        print(f"  ⏹️ 消费者 {self.consumer_id} 已停止")

    def _poll_loop(self, poll_interval_s: float):
        """轮询循环：持续从队列获取消息。"""
        while self._running:
            msg = self.queue.consume()
            if msg:
                print(f"\n  📥 [{self.consumer_id}] 收到消息: {msg.message_id[:8]}...")
                self._process_message(msg)
            else:
                time.sleep(poll_interval_s)

    def _process_message(self, message: QueueMessage):
        """
        处理单条消息：执行Graph → ack 或 nack。

        Args:
            message: 要处理的消息
        """
        state = message.payload.get("state", {})
        config = message.payload.get("config", {})

        try:
            # 执行 Graph
            result = self.graph.invoke(state, config)

            # 成功 → ack
            self.queue.ack(message.message_id)
            self.processed_count += 1
            print(f"  ✅ [{self.consumer_id}] 处理成功: {message.message_id[:8]}")

            # 存储结果
            message.payload["result"] = result
            message.status = TaskStatus.COMPLETED

        except Exception as e:
            # 失败 → 计算退避时间
            backoff = self.base_backoff_s * (2 ** message.retry_count)

            print(f"  ❌ [{self.consumer_id}] 处理失败 (重试#{message.retry_count + 1}): {e}")
            print(f"     退避: {backoff:.1f}s 后重试")

            # 退避等待
            time.sleep(backoff)

            # nack（可能触发重试或移至DLQ）
            self.queue.nack(message, str(e))
            self.failed_count += 1

    def process_single_sync(self, message: QueueMessage) -> dict:
        """
        同步处理单条消息（非线程模式）。

        Args:
            message: 要处理的消息

        Returns:
            处理结果或错误信息
        """
        state = message.payload.get("state", {})
        config = message.payload.get("config", {})

        for attempt in range(self.max_retries + 1):
            try:
                result = self.graph.invoke(state, config)
                self.queue.ack(message.message_id)
                self.processed_count += 1
                return {"status": "success", "message_id": message.message_id, "result": result}
            except Exception as e:
                if attempt < self.max_retries:
                    backoff = self.base_backoff_s * (2 ** attempt)
                    print(f"  🔄 重试 {attempt + 1}/{self.max_retries} "
                          f"(退避{backoff:.1f}s): {e}")
                    time.sleep(backoff)
                else:
                    self.queue.nack(message, str(e))
                    self.failed_count += 1
                    return {"status": "failed", "message_id": message.message_id, "error": str(e)}

        return {"status": "unknown", "message_id": message.message_id}

    def get_stats(self) -> dict:
        return {
            "consumer_id": self.consumer_id,
            "processed": self.processed_count,
            "failed": self.failed_count,
            "success_rate": round(
                self.processed_count / max(1, self.processed_count + self.failed_count), 3
            ),
        }


# ============================================================================
# Publisher（发布者 / API 端）
# ============================================================================

class TaskPublisher:
    """
    任务发布者：API 端接收用户请求，发布到消息队列。

    使用模式：
    POST /api/analyze → 返回 202 Accepted
    后台 Consumer 异步处理
    客户端轮询 GET /api/status/{message_id} 获取结果
    """

    def __init__(self, queue: MockRabbitMQ):
        self.queue = queue

    def submit_task(
        self,
        task_type: str,
        state: dict,
        config: dict = None,
        max_retries: int = 3,
    ) -> dict:
        """
        提交异步任务。

        Args:
            task_type: 任务类型
            state: Graph 的初始状态
            config: Graph 配置（如 thread_id）
            max_retries: 最大重试次数

        Returns:
            包含 message_id 和状态查询URL的响应
        """
        message = QueueMessage(
            task_type=task_type,
            payload={
                "state": state,
                "config": config or {},
            },
            max_retries=max_retries,
        )

        message_id = self.queue.publish(message)

        return {
            "status": "accepted",
            "message_id": message_id,
            "message": f"任务已提交 ({task_type})",
            "status_url": f"/api/tasks/{message_id}/status",
            "retry_policy": {
                "max_retries": max_retries,
                "strategy": "exponential_backoff",
            },
        }

    def submit_batch(
        self,
        tasks: list[tuple[str, dict]],
        max_retries: int = 3,
    ) -> list[dict]:
        """批量提交任务。"""
        results = []
        for task_type, state in tasks:
            result = self.submit_task(task_type, state, max_retries=max_retries)
            results.append(result)
        return results

    def get_task_status(self, message_id: str) -> dict:
        """查询任务状态。"""
        # 在真实系统中，这里会查询数据库
        # 在模拟系统中，检查队列和DLQ
        for msg in list(self.queue._queue) + list(self.queue._dlq):
            if msg.message_id == message_id:
                return {
                    "message_id": message_id,
                    "status": msg.status.value,
                    "retry_count": msg.retry_count,
                    "last_error": msg.last_error,
                    "created_at": msg.created_at,
                }

        return {
            "message_id": message_id,
            "status": "completed" if message_id in self.queue._results else "unknown",
        }


# ============================================================================
# 简单 Graph（用于演示）
# ============================================================================

class AnalysisState(TypedDict):
    query: str
    result: str
    error: str
    attempt_count: int


def node_analyze(state: AnalysisState) -> dict:
    """分析节点。"""
    query = state['query']
    attempt = state.get('attempt_count', 0) + 1

    print(f"    [analyze] 分析查询 (尝试#{attempt}): {query[:50]}...")

    # 模拟偶尔失败
    if attempt < 2 and "fail" in query.lower():
        raise RuntimeError(f"分析失败（模拟）: 查询 '{query[:30]}...' 不完整")

    return {
        "result": f"分析结果: {query[:50]}... = 通过",
        "attempt_count": attempt,
    }


def build_analysis_graph() -> StateGraph:
    """构建简单的分析图。"""
    builder = StateGraph(AnalysisState)
    builder.add_node("analyze", node_analyze)
    builder.set_entry_point("analyze")
    builder.add_edge("analyze", END)
    return builder.compile()


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  LangGraph 企业级 - 消息队列集成 (RabbitMQ) 演示")
    print("=" * 72)

    # 创建队列和Graph
    queue = MockRabbitMQ()
    graph = build_analysis_graph()

    publisher = TaskPublisher(queue)
    consumer = QueueConsumer(queue, graph, consumer_id="worker-01")

    # ---- Demo 1: 基本发布-消费流程 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 异步任务提交 → 消费 → 完成")
    print("=" * 72)

    # API 端：提交任务
    response1 = publisher.submit_task(
        task_type="analysis",
        state={"query": "分析2025年AI市场趋势", "result": "", "error": "", "attempt_count": 0},
    )
    print(f"\n  API 响应 (202 Accepted):")
    print(f"    {json.dumps(response1, ensure_ascii=False, indent=2)}")

    # Worker 端：同步处理
    msg = queue.consume()
    if msg:
        result = consumer.process_single_sync(msg)
        print(f"\n  处理结果: {result.get('status')}")

    # ---- Demo 2: 失败重试 + 指数退避 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 失败 → 指数退避重试 → DLQ")
    print("=" * 72)

    response2 = publisher.submit_task(
        task_type="analysis",
        state={"query": "这个查询会fail因为格式错误", "result": "", "error": "", "attempt_count": 0},
        max_retries=3,
    )
    print(f"\n  提交任务: {response2['message_id'][:8]}...")

    # 模拟 Consumer 处理（会失败并重试）
    msg = queue.consume()
    if msg:
        result = consumer.process_single_sync(msg)
        print(f"\n  最终结果: {result.get('status')}")
        if result["status"] == "failed":
            print(f"  错误: {result.get('error', '')[:100]}")

    # ---- Demo 3: 死信队列 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 死信队列 (DLQ) 管理")
    print("=" * 72)

    dlq_stats = queue.get_stats()
    print(f"\n  📊 队列状态:")
    print(f"    主队列: {dlq_stats['queue_length']}")
    print(f"    DLQ: {dlq_stats['dlq_length']} (死信)")
    print(f"    处理中: {dlq_stats['processing']}")

    if dlq_stats['total_dlq_messages']:
        print(f"\n  💀 DLQ 中的消息:")
        for dm in dlq_stats['total_dlq_messages']:
            print(f"    ID={dm['id']} 类型={dm['task_type']} "
                  f"重试={dm['retries']} 错误={dm['error']}")

        # 重新入队一条DLQ消息
        dlq_msg = queue._dlq[0] if queue._dlq else None
        if dlq_msg:
            requeued = queue.requeue_from_dlq(dlq_msg.message_id)
            print(f"\n  🔄 重新入队: {dlq_msg.message_id[:8]}... "
                  f"{'✅ 成功' if requeued else '❌ 失败'}")

    # ---- Demo 4: 批量提交 ----

    print("\n\n" + "=" * 72)
    print("  Demo 4: 批量任务提交")
    print("=" * 72)

    batch_tasks = [
        ("analysis", {"query": "批量任务1", "result": "", "error": "", "attempt_count": 0}),
        ("analysis", {"query": "批量任务2", "result": "", "error": "", "attempt_count": 0}),
        ("analysis", {"query": "批量任务3", "result": "", "error": "", "attempt_count": 0}),
    ]

    batch_response = publisher.submit_batch(batch_tasks)
    print(f"\n  提交了 {len(batch_response)} 个任务:")
    for br in batch_response:
        print(f"    {br['message_id'][:8]}... [{br['status']}]")

    # 处理所有
    print(f"\n  处理所有任务...")
    while True:
        msg = queue.consume()
        if msg is None:
            break
        result = consumer.process_single_sync(msg)

    # ---- 统计 ----

    print(f"\n\n📊 最终统计:")
    print(f"  消费者: {json.dumps(consumer.get_stats(), ensure_ascii=False, indent=2)}")
    print(f"  队列: {json.dumps(queue.get_stats(), ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 72)
    print("  消息队列集成演示完成！")
    print("=" * 72)
