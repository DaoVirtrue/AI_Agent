#!/usr/bin/env python3
"""
Phase 07 - 多智能体系统：06 - Agent通信协议（Agent Communication Protocol）

标准化 AgentMessage 模式，发布-订阅通信总线。
MCP (Model Context Protocol) 和 A2A (Agent-to-Agent) 协议概述。

核心组件：
1. AgentMessage：标准化消息格式（sender_id, receiver_id, message_type,
   content, timestamp, correlation_id, metadata）
2. CommunicationBus：发布-订阅总线（publish, subscribe, unsubscribe）
3. MessageRouter：消息路由（direct, broadcast, topic-based）
4. ProtocolHandler：协议处理（序列化、验证、日志）

通信模式：
  ┌──────────┐   AgentMessage    ┌──────────────┐   AgentMessage    ┌──────────┐
  │ Agent A  │ ────────────────→ │ CommBus      │ ────────────────→ │ Agent B  │
  │(发布者)  │                   │ (消息总线)    │                   │(订阅者)  │
  └──────────┘                   └──────┬───────┘                   └──────────┘
                                       │
                                ┌──────▼───────┐
                                │ MessageLog   │
                                │ (持久化日志)  │
                                └──────────────┘
"""

import json
import time
import uuid
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue, Empty
from typing import Any, Callable, Optional, Union
from concurrent.futures import ThreadPoolExecutor


# ============================================================================
# 消息数据模型
# ============================================================================

class MessageType(Enum):
    """标准化消息类型。"""
    # 任务相关
    TASK_REQUEST = "task_request"          # 任务请求
    TASK_RESPONSE = "task_response"        # 任务响应
    TASK_PROGRESS = "task_progress"        # 进度更新
    TASK_ERROR = "task_error"              # 任务错误

    # 协作相关
    QUERY = "query"                         # 查询/问询
    ANSWER = "answer"                       # 回答
    DEBATE_STATEMENT = "debate_statement"  # 辩论发言
    VOTE = "vote"                          # 投票
    ESCALATION = "escalation"              # 升级请求
    APPROVAL_REQUIRED = "approval_required"  # 需要审批

    # 系统相关
    HEARTBEAT = "heartbeat"                # 心跳
    ACK = "ack"                            # 确认收到
    NACK = "nack"                          # 否定确认
    AGENT_REGISTER = "agent_register"      # Agent注册
    AGENT_UNREGISTER = "agent_unregister"  # Agent注销
    SYSTEM_ALERT = "system_alert"          # 系统告警

    # 数据相关
    DATA_UPDATE = "data_update"            # 数据更新
    STATE_CHANGE = "state_change"          # 状态变更
    BLACKBOARD_WRITE = "blackboard_write"  # 黑板写入


class MessagePriority(Enum):
    """消息优先级。"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AgentMessage:
    """
    标准化Agent消息格式。

    Attributes:
        message_id: 唯一消息ID（UUID）
        sender_id: 发送者Agent标识
        receiver_id: 接收者Agent标识（空字符串=广播）
        message_type: 消息类型
        content: 消息内容（任意JSON可序列化数据）
        timestamp: Unix时间戳
        correlation_id: 关联ID（用于追踪请求-响应链）
        priority: 消息优先级
        ttl: 消息生存时间（秒，0=永不过期）
        metadata: 附加元数据
    """

    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sender_id: str = ""
    receiver_id: str = ""                   # "" = 广播给所有订阅者
    message_type: MessageType = MessageType.QUERY
    content: Any = None
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = ""                # 用于将响应关联到原始请求
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: float = 0.0                        # 0 = 永不过期
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """检查消息是否已过期。"""
        if self.ttl <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl

    @property
    def is_broadcast(self) -> bool:
        """是否为广播消息。"""
        return self.receiver_id == ""

    @property
    def datetime_str(self) -> str:
        """人类可读的时间字符串。"""
        return datetime.fromtimestamp(self.timestamp).isoformat()

    def create_reply(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType = None,
    ) -> "AgentMessage":
        """
        创建对此消息的回复。

        Args:
            sender_id: 回复的发送者（通常是原消息的接收者）
            content: 回复内容
            message_type: 回复的消息类型（默认与请求对应）

        Returns:
            回复消息
        """
        # 自动选择回复类型
        if message_type is None:
            reply_type_map = {
                MessageType.QUERY: MessageType.ANSWER,
                MessageType.TASK_REQUEST: MessageType.TASK_RESPONSE,
                MessageType.HEARTBEAT: MessageType.ACK,
                MessageType.DEBATE_STATEMENT: MessageType.DEBATE_STATEMENT,
                MessageType.VOTE: MessageType.ACK,
                MessageType.ESCALATION: MessageType.ACK,
            }
            message_type = reply_type_map.get(self.message_type, MessageType.ACK)

        return AgentMessage(
            message_id=uuid.uuid4().hex,
            sender_id=sender_id,
            receiver_id=self.sender_id,
            message_type=message_type,
            content=content,
            correlation_id=self.message_id,  # 关联到原始消息
        )

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "priority": self.priority.value,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """序列化为JSON字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMessage":
        """从字典反序列化。"""
        return cls(
            message_id=data.get("message_id", ""),
            sender_id=data.get("sender_id", ""),
            receiver_id=data.get("receiver_id", ""),
            message_type=MessageType(data.get("message_type", "query")),
            content=data.get("content"),
            timestamp=data.get("timestamp", time.time()),
            correlation_id=data.get("correlation_id", ""),
            priority=MessagePriority(data.get("priority", 1)),
            ttl=data.get("ttl", 0.0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AgentMessage":
        """从JSON字符串反序列化。"""
        return cls.from_dict(json.loads(json_str))

    def validate(self) -> list[str]:
        """
        验证消息字段的有效性。
        返回问题列表（空列表表示有效）。
        """
        issues = []
        if not self.sender_id:
            issues.append("sender_id 不能为空")
        if not self.message_id:
            issues.append("message_id 不能为空")
        if self.priority not in MessagePriority:
            issues.append(f"无效的 priority: {self.priority}")
        if self.ttl < 0:
            issues.append(f"ttl 不能为负数: {self.ttl}")
        return issues

    def __repr__(self) -> str:
        return (
            f"<AgentMessage id={self.message_id[:8]} "
            f"from={self.sender_id} to={self.receiver_id or '[broadcast]'} "
            f"type={self.message_type.value}>"
        )


# ============================================================================
# 通信总线
# ============================================================================

class MessageSubscriber(ABC):
    """消息订阅者基类。"""

    @abstractmethod
    def on_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        处理收到的消息。

        Args:
            message: 收到的消息

        Returns:
            可选的回复消息
        """
        ...

    @abstractmethod
    def get_agent_id(self) -> str:
        """获取Agent标识。"""
        ...


class CommunicationBus:
    """
    发布-订阅通信总线。

    特性：
    - 主题（Topic）订阅：按 message_type 订阅
    - 定向消息：指定 receiver_id
    - 广播消息：receiver_id 为空
    - 请求-响应追踪（correlation_id）
    - 消息日志（可持久化）
    - 线程安全
    """

    def __init__(self, name: str = "DefaultBus"):
        self.name = name
        self._subscribers: dict[str, set[MessageSubscriber]] = defaultdict(set)
        # _subscribers[message_type.value] = {subscriber1, subscriber2, ...}

        self._message_log: list[AgentMessage] = []
        self._correlation_map: dict[str, AgentMessage] = {}
        self._lock = threading.RLock()
        self._message_counter: int = 0

        # 消息处理线程池
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="commbus")

    # ---- 发布 ----

    def publish(
        self,
        message: AgentMessage,
        persist: bool = True,
    ) -> int:
        """
        发布消息到总线。

        Args:
            message: 要发布的消息
            persist: 是否持久化到消息日志

        Returns:
            消息被投递到的订阅者数量

        Raises:
            ValueError: 如果消息验证失败
        """
        # 验证消息
        issues = message.validate()
        if issues:
            raise ValueError(f"消息验证失败: {issues}")

        with self._lock:
            self._message_counter += 1

            # 过滤过期消息
            if message.is_expired:
                return 0

            # 持久化
            if persist:
                self._message_log.append(message)

            # 追踪关联
            if message.correlation_id:
                self._correlation_map[message.correlation_id] = message

            delivery_count = 0

            # 定向消息
            if not message.is_broadcast:
                # 在所有订阅者中查找目标
                for subscribers in self._subscribers.values():
                    for sub in subscribers:
                        if sub.get_agent_id() == message.receiver_id:
                            self._deliver(sub, message)
                            delivery_count += 1
                            break
            else:
                # 广播：投递给所有匹配 message_type 的订阅者
                subscribers = self._subscribers.get(
                    message.message_type.value, set()
                )
                for sub in subscribers:
                    self._deliver(sub, message)
                    delivery_count += 1

            return delivery_count

    def _deliver(self, subscriber: MessageSubscriber, message: AgentMessage):
        """
        将消息投递给订阅者（异步）。

        Args:
            subscriber: 订阅者
            message: 消息
        """
        def _handle():
            try:
                reply = subscriber.on_message(message)
                if reply is not None:
                    # 订阅者返回了回复消息，自动发布
                    self.publish(reply, persist=True)
            except Exception as e:
                # 错误消息
                error_msg = AgentMessage(
                    sender_id="system",
                    receiver_id=message.sender_id,
                    message_type=MessageType.TASK_ERROR,
                    content={
                        "error": str(e),
                        "original_message_id": message.message_id,
                    },
                    correlation_id=message.message_id,
                    priority=MessagePriority.HIGH,
                )
                self.publish(error_msg, persist=True)

        self._executor.submit(_handle)

    # ---- 请求-响应 ----

    def request(
        self,
        receiver_id: str,
        content: Any,
        message_type: MessageType = MessageType.QUERY,
        timeout: float = 30.0,
    ) -> Optional[AgentMessage]:
        """
        发送请求并等待响应（同步）。

        Args:
            receiver_id: 接收者ID
            content: 请求内容
            message_type: 消息类型
            timeout: 超时时间（秒）

        Returns:
            响应消息，超时返回None
        """
        msg = AgentMessage(
            sender_id="requester",
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
        )

        # 发布请求
        self.publish(msg)

        # 等待响应
        start = time.time()
        while (time.time() - start) < timeout:
            with self._lock:
                if msg.message_id in self._correlation_map:
                    return self._correlation_map[msg.message_id]
            time.sleep(0.05)

        return None  # 超时

    # ---- 订阅 ----

    def subscribe(
        self,
        subscriber: MessageSubscriber,
        message_types: list[MessageType] = None,
    ) -> bool:
        """
        订阅特定类型的消息。

        Args:
            subscriber: 订阅者对象
            message_types: 要订阅的消息类型列表（None=订阅所有类型）

        Returns:
            是否成功
        """
        if message_types is None:
            message_types = list(MessageType)

        with self._lock:
            for mt in message_types:
                self._subscribers[mt.value].add(subscriber)

        return True

    def unsubscribe(
        self,
        subscriber: MessageSubscriber,
        message_types: list[MessageType] = None,
    ) -> bool:
        """取消订阅。"""
        if message_types is None:
            message_types = list(MessageType)

        with self._lock:
            for mt in message_types:
                self._subscribers[mt.value].discard(subscriber)

        return True

    # ---- 查询 ----

    def get_subscribers(self, message_type: MessageType = None) -> list[str]:
        """获取订阅者列表。"""
        with self._lock:
            if message_type:
                return [s.get_agent_id() for s in self._subscribers.get(message_type.value, set())]
            else:
                all_subs = set()
                for subs in self._subscribers.values():
                    all_subs.update(subs)
                return [s.get_agent_id() for s in all_subs]

    def get_message_log(
        self,
        sender_id: str = None,
        message_type: MessageType = None,
        limit: int = 100,
    ) -> list[AgentMessage]:
        """
        查询消息日志。

        Args:
            sender_id: 按发送者过滤（可选）
            message_type: 按类型过滤（可选）
            limit: 返回条数限制

        Returns:
            符合条件的消息列表
        """
        with self._lock:
            messages = self._message_log

            if sender_id:
                messages = [m for m in messages if m.sender_id == sender_id]
            if message_type:
                messages = [m for m in messages if m.message_type == message_type]

            return messages[-limit:]

    def get_statistics(self) -> dict:
        """获取通信总线统计信息。"""
        with self._lock:
            type_counts = defaultdict(int)
            for msg in self._message_log:
                type_counts[msg.message_type.value] += 1

            return {
                "bus_name": self.name,
                "total_messages": self._message_counter,
                "logged_messages": len(self._message_log),
                "active_subscribers": sum(
                    len(subs) for subs in self._subscribers.values()
                ),
                "message_types_distribution": dict(type_counts),
                "pending_correlations": len(self._correlation_map),
            }

    def shutdown(self):
        """关闭通信总线。"""
        self._executor.shutdown(wait=True)


# ============================================================================
# 协议概述：MCP 和 A2A
# ============================================================================

class ProtocolOverview:
    """
    MCP (Model Context Protocol) 和 A2A (Agent-to-Agent) 协议概述。

    这些是行业标准协议，本文件中的 CommunicationBus 实现了类似概念的精简版。
    """

    MCP_OVERVIEW = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         MCP (Model Context Protocol) 概述                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  发起方: Anthropic                                            ║
    ║  定位: LLM/Agent 与外部工具/数据源之间的标准化通信协议          ║
    ║                                                              ║
    ║  核心概念：                                                    ║
    ║  ┌──────────────────────────────────────────────────────┐     ║
    ║  │ MCP Client (Host) ←→ MCP Server                      │     ║
    ║  │   (Claude/GPT)         (Tool/Data Server)             │     ║
    ║  │                                                        │     ║
    ║  │ 传输层: stdio / HTTP+SSE / WebSocket                   │     ║
    ║  │ 消息格式: JSON-RPC 2.0                                │     ║
    ║  │                                                        │     ║
    ║  │ 核心能力:                                              │     ║
    ║  │  - tools/list: 获取可用工具列表                        │     ║
    ║  │  - tools/call: 调用工具                               │     ║
    ║  │  - resources/read: 读取资源                           │     ║
    ║  │  - prompts/get: 获取提示词模板                         │     ║
    ║  └──────────────────────────────────────────────────────┘     ║
    ║                                                              ║
    ║  与本课程对应关系：                                            ║
    ║  本文件的 AgentMessage 结构 ≈ MCP 的 JSON-RPC 消息格式        ║
    ║  CommunicationBus 的 publish/subscribe ≈ MCP 的传输层抽象     ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    A2A_OVERVIEW = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         A2A (Agent-to-Agent) 协议概述                        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  发起方: Google                                               ║
    ║  定位: 多Agent之间的标准化通信协议（Agent联邦）                 ║
    ║                                                              ║
    ║  核心概念：                                                    ║
    ║  ┌──────────────────────────────────────────────────────┐     ║
    ║  │ Agent Card: Agent的"名片"                             │     ║
    ║  │  - name, description, url                            │     ║
    ║  │  - capabilities (skills)                             │     ║
    ║  │  - defaultInputModes, defaultOutputModes              │     ║
    ║  │                                                        │     ║
    ║  │ Task: 工作单元                                         │     ║
    ║  │  - id, sessionId, status                              │     ║
    ║  │  - artifacts (产出的工件)                               │     ║
    ║  │  - history (消息历史)                                  │     ║
    ║  │                                                        │     ║
    ║  │ 通信模式:                                              │     ║
    ║  │  - HTTP + JSON (client → remote agent)                │     ║
    ║  │  - Server-Sent Events (remote agent → client)         │     ║
    ║  │  - Webhook (push notifications)                       │     ║
    ║  └──────────────────────────────────────────────────────┘     ║
    ║                                                              ║
    ║  与本课程对应关系：                                            ║
    ║  Phase 07 的多Agent协作模式 ≈ A2A协议的应用场景               ║
    ║  AgentMessage.correlation_id ≈ A2A的Task.sessionId           ║
    ║  后续Phase 08的LangGraph ≈ 更工程化的Agent编排实现            ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    @staticmethod
    def print_overview():
        """打印协议概述。"""
        print(ProtocolOverview.MCP_OVERVIEW)
        print(ProtocolOverview.A2A_OVERVIEW)


# ============================================================================
# 简单Agent实现（用于演示）
# ============================================================================

class DemoAgent(MessageSubscriber):
    """
    演示用Agent：实现MessageSubscriber接口。
    在生产环境中，Agent会包含LLM调用和业务逻辑。
    """

    def __init__(self, agent_id: str, bus: CommunicationBus):
        self.agent_id = agent_id
        self.bus = bus
        self.received_messages: list[AgentMessage] = []
        self._active = True

    def get_agent_id(self) -> str:
        return self.agent_id

    def on_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理收到的消息。"""
        self.received_messages.append(message)

        # 对不同类型的消息做出不同响应
        if message.message_type == MessageType.QUERY:
            return message.create_reply(
                sender_id=self.agent_id,
                content=f"[{self.agent_id}] 已收到查询: {str(message.content)[:100]}",
                message_type=MessageType.ANSWER,
            )
        elif message.message_type == MessageType.HEARTBEAT:
            return message.create_reply(
                sender_id=self.agent_id,
                content="alive",
                message_type=MessageType.ACK,
            )
        elif message.message_type == MessageType.TASK_REQUEST:
            # 模拟任务处理
            return message.create_reply(
                sender_id=self.agent_id,
                content={
                    "status": "completed",
                    "result": f"[{self.agent_id}] 任务已处理",
                    "task_metadata": {"processing_time_ms": 150},
                },
                message_type=MessageType.TASK_RESPONSE,
            )
        return None

    def send_message(
        self,
        receiver_id: str,
        content: Any,
        message_type: MessageType = MessageType.QUERY,
    ) -> AgentMessage:
        """
        发送消息到总线。

        Args:
            receiver_id: 接收者ID（空=广播）
            content: 消息内容
            message_type: 消息类型

        Returns:
            发送的消息对象
        """
        msg = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            metadata={"sent_by": self.agent_id},
        )
        self.bus.publish(msg)
        return msg


# ============================================================================
# 消息序列化工具
# ============================================================================

class MessageSerializer:
    """
    消息序列化/反序列化工具。
    支持JSON和二进制格式。
    """

    @staticmethod
    def to_json(message: AgentMessage, indent: int = None) -> str:
        """序列化为JSON字符串。"""
        return json.dumps(
            message.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )

    @staticmethod
    def from_json(json_str: str) -> AgentMessage:
        """从JSON字符串反序列化。"""
        data = json.loads(json_str)
        return AgentMessage.from_dict(data)

    @staticmethod
    def to_bytes(message: AgentMessage) -> bytes:
        """序列化为字节（用于网络传输）。"""
        return MessageSerializer.to_json(message).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> AgentMessage:
        """从字节反序列化。"""
        return MessageSerializer.from_json(data.decode("utf-8"))

    @staticmethod
    def batch_to_json(messages: list[AgentMessage], indent: int = None) -> str:
        """批量序列化。"""
        return json.dumps(
            [m.to_dict() for m in messages],
            ensure_ascii=False,
            indent=indent,
            default=str,
        )


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  多智能体系统 - Agent通信协议演示")
    print("=" * 72)

    # ---- Demo 1: 发布-订阅 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 发布-订阅通信")
    print("=" * 72)

    bus = CommunicationBus(name="MainBus")

    # 创建Agent
    agent_a = DemoAgent("research_agent", bus)
    agent_b = DemoAgent("writing_agent", bus)
    agent_c = DemoAgent("review_agent", bus)

    # 订阅消息类型
    bus.subscribe(agent_a, [MessageType.QUERY, MessageType.TASK_REQUEST])
    bus.subscribe(agent_b, [MessageType.QUERY])
    bus.subscribe(agent_c, [MessageType.QUERY, MessageType.TASK_REQUEST])

    print(f"  订阅者: {bus.get_subscribers()}")

    # A发送定向消息给B
    print("\n  [A → B] 定向查询")
    agent_a.send_message(
        receiver_id="writing_agent",
        content="请根据研究报告撰写初稿",
        message_type=MessageType.QUERY,
    )

    # A广播任务请求
    print("  [A → ALL] 广播任务请求")
    agent_a.send_message(
        receiver_id="",  # 广播
        content="所有Agent注意：新项目已启动，请在1小时内完成初始任务",
        message_type=MessageType.TASK_REQUEST,
    )

    time.sleep(0.2)  # 等待异步处理

    # 查看各Agent收到的消息
    print(f"\n  📬 消息收取统计:")
    for agent in [agent_a, agent_b, agent_c]:
        print(f"    {agent.agent_id}: 收到 {len(agent.received_messages)} 条消息")
        for msg in agent.received_messages:
            print(f"      ← {msg.sender_id}: [{msg.message_type.value}] "
                  f"{str(msg.content)[:60]}")

    # ---- Demo 2: 请求-响应 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 请求-响应模式")
    print("=" * 72)

    # 创建Agent D（服务提供者）
    agent_d = DemoAgent("data_service", bus)
    bus.subscribe(agent_d, [MessageType.QUERY])

    # 发送请求并等待响应
    print("  发送请求到 data_service...")
    response = bus.request(
        receiver_id="data_service",
        content={"query": "select * from users where active=true"},
        message_type=MessageType.QUERY,
        timeout=2.0,
    )

    if response:
        print(f"  ✅ 收到响应: {response.content}")
        print(f"     correlation_id: {response.correlation_id[:12]}...")
    else:
        print(f"  ❌ 请求超时")

    # ---- Demo 3: 消息序列化 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 消息序列化/反序列化")
    print("=" * 72)

    original_msg = AgentMessage(
        sender_id="test_agent",
        receiver_id="target_agent",
        message_type=MessageType.TASK_REQUEST,
        content={"task": "analyze_data", "params": {"source": "db1"}},
        priority=MessagePriority.HIGH,
        metadata={"trace_id": "abc-123"},
    )

    # JSON序列化
    json_str = MessageSerializer.to_json(original_msg, indent=2)
    print(f"  📝 序列化JSON ({len(json_str)} bytes):")
    print(json_str[:300])

    # JSON反序列化
    restored_msg = MessageSerializer.from_json(json_str)
    print(f"\n  🔄 反序列化验证:")
    print(f"    sender_id 匹配: {original_msg.sender_id == restored_msg.sender_id}")
    print(f"    message_type 匹配: {original_msg.message_type == restored_msg.message_type}")
    print(f"    content 匹配: {original_msg.content == restored_msg.content}")
    print(f"    priority 匹配: {original_msg.priority == restored_msg.priority}")

    # 字节序列化
    bytes_data = MessageSerializer.to_bytes(original_msg)
    restored_from_bytes = MessageSerializer.from_bytes(bytes_data)
    print(f"    字节序列化往返: {original_msg.content == restored_from_bytes.content}")

    # ---- Demo 4: 协议概述 ----

    print("\n\n" + "=" * 72)
    print("  Demo 4: MCP 和 A2A 协议概述")
    print("=" * 72)

    ProtocolOverview.print_overview()

    # ---- 统计 ----

    print("\n\n" + "=" * 72)
    print("  📊 通信总线统计")
    print("=" * 72)

    stats = bus.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    bus.shutdown()

    print("\n" + "=" * 72)
    print("  通信协议演示完成！")
    print("=" * 72)
