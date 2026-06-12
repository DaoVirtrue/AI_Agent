# 坑点 #4：消息格式不匹配（Message Format Mismatch）

> "Agent A发送了JSON格式的数据，Agent B期望XML格式。Agent C在中间尝试转换，但丢失了一个关键字段。最终结果：Agent D收到了一个空的字典，默默生成了完全无关的回答。"

---

## 症状

1. **静默失败**：消息被接收但解析失败，Agent不报错而是用默认值/空值继续
2. **字段丢失**：消息转换过程中某些字段被丢弃，导致下游判断错误
3. **类型混乱**：Agent A发送 `"count": 42`（整数），Agent B期望 `"count": "42"`（字符串）
4. **版本不兼容**：Agent A使用消息格式v2，Agent B只支持v1
5. **调试困难**：消息经过了3个Agent的传递和转换，难以确定是哪一步出了问题

---

## 根因

**缺乏标准化的消息Schema和强制验证机制。**

在快速迭代的多Agent开发中，每个Agent开发者独立定义消息格式，缺少统一的Schema注册和版本管理。当某个Agent的消息格式变更时，依赖它的Agent不会被自动通知——导致运行时的静默不兼容。

```
          Agent A (v2格式)                   Agent B (v1格式)
        {                             {
          "query": "hello",    →        "question": "",    ← 字段名不匹配！
          "context": {...},             "ctx": null,        ← 类型不匹配！
          "meta": {...}                }                    ← 新字段被丢弃！
        }
```

---

## 解法

### 1. 强制Schema验证（使用Pydantic）

```python
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Any, Optional
from enum import Enum

class AgentMessageType(str, Enum):
    QUERY = "query"
    ANSWER = "answer"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    ERROR = "error"

class AgentMessageV1(BaseModel):
    """
    V1消息格式：所有Agent必须遵守的标准化协议。
    使用Pydantic进行自动验证，任何不符合此格式的消息都会被拒绝。
    """
    message_id: str = Field(..., min_length=1, description="唯一消息ID")
    sender_id: str = Field(..., min_length=1)
    receiver_id: str = Field(default="")
    message_type: AgentMessageType
    content: Any = Field(...)
    timestamp: float = Field(..., ge=0)
    version: str = Field(default="1.0")

    @field_validator("sender_id", "receiver_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        if not v or not v.strip():
            return v
        # Agent ID必须是字母数字 + 下划线 + 连字符
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(f"Invalid agent_id format: {v}")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        # 时间戳不能在将来超1分钟，也不能在过去超24小时
        now = __import__('time').time()
        if v > now + 60:
            raise ValueError(f"Timestamp in the far future: {v}")
        if v < now - 86400:
            raise ValueError(f"Timestamp too old: {v}")
        return v


class SafeCommunicationBus:
    """
    带强制Schema验证的通信总线。
    所有消息在发布前和接收后都必须通过Schema验证。
    """

    def validate_message(self, raw_data: dict) -> AgentMessageV1:
        """验证并解析消息。验证失败时抛出ValidationError。"""
        try:
            return AgentMessageV1(**raw_data)
        except ValidationError as e:
            # 详细记录验证失败的信息
            logger.error(f"Message validation failed: {e.errors()}")
            logger.error(f"Raw data: {raw_data}")
            raise

    def publish(self, raw_message: dict) -> None:
        # 发送端验证
        try:
            validated = self.validate_message(raw_message)
        except ValidationError:
            # 发送失败，不传播无效消息
            return

        # 转换为传输格式（保留所有字段）
        transport_format = validated.model_dump()

        # 投递给订阅者
        for subscriber in self.subscribers:
            self.deliver(transport_format, subscriber)

    def deliver(self, raw_data: dict, subscriber) -> None:
        # 接收端再次验证（双保险）
        try:
            validated = self.validate_message(raw_data)
        except ValidationError as e:
            # 通知订阅者消息格式错误
            subscriber.on_error(e)
            return

        subscriber.on_message(validated)
```

### 2. 消息版本协商（Version Negotiation）

```python
class VersionNegotiator:
    """
    消息版本协商器。
    发送方和接收方自动协商共同支持的最高版本。
    """

    SUPPORTED_VERSIONS = ["1.0", "1.1", "2.0"]

    @classmethod
    def negotiate(cls, sender_versions: list[str], receiver_versions: list[str]) -> Optional[str]:
        """
        协商双方都能接受的最高版本。

        Returns:
            协商后的版本号，如果无法达成一致则返回None
        """
        common = set(sender_versions) & set(receiver_versions)
        if not common:
            return None

        # 按版本号排序，选择最高的
        sorted_versions = sorted(
            common,
            key=lambda v: [int(x) for x in v.split('.')],
            reverse=True,
        )
        return sorted_versions[0]

    @classmethod
    def convert_message(cls, message: dict, from_version: str, to_version: str) -> dict:
        """将消息从源版本转换为目标版本。"""
        # V1 → V2: 添加新字段，使用默认值
        if from_version == "1.0" and to_version == "2.0":
            message = dict(message)
            message.setdefault("metadata", {})
            message.setdefault("ttl", 0)
            message.setdefault("priority", "normal")
            return message

        # V2 → V1: 移除V2独有的字段
        if from_version == "2.0" and to_version == "1.0":
            message = dict(message)
            message.pop("metadata", None)
            message.pop("ttl", None)
            message.pop("priority", None)
            return message

        # 同版本，直接返回
        return message
```

### 3. 集成测试：消息格式兼容性

```python
import pytest

class TestMessageCompatibility:
    """验证所有Agent之间的消息格式兼容性。"""

    @pytest.mark.parametrize("sender_agent, receiver_agent", [
        ("research_agent", "writing_agent"),
        ("writing_agent", "review_agent"),
        ("review_agent", "polish_agent"),
    ])
    def test_message_roundtrip(self, sender_agent, receiver_agent):
        """验证任意两个Agent之间的消息往返。"""
        sender = AgentRegistry.get(sender_agent)
        receiver = AgentRegistry.get(receiver_agent)

        # 创建消息
        msg = sender.create_message(content="test content")
        raw = msg.to_dict()

        # 验证
        assert "message_id" in raw
        assert "sender_id" in raw
        assert "content" in raw
        assert raw["sender_id"] == sender_agent

        # 接收方解析
        parsed = receiver.parse_message(raw)
        assert parsed.content == "test content"
        assert parsed.sender_id == sender_agent

    def test_all_agents_use_same_schema_version(self):
        """验证所有Agent使用相同的消息Schema版本。"""
        versions = set()
        for agent in AgentRegistry.all():
            versions.add(agent.supported_message_version)

        assert len(versions) == 1, (
            f"Agent版本不一致！发现的版本: {versions}"
        )
```

---

## 检查清单

- [ ] 使用Pydantic（或类似工具）对消息进行运行时强制验证
- [ ] 定义了消息Schema的版本号，并在所有Agent中一致
- [ ] 实现了消息格式不匹配时的降级策略（而非静默失败）
- [ ] 有集成测试验证所有Agent之间的消息兼容性
- [ ] 记录了消息Schema的变更历史和兼容性矩阵
- [ ] 发送端和接收端都进行消息验证（双重保险）
- [ ] 消息格式错误时触发告警（而非仅记录日志）

---

**一句话总结**：在多Agent系统中，消息格式就是"通用语言"。没有统一的Schema和严格验证，你的Agent们就像说着八种不同语言的人被要求协作——沟通失败是必然的。
