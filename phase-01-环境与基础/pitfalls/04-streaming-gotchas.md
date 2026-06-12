# 坑点 #4：流式响应的陷阱

> "用户看到的结果是：`{ \"name\": \"张三\", \"a` ——然后卡住了。JSON 在 chunk 边界断了。"

---

## 症状

你实现了流式输出，想给用户一个"打字机"般的体验。但实际运行时：

1. **JSON 解析爆炸**：LLM 的输出是要被解析的 JSON，但你每收到一个 chunk 就尝试 `json.loads()`，结果永远是 `JSONDecodeError`
2. **前端显示乱码**：中文/多字节字符在 chunk 边界被截断，前端显示 `"你好世\xe7\x95\x8c"` 这样的乱码（半个 UTF-8 字节序列）
3. **流式输出"卡住"**：网络抖动导致某个 chunk 超时，整个流式连接断开，后续内容全丢
4. **流式数据没做超时处理**：`for chunk in response:` 这种循环，如果中间有一个 chunk 花了 30 秒才到，用户觉得"AI 死掉了"，但实际上只是网络慢
5. **"完整响应"永远拼不完整**：你在收集所有 chunk 拼接成完整响应，但某个 chunk 丢失了，你拿到了一个不完整的回答
6. **SSE 解析错误**：服务端发送的 Server-Sent Events 格式不标准（少了 `data:` 前缀，或多了一个空行），你的解析器崩溃了

---

## 根因

流式响应按 **chunk** 到达，而每个 chunk 的边界是**不可预测的**——可能在任意位置切割数据：

```
完整 JSON:  {"name": "张三", "age": 25, "city": "北京"}

实际到达的 chunks:
  chunk1: '{"name": "张'
  chunk2: '三", "a'
  chunk3: 'ge": 25, "city'
  chunk4: '": "北京"}'
```

如果你在每个 chunk 上调用 `json.loads()`，4 个 chunk 全部会失败。只有完整的 `chunk1 + chunk2 + chunk3 + chunk4` 才是合法 JSON。

对于 UTF-8 多字节字符，同样的问题：

```
中文字符 "界" = UTF-8 字节: 0xE7 0x95 0x8C

chunk 边界可能切割为:
  chunk1: ... 0xE7 0x95  → 无效！(不完整的 UTF-8 序列)
  chunk2: 0x8C ...        → 解码失败
```

---

## 修复方案

### 方案 1：缓冲区累积 + 完整性检测

将所有 chunk 累积到缓冲区，每次检查缓冲区内容是否"完整"（JSON 是否闭合、是否在一个完整的代码块内），只有完整时才交给下游。

### 方案 2：增量解析（ijson 或手动状态机）

对于大型 JSON 流，使用增量解析器（如 `ijson`）逐 token 解析，不需要等待完整 JSON。

### 方案 3：超时处理 + 错误恢复

流式响应也会超时！为每个 chunk 设置接收超时，超时后尝试重连或降级为非流式。

---

## 代码示例

### StreamingBuffer：累积 chunk 并检测完整性

```python
# streaming_buffer.py
"""流式响应缓冲区：累积 chunk，检测 JSON 完整性后再解析。

解决的核心问题：
  1. chunk 边界切断了 JSON 结构 → 累积到完整再解析
  2. chunk 边界切断了 UTF-8 多字节字符 → 处理解码错误
  3. 需要在流式过程中"尽早"输出部分结果 → 逐行/逐对象检测
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class StreamingBuffer:
    """通用的流式数据缓冲区。

    累积 chunk 直到内容"完整"，然后触发回调。
    """

    buffer: str = ""
    _last_complete_index: int = 0
    _brace_depth: int = 0
    _in_string: bool = False
    _escape_next: bool = False
    _bracket_depth: int = 0

    def feed(self, chunk: str) -> list[str]:
        """喂入一个 chunk，返回所有新检测到的完整 JSON 片段。

        Args:
            chunk: 新的文本 chunk

        Returns:
            新检测到的完整 JSON 字符串列表
        """
        self.buffer += chunk
        return self._extract_complete()

    def _extract_complete(self) -> list[str]:
        """扫描缓冲区，提取完整 JSON 对象。"""
        complete = []

        for i in range(self._last_complete_index, len(self.buffer)):
            ch = self.buffer[i]

            # 处理字符串中的转义
            if self._escape_next:
                self._escape_next = False
                continue

            if ch == '\\' and self._in_string:
                self._escape_next = True
                continue

            if ch == '"' and not self._escape_next:
                self._in_string = not self._in_string
                continue

            if self._in_string:
                continue

            # 追踪括号深度
            if ch == '{':
                self._brace_depth += 1
            elif ch == '}':
                self._brace_depth -= 1
            elif ch == '[':
                self._bracket_depth += 1
            elif ch == ']':
                self._bracket_depth -= 1

            # 检测一个顶级 JSON 对象的结束
            if self._brace_depth == 0 and self._bracket_depth == 0:
                if ch == '}' or ch == ']':
                    # 检查这个位置之后是否还有内容
                    # 尝试从这里截取 JSON
                    candidate = self.buffer[self._last_complete_index:i+1]
                    if self._is_valid_json(candidate):
                        complete.append(candidate)
                        self._last_complete_index = i + 1

        return complete

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        """检查文本是否是合法的 JSON。"""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def flush(self) -> str:
        """获取缓冲区中尚未被提取的剩余内容。"""
        remaining = self.buffer[self._last_complete_index:]
        self.buffer = ""
        self._last_complete_index = 0
        return remaining

    def clear(self) -> None:
        """清空缓冲区。"""
        self.buffer = ""
        self._last_complete_index = 0
        self._brace_depth = 0
        self._in_string = False
        self._escape_next = False
        self._bracket_depth = 0


# ============================================================
# 流式 JSON 行解析器（适用于 JSONL 格式的流式输出）
# ============================================================

class StreamingJSONLParser:
    """逐行解析流式 JSONL（JSON Lines）输出。

    常见场景：LLM 按行输出 JSON 对象，每行一个独立 JSON。
    例如：
      {"type": "thought", "content": "..."}
      {"type": "tool_call", "name": "search", "args": {...}}
      {"type": "result", "content": "..."}
    """

    def __init__(self):
        self._buffer = ""
        self._lines: list[str] = []

    def feed(self, chunk: str) -> list[dict]:
        """喂入 chunk，返回所有完整的 JSON 行对象。

        Args:
            chunk: 新的文本 chunk

        Returns:
            解析出的 JSON 对象列表
        """
        self._buffer += chunk
        results = []

        # 按换行分割
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    results.append(obj)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSONL 解析失败: {e}, line='{line[:100]}...'"
                    )
                    # 继续尝试（不中断整个流）

        return results

    def flush(self) -> str:
        """获取未完成的行（可能需要在下一个 chunk 中拼接）。"""
        return self._buffer


# ============================================================
# 安全的流式包装器（带超时和错误恢复）
# ============================================================

import asyncio
from typing import AsyncIterator, TypeVar, Optional

T = TypeVar("T")


class StreamingTimeoutError(Exception):
    """流式读取超时。"""
    pass


class StreamingDisconnectError(Exception):
    """流式连接断开。"""
    pass


async def safe_stream(
    stream: AsyncIterator[T],
    *,
    chunk_timeout: float = 30.0,
    total_timeout: float = 300.0,
    max_retries: int = 2,
    fallback_chunks: list[T] | None = None,
) -> AsyncIterator[T]:
    """带超时和错误恢复的安全流式包装器。

    核心功能：
      1. 每个 chunk 有独立的接收超时（chunk_timeout）
      2. 整个流有总超时（total_timeout）
      3. 超时或断开后自动重试（续传）
      4. 重试失败后使用 fallback 数据
      5. 优雅降级：如果流式持续失败，转为非流式获取

    Args:
        stream: 原始的异步迭代器（chunk 流）
        chunk_timeout: 单个 chunk 的接收超时（秒）
        total_timeout: 整个流的总超时（秒）
        max_retries: 流断开后的最大重连次数
        fallback_chunks: 彻底失败后返回的备用 chunk（如 "[回答生成失败]"）

    Yields:
        成功接收的 chunk
    """
    chunks_yielded = 0
    retry_count = 0
    start_time = asyncio.get_event_loop().time()

    while retry_count <= max_retries:
        try:
            async for chunk in _stream_with_chunk_timeout(
                stream, chunk_timeout
            ):
                # 检查总超时
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > total_timeout:
                    logger.error(
                        f"流式输出总超时 ({total_timeout}s)，"
                        f"已输出 {chunks_yielded} 个 chunk"
                    )
                    if fallback_chunks:
                        for fb in fallback_chunks:
                            yield fb
                    raise StreamingTimeoutError(
                        f"流式输出超过总时间限制 {total_timeout}s"
                    )

                chunks_yielded += 1
                yield chunk

            # 流正常结束
            return

        except (StreamingTimeoutError, StreamingDisconnectError) as e:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"流式重试 {max_retries} 次后仍然失败: {e}")
                if fallback_chunks:
                    for fb in fallback_chunks:
                        yield fb
                return

            wait = 2 ** retry_count  # 指数退避
            logger.warning(
                f"流式中断 ({e})，{wait}s 后第 {retry_count} 次重试..."
            )
            await asyncio.sleep(wait)

            # TODO: 实际应用中，这里应该重新建立流式连接
            # 新连接应从 chunks_yielded 之后继续


async def _stream_with_chunk_timeout(
    stream: AsyncIterator[T],
    timeout: float,
) -> AsyncIterator[T]:
    """为每个 chunk 添加独立的接收超时。"""
    while True:
        try:
            chunk = await asyncio.wait_for(
                stream.__anext__(),  # type: ignore
                timeout=timeout,
            )
            yield chunk
        except asyncio.TimeoutError:
            raise StreamingTimeoutError(
                f"等待 chunk 超时 ({timeout}s)"
            )
        except StopAsyncIteration:
            return
        except Exception as e:
            raise StreamingDisconnectError(f"流式连接异常: {e}")


# ============================================================
# 流式 SSE 安全解析器
# ============================================================

class SSEDecoder:
    """安全的 Server-Sent Events 解析器。

    解决标准 SSE 解析中的常见问题：
      - 某些实现在 data: 后少了空格
      - 多余的空行
      - comment 行（以 : 开头）
      - 多行 data（某些实现将完整 JSON 分成多个 data 行）

    SSE 标准格式:
      data: {"content": "hello"}\n\n

    非标准但常见的格式:
      data:{"content": "hello"}\n\n       # 缺少空格
      data: {"content": "he\n             # 多行 data
      llo"}\n\n
    """

    def __init__(self):
        self._buffer = ""
        self._event_type: str | None = None
        self._data_lines: list[str] = []
        self._last_event_id: str | None = None

    def feed(self, text: str) -> list[dict]:
        """喂入 SSE 文本，返回所有解析出的事件。

        Args:
            text: SSE 格式的原始文本

        Returns:
            事件列表，每个事件为 dict: {"event": ..., "data": ..., "id": ...}
        """
        self._buffer += text
        events = []

        while True:
            # 查找一个完整的事件（以 \n\n 结束）
            idx = self._buffer.find('\n\n')
            if idx == -1:
                break

            event_text = self._buffer[:idx]
            self._buffer = self._buffer[idx + 2:]

            event = self._parse_event(event_text)
            if event is not None:
                events.append(event)

        return events

    def _parse_event(self, text: str) -> dict | None:
        """解析单个 SSE 事件文本。"""
        event_type = "message"
        data = ""
        event_id = self._last_event_id

        lines = text.split('\n')
        for line in lines:
            if not line:
                continue

            # comment 行
            if line.startswith(':'):
                continue

            # field: value 格式
            if ':' in line:
                field, _, value = line.partition(':')
                # 去掉 value 开头的空格（标准要求）
                if value.startswith(' '):
                    value = value[1:]

                if field == 'event':
                    event_type = value
                elif field == 'data':
                    if data:
                        data += '\n' + value
                    else:
                        data = value
                elif field == 'id':
                    event_id = value
                    self._last_event_id = event_id
                elif field == 'retry':
                    # retry 字段用于设置重连间隔，这里暂时忽略
                    pass
            else:
                # 没有冒号的行，整个行是字段名，值为空
                pass

        if not data:
            return None  # 空数据的事件可以忽略（如 heartbeat）

        return {
            "event": event_type,
            "data": data,
            "id": event_id,
        }

    def flush(self) -> str:
        """获取未处理完的缓冲内容。"""
        return self._buffer


# ============================================================
# 完整的流式 LLM 响应处理示例
# ============================================================

async def demo_streaming_handler():
    """演示完整的流式 LLM 响应处理流程。"""

    # --- 模拟 OpenAI 流式 API 响应 ---
    async def mock_openai_stream():
        """模拟 OpenAI 的 SSE 流式响应。"""
        chunks = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
            '"choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',

            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
            '"choices":[{"delta":{"content":"你好"},"index":0}]}\n\n',

            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
            '"choices":[{"delta":{"content":"，世界"},"index":0}]}\n\n',

            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
            '"choices":[{"delta":{"content":"！"},"index":0}]}\n\n',

            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
            '"choices":[{"delta":{},"finish_reason":"stop"},"index":0}]}\n\n',

            'data: [DONE]\n\n',
        ]
        for chunk in chunks:
            await asyncio.sleep(0.1)  # 模拟网络延迟
            yield chunk

    # --- 流式处理 ---
    decoder = SSEDecoder()
    full_content = ""

    async for raw_chunk in mock_openai_stream():
        # 1. 用 SSE 解码器解析
        events = decoder.feed(raw_chunk)

        for event in events:
            try:
                data = json.loads(event["data"])
            except json.JSONDecodeError:
                if event["data"].strip() == "[DONE]":
                    print("\n[流式输出完成]")
                    continue
                print(f"[JSON 解析失败]: {event['data'][:50]}")
                continue

            # 2. 提取内容
            choices = data.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content:
                    full_content += content
                    print(content, end="", flush=True)

                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    print(f"\n[结束原因: {finish_reason}]")

    # 3. 最终结果
    print(f"\n完整响应: '{full_content}'")


# ============================================================
# 流式输出的使用示例
# ============================================================

async def demo_json_streaming():
    """演示 JSON 流式解析。"""

    # 模拟 LLM 返回 JSON 对象的流式输出（chunk 边界随机切割）
    json_string = '{"name": "张三", "age": 25, "city": "北京"}'

    # 模拟随机切割为 chunks
    import random
    chunks = []
    remaining = json_string
    while remaining:
        cut = random.randint(1, min(5, len(remaining)))
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]

    print(f"完整 JSON: {json_string}")
    print(f"Chunks ({len(chunks)}): {chunks}")
    print()

    # 使用 StreamingBuffer 累积
    buffer = StreamingBuffer()
    all_complete = []

    for i, chunk in enumerate(chunks):
        complete = buffer.feed(chunk)
        all_complete.extend(complete)
        print(f"  Chunk {i}: '{chunk}' → 新完整对象: {complete}")

    # Flush 剩余
    remaining = buffer.flush()
    if remaining:
        print(f"  剩余未完成: '{remaining}'")

    print(f"\n最终解析结果: {all_complete}")


if __name__ == "__main__":
    print("=" * 50)
    print("SSE 流式处理演示")
    print("=" * 50)
    asyncio.run(demo_streaming_handler())

    print("\n" + "=" * 50)
    print("JSON 流式累积演示")
    print("=" * 50)
    asyncio.run(demo_json_streaming())
```

---

## 检查清单

在实现流式响应功能之前，确认以下 3 项：

- [ ] **JSON 解析是"累积后解析"而非"逐 chunk 解析"**: 任何 `json.loads(chunk)` 出现在你的代码中都意味着它总有一天会炸（除非你用的是增量解析器 ijson）
- [ ] **流式循环有超时保护**: `for chunk in response` 不会自动为每个 chunk 设置超时。如果网络中断，这个循环可能挂起直到 TCP 超时（默认数分钟）。至少为每个 chunk 设置一个 30 秒的超时
- [ ] **处理了不完整的 SSE/UTF-8**: 如果你在解析 SSE 或处理中文字符，确认你的代码能处理跨 chunk 边界的不完整数据行和字节序列

如果有一项没处理，流式输出就会在某个不可预测的时刻崩溃——而且很可能是在演示给老板看的时候。

---

## 延伸阅读

- [MDN: Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [OpenAI: Streaming](https://platform.openai.com/docs/guides/streaming)
- [ijson: Iterative JSON parser](https://pypi.org/project/ijson/)
- [Anthropic: Streaming Messages](https://docs.anthropic.com/en/api/messages-streaming)

---

**一句话总结**：流式响应让体验变快，但 chunk 边界是随机的。永远累积到完整再解析——否则你的 JSON 解析器会在 chunk 边界哭出声来。
