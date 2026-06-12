#!/usr/bin/env python3
"""
A3-02: FastAPI流式端点 (FastAPI Streaming Endpoints)
========================================================
学习目标:
  1. StreamingResponse + async generator实现token-by-token输出
  2. asyncio.Queue背压控制
  3. 连接断开处理
  4. Heartbeat心跳保活
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
import uvicorn

# ============================================================
# Section 1: SSE格式化工具
# ============================================================

class SSEFormatter:
    """SSE (Server-Sent Events) 格式工具"""

    @staticmethod
    def format_event(event: str, data: str,
                     event_id: str = None,
                     retry: int = None) -> str:
        """
        格式化为SSE标准格式

        SSE格式:
        id: <event_id>\n
        event: <event_type>\n
        retry: <milliseconds>\n
        data: <json_string>\n
        \n
        """
        lines = []
        if event_id is not None:
            lines.append(f"id: {event_id}")
        if event is not None:
            lines.append(f"event: {event}")
        if retry is not None:
            lines.append(f"retry: {retry}")
        # data可以跨多行（以\n\n结束）
        for line in data.split("\n"):
            lines.append(f"data: {line}")
        lines.append("")  # 空行表示事件结束
        return "\n".join(lines)

    @staticmethod
    def token_event(token: str, event_id: int) -> str:
        """token流式事件"""
        return SSEFormatter.format_event(
            event="token",
            data=json.dumps({"token": token, "timestamp": time.time()}),
            event_id=str(event_id),
        )

    @staticmethod
    def done_event(finish_reason: str = "stop",
                   usage: Dict = None) -> str:
        """完成事件"""
        return SSEFormatter.format_event(
            event="done",
            data=json.dumps({
                "finish_reason": finish_reason,
                "usage": usage or {},
            }),
        )

    @staticmethod
    def error_event(message: str, code: str = "unknown_error") -> str:
        """错误事件"""
        return SSEFormatter.format_event(
            event="error",
            data=json.dumps({"error": message, "code": code}),
        )

    @staticmethod
    def heartbeat_event() -> str:
        """心跳事件（注释行不会触发EventSource handler）"""
        return ": heartbeat\n\n"


# ============================================================
# Section 2: Token流式生成引擎
# ============================================================

class TokenStreamEngine:
    """
    Token流式生成引擎（模拟LLM输出）
    支持：
    - 模拟逐个token输出
    - 背压控制
    - 中断处理
    """

    def __init__(self, token_delay: float = 0.05):
        """
        参数:
            token_delay: 每个token之间的延迟（秒），模拟LLM推理
        """
        self.token_delay = token_delay
        self._interrupted = False

    def interrupt(self):
        """中断当前生成"""
        self._interrupted = True

    async def generate_tokens(self, query: str,
                              context: str = "",
                              max_tokens: int = 100) -> AsyncGenerator[str, None]:
        """
        异步生成token流

        实际生产中这里会调用 LLM API 的 stream=True
        例如: openai.chat.completions.create(stream=True)
        """
        self._interrupted = False

        # 模拟答案（实际生产中用LLM）
        full_answer = (
            f"关于您的问题'{query[:30]}'，"
            f"基于检索到的{len(context)}字符上下文，"
            f"以下是回答：RAG（检索增强生成）系统通过将检索到的文档片段"
            f"与大型语言模型结合，能够生成更准确、更可靠的回答。"
            f"该系统首先从知识库中检索最相关的信息，"
            f"然后将这些信息作为上下文提供给LLM进行生成。"
        )

        # 按中文字/英文单词分割为token
        tokens = self._tokenize(full_answer, max_tokens)

        event_id = 0
        for token in tokens:
            # 检查中断
            if self._interrupted:
                yield "[生成已中断]"
                return

            # 模拟推理延迟
            await asyncio.sleep(self.token_delay)

            event_id += 1
            yield token

    def _tokenize(self, text: str, max_tokens: int) -> List[str]:
        """
        简易分词（中文按字，英文按空格+标点）
        实际应用中这是LLM tokenizer的工作
        """
        tokens = []
        for char in text:
            if len(tokens) >= max_tokens:
                break
            # 中文字符单独算一个token
            if '一' <= char <= '鿿' or '㐀' <= char <= '䶿':
                tokens.append(char)
            elif char in ('，', '。', '！', '？', '、', '；', '：'):
                tokens.append(char)
            else:
                # 英文/数字累积
                if tokens and tokens[-1] and tokens[-1][-1].isascii() and char.isascii() and char != ' ':
                    tokens[-1] += char
                else:
                    if char.strip():
                        tokens.append(char)
        return tokens[:max_tokens]


# ============================================================
# Section 3: 背压控制（asyncio.Queue）
# ============================================================

class BackpressureController:
    """
    背压控制器

    问题: 生产速度 > 消费速度时，内存无限增长
    解决: 使用有界asyncio.Queue限制缓冲区大小

    策略:
    1. Queue最大长度限制
    2. 达到限制时生产者阻塞（等待消费）
    3. 消费者断开时通知生产者停止
    """

    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._consumer_connected = True
        self._stats = {
            "produced": 0,
            "consumed": 0,
            "dropped": 0,
            "high_water_mark": 0,
        }

    async def produce(self, item: str, timeout: float = 5.0) -> bool:
        """
        生产者放入数据
        队列满时阻塞等待
        消费者断开时快速失败
        """
        if not self._consumer_connected:
            return False

        try:
            await asyncio.wait_for(
                self.queue.put(item),
                timeout=timeout,
            )
            self._stats["produced"] += 1
            current_size = self.queue.qsize()
            if current_size > self._stats["high_water_mark"]:
                self._stats["high_water_mark"] = current_size
            return True
        except asyncio.TimeoutError:
            self._stats["dropped"] += 1
            return False
        except asyncio.CancelledError:
            self._stats["dropped"] += 1
            return False

    async def consume(self) -> Optional[str]:
        """消费者获取数据"""
        try:
            item = await self.queue.get()
            self._stats["consumed"] += 1
            return item
        except asyncio.CancelledError:
            return None

    def disconnect(self):
        """消费者断开"""
        self._consumer_connected = False

    def get_stats(self) -> Dict:
        return {
            **self._stats,
            "current_queue_size": self.queue.qsize(),
        }


# ============================================================
# Section 4: 流式RAG服务
# ============================================================

class StreamingRAGService:
    """流式RAG服务核心"""

    def __init__(self):
        self.engine = TokenStreamEngine(token_delay=0.03)

    async def stream_chat_response(self,
                                   query: str,
                                   context: str = "",
                                   heartbeat_interval: float = 15.0
                                   ) -> AsyncGenerator[str, None]:
        """
        流式聊天响应生成器
        每个yield是一个SSE格式的事件字符串

        功能:
        - Token-by-token SSE事件
        - 心跳保活
        - 连接断开检测
        - 背压控制
        """
        controller = BackpressureController(max_queue_size=50)
        event_id = 0
        last_heartbeat = time.time()

        # 启动生产者协程
        async def produce_tokens():
            async for token in self.engine.generate_tokens(query, context):
                sse_event = SSEFormatter.token_event(token, event_id)
                nonlocal event_id
                event_id += 1
                await controller.produce(sse_event)

            # 发送完成事件
            done_event = SSEFormatter.done_event("stop")
            await controller.produce(done_event)

        producer_task = asyncio.create_task(produce_tokens())

        try:
            while True:
                # 从队列消费
                try:
                    sse_data = await asyncio.wait_for(
                        controller.consume(), timeout=heartbeat_interval
                    )
                except asyncio.TimeoutError:
                    # 超时发送心跳
                    heartbeat = SSEFormatter.heartbeat_event()
                    yield heartbeat
                    last_heartbeat = time.time()
                    continue

                if sse_data is None:
                    break  # 消费者已断开

                yield sse_data

                # 检查是否是完成事件
                if '"finish_reason"' in sse_data:
                    break

            # 等待生产者完成
            await producer_task

        except asyncio.CancelledError:
            # 客户端断开
            controller.disconnect()
            self.engine.interrupt()
            producer_task.cancel()
            print(f"  [断开] 连接被取消, 背压统计: {controller.get_stats()}")
            raise

        except Exception as e:
            error_event = SSEFormatter.error_event(str(e))
            yield error_event

        finally:
            if not producer_task.done():
                producer_task.cancel()


# ============================================================
# Section 5: FastAPI应用
# ============================================================

from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG Streaming API", version="1.0")

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
rag_service = StreamingRAGService()


@app.get("/")
async def root():
    """API文档页"""
    return HTMLResponse("""
    <html>
    <head><title>RAG Streaming API</title></head>
    <body>
        <h1>RAG Streaming API</h1>
        <ul>
            <li><a href="/docs">/docs</a> - Swagger API文档</li>
            <li>POST /api/chat/stream - SSE流式聊天</li>
            <li>GET /api/chat/demo-stream - 演示流式输出</li>
            <li>GET /api/health - 健康检查</li>
        </ul>
        <h2>测试流式端点</h2>
        <div id="output" style="border:1px solid #ccc; padding:10px; min-height:200px; white-space:pre-wrap;"></div>
        <button onclick="testStream()">开始流式测试</button>
        <script>
        function testStream() {
            const output = document.getElementById('output');
            output.textContent = '';
            const evtSource = new EventSource('/api/chat/demo-stream');
            evtSource.addEventListener('token', (e) => {
                const data = JSON.parse(e.data);
                output.textContent += data.token;
            });
            evtSource.addEventListener('done', (e) => {
                const data = JSON.parse(e.data);
                output.textContent += '\\n\\n[完成] ' + data.finish_reason;
                evtSource.close();
            });
            evtSource.addEventListener('error', (e) => {
                output.textContent += '\\n[错误]';
                evtSource.close();
            });
        }
        </script>
    </body>
    </html>
    """)


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """
    POST方式SSE流式端点

    请求体：
    {
        "query": "什么是RAG?",
        "context": "检索到的上下文...",
        "stream": true
    }

    响应: text/event-stream (SSE)
    """
    body = await request.json()
    query = body.get("query", "")
    context = body.get("context", "")
    stream = body.get("stream", True)

    if not query:
        return {"error": "query is required"}

    if not stream:
        return {"answer": f"非流式回答: {query}"}

    async def event_generator():
        """包装rag_service的异步生成器"""
        async for sse_event in rag_service.stream_chat_response(query, context):
            # 检查客户端是否断开
            if await request.is_disconnected():
                break
            yield sse_event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # Nginx禁用缓冲
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/chat/demo-stream")
async def demo_stream():
    """
    GET方式演示流式端点（使用EventSource测试）
    这是最简单的SSE测试方式
    """
    async def event_generator():
        async for sse_event in rag_service.stream_chat_response(
            "什么是RAG系统？", "RAG is Retrieval-Augmented Generation..."
        ):
            yield sse_event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "RAG Streaming API",
    }


# ============================================================
# Section 6: 主流程
# ============================================================

def main():
    """主函数：启动FastAPI服务器"""
    print("=" * 70)
    print("A3-02: FastAPI流式端点 — 演示")
    print("=" * 70)

    print("\n启动FastAPI服务器...")
    print("  访问 http://localhost:8000 查看演示页面")
    print("  访问 http://localhost:8000/docs 查看API文档")
    print("  测试流式: curl -N http://localhost:8000/api/chat/demo-stream")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=30,  # keep-alive超时
    )


if __name__ == "__main__":
    main()
