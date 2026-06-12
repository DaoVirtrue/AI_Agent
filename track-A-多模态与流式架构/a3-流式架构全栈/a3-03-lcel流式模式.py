#!/usr/bin/env python3
"""
A3-03: LCEL流式模式 (LangChain Expression Language Streaming)
===============================================================
学习目标:
  1. .astream() vs .astream_events() 的区别与用法
  2. on_llm_new_token 回调
  3. 流式输出 + 并行工具调用
  4. JSON流式缓冲
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass, field

# ============================================================
# Section 1: 模拟LCEL链（纯Python实现，无LangChain依赖）
# ============================================================

@dataclass
class StreamEvent:
    """流式事件"""
    event_type: str  # "on_chat_model_stream", "on_tool_start", "on_tool_end"
    name: str        # 事件名称
    data: Any        # 事件数据
    run_id: str      # 运行ID


class MockLLM:
    """模拟LLM（代替真实LLM，演示流式模式）"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    async def astream(self, messages: List[Dict],
                      **kwargs) -> AsyncIterator[str]:
        """
        模拟 .astream() 流式输出
        每步yield一个token字符串
        """
        full_response = (
            "RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的AI架构。"
            "它首先从知识库中检索相关文档，然后将这些文档与用户查询一起输入到大语言模型中，"
            "从而生成更准确、更可靠的回答。RAG的主要优势包括：减少幻觉、"
            "知识可更新、可溯源引用。"
        )
        for char in full_response:
            await asyncio.sleep(0.02)
            yield char

    async def astream_events(self, messages: List[Dict],
                             **kwargs) -> AsyncIterator[StreamEvent]:
        """
        模拟 .astream_events() 流式输出
        相比于 .astream()，增加了事件类型标记
        """
        run_id = f"run_{time.time_ns()}"

        # 1. 开始事件
        yield StreamEvent("on_chain_start", "rag_chain", {}, run_id)

        # 2. 检索步骤
        yield StreamEvent(
            "on_retriever_start", "vector_retriever",
            {"query": messages[-1].get("content", "")}, run_id
        )
        await asyncio.sleep(0.1)
        yield StreamEvent(
            "on_retriever_end", "vector_retriever",
            {"documents": ["文档1: RAG概述...", "文档2: RAG实现方法..."]}, run_id
        )

        # 3. LLM开始
        yield StreamEvent(
            "on_chat_model_start", "gpt-4o-mini",
            {"messages": messages}, run_id
        )

        # 4. Token-by-token流式
        full_response = (
            "基于检索到的文档，RAG系统具有以下特点：\n"
            "1. 检索增强：通过向量检索获取相关知识\n"
            "2. 生成准确：LLM基于检索结果生成回答\n"
            "3. 可溯源：每个回答都可以追溯到来源文档"
        )
        for char in full_response:
            await asyncio.sleep(0.015)
            yield StreamEvent(
                "on_chat_model_stream", "gpt-4o-mini",
                {"chunk": char, "delta": char}, run_id
            )

        # 5. LLM结束
        yield StreamEvent(
            "on_chat_model_end", "gpt-4o-mini",
            {"total_tokens": len(full_response)}, run_id
        )

        # 6. 链结束
        yield StreamEvent(
            "on_chain_end", "rag_chain",
            {"output": full_response}, run_id
        )


class MockTool:
    """模拟工具调用"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def run(self, input_data: str) -> str:
        """模拟工具执行"""
        await asyncio.sleep(0.3)
        return f"[{self.name}] 处理结果: 已分析'{input_data[:30]}...'"


# ============================================================
# Section 2: .astream() vs .astream_events() 对比
# ============================================================

class StreamModeComparer:
    """对比两种流式模式的用法"""

    def __init__(self):
        self.llm = MockLLM()

    async def demo_astream(self, query: str) -> str:
        """
        .astream() 模式:
        - 最简单，只返回LLM输出的token流
        - 不包含内部步骤的事件
        - 适合只需要最终输出的场景
        """
        print("\n  [.astream() 模式] 仅输出token:")
        print("  " + "-" * 40)

        full_text = ""
        async for token in self.llm.astream([{"role": "user", "content": query}]):
            print(token, end="", flush=True)
            full_text += token

        print("\n  " + "-" * 40)
        return full_text

    async def demo_astream_events(self, query: str) -> tuple[str, List[StreamEvent]]:
        """
        .astream_events() 模式:
        - 返回所有内部步骤的事件
        - 可以监控检索、工具调用、LLM token等多个步骤
        - 适合需要展示进度的场景
        """
        print("\n  [.astream_events() 模式] 完整事件流:")
        print("  " + "-" * 40)

        events = []
        full_text = ""
        retrieval_docs = []

        async for event in self.llm.astream_events(
            [{"role": "user", "content": query}]
        ):
            events.append(event)

            if event.event_type == "on_retriever_end":
                retrieval_docs = event.data.get("documents", [])
                print(f"  [检索完成] 找到 {len(retrieval_docs)} 个文档")

            elif event.event_type == "on_chat_model_stream":
                token = event.data.get("chunk", "")
                print(token, end="", flush=True)
                full_text += token

            elif event.event_type == "on_chain_end":
                print(f"\n  [链完成] 总tokens: {event.data.get('total_tokens', '?')}")

        print("\n  " + "-" * 40)
        return full_text, events


# ============================================================
# Section 3: on_llm_new_token 回调模式
# ============================================================

class TokenCallbackHandler:
    """
    on_llm_new_token 回调模式

    这是另一种获取流式token的方式：
    不通过async generator，而是通过回调函数

    用途: 在非流式框架中模拟流式效果
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.subscribers: List[callable] = []

    def on_llm_new_token(self, token: str, **kwargs):
        """
        每当LLM生成新token时调用
        """
        self.tokens.append(token)

        # 通知所有订阅者
        for subscriber in self.subscribers:
            subscriber(token, **kwargs)

    def subscribe(self, callback: callable):
        """注册回调订阅者"""
        self.subscribers.append(callback)

    def get_full_text(self) -> str:
        return "".join(self.tokens)

    async def run_with_callback(self, llm: MockLLM, query: str):
        """使用回调模式运行流式生成"""
        print("\n  [回调模式] on_llm_new_token:")
        print("  " + "-" * 40)

        # 注册一个打印订阅者
        def print_token(token, **kwargs):
            print(token, end="", flush=True)

        self.subscribe(print_token)

        async for token in llm.astream([{"role": "user", "content": query}]):
            self.on_llm_new_token(token)

        print("\n  " + "-" * 40)
        return self.get_full_text()


# ============================================================
# Section 4: 流式输出 + 并行工具调用
# ============================================================

class StreamingWithTools:
    """
    流式输出 + 并行工具调用

    场景: LLM在生成答案时，同时需要调用外部工具
    挑战: 如何在流式输出的同时，管理并发的工具调用？

    策略:
    1. 分析LLM输出中是否需要工具调用
    2. 如果需要，在后台启动工具调用（不阻塞token流）
    3. 工具结果回来后，插入到流中作为事件
    """

    def __init__(self):
        self.tools = {
            "search": MockTool("search", "搜索知识库"),
            "calculator": MockTool("calculator", "执行数学计算"),
            "translate": MockTool("translate", "翻译文本"),
        }

    async def stream_with_tools(self, query: str,
                                tool_name: str = None,
                                tool_input: str = None
                                ) -> AsyncIterator[StreamEvent]:
        """
        流式输出 + 并行工具调用
        """
        run_id = f"run_{time.time_ns()}"

        # 1. 开始事件
        yield StreamEvent("on_chain_start", "agent_chain",
                          {"query": query}, run_id)

        # 2. 如果需要工具调用，在流式输出的同时并行执行
        tool_task = None
        if tool_name and tool_name in self.tools:
            tool = self.tools[tool_name]
            yield StreamEvent("on_tool_start", tool_name,
                              {"input": tool_input}, run_id)
            # 启动并行工具任务
            tool_task = asyncio.create_task(tool.run(tool_input or query))

        # 3. 流式输出（不等待工具）
        mock_answer = f"正在处理您的查询'{query[:20]}...'..."

        if tool_name:
            mock_answer += f"\n同时，正在调用{tool_name}工具获取外部信息..."

        for char in mock_answer:
            await asyncio.sleep(0.02)
            yield StreamEvent("on_chat_model_stream", "agent_llm",
                              {"chunk": char}, run_id)

        # 4. 等待工具结果
        if tool_task:
            tool_result = await tool_task
            yield StreamEvent("on_tool_end", tool_name,
                              {"result": tool_result}, run_id)

            # 将工具结果注入继续生成
            continuation = f"\n\n工具返回结果：{tool_result}"
            for char in continuation:
                await asyncio.sleep(0.02)
                yield StreamEvent("on_chat_model_stream", "agent_llm",
                                  {"chunk": char}, run_id)

        # 5. 完成
        yield StreamEvent("on_chain_end", "agent_chain",
                          {"status": "completed"}, run_id)


# ============================================================
# Section 5: JSON流式缓冲
# ============================================================

class JSONStreamBuffer:
    """
    JSON流式缓冲

    问题: LLM输出的JSON常常不完整（流式输出时）
    解决: 累积token直到JSON解析成功，再发给客户端

    应用场景:
    - 结构化输出（如JSON schema约束的回复）
    - 工具调用（function call的JSON参数）
    """

    def __init__(self, max_buffer_size: int = 4096):
        self.buffer: str = ""
        self.max_buffer_size = max_buffer_size
        self.json_complete: bool = False
        self.last_valid_json: Optional[Dict] = None

    def feed(self, token: str) -> Optional[Dict]:
        """
        喂入token，尝试解析JSON
        返回: 如果JSON完整返回dict，否则返回None
        """
        self.buffer += token

        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]

        # 尝试提取JSON块（寻找匹配的{}对）
        result = self._try_extract_json()

        if result is not None:
            self.json_complete = True
            self.last_valid_json = result

        return result

    def _try_extract_json(self) -> Optional[Dict]:
        """尝试从buffer中提取JSON"""
        # 寻找第一个 { 到最后一个匹配的 }
        start = self.buffer.find("{")
        if start == -1:
            return None

        # 简单的括号匹配
        depth = 0
        for i in range(start, len(self.buffer)):
            if self.buffer[i] == "{":
                depth += 1
            elif self.buffer[i] == "}":
                depth -= 1
                if depth == 0:
                    json_str = self.buffer[start:i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
        return None

    def reset(self):
        """重置缓冲区"""
        self.buffer = ""
        self.json_complete = False
        self.last_valid_json = None

    def get_partial_text(self) -> str:
        """获取缓冲区中非JSON的文本部分"""
        # 返回最后一个完整JSON之后的内容
        if self.last_valid_json:
            last_json_str = json.dumps(self.last_valid_json, ensure_ascii=False)
            idx = self.buffer.find(last_json_str)
            if idx >= 0:
                return self.buffer[idx + len(last_json_str):]
        return self.buffer

    async def stream_structured_output(self,
                                       llm: MockLLM,
                                       query: str
                                       ) -> AsyncIterator[Dict]:
        """
        流式结构化输出
        等到JSON完整才yield
        """
        self.reset()

        async for token in llm.astream([
            {"role": "user", "content": f"{query}\n请以JSON格式回答。"}
        ]):
            # 非结构化token直接输出
            json_result = self.feed(token)

            if json_result is not None:
                yield {"type": "structured", "data": json_result}
                self.buffer = self.get_partial_text()  # 保留剩余文本

            elif not self.json_complete:
                yield {"type": "streaming", "token": token}

        # 输出剩余文本
        remaining = self.buffer.strip()
        if remaining and not self.json_complete:
            yield {"type": "text", "data": remaining}


# ============================================================
# Section 6: 主流程
# ============================================================

async def main_async():
    """异步主函数"""
    print("=" * 70)
    print("A3-03: LCEL流式模式 — 完整演示")
    print("=" * 70)

    query = "什么是RAG？它有哪些优势？"

    # 1. .astream() 演示
    print("\n[测试1] .astream() 模式")
    comparer = StreamModeComparer()
    full_text = await comparer.demo_astream(query)
    print(f"\n  总字符数: {len(full_text)}")

    # 2. .astream_events() 演示
    print("\n[测试2] .astream_events() 模式")
    full_text2, events = await comparer.demo_astream_events(query)
    print(f"  捕获事件数: {len(events)}")
    event_types = set(e.event_type for e in events)
    print(f"  事件类型: {event_types}")

    # 3. 回调模式
    print("\n[测试3] on_llm_new_token 回调模式")
    handler = TokenCallbackHandler()
    callback_text = await handler.run_with_callback(MockLLM(), query)
    print(f"\n  回调捕获tokens: {len(handler.tokens)}")

    # 4. 流式+并行工具调用
    print("\n[测试4] 流式输出 + 并行工具调用")
    stream_with_tools = StreamingWithTools()
    print("  " + "-" * 40)
    async for event in stream_with_tools.stream_with_tools(
        query, tool_name="search", tool_input="RAG系统架构"
    ):
        if event.event_type in ("on_tool_start", "on_tool_end"):
            print(f"\n  [{event.name}] {event.data}")
        elif event.event_type == "on_chat_model_stream":
            print(event.data.get("chunk", ""), end="", flush=True)
    print("\n  " + "-" * 40)

    # 5. JSON流式缓冲
    print("\n[测试5] JSON流式缓冲")
    json_buffer = JSONStreamBuffer()
    print("  " + "-" * 40)
    async for item in json_buffer.stream_structured_output(MockLLM(), query):
        if item["type"] == "structured":
            print(f"\n  [JSON完成] {json.dumps(item['data'], ensure_ascii=False)[:200]}")
        elif item["type"] == "streaming":
            print(item["token"], end="", flush=True)
    print("\n  " + "-" * 40)

    print("\n" + "=" * 70)
    print("A3-03 演示完成！")
    print("=" * 70)


def main():
    """入口"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
