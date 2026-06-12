#!/usr/bin/env python3
"""
C5-02：MCP 传输模式对比
==========================
MCP 支持三种传输模式，各有适用场景：

1. stdio（标准输入输出）———— 本地进程通信
2. HTTP/SSE（Server-Sent Events）———— 远程服务
3. Streamable HTTP ———— 新一代流式 HTTP（2025+）

本文件演示三种模式的代码实现、对比和选择指南。

依赖：pip install mcp uvicorn starlette httpx
"""

import os
import asyncio
import json
from typing import Optional


# ============================================================================
# 第一部分：传输模式概述
# ============================================================================

def explain_transport_modes():
    """三种传输模式的概述。"""
    print("=" * 60)
    print("【MCP 三种传输模式】")
    print("=" * 60)

    print("""
    ┌──────────────┬─────────────────┬──────────────────┬─────────────────┐
    │   特性       │    stdio         │   HTTP/SSE        │ Streamable HTTP │
    ├──────────────┼─────────────────┼──────────────────┼─────────────────┤
    │ 通信方式     │ stdin/stdout     │ HTTP POST + SSE   │ HTTP + 双向流   │
    │ 连接类型     │ 进程内           │ 客户端→服务端      │ 双向            │
    │ 延迟         │ 极低 (<1ms)      │ 低 (5-50ms)      │ 低 (5-30ms)    │
    │ 适用场景     │ 本地工具/CLI     │ Web 服务/远程工具  │ 生产级远程服务  │
    │ 服务器推送   │ ❌ 不适用       │ ✅ SSE 推送       │ ✅ 原生支持     │
    │ 断线重连     │ ❌ 进程结束即断  │ ⚠️ 需自行实现     │ ✅ 原生支持     │
    │ 负载均衡     │ ❌               │ ✅ 可负载均衡     │ ✅ 可负载均衡   │
    │ 认证         │ ❌ 本地信任     │ ✅ Bearer/API Key │ ✅ 同上         │
    │ 多客户端     │ ❌ 单客户端     │ ✅ 多客户端       │ ✅ 多客户端     │
    │ 协议版本     │ MCP 2024-11-05  │ MCP 2024-11-05   │ MCP 2025+      │
    │ 生产就绪度   │ ✅ (本地场景)   │ ✅               │ ⚠️ (较新)      │
    └──────────────┴─────────────────┴──────────────────┴─────────────────┘
    """)


# ============================================================================
# 第二部分：stdio 传输模式
# ============================================================================

async def demo_stdio_transport():
    """
    stdio 传输模式：通过标准输入输出流进行通信。

    适用场景：
    - Claude Desktop 集成（本地运行 MCP Server）
    - 命令行工具
    - 开发调试

    优点：零网络开销、无需认证、部署最简单
    缺点：不能远程访问、单客户端、进程生命周期管理
    """
    print("=" * 60)
    print("【1】stdio 传输模式 —— 本地进程通信")
    print("=" * 60)

    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    # 创建简单的 Echo Server
    server = Server("stdio-echo-demo")

    @server.tool()
    async def echo(message: str) -> str:
        """回显消息（用于测试 stdio 通信）。"""
        return f"[Echo] {message}"

    @server.tool()
    async def get_env(env_var: str) -> str:
        """获取环境变量（本地访问）。"""
        return os.environ.get(env_var, f"环境变量 '{env_var}' 未设置")

    print("\n  Server 'stdio-echo-demo' 已创建")
    print("  stdio 传输模式特点:")
    print("    - 通过 stdin 接收 JSON-RPC 请求")
    print("    - 通过 stdout 发送 JSON-RPC 响应")
    print("    - 一行一个 JSON 对象（jsonlines 格式）")
    print("\n  实际代码:")
    print("""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, ...)
    """)

    print("\n  Claude Desktop 配置:")
    print("""
    {
      "mcpServers": {
        "echo-demo": {
          "command": "python",
          "args": ["c5-02-mcp传输模式.py", "--mode", "stdio"]
        }
      }
    }
    """)

    return server


# ============================================================================
# 第三部分：HTTP/SSE 传输模式
# ============================================================================

async def demo_http_sse_transport():
    """
    HTTP/SSE 传输模式：通过 HTTP 请求 + SSE 推送进行通信。

    架构：
    - POST /mcp: 客户端发送 JSON-RPC 请求
    - GET /sse: 服务端通过 SSE 推送通知（工具结果、进度更新）

    适用场景：
    - Web 应用集成
    - 微服务架构中的远程 MCP 服务
    - 需要多客户端访问的场景
    """
    print("\n" + "=" * 60)
    print("【2】HTTP/SSE 传输模式 —— 远程服务通信")
    print("=" * 60)

    from mcp.server import Server

    server = Server("http-sse-demo")

    @server.tool()
    async def query_database(sql: str) -> str:
        """执行 SQL 查询（远程数据库访问）。"""
        return f"[模拟] SQL 执行结果: {sql[:50]}..."

    @server.tool()
    async def long_running_task(task_name: str, duration: int = 3) -> str:
        """
        长时间运行的任务（演示 SSE 进度推送）。

        在实际实现中，服务端会通过 SSE 推送进度。
        """
        import asyncio
        await asyncio.sleep(min(duration, 1))  # 模拟耗时
        return f"任务 '{task_name}' 已完成"

    print(f"\n  Server 'http-sse-demo' 已创建")
    print("\n  Starlette/FastAPI 集成代码:")
    code = '''
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import StreamingResponse
from mcp.server.sse import SseServerTransport

# 创建 SSE 传输
sse = SseServerTransport("/messages/")

app = Starlette(
    routes=[
        Route("/sse", endpoint=sse.handle_sse),
        Route("/messages/", endpoint=sse.handle_post_message),
    ]
)

# 启动服务
# uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    print(code)

    return server


# ============================================================================
# 第四部分：Streamable HTTP 传输模式
# ============================================================================

def demo_streamable_http():
    """
    Streamable HTTP：MCP 2025 年引入的新传输模式。

    特点：
    - 单一端点：不需要分离 SSE + POST
    - 双向流：客户端和服务端都可以推送消息
    - 原生重连：断线后自动恢复会话
    - 更简单的部署：不需要反向代理配置 SSE
    """
    print("\n" + "=" * 60)
    print("【3】Streamable HTTP —— 新一代流式 HTTP")
    print("=" * 60)

    print("""
    Streamable HTTP 架构：

    ┌──────────┐                ┌──────────────────┐
    │  Client  │   HTTP POST    │  MCP Server       │
    │          │ ──────────────>│  (单一端点 /mcp)  │
    │          │                │                   │
    │          │ <─ 流式响应 ───│                   │
    │          │   (JSON Lines) │                   │
    │          │                │                   │
    │          │ <─ 服务端推送 ─│ (原生支持)        │
    │          │                │                   │
    │          │ ── 心跳/续约 ─>│ (会话管理)        │
    └──────────┘                └──────────────────┘

    与 HTTP/SSE 的关键区别：

    ┌──────────────────┬─────────────────┬─────────────────┐
    │   特性           │  HTTP/SSE       │ Streamable HTTP │
    ├──────────────────┼─────────────────┼─────────────────┤
    │ 端点数量         │  2 (POST + SSE) │  1 (单一)       │
    │ 服务端→客户端    │  SSE 通道       │  同一流          │
    │ 客户端→服务端    │  POST 请求      │  同一流 + POST   │
    │ 会话恢复         │  需自行实现     │  原生支持        │
    │ 反向代理配置     │  需特殊配置     │  标准即可        │
    │ 协议开销         │  较高           │  较低            │
    └──────────────────┴─────────────────┴─────────────────┘

    未来趋势：Streamable HTTP 将成为 MCP 远程通信的首选模式。
    目前状态：2025年引入，逐步被主流 SDK 支持。
    """)


# ============================================================================
# 第五部分：传输模式选择决策树
# ============================================================================

def print_transport_decision_tree():
    """传输模式选择指南。"""
    print("\n" + "=" * 60)
    print("【4】传输模式选择决策树")
    print("=" * 60)

    print("""
    开始 ──> MCP Server 部署在哪里？

      ├── "本地机器（与 Client 同一台）"
      │   └── → stdio
      │       理由：零网络开销、无需认证、最简部署
      │       典型：Claude Desktop 插件、本地 CLI 工具
      │
      ├── "远程服务器（需要跨网络访问）"
      │   ├── "需要服务端主动推送（如流式进度）"
      │   │   ├── "已有基础设施支持 SSE" → HTTP/SSE
      │   │   └── "全新项目，追求简洁" → Streamable HTTP
      │   │
      │   └── "不需要服务端推送"
      │       └── → HTTP/SSE 或 Streamable HTTP
      │
      └── "既需要本地又需要远程访问"
          └── → 同时支持 stdio + HTTP/SSE
              理由：模块化设计，适配不同客户端

    快速决策表：

    ┌───────────────────────────┬──────────────┐
    │ 你的情况                  │ 推荐传输模式  │
    ├───────────────────────────┼──────────────┤
    │ 给 Claude Desktop 写插件 │ stdio         │
    │ 构建内部 API 的 MCP 包装  │ HTTP/SSE      │
    │ 微服务间的 MCP 通信      │ Streamable HTTP│
    │ 快速开发调试              │ stdio         │
    │ 生产环境多客户端访问      │ HTTP/SSE      │
    │ 需要服务端推送进度        │ HTTP/SSE      │
    │ 追求最简部署              │ stdio         │
    └───────────────────────────┴──────────────┘
    """)


# ============================================================================
# 第六部分：实现多模式支持的 MCP Server
# ============================================================================

async def demo_multi_transport_server():
    """演示如何构建同时支持 stdio 和 HTTP/SSE 的 MCP Server。"""
    print("\n" + "=" * 60)
    print("【5】多模式 MCP Server —— 同时支持 stdio + HTTP/SSE")
    print("=" * 60)

    from mcp.server import Server

    # 公共的 Server 定义（与传输模式无关）
    server = Server("multi-transport-demo")

    @server.tool()
    async def ping() -> str:
        return "pong"

    @server.tool()
    async def get_server_time() -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    print("""
    同时支持 stdio 和 HTTP/SSE 的代码结构：

    # ----- mcp_server.py -----
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette

    # 1. 定义 Server（公共）
    server = Server("my-server")

    @server.tool()
    async def my_tool(param: str) -> str:
        ...

    # 2. stdio 入口
    async def run_stdio():
        async with stdio_server() as (read, write):
            await server.run(read, write, ...)

    # 3. HTTP/SSE 入口
    sse = SseServerTransport("/messages/")
    app = Starlette(routes=[
        Route("/sse", endpoint=sse.handle_sse),
        Route("/messages/", endpoint=sse.handle_post_message),
    ])

    # 4. 命令行切换
    if __name__ == "__main__":
        import sys
        if "--http" in sys.argv:
            uvicorn.run(app, port=8000)
        else:
            asyncio.run(run_stdio())
    """)

    return server


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("MCP 传输模式对比实战")
    print("=" * 60)

    explain_transport_modes()

    async def main():
        await demo_stdio_transport()
        await demo_http_sse_transport()
        demo_streamable_http()
        print_transport_decision_tree()
        await demo_multi_transport_server()

    asyncio.run(main())

    print("\n[完成] MCP 传输模式对比演示结束。")
    print("  建议: 先用 stdio 开发调试，再用 HTTP/SSE 部署生产。")
