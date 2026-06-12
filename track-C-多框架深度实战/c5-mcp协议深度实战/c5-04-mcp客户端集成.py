#!/usr/bin/env python3
"""
C5-04：MCP 客户端集成
========================
从客户端角度使用 MCP Server。涵盖三种客户端模式：

1. stdio 客户端 —— 管理子进程，与本地 MCP Server 通信
2. HTTP 客户端 —— 与远程 MCP Server 通信
3. 工具发现 (tools/list) —— 动态发现可用工具
4. LangChain 适配器 —— 将 MCP Tool 包装为 LangChain Tool
5. LlamaIndex 适配器 —— 将 MCP Tool 包装为 LlamaIndex Tool

依赖：pip install mcp httpx langchain-mcp-adapters
"""

import os
import json
import asyncio
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path


# ============================================================================
# 第一部分：MCP 客户端基础架构
# ============================================================================

class MCPClientBase:
    """MCP 客户端基类，定义统一的接口。"""

    async def connect(self):
        """建立连接。"""
        raise NotImplementedError

    async def disconnect(self):
        """断开连接。"""
        raise NotImplementedError

    async def list_tools(self) -> List[dict]:
        """发现可用工具。"""
        raise NotImplementedError

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用工具。"""
        raise NotImplementedError

    async def list_resources(self) -> List[dict]:
        """发现可用资源。"""
        raise NotImplementedError

    async def read_resource(self, uri: str) -> str:
        """读取资源。"""
        raise NotImplementedError


# ============================================================================
# 第二部分：stdio 客户端 —— 子进程管理
# ============================================================================

class StdioMCPClient(MCPClientBase):
    """
    stdio 传输的 MCP 客户端。

    负责：
    1. 启动 MCP Server 子进程
    2. 通过 stdin/stdout 进行 JSON-RPC 通信
    3. 管理子进程生命周期

    适用于：Claude Desktop 插件、本地 CLI 工具
    """

    def __init__(self, command: str, args: List[str] = None,
                 env: Dict[str, str] = None):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self._tools_cache: Optional[List[dict]] = None

    async def connect(self):
        """启动子进程并建立连接。"""
        full_env = {**os.environ, **self.env}

        # 启动 MCP Server 子进程
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        print(f"  [StdioClient] 已启动子进程: {self.command} "
              f"(PID: {self.process.pid})")

        # MCP 协议初始化（简化演示）
        # 实际需要: initialize → initialized 握手
        await self._send_json_rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-client", "version": "1.0"},
        })

    async def _send_json_rpc(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        # 发送请求
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()

        # 读取响应（实际应处理 jsonlines 格式）
        response_line = await self.process.stdout.readline()
        if response_line:
            return json.loads(response_line.decode())
        return {"error": "no response"}

    async def list_tools(self) -> List[dict]:
        """发现 MCP Server 提供的所有工具。"""
        print(f"  [StdioClient] 请求 tools/list...")
        # response = await self._send_json_rpc("tools/list", {})
        # return response.get("result", {}).get("tools", [])

        # 模拟返回
        tools = [
            {
                "name": "search_knowledge_base",
                "description": "搜索企业内部知识库",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "top_k": {"type": "integer", "default": 5},
                        "category": {"type": "string", "default": "all"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "create_support_ticket",
                "description": "创建技术支持工单",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    },
                    "required": ["title", "description"],
                },
            },
            {
                "name": "check_system_status",
                "description": "检查系统运行状态",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        self._tools_cache = tools
        return tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用指定的工具。"""
        print(f"  [StdioClient] 调用工具: {tool_name}({arguments})")
        # response = await self._send_json_rpc("tools/call", {
        #     "name": tool_name,
        #     "arguments": arguments,
        # })
        # return response.get("result", {}).get("content", [{}])[0].get("text", "")

        # 模拟返回
        return f"[{tool_name}] 执行结果: {json.dumps(arguments)}"

    async def disconnect(self):
        """终止子进程。"""
        if self.process:
            print(f"  [StdioClient] 终止子进程 (PID: {self.process.pid})")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
            self.process = None


async def demo_stdio_client():
    """stdio 客户端演示。"""
    print("=" * 60)
    print("【1】stdio 客户端 —— 子进程管理")
    print("=" * 60)

    # 创建客户端（连接一个 Python MCP Server）
    client = StdioMCPClient(
        command="python",
        args=["-c", "print('MCP Server running...')"],
        env={"MCP_MODE": "stdio"},
    )

    # 实际使用中：
    # await client.connect()
    # tools = await client.list_tools()
    # result = await client.call_tool("search", {"query": "RAG架构"})
    # await client.disconnect()

    print("\n  [演示模式] stdio 客户端已定义")
    print("  生产环境使用模式:")
    print("""
    async with stdio_client(
        ServerParameters(
            command="python",
            args=["mcp_server.py"],
            env={"API_KEY": "xxx"}
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            tools = await session.list_tools()
            result = await session.call_tool("tool_name", {"arg": "val"})
    """)

    return client


# ============================================================================
# 第三部分：HTTP 客户端
# ============================================================================

class HTTPMCPClient(MCPClientBase):
    """
    HTTP/SSE 传输的 MCP 客户端。

    负责：
    1. 通过 HTTP 与远程 MCP Server 通信
    2. 自动处理认证（Bearer Token / API Key）
    3. 支持工具发现和调用
    """

    def __init__(self, base_url: str,
                 auth_token: str = None,
                 api_key: str = None,
                 timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.api_key = api_key
        self.timeout = timeout
        self._session = None

    async def connect(self):
        """建立 HTTP 连接。"""
        try:
            import httpx
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            if self.api_key:
                headers["X-API-Key"] = self.api_key

            self._session = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
            print(f"  [HTTPClient] 已连接到: {self.base_url}")
        except ImportError:
            print("  [WARN] httpx 未安装，HTTP 客户端不可用")

    async def list_tools(self) -> List[dict]:
        """通过 HTTP 发现工具。"""
        print(f"  [HTTPClient] GET /tools/list...")
        if self._session:
            # resp = await self._session.post("/mcp", json={
            #     "jsonrpc": "2.0", "method": "tools/list", "id": 1
            # })
            # return resp.json()["result"]["tools"]
            pass
        # 模拟返回
        return [
            {"name": "remote_search", "description": "远程搜索"},
            {"name": "remote_analyze", "description": "远程分析"},
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """通过 HTTP 调用工具。"""
        print(f"  [HTTPClient] POST 调用 {tool_name}...")
        if self._session:
            # resp = await self._session.post("/mcp", json={
            #     "jsonrpc": "2.0", "method": "tools/call",
            #     "params": {"name": tool_name, "arguments": arguments},
            #     "id": 1,
            # })
            # return resp.json()["result"]["content"][0]["text"]
            pass
        return f"远程调用 {tool_name} 结果"

    async def disconnect(self):
        """关闭 HTTP 连接。"""
        if self._session:
            await self._session.aclose()
            self._session = None
            print(f"  [HTTPClient] 已断开连接")


async def demo_http_client():
    """HTTP 客户端演示。"""
    print("\n" + "=" * 60)
    print("【2】HTTP 客户端 —— 远程服务调用")
    print("=" * 60)

    client = HTTPMCPClient(
        base_url="https://mcp.example.com",
        auth_token=os.environ.get("MCP_AUTH_TOKEN", "demo-token"),
    )

    print(f"  目标服务: {client.base_url}")
    print(f"  认证方式: Bearer Token")

    # 实际使用：
    # await client.connect()
    # tools = await client.list_tools()
    # result = await client.call_tool("search", {"query": "test"})
    # await client.disconnect()

    print("\n  HTTP 客户端完整请求示例:")
    print("""
    POST /mcp HTTP/1.1
    Host: mcp.example.com
    Authorization: Bearer eyJhbGciOi...
    Content-Type: application/json

    {
      "jsonrpc": "2.0",
      "method": "tools/call",
      "params": {
        "name": "search_knowledge_base",
        "arguments": {"query": "RAG架构", "top_k": 5}
      },
      "id": 1
    }
    """)

    return client


# ============================================================================
# 第四部分：工具发现与动态集成
# ============================================================================

async def demo_tool_discovery():
    """工具发现：动态获取 MCP Server 的所有工具并自动集成。"""
    print("\n" + "=" * 60)
    print("【3】工具发现 —— tools/list 动态集成")
    print("=" * 60)

    # 模拟发现过程
    discovered_tools = [
        {
            "name": "search_knowledge_base",
            "description": "搜索企业内部知识库",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_document_detail",
            "description": "获取文档详细内容",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "文档ID"},
                },
                "required": ["document_id"],
            },
        },
        {
            "name": "check_system_status",
            "description": "检查系统状态",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]

    print(f"  发现 {len(discovered_tools)} 个工具:\n")
    for tool in discovered_tools:
        params = tool["inputSchema"].get("properties", {}).keys()
        required = tool["inputSchema"].get("required", [])
        print(f"  📎 {tool['name']}")
        print(f"     {tool['description']}")
        print(f"     参数: {', '.join(params) if params else '无'}")
        if required:
            print(f"     必填: {', '.join(required)}")
        print()

    # 工具发现后的动态集成代码
    print("  工具发现后的集成模式:")
    code = '''
# 自动将 MCP Tools 转换为可调用函数
class MCPToolWrapper:
    """将 MCP Tool Schema 转换为 Python 可调用对象。"""
    def __init__(self, client, tool_def):
        self.client = client
        self.name = tool_def["name"]
        self.description = tool_def["description"]
        self.schema = tool_def["inputSchema"]

    async def __call__(self, **kwargs):
        # 参数校验
        required = self.schema.get("required", [])
        for param in required:
            if param not in kwargs:
                raise ValueError(f"缺少必填参数: {param}")
        return await self.client.call_tool(self.name, kwargs)

# 自动为 LLM Agent 生成工具列表
def build_tools_for_agent(client):
    tools = []
    for tool_def in await client.list_tools():
        wrapped = MCPToolWrapper(client, tool_def)
        tools.append(wrapped)
    return tools
'''
    print(code)


# ============================================================================
# 第五部分：LangChain 适配器
# ============================================================================

def demo_langchain_adapter():
    """将 MCP Tool 适配为 LangChain Tool。"""
    print("\n" + "=" * 60)
    print("【4】LangChain 适配器 —— MCP Tool → LangChain Tool")
    print("=" * 60)

    code = '''
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# 方案1：使用 langchain-mcp-adapters（推荐）
async def build_agent_with_mcp():
    # 连接多个 MCP Server
    client = MultiServerMCPClient({
        "knowledge_base": {
            "command": "python",
            "args": ["kb_mcp_server.py"],
            "transport": "stdio",
        },
        "analytics": {
            "url": "http://analytics:8000/mcp",
            "transport": "sse",
            "headers": {"Authorization": "Bearer token"},
        },
    })

    # 获取所有工具（自动发现）
    tools = await client.get_tools()

    # 创建 LangChain Agent
    from langchain_openai import ChatOpenAI
    agent = create_agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=tools,
        system_prompt="你是企业助手，可以使用知识库和分析工具。",
    )

    return agent


# 方案2：手动包装
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def create_langchain_tool_from_mcp(
    server_params: StdioServerParameters,
    tool_name: str,
):
    """将单个 MCP Tool 手动包装为 LangChain Tool。"""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            @tool
            async def mcp_tool(**kwargs):
                """从 MCP Server 调用的工具。"""
                result = await session.call_tool(tool_name, kwargs)
                return result.content[0].text

            mcp_tool.name = tool_name
            return mcp_tool
'''
    print(code)


# ============================================================================
# 第六部分：LlamaIndex 适配器
# ============================================================================

def demo_llamaindex_adapter():
    """将 MCP Tool 适配为 LlamaIndex Tool。"""
    print("\n" + "=" * 60)
    print("【5】LlamaIndex 适配器 —— MCP Tool → LlamaIndex Tool")
    print("=" * 60)

    code = '''
from llama_index.core.tools import FunctionTool
from mcp import ClientSession

async def mcp_tool_to_llamaindex(
    session: ClientSession,
    tool_def: dict,
) -> FunctionTool:
    """将 MCP Tool 定义转换为 LlamaIndex FunctionTool。"""

    tool_name = tool_def["name"]
    tool_desc = tool_def["description"]

    async def tool_fn(**kwargs) -> str:
        result = await session.call_tool(tool_name, kwargs)
        return result.content[0].text

    return FunctionTool.from_defaults(
        name=tool_name,
        description=tool_desc,
        fn=tool_fn,
        async_fn=tool_fn,
    )

# 在 LlamaIndex Agent 中使用
async def build_llamaindex_agent_with_mcp():
    from llama_index.core.agent import ReActAgent
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="python",
        args=["kb_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 发现并转换所有工具
            tools_response = await session.list_tools()
            tools = []
            for tool_def in tools_response.tools:
                llama_tool = await mcp_tool_to_llamaindex(session, {
                    "name": tool_def.name,
                    "description": tool_def.description,
                })
                tools.append(llama_tool)

            # 创建 Agent
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                verbose=True,
            )
            return agent
'''
    print(code)


# ============================================================================
# 第七部分：多 MCP Server 编排
# ============================================================================

def demo_multi_server_orchestration():
    """同时连接多个 MCP Server 进行编排。"""
    print("\n" + "=" * 60)
    print("【6】多 MCP Server 编排 —— 统一工具网关")
    print("=" * 60)

    print("""
    多 MCP Server 编排架构：

    ┌─────────────────────────────────────────────┐
    │              MCP Client (Agent)              │
    │                                               │
    │   tools = [                                    │
    │     knowledge_base.search_documents,   ← MCP Server A
    │     knowledge_base.get_document,       ← MCP Server A
    │     analytics.run_query,               ← MCP Server B
    │     analytics.generate_report,         ← MCP Server B
    │     email.send,                         ← MCP Server C
    │   ]                                           │
    └─────────────────────────────────────────────┘

    工具命名空间策略：
    - 每个 MCP Server 的工具加前缀（如 knowledge_base.xxx）
    - 避免工具名冲突
    - 统一认证管理
    - 统一错误处理和重试
    """)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("MCP 客户端集成实战")
    print("=" * 60)

    async def main():
        await demo_stdio_client()
        await demo_http_client()
        await demo_tool_discovery()
        demo_langchain_adapter()
        demo_llamaindex_adapter()
        demo_multi_server_orchestration()

    asyncio.run(main())

    print("\n[完成] MCP 客户端集成演示结束。")
    print("  推荐: 使用 langchain-mcp-adapters 快速集成到 Agent。")
