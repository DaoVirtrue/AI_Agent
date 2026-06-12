#!/usr/bin/env python3
"""
C5-01：MCP 服务端从零构建
============================
MCP（Model Context Protocol）是 Anthropic 发布的开放协议，
用于标准化 LLM 与外部工具/数据源的交互。

核心理念：像 USB-C 一样，为 AI 应用提供统一的"接口标准"。

本文件演示：
1. MCP 协议核心概念（Server/Tool/Resource/Prompt）
2. 从零构建 MCP 服务端（mcp.server.Server）
3. 注册工具（@server.tool()）
4. 注册资源（@server.resource()）
5. 注册提示词模板（@server.prompt()）
6. 在 stdio 传输上运行完整服务

依赖：pip install mcp
"""

import os
import json
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path


# ============================================================================
# 第一部分：MCP 协议核心概念
# ============================================================================

def explain_mcp_concepts():
    """MCP 协议的核心概念。"""
    print("=" * 60)
    print("【MCP 协议核心概念】")
    print("=" * 60)

    print("""
    MCP 架构图：

    ┌──────────────────────────────────────────────────┐
    │                  MCP 生态系统                      │
    │                                                    │
    │  ┌──────────────┐         ┌──────────────┐        │
    │  │   Host        │         │   MCP Client │        │
    │  │  (Claude/IDE) │◄───────►│   (应用集成) │        │
    │  └──────┬───────┘         └──────┬───────┘        │
    │         │ MCP Protocol           │ MCP Protocol     │
    │         ▼                        ▼                 │
    │  ┌──────────────────────────────────────┐         │
    │  │          MCP Server                   │         │
    │  │                                       │         │
    │  │  ┌─────────┐ ┌──────────┐ ┌───────┐ │         │
    │  │  │ Tools   │ │ Resources│ │Prompts│ │         │
    │  │  │(可调用) │ │(可读取)  │ │(模板) │ │         │
    │  │  └─────────┘ └──────────┘ └───────┘ │         │
    │  └──────────────────────────────────────┘         │
    └──────────────────────────────────────────────────┘

    三大原语：
    1. Tool —— LLM 可以调用的函数（带参数和返回值的操作）
       模型控制：由 LLM 决定何时调用、用什么参数
       示例：search_documents(), create_ticket(), send_email()

    2. Resource —— LLM 可以读取的数据（类似 GET 请求）
       应用控制：由客户端决定何时读取
       示例：file://documents/project.txt, postgres://users/table

    3. Prompt —— 预定义的提示词模板（可参数化）
       用户控制：由用户选择使用
       示例：代码审查模板、文档生成模板、翻译模板
    """)


# ============================================================================
# 第二部分：创建 MCP 服务端
# ============================================================================

def create_mcp_server():
    """
    从零构建 MCP 服务端。

    这是一个完整的企业内部 API MCP-ification 示例：
    将一个企业知识库 API 包装为 MCP Server。
    """
    print("=" * 60)
    print("【构建 MCP 服务端】—— 企业内部 API MCP-ification")
    print("=" * 60)

    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationCapabilities

    # 创建 Server 实例
    server = Server("enterprise-knowledge-base")

    print(f"  Server 已创建: {server.name}")

    # ---- 注册 Tool ----

    @server.tool()
    async def search_knowledge_base(
        query: str,
        top_k: int = 5,
        category: str = "all",
    ) -> str:
        """
        搜索企业内部知识库。

        支持全文检索 + 语义检索混合，返回最相关的文档片段。

        Args:
            query: 搜索关键词或自然语言问题
            top_k: 返回结果数量（默认5，最大20）
            category: 搜索范围（all/技术文档/产品手册/规章制度）
        """
        # 模拟知识库搜索
        mock_results = [
            {
                "title": "RAG 系统架构设计规范 v2.1",
                "content": "RAG 系统采用分层架构：接入层→业务层→数据层...",
                "score": 0.95,
                "category": "技术文档",
                "updated_at": "2026-05-15",
            },
            {
                "title": "企业AI平台部署指南",
                "content": "部署前请确保：Docker >= 24.0, K8s >= 1.28, GPU Driver >= 535...",
                "score": 0.88,
                "category": "技术文档",
                "updated_at": "2026-06-01",
            },
            {
                "title": "智能客服使用手册",
                "content": "智能客服系统支持多渠道接入，包括网页、微信、API...",
                "score": 0.76,
                "category": "产品手册",
                "updated_at": "2026-04-20",
            },
        ]

        # 过滤分类
        if category != "all":
            filtered = [r for r in mock_results if r["category"] == category]
        else:
            filtered = mock_results

        results = filtered[:top_k]

        # 格式化为 Markdown 返回（方便 LLM 阅读）
        output = f"## 知识库搜索结果\n\n**查询**: {query}\n**结果数**: {len(results)}\n\n"
        for i, r in enumerate(results, 1):
            output += f"### {i}. {r['title']}\n"
            output += f"- **相关性**: {r['score']:.0%}\n"
            output += f"- **分类**: {r['category']}\n"
            output += f"- **更新**: {r['updated_at']}\n"
            output += f"- **内容**: {r['content']}\n\n"

        return output

    # ---- 注册更多 Tool ----

    @server.tool()
    async def get_document_detail(document_id: str) -> str:
        """
        获取文档的详细内容。

        当 search_knowledge_base 返回的摘要不够时，
        使用此工具获取文档的完整内容。

        Args:
            document_id: 文档ID（从 search_knowledge_base 的结果中获取）
        """
        mock_documents = {
            "doc_001": {
                "title": "RAG 系统架构设计规范 v2.1",
                "author": "技术架构组",
                "content": "全文内容...（此处省略完整文档内容）\n\n"
                          "第一章：总体架构\nRAG系统由以下核心组件构成：\n"
                          "1. 文档处理层 2. 嵌入向量层 3. 检索层 4. 生成层\n\n"
                          "第二章：性能要求\nP99延迟<2秒，支持1000QPS...",
                "version": "2.1",
            },
        }
        doc = mock_documents.get(document_id, {"title": "未知文档", "content": "文档不存在"})
        return f"# {doc['title']}\n\n{doc['content']}"

    @server.tool()
    async def create_support_ticket(
        title: str,
        description: str,
        priority: str = "medium",
        assignee: str = "",
    ) -> str:
        """
        创建技术支持工单。

        Args:
            title: 工单标题（简短描述问题）
            description: 详细问题描述
            priority: 优先级（low/medium/high/critical）
            assignee: 指派人（可选，留空则自动分配）
        """
        ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return json.dumps({
            "ticket_id": ticket_id,
            "status": "created",
            "title": title,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "message": f"工单 {ticket_id} 已创建，优先级 {priority}。"
                       f"预计响应时间: {'5分钟' if priority == 'critical' else '1小时'}",
        }, ensure_ascii=False, indent=2)

    @server.tool()
    async def check_system_status() -> str:
        """检查系统各组件的运行状态。"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "knowledge_base": {"status": "healthy", "documents": 15234},
                "vector_db": {"status": "healthy", "collections": 12},
                "llm_service": {"status": "healthy", "model": "gpt-4o"},
                "search_engine": {"status": "healthy", "index_size": "2.4GB"},
                "api_gateway": {"status": "healthy", "qps": 342},
            },
            "overall": "healthy",
        }
        return json.dumps(status, ensure_ascii=False, indent=2)

    @server.tool()
    async def summarize_document(document_id: str, max_length: int = 200) -> str:
        """
        对文档进行智能摘要。

        Args:
            document_id: 要摘要的文档ID
            max_length: 摘要最大长度（字符数）
        """
        # 模拟 LLM 摘要
        return f"[摘要] 文档 {document_id} 的主要内容涉及..."
               f"(这是一个 {max_length} 字以内的摘要)"

    # ---- 注册 Resource ----

    @server.resource("kb://statistics")
    async def get_kb_statistics() -> str:
        """知识库统计信息（Resource：应用端主动拉取）。"""
        return json.dumps({
            "total_documents": 15234,
            "total_categories": 8,
            "last_updated": "2026-06-11T10:30:00Z",
            "storage_size": "5.2GB",
            "index_type": "HNSW",
        })

    @server.resource("kb://categories")
    async def get_categories() -> str:
        """知识库分类列表。"""
        return json.dumps([
            "技术文档", "产品手册", "规章制度",
            "运维指南", "API文档", "FAQ",
            "培训材料", "竞品分析",
        ], ensure_ascii=False)

    # ---- 注册 Prompt ----

    @server.prompt()
    async def code_review_prompt(code: str, language: str = "python") -> str:
        """代码审查提示词模板。"""
        return f"""请审查以下 {language} 代码，从以下维度评估：
1. 代码质量（可读性、命名规范）
2. 安全性（注入攻击、权限检查）
3. 性能（算法复杂度、资源使用）
4. 错误处理（异常捕获、边界条件）

代码：
```{language}
{code}
```

请给出具体的改进建议。"""

    @server.prompt()
    async def document_writer_prompt(topic: str, doc_type: str = "技术方案") -> str:
        """文档撰写提示词模板。"""
        return f"""请撰写一份关于"{topic}"的{doc_type}文档。

文档结构要求：
1. 概述（背景、目标、范围）
2. 详细内容（分章节论述）
3. 实施建议
4. 风险与注意事项

请使用专业、清晰的语言。"""

    print(f"  已注册 Tool: 5 个")
    print(f"  已注册 Resource: 2 个")
    print(f"  已注册 Prompt: 2 个")

    return server


# ============================================================================
# 第三部分：运行 MCP 服务端
# ============================================================================

async def run_mcp_server(server: Server):
    """
    启动 MCP 服务端（stdio 传输）。

    生产环境运行方式：
        python c5-01-mcp服务端从零构建.py

    然后在 Claude Desktop 配置中添加：
    {
        "mcpServers": {
            "enterprise-kb": {
                "command": "python",
                "args": ["c5-01-mcp服务端从零构建.py"]
            }
        }
    }
    """
    print("\n" + "=" * 60)
    print("【启动 MCP 服务端】—— stdio 传输模式")
    print("=" * 60)

    from mcp.server.stdio import stdio_server

    print("  MCP 服务端已就绪，等待客户端连接...")
    print("  传输模式: stdio (标准输入输出)")
    print("  按 Ctrl+C 停止服务")

    # 实际启动代码（在真实环境中取消注释）：
    # async with stdio_server() as (read_stream, write_stream):
    #     await server.run(
    #         read_stream,
    #         write_stream,
    #         InitializationCapabilities(
    #             sampling=...,
    #             logging=...,
    #         ),
    #     )

    print("\n  [演示模式] 服务端已定义但未启动真实传输。")
    print("  要启动真实服务，请运行本文件。")
    print("  然后在 Claude Desktop 中配置 MCP Server。")


# ============================================================================
# 第四部分：MCP 服务端最佳实践
# ============================================================================

def print_mcp_server_best_practices():
    """MCP 服务端开发最佳实践。"""
    print("\n" + "=" * 60)
    print("【MCP 服务端开发最佳实践】")
    print("=" * 60)

    tips = """
    1. 【Tool 设计】
       - 函数名即工具名，要简洁明了（如 search_documents）
       - docstring 要详细（MCP 用它生成 tool description）
       - 参数类型注解要完整（MCP 用它生成 JSON Schema）
       - 返回值用 Markdown 格式（方便 LLM 渲染）
       - 错误时返回友好的错误信息（不要抛异常）

    2. 【Tool vs Resource 的选择】
       - 需要 LLM 主动决策何时调用 → Tool
       - 需要客户端按需获取数据 → Resource
       - Tool 可以有副作用（创建、修改），Resource 应该是只读的

    3. 【安全性】
       - 输入参数要校验（防止注入攻击）
       - 敏感操作（删除/修改）要二次确认
       - API Key 等敏感信息不要硬编码
       - 设置超时时间，防止 LLM 无限循环调用
       - 限制 Tool 调用频率（rate limiting）

    4. 【性能】
       - Tool 函数应该是异步的（async def）
       - 耗时操作要有进度反馈
       - 结果可以分页返回（limit + offset）
       - 使用缓存减少重复调用

    5. 【测试】
       - 用 MCP Inspector 调试（npx @modelcontextprotocol/inspector）
       - 编写 Tool 调用的单元测试
       - 模拟客户端进行集成测试

    6. 【部署】
       - stdio 模式适合本地工具（低延迟、无网络开销）
       - HTTP/SSE 模式适合远程工具（C5-02 详细讲解）
       - 生产环境用容器化部署
       - 配置健康检查和优雅关闭
    """
    print(tips)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("MCP 服务端从零构建实战")
    print("=" * 60)

    explain_mcp_concepts()
    server = create_mcp_server()

    # 演示 Tool 调用（无需启动真实传输）
    print("\n" + "=" * 60)
    print("【本地演示 Tool 调用】")
    print("=" * 60)

    async def demo_tools():
        """本地演示 Tool 函数调用（不通过 MCP 协议）。"""
        # 模拟 search_knowledge_base
        result = await search_knowledge_base("RAG 架构", top_k=2)
        print(f"  search_knowledge_base('RAG 架构', top_k=2):")
        print(f"  {result[:200]}...")

        # 模拟 check_system_status
        result = await check_system_status()
        print(f"\n  check_system_status():")
        print(f"  {result[:200]}...")

        # 模拟 create_support_ticket
        result = await create_support_ticket(
            title="知识库检索超时",
            description="查询响应时间超过5秒",
            priority="high",
        )
        print(f"\n  create_support_ticket(...):")
        print(f"  {result}")

    asyncio.run(demo_tools())

    print_mcp_server_best_practices()

    print("\n[完成] MCP 服务端构建演示结束。")
    print("  要启动真实服务: python c5-01-mcp服务端从零构建.py")
    print("  调试工具: npx @modelcontextprotocol/inspector")
