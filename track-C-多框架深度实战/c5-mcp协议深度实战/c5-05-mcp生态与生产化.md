# C5-05：MCP 生态与生产化指南

> **适用读者**：需要将 MCP 落地到生产环境的架构师和平台工程师
> **前置阅读**：C5-01 至 C5-04

---

## 一、现有 MCP Server 生态目录

### 1.1 官方与主流 MCP Server

| MCP Server | 提供者 | 功能 | 安装方式 |
|-----------|--------|------|----------|
| **Filesystem** | Anthropic | 安全的文件系统读写 | `npx @modelcontextprotocol/server-filesystem` |
| **GitHub** | Anthropic | 仓库管理、PR、Issues | `npx @modelcontextprotocol/server-github` |
| **PostgreSQL** | Anthropic | 数据库查询（只读） | `npx @modelcontextprotocol/server-postgres` |
| **SQLite** | Anthropic | SQLite 数据库操作 | `npx @modelcontextprotocol/server-sqlite` |
| **Brave Search** | Anthropic | 网页搜索 | `npx @modelcontextprotocol/server-brave-search` |
| **Puppeteer** | Anthropic | 浏览器自动化 | `npx @modelcontextprotocol/server-puppeteer` |
| **Slack** | Community | Slack 消息管理 | `npx @modelcontextprotocol/server-slack` |
| **GitLab** | Community | GitLab API 操作 | `npx @modelcontextprotocol/server-gitlab` |
| **Jira** | Community | Jira Issue 管理 | `npx @modelcontextprotocol/server-jira` |
| **Obsidian** | Community | Obsidian 笔记管理 | `npx mcp-obsidian` |

### 1.2 Python SDK 生态

```bash
# MCP Python SDK 生态
pip install mcp                    # 核心 SDK
pip install langchain-mcp-adapters # LangChain 适配器
pip install mcp-server-fastmcp     # FastMCP 简化开发
```

### 1.3 LlamaIndex 集成

```python
# LlamaIndex 内置了 MCP 集成
from llama_index.tools.mcp import McpToolSpec

# 从 MCP Server 自动发现和加载工具
mcp_spec = McpToolSpec(
    command="python",
    args=["kb_mcp_server.py"],
)
tools = mcp_spec.to_tool_list()
```

---

## 二、内部 API MCP-ification 指南

### 2.1 什么 API 适合 MCP-化

```
评估标准：

✅ 适合 MCP-化：
  - 只读查询（搜索、获取、统计）→ Tool
  - 数据资源（文档、配置、数据库表）→ Resource
  - 有结构化输入输出的操作（创建工单、发送通知）→ Tool
  - 需要 LLM 根据上下文自动决策调用的 API

❌ 不适合 MCP-化：
  - 实时流式数据（Kafka/Flink）
  - 需要复杂 UI 交互的操作
  - 批量数据处理（更适合 Spark/Flink Job）
  - 超大文件传输（>100MB）
  - 需要长时间运行的异步任务（>5分钟）
```

### 2.2 MCP-ification 步骤

```
第1步：API 盘点
  ├── 列出所有内部 API
  ├── 分类：查询类 / 操作类 / 资源类
  └── 优先级排序：AI 最需要什么能力

第2步：设计 Tool/Resource/Prompt
  ├── 每个 API 端点 → 对应一个 Tool 或 Resource
  ├── Tool 描述要详细（LLM 据此决定何时调用）
  ├── 参数设计要简单（1-3 个参数最佳）
  └── 返回格式用 Markdown（方便 LLM 理解）

第3步：实现 MCP Server
  ├── 用 mcp.server.Server 框架
  ├── 添加认证层（OAuth2 / API Key）
  ├── 添加速率限制
  ├── 添加审计日志
  └── 添加错误处理（友好错误信息）

第4步：测试
  ├── 用 MCP Inspector 调试
  ├── 编写单元测试
  ├── 集成测试（与真实 LLM 测试）
  └── 性能测试（并发、延迟）

第5步：部署
  ├── stdio 模式：本地工具
  ├── HTTP/SSE 模式：远程服务
  ├── 容器化（Docker）
  └── 配置监控和告警

第6步：注册到 AI 平台
  ├── Claude Desktop 配置
  ├── 企业内部 AI 中台注册
  └── 文档和示例
```

### 2.3 MCP Server 模板

```python
# 内部 API MCP-ification 标准模板

from mcp.server import Server
from mcp.server.stdio import stdio_server
import asyncio

# ===== 第1部分：定义 Server =====
server = Server("internal-api-bridge")

# ===== 第2部分：注册 Tools =====
@server.tool()
async def search_products(query: str, category: str = "all") -> str:
    """搜索产品信息。当用户询问产品相关问题时的首选工具。"""
    # 调用内部 API
    result = await internal_api.search_products(query, category)
    # 格式化为 Markdown
    return format_as_markdown(result)

@server.tool()
async def get_order_status(order_id: str) -> str:
    """查询订单状态。需要提供订单号。"""
    result = await internal_api.get_order(order_id)
    return format_as_markdown(result)

# ===== 第3部分：注册 Resources =====
@server.resource("products://categories")
async def get_product_categories() -> str:
    """产品分类列表。"""
    return await internal_api.get_categories()

# ===== 第4部分：注册 Prompts =====
@server.prompt()
async def product_analysis_prompt(product_id: str) -> str:
    """产品分析提示词模板。"""
    return f"请分析产品 {product_id} 的市场表现、用户评价和改进建议。"

# ===== 第5部分：启动 =====
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 三、生产化清单

### 3.1 基础设施

| 组件 | 方案 | 说明 |
|------|------|------|
| 容器化 | Docker + K8s | 统一部署和扩缩容 |
| API 网关 | Kong / Traefik / Nginx | 统一入口、限流、认证 |
| 服务发现 | Consul / K8s DNS | MCP Server 注册与发现 |
| 配置管理 | 环境变量 / ConfigMap / Vault | 敏感信息保护 |
| 日志 | ELK / Loki | 集中日志管理 |
| 监控 | Prometheus + Grafana | 健康检查、延迟、错误率 |
| 追踪 | Jaeger / OpenTelemetry | 分布式调用链追踪 |

### 3.2 可靠性

```yaml
高可用策略：
  - 多副本部署（至少 2 副本）
  - 健康检查 + 自动重启
  - 优雅关闭（drain connections before shutdown）
  - 超时控制（请求级 + 会话级）
  - 断路器（circuit breaker on internal API failure）
  - 降级策略（graceful degradation）

容灾策略：
  - 跨可用区部署
  - 会话状态外部化（Redis）
  - 定期备份配置和元数据
```

### 3.3 安全加固

```yaml
安全检查清单：
  - [ ] TLS 1.3 加密所有传输
  - [ ] API 认证（OAuth2 / API Key / mTLS）
  - [ ] 工具级授权（RBAC）
  - [ ] 输入参数校验（防注入）
  - [ ] 速率限制（per user / per IP）
  - [ ] 审计日志（全量记录）
  - [ ] 敏感数据脱敏（日志中自动脱敏）
  - [ ] 定期安全扫描（依赖漏洞检查）
  - [ ] 渗透测试
```

### 3.4 可观测性

```python
# MCP Server 关键监控指标
metrics = {
    "mcp_server_uptime": "服务运行时长",
    "mcp_tools_count": "注册的工具数量",
    "mcp_connections_active": "活跃连接数",
    "mcp_tool_calls_total": "工具调用总数（按工具名分组）",
    "mcp_tool_call_duration_seconds": "工具调用延迟（P50/P95/P99）",
    "mcp_tool_call_errors_total": "工具调用错误数",
    "mcp_auth_failures_total": "认证失败次数",
    "mcp_rate_limited_total": "被限流的请求数",
}

# 告警规则
alerts = {
    "high_error_rate": "错误率 > 1% → P1 告警",
    "high_latency": "P99 延迟 > 3s → P2 告警",
    "connection_spike": "活跃连接突增 200% → P3 告警",
    "auth_attack": "认证失败 > 50次/分钟 → P1 告警",
}
```

---

## 四、企业级 MCP 平台架构

```
┌─────────────────────────────────────────────────────────┐
│                   企业 MCP 平台                           │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │              MCP 网关 (Gateway)               │       │
│  │  - 统一认证（OAuth2 SSO）                     │       │
│  │  - 限流/配额管理                              │       │
│  │  - 请求路由                                   │       │
│  │  - 协议转换（HTTP→MCP）                       │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│    ┌────────────┼────────────┐                          │
│    ▼            ▼            ▼                          │
│  ┌────────┐ ┌────────┐ ┌────────┐                       │
│  │MCP     │ │MCP     │ │MCP     │                       │
│  │Server A│ │Server B│ │Server C│  ... (N个服务)         │
│  └────────┘ └────────┘ └────────┘                       │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │              MCP 注册中心                     │       │
│  │  - 服务注册/发现                              │       │
│  │  - 工具目录（Tools Catalog）                  │       │
│  │  - 版本管理                                   │       │
│  └──────────────────────────────────────────────┘       │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │              MCP 管理后台                     │       │
│  │  - 工具审批（上线/下线）                      │       │
│  │  - 权限配置                                   │       │
│  │  - 使用统计                                   │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## 五、MCP 未来展望（2026+）

```
当前趋势：
  1. Auth 标准化：MCP Auth 扩展（RFC）
  2. Streamable HTTP：取代 HTTP/SSE 成为标准远程传输
  3. MCP Marketplace：工具市场和共享生态
  4. 多模态扩展：图片、音频、视频 Tool
  5. Agent-to-Agent Protocol：多个 Agent 之间通过 MCP 通信
  6. 企业级特性：多租户、审计、合规

给架构师的建议：
  - 现在就投入 MCP，它是 AI 应用的标准协议
  - 设计内部 API 时就考虑 MCP-ification
  - 建立企业 MCP 工具目录和管理规范
  - 关注 MCP Auth 和 Streamable HTTP 的标准化进展
```

---

## 六、MCP 资源链接

```
官方资源：
  - 规范: https://spec.modelcontextprotocol.io
  - GitHub: https://github.com/modelcontextprotocol
  - Python SDK: https://github.com/modelcontextprotocol/python-sdk
  - TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk

学习资源：
  - 官方文档: https://modelcontextprotocol.io
  - MCP Inspector: npx @modelcontextprotocol/inspector
  - 示例仓库: https://github.com/modelcontextprotocol/servers

社区：
  - Discord: https://discord.gg/modelcontextprotocol
  - Awesome MCP: https://github.com/punkpeye/awesome-mcp-servers
```
