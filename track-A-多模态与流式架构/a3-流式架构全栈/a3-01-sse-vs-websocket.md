# A3-01: SSE vs WebSocket 协议选型

## 1. 协议对比表

| 特性 | SSE (Server-Sent Events) | WebSocket |
|------|--------------------------|-----------|
| **通信方向** | 单向：服务器→客户端 | 双向：服务器↔客户端 |
| **协议层** | HTTP/1.1 或 HTTP/2 | 独立协议 (ws:// wss://) |
| **连接建立** | 标准HTTP请求，自动重连 | HTTP Upgrade握手 (101 Switching Protocols) |
| **数据格式** | 纯文本 (text/event-stream) | 文本或二进制帧 |
| **浏览器支持** | EventSource API (所有现代浏览器) | WebSocket API (所有现代浏览器) |
| **代理/防火墙** | 完全兼容HTTP代理 | 部分企业代理需要配置 |
| **自动重连** | EventSource内置3秒自动重连 | 需要手动实现 |
| **消息ID** | 原生支持 `id:` 字段（断线续传） | 需应用层实现 |
| **并发连接** | HTTP/2下可多路复用 | 独立连接 |
| **负载均衡** | 标准HTTP LB即可 | 需要支持WebSocket的LB |
| **Cookie/认证** | 自动携带（同域） | 需手动在握手中携带 |
| **Nginx配置** | 无需特殊配置 | 需proxy_set_header Upgrade等 |
| **服务端推送** | 原生支持 | 原生支持 |
| **客户端推送** | 不支持（需额外HTTP请求） | 原生双向 |
| **开销** | 轻量：HTTP头+文本行 | 较重：帧头+心跳 |
| **状态管理** | 无状态（每个event独立） | 有状态（持久连接） |
| **Typical RAG Use** | LLM token流式输出 | Agent对话交互 |
| **最大并发** | 高（HTTP连接池复用） | 中（每连接一个线程） |

## 2. 决策树

```
开始 → 是否需要客户端向服务器发送流式数据？
    ├── 是 → 是否需要低延迟双向交互？
    │   ├── 是 → WebSocket ✓
    │   └── 否 → 评估HTTP/2 Streams or WebSocket
    └── 否 → 主要是服务器推送？
        ├── 是 → 是否需要二进制数据传输？
        │   ├── 是 → WebSocket ✓
        │   └── 否 → 是否需要自动重连+事件ID？
        │       ├── 是 → SSE ✓ (RAG Token Streaming)
        │       └── 否 → 考虑 polling 或 SSE
        └── 否 → 用普通HTTP请求即可
```

## 3. RAG系统中的推荐场景

### SSE 适用场景（推荐用于RAG的LLM输出流式）

```
客户端                        服务器 (FastAPI)
  │                              │
  │  POST /chat/stream           │
  │  {"query": "什么是RAG?"}      │
  │─────────────────────────────>│
  │                              │
  │  event: token                │
  │  data: {"token": "RAG"}      │
  │<─────────────────────────────│
  │  event: token                │
  │  data: {"token": "是"}       │
  │<─────────────────────────────│
  │  event: token                │
  │  data: {"token": "检索"}     │
  │<─────────────────────────────│
  │  ...                         │
  │  event: done                 │
  │  data: {"finish_reason": "stop"}
  │<─────────────────────────────│
  │                              │
  │  (连接自动关闭或保持open)      │
```

优点：
- 天然适合LLM token-by-token输出
- 浏览器EventSource API零依赖
- 断线自动重连（Last-Event-ID恢复上下文）
- Nginx/CDN友好

### WebSocket 适用场景（RAG Agent交互）

```
客户端                        服务器
  │                              │
  │  WS Upgrade                  │
  │─────────────────────────────>│
  │<───────── 101 Switching ─────│
  │                              │
  │  {"type": "user_message",    │
  │   "content": "查询", ...}     │
  │─────────────────────────────>│
  │  {"type": "agent_thought",   │
  │   "content": "思考中..."}     │
  │<─────────────────────────────│
  │  {"type": "tool_call",       │
  │   "tool": "search", ...}     │
  │<─────────────────────────────│
  │  {"type": "tool_result",     │
  │   "data": {...}}             │
  │<─────────────────────────────│
  │  {"type": "token", "data":"..."}
  │<─────────────────────────────│
  │                              │
  │  {"type": "interrupt", ...}  │
  │─────────────────────────────>│
  │  (取消当前生成)               │
```

优点：
- 双向通信：用户可中途打断/修改
- Agent工具调用+结果回传在同一条连接中
- 适合多轮交互式对话

## 4. 实际选择建议

| RAG场景 | 推荐协议 | 理由 |
|---------|---------|------|
| 简单问答（LLM流式输出） | SSE | 轻量，HTTP友好，token-by-token |
| 带检索引用的问答 | SSE | 引用可以与token同流下发 |
| Agent工具调用 | WebSocket | 双向交互，tool_call→tool_result |
| 实时协同编辑 | WebSocket | 低延迟双向同步 |
| 进度条/状态更新 | SSE | 纯推送，简单可靠 |
| 多模态流式（图片+文本） | SSE | 图片可用event.data内嵌或单独HTTP |
| TTS语音流式输出 | WebSocket | 二进制音频帧 |
| 聊天机器人（无工具） | SSE | 最简单部署 |
| 聊天机器人（有工具） | WebSocket | 需要双向 |

## 5. 混合方案

对于复杂RAG系统，可同时使用两种协议：

```
├── SSE端点: /api/chat/stream        # 简单问答
├── SSE端点: /api/progress/{task_id} # 任务进度
└── WebSocket: /ws/agent/chat       # Agent对话
```

## 6. 总结

| 对比维度 | Winner |
|---------|--------|
| 部署复杂度 | SSE |
| 浏览器兼容性 | SSE（老IE除外） |
| 自动重连 | SSE |
| 双向通信 | WebSocket |
| 二进制数据 | WebSocket |
| HTTP生态兼容 | SSE |
| 低延迟 | WebSocket（略优） |

**对于90%的RAG Token流式场景：首选SSE。**
