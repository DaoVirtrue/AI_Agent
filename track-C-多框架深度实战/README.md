# Track C：多框架深度实战

> **对应阶段**：Phase 03 - Phase 07
> **前置要求**：已完成 Track A（RAG 理论基石）与 Track B（LangChain 全栈实战）
> **学习目标**：掌握 LlamaIndex、RAGFlow、Dify、多 Agent 框架、MCP 协议五大技术栈的生产级实战能力

---

## 课程地图

```
Track C：多框架深度实战 (5周)
│
├── C1：LlamaIndex 深度实战 ──────────── Phase 03
│   ├── 数据连接器（30+ LLM 可用数据源）
│   ├── 索引类型对比（Vector/Summary/Tree/Keyword）
│   ├── 查询引擎（Router/SubQuestion/SQL）
│   ├── 高级检索（递归检索/混合检索/重排序）
│   └── vs LangChain：终极对比指南
│
├── C2：RAGFlow 深度实战 ────────────── Phase 04
│   ├── 无切片范式（文档理解驱动的 RAG）
│   ├── 知识图谱增强 RAG（GraphRAG）
│   ├── 部署与 API（Docker/API 封装）
│   └── vs DIY RAG：何时用 RAGFlow
│
├── C3：Dify 低代码实战 ────────────── Phase 05
│   ├── 知识库搭建（拖拽式 RAG）
│   ├── 工作流引擎（条件分支/代码节点/插件）
│   ├── Agent 构建（工具调用+可视化配置）
│   ├── API 与应用集成
│   └── 何时用 Dify，何时自建
│
├── C4：多 Agent 框架实战 ───────────── Phase 06
│   ├── CrewAI 实战（Role/Goal/Backstory/Task/Crew）
│   ├── AutoGen 实战（ConversableAgent/GroupChat/HITL）
│   ├── MetaGPT 实战（SOP 驱动的软件公司模拟）
│   └── 框架对比与选型矩阵
│
└── C5：MCP 协议深度实战 ───────────── Phase 07
    ├── 服务端从零构建（mcp.server.Server + @server.tool()）
    ├── 传输模式对比（stdio vs HTTP/SSE vs Streamable HTTP）
    ├── 安全认证（OAuth/API Key/会话级/工具级权限）
    ├── 客户端集成（子进程管理/HTTP 客户端/工具发现/LangChain 适配）
    └── MCP 生态与生产化指南
```

---

## 学习路线建议

| 角色 | 重点章节 | 可跳过 |
|------|----------|--------|
| **RAG 工程师** | C1 全部 + C2 + C5 | C3（了解即可） |
| **全栈工程师** | C1 + C3 + C4 + C5 | C2（了解即可） |
| **架构师** | 所有总结性 .md 文件 | 代码可略读 |
| **产品经理** | C3 + C4 对比表格 | 代码可跳过 |

---

## 环境准备

```bash
# 所有 C1-C5 的公共依赖
pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
pip install langchain langchain-openai langchain-community
pip install crewai autogen-agentchat
pip install mcp anthropic
pip install ragflow-client  # C2
pip install dify-client     # C3

# 各章特有依赖见各章文件头部注释
```

---

## 每章文件结构约定

- **.py 文件**：完整可运行代码，包含 `if __name__ == "__main__":` 入口块
- **.md 文件**：理论/对比/指南类内容，生产级深度
- **pitfalls/ 目录**：常见陷阱与反模式（留空，待后续填充）

---

## 产出物检查清单

完成 Track C 后，你应当能够：

- [ ] 在 LlamaIndex 中构建包含路由、子问题分解、递归检索的复杂查询引擎
- [ ] 在 RAGFlow 中部署无切片 RAG 与知识图谱增强 RAG
- [ ] 在 Dify 中搭建完整知识库+工作流+Agent 应用并对外提供 API
- [ ] 分别用 CrewAI、AutoGen、MetaGPT 构建多 Agent 协作系统
- [ ] 从零构建 MCP 服务端，并集成到 LangChain/LlamaIndex 客户端
- [ ] 基于业务场景做出框架选型决策

---

## 同步说明

本 README 与以下内容保持同步：
- Phase 03-07 主课程大纲
- Track A/B 前置知识体系
- Track D（企业合规与商业）商业应用场景
- Supplement 01-03（测试/运维/案例）补充资料
