# Phase 01：环境与基础

> **预计时间**：1 周 | **难度**：⭐ | **前置要求**：基本的 Python 语法知识

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 使用 **async/await** 编写异步 Python 代码（后续所有 LLM 调用的基础）
2. 掌握 **TypedDict、Pydantic、dataclasses**（LangGraph State 和结构化输出的基础）
3. 调用 **OpenAI 和 Anthropic API**，处理流式输出、错误重试、Token 计数
4. 在本地部署 **Ollama** 并运行开源模型（qwen2.5、llama3）
5. 使用 **pydantic-settings** 管理环境配置

---

## 📂 文件清单

| 文件 | 类型 | 内容 |
|------|------|------|
| `01-python-essentials.ipynb` | Notebook | async/await、TypedDict、Pydantic、dataclasses |
| `02-llm-api-basics.ipynb` | Notebook | OpenAI/Anthropic API 调用、Token、流式、错误处理 |
| `03-local-model-setup.ipynb` | Notebook | Ollama 安装、模型拉取、GGUF 基础 |
| `04-environment-config.py` | Python 脚本 | pydantic-settings、.env 管理、API Key 轮换 |
| `pitfalls/` | 目录 | 4 个坑点分析 + 解决方案 |
| `checkpoint/` | 目录 | 10 道练习题 + 参考解答 |

---

## 🔑 核心概念

### 为什么从这些开始？

```
Phase 01 是所有后续阶段的基石：
├── async/await ──────→ Phase 05 LCEL 异步流式
├── Pydantic ─────────→ Phase 02 结构化输出
├── TypedDict ────────→ Phase 08 LangGraph State
├── API 调用 ─────────→ 全部阶段
├── Token 计数 ───────→ Phase 02 上下文窗口管理
└── 环境配置 ─────────→ Phase 09 生产部署
```

### 模型选择决策（API vs 本地）

| 维度 | API（OpenAI/Anthropic） | 本地（Ollama） |
|------|------------------------|----------------|
| 推理质量 | ★★★★★ | ★★★~★★★★ |
| Function Calling | 原生支持，稳定 | 需适配，不稳定 |
| 延迟 | 网络 RTT + 推理 | 仅推理（视 GPU 而定） |
| 成本 | 按 Token 付费 | 固定硬件成本 |
| 数据隐私 | 数据离开本机 | 完全本地 |
| 运维 | 零运维 | 需 GPU 运维 |

> 💡 **建议**：学习阶段用 API（快速迭代），生产环境敏感数据用本地模型。

---

## ⚠️ 本阶段坑点预览

| # | 坑点 | 根因 | 后果 |
|---|------|------|------|
| 1 | API Key 泄露 | 误提交 `.env` 到 Git | 密钥被盗，账单飙升 |
| 2 | Token 超限 | 不了解模型上下文窗口 | 响应被截断 |
| 3 | 429 限流 | 请求过快 | 服务拒绝 |
| 4 | 流式 JSON 不完整 | 流式响应分片截断 JSON | 解析失败 |

---

## ✅ 检验标准

完成 `checkpoint/exercises.md` 中的 10 道练习，确保：

- [ ] 能正确调用 OpenAI 和 Anthropic API
- [ ] 能用 tiktoken 精确计算 Token 数
- [ ] 能实现指数退避重试
- [ ] 能在本地 Ollama 运行模型并对话
- [ ] 能使用 pydantic-settings 管理配置

---

## 🚀 快速开始

```bash
# 1. 激活虚拟环境
venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install openai anthropic python-dotenv pydantic pydantic-settings tiktoken httpx

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 4. 启动 Jupyter
jupyter notebook

# 5. 打开 01-python-essentials.ipynb 开始学习！
```

---

准备好了吗？打开 `01-python-essentials.ipynb` 开始你的 AI 之旅！🚀
