# Phase 06: Agent 基础

**时长**: 2 周  
**级别**: 中级 → 高级  
**前置条件**: RAG 基础、Prompt Engineering、Python 异步编程基础

---

## 学习目标

通过本阶段学习，你将能够：

1. **理解 Agent 控制循环**：掌握 Agent = LLM + Tools + Memory + Control Loop 的核心架构，理解感知→规划→行动→反思的完整循环
2. **从零实现 ReAct 模式**：不依赖任何框架，用纯 Python 实现完整的 ReAct（Reasoning + Acting）Agent，深入理解 Thought→Action→Observation 循环
3. **精通工具执行机制**：掌握 JSON Schema 设计原则、函数路由策略、并行工具调用、工具返回结果处理等核心技能
4. **构建记忆系统**：理解短期记忆（scratchpad）、工作记忆（context window）和长期记忆（向量数据库）的分层架构
5. **使用 LangChain Agent Builder**：掌握 LangChain 的 Agent 构建工具，能够快速搭建生产级 Agent 应用

---

## 文件说明

| 序号 | 文件名 | 类型 | 内容概要 |
|------|--------|------|----------|
| 1 | `01-agent-control-loop.ipynb` | Notebook | Agent 核心架构：BaseAgent 类设计、AgentState 状态机、Perception→Planning→Action→Reflection 主循环、状态转移可视化 |
| 2 | `02-react-agent-from-scratch.py` | Python 脚本 | **从零实现 ReAct**：完整的 ReActAgent 类、MockLLM 模拟推理、5 个可执行工具（计算器、搜索、天气、数据库、文件读取）、3 个演示任务、CLI 接口 |
| 3 | `03-function-calling-deep-dive.ipynb` | Notebook | **函数调用深潜**：JSON Schema 设计原则、函数路由策略对比、并行函数调用与依赖检测、工具结果处理（错误、重试、截断） |
| 4 | `pitfalls/01-infinite-loop.md` | 陷阱分析 | **无限循环陷阱**：症状、根因、4 种解决方案（硬限制、Prompt 工程、反思检查、步骤预算）、完整修复代码 |
| 5 | `pitfalls/02-tool-call-failure.md` | 陷阱分析 | **工具调用失败陷阱**：异常类型分析、4 种恢复策略（结构化错误、Agent 级错误处理、参数预校验、回退工具）、完整代码示例 |
| 6 | `pitfalls/03-wrong-tool-selection.md` | 陷阱分析 | **工具选择错误陷阱**：歧义描述问题、4 种改进方案（描述优化、预路由分类、置信度阈值、输出类型验证）、完整对比代码 |
| 7 | `README.md` | 文档 | 本文件：阶段概览、学习路径、评估标准 |

---

## 四种 Agent 循环模式预览

### 1. ReAct（Thought → Action → Observation → Thought 循环）

```
Thought: 我需要知道巴黎的天气，然后根据天气建议穿着
Action: get_weather[city=Paris]
Observation: 巴黎今天晴天，22°C
Thought: 天气信息已获取，现在给出穿着建议
Final Answer: 巴黎今天22°C晴天，建议穿薄外套...
```

**特点**：
- **迭代次数**：中等（2-6 轮）
- **可靠性**：高（每步人类可理解、可调试）
- **延迟**：中等（串行 thought-action 交替）
- **适用场景**：需要多步推理的问答、信息检索、需要工具辅助的对话

### 2. Plan-Execute（Plan → Execute → Evaluate → Replan 循环）

```
Plan: [搜索巴黎天气, 搜索巴黎景点, 搜索巴黎交通]
Execute: get_weather[city=Paris] → 22°C晴天
Evaluate: 天气信息已获取，继续执行
Execute: web_search[query=巴黎景点] → {...}
Evaluate: 所有信息收集完毕
Answer: 综合天气、景点、交通信息...
```

**特点**：
- **迭代次数**：少（1-2 轮规划 + N 轮执行）
- **可靠性**：高（计划可预测、可审查）
- **延迟**：较低（同批工具可并行执行）
- **适用场景**：多步骤数据收集、报告生成、有明确子任务的任务

### 3. Reflection（Act → Evaluate → Reflect → Refine 循环）

```
Act: 生成初始回答
Evaluate: 回答缺少具体数据支撑
Reflect: 我应该在回答前先查询数据
Refine: [修改后的回答，包含具体数据]
Evaluate: 回答质量达标
```

**特点**：
- **迭代次数**：较高（3-8 轮自我修正）
- **可靠性**：很高（自我检查机制）
- **延迟**：高（多轮自我修正）
- **适用场景**：代码生成、内容创作、需要高质量输出的任务

### 4. ReWOO（Plan → Worker → Solver 一步到位）

```
Planner: 需要{巴黎天气, 巴黎景点, 巴黎交通} → 将工具调用分配给 Workers
Worker1: get_weather[city=Paris] → 22°C
Worker2: web_search[query=巴黎景点] → {...}
Worker3: web_search[query=巴黎交通] → {...}
Solver: 综合所有 Worker 结果，一步生成最终回答
```

**特点**：
- **迭代次数**：最少（1 轮规划 + 1 轮并行执行 + 1 轮求解）
- **可靠性**：中等（并行执行无依赖时正确，有依赖时可能出错）
- **延迟**：最低（工具调用完全并行）
- **适用场景**：独立子任务、大规模信息收集、对延迟敏感的场景

### 四种模式对比总结

| 维度 | ReAct | Plan-Execute | Reflection | ReWOO |
|------|-------|-------------|------------|-------|
| 迭代次数 | 中等 | 少 | 高 | 最少 |
| 可靠性 | 高 | 高 | 很高 | 中等 |
| 延迟 | 中等 | 较低 | 高 | 最低 |
| 可调试性 | 优秀 | 优秀 | 良好 | 一般 |
| 并行能力 | 无 | 部分 | 无 | 完全 |
| 典型场景 | 问答推理 | 多步任务 | 内容创作 | 信息收集 |

### 选择指南

- **需要可解释性** → ReAct（每步可见推理过程）
- **任务步骤相互独立** → ReWOO（并行最大化效率）
- **输出质量至关重要** → Reflection（自我修正打磨）
- **步骤之间有依赖关系** → Plan-Execute（先规划再执行）
- **简单工具调用** → 直接用 Function Calling，不需要 Agent 循环

---

## 环境准备

### 前置条件

- Python 3.10+
- 基本的 LangChain 知识
- 了解异步编程（asyncio）

### 安装依赖

```bash
# 核心依赖
pip install langchain langchain-openai langchain-community

# OpenAI SDK
pip install openai

# 向量数据库（用于记忆系统）
pip install chromadb

# Jupyter（运行 notebook）
pip install jupyter

# 可选：更多工具集成
pip install wikipedia duckduckgo-search arxiv
```

### 环境变量

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key"

# Windows
set OPENAI_API_KEY=your-api-key

# 或在代码中设置（不推荐用于生产）
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

---

## 学习路径

建议按以下顺序学习（总时长约 14 天）：

### 第 1 天：Agent 架构概览
- 阅读本 README 完整内容
- 理解四种 Agent 循环模式
- 运行 `01-agent-control-loop.ipynb`，观察控制循环

### 第 2-3 天：从零实现 ReAct
- 逐行阅读 `02-react-agent-from-scratch.py`
- 理解 ReAct prompt 结构
- 理解 Action 解析正则
- 运行 3 个 Demo，修改任务测试

### 第 4-5 天：工具调用深入
- 学习 `03-function-calling-deep-dive.ipynb` Part 1：Schema 设计
- 实现 Part 2：路由策略对比
- 实践 Part 3：并行调用
- 掌握 Part 4：错误处理模式

### 第 6-7 天：陷阱分析
- 阅读 3 个 pitfalls 文档
- 在自己的代码中复现陷阱
- 实际应用解决方案
- 编写总结：避免这些陷阱的最佳实践

### 第 8-14 天：实战项目
- 选一个真实场景（客服、数据分析、代码助手等）
- 设计 Agent 架构
- 实现并测试
- 对比不同 Agent 模式的效果
- 编写项目文档

---

## 评估标准

完成本阶段后，你应该能够：

### 理论（40%）
- [ ] 画出 Agent 控制循环的状态转移图
- [ ] 解释 ReAct 中 Thought/Action/Observation 的关系
- [ ] 比较四种 Agent 模式的优劣
- [ ] 说明记忆在 Agent 中的作用

### 实践（40%）
- [ ] 从零实现一个 ReAct Agent（不用框架）
- [ ] 设计 3 个以上的工具 JSON Schema
- [ ] 实现带错误处理的工具执行器
- [ ] 处理工具调用的依赖关系和并行执行
- [ ] 识别并修复无限循环问题

### 项目（20%）
- [ ] 完成一个端到端的 Agent 应用
- [ ] Agent 能正确处理 80% 以上的测试用例
- [ ] 代码包含完善的错误处理和日志
- [ ] 有运行日志证明 Agent 的推理过程

---

## 参考资源

- [ReAct Paper](https://arxiv.org/abs/2210.03629) - ReAct: Synergizing Reasoning and Acting in Language Models
- [Plan-and-Execute Paper](https://arxiv.org/abs/2305.04091) - Plan-and-Solve Prompting
- [Reflexion Paper](https://arxiv.org/abs/2303.11366) - Reflexion: Language Agents with Verbal Reinforcement Learning
- [ReWOO Paper](https://arxiv.org/abs/2305.18323) - ReWOO: Decoupling Reasoning from Observations
- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

---

> **下一步**: 打开 `01-agent-control-loop.ipynb` 开始学习 Agent 的核心架构。
