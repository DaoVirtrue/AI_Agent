# 🧠 Agent 超长上下文与记忆体系

## 模块概述

本模块深入探讨 AI Agent 在超长上下文场景下的记忆管理、状态追踪、对话流程控制以及跨 Agent 记忆共享等核心话题。随着大语言模型上下文窗口从 4K 扩展到 128K 甚至 1M tokens，如何高效组织和利用上下文资源、如何在长对话中保持记忆一致性、以及如何实现持久化记忆存储，成为了构建可靠 Agent 系统的关键挑战。

本模块将从**有限状态机驱动的对话流程**开始，逐步深入到**记忆压缩与摘要技术**、**遗忘曲线建模与记忆巩固**、**超长上下文窗口编排策略**，最终构建一个完整的**跨 Agent 记忆路由与持久化存储体系**。

## 前置知识

在开始本模块之前，你应该已经完成以下 Phase 的学习：

| Phase | 内容 | 与本模块的关联 |
|-------|------|---------------|
| **Phase 02** | 向量检索与语义搜索 | 记忆存储的底层检索引擎 |
| **Phase 04** | Agent 工具调用与编排 | Agent 基础架构与工具集成 |
| **Phase 06** | 多 Agent 协作与通信 | 跨 Agent 记忆路由与联邦查询 |
| **Phase 07** | 对话管理与用户交互 | 对话状态机与槽位填充的基础 |
| **Phase 08** | 安全与隐私保护 | PII 检测、记忆隐私与审计日志 |

## 子模块结构（6 个子模块）

### 子模块 1：对话状态管理与槽位填充（文件 01-03）
- **01-dialog-fsm-core.py** — 对话有限状态机核心，8 种状态定义，状态转移管理，ASCII 状态图
- **02-slot-filling-fsm.py** — 基于 FSM 的槽位填充系统，支持必填/可选槽位，验证器，默认值
- **03-task-oriented-dialogue.py** — 面向任务的对话 Agent 集成，模拟餐厅预订完整流程

### 子模块 2：记忆压缩与上下文编排（文件 04-07）
- **04-llm-summarization-compressor.py** — 三种摘要风格（层级/增量/结构化），递归摘要压缩
- **05-memory-consolidation-engine.py** — 艾宾浩斯遗忘曲线建模，STM→LTM 分级迁移，睡眠巩固模拟
- **06-ultra-long-context-strategies.py** — 结构化上下文窗口（5 区预算分配），KV-Cache 复用，注意力预算调度
- **07-context-window-orchestrator.py** — 上下文窗口编排器，预算协商，溢出降级处理，上下文版本管理

### 子模块 3：跨 Agent 记忆共享（文件 08-09）
- **08-shared-memory-blackboard.py** — 共享内存黑板，命名空间隔离，发布/订阅模式，事务管理，冲突检测
- **09-cross-agent-memory-routing.py** — 联邦记忆查询，查询路由，结果合并，多 Agent 记忆路由

### 子模块 4：记忆持久化与评估（文件 10-11）
- **10-persistent-memory-store.py** — SQLite 持久化记忆存储，版本管理，增量同步，向量搜索
- **11-memory-evaluation-metrics.py** — 检索质量指标（Precision@K, Recall@K, MRR, NDCG），压缩保真度，幻觉检测

### 子模块 5：记忆安全与自省（文件 12-13）
- **12-memory-privacy-security.py** — PII 检测与脱敏（9 种类型），保留策略，访问控制，审计日志哈希链
- **13-agent-memory-reflection.py** — 记忆自评估，矛盾检测，再巩固控制，记忆感知规划

### 子模块 6：流式交互与避坑指南（文件 14-21）
- **14-streaming-memory-interaction.py** — 流式记忆积累，早退检索，背压感知写入
- **pitfalls/01-05** — 五大常见陷阱分析与解决方案
- **checkpoint/** — 10 道综合练习 + 完整解答

## 三周学习计划

### 第 1 周：对话状态机与记忆压缩（文件 01-05）
| 天 | 内容 | 重点 |
|----|------|------|
| 1 | 01-dialog-fsm-core.py | 理解 FSM 核心概念，状态转移，ASCII 图 |
| 2 | 02-slot-filling-fsm.py | 槽位填充，验证器，必填检查 |
| 3 | 03-task-oriented-dialogue.py | 完整对话流程，意图分类 |
| 4 | 04-llm-summarization-compressor.py | 三种摘要策略，压缩验证 |
| 5 | 05-memory-consolidation-engine.py | 遗忘曲线，记忆巩固模拟 |

### 第 2 周：超长上下文与跨 Agent 记忆（文件 06-10）
| 天 | 内容 | 重点 |
|----|------|------|
| 1 | 06-ultra-long-context-strategies.py | 上下文窗口分区，KV-Cache |
| 2 | 07-context-window-orchestrator.py | 预算协商，降级处理 |
| 3 | 08-shared-memory-blackboard.py | 黑板模式，发布订阅 |
| 4 | 09-cross-agent-memory-routing.py | 联邦查询，结果合并 |
| 5 | 10-persistent-memory-store.py | SQLite 存储，版本管理 |

### 第 3 周：评估、安全、流式与综合（文件 11-21）
| 天 | 内容 | 重点 |
|----|------|------|
| 1 | 11-memory-evaluation-metrics.py | 检索与压缩质量指标 |
| 2 | 12-memory-privacy-security.py | PII 检测，访问控制 |
| 3 | 13-agent-memory-reflection.py | 记忆自省与再巩固 |
| 4 | 14-streaming-memory-interaction.py | 流式记忆处理 |
| 5 | pitfalls/ + checkpoint/ | 避坑指南与综合练习 |

## 学习目标

完成本模块后，你将能够：

1. **设计生产级对话状态机**：实现支持 8+ 状态、超时检测、死锁预防的 FSM
2. **实现多策略记忆压缩**：在层级、增量、结构化三种范式间灵活切换
3. **建模记忆遗忘与巩固**：基于艾宾浩斯曲线实现 STM→LTM 分级迁移
4. **编排超长上下文窗口**：实现 128K+ 上下文的区域预算分配与 KV-Cache 复用
5. **构建联邦记忆路由**：跨多个专业 Agent 实现统一查询路由与结果合并
6. **实现持久化记忆存储**：SQLite 全量 CRUD、版本管理、增量同步
7. **评估记忆系统质量**：Precision@K、MRR、NDCG、幻觉检测等完整指标体系
8. **保障记忆隐私安全**：PII 检测脱敏、保留策略、审计日志哈希链
9. **识别并解决常见陷阱**：遗忘症、记忆膨胀、摘要幻觉、跨会话泄漏、状态卡死

## 技术栈

- **Python 3.10+**：dataclasses, enum, asyncio, sqlite3, hashlib, re, json, typing
- **数学库**：math（指数衰减、对数计算）
- **数据结构**：deque, heapq, defaultdict, Counter

本模块为纯 Python 实现，不依赖任何第三方库，便于深入理解每个算法的内部机制。
