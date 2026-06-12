# Phase 07：多智能体系统（Multi-Agent Systems）

> **预计时间**：2 周 | **难度**：⭐⭐⭐⭐ | **前置要求**：Phase 06（Agent基础）

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 理解并实现 **7 种多智能体协作模式**，能根据业务场景选择最优架构
2. 设计 **层级化 Manager-Worker** 系统，实现任务的自动分解与并行调度
3. 构建 **多智能体辩论与投票机制**，通过对抗提升输出质量
4. 实现 **三级升级（Escalation）模式**，在自动化与人工介入之间取得平衡
5. 使用 **黑板模式（Blackboard）** 实现多智能体共享状态与冲突检测
6. 设计 **标准化 Agent 通信协议**，理解 MCP 和 A2A 协议的设计思路
7. 诊断并修复多智能体系统中 5 个常见致命故障

---

## 📂 文件清单

| 文件 | 类型 | 核心内容 | 建议时长 |
|------|------|----------|----------|
| `01-sequential-pipeline.py` | Python脚本 | SequentialPipeline类，串行流水线，各阶段计时 | 1-2小时 |
| `02-hierarchical-manager-worker.py` | Python脚本 | Manager分解任务→Worker并行执行→合成结果 | 2-3小时 |
| `03-debate-ensemble.py` | Python脚本 | 多角色辩论，结构化轮次，投票与裁决 | 2-3小时 |
| `04-escalation-pattern.py` | Python脚本 | L1→L2→L3三级升级，置信度阈值，人工队列 | 1-2小时 |
| `05-blackboard-pattern.py` | Python脚本 | 共享黑板，线程安全读写，冲突检测 | 1-2小时 |
| `06-agent-communication-protocol.py` | Python脚本 | 标准化消息格式，发布订阅总线，MCP/A2A | 1-2小时 |
| `07-role-playing-agents.py` | Python脚本 | Persona定义，角色工具分配，团队协作 | 2-3小时 |
| `pitfalls/` | 参考文档 | 5个多智能体致命陷阱诊断与修复 | 1-2小时 |
| `checkpoint/` | 练习 | 7道练习题+详解答案 | 2-3小时 |

**总计：约14-24小时**

---

## 🔑 核心概念

### 多智能体系统的动机

```
单一Agent的局限：
├── 上下文窗口有限（一个Agent只能处理有限信息）
├── 能力单一（一个Agent很难同时擅长代码、写作、分析）
├── 缺乏制衡（没有其他Agent质疑和纠正错误）
└── 无法并行（一个Agent一次只能做一件事）

多Agent系统的优势：
├── 专业化分工（每个Agent只做自己最擅长的事）
├── 并行处理（多个Agent同时工作，加速执行）
├── 相互制衡（辩论、投票、审查 → 减少幻觉）
└── 弹性扩展（新增Agent类型不影响现有系统）
```

### 七大协作模式一览

```
1. Sequential（串行）
   Agent A → Agent B → Agent C → Agent D
   适用：线性任务（研究→写作→审校→润色）

2. Hierarchical（层级管理）
   Manager → [Worker1, Worker2, Worker3] → Manager合成
   适用：复杂项目分解（代码项目、研究报告）

3. Debate（辩论）
   Agent Pro ↔ Agent Con → Judge裁决
   适用：需要多角度分析（政策评估、投资决策）

4. Escalation（升级）
   L1简单Agent → L2复杂Agent → L3人工
   适用：客服系统、工单处理

5. Blackboard（黑板）
   共享数据 ←→ [Agent A, Agent B, Agent C]
   适用：旅行规划、协同设计

6. Ensemble（集成）
   多个Agent独立回答 → 投票/加权 → 最终答案
   适用：高可靠性需求（医疗、法律）

7. Role-Playing（角色扮演）
   PM ↔ Dev ↔ QA ↔ DevOps（团队仿真）
   适用：软件工程、创意写作
```

### 选择决策矩阵

| 场景特征 | 推荐模式 | 原因 |
|----------|----------|------|
| 任务有明确顺序依赖 | Sequential | 简单，无额外通信开销 |
| 任务可独立分解 | Hierarchical | 并行加速 |
| 需要高可靠性 | Ensemble/Debate | 多角度验证 |
| 部分问题需人工处理 | Escalation | 成本控制 |
| 多Agent需共享数据 | Blackboard | 集中数据管理 |
| 仿真真实团队 | Role-Playing | 自然交互模式 |

---

## ⚠️ 本阶段坑点预览

| # | 坑点 | 根因 | 后果 |
|---|------|------|------|
| 1 | 协调开销爆炸 | 过多Agent来回通信 | 延迟指数增长，成本飙升 |
| 2 | 幻觉级联放大 | 上游Agent的错误→下游放大 | 最终输出严重偏离事实 |
| 3 | 辩论死锁 | 双方观点势均力敌，无法收敛 | 无限循环，无结果返回 |
| 4 | 消息格式不匹配 | 各Agent消息Schema不一致 | 解析失败，通信中断 |
| 5 | 资源争抢 | 多Agent同时访问共享资源 | 死锁、超时、数据不一致 |

---

## ✅ 检验标准

完成 `checkpoint/exercises.md` 中的 7 道练习，确保：

- [ ] 能实现 Sequential Pipeline 并测量各阶段耗时
- [ ] 能设计 Manager-Worker 架构并处理依赖关系
- [ ] 能构建 Debate 机制并实现加权投票
- [ ] 能实现三级升级策略并定义清晰的升级条件
- [ ] 能使用 Blackboard 模式共享状态且无竞争条件
- [ ] 能定义标准化消息格式并搭建发布订阅总线
- [ ] 能创建角色扮演团队并完成协作任务

---

## 🚀 快速开始

```bash
# 1. 激活虚拟环境
venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install openai anthropic python-dotenv pydantic

# 3. 配置 API Key
export OPENAI_API_KEY=sk-your-key-here

# 4. 运行示例
python 01-sequential-pipeline.py
```

---

准备好了吗？从 `01-sequential-pipeline.py` 开始探索多智能体的世界！
