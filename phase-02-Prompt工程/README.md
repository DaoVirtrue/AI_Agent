# Phase 02: Prompt Engineering (提示工程)

**Duration**: 1-2 weeks | **Difficulty**: 中级 | **Type**: 核心技能

## 阶段概述

本阶段系统性地教授提示工程（Prompt Engineering），从基础设计模式到自动化优化管道。你将掌握如何编写高效提示词、管理上下文窗口、构建结构化输出，以及使用 DSPy 实现提示自动优化。

## 学习目标

完成本阶段后，你将能够：

1. **设计多模式提示词**：掌握 Zero-shot → Few-shot → Chain-of-Thought → ReAct → Self-Ask 五种核心模式，并能为任意任务选择最佳模式
2. **构建模板引擎**：使用 Jinja2 创建可复用的提示模板，支持条件逻辑、动态 Few-shot 注入和角色框架
3. **强制结构化输出**：使用 OpenAI Structured Outputs 和 Pydantic 约束输出格式，实现多级回退解析器
4. **管理上下文窗口**：精确控制 token 预算，实现滑动窗口、分层记忆和溢出防护
5. **实现自动化优化**：使用统计学方法（Welch's t-test, Cohen's d）比较提示变体，并通过 DSPy 编译器实现端到端自动化

## 文件清单

| 文件 | 内容 | 类型 |
|------|------|------|
| `README.md` | 阶段概述、学习目标、概念预览 | 文档 |
| `01-prompt-design-patterns.ipynb` | Zero-shot / Few-shot / CoT / ReAct / Self-Ask 五大模式 | 交互式笔记本 |
| `02-template-engines.ipynb` | Jinja2 模板引擎、角色框架、动态 Few-shot 注入 | 交互式笔记本 |
| `03-output-structuring.ipynb` | Pydantic 结构化输出、JSON Schema 约束、多级回退解析 | 交互式笔记本 |
| `04-context-window-management.ipynb` | Token 预算、滑动窗口、分层记忆、溢出防护 | 交互式笔记本 |
| `05-few-shot-selection.py` | FewShotSelector、ExampleOrderingStrategies、HardExampleMiner | Python 模块 |
| `06-prompt-optimization-pipeline.py` | PromptOptimizationPipeline、EvaluationResult、统计分析 | Python 模块 |
| `07-dspy-automated-optimization.py` | DSPy 编译器、Signature、Module、MIPROv2 优化器 | Python 模块 |

## 核心概念预览

### 1. Zero-shot → Few-shot → CoT → ReAct

```
复杂度和推理能力逐步增强：

Zero-shot:  任务描述 → 直接输出
    ↓ + 2-5个示例
Few-shot:   任务描述 + 示例 → 模式匹配 → 输出
    ↓ + "Let's think step by step"
CoT:        任务描述 + 推理步骤 → 链式推理 → 输出
    ↓ + Thought/Action/Observation 循环
ReAct:      任务描述 + 外部工具 → 思考-行动-观察 → 输出
```

### 2. 提示模板引擎

```
提示构建的工程化方法：

模板渲染 (Jinja2):
  原始模板字符串 + 变量上下文 → 渲染后的提示

角色框架 (Role-Framing):
  "你是一位{role}，拥有{years}年经验..." → 设定行为预期

动态 Few-shot (Dynamic Shot Injection):
  查询向量 → 检索相似示例 → 注入模板 → 生成提示
```

### 3. 结构化输出管道

```
用户输入 → LLM + JSON Schema → 原始输出
  → 主解析器 (json.loads) ──失败──→ 回退层1 (正则提取)
  → 回退层2 (JSON修复) ──失败──→ 回退层3 (重试+提示修正)
  → Pydantic 验证 → 结构化结果
```

### 4. Token 预算分配策略

```
+------------------+-------+
| 组件              | 占比   |
+------------------+-------+
| System Prompt    | 15%   |
| Few-shot Examples|  8%   |
| Conversation Hist| 50%   |
| Knowledge/RAG    | 15%   |
| Output Allocation| 12%   |
+------------------+-------+
```

### 5. DSPy 自动优化

```
任务定义 (Signature)
  → 初始模块 (Module)  
  → 训练数据 (DevSet)  
  → 优化器 (BootstrapFewShot / MIPROv2)  
  → 编译后的最优程序 (CompiledProgram)
```

## 常见陷阱预览

1. **过度 Few-shot**：超过 5 个示例后边际收益递减，且消耗大量 token
2. **CoT 过度使用**：简单任务使用 CoT 反而增加错误率和延迟
3. **忽略 token 预算**：对话历史膨胀导致上下文溢出，静默截断关键信息
4. **硬编码提示**：提示字符串硬编码在代码中，维护和 A/B 测试极其困难
5. **无结构化输出**：依赖正则匹配解析 LLM 输出，脆弱且不可靠
6. **模板注入漏洞**：用户输入直接拼接到提示中，可能绕过系统指令
7. **忽视输出格式变化**：模型更新后输出格式可能变化，需要监控和回退
8. **过度优化特定测试集**：在小型测试集上过拟合，泛化性能差

## 前置要求

- Python 3.10+
- 已安装依赖：`openai`, `jinja2`, `pydantic`, `tiktoken`, `numpy`, `scipy`, `dspy-ai`
- 有效的 OpenAI API Key（设置为环境变量 `OPENAI_API_KEY`）

## 快速安装

```bash
pip install openai jinja2 pydantic tiktoken numpy scipy dspy-ai rich tabulate
```
