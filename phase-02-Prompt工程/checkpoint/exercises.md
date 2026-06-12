# Phase 02 Checkpoint：Prompt 工程 - 练习题

本 checkpoint 包含 8 个练习，涵盖 Zero-shot/Few-shot 提示、Chain-of-Thought 推理、Token 预算管理、Jinja2 模板化、鲁棒 JSON 解析、A/B 测试流水线和 DSPy 自动优化器。每个练习应独立完成，参考答案见 `solutions.md`。

---

## 练习 1：Zero-shot 分类器

**背景**：你有一个电商客服系统，需要将用户消息分为 `refund`（退款）、`shipping`（物流）、`product_info`（商品咨询）、`complaint`（投诉）四类。

**任务**：
1. 编写一个 `zero_shot_classifier()` 函数，接收用户消息和类别列表，调用 LLM 返回分类结果
2. 分类结果必须是纯类别名（不含解释），并包装为 `ClassifyResult` 数据类
3. Prompt 中需要包含输出格式约束（仅输出类别名）

**要求**：
- 使用 `ClassifyResult` 数据类封装结果
- 结果包含：类别名（str）、置信度（0~1 float）、原始响应（str）
- 处理 LLM 可能返回多余文本的情况

**测试案例**：
```python
msg1 = "我的包裹已经一个星期了还没到，能帮我查一下吗？"
msg2 = "这个商品质量太差了，用了两天就坏了，我要退款！"
```

**期望输出格式**：
```python
ClassifyResult(category="shipping", confidence=0.9, raw_response="shipping")
```

---

## 练习 2：Few-shot 命名实体识别（NER）

**背景**：你需要从技术文档中提取软件名称、版本号和公司名称，但 LLM 直接输出的格式不稳定。

**任务**：
1. 编写 `few_shot_ner_extractor()` 函数
2. 构建 3 个 Few-shot 示例，每个示例包含输入文本和期望的 JSON 输出
3. 使用这些示例引导 LLM 输出符合格式的 NER 结果
4. 用 `RobustJSONParser` 解析 LLM 输出

**要求**：
- Few-shot 示例必须覆盖：单个实体、多个实体、无实体三种情况
- 输出 JSON 格式：`{"entities": [{"text": "...", "type": "...", "start": 0, "end": 5}]}`
- 使用鲁棒解析器处理输出

**测试案例**：
```python
text = "Apache Hadoop 3.3.6 由 Apache Software Foundation 发布，包含 HDFS 和 YARN 组件。"
```

---

## 练习 3：Chain-of-Thought 推理

**背景**：用户问了一个需要多步推理的问题，直接回答准确率低。

**任务**：
1. 编写 `cot_reasoner()` 函数，实现 Chain-of-Thought 推理
2. Prompt 要求 LLM 先输出推理步骤（用 `[REASONING]` 标记），再输出最终答案（用 `[ANSWER]` 标记）
3. 解析输出，分别提取推理过程和最终答案
4. 实现结果类 `CoTResult` 包含 reasoning_steps 和 final_answer

**要求**：
- 使用明确的标记系统（`[REASONING]` 和 `[ANSWER]`）
- 解析函数需处理 LLM 可能遗漏标记的情况
- 支持自定义推理步骤（如 "第一步分析、第二步计算、第三步总结"）

**测试案例**：
```python
question = "一个商店周一卖出120件商品，周二比周一多卖15%，周三比周二少卖10%。三天总共卖出多少件？"
```

---

## 练习 4：Token 预算控制器

**背景**：你的系统需要将多篇检索文档传给 LLM，但不能超过 8000 token 的输入限制。

**任务**：
1. 编写 `TokenBudget` 类
2. 实现 `allocate()` 方法，根据 system prompt、history、query 的长度计算文档可用 token 数
3. 实现 `truncate_docs_to_budget()` 方法，按预算截断文档列表
4. 实现粗略的 token 计数函数（`count_tokens`），中文按字符数、英文按 4 字符/token

**要求**：
- 截断策略：优先保留前面的文档，最后一份文档可截断填充剩余空间
- 至少保留一份完整文档
- 截断时在句子边界处断开（以句号、问号、感叹号分隔）
- 返回截断统计（原文档数、保留文档数、截断字符数）

**测试案例**：
```python
docs = ["文档1内容" * 100, "文档2内容" * 200, "文档3内容" * 300]
system_prompt = "你是客服助手" * 50
budget = TokenBudget(max_input_tokens=500)
result = budget.truncate_docs_to_budget(docs, budget.allocate(system_prompt, "查询", None, 3)["docs_budget"])
```

---

## 练习 5：Jinja2 模板化 Prompt 引擎

**背景**：你的团队需要管理数十个 prompt 模板，不同场景使用不同模板，且需要支持变量替换和条件渲染。

**任务**：
1. 编写 `PromptTemplateEngine` 类，基于 Jinja2
2. 支持 `register_template(name, template_string)` 注册模板
3. 支持 `render(name, **variables)` 渲染模板
4. 实现模板继承：子模板可以继承父模板的 system_prompt 部分
5. 支持条件渲染：`{% if context_docs %}...{% endif %}`

**要求**：
- 使用 Jinja2 的 `Environment` 和 `FileSystemLoader`（或 `DictLoader`）
- 模板包含：system_prompt, user_prompt, 可选的 few_shot_examples
- `render()` 返回渲染后的完整 prompt 字典
- 变量未提供时抛出明确的错误信息

**模板示例**：
```jinja2
{% extends "base_template" %}
{% block system %}
你是一个{{ role }}。回答时请遵循以下规则：
{{ rules | join('\n') }}
{% endblock %}

{% block user %}
用户问题：{{ query }}
{% if docs %}
参考文档：
{% for doc in docs %}
- {{ doc.title }}: {{ doc.snippet }}
{% endfor %}
{% endif %}
{% endblock %}
```

---

## 练习 6：RobustJSONParser（完整实现）

**背景**：基于课程中的五级回退解析器概念，实现一个完整的鲁棒JSON解析器。

**任务**：
1. 实现 `RobustJSONParser` 类，包含完整的五级回退策略
2. Level 1: 直接 `json.loads()`
3. Level 2: 提取 Markdown 代码块（```json...```）
4. Level 3: 大括号/方括号边界提取（含深度追踪）
5. Level 4: 启发式错误修复（尾逗号、单引号、无引号键名、缺失闭合）
6. Level 5: `ast.literal_eval()` 兜底

**要求**：
- 每级失败后自动降级，记录修复日志
- 返回 `ParseResult` 数据类（success, data, level, repair_log）
- 提供统计功能：总解析次数、各级成功次数、成功率

**测试案例**：
```python
parser = RobustJSONParser()
# 应分别在 Level 1, 2, 3, 4 成功
tests = [
    '{"name": "Alice"}',                           # Level 1
    '```json\n{"name": "Bob"}\n```',               # Level 2
    '结果是：\n{"items": [1,2,3]}\n完成',          # Level 3
    "{'name': 'Charlie', 'age': 25,}",              # Level 4
]
```

---

## 练习 7：A/B 测试流水线

**背景**：你想比较两个 prompt 变体（Variant A 和 Variant B）对同一个 QA 数据集的效果。

**任务**：
1. 编写 `ABTestPipeline` 类
2. 接收两个 prompt 模板（template_a, template_b）和一个测试数据集
3. 对每个测试样本，分别用两个模板生成回答
4. 用 LLM 作为评判器（LLM-as-Judge），比较两个回答的质量
5. 统计结果：A胜次数、B胜次数、平局次数、A的平均分数、B的平均分数

**要求**：
- 评测维度：准确性（accuracy）、完整性（completeness）、简洁性（conciseness）
- 每个维度 1-5 分
- 评判器的 prompt 需要独立且可复现
- 输出完整的比较报告

**测试案例**：
```python
qa_pairs = [
    {"question": "什么是机器学习？", "reference": "机器学习是AI的一个分支..."},
    {"question": "Python中list和tuple的区别？", "reference": "list可变，tuple不可变..."},
]

template_a = "简洁回答：{{ question }}"
template_b = "详细回答：{{ question }}。请从定义、特点、示例三方面说明。"

pipeline = ABTestPipeline(template_a, template_b, qa_pairs)
report = pipeline.run()
```

---

## 练习 8：DSPy 自动优化器

**背景**：手动调整 prompt 费时且容易过拟合特定案例。使用 DSPy 框架自动优化 Few-shot 示例的选择和组合。

**任务**：
1. 使用 DSPy 定义签名（Signature）：Question -> Answer
2. 编写 `dspy_optimize()` 函数，接收训练集（问答对）和验证集
3. 使用 `BootstrapFewShot` 优化器自动选择和组合最佳 Few-shot 示例
4. 比较优化前后的模型准确率

**要求**：
- 定义 DSPy Module（如 `dspy.ChainOfThought`）
- 使用 `BootstrapFewShot` 进行 Few-shot 示例的自动选择
- 训练后在验证集上评估
- 输出优化前后对比（准确率、使用的示例数量）
- 如果 DSPy 不可用，提供模拟实现，展示工作原理

**提示**：
```python
import dspy

class QA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

# 或使用内置
qa_module = dspy.ChainOfThought("question -> answer")
```

---

## 提交要求

1. 所有代码放在 `phase-02-Prompt工程/checkpoint/` 目录下
2. 每个练习一个 `.py` 文件（如 `ex1_zero_shot.py`）
3. 包含 `if __name__ == "__main__":` 测试代码块
4. 代码注释覆盖关键逻辑
5. 参考 `solutions.md` 对照检查
