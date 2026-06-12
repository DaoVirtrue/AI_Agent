# 陷阱 3：工具选择错误 (Wrong Tool Selection)

## 概述

Agent 在面对多个可用工具时，可能选择错误的工具来处理任务。例如：用计算器查天气、用文件读取器做数学计算、用搜索处理数据库查询。工具选择错误会导致无效的工具调用、浪费 API 费用、产生错误回答。

---

## 症状 (Symptoms)

1. **工具-任务不匹配**：Agent 用 calculator 处理 "What's the weather?" 类型的查询
2. **跨域调用**：Agent 用 web_search 做数学计算（搜不到有效结果后又调用多次）
3. **功能重叠困惑**：有两个功能类似的工具时（如 web_search vs wikipedia_search），Agent 选择了功能较弱或数据较旧的那个
4. **忽略专用工具**：存在专用的 database_query 工具，Agent 却用 web_search 来"搜索"数据
5. **参数映射错误**：选了正确的工具但用完全错误的参数（如 weather(city="25*4+10")）

---

## 根本原因 (Root Cause)

### 1. 工具描述模糊或误导

工具的描述不清晰，没有明确说明"何时使用"和"何时不使用"。多个工具的描述之间存在语义重叠。

### 2. LLM 缺乏工具选择训练

通用 LLM 没有针对特定工具集进行微调，在面对语义相近的问题时容易混淆。

### 3. 缺少工具分类和路由

所有工具以扁平列表呈现给 LLM，没有按照功能域分组。LLM 需要在 10+ 个工具中做选择，错误概率随工具数量增加。

### 4. 用户输入与工具之间的语义鸿沟

用户的自然语言查询可能映射到多个工具域，LLM 选择了最表层的匹配。

---

## 真实场景 (Real Scenario)

### 场景 1：天气查询被当作计算

```
用户提问: "What's the weather in Paris?"

Agent 错误行为:
  Thought: 用户想知道 Paris 的天气..."Paris"可能是一个变量...
  Action: calculator[expression=weather in Paris]
  Observation: 错误: 无法计算表达式 'weather in Paris' - 表达式语法错误

  Thought: 让我尝试不同的写法
  Action: calculator[expression=weather(Paris)]
  Observation: 错误: 不支持的函数: weather()

  ... (继续浪费步数)
```

**问题分析**：工具描述不够清晰。"calculator"的描述是"执行数学计算"，对于 "weather" 这个词出现在问题中，LLM 应该调用 `get_weather` 而不是 `calculator`。但描述中没有明确的"不要用此工具处理天气查询"的否定示例。

### 场景 2：简单运算当作搜索

```
用户提问: "25 + 17 等于多少？"

Agent 错误行为:
  Thought: 用户想知道 25+17，让我搜索一下
  Action: web_search[query=25+17等于多少]
  Observation: 搜索结果: 未找到精确匹配的信息。

  Thought: 搜索没找到，让我换个方式搜
  Action: web_search[query=25 plus 17 result]
  Observation: 搜索结果: 未找到精确匹配的信息。

  ... (浪费 3 步后才尝试 calculator)
```

### 场景 3：数据库查询用搜索代替

```
用户提问: "技术部的平均工资是多少？"

Agent 错误行为:
  Thought: 用户想知道技术部工资，让我搜索相关信息
  Action: web_search[query=技术部 平均工资]
  Observation: 搜索结果: 未找到精确匹配（搜到的是市场薪资报告，不是数据库数据）

  Thought: 搜索结果不准确，让我再搜索
  Action: web_search[query=average salary tech department]
  ...

正确行为应该:
  Thought: 用户想查询数据库中的薪资信息，我应该使用 database_query
  Action: database_query[sql=SELECT department, AVG(salary) FROM employees WHERE department='技术部']
```

---

## 解决方案

### 方案 1：优化工具描述 —— 加入正向触发和负向排除

工具描述应该清楚地说明三件事：**这个工具做什么**、**何时使用它**、**何时绝对不要使用它**。

```python
def get_well_described_tools() -> dict:
    """返回具有精良描述的工具集。

    每个工具的 docstring 都遵循格式:
    1. 一句话总结功能
    2. 【使用场景】明确的正向触发条件
    3. 【不要使用】明确的负向排除条件
    4. 【示例输入】典型用法示例
    """

    def calculator(expression: str) -> str:
        """安全的数学计算器，支持基本运算和数学函数。

        【使用场景】
        - 用户问题中出现数字和运算符（+, -, *, /, **, sqrt, sin, cos 等）
        - 用户明确要求"计算"、"算一下"、"求值"、"等于多少"
        - 需要对数值进行加减乘除、百分比、幂运算等
        - 用户问题类似 "X + Y 等于多少"、"计算 sqrt(144)"

        【绝对不要使用此工具】
        - 不要用于非数学问题（如查询天气、搜索信息、读取文件）
        - 不要用于文本处理、语言翻译、情感分析
        - 不要将非数字的用户输入作为 expression 参数
        - 如果用户问题中没有明确的数学运算，请考虑其他工具

        【示例输入】
        - expression="25 * 4 + 10"
        - expression="sqrt(144) + sin(pi/2)"
        - expression="100 / 3"

        Args:
            expression: 纯数学表达式（仅包含数字、运算符、数学函数）
        """
        # ... 实现代码 ...

    def get_weather(city: str, date: str = "today") -> str:
        """查询指定城市的天气信息。

        【使用场景】
        - 用户问题中包含城市名称 + "天气"、"气温"、"下雨"、"晴天"等
        - 用户询问"某地今天/明天/昨天天气怎么样"
        - 用户想了解出行目的地的天气状况
        - 用户询问"带伞"、"穿什么"等与天气相关的建议

        【绝对不要使用此工具】
        - 不要用于非天气相关的城市信息查询（如人口、GDP、景点）
        - 不要将数学表达式作为 city 参数
        - 不要用于时间查询、日期计算
        - 如果用户没有提到具体城市，先询问城市名称再调用

        【示例输入】
        - city="北京", date="today"
        - city="巴黎", date="tomorrow"
        - city="东京"

        Args:
            city: 城市名称（中文或英文），如"北京"、"Paris"
            date: 日期，可选 "today"（默认）、"tomorrow"、"yesterday"
        """
        # ... 实现代码 ...

    def web_search(query: str) -> str:
        """在互联网上搜索信息。

        【使用场景】
        - 用户想知道某个概念的定义或解释
        - 用户询问事实性问题（如"故宫门票价格"、"Python 作者是谁"）
        - 用户需要了解最新资讯、新闻、事件
        - 用户查询不属于其他专用工具范畴的信息

        【绝对不要使用此工具】
        - 不要用于数学计算（使用 calculator）
        - 不要用于天气查询（使用 get_weather）
        - 不要用于数据库查询（使用 database_query）
        - 不要用于文件读取（使用 file_reader）
        - 不要将完整的数学表达式作为搜索查询

        【示例输入】
        - query="Python 编程语言简介"
        - query="故宫门票价格"
        - query="机器学习基本概念"

        Args:
            query: 搜索关键词或问题，尽量简洁精准
        """
        # ... 实现代码 ...

    def database_query(sql: str) -> str:
        """查询内部数据库（员工信息、部门数据）。

        数据库表结构:
        - employees: id, name, department, salary, hire_date
        - departments: id, name, manager, location

        【使用场景】
        - 用户询问公司内部的员工信息
        - 用户需要统计部门数据（平均薪资、人数统计等）
        - 用户查询包含"员工"、"部门"、"薪资"、"工资"、"技术部"、"市场部"等关键词
        - 用户想知道"谁在哪个部门"、"某部门有多少人"

        【绝对不要使用此工具】
        - 不要用于数学计算
        - 不要用于互联网搜索（内部数据库不包含外部信息）
        - 不要用于天气、文件、时间等其他工具的领域
        - 不要尝试 INSERT/UPDATE/DELETE（仅支持 SELECT）

        【示例输入】
        - sql="SELECT * FROM employees WHERE department='技术部'"
        - sql="SELECT department, AVG(salary) FROM employees GROUP BY department"
        - sql="SELECT * FROM employees ORDER BY salary DESC"

        Args:
            sql: SELECT 查询语句（仅支持 SELECT）
        """
        # ... 实现代码 ...

    def file_reader(filepath: str) -> str:
        """读取本地文件的内容。

        【使用场景】
        - 用户明确要求"读取文件"、"查看文件内容"
        - 用户提供的路径以 .py/.txt/.md/.json/.csv 等文本文件扩展名结尾
        - 用户想了解本地项目代码或配置文件的内容

        【绝对不要使用此工具】
        - 不要用于数学计算（使用 calculator）
        - 不要用于搜索互联网信息（使用 web_search）
        - 不要用于查询天气（使用 get_weather）
        - 不要将非文件路径的文本作为 filepath 参数

        【示例输入】
        - filepath="README.md"
        - filepath="/home/user/config.json"
        - filepath="./src/main.py"

        Args:
            filepath: 文件路径（绝对路径或相对路径）
        """
        # ... 实现代码 ...

    return {
        "calculator": calculator,
        "get_weather": get_weather,
        "web_search": web_search,
        "database_query": database_query,
        "file_reader": file_reader,
    }
```

### 方案 2：工具分类 + 预路由

在 Agent 推理之前，先用一个轻量分类步骤决定使用哪个工具类别，然后只给 LLM 展示相关工具。

```python
from enum import Enum
import re

class ToolCategory(Enum):
    """工具功能域分类。"""
    MATH = "math"           # 数学计算
    WEATHER = "weather"     # 天气查询
    SEARCH = "search"       # 信息搜索
    DATABASE = "database"   # 数据库查询
    FILE = "file"           # 文件操作
    UNKNOWN = "unknown"     # 无法分类

class ToolRouter:
    """预路由分类器：在 Agent 推理之前，先确定任务属于哪个工具类别。

    使用基于规则的方法（快速、零成本）做初步分类。
    实际生产环境中可以替换为轻量分类模型（如 DistilBERT）。
    """

    # 每个类别的关键词匹配规则
    CATEGORY_RULES = {
        ToolCategory.MATH: {
            "keywords": [
                "计算", "算", "等于", "求值", "加", "减", "乘", "除",
                "平方", "开方", "sqrt", "sin", "cos", "公式",
                "calculate", "compute", "result",
            ],
            "patterns": [
                r'\d+\s*[\+\-\*\/\*\*]\s*\d+',  # 数字 + 运算符 + 数字
                r'sqrt\(', r'sin\(', r'cos\(', r'log\d*\(',
            ],
        },
        ToolCategory.WEATHER: {
            "keywords": [
                "天气", "气温", "温度", "湿度", "下雨", "晴天", "阴天",
                "刮风", "台风", "雾霾", "穿衣", "带伞", "降水",
                "weather", "temperature", "rain", "sunny", "cloudy",
            ],
            "patterns": [
                r'.*(天气|weather).*',
            ],
        },
        ToolCategory.DATABASE: {
            "keywords": [
                "员工", "部门", "工资", "薪资", "入职", "人事",
                "财务", "技术部", "市场部", "人事部",
                "查询", "统计", "平均", "总和",
                "employee", "salary", "department", "SQL",
            ],
            "patterns": [
                r'(平均|总|最高|最低).*(工资|薪资|salary)',
                r'.*部门.*(工资|薪资|人|员工)',
            ],
        },
        ToolCategory.FILE: {
            "keywords": [
                "文件", "读取", "查看文件", "打开文件", "文件内容",
                "read file", "open file", "file content",
            ],
            "patterns": [
                r'读取.*文件', r'查看.*文件', r'打开.*文件',
                r'.*\.(py|txt|md|json|csv|yaml|yml|toml|ini|cfg|log)',
            ],
        },
        ToolCategory.SEARCH: {
            "keywords": [
                "搜索", "查找", "什么是", "介绍", "了解", "谁", "哪里",
                "什么", "何时", "为什么", "怎么", "如何",
                "search", "find", "what is", "who is", "how to",
                "门票", "价格", "开放时间", "地址", "电话",
            ],
            "patterns": [],
        },
    }

    def __init__(self, default_category: ToolCategory = ToolCategory.SEARCH):
        self.default_category = default_category

    def classify(self, task: str) -> ToolCategory:
        """将用户任务分类到工具类别。

        基于关键词匹配和正则表达式做快速分类。
        多个类别匹配时，按优先级返回第一个匹配的。

        Args:
            task: 用户任务描述

        Returns:
            最匹配的工具类别
        """
        task_lower = task.lower()
        scores = {}

        for category, rules in self.CATEGORY_RULES.items():
            score = 0

            # 关键词匹配
            for keyword in rules["keywords"]:
                if keyword.lower() in task_lower:
                    score += 1

            # 正则匹配（权重更高）
            for pattern in rules["patterns"]:
                if re.search(pattern, task, re.IGNORECASE):
                    score += 3

            if score > 0:
                scores[category] = score

        if not scores:
            return self.default_category

        # 返回得分最高的类别
        return max(scores, key=scores.get)

    def get_tools_for_category(self, category: ToolCategory,
                                all_tools: dict) -> dict:
        """获取指定类别的工具子集。

        每个类别包含主要工具 + 回退工具（web_search 作为万能回退）。

        Args:
            category: 工具类别
            all_tools: 完整的工具字典

        Returns:
            该类别的工具子集
        """
        category_tools = {
            ToolCategory.MATH: ["calculator", "web_search"],
            ToolCategory.WEATHER: ["get_weather", "web_search"],
            ToolCategory.SEARCH: ["web_search", "file_reader"],
            ToolCategory.DATABASE: ["database_query", "web_search"],
            ToolCategory.FILE: ["file_reader", "web_search"],
            ToolCategory.UNKNOWN: list(all_tools.keys()),
        }

        tool_names = category_tools.get(category, list(all_tools.keys()))
        return {name: all_tools[name] for name in tool_names if name in all_tools}


# 使用预路由的 Agent
class ReActAgentWithRouting(ReActAgent):
    """带工具预路由的 ReAct Agent。

    优势:
    1. 减少无关工具干扰（LLM 只看到相关工具）
    2. 降低选错工具的概率
    3. 减少 Prompt Token 消耗（更少的工具描述）
    """

    def __init__(self, llm_client, tools: dict, max_steps: int = 15):
        super().__init__(llm_client, tools, max_steps)
        self.router = ToolRouter()
        self.all_tools = tools  # 保留完整工具集用于回退

    def _classify_and_select_tools(self, task: str) -> dict:
        """对任务进行分类并选择对应工具集。"""
        category = self.router.classify(task)
        selected_tools = self.router.get_tools_for_category(category, self.all_tools)

        print(f"[预路由] 任务分类: {category.value}")
        print(f"[预路由] 可用工具: {list(selected_tools.keys())}")

        return selected_tools

    def run(self, task: str, verbose: bool = True) -> str:
        """带预路由的 ReAct 执行。"""
        # 步骤 1: 预路由 —— 选择相关工具
        self.tools = self._classify_and_select_tools(task)

        # 步骤 2: 正常 ReAct 循环
        result = super().run(task, verbose=verbose)

        # 步骤 3: 检查是否需要扩大工具集
        # 如果 Agent 没有完成任务且未使用全部工具，用完整工具集重试
        if self._indicates_wrong_tools(result):
            print("[预路由] 检测到可能需要更多工具，使用完整工具集重试...")
            self.tools = self.all_tools
            # 重置并重试（实际中可能不需要完整重试，这里简化处理）
            result = super().run(task, verbose=verbose)

        return result

    def _indicates_wrong_tools(self, result: str) -> bool:
        """检查结果是否表明工具集不足。"""
        indicators = [
            "无法完成", "需要更多", "不能处理",
            "没有合适的工具", "工具不足",
        ]
        return any(ind in result for ind in indicators)
```

### 方案 3：置信度阈值 + 澄清机制

在每次工具选择后，评估模型的置信度。如果置信度过低，暂停并让用户澄清。

```python
class ConfidenceToolSelector:
    """带置信度评估的工具选择器。

    在让 LLM 选择工具后，用额外的 prompt 评估其选择置信度。
    低置信度时触发澄清机制，避免错误执行。
    """

    def __init__(self, llm_client, confidence_threshold: float = 0.7):
        self.llm = llm_client
        self.confidence_threshold = confidence_threshold
        self.selection_history: list = []  # 记录历史选择

    def evaluate_selection(self, task: str, selected_tool: str,
                           available_tools: list[str]) -> dict:
        """评估工具选择的合理性。

        让 LLM 对自己的选择进行二次评估。

        Returns:
            {
                "confidence": 0.85,  # 0-1
                "reasoning": "选择 calculator 因为...",
                "alternative": "web_search",  # 次优选择
                "should_proceed": True,
                "clarification_needed": False,
                "clarification_question": "",
            }
        """
        evaluation_prompt = f"""你是一个工具选择评估器。

用户任务: "{task}"

Agent 选择的工具: "{selected_tool}"

可用工具列表: {available_tools}

请评估这个工具选择是否合理。以 JSON 格式回答:
{{
    "confidence": 0.0-1.0,  // 你对选择的信心程度
    "reasoning": "选择原因或问题所在",
    "alternative": "如果选错了，应该选哪个工具？",
    "is_correct": true/false,
    "should_ask_clarification": true/false,
    "clarification_question": "如果需要用户澄清，要问什么？没有则留空"
}}

如果 confidence < 0.5 且 is_correct = false，必须设置 should_ask_clarification = true。
"""
        response = self.llm.generate(evaluation_prompt)
        try:
            import json
            # 提取 JSON（处理 LLM 可能输出的非 JSON 文本）
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"confidence": 0.5, "is_correct": True, "should_ask_clarification": False}
        except (json.JSONDecodeError, AttributeError):
            result = {"confidence": 0.5, "is_correct": True, "should_ask_clarification": False}

        # 记录选择历史
        self.selection_history.append({
            "task": task,
            "selected": selected_tool,
            "evaluation": result,
        })

        return result

    def should_proceed(self, evaluation: dict) -> bool:
        """根据评估结果决定是否继续执行。"""
        confidence = evaluation.get("confidence", 0.5)
        is_correct = evaluation.get("is_correct", True)

        # 高置信度 + 选择正确 → 执行
        if is_correct and confidence >= self.confidence_threshold:
            return True

        # 中等置信度 + 选择正确 → 执行但记录警告
        if is_correct and confidence >= 0.4:
            print(f"[工具选择警告] 选择 '{evaluation.get('selected', 'N/A')}' 的置信度仅为 {confidence:.0%}")
            return True

        # 低置信度或选择错误 → 不执行
        return False

    def get_clarification(self, evaluation: dict) -> str:
        """获取需要向用户澄清的问题。"""
        question = evaluation.get("clarification_question", "")
        if not question:
            alternative = evaluation.get("alternative", "")
            reasoning = evaluation.get("reasoning", "")
            question = (
                f"我不太确定应该用什么方法来处理你的请求。\n"
                f"我的初步判断是使用 '{evaluation.get('selected', '?')}'，"
                f"但信心不足（{evaluation.get('confidence', 0):.0%}）。\n"
                f"原因: {reasoning}\n"
            )
            if alternative:
                question += f"备选方案: {alternative}\n"
            question += "\n你能帮我确认一下，你希望我：\n"
            question += f"1. 继续用 '{evaluation.get('selected', '?')}' 处理？\n"
            if alternative:
                question += f"2. 改用 '{alternative}'？\n"
            question += "3. 提供更多信息？"
        return question


# 集成到 Agent 中
class ReActAgentWithConfidence(ReActAgent):
    """带置信度评估的 ReAct Agent。"""

    def __init__(self, llm_client, tools: dict, max_steps: int = 15,
                 confidence_threshold: float = 0.7):
        super().__init__(llm_client, tools, max_steps)
        self.selector = ConfidenceToolSelector(llm_client, confidence_threshold)

    def _prepare_tool_call(self, task: str, action: tuple) -> tuple[bool, str]:
        """在工具调用前评估选择是否正确。

        Returns:
            (是否继续执行, 消息)
        """
        tool_name = action[0]
        evaluation = self.selector.evaluate_selection(
            task, tool_name, list(self.tools.keys())
        )

        if self.selector.should_proceed(evaluation):
            return True, ""
        else:
            clarification = self.selector.get_clarification(evaluation)
            return False, clarification
```

### 方案 4：工具选择历史验证

维护工具选择的历史记录，在当前选择与历史模式矛盾时发出警告。

```python
class ToolSelectionValidator:
    """基于历史模式的工具选择验证器。

    检测异常的工具选择模式，如:
    - 突然从专用工具切换到不相关的通用工具
    - 对相似问题选择不同的工具
    - 频繁切换工具类型
    """

    def __init__(self):
        self.history: list[dict] = []

    def record(self, task: str, selected_tool: str, result: str):
        """记录一次工具选择。"""
        self.history.append({
            "task": task[:100],
            "tool": selected_tool,
            "result_preview": result[:100],
        })

    def validate(self, task: str, selected_tool: str,
                 available_tools: list[str]) -> list[str]:
        """验证当前工具选择，返回警告列表。

        Returns:
            警告信息列表（空列表表示无问题）
        """
        warnings = []

        if not self.history:
            return warnings

        # 检查 1: 同一任务重复选择不同工具
        last_selections = [h["tool"] for h in self.history[-3:]]
        if (len(last_selections) >= 2 and
            len(set(last_selections)) == len(last_selections) and
            selected_tool not in last_selections):
            # 每次都在换工具，且现在又要换一个新的
            warnings.append(
                f"[模式警告] 最近 3 步使用了 3 个不同的工具（{', '.join(last_selections)}），"
                f"现在又要使用 '{selected_tool}'。建议回顾之前的工具结果，考虑是否可以给出答案。"
            )

        # 检查 2: 功能相似的工具有更优选择
        similar_groups = [
            (["web_search", "wikipedia_search", "google_search"], "信息搜索"),
            (["calculator", "math_solver", "equation_solver"], "数学计算"),
        ]
        for group, domain in similar_groups:
            available_in_group = [t for t in group if t in available_tools]
            if selected_tool in group and len(available_in_group) > 1:
                # 多个相关工具可用，检查是否选了最优的
                better = available_in_group[0]  # 假设按优先级排序
                if selected_tool != better:
                    warnings.append(
                        f"[工具建议] 在{domain}领域，'{better}' 可能比 '{selected_tool}' 更合适。"
                    )

        # 检查 3: 输出类型不匹配
        # 例如：需要数值结果的任务却选择了返回文本的工具
        numeric_keywords = ["多少", "计算", "等于", "总和", "平均", "统计"]
        if any(kw in task for kw in numeric_keywords):
            numeric_tools = ["calculator", "database_query"]
            if selected_tool not in numeric_tools:
                # 检查是否所有数字工具都不可用
                available_numeric = [t for t in numeric_tools if t in available_tools]
                if available_numeric:
                    warnings.append(
                        f"[类型不匹配] 用户任务似乎需要数值结果（关键词: 多少/计算/等于），"
                        f"但你选择了文本型工具 '{selected_tool}'。"
                        f"建议使用: {available_numeric}"
                    )

        return warnings
```

---

## 预防清单 (Checklist)

在配置 Agent 工具集时，请确保：

- [ ] 1. 每个工具的 docstring 包含"使用场景"和"绝对不要使用"两部分
- [ ] 2. 工具描述中包含正向触发关键词和负向排除示例
- [ ] 3. 实现了工具预路由机制，减少无关工具对 LLM 的干扰
- [ ] 4. 关键工具有充分的示例输入（帮助 LLM 理解参数格式）
- [ ] 5. 对功能相似的工具有明确的优先级说明（如"web_search 优于 wikipedia_search"）
- [ ] 6. 测试过常见查询，确认每个查询都会路由到正确的工具
- [ ] 7. 记录了工具选择的日志，方便事后分析选错原因

---

## 总结

工具选择错误源于 LLM 对工具功能的理解不足。核心解决方案：

| 方案 | 实施位置 | 效果 | 成本 |
|------|---------|------|------|
| 优化工具描述 | 工具定义 | 中 | 低（一次性编写） |
| 预路由分类 | Agent 执行前 | 高 | 低（规则匹配） |
| 置信度评估 | 工具选择时 | 高 | 中（额外 LLM 调用） |
| 历史验证 | 工具选择时 | 中 | 低（本地计算） |

推荐组合：**优化工具描述（静态预防）+ 预路由分类（动态引导）+ 历史验证（运行时检查）**。

---

> **陷阱系列完成！** 回顾三个陷阱，建立完整的 Agent 健壮性防线：
> 1. [01-infinite-loop.md](./01-infinite-loop.md) —— 无限循环
> 2. [02-tool-call-failure.md](./02-tool-call-failure.md) —— 工具调用失败
> 3. [03-wrong-tool-selection.md](./03-wrong-tool-selection.md) —— 工具选择错误
