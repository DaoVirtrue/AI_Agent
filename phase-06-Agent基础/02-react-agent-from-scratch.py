#!/usr/bin/env python3
"""
ReAct Agent - 从零实现 (From-Scratch Implementation)
====================================================
ReAct: Reasoning + Acting 模式
循环: Thought → Action → Tool → Observation → Thought... → Final Answer

输出格式 (通过prompt约束):
Thought: <推理过程>
Action: tool_name[param1=value1, param2=value2]
Observation: <系统填充工具返回结果>
... (重复直到得到答案)
Final Answer: <最终回答>

参考论文: ReAct: Synergizing Reasoning and Acting in Language Models
arXiv: https://arxiv.org/abs/2210.03629

作者: Phase 06 - Agent 基础
运行方式:
    python 02-react-agent-from-scratch.py
    python 02-react-agent-from-scratch.py --task "What is 25 * 4 + 10?"
    python 02-react-agent-from-scratch.py --task "查询北京天气并搜索故宫门票价格"
    python 02-react-agent-from-scratch.py --max-steps 20 --demo 3
"""

import re
import math
import json
import sqlite3
import argparse
import ast
import operator as op
from typing import Any, Callable, Optional


# ============================================================================
# 安全的数学表达式求值器
# ============================================================================

# 允许的运算符白名单
_ALLOWED_OPERATORS: dict = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
}

# 允许的数学函数
_MATH_FUNCTIONS: dict = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> Any:
    """递归求值 AST 节点，仅允许白名单中的运算符和函数。"""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left_val = _safe_eval_node(node.left)
        right_val = _safe_eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"不允许的运算符: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        operand_val = _safe_eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"不允许的一元运算符: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](operand_val)
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in _MATH_FUNCTIONS:
            raise ValueError(f"不允许的函数: {func_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return _MATH_FUNCTIONS[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in _MATH_FUNCTIONS:
            return _MATH_FUNCTIONS[node.id]
        raise ValueError(f"未定义的变量: {node.id}")
    elif isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    else:
        raise ValueError(f"不支持的 AST 节点类型: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """安全地求值数学表达式。

    使用 Python AST 模块解析表达式，仅允许白名单中的
    运算符和数学函数，防止代码注入攻击。

    Args:
        expression: 数学表达式字符串，如 "25 * 4 + 10"

    Returns:
        计算结果

    Raises:
        ValueError: 表达式包含不允许的操作
        SyntaxError: 表达式语法错误
    """
    # 移除空白字符
    expression = expression.strip()
    # 空表达式检查
    if not expression:
        raise ValueError("表达式为空")
    # 解析并求值
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise SyntaxError(f"表达式语法错误: {e.msg}")
    return _safe_eval_node(tree)


# ============================================================================
# 工具集 (5个演示工具)
# ============================================================================

# 模拟的天气数据库
_WEATHER_DB: dict = {
    "北京": {"today": "晴天，22°C，湿度45%，北风3级", "tomorrow": "多云，18°C，湿度60%", "yesterday": "小雨，15°C"},
    "上海": {"today": "阴天，25°C，湿度70%，东南风2级", "tomorrow": "小雨，23°C，湿度80%", "yesterday": "晴天，27°C"},
    "广州": {"today": "雷阵雨，28°C，湿度85%，南风4级", "tomorrow": "大雨，26°C，湿度90%", "yesterday": "多云，30°C"},
    "深圳": {"today": "晴转多云，27°C，湿度65%，东风3级", "tomorrow": "晴天，29°C，湿度55%", "yesterday": "阴天，26°C"},
    "杭州": {"today": "小雨，20°C，湿度75%，北风2级", "tomorrow": "阴转晴，22°C，湿度60%", "yesterday": "晴天，24°C"},
    "成都": {"today": "多云，23°C，湿度65%，无持续风向", "tomorrow": "晴天，25°C，湿度55%", "yesterday": "阴天，21°C"},
    "武汉": {"today": "晴，26°C，湿度50%，东风2级", "tomorrow": "多云，24°C，湿度60%", "yesterday": "晴天，27°C"},
    "西安": {"today": "扬沙，18°C，湿度30%，北风5级", "tomorrow": "晴，20°C，湿度35%", "yesterday": "多云，22°C"},
    "巴黎": {"today": "晴天，22°C，湿度50%", "tomorrow": "多云，19°C", "yesterday": "小雨，17°C"},
    "东京": {"today": "多云，24°C，湿度60%", "tomorrow": "晴天，26°C", "yesterday": "阴天，23°C"},
    "纽约": {"today": "雷阵雨，20°C，湿度80%", "tomorrow": "阴天，18°C", "yesterday": "晴天，22°C"},
    "伦敦": {"today": "小雨，15°C，湿度85%", "tomorrow": "阴天，16°C", "yesterday": "多云，17°C"},
    "悉尼": {"today": "晴，28°C，湿度40%", "tomorrow": "晴，30°C", "yesterday": "晴天，29°C"},
    "莫斯科": {"today": "阴天，10°C，湿度55%", "tomorrow": "小雨，8°C", "yesterday": "多云，12°C"},
    "新加坡": {"today": "雷阵雨，30°C，湿度90%", "tomorrow": "阵雨，29°C", "yesterday": "多云，31°C"},
}


def calculator(expression: str) -> str:
    """安全的数学计算器，可以执行各种数学运算。

    用途：当用户需要算术计算、数学表达式求值时使用此工具。
    不要用此工具处理非数学问题。

    Args:
        expression: 数学表达式，如 "25 * 4 + 10" 或 "sqrt(144) + sin(pi/2)"

    Returns:
        计算结果字符串
    """
    try:
        result = safe_eval(expression)
        # 整数结果不显示小数点
        if isinstance(result, float) and result == int(result) and not math.isinf(result):
            result = int(result)
        return f"计算结果: {expression} = {result}"
    except ZeroDivisionError:
        return f"错误: 表达式 '{expression}' 包含除以零操作"
    except (ValueError, SyntaxError) as e:
        return f"错误: 无法计算表达式 '{expression}' - {str(e)}"
    except Exception as e:
        return f"错误: 计算 '{expression}' 时发生未知错误 - {str(e)}"


# 模拟的搜索数据库
_SEARCH_DB: dict = {
    "故宫门票": "故宫博物院门票价格：旺季（4月1日-10月31日）60元/人，淡季（11月1日-3月31日）40元/人。"
                "学生票半价。需提前在官网或微信小程序预约。开放时间：8:30-17:00（旺季延长至17:30）。"
                "周一闭馆（法定节假日除外）。",
    "python教程": "Python 是一种解释型、面向对象的高级编程语言。由 Guido van Rossum 于 1991 年发布。"
                  "Python 以简洁易读的语法著称。广泛应用于 Web 开发（Django、Flask）、数据科学（NumPy、Pandas）、"
                  "人工智能（TensorFlow、PyTorch）、自动化运维等领域。",
    "长城": "长城是中国古代的军事防御工程，总长度超过 2.1 万公里。最著名的段落包括八达岭长城（北京）、"
           "慕田峪长城（北京）、司马台长城（北京）、山海关（河北）、嘉峪关（甘肃）。八达岭长城门票："
           "旺季 40 元，淡季 35 元。",
    "react agent": "ReAct (Reasoning + Acting) 是一种将推理和行动结合的 Agent 模式。由 Google Research 在 2022 年提出。"
                   "核心思想是让 LLM 交替生成推理轨迹（Thought）和行动（Action），"
                   "通过与外部工具交互来获取信息，最终给出有根据的回答。",
    "langchain": "LangChain 是一个用于构建 LLM 应用的开源框架。提供了 Chains、Agents、Tools、Memory 等核心组件。"
                "支持 Python 和 JavaScript。由 Harrison Chase 于 2022 年创建。",
    "openai": "OpenAI 是一家人工智能研究公司，开发了 GPT 系列大语言模型。"
             "主要产品包括 GPT-4、ChatGPT、DALL-E、Whisper。API 提供文本生成、图像生成、语音识别等服务。",
    "巴黎景点": "巴黎著名景点包括：埃菲尔铁塔（登塔 26.80 欧元）、卢浮宫（17 欧元，18 岁以下免费）、"
               "凯旋门（13 欧元）、巴黎圣母院（免费，火灾后修复中，预计 2024 年 12 月重新开放）、"
               "凡尔赛宫（19.50 欧元）、奥赛博物馆（16 欧元）。",
    "机器学习": "机器学习是人工智能的一个分支，让计算机从数据中学习规律。主要分为三类："
              "监督学习（分类、回归）、无监督学习（聚类、降维）、强化学习（决策优化）。"
              "常用算法包括：线性回归、决策树、SVM、神经网络、随机森林、XGBoost。",
}


def web_search(query: str) -> str:
    """模拟网络搜索，从知识库中检索信息。

    用途：当需要查找事实、获取信息、了解概念时使用此工具。
    不要用此工具进行数学计算。

    Args:
        query: 搜索关键词，如 "故宫门票价格" 或 "Python 简介"

    Returns:
        搜索结果字符串
    """
    # 精确匹配
    if query.strip() in _SEARCH_DB:
        return f"搜索结果 (关于 '{query}'):\n{_SEARCH_DB[query.strip()]}"

    # 模糊匹配：查找包含关键词的条目
    query_lower = query.strip().lower()
    matches = []
    for key, value in _SEARCH_DB.items():
        # 检查查询是否包含在键中或键包含在查询中
        if query_lower in key.lower() or any(word in key.lower() for word in query_lower.split()):
            matches.append((key, value))

    if matches:
        results = []
        for key, value in matches[:3]:  # 最多返回 3 条
            results.append(f"【{key}】: {value}")
        return f"搜索结果 (关于 '{query}'):\n" + "\n\n".join(results)

    # 无匹配时返回通用信息
    return f"搜索结果 (关于 '{query}'):\n未找到精确匹配的信息。这可能是一个需要进一步搜索的话题。"


def get_weather(city: str, date: str = "today") -> str:
    """获取指定城市的天气信息。

    用途：当用户询问天气情况时使用此工具。
    不要用此工具进行数学计算或搜索其他信息。

    Args:
        city: 城市名称，如 "北京"、"上海"、"巴黎"
        date: 日期，可选 "today"（默认）、"tomorrow"、"yesterday"

    Returns:
        天气信息字符串
    """
    if not city or not city.strip():
        return "错误: 请指定城市名称"

    city = city.strip()
    date = date.strip().lower() if date else "today"

    if date not in ("today", "tomorrow", "yesterday"):
        return f"错误: 不支持的日期 '{date}'，仅支持 today/tomorrow/yesterday"

    if city in _WEATHER_DB:
        weather_info = _WEATHER_DB[city].get(date, "暂无数据")
        return f"{city} 的天气 ({date}): {weather_info}"
    else:
        # 对于不在数据库中的城市，返回模拟数据
        import hashlib
        city_hash = int(hashlib.md5(city.encode()).hexdigest()[:8], 16)
        temp_base = 15 + (city_hash % 20)
        conditions = ["晴天", "多云", "阴天", "小雨", "阵雨"]
        condition = conditions[city_hash % len(conditions)]
        humidity = 40 + (city_hash % 50)
        return f"{city} 的天气 ({date}): {condition}，{temp_base}°C，湿度{humidity}% (模拟数据)"


# 内存中的 SQLite 数据库（演示用）
# 创建演示数据库表和数据
_DEMO_DB: sqlite3.Connection = sqlite3.connect(":memory:")
_DEMO_DB.row_factory = sqlite3.Row

# 初始化演示数据
_DEMO_DB.executescript("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL
    );

    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        manager TEXT,
        location TEXT
    );

    INSERT INTO employees VALUES
        (1, '张三', '技术部', 25000, '2020-03-15'),
        (2, '李四', '市场部', 22000, '2020-06-01'),
        (3, '王五', '技术部', 28000, '2019-01-10'),
        (4, '赵六', '人事部', 20000, '2021-09-20'),
        (5, '钱七', '市场部', 24000, '2020-11-05'),
        (6, '孙八', '技术部', 32000, '2018-07-01'),
        (7, '周九', '财务部', 26000, '2019-04-15'),
        (8, '吴十', '技术部', 18000, '2022-02-28');

    INSERT INTO departments VALUES
        (1, '技术部', '孙八', '3楼A区'),
        (2, '市场部', '钱七', '2楼B区'),
        (3, '人事部', '赵六', '1楼C区'),
        (4, '财务部', '周九', '2楼A区');
""")
_DEMO_DB.commit()


def database_query(sql: str) -> str:
    """在演示数据库中执行 SQL 查询。

    用途：当需要查询员工信息、部门数据、薪资统计时使用。
    不要用此工具进行数学计算。
    数据库包含 employees 表 (id, name, department, salary, hire_date)
    和 departments 表 (id, name, manager, location)。

    Args:
        sql: SQL 查询语句（仅支持 SELECT，不支持 INSERT/UPDATE/DELETE/DROP）

    Returns:
        查询结果字符串
    """
    sql = sql.strip()

    # 安全检查：只允许 SELECT 语句
    if not sql.upper().startswith("SELECT"):
        return "错误: 仅允许 SELECT 查询。不支持修改数据库操作（INSERT/UPDATE/DELETE/DROP）。"

    # 禁止危险操作
    dangerous_keywords = ["DROP", "ALTER", "CREATE", "INSERT", "UPDATE", "DELETE", "EXEC", "EXECUTE"]
    sql_upper = sql.upper()
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return f"错误: 查询中包含不允许的关键词 '{keyword}'。仅支持 SELECT 查询。"

    try:
        cursor = _DEMO_DB.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            return "查询结果: 未找到匹配的记录。"

        # 获取列名
        columns = [desc[0] for desc in cursor.description]

        # 格式化输出
        result_lines = [f"查询结果 (共 {len(rows)} 条记录):"]
        result_lines.append(" | ".join(columns))
        result_lines.append("-" * (len(" | ".join(columns))))

        for row in rows:
            result_lines.append(" | ".join(str(row[col]) for col in columns))

        return "\n".join(result_lines)

    except sqlite3.Error as e:
        return f"SQL 错误: {str(e)}"


def file_reader(filepath: str) -> str:
    """读取文件内容。

    用途：当需要查看文件内容时使用。
    不要用此工具进行数学计算或搜索。

    Args:
        filepath: 文件路径，可以是绝对路径或相对于当前目录的路径

    Returns:
        文件内容字符串（最多显示 2000 个字符）
    """
    import os

    try:
        if not os.path.exists(filepath):
            return f"错误: 文件不存在 - '{filepath}'"

        if not os.path.isfile(filepath):
            return f"错误: 路径不是文件 - '{filepath}'"

        # 检查文件大小（限制 10MB）
        file_size = os.path.getsize(filepath)
        if file_size > 10 * 1024 * 1024:
            return f"错误: 文件过大 ({file_size} 字节)，最大支持 10MB"

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # 截断过长的内容
        if len(content) > 2000:
            content = content[:2000] + f"\n\n... (文件被截断，总长度 {len(content)} 字符)"

        filename = os.path.basename(filepath)
        return f"文件内容 ({filename}, {file_size} 字节):\n\n{content}"

    except PermissionError:
        return f"错误: 没有权限读取文件 - '{filepath}'"
    except UnicodeDecodeError:
        return f"错误: 无法以 UTF-8 解码文件 - '{filepath}'，请确认是文本文件"
    except Exception as e:
        return f"错误: 读取文件 '{filepath}' 时发生错误 - {str(e)}"


# 工具注册表
def get_default_tools() -> dict:
    """返回默认工具集。"""
    return {
        "calculator": calculator,
        "web_search": web_search,
        "get_weather": get_weather,
        "database_query": database_query,
        "file_reader": file_reader,
    }


# ============================================================================
# ReAct Agent 实现
# ============================================================================

class ReActAgent:
    """实现 ReAct 模式的 Agent。

    ReAct 循环流程:
        Thought → Action → Tool → Observation → Thought → ... → Final Answer

    工作方式:
        1. 构建包含工具说明和 ReAct 格式要求的 System Prompt
        2. 将用户任务发送给 LLM
        3. 解析 LLM 输出中的 Action 指令
        4. 执行对应的工具，获取 Observation
        5. 将 Observation 注入对话，再次请求 LLM
        6. 重复步骤 3-5，直到 LLM 输出 Final Answer 或达到最大步数

    Attributes:
        llm: LLM 客户端（需实现 generate 方法）
        tools: 工具字典 {名称: 可调用对象}
        max_steps: 最大推理步数（防止无限循环）
        scratchpad: 存储所有推理步骤的列表
    """

    def __init__(self, llm_client, tools: dict, max_steps: int = 15):
        """初始化 ReAct Agent。

        Args:
            llm_client: LLM 客户端，需有 generate(messages) -> str 方法
            tools: 工具字典，{工具名: 可调用对象}
            max_steps: 最大推理步数，默认 15
        """
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps
        self.scratchpad: list = []  # 记录所有推理步骤

    def _build_tools_description(self) -> str:
        """构建工具描述文本，用于 System Prompt。

        Returns:
            格式化的工具列表字符串
        """
        tool_descriptions = []
        for name, func in self.tools.items():
            # 获取函数的文档字符串
            doc = func.__doc__ or "无描述"
            # 获取函数签名
            import inspect
            try:
                sig = inspect.signature(func)
                params = []
                for pname, param in sig.parameters.items():
                    if param.default is inspect.Parameter.empty:
                        params.append(pname)
                    else:
                        params.append(f"{pname}={param.default}")
                signature = f"{name}({', '.join(params)})"
            except (ValueError, TypeError):
                signature = name

            tool_descriptions.append(f"- {signature}: {doc.strip()}")

        return "\n".join(tool_descriptions)

    def _build_prompt(self, task: str) -> str:
        """构建 ReAct 格式的 System Prompt。

        这个 Prompt 是 ReAct 模式的核心：它告诉 LLM 必须按照
        Thought → Action → Observation 的格式输出，并且在获得
        足够信息后输出 Final Answer。

        Args:
            task: 用户任务描述

        Returns:
            完整的 System Prompt 字符串
        """
        tools_desc = self._build_tools_description()

        prompt = f"""你是一个具备推理和行动能力的 AI Agent。

## 可用工具
你可以使用以下工具来解决用户的问题：
{tools_desc}

## 输出格式要求
你必须严格按照以下格式交替进行"推理"和"行动"：

Thought: <你的推理过程——接下来应该做什么、为什么要这样做>
Action: <工具名称>[参数1=值1, 参数2=值2]
Observation: <系统会自动填充工具返回的结果>
... (这个 Thought/Action/Observation 循环可以重复多次)

当你已经收集到足够的信息来回答用户的问题时，你必须输出：
Final Answer: <你的最终回答——清晰、完整、直接回答用户的问题>

## 重要规则
1. 每次只能调用一个工具
2. 每次 Action 之后，你会收到 Observation，然后继续 Thought
3. 参数格式必须是: param1=value1, param2=value2
4. 如果工具返回错误，请尝试用不同的参数重试，或者换用其他工具
5. 如果已经获得足够信息，请立即输出 Final Answer，不要继续调用工具
6. 不要在没有工具返回错误时重复调用同一个工具
7. Final Answer 必须是针对用户问题的直接回答
8. 你必须确保在第三步之前有 Action 行为

现在，请解决以下问题：

Question: {task}

请开始推理："""
        return prompt

    def _parse_action(self, text: str) -> Optional[tuple]:
        """从 LLM 输出中解析 Action 指令。

        支持的格式:
            Action: tool_name[param1=value1, param2=value2]
            Action: tool_name[param1=value1]
            Action: tool_name[]

        解析逻辑:
            1. 使用正则表达式匹配 "Action: 工具名[参数]"
            2. 解析参数部分，支持逗号分隔的 key=value 格式
            3. 支持引号包裹的参数值

        Args:
            text: LLM 输出文本

        Returns:
            (工具名称, 参数字典) 或 None（未找到 Action）
        """
        # 匹配 Action: tool_name[params]
        pattern = r"Action:\s*(\w+)\s*\[(.*?)\]"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return None

        tool_name = match.group(1).strip()
        params_str = match.group(2).strip()

        # 解析参数
        params = {}
        if params_str:
            # 使用简单的状态机解析参数（处理引号内的逗号）
            current_key = ""
            current_value = ""
            in_quotes = False
            quote_char = None
            parsing_key = True
            param_parts = []

            for char in params_str:
                if parsing_key:
                    if char == "=":
                        parsing_key = False
                    else:
                        current_key += char
                else:
                    if char in ('"', "'") and not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char and in_quotes:
                        in_quotes = False
                        quote_char = None
                    elif char == "," and not in_quotes:
                        param_parts.append((current_key.strip(), current_value.strip()))
                        current_key = ""
                        current_value = ""
                        parsing_key = True
                        continue
                    else:
                        current_value += char

            # 添加最后一个参数
            if current_key.strip():
                param_parts.append((current_key.strip(), current_value.strip()))

            for key, value in param_parts:
                # 去除值两端的引号
                if value and value[0] in ('"', "'") and value[-1] in ('"', "'") and len(value) >= 2:
                    value = value[1:-1]
                params[key] = value

        return (tool_name, params)

    def _has_final_answer(self, text: str) -> Optional[str]:
        """检查并提取 Final Answer。

        Args:
            text: LLM 输出文本

        Returns:
            Final Answer 文本，如果不存在则返回 None
        """
        pattern = r"Final\s*Answer\s*:\s*(.*?)$"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _execute_tool(self, name: str, params: dict) -> str:
        """执行工具并返回 Observation。

        安全执行工具调用，捕获所有异常并返回格式化的错误信息。

        Args:
            name: 工具名称
            params: 参数字典

        Returns:
            工具执行结果（成功）或错误信息（失败）
        """
        if name not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"错误: 未找到工具 '{name}'。可用工具: {available}"

        tool_func = self.tools[name]

        try:
            # 执行工具
            result = tool_func(**params)
            return str(result)
        except TypeError as e:
            # 参数不匹配
            import inspect
            try:
                sig = inspect.signature(tool_func)
                expected_params = list(sig.parameters.keys())
                return f"错误: 工具 '{name}' 参数不匹配。期望参数: {expected_params}，提供参数: {list(params.keys())}。详情: {str(e)}"
            except (ValueError, TypeError):
                return f"错误: 工具 '{name}' 执行失败 - 参数类型错误: {str(e)}"
        except Exception as e:
            return f"错误: 工具 '{name}' 执行异常: {str(e)}"

    def run(self, task: str, verbose: bool = True) -> str:
        """主控制循环：执行 ReAct 推理。

        流程:
        1. 构建 System Prompt
        2. 循环: 调用 LLM → 解析 Action → 执行工具 → 注入 Observation
        3. 检测到 Final Answer 时停止
        4. 达到最大步数时强制终止

        Args:
            task: 用户任务描述
            verbose: 是否打印详细推理过程

        Returns:
            最终回答字符串
        """
        self.scratchpad = []

        # 构建初始 prompt
        system_prompt = self._build_prompt(task)

        # 对话历史（用于多轮交互）
        conversation = [system_prompt]

        if verbose:
            print("=" * 70)
            print("  ReAct Agent - 开始执行任务")
            print("=" * 70)
            print(f"\n[任务] {task}\n")
            print(f"[最大步数] {self.max_steps}\n")

        for step in range(1, self.max_steps + 1):
            if verbose:
                print(f"{'─' * 70}")
                print(f"  Step {step}/{self.max_steps}")
                print(f"{'─' * 70}")

            # 调用 LLM
            full_prompt = "\n".join(conversation)
            response = self.llm.generate(full_prompt)

            # 记录这一步
            step_record = {
                "step": step,
                "prompt_suffix": conversation[-1] if conversation else "",
                "response": response,
            }
            self.scratchpad.append(step_record)

            # 检查是否有 Final Answer
            final_answer = self._has_final_answer(response)
            if final_answer:
                if verbose:
                    print(f"\n{response}")
                    print(f"\n{'=' * 70}")
                    print(f"  任务完成! (共 {step} 步)")
                    print(f"{'=' * 70}")
                return final_answer

            # 解析 Action
            action = self._parse_action(response)
            if action is None:
                # 没有 Action 也没有 Final Answer —— 可能是 LLM 输出了不规范的格式
                if verbose:
                    print(f"\n[LLM 输出]\n{response}")
                    print(f"\n[警告] 未检测到 Action 或 Final Answer，尝试引导 LLM 继续...")

                # 提示 LLM 继续
                conversation.append(response)
                conversation.append(
                    "Observation: 请严格按照格式输出。"
                    "如果你已经可以回答问题，请输出 'Final Answer: ...'；"
                    "否则，请输出 'Action: tool_name[param=value]' 来使用工具。"
                )
                continue

            tool_name, params = action

            # 执行工具
            observation = self._execute_tool(tool_name, params)

            if verbose:
                print(f"\n[Thought] {self._extract_thought(response)}")
                print(f"\n[Action] {tool_name}({params})")
                print(f"\n[Observation] {observation}")

            # 更新对话历史
            conversation.append(response)
            conversation.append(f"Observation: {observation}")

            step_record["action"] = action
            step_record["observation"] = observation

        # 达到最大步数，强制终止
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  达到最大步数 ({self.max_steps}) - 强制终止")
            print(f"{'=' * 70}")

        # 最后尝试让 LLM 给出回答
        conversation.append(
            f"你已经执行了 {self.max_steps} 步，已经达到最大限制。"
            "请基于目前收集到的所有信息，给出你的最终回答。\n"
            "Final Answer:"
        )
        final_prompt = "\n".join(conversation)
        final_response = self.llm.generate(final_prompt)
        final_answer = self._has_final_answer(final_response)

        if final_answer:
            return final_answer
        return "无法在最大步数内完成任务。请增加 max_steps 或简化任务。"

    def _extract_thought(self, text: str) -> str:
        """从 LLM 输出中提取 Thought 部分（用于显示）。"""
        pattern = r"Thought\s*:\s*(.*?)(?=Action\s*:|Final\s*Answer\s*:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text[:200] + "..." if len(text) > 200 else text


# ============================================================================
# Mock LLM（模拟 LLM 响应，用于演示 ReAct 循环）
# ============================================================================

class MockLLM:
    """模拟 LLM 客户端，用于在不调用真实 API 的情况下演示 ReAct 循环。

    工作原理:
    - 根据用户任务的关键词，预定义多步推理路径
    - 第一步返回带 Action 的响应
    - 收到 Observation 后，下一步根据上下文返回新的 Action 或 Final Answer
    - 展示完整的 Thought→Action→Observation 循环

    这不是真实的 AI 推理，而是演示 ReAct Agent 的控制流程。
    """

    def __init__(self):
        """初始化 MockLLM，设置内置的任务响应模板。"""
        self._call_count = 0
        self._task_context = ""
        self._accumulated_info = []

    def generate(self, prompt: str) -> str:
        """模拟 LLM 的 generate 方法。

        根据当前调用次数和 prompt 内容，返回合适的响应。
        模拟真实的 ReAct 推理过程。

        Args:
            prompt: 完整的 prompt（包括 system prompt 和对话历史）

        Returns:
            模拟的 LLM 响应字符串
        """
        self._call_count += 1

        # 提取任务
        task = self._extract_task(prompt)

        # 检查对话轮数（从 Observation 出现次数推断已执行的工具调用次数）
        observation_count = prompt.count("Observation:")
        # 减去可能导致的重试提示
        retry_hints = prompt.count("请严格按照格式输出")
        effective_steps = observation_count - retry_hints

        # 根据任务类型选择响应策略
        if "25 * 4 + 10" in task or "25*4+10" in task or "计算" in task and ("25" in task or "15" in task):
            return self._math_demo_response(prompt, effective_steps)
        elif "天气" in task and "故宫" in task:
            return self._weather_search_demo_response(prompt, effective_steps)
        elif "天气" in task and "门票" in task:
            return self._weather_search_demo_response(prompt, effective_steps)
        elif "平均" in task or "薪资" in task or "工资" in task or "salary" in task.lower():
            return self._database_demo_response(prompt, effective_steps)
        elif "天气" in task:
            return self._weather_demo_response(prompt, effective_steps)
        elif "计算" in task or any(op in task for op in ["+", "-", "*", "/", "sqrt", "sin", "cos"]):
            return self._math_demo_response(prompt, effective_steps)
        elif "搜索" in task or "查" in task or "介绍" in task or "什么是" in task:
            return self._search_demo_response(prompt, effective_steps)
        elif "文件" in task or "读取" in task:
            return self._file_demo_response(prompt, effective_steps)
        else:
            return self._generic_demo_response(prompt, effective_steps)

    def _extract_task(self, prompt: str) -> str:
        """从 prompt 中提取任务描述。"""
        for marker in ["Question:", "问题：", "任务："]:
            if marker in prompt:
                parts = prompt.split(marker)
                if len(parts) > 1:
                    # 取到下一个明显标记之前
                    task_part = parts[1].strip()
                    for end_marker in ["请开始推理", "请解决以下问题", "Thought:", "Action:", "Final Answer:"]:
                        if end_marker in task_part:
                            task_part = task_part.split(end_marker)[0]
                    return task_part.strip()
        return ""

    def _math_demo_response(self, prompt: str, step: int) -> str:
        """模拟数学计算任务的响应。"""
        task = self._extract_task(prompt)

        if step == 0:
            # 第一步：分析任务，决定使用计算器
            return (
                f"Thought: 用户需要进行数学计算。我应该先检查表达式，然后使用 calculator 工具来求值。\n\n"
                f"首先，让我用计算器计算这个表达式。\n"
                f"Action: calculator[expression={task.replace('计算', '').replace(' ', '')}]"
            )
        elif step == 1:
            # 获取结果后，返回答案
            # 从 Observation 中提取计算结果
            return (
                f"Thought: 我已经通过 calculator 工具获得了计算结果。"
                f"现在我可以直接给用户提供答案了。\n\n"
                f"Final Answer: 根据计算器的结果，表达式的计算结果是当前工具返回的值。"
                f"计算过程已完成。"
            )
        else:
            return (
                f"Thought: 我已经完成了计算。\n\n"
                f"Final Answer: 计算已完成。"
            )

    def _weather_demo_response(self, prompt: str, step: int) -> str:
        """模拟天气查询任务的响应。"""
        task = self._extract_task(prompt)

        if step == 0:
            # 第一步：查询天气
            # 提取城市名
            cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "巴黎", "东京", "纽约"]
            found_city = "北京"
            for city in cities:
                if city in task:
                    found_city = city
                    break
            return (
                f"Thought: 用户想了解天气情况。让我先查询{found_city}的天气信息。\n"
                f"Action: get_weather[city={found_city}, date=today]"
            )
        elif step == 1:
            # 有了天气信息，给出穿着建议
            return (
                f"Thought: 我已经获取了天气数据。现在可以根据天气信息给用户提供完整的回答，"
                f"包括天气状况和穿着建议。\n\n"
                f"Final Answer: 根据查询结果，今天天气如上所示。"
                f"如果是晴天且温度较高（20°C以上），建议穿轻薄外套或单衣；"
                f"如果是阴天或温度适中（15-20°C），建议穿夹克或风衣；"
                f"如果是雨天，记得带伞并穿防滑鞋；"
                f"如果温度较低（15°C以下），建议穿厚外套或羽绒服。"
                f"请根据具体天气数据合理选择穿着。"
            )
        else:
            return f"Final Answer: 天气查询已完成，请参考之前获取的天气数据。"

    def _search_demo_response(self, prompt: str, step: int) -> str:
        """模拟信息搜索任务的响应。"""
        task = self._extract_task(prompt)

        if step == 0:
            # 提取搜索词
            search_terms = []
            for keyword in ["故宫", "python", "长城", "langchain", "巴黎", "机器学习", "openai", "react"]:
                if keyword.lower() in task.lower():
                    search_terms.append(keyword)

            if not search_terms:
                search_terms = [task[:20].strip()]

            query = " ".join(search_terms[:2])
            return (
                f"Thought: 用户想知道关于 '{query}' 的信息。让我使用 web_search 工具来查找相关数据。\n"
                f"Action: web_search[query={query}]"
            )
        elif step == 1:
            return (
                f"Thought: 我已经通过搜索获得了相关信息。现在可以基于搜索结果给用户一个全面的回答了。\n\n"
                f"Final Answer: 根据搜索结果，我已经获取了相关信息的详细内容。"
                f"上述信息涵盖了用户询问的内容。如果需要更多细节，建议进一步搜索。"
            )
        else:
            return f"Final Answer: 信息搜索已完成。"

    def _weather_search_demo_response(self, prompt: str, step: int) -> str:
        """模拟天气+搜索组合任务（需要多步推理）。"""
        task = self._extract_task(prompt)

        if step == 0:
            # 第一步：先查询天气
            cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "巴黎", "东京", "纽约"]
            found_city = "北京"
            for city in cities:
                if city in task:
                    found_city = city
                    break
            return (
                f"Thought: 用户提出了一个复合任务。我需要分步骤来解决。"
                f"首先查询{found_city}的天气，然后再进行其他搜索。\n"
                f"Action: get_weather[city={found_city}, date=today]"
            )
        elif step == 1:
            # 第二步：搜索另外的信息
            # 找到非天气相关的搜索词
            search_terms = []
            for keyword in ["故宫", "门票", "景点", "美食", "交通", "酒店"]:
                if keyword in task:
                    search_terms.append(keyword)

            if not search_terms:
                # 直接返回答案
                return (
                    f"Thought: 我已经获取了天气信息。现在可以给用户提供完整的回答了。\n\n"
                    f"Final Answer: 根据查询结果，天气情况已获取。"
                    f"建议用户关注实时天气更新，并根据天气合理安排出行。"
                )

            query = " ".join(search_terms)
            return (
                f"Thought: 天气信息已获取。现在让我搜索关于 '{query}' 的信息。\n"
                f"Action: web_search[query={query}]"
            )
        elif step == 2:
            return (
                f"Thought: 我已经获取了天气和搜索结果。现在可以综合这些信息给用户一个完整的回答了。\n\n"
                f"Final Answer: 综合天气和搜索结果，我已经获取了所有需要的信息。"
                f"请根据天气数据安排出行，并根据搜索结果了解相关景点或服务的详情。"
                f"建议提前规划行程，关注实时信息更新。"
            )
        else:
            return f"Final Answer: 复合任务已完成。"

    def _database_demo_response(self, prompt: str, step: int) -> str:
        """模拟数据库查询任务的响应。"""
        task = self._extract_task(prompt)

        if step == 0:
            # 分析 SQL 需求
            if "平均" in task or "average" in task.lower() or "avg" in task.lower():
                sql = "SELECT department, AVG(salary) as avg_salary, COUNT(*) as count FROM employees GROUP BY department"
            elif "总" in task or "sum" in task.lower():
                sql = "SELECT department, SUM(salary) as total_salary FROM employees GROUP BY department"
            elif "最高" in task or "max" in task.lower():
                sql = "SELECT * FROM employees WHERE salary = (SELECT MAX(salary) FROM employees)"
            elif "所有" in task or "全部" in task or "列表" in task:
                sql = "SELECT * FROM employees ORDER BY department, name"
            else:
                sql = "SELECT * FROM employees"

            return (
                f"Thought: 用户想查询数据库中的数据。我需要构建合适的 SQL 查询。"
                f"根据用户的问题，我将查询员工信息。\n"
                f"Action: database_query[sql={sql}]"
            )
        elif step == 1:
            return (
                f"Thought: 数据库查询已完成。我可以基于查询结果给用户提供分析报告了。\n\n"
                f"Final Answer: 根据数据库查询结果，我已经获取了所需的数据。"
                f"以上数据显示了员工/部门的详细信息。如果需要进一步分析，请告诉我具体需求。"
            )
        else:
            return f"Final Answer: 数据库查询已完成。"

    def _file_demo_response(self, prompt: str, step: int) -> str:
        """模拟文件读取任务的响应。"""
        if step == 0:
            return (
                f"Thought: 用户想要读取文件。我需要使用 file_reader 工具来获取文件内容。\n"
                f"Action: file_reader[filepath=02-react-agent-from-scratch.py]"
            )
        elif step == 1:
            return (
                f"Thought: 文件内容已获取。我可以总结文件内容或回答用户的问题了。\n\n"
                f"Final Answer: 文件读取成功。文件内容已显示在上方。"
                f"你可以查看文件的具体内容来了解详细信息。"
            )
        else:
            return f"Final Answer: 文件读取操作已完成。"

    def _generic_demo_response(self, prompt: str, step: int) -> str:
        """通用任务的模拟响应。"""
        task = self._extract_task(prompt)

        if step == 0:
            return (
                f"Thought: 用户提出了一个问题。让我先分析一下任务需求。"
                f"我可能需要搜索相关信息或进行计算。让我先尝试搜索。\n"
                f"Action: web_search[query={task[:50]}]"
            )
        elif step == 1:
            return (
                f"Thought: 已经获取了一些信息。现在我有了足够的信息来回答问题。\n\n"
                f"Final Answer: 根据获取的信息，我已经能够回答你的问题。"
                f"如果还需要更详细的信息，请告诉我具体方面。"
            )
        else:
            return f"Final Answer: 任务已完成。"


# ============================================================================
# 演示函数
# ============================================================================

def run_demo(demo_id: int):
    """运行指定的演示任务。

    三个演示分别展示:
    1. 简单数学计算（单步工具调用）
    2. 信息查询（两步推理）
    3. 多步骤复合任务（天气 + 搜索）

    Args:
        demo_id: 演示编号 (1-3)
    """
    tools = get_default_tools()
    model = MockLLM()

    if demo_id == 1:
        task = "计算 25 * 4 + 10 的结果"
        title = "Demo 1: 简单数学计算"
        description = "演示单步工具调用 —— 一次 calculator 调用即可得到答案"
    elif demo_id == 2:
        task = "查询一下故宫的门票价格是多少"
        title = "Demo 2: 信息查询"
        description = "演示单步工具调用 —— 一次 web_search 调用获取信息"
    elif demo_id == 3:
        task = "帮我查询北京的天气，然后搜索故宫门票价格和景点信息"
        title = "Demo 3: 多步骤复合任务"
        description = "演示多步推理 —— 先查天气，再搜索门票，最后综合回答"
    elif demo_id == 4:
        task = "查询技术部所有员工的平均薪资"
        title = "Demo 4: 数据库查询"
        description = "演示数据库工具调用 —— SQL 查询员工薪资数据"
    elif demo_id == 5:
        task = "读取 02-react-agent-from-scratch.py 文件的内容"
        title = "Demo 5: 文件读取"
        description = "演示文件工具调用 —— 读取本地文件内容"
    else:
        print(f"无效的演示编号: {demo_id}（支持 1-5）")
        return

    print(f"\n{'#' * 70}")
    print(f"  {title}")
    print(f"  {description}")
    print(f"{'#' * 70}")

    agent = ReActAgent(llm_client=model, tools=tools, max_steps=10)
    result = agent.run(task, verbose=True)

    print(f"\n{'=' * 70}")
    print(f"  [最终回答]")
    print(f"{'=' * 70}")
    print(result)

    print(f"\n{'=' * 70}")
    print(f"  [Scratchpad 摘要]")
    print(f"{'=' * 70}")
    for record in agent.scratchpad:
        step_num = record["step"]
        has_action = "action" in record
        has_answer = "Final Answer" in record.get("response", "")
        action_desc = f"Action: {record.get('action', 'N/A')}" if has_action else "N/A"
        print(f"  Step {step_num}: {action_desc}")


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    """主入口函数。

    支持两种运行模式:
    1. 演示模式：运行内置的 Demo 任务
    2. 交互/单任务模式：运行用户指定的任务
    """
    parser = argparse.ArgumentParser(
        description="ReAct Agent - 从零实现 (From-Scratch Implementation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 运行所有演示
    python 02-react-agent-from-scratch.py --demo all

    # 运行单个演示
    python 02-react-agent-from-scratch.py --demo 1

    # 运行自定义任务
    python 02-react-agent-from-scratch.py --task "What is 25 * 4 + 10?"

    # 运行中文任务
    python 02-react-agent-from-scratch.py --task "查询北京天气并搜索故宫门票价格"

    # 指定最大步数
    python 02-react-agent-from-scratch.py --task "复杂的多步任务" --max-steps 20
        """,
    )

    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help="要执行的任务描述（中英文均可）",
    )
    parser.add_argument(
        "--demo", "-d",
        type=str,
        default=None,
        help="运行演示任务 (1-5 或 all)",
    )
    parser.add_argument(
        "--max-steps", "-m",
        type=int,
        default=10,
        help="最大推理步数 (默认: 10)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="安静模式，不打印详细推理过程",
    )

    args = parser.parse_args()

    tools = get_default_tools()

    # 演示模式
    if args.demo:
        if args.demo.lower() == "all":
            for i in range(1, 6):
                run_demo(i)
                print("\n\n")
        else:
            try:
                demo_id = int(args.demo)
                run_demo(demo_id)
            except ValueError:
                print(f"错误: 无效的演示编号 '{args.demo}'。请使用 1-5 或 'all'。")
                return
        return

    # 自定义任务模式
    if args.task:
        task = args.task
    else:
        # 交互模式
        print("\n" + "=" * 70)
        print("  ReAct Agent - 交互模式")
        print("  输入 'exit' 或 'quit' 退出")
        print("  输入 'demo' 运行所有演示")
        print("=" * 70 + "\n")
        task = input("请输入你的任务: ").strip()

        if not task:
            print("任务不能为空。")
            return

        if task.lower() in ("exit", "quit"):
            print("再见！")
            return

        if task.lower() == "demo":
            for i in range(1, 6):
                run_demo(i)
                print("\n\n")
            return

    # 执行任务
    model = MockLLM()
    agent = ReActAgent(llm_client=model, tools=tools, max_steps=args.max_steps)

    print(f"\n[任务] {task}")
    result = agent.run(task, verbose=not args.quiet)

    if args.quiet:
        print(result)
    else:
        print(f"\n{'=' * 70}")
        print(f"  [最终回答]")
        print(f"{'=' * 70}")
        print(result)


if __name__ == "__main__":
    main()
