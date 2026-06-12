# 陷阱 2：工具调用失败 (Tool Call Failure)

## 概述

Agent 依赖工具来获取信息或执行操作。当工具调用失败时（参数错误、网络超时、权限不足、除零异常等），如果 Agent 不能优雅地恢复，就会传播错误、给出错误答案、或者直接崩溃。

---

## 症状 (Symptoms)

1. **Agent 直接崩溃**：工具抛出未捕获异常，导致整个 Agent 循环终止
2. **错误传播**：工具的原始错误信息直接传递给 LLM，LLM 将其当作正常数据用于推理
3. **无法恢复**：第一次工具调用失败后，Agent 不知道如何调整参数重试
4. **错误回答**：Agent 基于错误的 Observation 生成了看似合理但实际错误的 Final Answer
5. **静默失败**：工具返回了空字符串或含糊不清的错误信息，Agent 将其解读为"没有数据"

---

## 根本原因 (Root Cause)

### 1. 工具缺乏异常处理

工具函数直接暴露原始异常（如 `ZeroDivisionError`、`KeyError`），没有包装为 Agent 可理解的错误信息。

### 2. Agent 未处理工具错误

Agent 的 `_execute_tool` 方法没有 try-except，也没有对错误返回值做特殊标记。

### 3. 参数校验缺失

Agent 在调用工具前不验证参数的有效性，将格式错误或范围外的参数直接传递给工具。

### 4. 缺少重试和回退机制

一次失败后没有重试策略，也没有备选工具可用。

---

## 真实场景 (Real Scenario)

### 场景描述

用户询问：**"帮我计算 100 除以 0 再乘以 5"**

Agent 的错误行为：

```
Step 1:
  Thought: 用户需要计算 100/0*5，让我使用计算器
  Action: calculator[expression=100/0*5]
  Observation: ZeroDivisionError: division by zero
  → Agent 崩溃！整个循环终止。

# 或者更糟的情况：
Step 1:
  Thought: 计算这个表达式
  Action: calculator[expression=100/0*5]
  Observation: 计算结果: 100/0*5 = 0  ← 工具没有正确处理除零，返回了错误值

Step 2:
  Thought: 100/0*5 的结果是 0，这似乎是对的...
  Final Answer: 100 除以 0 再乘以 5 的结果是 0
  → 完全错误的答案！
```

### 更多失败场景

| 工具 | 失败方式 | 原始错误 | 用户影响 |
|------|---------|---------|---------|
| calculator | 除零 | ZeroDivisionError | Agent 崩溃 |
| web_search | 网络超时 | TimeoutError | 无搜索结果 |
| get_weather | 城市名拼写错误 | KeyError | 返回通用数据 |
| database_query | SQL 语法错误 | sqlite3.OperationalError | 查询失败 |
| file_reader | 文件不存在 | FileNotFoundError | 无法读取 |
| API 工具 | 认证过期 | HTTP 401 | 调用被拒 |

---

## 解决方案

### 方案 1：每个工具内部 Try-Except，返回结构化错误

最基础也是最重要的防线：每个工具函数内部捕获所有异常，返回结构化 JSON 错误。

```python
import json
import traceback

class ToolError:
    """标准化的工具错误响应。"""

    def __init__(self, message: str, error_type: str, suggestion: str = "",
                 retryable: bool = False, fallback_tool: str = ""):
        self.message = message
        self.error_type = error_type
        self.suggestion = suggestion
        self.retryable = retryable
        self.fallback_tool = fallback_tool

    def to_response(self) -> str:
        """生成给 LLM 看的结构化错误响应。"""
        response = {
            "success": False,
            "error": self.message,
            "error_type": self.error_type,
        }
        if self.suggestion:
            response["suggestion"] = self.suggestion
        if self.retryable:
            response["retryable"] = True
            response["retry_hint"] = "此错误可以重试，请调整参数后再次调用"
        if self.fallback_tool:
            response["fallback_tool"] = self.fallback_tool
            response["fallback_hint"] = f"建议尝试使用工具: {self.fallback_tool}"

        return json.dumps(response, ensure_ascii=False, indent=2)


def safe_calculator(expression: str) -> str:
    """带完整错误处理的数学计算器。

    相比原始版本的改进:
    - 捕获所有异常类型
    - 返回结构化 JSON 错误（而非崩溃）
    - 提供修复建议
    """
    import ast
    import math
    import operator as op

    _ALLOWED_OPS = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow,
    }

    def _eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            # 除零检查
            if op_type == ast.Div and right == 0:
                raise ZeroDivisionError(f"除数不能为零: {left}/{right}")
            return _ALLOWED_OPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            return operand
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else "unknown"
            raise ValueError(f"不支持的函数: {func_name}()")
        elif isinstance(node, ast.Expression):
            return _eval_node(node.body)
        else:
            raise ValueError(f"不支持的语法: {type(node).__name__}")

    try:
        # 输入验证
        if not expression or not expression.strip():
            return ToolError(
                message="表达式为空",
                error_type="invalid_input",
                suggestion="请提供一个有效的数学表达式，例如 '2 + 3' 或 'sqrt(16)'",
                retryable=True,
            ).to_response()

        expression = expression.strip()

        # 长度限制
        if len(expression) > 500:
            return ToolError(
                message="表达式过长（超过 500 字符）",
                error_type="input_too_long",
                suggestion="请简化表达式，分解为多个较小的计算步骤",
                retryable=True,
            ).to_response()

        # 解析并求值
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)

        # 结果验证
        if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
            return ToolError(
                message=f"计算结果为 {'NaN' if math.isnan(result) else '无穷大'}（{expression}）",
                error_type="math_error",
                suggestion="表达式在数学上无定义或溢出，请检查输入值",
                retryable=False,
            ).to_response()

        # 格式化结果
        if isinstance(result, float) and result == int(result) and not math.isinf(result):
            result = int(result)

        return json.dumps({
            "success": True,
            "expression": expression,
            "result": result,
        }, ensure_ascii=False)

    except ZeroDivisionError as e:
        return ToolError(
            message=str(e),
            error_type="division_by_zero",
            suggestion="除数不能为零。请检查表达式，避免除以零。",
            retryable=True,
        ).to_response()
    except (ValueError, SyntaxError) as e:
        return ToolError(
            message=f"表达式无效: {str(e)}",
            error_type="invalid_expression",
            suggestion="请检查表达式语法。支持的运算符: +, -, *, /, **。示例: '2 + 3 * 4'",
            retryable=True,
        ).to_response()
    except OverflowError as e:
        return ToolError(
            message=f"数值溢出: {str(e)}",
            error_type="overflow",
            suggestion="计算结果过大。请使用较小的数值，或分步计算。",
            retryable=False,
        ).to_response()
    except Exception as e:
        return ToolError(
            message=f"未知计算错误: {str(e)}",
            error_type="unknown",
            suggestion="请尝试用不同的表达式重新计算。",
            retryable=True,
        ).to_response()


def safe_web_search(query: str) -> str:
    """带完整错误处理的网络搜索。

    处理网络超时、无结果、参数错误等情况。
    """
    import time

    # 输入验证
    if not query or not query.strip():
        return ToolError(
            message="搜索关键词为空",
            error_type="invalid_input",
            suggestion="请提供有效的搜索关键词",
            retryable=True,
        ).to_response()

    query = query.strip()

    # 长度限制
    if len(query) > 300:
        query = query[:300]
        # 不报错，但截断

    try:
        # 模拟网络请求（实际应用中使用 httpx 或 requests）
        # 这里用我们之前定义的 _SEARCH_DB 或类似逻辑
        results = _perform_search(query)

        if not results:
            return ToolError(
                message=f"未找到关于 '{query}' 的搜索结果",
                error_type="no_results",
                suggestion=f"请尝试以下操作: 1) 使用更通用的关键词 2) 检查拼写 3) 尝试英文搜索",
                retryable=True,
                fallback_tool="wikipedia_search",
            ).to_response()

        return json.dumps({
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results,
        }, ensure_ascii=False)

    except TimeoutError:
        return ToolError(
            message=f"搜索 '{query}' 超时（超过 10 秒无响应）",
            error_type="timeout",
            suggestion="请尝试以下操作: 1) 使用更简短的查询 2) 稍后重试 3) 使用其他搜索工具",
            retryable=True,
            fallback_tool="cached_search",
        ).to_response()
    except ConnectionError:
        return ToolError(
            message=f"网络连接失败，无法完成搜索 '{query}'",
            error_type="network_error",
            suggestion="请检查网络连接，或使用离线缓存数据。",
            retryable=True,
        ).to_response()
    except Exception as e:
        return ToolError(
            message=f"搜索异常: {str(e)}",
            error_type="search_error",
            suggestion="请调整搜索关键词后重试。",
            retryable=True,
        ).to_response()


def _perform_search(query: str) -> list:
    """模拟搜索（实际应用中使用真实 API）。"""
    # 这里是演示代码，实际中替换为真实搜索 API 调用
    return [{"title": "示例结果", "snippet": "这是一个模拟的搜索结果"}]
```

### 方案 2：Agent 级错误处理器

在 Agent 层面的 `_execute_tool` 方法中增加错误处理，将工具错误重新格式化为 LLM 能理解的 Observation。

```python
class ReActAgentWithErrorHandler(ReActAgent):
    """带有 Agent 级别错误处理的 ReAct Agent。"""

    def __init__(self, llm_client, tools: dict, max_steps: int = 15,
                 max_retries: int = 2):
        super().__init__(llm_client, tools, max_steps)
        self.max_retries = max_retries
        self.retry_count: dict[str, int] = {}  # 每步的重试计数

    def _execute_tool_with_recovery(self, tool_name: str, params: dict,
                                     step: int) -> str:
        """执行工具，带自动恢复逻辑。

        增强功能:
        1. 参数预校验
        2. 执行工具 + 异常捕获
        3. 解析错误响应（JSON 或纯文本）
        4. 对可重试错误自动重试
        5. 对不可重试错误提供回退建议

        Args:
            tool_name: 工具名称
            params: 参数字典
            step: 当前步骤编号（用于重试计数）

        Returns:
            增强后的 Observation 字符串（包含错误上下文和建议）
        """
        retry_key = f"{step}_{tool_name}"
        self.retry_count[retry_key] = self.retry_count.get(retry_key, 0)

        # === 阶段 1: 参数预校验 ===
        validation_error = self._validate_params(tool_name, params)
        if validation_error:
            return (
                f"[参数校验失败] {validation_error}\n"
                f"请检查参数格式后重试。"
            )

        # === 阶段 2: 执行工具 ===
        tool_func = self.tools[tool_name]
        try:
            result = tool_func(**params)
        except TypeError as e:
            # 参数数量或名称不匹配
            expected = self._get_tool_signature(tool_name)
            return (
                f"[参数错误] 工具 '{tool_name}' 参数不匹配: {str(e)}\n"
                f"期望参数: {expected}\n"
                f"实际提供: {list(params.keys())}\n"
                f"请修正参数后重试。"
            )
        except Exception as e:
            result = json.dumps({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }, ensure_ascii=False)

        # === 阶段 3: 解析结果 ===
        observation = str(result)

        # 尝试解析为 JSON 以检查是否是结构化错误
        try:
            error_data = json.loads(observation)
        except (json.JSONDecodeError, TypeError):
            error_data = None

        # === 阶段 4: 错误恢复 ===
        if error_data and isinstance(error_data, dict):
            if not error_data.get("success", True):
                retryable = error_data.get("retryable", False)
                error_msg = error_data.get("error", "未知错误")
                suggestion = error_data.get("suggestion", "")
                fallback = error_data.get("fallback_tool", "")

                current_retries = self.retry_count[retry_key]

                if retryable and current_retries < self.max_retries:
                    # 自动重试逻辑
                    self.retry_count[retry_key] += 1

                    # 尝试调整参数
                    adjusted_params = self._adjust_params_for_retry(
                        tool_name, params, error_data
                    )

                    retry_msg = (
                        f"[工具错误 - 第 {current_retries + 1}/{self.max_retries} 次重试]\n"
                        f"错误类型: {error_data.get('error_type', 'unknown')}\n"
                        f"错误详情: {error_msg}\n"
                    )
                    if suggestion:
                        retry_msg += f"建议: {suggestion}\n"
                    if adjusted_params != params:
                        retry_msg += f"自动调整参数: {params} → {adjusted_params}\n"
                        retry_msg += f"正在使用新参数重试...\n"

                    # 递归重试
                    retry_result = self._execute_tool_with_recovery(
                        tool_name, adjusted_params, step
                    )
                    return retry_msg + "\n" + retry_result

                elif fallback and fallback in self.tools:
                    # 尝试回退工具
                    return (
                        f"[工具 '{tool_name}' 失败] {error_msg}\n"
                        f"[自动回退] 尝试使用备选工具 '{fallback}'...\n"
                        f"{self._execute_tool_with_recovery(fallback, params, step)}"
                    )

                else:
                    # 不可重试，返回增强的错误信息
                    enhanced_error = (
                        f"[工具执行失败] '{tool_name}' 返回错误\n"
                        f"错误详情: {error_msg}\n"
                        f"错误类型: {error_data.get('error_type', 'unknown')}\n"
                    )
                    if suggestion:
                        enhanced_error += f"修复建议: {suggestion}\n"
                    enhanced_error += (
                        f"\n你可以选择:\n"
                        f"1. 调整参数后重试调用 '{tool_name}'\n"
                        f"2. 使用其他工具获取所需信息\n"
                        f"3. 如果已有足够信息，直接输出 Final Answer\n"
                    )
                    return enhanced_error

        # === 阶段 5: 成功返回 ===
        return f"[工具调用成功] {tool_name}:\n{observation}"

    def _validate_params(self, tool_name: str, params: dict) -> str:
        """预校验工具参数。

        在调用工具之前检查参数的有效性，避免常见的参数错误。

        Returns:
            错误描述字符串，或空字符串表示验证通过
        """
        # 可为不同工具定义参数约束
        param_constraints = {
            "calculator": {
                "expression": {
                    "required": True,
                    "min_length": 1,
                    "max_length": 500,
                }
            },
            "web_search": {
                "query": {
                    "required": True,
                    "min_length": 1,
                    "max_length": 300,
                }
            },
            "get_weather": {
                "city": {
                    "required": True,
                    "min_length": 1,
                }
            },
            "database_query": {
                "sql": {
                    "required": True,
                    "min_length": 1,
                }
            },
        }

        constraints = param_constraints.get(tool_name, {})

        for param_name, rules in constraints.items():
            value = params.get(param_name, None)

            if rules.get("required") and (value is None or str(value).strip() == ""):
                return f"参数 '{param_name}' 是必填的，但未提供或为空。"

            if value is not None:
                str_value = str(value)
                if "min_length" in rules and len(str_value) < rules["min_length"]:
                    return f"参数 '{param_name}' 长度不足（最小 {rules['min_length']} 字符）。"
                if "max_length" in rules and len(str_value) > rules["max_length"]:
                    return f"参数 '{param_name}' 长度超限（最大 {rules['max_length']} 字符）。"

        return ""  # 验证通过

    def _adjust_params_for_retry(self, tool_name: str, original_params: dict,
                                  error_data: dict) -> dict:
        """根据错误类型智能调整参数用于重试。

        这是 Agent 自动恢复的关键：不是盲目重试，而是有针对性地调整参数。
        """
        error_type = error_data.get("error_type", "")
        adjusted = dict(original_params)  # 浅拷贝

        if error_type == "division_by_zero":
            # 除零错误：无法自动修复，原样返回（Agent 会收到错误提示）
            pass

        elif error_type == "invalid_expression":
            # 表达式语法错误：尝试添加括号或简化
            if "expression" in adjusted:
                expr = adjusted["expression"]
                # 简单的修复尝试（实际中可以有更复杂的逻辑）
                # 例如：去掉多余空格、确保括号配对
                expr = expr.replace("  ", " ")
                if expr.count("(") != expr.count(")"):
                    # 尝试在末尾补充右括号
                    diff = expr.count("(") - expr.count(")")
                    expr = expr + ")" * diff
                adjusted["expression"] = expr

        elif error_type == "timeout":
            # 超时：尝试缩短查询
            if "query" in adjusted:
                query = adjusted["query"]
                if len(query) > 100:
                    adjusted["query"] = query[:100]

        elif error_type == "no_results":
            # 无结果：使用更短的查询
            if "query" in adjusted:
                query = adjusted["query"]
                words = query.split()
                if len(words) > 3:
                    adjusted["query"] = " ".join(words[:3])

        elif error_type == "invalid_input":
            # 输入无效：尝试去除特殊字符
            for key in adjusted:
                if isinstance(adjusted[key], str):
                    # 去除潜在的问题字符
                    adjusted[key] = adjusted[key].strip().strip("'\"`")

        return adjusted

    def _get_tool_signature(self, tool_name: str) -> str:
        """获取工具的函数签名（用于错误提示）。"""
        import inspect
        func = self.tools.get(tool_name)
        if func is None:
            return "未知"
        try:
            sig = inspect.signature(func)
            return str(sig)
        except (ValueError, TypeError):
            return str(func.__name__)
```

### 方案 3：自动参数校验装饰器

使用 Python 装饰器在工具执行前自动校验参数，避免工具内部因参数问题崩溃。

```python
from functools import wraps
from typing import get_type_hints, Any
import inspect

def validate_params(**constraints):
    """参数校验装饰器。

    用法:
        @validate_params(
            expression={"required": True, "type": str, "min_length": 1, "max_length": 500},
        )
        def calculator(expression: str) -> str:
            ...

    在函数执行前自动检查参数是否满足约束。
    校验失败时返回结构化错误而非抛出异常。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            all_params = bound.arguments

            for param_name, rules in constraints.items():
                value = all_params.get(param_name)

                # 必填检查
                if rules.get("required", False) and (value is None or
                    (isinstance(value, str) and value.strip() == "")):
                    return json.dumps({
                        "success": False,
                        "error": f"参数 '{param_name}' 是必填的，但未提供",
                        "error_type": "invalid_input",
                        "suggestion": f"请为 '{param_name}' 提供有效值",
                        "retryable": True,
                    }, ensure_ascii=False)

                if value is None:
                    continue

                # 类型检查
                expected_type = rules.get("type")
                if expected_type and not isinstance(value, expected_type):
                    return json.dumps({
                        "success": False,
                        "error": f"参数 '{param_name}' 类型错误（期望 {expected_type.__name__}，实际 {type(value).__name__}）",
                        "error_type": "type_error",
                        "suggestion": f"请确保 '{param_name}' 为 {expected_type.__name__} 类型",
                        "retryable": True,
                    }, ensure_ascii=False)

                # 长度检查（字符串）
                if isinstance(value, str):
                    if "min_length" in rules and len(value) < rules["min_length"]:
                        return json.dumps({
                            "success": False,
                            "error": f"参数 '{param_name}' 长度不足",
                            "error_type": "invalid_input",
                            "retryable": True,
                        }, ensure_ascii=False)
                    if "max_length" in rules and len(value) > rules["max_length"]:
                        return json.dumps({
                            "success": False,
                            "error": f"参数 '{param_name}' 长度超限（最大 {rules['max_length']}）",
                            "error_type": "input_too_long",
                            "suggestion": f"请缩短 '{param_name}' 的长度",
                            "retryable": True,
                        }, ensure_ascii=False)

                # 数值范围检查
                if isinstance(value, (int, float)):
                    if "min_value" in rules and value < rules["min_value"]:
                        return json.dumps({
                            "success": False,
                            "error": f"参数 '{param_name}' 值过小（最小 {rules['min_value']}）",
                            "error_type": "out_of_range",
                            "retryable": True,
                        }, ensure_ascii=False)
                    if "max_value" in rules and value > rules["max_value"]:
                        return json.dumps({
                            "success": False,
                            "error": f"参数 '{param_name}' 值过大（最大 {rules['max_value']}）",
                            "error_type": "out_of_range",
                            "retryable": True,
                        }, ensure_ascii=False)

                # 枚举检查
                allowed = rules.get("enum")
                if allowed and value not in allowed:
                    return json.dumps({
                        "success": False,
                        "error": f"参数 '{param_name}' 值 '{value}' 不在允许范围内",
                        "error_type": "invalid_choice",
                        "suggestion": f"允许的值: {allowed}",
                        "retryable": True,
                    }, ensure_ascii=False)

            # 校验通过，执行原函数
            return func(*args, **kwargs)

        return wrapper
    return decorator


# 使用示例
@validate_params(
    city={"required": True, "type": str, "min_length": 1, "max_length": 50},
    date={"required": False, "type": str, "enum": ["today", "tomorrow", "yesterday"]},
)
def validated_weather(city: str, date: str = "today") -> str:
    """带参数校验的天气查询工具。"""
    # 由于装饰器已经校验了参数，这里可以安全地执行业务逻辑
    weather_data = _get_weather_data(city, date)
    return json.dumps({
        "success": True,
        "city": city,
        "date": date,
        "weather": weather_data,
    }, ensure_ascii=False)
```

### 方案 4：回退工具链 (Fallback Chain)

当一个工具失败时，自动尝试备选工具。

```python
class FallbackChain:
    """工具回退链：按优先级依次尝试工具，直到有一个成功。

    用法:
        chain = FallbackChain([
            ("primary_search", primary_search_func),
            ("cached_search", cached_search_func),
            ("basic_search", basic_search_func),
        ])
        result = chain.execute(query="python tutorial")
    """

    def __init__(self, tools: list[tuple[str, callable]],
                 max_retries_per_tool: int = 1):
        """
        Args:
            tools: [(工具名, 工具函数)] 按优先级降序排列
            max_retries_per_tool: 每个工具的最大重试次数
        """
        self.tools = tools
        self.max_retries_per_tool = max_retries_per_tool

    def execute(self, **kwargs) -> str:
        """依次尝试每个工具，返回第一个成功的结果。"""
        errors = []

        for tool_name, tool_func in self.tools:
            for attempt in range(self.max_retries_per_tool + 1):
                try:
                    result = tool_func(**kwargs)
                    # 检查是否为错误响应
                    try:
                        result_data = json.loads(result)
                        if isinstance(result_data, dict) and not result_data.get("success", True):
                            errors.append(f"{tool_name} (attempt {attempt+1}): {result_data.get('error', 'unknown')}")
                            continue  # 尝试下一次
                    except (json.JSONDecodeError, TypeError):
                        pass  # 不是 JSON，当作正常结果

                    # 成功
                    if len(errors) > 0:
                        result = f"[经过 {len(errors)} 次失败后由 {tool_name} 成功]\n{result}"
                    return result

                except Exception as e:
                    errors.append(f"{tool_name} (attempt {attempt+1}): {str(e)}")

        # 所有工具都失败
        return json.dumps({
            "success": False,
            "error": "所有备选工具均调用失败",
            "error_type": "all_failed",
            "attempted_tools": [t[0] for t in self.tools],
            "error_log": errors,
            "suggestion": "请手动提供所需信息，或稍后重试。",
        }, ensure_ascii=False, indent=2)


# 使用示例：搜索回退链
search_fallback = FallbackChain([
    ("web_search", web_search),
    ("file_reader", file_reader),  # 如果网络搜索失败，尝试本地文件
])
```

---

## 预防清单 (Checklist)

在部署 Agent 工具前，请确保：

- [ ] 1. 每个工具函数都有完整的 try-except，覆盖所有可能的异常类型
- [ ] 2. 工具返回结构化 JSON：`{"success": True/False, "error": "...", "suggestion": "..."}`
- [ ] 3. Agent 的 `_execute_tool` 方法会解析错误响应，并给 LLM 提供恢复建议
- [ ] 4. 工具函数包含输入参数校验（类型检查、范围检查、格式校验）
- [ ] 5. 对关键工具有回退工具链（主工具失败 → 备选工具 → 兜底工具）
- [ ] 6. 除零、网络超时、权限错误等常见失败有专门的处理逻辑
- [ ] 7. 工具执行的错误日志被记录，便于事后排查
- [ ] 8. Agent 在遇到不可恢复错误时，能向用户诚实说明而非编造答案

---

## 总结

工具调用失败是必然的，关键是设计好防御层次：

| 防御层 | 位置 | 负责内容 |
|--------|------|----------|
| 参数校验 | 工具入口 | 检查输入的有效性 |
| Try-Except | 工具内部 | 捕获所有业务异常 |
| 结构化错误 | 工具返回 | 标准化错误信息格式 |
| Agent 错误处理 | Agent 层 | 解析错误、格式化 Observation |
| 重试逻辑 | Agent 层 | 智能调整参数后重试 |
| 回退链 | 工具层 | 主工具失败后自动切换备选 |

---

> **下一陷阱**: [03-wrong-tool-selection.md](./03-wrong-tool-selection.md) —— 当 Agent 选择了错误的工具
