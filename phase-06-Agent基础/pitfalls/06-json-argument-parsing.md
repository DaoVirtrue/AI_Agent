# JSON参数解析失败 (JSON Argument Parsing Failure)

> 陷阱编号: #06 | 难度: 高 | 影响: 高

---

## 1. 症状 (Symptoms)

Agent生成工具调用参数时JSON解析失败：

- **JSON格式错误**: 模型输出的JSON有语法错误，Python的 `json.loads()` 抛出 `JSONDecodeError`
- **无限重试循环**: 解析失败 → 告诉模型错误 → 模型再次生成错误JSON → 再次失败
- **静默fallback失败**: 回退逻辑本身也有bug，整个Agent执行中断
- **部分成功**: 有时能解析，有时不能，行为不一致

### 典型错误日志

```
ERROR: JSONDecodeError at line 1, column 45: Expecting ',' delimiter
ERROR: Failed to parse arguments after 3 attempts: {expression: "2+2", mode: "scientific",}
                                                                         ↑ 多余逗号
ERROR: Invalid JSON: Unterminated string starting at line 1
ERROR: Parse failure for tool 'calculator'. Raw output: {expression: 2+2, mode: scientific}
                                                                       ↑ 缺少引号
```

---

## 2. 根本原因 (Root Cause)

### 2.1 LLM不是JSON生成器

LLM被训练来生成自然语言，而非结构化数据。JSON要求精确的语法，而LLM本质上是概率性的。

### 2.2 常见JSON错误类型

```python
# 错误1: 多余逗号
{"city": "北京", "units": "celsius",}  # ← JSON不允许尾随逗号

# 错误2: 缺少引号
{city: "北京", units: "celsius"}  # ← key必须有双引号

# 错误3: 单引号
{'city': '北京', 'units': 'celsius'}  # ← JSON必须是双引号

# 错误4: 未转义的特殊字符
{"path": "C:\Users\name\file.txt"}  # ← 反斜杠需要转义

# 错误5: 非标准值
{"count": NaN, "flag": undefined}  # ← JSON不支持NaN/undefined

# 错误6: 注释
{"city": "北京",  // 中国首都  ← JSON不支持注释
 "units": "celsius"}

# 错误7: 方括号混用
[city=北京, units=celsius]  # ← 不是合法JSON，是自定义格式
```

---

## 3. 真实场景 (Real-World Scenario)

### 场景: 多步骤推理中的JSON解析失败

```
Agent思考: 我需要计算 2+2*3
Agent行动: calculator(expression=2+2*3, mode="scientific")

[解析失败]
错误: JSONDecodeError at line 1, char 12: Expecting property name enclosed in double quotes

Agent思考: 解析出错了...我调整一下格式
Agent行动: calculator[expression=2+2*3, mode="scientific"]

[解析失败]  
错误: JSONDecodeError ...

Agent思考: 再试一次
Agent行动: Action: calculator(expression: 2+2*3, mode: scientific)

[解析失败]
Agent思考: 我无法使用这个工具...

→ 3次失败后，任务放弃，用户看到"抱歉，我无法完成计算"
```

---

## 4. 解决方案 (Solutions)

### 方案1: 多层解析策略 (Strict → Lenient → Regex)

```python
import json
import re
import ast

class RobustArgumentParser:
    """多层解析策略: 从严格到宽松逐步尝试"""
    
    def __init__(self):
        self.attempts = []
    
    def parse(self, raw_text: str) -> dict:
        """
        多层解析参数
        1. 严格JSON解析
        2. 宽松JSON (修复常见错误)
        3. Python dict解析 (ast.literal_eval)
        4. 正则提取
        """
        self.attempts = []
        
        # Layer 1: 严格JSON
        result = self._try_strict_json(raw_text)
        if result is not None:
            return result
        
        # Layer 2: 修复后JSON
        result = self._try_fixed_json(raw_text)
        if result is not None:
            return result
        
        # Layer 3: Python literal
        result = self._try_python_dict(raw_text)
        if result is not None:
            return result
        
        # Layer 4: 正则提取
        result = self._try_regex_extraction(raw_text)
        if result is not None:
            return result
        
        raise ValueError(
            f"所有解析策略均失败。原始文本: {raw_text[:200]}\n"
            f"尝试记录: {self.attempts}"
        )
    
    def _try_strict_json(self, text: str) -> Optional[dict]:
        """Layer 1: 严格JSON解析"""
        self.attempts.append("strict_json")
        try:
            # 提取JSON对象 {...}
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, ValueError) as e:
            self.attempts[-1] += f"(failed: {e})"
        return None
    
    def _try_fixed_json(self, text: str) -> Optional[dict]:
        """Layer 2: 修复常见错误后解析"""
        self.attempts.append("fixed_json")
        
        try:
            # 提取潜在JSON部分
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if not match:
                return None
            
            json_str = match.group()
            
            # 修复1: 移除尾随逗号
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            
            # 修复2: 单引号 → 双引号 (小心处理已有双引号)
            json_str = self._fix_quotes(json_str)
            
            # 修复3: 为无引号的key添加双引号
            json_str = re.sub(
                r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
                r'\1"\2":',
                json_str
            )
            
            # 修复4: NaN → null, undefined → null
            json_str = json_str.replace('NaN', 'null')
            json_str = json_str.replace('undefined', 'null')
            json_str = json_str.replace('Infinity', 'null')
            
            # 修复5: 移除注释
            json_str = re.sub(r'//.*?\n', '\n', json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            self.attempts[-1] += f"(failed: {e})"
        return None
    
    def _fix_quotes(self, text: str) -> str:
        """智能修复引号"""
        # 简单策略: 如果文本包含双引号，保持；否则单引号→双引号
        if '"' in text:
            # 已经使用双引号，不处理（避免破坏嵌套引号）
            return text
        # 没有双引号的情况下，把单引号换成双引号
        return text.replace("'", '"')
    
    def _try_python_dict(self, text: str) -> Optional[dict]:
        """Layer 3: 当作Python字典解析"""
        self.attempts.append("python_dict")
        
        try:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                return ast.literal_eval(match.group())
        except (ValueError, SyntaxError) as e:
            self.attempts[-1] += f"(failed: {e})"
        return None
    
    def _try_regex_extraction(self, text: str) -> Optional[dict]:
        """Layer 4: 正则表达式提取键值对"""
        self.attempts.append("regex_extraction")
        
        result = {}
        
        # 模式1: key=value
        for match in re.finditer(
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*'
            r'(?:"([^"]*)"|\'([^\']*)\'|([^,\s\]\)\}]+))',
            text
        ):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4)
            result[key] = self._coerce_type(value)
        
        # 模式2: "key": value (JSON-like)
        if not result:
            for match in re.finditer(
                r'["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?\s*[:=]\s*'
                r'(?:"([^"]*)"|\'([^\']*)\'|([^,\s\]\)\}]+))',
                text
            ):
                key = match.group(1)
                value = match.group(2) or match.group(3) or match.group(4)
                result[key] = self._coerce_type(value)
        
        if result:
            return result
        
        self.attempts[-1] += "(no matches)"
        return None
    
    def _coerce_type(self, value: str) -> Any:
        """尝试推断值的类型"""
        value = value.strip().rstrip(',').rstrip(';')
        
        # 布尔值
        if value.lower() in ('true', 'yes'):
            return True
        if value.lower() in ('false', 'no'):
            return False
        if value.lower() in ('null', 'none', 'nil'):
            return None
        
        # 整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 字符串
        return value
```

---

### 方案2: 正则表达式回退提取器

作为最后手段，用正则提取非标准格式。

```python
class RegexFallbackParser:
    """支持多种非标准工具调用格式的正则提取器"""
    
    # 支持的各种格式
    PATTERNS = [
        # 格式1: tool(key1=val1, key2=val2)
        r'(\w+)\s*\(\s*(.+?)\s*\)',
        # 格式2: tool[key1=val1, key2=val2]
        r'(\w+)\s*\[\s*(.+?)\s*\]',
        # 格式3: Action: tool(key1=val1, ...)
        r'Action:\s*(\w+)\s*[\(\[]\s*(.+?)\s*[\)\]]',
        # 格式4: tool {key1: val1, key2: val2}
        r'(\w+)\s*\{\s*(.+?)\s*\}',
        # 格式5: JSON直接
        r'"name":\s*"(\w+)".*?"arguments":\s*(\{.+?\})',
    ]
    
    def parse(self, text: str) -> tuple:
        """
        返回: (tool_name, arguments_dict) 或 (None, None)
        """
        for pattern in self.PATTERNS:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                tool_name = match.group(1).strip()
                args_str = match.group(2).strip()
                
                args = self._parse_kv_pairs(args_str)
                return (tool_name, args)
        
        return (None, None)
    
    def _parse_kv_pairs(self, text: str) -> dict:
        """解析 key=value 或 key: value 对"""
        args = {}
        
        # 模式: key=value 或 key: value
        for match in re.finditer(
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*'
            r'(?:'
            r'"([^"]*)"'        # 双引号字符串
            r'|'                 # 或
            r'\'([^\']*)\''      # 单引号字符串
            r'|'                 # 或
            r'([^,\s\)\]\}]+)'   # 无引号值
            r')',
            text
        ):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4)
            args[key] = self._coerce(value)
        
        return args
    
    def _coerce(self, value: str) -> Any:
        """类型推断"""
        value = value.strip()
        if value.lower() == 'true': return True
        if value.lower() == 'false': return False
        if value.lower() in ('null', 'none'): return None
        try: return int(value)
        except ValueError:
            try: return float(value)
            except ValueError:
                return value
```

---

### 方案3: LLM自我纠正

将错误信息喂回给模型，让模型纠正自己的输出。

```python
class LLMSelfCorrection:
    """LLM自我纠正: 解析失败时让模型重新生成"""
    
    def __init__(self, llm_call, max_retries=3):
        self.llm_call = llm_call
        self.max_retries = max_retries
    
    def parse_with_correction(self, raw_response: str) -> dict:
        """带自我纠正的解析"""
        
        for attempt in range(self.max_retries):
            try:
                # 尝试直接解析
                parser = RobustArgumentParser()
                result = parser.parse(raw_response)
                return result
            
            except ValueError as parse_error:
                if attempt < self.max_retries - 1:
                    # 构建纠正提示
                    correction_prompt = self._build_correction_prompt(
                        raw_response, str(parse_error), attempt + 1
                    )
                    
                    # 让模型重新生成
                    raw_response = self.llm_call(correction_prompt)
                    
                    print(f"  [纠正] 尝试 {attempt + 1}/{self.max_retries}")
                else:
                    raise ValueError(
                        f"经过{self.max_retries}次尝试仍无法解析JSON参数\n"
                        f"最后一次错误: {parse_error}\n"
                        f"原始输出: {raw_response}"
                    )
        
        return {}
    
    def _build_correction_prompt(self, original: str, error: str, attempt: int):
        """构建纠正提示"""
        return f"""
你的上一次JSON输出解析失败。请修正以下错误。

## 错误信息
{error}

## 你的上一次输出
{original}

## 正确的JSON格式要求
1. 使用双引号 "" 包裹所有key和字符串value
2. 不允许尾随逗号
3. 特殊字符需要转义 (如 \\n, \\t, \\")
4. 不允许注释
5. 使用 null 而非 None/NaN/undefined

## 请重新输出正确的JSON
只输出JSON对象，不要包含其他内容。
"""
```

---

### 方案4: 使用原生Function Calling API（推荐）

使用支持原生函数调用的模型API，避免文本解析。

```python
class FunctionCallingToolExecutor:
    """使用原生Function Calling API，避免JSON解析问题"""
    
    def __init__(self, client):
        self.client = client
        self.tool_schemas = []
    
    def register_tool(self, name: str, description: str, parameters: dict):
        """注册工具的OpenAI格式schema"""
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": list(parameters.keys()),
                },
            },
        }
        self.tool_schemas.append(schema)
    
    def execute_with_function_calling(self, messages: list) -> dict:
        """使用原生Function Calling执行工具"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tool_schemas,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        
        # 模型决定调用工具
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                # 参数已经是解析好的dict，无需手动解析JSON！
                function_args = json.loads(tool_call.function.arguments)
                
                # 执行工具
                result = self._execute_tool(function_name, **function_args)
                return result
        
        # 模型直接回复文本
        return {"type": "text", "content": response_message.content}
```

---

## 5. 检查清单 (Checklist)

- [ ] **多层解析**: 是否实现了 "严格→宽松→正则" 的多层回退策略？
- [ ] **常见错误修复**: 是否自动修复尾随逗号、单引号、无引号key等常见问题？
- [ ] **自我纠正机制**: 解析失败时是否将错误信息反馈给LLM进行修正？
- [ ] **超时和重试上限**: 是否设置了最大重试次数，防止无限循环？
- [ ] **原始输出保存**: 是否保存了解析失败的原始文本，便于调试？
- [ ] **考虑Function Calling API**: 是否评估了使用原生Function Calling API的可行性？

---

## 6. 完整代码示例: 鲁棒JSON参数解析器

```python
class ProductionArgumentParser:
    """生产级参数解析器，综合所有方案"""
    
    def __init__(self, llm_callback=None, max_retries=3):
        self.llm_callback = llm_callback
        self.max_retries = max_retries
        self.parser = RobustArgumentParser()
        self.regex_parser = RegexFallbackParser()
        self.correction = LLMSelfCorrection(llm_callback) if llm_callback else None
        self.failed_parses: list = []
    
    def parse(self, raw_text: str) -> dict:
        """主解析入口"""
        self.failed_parses = []
        
        for attempt in range(self.max_retries):
            try:
                # Step 1: 多层JSON解析
                result = self.parser.parse(raw_text)
                return result
                
            except ValueError as e1:
                self.failed_parses.append({
                    "attempt": attempt + 1,
                    "text": raw_text[:500],
                    "error": str(e1),
                })
                
                # Step 2: 正则回退
                tool_name, args = self.regex_parser.parse(raw_text)
                if tool_name and args:
                    print(f"[警告] 正则回退解析成功 (尝试 {attempt + 1})")
                    return {"_tool": tool_name, **args}
                
                # Step 3: LLM自我纠正
                if self.correction and attempt < self.max_retries - 1:
                    try:
                        raw_text = self._call_llm_correction(raw_text, str(e1))
                    except Exception:
                        break  # LLM纠正也失败了
        
        # 所有策略都失败
        raise ValueError(
            f"参数解析失败 ({self.max_retries}次尝试)\n"
            f"原始文本: {raw_text}\n"
            f"详细错误: {json.dumps(self.failed_parses, ensure_ascii=False, indent=2)}"
        )
    
    def _call_llm_correction(self, text, error):
        """调用LLM自我纠正"""
        if self.llm_callback is None:
            raise RuntimeError("LLM回调未设置")
        
        prompt = (
            "你的JSON输出有格式错误。请修正并只返回有效的JSON对象。\n\n"
            f"错误: {error}\n\n"
            f"你的输出: {text}\n\n"
            "正确格式示例: {\"key\": \"value\"}\n"
        )
        
        return self.llm_callback(prompt)


# ============================================================
# 使用示例和测试
# ============================================================
if __name__ == "__main__":
    parser = ProductionArgumentParser()
    
    # 各种格式的测试用例
    test_cases = [
        # 正确格式
        '{"city": "北京", "units": "celsius"}',
        # 多余逗号
        '{"city": "北京", "units": "celsius",}',
        # 单引号
        "{'city': '北京', 'units': 'celsius'}",
        # 无引号key
        '{city: "北京", units: "celsius"}',
        # 自定义格式
        'calculator(expression=2+2*3, mode="scientific")',
        # 方括号格式
        'weather[city=上海, units=fahrenheit]',
        # NaN / undefined
        '{"value": NaN, "flag": undefined}',
        # 路径中的反斜杠
        '{"path": "C:\\Users\\name\\file.txt"}',
        # 包含注释
        '{"city": "北京", /* 首都 */ "units": "celsius"}',
    ]
    
    print("=" * 60)
    print("鲁棒JSON参数解析器 - 测试")
    print("=" * 60)
    
    for i, test_input in enumerate(test_cases):
        print(f"\n测试 {i+1}: {test_input[:60]}...")
        try:
            result = parser.parse(test_input)
            print(f"  ✓ 成功: {result}")
        except ValueError as e:
            print(f"  ✗ 失败: {str(e)[:100]}")
```

---

## 关键要点

1. **JSON是脆弱的** -- LLM生成JSON本质上不可靠，需要多层防御
2. **从严格到宽松** -- 先尝试严格解析，再逐步放宽约束
3. **正则是最佳回退** -- 对于Key-Value格式，正则比JSON解析更鲁棒
4. **自我纠正是双刃剑** -- 有效但增加延迟和成本
5. **Function Calling API是最佳方案** -- 如果可以，优先使用原生API
6. **保存失败的原始文本** -- 调试时无价的信息来源
