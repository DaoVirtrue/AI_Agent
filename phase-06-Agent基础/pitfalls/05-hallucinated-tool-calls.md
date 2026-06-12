# 幻觉工具调用 (Hallucinated Tool Calls)

> 陷阱编号: #05 | 难度: 中等 | 影响: 高

---

## 1. 症状 (Symptoms)

Agent运行时生成不存在的工具调用：

- **虚构工具名**: Agent调用 `send_email()` 但注册表中根本没有这个工具
- **虚构参数名**: Agent调用 `calculator(expression="2+2")` 但工具定义的参数是 `query` 而不是 `expression`
- **错误参数类型**: 参数需要 `int` 但Agent传了字符串 `"five"`
- **不存在的API端点**: Agent尝试调用 `/api/v2/send` 但只有 `/api/v1/message`
- **工具调用失败后陷入循环**: 失败后重试相同错误的调用，消耗大量资源

### 典型错误日志

```
ERROR: Tool 'send_email' not found in registry. Available: [weather, calculator, search]
ERROR: Invalid parameter 'expression' for tool 'calculator'. Expected: 'formula'
WARNING: Tool call failed 3 times. Possible hallucination detected.
```

---

## 2. 根本原因 (Root Cause)

### 2.1 模型对工具名产生幻觉

LLM根据训练数据中的常见模式"猜测"工具名，而非严格基于提供的工具描述。

```
System Prompt: "You have access to: weather(query), calculator(formula)"
Model Output:   "I'll use the math_tool to calculate..."  ← 幻觉!
```

### 2.2 工具描述不够充分

```python
# 描述太简短，模型不知道如何正确使用
ToolDefinition(
    name="calc",
    description="计算器",  # ← 太模糊！没有说明参数和用法
    func=calculate,
)
```

### 2.3 参数名与模型预期不匹配

```python
# 模型"期望"参数叫 'expression' 或 'formula'...
def calculate(my_custom_param_name: str):  # ← 模型猜不到这个名字
    pass
```

---

## 3. 真实场景 (Real-World Scenario)

### 场景: Agent试图发送邮件

```
用户: "将计算结果发送给张三"

Agent思考: 用户要发送结果...我需要send_email工具
Agent行动: send_email(to="zhangsan@example.com", body="计算结果: 42")

系统: 错误！工具 'send_email' 不存在

Agent思考: 我用了send_email但出错了...也许叫email_send?
Agent行动: email_send(to="zhangsan@example.com", body="计算结果: 42")

系统: 错误！工具 'email_send' 不存在

Agent思考: 也许应该用mail?
Agent行动: mail(action="send", to="zhangsan@example.com", ...)

系统: 错误！工具 'mail' 不存在

→ 循环3-5次后Agent放弃，返回"抱歉，我无法发送邮件"
```

### 场景: 计算器参数名错误

```
Agent: calculator(expression="2+2*3")   ← LLM幻觉: 多数教程用 expression
实际工具签名: def calculator(formula: str) → 参数名是 formula
结果: 参数名不匹配，工具执行失败
```

---

## 4. 解决方案 (Solutions)

### 方案1: 工具名白名单验证

在执行前验证工具名是否存在于注册表中。

```python
class ValidatedToolRegistry:
    """在执行前验证工具名的注册表"""
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """安全执行: 先验证工具名"""
        # 严格验证: 工具名必须在注册表中
        if tool_name not in self._tools:
            available = list(self._tools.keys())
            suggestion = self._fuzzy_match(tool_name, available)
            
            error_msg = f"工具 '{tool_name}' 不存在。可用工具: {available}"
            if suggestion:
                error_msg += f"\n你是不是想用 '{suggestion}'?"
            
            return ToolResult(
                status=ToolStatus.INVALID_ARGS,
                error=error_msg,
            )
        
        # 工具存在，继续执行
        return self._execute_internal(tool_name, kwargs)
    
    def _fuzzy_match(self, name: str, candidates: list) -> Optional[str]:
        """模糊匹配建议最近似的工具名"""
        name_lower = name.lower()
        
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # 完全包含
            if name_lower in candidate_lower or candidate_lower in name_lower:
                score = 0.8
            else:
                # 简单的编辑距离近似
                common_chars = sum(1 for c in name_lower if c in candidate_lower)
                score = common_chars / max(len(name_lower), len(candidate_lower))
            
            if score > best_score and score > 0.5:
                best_score = score
                best_match = candidate
        
        return best_match
```

---

### 方案2: 返回清晰错误信息

当工具不存在时，给Agent提供足够的信息来纠正。

```python
class ClearErrorToolExecutor:
    """返回清晰、可操作的错误信息"""
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self.registry:
            # 提供完整的上下文信息
            available_tools_desc = []
            for name, td in self.registry.items():
                params = self._get_param_names(td.func)
                available_tools_desc.append(
                    f"  - {name}({', '.join(params)}): {td.description}"
                )
            
            return ToolResult(
                status=ToolStatus.INVALID_ARGS,
                error=(
                    f"错误: 工具 '{tool_name}' 不存在。\n"
                    f"可用工具列表:\n"
                    f"{chr(10).join(available_tools_desc)}\n"
                    f"请从以上列表中选择正确的工具并重试。"
                ),
            )
        
        # 验证参数
        td = self.registry[tool_name]
        valid_params = self._get_param_names(td.func)
        invalid_params = [k for k in kwargs if k not in valid_params]
        
        if invalid_params:
            return ToolResult(
                status=ToolStatus.INVALID_ARGS,
                error=(
                    f"错误: 工具 '{tool_name}' 不接受以下参数: {invalid_params}\n"
                    f"正确参数: {valid_params}\n"
                    f"正确用法: {tool_name}({', '.join(f'{p}=...' for p in valid_params)})"
                ),
            )
        
        # 全部通过验证，执行
        return self._do_execute(td, kwargs)
    
    def _get_param_names(self, func) -> list:
        """获取函数的参数名列表"""
        import inspect
        sig = inspect.signature(func)
        return list(sig.parameters.keys())
```

---

### 方案3: 模糊匹配 + 自动修正

当工具名接近但不同时，自动建议或自动修正。

```python
class FuzzyToolMatcher:
    """模糊匹配工具名，支持自动修正"""
    
    def __init__(self, auto_correct: bool = False, threshold: float = 0.7):
        self.auto_correct = auto_correct
        self.threshold = threshold
    
    def find_tool(self, requested_name: str, registry: dict) -> tuple:
        """
        返回: (matched_name, confidence, is_exact_match)
        """
        # 1. 精确匹配
        if requested_name in registry:
            return (requested_name, 1.0, True)
        
        # 2. 规范化匹配（移除空格、下划线、连字符）
        normalized = requested_name.lower().replace(' ', '').replace('_', '').replace('-', '')
        for name in registry:
            norm_name = name.lower().replace('_', '').replace('-', '')
            if norm_name == normalized:
                return (name, 0.95, False)
        
        # 3. 子串匹配
        for name in registry:
            if normalized in name.lower() or name.lower() in normalized:
                return (name, 0.85, False)
        
        # 4. 常见别名映射
        aliases = {
            'calc': 'calculator',
            'compute': 'calculator',
            'math': 'calculator',
            'weather_query': 'weather',
            'get_weather': 'weather',
            'search_web': 'search',
            'find': 'search',
            'read_file': 'file_reader',
            'open_file': 'file_reader',
            'get_time': 'datetime',
            'current_time': 'datetime',
        }
        if requested_name.lower() in aliases:
            matched = aliases[requested_name.lower()]
            if matched in registry:
                return (matched, 0.9, False)
        
        # 5. 编辑距离近似
        best_name = None
        best_score = 0
        for name in registry:
            score = self._similarity(requested_name.lower(), name.lower())
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_score >= self.threshold:
            return (best_name, best_score, False)
        
        return (None, 0.0, False)
    
    def _similarity(self, a: str, b: str) -> float:
        """简单的字符串相似度"""
        # Jaccard相似度 on character bigrams
        a_bigrams = {a[i:i+2] for i in range(len(a)-1)}
        b_bigrams = {b[i:i+2] for i in range(len(b)-1)}
        
        if not a_bigrams or not b_bigrams:
            return 0.0
        
        intersection = a_bigrams & b_bigrams
        union = a_bigrams | b_bigrams
        return len(intersection) / len(union)
```

---

### 方案4: Pydantic参数Schema验证

使用Pydantic进行严格的参数类型和结构验证。

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class WeatherToolParams(BaseModel):
    """天气查询工具的参数schema"""
    city: str = Field(..., description="城市名称，如'北京'、'上海'", min_length=1)
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位"
    )
    
    @validator('city')
    def city_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('城市名不能为空')
        if any(char.isdigit() for char in v):
            raise ValueError(f'城市名不应包含数字: {v}')
        return v.strip()


class CalculatorToolParams(BaseModel):
    """计算器工具的参数schema"""
    formula: str = Field(
        ...,
        description="数学表达式，如 '2+3*4', 'sqrt(144)', 'sin(pi/2)'"
    )
    
    @validator('formula')
    def validate_formula(cls, v):
        # 安全检查
        dangerous = ['__', 'import', 'exec', 'eval', 'system', 'subprocess']
        for d in dangerous:
            if d in v.lower():
                raise ValueError(f'表达式包含禁止的内容: {d}')
        return v


class PydanticToolExecutor:
    """使用Pydantic验证参数的执行器"""
    
    def __init__(self):
        self.registry = {}
        self.schemas = {}
    
    def register(self, definition: ToolDefinition, schema_class=None):
        """注册工具及其参数schema"""
        self.registry[definition.name] = definition
        if schema_class:
            self.schemas[definition.name] = schema_class
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self.registry:
            return ToolResult(
                status=ToolStatus.INVALID_ARGS,
                error=f"工具 '{tool_name}' 不存在",
            )
        
        # Pydantic验证
        if tool_name in self.schemas:
            try:
                validated = self.schemas[tool_name](**kwargs)
                kwargs = validated.dict()
            except Exception as e:
                return ToolResult(
                    status=ToolStatus.INVALID_ARGS,
                    error=f"参数验证失败: {str(e)}",
                )
        
        td = self.registry[tool_name]
        try:
            result = td.func(**kwargs)
            return ToolResult(status=ToolStatus.SUCCESS, data=result)
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FATAL_ERROR,
                error=str(e),
            )
```

---

### 方案5: Few-shot示例抑制幻觉

在System Prompt中添加正确的工具使用示例。

```python
FEW_SHOT_SYSTEM_PROMPT = """
你是一个AI助手，可以使用以下工具完成任务。

## 可用工具
- weather(city: str, units: str = "celsius") - 查询城市天气
- calculator(formula: str) - 计算数学表达式
- search(query: str, max_results: int = 5) - 搜索知识库
- file_reader(path: str, encoding: str = "utf-8") - 读取文件内容

## 工具使用格式
当需要使用工具时，请使用以下格式:
Action: 工具名[参数名1=值1, 参数名2=值2]

## 正确示例
用户: 北京天气怎么样？
助手: Action: weather[city=北京, units=celsius]

用户: 计算 2的10次方
助手: Action: calculator[formula=2**10]

用户: 搜索机器学习相关内容
助手: Action: search[query=机器学习, max_results=5]

## 错误示例（不要这样做！）
❌ Action: get_weather[location=北京]        → 工具名错误，没有get_weather
❌ Action: calculator[expression=2+2]        → 参数名错误，应该用formula
❌ Action: send_email[to=user@email.com]     → send_email工具不存在
❌ Action: weather(city=北京)                 → 格式错误，应该用方括号[]

## 重要规则
1. 只使用上面列出的工具，不要编造工具名
2. 严格使用每个工具的正确参数名
3. 如果工具不存在，告诉用户你无法完成该操作
4. 如果参数不匹配，检查上面的工具定义
"""
```

---

## 5. 检查清单 (Checklist)

- [ ] **工具名验证**: 是否在执行前验证工具名存在于注册表中？
- [ ] **参数schema验证**: 是否使用Pydantic或类似机制验证参数类型和格式？
- [ ] **模糊匹配**: 是否提供工具名的模糊匹配和建议功能？
- [ ] **错误信息质量**: 错误信息是否包含正确的工具列表和参数说明？
- [ ] **Few-shot示例**: System Prompt中是否包含了正确的工具使用示例？

---

## 6. 完整代码示例

```python
class RobustToolExecutor:
    """综合所有方案的健壮工具执行器"""
    
    def __init__(self):
        self.registry: dict[str, ToolDefinition] = {}
        self.schemas: dict = {}
        self.fuzzy_matcher = FuzzyToolMatcher(auto_correct=True)
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """执行工具，带完整的验证和修正"""
        
        # Step 1: 工具名验证 + 模糊匹配
        matched_name, confidence, is_exact = self.fuzzy_matcher.find_tool(
            tool_name, self.registry
        )
        
        if matched_name is None:
            available = list(self.registry.keys())
            return ToolResult(
                status=ToolStatus.INVALID_ARGS,
                error=(
                    f"工具 '{tool_name}' 不存在，且没有找到相似的替代工具。\n"
                    f"可用工具: {available}\n"
                    f"请选择以上工具之一重试。"
                ),
            )
        
        if not is_exact:
            print(f"[自动修正] '{tool_name}' → '{matched_name}' (置信度: {confidence:.0%})")
        
        # Step 2: 参数验证 (Pydantic)
        td = self.registry[matched_name]
        validated_kwargs = kwargs
        
        if tool_name in self.schemas:
            try:
                validated_data = self.schemas[tool_name](**kwargs)
                validated_kwargs = validated_data.dict()
            except Exception as e:
                params_desc = self._describe_parameters(td)
                return ToolResult(
                    status=ToolStatus.INVALID_ARGS,
                    error=(
                        f"工具 '{matched_name}' 参数验证失败: {str(e)}\n"
                        f"正确的参数格式: {params_desc}"
                    ),
                )
        
        # Step 3: 执行
        try:
            start = time.time()
            result = td.func(**validated_kwargs)
            elapsed = time.time() - start
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                execution_time=elapsed,
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FATAL_ERROR,
                error=f"工具执行错误: {str(e)}",
            )
    
    def _describe_parameters(self, td: ToolDefinition) -> str:
        """生成参数描述"""
        import inspect
        sig = inspect.signature(td.func)
        params = []
        for name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                params.append(f"{name} (必需)")
            else:
                params.append(f"{name} = {param.default} (可选)")
        return f"{td.name}({', '.join(params)})"


# 使用示例
if __name__ == "__main__":
    executor = RobustToolExecutor()
    
    # 注册工具
    executor.registry["calculator"] = ToolDefinition(
        name="calculator",
        description="数学计算器",
        func=calculate_tool,
    )
    
    # 测试: Agent幻觉出来的工具名
    result = executor.execute("calc")  # 自动修正为 calculator
    result = executor.execute("send_email")  # 返回清晰错误
    
    # 测试: 错误的参数名
    result = executor.execute("calculator", expression="2+2")  # 参数验证失败
```

---

## 关键要点

1. **永远验证，不要信任** -- 模型输出是不可靠的，必须在执行前验证
2. **模糊匹配是用户体验** -- 自动修正常见别名提升体验
3. **错误信息是给模型看的** -- 清晰的错误帮助模型自我纠正
4. **Pydantic是最佳实践** -- 类型安全 + 自动验证 + 清晰错误
5. **Few-shot示例减少幻觉** -- 正确的示例比规则描述更有效
