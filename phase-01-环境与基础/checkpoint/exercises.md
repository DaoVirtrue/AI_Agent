# Phase 01 检验点：练习题

> 以下 10 道练习题涵盖了 Phase 01（环境与基础）中最重要的技能。
> 建议按顺序完成，每道题都聚焦一个关键概念。
> 预计完成时间：2-4 小时（取决于你的 Python/异步编程基础）。

---

## 练习概览

| 编号 | 主题                        | 核心技能                          | 难度  |
|------|----------------------------|-----------------------------------|------|
| 01   | 并行异步调用                | `asyncio.gather`, `wait`          | 中等  |
| 02   | Pydantic 数据模型            | 类型验证、字段定义                 | 简单  |
| 03   | Token 精确计数               | `tiktoken`, 编码器选择             | 中等  |
| 04   | 指数退避重试装饰器           | 装饰器、指数退避、jitter           | 较高  |
| 05   | Token Bucket 限流器          | 令牌桶算法、客户端限流             | 较高  |
| 06   | 安全流式 JSON 解析器         | 缓冲区管理、JSON 完整性检测        | 中等  |
| 07   | pydantic-settings 配置类     | 环境变量读取、类型安全配置         | 简单  |
| 08   | Ollama vs OpenAI 延迟对比    | API 调用、计时、对比分析           | 简单  |
| 09   | API Key 轮换管理器           | 故障转移、密钥管理                 | 中等  |
| 10   | 成本估算函数                 | API 定价、数学计算                 | 简单  |

---

## 练习 01：并行调用 3 个 LLM API，返回最快结果

### 问题描述

编写一个异步函数 `race_call(prompt: str, models: list[str]) -> dict`，该函数：
- 同时向 3 个不同的模型 API 发送相同的 prompt
- 返回**第一个成功响应**的结果（"race" 模式）
- 如果所有模型都失败，抛出异常
- 其他还在进行中的请求应被**取消**（不浪费资源）

### 输入示例

```python
result = await race_call(
    prompt="What is the capital of France?",
    models=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
)
```

### 输出示例

```python
{
    "model": "gpt-4o-mini",
    "content": "The capital of France is Paris.",
    "latency": 1.23,  # 秒
}
```

### 提示

1. 使用 `asyncio.wait(..., return_when=asyncio.FIRST_COMPLETED)` 实现 race
2. 记得取消未完成的任务，避免资源泄露
3. 模拟 API 调用即可（不需要真实的 API Key）——用 `asyncio.sleep(random.random())` 模拟不同延迟
4. 考虑使用 `asyncio.create_task()` 和 `task.cancel()`

---

## 练习 02：用 Pydantic 定义 `APICallResult` 模型

### 问题描述

用 Pydantic v2 定义一个 `APICallResult` 数据模型，字段如下：

- `model_name: str` — 调用的模型名称
- `tokens_used: int` — 总消耗 token 数（input + output）
- `input_tokens: int` — 输入 token 数
- `output_tokens: int` — 输出 token 数
- `cost: float` — 费用（美元）
- `latency: float` — 延迟（秒）
- `content: str` — 模型返回的文本内容
- `timestamp: datetime` — 调用时间（自动设为当前时间）
- `success: bool` — 调用是否成功（默认 True）
- `error_message: str | None` — 错误信息（仅在 success=False 时有意义）

额外要求：
- 添加一个 `cost_per_1k_tokens: float` 计算属性，返回每 1000 token 的平均费用
- 添加一个类方法 `from_failed_call()` 用于快速创建失败记录

### 输入示例

```python
result = APICallResult(
    model_name="gpt-4o-mini",
    tokens_used=1500,
    input_tokens=1000,
    output_tokens=500,
    cost=0.00045,
    latency=1.23,
    content="The answer is 42.",
)
print(result.model_dump_json(indent=2))
```

### 输出示例

```json
{
  "model_name": "gpt-4o-mini",
  "tokens_used": 1500,
  "input_tokens": 1000,
  "output_tokens": 500,
  "cost": 0.00045,
  "latency": 1.23,
  "content": "The answer is 42.",
  "timestamp": "2026-06-11T10:30:00",
  "success": true,
  "error_message": null
}
```

### 提示

1. 使用 `pydantic.BaseModel` 和 `Field` 定义字段
2. `timestamp` 用 `Field(default_factory=datetime.now)`
3. `cost_per_1k_tokens` 用 `@property` 或 `@computed_field`（Pydantic v2）
4. `from_failed_call()` 用 `@classmethod`

---

## 练习 03：实现 `count_tokens()` 函数

### 问题描述

编写两个函数：
1. `count_tokens(text: str, model: str = "gpt-4o-mini") -> int` — 计算单段文本的 token 数
2. `count_messages_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int` — 计算消息列表的 token 数（含格式开销）

支持以下模型：
- `gpt-4o`, `gpt-4o-mini` → 使用 `o200k_base` 编码器
- `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo` → 使用 `cl100k_base` 编码器

### 输入示例

```python
text = "Hello, world! 你好，世界！"
tokens = count_tokens(text, "gpt-4o-mini")
print(f"'{text}' = {tokens} tokens")

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is RAG?"},
]
msg_tokens = count_messages_tokens(messages)
print(f"Messages = {msg_tokens} tokens")
```

### 输出示例

```
'Hello, world! 你好，世界！' = 15 tokens
Messages = 28 tokens
```

（实际 token 数可能略有差异，因为 tiktoken 版本和具体编码器实现有关）

### 提示

1. 安装 `tiktoken`: `pip install tiktoken`
2. 消息列表的格式开销：每条消息约 4 token，整个请求额外 2 token
3. 使用字典映射 model → encoding name
4. 测试中英文混合文本，观察 token 比例

---

## 练习 04：指数退避重试装饰器

### 问题描述

实现一个装饰器 `@with_backoff()`，为任意异步函数添加指数退避重试功能：

- **等待策略**：1s → 2s → 4s → 8s → 16s（指数增长，最大 60s）
- **随机抖动**：每次等待时间 ±25% 随机偏移（避免惊群效应）
- **可重试条件**：默认仅 HTTP 429（Rate Limit）和 5xx（Server Error）可重试
- **配置参数**：`max_retries`（最大重试次数）、`base_delay`（基础等待时间）、`max_delay`（最大等待时间）
- **日志**：每次重试时打印等待时间

不要依赖 `tenacity` 库，自己实现核心逻辑。

### 输入示例

```python
import random

call_count = 0

@with_backoff(max_retries=5, base_delay=1.0)
async def flaky_api_call():
    global call_count
    call_count += 1
    if call_count < 4:
        raise Exception("429 Rate limit exceeded")
    return {"status": "success", "attempt": call_count}

result = await flaky_api_call()
print(result)  # 预期第 4 次才成功
```

### 输出示例

```
[重试 1/5] 失败: 429 Rate limit exceeded, 1.2s 后重试
[重试 2/5] 失败: 429 Rate limit exceeded, 2.3s 后重试
[重试 3/5] 失败: 429 Rate limit exceeded, 4.1s 后重试
{'status': 'success', 'attempt': 4}
```

### 提示

1. 使用 `functools.wraps` 保留原函数的元数据
2. 随机抖动：`actual_delay = delay * (1 + random.uniform(-0.25, 0.25))`
3. 用 `inspect.iscoroutinefunction()` 检测函数是否为 async
4. 装饰器应同时支持带参数和不带参数的使用方式

---

## 练习 05：Token Bucket 限流器

### 问题描述

实现一个 `TokenBucket` 类，支持以下限流策略：

- **配置**：60 秒内最多 100 个请求（即每 0.6 秒补充 1 个 token）
- **突发容量**：桶的最大容量为 rate 值（允许短时突发）
- **同步和异步**：同时提供 `TokenBucket`（异步）和 `TokenBucketSync`（同步）两个版本
- **上下文管理器**：支持 `async with bucket:` 和 `with bucket:` 语法
- **主动查询**：提供 `acquire()` 方法返回等待秒数（0 = 立即可用）

### 输入示例

```python
import time
import asyncio

async def demo():
    bucket = TokenBucket(rate=100, period=60.0)  # 60秒100个请求

    start = time.monotonic()
    for i in range(105):  # 尝试发送 105 个请求
        async with bucket:
            pass  # 模拟发送请求
    elapsed = time.monotonic() - start
    print(f"105 个请求耗时: {elapsed:.2f}s")
    # 预期：最后 5 个请求需要等待桶补充 token

asyncio.run(demo())
```

### 输出示例

```
105 个请求耗时: ~3.0s
# （前 100 个瞬间完成，后 5 个等待桶补充，每个约 0.6s）
```

### 提示

1. 核心公式：`new_tokens = min(capacity, current_tokens + elapsed_time * refill_rate)`
2. 用 `time.monotonic()` 计时（不受系统时间调整影响）
3. 异步版用 `asyncio.Lock`，同步版用 `threading.Lock`
4. `refill_rate = rate / period`（每秒补充的 token 数）

---

## 练习 06：安全流式 JSON 解析器

### 问题描述

实现一个 `StreamingJSONParser` 类，能处理跨 chunk 边界的 JSON 数据：

- **核心方法** `feed(chunk: str) -> list[dict]`：喂入一个文本 chunk，返回所有新检测到的完整 JSON 对象
- **完整性检测**：通过追踪括号深度（`{}` 和 `[]`）来判断 JSON 是否闭合
- **字符串处理**：正确处理 JSON 字符串中的转义字符和嵌套引号
- **flush() 方法**：返回缓冲区中尚未完成的内容

### 输入示例

```python
parser = StreamingJSONParser()

# 模拟 chunk 边界任意切割 JSON
chunks = [
    '{"name": "Alice", "data": {"c',
    'ity": "Beijing"}} {"name": "Bob',
    '", "age": 30}',
]

for chunk in chunks:
    results = parser.feed(chunk)
    for obj in results:
        print(f"解析到: {obj}")

# 预期输出 2 个完整 JSON 对象
```

### 输出示例

```
解析到: {'name': 'Alice', 'data': {'city': 'Beijing'}}
解析到: {'name': 'Bob', 'age': 30}
```

### 提示

1. 维护 `brace_depth` 和 `bracket_depth` 状态变量
2. 维护 `in_string` 和 `escape_next` 标志位追踪字符串状态
3. 当 `brace_depth == 0` 且 `bracket_depth == 0` 时尝试解析
4. 用 `json.loads()` 做最终验证（捕获 `JSONDecodeError`）

---

## 练习 07：用 pydantic-settings 创建配置类

### 问题描述

用 `pydantic-settings` 创建一个 `AppSettings` 配置类，从 `.env` 文件和环境变量读取配置。包含以下字段：

| 字段名              | 类型   | 环境变量              | 默认值                | 说明                     |
|--------------------|--------|----------------------|----------------------|-------------------------|
| `openai_api_key`   | str    | `OPENAI_API_KEY`     | ""                   | OpenAI API 密钥          |
| `anthropic_api_key`| str    | `ANTHROPIC_API_KEY`  | ""                   | Anthropic API 密钥       |
| `default_model`    | str    | `DEFAULT_MODEL`      | "gpt-4o-mini"        | 默认使用的模型            |
| `max_retries`      | int    | `MAX_RETRIES`        | 3                    | 最大重试次数             |
| `log_level`        | str    | `LOG_LEVEL`          | "INFO"               | 日志级别                 |
| `temperature`      | float  | `TEMPERATURE`        | 0.7                  | LLM temperature          |
| `max_tokens`       | int    | `MAX_OUTPUT_TOKENS`  | 4096                 | 最大输出 token 数         |
| `enable_cache`     | bool   | `ENABLE_CACHE`       | True                 | 是否启用缓存             |

额外要求：
- 使用 `model_config = SettingsConfigDict(env_file=".env")` 加载 .env
- 添加一个 `model_post_init` 钩子，验证 `temperature` 在 0.0-2.0 范围内
- 添加一个方法 `masked_keys()` 返回脱敏后的 API Key（只显示前 8 和后 4 字符）
- 如果 `openai_api_key` 和 `anthropic_api_key` 都为空，打印警告

### 输入示例

```python
# .env 文件内容:
# OPENAI_API_KEY=sk-proj-abcdefghijklmnop
# DEFAULT_MODEL=gpt-4o
# TEMPERATURE=0.3

settings = AppSettings()
print(settings.model_dump())
print(f"OpenAI Key: {settings.masked_keys()['openai_api_key']}")
```

### 输出示例

```
openai_api_key='sk-proj-abcdefghijklmnop'
anthropic_api_key=''
default_model='gpt-4o'
max_retries=3
log_level='INFO'
temperature=0.3
max_tokens=4096
enable_cache=True
OpenAI Key: sk-proj-...mnop
```

### 提示

1. 安装 `pydantic-settings`: `pip install pydantic-settings`
2. 从 `pydantic_settings import BaseSettings, SettingsConfigDict`
3. 使用 `Field(alias="ENV_VAR_NAME")` 映射不同的环境变量名
4. `model_post_init` 在 Pydantic v2 中替代了 `__init__` 后钩子

---

## 练习 08：Ollama 本地模型 vs OpenAI API 延迟对比

### 问题描述

编写一个脚本，对比本地 Ollama 模型和 OpenAI API 的响应延迟：

- 测试相同的 prompt（3 个不同长度：短/中/长）
- 对每个 prompt，各调用 5 次取平均值
- 测量指标：首次 token 时间（TTFT）、总延迟、token 生成速度（token/s）
- 输出格式化的对比表格

如果不能实际连接 Ollama/OpenAI，则构建一个模拟框架，用合理的时间值模拟结果。

### 输入示例

```python
prompts = {
    "short": "你好，请问今天天气怎么样？",
    "medium": "请详细解释什么是 RAG（Retrieval-Augmented Generation），包括它的工作流程和优缺点。" * 2,
    "long": "请从以下长篇文档中提取关键信息..." * 10,
}

results = await benchmark(prompts, models=["ollama:llama3.1", "openai:gpt-4o-mini"])
print_results_table(results)
```

### 输出示例

```
模型                      提示词长度    平均TTFT     平均延迟     Token/s
───────────────────────────────────────────────────────────────────
ollama:llama3.1           短 (15 tok)   0.05s       0.30s       82.5
openai:gpt-4o-mini        短 (15 tok)   0.80s       1.20s       45.2
ollama:llama3.1           中 (120 tok)  0.06s       1.80s       78.3
openai:gpt-4o-mini        中 (120 tok)  0.85s       2.50s       42.1
ollama:llama3.1           长 (500 tok)  0.08s       6.50s       72.1
openai:gpt-4o-mini        长 (500 tok)  0.90s       8.20s       40.8
```

### 提示

1. `time.monotonic()` 或 `time.perf_counter()` 用于高精度计时
2. TTFT（Time To First Token）：从发送请求到收到第一个 token 的时间
3. 模拟 Ollama 时，TTFT 设为 0.05-0.1s（本地模型没有网络延迟），token 生成速度 70-90 token/s
4. 模拟 OpenAI 时，TTFT 设为 0.5-1.2s（网络延迟），token 生成速度 40-50 token/s

---

## 练习 09：API Key 轮换管理器

### 问题描述

实现一个 `APIKeyManager` 类，管理多个 API Key 的轮换和故障转移：

- **多 Key 存储**：维护一个主 Key 和多个备用 Key
- **故障检测**：当某个 Key 返回 401/403/429 时自动切换到下一个
- **冷却时间**：被暂时禁用的 Key 在冷却时间（默认 60 秒）后重新可用
- **优先级策略**：优先使用主 Key，主 Key 可用时始终用主 Key
- **线程安全**：使用锁保证并发场景下的 Key 分配安全

### 输入示例

```python
manager = APIKeyManager(
    primary_key="sk-primary-key-123",
    backup_keys=["sk-backup-456", "sk-backup-789"],
    cooldown_seconds=60.0,
)

# 正常情况
key = await manager.get_key()  # 返回主 Key
print(f"使用 Key: {key}")

# 主 Key 故障
manager.mark_failed("sk-primary-key-123", reason="429")
key = await manager.get_key()  # 返回第一个备用 Key
print(f"故障转移 Key: {key}")

# 60 秒后主 Key 恢复
await asyncio.sleep(61)
key = await manager.get_key()  # 返回主 Key
print(f"恢复 Key: {key}")

print(manager.status())  # 打印所有 Key 的状态
```

### 输出示例

```
使用 Key: sk-primary-key-123
故障转移 Key: sk-backup-456
恢复 Key: sk-primary-key-123

=== API Key 状态 ===
sk-primary...y-123: ✅ active (failures: 1)
sk-backup...p-456: ✅ active (failures: 0)
sk-backup...p-789: ✅ active (failures: 0)
```

### 提示

1. 用 `asyncio.Lock` 保证线程安全
2. 用 `dataclass` 追踪每个 Key 的状态（active / cooldown / revoked）
3. `mark_failed()` 记录失败时间和原因
4. 冷却时间过后自动将状态恢复为 active
5. 考虑添加 `revoke_key()` 方法用于永久禁用某个 Key

---

## 练习 10：费用估算函数

### 问题描述

实现 `estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float` 函数，支持以下模型的费用估算：

| 模型             | 输入价格 ($/1M tokens) | 输出价格 ($/1M tokens) |
|-----------------|----------------------|-----------------------|
| gpt-4o-mini     | $0.15                | $0.60                 |
| gpt-4o          | $2.50                | $10.00                |
| gpt-4-turbo     | $10.00               | $30.00                |
| gpt-3.5-turbo   | $0.50                | $1.50                 |
| claude-3.5-sonnet| $3.00               | $15.00                |
| claude-3.5-haiku | $0.80               | $4.00                 |

额外要求：
- 添加 `format_cost(cost: float) -> str` 辅助函数，格式化输出："$0.0045" 或 "$1.23"
- 添加 `compare_models(input_tokens: int, output_tokens: int) -> dict[str, float]` 函数，对同一个 (input, output) 组合计算所有模型的费用
- 处理未知模型的情况（返回 0.0 并打印警告）

### 输入示例

```python
# 单次 RAG 查询：3000 input + 500 output
cost = estimate_cost("gpt-4o-mini", input_tokens=3000, output_tokens=500)
print(f"gpt-4o-mini: {format_cost(cost)}")

comparison = compare_models(input_tokens=3000, output_tokens=500)
for model, cost in sorted(comparison.items(), key=lambda x: x[1]):
    print(f"  {model:25s}: {format_cost(cost)}")
```

### 输出示例

```
gpt-4o-mini: $0.00075

  各模型费用对比（3000 input + 500 output）:
  gpt-4o-mini              : $0.00075
  gpt-3.5-turbo            : $0.00225
  claude-3.5-haiku         : $0.00440
  gpt-4o                   : $0.01250
  claude-3.5-sonnet        : $0.01650
  gpt-4-turbo              : $0.04500
```

### 提示

1. 用字典存储模型定价（key: model_name, value: (input_price, output_price)，单位：$/1M tokens）
2. 公式：`cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price`
3. 注意单位换算：$0.15/1M tokens = $0.00000015 per token
4. `format_cost()` 用于友好显示极小金额（如 $0.00075）

---

## 提交要求

- 所有代码用 Python 3.10+ 编写
- 类型注解完整（函数参数和返回值）
- 包含 docstring
- 异步代码用 `async/await`
- 允许使用标准库 + `tiktoken` + `pydantic` + `pydantic-settings`

对照参考解答时，重点检查：
1. 你的代码能否通过所有示例输入输出测试
2. 错误处理是否完备
3. 代码风格是否清晰（命名、注释、结构）
