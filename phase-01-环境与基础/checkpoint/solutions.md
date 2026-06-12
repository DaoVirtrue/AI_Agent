# Phase 01 检验点：参考解答

> 以下提供所有 10 道练习题的完整参考解答，包括代码、思路讲解、替代方案和常见错误。

---

## 练习 01：并行调用 3 个 LLM API，返回最快结果

### 思路

核心需求是"并发调用，取第一个成功的结果"。这里的关键技术点是：
1. 使用 `asyncio.wait(return_when=FIRST_COMPLETED)` 等待第一个完成的任务
2. 处理完成后立即取消其余未完成的任务
3. 处理全部失败的边界情况

### 参考解答

```python
"""练习 01：并行调用多个 LLM API，返回最快的结果。"""

import asyncio
import logging
import random
import time
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 模拟的 API 调用函数（实际使用时替换为真实 API 调用）
# ============================================================

async def mock_llm_call(
    model: str,
    prompt: str,
    *,
    fail_rate: float = 0.2,
) -> dict[str, Any]:
    """模拟 LLM API 调用，带有随机延迟和失败概率。

    Args:
        model: 模型名称
        prompt: 用户提示词
        fail_rate: 调用失败的概率（模拟网络错误或 429）

    Returns:
        包含 model, content, latency 的字典

    Raises:
        Exception: 模拟调用失败
    """
    # 模拟不同的响应延迟
    # gpt-4o-mini 通常最快，gpt-4o 较慢
    base_delays = {
        "gpt-4o-mini": (0.3, 0.8),
        "gpt-3.5-turbo": (0.2, 0.6),
        "gpt-4o": (0.8, 2.0),
    }
    min_d, max_d = base_delays.get(model, (0.5, 1.5))
    delay = min_d + random.random() * (max_d - min_d)

    await asyncio.sleep(delay)

    # 模拟随机失败
    if random.random() < fail_rate:
        raise Exception(f"[{model}] Server error (simulated)")

    return {
        "model": model,
        "content": f"[{model}] Response to: {prompt[:30]}...",
        "latency": round(delay, 3),
    }


# ============================================================
# 核心实现：race_call
# ============================================================

async def race_call(
    prompt: str,
    models: list[str],
    *,
    fail_rate: float = 0.2,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """向多个模型并发发送请求，返回第一个成功的结果。

    策略：
      1. 为每个模型创建一个 Task
      2. 用 asyncio.wait(FIRST_COMPLETED) 等待第一个完成
      3. 如果是成功的结果，取消其余任务并返回
      4. 如果是失败的结果，从 pending 中移除，继续等待

    这样可以处理"最快的模型恰好失败了"的情况。

    Args:
        prompt: 用户提示词
        models: 模型名称列表
        fail_rate: 模拟的失败概率
        timeout: 总超时时间（秒）

    Returns:
        第一个成功响应的结果

    Raises:
        RuntimeError: 所有模型都调用失败
        asyncio.TimeoutError: 超过总超时时间
    """
    if not models:
        raise ValueError("models 列表不能为空")

    # 创建所有任务
    tasks = {
        asyncio.create_task(
            mock_llm_call(model, prompt, fail_rate=fail_rate),
            name=model,
        ): model
        for model in models
    }

    results: list[dict] = []
    errors: list[tuple[str, str]] = []

    try:
        # 外层总超时
        async with asyncio.timeout(timeout):
            pending = set(tasks.keys())

            while pending:
                # 等待任意一个完成
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # 检查每个完成的任务
                for task in done:
                    model = tasks[task]
                    try:
                        result = task.result()
                        # ✅ 成功！取消其余任务
                        logger.info(
                            f"🏆 最快响应: {model} ({result['latency']}s)"
                        )
                        _cancel_all(pending)
                        return result
                    except Exception as e:
                        # ❌ 这个模型失败了，记录并继续等待
                        logger.warning(f"  {model} 失败: {e}")
                        errors.append((model, str(e)))
                        # 如果所有任务都完成了且全失败
                        if not pending:
                            raise RuntimeError(
                                f"所有 {len(models)} 个模型均调用失败。\n"
                                f"错误详情: {errors}"
                            )

    except asyncio.TimeoutError:
        _cancel_all(pending)
        raise asyncio.TimeoutError(
            f"race_call 超时 ({timeout}s)，"
            f"已完成但失败: {len(errors)}，剩余: {len(pending)}"
        )

    # 这行理论上不会执行到
    raise RuntimeError("Unexpected: race_call 未能返回结果")


def _cancel_all(tasks: set[asyncio.Task]) -> None:
    """取消一组未完成的 asyncio Task。"""
    for task in tasks:
        if not task.done():
            task.cancel()


# ============================================================
# 演示
# ============================================================

async def demo():
    print("=" * 50)
    print("练习 01: 并行调用，返回最快结果")
    print("=" * 50)

    prompt = "What is the capital of France?"

    # 场景 1: 正常情况
    print("\n--- 场景 1: 正常竞赛 ---")
    try:
        result = await race_call(prompt, ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"])
        print(f"\n最终结果:")
        print(f"  模型:   {result['model']}")
        print(f"  内容:   {result['content']}")
        print(f"  延迟:   {result['latency']}s")
    except RuntimeError as e:
        print(f"失败: {e}")

    # 场景 2: 前两个模型都失败
    print("\n--- 场景 2: 前两个失败，第三个成功 ---")
    try:
        result = await race_call(
            prompt,
            ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
            fail_rate=0.9,  # 高失败率，测试 fallback
        )
        print(f"最终成功模型: {result['model']}")
    except RuntimeError as e:
        print(f"所有模型均失败: {e}")


if __name__ == "__main__":
    asyncio.run(demo())
```

### 替代方案

**方案 B：`asyncio.as_completed()`** — 更简洁但语义略有不同，它按完成顺序返回，天然适合 race 场景。

```python
async def race_call_alt(prompt, models):
    tasks = [
        asyncio.create_task(mock_llm_call(m, prompt))
        for m in models
    ]
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            for t in tasks:
                t.cancel()
            return result
        except Exception:
            continue
    raise RuntimeError("All models failed")
```

### 常见错误

1. **忘记取消未完成的任务**：`asyncio.wait(FIRST_COMPLETED)` 返回后，pending 中的任务仍在运行，造成资源泄露
2. **没有处理"最快响应是错误"的情况**：最快的任务可能返回错误，需要继续等待第二个
3. **使用 `asyncio.gather()` 代替 `asyncio.wait()`**：`gather` 会等待所有任务完成（或一个失败就全部取消），不适合 race 场景

### 测试用例

```python
# 测试 1: 正常竞赛
result = await race_call("test", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"])
assert "model" in result
assert "latency" in result

# 测试 2: 空模型列表
try:
    await race_call("test", [])
    assert False, "应该抛出异常"
except ValueError:
    pass

# 测试 3: 所有模型都失败
try:
    await race_call("test", ["gpt-4o-mini"], fail_rate=1.0)
    assert False, "应该抛出异常"
except RuntimeError:
    pass
```

---

## 练习 02：用 Pydantic 定义 `APICallResult` 模型

### 思路

Pydantic v2 提供了 `BaseModel`、`Field`、`computed_field` 等能力。模型设计的关键是：
1. 使用合适的类型注解（`str | None` 而非 `Optional[str]`）
2. 时间戳用 `default_factory` 而非 `default`（避免类定义时固定时间）
3. `computed_field` 用于计算属性（会被序列化）
4. 类方法用于构造辅助逻辑

### 参考解答

```python
"""练习 02：用 Pydantic 定义 APICallResult 模型。"""

from datetime import datetime, timezone
from typing import Self

from pydantic import BaseModel, Field, computed_field, model_validator


class APICallResult(BaseModel):
    """单次 LLM API 调用的结果记录。

    用于追踪每次 API 调用的详细信息，包括费用、延迟和结果。
    """

    model_name: str = Field(
        ...,  # ... 表示必填
        description="调用的模型名称",
        examples=["gpt-4o-mini"],
    )
    tokens_used: int = Field(
        ...,
        ge=0,
        description="总消耗 token 数（input + output）",
    )
    input_tokens: int = Field(
        ...,
        ge=0,
        description="输入 token 数",
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description="输出 token 数",
    )
    cost: float = Field(
        ...,
        ge=0.0,
        description="费用（美元）",
    )
    latency: float = Field(
        ...,
        ge=0.0,
        description="调用延迟（秒）",
    )
    content: str = Field(
        ...,
        description="模型返回的文本内容",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="调用时间（UTC）",
    )
    success: bool = Field(
        default=True,
        description="调用是否成功",
    )
    error_message: str | None = Field(
        default=None,
        description="错误信息（仅在 success=False 时有效）",
    )

    # ---- 计算属性 ----

    @computed_field
    @property
    def cost_per_1k_tokens(self) -> float:
        """每 1000 token 的平均费用。

        Returns:
            费用/千token，如果 tokens_used 为 0 则返回 0.0
        """
        if self.tokens_used == 0:
            return 0.0
        return round(self.cost / (self.tokens_used / 1000), 8)

    @computed_field
    @property
    def tokens_per_second(self) -> float:
        """每秒生成的 token 数。

        Returns:
            token/s，如果延迟为 0 则返回 0.0
        """
        if self.latency == 0:
            return 0.0
        return round(self.output_tokens / self.latency, 2)

    # ---- 类方法（工厂） ----

    @classmethod
    def from_failed_call(
        cls,
        model_name: str,
        error_message: str,
        latency: float = 0.0,
        **kwargs,
    ) -> "APICallResult":
        """快速创建失败调用的记录。

        Args:
            model_name: 模型名称
            error_message: 错误描述
            latency: 调用延迟（如果有）
            **kwargs: 其他可选的字段值

        Returns:
            标记为失败的 APICallResult 实例
        """
        return cls(
            model_name=model_name,
            tokens_used=0,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            latency=latency,
            content="",
            success=False,
            error_message=error_message,
            **kwargs,
        )

    @classmethod
    def from_successful_call(
        cls,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        content: str,
        latency: float,
        cost: float,
    ) -> "APICallResult":
        """快速创建成功调用的记录。

        Args:
            model_name: 模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            content: 模型返回内容
            latency: 调用延迟（秒）
            cost: 费用（美元）

        Returns:
            APICallResult 实例
        """
        return cls(
            model_name=model_name,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency=latency,
            content=content,
            success=True,
        )

    # ---- 验证器 ----

    @model_validator(mode="after")
    def validate_tokens_consistency(self) -> "APICallResult":
        """验证 tokens_used 与 input/output 的一致性。"""
        if self.success and self.tokens_used != self.input_tokens + self.output_tokens:
            raise ValueError(
                f"tokens_used ({self.tokens_used}) 应等于 "
                f"input_tokens ({self.input_tokens}) + "
                f"output_tokens ({self.output_tokens})"
            )
        return self

    # ---- 方法 ----

    def summary(self) -> str:
        """生成人类可读的调用摘要。"""
        status = "✅" if self.success else "❌"
        if self.success:
            return (
                f"{status} [{self.model_name}] "
                f"{self.tokens_used} tokens | "
                f"${self.cost:.6f} | "
                f"{self.latency:.2f}s | "
                f"${self.cost_per_1k_tokens:.6f}/1K"
            )
        else:
            return (
                f"{status} [{self.model_name}] "
                f"FAILED: {self.error_message}"
            )


# ============================================================
# 演示
# ============================================================

def demo():
    print("=" * 50)
    print("练习 02: Pydantic APICallResult 模型")
    print("=" * 50)

    # 成功调用
    result = APICallResult(
        model_name="gpt-4o-mini",
        tokens_used=1500,
        input_tokens=1000,
        output_tokens=500,
        cost=0.00045,
        latency=1.23,
        content="The answer is 42.",
    )
    print("\n成功调用:")
    print(result.model_dump_json(indent=2))
    print(f"\n摘要: {result.summary()}")
    print(f"每1K token费用: ${result.cost_per_1k_tokens:.6f}")
    print(f"生成速度: {result.tokens_per_second} tok/s")

    # 失败调用
    failed = APICallResult.from_failed_call(
        model_name="gpt-4o",
        error_message="429 Rate limit exceeded",
        latency=0.5,
    )
    print(f"\n失败调用:")
    print(f"摘要: {failed.summary()}")
    print(f"success: {failed.success}, error: {failed.error_message}")

    # 验证：tokens 一致性检查
    try:
        APICallResult(
            model_name="gpt-4o-mini",
            tokens_used=100,  # 不一致！
            input_tokens=1000,
            output_tokens=500,
            cost=0.1,
            latency=1.0,
            content="test",
        )
    except ValueError as e:
        print(f"\n验证捕获: {e}")


if __name__ == "__main__":
    demo()
```

### 替代方案

**使用 dataclass + 手动验证**（不需要 pydantic 时）：

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class APICallResultSimple:
    model_name: str
    tokens_used: int
    input_tokens: int
    output_tokens: int
    cost: float
    latency: float
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    error_message: str | None = None
```

### 常见错误

1. **`default=datetime.now()` 而非 `default_factory`**：前者会导致所有实例共享同一个时间戳（类定义时的时刻）
2. **忘记 `computed_field`**：普通 `@property` 不会被 Pydantic 序列化到 JSON
3. **`Optional[str]` vs `str | None`**：Python 3.10+ 推荐后者，更简洁

---

## 练习 03：实现 `count_tokens()` 函数

### 思路

核心是建立 model_name -> tiktoken encoding name 的映射。需要注意：
1. GPT-4o 系列使用较新的 `o200k_base` 编码器
2. GPT-4 / GPT-3.5 系列使用 `cl100k_base`
3. 消息列表要加上格式开销（每条消息 ~4 token + 整体 ~2 token）
4. 如果传入未知模型，用 `cl100k_base` 作为回退

### 参考解答

```python
"""练习 03：用 tiktoken 精确计算 token 数。"""

import tiktoken
import logging

logger = logging.getLogger(__name__)

# 模型 → 编码器名称映射
MODEL_TO_ENCODING: dict[str, str] = {
    # GPT-4o 系列 → o200k_base
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",
    # GPT-4 系列 → cl100k_base
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    # GPT-3.5 系列 → cl100k_base
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    # Embedding 模型 → cl100k_base
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
}

# 缓存已创建的编码器实例（避免重复创建）
_ENCODER_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoder(model: str) -> tiktoken.Encoding:
    """获取或创建模型的 tiktoken 编码器。

    Args:
        model: 模型名称

    Returns:
        tiktoken.Encoding 实例
    """
    # 规范化模型名称（去掉版本后缀）
    model_base = model.lower().strip()

    # 查找编码器名称
    encoding_name = None
    for key, enc in MODEL_TO_ENCODING.items():
        if model_base.startswith(key):
            encoding_name = enc
            break

    if encoding_name is None:
        # fallback：尝试 o200k_base 或 cl100k_base
        if "gpt-4o" in model_base:
            encoding_name = "o200k_base"
        else:
            encoding_name = "cl100k_base"
        logger.debug(
            f"模型 '{model}' 不在预定义列表中，使用 '{encoding_name}' 编码器"
        )

    # 从缓存获取
    if encoding_name not in _ENCODER_CACHE:
        _ENCODER_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)

    return _ENCODER_CACHE[encoding_name]


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """计算单段文本的 token 数。

    Args:
        text: 要计算的文本
        model: 模型名称（决定使用哪种编码器）

    Returns:
        token 数量

    Examples:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens("你好，世界！", "gpt-4o-mini")
        5
    """
    if not text:
        return 0

    encoder = _get_encoder(model)
    tokens = encoder.encode(text)
    return len(tokens)


def count_messages_tokens(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
) -> int:
    """计算 OpenAI 格式消息列表的 token 数（含格式开销）。

    参考 OpenAI 官方文档：
    - 每条消息有固定开销（约 4 token）：用于分隔符和 role 标识
    - 整个请求有额外开销（约 2-3 token）
    - 如果消息包含 "name" 字段，还有额外开销

    Args:
        messages: 消息列表，格式 [{"role": "...", "content": "..."}]
        model: 模型名称

    Returns:
        总 token 数

    Examples:
        >>> msgs = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hi"},
        ... ]
        >>> count_messages_tokens(msgs)
        21  # (约值：含格式开销)
    """
    if not messages:
        return 0

    encoder = _get_encoder(model)
    total = 0

    for msg in messages:
        # 每条消息的基础开销（role 标记 + 分隔符）
        total += 4

        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(encoder.encode(content))
        elif isinstance(content, list):
            # 多模态消息（文本 + 图片）
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(encoder.encode(part.get("text", "")))

        # name 字段的额外开销
        if "name" in msg:
            total += -1  # role 已被包含，name 额外占用约 -1+1=0 差异
            name = msg["name"]
            total += len(encoder.encode(name))

    # 整个请求的固定开销
    total += 2

    return total


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """不依赖 tiktoken 的粗略估算（用于无网络/无库的场景）。

    粗略规则：
      - 英文：~4 字符/token
      - 中文：~1.5 字符/token
      - 混合内容：按比例加权

    Args:
        text: 文本
        model: 模型名称（仅用于日志）

    Returns:
        估算的 token 数
    """
    # 简单策略：对中文和其他 CJK 字符单独计数
    import unicodedata

    english_chars = 0
    cjk_chars = 0
    other_chars = 0

    for ch in text:
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            cjk_chars += 1
        elif '぀' <= ch <= 'ヿ':
            cjk_chars += 1  # 日文假名
        elif '가' <= ch <= '힯':
            cjk_chars += 1  # 韩文
        elif ord(ch) < 128:
            english_chars += 1
        else:
            other_chars += 1

    # 估算公式
    estimated = (english_chars / 4) + (cjk_chars / 1.5) + (other_chars / 3)
    return max(1, int(estimated))


# ============================================================
# 演示
# ============================================================

def demo():
    print("=" * 50)
    print("练习 03: Token 精确计数")
    print("=" * 50)

    # 测试纯英文
    en_text = "Hello, world! This is a test."
    en_tokens = count_tokens(en_text, "gpt-4o-mini")
    print(f"\n英文文本: '{en_text}'")
    print(f"  长度: {len(en_text)} 字符 → {en_tokens} tokens")
    print(f"  比例: {len(en_text) / en_tokens:.1f} 字符/token")

    # 测试纯中文
    cn_text = "你好，世界！这是一个测试。"
    cn_tokens = count_tokens(cn_text, "gpt-4o-mini")
    print(f"\n中文文本: '{cn_text}'")
    print(f"  长度: {len(cn_text)} 字符 → {cn_tokens} tokens")
    print(f"  比例: {len(cn_text) / cn_tokens:.1f} 字符/token")

    # 测试混合文本
    mix_text = "Hello! 你好！こんにちは！"
    mix_tokens = count_tokens(mix_text, "gpt-4o-mini")
    print(f"\n混合文本: '{mix_text}'")
    print(f"  长度: {len(mix_text)} 字符 → {mix_tokens} tokens")

    # 测试消息列表
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    msg_tokens = count_messages_tokens(messages)
    print(f"\n消息列表 ({len(messages)} 条):")
    print(f"  总 token 数: {msg_tokens}")

    # 分解每条消息
    for i, msg in enumerate(messages):
        content_tokens = count_tokens(msg["content"], "gpt-4o-mini")
        print(f"  [{i}] {msg['role']}: {content_tokens} tokens (content only)")

    # 对比不同模型的编码差异
    text = "Hello, world! 你好世界！"
    print(f"\n同一文本在不同模型下的 token 数: '{text}'")
    for model in ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]:
        tokens = count_tokens(text, model)
        print(f"  {model:20s}: {tokens} tokens")

    # 粗略估算比较
    estimated = estimate_tokens(text)
    actual = count_tokens(text, "gpt-4o-mini")
    print(f"\n粗略估算 vs 实际: {estimated} vs {actual} tokens")


if __name__ == "__main__":
    demo()
```

### 替代方案

**使用 `transformers` 库的 tokenizer**（适用于开源模型）：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
tokens = tokenizer.encode("你好世界")
print(len(tokens))  # → 3
```

**注意事项**：不同模型的 tokenizer 结果不同，tiktoken 只适用于 OpenAI 模型。

### 常见错误

1. **用 `len(text)` 估算 token**：中文 1 字符约等于 1.5-2 token，英文约 0.25 token/字符，差距巨大
2. **忘记消息格式开销**：每条消息约 4 token 的格式开销，消息多时累积可观
3. **编码器选择错误**：GPT-4o 系列必须用 `o200k_base`，用 `cl100k_base` 会得到不同的（不准确的）计数

---

## 练习 04：指数退避重试装饰器

### 思路

装饰器的核心挑战是同时支持 `@with_backoff` 和 `@with_backoff(max_retries=5)` 两种用法。这需要对 `__call__` 参数进行判断。

指数退避公式：`delay = base * 2^(attempt-1)`，然后添加 ±25% 的 jitter。

### 参考解答

```python
"""练习 04：指数退避重试装饰器。"""

import asyncio
import functools
import inspect
import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# 核心重试逻辑（提取为独立函数，可被装饰器和手动调用共用）
# ============================================================

def _is_retryable(exception: Exception) -> bool:
    """判断异常是否可重试。

    基于异常的 HTTP 状态码（如果可获取）：
      - 429: Rate Limit → 可重试
      - 5xx: Server Error → 可重试
      - 其他 → 不可重试
    """
    status_code = (
        getattr(exception, 'status_code', None)
        or getattr(getattr(exception, 'response', None), 'status_code', None)
    )
    if status_code is not None:
        return status_code == 429 or 500 <= status_code < 600
    # 无法获取状态码时，默认可重试
    return True


def _calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    jitter: float = 0.25,
) -> float:
    """计算第 attempt 次重试的等待时间。

    Args:
        attempt: 第几次重试（1-based）
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        jitter: 抖动比例（0.25 = ±25%）

    Returns:
        等待时间（秒）
    """
    # 指数增长
    delay = base_delay * (2 ** (attempt - 1))
    # 封顶
    delay = min(delay, max_delay)
    # 添加随机抖动
    jitter_amount = delay * jitter
    actual = delay + random.uniform(-jitter_amount, jitter_amount)
    return max(0.1, actual)


# ============================================================
# 异步版装饰器
# ============================================================

def with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """为异步函数添加指数退避重试能力。

    支持两种使用方式：
      @with_backoff                    # 使用默认参数
      @with_backoff(max_retries=5)     # 自定义参数

    Args:
        max_retries: 最大重试次数
        base_delay: 基础等待时间（秒）
        max_delay: 最大等待时间（秒）
        jitter: 抖动比例
        retryable_exceptions: 哪些异常类型可触发重试

    Returns:
        装饰器函数
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_retries + 2):  # 1 次初始 + N 次重试
                try:
                    return await func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    # 检查是否可重试
                    if not _is_retryable(e):
                        logger.error(
                            f"[{func.__name__}] 不可重试的错误: {e}"
                        )
                        raise

                    if attempt > max_retries:
                        logger.error(
                            f"[{func.__name__}] "
                            f"已达最大重试次数 ({max_retries}): {e}"
                        )
                        raise

                    # 计算并等待
                    delay = _calculate_delay(attempt, base_delay, max_delay, jitter)
                    logger.warning(
                        f"[{func.__name__}] "
                        f"重试 {attempt}/{max_retries}: "
                        f"{type(e).__name__}: {e}，"
                        f"{delay:.1f}s 后重试..."
                    )
                    await asyncio.sleep(delay)

            # 理论上不会执行到这里
            if last_exception:
                raise last_exception

        return wrapper

    # 支持 @with_backoff 不带括号的用法
    if callable(max_retries):
        # max_retries 实际上是被装饰的函数
        func = max_retries
        max_retries = 3
        return decorator(func)

    return decorator


# ============================================================
# 同步版装饰器
# ============================================================

def with_backoff_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """同步版指数退避重试装饰器。"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_retries + 2):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if not _is_retryable(e):
                        logger.error(f"[{func.__name__}] 不可重试: {e}")
                        raise

                    if attempt > max_retries:
                        logger.error(
                            f"[{func.__name__}] 已达最大重试 ({max_retries}): {e}"
                        )
                        raise

                    delay = _calculate_delay(attempt, base_delay, max_delay, jitter)
                    logger.warning(
                        f"[{func.__name__}] "
                        f"重试 {attempt}/{max_retries}: {e}，"
                        f"{delay:.1f}s 后重试..."
                    )
                    time.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper

    if callable(max_retries):
        func = max_retries
        max_retries = 3
        return decorator(func)

    return decorator


# ============================================================
# 演示
# ============================================================

async def demo():
    print("=" * 50)
    print("练习 04: 指数退避重试装饰器")
    print("=" * 50)

    # --- 测试 1: 几次失败后成功 ---
    print("\n--- 测试 1: 第 4 次成功 ---")
    call_count_a = 0

    @with_backoff(max_retries=5, base_delay=0.5)
    async def flaky_api_a():
        nonlocal call_count_a
        call_count_a += 1
        if call_count_a < 4:
            raise Exception("429 Rate limit exceeded")
        return {"status": "success", "attempt": call_count_a}

    result = await flaky_api_a()
    print(f"结果: {result}")

    # --- 测试 2: 超过最大重试次数 ---
    print("\n--- 测试 2: 超过最大重试次数 ---")
    call_count_b = 0

    @with_backoff(max_retries=3, base_delay=0.5)
    async def flaky_api_b():
        nonlocal call_count_b
        call_count_b += 1
        raise Exception(f"Error #{call_count_b}")

    try:
        await flaky_api_b()
    except Exception as e:
        print(f"预期失败: {e}")

    # --- 测试 3: 首次成功（不重试）---
    print("\n--- 测试 3: 首次成功 ---")

    @with_backoff(max_retries=3, base_delay=0.5)
    async def success_first_try():
        return {"status": "success", "attempt": 1}

    result = await success_first_try()
    print(f"结果: {result}")


if __name__ == "__main__":
    asyncio.run(demo())
```

### 替代方案

**使用 `tenacity` 库**（生产环境推荐，功能更完善）：

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(Exception),
)
async def call_api():
    ...
```

### 常见错误

1. **忘记添加 jitter**：没有 jitter 的退避在并发场景下会导致"惊群效应"——所有请求在相同时刻重试，再次撞墙
2. **退避时间太短**：base_delay=0.1s 对速率限制基本无效
3. **对所有异常都重试**：401（未授权）、403（禁止访问）不应该重试，重试一万次也不会成功
4. **忘了 `functools.wraps`**：不加的话装饰后的函数会丢失原函数的 `__name__`、`__doc__` 等信息

---

## 练习 05：Token Bucket 限流器

### 思路

Token Bucket（令牌桶）是最经典的限流算法之一：
- 桶有一个最大容量（burst）
- 以恒定速率（refill_rate）向桶中添加 token
- 每次请求需要消耗 1 个 token
- 如果桶中 token 不足，需要等待补充

关键实现细节：
- 使用 `time.monotonic()` 而不是 `time.time()`（不受系统时钟调整影响）
- 补充量 = 经过时间 * 补充速率
- 异步版用 `asyncio.Lock`，同步版用 `threading.Lock`

### 参考解答

```python
"""练习 05：Token Bucket 限流器。"""

import asyncio
import time
import threading
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 异步 Token Bucket
# ============================================================

class TokenBucket:
    """异步 Token Bucket 限流器。

    使用令牌桶算法平滑控制请求速率。支持上下文管理器。

    用法:
        bucket = TokenBucket(rate=100, period=60.0)  # 60秒100个请求

        async with bucket:
            await call_api()  # 如果速率超限会自动等待

    """

    def __init__(
        self,
        rate: int = 100,
        period: float = 60.0,
        burst: int | None = None,
    ):
        """
        Args:
            rate: 在 period 时间内允许的最大请求数
            period: 时间窗口（秒）
            burst: 桶的最大容量（允许的突发请求数），默认等于 rate
        """
        if rate <= 0:
            raise ValueError(f"rate 必须 > 0, 实际: {rate}")
        if period <= 0:
            raise ValueError(f"period 必须 > 0, 实际: {period}")

        self.rate = rate
        self.period = period
        self._refill_rate = rate / period  # 每秒补充的 token 数
        self._max_tokens = float(burst if burst is not None else rate)
        self._tokens = self._max_tokens  # 当前可用 token 数
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

        # 统计信息
        self._total_requests = 0
        self._total_wait_time = 0.0

    def _refill(self) -> None:
        """根据经过的时间补充 token。"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._max_tokens,
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> float:
        """获取请求许可。

        Args:
            tokens: 需要消耗的 token 数（默认 1）

        Returns:
            需要等待的秒数（0 = 立即可用）
        """
        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_requests += 1
                return 0.0

            # Token 不足，需要等待
            wait_time = (tokens - self._tokens) / self._refill_rate
            self._tokens = 0.0
            self._total_requests += 1
            self._total_wait_time += wait_time
            return wait_time

    async def __aenter__(self):
        """async with 上下文管理器入口。"""
        wait = await self.acquire()
        if wait > 0:
            logger.debug(f"TokenBucket: 等待 {wait:.2f}s")
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *args):
        """async with 上下文管理器出口。"""
        pass

    @property
    def available(self) -> float:
        """当前可用 token 数（近似值，调用时可能已变化）。"""
        return self._tokens

    @property
    def stats(self) -> dict:
        """统计信息。"""
        return {
            "total_requests": self._total_requests,
            "total_wait_time": round(self._total_wait_time, 3),
            "avg_wait_time": round(
                self._total_wait_time / max(1, self._total_requests), 3
            ),
            "available_tokens": round(self._tokens, 2),
            "refill_rate": round(self._refill_rate, 2),
        }


# ============================================================
# 同步 Token Bucket
# ============================================================

class TokenBucketSync:
    """同步版 Token Bucket 限流器。"""

    def __init__(self, rate: int = 100, period: float = 60.0, burst: int | None = None):
        if rate <= 0:
            raise ValueError(f"rate 必须 > 0, 实际: {rate}")
        if period <= 0:
            raise ValueError(f"period 必须 > 0, 实际: {period}")

        self.rate = rate
        self.period = period
        self._refill_rate = rate / period
        self._max_tokens = float(burst if burst is not None else rate)
        self._tokens = self._max_tokens
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

        self._total_requests = 0
        self._total_wait_time = 0.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._max_tokens,
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> float:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_requests += 1
                return 0.0
            wait_time = (tokens - self._tokens) / self._refill_rate
            self._tokens = 0.0
            self._total_requests += 1
            self._total_wait_time += wait_time
            return wait_time

    def __enter__(self):
        wait = self.acquire()
        if wait > 0:
            time.sleep(wait)
        return self

    def __exit__(self, *args):
        pass


# ============================================================
# 演示
# ============================================================

async def demo():
    print("=" * 50)
    print("练习 05: Token Bucket 限流器")
    print("=" * 50)

    # --- 测试 1: 基本限流 ---
    print("\n--- 测试 1: 105 个请求，60秒100个限制 ---")
    bucket = TokenBucket(rate=100, period=60.0)

    start = time.monotonic()
    for i in range(105):
        async with bucket:
            pass  # 模拟请求
    elapsed = time.monotonic() - start

    print(f"  105 个请求完成")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  预期额外等待: ~3s（多出的 5 个请求各等 0.6s）")
    print(f"  统计: {bucket.stats}")

    # --- 测试 2: 突发容量 ---
    print("\n--- 测试 2: 突发请求（burst=20）---")
    bucket2 = TokenBucket(rate=100, period=60.0, burst=30)

    start = time.monotonic()
    # 一口气发 25 个（在 burst 容量内）
    for i in range(25):
        async with bucket2:
            pass
    elapsed1 = time.monotonic() - start
    print(f"  前 25 个请求（突发）: {elapsed1:.3f}s（应几乎无等待）")

    # 再多发 10 个（超出 burst，需要等待补充）
    for i in range(10):
        async with bucket2:
            pass
    elapsed2 = time.monotonic() - start
    print(f"  总共 35 个请求: {elapsed2:.3f}s（后 10 个需要等待）")

    # --- 测试 3: 同步版本 ---
    print("\n--- 测试 3: 同步 TokenBucket ---")
    sync_bucket = TokenBucketSync(rate=5, period=1.0)  # 每秒 5 个

    start = time.monotonic()
    for i in range(8):
        with sync_bucket:
            pass  # 模拟请求
    elapsed = time.monotonic() - start
    print(f"  8 个请求（限制 5/s）: {elapsed:.2f}s")
    print(f"  预期: ~0.6s（后 3 个需要等待）")


if __name__ == "__main__":
    asyncio.run(demo())
```

### 替代方案

**滑动窗口限流**（更精确但实现更复杂）：

```python
class SlidingWindowLimiter:
    """维护一个时间戳队列，窗口外的旧请求自动过期。"""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: list[float] = []

    async def acquire(self) -> float:
        now = time.monotonic()
        # 清理窗口外的旧请求
        self._timestamps = [
            t for t in self._timestamps
            if now - t < self.window
        ]
        if len(self._timestamps) < self.max_requests:
            self._timestamps.append(now)
            return 0.0
        # 需要等待最早的请求过期
        wait = self._timestamps[0] + self.window - now
        self._timestamps.append(now + wait)
        return max(0, wait)
```

### 常见错误

1. **用 `time.time()` 而非 `time.monotonic()`**：`time.time()` 可能因为 NTP 校时或用户调整系统时钟而产生跳跃，导致补充量计算错误
2. **锁的粒度不对**：锁应该保护整段 `_refill()` + 修改 `_tokens` 的代码，否则可能出现竞态条件
3. **burst 过大**：burst=1000 但 rate=10/min，意味着短时间内可以发 1000 个请求，可能触发 API 的硬限制

---

## 练习 06：安全流式 JSON 解析器

### 思路

核心是一个状态机，追踪：
- 是否在字符串内部（`in_string`）
- 是否遇到转义字符（`escape_next`）
- 括号深度（`brace_depth` 和 `bracket_depth`）

只有当 `brace_depth == 0` 且 `bracket_depth == 0` 时，才尝试将缓冲区内容作为 JSON 解析。

### 参考解答

```python
"""练习 06：安全流式 JSON 解析器。"""

import json
import logging

logger = logging.getLogger(__name__)


class StreamingJSONParser:
    """累积 chunk 并按完整性检测提取完整 JSON 对象。

    解决了流式场景下的核心问题：
      - chunk 边界切割 JSON 结构
      - 字符串中的 '{' '}' 误触发深度变化
      - 多对象连续到达

    用法:
        parser = StreamingJSONParser()
        for chunk in stream:
            for obj in parser.feed(chunk):
                process(obj)
        remaining = parser.flush()
    """

    def __init__(self):
        self._buffer = ""            # 累积的所有字符
        self._start_idx = 0          # 当前待检测区域的起始索引
        self._brace_depth = 0        # 花括号深度
        self._bracket_depth = 0      # 方括号深度
        self._in_string = False      # 当前是否在字符串内
        self._escape_next = False    # 下一个字符是否为转义字符

        # 统计（调试用）
        self._total_objects = 0
        self._total_chunks = 0

    def feed(self, chunk: str) -> list[dict | list]:
        """喂入一个文本 chunk，返回新检测到的完整 JSON 对象。

        Args:
            chunk: 新的文本片段

        Returns:
            解析出的 JSON 对象列表
        """
        self._buffer += chunk
        self._total_chunks += 1
        return self._extract_complete()

    def _extract_complete(self) -> list[dict | list]:
        """扫描缓冲区，检测并提取所有完整的 JSON 对象。"""
        results = []
        i = self._start_idx

        while i < len(self._buffer):
            ch = self._buffer[i]

            # --- 处理转义 ---
            if self._escape_next:
                self._escape_next = False
                i += 1
                continue

            # --- 字符串边界 ---
            if ch == '\\' and self._in_string:
                self._escape_next = True
                i += 1
                continue

            if ch == '"':
                self._in_string = not self._in_string
                i += 1
                continue

            # 在字符串内部，忽略括号
            if self._in_string:
                i += 1
                continue

            # --- 追踪括号深度 ---
            if ch == '{':
                self._brace_depth += 1
            elif ch == '}':
                self._brace_depth -= 1
            elif ch == '[':
                self._bracket_depth += 1
            elif ch == ']':
                self._bracket_depth -= 1

            # --- 检测顶级对象完成 ---
            if (self._brace_depth == 0
                    and self._bracket_depth == 0
                    and ch in ('}', ']')
                    and not self._in_string):
                # 从 start_idx 到 i+1 是一个候选完整 JSON
                candidate = self._buffer[self._start_idx:i + 1]

                # 尝试解析
                try:
                    obj = json.loads(candidate)
                    results.append(obj)
                    self._total_objects += 1
                    self._start_idx = i + 1
                except json.JSONDecodeError:
                    # 可能不是有效的 JSON（例如在文本中出现的零散括号）
                    # 忽略，继续扫描
                    pass

            i += 1

        return results

    def flush(self) -> str:
        """返回缓冲区中尚未完成的剩余内容，并重置状态。

        Returns:
            未完成的文本片段
        """
        remaining = self._buffer[self._start_idx:]
        self._buffer = ""
        self._start_idx = 0
        self._brace_depth = 0
        self._bracket_depth = 0
        self._in_string = False
        self._escape_next = False
        return remaining

    def reset(self) -> None:
        """完全重置解析器状态。"""
        self._buffer = ""
        self._start_idx = 0
        self._brace_depth = 0
        self._bracket_depth = 0
        self._in_string = False
        self._escape_next = False

    @property
    def stats(self) -> dict:
        """获取统计信息。"""
        return {
            "total_chunks": self._total_chunks,
            "total_objects": self._total_objects,
            "buffer_size": len(self._buffer),
            "pending_start": self._start_idx,
            "pending_size": len(self._buffer) - self._start_idx,
        }


# ============================================================
# 扩展：支持部分对象回调（在对象还未完成时就能预览内容）
# ============================================================

class StreamingJSONParserWithProgress(StreamingJSONParser):
    """带进度回调的流式解析器。

    在检测到对象的键值对部分完成时触发回调，
    适合"边解析边更新 UI"的场景。
    """

    def __init__(self, on_key_value=None):
        super().__init__()
        self._on_key_value = on_key_value
        self._current_keys: set[str] = set()

    def feed(self, chunk: str) -> list[dict | list]:
        results = super().feed(chunk)

        # 扫描未完成区域中的键值对
        if self._on_key_value and self._start_idx < len(self._buffer):
            pending = self._buffer[self._start_idx:]
            # 简单正则在未完成 JSON 中找键值对
            import re
            for match in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', pending):
                key = match.group(1)
                value = match.group(2)
                if key not in self._current_keys:
                    self._current_keys.add(key)
                    self._on_key_value(key, value)

        return results

    def reset(self) -> None:
        super().reset()
        self._current_keys.clear()


# ============================================================
# 演示
# ============================================================

def demo():
    print("=" * 50)
    print("练习 06: 安全流式 JSON 解析器")
    print("=" * 50)

    # --- 测试 1: 嵌套对象 ---
    print("\n--- 测试 1: 嵌套 JSON 对象 ---")
    parser = StreamingJSONParser()

    chunks = [
        '{"name": "Alice", "data": {"c',
        'ity": "Beijing"}} {"name": "Bob',
        '", "age": 30, "hobbies": ["re',
        'ading", "coding"]}',
    ]

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: '{chunk[:40]}...'")
        results = parser.feed(chunk)
        for obj in results:
            print(f"    → 完整对象: {obj}")

    remaining = parser.flush()
    if remaining:
        print(f"  剩余: '{remaining}'")
    print(f"  统计: {parser.stats}")

    # --- 测试 2: 数组 ---
    print("\n--- 测试 2: JSON 数组 ---")
    parser2 = StreamingJSONParser()

    chunks2 = ['[1, 2, ', '3, {"key": "v', 'alue"}]']

    for i, chunk in enumerate(chunks2):
        results = parser2.feed(chunk)
        for obj in results:
            print(f"  Chunk {i} → {obj}")

    # --- 测试 3: 字符串中的括号 ---
    print("\n--- 测试 3: 字符串内含括号 ---")
    parser3 = StreamingJSONParser()

    # 字符串中有 '{' 和 '}'，不应影响深度计数
    tricky = '{"text": "this is {not} a [json] object", "valid": true}'
    # 完整发送
    results = parser3.feed(tricky)
    for obj in results:
        print(f"  正确解析: {obj}")
        assert obj["text"] == "this is {not} a [json] object"
        print(f"  ✅ 字符串内的括号未被误判为结构边界")

    # --- 测试 4: 真实场景模拟 ---
    print("\n--- 测试 4: 模拟 LLM 流式输出 ---")

    # 模拟 LLM 返回的 JSONL 格式（每行一个 JSON）
    llm_output = (
        '{"type": "thought", "content": "我需要查询数据库"}\n'
        '{"type": "tool_call", "name": "search", "args": {"query": "RAG"}}\n'
        '{"type": "observation", "result": ["doc1", "doc2"]}\n'
        '{"type": "answer", "content": "RAG是检索增强生成...", "sources": [1, 2]}\n'
    )

    # 模拟随机切割
    import random
    chunks4 = []
    remaining = llm_output
    while remaining:
        cut = random.randint(3, 15)
        chunks4.append(remaining[:cut])
        remaining = remaining[cut:]

    print(f"  原始文本长度: {len(llm_output)} 字符")
    print(f"  切割为 {len(chunks4)} 个 chunks: {[c[:10] for c in chunks4]}")

    parser4 = StreamingJSONParser()
    all_objects = []
    for chunk in chunks4:
        objs = parser4.feed(chunk)
        all_objects.extend(objs)

    print(f"  解析出 {len(all_objects)} 个对象:")
    for obj in all_objects:
        print(f"    type={obj.get('type', '?')}, content={str(obj)[:60]}...")


if __name__ == "__main__":
    demo()
```

### 替代方案

**使用 `ijson` 增量解析器**（适合超大 JSON）：

```python
import ijson
import io

# ijson 可以逐元素解析 JSON，不需要完整加载
# 但需要完整的 JSON 字节流，对于 chunk 拼接场景帮助有限
data = io.BytesIO(large_json_bytes)
parser = ijson.parse(data)
for prefix, event, value in parser:
    if event == 'map_key':
        ...
```

**使用 `json.JSONDecoder.raw_decode()`**（适合简单场景）：

```python
decoder = json.JSONDecoder()
buffer = ""
for chunk in chunks:
    buffer += chunk
    try:
        obj, idx = decoder.raw_decode(buffer)
        # 成功！从 buffer 中移除已解析的部分
        buffer = buffer[idx:].lstrip()
        yield obj
    except json.JSONDecodeError:
        pass  # 不完整，继续累积
```

### 常见错误

1. **直接对每个 chunk 调用 `json.loads()`**：这是最常见的错误。JSON 在 chunk 边界可能不完整。
2. **忘记追踪字符串状态**：JSON 字符串中的 `{}[]` 不应影响深度计数
3. **忘记转义字符**：`\"` 不是字符串结束，`\\` 后的 `"` 也不是
4. **没有 `flush()` 方法**：流结束后可能存在未完成的数据

---

## 练习 07：用 pydantic-settings 创建配置类

### 思路

`pydantic-settings` 的 `BaseSettings` 自动从环境变量和 `.env` 文件读取值。关键是：
1. 使用 `SettingsConfigDict` 配置 `.env` 文件路径
2. 使用 `model_post_init` 做初始化后验证
3. 字段名自动映射为大写环境变量名

### 参考解答

```python
"""练习 07：用 pydantic-settings 创建配置类。"""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    """应用程序配置。

    从 .env 文件和系统环境变量自动加载。优先级：环境变量 > .env 文件 > 默认值。

    Usage:
        settings = AppSettings()
        settings = AppSettings(_env_file=".env.production")
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # 忽略未知的环境变量（不报错）
        case_sensitive=False,    # 环境变量名不区分大小写
    )

    # ---- API Keys ----
    openai_api_key: str = Field(
        default="",
        description="OpenAI API 密钥",
        min_length=0,
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API 密钥",
    )

    # ---- 模型配置 ----
    default_model: str = Field(
        default="gpt-4o-mini",
        description="默认使用的 LLM 模型",
        examples=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"],
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature 参数（0.0-2.0）",
    )
    max_output_tokens: int = Field(
        default=4096,
        ge=1,
        le=16384,
        validation_alias="MAX_OUTPUT_TOKENS",
        description="最大输出 token 数",
    )

    # ---- 请求配置 ----
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="API 调用失败后的最大重试次数",
    )
    request_timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="API 请求超时时间（秒）",
    )

    # ---- 日志和调试 ----
    log_level: str = Field(
        default="INFO",
        description="日志级别（DEBUG, INFO, WARNING, ERROR）",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    enable_cache: bool = Field(
        default=True,
        description="是否启用响应缓存",
    )

    # ---- 向量数据库配置（演示用）----
    vector_db_url: str = Field(
        default="",
        description="向量数据库连接 URL",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding 模型名称",
    )

    # ============================================================
    # 初始化后验证
    # ============================================================

    @model_validator(mode="after")
    def validate_config(self) -> "AppSettings":
        """在模型初始化后执行额外的验证。"""
        # 警告：两个 API Key 都未设置
        if not self.openai_api_key and not self.anthropic_api_key:
            logger.warning(
                "⚠️  openai_api_key 和 anthropic_api_key 都未设置。\n"
                "   请设置至少一个 API Key：\n"
                "   1. 在 .env 文件中添加 OPENAI_API_KEY=xxx\n"
                "   2. 或设置环境变量 export OPENAI_API_KEY=xxx\n"
                "   3. 或设置环境变量 export ANTHROPIC_API_KEY=xxx"
            )

        # 验证 temperature 在合理范围
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"temperature 必须在 0.0-2.0 之间，实际: {self.temperature}"
            )

        # 验证日志级别
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"log_level 必须是 {valid_levels} 之一，实际: {self.log_level}"
            )

        return self

    # ============================================================
    # 辅助方法
    # ============================================================

    def masked_keys(self) -> dict[str, str]:
        """返回脱敏后的 API Key（只显示前 8 和后 4 字符）。

        Returns:
            字典，key 为 key 名称（不含 "api_key" 后缀），value 为脱敏后的 key

        Example:
            >>> settings.masked_keys()
            {'openai': 'sk-proj-...mnop', 'anthropic': '(未设置)'}
        """
        masked = {}

        for key_name in ["openai_api_key", "anthropic_api_key"]:
            short_name = key_name.replace("_api_key", "")
            value = getattr(self, key_name, "")

            if not value:
                masked[short_name] = "(未设置)"
            elif len(value) <= 12:
                masked[short_name] = value[:4] + "****"
            else:
                masked[short_name] = f"{value[:8]}...{value[-4:]}"

        return masked

    def get_active_key(self, provider: str) -> str:
        """根据模型选择对应的 API Key。

        Args:
            provider: "openai" 或 "anthropic"

        Returns:
            API Key 字符串

        Raises:
            ValueError: 如果对应 Key 未设置
        """
        key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }

        key = key_map.get(provider.lower(), "")
        if not key:
            raise ValueError(
                f"未找到 {provider} API Key。"
                f" 请在 .env 或环境变量中设置。"
            )
        return key

    def to_safe_dict(self) -> dict[str, Any]:
        """导出为字典，API Key 被脱敏处理。"""
        data = self.model_dump()
        # 脱敏 API Key
        masked = self.masked_keys()
        data["openai_api_key"] = masked.get("openai", "")
        data["anthropic_api_key"] = masked.get("anthropic", "")
        return data

    def display(self) -> None:
        """打印友好的配置概览。"""
        print("=" * 50)
        print("应用配置")
        print("=" * 50)
        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name)

            # 脱敏 API Key
            if "api_key" in field_name:
                masked = self.masked_keys()
                short_name = field_name.replace("_api_key", "")
                display_value = masked.get(short_name, "***")
            else:
                display_value = value

            desc = field_info.description or ""
            print(f"  {field_name:25s} = {display_value}")
            if desc:
                print(f"  {'':25s}   ({desc})")
        print()


# ============================================================
# 演示
# ============================================================

def demo():
    print("=" * 50)
    print("练习 07: pydantic-settings 配置类")
    print("=" * 50)

    # 模拟设置环境变量
    import os
    os.environ["OPENAI_API_KEY"] = "sk-proj-abcdefghijklmnopqrstuvwxyz"
    os.environ["DEFAULT_MODEL"] = "gpt-4o"
    os.environ["TEMPERATURE"] = "0.3"
    os.environ["LOG_LEVEL"] = "DEBUG"

    # 如果想完全使用 .env 文件，模拟写入临时文件
    temp_env = Path(__file__).parent / ".env.test"
    temp_env.write_text(
        "ANTHROPIC_API_KEY=sk-ant-test123456789\n"
        "MAX_OUTPUT_TOKENS=8192\n"
        "ENABLE_CACHE=false\n"
    )

    # 使用 .env.test 文件初始化
    settings = AppSettings(_env_file=temp_env)

    # 打印配置
    settings.display()

    # 脱敏 Key
    print("脱敏 Key:", settings.masked_keys())

    # 导出安全字典
    print("\n安全导出:", settings.to_safe_dict())

    # 获取 Provider Key
    try:
        key = settings.get_active_key("openai")
        print(f"\nOpenAI Key: {key[:10]}...")
    except ValueError as e:
        print(f"\n{e}")

    # 测试验证
    print("\n--- 测试验证 ---")
    try:
        bad = AppSettings(temperature=5.0)  # 超出范围
    except Exception as e:
        print(f"Temperature=5.0 被拒绝: {type(e).__name__}")

    # 清理
    temp_env.unlink(missing_ok=True)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("DEFAULT_MODEL", None)
    os.environ.pop("TEMPERATURE", None)
    os.environ.pop("LOG_LEVEL", None)


if __name__ == "__main__":
    # 如果没有 pydantic-settings，用纯 Pydantic 替代
    try:
        demo()
    except ImportError:
        print("请安装 pydantic-settings: pip install pydantic-settings")
```

### 替代方案

**不使用 pydantic-settings，手动读 .env**：

```python
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY", "")
default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
max_retries = int(os.getenv("MAX_RETRIES", "3"))
```

缺点：没有类型检查、没有验证、没有自动补全。

### 常见错误

1. **环境变量名不匹配**：`AppSettings.openai_api_key` 对应环境变量 `OPENAI_API_KEY`（全大写），不是 `openai_api_key`
2. **bool 类型的环境变量**：`ENABLE_CACHE=false` 读入时为字符串 `"false"`，`pydantic-settings` 只把 `"true"`, `"1"`, `"yes"`, `"on"` 解析为 True
3. **`.env` 文件路径错误**：默认从当前工作目录读取，不是从项目根目录

---

## 练习 08：Ollama 本地模型 vs OpenAI API 延迟对比

### 思路

构建一个模拟框架（或真实测试框架），对比：
- **TTFT（Time To First Token）**：第一个 token 到达的时间
- **总延迟**：从发送请求到接收完整响应
- **生成速度**：output_tokens / (总延迟 - TTFT)

### 参考解答

```python
"""练习 08：Ollama 本地模型 vs OpenAI API 延迟对比。"""

import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# 数据模型
# ============================================================

@dataclass
class BenchmarkResult:
    """单次 benchmark 的结果。"""
    model: str
    prompt_name: str
    prompt_tokens: int
    ttft: float           # Time To First Token（秒）
    total_latency: float  # 总延迟（秒）
    output_tokens: int
    tokens_per_second: float

    @property
    def generation_latency(self) -> float:
        """纯生成阶段的延迟（总延迟 - TTFT）。"""
        return self.total_latency - self.ttft


@dataclass
class ModelSimConfig:
    """模拟模型的配置参数。"""
    name: str
    ttft_base: float      # 基础 TTFT（秒）
    ttft_jitter: float    # TTFT 的随机抖动
    tokens_per_second: float  # token 生成速度（tok/s）
    tps_jitter: float     # 生成速度的随机抖动


# ============================================================
# 模拟配置
# ============================================================

# 实际场景参考值（基于社区报告和基准测试）：
# - Ollama 本地（RTX 4090, llama3.1 8B）：TTFT ~0.05s, 速度 ~80 tok/s
# - OpenAI gpt-4o-mini：TTFT ~0.5-0.8s, 速度 ~45 tok/s
# - OpenAI gpt-4o：TTFT ~0.8-1.5s, 速度 ~30 tok/s

MODEL_CONFIGS: dict[str, ModelSimConfig] = {
    "ollama:llama3.1:8b": ModelSimConfig(
        name="ollama:llama3.1:8b",
        ttft_base=0.06,
        ttft_jitter=0.03,
        tokens_per_second=80.0,
        tps_jitter=10.0,
    ),
    "ollama:qwen2.5:7b": ModelSimConfig(
        name="ollama:qwen2.5:7b",
        ttft_base=0.05,
        ttft_jitter=0.03,
        tokens_per_second=75.0,
        tps_jitter=8.0,
    ),
    "ollama:deepseek-r1:8b": ModelSimConfig(
        name="ollama:deepseek-r1:8b",
        ttft_base=0.08,
        ttft_jitter=0.04,
        tokens_per_second=60.0,
        tps_jitter=12.0,
    ),
    "openai:gpt-4o-mini": ModelSimConfig(
        name="openai:gpt-4o-mini",
        ttft_base=0.60,
        ttft_jitter=0.20,
        tokens_per_second=45.0,
        tps_jitter=5.0,
    ),
    "openai:gpt-4o": ModelSimConfig(
        name="openai:gpt-4o",
        ttft_base=1.0,
        ttft_jitter=0.30,
        tokens_per_second=30.0,
        tps_jitter=5.0,
    ),
    "openai:gpt-3.5-turbo": ModelSimConfig(
        name="openai:gpt-3.5-turbo",
        ttft_base=0.40,
        ttft_jitter=0.15,
        tokens_per_second=55.0,
        tps_jitter=5.0,
    ),
}


# ============================================================
# 模拟 API 调用（可替换为真实 API 调用）
# ============================================================

async def simulate_call(
    config: ModelSimConfig,
    prompt_tokens: int,
    output_tokens: int,
) -> BenchmarkResult:
    """模拟一次 API 调用并测量延迟。

    Args:
        config: 模型配置
        prompt_tokens: 输入 token 数（影响处理时间）
        output_tokens: 预期输出 token 数

    Returns:
        BenchmarkResult
    """
    # 1. TTFT = 基础值 + token 处理时间 + 随机抖动
    ttft = (
        config.ttft_base
        + prompt_tokens * 0.0001  # 更大的 prompt 需要更长的首 token 时间
        + random.uniform(-config.ttft_jitter, config.ttft_jitter)
    )
    ttft = max(0.02, ttft)  # 最小 20ms

    await asyncio.sleep(ttft)  # 模拟 TTFT 等待

    # 2. 生成阶段
    tps = max(
        5.0,
        config.tokens_per_second + random.uniform(-config.tps_jitter, config.tps_jitter)
    )
    generation_time = output_tokens / tps
    await asyncio.sleep(generation_time)

    # 3. 总延迟
    total_latency = ttft + generation_time

    return BenchmarkResult(
        model=config.name,
        prompt_name="",  # 将由外部设置
        prompt_tokens=prompt_tokens,
        ttft=round(ttft, 4),
        total_latency=round(total_latency, 4),
        output_tokens=output_tokens,
        tokens_per_second=round(output_tokens / generation_time, 2) if generation_time > 0 else 0,
    )


# ============================================================
# Benchmark 主逻辑
# ============================================================

@dataclass
class BenchmarkSummary:
    """多次测试的汇总统计。"""
    model: str
    prompt_name: str
    prompt_tokens: int
    avg_ttft: float
    avg_latency: float
    avg_tps: float
    min_latency: float
    max_latency: float
    std_latency: float
    sample_count: int


async def benchmark_single(
    model_key: str,
    prompt_name: str,
    prompt_tokens: int,
    output_tokens: int,
    runs: int = 5,
) -> BenchmarkSummary:
    """对单个模型 × 单个 prompt 进行多次测试。

    Args:
        model_key: 模型标识（如 "openai:gpt-4o-mini"）
        prompt_name: 提示词名称（"short", "medium", "long"）
        prompt_tokens: 输入 token 数
        output_tokens: 预期输出 token 数
        runs: 测试次数

    Returns:
        BenchmarkSummary
    """
    config = MODEL_CONFIGS.get(model_key)
    if config is None:
        raise ValueError(f"未知模型: {model_key}")

    results: list[BenchmarkResult] = []

    for _ in range(runs):
        result = await simulate_call(config, prompt_tokens, output_tokens)
        result.prompt_name = prompt_name
        results.append(result)

    latencies = [r.total_latency for r in results]
    ttfts = [r.ttft for r in results]
    tpss = [r.tokens_per_second for r in results]

    return BenchmarkSummary(
        model=model_key,
        prompt_name=prompt_name,
        prompt_tokens=prompt_tokens,
        avg_ttft=round(statistics.mean(ttfts), 4),
        avg_latency=round(statistics.mean(latencies), 4),
        avg_tps=round(statistics.mean(tpss), 2),
        min_latency=round(min(latencies), 4),
        max_latency=round(max(latencies), 4),
        std_latency=round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0,
        sample_count=len(results),
    )


async def benchmark(
    prompts: dict[str, tuple[int, int]],  # name → (input_tokens, output_tokens)
    models: list[str],
    runs: int = 5,
) -> list[BenchmarkSummary]:
    """完整的 benchmark 运行。

    Args:
        prompts: 提示词字典，name → (input_tokens, output_tokens)
        models: 要测试的模型列表
        runs: 每个组合的测试次数

    Returns:
        所有测试结果的列表
    """
    tasks = []

    for model_key in models:
        for prompt_name, (input_tok, output_tok) in prompts.items():
            tasks.append(
                benchmark_single(
                    model_key=model_key,
                    prompt_name=prompt_name,
                    prompt_tokens=input_tok,
                    output_tokens=output_tok,
                    runs=runs,
                )
            )

    results = await asyncio.gather(*tasks)
    return results


# ============================================================
# 结果展示
# ============================================================

def print_results_table(results: list[BenchmarkSummary]) -> None:
    """格式化打印 benchmark 结果。"""

    # 表头
    header = (
        f"{'Model':<28s} {'Prompt':<10s} {'#Tok':>5s} "
        f"{'TTFT':>8s} {'Latency':>8s} {'Min':>8s} {'Max':>8s} "
        f"{'Std':>7s} {'Tok/s':>7s}"
    )
    sep = "─" * len(header)

    print(sep)
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r.model:<28s} {r.prompt_name:<10s} {r.prompt_tokens:>5d} "
            f"{r.avg_ttft:>7.3f}s {r.avg_latency:>7.3f}s "
            f"{r.min_latency:>7.3f}s {r.max_latency:>7.3f}s "
            f"{r.std_latency:>6.3f}s {r.avg_tps:>6.1f}"
        )

    print(sep)


def analyze_results(results: list[BenchmarkSummary]) -> None:
    """输出分析结论。"""

    # 按 prompt 长度分组
    by_prompt: dict[str, list[BenchmarkSummary]] = {}
    for r in results:
        by_prompt.setdefault(r.prompt_name, []).append(r)

    print("\n=== 分析 ===")

    for prompt_name, group in by_prompt.items():
        # 找出最快和最慢
        fastest = min(group, key=lambda x: x.avg_latency)
        slowest = max(group, key=lambda x: x.avg_latency)

        print(f"\n{prompt_name} ({group[0].prompt_tokens} tokens):")
        print(f"  最快: {fastest.model} ({fastest.avg_latency:.3f}s)")
        print(f"  最慢: {slowest.model} ({slowest.avg_latency:.3f}s)")
        print(f"  差距: {slowest.avg_latency / fastest.avg_latency:.1f}x")

        # 本地 vs 云端对比
        local = [r for r in group if r.model.startswith("ollama")]
        cloud = [r for r in group if r.model.startswith("openai")]

        if local and cloud:
            avg_local = statistics.mean([r.avg_latency for r in local])
            avg_cloud = statistics.mean([r.avg_latency for r in cloud])
            ttft_local = statistics.mean([r.avg_ttft for r in local])
            ttft_cloud = statistics.mean([r.avg_ttft for r in cloud])

            print(f"\n  本地模型 (Ollama):")
            print(f"    平均延迟: {avg_local:.3f}s, 平均 TTFT: {ttft_local:.3f}s")
            print(f"  云端模型 (OpenAI):")
            print(f"    平均延迟: {avg_cloud:.3f}s, 平均 TTFT: {ttft_cloud:.3f}s")
            print(f"  本地/云端延迟比: {avg_local / avg_cloud:.2f}x")
            print(f"  本地/云端 TTFT 比: {ttft_local / ttft_cloud:.2f}x")


# ============================================================
# 演示
# ============================================================

async def demo():
    print("=" * 50)
    print("练习 08: Ollama vs OpenAI 延迟对比")
    print("=" * 50)

    # 定义测试场景
    prompts = {
        "短": (15, 50),      # (input_tokens, expected_output_tokens)
        "中": (120, 200),
        "长": (500, 400),
    }

    models = [
        "ollama:llama3.1:8b",
        "ollama:qwen2.5:7b",
        "openai:gpt-4o-mini",
        "openai:gpt-3.5-turbo",
    ]

    print(f"\n测试 {len(models)} 个模型 × {len(prompts)} 种 prompt × 5 次重复")
    print("模拟进行中...\n")

    # 运行 benchmark
    results = await benchmark(prompts, models, runs=5)

    # 打印结果
    print_results_table(results)

    # 分析
    analyze_results(results)


if __name__ == "__main__":
    asyncio.run(demo())
```

### 替代方案

**真实 API 测试版本**（需要实际 API Key）：

```python
import time
from openai import AsyncOpenAI

async def real_openai_call(prompt: str, model: str = "gpt-4o-mini"):
    client = AsyncOpenAI()
    start = time.monotonic()
    first_token_time = None

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    output_tokens = 0
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.monotonic()
        if chunk.choices[0].delta.content:
            output_tokens += 1

    total_time = time.monotonic() - start
    ttft = first_token_time - start if first_token_time else 0

    return {
        "ttft": ttft,
        "total_latency": total_time,
        "output_tokens": output_tokens,
    }
```

### 常见错误

1. **单次测试做结论**：LLM API 延迟抖动很大，至少跑 5 次取平均
2. **忽略 TTFT**：对于流式场景，TTFT 比总延迟更重要（用户体验）
3. **不同时间段延迟不同**：API 延迟在高峰时段可能翻倍，benchmark 应在多个时段各跑一次

---

## 练习 09：API Key 轮换管理器

### 思路

核心设计：
1. 每个 Key 有一个状态：`active`, `cooldown`, `revoked`
2. 优先返回主 Key（primary），主 Key 不可用时 fallback 到备用 Key
3. 冷却时间过后自动恢复
4. 线程安全用 `asyncio.Lock`

### 参考解答

```python
"""练习 09：API Key 轮换管理器。"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class KeyStatus(str, Enum):
    """API Key 的状态。"""
    ACTIVE = "active"        # 正常可用
    COOLDOWN = "cooldown"   # 暂时不可用（冷却中）
    REVOKED = "revoked"     # 永久禁用


@dataclass
class KeyInfo:
    """单个 API Key 的元信息。"""
    key: str
    status: KeyStatus = KeyStatus.ACTIVE
    failures: int = 0
    last_failure_time: float = 0.0
    last_failure_reason: str = ""
    cooldown_until: float = 0.0
    total_uses: int = 0

    @property
    def masked(self) -> str:
        """脱敏显示 Key。"""
        if len(self.key) <= 12:
            return self.key[:4] + "****"
        return self.key[:8] + "..." + self.key[-4:]


class APIKeyManager:
    """API Key 轮换和故障转移管理器。

    特性：
      - 主 Key + 多个备用 Key
      - 自动故障检测和切换
      - 冷却时间机制
      - 线程安全

    Usage:
        manager = APIKeyManager(
            primary_key="sk-primary-xxx",
            backup_keys=["sk-backup-yyy", "sk-backup-zzz"],
        )

        key = await manager.get_key()
        try:
            result = await call_api(key)
            manager.mark_success(key)
        except Exception as e:
            manager.mark_failed(key, str(e))
    """

    def __init__(
        self,
        primary_key: str,
        backup_keys: list[str] | None = None,
        *,
        cooldown_seconds: float = 60.0,
    ):
        """
        Args:
            primary_key: 主 API Key
            backup_keys: 备用 Key 列表
            cooldown_seconds: 被标记为失败的 Key 的冷却时间（秒）
        """
        if not primary_key:
            raise ValueError("primary_key 不能为空")

        self._cooldown_seconds = cooldown_seconds
        self._lock = asyncio.Lock()

        # 初始化 Key 信息
        self._keys: dict[str, KeyInfo] = {}
        self._primary = primary_key

        self._add_key(primary_key, is_primary=True)
        for bk in (backup_keys or []):
            self._add_key(bk, is_primary=False)

        # 统计
        self._total_switches = 0

    def _add_key(self, key: str, is_primary: bool = False) -> None:
        """注册一个新的 Key。"""
        if key in self._keys:
            return
        self._keys[key] = KeyInfo(key=key)

    # ============================================================
    # Key 获取
    # ============================================================

    async def get_key(self) -> str:
        """获取当前可用的 API Key。

        优先级：
          1. 主 Key（如果可用）
          2. 第一个可用的备用 Key
          3. 如果所有 Key 都在冷却中，等待第一个冷却完毕

        Returns:
            API Key 字符串

        Raises:
            RuntimeError: 所有 Key 都被永久禁用
        """
        async with self._lock:
            # 检查主 Key
            if self._is_key_available(self._primary):
                return self._primary

            # 检查备用 Key
            for key, info in self._keys.items():
                if key == self._primary:
                    continue
                if self._is_key_available(key):
                    return key

            # 所有 Key 都被禁用或冷却中
            # 找到最早恢复的冷却 Key
            soonest_key = None
            soonest_time = float('inf')

            for key, info in self._keys.items():
                if info.status == KeyStatus.COOLDOWN:
                    if info.cooldown_until < soonest_time:
                        soonest_time = info.cooldown_until
                        soonest_key = key

            if soonest_key:
                wait_time = max(0, soonest_time - time.monotonic())
                self._total_switches += 1
                # 释放锁后等待，然后递归获取
                # 注意：实际场景中应由调用方处理等待
                if wait_time > 0:
                    raise KeyCoolingDownError(
                        f"所有 Key 都在冷却中。最早恢复: {soonest_key[:8]}... "
                        f"({wait_time:.1f}s 后)",
                        retry_after=wait_time,
                    )

            # 检查是否有永久禁用的 Key
            active_or_cooling = [
                k for k, i in self._keys.items()
                if i.status != KeyStatus.REVOKED
            ]
            if not active_or_cooling:
                raise RuntimeError("所有 API Key 都被永久禁用")

            # fallback：返回任意冷却中的 Key（调用方自行处理）
            for key, info in self._keys.items():
                if info.status == KeyStatus.COOLDOWN:
                    return key

            raise RuntimeError("无法获取任何可用的 API Key")

    async def get_key_nonblocking(self) -> str | None:
        """非阻塞获取 Key（如果全部不可用则返回 None）。"""
        try:
            return await self.get_key()
        except (KeyCoolingDownError, RuntimeError):
            return None

    def _is_key_available(self, key: str) -> bool:
        """检查 Key 是否可用。"""
        info = self._keys.get(key)
        if info is None:
            return False

        if info.status == KeyStatus.ACTIVE:
            return True

        if info.status == KeyStatus.COOLDOWN:
            # 检查冷却是否已过
            if time.monotonic() >= info.cooldown_until:
                info.status = KeyStatus.ACTIVE
                info.cooldown_until = 0.0
                return True

        return False

    # ============================================================
    # 状态管理
    # ============================================================

    async def mark_failed(self, key: str, reason: str = "") -> None:
        """标记 Key 为失败状态，将其置入冷却。

        Args:
            key: 失败的 Key
            reason: 失败原因（用于日志）
        """
        async with self._lock:
            info = self._keys.get(key)
            if info is None:
                return

            info.failures += 1
            info.last_failure_time = time.monotonic()
            info.last_failure_reason = reason

            # 如果连续失败超过 3 次，延长冷却时间
            cooldown = self._cooldown_seconds
            if info.failures > 3:
                cooldown = self._cooldown_seconds * (info.failures - 2)

            info.status = KeyStatus.COOLDOWN
            info.cooldown_until = time.monotonic() + cooldown

    async def mark_success(self, key: str) -> None:
        """标记 Key 使用成功（重置失败计数）。"""
        async with self._lock:
            info = self._keys.get(key)
            if info is None:
                return
            info.failures = max(0, info.failures - 1)  # 逐渐恢复
            info.total_uses += 1

    async def revoke_key(self, key: str, reason: str = "手动吊销") -> None:
        """永久禁用某个 Key。"""
        async with self._lock:
            info = self._keys.get(key)
            if info is None:
                return
            info.status = KeyStatus.REVOKED
            info.last_failure_reason = reason

    # ============================================================
    # 查询
    # ============================================================

    async def status(self) -> str:
        """返回所有 Key 的状态摘要。"""
        async with self._lock:
            lines = ["=== API Key 状态 ==="]

            for key, info in self._keys.items():
                status_icon = {
                    KeyStatus.ACTIVE: "✅",
                    KeyStatus.COOLDOWN: "⏳",
                    KeyStatus.REVOKED: "❌",
                }.get(info.status, "❓")

                mark = "primary" if key == self._primary else "backup"
                line = (
                    f"{status_icon} [{mark}] {info.masked}: {info.status.value}"
                )

                if info.failures > 0:
                    line += f" (failures: {info.failures}, uses: {info.total_uses})"
                if info.status == KeyStatus.COOLDOWN:
                    remaining = max(0, info.cooldown_until - time.monotonic())
                    line += f" (冷却剩余: {remaining:.0f}s)"
                if info.last_failure_reason:
                    line += f" [{info.last_failure_reason[:40]}]"

                lines.append(line)

            lines.append(f"总切换次数: {self._total_switches}")
            return "\n".join(lines)

    @property
    def active_key_count(self) -> int:
        """当前可用的 Key 数量。"""
        return sum(
            1 for key, info in self._keys.items()
            if info.status == KeyStatus.ACTIVE
        )


class KeyCoolingDownError(Exception):
    """所有 Key 都在冷却中的异常。"""

    def __init__(self, message: str, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


# ============================================================
# 演示
# ============================================================

async def demo():
    print("=" * 50)
    print("练习 09: API Key 轮换管理器")
    print("=" * 50)

    manager = APIKeyManager(
        primary_key="sk-primary-key-abcdef123456",
        backup_keys=[
            "sk-backup-key-ghijkl789012",
            "sk-backup-key-mnopqr345678",
        ],
        cooldown_seconds=2.0,  # 2 秒冷却（演示用）
    )

    # 1. 正常获取
    print("\n--- 1. 正常获取 ---")
    key = await manager.get_key()
    print(f"获取 Key: {key[:15]}...")

    # 2. 主 Key 失败
    print("\n--- 2. 主 Key 失败，自动切换 ---")
    await manager.mark_failed(key, reason="429 Rate limit exceeded")

    try:
        key2 = await manager.get_key()
        print(f"故障转移 Key: {key2[:15]}...")
    except KeyCoolingDownError as e:
        print(f"冷却中: {e}")

    # 3. 等待冷却后恢复
    print("\n--- 3. 等待冷却后主 Key 恢复 ---")
    await asyncio.sleep(2.5)

    key3 = await manager.get_key()
    print(f"恢复 Key: {key3[:15]}...")

    # 4. 状态查询
    print(f"\n{await manager.status()}")

    # 5. 吊销备用 Key
    print("\n--- 4. 吊销备用 Key ---")
    await manager.revoke_key("sk-backup-key-ghijkl789012", reason="安全策略更新")
    print(f"{await manager.status()}")

    # 6. 所有 Key 都失败
    print("\n--- 5. 所有 Key 都失败 ---")
    await manager.mark_failed("sk-primary-key-abcdef123456", "Server error")
    await manager.mark_failed("sk-backup-key-mnopqr345678", "Server error")

    try:
        await manager.get_key()
    except KeyCoolingDownError as e:
        print(f"预期异常: {e}")


if __name__ == "__main__":
    asyncio.run(demo())
```

### 替代方案

**简化版（只轮换不冷却）**：

```python
import itertools

class SimpleKeyRotator:
    def __init__(self, keys: list[str]):
        self._keys = keys
        self._cycle = itertools.cycle(keys)

    def get_key(self) -> str:
        return next(self._cycle)
```

### 常见错误

1. **忘记线程安全**：并发调用时需要锁保护 Key 状态
2. **冷却时间太短**：如果 API 的 429 要求等待 30 秒，但你只等了 5 秒，会再次触发
3. **没有重置失败计数**：成功使用后应该逐渐恢复失败计数，否则正常使用的 Key 也会因为历史失败被过度惩罚

---

## 练习 10：费用估算函数

### 思路

核心是维护一个定价表。注意单位换算：$X/1M tokens = $X/1,000,000 tokens。

```
cost = (input_tokens / 1_000_000) * input_price_per_million
     + (output_tokens / 1_000_000) * output_price_per_million
```

对于极小的金额（如 $0.00075），需要用 `{:.6f}` 格式化显示。

### 参考解答

```python
"""练习 10：LLM API 费用估算函数。"""

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ============================================================
# 定价数据（价格单位：$/1M tokens）
# ============================================================

class Pricing(NamedTuple):
    """单个模型的定价信息。"""
    model: str
    provider: str
    input_price_per_1m: float  # 每百万输入 token 价格（美元）
    output_price_per_1m: float  # 每百万输出 token 价格（美元）


# 价格数据（最后更新：2026年6月）
# 来源：各平台官方定价页面
PRICING_TABLE: dict[str, Pricing] = {
    # ---- OpenAI ----
    "gpt-4o-mini": Pricing(
        model="gpt-4o-mini",
        provider="openai",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    "gpt-4o": Pricing(
        model="gpt-4o",
        provider="openai",
        input_price_per_1m=2.50,
        output_price_per_1m=10.00,
    ),
    "gpt-4-turbo": Pricing(
        model="gpt-4-turbo",
        provider="openai",
        input_price_per_1m=10.00,
        output_price_per_1m=30.00,
    ),
    "gpt-3.5-turbo": Pricing(
        model="gpt-3.5-turbo",
        provider="openai",
        input_price_per_1m=0.50,
        output_price_per_1m=1.50,
    ),
    # ---- Anthropic ----
    "claude-3.5-sonnet": Pricing(
        model="claude-3.5-sonnet",
        provider="anthropic",
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
    ),
    "claude-3.5-haiku": Pricing(
        model="claude-3.5-haiku",
        provider="anthropic",
        input_price_per_1m=0.80,
        output_price_per_1m=4.00,
    ),
    "claude-3-opus": Pricing(
        model="claude-3-opus",
        provider="anthropic",
        input_price_per_1m=15.00,
        output_price_per_1m=75.00,
    ),
    # ---- Google ----
    "gemini-1.5-flash": Pricing(
        model="gemini-1.5-flash",
        provider="google",
        input_price_per_1m=0.075,   # < 128K context
        output_price_per_1m=0.30,
    ),
    "gemini-1.5-pro": Pricing(
        model="gemini-1.5-pro",
        provider="google",
        input_price_per_1m=1.25,    # < 128K context
        output_price_per_1m=5.00,
    ),
    # ---- DeepSeek ----
    "deepseek-v3": Pricing(
        model="deepseek-v3",
        provider="deepseek",
        input_price_per_1m=0.27,
        output_price_per_1m=1.10,
    ),
}


# ============================================================
# 核心函数
# ============================================================

def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """估算单次 API 调用的费用。

    Args:
        model: 模型名称（如 "gpt-4o-mini", "claude-3.5-sonnet"）
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数

    Returns:
        估算费用（美元），精度到小数点后 6 位

    Raises:
        ValueError: 如果 token 数为负

    Examples:
        >>> estimate_cost("gpt-4o-mini", 1000, 500)
        0.00045
        >>> estimate_cost("gpt-4o", 5000, 1000)
        0.0225
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            f"token 数不能为负: input={input_tokens}, output={output_tokens}"
        )

    # 查找定价
    pricing = PRICING_TABLE.get(model.lower().strip())
    if pricing is None:
        logger.warning(
            f"未知模型 '{model}'，无法估算费用。返回 0.0。\n"
            f"已知模型: {list(PRICING_TABLE.keys())}"
        )
        return 0.0

    # 计算费用
    input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_1m

    return round(input_cost + output_cost, 10)


def format_cost(cost: float) -> str:
    """友好格式化费用金额。

    Args:
        cost: 费用（美元）

    Returns:
        格式化的字符串

    Examples:
        >>> format_cost(0.00045)
        '$0.00045'
        >>> format_cost(1.2345)
        '$1.23'
        >>> format_cost(0.0)
        '$0.00'
    """
    if cost == 0.0:
        return "$0.00"
    elif cost < 0.0001:
        return f"${cost:.8f}"  # 非常小的金额显示更多位
    elif cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def compare_models(
    input_tokens: int,
    output_tokens: int,
    *,
    sort_by: str = "cost",
) -> dict[str, float]:
    """对比所有已知模型的费用。

    Args:
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        sort_by: 排序方式 ("cost" | "model" | "provider")

    Returns:
        按费用从低到高排序的 {model_name: cost} 字典
    """
    costs: dict[str, float] = {}

    for model_name in PRICING_TABLE:
        costs[model_name] = estimate_cost(model_name, input_tokens, output_tokens)

    # 按费用从低到高排序
    if sort_by == "cost":
        return dict(sorted(costs.items(), key=lambda x: x[1]))

    # 按模型名排序
    return dict(sorted(costs.items()))


def estimate_monthly_cost(
    model: str,
    queries_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    days: int = 30,
) -> dict:
    """估算月度费用。

    Args:
        model: 模型名称
        queries_per_day: 每天查询次数
        avg_input_tokens: 每次查询的平均输入 token 数
        avg_output_tokens: 每次查询的平均输出 token 数
        days: 天数（默认 30）

    Returns:
        包含详细分解的字典
    """
    per_query = estimate_cost(model, avg_input_tokens, avg_output_tokens)
    total_queries = queries_per_day * days
    total_cost = per_query * total_queries
    total_input_tokens = avg_input_tokens * total_queries
    total_output_tokens = avg_output_tokens * total_queries

    return {
        "model": model,
        "per_query_cost": per_query,
        "queries_per_day": queries_per_day,
        "total_queries": total_queries,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost": round(total_cost, 4),
        "period": f"{days} 天",
    }


# ============================================================
# 演示
# ============================================================

def demo():
    print("=" * 50)
    print("练习 10: 费用估算")
    print("=" * 50)

    # --- 单次查询费用 ---
    print("\n--- 单次 RAG 查询费用（3000 input + 500 output）---")
    input_t = 3000
    output_t = 500

    for model_name in ["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet", "gpt-3.5-turbo"]:
        cost = estimate_cost(model_name, input_t, output_t)
        print(f"  {model_name:25s}: {format_cost(cost)}")

    # --- 所有模型对比 ---
    print(f"\n--- 所有模型费用对比 ---")
    comparison = compare_models(input_t, output_t)
    for model, cost in comparison.items():
        print(f"  {model:25s}: {format_cost(cost)}")

    # --- 月度费用估算 ---
    print(f"\n--- 月度费用估算 ---")
    monthly = estimate_monthly_cost(
        model="gpt-4o-mini",
        queries_per_day=1000,
        avg_input_tokens=3000,
        avg_output_tokens=500,
    )
    for k, v in monthly.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}" if k == "per_query_cost" else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # --- 费用敏感度分析 ---
    print(f"\n--- 费用敏感度分析：选择哪个模型？---")
    scenarios = [
        ("小规模 (100次/天)", 100, 2000, 500),
        ("中等规模 (1000次/天)", 1000, 2000, 500),
        ("大规模 (10000次/天)", 10000, 2000, 500),
    ]

    for scenario_name, qpd, inp, out in scenarios:
        print(f"\n  {scenario_name}:")
        costs = {}
        for model_name in ["gpt-4o-mini", "gpt-3.5-turbo", "claude-3.5-haiku"]:
            result = estimate_monthly_cost(model_name, qpd, inp, out)
            costs[model_name] = result["total_cost"]
        for m, c in sorted(costs.items(), key=lambda x: x[1]):
            print(f"    {m:25s}: {format_cost(c)}/月")

    # --- 边界测试 ---
    print(f"\n--- 边界测试 ---")
    print(f"0 tokens: {format_cost(estimate_cost('gpt-4o-mini', 0, 0))}")
    print(f"1M tokens: {format_cost(estimate_cost('gpt-4o', 1_000_000, 500_000))}")
    print(f"未知模型: {format_cost(estimate_cost('unknown-model', 1000, 500))}")


if __name__ == "__main__":
    demo()
```

### 替代方案

**使用官方 SDK 的 usage 字段**（最准确）：

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)

usage = response.usage
# usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
# 费用 = prompt_tokens * input_rate + completion_tokens * output_rate
```

### 常见错误

1. **单位混淆**：价格是 $/1M tokens（不是 $/1K tokens），常有新人把 `$0.15/1M` 当成 `$0.15/1K` 导致估算高 1000 倍
2. **忘记 output tokens 的价格通常比 input 贵 3-5 倍**：在做总费用估算时，output 部分的权重不应被忽视
3. **硬编码价格**：各厂商频繁调价，应把定价表放在配置中方便更新
4. **缓存命中不计费**：OpenAI 的 prompt caching 有不同价格（更低），Anthropic 的 cache 命中价格也不同

---

## 总结

这 10 道练习题覆盖了 LLM 开发中最基础的工程问题：

| 编号 | 技能                   | 应用场景                         |
|------|----------------------|--------------------------------|
| 01   | 异步并发控制           | 多路调用、race 模式             |
| 02   | 数据建模               | API 响应结构化、成本追踪         |
| 03   | Token 计数             | 预算管理、成本控制、截断检测     |
| 04   | 指数退避重试           | 网络不稳定、429 处理             |
| 05   | 速率限制               | 大批量处理、避免被封             |
| 06   | 流式解析               | 实时输出、SSE 处理               |
| 07   | 配置管理               | 环境切换、密钥管理               |
| 08   | 性能对比               | 本地 vs 云端选择、模型选型       |
| 09   | 故障转移               | 高可用、Key 轮换                 |
| 10   | 成本估算               | 预算规划、模型选择               |

掌握这些技能后，你可以自信地开始构建生产级的 LLM 应用。
