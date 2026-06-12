# 坑点 #2：Token 超限导致响应截断

> "你给的上下文太长了，我处理不了。以下是截断后的内容..."
> —— 这不是 AI 在耍脾气，是你的 token 预算爆了。

---

## 症状

你在构建一个 RAG 问答系统，一切看起来都正常，直到用户问了一个需要大量上下文的问题。然后：

1. **回答被截断**：LLM 的回答说到一半突然中断，最后一个词可能是 `"the answer is"` 后面就没了
2. **API 返回 400 错误**：响应体里写着 `"context_length_exceeded"` 或 `"This model's maximum context length is X tokens"`
3. **诡异的输出**：LLM 开始"胡言乱语"，输出重复的文本或完全不相关的内容（这是上下文窗口接近饱和时的退化现象）
4. **检索到的文档悄悄丢失**：你检索了 10 个文档片段，但实际只有前 5 个被送进了 LLM——后面的被静默截断了，而你毫不知情
5. **费用暴涨却不自知**：如果你没有做 token 预算，你可能每次调用都在发送远超所需的内容，烧掉大量费用

---

## 根因

不了解各模型的上下文窗口限制，以及"可用空间"远小于"标称值"的事实。

**关键认知**：模型的"总上下文窗口"不等于"你能用的空间"。实际可用的 token 数 = 总窗口 - 输出预留 - prompt 框架 - 系统消息 - 历史对话。

---

## 各模型限制速查表

| 模型                     | 总上下文窗口 | 最大输出  | 实际可用输入（估算） | 典型使用建议     |
|--------------------------|-------------|----------|-------------------|----------------|
| GPT-4o-mini              | 128,000     | 16,384   | ~100,000          | 日常大批量处理   |
| GPT-4o                   | 128,000     | 16,384   | ~100,000          | 复杂推理        |
| GPT-4 Turbo              | 128,000     | 4,096    | ~115,000          | 长文档分析      |
| GPT-3.5 Turbo            | 16,385      | 4,096    | ~10,000           | 简单任务（已过时）|
| Claude 3.5 Sonnet        | 200,000     | 8,192    | ~180,000          | 代码 & 推理     |
| Claude 3.5 Haiku         | 200,000     | 8,192    | ~180,000          | 快速响应        |
| Claude 3 Opus            | 200,000     | 4,096    | ~190,000          | 深度分析        |
| Gemini 1.5 Pro           | 1,000,000   | 8,192    | ~980,000          | 超长上下文      |
| Gemini 1.5 Flash         | 1,000,000   | 8,192    | ~980,000          | 超长上下文快速版 |
| Llama 3.1 8B (本地)      | 128,000     | 取决于部署 | ~100,000         | 本地推理        |
| DeepSeek-V3              | 128,000     | 8,192    | ~110,000          | 性价比高       |

**注意**：
- "最大输出"通常可以在 API 参数中设置，但不能超过模型限制
- OpenAI 的 GPT-4 系列输出上限可通过 `max_tokens` 参数控制
- Anthropic 的 `max_tokens` 参数是必需的（不设置不会自动使用最大值）
- 本地模型（Ollama 等）的限制取决于显存和部署配置

---

## Token 消耗的本质：为什么代码和数据更"贵"

了解 token 消耗的不对等性，才能做出正确的优化决策。同样的字符数，不同内容的 token 消耗可能相差 2-5 倍。

### 直观对比

| 内容 | 每 100 字符消耗的 token 数 | 压缩效率 |
|------|---------------------------|---------|
| 英文段落 | ~25 | 高（~4 字符/token） |
| 中文段落 | ~50 | 中（~2 字符/token） |
| JSON 数据 | ~60 | 低（~1.7 字符/token） |
| 代码片段 | ~70 | 最低（~1.4 字符/token） |

### 为什么代码消耗更多 token

**原因 1：特殊符号几乎每个都是独立 token**

自然语言中，空格和标点往往与相邻单词合并成一个 token。但代码中的 `{ } ( ) [ ] ; : => === != ->` 等符号几乎每个都独立成 token：

```
自然语言: "Hello, world!"  →  2 tokens  （Hello, | world!）
代码:      { x => x + 1 }   →  7 tokens  （{ | x | => | x | + | 1 | }）
```

同样的字符数，代码的 token 数可能是自然语言的 2-3 倍。

**原因 2：缩进空格被重复编码**

代码大量使用空格/制表符缩进。多级缩进的长空白前缀往往无法和后续内容合并为单个 token，导致每行缩进都在额外消耗 token。一个 4 级缩进的代码块，仅缩进空格就可能消耗 10+ token。

**原因 3：变量名拆分严重**

自然语言中的高频词（"the"、"information"、"because"）是固定 token。但代码中的组合变量名——`getUserById`、`fetchOrderList`、`handleSubmitClick`——在训练数据中罕见，tokenizer 会将它们拆成碎片：

```
getUserById  →  get + User + By + Id   （4 tokens，每次大小写切换都可能触发拆分）
```

驼峰命名、下划线命名的每个词段都可能独立成 token。

**原因 4：UUID、哈希值、数字字面量**

`d4f8a2b1-9c3e-4e5f-a7b1-6d2e8f9a0c3b` 这种字符串的几乎所有字符组合都罕见，会被拆得非常细。长数字（如毫秒时间戳 `1718092800000`）同理。

**原因 5：模板字符串和正则表达式**

模板字符串如 `` `Hello, ${name}! You have ${count} messages` `` 中的 `${}` 语法与普通文本交替出现，tokenizer 难以将整段合并。正则表达式如 `/^[\w.-]+@[\w.-]+\.\w{2,}$/` 符号更为密集，几乎逐字符编码。

### 为什么中文消耗更多 token

英文的高频词和常见词根可以被编码为单个 token，但中文每个汉字通常需要 1-2 个 token（常用字可能 1 token，生僻字可能 2-3 token），标点符号也不算在内。简而言之：**英文靠"词"压缩，中文靠"字"编码，后者天然密度低。**

---

## 修复方案

### 方案 0：从源头精简 Prompt

做好 token 预算管理的前提是 prompt 本身已经足够紧凑。以下是经过验证的优化策略。

**语言层面**

| 策略 | 效果 | 示例 |
|------|------|------|
| 用英文写指令（中文仅在必要时用） | 节省 ~50% | `翻译下列文本为英文：`（11 tokens）→ `Translate to English:`（4 tokens） |
| 删掉礼貌用语和填充词 | 节省 ~10-20% | `Could you please help me...` → `Analyze this code:` |
| 使用缩写和约定符号 | 节省 ~5-10% | `w/` 替代 `with`、`b/c` 替代 `because`、`btw` 替代 `between`、`→` 替代 `becomes` |

**数据格式层面**

| 策略 | 效果 | 说明 |
|------|------|------|
| 紧凑 JSON 替代 pretty-print | 节省 ~40% | 删除换行和缩进空格；`{"a":1,"b":2}` vs 带缩进的版本 |
| CSV/TSV 替代 JSON 数组 | 节省 ~30-50% | 大量结构化数据时，CSV 的表头+数据比 JSON 的每行 key-value 省得多 |
| 代码示例精简 | 节省 ~15-30% | 删掉无关的 import、注释、模板代码，用 `...` 省略无关部分 |

**架构层面**

- **System prompt 放静态信息**：角色设定、规则、格式要求等不变内容放进 system prompt，避免每条 user message 重复携带
- **引用而非复制**：多轮对话中用"你上一条回复中提到的 X"代替重新粘贴大段前文
- **分阶段处理长文档**：先分段总结 → 再汇总 → 最后分析，每一轮只占用一小部分窗口

> **核心原则：tokenizer 不关心排版美观，只关心信息密度。每一对多余的引号空格、每一个"请"和"谢谢"，都是在为排版付费而不是为智能付费。**

**一个直观的改造例子**

改造前（~95 tokens）：

> 请你作为一个经验丰富的代码审查专家，仔细帮我检查下面这段 JavaScript 代码。请重点关注潜在的安全漏洞、性能问题，以及不符合最佳实践的地方。如果发现问题，请详细说明并提供修改建议。代码如下：
> ```javascript
> function processUserData(userData) {
>     // First we validate the input
>     if (userData && userData.name && userData.email) {
>         // Then we save to database
>         ...
>     }
> }
> ```

改造后（~25 tokens）：

> Review for bugs, security, perf:
> ```js
> function processUserData(userData) {
>   if (userData && userData.name && userData.email) { ... }
> }
> ```

**适用场景判断**

| 场景 | 是否优先精简 Prompt |
|------|--------------------|
| 日常对话/简单任务 | ✅ 精简即可，不需要预算管理器 |
| 单次问答 | ✅ 精简优先，预算管理器过重 |
| 生产环境 RAG 管道 | 精简 + 预算管理器并用 |
| 长文档分析 | 精简 + 分阶段处理 |
| 多轮对话 Agent | 精简 + 历史截断策略 |

### 方案 1：发送前精确计算 Token 数

不要凭"感觉"判断内容长度。用 `tiktoken` 精确计算。

### 方案 2：设置 `max_tokens` 控制输出

永远不要依赖默认值。显式设置输出 token 上限，防止输出也被截断。

### 方案 3：实现 Token 预算管理器

这是最完整的解决方案：把上下文窗口当作"预算"，精确分配输入、输出、系统消息、历史对话的份额。

---

## 代码示例：完整的 Token 预算管理器

```python
# token_budget.py
"""Token 预算管理器：确保不超出模型的上下文窗口限制。

核心思想：
  总预算 = 上下文窗口上限
  分配策略：
    1. 预留输出空间（max_tokens 或估算值）
    2. 系统消息固定占用
    3. 历史对话按需分配（可以从旧到新截断）
    4. 检索文档分配剩余空间
"""

import tiktoken
from dataclasses import dataclass, field
from typing import Any, Literal
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 模型配置表
# ============================================================

@dataclass
class ModelConfig:
    """单个模型的配置信息。"""
    name: str
    context_window: int       # 总上下文窗口
    max_output: int           # 最大输出 token
    encoding_name: str        # tiktoken 编码器名称
    provider: Literal["openai", "anthropic", "google", "local"]
    input_cost_per_1k: float = 0.0   # 每 1K input token 成本（美元）
    output_cost_per_1k: float = 0.0  # 每 1K output token 成本（美元）


# 预定义的模型配置
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        context_window=128_000,
        max_output=16_384,
        encoding_name="o200k_base",
        provider="openai",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
    ),
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        context_window=128_000,
        max_output=16_384,
        encoding_name="o200k_base",
        provider="openai",
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.01,
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        context_window=16_385,
        max_output=4_096,
        encoding_name="cl100k_base",
        provider="openai",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        context_window=128_000,
        max_output=4_096,
        encoding_name="cl100k_base",
        provider="openai",
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
    ),
}

# Anthropic 模型（注意：Anthropic 不使用 tiktoken，这里用 cl100k_base 近似估算）
ANTHROPIC_CONFIGS = {
    "claude-3-5-sonnet": {"context_window": 200_000, "max_output": 8_192},
    "claude-3-5-haiku":  {"context_window": 200_000, "max_output": 8_192},
    "claude-3-opus":     {"context_window": 200_000, "max_output": 4_096},
}


# ============================================================
# Token 计数工具
# ============================================================

class TokenCounter:
    """基于 tiktoken 的精确 token 计数器。"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.config = MODEL_CONFIGS.get(model)
        if self.config is None:
            # fallback: 使用 cl100k_base（GPT-3.5/GPT-4 系列）
            logger.warning(
                f"模型 '{model}' 不在预定义列表中，使用 cl100k_base 估算"
            )
            self.config = ModelConfig(
                name=model,
                context_window=128_000,
                max_output=4_096,
                encoding_name="cl100k_base",
                provider="openai",
            )

        try:
            self.encoder = tiktoken.get_encoding(self.config.encoding_name)
        except Exception:
            logger.warning(
                f"无法加载编码器 '{self.config.encoding_name}'，"
                f"使用 cl100k_base 作为回退"
            )
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """计算单个文本的 token 数。"""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def count_messages(
        self,
        messages: list[dict[str, str]],
        *,
        include_overhead: bool = True,
    ) -> int:
        """计算消息列表的总 token 数。

        参考 OpenAI 官方计算方式：
        - 每条消息有固定开销（约 3-4 token）
        - role 名称不计入（但实际 API 中有少量开销）
        - 使用 tiktoken 精确编码

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            include_overhead: 是否包含每条消息的固定开销

        Returns:
            总 token 数
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                # 多模态消息（图片 + 文本）
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += self.count(part.get("text", ""))

            if include_overhead:
                # 每条消息约 4 token 的格式开销
                total += 4

        if include_overhead:
            # 整个请求的固定开销
            total += 2

        return total


# ============================================================
# Token 预算管理器
# ============================================================

@dataclass
class BudgetAllocation:
    """token 预算的分配方案。"""
    total_budget: int          # 总预算（上下文窗口）
    system_prompt_tokens: int = 0
    output_reservation: int = 0
    history_tokens: int = 0
    retrieval_tokens: int = 0
    available_for_user: int = 0


class TokenBudget:
    """Token 预算管理器。

    用法：
        budget = TokenBudget("gpt-4o-mini", output_reserve=4096)
        budget.set_system_prompt("你是一个有帮助的助手...")

        # 检查是否可以添加更多内容
        if budget.can_fit(extra_tokens=500):
            ...

        # 获取当前预算使用情况
        print(budget.report())
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        output_reserve: int | None = None,
        safety_margin: float = 0.05,  # 5% 安全余量
    ):
        """
        Args:
            model: 模型名称
            output_reserve: 输出预留 token 数。None 则使用模型默认 max_output
            safety_margin: 安全余量比例（避免边界情况）
        """
        self.model = model
        self.counter = TokenCounter(model)
        self.config = self.counter.config

        # 总预算（扣除安全余量）
        self.safety_margin = safety_margin
        self._total_budget = int(
            self.config.context_window * (1 - safety_margin)
        )

        # 输出预留
        self.output_reserve = (
            output_reserve
            if output_reserve is not None
            else min(self.config.max_output, 4096)
        )

        # 分配追踪
        self._system_prompt: str = ""
        self._system_tokens: int = 0
        self._history: list[dict[str, str]] = []
        self._history_tokens: int = 0
        self._retrieval_docs: list[str] = []
        self._retrieval_tokens: int = 0

    # ---- 属性 ----

    @property
    def total_budget(self) -> int:
        return self._total_budget

    @property
    def used(self) -> int:
        """已使用的 token 数。"""
        return (
            self._system_tokens
            + self._history_tokens
            + self._retrieval_tokens
            + self.output_reserve
        )

    @property
    def remaining(self) -> int:
        """剩余可用的 token 数。"""
        return max(0, self._total_budget - self.used)

    # ---- 设置内容 ----

    def set_system_prompt(self, prompt: str) -> "TokenBudget":
        """设置系统消息并计算其 token 占用。"""
        self._system_prompt = prompt
        self._system_tokens = self.counter.count(prompt)
        return self

    def set_history(self, messages: list[dict[str, str]]) -> "TokenBudget":
        """设置对话历史。如果超出预算，自动从最早的消息开始截断。"""
        self._history = messages
        self._history_tokens = self.counter.count_messages(messages)
        # 如果历史太长，截断
        self._history, self._history_tokens = self._truncate_history(
            messages, self._available_for_history()
        )
        return self

    def set_retrieval_docs(
        self, documents: list[str], *, max_tokens: int | None = None
    ) -> "TokenBudget":
        """设置检索到的文档列表。按顺序填充直到预算耗尽。

        文档按优先级从高到低排列（通常是检索分数的顺序）。
        """
        available = (
            min(max_tokens, self._available_for_retrieval())
            if max_tokens is not None
            else self._available_for_retrieval()
        )

        accepted = []
        used = 0
        for doc in documents:
            doc_tokens = self.counter.count(doc)
            if used + doc_tokens <= available:
                accepted.append(doc)
                used += doc_tokens
            else:
                # 尝试截断单个文档（只取能放下的部分）
                if available - used > 100:  # 至少留 100 token 才有意义
                    truncated = self._truncate_text(doc, available - used)
                    accepted.append(truncated + "\n[...文档被截断]")
                    used = available
                break

        self._retrieval_docs = accepted
        self._retrieval_tokens = used
        return self

    # ---- 检查和查询 ----

    def can_fit(self, *, extra_tokens: int = 0) -> bool:
        """检查是否还有空间容纳额外的 token。"""
        return self.remaining >= extra_tokens

    def report(self) -> str:
        """生成可读的预算使用报告。"""
        pct = self.used / self._total_budget * 100 if self._total_budget > 0 else 0
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        return (
            f"=== Token 预算报告: {self.model} ===\n"
            f"  使用率: [{bar}] {pct:.1f}%\n"
            f"  总预算:   {self._total_budget:>8,} tokens\n"
            f"  已使用:   {self.used:>8,} tokens\n"
            f"  ─────────────────────────────\n"
            f"  系统消息:  {self._system_tokens:>6,} tokens\n"
            f"  对话历史:  {self._history_tokens:>6,} tokens ({len(self._history)} 条)\n"
            f"  检索文档:  {self._retrieval_tokens:>6,} tokens ({len(self._retrieval_docs)} 篇)\n"
            f"  输出预留:  {self.output_reserve:>6,} tokens\n"
            f"  ─────────────────────────────\n"
            f"  剩余可用:  {self.remaining:>6,} tokens\n"
        )

    def build_messages(
        self, user_query: str
    ) -> list[dict[str, str]]:
        """构建发送给 LLM 的完整消息列表。

        组合：系统消息 + 检索文档 + 历史 + 用户查询
        """
        messages: list[dict[str, str]] = []

        # 系统消息
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # 检索文档作为上下文（拼接到系统消息后面或用户消息前面）
        if self._retrieval_docs:
            context = self._format_context(self._retrieval_docs)
            # 把检索上下文追加到第一条系统消息，或作为新的系统消息
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += "\n\n" + context
            else:
                messages.insert(0, {"role": "system", "content": context})

        # 历史对话
        messages.extend(self._history)

        # 用户当前查询
        messages.append({"role": "user", "content": user_query})

        # 最终安全检查
        total = self.counter.count_messages(messages)
        if total > self._total_budget:
            logger.error(
                f"构建的消息超出预算！总计 {total} tokens，"
                f"预算 {self._total_budget} tokens。"
            )
            # 紧急截断：从检索文档中移除内容
            excess = total - self._total_budget
            if self._retrieval_docs and messages:
                self._emergency_truncate_context(messages, excess)

        return messages

    # ---- 内部方法 ----

    def _available_for_history(self) -> int:
        """对话历史可用的 token 空间。"""
        # 给历史分配：剩余空间的 30%
        base = self._total_budget - self._system_tokens - self.output_reserve
        return max(0, int(base * 0.3))

    def _available_for_retrieval(self) -> int:
        """检索文档可用的 token 空间。"""
        base = (
            self._total_budget
            - self._system_tokens
            - self._history_tokens
            - self.output_reserve
        )
        return max(0, base)

    def _truncate_history(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[list[dict[str, str]], int]:
        """截断对话历史，保留最近的 N 条消息。"""
        if not messages:
            return [], 0

        kept = []
        total = 0

        # 从最新到最旧遍历（保留最近的）
        for msg in reversed(messages):
            msg_tokens = self.counter.count(msg.get("content", "")) + 4
            if total + msg_tokens <= max_tokens:
                kept.insert(0, msg)
                total += msg_tokens
            else:
                break

        return kept, total

    @staticmethod
    def _truncate_text(text: str, max_tokens: int) -> str:
        """按 token 数截断文本（粗略方法——按字符比例估算并保留前半部分）。"""
        # 粗略估算：英文约 4 字符/token，中文约 1.5 字符/token
        # 这里使用保守的策略
        if not text:
            return text

        # 先按字符粗略估算，然后逐步精确
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # 取前 max_tokens 个 token 解码回去
        truncated_tokens = tokens[:max_tokens]
        return encoder.decode(truncated_tokens)

    @staticmethod
    def _format_context(documents: list[str]) -> str:
        """将检索到的文档格式化为 LLM 可理解的上下文。"""
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(f"【参考资料 {i}】\n{doc}\n")
        return (
            "以下是从知识库中检索到的参考资料，用于回答用户问题：\n\n"
            + "\n".join(parts)
        )

    @staticmethod
    def _emergency_truncate_context(
        messages: list[dict[str, str]], excess_tokens: int
    ) -> None:
        """紧急情况：从消息列表中截断内容。

        注意：这是最后的手段，会修改传入的 messages 列表。
        """
        encoder = tiktoken.get_encoding("cl100k_base")
        for msg in messages:
            content = msg.get("content", "")
            if len(content) > excess_tokens * 3:  # 粗略字符估算
                tokens = encoder.encode(content)
                truncated = encoder.decode(tokens[:-excess_tokens])
                msg["content"] = truncated + "\n[...紧急截断]"
                logger.warning(f"紧急截断：移除了 {excess_tokens} tokens")
                break


# ============================================================
# 使用示例
# ============================================================

def demo():
    """演示 Token 预算管理器的完整使用流程。"""

    # 1. 创建预算管理器
    budget = TokenBudget("gpt-4o-mini", output_reserve=4096)
    print(f"模型上下文窗口: {budget.config.context_window:,} tokens")
    print(f"扣除 5% 安全余量后: {budget.total_budget:,} tokens\n")

    # 2. 设置系统提示
    system_prompt = (
        "你是一个专业的法律文件分析助手。"
        "请基于提供的参考资料，准确、简洁地回答用户的问题。"
        "如果参考资料中没有相关信息，请明确告知用户。"
    )
    budget.set_system_prompt(system_prompt)
    print(f"系统消息: {budget._system_tokens} tokens")

    # 3. 加载"检索"到的文档
    fake_docs = [
        "《民法典》第一千零一十条：违背他人意愿，以言语、文字、图像、"
        "肢体行为等方式对他人实施性骚扰的，受害人有权依法请求行为人承担民事责任。"
        "机关、企业、学校等单位应当采取合理的预防、受理投诉、调查处置等措施，"
        "防止和制止利用职权、从属关系等实施性骚扰。",

        "《民法典》第一千零一十一条：自然人享有姓名权，有权依法决定、使用、"
        "变更或者许可他人使用自己的姓名，但是不得违背公序良俗。",

        "《民法典》第一千零一十二条：自然人享有身体权。"
        "自然人的身体完整和行动自由受法律保护。"
        "任何组织或者个人不得侵害他人的身体权。",

        # 这是一段很长的文档，可能导致预算紧张
        "长文档内容..." * 500,
    ]

    budget.set_retrieval_docs(fake_docs, max_tokens=8000)
    print(f"检索文档: {len(budget._retrieval_docs)} 篇, "
          f"{budget._retrieval_tokens} tokens")

    # 4. 设置对话历史
    history = [
        {"role": "user", "content": "什么是性骚扰的法律定义？"},
        {"role": "assistant",
         "content": "根据《民法典》第一千零一十条，性骚扰是指违背他人意愿，"
                    "以言语、文字、图像、肢体行为等方式对他人实施的行为。"},
    ]
    budget.set_history(history)
    print(f"对话历史: {len(budget._history)} 条, "
          f"{budget._history_tokens} tokens")

    # 5. 打印预算报告
    print()
    print(budget.report())

    # 6. 检查是否可以添加用户问题
    user_query = "请问身体权包括哪些内容？"
    query_tokens = budget.counter.count(user_query)
    if budget.can_fit(extra_tokens=query_tokens):
        messages = budget.build_messages(user_query)
        final_count = budget.counter.count_messages(messages)
        print(f"✅ 可以发送！最终消息共 {final_count} tokens")
        print(f"   消息数: {len(messages)}")
    else:
        print(f"❌ 超出预算！剩余 {budget.remaining} tokens，"
              f"需要 {query_tokens} tokens")


if __name__ == "__main__":
    demo()
```

### 快速 Token 计数（不依赖预算管理器）

```python
"""最简 token 计数工具函数。"""

import tiktoken


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """快速计算文本的 token 数。

    Args:
        text: 要计算的文本
        model: 模型名称（决定使用哪个编码器）

    Returns:
        token 数量
    """
    # 模型 → 编码器映射
    encoding_map = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
    }
    encoding_name = encoding_map.get(model, "cl100k_base")
    encoder = tiktoken.get_encoding(encoding_name)
    return len(encoder.encode(text))


def count_messages(
    messages: list[dict[str, str]], model: str = "gpt-4o-mini"
) -> int:
    """计算消息列表的 token 数（含格式开销）。"""
    encoder_name = "o200k_base" if "gpt-4o" in model else "cl100k_base"
    encoder = tiktoken.get_encoding(encoder_name)

    total = 0
    for msg in messages:
        total += len(encoder.encode(msg.get("content", "")))
        total += 4  # 每条消息的格式开销

    total += 2  # 整个请求的格式开销
    return total


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """估算 API 调用的费用（美元）。

    Args:
        model: 模型名称
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数

    Returns:
        估算费用（美元）
    """
    pricing = {
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4o": (0.0025, 0.01),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0005, 0.0015),
    }

    input_price, output_price = pricing.get(
        model, (0.0, 0.0)
    )
    cost = (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
    return round(cost, 6)


# 演示
if __name__ == "__main__":
    text = "Hello, world! 你好，世界！" * 100
    tokens = count_tokens(text, "gpt-4o-mini")
    print(f"文本长度: {len(text)} 字符 → {tokens} tokens")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is RAG?"},
    ]
    msg_tokens = count_messages(messages)
    print(f"消息 token 数: {msg_tokens}")

    cost = estimate_cost("gpt-4o-mini", input_tokens=5000, output_tokens=1000)
    print(f"估算费用: ${cost:.6f} (5000 input + 1000 output)")
```

---

## 检查清单

在每次调用 LLM API 之前，确认以下 3 项：

- [ ] **已知晓所用模型的总上下文窗口**: 不是凭记忆，是实际查了文档（建议把上方的速查表贴在显示器旁边）
- [ ] **已计算输入内容的 token 数**: 不要用"大概 2000 字"来估算——中文和英文的 token 密度完全不同（1 个中文字约 1.5-2 token，1 个英文单词约 1-1.3 token）
- [ ] **已设置 `max_tokens` 参数**: 不要依赖默认值，显式设置输出上限，确保总输入 + 输出 < 上下文窗口

如果一项都没满足，你很可能在下一次长文档处理时撞到 `context_length_exceeded`。

---

## 延伸阅读

- [OpenAI: Managing Tokens](https://platform.openai.com/docs/guides/tokens)
- [OpenAI: Tokenizer Tool](https://platform.openai.com/tokenizer)
- [tiktoken 官方仓库](https://github.com/openai/tiktoken)
- [Anthropic: Context Windows](https://docs.anthropic.com/en/docs/build-with-claude/context-windows)

---

**一句话总结**：模型不会告诉你"内容放不下了"——它只会默默截断，然后给你一个残缺的答案。Token 预算不是优化，是必需品。
