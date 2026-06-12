# Token 成本爆炸问题与优化策略

## 一、问题本质

在大规模RAG系统中，Token消耗往往是最大的可变成本。一个看似无害的优化（如"把更多上下文塞进prompt"）可能导致成本呈指数级增长。

### 经典爆发现场

```
初始设计:
  用户query (100 tokens) + 检索3篇文档 (1500 tokens) = 1600 tokens/请求
  日请求量: 10,000 -> 日消耗: 16M tokens

"优化"后:
  "多检索几篇文档，结果更准确！"
  用户query (100) + 检索20篇文档 (15,000) + chain-of-thought (500) = 15,600 tokens/请求
  日请求量: 10,000 -> 日消耗: 156M tokens

成本爆炸: 10倍！！！
```

| 阶段 | 每请求Token | 日请求量 | 日Token消耗 | 月成本(Claude 3.5 Sonnet) |
|------|------------|---------|------------|--------------------------|
| 初始 | 1,600 | 10,000 | 16M | ~$240 |
| "优化" | 15,600 | 10,000 | 156M | ~$2,340 |
| 爆炸 | 50,000 | 10,000 | 500M | ~$7,500 |

---

## 二、症状与影响

| 症状 | 表现 | 根因 |
|------|------|------|
| **账单飙升** | 月成本从$500涨到$5,000 | prompt膨胀+请求量增长 |
| **延迟增加** | P99延迟从2s涨到8s | 处理更多token需要更多时间 |
| **无节制检索** | 每次都检索最大数量的文档 | 缺乏动态检索策略 |
| **重复计算** | 相同前缀每次请求都重新处理 | 未使用Prompt Caching |
| **模型选择不当** | 简单问题用GPT-4o | 缺乏模型路由器 |

---

## 三、策略1：Prompt Caching（提示缓存）

### 原理

对于RAG系统，System Prompt、Few-shot示例、检索文档的"外壳"（非文档内容部分）在每次请求中是相同的。Prompt Caching可以将这些重复内容的KV-Cache保留在服务器端，后续请求只需处理增量部分。

```
请求1: [System Prompt (5000 tokens)] [User Query (100 tokens)]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        缓存这部分，写入KV-Cache（写入成本：正常价格的1.25x）

请求2: [System Prompt (5000 tokens)] [User Query (150 tokens)]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        命中缓存！只计费 User Query 部分（读取成本：正常价格的0.1x）

节省: 90%的System Prompt token成本
```

### 实现代码

```python
"""
Prompt Caching Strategy -- 利用Anthropic/OpenAI的Prompt Caching特性

Anthropic Claude: 自动缓存 >1024 token的前缀，cache_write 1.25x, cache_read 0.1x
OpenAI: 自动缓存 >1024 token的前缀，cache_write 全价, cache_read 0.5x

策略：
1. 将System Prompt放在最前面（static prefix）
2. 将Few-shot示例紧随其后
3. 将Tool Definitions放在一起
4. 用户查询放在最后（cache miss部分）
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time


@dataclass
class CacheMetrics:
    """缓存性能指标"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_saved: int = 0
    cost_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def record_hit(self, tokens: int, cost_saved: float):
        self.total_requests += 1
        self.cache_hits += 1
        self.tokens_saved += tokens
        self.cost_saved += cost_saved

    def record_miss(self):
        self.total_requests += 1
        self.cache_misses += 1


class PromptCacheOptimizer:
    """
    Prompt缓存优化器。

    负责组织prompt结构以最大化缓存命中率。
    """

    def __init__(
        self,
        cache_breakpoint_threshold: int = 1024,  # 缓存的断点阈值（token）
    ):
        self.cache_breakpoint_threshold = cache_breakpoint_threshold
        self.metrics = CacheMetrics()

    def build_cache_friendly_prompt(
        self,
        system_prompt: str,
        user_query: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        tool_definitions: Optional[List[Dict]] = None,
        retrieved_docs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        构建缓存友好的prompt结构。

        顺序原则：最静态的在前，最动态的在后
        1. System Prompt       <- 完全静态，100%缓存命中
        2. Tool Definitions    <- 半静态（很少变）
        3. Few-shot Examples   <- 半静态（偶尔更新）
        4. Retrieved Docs      <- 半动态（每次不同但可以在请求间部分共享）
        5. User Query          <- 完全动态，每次不同

        Returns:
            包含messages列表和缓存元信息的字典
        """
        messages = []
        cache_boundaries = []  # 记录缓存断点位置

        # 1. System Prompt（最高缓存优先级）
        system_block = [{"role": "system", "content": system_prompt}]
        messages.extend(system_block)
        cache_boundaries.append(("system_prompt", len(system_prompt)))

        # 2. Tool Definitions（并发请求间共享）
        if tool_definitions:
            # 在支持cache_control的API中标记断点
            tool_block = [{
                "role": "system",
                "content": f"[TOOLS]\n{tool_definitions}",
            }]
            messages.extend(tool_block)

        # 3. Few-shot Examples（标为可缓存）
        if few_shot_examples:
            for example in few_shot_examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

        # 4. Retrieved Documents（按固定模板组织以提高缓存命中率）
        if retrieved_docs:
            # 文档外壳是静态的，内容是动态的
            # 更好的策略：文档编号+title模式
            docs_text = self._format_docs_for_caching(retrieved_docs)
            messages.append({"role": "user", "content": docs_text})

        # 5. User Query（动态部分，放在最后）
        messages.append({"role": "user", "content": user_query})

        # 估算缓存命中情况
        cacheable_tokens = sum(
            len(content) // 4  # 粗略token估算
            for msg in messages[:-1]  # 除了最后一条（user query）都是可缓存的
            for content in [msg.get("content", "")]
        )

        return {
            "messages": messages,
            "cache_boundaries": cache_boundaries,
            "estimated_cacheable_tokens": cacheable_tokens,
            "estimated_dynamic_tokens": len(user_query) // 4,
        }

    def _format_docs_for_caching(self, docs: List[str]) -> str:
        """以缓存友好的格式组织文档"""
        formatted = ["[RETRIEVED DOCUMENTS]\n"]
        for i, doc in enumerate(docs, 1):
            formatted.append(f"--- Document {i} ---")
            formatted.append(doc)
            formatted.append("")
        return "\n".join(formatted)

    def calculate_savings(
        self,
        input_tokens: int,
        cache_hit_tokens: int,
        model_pricing: Dict[str, float],
    ) -> Dict[str, float]:
        """
        计算缓存带来的成本节省。

        Args:
            input_tokens: 总输入token数
            cache_hit_tokens: 命中缓存的token数
            model_pricing: {"input_per_1k": 3.0, "cache_write_per_1k": 3.75, "cache_read_per_1k": 0.3}

        Returns:
            {"without_cache": cost, "with_cache": cost, "savings": amount, "savings_pct": percent}
        """
        # 无缓存的成本
        no_cache_cost = (input_tokens / 1000) * model_pricing["input_per_1k"]

        # 有缓存的成本
        cache_miss_tokens = input_tokens - cache_hit_tokens
        write_cost = (cache_hit_tokens / 1000) * model_pricing.get("cache_write_per_1k", model_pricing["input_per_1k"])
        read_cost = (cache_hit_tokens / 1000) * model_pricing.get("cache_read_per_1k", model_pricing["input_per_1k"] * 0.1)
        miss_cost = (cache_miss_tokens / 1000) * model_pricing["input_per_1k"]

        with_cache_cost = write_cost + read_cost + miss_cost

        savings = no_cache_cost - with_cache_cost
        savings_pct = (savings / no_cache_cost * 100) if no_cache_cost > 0 else 0.0

        return {
            "without_cache": round(no_cache_cost, 4),
            "with_cache": round(with_cache_cost, 4),
            "savings": round(savings, 4),
            "savings_pct": round(savings_pct, 2),
        }


# ============================================================
# 策略2：Model Router（模型路由器）
# ============================================================

class ModelTier(Enum):
    """模型层级"""
    LIGHT = "light"     # 轻量级：简单分类、关键词提取
    STANDARD = "standard"  # 标准级：一般RAG问答
    PREMIUM = "premium"    # 高级：复杂推理、多步链式


@dataclass
class RouterDecision:
    """路由决策"""
    model: str
    tier: ModelTier
    reason: str
    estimated_cost: float


class ModelRouter:
    """
    智能模型路由器。

    根据任务复杂度自动选择合适的模型层级，避免用大炮打蚊子。
    """

    # 模型定价（美元/1M tokens）
    PRICING = {
        "claude-haiku":    {"input": 0.25, "output": 1.25},
        "claude-sonnet":   {"input": 3.0,  "output": 15.0},
        "gpt-4o-mini":     {"input": 0.15, "output": 0.60},
        "gpt-4o":          {"input": 2.5,  "output": 10.0},
    }

    # 路由规则
    RULES = [
        # (条件函数, 目标模型, 层级)
        (lambda q, d: len(d) == 0, "claude-haiku", ModelTier.LIGHT),
        (lambda q, d: len(q.split()) < 5, "claude-haiku", ModelTier.LIGHT),
        (lambda q, d: any(kw in q.lower() for kw in [
            "你好", "谢谢", "再见", "什么", "谁"
        ]), "gpt-4o-mini", ModelTier.LIGHT),
        (lambda q, d: len(d) > 0 and len(d) < 3, "claude-sonnet", ModelTier.STANDARD),
        (lambda q, d: any(kw in q.lower() for kw in [
            "分析", "比较", "推理", "总结", "评估"
        ]) and len(d) > 2, "claude-sonnet", ModelTier.PREMIUM),
        (lambda q, d: any(kw in q.lower() for kw in [
            "代码", "编程", "debug", "算法"
        ]), "gpt-4o", ModelTier.PREMIUM),
    ]

    def __init__(self, default_model: str = "claude-sonnet"):
        self.default_model = default_model
        self.stats: Dict[str, int] = {}

    def route(
        self,
        query: str,
        retrieved_docs: Optional[List[str]] = None,
    ) -> RouterDecision:
        """
        根据查询和检索文档决定使用哪个模型。

        Args:
            query: 用户查询
            retrieved_docs: 检索到的文档列表

        Returns:
            RouterDecision 包含选中的模型和理由
        """
        docs = retrieved_docs or []

        for condition, model, tier in self.RULES:
            if condition(query, docs):
                self.stats[model] = self.stats.get(model, 0) + 1
                return RouterDecision(
                    model=model,
                    tier=tier,
                    reason=f"Matched routing rule: {tier.value}",
                    estimated_cost=self._estimate_cost(model, query, docs),
                )

        # 默认
        self.stats[self.default_model] = self.stats.get(self.default_model, 0) + 1
        return RouterDecision(
            model=self.default_model,
            tier=ModelTier.STANDARD,
            reason="Default routing (no rule matched)",
            estimated_cost=self._estimate_cost(self.default_model, query, docs),
        )

    def _estimate_cost(
        self, model: str, query: str, docs: List[str]
    ) -> float:
        """估算此次请求的成本"""
        pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
        input_tokens = (len(query) + sum(len(d) for d in docs)) // 4
        output_tokens = 200  # 假设平均输出200 tokens
        cost = (input_tokens / 1_000_000) * pricing["input"] + \
               (output_tokens / 1_000_000) * pricing["output"]
        return round(cost, 6)

    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        total = sum(self.stats.values())
        return {
            "total_requests": total,
            "distribution": {
                model: {"count": count, "pct": round(count/total*100, 1)}
                for model, count in self.stats.items()
            },
        }


# ============================================================
# 策略3：TokenBudget Controller
# ============================================================

class TokenBudget:
    """
    Token预算控制器。

    功能：
    1. 设置每个请求的token上限
    2. 动态计算分配给各部分（system, docs, history, output）的配额
    3. 超过配额时自动截断
    """

    def __init__(
        self,
        max_input_tokens: int = 100000,
        max_output_tokens: int = 4096,
        system_reserve: int = 2000,
        history_reserve: int = 3000,
        output_reserve: int = 2000,
    ):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.system_reserve = system_reserve
        self.history_reserve = history_reserve
        self.output_reserve = output_reserve

    def allocate(
        self,
        system_prompt: str,
        query: str,
        history: Optional[List[str]] = None,
        num_docs: int = 0,
    ) -> Dict[str, Any]:
        """
        为各部分分配token配额。

        Returns:
            {
                "system_tokens": int,
                "history_tokens": int,
                "query_tokens": int,
                "docs_budget": int,      <- 可分配给文档的token数
                "output_tokens": int,
                "is_over_budget": bool,
                "warnings": List[str],
            }
        """
        warnings = []
        system_tokens = self._count_tokens(system_prompt)
        query_tokens = self._count_tokens(query)

        # 分配
        alloc_system = min(system_tokens, self.system_reserve)
        if system_tokens > self.system_reserve:
            warnings.append(f"System prompt ({system_tokens}t) exceeds reserve ({self.system_reserve}t)")

        alloc_history = self._count_tokens("\n".join(history or []))
        alloc_history = min(alloc_history, self.history_reserve)

        alloc_query = query_tokens

        # 文档预算 = 总预算 - 已分配
        used = alloc_system + alloc_history + alloc_query + self.output_reserve
        docs_budget = max(0, self.max_input_tokens - used)
        is_over_budget = used > self.max_input_tokens

        if is_over_budget:
            warnings.append(f"Over budget: used {used}t > max {self.max_input_tokens}t")

        return {
            "system_tokens": alloc_system,
            "history_tokens": alloc_history,
            "query_tokens": alloc_query,
            "docs_budget": docs_budget,
            "output_tokens": self.output_reserve,
            "is_over_budget": is_over_budget,
            "warnings": warnings,
        }

    def truncate_docs_to_budget(
        self, documents: List[str], budget_tokens: int
    ) -> List[str]:
        """
        按token预算截断文档列表。

        策略：
        1. 逐文档添加，直到超出预算
        2. 最后一个文档截断以填充剩余空间
        3. 保证至少包含一篇完整文档
        """
        if not documents:
            return []

        kept = []
        used = 0

        for doc in documents:
            doc_tokens = self._count_tokens(doc)
            if used + doc_tokens <= budget_tokens:
                kept.append(doc)
                used += doc_tokens
            else:
                remaining = budget_tokens - used
                if remaining > 50 and not kept:
                    # 第一篇文档就必须截断
                    truncated = self._truncate_text(doc, remaining)
                    kept.append(truncated)
                elif remaining > 100 and kept:
                    truncated = self._truncate_text(doc, remaining)
                    kept.append(truncated)
                break

        return kept

    @staticmethod
    def _count_tokens(text: str) -> int:
        """粗略token计数（中文按字符，英文按4字符/token）"""
        if not text:
            return 0
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return chinese_chars + (other_chars // 4)

    @staticmethod
    def _truncate_text(text: str, max_tokens: int) -> str:
        """按token数截断文本"""
        max_chars = max_tokens * 3  # 粗略转换
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...[truncated]"


# ============================================================
# 综合示例：成本感知的RAG流水线
# ============================================================

class CostAwareRAGPipeline:
    """
    成本感知的RAG流水线。

    集成三个成本优化策略：
    1. Prompt Caching - 减少重复token费用
    2. Model Router - 为简单任务选择便宜模型
    3. TokenBudget - 控制每个请求的token消耗
    """

    def __init__(
        self,
        max_input_tokens: int = 100000,
        default_model: str = "claude-sonnet",
    ):
        self.cache_optimizer = PromptCacheOptimizer()
        self.router = ModelRouter(default_model=default_model)
        self.budget = TokenBudget(max_input_tokens=max_input_tokens)

        self.cost_records: List[Dict[str, Any]] = []

    def process(
        self,
        query: str,
        system_prompt: str,
        retrieved_docs: List[str],
        history: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        处理一次RAG请求，全程成本感知。

        Returns:
            包含响应、实际模型、成本分解的完整结果
        """
        # Step 1: Token预算分配
        allocation = self.budget.allocate(
            system_prompt, query, history, len(retrieved_docs)
        )

        # Step 2: 按预算截断文档
        docs_to_use = self.budget.truncate_docs_to_budget(
            retrieved_docs, allocation["docs_budget"]
        )

        # Step 3: 模型路由
        decision = self.router.route(query, docs_to_use)

        # Step 4: 构建缓存友好prompt
        prompt = self.cache_optimizer.build_cache_friendly_prompt(
            system_prompt=system_prompt,
            user_query=query,
            retrieved_docs=docs_to_use,
        )

        # Step 5: 记录成本
        cost_record = {
            "query": query[:80],
            "model": decision.model,
            "tier": decision.tier.value,
            "docs_count": len(docs_to_use),
            "docs_tokens": sum(self.budget._count_tokens(d) for d in docs_to_use),
            "budget_warnings": allocation["warnings"],
            "estimated_cost": decision.estimated_cost,
            "cacheable_tokens": prompt["estimated_cacheable_tokens"],
        }
        self.cost_records.append(cost_record)

        return {
            "prompt": prompt,
            "allocation": allocation,
            "router_decision": decision,
            "cost_record": cost_record,
        }

    def get_total_cost(self) -> float:
        """获取累计成本"""
        return round(sum(r["estimated_cost"] for r in self.cost_records), 4)

    def get_cost_summary(self) -> Dict[str, Any]:
        """获取成本摘要"""
        if not self.cost_records:
            return {"total_requests": 0, "total_cost": 0}

        model_costs = {}
        for r in self.cost_records:
            m = r["model"]
            model_costs[m] = model_costs.get(m, 0.0) + r["estimated_cost"]

        return {
            "total_requests": len(self.cost_records),
            "total_cost": self.get_total_cost(),
            "avg_cost_per_request": round(
                self.get_total_cost() / len(self.cost_records), 6
            ),
            "cost_by_model": model_costs,
            "router_distribution": self.router.get_routing_stats(),
        }


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    pipeline = CostAwareRAGPipeline(max_input_tokens=50000)

    system = "你是一个技术文档助手。请用中文回答用户问题。"
    docs = [f"文档{i}的内容" * 50 for i in range(10)]

    # 简单查询 -> 便宜模型
    r1 = pipeline.process("你好", system, docs[:2])
    print(f"简单查询 -> {r1['router_decision'].model} (${r1['router_decision'].estimated_cost})")

    # 复杂查询 -> 高级模型
    r2 = pipeline.process("请分析并比较以下技术方案的优缺点", system, docs)
    print(f"复杂查询 -> {r2['router_decision'].model} (${r2['router_decision'].estimated_cost})")

    print(f"\n总成本: ${pipeline.get_total_cost()}")
    print(pipeline.get_cost_summary())
```

---

## 四、策略对比与组合

```
                  ┌─────────────────────────────────┐
                  │   用户请求到达                    │
                  └────────────┬────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────────────┐
                  │ TokenBudget.allocate()          │
                  │ 计算各部分配额                   │
                  └────────────┬────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────────────┐
                  │ ModelRouter.route()             │
                  │ 选择合适的模型层级               │
                  └────────────┬────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────────────┐
                  │ PromptCacheOptimizer            │
                  │ 构建缓存友好的prompt结构          │
                  └────────────┬────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────────────┐
                  │ 调用LLM API                      │
                  └─────────────────────────────────┘
```

| 策略 | 节省比例 | 适用场景 | 实施难度 |
|------|---------|---------|---------|
| **Prompt Caching** | 50-90% | 有大量静态前缀的场景 | 低（API原生支持） |
| **Model Router** | 30-70% | 请求复杂度差异大的场景 | 中（需要规则/分类器） |
| **TokenBudget** | 10-50% | 文档数量不固定的场景 | 低（简单截断逻辑） |

---

## 五、关键设计决策

1. **缓存收益递减** -- Prompt Caching的收益取决于静态部分占比。如果每次请求的system prompt只有100 tokens而检索文档有10000 tokens，缓存收益有限。

2. **模型路由的"漏判成本"** -- 如果简单查询被错误地路由到了小模型，导致回答问题质量差需要重试，那么总成本可能更高。

3. **预算截断的信息损失** -- TokenBudget截断文档时，重要的信息可能被截掉。应结合LongContextReorder确保最重要的文档优先保留。

4. **监控成本趋势** -- 建议在生产环境中记录每次请求的实际token消耗和成本，建立成本看板，设置预算警报。

---

*上一篇：[03 位置偏差 Lost in the Middle](03-position-bias.md)*
