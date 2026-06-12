#!/usr/bin/env python3
"""
Prompt缓存实现 (Anthropic Prompt Caching)
Prompt Caching Implementation with Cost Analysis

特性:
  - Anthropic prompt caching with cache_control: {"type": "ephemeral"}
  - 缓存断点 (Cache Breakpoint) 策略放置
  - 成本计算 (90% 节省于缓存token)
  - TTL 5分钟，使用后刷新
  - 对比表: Anthropic vs OpenAI vs DeepSeek
"""

import time
import hashlib
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class CacheStrategy(Enum):
    """缓存策略"""
    FULL_PROMPT = "full_prompt"           # 缓存整个系统提示
    PREFIX = "prefix"                    # 缓存前缀（系统提示+固定前缀）
    SEGMENT = "segment"                  # 缓存分段
    DYNAMIC_TAIL = "dynamic_tail"       # 前缀缓存 + 动态尾部


@dataclass
class CacheBreakpoint:
    """缓存断点定义"""
    position: int          # 在文本中的字节/字符位置
    label: str             # 断点标签
    token_count: int       # 该段落的token数
    expected_hit_rate: float  # 预期命中率

    def to_anthropic(self) -> dict:
        """转换为Anthropic API格式"""
        return {
            "type": "ephemeral",
        }


@dataclass
class CachedPrompt:
    """缓存提示词"""
    content: str
    breakpoints: list[CacheBreakpoint]
    strategy: CacheStrategy
    cache_id: Optional[str] = None  # 缓存ID（服务端返回）
    cached_at: float = 0.0
    expires_at: float = 0.0
    access_count: int = 0
    token_count: int = 0

    def is_expired(self, ttl: int = 300) -> bool:
        """检查是否过期（默认5分钟）"""
        return time.time() > self.expires_at

    def refresh(self, ttl: int = 300):
        """刷新过期时间（每次使用后刷新）"""
        self.expires_at = time.time() + ttl
        self.access_count += 1


@dataclass
class CostBreakdown:
    """成本分解"""
    total_tokens: int
    cached_tokens: int
    uncached_tokens: int
    output_tokens: int
    cached_input_cost: float
    uncached_input_cost: float
    output_cost: float
    total_cost: float
    savings_pct: float

    def summary(self) -> str:
        return (
            f"总Token: {self.total_tokens}, "
            f"缓存命中: {self.cached_tokens} ({self.savings_pct:.0f}% 节省), "
            f"总费用: ${self.total_cost:.6f}"
        )


# ============================================================
# 断点策略引擎 (Breakpoint Strategy Engine)
# ============================================================

class BreakpointPlacer:
    """缓存断点放置策略"""

    @staticmethod
    def for_strategy(strategy: CacheStrategy, content: str,
                     system_prompt: str = "") -> list[CacheBreakpoint]:
        """根据策略生成断点位置"""
        if strategy == CacheStrategy.FULL_PROMPT:
            return BreakpointPlacer._full_prompt(content)
        elif strategy == CacheStrategy.PREFIX:
            return BreakpointPlacer._prefix_only(content, system_prompt)
        elif strategy == CacheStrategy.SEGMENT:
            return BreakpointPlacer._segmented(content)
        elif strategy == CacheStrategy.DYNAMIC_TAIL:
            return BreakpointPlacer._dynamic_tail(content)
        return []

    @staticmethod
    def _full_prompt(content: str) -> list[CacheBreakpoint]:
        """整个提示作为一个缓存块"""
        tokens = BreakpointPlacer._estimate_tokens(content)
        return [
            CacheBreakpoint(
                position=0,
                label="full_prompt",
                token_count=tokens,
                expected_hit_rate=0.9,
            )
        ]

    @staticmethod
    def _prefix_only(content: str, system_prompt: str) -> list[CacheBreakpoint]:
        """仅缓存系统提示前缀"""
        breakpoints = []
        pos = 0

        if system_prompt:
            tokens = BreakpointPlacer._estimate_tokens(system_prompt)
            breakpoints.append(
                CacheBreakpoint(
                    position=pos,
                    label="system_prompt",
                    token_count=tokens,
                    expected_hit_rate=0.95,  # 系统提示变化少
                )
            )
            pos += len(system_prompt)

        return breakpoints

    @staticmethod
    def _segmented(content: str) -> list[CacheBreakpoint]:
        """分段缓存: 系统提示、上下文、用户问题分别缓存"""
        import re

        breakpoints = []
        segments = re.split(r"\n(?=System:|User:|Assistant:)", content)

        pos = 0
        for i, segment in enumerate(segments):
            tokens = BreakpointPlacer._estimate_tokens(segment)
            if i == 0:
                # 系统提示几乎不变
                hit_rate = 0.95
            elif "User:" in segment and i == len(segments) - 1:
                # 最后的用户问题变化最多
                hit_rate = 0.2
            else:
                hit_rate = 0.6

            breakpoints.append(
                CacheBreakpoint(
                    position=pos,
                    label=f"segment_{i}",
                    token_count=tokens,
                    expected_hit_rate=hit_rate,
                )
            )
            pos += len(segment)

        return breakpoints

    @staticmethod
    def _dynamic_tail(content: str) -> list[CacheBreakpoint]:
        """前缀缓存 + 动态尾部（最适合RAG）
        缓存: 系统提示 + 检索到的上下文
        不缓存: 用户问题的动态部分
        """
        breakpoints = []

        # 尝试在最后一个"User:"之前放置缓存断点
        import re
        user_matches = list(re.finditer(r"\nUser:", content))

        if user_matches:
            last_user = user_matches[-1]
            prefix = content[: last_user.start()]

            prefix_tokens = BreakpointPlacer._estimate_tokens(prefix)
            breakpoints.append(
                CacheBreakpoint(
                    position=0,
                    label="prefix_before_last_user",
                    token_count=prefix_tokens,
                    expected_hit_rate=0.7,
                )
            )

        return breakpoints if breakpoints else BreakpointPlacer._full_prompt(content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算token数（近似: 英文1token≈4字符, 中文1token≈1.5字符）"""
        import re
        chinese_chars = len(re.findall(r"[一-鿿]", text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


# ============================================================
# Anthropic Prompt 缓存实现
# ============================================================

class AnthropicPromptCache:
    """Anthropic Prompt缓存管理器

    API格式:
    {
        "model": "claude-sonnet-4-20250514",
        "system": [
            {
                "type": "text",
                "text": "You are an AI assistant...",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": [...]
    }
    """

    # Anthropic 价格 (per 1M tokens)
    PRICING = {
        "claude-sonnet-4-20250514": {
            "input_cached": 0.30,    # $0.30/MTok (缓存写入)
            "input_uncached": 3.00,  # $3.00/MTok (未缓存输入)
            "output": 15.00,         # $15.00/MTok (输出)
        },
        "claude-opus-4-20250514": {
            "input_cached": 1.50,
            "input_uncached": 15.00,
            "output": 75.00,
        },
    }

    CACHE_TTL = 300  # 5分钟

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._cache_store: dict[str, CachedPrompt] = {}
        self._lock = threading.RLock()
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0,
        }

    # ========== 断点插入 ==========

    def apply_cache_control(self, content: str, breakpoints: list[CacheBreakpoint]) -> list[dict]:
        """将缓存断点应用到文本内容，生成Anthropic API格式的content块

        Args:
            content: 完整文本
            breakpoints: 断点列表

        Returns:
            Anthropic API格式的内容块列表
        """
        if not breakpoints:
            return [{"type": "text", "text": content}]

        blocks = []
        sorted_bp = sorted(breakpoints, key=lambda b: b.position)

        for i, bp in enumerate(sorted_bp):
            start = bp.position
            end = sorted_bp[i + 1].position if i + 1 < len(sorted_bp) else len(content)
            segment_text = content[start:end]

            block = {
                "type": "text",
                "text": segment_text,
            }

            # 对高命中率预期的段添加缓存控制
            if bp.expected_hit_rate > 0.5:
                block["cache_control"] = bp.to_anthropic()

            blocks.append(block)

        return blocks

    def build_cached_request(self, system_prompt: str, messages: list[dict],
                             strategy: CacheStrategy = CacheStrategy.DYNAMIC_TAIL) -> dict:
        """构建带缓存控制的Anthropic API请求

        Args:
            system_prompt: 系统提示
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
            strategy: 缓存策略

        Returns:
            Anthropic API格式的请求体
        """
        # 对系统提示放置缓存断点
        system_breakpoints = BreakpointPlacer.for_strategy(
            CacheStrategy.PREFIX, system_prompt, system_prompt
        )
        system_blocks = self.apply_cache_control(system_prompt, system_breakpoints)

        # 对消息内容放置缓存断点
        combined_user_content = "\n\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        msg_breakpoints = BreakpointPlacer.for_strategy(strategy, combined_user_content)
        msg_blocks = self.apply_cache_control(combined_user_content, msg_breakpoints)

        return {
            "model": self.model,
            "system": system_blocks,
            "messages": [
                {"role": "user", "content": msg_blocks}
            ],
        }

    # ========== 成本计算 ==========

    def calculate_cost(self, input_tokens: int, cached_input_tokens: int,
                       output_tokens: int) -> CostBreakdown:
        """计算使用缓存的成本

        缓存token成本仅为未缓存token的10% (90%节省)
        """
        pricing = self.PRICING.get(self.model, self.PRICING["claude-sonnet-4-20250514"])

        uncached_input = input_tokens - cached_input_tokens

        cached_cost = (cached_input_tokens / 1_000_000) * pricing["input_cached"]
        uncached_cost = (uncached_input / 1_000_000) * pricing["input_uncached"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        total = cached_cost + uncached_cost + output_cost

        # 计算节省（与不使用缓存对比）
        no_cache_cost = (input_tokens / 1_000_000) * pricing["input_uncached"] + output_cost
        savings = no_cache_cost - total
        savings_pct = (savings / no_cache_cost * 100) if no_cache_cost > 0 else 0

        return CostBreakdown(
            total_tokens=input_tokens + output_tokens,
            cached_tokens=cached_input_tokens,
            uncached_tokens=uncached_input,
            output_tokens=output_tokens,
            cached_input_cost=cached_cost,
            uncached_input_cost=uncached_cost,
            output_cost=output_cost,
            total_cost=total,
            savings_pct=savings_pct,
        )

    # ========== 缓存管理 ==========

    def cache_prompt(self, content: str, cache_id: str,
                     strategy: CacheStrategy = CacheStrategy.DYNAMIC_TAIL) -> CachedPrompt:
        """缓存一个提示"""
        breakpoints = BreakpointPlacer.for_strategy(strategy, content)
        token_count = BreakpointPlacer._estimate_tokens(content)

        cached = CachedPrompt(
            content=content,
            breakpoints=breakpoints,
            strategy=strategy,
            cache_id=cache_id,
            cached_at=time.time(),
            expires_at=time.time() + self.CACHE_TTL,
            token_count=token_count,
        )

        with self._lock:
            self._cache_store[cache_id] = cached

        return cached

    def get_cached(self, cache_id: str) -> Optional[CachedPrompt]:
        """获取缓存的提示（如果未过期）"""
        with self._lock:
            self._stats["total_requests"] += 1
            cached = self._cache_store.get(cache_id)

            if cached and not cached.is_expired(self.CACHE_TTL):
                cached.refresh(self.CACHE_TTL)
                self._stats["cache_hits"] += 1
                self._stats["total_tokens_saved"] += cached.token_count
                return cached

            if cached:
                # 已过期，移除
                del self._cache_store[cache_id]

            self._stats["cache_misses"] += 1
            return None

    def invalidate(self, cache_id: str) -> bool:
        """手动失效缓存"""
        with self._lock:
            if cache_id in self._cache_store:
                del self._cache_store[cache_id]
                return True
        return False

    def clear_expired(self) -> int:
        """清理过期缓存"""
        count = 0
        now = time.time()
        with self._lock:
            expired = [
                cid for cid, cp in self._cache_store.items()
                if cp.is_expired(self.CACHE_TTL)
            ]
            for cid in expired:
                del self._cache_store[cid]
                count += 1
        return count

    # ========== 统计 ==========

    def get_stats(self) -> dict:
        """获取缓存统计"""
        with self._lock:
            cache_hit_rate = (
                self._stats["cache_hits"] / max(self._stats["total_requests"], 1)
            )
            active_caches = len(self._cache_store)
        return {
            **self._stats,
            "cache_hit_rate": f"{cache_hit_rate:.1%}",
            "active_caches": active_caches,
            "model": self.model,
        }


# ============================================================
# 通用Prompt缓存门面 (Universal Prompt Cache Facade)
# ============================================================

class PromptCacheFacade:
    """Prompt缓存门面 - 支持多提供商"""

    def __init__(self, provider: str = "anthropic", model: str = ""):
        self.provider = provider.lower()
        self.model = model

        if self.provider == "anthropic":
            self._cache = AnthropicPromptCache(model or "claude-sonnet-4-20250514")
        elif self.provider == "openai":
            self._cache = _OpenAIPromptCache(model or "gpt-4o")
        elif self.provider == "deepseek":
            self._cache = _DeepSeekPromptCache(model or "deepseek-chat")
        else:
            self._cache = AnthropicPromptCache("claude-sonnet-4-20250514")

    def get_cached(self, cache_id: str) -> Optional[Any]:
        return self._cache.get_cached(cache_id) if hasattr(self._cache, "get_cached") else None

    def calculate_cost(self, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> CostBreakdown:
        return self._cache.calculate_cost(input_tokens, cached_input_tokens, output_tokens)


class _OpenAIPromptCache:
    """OpenAI自动缓存（无需手动管理断点）"""

    PRICING = {
        "gpt-4o": {"input_cached": 1.25, "input_uncached": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input_cached": 0.075, "input_uncached": 0.15, "output": 0.60},
    }

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        # OpenAI自动缓存: 对相同前缀自动命中，无需手动管理
        # 缓存TTL: 5-10分钟（与使用量相关）

    def calculate_cost(self, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> CostBreakdown:
        pricing = self.PRICING.get(self.model, self.PRICING["gpt-4o"])
        uncached_input = input_tokens - cached_input_tokens

        cached_cost = (cached_input_tokens / 1_000_000) * pricing["input_cached"]
        uncached_cost = (uncached_input / 1_000_000) * pricing["input_uncached"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total = cached_cost + uncached_cost + output_cost

        no_cache_cost = (input_tokens / 1_000_000) * pricing["input_uncached"] + output_cost
        savings_pct = ((no_cache_cost - total) / no_cache_cost * 100) if no_cache_cost > 0 else 0

        return CostBreakdown(
            total_tokens=input_tokens + output_tokens,
            cached_tokens=cached_input_tokens,
            uncached_tokens=uncached_input,
            output_tokens=output_tokens,
            cached_input_cost=cached_cost,
            uncached_input_cost=uncached_cost,
            output_cost=output_cost,
            total_cost=total,
            savings_pct=savings_pct,
        )


class _DeepSeekPromptCache:
    """DeepSeek Prompt缓存（支持context caching）"""

    PRICING = {
        "deepseek-chat": {"input_cached": 0.014, "input_uncached": 0.27, "output": 1.10},
        "deepseek-reasoner": {"input_cached": 0.014, "input_uncached": 0.55, "output": 2.19},
    }

    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        # DeepSeek支持磁盘缓存（KV-cache），需要在API中开启
        # 缓存TTL: 依赖服务端策略

    def calculate_cost(self, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> CostBreakdown:
        pricing = self.PRICING.get(self.model, self.PRICING["deepseek-chat"])
        uncached_input = input_tokens - cached_input_tokens

        cached_cost = (cached_input_tokens / 1_000_000) * pricing["input_cached"]
        uncached_cost = (uncached_input / 1_000_000) * pricing["input_uncached"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total = cached_cost + uncached_cost + output_cost

        no_cache_cost = (input_tokens / 1_000_000) * pricing["input_uncached"] + output_cost
        savings_pct = ((no_cache_cost - total) / no_cache_cost * 100) if no_cache_cost > 0 else 0

        return CostBreakdown(
            total_tokens=input_tokens + output_tokens,
            cached_tokens=cached_input_tokens,
            uncached_tokens=uncached_input,
            output_tokens=output_tokens,
            cached_input_cost=cached_cost,
            uncached_input_cost=uncached_cost,
            output_cost=output_cost,
            total_cost=total,
            savings_pct=savings_pct,
        )


# ============================================================
# 对比表生成 (Provider Comparison)
# ============================================================

def generate_comparison_table():
    """生成Anthropic vs OpenAI vs DeepSeek缓存对比表"""
    # 场景: 1000 token输入, 60%缓存命中, 200 token输出
    input_tokens = 1000
    cached_pct = 0.6
    cached_input = int(input_tokens * cached_pct)
    output_tokens = 200

    scenarios = [
        ("Anthropic\nClaude Sonnet 4", "anthropic", "claude-sonnet-4-20250514"),
        ("Anthropic\nClaude Opus 4", "anthropic", "claude-opus-4-20250514"),
        ("OpenAI\nGPT-4o", "openai", "gpt-4o"),
        ("OpenAI\nGPT-4o Mini", "openai", "gpt-4o-mini"),
        ("DeepSeek\nDeepSeek-Chat", "deepseek", "deepseek-chat"),
        ("DeepSeek\nDeepSeek-R1", "deepseek", "deepseek-reasoner"),
    ]

    results = []
    for name, provider, model in scenarios:
        facade = PromptCacheFacade(provider, model)
        cost = facade.calculate_cost(input_tokens, cached_input, output_tokens)
        results.append({
            "provider": name,
            "uncached_cost": f"${cost.uncached_input_cost + cost.output_cost:.4f}",
            "cached_cost": f"${cost.total_cost:.4f}",
            "savings": f"{cost.savings_pct:.0f}%",
            "cached_input_price": f"${cost.cached_input_cost / cached_input * 1e6:.2f}/MTok",
            "uncached_input_price": f"${cost.uncached_input_cost / (input_tokens - cached_input) * 1e6:.2f}/MTok",
        })

    return results


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Prompt缓存系统 - 演示运行")
    print("=" * 60)

    # 1. 断点策略演示
    print("\n--- 缓存断点放置策略 ---")

    system_prompt = "You are an expert RAG system assistant. You help users with document retrieval and question answering."
    user_context = """
Retrieved Documents:
1. RAG combines retrieval with generation for better accuracy.
2. Vector databases are essential for efficient similarity search.
3. Chunking strategies affect retrieval quality significantly.
"""
    user_question = "What is the best chunking strategy for technical documentation?"

    full_content = f"""{system_prompt}

Context:
{user_context}

User Question: {user_question}"""

    for strategy in CacheStrategy:
        bps = BreakpointPlacer.for_strategy(strategy, full_content, system_prompt)
        print(f"\n{strategy.value}:")
        for bp in bps:
            print(f"  位置={bp.position}, 标签='{bp.label}', Tokens≈{bp.token_count}, 预期命中率={bp.expected_hit_rate:.0%}")

    # 2. Anthropic缓存API格式演示
    print("\n--- Anthropic API缓存格式 ---")
    anthropic_cache = AnthropicPromptCache()

    cached = anthropic_cache.cache_prompt(full_content, "demo_cache_001", CacheStrategy.DYNAMIC_TAIL)
    print(f"缓存ID: {cached.cache_id}")
    print(f"Token数: {cached.token_count}")
    print(f"过期时间: {time.strftime('%H:%M:%S', time.localtime(cached.expires_at))}")

    # 模拟API请求格式
    request = anthropic_cache.build_cached_request(
        system_prompt,
        [{"role": "user", "content": f"Context:\n{user_context}\n\nQuestion: {user_question}"}],
        CacheStrategy.DYNAMIC_TAIL,
    )
    print(f"API请求模型: {request['model']}")
    print(f"系统提示块数: {len(request['system'])}")
    has_cache = any("cache_control" in b for b in request["system"])
    print(f"包含cache_control: {has_cache}")

    # 3. 缓存命中演示
    print("\n--- 缓存命中测试 ---")
    # 首次获取
    result1 = anthropic_cache.get_cached("demo_cache_001")
    print(f"首次获取: {'命中 ✅' if result1 else '未命中 ❌'}")

    # 等待1秒（模拟短时间重用）
    time.sleep(0.1)
    result2 = anthropic_cache.get_cached("demo_cache_001")
    print(f"再次获取: {'命中 ✅' if result2 else '未命中 ❌'}")
    print(f"访问次数: {result2.access_count if result2 else 0}")

    # 不存在的缓存
    result3 = anthropic_cache.get_cached("nonexistent")
    print(f"不存在缓存: {'命中' if result3 else '正确未命中 ✅'}")

    # 4. 成本计算与对比
    print("\n--- 成本计算示例 ---")

    # 场景: 1000 token输入, 60%缓存命中(600tokens), 200 token输出
    cost = anthropic_cache.calculate_cost(
        input_tokens=1000,
        cached_input_tokens=600,
        output_tokens=200,
    )
    print(f"场景: 1000输入(60%缓存) + 200输出")
    print(f"  总Token:       {cost.total_tokens}")
    print(f"  缓存输入Token: {cost.cached_tokens}")
    print(f"  未缓存Token:   {cost.uncached_tokens}")
    print(f"  缓存输入费用:  ${cost.cached_input_cost:.6f}")
    print(f"  未缓存输入费:  ${cost.uncached_input_cost:.6f}")
    print(f"  输出费用:      ${cost.output_cost:.6f}")
    print(f"  总费用:        ${cost.total_cost:.6f}")
    print(f"  节省:          {cost.savings_pct:.0f}%")

    # 5. 多提供商对比表
    print("\n--- 多提供商Prompt缓存对比 ---")
    print(f"（场景: {1000}输入token, {cached_pct:.0%}缓存命中, {output_tokens}输出token）")
    print()

    comparison = generate_comparison_table()
    # 表头
    print(f"{'提供商':<20} {'无缓存费用':>12} {'缓存费用':>12} {'节省':>8} {'缓存价格':>15} {'未缓存价格':>15}")
    print("-" * 85)

    for row in comparison:
        provider_lines = row["provider"].split("\n")
        print(f"{provider_lines[0]:<20} {row['uncached_cost']:>12} {row['cached_cost']:>12} {row['savings']:>8} {row['cached_input_price']:>15} {row['uncached_input_price']:>15}")
        if len(provider_lines) > 1:
            for line in provider_lines[1:]:
                print(f"  {line}")

    print()
    print("  * Anthropic: 显式cache_control，10%缓存token价格")
    print("  * OpenAI: 自动前缀缓存，50%缓存token价格")
    print("  * DeepSeek: KV-cache磁盘缓存，5%缓存token价格（最高性价比）")

    # 6. 缓存统计
    print("\n--- 缓存统计 ---")
    stats = anthropic_cache.get_stats()
    print(f"总请求:     {stats['total_requests']}")
    print(f"缓存命中:   {stats['cache_hits']}")
    print(f"缓存未命中: {stats['cache_misses']}")
    print(f"命中率:     {stats['cache_hit_rate']}")
    print(f"生效缓存数: {stats['active_caches']}")
    print(f"已省token:  {stats['total_tokens_saved']}")

    # 7. 清理过期
    cleared = anthropic_cache.clear_expired()
    print(f"\n清理过期缓存: {cleared} 条")

    print("\n" + "=" * 60)
    print("Prompt缓存演示完成！")
    print("=" * 60)
