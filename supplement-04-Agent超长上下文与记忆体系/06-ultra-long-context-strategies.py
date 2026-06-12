#!/usr/bin/env python3
"""
超长上下文策略 (Ultra-Long Context Strategies)
================================================
实现面向 128K+ token 上下文窗口的结构化分区管理、KV-Cache 复用
和注意力预算分配策略。

核心组件：
  - StructuredContextWindow: 5区上下文窗口管理器
  - PromptCompressor: 基于Token重要性的压缩器
  - AttentionBudgetAllocator: 注意力预算分配器

五个上下文区域：
  ┌─────────────────────────────────────────────┐
  │  Zone 1: PINNED_PREFIX (5%)    - 系统指令   │
  │  Zone 2: RECENT_HISTORY (20%)  - 最近对话   │
  │  Zone 3: RETRIEVED_DOCS (35%)  - 检索文档   │
  │  Zone 4: AGENT_SCRATCH (25%)   - Agent思考  │
  │  Zone 5: RESERVED_FUTURE (15%) - 预留空间   │
  └─────────────────────────────────────────────┘

关键技术：
  - KV-Cache 复用：pinned_prefix 在多次推理间保持不动，节省重复计算
  - Token 重要性评分：基于位置、语义角色、TF-IDF 等维度的综合评分
  - 注意力预算：根据区域重要性分配注意力权重
"""

from __future__ import annotations

import re
import math
import json
import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict


# ============================================================================
# 上下文区域定义
# ============================================================================

class ContextZone(Enum):
    """上下文窗口的五个功能区域"""
    PINNED_PREFIX = auto()      # 固定前缀（系统指令、角色定义）
    RECENT_HISTORY = auto()     # 近期对话历史
    RETRIEVED_DOCS = auto()     # 检索到的文档/知识
    AGENT_SCRATCH = auto()      # Agent 的中间推理空间
    RESERVED_FUTURE = auto()    # 预留给未来使用的空间

    @property
    def cn_name(self) -> str:
        names = {
            ContextZone.PINNED_PREFIX: "固定前缀",
            ContextZone.RECENT_HISTORY: "近期历史",
            ContextZone.RETRIEVED_DOCS: "检索文档",
            ContextZone.AGENT_SCRATCH: "Agent暂存",
            ContextZone.RESERVED_FUTURE: "预留空间",
        }
        return names.get(self, "未知")

    @property
    def typical_budget(self) -> float:
        """典型预算比例"""
        return {
            ContextZone.PINNED_PREFIX: 0.05,
            ContextZone.RECENT_HISTORY: 0.20,
            ContextZone.RETRIEVED_DOCS: 0.35,
            ContextZone.AGENT_SCRATCH: 0.25,
            ContextZone.RESERVED_FUTURE: 0.15,
        }[self]


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ContextAllocation:
    """上下文区域的一个分配项"""
    content: str                    # 文本内容
    token_count: int               # 估算Token数
    timestamp: float               # 创建/更新时间戳
    priority: float = 0.5          # 优先级 [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenImportanceScore:
    """单个 Token 的重要性评分"""
    token: str
    position_score: float = 0.0     # 位置重要性
    semantic_score: float = 0.0     # 语义重要性
    frequency_score: float = 0.0    # 频率重要性
    composite_score: float = 0.0    # 综合评分

    def __post_init__(self):
        if self.composite_score == 0.0:
            self.composite_score = (
                0.3 * self.position_score +
                0.5 * self.semantic_score +
                0.2 * self.frequency_score
            )


@dataclass
class AttentionWeight:
    """注意力权重分配"""
    zone: ContextZone
    weight: float                   # 归一化权重
    num_tokens: int                 # Token数

# ============================================================================
# 结构化上下文窗口
# ============================================================================

class StructuredContextWindow:
    """
    结构化上下文窗口管理器。

    特性：
    - 5个语义区域，每个区域独立的Token预算
    - pinned_prefix 支持 KV-Cache 复用
    - 动态配额重分配
    - 溢出时的内容精简策略
    """

    def __init__(self, total_token_budget: int = 128000,
                 budget_distribution: Optional[Dict[ContextZone, float]] = None):
        """
        Args:
            total_token_budget: 总Token预算（如 128K）
            budget_distribution: 各区域预算比例，None时使用默认值
        """
        self.total_budget = total_token_budget
        self.budget_distribution = budget_distribution or {
            zone: zone.typical_budget for zone in ContextZone
        }
        self._zones: Dict[ContextZone, List[ContextAllocation]] = {
            zone: [] for zone in ContextZone
        }
        self._zone_budgets: Dict[ContextZone, int] = {}
        self._pinned_prefix_hash: Optional[str] = None
        self._kv_cache_hits: int = 0
        self._kv_cache_misses: int = 0
        self._recalculate_budgets()

    def _recalculate_budgets(self):
        """根据比例重新计算各区域Token预算"""
        total_ratio = sum(self.budget_distribution.values())
        for zone in ContextZone:
            ratio = self.budget_distribution.get(zone, 0.0)
            self._zone_budgets[zone] = int(
                self.total_budget * ratio / total_ratio
            )

    # ---- Token估算 ----

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估算文本的Token数量。

        简化估算（实际使用tiktoken或类似库）：
        - 英文：words * 1.3 (考虑子词切分)
        - 中文：characters * 0.6 (一个中文字约0.6 token)
        - 混合：取加权平均
        """
        if not text:
            return 0

        chinese_chars = len(re.findall(r'[一-鿿]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other = max(len(text) - chinese_chars - english_words, 0)

        # 中文约0.6 token/字，英文约1.3 token/词，其他约0.25 token/字符
        return int(chinese_chars * 0.6 + english_words * 1.3 + other * 0.25)

    def get_zone_usage(self, zone: ContextZone) -> int:
        """获取某区域当前使用的Token数"""
        return sum(a.token_count for a in self._zones[zone])

    def get_total_usage(self) -> int:
        """获取总Token使用量"""
        return sum(self.get_zone_usage(z) for z in ContextZone)

    # ---- 添加内容到区域 ----

    def add_to_zone(self, zone: ContextZone, content: str,
                    priority: float = 0.5,
                    metadata: Optional[Dict] = None) -> bool:
        """
        向指定区域添加内容。

        Args:
            zone: 目标区域
            content: 文本内容
            priority: 优先级
            metadata: 附加元数据

        Returns:
            True 如果添加成功，False 如果区域已满且无法替换低优先级内容
        """
        token_count = self.estimate_tokens(content)
        budget = self._zone_budgets[zone]
        current_usage = self.get_zone_usage(zone)

        # 检查是否超出预算
        if current_usage + token_count > budget:
            # 如果空间不足，尝试回收低优先级内容
            freed = self._trim_zone(zone, current_usage + token_count - budget)
            if not freed:
                return False

        allocation = ContextAllocation(
            content=content,
            token_count=token_count,
            timestamp=__import__('time').time(),
            priority=priority,
            metadata=metadata or {},
        )
        self._zones[zone].append(allocation)
        return True

    def _trim_zone(self, zone: ContextZone, tokens_needed: int) -> bool:
        """
        在区域内释放指定数量的Token空间。

        策略：优先移除优先级最低、最旧的内容。
        """
        allocations = self._zones[zone]
        if not allocations:
            return False

        # 按优先级升序排序（低优先级先删）
        sorted_items = sorted(enumerate(allocations), key=lambda x: (
            x[1].priority,
            -x[1].timestamp,
        ))

        tokens_freed = 0
        indices_to_remove = set()
        for idx, alloc in sorted_items:
            if tokens_freed >= tokens_needed:
                break
            tokens_freed += alloc.token_count
            indices_to_remove.add(idx)

        if tokens_freed < tokens_needed:
            return False

        # 执行删除
        self._zones[zone] = [
            alloc for i, alloc in enumerate(allocations)
            if i not in indices_to_remove
        ]
        return True

    # ---- KV-Cache 复用 ----

    def set_pinned_prefix(self, content: str):
        """
        设置固定前缀（如系统提示词），支持KV-Cache复用。

        当 pinned_prefix 内容未变时，KV-Cache可直接复用，
        无需重复计算前缀部分的 Key-Value 张量。
        """
        new_hash = hashlib.md5(content.encode()).hexdigest()

        if new_hash == self._pinned_prefix_hash:
            self._kv_cache_hits += 1
        else:
            self._kv_cache_misses += 1
            self._pinned_prefix_hash = new_hash
            # 清空旧前缀
            self._zones[ContextZone.PINNED_PREFIX].clear()
            self.add_to_zone(ContextZone.PINNED_PREFIX, content, priority=1.0,
                             metadata={"pinned": True, "hash": new_hash})

    def _calculate_kv_cache_reuse(self) -> Dict[str, Any]:
        """
        计算KV-Cache复用统计。

        KV-Cache 复用率 = hits / (hits + misses)
        """
        total = self._kv_cache_hits + self._kv_cache_misses
        rate = self._kv_cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._kv_cache_hits,
            "misses": self._kv_cache_misses,
            "reuse_rate": round(rate, 4),
            "estimated_savings_tokens": self._kv_cache_hits * self.estimate_tokens(
                " ".join(a.content for a in self._zones[ContextZone.PINNED_PREFIX])
            ),
        }

    # ---- 上下文组装 ----

    def get_assembled_context(self) -> str:
        """
        按正确的顺序组装完整的上下文。

        顺序：PINNED_PREFIX → RETRIEVED_DOCS → RECENT_HISTORY → AGENT_SCRATCH

        RESERVED_FUTURE 不参与当前组装。
        """
        parts = []
        order = [
            ContextZone.PINNED_PREFIX,
            ContextZone.RETRIEVED_DOCS,
            ContextZone.RECENT_HISTORY,
            ContextZone.AGENT_SCRATCH,
        ]

        separators = {
            ContextZone.PINNED_PREFIX: "",
            ContextZone.RETRIEVED_DOCS: "\n\n--- 相关文档 ---\n",
            ContextZone.RECENT_HISTORY: "\n\n--- 对话历史 ---\n",
            ContextZone.AGENT_SCRATCH: "\n\n--- Agent思考 ---\n",
        }

        for zone in order:
            allocs = self._zones[zone]
            if allocs:
                zone_text = "\n".join(a.content for a in allocs)
                if separators.get(zone):
                    parts.append(separators[zone])
                parts.append(zone_text)

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """获取上下文窗口统计"""
        stats = {
            "total_budget": self.total_budget,
            "total_usage": self.get_total_usage(),
            "utilization": round(self.get_total_usage() / self.total_budget * 100, 2),
            "zones": {},
            "kv_cache": self._calculate_kv_cache_reuse(),
        }
        for zone in ContextZone:
            stats["zones"][zone.name] = {
                "budget": self._zone_budgets[zone],
                "usage": self.get_zone_usage(zone),
                "allocation_count": len(self._zones[zone]),
                "utilization_pct": round(
                    self.get_zone_usage(zone) / max(self._zone_budgets[zone], 1) * 100, 1
                ),
            }
        return stats


# ============================================================================
# Prompt 压缩器
# ============================================================================

class PromptCompressor:
    """
    基于 Token 重要性的 Prompt 压缩器。

    支持：
    - 按比例压缩（compress_by_ratio）
    - Token 重要性评分（_token_importance_scoring）
    - 保留关键 Token，丢弃低重要性 Token
    """

    # 高重要性关键词（语义负载高的词）
    HIGH_IMPORTANCE_PATTERNS = [
        r'\b(must|required|crucial|critical|mandatory|essential|important)\b',
        r'\b(禁止|必须|重要|关键|核心|强制)\b',
        r'\b(error|failure|exception|bug|issue)\b',
        r'\b(错误|失败|异常|问题|故障)\b',
    ]

    # 低重要性停用词模式
    STOPWORD_PATTERNS = [
        r'\b(the|a|an|is|are|was|were|be|been|being|have|has|had|do|does|did)\b',
        r'\b(的|了|在|是|我|你|他|她|它|们|这|那|吗|呢|吧|啊|哦|嗯)\b',
    ]

    def compress_by_ratio(self, text: str, target_ratio: float) -> str:
        """
        按目标压缩比压缩文本。

        算法：
        1. 分Token并评分
        2. 按分数排序，保留 top target_ratio 部分
        3. 重新拼接

        Args:
            text: 原始文本
            target_ratio: 目标压缩比 (0-1)，如0.2表示保留20%

        Returns:
            压缩后的文本
        """
        if target_ratio >= 1.0:
            return text
        if target_ratio <= 0.0:
            return ""

        # 分句
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text[:int(len(text) * target_ratio)]

        # 句子重要性评分
        scored_sentences = []
        for sentence in sentences:
            scores = self._token_importance_scoring(sentence)
            if scores:
                avg_score = sum(s.composite_score for s in scores) / len(scores)
            else:
                avg_score = 0.5
            # 位置加权：首尾句更高
            idx = sentences.index(sentence)
            norm_pos = idx / max(len(sentences) - 1, 1)
            pos_weight = 1.0 - abs(norm_pos - 0.5) * 0.5  # 首尾权重高
            final_score = avg_score * pos_weight
            scored_sentences.append((sentence, final_score, idx))

        # 按分数排序
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # 选择top-k句子，达到目标压缩比
        target_chars = max(int(len(text) * target_ratio), 10)
        selected = []
        current_chars = 0

        for sentence, score, orig_idx in scored_sentences:
            if current_chars >= target_chars:
                break
            selected.append((orig_idx, sentence))
            current_chars += len(sentence)

        # 按原始顺序排列
        selected.sort(key=lambda x: x[0])

        return "".join(s for _, s in selected)

    def _token_importance_scoring(self, text: str) -> List[TokenImportanceScore]:
        """
        Token 重要性评分算法。

        评分维度：
        1. 位置分数：位于句首/句末的Token更重要
        2. 语义分数：是否匹配高重要性模式
        3. 频率分数：TF-IDF 风格的反文档频率
        """
        # 简单分词（空格/标点分割）
        tokens = re.findall(r'[\w一-鿿]+|[^\w\s]', text)
        if not tokens:
            return []

        n = len(tokens)
        scores = []

        # 全局Token频率（用于IDF计算）
        token_freq = Counter(t.lower() for t in tokens)

        for i, token in enumerate(tokens):
            # 位置分数：越靠首尾越重要
            norm_pos = i / max(n - 1, 1)
            position_score = 1.0 - abs(norm_pos - 0.5) * 1.5
            position_score = max(position_score, 0.1)

            # 语义分数
            semantic_score = 0.5  # 默认值
            for pattern in self.HIGH_IMPORTANCE_PATTERNS:
                if re.search(pattern, token, re.IGNORECASE):
                    semantic_score = 0.9
                    break

            # 频率分数（TF-IDF风格）
            freq = token_freq[token.lower()]
            freq_score = 1.0 - (freq / max(len(tokens), 1)) * 2.0
            freq_score = max(freq_score, 0.1)

            # 复合分数
            composite = (
                0.3 * position_score +
                0.5 * semantic_score +
                0.2 * freq_score
            )

            scores.append(TokenImportanceScore(
                token=token,
                position_score=round(position_score, 3),
                semantic_score=round(semantic_score, 3),
                frequency_score=round(freq_score, 3),
                composite_score=round(composite, 3),
            ))

        return scores

    def compress_with_keyword_boost(self, text: str, target_ratio: float,
                                     keywords: List[str]) -> str:
        """
        带关键词加权的压缩。

        包含关键词的句子获得更高保留优先级。

        Args:
            text: 原始文本
            target_ratio: 目标压缩比
            keywords: 需要优先保留的关键词列表
        """
        # 对包含关键词的句子附加更高分数
        # 复用 compress_by_ratio 但增强关键词句子得分
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        scored = []
        for s in sentences:
            base_score = 0.5
            for kw in keywords:
                if kw.lower() in s.lower():
                    base_score += 0.3
            scored.append((s, min(base_score, 1.0)))

        # 按得分排序
        scored.sort(key=lambda x: x[1], reverse=True)
        target_chars = int(len(text) * target_ratio)
        selected = []
        current = 0
        for s, sc in scored:
            if current >= target_chars:
                break
            selected.append(s)
            current += len(s)

        # 按原始顺序排列
        result = []
        for s in sentences:
            if s in selected:
                result.append(s)
                selected.remove(s)

        return "".join(result)


# ============================================================================
# 注意力预算分配器
# ============================================================================

class AttentionBudgetAllocator:
    """
    注意力预算分配器。

    在LLM推理时，不同区域的Token获得不同的注意力权重。

    原理：
    - 重要区域（如系统指令、近期对话）获得更高注意力
    - 检索文档中的关键段落获得聚焦注意力
    - 预留空间零注意力（不参与计算，仅占位）
    """

    def __init__(self):
        self._attention_weights: Dict[ContextZone, float] = {}
        self._token_allocations: Dict[ContextZone, int] = {}

    def allocate_attention_weights(
        self,
        zone_allocations: Dict[ContextZone, List[ContextAllocation]],
        total_attention_budget: float = 1.0,
    ) -> Dict[ContextZone, AttentionWeight]:
        """
        分配注意力权重。

        Args:
            zone_allocations: 各区域的分配项
            total_attention_budget: 总注意力预算（通常为1.0）

        Returns:
            {zone: AttentionWeight}
        """
        # 区域基础权重
        base_weights = {
            ContextZone.PINNED_PREFIX: 0.30,     # 系统指令最重要
            ContextZone.RECENT_HISTORY: 0.30,     # 近期对话同样重要
            ContextZone.RETRIEVED_DOCS: 0.25,     # 检索文档
            ContextZone.AGENT_SCRATCH: 0.10,      # Agent思考
            ContextZone.RESERVED_FUTURE: 0.05,    # 预留空间最少
        }

        # 根据实际内容调整
        adjusted = {}
        for zone in ContextZone:
            allocs = zone_allocations.get(zone, [])
            token_count = sum(a.token_count for a in allocs)
            priority_sum = sum(a.priority for a in allocs)
            avg_priority = priority_sum / max(len(allocs), 1)

            # 有内容 + 高优先级 = 更多注意力
            content_bonus = min(token_count / 10000, 0.1)  # 上限0.1
            priority_bonus = avg_priority * 0.1             # 上限0.1

            adjusted[zone] = base_weights.get(zone, 0.1) + content_bonus + priority_bonus

        # 归一化
        total = sum(adjusted.values())
        normalized = {z: w / total * total_attention_budget for z, w in adjusted.items()}
        self._attention_weights = normalized

        # 生成权重对象
        result = {}
        for zone in ContextZone:
            allocs = zone_allocations.get(zone, [])
            token_count = sum(a.token_count for a in allocs)
            result[zone] = AttentionWeight(
                zone=zone,
                weight=round(normalized[zone], 4),
                num_tokens=token_count,
            )

        return result

    def reorder_for_attention(self,
                               zone_allocations: Dict[ContextZone, List[ContextAllocation]]
                               ) -> List[ContextAllocation]:
        """
        按注意力优先级重新排序分配项。

        高注意力区域内的分配项排在前面，以便：
        - LLM的recency bias优先处理重要内容
        - 对抗"Lost in the Middle"效应
        """
        zone_order = [
            ContextZone.PINNED_PREFIX,
            ContextZone.RECENT_HISTORY,
            ContextZone.RETRIEVED_DOCS,
            ContextZone.AGENT_SCRATCH,
        ]

        result = []
        for zone in zone_order:
            allocs = zone_allocations.get(zone, [])
            # 区域内按优先级排序
            sorted_allocs = sorted(allocs, key=lambda a: a.priority, reverse=True)
            result.extend(sorted_allocs)

        return result

    def get_attention_report(self) -> Dict[str, Any]:
        """生成注意力分配报告"""
        return {
            "zone_weights": {
                z.name: round(w, 4)
                for z, w in self._attention_weights.items()
            },
            "total_attention": round(sum(self._attention_weights.values()), 4),
        }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示"""
    print("=" * 70)
    print("  超长上下文策略 (Ultra-Long Context Strategies) 演示")
    print("=" * 70)

    # ---- 1. 结构化上下文窗口 ----
    print("\n【1. 结构化上下文窗口 (128K Budget)】")
    print("-" * 50)

    window = StructuredContextWindow(total_token_budget=128000)

    # 设置固定前缀（系统提示词）
    system_prompt = (
        "你是一个专业的AI助手。你必须：\n"
        "1. 始终以中文回复用户\n"
        "2. 保持专业和礼貌的语气\n"
        "3. 在不确定时主动请求澄清\n"
        "4. 严禁提供有害信息\n"
        "5. 引用相关文档中的信息来支持回答"
    )
    window.set_pinned_prefix(system_prompt)
    print(f"  固定前缀设置完成，Token估计: {window.estimate_tokens(system_prompt)}")

    # 添加近期历史
    for i in range(5):
        turn = f"用户回合{i+1}: " + f"请问关于产品X的第{i+1}个问题？" * 3
        window.add_to_zone(ContextZone.RECENT_HISTORY, turn, priority=0.7)

    # 添加检索文档
    for i in range(3):
        doc = f"文档{i+1}: " + f"产品X的详细规格说明，版本{i+1}.0，支持多种配置..." * 5
        window.add_to_zone(ContextZone.RETRIEVED_DOCS, doc, priority=0.6)

    # 添加Agent思考空间
    window.add_to_zone(ContextZone.AGENT_SCRATCH,
                       "正在分析: 用户询问产品X性能参数... 步骤1: 检索规格文档完成"
                       " 步骤2: 对比竞品数据 步骤3: 组织回复结构",
                       priority=0.8)

    print(f"  总使用量: {window.get_total_usage()} / {window.total_budget} tokens")
    print(f"  利用率: {window.get_total_usage()/window.total_budget*100:.1f}%")

    # KV-Cache 模拟
    window.set_pinned_prefix(system_prompt)  # 相同前缀，应命中缓存
    window.set_pinned_prefix(system_prompt + " 新增: 支持图像输入")  # 修改前缀
    window.set_pinned_prefix(system_prompt)  # 恢复，应miss

    kv_stats = window._calculate_kv_cache_reuse()
    print(f"\n  KV-Cache 统计:")
    print(f"    Hits: {kv_stats['hits']}")
    print(f"    Misses: {kv_stats['misses']}")
    print(f"    复用率: {kv_stats['reuse_rate']:.1%}")

    # ---- 2. Prompt 压缩 ----
    print("\n【2. Prompt 压缩演示 (5K → 1K)】")
    print("-" * 50)

    # 生成5K字符的模拟文本
    original_text = (
        "在人工智能快速发展的今天，大语言模型(LLM)已经成为了自然语言处理领域的核心技术。"
        "我们必须认识到，上下文窗口的管理对于构建可靠的生产级AI应用至关重要。"
        "关键的是，有效的压缩策略可以大幅降低计算成本，同时保持回答质量。"
        "系统指令部分包含了核心的安全约束和行为规范，这部分内容绝对不能丢失。"
        "近期对话历史记录了用户的当前意图和上下文，对于理解用户需求非常关键。"
        "然而，早期的对话细节可能已经不再重要，可以被安全地压缩或丢弃。"
        "错误处理是系统设计中不可忽视的一环，必须建立完善的异常捕获和恢复机制。"
        "重要的事情需要再强调一遍：用户隐私保护是最高优先级的安全要求。"
    ) * 150  # 扩展到约5K chars

    original_tokens = window.estimate_tokens(original_text)
    print(f"  原始文本: {len(original_text)} 字符 (~{original_tokens} tokens)")

    compressor = PromptCompressor()
    compressed = compressor.compress_by_ratio(original_text, target_ratio=0.2)
    compressed_tokens = window.estimate_tokens(compressed)
    print(f"  压缩后: {len(compressed)} 字符 (~{compressed_tokens} tokens)")
    print(f"  压缩比: {len(compressed)/max(len(original_text),1)*100:.1f}%")
    print(f"  压缩后内容预览: {compressed[:200]}...")

    # 关键词加权压缩
    compressed_kw = compressor.compress_with_keyword_boost(
        original_text, target_ratio=0.2,
        keywords=["安全", "隐私", "错误", "关键", "重要"],
    )
    print(f"\n  关键词加权压缩: {len(compressed_kw)} 字符")
    print(f"  内容预览: {compressed_kw[:200]}...")

    # ---- 3. 注意力预算分配 ----
    print("\n【3. 注意力预算分配】")
    print("-" * 50)

    allocator = AttentionBudgetAllocator()
    weights = allocator.allocate_attention_weights(window._zones)

    for zone, aw in weights.items():
        bar = "█" * int(aw.weight * 50)
        print(f"  {zone.cn_name:10s}: {aw.weight:.3f} | {bar} | {aw.num_tokens} tokens")

    # 重排序
    reordered = allocator.reorder_for_attention(window._zones)
    print(f"\n  重排序后前5项:")
    for i, alloc in enumerate(reordered[:5]):
        content_preview = alloc.content[:60]
        print(f"  [{i+1}] P={alloc.priority:.2f} | {content_preview}...")

    # ---- 4. 完整统计 ----
    print("\n【4. 上下文窗口完整统计】")
    print("-" * 50)
    stats = window.get_stats()
    # 简化显示
    for zone_name, zone_stats in stats["zones"].items():
        print(f"  {zone_name:20s}: {zone_stats['usage']:>6d}/{zone_stats['budget']:>6d} "
              f"({zone_stats['utilization_pct']:>5.1f}%) [{zone_stats['allocation_count']}项]")

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
