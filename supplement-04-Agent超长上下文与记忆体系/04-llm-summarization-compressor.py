#!/usr/bin/env python3
"""
LLM摘要压缩器 (LLM Summarization Compressor)
===============================================
实现三种摘要压缩策略，用于在超长对话中高效管理上下文窗口。

三种摘要风格：
  1. hierarchical（层级摘要）: 先分组摘要，再对组摘要进行二次摘要
  2. incremental（增量摘要）: 每次只对新内容进行摘要，与之前摘要合并
  3. structured（结构化摘要）: 按实体/关系/事件/意图等维度组织摘要

核心组件：
  - LLMSummarizationCompressor: 摘要压缩器
  - StructuredSummary: 结构化摘要数据结构
  - incremental_summarize: 增量摘要算法
  - recursive_summarize: 递归摘要算法（层级）
  - _validate_summary: 摘要质量验证
"""

from __future__ import annotations

import re
import math
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto
from collections import OrderedDict


# ============================================================================
# 摘要风格枚举
# ============================================================================

class SummaryStyle(Enum):
    """摘要风格"""
    HIERARCHICAL = auto()     # 层级摘要
    INCREMENTAL = auto()      # 增量摘要
    STRUCTURED = auto()       # 结构化摘要


# ============================================================================
# 结构化摘要数据类
# ============================================================================

@dataclass
class StructuredSummary:
    """结构化摘要，按维度组织信息"""
    entities: List[Dict[str, str]] = field(default_factory=list)
    # [{name, type, attributes}]
    events: List[Dict[str, str]] = field(default_factory=list)
    # [{action, subject, object, timestamp}]
    decisions: List[str] = field(default_factory=list)
    # ["decision1", "decision2"]
    intents: List[str] = field(default_factory=list)
    # ["intent1", ...]
    facts: List[str] = field(default_factory=list)
    # ["fact1", "fact2"]
    open_questions: List[str] = field(default_factory=list)
    # ["question1", ...]
    summary: str = ""
    # 自由文本摘要

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "entities": self.entities,
            "events": self.events,
            "decisions": self.decisions,
            "intents": self.intents,
            "facts": self.facts,
            "open_questions": self.open_questions,
            "summary": self.summary,
        }

    def word_count(self) -> int:
        """总词数估算"""
        total = len(self.summary.split())
        total += sum(len(e.get("attributes", "").split()) for e in self.entities)
        total += sum(len(e.get("action", "").split()) for e in self.events)
        total += sum(len(d.split()) for d in self.decisions)
        total += sum(len(i.split()) for i in self.intents)
        total += sum(len(f.split()) for f in self.facts)
        total += sum(len(q.split()) for q in self.open_questions)
        return total


# ============================================================================
# Mock LLM 备用摘要引擎
# ============================================================================

class MockLLM:
    """
    模拟LLM的摘要生成器。
    在无真实LLM API时使用基于规则的抽取式+启发式摘要。
    实际生产环境中替换为 GPT-4 / Claude API 调用。
    """

    @staticmethod
    def summarize(messages: List[Dict[str, str]], max_words: int = 100) -> str:
        """
        基于规则的摘要生成。

        策略：
        1. 提取首句和末句（通常是意图和结论）
        2. 提取包含数字/日期/专有名词的句子
        3. 使用 TextRank 式关键词加权的句子排序
        """
        if not messages:
            return "（无内容）"

        # 提取所有文本内容
        texts = [m.get("content", "") for m in messages if "content" in m]
        if not texts:
            return "（无文本内容）"

        combined = " | ".join(texts)

        # 分句
        sentences = re.split(r'[。！？.!?\n]+', combined)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

        if not sentences:
            return combined[:max_words]

        # 句子评分
        scored = []
        for s in sentences:
            score = 1.0
            # 包含数字的句子权重高
            if re.search(r'\d+', s):
                score += 2.0
            # 包含专有名词（大写字母/中文名）权重高
            if re.search(r'[A-Z][a-z]+|[一-鿿]{2,3}', s):
                score += 1.5
            # 否定/转折词权重高
            if re.search(r'(不|未|没有|但是|然而|可是|except|but|however)', s):
                score += 1.0
            # 位置加权：首尾句重要
            idx = sentences.index(s)
            norm_idx = idx / max(len(sentences) - 1, 1)
            if norm_idx < 0.2 or norm_idx > 0.8:
                score += 1.0
            scored.append((s, score))

        # 按分数排序，选 top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in scored[:min(5, len(scored))]]

        summary = "。".join(selected) + "。"
        return summary[:max_words * 3]  # Rough char limit

    @staticmethod
    def extract_entities(text: str) -> List[Dict[str, str]]:
        """提取实体（简单规则）"""
        entities = []

        # 提取英文大写词和中文专名
        for match in re.finditer(r'[A-Z][a-z]+', text):
            name = match.group()
            if name not in [e["name"] for e in entities]:
                entities.append({"name": name, "type": "unknown", "attributes": ""})

        # 提取日期
        for match in re.finditer(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', text):
            entities.append({"name": match.group(), "type": "date", "attributes": ""})

        return entities[:10]

    @staticmethod
    def extract_facts(text: str) -> List[str]:
        """提取事实性陈述"""
        facts = []
        sentences = re.split(r'[。！？.!?\n]+', text)
        for s in sentences:
            s = s.strip()
            # 包含数字或"是"的判断句为事实
            if re.search(r'\d+', s) or '是' in s:
                facts.append(s[:100])
        return facts[:5]


# ============================================================================
# LLM摘要压缩器
# ============================================================================

class LLMSummarizationCompressor:
    """
    LLM摘要压缩器。

    支持三种摘要风格和完整的质量验证管线。

    使用方法:
        compressor = LLMSummarizationCompressor()
        summary = compressor.compress(messages, style=SummaryStyle.HIERARCHICAL)
    """

    def __init__(self, llm: Optional[Any] = None, max_segment_size: int = 5):
        """
        Args:
            llm: LLM客户端（None 时使用 MockLLM 备用）
            max_segment_size: 层级摘要时每组最大消息数
        """
        self.llm = llm or MockLLM()
        self.max_segment_size = max_segment_size
        self._compression_history: List[Dict[str, Any]] = []
        self._current_summary: Optional[str] = None

    def compress(self,
                 messages: List[Dict[str, str]],
                 style: SummaryStyle = SummaryStyle.HIERARCHICAL,
                 target_ratio: float = 0.2) -> str:
        """
        对消息列表进行摘要压缩。

        Args:
            messages: 消息列表 [{"role": "user/assistant", "content": "..."}, ...]
            style: 摘要风格
            target_ratio: 目标压缩比（摘要长度/原始长度）

        Returns:
            压缩后的摘要文本
        """
        if style == SummaryStyle.HIERARCHICAL:
            result = self._compress_hierarchical(messages)
        elif style == SummaryStyle.INCREMENTAL:
            result = self._compress_incremental(messages)
        elif style == SummaryStyle.STRUCTURED:
            result = self._compress_structured(messages)
        else:
            result = self._compress_hierarchical(messages)

        self._current_summary = result
        self._compression_history.append({
            "style": style.name,
            "input_messages": len(messages),
            "output_length": len(result),
            "timestamp": __import__('time').time(),
        })

        return result

    def _compress_hierarchical(self, messages: List[Dict[str, str]]) -> str:
        """层级摘要：分组 -> 组摘要 -> 元摘要"""
        # 分组
        segments = []
        for i in range(0, len(messages), self.max_segment_size):
            segment = messages[i:i + self.max_segment_size]
            segments.append(segment)

        # 每组摘要
        segment_summaries = []
        for seg in segments:
            seg_summary = self.llm.summarize(seg, max_words=50)
            segment_summaries.append(seg_summary)

        # 元摘要
        if len(segment_summaries) <= 1:
            return segment_summaries[0] if segment_summaries else ""

        # 将组摘要作为新的"消息"列表进行二次摘要
        pseudo_messages = [{"role": "assistant", "content": s}
                           for s in segment_summaries]
        return self.llm.summarize(pseudo_messages, max_words=100)

    def _compress_incremental(self, messages: List[Dict[str, str]]) -> str:
        """增量摘要：逐条加入并更新摘要"""
        if not messages:
            return ""

        prev_summary = ""
        for i in range(0, len(messages), 3):
            chunk = messages[i:i + 3]
            chunk_text = " ".join(m.get("content", "") for m in chunk)
            prev_summary = self.incremental_summarize(prev_summary, chunk_text)

        return prev_summary

    def _compress_structured(self, messages: List[Dict[str, str]]) -> str:
        """结构化摘要：按维度组织"""
        combined = " ".join(m.get("content", "") for m in messages if "content" in m)

        entities = self.llm.extract_entities(combined)
        facts = self.llm.extract_facts(combined)
        summary = self.llm.summarize(messages, max_words=100)

        structured = StructuredSummary(
            entities=entities,
            facts=facts,
            summary=summary,
        )

        return self._format_structured_summary(structured)

    def _format_structured_summary(self, ss: StructuredSummary) -> str:
        """格式化结构化摘要为可读文本"""
        lines = ["【对话摘要】", ss.summary]

        if ss.entities:
            lines.append("\n【关键实体】")
            for e in ss.entities:
                lines.append(f"  - {e['name']} ({e['type']})")

        if ss.facts:
            lines.append("\n【关键事实】")
            for f in ss.facts:
                lines.append(f"  - {f}")

        if ss.decisions:
            lines.append("\n【决策记录】")
            for d in ss.decisions:
                lines.append(f"  - {d}")

        if ss.open_questions:
            lines.append("\n【待解决问题】")
            for q in ss.open_questions:
                lines.append(f"  - {q}")

        return "\n".join(lines)

    # ---- 增量摘要算法 ----

    def incremental_summarize(self, prev_summary: str, new_content: str) -> str:
        """
        增量摘要核心算法。

        公式：new_summary = merge(prev_summary * decay + new_content_extract)

        Args:
            prev_summary: 之前的摘要
            new_content: 新增内容

        Returns:
            更新后的摘要
        """
        if not prev_summary:
            # 首次摘要
            return self.llm.summarize(
                [{"role": "assistant", "content": new_content}],
                max_words=80,
            )

        # 提取新旧关键信息
        new_extract = self.llm.summarize(
            [{"role": "assistant", "content": new_content}],
            max_words=40,
        )

        # 合并策略：旧摘要取关键句 + 新摘要
        old_sentences = re.split(r'[。！？.!?]+', prev_summary)
        old_sentences = [s.strip() for s in old_sentences if len(s.strip()) > 5]

        # 时间衰减：越旧的句子保留越少
        kept_sentences = []
        for i, s in enumerate(old_sentences):
            decay = 1.0 / (1.0 + i * 0.3)  # 越靠后（越旧）衰减越多
            if decay > 0.4 and len(kept_sentences) < 3:
                kept_sentences.append(s)

        combined = "。".join(kept_sentences) + "。" + new_extract
        return combined

    # ---- 递归摘要算法 ----

    def recursive_summarize(self, segments: List[str], depth: int = 0,
                            max_depth: int = 3) -> str:
        """
        递归摘要（层级变体）。

        将文本段列表递归合并摘要，直至只剩一个摘要。

        Args:
            segments: 待摘要的文本段列表
            depth: 当前递归深度
            max_depth: 最大递归深度

        Returns:
            最终摘要
        """
        if len(segments) <= 1 or depth >= max_depth:
            combined = "\n".join(segments)
            return self.llm.summarize(
                [{"role": "assistant", "content": combined}],
                max_words=100,
            )

        # 两两合并
        merged = []
        for i in range(0, len(segments), 2):
            pair = segments[i:i + 2]
            if len(pair) == 1:
                merged.append(pair[0])
            else:
                combined = "\n".join(pair)
                summary = self.llm.summarize(
                    [{"role": "assistant", "content": combined}],
                    max_words=60,
                )
                merged.append(summary)

        return self.recursive_summarize(merged, depth + 1, max_depth)

    # ---- 摘要质量验证 ----

    def _validate_summary(self, original: List[Dict[str, str]],
                          summary: str) -> dict:
        """
        验证摘要质量。

        Returns:
            {
                "compression_ratio": float,      # 压缩比
                "fact_retention_estimate": float, # 事实保留率估算
                "length_ok": bool,               # 长度是否合理
                "issues": list[str],              # 问题列表
            }
        """
        original_text = " ".join(m.get("content", "") for m in original)
        original_words = len(original_text)
        summary_words = len(summary)

        compression_ratio = summary_words / max(original_words, 1)

        # 检查是否过短/过长
        issues = []
        if compression_ratio < 0.02:
            issues.append("摘要过短，可能丢失大量信息")
        elif compression_ratio > 0.8:
            issues.append("摘要过长，压缩效果不佳")

        # 事实保留率估算（基于关键词重叠）
        original_keywords = set(
            w for w in re.findall(r'[\w一-鿿]{2,}', original_text.lower())
            if len(w) > 1
        )
        summary_keywords = set(
            w for w in re.findall(r'[\w一-鿿]{2,}', summary.lower())
            if len(w) > 1
        )

        if original_keywords:
            fact_retention = len(summary_keywords & original_keywords) / len(original_keywords)
        else:
            fact_retention = 1.0

        return {
            "compression_ratio": round(compression_ratio, 4),
            "fact_retention_estimate": round(fact_retention, 4),
            "length_ok": 0.02 <= compression_ratio <= 0.8,
            "issues": issues,
            "original_chars": original_words,
            "summary_chars": summary_words,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取压缩器统计"""
        return {
            "total_compressions": len(self._compression_history),
            "styles_used": list(set(
                h["style"] for h in self._compression_history
            )),
            "has_current_summary": self._current_summary is not None,
        }


# ============================================================================
# 模拟对话数据
# ============================================================================

def generate_simulated_conversation(turns: int = 20) -> List[Dict[str, str]]:
    """
    生成一段20轮的模拟技术支持对话。

    模拟场景：用户报告系统故障，技术支持Agent逐步排查。
    """
    conversation = [
        {"role": "user", "content": "你好，我们的支付系统好像出了问题，订单一直在pending状态。"},
        {"role": "assistant", "content": "您好！我来帮您排查。请问这个问题是从什么时候开始的？"},
        {"role": "user", "content": "大概今天下午3点左右，到现在已经有2个多小时了。"},
        {"role": "assistant", "content": "了解。请问是所有订单都受影响还是部分订单？"},
        {"role": "user", "content": "看起来是所有新订单，已经确认的订单没有影响。"},
        {"role": "assistant", "content": "好的，让我检查一下支付网关的状态。请您稍等..."},
        {"role": "assistant", "content": "我检查了支付网关的日志，发现在14:58开始有大量超时错误，与您描述的时间吻合。"},
        {"role": "user", "content": "那是什么原因导致的呢？是第三方支付平台的问题吗？"},
        {"role": "assistant", "content": "从日志来看，是数据库连接池耗尽导致的。在14:55左右有一个批量结算任务启动，占用了大量连接。"},
        {"role": "user", "content": "那现在怎么处理？我们有一批订单急需处理。"},
        {"role": "assistant", "content": "我已经临时扩大了连接池上限，现在新订单应该可以正常处理了。您可以试一下吗？"},
        {"role": "user", "content": "好的，我试一下... 嗯，现在可以了！新订单状态变为processing了。"},
        {"role": "assistant", "content": "很好！不过这个只是临时方案。根本原因需要优化那个批量结算任务，建议给它配置独立的数据库连接池。"},
        {"role": "user", "content": "明白了，我们会安排开发团队跟进。之前pending的那些订单怎么办？"},
        {"role": "assistant", "content": "pending的订单我已经手动触发了重试，应该会在接下来的几分钟内自动处理。"},
        {"role": "user", "content": "好的，我看到部分订单已经开始处理了。谢谢你的帮助！"},
        {"role": "assistant", "content": "不客气！我会将这个事件记录到知识库中，以后遇到类似情况可以直接参考。请问还有别的问题吗？"},
        {"role": "user", "content": "没有了。是否可以发一份今天的处理报告到我的邮箱？"},
        {"role": "assistant", "content": "当然可以。报告已生成并发送到您的邮箱，请查收。"},
        {"role": "user", "content": "收到，谢谢。再见！"},
        {"role": "assistant", "content": "再见！如有问题随时联系。"},
    ]

    # 扩展到所需轮数
    result = conversation[:turns]
    return result


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示摘要压缩器"""
    print("=" * 70)
    print("  LLM摘要压缩器 (Summarization Compressor) 演示")
    print("=" * 70)

    # 生成20轮对话
    print("\n【步骤1】生成20轮模拟对话...")
    conversation = generate_simulated_conversation(20)
    total_chars = sum(len(m.get("content", "")) for m in conversation)
    print(f"  对话轮数: {len(conversation)}")
    print(f"  原始总字符数: {total_chars}")

    compressor = LLMSummarizationCompressor(max_segment_size=5)

    # 测试三种风格
    for style in SummaryStyle:
        print(f"\n{'='*50}")
        print(f"  【{style.name} 摘要】")
        print(f"{'='*50}")

        summary = compressor.compress(conversation, style=style)
        validation = compressor._validate_summary(conversation, summary)

        print(f"  摘要长度: {len(summary)} 字符")
        print(f"  压缩比: {validation['compression_ratio']:.2%}")
        print(f"  事实保留率: {validation['fact_retention_estimate']:.2%}")
        print(f"  摘要内容:")
        # 截断显示
        display = summary[:300] + "..." if len(summary) > 300 else summary
        for line in display.split("\n")[:10]:
            print(f"    {line}")

    # 增量摘要演示
    print(f"\n{'='*50}")
    print(f"  【增量摘要演示】")
    print(f"{'='*50}")

    prev = ""
    for i in range(0, len(conversation), 5):
        chunk = conversation[i:i + 5]
        chunk_text = " ".join(m.get("content", "") for m in chunk)
        prev = compressor.incremental_summarize(prev, chunk_text)
        print(f"  第{i//5+1}轮增量后摘要: {prev[:100]}...")

    # 递归摘要演示
    print(f"\n{'='*50}")
    print(f"  【递归摘要演示】")
    print(f"{'='*50}")

    segments = [
        "用户报告支付系统订单pending问题",
        "技术支持检查发现问题从下午3点开始",
        "排查发现数据库连接池被批量任务耗尽",
        "临时扩大连接池并手动重试pending订单",
        "生成处理报告发送给用户",
    ]
    final = compressor.recursive_summarize(segments)
    print(f"  输入段数: {len(segments)}")
    print(f"  递归摘要结果: {final}")

    # 统计
    print(f"\n{'='*50}")
    print(f"  【压缩器统计】")
    print(f"{'='*50}")
    import json
    print(json.dumps(compressor.get_stats(), indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
