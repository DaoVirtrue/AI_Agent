#!/usr/bin/env python3
"""
Phase 04 Module 13: Lost-in-Middle Ranker
============================================
- Detect content position within original document
- Reprioritization: items from middle positions promoted
- LongContextReorder implementation (LangChain compatible)
- Before/after comparison: retrieval quality with and without position-aware reordering
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# Data Classes
# ===========================================================================

class PositionZone(Enum):
    BEGINNING = "beginning"   # First 20%
    EARLY_MIDDLE = "early_middle"  # 20-40%
    MIDDLE = "middle"         # 40-70% — The "lost" zone
    LATE_MIDDLE = "late_middle"  # 70-90%
    END = "end"               # Last 10%


@dataclass
class PositionInfo:
    """Position information for a chunk within its source document."""
    chunk_id: str
    doc_id: str
    position_ratio: float  # 0.0 = start, 1.0 = end
    position_zone: PositionZone
    total_chunks: int = 1
    chunk_index: int = 0


@dataclass
class ScoredChunk:
    """A chunk with position and relevance scores."""
    chunk_id: str
    text: str
    relevance_score: float
    position_info: PositionInfo
    adjusted_score: float = 0.0
    original_rank: int = 0
    adjusted_rank: int = 0


# ===========================================================================
# Position Detector
# ===========================================================================

class PositionDetector:
    """Detect content position within the original document."""

    @staticmethod
    def detect(
        chunk_id: str,
        doc_id: str,
        chunk_index: int,
        total_chunks: int,
    ) -> PositionInfo:
        """Calculate position information for a chunk.

        Args:
            chunk_index: 0-based index of this chunk in the document.
            total_chunks: Total number of chunks in the document.

        Returns:
            PositionInfo with position_ratio and zone classification.
        """
        if total_chunks <= 1:
            position_ratio = 0.0
            zone = PositionZone.BEGINNING
        else:
            position_ratio = chunk_index / (total_chunks - 1)
            zone = PositionDetector._classify_zone(position_ratio)

        return PositionInfo(
            chunk_id=chunk_id,
            doc_id=doc_id,
            position_ratio=round(position_ratio, 3),
            position_zone=zone,
            total_chunks=total_chunks,
            chunk_index=chunk_index,
        )

    @staticmethod
    def _classify_zone(ratio: float) -> PositionZone:
        """Classify a position ratio into a zone."""
        if ratio <= 0.20:
            return PositionZone.BEGINNING
        elif ratio <= 0.40:
            return PositionZone.EARLY_MIDDLE
        elif ratio <= 0.70:
            return PositionZone.MIDDLE
        elif ratio <= 0.90:
            return PositionZone.LATE_MIDDLE
        else:
            return PositionZone.END


# ===========================================================================
# Lost-in-Middle Position Adjuster
# ===========================================================================

class LostInMiddleAdjuster:
    """Adjust relevance scores based on the 'lost-in-middle' phenomenon.

    Research shows that LLMs pay more attention to content at the beginning
    and end of the context window. Content in the middle is often overlooked.

    This adjuster:
    1. Identifies chunks from the middle of documents
    2. Applies a promotion bonus to middle-positioned chunks
    3. Applies a slight demotion to beginning/end chunks (already favored)
    4. Ensures balanced representation across document positions
    """

    def __init__(
        self,
        middle_promotion_factor: float = 0.15,   # Bonus multiplier for middle chunks
        diversity_penalty_factor: float = 0.05,   # Penalty for over-represented zones
        max_promotion: float = 0.20,             # Max score increase
        enable_diversity: bool = True,
    ):
        """
        Args:
            middle_promotion_factor: How much to promote middle-positioned chunks.
            diversity_penalty_factor: Penalty for zone over-representation.
            max_promotion: Maximum score increase for any chunk.
            enable_diversity: Whether to enforce zone diversity.
        """
        self.middle_promotion_factor = middle_promotion_factor
        self.diversity_penalty_factor = diversity_penalty_factor
        self.max_promotion = max_promotion
        self.enable_diversity = enable_diversity

    def adjust(
        self,
        chunks: List[ScoredChunk],
    ) -> List[ScoredChunk]:
        """Apply position-aware score adjustments.

        Strategy:
        - MIDDLE zone chunks: +promotion (they're often overlooked)
        - EARLY_MIDDLE / LATE_MIDDLE: slight promotion
        - BEGINNING / END: neutral (already favored by LLM attention)
        - Apply diversity penalties if one zone is over-represented
        """
        if not chunks:
            return chunks

        n = len(chunks)
        max_score = max(c.score for c in chunks) if chunks else 1.0

        # Step 1: Apply position-based promotion
        for chunk in chunks:
            zone = chunk.position_info.position_zone
            promotion = self._get_promotion(zone)

            # Scale promotion by max_score
            chunk.adjusted_score = chunk.relevance_score + promotion * max_score

            # Clamp: don't exceed 1.0, don't go below 0
            chunk.adjusted_score = max(0.0, min(1.0, chunk.adjusted_score))

        # Step 2: Apply diversity adjustments
        if self.enable_diversity and n > 3:
            self._apply_diversity(chunks)

        # Step 3: Re-rank
        chunks.sort(key=lambda c: c.adjusted_score, reverse=True)

        for rank, chunk in enumerate(chunks, 1):
            chunk.adjusted_rank = rank

        return chunks

    def _get_promotion(self, zone: PositionZone) -> float:
        """Get the promotion factor for a position zone."""
        promotion_map = {
            PositionZone.MIDDLE: self.middle_promotion_factor,
            PositionZone.EARLY_MIDDLE: self.middle_promotion_factor * 0.6,
            PositionZone.LATE_MIDDLE: self.middle_promotion_factor * 0.6,
            PositionZone.BEGINNING: 0.0,
            PositionZone.END: 0.0,
        }
        return promotion_map.get(zone, 0.0)

    def _apply_diversity(self, chunks: List[ScoredChunk]) -> None:
        """Apply diversity penalty to over-represented zones.

        Ensures results aren't all from one document position.
        """
        # Count chunks per zone
        zone_counts: Dict[PositionZone, int] = defaultdict(int)
        for chunk in chunks:
            zone_counts[chunk.position_info.position_zone] += 1

        n = len(chunks)
        n_zones = len(PositionZone)
        expected_per_zone = n / n_zones

        # Penalize zones that exceed expected count
        for chunk in chunks:
            zone = chunk.position_info.position_zone
            count = zone_counts[zone]
            if count > expected_per_zone:
                overage_ratio = (count - expected_per_zone) / max(count, 1)
                penalty = overage_ratio * self.diversity_penalty_factor
                chunk.adjusted_score -= penalty
                chunk.adjusted_score = max(0.0, chunk.adjusted_score)


# ===========================================================================
# LongContextReorder (LangChain compatible)
# ===========================================================================

class LongContextReorder:
    """Reorder retrieved documents to mitigate the 'lost in the middle' effect.

    Compatible with LangChain's LongContextReorder pattern.

    Strategy: Place the most relevant document at the beginning,
    then alternate between less relevant ones. This ensures
    important information is seen first while maintaining variety.
    """

    @staticmethod
    def reorder(
        documents: List[Dict[str, Any]],
        score_key: str = 'score',
    ) -> List[Dict[str, Any]]:
        """Reorder documents to optimize for LLM attention distribution.

        Reordering pattern:
        - Position 0: Most relevant
        - Position 1: Least relevant (among all)
        - Position 2: 2nd most relevant
        - Position 3: 2nd least relevant
        - ...
        This creates a "U-shaped" relevance curve where the LLM
        can attend to both high and low relevance items alternately.
        """
        if not documents or len(documents) <= 2:
            return documents

        # Sort by score descending
        sorted_docs = sorted(documents, key=lambda d: d.get(score_key, 0.0),
                            reverse=True)
        n = len(sorted_docs)

        reordered: List[Dict[str, Any]] = []
        left, right = 0, n - 1
        take_from_left = True

        while left <= right:
            if take_from_left:
                reordered.append(sorted_docs[left])
                left += 1
            else:
                reordered.append(sorted_docs[right])
                right -= 1
            take_from_left = not take_from_left

        return reordered

    @staticmethod
    def reorder_with_positions(
        documents: List[Dict[str, Any]],
        score_key: str = 'score',
        position_key: str = 'position_ratio',
    ) -> List[Dict[str, Any]]:
        """Enhanced reorder that also considers document position.

        First promotes middle-position content, then applies
        the U-shaped relevance reordering.
        """
        if not documents:
            return documents

        n = len(documents)

        # Promote middle-position content
        for doc in documents:
            pos = doc.get(position_key, 0.5)
            # U-shaped attention curve: beginning and end get natural attention
            # Middle positions need promotion
            middle_distance = abs(pos - 0.5)  # How far from exact middle
            promotion = (1.0 - 2.0 * middle_distance) * 0.15  # Max 0.15 at middle
            original_score = doc.get(score_key, 0.0)
            doc['adjusted_score'] = original_score + promotion
            doc['position_promotion'] = promotion

        # Sort by adjusted score for the initial relevance ordering
        sorted_docs = sorted(documents, key=lambda d: d.get('adjusted_score',
                            d.get(score_key, 0.0)), reverse=True)

        # Apply U-shaped reordering
        return LongContextReorder.reorder(sorted_docs, score_key='adjusted_score')


# ===========================================================================
# Complete Lost-in-Middle Pipeline
# ===========================================================================

class LostInMiddlePipeline:
    """End-to-end lost-in-middle mitigation pipeline.

    Workflow:
    1. Detect position of each retrieved chunk
    2. Apply position-based score adjustments
    3. Apply LongContextReorder for optimal LLM context
    4. Output re-ranked results
    """

    def __init__(
        self,
        middle_promotion_factor: float = 0.15,
        enable_diversity: bool = True,
        enable_long_context_reorder: bool = True,
    ):
        self.adjuster = LostInMiddleAdjuster(
            middle_promotion_factor=middle_promotion_factor,
            enable_diversity=enable_diversity,
        )
        self.enable_long_context_reorder = enable_long_context_reorder

    def process(
        self,
        query: str,
        retrieved_chunks: List[Tuple[str, str, float, int, int, str]],
        top_k: int = 10,
    ) -> Tuple[List[ScoredChunk], Dict[str, Any]]:
        """Process retrieved chunks through the pipeline.

        Args:
            query: The search query (for logging).
            retrieved_chunks: List of (chunk_id, text, score,
                              chunk_index, total_chunks, doc_id).
            top_k: Number of results to return.

        Returns:
            (adjusted_chunks, stats_dict)
        """
        # Step 1: Create ScoredChunk objects with position info
        scored: List[ScoredChunk] = []
        for rank, (cid, text, score, chunk_idx, total, doc_id) in enumerate(
            retrieved_chunks, 1
        ):
            pos_info = PositionDetector.detect(cid, doc_id, chunk_idx, total)
            scored.append(ScoredChunk(
                chunk_id=cid,
                text=text,
                relevance_score=score,
                position_info=pos_info,
                original_rank=rank,
            ))

        # Step 2: Apply position-aware adjustments
        adjusted = self.adjuster.adjust(scored)

        # Step 3: Apply LongContextReorder
        if self.enable_long_context_reorder and len(adjusted) > 2:
            doc_dicts = [
                {
                    'chunk_id': c.chunk_id,
                    'text': c.text,
                    'score': c.adjusted_score,
                    'position_ratio': c.position_info.position_ratio,
                    'position_zone': c.position_info.position_zone.value,
                }
                for c in adjusted
            ]
            reordered_dicts = LongContextReorder.reorder_with_positions(
                doc_dicts, score_key='score'
            )
            # Map back — reordering preserves objects
            # Build lookup
            lookup = {c.chunk_id: c for c in adjusted}
            final_chunks: List[ScoredChunk] = []
            for rd in reordered_dicts[:top_k]:
                c = lookup.get(rd['chunk_id'])
                if c:
                    c.adjusted_score = rd.get('adjusted_score', c.adjusted_score)
                    final_chunks.append(c)
            # Re-rank
            for rank, c in enumerate(final_chunks, 1):
                c.adjusted_rank = rank
            adjusted = final_chunks
        else:
            adjusted = adjusted[:top_k]

        # Stats
        stats = self._compute_stats(adjusted, retrieved_chunks)

        return adjusted, stats

    def _compute_stats(
        self,
        adjusted: List[ScoredChunk],
        original: List,
    ) -> Dict[str, Any]:
        """Compute statistics about the adjustment."""
        if not adjusted:
            return {}

        zone_counts = defaultdict(int)
        for c in adjusted:
            zone_counts[c.position_info.position_zone.value] += 1

        promotions = sum(
            1 for c in adjusted
            if c.position_info.position_zone in (PositionZone.MIDDLE,
                                                  PositionZone.EARLY_MIDDLE,
                                                  PositionZone.LATE_MIDDLE)
        )

        return {
            'total_chunks': len(adjusted),
            'zone_distribution': dict(zone_counts),
            'middle_promoted': promotions,
            'promotion_ratio': promotions / len(adjusted) if adjusted else 0.0,
            'avg_adjustment': (
                sum(c.adjusted_score - c.relevance_score for c in adjusted) / len(adjusted)
                if adjusted else 0.0
            ),
        }


# ===========================================================================
# Comparison tools
# ===========================================================================

def compare_before_after(
    before: List[ScoredChunk],
    after: List[ScoredChunk],
) -> Dict[str, Any]:
    """Compare retrieval results before and after lost-in-middle adjustment.

    Returns:
        Comparison metrics dict.
    """
    before_ids = [c.chunk_id for c in before]
    after_ids = [c.chunk_id for c in after]

    # Zone distribution
    before_zones = defaultdict(int)
    after_zones = defaultdict(int)
    for c in before:
        before_zones[c.position_info.position_zone.value] += 1
    for c in after:
        after_zones[c.position_info.position_zone.value] += 1

    # How many middle-position chunks moved up
    middle_before = {c.chunk_id for c in before
                     if c.position_info.position_zone in
                     (PositionZone.MIDDLE, PositionZone.EARLY_MIDDLE, PositionZone.LATE_MIDDLE)}
    middle_after = {c.chunk_id for c in after
                    if c.position_info.position_zone in
                    (PositionZone.MIDDLE, PositionZone.EARLY_MIDDLE, PositionZone.LATE_MIDDLE)}

    # Check if more middle chunks are in top positions
    top_n = min(5, len(after))
    middle_in_top_after = sum(1 for c in after[:top_n]
                              if c.position_info.position_zone in
                              (PositionZone.MIDDLE, PositionZone.EARLY_MIDDLE, PositionZone.LATE_MIDDLE))
    middle_in_top_before = sum(1 for c in before[:top_n]
                               if c.position_info.position_zone in
                               (PositionZone.MIDDLE, PositionZone.EARLY_MIDDLE, PositionZone.LATE_MIDDLE))

    return {
        'zone_distribution_before': dict(before_zones),
        'zone_distribution_after': dict(after_zones),
        'middle_chunks_in_top5_before': middle_in_top_before,
        'middle_chunks_in_top5_after': middle_in_top_after,
        'middle_representation_change': middle_in_top_after - middle_in_top_before,
        'diversity_improvement': len(after_zones) - len(before_zones),
    }


# ===========================================================================
# __main__: Demo
# ===========================================================================

def _create_sample_chunks() -> List[Tuple[str, str, float, int, int, str]]:
    """Create sample retrieved chunks with position info."""
    samples = [
        ("c01", "Transformer架构由编码器和解码器组成，核心是自注意力机制。这是文档开头部分的内容概述。", 0.92, 0, 8, "doc_transformer"),
        ("c02", "自注意力机制的计算公式为Attention(Q,K,V)=softmax(QK^T/√d_k)V。这是文档第二部分。", 0.88, 2, 8, "doc_transformer"),
        ("c03", "多头注意力通过并行计算多个注意力头来增强模型表达能力。这是文档中间偏前的内容。", 0.85, 3, 8, "doc_transformer"),
        ("c04", "在训练过程中需要注意梯度消失和梯度爆炸问题，可以使用层归一化来缓解。文档中间部分。", 0.83, 4, 8, "doc_transformer"),
        ("c05", "位置编码为Transformer提供序列位置信息，包括正弦位置编码和可学习位置编码。文档中后部。", 0.80, 5, 8, "doc_transformer"),
        ("c06", "残差连接是Transformer稳定训练的关键技术，允许梯度直接传播。文档中后部。", 0.78, 6, 8, "doc_transformer"),
        ("c07", "Transformer在机器翻译、文本摘要、问答系统等任务中都取得了优异表现。文档结尾。", 0.75, 7, 8, "doc_transformer"),
        ("c08", "BERT使用Transformer编码器进行双向预训练。这是另一个文档的开头。", 0.72, 0, 5, "doc_bert"),
        ("c09", "预训练任务包括掩码语言模型和下一句预测。文档中间的细节内容被LLM容易忽略。", 0.70, 2, 5, "doc_bert"),
        ("c10", "BERT可以通过微调适应各种下游NLP任务，如分类、序列标注等。文档结尾。", 0.68, 4, 5, "doc_bert"),
        ("c11", "GPT使用自回归方式生成文本，只能看到左边的上下文。这是GPT文档开头。", 0.65, 0, 3, "doc_gpt"),
        ("c12", "GPT-4在多项基准测试中达到了人类水平的表现。GPT文档中间部分。", 0.62, 1, 3, "doc_gpt"),
    ]
    return samples


if __name__ == '__main__':
    print("=" * 60)
    print("Lost-in-Middle Ranker - Self-Test")
    print("=" * 60)

    # --- Load sample data ---
    sample_chunks = _create_sample_chunks()

    # --- Test 1: Position Detection ---
    print(f"\n[Test 1] Position Detection")
    print("-" * 40)

    for cid, text, score, idx, total, did in sample_chunks[:5]:
        pos = PositionDetector.detect(cid, did, idx, total)
        print(f"  {cid}: pos={pos.position_ratio:.2f} zone={pos.position_zone.value} "
              f"(chunk {idx+1}/{total} of {did})")

    # --- Test 2: LongContextReorder ---
    print(f"\n[Test 2] LongContextReorder")
    print("-" * 40)

    test_docs = [
        {'id': 'd1', 'text': 'Most relevant (score 0.95)', 'score': 0.95},
        {'id': 'd2', 'text': 'Relevant (score 0.82)', 'score': 0.82},
        {'id': 'd3', 'text': 'Somewhat relevant (score 0.71)', 'score': 0.71},
        {'id': 'd4', 'text': 'Slightly relevant (score 0.55)', 'score': 0.55},
        {'id': 'd5', 'text': 'Marginally relevant (score 0.43)', 'score': 0.43},
    ]

    reordered = LongContextReorder.reorder(test_docs)
    print("\n  Original order (by score):")
    for d in test_docs:
        print(f"    {d['id']}: {d['score']:.2f} - {d['text']}")
    print("\n  Reordered (U-shaped):")
    for d in reordered:
        print(f"    {d['id']}: {d['score']:.2f} - {d['text']}")

    # --- Test 3: Position-based Score Adjustment ---
    print(f"\n[Test 3] Position-based Score Adjustment")
    print("-" * 40)

    # Create ScoredChunks
    scored_chunks: List[ScoredChunk] = []
    for rank, (cid, text, score, idx, total, did) in enumerate(sample_chunks, 1):
        pos_info = PositionDetector.detect(cid, did, idx, total)
        scored_chunks.append(ScoredChunk(
            chunk_id=cid,
            text=text,
            relevance_score=score,
            position_info=pos_info,
            original_rank=rank,
        ))

    # Apply adjustment (before reorder)
    adjuster = LostInMiddleAdjuster(
        middle_promotion_factor=0.15,
        enable_diversity=True,
    )
    adjusted_before = adjuster.adjust([ScoredChunk(
        chunk_id=c.chunk_id,
        text=c.text,
        relevance_score=c.relevance_score,
        position_info=c.position_info,
        original_rank=c.original_rank,
    ) for c in scored_chunks])

    print(f"\n  {'Chunk':<6} {'Orig Score':<12} {'Adj Score':<12} "
          f"{'Delta':<10} {'Zone':<15} {'Pos':<8} {'Orig Rk':<8} {'Adj Rk':<8}")
    print(f"  {'-'*75}")
    for c in sorted(adjusted_before, key=lambda x: x.adjusted_score, reverse=True):
        delta = c.adjusted_score - c.relevance_score
        print(f"  {c.chunk_id:<6} {c.relevance_score:<12.4f} {c.adjusted_score:<12.4f} "
              f"{delta:+<10.4f} {c.position_info.position_zone.value:<15} "
              f"{c.position_info.position_ratio:<8.2f} {c.original_rank:<8} {c.adjusted_rank:<8}")

    # --- Test 4: Full Pipeline ---
    print(f"\n[Test 4] Full Lost-in-Middle Pipeline")
    print("-" * 40)

    pipeline = LostInMiddlePipeline(
        middle_promotion_factor=0.15,
        enable_diversity=True,
        enable_long_context_reorder=True,
    )

    final_chunks, stats = pipeline.process(
        "Transformer注意力机制",
        sample_chunks,
        top_k=8,
    )

    print(f"\n  Stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print(f"\n  Final Order:")
    print(f"  {'Rank':<6} {'Chunk':<6} {'Adj Score':<12} "
          f"{'Orig Score':<12} {'Zone':<15} {'Text'}")
    print(f"  {'-'*75}")
    for c in final_chunks:
        print(f"  {c.adjusted_rank:<6} {c.chunk_id:<6} {c.adjusted_score:<12.4f} "
              f"{c.relevance_score:<12.4f} {c.position_info.position_zone.value:<15} "
              f"{c.text[:50]}...")

    # --- Test 5: Before/After Comparison ---
    print(f"\n{'='*60}")
    print("Before/After Comparison")
    print(f"{'='*60}")

    # Simulate "before" — original ranking by relevance only
    before_chunks = sorted(scored_chunks, key=lambda c: c.relevance_score, reverse=True)[:8]
    for rank, c in enumerate(before_chunks, 1):
        c.adjusted_rank = rank

    comparison = compare_before_after(before_chunks, final_chunks)

    print(f"\n  Zone Distribution:")
    print(f"    Before: {comparison['zone_distribution_before']}")
    print(f"    After:  {comparison['zone_distribution_after']}")

    print(f"\n  Middle Chunks in Top-5:")
    print(f"    Before: {comparison['middle_chunks_in_top5_before']} (often overlooked)")
    print(f"    After:  {comparison['middle_chunks_in_top5_after']} (promoted)")
    print(f"    Change: {comparison['middle_representation_change']:+d}")

    print(f"\n  Diversity (zone types represented):")
    print(f"    Before: {comparison['diversity_improvement']}")
    print(f"    After:  (see zone distribution above)")

    # --- Test 6: LongContextReorder with position awareness ---
    print(f"\n{'='*60}")
    print("LongContextReorder with Position Awareness")
    print(f"{'='*60}")

    docs_with_pos = [
        {'id': 'd1', 'score': 0.95, 'position_ratio': 0.1},
        {'id': 'd2', 'score': 0.88, 'position_ratio': 0.5},  # Middle!
        {'id': 'd3', 'score': 0.85, 'position_ratio': 0.3},
        {'id': 'd4', 'score': 0.72, 'position_ratio': 0.6},  # Middle!
        {'id': 'd5', 'score': 0.68, 'position_ratio': 0.9},
    ]

    reordered_pos = LongContextReorder.reorder_with_positions(docs_with_pos)
    print("\n  After position-aware reordering:")
    for d in reordered_pos:
        pos_note = " [MIDDLE PROMOTED]" if d.get('position_promotion', 0) > 0.05 else ""
        print(f"    {d['id']}: score={d.get('score', 0):.3f} "
              f"(orig={d.get('relevance_score', d.get('score', 0)):.3f}) "
              f"pos={d['position_ratio']:.2f}{pos_note}")

    print(f"\n{'='*60}")
    print("Lost-in-Middle Ranker self-test completed!")
    print(f"{'='*60}")
