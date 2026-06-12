#!/usr/bin/env python3
"""
Phase 04 Module 10: Dynamic Top-K Selection
==============================================
- Query complexity scoring (word count, entities, conditions, logical operators)
- Map to Top-K: simple 3-5, complex 8-12
- "Cliff" detection: find steepest drop in similarity scores, cut there
- Min/max K guards: min_k=3, max_k=10
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# Enums and Data Classes
# ===========================================================================

class QueryComplexity(Enum):
    SIMPLE = "simple"           # Short factoid: 3 results
    MODERATE = "moderate"       # Standard question: 5 results
    COMPLEX = "complex"         # Multi-aspect: 8-10 results
    VERY_COMPLEX = "very_complex"  # Multi-hop reasoning: 10-12 results


@dataclass
class ComplexityScore:
    """Breakdown of query complexity scoring factors."""
    word_count: int = 0
    entity_count: int = 0
    condition_count: int = 0
    logical_operator_count: int = 0
    intent_count: int = 0          # Number of distinct intents
    total_score: float = 0.0
    complexity: QueryComplexity = QueryComplexity.SIMPLE


@dataclass
class DynamicTopKResult:
    """Result of dynamic top-K calculation."""
    recommended_k: int
    complexity: QueryComplexity
    complexity_score: ComplexityScore
    method: str = "scoring"  # "scoring" or "cliff" or "hybrid"
    cliff_position: Optional[int] = None
    cliff_score_drop: Optional[float] = None


# ===========================================================================
# Query Complexity Scorer
# ===========================================================================

class QueryComplexityScorer:
    """Score query complexity based on multiple dimensions."""

    # Entity patterns
    ENTITY_PATTERNS = [
        re.compile(r'\b[A-Z]{2,6}\b'),                           # Acronyms
        re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'),         # CamelCase
        re.compile(r'[一-鿿]{2,6}(?:模型|系统|架构|算法|框架|网络)'),  # Chinese tech terms
        re.compile(r'"([^"]+)"'),                                # Quoted terms
        re.compile(r'[「「]([^」」]+)[」」]'),                       # Chinese quoted
    ]

    # Condition/constraint patterns
    CONDITION_PATTERNS = [
        re.compile(r'(?:当|如果|若|在.*情况下|在.*条件下|when|if|given\s+that|'
                     r'under\s+the\s+condition)', re.IGNORECASE),
        re.compile(r'(?:大于|小于|等于|不超过|至少|最多|between|greater|less|'
                     r'at\s+least|at\s+most|>=|<=|>|<)', re.IGNORECASE),
        re.compile(r'(?:要求|需要|必须|Require|Must|Should|Need\s+to)',
                    re.IGNORECASE),
        re.compile(r'(?:限制|约束|Constraint|Limitation|Restriction)',
                    re.IGNORECASE),
    ]

    # Logical operators
    LOGICAL_PATTERNS = [
        re.compile(r'(?:和|与|及|以及|并且|而且|同时|and|also|in\s+addition)',
                     re.IGNORECASE),
        re.compile(r'(?:或者|或|还是|or|either)', re.IGNORECASE),
        re.compile(r'(?:不是|并非|除了|not|except|excluding)', re.IGNORECASE),
        re.compile(r'(?:但是|但|然而|不过|but|however|although)',
                     re.IGNORECASE),
    ]

    # Intent diversity patterns
    INTENT_PATTERNS = [
        (re.compile(r'(?:什么是|定义|概念|What\s+is|Define)'), 'definition'),
        (re.compile(r'(?:如何|怎么|步骤|How\s+to)'), 'process'),
        (re.compile(r'(?:区别|对比|比较|vs\.?|Compare)'), 'comparison'),
        (re.compile(r'(?:为什么|原因|Why|Cause)'), 'reasoning'),
        (re.compile(r'(?:案例|示例|Example|Case)'), 'example'),
    ]

    @classmethod
    def score(cls, query: str) -> ComplexityScore:
        """Compute complexity score for a query."""
        result = ComplexityScore()

        # 1. Word count (normalized)
        words = query.split()
        chinese_chars = len(re.findall(r'[一-鿿]', query))
        result.word_count = len(words) + chinese_chars  # Mixed word/chars

        # 2. Entity count
        entities: set = set()
        for pattern in cls.ENTITY_PATTERNS:
            for match in pattern.findall(query):
                if isinstance(match, tuple):
                    entities.update(m for m in match if m)
                else:
                    entities.add(match)
        result.entity_count = len(entities)

        # 3. Condition count
        conditions = 0
        for pattern in cls.CONDITION_PATTERNS:
            conditions += len(pattern.findall(query))
        result.condition_count = conditions

        # 4. Logical operator count
        logical_ops = 0
        for pattern in cls.LOGICAL_PATTERNS:
            logical_ops += len(pattern.findall(query))
        result.logical_operator_count = logical_ops

        # 5. Intent diversity
        intents: set = set()
        for pattern, intent_name in cls.INTENT_PATTERNS:
            if pattern.search(query):
                intents.add(intent_name)
        result.intent_count = len(intents)

        # ---- Weighted scoring ----
        score = 0.0
        # Word count: 0-3 points (log scale)
        score += min(math.log(max(result.word_count, 1)) / math.log(10), 3.0)
        # Entities: 0-2 points
        score += min(result.entity_count * 0.5, 2.0)
        # Conditions: 0-3 points
        score += min(result.condition_count * 1.5, 3.0)
        # Logical ops: 0-2 points
        score += min(result.logical_operator_count * 0.5, 2.0)
        # Intent diversity: 0-2 points
        score += min(result.intent_count * 1.0, 2.0)

        result.total_score = round(score, 2)

        # Classify complexity
        if score <= 2.0:
            result.complexity = QueryComplexity.SIMPLE
        elif score <= 4.0:
            result.complexity = QueryComplexity.MODERATE
        elif score <= 7.0:
            result.complexity = QueryComplexity.COMPLEX
        else:
            result.complexity = QueryComplexity.VERY_COMPLEX

        return result


# ===========================================================================
# Top-K Mapping
# ===========================================================================

class TopKMapper:
    """Map query complexity to recommended Top-K value."""

    # Default mapping: complexity → (min_k, default_k, max_k)
    DEFAULT_MAPPING: Dict[QueryComplexity, Tuple[int, int, int]] = {
        QueryComplexity.SIMPLE:       (3, 3, 5),
        QueryComplexity.MODERATE:     (4, 5, 7),
        QueryComplexity.COMPLEX:      (6, 8, 10),
        QueryComplexity.VERY_COMPLEX: (8, 10, 12),
    }

    def __init__(
        self,
        mapping: Optional[Dict[QueryComplexity, Tuple[int, int, int]]] = None,
        min_k: int = 3,
        max_k: int = 10,
    ):
        self.mapping = mapping or self.DEFAULT_MAPPING
        self.min_k = min_k
        self.max_k = max_k

    def map(self, complexity: QueryComplexity,
            complexity_score: Optional[ComplexityScore] = None) -> int:
        """Map complexity to a recommended K value."""
        min_k, default_k, cmax_k = self.mapping.get(
            complexity, (3, 5, 10)
        )

        # Clamp to global min/max
        k = max(self.min_k, min(default_k, self.max_k))

        # Fine-tune based on score within complexity range
        if complexity_score and complexity_score.total_score > 0:
            # Linear interpolation within the complexity band
            if complexity == QueryComplexity.SIMPLE:
                range_size = 2.0  # score 0-2
                offset = min(complexity_score.total_score, range_size) / range_size
                k = int(3 + offset * 2)  # 3-5
            elif complexity == QueryComplexity.MODERATE:
                range_size = 2.0  # score 2-4
                offset = max(0, complexity_score.total_score - 2.0) / range_size
                k = int(5 + offset * 2)  # 5-7
            elif complexity == QueryComplexity.COMPLEX:
                range_size = 3.0  # score 4-7
                offset = max(0, complexity_score.total_score - 4.0) / range_size
                k = int(7 + offset * 3)  # 7-10
            else:  # VERY_COMPLEX
                k = int(10 + min(complexity_score.total_score - 7.0, 3.0))  # 10-12

        return max(self.min_k, min(k, self.max_k))


# ===========================================================================
# Cliff Detector
# ===========================================================================

class CliffDetector:
    """Detect the 'cliff' — the steepest drop in similarity scores.

    Instead of using a fixed K, find the natural cutoff point where
    relevance drops sharply.
    """

    @staticmethod
    def detect(scores: List[float], min_k: int = 3,
               max_k: int = 10) -> Tuple[int, Optional[int], Optional[float]]:
        """
        Args:
            scores: List of similarity/relevance scores (descending).
            min_k: Minimum number of results to return.
            max_k: Maximum number of results to consider.

        Returns:
            (recommended_k, cliff_position, cliff_drop_magnitude)
        """
        if not scores:
            return min_k, None, None

        n = min(len(scores), max_k)

        if n <= min_k:
            return n, None, None

        # Compute score drops between consecutive positions
        drops: List[float] = []
        for i in range(1, n):
            drop = scores[i - 1] - scores[i]
            drops.append(drop)

        if not drops:
            return min_k, None, None

        # Find the maximum drop (the "cliff")
        max_drop_idx = int(np.argmax(drops))
        max_drop_val = drops[max_drop_idx]

        # The cliff position is the index AFTER the drop
        cliff_pos = max_drop_idx + 1  # +1 because drops are between i-1 and i

        # Only cut at the cliff if:
        # 1. The drop is significant (top 30% of all drops)
        # 2. The cliff position is at least min_k
        if len(drops) >= 2:
            drop_threshold = np.percentile(drops, 70)  # Top 30%
            is_significant = max_drop_val >= drop_threshold
        else:
            is_significant = max_drop_val > 0.1

        if is_significant and cliff_pos >= min_k:
            recommended_k = min(cliff_pos, max_k)
        else:
            # No significant cliff found, use default K based on score distribution
            # Use the position where cumulative score reaches 90% of total
            total_score = sum(scores[:n])
            if total_score > 0:
                cumsum = np.cumsum(scores[:n])
                threshold_90 = 0.90 * total_score
                for i, cs in enumerate(cumsum):
                    if cs >= threshold_90:
                        recommended_k = max(min_k, min(i + 1, max_k))
                        break
                else:
                    recommended_k = min_k
            else:
                recommended_k = min_k

        return recommended_k, cliff_pos, max_drop_val

    @staticmethod
    def visualize(scores: List[float], cliff_pos: Optional[int] = None) -> str:
        """Generate an ASCII visualization of the score cliff."""
        if not scores:
            return ""

        n = min(len(scores), 15)
        max_score = max(scores[:n]) if scores else 1.0
        bar_width = 40

        lines: List[str] = ["Rank | Score    | Distribution", "-" * 55]

        for i in range(n):
            bar_len = int(scores[i] / max(max_score, 1e-8) * bar_width) if max_score > 0 else 0
            bar = '█' * bar_len
            marker = ' <-- CLIFF' if cliff_pos and i == cliff_pos else ''
            lines.append(f"  {i+1:2d} | {scores[i]:.4f}  | {bar}{marker}")

        return '\n'.join(lines)


# ===========================================================================
# Dynamic Top-K Calculator
# ===========================================================================

class DynamicTopK:
    """Complete dynamic Top-K calculation combining complexity scoring
    and cliff detection."""

    def __init__(
        self,
        min_k: int = 3,
        max_k: int = 10,
        method: str = "hybrid",  # "scoring", "cliff", or "hybrid"
        cliff_threshold: float = 0.05,
    ):
        """
        Args:
            min_k: Absolute minimum K.
            max_k: Absolute maximum K.
            method: Which method to use.
            cliff_threshold: Minimum score drop to consider a cliff.
        """
        self.min_k = max(1, min_k)
        self.max_k = max(self.min_k, max_k)
        self.method = method
        self.cliff_threshold = cliff_threshold

        self.scorer = QueryComplexityScorer()
        self.mapper = TopKMapper(min_k=self.min_k, max_k=self.max_k)
        self.detector = CliffDetector()

    def calculate(
        self,
        query: str,
        scores: Optional[List[float]] = None,
    ) -> DynamicTopKResult:
        """Calculate the optimal Top-K for a query.

        Args:
            query: The search query string.
            scores: Retrieved similarity scores (if already available).

        Returns:
            DynamicTopKResult with recommended K.
        """
        # Step 1: Score query complexity
        complexity_score = self.scorer.score(query)
        complexity = complexity_score.complexity

        # Step 2: Map complexity to K (scoring method)
        scoring_k = self.mapper.map(complexity, complexity_score)

        if self.method == "scoring":
            return DynamicTopKResult(
                recommended_k=scoring_k,
                complexity=complexity,
                complexity_score=complexity_score,
                method="scoring",
            )

        if self.method == "cliff" and scores:
            cliff_k, cliff_pos, cliff_drop = self.detector.detect(
                scores, self.min_k, self.max_k
            )
            return DynamicTopKResult(
                recommended_k=cliff_k,
                complexity=complexity,
                complexity_score=complexity_score,
                method="cliff",
                cliff_position=cliff_pos,
                cliff_score_drop=cliff_drop,
            )

        if self.method == "hybrid":
            if scores:
                cliff_k, cliff_pos, cliff_drop = self.detector.detect(
                    scores, self.min_k, self.max_k
                )

                # Hybrid: blend scoring-based and cliff-based K
                # If cliff is clear, lean toward cliff; otherwise lean toward scoring
                if cliff_drop and cliff_drop > self.cliff_threshold:
                    # Significant cliff: weight cliff more heavily
                    hybrid_k = int(0.7 * cliff_k + 0.3 * scoring_k)
                else:
                    # No clear cliff: weight scoring more heavily
                    hybrid_k = int(0.3 * (cliff_k if scores else scoring_k) + 0.7 * scoring_k)

                return DynamicTopKResult(
                    recommended_k=max(self.min_k, min(hybrid_k, self.max_k)),
                    complexity=complexity,
                    complexity_score=complexity_score,
                    method="hybrid",
                    cliff_position=cliff_pos,
                    cliff_score_drop=cliff_drop,
                )
            else:
                # No scores available, use scoring only
                return DynamicTopKResult(
                    recommended_k=scoring_k,
                    complexity=complexity,
                    complexity_score=complexity_score,
                    method="scoring",
                )

        # Default: scoring method
        return DynamicTopKResult(
            recommended_k=scoring_k,
            complexity=complexity,
            complexity_score=complexity_score,
            method="scoring",
        )

    def calculate_from_scores(
        self,
        scores: List[float],
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
    ) -> int:
        """Calculate optimal K purely from score distribution (no query needed).

        Uses cliff detection + cumulative score threshold.
        """
        _min_k = min_k or self.min_k
        _max_k = max_k or self.max_k

        k, cliff_pos, drop = self.detector.detect(scores, _min_k, _max_k)
        return k


# ===========================================================================
# __main__: Demo
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Dynamic Top-K Selection - Self-Test")
    print("=" * 60)

    # --- Test queries with varying complexity ---
    test_cases = [
        ("什么是AI？", None),
        ("如何训练一个Transformer模型？", None),
        ("BERT和GPT在架构、训练目标、适用场景方面有什么区别？", None),
        ("为什么模型部署到生产环境后推理延迟显著增加，应该如何诊断问题"
         "并优化推理性能，同时确保不降低模型精度？", None),
        ("什么是注意力机制以及它如何提升NLP模型性能，"
         "另外在实际部署中需要注意哪些工程问题？", None),
    ]

    # Simulated score distributions for each test case
    simulated_scores = [
        [0.95, 0.60, 0.45, 0.40, 0.35, 0.30, 0.28, 0.25, 0.22, 0.20],
        [0.92, 0.88, 0.82, 0.75, 0.50, 0.42, 0.38, 0.35, 0.30, 0.28],
        [0.90, 0.85, 0.80, 0.78, 0.75, 0.55, 0.48, 0.42, 0.38, 0.35],
        [0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.65, 0.60, 0.55, 0.50],
        [0.95, 0.92, 0.88, 0.60, 0.45, 0.42, 0.40, 0.38, 0.35, 0.30],
    ]

    for (query, _), scores in zip(test_cases, simulated_scores):
        print(f"\n{'─'*60}")
        print(f"Query: {query[:80]}...")

        # --- Method comparison ---
        for method in ["scoring", "cliff", "hybrid"]:
            dtk = DynamicTopK(min_k=3, max_k=10, method=method)
            result = dtk.calculate(query, scores)

            print(f"\n  Method: {result.method}")
            print(f"    Complexity:    {result.complexity.value} "
                  f"(score={result.complexity_score.total_score})")
            print(f"    Word count:    {result.complexity_score.word_count}")
            print(f"    Entities:      {result.complexity_score.entity_count}")
            print(f"    Conditions:    {result.complexity_score.condition_count}")
            print(f"    Logical ops:   {result.complexity_score.logical_operator_count}")
            print(f"    Intents:       {result.complexity_score.intent_count}")
            print(f"    Recommended K: {result.recommended_k}")
            if result.cliff_position:
                print(f"    Cliff at:      position {result.cliff_position} "
                      f"(drop={result.cliff_score_drop:.4f})")

        # --- Show cliff visualization ---
        print(f"\n  Score Distribution:")
        print(CliffDetector.visualize(scores, cliff_pos=None))

    # --- Edge case tests ---
    print(f"\n{'='*60}")
    print("Edge Case Tests")
    print(f"{'='*60}")

    edge_cases = [
        ("Empty scores", [], 3),
        ("Single result", [0.95], 1),
        ("Flat scores (no cliff)", [0.5, 0.49, 0.48, 0.47, 0.46, 0.45], 3),
        ("Sharp cliff", [0.95, 0.92, 0.90, 0.30, 0.25, 0.20], 3),
        ("Gradual decline", [0.9, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60], 3),
        ("Very long list", [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60,
                           0.55, 0.50, 0.48, 0.45, 0.42, 0.40, 0.38], 3),
    ]

    for label, scores, expected_min in edge_cases:
        dtk = DynamicTopK(method="hybrid")
        k = dtk.calculate_from_scores(scores, min_k=3, max_k=10)
        print(f"\n  {label}:")
        print(f"    Scores: {scores[:8]}...")
        print(f"    Recommended K: {k}")
        if scores:
            print(CliffDetector.visualize(scores, cliff_pos=k))

    # --- Complexity score breakdown for a single query ---
    print(f"\n{'='*60}")
    print("Detailed Complexity Score Breakdown")
    print(f"{'='*60}")

    detailed_query = ("当模型在GPU服务器上部署后推理延迟超过100ms时，应该如何诊断性能瓶颈，"
                      "同时确保模型精度不低于95%，并且内存占用不超过16GB？")
    score = QueryComplexityScorer.score(detailed_query)
    print(f"Query: {detailed_query}")
    print(f"\n  Complexity: {score.complexity.value}")
    print(f"  Total Score: {score.total_score}")
    print(f"  Word Count: {score.word_count}")
    print(f"  Entities: {score.entity_count}")
    print(f"  Conditions: {score.condition_count}")
    print(f"  Logical Operators: {score.logical_operator_count}")
    print(f"  Distinct Intents: {score.intent_count}")

    print(f"\n{'='*60}")
    print("Dynamic Top-K self-test completed!")
    print(f"{'='*60}")
