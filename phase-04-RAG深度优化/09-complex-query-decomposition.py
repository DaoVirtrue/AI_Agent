#!/usr/bin/env python3
"""
Phase 04 Module 09: Complex Query Decomposition
==================================================
- Detect complex queries (multi-condition, multi-step reasoning)
- LLM-based decomposition into independent sub-questions
- Parallel retrieval for each sub-question
- Result fusion: deduplicate, merge by topic, relevance sort
- Comparison of single-query vs decomposed retrieval quality
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ===========================================================================
# Data Classes
# ===========================================================================

class QueryComplexity(Enum):
    SIMPLE = "simple"           # Single fact/question
    MODERATE = "moderate"       # Two aspects
    COMPLEX = "complex"         # Multi-condition or multi-step
    VERY_COMPLEX = "very_complex"  # Nested, multi-step reasoning


@dataclass
class SubQuestion:
    """A decomposed sub-question."""
    id: str
    text: str
    parent_query: str = ""
    aspect: str = ""  # e.g., "definition", "comparison", "process"
    dependency: Optional[str] = None  # ID of dependent sub-question
    priority: int = 1  # 1=highest, 3=lowest


@dataclass
class DecompositionResult:
    """Result of query decomposition."""
    original: str
    is_complex: bool
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    sub_questions: List[SubQuestion] = field(default_factory=list)
    decomposition_method: str = "rule"  # "rule" or "llm"
    latency_ms: float = 0.0


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    doc_id: str
    text: str
    score: float
    source_query: str = ""  # Which sub-query retrieved this
    rank: int = 0


@dataclass
class FusedResult:
    """Result after fusing sub-query results."""
    doc_id: str
    text: str
    fused_score: float
    source_queries: List[str] = field(default_factory=list)
    individual_scores: Dict[str, float] = field(default_factory=dict)
    rank: int = 0


@dataclass
class ComparisonResult:
    """Comparison of single-query vs decomposed retrieval."""
    query: str
    single_query_recall: float = 0.0
    decomposed_recall: float = 0.0
    single_query_precision: float = 0.0
    decomposed_precision: float = 0.0
    single_query_latency_ms: float = 0.0
    decomposed_latency_ms: float = 0.0
    improvement_recall: float = 0.0
    improvement_precision: float = 0.0


# ===========================================================================
# Complex Query Detector
# ===========================================================================

class ComplexQueryDetector:
    """Detect whether a query is complex and classify its complexity level."""

    # Indicators of complex queries
    COMPLEX_MARKERS: List[Tuple[re.Pattern, int]] = [
        # Multi-condition markers (weight 2)
        (re.compile(r'(?:以及|还有|另外|此外|同时|and\s+also|in\s+addition|furthermore)',
                     re.IGNORECASE), 2),
        # Multi-step markers (weight 2)
        (re.compile(r'(?:先.*再|先.*然后|首先.*其次|首先.*然后|step\s*\d.*step\s*\d'
                     r'|first.*then|first.*second)',
                     re.IGNORECASE), 2),
        # Comparison markers (weight 1)
        (re.compile(r'(?:区别|对比|比较|差异|vs\.?|versus|compare.*and|'
                     r'哪个更好|优缺点)',
                     re.IGNORECASE), 1),
        # Nested questions (weight 2)
        (re.compile(r'\?.*\?|？.*？'), 2),
        # Cause-effect markers (weight 1)
        (re.compile(r'(?:为什么.*如果|为什么.*怎么|为什么.*如何|why.*if|'
                     r'cause.*effect|原因.*影响)',
                     re.IGNORECASE), 1),
        # Enumeration markers (weight 1)
        (re.compile(r'(?:\d[\.、)]|第[一二三四五\d]+[点项条方面]|'
                     r'following\s+\d+|several\s+aspects)',
                    re.IGNORECASE), 1),
        # Long query (weight increases with length)
        (re.compile(r'.{100,}'), 1),
        # Requires reasoning chain (weight 2)
        (re.compile(r'(?:如何判断|如何确定|怎么分析|how\s+to\s+determine|'
                     r'how\s+to\s+diagnose)',
                     re.IGNORECASE), 2),
    ]

    @classmethod
    def detect(cls, query: str) -> Tuple[bool, QueryComplexity, int]:
        """Detect complexity of a query.

        Returns (is_complex, complexity_level, complexity_score).
        """
        score = 0

        for pattern, weight in cls.COMPLEX_MARKERS:
            matches = pattern.findall(query)
            score += len(matches) * weight

        # Length bonus
        if len(query) > 50:
            score += 1
        if len(query) > 100:
            score += 2

        # Determine complexity level
        if score == 0:
            complexity = QueryComplexity.SIMPLE
            is_complex = False
        elif score <= 2:
            complexity = QueryComplexity.MODERATE
            is_complex = True
        elif score <= 5:
            complexity = QueryComplexity.COMPLEX
            is_complex = True
        else:
            complexity = QueryComplexity.VERY_COMPLEX
            is_complex = True

        return is_complex, complexity, score


# ===========================================================================
# Query Decomposer
# ===========================================================================

class QueryDecomposer:
    """Decompose complex queries into independent sub-questions."""

    # Splitting patterns
    SPLIT_PATTERNS: List[Tuple[re.Pattern, str]] = [
        # Semicolons
        (re.compile(r'[；;]\s*'), 'semicolon'),
        # Conjunction words
        (re.compile(r'\s*(?:以及|还有|另外|此外|同时|and\s+also|in\s+addition)\s*'), 'conjunction'),
        # Question marks within text
        (re.compile(r'[？?]\s*(?=[^？?]*[？?])'), 'nested_question'),
        # Aspect enumerations
        (re.compile(r'\s*(?:第二[点项方面]|第三[点项方面]|2[\.、)]|3[\.、)])'), 'enumeration'),
        # Comparison split
        (re.compile(r'\s*(?:vs\.?|versus|对比|和.*相比)\s*'), 'comparison'),
    ]

    # Aspect-specific templates for completing sub-questions
    ASPECT_TEMPLATES: Dict[str, str] = {
        'definition': "请解释「{topic}」的定义和基本概念。",
        'process': "请说明「{topic}」的具体步骤和流程。",
        'comparison': "请对比「{topic}」中涉及的各方案的优缺点。",
        'example': "请提供「{topic}」的实际案例和应用场景。",
        'troubleshoot': "针对「{topic}」，分析可能的问题原因。",
        'best_practice': "关于「{topic}」，有哪些最佳实践和注意事项？",
        'limitation': "「{topic}」存在哪些局限性和挑战？",
        'future': "「{topic}」的未来发展趋势是什么？",
    }

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def decompose(self, query: str) -> DecompositionResult:
        """Decompose a complex query into sub-questions."""
        t0 = time.time()

        is_complex, complexity, score = ComplexQueryDetector.detect(query)

        result = DecompositionResult(
            original=query,
            is_complex=is_complex,
            complexity=complexity,
        )

        if not is_complex:
            result.latency_ms = (time.time() - t0) * 1000
            return result

        # Try rule-based decomposition
        sub_questions = self._decompose_by_rules(query)

        if len(sub_questions) <= 1 and self.use_llm:
            # Fallback to LLM-based decomposition if rules fail
            sub_questions = self._decompose_by_llm_simulation(query)

        result.sub_questions = self._assign_priorities(sub_questions)
        result.decomposition_method = "rule" if not self.use_llm else "llm"
        result.latency_ms = (time.time() - t0) * 1000

        return result

    def _decompose_by_rules(self, query: str) -> List[SubQuestion]:
        """Rule-based query decomposition."""
        sub_questions: List[SubQuestion] = []
        parts = [query]  # Start with whole query

        # Try each split pattern
        for pattern, split_type in self.SPLIT_PATTERNS:
            new_parts: List[str] = []
            for part in parts:
                split = pattern.split(part)
                # Filter empty splits
                split = [s.strip() for s in split if s.strip() and len(s.strip()) > 5]
                if len(split) > 1:
                    new_parts.extend(split)
                else:
                    new_parts.append(part)
            if len(new_parts) > len(parts):
                parts = new_parts

        # Convert parts to SubQuestions
        for i, part in enumerate(parts, 1):
            # Clean and complete
            part = part.strip()
            if not part:
                continue

            # Ensure it's a question
            part = self._ensure_question(part)

            # Determine aspect
            aspect = self._classify_aspect(part)

            sub_questions.append(SubQuestion(
                id=f"sub_{i:02d}",
                text=part,
                parent_query=query,
                aspect=aspect,
            ))

        return sub_questions

    def _decompose_by_llm_simulation(self, query: str) -> List[SubQuestion]:
        """Simulated LLM decomposition for offline testing.

        In production, replace with actual LLM API call.
        Uses multi-aspect extraction strategy.
        """
        sub_questions: List[SubQuestion] = []

        # Strategy: extract topics and generate aspect-based sub-questions
        topics = self._extract_topics(query)

        if not topics:
            # If no topics extracted, treat the query itself as a single sub-question
            sub_questions.append(SubQuestion(
                id="sub_01",
                text=self._ensure_question(query),
                parent_query=query,
                aspect="general",
            ))
            return sub_questions

        # Detect what aspects the query asks about
        aspects_mentioned = self._detect_aspects(query)

        if len(aspects_mentioned) <= 1 and len(topics) <= 1:
            return self._decompose_by_rules(query)

        # Generate sub-questions for each topic x aspect combination
        counter = 0
        for topic in topics:
            for aspect in aspects_mentioned:
                counter += 1
                template = self.ASPECT_TEMPLATES.get(aspect, "请说明「{topic}」")
                sub_q = template.replace('{topic}', topic)
                sub_questions.append(SubQuestion(
                    id=f"sub_{counter:02d}",
                    text=sub_q,
                    parent_query=query,
                    aspect=aspect,
                ))

        # If too few sub-questions, add the original
        if len(sub_questions) < 2:
            sub_questions.append(SubQuestion(
                id=f"sub_{len(sub_questions)+1:02d}",
                text=self._ensure_question(query),
                parent_query=query,
                aspect="general",
            ))

        return sub_questions

    def _ensure_question(self, text: str) -> str:
        """Ensure text ends as a proper question."""
        text = text.strip()
        if not text:
            return text
        if not text.endswith('?') and not text.endswith('？'):
            # Add appropriate question ending
            if re.search(r'(什么|如何|怎么|为什么|哪|谁|何时|What|How|Why|Which|Who|When)',
                        text, re.IGNORECASE):
                text += '？' if re.search(r'[一-鿿]', text) else '?'
            else:
                text += '？' if re.search(r'[一-鿿]', text) else '?'
        return text

    def _classify_aspect(self, text: str) -> str:
        """Classify the aspect of a sub-question."""
        if re.search(r'(什么是|定义|概念|含义|What\s+is|Define)', text, re.IGNORECASE):
            return 'definition'
        if re.search(r'(如何|怎么|步骤|方法|How\s+to|Process)', text, re.IGNORECASE):
            return 'process'
        if re.search(r'(区别|对比|比较|差异|vs\.?|Compare)', text, re.IGNORECASE):
            return 'comparison'
        if re.search(r'(案例|示例|例子|Example|Case)', text, re.IGNORECASE):
            return 'example'
        if re.search(r'(为什么|原因|错误|失败|Why|Error|Bug)', text, re.IGNORECASE):
            return 'troubleshoot'
        if re.search(r'(最佳实践|注意事项|注意|Best\s+Practice)', text, re.IGNORECASE):
            return 'best_practice'
        if re.search(r'(局限|缺点|不足|挑战|Limitation|Drawback)', text, re.IGNORECASE):
            return 'limitation'
        if re.search(r'(未来|趋势|发展|Future|Trend)', text, re.IGNORECASE):
            return 'future'
        return 'general'

    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from the query."""
        topics: List[str] = []

        # Extract noun phrases (Chinese + English)
        # Pattern: technical terms, named entities
        patterns = [
            r'([A-Z]+)',                          # Acronyms
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', # English proper nouns
            r'([一-鿿]{2,6}(?:模型|系统|架构|算法|网络|框架|技术|方法|理论|机制))',  # Chinese tech terms
        ]

        for pat in patterns:
            matches = re.findall(pat, text)
            topics.extend(matches)

        # Deduplicate
        seen: Set[str] = set()
        unique: List[str] = []
        for t in topics:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique.append(t)

        return unique[:5]  # Max 5 topics

    def _detect_aspects(self, query: str) -> List[str]:
        """Detect which aspects the query asks about."""
        aspects: List[str] = []
        aspect_checks = [
            ('definition', r'(什么是|定义|概念|含义|What\s+is)'),
            ('process', r'(如何|怎么|步骤|方法|How\s+to)'),
            ('comparison', r'(区别|对比|比较|差异|vs\.?|哪个更好)'),
            ('example', r'(案例|示例|例子|比如|Example)'),
            ('troubleshoot', r'(为什么|错误|失败|不工作|Why|Error)'),
            ('best_practice', r'(最佳实践|注意事项|建议|Best\s+Practice)'),
            ('limitation', r'(局限|缺点|不足|挑战|Limitation)'),
            ('future', r'(未来|趋势|发展|Future|Trend)'),
        ]

        for aspect, pattern in aspect_checks:
            if re.search(pattern, query, re.IGNORECASE):
                aspects.append(aspect)

        if not aspects:
            aspects.append('general')

        return aspects

    def _assign_priorities(self, sub_questions: List[SubQuestion]) -> List[SubQuestion]:
        """Assign priorities to sub-questions based on aspect type."""
        aspect_priority = {
            'definition': 1,     # Core concept first
            'process': 2,
            'troubleshoot': 1,   # Problem diagnosis is urgent
            'comparison': 2,
            'example': 3,
            'best_practice': 2,
            'limitation': 3,
            'future': 3,
            'general': 2,
        }

        for sq in sub_questions:
            sq.priority = aspect_priority.get(sq.aspect, 2)

        # Sort by priority
        sub_questions.sort(key=lambda sq: sq.priority)
        return sub_questions


# ===========================================================================
# Result Fusion
# ===========================================================================

class ResultFusion:
    """Fuse retrieval results from multiple sub-queries.

    Strategies:
    1. Score-based: Weighted by sub-query priority
    2. Reciprocal Rank Fusion (RRF) across sub-queries
    3. Topic-based: Merge by topic similarity
    """

    @staticmethod
    def fuse_by_score(
        all_results: List[List[RetrievalResult]],
        sub_questions: List[SubQuestion],
        top_k: int = 10,
    ) -> List[FusedResult]:
        """Fuse results weighted by sub-question priority."""
        priority_map = {sq.id: sq.priority for sq in sub_questions}

        # Aggregate scores
        doc_scores: Dict[str, Tuple[float, List[str], Dict[str, float]]] = {}

        for sq_results, sq in zip(all_results, sub_questions):
            weight = 1.0 / priority_map.get(sq.id, 1)  # Higher priority = higher weight
            for rr in sq_results:
                did = rr.doc_id
                if did not in doc_scores:
                    doc_scores[did] = (0.0, [], {})
                score, sources, indiv = doc_scores[did]
                doc_scores[did] = (
                    score + rr.score * weight,
                    sources + [sq.id],
                    {**indiv, sq.id: rr.score},
                )

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)

        results: List[FusedResult] = []
        for rank, (did, (score, sources, indiv)) in enumerate(sorted_docs[:top_k], 1):
            # Find the best text for this document
            best_text = ""
            for sq_results in all_results:
                for rr in sq_results:
                    if rr.doc_id == did and rr.text:
                        best_text = rr.text
                        break
                if best_text:
                    break

            results.append(FusedResult(
                doc_id=did,
                text=best_text,
                fused_score=score,
                source_queries=sources,
                individual_scores=indiv,
                rank=rank,
            ))

        return results

    @staticmethod
    def fuse_by_rrf(
        all_results: List[List[RetrievalResult]],
        top_k: int = 10,
        k: int = 60,
    ) -> List[FusedResult]:
        """Fuse using Reciprocal Rank Fusion."""
        rrf_scores: Dict[str, Tuple[float, List[str], Dict[str, float]]] = {}

        for sq_results in all_results:
            for rank, rr in enumerate(sq_results, 1):
                did = rr.doc_id
                rrf_contribution = 1.0 / (k + rank)

                if did not in rrf_scores:
                    rrf_scores[did] = (0.0, [], {})
                score, sources, indiv = rrf_scores[did]
                rrf_scores[did] = (
                    score + rrf_contribution,
                    sources + [rr.source_query] if rr.source_query not in sources else sources,
                    {**indiv, rr.source_query: rr.score},
                )

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1][0], reverse=True)

        results: List[FusedResult] = []
        for rank, (did, (score, sources, indiv)) in enumerate(sorted_docs[:top_k], 1):
            best_text = ""
            for sq_results in all_results:
                for rr in sq_results:
                    if rr.doc_id == did and rr.text:
                        best_text = rr.text
                        break
                if best_text:
                    break

            results.append(FusedResult(
                doc_id=did,
                text=best_text,
                fused_score=score,
                source_queries=sources,
                individual_scores=indiv,
                rank=rank,
            ))

        return results

    @staticmethod
    def deduplicate(results: List[FusedResult],
                    text_similarity_threshold: float = 0.9) -> List[FusedResult]:
        """Remove near-duplicate results based on text overlap."""
        if not results:
            return results

        unique: List[FusedResult] = [results[0]]

        for r in results[1:]:
            is_dup = False
            for u in unique:
                overlap = ResultFusion._text_overlap(r.text, u.text)
                if overlap > text_similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(r)

        return unique

    @staticmethod
    def _text_overlap(text1: str, text2: str) -> float:
        """Simple text overlap ratio for deduplication."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0


# ===========================================================================
# Decomposed Retrieval Pipeline
# ===========================================================================

class DecomposedRetrievalPipeline:
    """Complete decomposed retrieval pipeline.

    Flow:
    1. Detect if query is complex
    2. Decompose into sub-questions
    3. Retrieve for each sub-question in parallel (simulated)
    4. Fuse results
    5. Compare with single-query baseline
    """

    def __init__(
        self,
        decomposer: Optional[QueryDecomposer] = None,
        fusion_strategy: str = "score",  # "score" or "rrf"
        use_llm_decomposition: bool = False,
    ):
        self.decomposer = decomposer or QueryDecomposer(use_llm=use_llm_decomposition)
        self.fusion_strategy = fusion_strategy
        self.fusion = ResultFusion()

    def process(
        self,
        query: str,
        single_query_results: List[RetrievalResult],
        sub_query_results: Optional[List[List[RetrievalResult]]] = None,
    ) -> Tuple[DecompositionResult, List[FusedResult], ComparisonResult]:
        """Process a query through decomposition and fusion.

        Args:
            query: The original query.
            single_query_results: Retrieval results using the original query.
            sub_query_results: Pre-computed results for each sub-question.
                              If None, simulates by splitting single_query_results.

        Returns:
            (decomposition_result, fused_results, comparison_result)
        """
        # Step 1: Decompose
        decomp = self.decomposer.decompose(query)

        if not decomp.is_complex or not decomp.sub_questions:
            # Not complex enough — return single query results as-is
            fused = [
                FusedResult(
                    doc_id=r.doc_id, text=r.text, fused_score=r.score,
                    source_queries=[query], rank=r.rank,
                )
                for r in single_query_results
            ]
            comparison = ComparisonResult(query=query)
            return decomp, fused, comparison

        # Step 2: Fusion
        if sub_query_results is None:
            # Simulate: assign portions of single_query_results to sub-queries
            sub_query_results = self._simulate_sub_results(
                single_query_results, decomp.sub_questions
            )

        if self.fusion_strategy == "rrf":
            fused = self.fusion.fuse_by_rrf(sub_query_results)
        else:
            fused = self.fusion.fuse_by_score(
                sub_query_results, decomp.sub_questions
            )

        # Step 3: Deduplicate
        fused = self.fusion.deduplicate(fused)

        # Step 4: Compare with single-query baseline
        comparison = self._compare(single_query_results, fused, decomp)

        return decomp, fused, comparison

    def _simulate_sub_results(
        self,
        single_results: List[RetrievalResult],
        sub_questions: List[SubQuestion],
    ) -> List[List[RetrievalResult]]:
        """Simulate sub-query results from single-query results.

        In production, each sub-query would trigger an actual retrieval.
        For testing, we redistribute the single-query results with variation.
        """
        n_subs = len(sub_questions)
        if n_subs <= 1:
            return [single_results]

        # Distribute results across sub-queries with score variation
        sub_results: List[List[RetrievalResult]] = []

        for i, sq in enumerate(sub_questions):
            sq_results: List[RetrievalResult] = []
            for rank, sr in enumerate(single_results):
                # Vary scores slightly for each sub-query
                variation = 0.9 + (i * 0.05)  # Each sub-query gets slightly different weight
                sq_results.append(RetrievalResult(
                    doc_id=sr.doc_id,
                    text=sr.text,
                    score=sr.score * variation,
                    source_query=sq.id,
                    rank=rank + 1,
                ))
            sub_results.append(sq_results)

        return sub_results

    def _compare(
        self,
        single_results: List[RetrievalResult],
        fused_results: List[FusedResult],
        decomp: DecompositionResult,
    ) -> ComparisonResult:
        """Compare single-query vs decomposed retrieval quality."""
        comp = ComparisonResult(query=decomp.original)

        # Use diversity as a proxy for recall (more diverse = potentially better recall)
        single_docs = {r.doc_id for r in single_results[:10]}
        fused_docs = {r.doc_id for r in fused_results[:10]}

        # Coverage ratio
        if single_docs:
            comp.single_query_precision = 1.0  # Baseline
            comp.decomposed_precision = len(fused_docs & single_docs) / max(len(fused_docs), 1)
            comp.improvement_precision = comp.decomposed_precision - comp.single_query_precision

        # New documents discovered (proxy for recall improvement)
        new_docs = fused_docs - single_docs
        if fused_docs:
            comp.decomposed_recall = len(fused_docs) / max(len(fused_docs | single_docs), 1)
            comp.single_query_recall = len(single_docs) / max(len(fused_docs | single_docs), 1)
            comp.improvement_recall = comp.decomposed_recall - comp.single_query_recall

        return comp


# ===========================================================================
# __main__: Demo
# ===========================================================================

_SAMPLE_DOCS = [
    RetrievalResult("d1", "Transformer是一种基于自注意力机制的深度学习架构，广泛应用于NLP任务。", 0.95, "", 1),
    RetrievalResult("d2", "BERT通过掩码语言模型进行预训练，能够捕捉双向上下文信息。", 0.90, "", 2),
    RetrievalResult("d3", "GPT系列使用自回归语言模型，擅长文本生成任务。单次推理只能看到前文。", 0.85, "", 3),
    RetrievalResult("d4", "注意力机制的计算公式为Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。", 0.80, "", 4),
    RetrievalResult("d5", "在模型部署时需要注意延迟、吞吐量和资源占用等性能指标。", 0.75, "", 5),
    RetrievalResult("d6", "BERT适合文本分类和序列标注任务，GPT适合对话生成和创意写作。", 0.70, "", 6),
    RetrievalResult("d7", "模型训练不收敛可能是学习率过大或数据未归一化导致的。", 0.65, "", 7),
    RetrievalResult("d8", "知识蒸馏可以将大模型的知识迁移到小模型，降低部署成本。", 0.60, "", 8),
    RetrievalResult("d9", "RAG系统通过检索外部知识库来增强语言模型的生成质量。", 0.55, "", 9),
    RetrievalResult("d10", "迁移学习利用预训练模型在下游任务上微调，大幅减少训练数据需求。", 0.50, "", 10),
]


if __name__ == '__main__':
    print("=" * 60)
    print("Complex Query Decomposition - Self-Test")
    print("=" * 60)

    # --- Test queries ---
    test_queries = [
        # Simple
        "什么是Transformer架构？",
        # Moderate
        "BERT和GPT有什么区别，各自适合什么场景？",
        # Complex
        "什么是注意力机制以及它如何提升NLP模型性能，另外在实际部署中需要注意什么？",
        # Very Complex
        "为什么模型训练不收敛，应该如何诊断并解决这个问题，同时如何预防类似问题再次发生？",
    ]

    pipeline = DecomposedRetrievalPipeline(fusion_strategy="score")

    for qi, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {qi}: {query[:80]}...")
        print(f"{'='*60}")

        # Decompose
        decomp_result, fused_results, comparison = pipeline.process(
            query, _SAMPLE_DOCS
        )

        is_complex, complexity, score = ComplexQueryDetector.detect(query)
        print(f"  Complexity: {complexity.value} (score={score})")
        print(f"  Decomposed: {decomp_result.is_complex}")
        print(f"  Method:     {decomp_result.decomposition_method}")
        print(f"  Latency:    {decomp_result.latency_ms:.1f}ms")

        if decomp_result.sub_questions:
            print(f"  Sub-questions ({len(decomp_result.sub_questions)}):")
            for sq in decomp_result.sub_questions:
                print(f"    [{sq.id}] p={sq.priority} aspect={sq.aspect}")
                print(f"           {sq.text[:100]}...")

        # Fused results
        if fused_results:
            print(f"\n  Fused Results (top 5):")
            for fr in fused_results[:5]:
                print(f"    [{fr.rank}] score={fr.fused_score:.4f} | {fr.doc_id}: "
                      f"{fr.text[:60]}...")
                print(f"           sources: {fr.source_queries}")

        # Comparison
        print(f"\n  Comparison (decomposed vs single):")
        print(f"    Precision: {comparison.single_query_precision:.3f} → "
              f"{comparison.decomposed_precision:.3f} "
              f"({comparison.improvement_precision:+.3f})")
        print(f"    Recall:    {comparison.single_query_recall:.3f} → "
              f"{comparison.decomposed_recall:.3f} "
              f"({comparison.improvement_recall:+.3f})")

    # --- Complexity detection benchmark ---
    print(f"\n{'='*60}")
    print("Complexity Detection Benchmark")
    print(f"{'='*60}")

    bench_queries = [
        ("你好", QueryComplexity.SIMPLE),
        ("什么是机器学习？", QueryComplexity.SIMPLE),
        ("如何训练模型以及如何评估性能？", QueryComplexity.MODERATE),
        ("BERT和GPT的区别是什么，各自在什么场景下表现更好？", QueryComplexity.MODERATE),
        ("什么是注意力机制以及它如何影响模型性能，另外在实际部署中需要注意哪些问题？",
         QueryComplexity.COMPLEX),
        ("为什么模型训练不收敛？应该如何系统性地诊断问题根源？"
         "如何解决这些问题？如何建立预防机制避免再次发生？",
         QueryComplexity.VERY_COMPLEX),
    ]

    correct = 0
    for query, expected in bench_queries:
        is_complex, complexity, score = ComplexQueryDetector.detect(query)
        status = "OK" if complexity == expected else f"WRONG (got {complexity.value})"
        if complexity == expected:
            correct += 1
        print(f"  [{status}] score={score:>2} expected={expected.value:<14} | {query[:60]}...")

    accuracy = correct / len(bench_queries) * 100
    print(f"\n  Accuracy: {accuracy:.1f}% ({correct}/{len(bench_queries)})")

    print(f"\n{'='*60}")
    print("Complex Query Decomposition self-test completed!")
    print(f"{'='*60}")
