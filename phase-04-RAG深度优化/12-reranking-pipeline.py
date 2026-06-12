#!/usr/bin/env python3
"""
Phase 04 Module 12: Reranking Pipeline
=========================================
- BGE-Reranker-v2-m3 integration (use_fp16=True)
- Cohere Rerank v3 API (simulated for offline)
- Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
- Comparative benchmark: precision@5 improvement from each
- Batch reranking for latency optimization
- Complete pipeline: 粗检索Top-20 → Rerank Top-5 → LLM生成
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Optional imports — degrade gracefully
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    CrossEncoder = None

try:
    from FlagEmbedding import FlagReranker
    HAS_BGE_RERANKER = True
except ImportError:
    HAS_BGE_RERANKER = False
    FlagReranker = None


# ===========================================================================
# Data Classes
# ===========================================================================

class RerankerType(Enum):
    BGE = "bge-reranker-v2-m3"
    COHERE = "cohere-rerank-v3"
    CROSS_ENCODER = "cross-encoder-msmarco"
    ENSEMBLE = "ensemble"


@dataclass
class RerankerResult:
    """Result from a single reranker."""
    doc_id: str
    text: str
    original_score: float
    rerank_score: float
    original_rank: int
    rerank_rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerBenchmark:
    """Benchmark metrics for a reranker."""
    reranker_type: RerankerType
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    avg_latency_ms: float = 0.0
    avg_items_reranked: int = 0
    improvement_over_baseline: float = 0.0  # % improvement in P@5


# ===========================================================================
# BGE Reranker Wrapper
# ===========================================================================

class BGEReranker:
    """Wrapper for BGE-Reranker-v2-m3 from FlagEmbedding.

    Model: BAAI/bge-reranker-v2-m3
    Capable of handling both Chinese and English.
    """

    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-v2-m3',
        use_fp16: bool = True,
        device: str = 'cpu',
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self.batch_size = batch_size
        self._model: Optional[Any] = None

        if HAS_BGE_RERANKER:
            try:
                self._model = FlagReranker(
                    model_name,
                    use_fp16=use_fp16,
                    device=device,
                )
            except Exception as e:
                print(f"[WARN] BGE Reranker init failed: {e}")
                self._model = None
        else:
            print("[WARN] FlagEmbedding not installed. Install with: "
                  "pip install FlagEmbedding")

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Rerank documents with BGE Reranker.

        Returns:
            List of (original_index, rerank_score) sorted by score descending.
        """
        if not self.is_available or not documents:
            return self._simulated_rerank(query, documents, top_k)

        # Create pairs
        pairs = [[query, doc] for doc in documents]

        # Compute scores in batches
        all_scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            try:
                scores = self._model.compute_score(batch)
                if isinstance(scores, (int, float)):
                    scores = [float(scores)]
                all_scores.extend(float(s) for s in scores)
            except Exception as e:
                print(f"[WARN] BGE batch failed: {e}")
                all_scores.extend([0.0] * len(batch))

        # Sort by score descending
        indexed_scores = list(enumerate(all_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    def _simulated_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Simulated reranking for offline testing.
        Uses keyword overlap + length penalty as a simple proxy.
        """
        scores: List[float] = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for doc in documents:
            doc_lower = doc.lower()
            # Term overlap score
            overlap = sum(1 for t in query_terms if t in doc_lower)
            # Length penalty (prefer concise docs)
            length_penalty = min(1.0, 200 / max(len(doc), 1))
            # Combine
            score = (overlap / max(len(query_terms), 1)) * 0.7 + length_penalty * 0.3
            scores.append(score)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed = indexed[:top_k]

        return indexed


# ===========================================================================
# Cohere Rerank v3 (Simulated)
# ===========================================================================

class CohereReranker:
    """Cohere Rerank v3 API wrapper (simulated for offline testing).

    In production, this would call the Cohere API:
    POST https://api.cohere.ai/v1/rerank
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'rerank-english-v3.0',
        batch_size: int = 32,
    ):
        self.api_key = api_key or os.environ.get('COHERE_API_KEY', '')
        self.model = model
        self.batch_size = batch_size
        self._is_available = bool(self.api_key)

    @property
    def is_available(self) -> bool:
        return self._is_available

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Rerank using Cohere API (simulated).

        Returns:
            List of (original_index, rerank_score).
        """
        if not self.is_available:
            return self._simulated_rerank(query, documents, top_k)

        # In production: actual API call
        # For now: use enhanced simulation
        return self._simulated_rerank(query, documents, top_k)

    def _simulated_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Enhanced simulation of Cohere reranking behavior."""
        import math
        scores: List[float] = []
        query_lower = query.lower()

        for doc in documents:
            doc_lower = doc.lower()

            # Simulate Cohere's semantic matching
            # 1. N-gram overlap
            query_bigrams = set(query_lower[i:i+2] for i in range(len(query_lower)-1))
            doc_bigrams = set(doc_lower[i:i+2] for i in range(len(doc_lower)-1))
            if query_bigrams:
                bigram_overlap = len(query_bigrams & doc_bigrams) / len(query_bigrams)
            else:
                bigram_overlap = 0.0

            # 2. Keyword density
            query_words = set(query_lower.split())
            word_matches = sum(1 for w in query_words if w in doc_lower)
            keyword_density = word_matches / max(len(query_words), 1)

            # 3. Position bonus (earlier = more relevant assumption)
            position_score = 1.0 / (1.0 + math.log(1 + len(doc) / 500))

            score = bigram_overlap * 0.4 + keyword_density * 0.4 + position_score * 0.2
            scores.append(score)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed = indexed[:top_k]

        return indexed


# ===========================================================================
# Cross-Encoder Reranker
# ===========================================================================

class CrossEncoderReranker:
    """Cross-Encoder based reranker using ms-marco-MiniLM-L-6-v2.

    Lightweight, fast, good for initial filtering before heavier rerankers.
    """

    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        device: str = 'cpu',
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Optional[Any] = None

        if HAS_CROSS_ENCODER:
            try:
                self._model = CrossEncoder(
                    model_name,
                    device=device,
                    max_length=max_length,
                )
            except Exception as e:
                print(f"[WARN] CrossEncoder init failed: {e}")
                self._model = None
        else:
            print("[WARN] sentence-transformers CrossEncoder not available.")

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Rerank using Cross-Encoder."""
        if not self.is_available or not documents:
            return self._simulated_rerank(query, documents, top_k)

        pairs = [[query, doc[:self.max_length]] for doc in documents]

        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            scores = [float(s) for s in scores]
        except Exception as e:
            print(f"[WARN] CrossEncoder predict failed: {e}")
            return self._simulated_rerank(query, documents, top_k)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed = indexed[:top_k]

        return indexed

    def _simulated_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Fallback simulation."""
        scores: List[float] = []
        query_terms = set(query.lower().split())

        for doc in documents:
            doc_terms = set(doc.lower().split())
            if query_terms:
                score = len(query_terms & doc_terms) / len(query_terms)
            else:
                score = 0.0
            scores.append(score)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed = indexed[:top_k]

        return indexed


# ===========================================================================
# Ensemble Reranker
# ===========================================================================

class EnsembleReranker:
    """Combine multiple rerankers using weighted score fusion."""

    def __init__(
        self,
        rerankers: List[Tuple[Any, float]],  # (reranker_instance, weight)
    ):
        self.rerankers = rerankers

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Ensemble reranking with weighted score combination."""
        if not self.rerankers or not documents:
            return [(i, 0.0) for i in range(len(documents))]

        n = len(documents)
        all_scores = np.zeros(n)

        for reranker, weight in self.rerankers:
            results = reranker.rerank(query, documents, top_k=None)
            for idx, score in results:
                all_scores[idx] += score * weight

        # Normalize
        if np.max(all_scores) > 0:
            all_scores = all_scores / np.max(all_scores)

        indexed = list(enumerate(float(s) for s in all_scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed = indexed[:top_k]

        return indexed


# ===========================================================================
# Complete Reranking Pipeline
# ===========================================================================

class RerankingPipeline:
    """Complete reranking pipeline: 粗检索 → 重排序 → Top-K 输出

    Flow:
    1. Coarse retrieval: Top-N candidates (e.g., 20)
    2. Optional threshold filter: discard low-relevance
    3. Reranker(s): re-score remaining candidates
    4. Output: Top-K results (e.g., 5) for LLM generation
    """

    def __init__(
        self,
        reranker_type: RerankerType = RerankerType.BGE,
        coarse_top_n: int = 20,
        final_top_k: int = 5,
        use_fp16: bool = True,
        device: str = 'cpu',
        cohere_api_key: Optional[str] = None,
    ):
        self.reranker_type = reranker_type
        self.coarse_top_n = coarse_top_n
        self.final_top_k = final_top_k
        self.device = device

        # Initialize rerankers
        self.bge_reranker = BGEReranker(use_fp16=use_fp16, device=device)
        self.cohere_reranker = CohereReranker(api_key=cohere_api_key)
        self.cross_encoder = CrossEncoderReranker(device=device)

        if reranker_type == RerankerType.ENSEMBLE:
            available_rerankers = []
            if self.bge_reranker.is_available:
                available_rerankers.append((self.bge_reranker, 0.4))
            if self.cross_encoder.is_available:
                available_rerankers.append((self.cross_encoder, 0.3))
            available_rerankers.append((self.cohere_reranker, 0.3))
            self.ensemble = EnsembleReranker(available_rerankers)

    def run(
        self,
        query: str,
        coarse_results: List[Tuple[str, float, str]],
        return_details: bool = False,
    ) -> List[RerankerResult]:
        """Run the complete reranking pipeline.

        Args:
            query: The search query.
            coarse_results: [(doc_id, score, text), ...] from coarse retrieval.
            return_details: Include intermediate scores.

        Returns:
            Reranked results, top K.
        """
        t0 = time.time()

        # Step 1: Take top-N from coarse retrieval
        coarse_top = coarse_results[:self.coarse_top_n]
        doc_ids = [r[0] for r in coarse_top]
        texts = [r[2] for r in coarse_top]
        original_scores = [r[1] for r in coarse_top]

        if not texts:
            return []

        # Step 2: Rerank
        reranker = self._get_reranker()
        reranked = reranker.rerank(query, texts, top_k=self.final_top_k)

        # Step 3: Build results
        results: List[RerankerResult] = []
        for new_rank, (orig_idx, rerank_score) in enumerate(reranked, 1):
            orig_score = original_scores[orig_idx] if orig_idx < len(original_scores) else 0.0
            results.append(RerankerResult(
                doc_id=doc_ids[orig_idx] if orig_idx < len(doc_ids) else f'doc_{orig_idx}',
                text=texts[orig_idx] if orig_idx < len(texts) else '',
                original_score=orig_score,
                rerank_score=rerank_score,
                original_rank=orig_idx + 1,
                rerank_rank=new_rank,
            ))

        elapsed = (time.time() - t0) * 1000

        if return_details:
            for r in results:
                r.metadata['pipeline_latency_ms'] = elapsed

        return results

    def _get_reranker(self) -> Any:
        """Get the active reranker."""
        if self.reranker_type == RerankerType.BGE:
            return self.bge_reranker
        elif self.reranker_type == RerankerType.COHERE:
            return self.cohere_reranker
        elif self.reranker_type == RerankerType.CROSS_ENCODER:
            return self.cross_encoder
        elif self.reranker_type == RerankerType.ENSEMBLE:
            return self.ensemble
        return self.bge_reranker  # Default

    def benchmark(
        self,
        queries: List[str],
        coarse_results_all: List[List[Tuple[str, float, str]]],
        relevant_doc_ids: List[List[str]],
    ) -> RerankerBenchmark:
        """Benchmark reranker performance.

        Compares before-rerank vs after-rerank Precision@5.
        """
        benchmark = RerankerBenchmark(reranker_type=self.reranker_type)

        total_p5_before = 0.0
        total_p5_after = 0.0
        total_p10_before = 0.0
        total_p10_after = 0.0
        total_mrr = 0.0
        total_latency = 0.0
        total_items = 0

        for query, coarse_results, relevant in zip(
            queries, coarse_results_all, relevant_doc_ids
        ):
            relevant_set = set(relevant)

            # Before rerank
            before_5 = [r[0] for r in coarse_results[:5]]
            before_10 = [r[0] for r in coarse_results[:10]]
            total_p5_before += len(set(before_5) & relevant_set) / max(len(before_5), 1)
            total_p10_before += len(set(before_10) & relevant_set) / max(len(before_10), 1)

            # After rerank
            t0 = time.time()
            after = self.run(query, coarse_results)
            latency = (time.time() - t0) * 1000
            total_latency += latency

            after_docs = [r.doc_id for r in after]
            after_5 = after_docs[:5]
            after_10 = after_docs[:10]
            total_p5_after += len(set(after_5) & relevant_set) / max(len(after_5), 1)
            total_p10_after += len(set(after_10) & relevant_set) / max(len(after_10), 1)

            # MRR
            for rank, did in enumerate(after_docs, 1):
                if did in relevant_set:
                    total_mrr += 1.0 / rank
                    break

            total_items += len(after)

        n = len(queries)
        if n > 0:
            benchmark.precision_at_5 = total_p5_after / n
            benchmark.precision_at_10 = total_p10_after / n
            benchmark.mrr = total_mrr / n
            benchmark.avg_latency_ms = total_latency / n
            benchmark.avg_items_reranked = total_items // n

            baseline_p5 = total_p5_before / n
            if baseline_p5 > 0:
                benchmark.improvement_over_baseline = (
                    (benchmark.precision_at_5 - baseline_p5) / baseline_p5 * 100
                )

        return benchmark


# ===========================================================================
# __main__: Demo
# ===========================================================================

_SAMPLE_COARSE_RESULTS = [
    ("doc_a", 0.92, "Transformer是一种基于自注意力机制的神经网络架构，广泛应用于自然语言处理任务。"),
    ("doc_b", 0.88, "BERT通过掩码语言模型预训练，能够捕捉双向上下文信息，在多项NLP基准上表现优异。"),
    ("doc_c", 0.85, "深度学习使用多层神经网络进行特征学习。CNN、RNN和Transformer是三种主要架构。"),
    ("doc_d", 0.82, "自注意力机制允许模型在处理序列时关注不同位置的相关信息。"),
    ("doc_e", 0.78, "模型评估是机器学习流程中的重要环节，常用指标包括准确率、召回率和F1分数。"),
    ("doc_f", 0.75, "在NLP任务中，数据预处理包括分词、去除停用词、构建词表等步骤。"),
    ("doc_g", 0.72, "机器学习是AI的一个分支，通过从数据中学习模式来改进性能。"),
    ("doc_h", 0.68, "Python是一种广泛使用的编程语言，在数据科学和机器学习领域特别流行。"),
    ("doc_i", 0.65, "云计算提供了弹性的计算资源，可以按需扩展。"),
    ("doc_j", 0.60, "软件工程中的最佳实践包括代码审查、单元测试和持续集成。"),
    ("doc_k", 0.55, "项目管理方法论如Scrum和Kanban可以提高团队效率。"),
    ("doc_l", 0.50, "版本控制系统如Git是现代软件开发的基础工具。"),
]


if __name__ == '__main__':
    print("=" * 60)
    print("Reranking Pipeline - Self-Test")
    print("=" * 60)

    query = "Transformer自注意力机制如何工作？"

    # --- Test 1: BGE Reranker ---
    print(f"\n[Test 1] BGE Reranker")
    print("-" * 40)

    pipeline_bge = RerankingPipeline(
        reranker_type=RerankerType.BGE,
        coarse_top_n=12,
        final_top_k=5,
    )

    results_bge = pipeline_bge.run(query, _SAMPLE_COARSE_RESULTS, return_details=True)
    print(f"\n  Query: {query}")
    print(f"  Reranker: {pipeline_bge.reranker_type.value}")
    print(f"  Coarse top-N: {pipeline_bge.coarse_top_n} → Final top-K: {pipeline_bge.final_top_k}")
    if results_bge and results_bge[0].metadata:
        print(f"  Latency: {results_bge[0].metadata.get('pipeline_latency_ms', 0):.1f}ms")

    print(f"\n  {'Rerank Rank':<12} {'Orig Rank':<12} {'Orig Score':<12} "
          f"{'Rerank Score':<14} {'Doc ID':<10} {'Text'}")
    print(f"  {'-'*70}")
    for r in results_bge:
        print(f"  {r.rerank_rank:<12} {r.original_rank:<12} {r.original_score:<12.4f} "
              f"{r.rerank_score:<14.4f} {r.doc_id:<10} {r.text[:50]}...")

    # --- Test 2: Cross-Encoder Reranker ---
    print(f"\n[Test 2] Cross-Encoder Reranker")
    print("-" * 40)

    pipeline_ce = RerankingPipeline(
        reranker_type=RerankerType.CROSS_ENCODER,
        coarse_top_n=12,
        final_top_k=5,
    )

    results_ce = pipeline_ce.run(query, _SAMPLE_COARSE_RESULTS, return_details=True)
    print(f"\n  Reranker: {pipeline_ce.reranker_type.value}")

    print(f"\n  {'Rerank Rank':<12} {'Orig Rank':<12} {'Orig Score':<12} "
          f"{'Rerank Score':<14} {'Doc ID':<10} {'Text'}")
    print(f"  {'-'*70}")
    for r in results_ce:
        print(f"  {r.rerank_rank:<12} {r.original_rank:<12} {r.original_score:<12.4f} "
              f"{r.rerank_score:<14.4f} {r.doc_id:<10} {r.text[:50]}...")

    # --- Test 3: Cohere Reranker (simulated) ---
    print(f"\n[Test 3] Cohere Reranker (simulated)")
    print("-" * 40)

    pipeline_ch = RerankingPipeline(
        reranker_type=RerankerType.COHERE,
        coarse_top_n=12,
        final_top_k=5,
    )

    results_ch = pipeline_ch.run(query, _SAMPLE_COARSE_RESULTS, return_details=True)
    print(f"\n  Reranker: {pipeline_ch.reranker_type.value}")

    print(f"\n  {'Rerank Rank':<12} {'Orig Rank':<12} {'Orig Score':<12} "
          f"{'Rerank Score':<14} {'Doc ID':<10} {'Text'}")
    print(f"  {'-'*70}")
    for r in results_ch:
        print(f"  {r.rerank_rank:<12} {r.original_rank:<12} {r.original_score:<12.4f} "
              f"{r.rerank_score:<14.4f} {r.doc_id:<10} {r.text[:50]}...")

    # --- Test 4: Rank Movement Analysis ---
    print(f"\n{'='*60}")
    print("Rank Movement Analysis")
    print(f"{'='*60}")

    for label, results in [("BGE", results_bge), ("Cross-Encoder", results_ce),
                           ("Cohere", results_ch)]:
        promoted = [r for r in results if r.rerank_rank < r.original_rank]
        demoted = [r for r in results if r.rerank_rank > r.original_rank]
        unchanged = [r for r in results if r.rerank_rank == r.original_rank]
        print(f"\n  {label}:")
        print(f"    Promoted:   {len(promoted)} docs (avg +{sum(r.original_rank - r.rerank_rank for r in promoted)/max(len(promoted),1):.1f} ranks)")
        print(f"    Demoted:    {len(demoted)} docs (avg -{sum(r.rerank_rank - r.original_rank for r in demoted)/max(len(demoted),1):.1f} ranks)")
        print(f"    Unchanged:  {len(unchanged)} docs")

    # --- Test 5: Benchmark comparison ---
    print(f"\n{'='*60}")
    print("Convergent Validity Check")
    print(f"{'='*60}")

    # Check that all rerankers agree on top-1 (convergent validity)
    top1_bge = results_bge[0].doc_id if results_bge else None
    top1_ce = results_ce[0].doc_id if results_ce else None
    top1_ch = results_ch[0].doc_id if results_ch else None

    agreement = len({top1_bge, top1_ce, top1_ch}) if all([top1_bge, top1_ce, top1_ch]) else 0
    print(f"\n  Top-1 agreement: {1 if agreement == 1 else 'PARTIAL' if agreement == 2 else 'LOW'}")
    print(f"    BGE:  {top1_bge}")
    print(f"    CE:   {top1_ce}")
    print(f"    Cohere: {top1_ch}")

    # Top-3 overlap
    top3_bge = {r.doc_id for r in results_bge[:3]}
    top3_ce = {r.doc_id for r in results_ce[:3]}
    top3_ch = {r.doc_id for r in results_ch[:3]}
    all_top3 = top3_bge | top3_ce | top3_ch
    intersection = top3_bge & top3_ce & top3_ch
    print(f"\n  Top-3 overlap: {len(intersection)}/{len(all_top3)} ({len(intersection)/max(len(all_top3),1)*100:.0f}%)")
    print(f"    BGE top-3:  {top3_bge}")
    print(f"    CE top-3:   {top3_ce}")
    print(f"    Cohere top-3: {top3_ch}")
    if intersection:
        print(f"    All agree:  {intersection}")

    print(f"\n{'='*60}")
    print("Reranking Pipeline self-test completed!")
    print(f"{'='*60}")
