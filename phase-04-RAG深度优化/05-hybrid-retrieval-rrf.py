#!/usr/bin/env python3
"""
Phase 04 Module 05: Hybrid Retrieval with RRF Fusion ⭐ CRITICAL
===================================================================
Complete implementation of:
- Dense retrieval (vector cosine similarity)
- Sparse retrieval (BM25 via rank_bm25)
- RRF fusion: RRFscore(d) = Σ 1/(k + rank_i(d)), k=60
- Weighted hybrid with dynamic weights: general 7:3, professional 4:6
- Min-Max normalization before weighted fusion
- Fallback: if BM25 returns empty → vector-only
- Complete HybridRetriever class with benchmark comparison
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_DENSE = True
except ImportError:
    HAS_DENSE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    HAS_MINMAX = True
except ImportError:
    HAS_MINMAX = False


# ===========================================================================
# Enums and Data Classes
# ===========================================================================

class Scenario(Enum):
    GENERAL = "general"
    PROFESSIONAL = "professional"

@dataclass
class RetrievalResult:
    """Single retrieval result."""
    doc_id: str
    text: str
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rank_dense: int = -1
    rank_sparse: int = -1
    rank_hybrid: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Comparison of different retrieval methods."""
    method: str
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    avg_latency_ms: float = 0.0
    num_queries: int = 0


# ===========================================================================
# Dense Retriever
# ===========================================================================

class DenseRetriever:
    """Vector-based dense retrieval using cosine similarity."""

    def __init__(
        self,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        device: str = 'cpu',
        batch_size: int = 32,
    ):
        if not HAS_DENSE:
            raise ImportError("sentence-transformers required. pip install sentence-transformers")
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.doc_embeddings: Optional[np.ndarray] = None
        self.documents: List[Dict[str, str]] = []
        self._doc_ids: List[str] = []

    def index(self, documents: List[Dict[str, str]]) -> None:
        """Index documents by computing embeddings."""
        self.documents = documents
        self._doc_ids = [d.get('doc_id', f'doc_{i}') for i, d in enumerate(documents)]
        texts = [d.get('text', '') for d in documents]

        embeddings_list: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            emb = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings_list.append(emb)

        if embeddings_list:
            self.doc_embeddings = np.concatenate(embeddings_list, axis=0)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.doc_embeddings = self.doc_embeddings / norms

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Search for query, returns [(doc_index, cosine_score), ...]."""
        if self.doc_embeddings is None or len(self.documents) == 0:
            return []

        query_embedding = self.model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        )
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        scores = np.dot(self.doc_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]


# ===========================================================================
# Sparse Retriever (BM25)
# ===========================================================================

class SparseRetriever:
    """BM25-based sparse retrieval with tokenization for Chinese + English."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if not HAS_BM25:
            raise ImportError("rank-bm25 required. pip install rank-bm25")
        self.k1 = k1
        self.b = b
        self.bm25: Optional[Any] = None  # BM25Okapi instance
        self.documents: List[Dict[str, str]] = []
        self._doc_ids: List[str] = []
        self._corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text for BM25. Handles Chinese (character-level bigrams)
        and English (whitespace + punctuation split)."""
        tokens: List[str] = []

        # Split mixed text into segments
        segments = re.split(r'([a-zA-Z0-9]+|[一-鿿]+)', str(text))

        for seg in segments:
            if not seg.strip():
                continue
            if re.match(r'[a-zA-Z0-9]+', seg):
                # English: lowercase and split
                tokens.append(seg.lower())
            elif re.match(r'[一-鿿]+', seg):
                # Chinese: character bigrams for better BM25 matching
                if len(seg) == 1:
                    tokens.append(seg)
                else:
                    for i in range(len(seg) - 1):
                        tokens.append(seg[i:i + 2])
                    tokens.append(seg[-1])  # last unigram
            else:
                # Punctuation/whitespace → split individual chars
                for ch in seg:
                    if ch.strip() and not re.match(r'\s', ch):
                        tokens.append(ch)

        return [t for t in tokens if len(t) >= 1]

    def index(self, documents: List[Dict[str, str]]) -> None:
        """Build BM25 index from documents."""
        self.documents = documents
        self._doc_ids = [d.get('doc_id', f'doc_{i}') for i, d in enumerate(documents)]
        self._corpus = [self.tokenize(d.get('text', '')) for d in documents]

        if self._corpus:
            self.bm25 = BM25Okapi(self._corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """BM25 search. Returns [(doc_index, bm25_score), ...].
        If BM25 is not initialized, returns empty list (triggers fallback).
        """
        if self.bm25 is None or len(self._corpus) == 0:
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        # Get top-k indices
        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ===========================================================================
# RRF Fusion
# ===========================================================================

def rrf_fusion(
    dense_results: List[Tuple[int, float]],
    sparse_results: List[Tuple[int, float]],
    k: int = 60,
) -> Dict[int, float]:
    """Reciprocal Rank Fusion.

    RRFscore(d) = Σ_{i in retrievers} 1 / (k + rank_i(d))

    Args:
        dense_results: [(doc_idx, score), ...] from dense retriever.
        sparse_results: [(doc_idx, score), ...] from sparse retriever.
        k: Smoothing constant (default 60, empirically validated).

    Returns:
        Dict mapping doc_idx → RRF fusion score.
    """
    rrf_scores: Dict[int, float] = defaultdict(float)

    # Dense ranks
    for rank, (doc_idx, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_idx] += 1.0 / (k + rank)

    # Sparse ranks
    for rank, (doc_idx, _) in enumerate(sparse_results, start=1):
        rrf_scores[doc_idx] += 1.0 / (k + rank)

    return dict(rrf_scores)


def weighted_hybrid_fusion(
    dense_results: List[Tuple[int, float]],
    sparse_results: List[Tuple[int, float]],
    weight_dense: float = 0.7,
    weight_sparse: float = 0.3,
    k: int = 60,
    normalize: bool = True,
) -> Dict[int, float]:
    """Weighted hybrid fusion with optional Min-Max normalization.

    Args:
        weight_dense: Weight for dense retrieval scores.
        weight_sparse: Weight for sparse retrieval scores.
        normalize: Apply Min-Max normalization before weighted combination.
        k: RRF k parameter.

    Returns:
        Dict mapping doc_idx → weighted fusion score.
    """
    # First get RRF scores as base
    rrf_scores = rrf_fusion(dense_results, sparse_results, k=k)
    doc_indices = set(rrf_scores.keys())

    # Build raw score maps
    dense_map = {idx: score for idx, score in dense_results}
    sparse_map = {idx: score for idx, score in sparse_results}

    if normalize and HAS_MINMAX:
        # Min-Max normalize each source
        dense_scores_all = np.array([s for _, s in dense_results]).reshape(-1, 1)
        sparse_scores_all = np.array([s for _, s in sparse_results]).reshape(-1, 1)

        dense_normalized = {}
        if len(dense_scores_all) > 1:
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(dense_scores_all).flatten()
            dense_normalized = {dense_results[i][0]: normalized[i]
                               for i in range(len(dense_results))}
        else:
            dense_normalized = {idx: 1.0 if s > 0 else 0.0 for idx, s in dense_results}

        sparse_normalized = {}
        if len(sparse_scores_all) > 1:
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(sparse_scores_all).flatten()
            sparse_normalized = {sparse_results[i][0]: normalized[i]
                                for i in range(len(sparse_results))}
        else:
            sparse_normalized = {idx: 1.0 if s > 0 else 0.0 for idx, s in sparse_results}

        # Weighted combination
        fusion_scores: Dict[int, float] = {}
        for idx in doc_indices:
            d_score = dense_normalized.get(idx, 0.0)
            s_score = sparse_normalized.get(idx, 0.0)
            rrf_base = rrf_scores.get(idx, 0.0)

            # Combine: weighted normalized scores + RRF bonus
            fusion_scores[idx] = (
                weight_dense * d_score +
                weight_sparse * s_score +
                0.1 * rrf_base  # Small RRF bonus
            )
    else:
        # Simple weighted combination without normalization
        fusion_scores: Dict[int, float] = {}
        max_dense = max([s for _, s in dense_results]) if dense_results else 1.0
        max_sparse = max([s for _, s in sparse_results]) if sparse_results else 1.0

        for idx in doc_indices:
            d_score = (dense_map.get(idx, 0.0) / max_dense) if max_dense > 0 else 0.0
            s_score = (sparse_map.get(idx, 0.0) / max_sparse) if max_sparse > 0 else 0.0
            fusion_scores[idx] = weight_dense * d_score + weight_sparse * s_score

    return fusion_scores


# ===========================================================================
# Main HybridRetriever
# ===========================================================================

class HybridRetriever:
    """Complete hybrid retrieval system combining dense + sparse retrieval
    with RRF fusion, weighted combination, and fallback strategies.

    Usage:
        retriever = HybridRetriever(scenario=Scenario.GENERAL)
        retriever.index(documents)
        results = retriever.search("什么是自然语言处理？", top_k=10)
    """

    def __init__(
        self,
        scenario: Scenario = Scenario.GENERAL,
        dense_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        normalize_scores: bool = True,
        device: str = 'cpu',
    ):
        """
        Args:
            scenario: GENERAL (dense:sparse=7:3) or PROFESSIONAL (4:6).
            dense_model: SentenceTransformer model name.
            bm25_k1, bm25_b: BM25 parameters.
            rrf_k: RRF smoothing constant.
            dense_weight, sparse_weight: Override scenario defaults.
            normalize_scores: Apply Min-Max normalization.
            device: 'cpu' or 'cuda'.
        """
        self.scenario = scenario

        # Set weights based on scenario
        if dense_weight is not None and sparse_weight is not None:
            self.dense_weight = dense_weight
            self.sparse_weight = sparse_weight
        elif scenario == Scenario.PROFESSIONAL:
            self.dense_weight = 0.4
            self.sparse_weight = 0.6
        else:  # GENERAL
            self.dense_weight = 0.7
            self.sparse_weight = 0.3

        self.rrf_k = rrf_k
        self.normalize_scores = normalize_scores

        # Initialize retrievers
        self.dense_retriever = DenseRetriever(model_name=dense_model, device=device)
        self.sparse_retriever = SparseRetriever(k1=bm25_k1, b=bm25_b)

        self.documents: List[Dict[str, str]] = []
        self.is_indexed: bool = False

    def index(self, documents: List[Dict[str, str]]) -> None:
        """Index a list of documents.

        Each document: {'doc_id': str, 'text': str, ...metadata...}
        """
        if not documents:
            print("[WARN] Empty document list, skipping index.")
            return

        self.documents = documents

        print(f"Indexing {len(documents)} documents...")
        t0 = time.time()

        self.dense_retriever.index(documents)
        self.sparse_retriever.index(documents)

        self.is_indexed = True
        elapsed = time.time() - t0
        print(f"Indexing complete in {elapsed:.2f}s "
              f"(dense: {self.dense_retriever.doc_embeddings.shape if self.dense_retriever.doc_embeddings is not None else 'N/A'}, "
              f"sparse: {len(self.sparse_retriever._corpus)} docs)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        return_details: bool = False,
    ) -> List[RetrievalResult]:
        """Hybrid search with RRF fusion.

        Args:
            query: Search query.
            top_k: Number of final results to return.
            dense_top_k: Candidates from dense retrieval.
            sparse_top_k: Candidates from sparse retrieval.
            return_details: Include individual dense/sparse scores.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        if not self.is_indexed:
            return []

        # Step 1: Dense retrieval
        dense_results = self.dense_retriever.search(query, top_k=dense_top_k)

        # Step 2: Sparse retrieval (BM25)
        sparse_results = self.sparse_retriever.search(query, top_k=sparse_top_k)

        # Step 3: Fallback if BM25 returns empty
        if not sparse_results:
            print(f"[FALLBACK] BM25 returned empty for query '{query[:50]}...'. "
                  "Using dense-only results.")
            return self._dense_only_results(query, dense_results, top_k)

        # Step 4: Weighted hybrid fusion
        fusion_scores = weighted_hybrid_fusion(
            dense_results,
            sparse_results,
            weight_dense=self.dense_weight,
            weight_sparse=self.sparse_weight,
            k=self.rrf_k,
            normalize=self.normalize_scores,
        )

        # Step 5: Sort and take top-k
        sorted_items = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_k]

        # Step 6: Build result objects
        dense_map = {idx: score for idx, score in dense_results}
        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_results, 1)}
        sparse_map = {idx: score for idx, score in sparse_results}
        sparse_rank = {idx: r for r, (idx, _) in enumerate(sparse_results, 1)}

        results: List[RetrievalResult] = []
        for rank, (doc_idx, fusion_score) in enumerate(top_items, 1):
            doc = self.documents[doc_idx] if doc_idx < len(self.documents) else {}
            result = RetrievalResult(
                doc_id=doc.get('doc_id', f'doc_{doc_idx}'),
                text=doc.get('text', ''),
                score=fusion_score,
                dense_score=dense_map.get(doc_idx, 0.0),
                sparse_score=sparse_map.get(doc_idx, 0.0),
                rrf_score=fusion_score,
                rank_dense=dense_rank.get(doc_idx, -1),
                rank_sparse=sparse_rank.get(doc_idx, -1),
                rank_hybrid=rank,
                metadata={k: v for k, v in doc.items()
                         if k not in ('doc_id', 'text')},
            )
            results.append(result)

        return results

    def _dense_only_results(
        self, query: str, dense_results: List[Tuple[int, float]], top_k: int
    ) -> List[RetrievalResult]:
        """Fallback: build results from dense-only scores."""
        results: List[RetrievalResult] = []
        for rank, (doc_idx, score) in enumerate(dense_results[:top_k], 1):
            doc = self.documents[doc_idx] if doc_idx < len(self.documents) else {}
            results.append(RetrievalResult(
                doc_id=doc.get('doc_id', f'doc_{doc_idx}'),
                text=doc.get('text', ''),
                score=score,
                dense_score=score,
                sparse_score=0.0,
                rrf_score=score,
                rank_dense=rank,
                rank_sparse=-1,
                rank_hybrid=rank,
                metadata={k: v for k, v in doc.items()
                         if k not in ('doc_id', 'text')},
            ))
        return results

    def benchmark(
        self,
        queries: List[str],
        relevant_doc_ids: List[Set[str]],
        top_k: int = 10,
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark different retrieval strategies.

        Args:
            queries: List of test queries.
            relevant_doc_ids: List of sets of relevant doc_ids for each query.
            top_k: Top-K for evaluation.

        Returns:
            Dict mapping method name → BenchmarkResult.
        """
        results: Dict[str, BenchmarkResult] = {
            'dense_only': BenchmarkResult(method='dense_only'),
            'sparse_only': BenchmarkResult(method='sparse_only'),
            'hybrid_rrf': BenchmarkResult(method='hybrid_rrf'),
        }

        for qi, (query, relevant) in enumerate(zip(queries, relevant_doc_ids)):
            # --- Dense only ---
            t0 = time.time()
            d_results = self.dense_retriever.search(query, top_k=top_k)
            d_latency = (time.time() - t0) * 1000
            d_doc_ids = [self.documents[idx].get('doc_id', '')
                        for idx, _ in d_results[:top_k]]
            self._update_benchmark(results['dense_only'], d_doc_ids, relevant,
                                  d_latency, top_k)

            # --- Sparse only ---
            t0 = time.time()
            s_results = self.sparse_retriever.search(query, top_k=top_k)
            s_latency = (time.time() - t0) * 1000
            s_doc_ids = [self.documents[idx].get('doc_id', '')
                        for idx, _ in s_results[:top_k]] if s_results else []
            self._update_benchmark(results['sparse_only'], s_doc_ids, relevant,
                                  s_latency, top_k)

            # --- Hybrid ---
            t0 = time.time()
            h_results = self.search(query, top_k=top_k)
            h_latency = (time.time() - t0) * 1000
            h_doc_ids = [r.doc_id for r in h_results[:top_k]]
            self._update_benchmark(results['hybrid_rrf'], h_doc_ids, relevant,
                                  h_latency, top_k)

        # Average metrics
        n = len(queries)
        for method, br in results.items():
            br.num_queries = n
            if n > 0:
                br.precision_at_5 /= n
                br.precision_at_10 /= n
                br.recall_at_10 /= n
                br.mrr /= n
                br.ndcg_at_10 /= n
                br.avg_latency_ms /= n

        return results

    def _update_benchmark(
        self, br: BenchmarkResult, retrieved_ids: List[str],
        relevant: Set[str], latency_ms: float, top_k: int
    ) -> None:
        """Update running benchmark metrics."""
        retrieved_5 = retrieved_ids[:5]
        retrieved_10 = retrieved_ids[:10]

        relevant_set = set(relevant)

        # Precision
        br.precision_at_5 += len(set(retrieved_5) & relevant_set) / max(len(retrieved_5), 1)
        br.precision_at_10 += len(set(retrieved_10) & relevant_set) / max(len(retrieved_10), 1)

        # Recall
        br.recall_at_10 += len(set(retrieved_10) & relevant_set) / max(len(relevant_set), 1)

        # MRR
        for rank, did in enumerate(retrieved_ids, 1):
            if did in relevant_set:
                br.mrr += 1.0 / rank
                break

        # NDCG@10 (simplified)
        dcg = 0.0
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), 10)))
        for rank, did in enumerate(retrieved_10, 1):
            if did in relevant_set:
                dcg += 1.0 / math.log2(rank + 1)
        br.ndcg_at_10 += dcg / max(idcg, 1e-8)

        br.avg_latency_ms += latency_ms


# ===========================================================================
# Utility: print results
# ===========================================================================

def print_results(results: List[RetrievalResult], title: str = "Results") -> None:
    """Pretty-print retrieval results."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Doc ID':<20} {'Score':<10} {'Dense':<10} {'Sparse':<10} {'Text'}")
    print(f"{'-'*70}")
    for r in results:
        text_preview = r.text[:50].replace('\n', ' ') + ('...' if len(r.text) > 50 else '')
        print(f"{r.rank_hybrid:<6} {r.doc_id:<20} {r.score:<10.4f} "
              f"{r.dense_score:<10.4f} {r.sparse_score:<10.4f} {text_preview}")


# ===========================================================================
# __main__: Demo
# ===========================================================================

# Need re import at top for Chinese tokenization
import re


def _create_sample_documents() -> List[Dict[str, str]]:
    """Create sample document collection for testing."""
    return [
        {"doc_id": "doc_nlp_intro", "text":
         "自然语言处理是人工智能的一个重要分支，研究计算机如何理解和生成人类语言。"
         "NLP包括机器翻译、情感分析、文本摘要、问答系统等应用。"},
        {"doc_id": "doc_ml_basics", "text":
         "机器学习是人工智能的核心技术，通过数据学习模式而无需显式编程。"
         "监督学习、无监督学习和强化学习是三种主要学习范式。"},
        {"doc_id": "doc_deep_learning", "text":
         "深度学习使用多层神经网络进行层次化特征学习。卷积神经网络适合图像处理，"
         "循环神经网络适合序列数据，Transformer架构在NLP中表现优异。"},
        {"doc_id": "doc_data_preprocessing", "text":
         "数据预处理包括清洗、标准化、特征工程等步骤。高质量数据是模型性能的基础。"
         "数据增强技术可以扩充训练集，提高模型泛化能力。"},
        {"doc_id": "doc_model_evaluation", "text":
         "模型评估指标包括准确率、精确率、召回率和F1分数。交叉验证是可靠的评估方法，"
         "通过多次训练测试减少方差。混淆矩阵可以直观展示分类结果。"},
        {"doc_id": "doc_transfer_learning", "text":
         "迁移学习利用预训练模型的知识应用到新任务，降低训练成本。在NLP中，"
         "BERT和GPT等模型可以通过微调适应各种下游任务。预训练加微调是主流范式。"},
        {"doc_id": "doc_attention_mechanism", "text":
         "注意力机制是Transformer的核心，允许模型关注相关信息。自注意力捕获序列内部依赖，"
         "多头注意力增强表达能力。缩放点积注意力是最常用的形式。"},
        {"doc_id": "doc_rag_system", "text":
         "RAG检索增强生成系统结合了信息检索和文本生成。通过检索相关文档作为上下文，"
         "提高生成的准确性和事实性。向量数据库和混合检索是RAG的关键技术。"},
        {"doc_id": "doc_hyperparameter", "text":
         "超参数调优对模型性能影响显著。学习率、批量大小、网络层数是关键超参数。"
         "网格搜索和贝叶斯优化是常用调优方法。自动机器学习可以自动搜索最优配置。"},
        {"doc_id": "doc_transformer_arch", "text":
         "Transformer架构由编码器和解码器组成。位置编码提供序列位置信息。"
         "残差连接和层归一化是稳定训练的关键。多头自注意力是核心创新。"},
    ]


if __name__ == '__main__':
    print("=" * 60)
    print("Hybrid Retrieval with RRF Fusion - Self-Test")
    print("=" * 60)

    # --- Create index ---
    docs = _create_sample_documents()
    print(f"\nLoaded {len(docs)} sample documents.")

    # Test General scenario (7:3 weights)
    print(f"\n{'─'*60}")
    print("Test 1: GENERAL Scenario (dense:sparse = 7:3)")
    print(f"{'─'*60}")
    retriever_general = HybridRetriever(scenario=Scenario.GENERAL)
    retriever_general.index(docs)

    queries_general = [
        "什么是自然语言处理？",
        "深度学习的主要架构有哪些？",
        "如何评估机器学习模型？",
    ]

    for q in queries_general:
        results = retriever_general.search(q, top_k=5)
        print_results(results, f"Query: {q}")

    # Test Professional scenario (4:6 weights)
    print(f"\n{'─'*60}")
    print("Test 2: PROFESSIONAL Scenario (dense:sparse = 4:6)")
    print(f"{'─'*60}")
    retriever_pro = HybridRetriever(scenario=Scenario.PROFESSIONAL)
    retriever_pro.index(docs)

    for q in queries_general[:2]:
        results = retriever_pro.search(q, top_k=5)
        print_results(results, f"Query: {q}")

    # Test Fallback scenario (query with no BM25 hits)
    print(f"\n{'─'*60}")
    print("Test 3: Fallback when BM25 returns empty")
    print(f"{'─'*60}")
    fallback_query = "xyzzy 不存在的罕见词汇查询测试"
    results_fb = retriever_general.search(fallback_query, top_k=3)
    print_results(results_fb, f"Query: {fallback_query}")
    if all(r.sparse_score == 0.0 for r in results_fb):
        print("  >> Fallback triggered: dense-only results (as expected)")

    # Benchmark comparison
    print(f"\n{'─'*60}")
    print("Test 4: Benchmark Comparison")
    print(f"{'─'*60}")

    benchmark_queries = [
        "自然语言处理的应用有哪些？",
        "Transformer架构的核心组件是什么？",
        "如何提高模型泛化能力？",
    ]
    # Define relevant docs for each query
    benchmark_relevant = [
        {"doc_nlp_intro", "doc_rag_system"},
        {"doc_attention_mechanism", "doc_transformer_arch", "doc_deep_learning"},
        {"doc_data_preprocessing", "doc_transfer_learning", "doc_hyperparameter"},
    ]

    bench_results = retriever_general.benchmark(
        benchmark_queries, benchmark_relevant, top_k=10
    )

    print(f"\n{'Method':<15} {'P@5':<10} {'P@10':<10} {'R@10':<10} "
          f"{'MRR':<10} {'NDCG@10':<10} {'Latency':<10}")
    print(f"{'-'*75}")
    for method, br in bench_results.items():
        print(f"{br.method:<15} {br.precision_at_5:<10.4f} {br.precision_at_10:<10.4f} "
              f"{br.recall_at_10:<10.4f} {br.mrr:<10.4f} {br.ndcg_at_10:<10.4f} "
              f"{br.avg_latency_ms:<10.1f}ms")

    # Detailed result inspection
    print(f"\n{'─'*60}")
    print("Test 5: Detailed scores for a single query")
    print(f"{'─'*60}")
    detailed = retriever_general.search(
        "注意力机制是怎么工作的？", top_k=5, return_details=True
    )
    for r in detailed:
        print(f"\n  Doc: {r.doc_id}")
        print(f"    Hybrid Score:  {r.score:.4f} (rank={r.rank_hybrid})")
        print(f"    Dense  Score:  {r.dense_score:.4f} (rank={r.rank_dense})")
        print(f"    Sparse Score:  {r.sparse_score:.4f} (rank={r.rank_sparse})")
        print(f"    Text:          {r.text[:80]}...")

    print(f"\n{'='*60}")
    print("Hybrid Retrieval self-test completed successfully!")
    print(f"{'='*60}")
