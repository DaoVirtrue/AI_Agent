#!/usr/bin/env python3
"""
Phase 04 Module 14: Parent-Child Retrieval
=============================================
- Two-stage retrieval: child chunks (256 tokens) for precision
  → parent chunks (1024 tokens) for context
- Metadata linking between parent and child chunks
- ParentChildRetriever class as a retriever wrapper
- Comparison: standard retrieval vs parent-child retrieval quality
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ===========================================================================
# Data Classes
# ===========================================================================

class ChunkKind(Enum):
    PARENT = "parent"  # 1024 tokens — provides full context
    CHILD = "child"    # 256 tokens — used for precise retrieval


@dataclass
class DocumentChunk:
    """A chunk with parent-child relationships."""
    chunk_id: str
    text: str
    kind: ChunkKind
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    doc_id: str = ""
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A retrieval result with parent-child context."""
    child_chunk: DocumentChunk
    parent_chunk: Optional[DocumentChunk] = None
    child_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0


@dataclass
class RetrievalComparison:
    """Comparison of standard vs parent-child retrieval."""
    query: str
    standard_top_k_ids: List[str] = field(default_factory=list)
    parent_child_top_k_ids: List[str] = field(default_factory=list)
    standard_precision: float = 0.0
    parent_child_precision: float = 0.0
    context_coverage_standard: float = 0.0
    context_coverage_parent_child: float = 0.0


# ===========================================================================
# Chunk Builder: Create parent-child chunk hierarchy
# ===========================================================================

class ChunkHierarchyBuilder:
    """Build a two-level parent-child chunk hierarchy from documents.

    Strategy:
    - Parent chunks: ~1024 tokens (for context)
    - Child chunks: ~256 tokens (for precise retrieval)
    - Each parent has 3-4 children
    - Overlap between siblings to prevent boundary issues
    """

    def __init__(
        self,
        parent_size_tokens: int = 1024,
        child_size_tokens: int = 256,
        children_per_parent: int = 4,
        overlap_tokens: int = 32,
    ):
        self.parent_size_tokens = parent_size_tokens
        self.child_size_tokens = child_size_tokens
        self.children_per_parent = children_per_parent
        self.overlap_tokens = overlap_tokens

    def build(self, doc_id: str, text: str) -> List[DocumentChunk]:
        """Build parent-child chunk hierarchy for a document.

        Returns:
            List of all chunks (both parent and child).
        """
        chunks: List[DocumentChunk] = []
        sentences = self._split_sentences(text)

        if not sentences:
            return chunks

        # Step 1: Create parent chunks
        parent_texts = self._chunk_by_tokens(
            sentences, self.parent_size_tokens, overlap=0
        )

        parent_chunks: List[DocumentChunk] = []
        for pi, ptext in enumerate(parent_texts):
            parent = DocumentChunk(
                chunk_id=f"{doc_id}_parent_{pi:03d}",
                text=ptext,
                kind=ChunkKind.PARENT,
                doc_id=doc_id,
                chunk_index=pi,
            )
            parent_chunks.append(parent)
            chunks.append(parent)

        # Step 2: Create child chunks within each parent
        child_index = 0
        for parent in parent_chunks:
            parent_sentences = self._split_sentences(parent.text)
            child_texts = self._chunk_by_tokens(
                parent_sentences,
                self.child_size_tokens,
                overlap=self.overlap_tokens,
                max_chunks=self.children_per_parent,
            )

            for ct in child_texts:
                child = DocumentChunk(
                    chunk_id=f"{doc_id}_child_{child_index:03d}",
                    text=ct,
                    kind=ChunkKind.CHILD,
                    parent_id=parent.chunk_id,
                    doc_id=doc_id,
                    chunk_index=child_index,
                )
                parent.child_ids.append(child.chunk_id)
                chunks.append(child)
                child_index += 1

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        text = text.replace('\n', ' ')
        pattern = r'(?<=[。！？.!?；;])\s*'
        raw = re.split(pattern, text)
        return [s.strip() for s in raw if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4.0)

    def _chunk_by_tokens(
        self,
        sentences: List[str],
        max_tokens: int,
        overlap: int = 0,
        max_chunks: Optional[int] = None,
    ) -> List[str]:
        """Chunk sentences into token-limited groups."""
        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)

            if current_tokens + sent_tokens > max_tokens and current:
                chunks.append(' '.join(current))
                # Overlap: keep last sentence if overlap requested
                if overlap > 0 and len(current) > 1:
                    current = [current[-1]]
                    current_tokens = self._estimate_tokens(current[0])
                else:
                    current = []
                    current_tokens = 0

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(' '.join(current))

        if max_chunks and len(chunks) > max_chunks:
            # Merge excess chunks
            while len(chunks) > max_chunks:
                chunks[-2] = chunks[-2] + ' ' + chunks[-1]
                chunks.pop()

        return chunks


# ===========================================================================
# Simple Embedding Retriever (for demo)
# ===========================================================================

class SimpleEmbeddingRetriever:
    """A minimal dense retriever for demonstration purposes.

    Uses character n-gram Jaccard similarity as a lightweight proxy
    for embeddings. In production, replace with actual vector search.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self._child_indices: List[int] = []
        self._parent_by_id: Dict[str, DocumentChunk] = {}

    def index(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks for retrieval."""
        self.chunks = chunks
        self._child_indices = [
            i for i, c in enumerate(chunks) if c.kind == ChunkKind.CHILD
        ]
        self._parent_by_id = {
            c.chunk_id: c for c in chunks if c.kind == ChunkKind.PARENT
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        chunk_kind: Optional[ChunkKind] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for relevant chunks.

        Args:
            query: Search query.
            top_k: Number of results.
            chunk_kind: Filter by chunk kind (None = all).

        Returns:
            List of (chunk, relevance_score).
        """
        scores: List[Tuple[int, float]] = []

        indices = range(len(self.chunks))
        if chunk_kind == ChunkKind.CHILD:
            indices = self._child_indices
        elif chunk_kind == ChunkKind.PARENT:
            indices = [i for i, c in enumerate(self.chunks)
                      if c.kind == ChunkKind.PARENT]

        for i in indices:
            chunk = self.chunks[i]
            score = self._compute_similarity(query, chunk.text)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]
        return [(self.chunks[i], score) for i, score in top]

    def _compute_similarity(self, query: str, text: str) -> float:
        """Compute n-gram Jaccard similarity between query and text."""
        def ngrams(s: str, n: int = 2) -> Set[str]:
            s = s.lower()
            return {s[i:i+n] for i in range(len(s)-n+1)}

        q_ngrams = ngrams(query)
        t_ngrams = ngrams(text)

        if not q_ngrams or not t_ngrams:
            return 0.0

        intersection = q_ngrams & t_ngrams
        union = q_ngrams | t_ngrams

        # TF-IDF-like weighting: prefer matches with longer query ngrams
        weighted = len(intersection) / len(union) if union else 0.0

        # Bonus for exact substring matches
        query_lower = query.lower()
        text_lower = text.lower()
        if query_lower in text_lower:
            weighted += 0.1

        return min(weighted, 1.0)


# ===========================================================================
# Parent-Child Retriever
# ===========================================================================

class ParentChildRetriever:
    """Two-stage parent-child retrieval.

    Stage 1: Retrieve using child chunks (256 tokens) — high precision
    Stage 2: Expand to parent chunks (1024 tokens) — full context

    Benefits:
    - Child chunks provide precise matching (small, focused)
    - Parent chunks provide complete context (large, comprehensive)
    - Reduces "lost in middle" by keeping parent chunks concise
    """

    def __init__(
        self,
        retriever: SimpleEmbeddingRetriever,
        child_top_k: int = 10,
        max_parents: int = 5,
        child_weight: float = 0.6,
        parent_weight: float = 0.4,
    ):
        """
        Args:
            retriever: The base retriever (with indexed chunks).
            child_top_k: Number of child chunks to retrieve in Stage 1.
            max_parents: Maximum number of parent chunks to include.
            child_weight: Weight for child scores in combined ranking.
            parent_weight: Weight for parent context scores.
        """
        self.retriever = retriever
        self.child_top_k = child_top_k
        self.max_parents = max_parents
        self.child_weight = child_weight
        self.parent_weight = parent_weight

    def search(
        self,
        query: str,
        final_top_k: int = 5,
        return_details: bool = False,
    ) -> List[RetrievalResult]:
        """Two-stage parent-child retrieval.

        Args:
            query: Search query.
            final_top_k: Number of final results.
            return_details: Include detailed scores.

        Returns:
            List of RetrievalResult with child+parent pairs.
        """
        # Stage 1: Retrieve child chunks (precise matching)
        child_results = self.retriever.search(
            query, top_k=self.child_top_k, chunk_kind=ChunkKind.CHILD
        )

        if not child_results:
            return []

        # Stage 2: Map children to parents and re-rank
        parent_scores: Dict[str, Tuple[DocumentChunk, List[Tuple[DocumentChunk, float]]]] = {}

        for child, child_score in child_results:
            parent_id = child.parent_id
            if parent_id is None:
                continue

            parent = self.retriever._parent_by_id.get(parent_id)
            if parent is None:
                continue

            if parent_id not in parent_scores:
                parent_scores[parent_id] = (parent, [])
            parent_scores[parent_id][1].append((child, child_score))

        # Compute combined scores
        combined: List[Tuple[str, float, DocumentChunk, DocumentChunk, float]] = []
        for parent_id, (parent, children) in parent_scores.items():
            # Child score: max or mean of child scores
            child_scores = [s for _, s in children]
            best_child_score = max(child_scores) if child_scores else 0.0
            mean_child_score = sum(child_scores) / len(child_scores) if child_scores else 0.0

            # Best child chunk
            best_child = max(children, key=lambda x: x[1])[0]

            # Combined score: weighted combination of child and parent factors
            combined_score = (
                self.child_weight * best_child_score +
                self.parent_weight * mean_child_score
            )

            combined.append((
                parent_id, combined_score, parent, best_child, best_child_score,
            ))

        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)

        # Top-K results
        top_combined = combined[:min(self.max_parents, final_top_k)]

        results: List[RetrievalResult] = []
        for rank, (pid, comb_score, parent, best_child, child_score) in enumerate(
            top_combined, 1
        ):
            results.append(RetrievalResult(
                child_chunk=best_child,
                parent_chunk=parent,
                child_score=child_score,
                combined_score=comb_score,
                rank=rank,
            ))

        return results

    def get_full_context(
        self,
        result: RetrievalResult,
        include_siblings: bool = True,
    ) -> str:
        """Get the full context for a retrieval result.

        Args:
            result: A retrieval result from search().
            include_siblings: Whether to include sibling child chunks.

        Returns:
            Full context text suitable for LLM generation.
        """
        parts: List[str] = []

        if result.parent_chunk:
            parts.append(f"[Context from: {result.parent_chunk.doc_id}]\n")
            parts.append(result.parent_chunk.text)

            if include_siblings and result.parent_chunk.child_ids:
                # Add sibling chunks for additional context
                sibling_texts: List[str] = []
                for cid in result.parent_chunk.child_ids:
                    for chunk in self.retriever.chunks:
                        if chunk.chunk_id == cid and chunk.chunk_id != result.child_chunk.chunk_id:
                            sibling_texts.append(chunk.text)
                            break

                if sibling_texts:
                    parts.append("\n\n[Additional context from siblings:]")
                    parts.extend(sibling_texts)
        else:
            parts.append(result.child_chunk.text)

        return '\n'.join(parts)


# ===========================================================================
# Comparison Tools
# ===========================================================================

class RetrievalComparator:
    """Compare standard retrieval vs parent-child retrieval."""

    def __init__(
        self,
        standard_retriever: SimpleEmbeddingRetriever,
        parent_child_retriever: ParentChildRetriever,
    ):
        self.standard_retriever = standard_retriever
        self.parent_child_retriever = parent_child_retriever

    def compare(
        self,
        query: str,
        relevant_chunk_ids: Set[str],
        top_k: int = 5,
    ) -> RetrievalComparison:
        """Compare retrieval methods on a single query."""
        comp = RetrievalComparison(query=query)

        # Standard retrieval (all chunks)
        std_results = self.standard_retriever.search(query, top_k=top_k)
        std_ids = [c.chunk_id for c, _ in std_results]
        comp.standard_top_k_ids = std_ids

        # Parent-child retrieval
        pc_results = self.parent_child_retriever.search(query, final_top_k=top_k)
        pc_ids = [r.child_chunk.chunk_id for r in pc_results]
        comp.parent_child_top_k_ids = pc_ids

        # Precision: how many retrieved chunks are relevant
        if std_ids:
            comp.standard_precision = len(set(std_ids) & relevant_chunk_ids) / len(std_ids)
        if pc_ids:
            comp.parent_child_precision = len(set(pc_ids) & relevant_chunk_ids) / len(pc_ids)

        # Context coverage: how much total context is provided
        std_total_tokens = sum(
            self._estimate_tokens(c.text) for c, _ in std_results
        )
        pc_total_tokens = sum(
            self._estimate_tokens(r.parent_chunk.text) if r.parent_chunk
            else self._estimate_tokens(r.child_chunk.text)
            for r in pc_results
        )
        comp.context_coverage_standard = std_total_tokens
        comp.context_coverage_parent_child = pc_total_tokens

        return comp

    def _estimate_tokens(self, text: str) -> int:
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4.0)


# ===========================================================================
# __main__: Demo
# ===========================================================================

_SAMPLE_DOCS = {
    "doc_ml_basics": (
        "机器学习是人工智能的核心技术之一。它通过从数据中学习模式和规律来改进性能。"
        "监督学习使用标注数据进行训练。常见的监督学习算法包括线性回归、决策树和支持向量机。"
        "无监督学习在没有标签的情况下发现数据结构。聚类和降维是无监督学习的典型任务。"
        "强化学习通过与环境的交互学习最优策略。智能体通过试错获得奖励信号来改进行为。"
        "深度强化学习结合了深度神经网络和强化学习。AlphaGo就是深度强化学习的成功案例。"
        "迁移学习允许将预训练模型的知识应用到新任务上。这大大降低了训练成本和时间。"
        "在NLP领域，BERT和GPT等预训练模型可以通过微调适应各种下游任务。"
        "模型评估是确保模型质量的关键环节。常用指标包括准确率、精确率、召回率和F1分数。"
        "交叉验证可以更可靠地评估模型性能。K折交叉验证是最常用的方法。"
        "超参数调优对模型性能有显著影响。网格搜索和贝叶斯优化是常用方法。"
        "自动机器学习(AutoML)可以自动完成特征工程、模型选择和超参数调优。"
    ),
    "doc_neural_networks": (
        "神经网络是深度学习的核心。它由多层神经元组成，模拟人脑的信息处理方式。"
        "激活函数为神经网络引入非线性。ReLU是最常用的激活函数，计算简单且效果好。"
        "反向传播算法是训练神经网络的关键。它通过链式法则计算梯度来更新网络参数。"
        "卷积神经网络(CNN)在计算机视觉领域表现出色。它使用卷积核提取局部特征。"
        "池化层用于降低特征图的尺寸。最大池化和平均池化是两种常见方式。"
        "循环神经网络(RNN)适合处理序列数据。但标准RNN存在梯度消失问题。"
        "LSTM和GRU通过门控机制解决了长距离依赖问题。它们在NLP和时间序列分析中广泛应用。"
        "Transformer架构完全基于注意力机制。它抛弃了循环结构，实现了并行计算。"
        "自注意力机制可以捕获序列内部的长距离依赖关系。这是Transformer成功的关键。"
        "多头注意力允许模型同时关注不同子空间的信息。这增强了模型的表达能力。"
        "预训练加微调已成为NLP的主流范式。BERT、GPT和T5都遵循这个范式。"
    ),
    "doc_rag_systems": (
        "RAG(检索增强生成)系统结合了信息检索和文本生成。它通过检索外部知识来增强回答质量。"
        "向量数据库是RAG系统的核心组件。它将文档编码为向量进行高效相似度搜索。"
        "Milvus和Pinecone是两个流行的向量数据库。它们支持十亿级别的向量检索。"
        "文档切分策略直接影响检索质量。基于语义的切分比固定长度切分效果更好。"
        "父子切片策略是最佳实践之一。小切片用于精确检索，大切片提供完整上下文。"
        "混合检索结合了稠密向量检索和稀疏BM25检索。这种融合策略能显著提升召回率。"
        "重排序是提升精度的关键步骤。BGE-Reranker和Cohere Rerank是常用的重排序模型。"
        "查询优化可以改善用户原始查询的质量。包括查询重写、扩展和分解。"
        "多轮对话中的指代消解是RAG系统的重要功能。它确保系统理解对话上下文。"
        "相似度阈值过滤可以排除低质量检索结果。通用场景建议阈值为0.75。"
        "RAG系统的评估指标包括准确率、召回率、MRR和NDCG。端到端评估需要考虑生成质量。"
    ),
}


if __name__ == '__main__':
    print("=" * 60)
    print("Parent-Child Retrieval - Self-Test")
    print("=" * 60)

    # --- Step 1: Build chunk hierarchy ---
    print(f"\n[Step 1] Building Parent-Child Chunk Hierarchy")
    print("-" * 40)

    builder = ChunkHierarchyBuilder(
        parent_size_tokens=1024,
        child_size_tokens=256,
        children_per_parent=3,
        overlap_tokens=30,
    )

    all_chunks: List[DocumentChunk] = []
    for doc_id, text in _SAMPLE_DOCS.items():
        chunks = builder.build(doc_id, text)
        all_chunks.extend(chunks)
        parent_count = sum(1 for c in chunks if c.kind == ChunkKind.PARENT)
        child_count = sum(1 for c in chunks if c.kind == ChunkKind.CHILD)
        print(f"\n  {doc_id}: {len(text)} chars → {parent_count} parents + {child_count} children")

    total_parents = sum(1 for c in all_chunks if c.kind == ChunkKind.PARENT)
    total_children = sum(1 for c in all_chunks if c.kind == ChunkKind.CHILD)
    print(f"\n  Total: {len(all_chunks)} chunks ({total_parents} parents + {total_children} children)")

    # --- Step 2: Index chunks ---
    print(f"\n[Step 2] Indexing Chunks")
    print("-" * 40)

    retriever = SimpleEmbeddingRetriever()
    retriever.index(all_chunks)
    print(f"  Indexed {len(all_chunks)} chunks")

    # --- Step 3: Standard retrieval ---
    print(f"\n[Step 3] Standard Retrieval (all chunks)")
    print("-" * 40)

    queries = [
        "什么是迁移学习？",
        "注意力机制如何工作？",
        "RAG系统的核心组件有哪些？",
    ]

    for query in queries:
        results = retriever.search(query, top_k=3)
        print(f"\n  Query: {query}")
        for rank, (chunk, score) in enumerate(results, 1):
            kind_tag = f"[{chunk.kind.value.upper()}]"
            print(f"    {rank}. {chunk.chunk_id} {kind_tag} (score={score:.4f})")
            print(f"       {chunk.text[:80]}...")

    # --- Step 4: Parent-child retrieval ---
    print(f"\n[Step 4] Parent-Child Retrieval (child→parent)")
    print("-" * 40)

    pc_retriever = ParentChildRetriever(
        retriever=retriever,
        child_top_k=8,
        max_parents=3,
        child_weight=0.6,
        parent_weight=0.4,
    )

    for query in queries:
        results = pc_retriever.search(query, final_top_k=3)
        print(f"\n  Query: {query}")
        for rr in results:
            print(f"    [{rr.rank}] Child: {rr.child_chunk.chunk_id} "
                  f"(score={rr.child_score:.4f})")
            print(f"         Parent: {rr.parent_chunk.chunk_id if rr.parent_chunk else 'N/A'}")
            print(f"         Combined score: {rr.combined_score:.4f}")
            print(f"         Child text: {rr.child_chunk.text[:80]}...")
            if rr.parent_chunk:
                print(f"         Parent text: {rr.parent_chunk.text[:80]}...")

    # --- Step 5: Full context generation ---
    print(f"\n[Step 5] Full Context Generation")
    print("-" * 40)

    query = "什么是迁移学习？"
    results = pc_retriever.search(query, final_top_k=1)
    if results:
        context = pc_retriever.get_full_context(results[0], include_siblings=True)
        print(f"\n  Full context for top-1 result:")
        print(f"  {'─'*50}")
        for line in context.split('\n')[:8]:
            print(f"  {line}")
        print(f"  ... ({len(context)} total chars, ~{builder._estimate_tokens(context)} tokens)")

    # --- Step 6: Comparison ---
    print(f"\n[Step 6] Standard vs Parent-Child Comparison")
    print("-" * 40)

    comparator = RetrievalComparator(retriever, pc_retriever)

    # Define some "relevant" child chunks for evaluation
    relevant_ids: Dict[str, Set[str]] = {
        "什么是迁移学习？": {
            "doc_ml_basics_child_002", "doc_ml_basics_child_003",
            "doc_ml_basics_child_001",
        },
        "注意力机制如何工作？": {
            "doc_neural_networks_child_003", "doc_neural_networks_child_004",
            "doc_neural_networks_child_005",
        },
        "RAG系统的核心组件有哪些？": {
            "doc_rag_systems_child_001", "doc_rag_systems_child_002",
            "doc_rag_systems_child_000",
        },
    }

    for query, relevant in relevant_ids.items():
        comp = comparator.compare(query, relevant, top_k=5)
        print(f"\n  Query: {query}")
        print(f"    Standard Precision:      {comp.standard_precision:.3f}")
        print(f"    Parent-Child Precision:  {comp.parent_child_precision:.3f}")
        print(f"    Standard Context (tokens):   {comp.context_coverage_standard}")
        print(f"    Parent-Child Context (tokens): {comp.context_coverage_parent_child}")
        improvement = (comp.parent_child_precision - comp.standard_precision) / max(comp.standard_precision, 1e-8) * 100
        print(f"    Precision Improvement:   {improvement:+.1f}%")

    # --- Test 7: Edge cases ---
    print(f"\n[Step 7] Edge Cases")
    print("-" * 40)

    # Empty query
    empty_results = pc_retriever.search("", final_top_k=3)
    print(f"  Empty query: {len(empty_results)} results (expected 0)")

    # Query with no matches
    no_match = pc_retriever.search("xyzzy不存在的词汇abcdefg", final_top_k=3)
    print(f"  No-match query: {len(no_match)} results")

    # Single document
    single_builder = ChunkHierarchyBuilder(parent_size_tokens=512, child_size_tokens=128)
    single_chunks = single_builder.build("tiny_doc", "这是一个很短的文档。")
    print(f"  Single-sentence doc: {len(single_chunks)} chunks "
          f"({sum(1 for c in single_chunks if c.kind==ChunkKind.PARENT)}P "
          f"+ {sum(1 for c in single_chunks if c.kind==ChunkKind.CHILD)}C)")

    print(f"\n{'='*60}")
    print("Parent-Child Retrieval self-test completed!")
    print(f"{'='*60}")
