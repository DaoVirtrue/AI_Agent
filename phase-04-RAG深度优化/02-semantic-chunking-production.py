#!/usr/bin/env python3
"""
Phase 04 Module 02: Semantic Chunking (Production)
====================================================
Production-quality semantic chunking with:
- Sentence embedding + similarity dip detection
- Configurable thresholds: general 85-90%, professional 90-95%
- Overlap: general 20%, professional 30%
- Three-level hierarchy: parent → middle → child
- Min/max chunk size guards
- Visualization of chunk boundaries
"""

from __future__ import annotations

import os
import re
import sys
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


# ===========================================================================
# Enums and Data Classes
# ===========================================================================

class Scenario(Enum):
    GENERAL = "general"            # 通用：客服FAQ, 百科问答
    PROFESSIONAL = "professional"  # 专业：法律, 医学, 金融

class ChunkLevel(Enum):
    PARENT = "parent"    # Full section (~1024 tokens)
    MIDDLE = "middle"    # Core paragraph (~512 tokens)
    CHILD = "child"      # Atomic fact (~256 tokens)


@dataclass
class ChunkConfig:
    """Configuration for chunking parameters based on scenario."""
    scenario: Scenario
    # Similarity dip threshold — chunk boundary when similarity drops below this
    similarity_threshold: float = 0.87
    # Overlap between adjacent chunks (fraction of chunk size)
    overlap_ratio: float = 0.20
    # Size constraints (in tokens, approx)
    min_chunk_tokens: int = 128
    max_chunk_tokens: int = 1024
    # Averaging window for similarity smoothing
    smooth_window: int = 3

    @classmethod
    def for_scenario(cls, scenario: Scenario) -> 'ChunkConfig':
        if scenario == Scenario.GENERAL:
            return cls(
                scenario=scenario,
                similarity_threshold=0.87,
                overlap_ratio=0.20,
                min_chunk_tokens=128,
                max_chunk_tokens=1024,
            )
        else:  # PROFESSIONAL
            return cls(
                scenario=scenario,
                similarity_threshold=0.92,
                overlap_ratio=0.30,
                min_chunk_tokens=256,
                max_chunk_tokens=2048,
            )


@dataclass
class Chunk:
    """A semantic chunk with hierarchy metadata."""
    chunk_id: str
    text: str
    level: ChunkLevel = ChunkLevel.MIDDLE
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    start_idx: int = 0
    end_idx: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Token estimation helpers
# ===========================================================================

def estimate_tokens(text: str, lang: str = 'zh') -> int:
    """Rough token count estimation.
    Chinese: ~1.5 chars per token. English: ~4 chars per token (subword avg).
    Mixed: heuristic based on character ranges.
    """
    if not text:
        return 0
    chinese_chars = len(re.findall(r'[一-鿿㐀-䶿]', text))
    total_chars = len(text)
    other_chars = total_chars - chinese_chars
    # Chinese: ~1.5 chars/token, English: ~4 chars/token
    return int(chinese_chars / 1.5 + other_chars / 4.0)


# ===========================================================================
# Sentence splitting
# ===========================================================================

def split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling Chinese and English punctuation."""
    # Replace newlines with spaces for sentence detection
    text = text.replace('\n', ' ')

    # Split on sentence-ending punctuation
    # Chinese: 。！？English: .!?  plus Chinese comma-like: ；：
    pattern = r'(?<=[。！？.!?；;])\s*'

    raw = re.split(pattern, text)

    sentences: List[str] = []
    for s in raw:
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


# ===========================================================================
# Similarity dip detection
# ===========================================================================

def compute_sentence_similarities(
    sentences: List[str],
    model: 'SentenceTransformer',
    smooth_window: int = 3,
) -> np.ndarray:
    """Compute pairwise cosine similarity between consecutive sentences.
    Returns array of length len(sentences)-1 with similarity of (s_i, s_{i+1}).
    Applies moving average smoothing.
    """
    if len(sentences) < 2:
        return np.array([])

    embeddings = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)

    n = len(sentences)
    sims = np.zeros(n - 1)
    for i in range(n - 1):
        a = embeddings[i].reshape(1, -1)
        b = embeddings[i + 1].reshape(1, -1)
        sims[i] = float(cosine_similarity(a, b)[0][0])

    # Smoothing
    if smooth_window > 1 and len(sims) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        sims = np.convolve(sims, kernel, mode='same')

    return sims


def detect_boundaries(
    sentences: List[str],
    sims: np.ndarray,
    threshold: float,
    min_tokens: int,
    max_tokens: int,
) -> List[int]:
    """
    Detect chunk boundaries where similarity dips below threshold.
    Also enforces min/max chunk size by inserting or merging boundaries.

    Returns list of sentence indices where chunks END (exclusive boundary).
    E.g., boundary at index 3 means sentences 0,1,2 form a chunk.
    """
    n = len(sentences)
    if n == 0:
        return []

    # Initial boundaries from similarity dip
    boundaries: List[int] = []
    current_tokens = 0
    chunk_start = 0

    for i in range(len(sims)):
        sent_tokens = estimate_tokens(sentences[i])
        current_tokens += sent_tokens

        is_boundary = False

        # Similarity dip
        if sims[i] < threshold:
            is_boundary = True

        # Max size guard: force boundary
        if current_tokens >= max_tokens:
            is_boundary = True

        # Min size guard: don't break too early
        if current_tokens < min_tokens and is_boundary and i < len(sims) - 1:
            is_boundary = False

        if is_boundary:
            boundaries.append(i + 1)  # End-exclusive boundary
            current_tokens = 0
            chunk_start = i + 1

    # Always end at last sentence
    if not boundaries or boundaries[-1] < n:
        boundaries.append(n)

    return boundaries


# ===========================================================================
# Three-level hierarchy builder
# ===========================================================================

def build_hierarchy(
    sentences: List[str],
    mid_boundaries: List[int],
    config: ChunkConfig,
    model: 'SentenceTransformer',
) -> List[Chunk]:
    """
    Build three-level chunk hierarchy:
    - Parent: groups of middle chunks (full sections)
    - Middle: core paragraphs (from boundary detection)
    - Child: sub-divisions within middle chunks (atomic facts)

    Also applies overlap between adjacent chunks.
    """
    chunks: List[Chunk] = []
    n = len(sentences)
    cid_counter = [0]

    def _next_id(level: ChunkLevel) -> str:
        cid_counter[0] += 1
        return f"{level.value}_{cid_counter[0]:04d}"

    # ---- Middle chunks (core) ----
    middle_chunks: List[Chunk] = []
    prev_end = 0
    for b in mid_boundaries:
        start = prev_end
        end = b
        text_sents = sentences[start:end]
        full_text = ' '.join(text_sents)
        mid = Chunk(
            chunk_id=_next_id(ChunkLevel.MIDDLE),
            text=full_text,
            level=ChunkLevel.MIDDLE,
            start_idx=start,
            end_idx=end,
            token_count=estimate_tokens(full_text),
        )
        middle_chunks.append(mid)
        # Overlap: include some sentences from next chunk
        overlap_sents = max(1, int(len(text_sents) * config.overlap_ratio))
        prev_end = max(start, end - overlap_sents)

    # ---- Parent chunks: group every 2-3 middle chunks ----
    parent_group_size = 3
    for pi in range(0, len(middle_chunks), parent_group_size):
        group = middle_chunks[pi:pi + parent_group_size]
        parent_text = ' '.join([m.text for m in group])
        parent = Chunk(
            chunk_id=_next_id(ChunkLevel.PARENT),
            text=parent_text,
            level=ChunkLevel.PARENT,
            start_idx=group[0].start_idx,
            end_idx=group[-1].end_idx,
            token_count=estimate_tokens(parent_text),
        )
        # Link children
        for m in group:
            m.parent_id = parent.chunk_id
            parent.child_ids.append(m.chunk_id)
        chunks.append(parent)
        chunks.extend(group)

    # ---- Child chunks: sub-divide middle chunks ----
    child_chunks: List[Chunk] = []
    for mid in middle_chunks:
        mid_sents = split_sentences(mid.text)
        if len(mid_sents) <= 2:
            continue  # Too small to sub-divide
        # Split into groups of 2-3 sentences
        child_size = max(2, len(mid_sents) // max(2, len(mid_sents) // 2))
        for ci in range(0, len(mid_sents), child_size):
            child_sents = mid_sents[ci:ci + child_size]
            child_text = ' '.join(child_sents)
            child = Chunk(
                chunk_id=_next_id(ChunkLevel.CHILD),
                text=child_text,
                level=ChunkLevel.CHILD,
                parent_id=mid.chunk_id,
                start_idx=mid.start_idx + ci,
                end_idx=mid.start_idx + ci + len(child_sents),
                token_count=estimate_tokens(child_text),
            )
            mid.child_ids.append(child.chunk_id)
            child_chunks.append(child)

    chunks.extend(child_chunks)
    return chunks


# ===========================================================================
# Main Semantic Chunker class
# ===========================================================================

class SemanticChunker:
    """Production semantic chunker with three-level hierarchy."""

    def __init__(
        self,
        scenario: Scenario = Scenario.GENERAL,
        config: Optional[ChunkConfig] = None,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        device: str = 'cpu',
    ):
        if config is None:
            config = ChunkConfig.for_scenario(scenario)

        self.config = config
        self.scenario = scenario
        self.model_name = model_name

        if not HAS_DEPS:
            raise ImportError(
                "sentence-transformers and scikit-learn are required. "
                "Install with: pip install sentence-transformers scikit-learn"
            )

        self.model = SentenceTransformer(model_name, device=device)

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk a document into a three-level hierarchy.

        Args:
            text: Input document text.

        Returns:
            List of Chunk objects at parent, middle, and child levels.
        """
        sentences = split_sentences(text)
        if not sentences:
            return []

        # Compute similarities
        sims = compute_sentence_similarities(
            sentences, self.model, smooth_window=self.config.smooth_window
        )

        # Detect boundaries
        boundaries = detect_boundaries(
            sentences, sims,
            threshold=self.config.similarity_threshold,
            min_tokens=self.config.min_chunk_tokens,
            max_tokens=self.config.max_chunk_tokens,
        )

        # Build hierarchy
        chunks = build_hierarchy(sentences, boundaries, self.config, self.model)

        return chunks

    def chunk_documents(self, documents: List[str]) -> List[List[Chunk]]:
        """Chunk multiple documents."""
        return [self.chunk(doc) for doc in documents]

    def get_similarities_for_visualization(self, text: str) -> Tuple[List[str], np.ndarray]:
        """Get sentences and similarity scores for visualization."""
        sentences = split_sentences(text)
        sims = compute_sentence_similarities(
            sentences, self.model, smooth_window=self.config.smooth_window
        )
        return sentences, sims


# ===========================================================================
# Visualization
# ===========================================================================

def visualize_chunk_boundaries(
    sentences: List[str],
    sims: np.ndarray,
    boundaries: List[int],
    threshold: float,
    title: str = "Semantic Chunk Boundaries",
    output_path: Optional[str] = None,
) -> None:
    """Visualize similarity dips and chunk boundaries."""
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not installed; skipping visualization.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2]})

    # --- Top: similarity scores ---
    x = list(range(len(sims)))
    colors = ['#d62728' if s < threshold else '#1f77b4' for s in sims]
    ax1.bar(x, sims, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.2f}')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)

    # Mark boundaries
    for b in boundaries:
        if b < len(sims):
            ax1.axvline(x=b - 0.5, color='green', linestyle=':', alpha=0.5, linewidth=1.5)

    # --- Bottom: sentence blocks with boundaries ---
    n_sents = len(sentences)
    ax2.set_xlim(0, max(n_sents, 1))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Sentence Index')
    ax2.set_ylabel('Chunks')
    ax2.set_yticks([])

    current_chunk = 0
    for i in range(n_sents):
        ax2.axvspan(i, i + 1, alpha=0.3, color=f'C{current_chunk % 10}')
        # Label short sentences
        label = sentences[i][:30] + ('...' if len(sentences[i]) > 30 else '')
        ax2.text(i + 0.5, 0.5 - 0.15 * (i % 3), label, ha='center', va='center',
                 fontsize=7, rotation=90 if len(sentences[i]) > 20 else 0)

    # Draw boundary lines
    for b in boundaries:
        if b < n_sents:
            ax2.axvline(x=b, color='red', linestyle='-', linewidth=2.5, alpha=0.7)
            ax2.text(b, 0.95, f'CHUNK SPLIT', ha='center', fontsize=8,
                     color='red', fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


# ===========================================================================
# Utility: chunks to dicts for serialization
# ===========================================================================

def chunks_to_dicts(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    """Convert Chunk objects to serializable dictionaries."""
    return [
        {
            'chunk_id': c.chunk_id,
            'level': c.level.value,
            'text': c.text,
            'parent_id': c.parent_id,
            'child_ids': c.child_ids,
            'start_idx': c.start_idx,
            'end_idx': c.end_idx,
            'token_count': c.token_count,
            'text_preview': c.text[:100] + ('...' if len(c.text) > 100 else ''),
            **c.metadata,
        }
        for c in chunks
    ]


# ===========================================================================
# __main__: self-test
# ===========================================================================

_SAMPLE_TEXT_ZH = """
自然语言处理是人工智能领域的一个重要分支。它研究如何让计算机理解和生成人类语言。
近年来，随着深度学习技术的发展，NLP取得了突破性进展。大型语言模型如GPT和BERT的出现，
彻底改变了这一领域的研究范式。

机器学习是人工智能的核心技术之一。通过从数据中学习模式和规律，
机器可以自动改进其性能而无需显式编程。监督学习、无监督学习和强化学习是三种主要范式。

深度学习是机器学习的一个子集。它使用多层神经网络来学习数据的层次化表示。
卷积神经网络在计算机视觉领域表现出色。循环神经网络适合处理序列数据。Transformer架构
已经成为自然语言处理的主流选择。

数据预处理是机器学习流程中的关键步骤。包括数据清洗、特征工程、标准化等环节。
高质量的训练数据是模型性能的基础保障。数据增强技术可以有效扩充训练集规模。

模型评估是确保模型质量的重要环节。常用的评估指标包括准确率、精确率、召回率和F1分数。
交叉验证是一种可靠的模型评估方法。它通过多次训练和测试来减少评估结果的方差。

超参数调优对模型性能有显著影响。网格搜索和贝叶斯优化是常用的调优方法。
学习率是最重要的超参数之一。批量大小和网络层数也需要仔细调整。

迁移学习允许我们将预训练模型的知识应用到新任务上。这大大降低了训练成本。
在NLP领域，BERT和GPT等预训练模型可以通过微调适应各种下游任务。

注意力机制是Transformer架构的核心组件。它允许模型在处理序列时关注相关信息。
自注意力机制可以捕获序列内部的依赖关系。多头注意力进一步增强了模型的表达能力。

知识图谱是结构化的语义知识库。它以图的形式表示实体及其关系。
知识图谱在搜索引擎、推荐系统和问答系统中都有广泛应用。实体链接和关系抽取是构建知识图谱的关键技术。

信息检索是RAG系统的重要组成部分。它负责从大规模文档库中找到相关的内容。
向量检索已经成为现代信息检索的主流方法。稠密检索和稀疏检索各有优势。
混合检索结合两者可以取得更好的效果。重排序技术进一步提升了检索质量。
"""

_SAMPLE_TEXT_EN = """
Natural language processing is a crucial branch of artificial intelligence. It focuses on
enabling computers to understand and generate human language. With the advancement of deep
learning, NLP has made remarkable progress in recent years.

Machine learning forms the foundation of modern AI systems. It allows machines to learn
patterns from data without explicit programming. There are three main paradigms: supervised
learning, unsupervised learning, and reinforcement learning.

Deep learning utilizes multi-layer neural networks for hierarchical feature learning.
Convolutional neural networks excel at computer vision tasks. Recurrent neural networks
are suitable for sequential data processing. The Transformer architecture has become
the dominant approach in natural language processing.

Data preprocessing is essential for machine learning workflows. It includes data cleaning,
feature engineering, and normalization. High-quality training data is fundamental to model
performance. Data augmentation techniques can effectively expand training datasets.

Model evaluation ensures the quality and reliability of machine learning models. Common metrics
include accuracy, precision, recall, and F1 score. Cross-validation provides reliable model
evaluation by reducing variance through multiple training and testing iterations.

Hyperparameter optimization significantly impacts model performance. Grid search and Bayesian
optimization are commonly used tuning methods. Learning rate is one of the most critical
hyperparameters. Batch size and network depth also require careful adjustment.

Transfer learning allows us to apply pre-trained model knowledge to new tasks. This greatly
reduces training costs. In NLP, pre-trained models like BERT and GPT can be fine-tuned
for various downstream tasks.

Attention mechanisms are core components of the Transformer architecture. They allow models
to focus on relevant information when processing sequences. Self-attention captures internal
dependencies within sequences. Multi-head attention further enhances model expressiveness.
"""


if __name__ == '__main__':
    print("=" * 60)
    print("Semantic Chunking - Self-Test")
    print("=" * 60)

    # ---------------------------------------------------
    # Test 1: General scenario (Chinese)
    # ---------------------------------------------------
    print("\n[Test 1] General scenario - Chinese text")
    print("-" * 40)

    chunker_general = SemanticChunker(scenario=Scenario.GENERAL)
    print(f"Config: threshold={chunker_general.config.similarity_threshold}, "
          f"overlap={chunker_general.config.overlap_ratio}, "
          f"min_tokens={chunker_general.config.min_chunk_tokens}, "
          f"max_tokens={chunker_general.config.max_chunk_tokens}")

    chunks = chunker_general.chunk(_SAMPLE_TEXT_ZH)

    # Count by level
    level_counts = {}
    for c in chunks:
        level_counts[c.level.value] = level_counts.get(c.level.value, 0) + 1
    print(f"\nChunk counts by level: {level_counts}")

    print(f"\nParent chunks ({level_counts.get('parent', 0)}):")
    for c in chunks:
        if c.level == ChunkLevel.PARENT:
            print(f"  [{c.chunk_id}] tokens={c.token_count} | {c.text[:80]}...")

    print(f"\nMiddle chunks ({level_counts.get('middle', 0)}):")
    for c in chunks:
        if c.level == ChunkLevel.MIDDLE:
            print(f"  [{c.chunk_id}] tokens={c.token_count} parent={c.parent_id} "
                  f"| {c.text[:80]}...")

    print(f"\nChild chunks ({level_counts.get('child', 0)}):")
    for c in chunks:
        if c.level == ChunkLevel.CHILD:
            print(f"  [{c.chunk_id}] tokens={c.token_count} parent={c.parent_id} "
                  f"| {c.text[:80]}...")

    # --- Visualization ---
    if HAS_MATPLOTLIB:
        sentences, sims = chunker_general.get_similarities_for_visualization(_SAMPLE_TEXT_ZH)
        boundaries = detect_boundaries(
            sentences, sims,
            threshold=chunker_general.config.similarity_threshold,
            min_tokens=chunker_general.config.min_chunk_tokens,
            max_tokens=chunker_general.config.max_chunk_tokens,
        )
        output_dir = os.path.dirname(os.path.abspath(__file__))
        viz_path = os.path.join(output_dir, 'chunk_viz_general.png')
        visualize_chunk_boundaries(
            sentences, sims, boundaries,
            threshold=chunker_general.config.similarity_threshold,
            title="Semantic Chunk Boundaries - General Scenario (Chinese)",
            output_path=viz_path,
        )

    # ---------------------------------------------------
    # Test 2: Professional scenario (English)
    # ---------------------------------------------------
    print("\n\n[Test 2] Professional scenario - English text")
    print("-" * 40)

    chunker_pro = SemanticChunker(scenario=Scenario.PROFESSIONAL)
    print(f"Config: threshold={chunker_pro.config.similarity_threshold}, "
          f"overlap={chunker_pro.config.overlap_ratio}, "
          f"min_tokens={chunker_pro.config.min_chunk_tokens}, "
          f"max_tokens={chunker_pro.config.max_chunk_tokens}")

    chunks_en = chunker_pro.chunk(_SAMPLE_TEXT_EN)

    level_counts_en = {}
    for c in chunks_en:
        level_counts_en[c.level.value] = level_counts_en.get(c.level.value, 0) + 1
    print(f"\nChunk counts by level: {level_counts_en}")

    print(f"\nAll middle chunks:")
    for c in chunks_en:
        if c.level == ChunkLevel.MIDDLE:
            print(f"  [{c.chunk_id}] tokens={c.token_count} | {c.text[:100]}...")

    # --- Visualization (professional) ---
    if HAS_MATPLOTLIB:
        sentences_en, sims_en = chunker_pro.get_similarities_for_visualization(_SAMPLE_TEXT_EN)
        boundaries_en = detect_boundaries(
            sentences_en, sims_en,
            threshold=chunker_pro.config.similarity_threshold,
            min_tokens=chunker_pro.config.min_chunk_tokens,
            max_tokens=chunker_pro.config.max_chunk_tokens,
        )
        output_dir = os.path.dirname(os.path.abspath(__file__))
        viz_path_en = os.path.join(output_dir, 'chunk_viz_professional.png')
        visualize_chunk_boundaries(
            sentences_en, sims_en, boundaries_en,
            threshold=chunker_pro.config.similarity_threshold,
            title="Semantic Chunk Boundaries - Professional Scenario (English)",
            output_path=viz_path_en,
        )

    # ---------------------------------------------------
    # Test 3: Serialization
    # ---------------------------------------------------
    print("\n\n[Test 3] Chunk serialization")
    print("-" * 40)
    dicts = chunks_to_dicts(chunks)
    print(f"Serialized {len(dicts)} chunks to JSON-compatible dicts.")
    sample = dicts[0].copy()
    # Truncate text for display
    sample['text'] = sample['text'][:60] + '...'
    print(f"Sample chunk dict: {json.dumps(sample, ensure_ascii=False, indent=2)}")

    # ---------------------------------------------------
    # Test 4: Edge cases
    # ---------------------------------------------------
    print("\n\n[Test 4] Edge cases")
    print("-" * 40)
    empty = chunker_general.chunk("")
    print(f"Empty input: {len(empty)} chunks (expected 0)")

    single = chunker_general.chunk("这是一个很短的句子。")
    print(f"Single sentence: {len(single)} chunks")
    for c in single:
        print(f"  [{c.chunk_id}] level={c.level.value} | {c.text}")

    print("\n" + "=" * 60)
    print("Semantic Chunking self-test completed successfully!")
    print("=" * 60)
