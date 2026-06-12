#!/usr/bin/env python3
"""
Phase 04 Module 01: Data Cleaning Pipeline
============================================
Complete production-quality document cleaning pipeline:
- Watermark removal (Chinese/English regex patterns)
- Header/footer stripping
- Blank line removal, special char normalization
- Content length filter (< 50 chars → discard)
- MD5 exact deduplication
- Semantic deduplication (cosine similarity > 0.92 → flag near-duplicate)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Try importing optional heavier deps; degrade gracefully
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class CleaningStats:
    """Aggregated statistics from the cleaning pipeline."""
    total_input: int = 0
    removed_watermark_lines: int = 0
    removed_header_footer_lines: int = 0
    removed_blank_lines: int = 0
    removed_short_docs: int = 0
    removed_exact_duplicates: int = 0
    flagged_near_duplicates: int = 0
    kept: int = 0


@dataclass
class CleanDocument:
    """A document that has been cleaned."""
    doc_id: str
    original_text: str
    cleaned_text: str
    md5: str
    is_flagged_near_duplicate: bool = False
    near_duplicate_of: Optional[str] = None
    near_duplicate_score: float = 0.0


# ===========================================================================
# Pattern collections
# ===========================================================================

WATERMARK_PATTERNS: List[re.Pattern] = [
    # --- Chinese watermarks ---
    re.compile(r'仅供.{0,10}(参考|使用|学习)', re.IGNORECASE),
    re.compile(r'内部资料.{0,10}(保密|严禁外传)', re.IGNORECASE),
    re.compile(r'版权(所有|归).{0,20}(公司|集团|有限公司)', re.IGNORECASE),
    re.compile(r'第[一二三四五六七八九十\d]+版', re.IGNORECASE),
    re.compile(r'机密.{0,5}(文件|档案)', re.IGNORECASE),
    re.compile(r'©\s*\d{4}.*', re.IGNORECASE),
    re.compile(r'Copyright\s+©?\s*\d{4}', re.IGNORECASE),
    re.compile(r'All\s+Rights?\s+Reserved', re.IGNORECASE),
    re.compile(r'Confidential\b', re.IGNORECASE),
    re.compile(r'DRAFT\b|草稿\b|未定稿\b', re.IGNORECASE),
    re.compile(r'水印|watermark|预览|preview', re.IGNORECASE),
    re.compile(r'仅供.{0,8}评估', re.IGNORECASE),
    re.compile(r'DO\s+NOT\s+DISTRIBUTE', re.IGNORECASE),
]

HEADER_FOOTER_PATTERNS: List[re.Pattern] = [
    re.compile(r'^[\s\-_=]{10,}$'),                        # Separator lines
    re.compile(r'^第\s*[一二三四五六七八九十\d]+\s*页\s*$'),     # Page number (CN)
    re.compile(r'^Page\s+\d+\s*(of\s+\d+)?$', re.IGNORECASE),  # Page number (EN)
    re.compile(r'^[-\s]*\d+[-\s]*$'),                        # Isolated page numbers
    re.compile(r'^(http|www\.)\S+', re.IGNORECASE),          # URLs
    re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4}.*$'),              # Date headers
    re.compile(r'^(Author|作者|版本|Version|修订)\s*[:：]', re.IGNORECASE),
    re.compile(r'^(Updated|Last Modified|更新时间)\s*[:：]', re.IGNORECASE),
]

SPECIAL_CHAR_MAP: Dict[str, str] = {
    '​': '',          # Zero-width space
    '‌': '',          # Zero-width non-joiner
    '‍': '',          # Zero-width joiner
    '﻿': '',          # BOM
    ' ': ' ',         # Non-breaking space → normal space
    '　': ' ',         # Full-width space → normal space
    '\r': '\n',            # CR → LF
    '–': '-',         # En dash
    '—': '--',        # Em dash
    '‘': "'",         # Left single quote
    '’': "'",         # Right single quote
    '“': '"',         # Left double quote
    '”': '"',         # Right double quote
    '…': '...',       # Ellipsis
    '·': '-',         # Middle dot
}


# ===========================================================================
# Core cleaning functions
# ===========================================================================

def remove_watermarks(text: str) -> Tuple[str, int]:
    """Remove watermark text using regex patterns. Returns (cleaned_text, removed_count)."""
    lines = text.split('\n')
    removed = 0
    cleaned_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        is_watermark = False
        for pat in WATERMARK_PATTERNS:
            if pat.search(stripped):
                is_watermark = True
                break
        if is_watermark:
            removed += 1
        else:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines), removed


def strip_headers_footers(text: str, max_header_lines: int = 5,
                          max_footer_lines: int = 5) -> Tuple[str, int]:
    """Remove header/footer lines (page numbers, urls, separator lines etc.)."""
    lines = text.split('\n')
    removed = 0
    n = len(lines)

    def _is_header_or_footer(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        for pat in HEADER_FOOTER_PATTERNS:
            if pat.search(s):
                return True
        return False

    # Mark positions to remove
    to_remove: Set[int] = set()

    # Top headers
    for i in range(min(max_header_lines, n)):
        if _is_header_or_footer(lines[i]):
            to_remove.add(i)
            removed += 1

    # Bottom footers
    for i in range(max(0, n - max_footer_lines), n):
        if _is_header_or_footer(lines[i]):
            to_remove.add(i)
            removed += 1

    cleaned_lines = [line for idx, line in enumerate(lines) if idx not in to_remove]
    return '\n'.join(cleaned_lines), removed


def normalize_whitespace_and_special_chars(text: str) -> Tuple[str, int]:
    """Remove blank lines, collapse repeated newlines, normalize special chars."""
    # Normalize special characters
    for char, replacement in SPECIAL_CHAR_MAP.items():
        text = text.replace(char, replacement)

    lines = text.split('\n')
    non_empty_lines: List[str] = []
    blank_removed = 0
    for line in lines:
        stripped = line.strip()
        if stripped == '':
            blank_removed += 1
        else:
            non_empty_lines.append(line)

    # Collapse multiple newlines into at most 2
    text = '\n'.join(non_empty_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove trailing/leading whitespace on each line but preserve indent intent
    cleaned_lines = [line.rstrip() for line in text.split('\n')]

    return '\n'.join(cleaned_lines), blank_removed


def filter_by_length(documents: List[Dict[str, str]],
                     min_chars: int = 50) -> Tuple[List[Dict[str, str]], int]:
    """Filter out documents shorter than min_chars."""
    kept = [doc for doc in documents if len(doc.get('cleaned_text', '')) >= min_chars]
    removed = len(documents) - len(kept)
    return kept, removed


def md5_exact_dedup(documents: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    """Remove exact duplicates based on MD5 hash of cleaned text."""
    seen: Set[str] = set()
    unique: List[Dict[str, str]] = []
    removed = 0
    for doc in documents:
        text = doc.get('cleaned_text', '')
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        if h in seen:
            removed += 1
        else:
            seen.add(h)
            doc['md5'] = h
            unique.append(doc)
    return unique, removed


def semantic_dedup(documents: List[Dict[str, str]],
                   threshold: float = 0.92,
                   model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                   batch_size: int = 64) -> Tuple[List[Dict[str, str]], int]:
    """
    Flag near-duplicate documents based on cosine similarity of embeddings.
    Returns documents with 'is_flagged_near_duplicate' and 'near_duplicate_of' set.

    Strategy: keep the first occurrence, flag subsequent near-duplicates.
    """
    if not HAS_SENTENCE_TRANSFORMERS or not HAS_SKLEARN:
        print("[WARN] sentence-transformers or sklearn not installed; "
              "skipping semantic dedup.")
        return documents, 0

    model = SentenceTransformer(model_name)
    texts = [doc.get('cleaned_text', '') for doc in documents]

    # Encode in batches
    embeddings_list: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings_list.append(emb)
    embeddings = np.concatenate(embeddings_list, axis=0) if embeddings_list else np.array([])

    n = len(documents)
    flagged = 0
    seen_embeddings: List[np.ndarray] = []
    seen_indices: List[int] = []

    for i in range(n):
        if not seen_embeddings:
            seen_embeddings.append(embeddings[i])
            seen_indices.append(i)
            documents[i]['is_flagged_near_duplicate'] = False
            continue

        sims = cosine_similarity(
            embeddings[i].reshape(1, -1),
            np.array(seen_embeddings)
        )[0]

        max_sim_idx = int(np.argmax(sims))
        max_sim = float(sims[max_sim_idx])

        if max_sim >= threshold:
            documents[i]['is_flagged_near_duplicate'] = True
            documents[i]['near_duplicate_of'] = documents[seen_indices[max_sim_idx]].get('doc_id', '')
            documents[i]['near_duplicate_score'] = round(max_sim, 4)
            flagged += 1
        else:
            seen_embeddings.append(embeddings[i])
            seen_indices.append(i)
            documents[i]['is_flagged_near_duplicate'] = False

    return documents, flagged


# ===========================================================================
# Full pipeline
# ===========================================================================

class DataCleaningPipeline:
    """Complete document cleaning pipeline with statistics tracking."""

    def __init__(
        self,
        min_content_length: int = 50,
        semantic_dedup_threshold: float = 0.92,
        semantic_dedup_enabled: bool = True,
        max_header_lines: int = 5,
        max_footer_lines: int = 5,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    ):
        self.min_content_length = min_content_length
        self.semantic_dedup_threshold = semantic_dedup_threshold
        self.semantic_dedup_enabled = semantic_dedup_enabled
        self.max_header_lines = max_header_lines
        self.max_footer_lines = max_footer_lines
        self.model_name = model_name
        self.stats = CleaningStats()

    def load_documents(self, directory: str, glob_pattern: str = '*.txt') -> List[Dict[str, str]]:
        """Load text files from a directory."""
        docs: List[Dict[str, str]] = []
        path = Path(directory)
        if not path.exists():
            print(f"[ERROR] Directory not found: {directory}")
            return docs

        for file_path in sorted(path.rglob(glob_pattern)):
            try:
                text = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    text = file_path.read_text(encoding='gbk')
                except Exception:
                    print(f"[WARN] Cannot read {file_path}, skipping.")
                    continue
            docs.append({
                'doc_id': str(file_path.relative_to(path)),
                'file_path': str(file_path),
                'original_text': text,
                'cleaned_text': text,
                'md5': '',
                'is_flagged_near_duplicate': False,
                'near_duplicate_of': '',
                'near_duplicate_score': 0.0,
            })
        self.stats.total_input = len(docs)
        return docs

    def run(self, directory: str, glob_pattern: str = '*.txt') -> List[Dict[str, Any]]:
        """Run the complete cleaning pipeline on a directory of documents."""
        print(f"\n{'='*60}")
        print(f"Data Cleaning Pipeline")
        print(f"{'='*60}\n")

        # Step 0: Load
        print("[0/6] Loading documents...")
        docs = self.load_documents(directory, glob_pattern)
        if not docs:
            print("[WARN] No documents loaded.")
            return []
        print(f"  Loaded {len(docs)} documents.")

        # Step 1: Remove watermarks
        print("[1/6] Removing watermarks...")
        for doc in docs:
            doc['cleaned_text'], removed = remove_watermarks(doc['cleaned_text'])
            self.stats.removed_watermark_lines += removed

        # Step 2: Strip headers/footers
        print("[2/6] Stripping headers and footers...")
        for doc in docs:
            doc['cleaned_text'], removed = strip_headers_footers(
                doc['cleaned_text'],
                max_header_lines=self.max_header_lines,
                max_footer_lines=self.max_footer_lines,
            )
            self.stats.removed_header_footer_lines += removed

        # Step 3: Normalize whitespace and special characters
        print("[3/6] Normalizing whitespace and special characters...")
        for doc in docs:
            doc['cleaned_text'], removed = normalize_whitespace_and_special_chars(
                doc['cleaned_text']
            )
            self.stats.removed_blank_lines += removed

        # Step 4: Length filter
        print("[4/6] Filtering short documents...")
        docs, removed = filter_by_length(docs, min_chars=self.min_content_length)
        self.stats.removed_short_docs = removed

        # Step 5: MD5 exact deduplication
        print("[5/6] MD5 exact deduplication...")
        docs, removed = md5_exact_dedup(docs)
        self.stats.removed_exact_duplicates = removed

        # Step 6: Semantic near-duplicate detection
        if self.semantic_dedup_enabled and docs:
            print("[6/6] Semantic near-duplicate detection...")
            docs, flagged = semantic_dedup(
                docs,
                threshold=self.semantic_dedup_threshold,
                model_name=self.model_name,
            )
            self.stats.flagged_near_duplicates = flagged
        else:
            print("[6/6] Semantic dedup skipped (disabled or no documents).")

        self.stats.kept = len(docs)

        # Print summary
        print(f"\n{'='*60}")
        print("Cleaning Summary")
        print(f"{'='*60}")
        print(f"  Total input:                {self.stats.total_input:>6}")
        print(f"  Watermark lines removed:    {self.stats.removed_watermark_lines:>6}")
        print(f"  Header/footer lines removed:{self.stats.removed_header_footer_lines:>6}")
        print(f"  Blank lines removed:        {self.stats.removed_blank_lines:>6}")
        print(f"  Short docs removed:         {self.stats.removed_short_docs:>6}")
        print(f"  Exact duplicates removed:   {self.stats.removed_exact_duplicates:>6}")
        print(f"  Near-duplicates flagged:    {self.stats.flagged_near_duplicates:>6}")
        print(f"  Final kept:                 {self.stats.kept:>6}")
        print(f"{'='*60}\n")

        return docs

    def save_cleaned(self, docs: List[Dict[str, Any]], output_dir: str) -> None:
        """Save cleaned documents to output directory."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save cleaned txt files
        txt_dir = out_path / 'cleaned'
        txt_dir.mkdir(exist_ok=True)
        for doc in docs:
            if doc.get('is_flagged_near_duplicate'):
                continue
            fname = Path(doc['doc_id']).name
            (txt_dir / fname).write_text(doc['cleaned_text'], encoding='utf-8')

        # Save metadata JSON
        meta = []
        for doc in docs:
            meta.append({
                'doc_id': doc['doc_id'],
                'md5': doc.get('md5', ''),
                'is_flagged_near_duplicate': doc.get('is_flagged_near_duplicate', False),
                'near_duplicate_of': doc.get('near_duplicate_of', ''),
                'near_duplicate_score': doc.get('near_duplicate_score', 0.0),
                'char_count': len(doc['cleaned_text']),
            })
        (out_path / 'metadata.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8'
        )

        # Save stats
        (out_path / 'stats.json').write_text(
            json.dumps(self.stats.__dict__, ensure_ascii=False, indent=2), encoding='utf-8'
        )

        print(f"Cleaned documents saved to: {out_path}")


# ===========================================================================
# __main__: Demo and self-test
# ===========================================================================

_SAMPLE_DOCS = {
    "doc1_watermark.txt": (
        "仅供内部参考使用\n"
        "=== CONFIDENTIAL ===\n\n"
        "第一章 引言\n\n"
        "这是关于自然语言处理技术的一篇介绍文章。\n"
        "本文涵盖了从基础理论到实际应用的多个方面。\n\n"
        "第 1 页\n"
        "Copyright © 2024 Example Corp.\n"
    ),
    "doc2_dup.txt": (
        "第一章 引言\n\n"
        "这是关于自然语言处理技术的一篇介绍文章。\n"
        "本文涵盖了从基础理论到实际应用的多个方面。\n"
    ),
    "doc3_short.txt": "太短",
    "doc4_english.txt": (
        "CONFIDENTIAL - DO NOT DISTRIBUTE\n\n"
        "# Introduction to Machine Learning\n\n"
        "Machine learning is a subset of artificial intelligence\n"
        "that enables systems to learn and improve from experience\n"
        "without being explicitly programmed.\n\n"
        "Page 1 of 10\n"
    ),
    "doc5_similar.txt": (
        "第一章 导论\n\n"
        "这是关于自然语言处理技术的一篇综合性介绍文章。\n"
        "本文涵盖了从基础理论知识到实际工程应用的各个方面。\n"
        "还包括了最新的研究进展和未来趋势分析。\n"
    ),
}


def _create_sample_dir(base_dir: str) -> str:
    """Create sample documents for testing."""
    sample_dir = os.path.join(base_dir, 'sample_docs')
    os.makedirs(sample_dir, exist_ok=True)
    for fname, content in _SAMPLE_DOCS.items():
        filepath = os.path.join(sample_dir, fname)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    return sample_dir


if __name__ == '__main__':
    import tempfile

    # --- Create a temporary directory with sample documents ---
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = _create_sample_dir(tmpdir)
        output_dir = os.path.join(tmpdir, 'output')

        print("=" * 60)
        print("Data Cleaning Pipeline - Self-Test")
        print("=" * 60)
        print(f"Sample directory: {sample_dir}")
        print(f"Sample files:")
        for f in sorted(os.listdir(sample_dir)):
            fpath = os.path.join(sample_dir, f)
            print(f"  {f} ({len(open(fpath, encoding='utf-8').read())} chars)")

        # Run pipeline
        pipeline = DataCleaningPipeline(
            min_content_length=50,
            semantic_dedup_threshold=0.92,
            semantic_dedup_enabled=True,
        )

        cleaned_docs = pipeline.run(sample_dir, '*.txt')

        # Save output
        pipeline.save_cleaned(cleaned_docs, output_dir)

        # Show cleaned results
        print("\nCleaned Documents Preview:\n")
        for i, doc in enumerate(cleaned_docs):
            flag = " [NEAR-DUP]" if doc.get('is_flagged_near_duplicate') else ""
            print(f"[{i}] {doc['doc_id']}{flag}")
            print(f"    MD5: {doc.get('md5', 'N/A')[:16]}...")
            print(f"    Text (first 120 chars):")
            text_preview = doc['cleaned_text'][:120].replace('\n', '\\n')
            print(f"    {text_preview}...")
            if doc.get('is_flagged_near_duplicate'):
                print(f"    Near-duplicate of: {doc['near_duplicate_of']} "
                      f"(score: {doc['near_duplicate_score']})")
            print()

        print("Pipeline self-test completed successfully!")
