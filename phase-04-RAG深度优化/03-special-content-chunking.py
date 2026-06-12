#!/usr/bin/env python3
"""
Phase 04 Module 03: Special Content Chunking
==============================================
Handles non-text content within documents:
- Table detection and extraction → Markdown table
- Formula extraction (LaTeX preservation)
- Code block isolation (language-tagged fence blocks)
- Image/Chart placeholder with description
- Special content gets standalone chunks with "context binding" text
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# Enums and Data Classes
# ===========================================================================

class ContentType(Enum):
    TEXT = "text"
    TABLE = "table"
    FORMULA = "formula"
    CODE = "code"
    IMAGE = "image"
    CHART = "chart"
    MIXED = "mixed"


@dataclass
class SpecialContentBlock:
    """A detected special content block."""
    content_type: ContentType
    original: str                          # Original raw text
    processed: str                         # Processed/standardized content
    context_binding: str                   # Context binding text (surrounding text summary)
    language: Optional[str] = None         # For code blocks: python, sql, etc.
    caption: Optional[str] = None          # For tables/images: caption
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ChunkResult:
    """Result of chunking a mixed-format document."""
    text_chunks: List[str] = field(default_factory=list)
    special_blocks: List[SpecialContentBlock] = field(default_factory=list)
    full_text: str = ""
    stats: Dict[str, int] = field(default_factory=dict)


# ===========================================================================
# Table detection and extraction
# ===========================================================================

TABLE_PATTERNS = [
    # Markdown table
    re.compile(r'^\|.+\|\s*$\n^\|[\s\-:]+\|\s*$', re.MULTILINE),
    # HTML table
    re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE),
    # ASCII table (pipe or plus-dash borders)
    re.compile(r'^\+[\-+]+\+\s*$\n^(\|.+\|\s*$\n)*\+[\-+]+\+', re.MULTILINE),
    # Tab-separated data (3+ columns, 2+ rows)
    re.compile(r'^[^\t]+\t[^\t]+\t[^\t]+.*$\n^[^\t]+\t[^\t]+\t[^\t]+.*$', re.MULTILINE),
    # CSV-like with 3+ commas per line in consecutive rows
    re.compile(r'^[^,]+,[^,]+,[^,]+.*$\n^[^,]+,[^,]+,[^,]+.*$', re.MULTILINE),
]


def detect_tables(text: str) -> List[SpecialContentBlock]:
    """Detect and extract tables from text.

    For tabular data:
    - Camelot/Tabula style extraction → Markdown table format
    - Preserve headers and alignments
    - Generate context binding from caption or surrounding text
    """
    blocks: List[SpecialContentBlock] = []

    # ---- Markdown tables ----
    md_table_pattern = re.compile(
        r'((?:^\|.+\|\s*$\n)+)', re.MULTILINE
    )
    for match in md_table_pattern.finditer(text):
        raw = match.group(0).strip()
        if raw.count('|') < 4:
            continue  # Too small to be a table

        # Normalize to clean Markdown table
        lines = raw.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Ensure line starts and ends with |
            if not line.startswith('|'):
                line = '| ' + line
            if not line.endswith('|'):
                line = line + ' |'
            cleaned_lines.append(line)

        processed = '\n'.join(cleaned_lines)

        # Try to find caption before the table
        caption = _find_caption(text, match.start(), before=True)

        block = SpecialContentBlock(
            content_type=ContentType.TABLE,
            original=raw,
            processed=processed,
            context_binding=_build_table_context(raw, caption),
            caption=caption,
            start_pos=match.start(),
            end_pos=match.end(),
        )
        blocks.append(block)

    # ---- ASCII tables ----
    ascii_pattern = re.compile(
        r'(\+[\-]+\+[\-]+\+[\-]+\+[\s\S]*?)(?=\n\n|\Z)', re.MULTILINE
    )
    for match in ascii_pattern.finditer(text):
        raw = match.group(0).strip()
        if raw.count('+') < 4:
            continue

        # Convert ASCII table to Markdown
        processed = _ascii_to_markdown_table(raw)
        caption = _find_caption(text, match.start(), before=True)

        block = SpecialContentBlock(
            content_type=ContentType.TABLE,
            original=raw,
            processed=processed,
            context_building=_build_table_context(raw, caption),
            caption=caption,
            start_pos=match.start(),
            end_pos=match.end(),
        )
        blocks.append(block)

    return blocks


def _ascii_to_markdown_table(ascii_table: str) -> str:
    """Convert ASCII art table to Markdown format."""
    lines = ascii_table.strip().split('\n')
    data_lines = [l for l in lines if l.startswith('|')]
    if not data_lines:
        return ascii_table

    md_lines = []
    for i, line in enumerate(data_lines):
        cells = [c.strip() for c in line.split('|')[1:-1]]
        md_lines.append('| ' + ' | '.join(cells) + ' |')
        if i == 0:
            # Add separator line
            md_lines.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')

    return '\n'.join(md_lines)


def _find_caption(text: str, pos: int, before: bool = True, max_dist: int = 200) -> Optional[str]:
    """Find a caption ('Table X:', '图 X:', 'Figure X:') near a position."""
    caption_patterns = [
        r'(?:Table|表|表格)\s*\d+[：:]\s*(.+?)(?:\n|$)',
        r'(?:Figure|图|图表|Chart)\s*\d+[：:]\s*(.+?)(?:\n|$)',
    ]
    search_start = max(0, pos - max_dist)
    search_end = min(len(text), pos + max_dist)
    context = text[search_start:search_end]

    for pat in caption_patterns:
        m = re.search(pat, context, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def _build_table_context(raw_table: str, caption: Optional[str]) -> str:
    """Build context binding text for a table."""
    lines = raw_table.strip().split('\n')
    data_lines = [l for l in lines if '|' in l or '\t' in l or ',' in l]
    n_rows = len(data_lines) - 1  # minus header
    header = data_lines[0] if data_lines else ""

    parts = [f"[TABLE]"]
    if caption:
        parts.append(f"Caption: {caption}")
    parts.append(f"Rows: {n_rows}")
    if header:
        parts.append(f"Header: {_extract_headers(header)}")
    return ' | '.join(parts)


def _extract_headers(line: str) -> str:
    """Extract column headers from a table row."""
    if '|' in line:
        cells = [c.strip() for c in line.split('|') if c.strip()]
    elif '\t' in line:
        cells = [c.strip() for c in line.split('\t') if c.strip()]
    else:
        cells = [c.strip() for c in line.split(',') if c.strip()]
    return ', '.join(cells[:5])  # Top 5 columns


# ===========================================================================
# Formula extraction (LaTeX)
# ===========================================================================

LATEX_PATTERNS = [
    # Display math: $$...$$ or \[...\]
    re.compile(r'\$\$(.+?)\$\$', re.DOTALL),
    re.compile(r'\\\[(.+?)\\\]', re.DOTALL),
    # Inline math: $...$
    re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', re.DOTALL),
    # LaTeX environments: \begin{equation}...\end{equation}
    re.compile(r'\\begin\{(equation|align|eqnarray)\*?\}(.+?)\\end\{\1\*?\}', re.DOTALL),
]


def extract_formulas(text: str) -> List[SpecialContentBlock]:
    """Extract LaTeX formulas from text."""
    blocks: List[SpecialContentBlock] = []

    for pattern in LATEX_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            inner = match.group(1).strip() if match.lastindex else raw

            # Preserve LaTeX, add a plain-text description
            description = _latex_to_text_description(inner)

            block = SpecialContentBlock(
                content_type=ContentType.FORMULA,
                original=raw,
                processed=raw,  # Keep LaTeX as-is
                context_binding=f"[FORMULA] {description}",
                start_pos=match.start(),
                end_pos=match.end(),
            )
            blocks.append(block)

    return blocks


def _latex_to_text_description(latex: str) -> str:
    """Create a simple text description from LaTeX content."""
    # Remove common LaTeX commands for readability
    text = latex.replace('\\frac', '').replace('\\sum', 'sum')
    text = re.sub(r'\\[a-zA-Z]+\{', '', text)
    text = text.replace('{', '').replace('}', '')
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 80:
        text = text[:77] + '...'
    return f"Formula: {text}"


# ===========================================================================
# Code block isolation
# ===========================================================================

CODE_FENCE_PATTERN = re.compile(
    r'```(\w*)\s*\n(.*?)```', re.DOTALL
)

CODE_INDENT_PATTERN = re.compile(
    r'(?:^|\n)((?:    |\t)[^\n]+(?:\n(?:    |\t)[^\n]+)*)', re.MULTILINE
)


def extract_code_blocks(text: str) -> List[SpecialContentBlock]:
    """Extract code blocks (fenced and indented) from text."""
    blocks: List[SpecialContentBlock] = []

    # ---- Fenced code blocks ----
    for match in CODE_FENCE_PATTERN.finditer(text):
        language = match.group(1).strip() or 'text'
        code = match.group(2).strip()
        raw = match.group(0)

        block = SpecialContentBlock(
            content_type=ContentType.CODE,
            original=raw,
            processed=code,
            context_binding=f"[CODE:{language}] {len(code.split(chr(10)))} lines, "
                          f"{len(code)} chars",
            language=language,
            start_pos=match.start(),
            end_pos=match.end(),
        )
        blocks.append(block)

    # ---- Indented code blocks ----
    for match in CODE_INDENT_PATTERN.finditer(text):
        raw = match.group(1)
        if len(raw.strip()) < 20:
            continue  # Too short
        # Check: is it really code? Heuristic: contains keywords or symbols
        code_lines = raw.strip().split('\n')
        code_text = '\n'.join(l[4:] if l.startswith('    ') else l[1:] for l in code_lines)

        # Skip if already inside a fenced block
        is_inside_fence = False
        for fb in blocks:
            if fb.start_pos <= match.start() <= fb.end_pos:
                is_inside_fence = True
                break
        if is_inside_fence:
            continue

        lang = _guess_language(code_text)

        block = SpecialContentBlock(
            content_type=ContentType.CODE,
            original=raw,
            processed=code_text,
            context_binding=f"[CODE:{lang}] {len(code_lines)} lines, "
                          f"{len(code_text)} chars",
            language=lang,
            start_pos=match.start(),
            end_pos=match.end(),
        )
        blocks.append(block)

    return blocks


def _guess_language(code: str) -> str:
    """Heuristic language detection for code."""
    code_lower = code.lower()
    if re.search(r'\b(def|class|import|from|print|lambda|yield|with)\b', code):
        return 'python'
    if re.search(r'\b(function|const|let|var|=>|console\.log|require)\b', code):
        return 'javascript'
    if re.search(r'\b(SELECT|FROM|WHERE|INSERT|UPDATE|CREATE TABLE)\b', code, re.IGNORECASE):
        return 'sql'
    if re.search(r'\b(public|class|static|void|String|import java)\b', code):
        return 'java'
    if re.search(r'\b(fn|impl|struct|enum|trait|pub|use|mod)\b', code):
        return 'rust'
    if re.search(r'</?[a-z]+', code):
        return 'html'
    if re.search(r'^[a-z]+:\s*$', code, re.MULTILINE) and '---' in code:
        return 'yaml'
    if re.search(r'\{[^}]*\}', code) and ':' in code:
        return 'json'
    return 'text'


# ===========================================================================
# Image/Chart placeholder
# ===========================================================================

IMAGE_PATTERNS = [
    # Markdown image: ![alt](url)
    re.compile(r'!\[([^\]]*)\]\(([^)]+)\)'),
    # HTML img tag
    re.compile(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*/?>', re.IGNORECASE),
    # Reference-style: [Figure X] or [Chart X]
    re.compile(r'\[(?:Figure|图表?|图|Chart|Fig\.?)\s*\d+[^\]]*\]', re.IGNORECASE),
    # Inline reference: (see Figure X) or (见图X)
    re.compile(r'\((?:see\s+)?(?:Figure|图表?|图)\s*\d+[^)]*\)', re.IGNORECASE),
]


def extract_image_references(text: str) -> List[SpecialContentBlock]:
    """Detect image/chart references and create placeholder chunks."""
    blocks: List[SpecialContentBlock] = []

    # ---- Markdown images ----
    for match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', text):
        alt_text = match.group(1) or 'Image'
        url = match.group(2)
        block = SpecialContentBlock(
            content_type=ContentType.IMAGE,
            original=match.group(0),
            processed=f"[IMAGE: {url}]",
            context_binding=f"[IMAGE] {alt_text} (source: {url})",
            caption=alt_text,
            metadata={'url': url, 'alt': alt_text},
            start_pos=match.start(),
            end_pos=match.end(),
        )
        blocks.append(block)

    # ---- Figure/Chart references ----
    for match in re.finditer(r'\[(Figure|图表?|图|Chart|Fig\.?)\s*(\d+)[^\]]*\]', text, re.IGNORECASE):
        fig_type = match.group(1)
        fig_num = match.group(2)
        raw = match.group(0)
        block = SpecialContentBlock(
            content_type=ContentType.CHART if 'chart' in fig_type.lower() else ContentType.IMAGE,
            original=raw,
            processed=f"[{fig_type.upper()} {fig_num}]",
            context_binding=f"[{fig_type.upper()}] Reference to {fig_type} {fig_num}",
            caption=raw,
            start_pos=match.start(),
            end_pos=match.end(),
        )
        blocks.append(block)

    return blocks


# ===========================================================================
# Main processor: Mixed-format document chunking
# ===========================================================================

class SpecialContentChunker:
    """Process mixed-format documents: detect, extract, and chunk special content.

    Strategy:
    1. Identify all special content blocks (tables, formulas, code, images)
    2. Remove them from main text → clean text for normal chunking
    3. Each special block becomes a standalone chunk with context binding
    4. Context binding includes: surrounding text summary, position info, type info
    """

    def __init__(
        self,
        extract_tables: bool = True,
        extract_formulas: bool = True,
        extract_code: bool = True,
        extract_images: bool = True,
        context_window: int = 200,  # chars of surrounding context
    ):
        self.extract_tables = extract_tables
        self.extract_formulas = extract_formulas
        self.extract_code = extract_code
        self.extract_images = extract_images
        self.context_window = context_window

    def process(self, text: str) -> ChunkResult:
        """Process a mixed-format document."""
        result = ChunkResult(full_text=text)
        all_special: List[SpecialContentBlock] = []

        # Step 1: Extract all special content
        if self.extract_tables:
            tables = detect_tables(text)
            all_special.extend(tables)
            result.stats['tables'] = len(tables)

        if self.extract_formulas:
            formulas = extract_formulas(text)
            all_special.extend(formulas)
            result.stats['formulas'] = len(formulas)

        if self.extract_code:
            code_blocks = extract_code_blocks(text)
            all_special.extend(code_blocks)
            result.stats['code_blocks'] = len(code_blocks)

        if self.extract_images:
            images = extract_image_references(text)
            all_special.extend(images)
            result.stats['images'] = len(images)

        # Sort by position
        all_special.sort(key=lambda b: b.start_pos)
        result.special_blocks = all_special

        # Step 2: Remove special blocks, keep text regions
        text_chunks = self._extract_text_regions(text, all_special)
        result.text_chunks = text_chunks
        result.stats['text_chunks'] = len(text_chunks)

        # Step 3: Enrich context binding with surrounding text
        self._enrich_context_bindings(text, all_special)

        return result

    def _extract_text_regions(
        self, text: str, blocks: List[SpecialContentBlock]
    ) -> List[str]:
        """Extract text regions between special blocks."""
        if not blocks:
            return [text.strip()] if text.strip() else []

        regions: List[str] = []
        cursor = 0

        for block in blocks:
            # Text before this block
            if block.start_pos > cursor:
                region = text[cursor:block.start_pos].strip()
                if region:
                    regions.append(region)
            cursor = block.end_pos

        # Remaining text after last block
        if cursor < len(text):
            region = text[cursor:].strip()
            if region:
                regions.append(region)

        return regions

    def _enrich_context_bindings(
        self, text: str, blocks: List[SpecialContentBlock]
    ) -> None:
        """Add surrounding context to each special block's context binding."""
        for block in blocks:
            # Get text before
            before_start = max(0, block.start_pos - self.context_window)
            before_text = text[before_start:block.start_pos].strip()
            # Get text after
            after_end = min(len(text), block.end_pos + self.context_window)
            after_text = text[block.end_pos:after_end].strip()

            surrounding = []
            if before_text:
                # Take last sentence or last 80 chars
                before_snippet = before_text[-80:].strip()
                surrounding.append(f"Before: ...{before_snippet}...")
            if after_text:
                after_snippet = after_text[:80].strip()
                surrounding.append(f"After: {after_snippet}...")

            if surrounding:
                block.context_binding += ' | ' + ' | '.join(surrounding)

    def to_chunk_texts(self, result: ChunkResult) -> List[str]:
        """Convert result to a list of chunk texts ready for embedding."""
        chunks: List[str] = []

        # Text chunks first
        for i, tc in enumerate(result.text_chunks):
            chunks.append(tc)

        # Special blocks with context binding
        for block in result.special_blocks:
            chunk_text = f"{block.context_binding}\n\n{block.processed}"
            chunks.append(chunk_text)

        return chunks


# ===========================================================================
# __main__: Demo
# ===========================================================================

_SAMPLE_MIXED_DOC = """
# 机器学习性能分析报告

## 数据集统计

以下是实验使用的数据集统计信息：

| 数据集 | 训练样本 | 测试样本 | 类别数 | 平均长度 |
|--------|----------|----------|--------|----------|
| SST-2  | 67,349   | 1,821    | 2      | 19       |
| MNLI   | 392,702  | 9,815    | 3      | 38       |
| QQP    | 363,846  | 40,430   | 2      | 22       |

表 1: GLUE 基准数据集统计

## 模型架构

Transformer 模型的核心是自注意力机制，其计算公式为：

$$Attention(Q, K, V) = softmax\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

多头注意力的计算可以表示为：

\\[MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O\\]

其中每个头的计算为 $head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$。

## 训练代码

训练循环的 Python 实现如下：

```python
def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

## 实验结果

![Accuracy Comparison](figures/accuracy_comparison.png)

图 2: 各模型在 GLUE 基准上的准确率对比

经过 3 个 epoch 的训练，模型在验证集上的表现如下：

+------------------+----------+----------+
| 指标             | 基线模型 | 优化模型 |
+------------------+----------+----------+
| Accuracy         | 0.8923   | 0.9341   |
| F1 Score         | 0.8856   | 0.9298   |
| Inference Time   | 12.3ms   | 9.7ms    |
+------------------+----------+----------+

## 结论

本实验表明，通过引入多头注意力机制和适当的超参数调优，
模型性能可以得到显著提升。未来工作将探索更大规模预训练模型的微调策略。
"""


if __name__ == '__main__':
    print("=" * 60)
    print("Special Content Chunking - Self-Test")
    print("=" * 60)

    chunker = SpecialContentChunker(
        extract_tables=True,
        extract_formulas=True,
        extract_code=True,
        extract_images=True,
    )

    result = chunker.process(_SAMPLE_MIXED_DOC)

    # --- Print stats ---
    print(f"\nDocument Stats:")
    print(f"  Total chars: {len(_SAMPLE_MIXED_DOC)}")
    print(f"  Text chunks: {result.stats.get('text_chunks', 0)}")
    print(f"  Tables:      {result.stats.get('tables', 0)}")
    print(f"  Formulas:    {result.stats.get('formulas', 0)}")
    print(f"  Code blocks: {result.stats.get('code_blocks', 0)}")
    print(f"  Images:      {result.stats.get('images', 0)}")

    # --- Print text regions ---
    print(f"\n{'='*60}")
    print("TEXT REGIONS (for normal chunking):")
    print(f"{'='*60}")
    for i, tc in enumerate(result.text_chunks):
        print(f"\n--- Text Region {i+1} ({len(tc)} chars) ---")
        print(tc[:300] + ('...' if len(tc) > 300 else ''))

    # --- Print special blocks ---
    print(f"\n{'='*60}")
    print("SPECIAL CONTENT BLOCKS:")
    print(f"{'='*60}")
    for i, block in enumerate(result.special_blocks):
        print(f"\n--- Block {i+1}: {block.content_type.value.upper()} ---")
        print(f"  Context: {block.context_binding}")
        if block.language:
            print(f"  Language: {block.language}")
        if block.caption:
            print(f"  Caption: {block.caption}")
        print(f"  Content preview ({len(block.processed)} chars):")
        for line in block.processed.split('\n')[:5]:
            print(f"    {line}")
        if len(block.processed.split('\n')) > 5:
            print(f"    ... ({len(block.processed.split(chr(10)))} total lines)")

    # --- Final chunk list for embedding ---
    print(f"\n{'='*60}")
    print("CHUNKS FOR EMBEDDING:")
    print(f"{'='*60}")
    all_chunks = chunker.to_chunk_texts(result)
    for i, chunk in enumerate(all_chunks):
        print(f"\n[Chunk {i+1}] ({len(chunk)} chars)")
        print(chunk[:200] + ('...' if len(chunk) > 200 else ''))

    print(f"\n{'='*60}")
    print("Special Content Chunking self-test completed!")
    print(f"{'='*60}")
