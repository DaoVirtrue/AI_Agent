#!/usr/bin/env python3
"""
Phase 04 Module 04: Metadata Enrichment
=========================================
Auto-generate rich metadata for each chunk:
- Source, page, date, domain classification
- Content type enum: definition/process/parameter/case
- LLM-generated chunk summary (for dual-vector indexing)
- Custom domain dictionary for entity linking
- Metadata filter builder for retrieval-time filtering
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ===========================================================================
# Enums and Data Classes
# ===========================================================================

class ContentType(Enum):
    DEFINITION = "definition"     # 定义
    PROCESS = "process"           # 流程
    PARAMETER = "parameter"       # 参数
    CASE = "case"                 # 案例
    REFERENCE = "reference"       # 参考
    INSTRUCTION = "instruction"   # 操作说明
    UNKNOWN = "unknown"

class Domain(Enum):
    GENERAL = "general"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    ACADEMIC = "academic"

@dataclass
class ChunkMetadata:
    """Rich metadata for a single chunk."""
    chunk_id: str
    source_document: str = ""
    page_number: Optional[int] = None
    date_published: Optional[str] = None
    date_processed: str = field(default_factory=lambda: datetime.now().isoformat())
    domain: Domain = Domain.GENERAL
    content_type: ContentType = ContentType.UNKNOWN
    entity_tags: List[str] = field(default_factory=list)
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    chunk_length_chars: int = 0
    chunk_length_tokens: int = 0
    language: str = "zh"
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainDictionaryEntry:
    """An entry in the domain-specific entity dictionary."""
    abbreviation: str
    full_name: str
    aliases: List[str] = field(default_factory=list)
    category: str = ""
    description: str = ""


# ===========================================================================
# Domain Dictionary (Entity Linking)
# ===========================================================================

# Pre-built domain dictionaries
DOMAIN_DICTIONARIES: Dict[str, List[DomainDictionaryEntry]] = {
    "legal": [
        DomainDictionaryEntry("民法典", "中华人民共和国民法典",
                             aliases=["民法", "Civil Code"], category="law"),
        DomainDictionaryEntry("刑法", "中华人民共和国刑法",
                             aliases=["Criminal Law"], category="law"),
        DomainDictionaryEntry("知产", "知识产权",
                             aliases=["IP", "Intellectual Property"], category="concept"),
        DomainDictionaryEntry("仲裁", "仲裁裁决",
                             aliases=["Arbitration"], category="process"),
        DomainDictionaryEntry("竞业限制", "竞业限制协议",
                             aliases=["Non-compete"], category="contract"),
    ],
    "medical": [
        DomainDictionaryEntry("CT", "计算机断层扫描",
                             aliases=["Computed Tomography", "CAT scan"], category="imaging"),
        DomainDictionaryEntry("MRI", "磁共振成像",
                             aliases=["Magnetic Resonance Imaging", "核磁共振"], category="imaging"),
        DomainDictionaryEntry("ICU", "重症监护室",
                             aliases=["Intensive Care Unit"], category="department"),
        DomainDictionaryEntry("NLP", "自然语言处理",
                             aliases=["Natural Language Processing"], category="ai"),
        DomainDictionaryEntry("EHR", "电子健康记录",
                             aliases=["Electronic Health Record", "电子病历"], category="system"),
    ],
    "financial": [
        DomainDictionaryEntry("ROI", "投资回报率",
                             aliases=["Return on Investment"], category="metric"),
        DomainDictionaryEntry("ROE", "净资产收益率",
                             aliases=["Return on Equity"], category="metric"),
        DomainDictionaryEntry("EPS", "每股收益",
                             aliases=["Earnings Per Share"], category="metric"),
        DomainDictionaryEntry("PE", "市盈率",
                             aliases=["P/E ratio", "Price to Earnings"], category="metric"),
        DomainDictionaryEntry("KYC", "了解你的客户",
                             aliases=["Know Your Customer"], category="compliance"),
    ],
    "technical": [
        DomainDictionaryEntry("API", "应用程序编程接口",
                             aliases=["Application Programming Interface"], category="interface"),
        DomainDictionaryEntry("SDK", "软件开发工具包",
                             aliases=["Software Development Kit"], category="tool"),
        DomainDictionaryEntry("RAG", "检索增强生成",
                             aliases=["Retrieval-Augmented Generation"], category="architecture"),
        DomainDictionaryEntry("LLM", "大型语言模型",
                             aliases=["Large Language Model", "大语言模型"], category="model"),
        DomainDictionaryEntry("GPU", "图形处理器",
                             aliases=["Graphics Processing Unit"], category="hardware"),
    ],
}


# ===========================================================================
# Content type classifier (rule-based)
# ===========================================================================

# Indicator patterns for each content type
CONTENT_TYPE_INDICATORS: Dict[ContentType, List[re.Pattern]] = {
    ContentType.DEFINITION: [
        re.compile(r'(是指|定义为|指的是|称为|is\s+defined\s+as|refers?\s+to|means?\s+)'),
        re.compile(r'^(什么是|What\s+is|Define)'),
        re.compile(r'(概念|定义|术语|Definition|Concept|Term)'),
    ],
    ContentType.PROCESS: [
        re.compile(r'(步骤|流程|过程|操作|Step|Process|Procedure|How\s+to)'),
        re.compile(r'(首先|然后|接着|最后|First|Then|Next|Finally)'),
        re.compile(r'^\d+[\.、\)]\s', re.MULTILINE),  # Numbered lists
    ],
    ContentType.PARAMETER: [
        re.compile(r'(参数|配置|设置|阈值|Parameter|Config|Setting|Threshold)'),
        re.compile(r'(默认值|取值范围|Default\s+value|Range)'),
        re.compile(r'(=?[\d.]+(?:ms|s|GB|MB|%|px)?\s*)'),
    ],
    ContentType.CASE: [
        re.compile(r'(案例|示例|案例研究|Example|Case\s+Study|Scenario)'),
        re.compile(r'(例如|比如|For\s+example|e\.g\.|i\.e\.)'),
        re.compile(r'(某公司|某医院|某项目|Company\s+X|Hospital\s+Y)'),
    ],
    ContentType.REFERENCE: [
        re.compile(r'\[(\d+|[\w\s]+)\]', re.MULTILINE),
        re.compile(r'(参考文献|Reference|Bibliography|引用|参见)'),
        re.compile(r'(http[s]?://|doi:|arXiv:)'),
    ],
    ContentType.INSTRUCTION: [
        re.compile(r'(请|需|应该|必须|Please|Should|Must|Need\s+to)'),
        re.compile(r'(注意|警告|Note|Warning|Caution|Important)'),
        re.compile(r'(单击|输入|选择|Click|Enter|Select|Press)'),
    ],
}

# Domain indicator patterns
DOMAIN_INDICATORS: Dict[Domain, List[re.Pattern]] = {
    Domain.LEGAL: [
        re.compile(r'(法律|法规|条款|判决|法院|合同|Law|Legal|Court|Contract|Clause)'),
        re.compile(r'(原告|被告|诉讼|起诉|Plaintiff|Defendant|Lawsuit|Sue)'),
    ],
    Domain.MEDICAL: [
        re.compile(r'(患者|诊断|治疗|药物|手术|Patient|Diagnos|Treat|Drug|Surgery)'),
        re.compile(r'(症状|疾病|临床|医疗|Symptom|Disease|Clinical|Medical)'),
    ],
    Domain.FINANCIAL: [
        re.compile(r'(股票|基金|投资|收益|风险|Stock|Fund|Invest|Return|Risk)'),
        re.compile(r'(财报|营收|利润|资产|Revenue|Profit|Asset|Liability)'),
    ],
    Domain.TECHNICAL: [
        re.compile(r'(算法|代码|函数|API|SDK|Algorithm|Code|Function|Framework)'),
        re.compile(r'(部署|架构|服务器|数据库|Deploy|Architecture|Server|Database)'),
    ],
    Domain.ACADEMIC: [
        re.compile(r'(实验|研究|理论|假设|Experiment|Research|Theory|Hypothesis)'),
        re.compile(r'(论文|期刊|学术|Pape|Journal|Academic|Scholar)'),
    ],
}


# ===========================================================================
# Metadata Enricher
# ===========================================================================

class MetadataEnricher:
    """Generate rich metadata for document chunks."""

    def __init__(
        self,
        domain: Domain = Domain.GENERAL,
        custom_dictionary: Optional[List[DomainDictionaryEntry]] = None,
        llm_summary_fn: Optional[Callable[[str], str]] = None,
    ):
        self.domain = domain
        self.llm_summary_fn = llm_summary_fn
        self.entity_dict: Dict[str, DomainDictionaryEntry] = {}

        # Load domain dictionaries
        for domain_key, entries in DOMAIN_DICTIONARIES.items():
            for entry in entries:
                self.entity_dict[entry.abbreviation.lower()] = entry
                for alias in entry.aliases:
                    self.entity_dict[alias.lower()] = entry

        # Add custom entries
        if custom_dictionary:
            for entry in custom_dictionary:
                self.entity_dict[entry.abbreviation.lower()] = entry
                for alias in entry.aliases:
                    self.entity_dict[alias.lower()] = entry

    def enrich(self, chunk_id: str, text: str, **kwargs: Any) -> ChunkMetadata:
        """Generate complete metadata for a chunk."""
        meta = ChunkMetadata(chunk_id=chunk_id)

        # From kwargs
        meta.source_document = kwargs.get('source_document', '')
        meta.page_number = kwargs.get('page_number')
        meta.date_published = kwargs.get('date_published')
        meta.chunk_length_chars = len(text)
        meta.chunk_length_tokens = self._estimate_tokens(text)
        meta.language = self._detect_language(text)

        # Classify
        meta.domain = self._classify_domain(text)
        meta.content_type = self._classify_content_type(text)

        # Entity linking
        meta.entity_tags = self._extract_entities(text)

        # Keywords
        meta.keywords = self._extract_keywords(text)

        # Summary (rule-based fallback if no LLM)
        meta.summary = self._generate_summary(text)

        # Confidence scores
        meta.confidence_scores = {
            'domain': self._domain_confidence(text, meta.domain),
            'content_type': self._type_confidence(text, meta.content_type),
        }

        return meta

    def enrich_batch(
        self, chunks: List[Tuple[str, str]], **kwargs: Any
    ) -> List[ChunkMetadata]:
        """Enrich a batch of chunks."""
        return [self.enrich(cid, text, **kwargs) for cid, text in chunks]

    # --- Internal classifiers ---

    def _classify_domain(self, text: str) -> Domain:
        scores: Dict[Domain, int] = defaultdict(int)
        for domain, patterns in DOMAIN_INDICATORS.items():
            for pat in patterns:
                scores[domain] += len(pat.findall(text))

        if not scores or max(scores.values()) == 0:
            return Domain.GENERAL

        best = max(scores, key=lambda k: scores[k])
        if scores[best] >= 3:
            return best
        return Domain.GENERAL

    def _classify_content_type(self, text: str) -> ContentType:
        scores: Dict[ContentType, int] = defaultdict(int)
        for ctype, patterns in CONTENT_TYPE_INDICATORS.items():
            for pat in patterns:
                scores[ctype] += len(pat.findall(text))

        if not scores or max(scores.values()) == 0:
            return ContentType.UNKNOWN

        best = max(scores, key=lambda k: scores[k])
        if scores[best] >= 2:
            return best
        # Second-best if first is UNKNOWN
        if best == ContentType.UNKNOWN:
            del scores[ContentType.UNKNOWN]
            if scores:
                return max(scores, key=lambda k: scores[k])
        return ContentType.UNKNOWN

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities using domain dictionary matching."""
        found: Set[str] = set()
        text_lower = text.lower()

        for key, entry in self.entity_dict.items():
            if key in text_lower:
                found.add(f"{entry.full_name} ({entry.abbreviation})")
                if entry.category:
                    found.add(entry.category)

        # Also extract capitalized acronyms
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        for acr in acronyms:
            acr_lower = acr.lower()
            if acr_lower in self.entity_dict:
                entry = self.entity_dict[acr_lower]
                found.add(f"{entry.full_name} ({acr})")

        return sorted(found)

    def _extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        """Simple keyword extraction using TF-like heuristics."""
        # Remove punctuation
        clean = re.sub(r'[^\w\s一-鿿]', ' ', text)
        words = clean.split()

        # Count frequencies
        freq: Dict[str, int] = defaultdict(int)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
            'and', 'or', 'but', 'not', 'this', 'that', 'it', 'as', 'we',
            '的', '是', '在', '了', '和', '与', '或', '也', '就', '都',
            '而', '及', '但', '被', '把', '从', '到', '对', '让', '向',
            '一个', '一种', '这个', '那个', '可以', '需要', '进行', '通过',
        }
        for w in words:
            wl = w.lower()
            if len(wl) < 2 or wl in stopwords:
                continue
            freq[wl] += 1

        # Sort by frequency and length bonus
        scored = sorted(freq.items(), key=lambda x: x[1] * (1 + min(len(x[0]) / 10, 1)),
                       reverse=True)
        return [w for w, _ in scored[:max_keywords]]

    def _generate_summary(self, text: str, max_len: int = 120) -> str:
        """Generate a simple summary (rule-based fallback)."""
        if self.llm_summary_fn:
            try:
                return self.llm_summary_fn(text)
            except Exception:
                pass

        # Fallback: first sentence + keyword context
        sentences = re.split(r'[。！？.!?\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text[:max_len]

        if len(sentences[0]) >= 20:
            summary = sentences[0][:max_len]
        else:
            summary = ' '.join(sentences[:2])[:max_len]

        if len(summary) < len(text):
            summary += '...'
        return summary

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English."""
        chinese_chars = len(re.findall(r'[一-鿿㐀-䶿]', text))
        total_chars = len(re.sub(r'\s', '', text))
        if total_chars == 0:
            return 'unknown'
        return 'zh' if chinese_chars / max(total_chars, 1) > 0.3 else 'en'

    def _estimate_tokens(self, text: str) -> int:
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4.0)

    def _domain_confidence(self, text: str, domain: Domain) -> float:
        total = sum(len(pat.findall(text)) for pats in DOMAIN_INDICATORS.values() for pat in pats)
        if total == 0:
            return 0.5
        match = sum(len(pat.findall(text)) for pat in DOMAIN_INDICATORS.get(domain, []))
        return round(float(match) / total, 3)

    def _type_confidence(self, text: str, ctype: ContentType) -> float:
        total = sum(len(pat.findall(text)) for pats in CONTENT_TYPE_INDICATORS.values() for pat in pats)
        if total == 0:
            return 0.3
        match = sum(len(pat.findall(text)) for pat in CONTENT_TYPE_INDICATORS.get(ctype, []))
        return round(float(match) / total, 3)


# ===========================================================================
# Metadata Filter Builder
# ===========================================================================

class MetadataFilterBuilder:
    """Build retrieval-time filters from metadata."""

    @staticmethod
    def build_filter(
        domain: Optional[Domain] = None,
        content_type: Optional[ContentType] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        entity_tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a metadata filter dict for vector DB queries."""
        filter_dict: Dict[str, Any] = {}

        if domain is not None:
            filter_dict['domain'] = domain.value

        if content_type is not None:
            filter_dict['content_type'] = content_type.value

        if date_from:
            filter_dict['date_published'] = filter_dict.get('date_published', {})
            filter_dict['date_published']['$gte'] = date_from

        if date_to:
            filter_dict['date_published'] = filter_dict.get('date_published', {})
            filter_dict['date_published']['$lte'] = date_to

        if entity_tags:
            filter_dict['entity_tags'] = {'$in': entity_tags}

        if language:
            filter_dict['language'] = language

        if min_length is not None:
            filter_dict['chunk_length_tokens'] = filter_dict.get('chunk_length_tokens', {})
            filter_dict['chunk_length_tokens']['$gte'] = min_length

        if max_length is not None:
            filter_dict['chunk_length_tokens'] = filter_dict.get('chunk_length_tokens', {})
            filter_dict['chunk_length_tokens']['$lte'] = max_length

        return filter_dict

    @staticmethod
    def to_query_string(filter_dict: Dict[str, Any]) -> str:
        """Convert filter dict to a human-readable query string."""
        parts: List[str] = []
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    op_map = {'$gte': '>=', '$lte': '<=', '$in': 'in', '$eq': '='}
                    parts.append(f"{key} {op_map.get(op, op)} {val}")
            else:
                parts.append(f"{key} = {value}")
        return ' AND '.join(parts) if parts else '(no filter)'


# ===========================================================================
# Dual-vector indexing helper
# ===========================================================================

def generate_dual_vector_text(chunk_text: str, summary: str) -> Tuple[str, str]:
    """Generate two vector representations for dual-indexing.

    Returns:
        - content_vector_text: the original chunk for semantic retrieval
        - summary_vector_text: the summary for fast filtering/pre-retrieval
    """
    return (
        chunk_text,                          # Dense embedding on full content
        summary if summary else chunk_text[:200],  # Lightweight embedding on summary
    )


# ===========================================================================
# __main__: Demo
# ===========================================================================

_SAMPLE_CHUNKS = [
    ("chunk_001", "自然语言处理是指通过计算机技术对人类语言进行自动分析和生成的一门学科。"
                  "它结合了语言学、计算机科学和人工智能的知识。NLP的主要应用包括机器翻译、"
                  "情感分析、文本摘要和问答系统。近年来，大型语言模型（LLM）的出现极大地推动了NLP的发展。"),
    ("chunk_002", "要训练一个文本分类模型，首先需要准备标注数据集。然后进行数据预处理，"
                  "包括分词、去除停用词和构建词汇表。接着选择合适的模型架构，如BERT或LSTM。"
                  "训练过程中需要设置学习率为0.0001，批量大小为32，训练3-5个epoch。"
                  "最后在测试集上评估模型的准确率和F1分数。"),
    ("chunk_003", "[案例] 某医疗机构引入了基于NLP的临床决策支持系统。该系统通过分析患者的"
                  "电子健康记录（EHR），自动提取关键症状和检查结果。在为期6个月的试验中，"
                  "诊断准确率提高了15%，平均诊断时间缩短了40%。该案例表明AI辅助诊断具有显著的临床价值。"),
    ("chunk_004", "ROI（投资回报率）是衡量投资效益的核心指标。计算公式为：ROI = (收益 - 成本) / 成本 × 100%。"
                  "在评估AI项目时，需要考虑直接收益（效率提升、成本节约）和间接收益（客户满意度、"
                  "品牌价值）两个方面。通常一个成功的AI项目的ROI在200%-500%之间。"),
    ("chunk_005", "Please note that this is a reference document for the API integration. "
                  "The SDK requires Python 3.10+ and an active API key. "
                  "For examples, see Section 3.2 of the developer guide."),
]


if __name__ == '__main__':
    print("=" * 60)
    print("Metadata Enrichment - Self-Test")
    print("=" * 60)

    # --- Create enricher ---
    enricher = MetadataEnricher(domain=Domain.TECHNICAL)

    # --- Enrich all sample chunks ---
    metadata_list = enricher.enrich_batch(_SAMPLE_CHUNKS)

    for i, meta in enumerate(metadata_list):
        print(f"\n{'─'*50}")
        print(f"Chunk: {meta.chunk_id} ({meta.chunk_length_chars} chars, "
              f"~{meta.chunk_length_tokens} tokens)")
        print(f"  Language:    {meta.language}")
        print(f"  Domain:      {meta.domain.value} "
              f"(confidence: {meta.confidence_scores.get('domain', 'N/A')})")
        print(f"  ContentType: {meta.content_type.value} "
              f"(confidence: {meta.confidence_scores.get('content_type', 'N/A')})")
        print(f"  Summary:     {meta.summary[:100]}...")
        print(f"  Keywords:    {', '.join(meta.keywords)}")
        print(f"  Entities:    {', '.join(meta.entity_tags) if meta.entity_tags else '(none)'}")

    # --- Test filter builder ---
    print(f"\n{'='*60}")
    print("Metadata Filter Builder Test")
    print(f"{'='*60}")

    filters = [
        MetadataFilterBuilder.build_filter(domain=Domain.TECHNICAL),
        MetadataFilterBuilder.build_filter(
            domain=Domain.MEDICAL,
            content_type=ContentType.CASE,
        ),
        MetadataFilterBuilder.build_filter(
            date_from='2024-01-01',
            date_to='2024-12-31',
            entity_tags=['自然语言处理', '大型语言模型'],
        ),
        MetadataFilterBuilder.build_filter(
            language='zh',
            min_length=50,
            max_length=500,
        ),
    ]

    for i, f in enumerate(filters, 1):
        print(f"\nFilter {i}: {MetadataFilterBuilder.to_query_string(f)}")
        print(f"  Raw: {json.dumps(f, ensure_ascii=False)}")

    # --- Test dual vector generation ---
    print(f"\n{'='*60}")
    print("Dual-Vector Indexing Test")
    print(f"{'='*60}")
    for cid, text in _SAMPLE_CHUNKS[:2]:
        content_vec, summary_vec = generate_dual_vector_text(
            text, enricher._generate_summary(text)
        )
        print(f"\n{cid}:")
        print(f"  Content vector text: {content_vec[:80]}...")
        print(f"  Summary vector text: {summary_vec[:80]}...")

    print(f"\n{'='*60}")
    print("Metadata Enrichment self-test completed!")
    print(f"{'='*60}")
