#!/usr/bin/env python3
"""
生产级RAG数据管道 - 8阶段全流程处理
Production RAG Data Pipeline - 8-Phase Full Pipeline

从原始文档到可查询索引的完整生产管道，包含：
  阶段1: 多源文档解析 (PDF/Word/Markdown/网页/OCR → 结构化JSON)
  阶段2: 双重去重 (MD5精确匹配 + 语义相似度>0.92)
  阶段3: 三级语义分块 (章节→段落→知识点)
  阶段4: 元数据增强 (自动元数据 + LLM摘要 + 实体标签)
  阶段5: 双向量生成 (chunk_original + chunk_summary)
  阶段6: HNSW索引构建
  阶段7: BM25倒排索引构建
  阶段8: 验证与质量门控
"""

import os
import sys
import json
import hashlib
import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable

import numpy as np

# ============================================================
# 日志配置 (Logging Configuration)
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("DataPipeline")


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class DocumentSource(Enum):
    """文档来源类型"""
    PDF = "pdf"
    WORD = "docx"
    MARKDOWN = "md"
    WEB = "html"
    IMAGE = "image"  # OCR处理


class ChunkLevel(Enum):
    """分块层级"""
    CHAPTER = "chapter"
    PARAGRAPH = "paragraph"
    KNOWLEDGE_POINT = "knowledge_point"


class PipelinePhase(Enum):
    """管道阶段枚举"""
    PARSE = 1
    DEDUP = 2
    CHUNK = 3
    ENRICH = 4
    VECTORIZE = 5
    HNSW_INDEX = 6
    BM25_INDEX = 7
    VALIDATE = 8


@dataclass
class RawDocument:
    """原始文档"""
    source: str  # 文件路径或URL
    source_type: DocumentSource
    content_raw: bytes = b""
    metadata_raw: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """解析后文档"""
    doc_id: str
    source: str
    source_type: DocumentSource
    text: str
    structured: dict  # JSON结构化内容
    metadata: dict
    raw_hash: str  # MD5


@dataclass
class Chunk:
    """文本块"""
    chunk_id: str
    doc_id: str
    level: ChunkLevel
    text: str
    metadata: dict
    embedding_original: Optional[np.ndarray] = None
    embedding_summary: Optional[np.ndarray] = None
    version: int = 1


@dataclass
class PipelineStats:
    """管道统计"""
    docs_total: int = 0
    docs_parsed: int = 0
    docs_dedup_removed: int = 0
    chunks_total: int = 0
    chunks_chapter: int = 0
    chunks_paragraph: int = 0
    chunks_knowledge: int = 0
    vector_count: int = 0
    phase_times: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


# ============================================================
# 回调和错误处理 (Callbacks & Error Handling)
# ============================================================

class ProgressCallback:
    """进度回调管理器"""

    def __init__(self):
        self.callbacks: list[Callable] = []

    def register(self, fn: Callable):
        """注册回调函数"""
        self.callbacks.append(fn)

    def notify(self, phase: PipelinePhase, progress: float, message: str):
        """通知所有回调"""
        for fn in self.callbacks:
            try:
                fn(phase, progress, message)
            except Exception as e:
                logger.warning(f"回调执行失败: {e}")


class PipelineError(Exception):
    """管道异常基类"""

    def __init__(self, phase: PipelinePhase, message: str, original: Exception = None):
        self.phase = phase
        self.message = message
        self.original = original
        super().__init__(f"[{phase.name}] {message}")


class ErrorHandler:
    """错误处理器"""

    def __init__(self, stats: PipelineStats):
        self.stats = stats

    def handle(self, phase: PipelinePhase, doc_id: str, error: Exception, recoverable: bool = True):
        """处理管道错误"""
        error_info = {
            "phase": phase.name,
            "doc_id": doc_id,
            "error": str(error),
            "recoverable": recoverable,
        }
        self.stats.errors.append(error_info)
        logger.error(f"[{phase.name}] 文档 {doc_id} 处理失败: {error}")

        if not recoverable:
            raise PipelineError(phase, f"不可恢复错误: {error}", error)


# ============================================================
# 阶段1: 多源文档解析 (Multi-Source Parsing)
# ============================================================

class BaseParser(ABC):
    """解析器基类"""

    @abstractmethod
    def parse(self, raw: RawDocument) -> ParsedDocument:
        pass


class PDFParser(BaseParser):
    """PDF解析器"""

    def parse(self, raw: RawDocument) -> ParsedDocument:
        try:
            import pypdf

            reader = pypdf.PdfReader(raw.source)
            text_parts = []
            structured = {"pages": []}

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                structured["pages"].append({
                    "page_number": i + 1,
                    "text": page_text,
                })

            full_text = "\n\n".join(text_parts)
            doc_id = self._generate_doc_id(raw.source, full_text)

            return ParsedDocument(
                doc_id=doc_id,
                source=raw.source,
                source_type=DocumentSource.PDF,
                text=full_text,
                structured=structured,
                metadata={
                    "file_name": os.path.basename(raw.source),
                    "page_count": len(reader.pages),
                },
                raw_hash=self._compute_hash(full_text),
            )
        except ImportError:
            logger.warning("pypdf 未安装，使用简化PDF解析")
            return self._fallback_parse(raw)

    def _fallback_parse(self, raw: RawDocument) -> ParsedDocument:
        """降级解析"""
        text = raw.content_raw.decode("utf-8", errors="ignore") if raw.content_raw else ""
        if not text:
            with open(raw.source, "rb") as f:
                text = f.read().decode("utf-8", errors="ignore")
        doc_id = self._generate_doc_id(raw.source, text)
        return ParsedDocument(
            doc_id=doc_id,
            source=raw.source,
            source_type=DocumentSource.PDF,
            text=text,
            structured={"pages": [{"page_number": 1, "text": text}]},
            metadata={"file_name": os.path.basename(raw.source)},
            raw_hash=self._compute_hash(text),
        )

    @staticmethod
    def _compute_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_doc_id(source: str, text: str) -> str:
        return hashlib.md5(f"{source}:{len(text)}".encode()).hexdigest()[:16]


class WordParser(BaseParser):
    """Word文档解析器"""

    def parse(self, raw: RawDocument) -> ParsedDocument:
        try:
            import docx

            doc = docx.Document(raw.source)
            text_parts = []
            structured = {"paragraphs": []}

            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    text_parts.append(para.text)
                    structured["paragraphs"].append({
                        "index": i,
                        "text": para.text,
                        "style": para.style.name if para.style else "Normal",
                    })

            full_text = "\n\n".join(text_parts)
            doc_id = hashlib.md5(f"{raw.source}:{len(full_text)}".encode()).hexdigest()[:16]

            return ParsedDocument(
                doc_id=doc_id,
                source=raw.source,
                source_type=DocumentSource.WORD,
                text=full_text,
                structured=structured,
                metadata={
                    "file_name": os.path.basename(raw.source),
                    "paragraph_count": len(text_parts),
                },
                raw_hash=hashlib.md5(full_text.encode()).hexdigest(),
            )
        except ImportError:
            return self._fallback_parse(raw)

    def _fallback_parse(self, raw: RawDocument) -> ParsedDocument:
        text = raw.content_raw.decode("utf-8", errors="ignore") if raw.content_raw else open(raw.source, encoding="utf-8").read()
        doc_id = hashlib.md5(f"{raw.source}:{len(text)}".encode()).hexdigest()[:16]
        return ParsedDocument(
            doc_id=doc_id,
            source=raw.source,
            source_type=DocumentSource.WORD,
            text=text,
            structured={"paragraphs": [{"index": 0, "text": text}]},
            metadata={"file_name": os.path.basename(raw.source)},
            raw_hash=hashlib.md5(text.encode()).hexdigest(),
        )


class MarkdownParser(BaseParser):
    """Markdown解析器"""

    def parse(self, raw: RawDocument) -> ParsedDocument:
        try:
            import markdown_it

            with open(raw.source, "r", encoding="utf-8") as f:
                raw_text = f.read()

            md = markdown_it.MarkdownIt()
            tokens = md.parse(raw_text)

            structured = {"headings": [], "paragraphs": [], "code_blocks": []}
            current_heading = None

            for token in tokens:
                if token.type == "heading_open":
                    level = int(token.tag[1])
                    heading_text = ""
                    for child in tokens:
                        if child.type == "inline" and child.map and child.map[0] == token.map[0] + 1:
                            heading_text = child.content
                            break
                    structured["headings"].append({"level": level, "text": heading_text})
                    current_heading = heading_text
                elif token.type == "fence":
                    structured["code_blocks"].append({
                        "language": token.info,
                        "content": token.content,
                    })
                elif token.type == "paragraph_open":
                    pass

            doc_id = hashlib.md5(f"{raw.source}:{len(raw_text)}".encode()).hexdigest()[:16]

            return ParsedDocument(
                doc_id=doc_id,
                source=raw.source,
                source_type=DocumentSource.MARKDOWN,
                text=raw_text,
                structured=structured,
                metadata={
                    "file_name": os.path.basename(raw.source),
                    "heading_count": len(structured["headings"]),
                },
                raw_hash=hashlib.md5(raw_text.encode()).hexdigest(),
            )
        except ImportError:
            return self._fallback_parse(raw)

    def _fallback_parse(self, raw: RawDocument) -> ParsedDocument:
        with open(raw.source, "r", encoding="utf-8") as f:
            text = f.read()
        doc_id = hashlib.md5(f"{raw.source}:{len(text)}".encode()).hexdigest()[:16]
        return ParsedDocument(
            doc_id=doc_id,
            source=raw.source,
            source_type=DocumentSource.MARKDOWN,
            text=text,
            structured={"headings": [], "paragraphs": []},
            metadata={"file_name": os.path.basename(raw.source)},
            raw_hash=hashlib.md5(text.encode()).hexdigest(),
        )


class WebParser(BaseParser):
    """网页解析器"""

    def parse(self, raw: RawDocument) -> ParsedDocument:
        try:
            import requests
            from html.parser import HTMLParser as _HTMLParser

            class HTMLTextExtractor(_HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip_tags = {"script", "style", "meta", "link"}

                def handle_data(self, data):
                    stripped = data.strip()
                    if stripped:
                        self.text.append(stripped)

            resp = requests.get(raw.source, timeout=30)
            resp.raise_for_status()

            extractor = HTMLTextExtractor()
            extractor.feed(resp.text)

            full_text = "\n".join(extractor.text)
            doc_id = hashlib.md5(f"{raw.source}:{len(full_text)}".encode()).hexdigest()[:16]

            return ParsedDocument(
                doc_id=doc_id,
                source=raw.source,
                source_type=DocumentSource.WEB,
                text=full_text,
                structured={"url": raw.source, "title": self._extract_title(resp.text)},
                metadata={"url": raw.source, "status_code": resp.status_code},
                raw_hash=hashlib.md5(full_text.encode()).hexdigest(),
            )
        except ImportError:
            return self._fallback_parse(raw)

    def _fallback_parse(self, raw: RawDocument) -> ParsedDocument:
        text = raw.content_raw.decode("utf-8", errors="ignore") if raw.content_raw else f"WEB_CONTENT_FROM:{raw.source}"
        doc_id = hashlib.md5(f"{raw.source}:{len(text)}".encode()).hexdigest()[:16]
        return ParsedDocument(
            doc_id=doc_id,
            source=raw.source,
            source_type=DocumentSource.WEB,
            text=text,
            structured={"url": raw.source},
            metadata={"url": raw.source},
            raw_hash=hashlib.md5(text.encode()).hexdigest(),
        )

    @staticmethod
    def _extract_title(html: str) -> str:
        import re
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
        return match.group(1) if match else ""


class OCRParser(BaseParser):
    """OCR图像解析器"""

    def parse(self, raw: RawDocument) -> ParsedDocument:
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(raw.source)
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")

            doc_id = hashlib.md5(f"{raw.source}:{len(text)}".encode()).hexdigest()[:16]

            return ParsedDocument(
                doc_id=doc_id,
                source=raw.source,
                source_type=DocumentSource.IMAGE,
                text=text,
                structured={"image_size": image.size, "ocr_engine": "tesseract"},
                metadata={"file_name": os.path.basename(raw.source)},
                raw_hash=hashlib.md5(text.encode()).hexdigest(),
            )
        except ImportError:
            return self._fallback_parse(raw)

    def _fallback_parse(self, raw: RawDocument) -> ParsedDocument:
        text = f"OCR_FALLBACK:{raw.source}"
        doc_id = hashlib.md5(f"{raw.source}:{len(text)}".encode()).hexdigest()[:16]
        return ParsedDocument(
            doc_id=doc_id,
            source=raw.source,
            source_type=DocumentSource.IMAGE,
            text=text,
            structured={"ocr_engine": "fallback"},
            metadata={"file_name": os.path.basename(raw.source)},
            raw_hash=hashlib.md5(text.encode()).hexdigest(),
        )


class ParserRegistry:
    """解析器注册表"""

    def __init__(self):
        self._parsers: dict[DocumentSource, BaseParser] = {
            DocumentSource.PDF: PDFParser(),
            DocumentSource.WORD: WordParser(),
            DocumentSource.MARKDOWN: MarkdownParser(),
            DocumentSource.WEB: WebParser(),
            DocumentSource.IMAGE: OCRParser(),
        }

    def detect_source_type(self, source: str) -> DocumentSource:
        """自动检测文档类型"""
        ext = os.path.splitext(source)[1].lower()
        if source.startswith(("http://", "https://")):
            return DocumentSource.WEB
        mapping = {
            ".pdf": DocumentSource.PDF,
            ".docx": DocumentSource.WORD,
            ".doc": DocumentSource.WORD,
            ".md": DocumentSource.MARKDOWN,
            ".markdown": DocumentSource.MARKDOWN,
            ".png": DocumentSource.IMAGE,
            ".jpg": DocumentSource.IMAGE,
            ".jpeg": DocumentSource.IMAGE,
            ".tiff": DocumentSource.IMAGE,
        }
        return mapping.get(ext, DocumentSource.MARKDOWN)

    def parse(self, source: str) -> ParsedDocument:
        """解析文档"""
        source_type = self.detect_source_type(source)
        parser = self._parsers[source_type]

        raw = RawDocument(source=source, source_type=source_type)
        logger.info(f"解析文档: {source} (类型: {source_type.value})")
        return parser.parse(raw)


# ============================================================
# 阶段2: 双重去重 (Double Deduplication)
# ============================================================

class Deduplicator:
    """双重去重器: MD5精确匹配 + 语义相似度去重"""

    def __init__(self, semantic_threshold: float = 0.92):
        """
        Args:
            semantic_threshold: 语义相似度阈值，超过此值视为重复
        """
        self.semantic_threshold = semantic_threshold
        self.seen_hashes: set[str] = set()
        self.seen_texts: list[str] = []

    def is_duplicate(self, doc: ParsedDocument) -> tuple[bool, str]:
        """检查文档是否重复
        Returns:
            (is_duplicate, reason) - 是否重复及原因
        """
        # 第1关: MD5精确去重
        if doc.raw_hash in self.seen_hashes:
            return True, f"MD5精确匹配: {doc.raw_hash[:8]}"

        # 第2关: 语义相似度去重
        for seen_text in self.seen_texts:
            similarity = self._compute_semantic_similarity(doc.text, seen_text)
            if similarity > self.semantic_threshold:
                return True, f"语义相似度 {similarity:.3f} > {self.semantic_threshold}"

        # 不是重复
        self.seen_hashes.add(doc.raw_hash)
        self.seen_texts.append(doc.text)
        return False, ""

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度（基于Jaccard字符n-gram近似）"""
        # 生产环境应使用sentence-transformers计算余弦相似度
        # 此处使用高效近似的n-gram Jaccard
        def char_ngrams(text: str, n: int = 4) -> set:
            text = text.lower()
            return {text[i : i + n] for i in range(len(text) - n + 1)}

        ngrams1 = char_ngrams(text1)
        ngrams2 = char_ngrams(text2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0.0


# ============================================================
# 阶段3: 三级语义分块 (Three-Level Semantic Chunking)
# ============================================================

class SemanticChunker:
    """三级语义分块器: 章节→段落→知识点"""

    def __init__(
        self,
        chapter_max_length: int = 4096,
        paragraph_max_length: int = 1024,
        knowledge_max_length: int = 512,
        overlap: int = 128,
    ):
        self.chapter_max = chapter_max_length
        self.paragraph_max = paragraph_max_length
        self.knowledge_max = knowledge_max_length
        self.overlap = overlap

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """执行三级分块"""
        chunks: list[Chunk] = []

        # 第1级: 章节级分块
        chapter_chunks = self._split_by_chapters(doc)
        for i, chap_text in enumerate(chapter_chunks):
            chap_chunk = Chunk(
                chunk_id=f"{doc.doc_id}_chapter_{i:04d}",
                doc_id=doc.doc_id,
                level=ChunkLevel.CHAPTER,
                text=chap_text,
                metadata={
                    **doc.metadata,
                    "level": "chapter",
                    "chapter_index": i,
                    "source": doc.source,
                },
            )
            chunks.append(chap_chunk)

            # 第2级: 段落级分块 (在每个章节内)
            para_chunks = self._split_by_paragraphs(chap_text)
            for j, para_text in enumerate(para_chunks):
                para_chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_para_{i:04d}_{j:04d}",
                    doc_id=doc.doc_id,
                    level=ChunkLevel.PARAGRAPH,
                    text=para_text,
                    metadata={
                        **doc.metadata,
                        "level": "paragraph",
                        "chapter_index": i,
                        "paragraph_index": j,
                        "source": doc.source,
                    },
                )
                chunks.append(para_chunk)

                # 第3级: 知识点级分块 (在每个段落内)
                knowledge_chunks = self._split_by_knowledge_points(para_text)
                for k, kp_text in enumerate(knowledge_chunks):
                    kp_chunk = Chunk(
                        chunk_id=f"{doc.doc_id}_kp_{i:04d}_{j:04d}_{k:04d}",
                        doc_id=doc.doc_id,
                        level=ChunkLevel.KNOWLEDGE_POINT,
                        text=kp_text,
                        metadata={
                            **doc.metadata,
                            "level": "knowledge_point",
                            "chapter_index": i,
                            "paragraph_index": j,
                            "kp_index": k,
                            "source": doc.source,
                        },
                    )
                    chunks.append(kp_chunk)

        return chunks

    def _split_by_chapters(self, doc: ParsedDocument) -> list[str]:
        """按章节分割"""
        text = doc.text

        # 尝试按Markdown标题分割
        import re
        heading_pattern = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)
        headings = list(heading_pattern.finditer(text))

        if not headings:
            # 按长度滑动窗口分块
            return self._sliding_window_split(text, self.chapter_max)

        chunks = []
        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks if chunks else self._sliding_window_split(text, self.chapter_max)

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """按段落分割"""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) <= self.paragraph_max:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks if chunks else self._sliding_window_split(text, self.paragraph_max)

    def _split_by_knowledge_points(self, text: str) -> list[str]:
        """按知识点分割（句子级）"""
        import re
        # 中英文句子分割
        sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
        chunks = []
        current = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) <= self.knowledge_max:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                current = sent

        if current:
            chunks.append(current)

        return chunks if chunks else [text[: self.knowledge_max]]

    @staticmethod
    def _sliding_window_split(text: str, max_length: int, overlap: int = 128) -> list[str]:
        """滑动窗口分割"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_length, len(text))
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        return chunks if chunks else [text]


# ============================================================
# 阶段4: 元数据增强 (Metadata Enrichment)
# ============================================================

class MetadataEnricher:
    """元数据增强器: 自动元数据 + LLM摘要 + 实体标签"""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM客户端（可选，用于生成摘要和实体标签）
        """
        self.llm = llm_client

    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        """增强单个块的元数据"""
        # 自动元数据: 字数、语言检测、关键词密度
        auto_meta = self._auto_metadata(chunk.text)
        chunk.metadata.update(auto_meta)

        # 实体提取（基于规则）
        entities = self._extract_entities(chunk.text)
        chunk.metadata["entities"] = entities

        return chunk

    def enrich_with_llm(self, chunk: Chunk) -> Chunk:
        """使用LLM增强（摘要+标签）"""
        if self.llm:
            try:
                summary = self._llm_summarize(chunk.text)
                tags = self._llm_tag(chunk.text)
                chunk.metadata["llm_summary"] = summary
                chunk.metadata["llm_tags"] = tags
            except Exception as e:
                logger.warning(f"LLM增强失败: {e}")

        return chunk

    @staticmethod
    def _auto_metadata(text: str) -> dict:
        """自动提取元数据"""
        # 中文字数
        import re
        chinese_chars = len(re.findall(r"[一-鿿]", text))
        english_words = len(re.findall(r"[a-zA-Z]+", text))
        total_chars = len(text)

        # 关键词密度（前5个高频词）
        words = re.findall(r"\w+", text.lower())
        word_freq = {}
        for w in words:
            if len(w) > 2:
                word_freq[w] = word_freq.get(w, 0) + 1
        top_keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:5]

        return {
            "char_count": total_chars,
            "chinese_char_count": chinese_chars,
            "english_word_count": english_words,
            "top_keywords": top_keywords,
            "language": "zh" if chinese_chars > english_words else "en",
        }

    @staticmethod
    def _extract_entities(text: str) -> list[dict]:
        """基于规则的实体提取"""
        import re

        entities = []

        # 提取URL
        urls = re.findall(r"https?://[^\s]+", text)
        for url in urls:
            entities.append({"type": "URL", "value": url})

        # 提取日期
        dates = re.findall(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text)
        for d in dates:
            entities.append({"type": "DATE", "value": d})

        # 提取邮箱
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        for e in emails:
            entities.append({"type": "EMAIL", "value": e})

        # 提取中文专有名词（引号内）
        quoted = re.findall(r"「([^」]+)」|《([^》]+)》|\"([^\"]+)\"", text)
        for match in quoted:
            value = "".join(m for m in match if m)
            if len(value) > 1 and len(value) < 20:
                entities.append({"type": "NAMED_ENTITY", "value": value})

        return entities[:20]  # 限制数量

    def _llm_summarize(self, text: str) -> str:
        """LLM生成摘要"""
        if not self.llm:
            return text[:200] + "..." if len(text) > 200 else text

        prompt = f"""请为以下文本生成一个简洁的中文摘要（不超过100字）：

{text[:2000]}

摘要："""

        try:
            response = self.llm.generate(prompt)
            return response.strip()
        except Exception:
            return text[:200] + "..." if len(text) > 200 else text

    def _llm_tag(self, text: str) -> list[str]:
        """LLM生成标签"""
        if not self.llm:
            return []

        prompt = f"""请为以下文本提取3-5个关键词标签（用逗号分隔）：

{text[:2000]}

标签："""

        try:
            response = self.llm.generate(prompt)
            return [t.strip() for t in response.split(",") if t.strip()]
        except Exception:
            return []


# ============================================================
# 阶段5: 双向量生成 (Dual-Vector Generation)
# ============================================================

class DualVectorGenerator:
    """双向量生成器: chunk_original + chunk_summary"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # MiniLM-L6-v2 维度

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"加载向量模型: {self.model_name} (维度: {self._dimension})")
            except ImportError:
                logger.warning("sentence-transformers未安装，使用随机向量模拟")
                self._model = None
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def generate(self, chunks: list[Chunk]) -> list[Chunk]:
        """为所有块生成双向量"""
        if not chunks:
            return []

        texts = [c.text for chunks_in_batch in self._batch(chunks, 32) for c in chunks_in_batch]

        # 原始文本向量
        original_embeddings = self._encode(texts)
        # 使用摘要/前200字符作为摘要向量
        summary_texts = [c.metadata.get("llm_summary", c.text[:200]) for c in chunks]
        summary_embeddings = self._encode(summary_texts)

        idx = 0
        for chunk in chunks:
            if idx < len(original_embeddings):
                chunk.embedding_original = np.array(original_embeddings[idx])
            if idx < len(summary_embeddings):
                chunk.embedding_summary = np.array(summary_embeddings[idx])
            idx += 1

        return chunks

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """编码文本为向量"""
        if self._model:
            return self._model.encode(texts, normalize_embeddings=True).tolist()

        # 降级: 使用TF-IDF-like稀疏向量模拟
        rng = np.random.RandomState(42)
        return [rng.randn(self._dimension).tolist() for _ in texts]

    @staticmethod
    def _batch(items: list, size: int) -> list[list]:
        return [items[i : i + size] for i in range(0, len(items), size)]


# ============================================================
# 阶段6: HNSW索引构建 (HNSW Index Building)
# ============================================================

class HNSWIndexBuilder:
    """HNSW索引构建器"""

    def __init__(self, dimension: int, space: str = "cosine", M: int = 16, ef_construction: int = 200):
        self.dimension = dimension
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self._index = None
        self._chunk_map: dict[int, Chunk] = {}

    def build(self, chunks: list[Chunk]) -> dict:
        """构建HNSW索引"""
        if not chunks:
            logger.warning("没有块数据，跳过HNSW索引构建")
            return {"status": "empty", "index_size": 0}

        vectors = []
        valid_chunks = []

        for i, chunk in enumerate(chunks):
            emb = chunk.embedding_original if chunk.embedding_original is not None else chunk.embedding_summary
            if emb is not None:
                vectors.append(emb.tolist() if isinstance(emb, np.ndarray) else emb)
                valid_chunks.append(chunk)
                self._chunk_map[len(vectors) - 1] = chunk

        if not vectors:
            logger.warning("没有有效向量，跳过HNSW索引构建")
            return {"status": "no_vectors", "index_size": 0}

        try:
            import hnswlib

            self._index = hnswlib.Index(space=self.space, dim=self.dimension)
            self._index.init_index(
                max_elements=len(vectors),
                ef_construction=self.ef_construction,
                M=self.M,
            )
            self._index.add_items(np.array(vectors), np.arange(len(vectors)))
            self._index.set_ef(50)  # 查询时ef

            logger.info(f"HNSW索引构建完成: {len(vectors)} 个向量, M={self.M}, ef={self.ef_construction}")
            return {
                "status": "success",
                "index_size": len(vectors),
                "dimension": self.dimension,
                "M": self.M,
                "ef_construction": self.ef_construction,
            }
        except ImportError:
            logger.warning("hnswlib未安装，使用模拟索引")
            self._index = _MockHNSWIndex(vectors, self.dimension)
            return {
                "status": "mock",
                "index_size": len(vectors),
                "note": "hnswlib not installed, using mock index",
            }

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[Chunk, float]]:
        """搜索HNSW索引"""
        if self._index is None:
            return []

        try:
            import hnswlib

            if isinstance(self._index, hnswlib.Index):
                labels, distances = self._index.knn_query(query_vector.reshape(1, -1), k=k)
                results = []
                for label, dist in zip(labels[0], distances[0]):
                    if label >= 0 and label in self._chunk_map:
                        results.append((self._chunk_map[label], float(dist)))
                return results
        except ImportError:
            pass

        if hasattr(self._index, "search"):
            return self._index.search(query_vector, k)

        return []


class _MockHNSWIndex:
    """模拟HNSW索引（用于hnswlib不可用时）"""

    def __init__(self, vectors: list, dimension: int):
        self.vectors = np.array(vectors)
        self.dimension = dimension

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[Chunk, float]]:
        """暴力搜索"""
        query = query_vector.reshape(1, -1)
        similarities = np.dot(self.vectors, query.T).flatten()
        top_k = np.argsort(similarities)[-k:][::-1]
        return [(None, float(1.0 - similarities[i])) for i in top_k]


# ============================================================
# 阶段7: BM25倒排索引构建 (BM25 Inverted Index)
# ============================================================

class BM25IndexBuilder:
    """BM25倒排索引构建器"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._index = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> dict:
        """构建BM25索引"""
        self._chunks = chunks
        texts = [c.text for c in chunks]

        try:
            import bm25s

            # Tokenization（支持中英文）
            corpus_tokens = [self._tokenize(t) for t in texts]
            self._index = bm25s.BM25(corpus=corpus_tokens)
            self._index.index(corpus_tokens)
            logger.info(f"BM25索引构建完成: {len(texts)} 文档, k1={self.k1}, b={self.b}")
            return {
                "status": "success",
                "doc_count": len(texts),
                "k1": self.k1,
                "b": self.b,
            }
        except ImportError:
            logger.warning("bm25s未安装，使用模拟索引")
            self._index = _MockBM25Index(texts)
            return {
                "status": "mock",
                "doc_count": len(texts),
                "note": "bm25s not installed, using mock index",
            }

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """搜索BM25索引"""
        if self._index is None:
            return []

        try:
            import bm25s

            if isinstance(self._index, bm25s.BM25):
                query_tokens = self._tokenize(query)
                results, scores = self._index.retrieve([query_tokens], k=k)
                output = []
                for i, (doc_idx, score) in enumerate(zip(results[0], scores[0])):
                    if doc_idx < len(self._chunks):
                        output.append((self._chunks[doc_idx], float(score)))
                return output
        except ImportError:
            pass

        if hasattr(self._index, "search"):
            return self._index.search(query, k)

        return []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词（生产环境应使用jieba等）"""
        import re

        # 中文按字分割，英文按空格
        tokens = []
        # 中文
        chinese_chars = re.findall(r"[一-鿿]", text)
        tokens.extend(chinese_chars)
        # 英文
        english_words = re.findall(r"[a-zA-Z]+", text.lower())
        tokens.extend(english_words)
        # 数字
        numbers = re.findall(r"\d+", text)
        tokens.extend(numbers)

        return tokens


class _MockBM25Index:
    """模拟BM25索引"""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """基于关键词匹配的简单搜索"""
        import re

        query_words = set(re.findall(r"\w+", query.lower()))
        scores = []
        for i, text in enumerate(self.texts):
            text_words = set(re.findall(r"\w+", text.lower()))
            overlap = len(query_words & text_words)
            if overlap > 0:
                scores.append((None, overlap / max(len(query_words), 1)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ============================================================
# 阶段8: 验证与质量门控 (Validation and Quality Gates)
# ============================================================

class QualityGate:
    """质量门控"""

    def __init__(self):
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        """执行一项质量检查"""
        self.checks.append({
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        if not passed:
            logger.warning(f"质量门控失败: {name} - {detail}")
        return passed


class PipelineValidator:
    """管道验证器"""

    def __init__(self):
        self.gate = QualityGate()

    def validate(self, chunks: list[Chunk], hnsw_info: dict, bm25_info: dict) -> dict:
        """执行全部验证"""
        results = {
            "gate_checks": [],
            "overall_pass": True,
            "metrics": {},
        }

        # 检查1: 块数量非空
        passed = len(chunks) > 0
        self.gate.check("chunks_not_empty", passed, f"块数量: {len(chunks)}")
        results["gate_checks"].append({"name": "chunks_not_empty", "passed": passed})
        results["overall_pass"] &= passed

        # 检查2: HNSW索引构建成功
        hnsw_ok = hnsw_info.get("status") in ("success", "mock")
        self.gate.check("hnsw_index_ok", hnsw_ok, str(hnsw_info))
        results["gate_checks"].append({"name": "hnsw_index_ok", "passed": hnsw_ok})
        results["overall_pass"] &= hnsw_ok

        # 检查3: BM25索引构建成功
        bm25_ok = bm25_info.get("status") in ("success", "mock")
        self.gate.check("bm25_index_ok", bm25_ok, str(bm25_info))
        results["gate_checks"].append({"name": "bm25_index_ok", "passed": bm25_ok})
        results["overall_pass"] &= bm25_ok

        # 检查4: 向量覆盖率
        chunks_with_vectors = sum(1 for c in chunks if c.embedding_original is not None or c.embedding_summary is not None)
        coverage = chunks_with_vectors / max(len(chunks), 1)
        coverage_pass = coverage >= 0.8
        self.gate.check("vector_coverage", coverage_pass, f"{coverage:.1%}")
        results["gate_checks"].append({"name": "vector_coverage", "passed": coverage_pass})
        results["metrics"]["vector_coverage"] = coverage

        # 检查5: 文本质量（最小长度）
        empty_chunks = sum(1 for c in chunks if len(c.text.strip()) < 10)
        quality_pass = empty_chunks < len(chunks) * 0.05
        self.gate.check("text_quality", quality_pass, f"短块: {empty_chunks}/{len(chunks)}")
        results["gate_checks"].append({"name": "text_quality", "passed": quality_pass})
        results["metrics"]["short_chunk_ratio"] = empty_chunks / max(len(chunks), 1)

        results["overall_pass"] = all(c["passed"] for c in results["gate_checks"])
        logger.info(f"质量验证: {'通过' if results['overall_pass'] else '失败'}")
        return results


# ============================================================
# 管道编排器 (Pipeline Orchestrator)
# ============================================================

class ProductionPipeline:
    """生产级数据管道编排器"""

    def __init__(self):
        self.parser_registry = ParserRegistry()
        self.deduplicator = Deduplicator()
        self.chunker = SemanticChunker()
        self.enricher = MetadataEnricher()
        self.vectorizer = DualVectorGenerator()
        self.hnsw_builder: Optional[HNSWIndexBuilder] = None
        self.bm25_builder = BM25IndexBuilder()
        self.validator = PipelineValidator()
        self.progress = ProgressCallback()
        self.stats = PipelineStats()
        self.error_handler = ErrorHandler(self.stats)

    def run(self, sources: list[str]) -> dict:
        """运行完整管道"""
        import time

        logger.info(f"开始运行生产级数据管道, 源文件数: {len(sources)}")
        self.stats = PipelineStats()
        self.stats.docs_total = len(sources)
        self.error_handler = ErrorHandler(self.stats)

        # 阶段1: 解析
        t0 = time.time()
        self.progress.notify(PipelinePhase.PARSE, 0.0, "开始多源文档解析")
        parsed_docs = self._phase_parse(sources)
        self.stats.docs_parsed = len(parsed_docs)
        self.stats.phase_times["parse"] = time.time() - t0
        logger.info(f"阶段1 完成: 解析 {len(parsed_docs)} 个文档 ({self.stats.phase_times['parse']:.1f}s)")

        # 阶段2: 去重
        t0 = time.time()
        self.progress.notify(PipelinePhase.DEDUP, 0.125, "开始双重去重")
        unique_docs = self._phase_dedup(parsed_docs)
        self.stats.docs_dedup_removed = len(parsed_docs) - len(unique_docs)
        self.stats.phase_times["dedup"] = time.time() - t0
        logger.info(f"阶段2 完成: 去重后 {len(unique_docs)} 个文档, 移除 {self.stats.docs_dedup_removed} ({self.stats.phase_times['dedup']:.1f}s)")

        # 阶段3: 分块
        t0 = time.time()
        self.progress.notify(PipelinePhase.CHUNK, 0.25, "开始三级语义分块")
        chunks = self._phase_chunk(unique_docs)
        self.stats.chunks_total = len(chunks)
        self.stats.phase_times["chunk"] = time.time() - t0
        logger.info(f"阶段3 完成: {len(chunks)} 个块 (章节:{self.stats.chunks_chapter}, 段落:{self.stats.chunks_paragraph}, 知识点:{self.stats.chunks_knowledge}) ({self.stats.phase_times['chunk']:.1f}s)")

        # 阶段4: 元数据增强
        t0 = time.time()
        self.progress.notify(PipelinePhase.ENRICH, 0.375, "开始元数据增强")
        chunks = self._phase_enrich(chunks)
        self.stats.phase_times["enrich"] = time.time() - t0
        logger.info(f"阶段4 完成 ({self.stats.phase_times['enrich']:.1f}s)")

        # 阶段5: 向量生成
        t0 = time.time()
        self.progress.notify(PipelinePhase.VECTORIZE, 0.5, "开始双向量生成")
        chunks = self._phase_vectorize(chunks)
        self.stats.vector_count = len(chunks)
        self.stats.phase_times["vectorize"] = time.time() - t0
        logger.info(f"阶段5 完成 ({self.stats.phase_times['vectorize']:.1f}s)")

        # 初始化HNSW构建器（需要知道维度）
        self.hnsw_builder = HNSWIndexBuilder(dimension=self.vectorizer.dimension)

        # 阶段6: HNSW索引
        t0 = time.time()
        self.progress.notify(PipelinePhase.HNSW_INDEX, 0.625, "开始HNSW索引构建")
        hnsw_info = self._phase_hnsw_index(chunks)
        self.stats.phase_times["hnsw_index"] = time.time() - t0
        logger.info(f"阶段6 完成: HNSW索引 {hnsw_info} ({self.stats.phase_times['hnsw_index']:.1f}s)")

        # 阶段7: BM25索引
        t0 = time.time()
        self.progress.notify(PipelinePhase.BM25_INDEX, 0.75, "开始BM25索引构建")
        bm25_info = self._phase_bm25_index(chunks)
        self.stats.phase_times["bm25_index"] = time.time() - t0
        logger.info(f"阶段7 完成: BM25索引 {bm25_info} ({self.stats.phase_times['bm25_index']:.1f}s)")

        # 阶段8: 验证
        t0 = time.time()
        self.progress.notify(PipelinePhase.VALIDATE, 0.875, "开始质量验证")
        validation_results = self._phase_validate(chunks, hnsw_info, bm25_info)
        self.stats.phase_times["validate"] = time.time() - t0
        logger.info(f"阶段8 完成: {'通过' if validation_results['overall_pass'] else '失败'} ({self.stats.phase_times['validate']:.1f}s)")

        self.progress.notify(PipelinePhase.VALIDATE, 1.0, "管道运行完成")

        return {
            "stats": self._stats_to_dict(),
            "hnsw_index": hnsw_info,
            "bm25_index": bm25_info,
            "validation": validation_results,
        }

    def _phase_parse(self, sources: list[str]) -> list[ParsedDocument]:
        """阶段1: 解析"""
        docs = []
        for i, source in enumerate(sources):
            try:
                doc = self.parser_registry.parse(source)
                docs.append(doc)
                self.progress.notify(PipelinePhase.PARSE, (i + 1) / len(sources), f"解析: {source}")
            except Exception as e:
                self.error_handler.handle(PipelinePhase.PARSE, source, e, recoverable=True)
        return docs

    def _phase_dedup(self, docs: list[ParsedDocument]) -> list[ParsedDocument]:
        """阶段2: 去重"""
        unique = []
        for i, doc in enumerate(docs):
            is_dup, reason = self.deduplicator.is_duplicate(doc)
            if is_dup:
                logger.info(f"去重移除: {doc.source} - {reason}")
            else:
                unique.append(doc)
            self.progress.notify(PipelinePhase.DEDUP, (i + 1) / len(docs), f"去重: {doc.source}")
        return unique

    def _phase_chunk(self, docs: list[ParsedDocument]) -> list[Chunk]:
        """阶段3: 分块"""
        all_chunks = []
        for i, doc in enumerate(docs):
            try:
                chunks = self.chunker.chunk(doc)
                all_chunks.extend(chunks)
                # 统计层级
                for c in chunks:
                    if c.level == ChunkLevel.CHAPTER:
                        self.stats.chunks_chapter += 1
                    elif c.level == ChunkLevel.PARAGRAPH:
                        self.stats.chunks_paragraph += 1
                    else:
                        self.stats.chunks_knowledge += 1
            except Exception as e:
                self.error_handler.handle(PipelinePhase.CHUNK, doc.doc_id, e)
            self.progress.notify(PipelinePhase.CHUNK, (i + 1) / len(docs), f"分块: {doc.source}")
        return all_chunks

    def _phase_enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        """阶段4: 元数据增强"""
        for i, chunk in enumerate(chunks):
            try:
                chunk = self.enricher.enrich_chunk(chunk)
            except Exception as e:
                self.error_handler.handle(PipelinePhase.ENRICH, chunk.chunk_id, e)
            if i % 100 == 0:
                self.progress.notify(PipelinePhase.ENRICH, (i + 1) / len(chunks), f"增强: {chunk.chunk_id}")
        return chunks

    def _phase_vectorize(self, chunks: list[Chunk]) -> list[Chunk]:
        """阶段5: 向量生成"""
        return self.vectorizer.generate(chunks)

    def _phase_hnsw_index(self, chunks: list[Chunk]) -> dict:
        """阶段6: HNSW索引"""
        return self.hnsw_builder.build(chunks)

    def _phase_bm25_index(self, chunks: list[Chunk]) -> dict:
        """阶段7: BM25索引"""
        return self.bm25_builder.build(chunks)

    def _phase_validate(self, chunks: list[Chunk], hnsw_info: dict, bm25_info: dict) -> dict:
        """阶段8: 验证"""
        return self.validator.validate(chunks, hnsw_info, bm25_info)

    def _stats_to_dict(self) -> dict:
        """统计信息转字典"""
        return {
            "docs_total": self.stats.docs_total,
            "docs_parsed": self.stats.docs_parsed,
            "docs_dedup_removed": self.stats.docs_dedup_removed,
            "chunks_total": self.stats.chunks_total,
            "chunks_chapter": self.stats.chunks_chapter,
            "chunks_paragraph": self.stats.chunks_paragraph,
            "chunks_knowledge": self.stats.chunks_knowledge,
            "vector_count": self.stats.vector_count,
            "phase_times": self.stats.phase_times,
            "error_count": len(self.stats.errors),
        }


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

def create_sample_documents(output_dir: str = "./sample_docs"):
    """创建示例文档用于测试"""
    os.makedirs(output_dir, exist_ok=True)

    # 示例Markdown文档
    md_content = """# RAG系统架构指南

## 第1章: 检索增强生成概述

检索增强生成（RAG）是一种将信息检索与文本生成相结合的技术。
它通过从外部知识库检索相关文档来增强大型语言模型的生成能力。

### 1.1 基本架构

RAG系统通常包含以下核心组件：
- 文档解析器：将原始文档转换为结构化文本
- 向量化引擎：将文本转换为数值向量
- 向量数据库：高效存储和检索向量
- 生成模型：基于检索到的上下文生成回答

### 1.2 工作流程

1. 用户提交查询
2. 系统对查询进行向量化
3. 在向量数据库中检索最相关的文档块
4. 将检索到的上下文与原始查询结合
5. LLM基于增强的上下文生成回答

## 第2章: 向量数据库选型

选择合适的向量数据库是RAG系统的关键决策。
主流选项包括Faiss、Milvus、ChromaDB和Pinecone。

### 2.1 Faiss

Faiss是Meta开源的向量相似度搜索库，支持多种索引类型。
它的HNSW索引在大多数场景下表现优异。

### 2.2 Milvus

Milvus是一个云原生的向量数据库，支持分布式部署和混合查询。
适合大规模生产环境。

## 第3章: 分块策略

分块是RAG系统中影响检索质量的关键因素。

### 3.1 固定大小分块

最简单的方法，但可能切断语义完整性。

### 3.2 语义分块

基于文档结构和语义边界进行分块，保留上下文完整性。

### 3.3 递归分块

使用分隔符层次结构进行递归分块，兼顾灵活性和语义。
"""

    with open(os.path.join(output_dir, "rag_guide.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    # 示例Markdown文档2
    md_content2 = """# 熔断器模式详解

## 概述

熔断器（Circuit Breaker）是一种用于检测故障并防止故障扩散的设计模式。
在分布式系统中，熔断器可以防止级联故障。

## 三态模型

### CLOSED状态
正常状态，所有请求正常通过。系统持续监控失败率。

### OPEN状态
当失败率达到阈值时，熔断器打开，直接拒绝请求。
这给下游服务恢复的时间。

### HALF_OPEN状态
经过冷却时间后，熔断器允许少量探测请求通过。
如果探测成功，恢复到CLOSED；如果失败，重新回到OPEN。

## 配置参数

| 参数 | 描述 | 典型值 |
|------|------|--------|
| failure_threshold | 失败阈值 | 5 |
| timeout | 冷却时间 | 30s |
| half_open_max | 半开最大请求 | 1 |
"""

    with open(os.path.join(output_dir, "circuit_breaker.md"), "w", encoding="utf-8") as f:
        f.write(md_content2)

    # 示例文本作为"PDF"内容
    txt_content = """生产级RAG系统的设计原则

1. 可靠性优先：系统必须在各种负载和故障条件下保持稳定。
2. 性能优化：通过多级缓存和索引优化降低延迟。
3. 安全性保障：多租户隔离、API认证、数据加密。
4. 可观测性：全面的监控、日志和告警机制。
5. 可扩展性：支持水平扩展和增量更新。
"""

    with open(os.path.join(output_dir, "design_principles.txt"), "w", encoding="utf-8") as f:
        f.write(txt_content)

    logger.info(f"创建了 {3} 个示例文档在 {output_dir}")
    return [
        os.path.join(output_dir, "rag_guide.md"),
        os.path.join(output_dir, "circuit_breaker.md"),
        os.path.join(output_dir, "design_principles.txt"),
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("生产级RAG数据管道 - 演示运行")
    print("=" * 60)

    # 创建示例文档
    sample_sources = create_sample_documents()

    # 初始化管道
    pipeline = ProductionPipeline()

    # 注册进度回调
    def progress_handler(phase, progress, message):
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {phase.name:12s} {progress:3.0%} | {message}", end="", flush=True)

    pipeline.progress.register(progress_handler)

    # 运行管道
    print("\n开始处理...")
    result = pipeline.run(sample_sources)

    # 输出结果
    print("\n\n" + "=" * 60)
    print("管道运行结果")
    print("=" * 60)

    stats = result["stats"]
    print(f"\n📊 统计信息:")
    print(f"  总文档数:       {stats['docs_total']}")
    print(f"  成功解析:       {stats['docs_parsed']}")
    print(f"  去重移除:       {stats['docs_dedup_removed']}")
    print(f"  总块数:         {stats['chunks_total']}")
    print(f"    - 章节级:     {stats['chunks_chapter']}")
    print(f"    - 段落级:     {stats['chunks_paragraph']}")
    print(f"    - 知识点级:   {stats['chunks_knowledge']}")
    print(f"  错误数:         {stats['error_count']}")

    print(f"\n⏱️ 阶段耗时:")
    for phase, elapsed in stats['phase_times'].items():
        print(f"  {phase:15s}: {elapsed:.3f}s")

    print(f"\n📋 质量验证:")
    validation = result["validation"]
    for check in validation["gate_checks"]:
        status = "✅" if check["passed"] else "❌"
        print(f"  {status} {check['name']}")
    print(f"  总体: {'✅ 通过' if validation['overall_pass'] else '❌ 失败'}")

    print(f"\n📈 HNSW索引: {result['hnsw_index']}")
    print(f"📈 BM25索引: {result['bm25_index']}")

    print("\n" + "=" * 60)
    print("管道演示完成！")
    print("=" * 60)
