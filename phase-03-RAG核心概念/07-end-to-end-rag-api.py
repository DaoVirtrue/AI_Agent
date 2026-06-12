"""
=============================================================================
07-end-to-end-rag-api.py — 完整的RAG（检索增强生成）FastAPI服务
=============================================================================

描述：
    从零构建的端到端RAG API服务，实现文档上传→解析→分块→嵌入→存储→检索→生成的
    完整流水线。支持 PDF / DOCX / MD / TXT / HTML 五种文档格式，使用 OpenAI 嵌入
    模型和 ChromaDB 向量数据库，提供 RESTful API 接口。

API端点：
    POST   /documents/upload    上传文档，后台解析→分块→嵌入→存储
    GET    /search              语义检索已索引的文档片段
    POST   /chat/completions    基于RAG的对话生成（检索增强生成）
    GET    /health              健康检查（OpenAI + ChromaDB 连通性）

启动方式：
    uvicorn 07-end-to-end-rag-api:app --reload --host 0.0.0.0 --port 8000

    或直接运行：
    python 07-end-to-end-rag-api.py

依赖安装：
    pip install fastapi uvicorn python-multipart openai chromadb pydantic
    pip install PyPDF2 python-docx beautifulsoup4 markdown

环境变量：
    OPENAI_API_KEY — 必需，OpenAI API密钥
    OPENAI_EMBEDDING_MODEL — 可选，默认 text-embedding-3-small
    OPENAI_CHAT_MODEL — 可选，默认 gpt-4o-mini
    LOG_LEVEL — 可选，日志级别，默认 INFO

示例curl命令：
    # 健康检查
    curl http://localhost:8000/health

    # 上传文档
    curl -X POST http://localhost:8000/documents/upload \
      -F "file=@example.pdf"

    # 语义检索
    curl "http://localhost:8000/search?q=什么是RAG&top_k=5"

    # RAG对话
    curl -X POST http://localhost:8000/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"query":"什么是检索增强生成？","top_k":5}'

    # 带对话历史的RAG对话
    curl -X POST http://localhost:8000/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "query":"它的优点是什么？",
        "conversation_history":[
          {"role":"user","content":"什么是RAG？"},
          {"role":"assistant","content":"RAG是检索增强生成..."}
        ],
        "top_k":5
      }'

=============================================================================
"""

import asyncio
import contextlib
import datetime
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb
import numpy as np
import uvicorn
from chromadb.config import Settings as ChromaSettings
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 配置
# =============================================================================

class Settings:
    """应用配置类，支持从环境变量读取"""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_FILE_SIZE: int = int(
        os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024))
    )  # 10 MB
    ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".docx", ".md", ".txt", ".html"}
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")
    TOP_K_DEFAULT: int = int(os.getenv("TOP_K_DEFAULT", "5"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()


# =============================================================================
# 日志设置
# =============================================================================

def setup_logging() -> logging.Logger:
    """配置并返回应用日志记录器。

    设置控制台输出，包含时间戳、日志级别、模块名和消息内容。
    日志级别从 settings.LOG_LEVEL 读取。

    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    logger = logging.getLogger("rag_api")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # 避免重复添加 handler
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# =============================================================================
# Pydantic 数据模型
# =============================================================================

class DocumentUploadResponse(BaseModel):
    """文档上传成功后返回的响应模型"""
    document_id: str = Field(..., description="文档唯一标识符（UUID）")
    filename: str = Field(..., description="原始文件名")
    chunks_count: int = Field(default=0, description="分块数量（处理完成后更新）")
    status: str = Field(..., description="处理状态：processing / completed / failed")
    message: str = Field(..., description="人类可读的状态消息")


class SearchResult(BaseModel):
    """单条检索结果模型"""
    content: str = Field(..., description="检索到的文本片段内容")
    score: float = Field(..., description="相似度分数（距离度量）")
    metadata: dict = Field(default_factory=dict, description="关联的元数据")
    document_id: str = Field(..., description="来源文档ID")


class SearchResponse(BaseModel):
    """语义检索响应模型"""
    query: str = Field(..., description="用户查询文本")
    results: List[SearchResult] = Field(default_factory=list, description="检索结果列表")
    total: int = Field(..., description="返回结果总数")
    search_time_ms: float = Field(..., description="检索耗时（毫秒）")


class ChatMessage(BaseModel):
    """对话消息模型"""
    role: str = Field(
        ...,
        pattern=r"^(user|assistant|system)$",
        description="消息角色：user / assistant / system",
    )
    content: str = Field(..., min_length=1, description="消息内容")


class ChatRequest(BaseModel):
    """RAG对话请求模型"""
    query: str = Field(..., min_length=1, description="用户查询文本")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        description="对话历史记录",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="检索结果数量（1-20）",
    )


class ChatResponse(BaseModel):
    """RAG对话响应模型"""
    answer: str = Field(..., description="LLM生成的回答")
    sources: List[SearchResult] = Field(
        default_factory=list, description="回答所引用的检索结果"
    )
    query: str = Field(..., description="原始查询文本")
    generation_time_ms: float = Field(..., description="生成耗时（毫秒）")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务整体状态")
    version: str = Field(..., description="API版本号")
    documents_count: int = Field(..., description="已索引文档数量")
    uptime_seconds: float = Field(..., description="服务运行时长（秒）")
    openai_available: bool = Field(..., description="OpenAI API是否可达")
    chroma_available: bool = Field(..., description="ChromaDB是否正常")


class ErrorResponse(BaseModel):
    """统一错误响应模型"""
    error: str = Field(..., description="错误类型")
    detail: str = Field(..., description="错误详情")
    timestamp: str = Field(..., description="错误发生时间（ISO 8601）")


# =============================================================================
# 文档处理服务
# =============================================================================

class DocumentProcessor:
    """文档处理核心服务。

    负责文档解析、文本分块、向量嵌入、ChromaDB存储、语义检索和RAG回答生成。
    封装了从原始文档到可检索知识库的完整流水线。

    Attributes:
        settings: 应用配置实例
        openai_client: OpenAI API客户端
        chroma_client: ChromaDB客户端
        collection: ChromaDB集合实例
        upload_dir: 上传文件存储目录的Path对象
    """

    def __init__(self, config: Settings):
        """初始化文档处理器。

        Args:
            config: 应用配置实例
        """
        self.settings = config
        self._logger = logging.getLogger("rag_api.processor")

        # 初始化 OpenAI 客户端
        if config.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self._logger.info("OpenAI客户端初始化完成，模型=%s", config.OPENAI_EMBEDDING_MODEL)
        else:
            self.openai_client = None
            self._logger.warning("未设置OPENAI_API_KEY，嵌入和生成功能将不可用")

        # 初始化 ChromaDB 客户端（持久化模式）
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._logger.info(
            "ChromaDB客户端初始化完成，持久化目录=%s", config.CHROMA_PERSIST_DIR
        )

        # 获取或创建集合
        self.collection = self._get_or_create_collection()

        # 确保上传目录存在
        self.upload_dir = Path(config.UPLOAD_DIR)
        os.makedirs(self.upload_dir, exist_ok=True)

        # 文件锁，防止并发写入冲突
        self._lock = asyncio.Lock()

    def _get_or_create_collection(self):
        """获取或创建ChromaDB集合。

        Returns:
            chromadb.Collection: ChromaDB集合实例
        """
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=self.settings.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            count = collection.count()
            self._logger.info(
                "ChromaDB集合'%s'就绪，当前文档数=%d",
                self.settings.COLLECTION_NAME,
                count,
            )
            return collection
        except Exception as e:
            self._logger.error("创建/获取ChromaDB集合失败: %s", e)
            raise

    # ---------- 文档解析 ----------

    def parse_file(self, file_path: str) -> List[dict]:
        """解析文档文件，提取文本内容和元数据。

        根据文件扩展名自动选择解析器：
        - .pdf  → PDF解析（PyPDF2）
        - .docx → Word解析（python-docx）
        - .md   → Markdown解析
        - .txt  → 纯文本解析
        - .html → HTML解析（BeautifulSoup）

        Args:
            file_path: 文档文件的绝对路径

        Returns:
            List[dict]: 解析结果列表，每项包含 "text" 和 "metadata" 字段

        Raises:
            ValueError: 不支持的文件类型
            Exception: 解析过程中的各类错误
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件类型: {extension}。"
                f"支持的类型: {self.settings.ALLOWED_EXTENSIONS}"
            )

        self._logger.info("开始解析文件: %s (类型=%s)", file_path.name, extension)

        metadata = {
            "filename": file_path.name,
            "file_type": extension,
            "file_size": file_path.stat().st_size,
            "parse_time": datetime.datetime.now().isoformat(),
        }

        try:
            if extension == ".pdf":
                text = self._parse_pdf(file_path)
            elif extension == ".docx":
                text = self._parse_docx(file_path)
            elif extension == ".md":
                text = self._parse_markdown(file_path)
            elif extension == ".txt":
                text = self._parse_txt(file_path)
            elif extension == ".html":
                text = self._parse_html(file_path)
            else:
                raise ValueError(f"未实现的解析器: {extension}")

        except Exception as e:
            self._logger.error("解析文件失败: %s, 错误: %s", file_path.name, e)
            raise

        if not text or not text.strip():
            self._logger.warning("文件 %s 解析后无有效文本内容", file_path.name)
            return [{"text": "", "metadata": metadata}]

        self._logger.info(
            "文件 %s 解析完成，提取字符数=%d", file_path.name, len(text)
        )
        return [{"text": text, "metadata": metadata}]

    def _parse_pdf(self, file_path: Path) -> str:
        """解析PDF文件，提取文本内容。

        Args:
            file_path: PDF文件路径

        Returns:
            str: 提取的文本内容
        """
        self._logger.debug("使用PyPDF2解析PDF: %s", file_path.name)
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "需要安装PyPDF2来解析PDF文件。请运行: pip install PyPDF2"
            )

        reader = PdfReader(str(file_path))
        text_parts = []
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            self._logger.debug("PDF第%d页提取完成，字符数=%d", page_num, len(page_text or ""))

        full_text = "\n\n".join(text_parts)
        return full_text

    def _parse_docx(self, file_path: Path) -> str:
        """解析DOCX文件，提取文本内容。

        同时提取段落和表格中的文本。

        Args:
            file_path: DOCX文件路径

        Returns:
            str: 提取的文本内容
        """
        self._logger.debug("使用python-docx解析DOCX: %s", file_path.name)
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "需要安装python-docx来解析DOCX文件。请运行: pip install python-docx"
            )

        doc = Document(str(file_path))
        text_parts = []

        # 提取段落文本
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        # 提取表格文本
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text for cell in row.cells if cell.text.strip()
                )
                if row_text.strip():
                    text_parts.append(row_text)

        full_text = "\n".join(text_parts)
        return full_text

    def _parse_markdown(self, file_path: Path) -> str:
        """解析Markdown文件，返回纯文本内容。

        Args:
            file_path: Markdown文件路径

        Returns:
            str: 文件的纯文本内容
        """
        self._logger.debug("解析Markdown: %s", file_path.name)
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_txt(self, file_path: Path) -> str:
        """解析纯文本文件。

        尝试多种编码方式读取（UTF-8 → GBK → Latin-1）。

        Args:
            file_path: 纯文本文件路径

        Returns:
            str: 文件文本内容
        """
        self._logger.debug("解析TXT: %s", file_path.name)
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # 最后尝试，忽略无法解码的字符
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _parse_html(self, file_path: Path) -> str:
        """解析HTML文件，提取纯文本内容。

        使用BeautifulSoup去除HTML标签，保留文本。

        Args:
            file_path: HTML文件路径

        Returns:
            str: 提取的纯文本内容
        """
        self._logger.debug("使用BeautifulSoup解析HTML: %s", file_path.name)
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "需要安装beautifulsoup4来解析HTML文件。"
                "请运行: pip install beautifulsoup4"
            )

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")
        # 移除 script 和 style 标签
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text

    # ---------- 文本分块 ----------

    def chunk_text(self, text: str, metadata: dict = None) -> List[dict]:
        """使用递归字符分割策略将文本切分为重叠的块。

        按层级分隔符递归分割文本：段落 → 换行 → 句子 → 空格 → 字符。
        每个块的大小为 settings.CHUNK_SIZE，重叠为 settings.CHUNK_OVERLAP。

        Args:
            text: 待分块的原始文本
            metadata: 与文本关联的元数据（会合并到每个块中）

        Returns:
            List[dict]: 分块结果列表，每项包含 "content", "metadata", "chunk_index"
        """
        chunk_size = self.settings.CHUNK_SIZE
        chunk_overlap = self.settings.CHUNK_OVERLAP

        if not text or not text.strip():
            self._logger.warning("分块文本为空，返回空列表")
            return []

        self._logger.debug(
            "开始分块: 文本长度=%d, chunk_size=%d, overlap=%d",
            len(text),
            chunk_size,
            chunk_overlap,
        )

        # 递归字符分割器所使用的分隔符（按优先级从高到低）
        separators = ["\n\n", "\n", "。", "；", "，", ".", ";", ",", " ", ""]

        chunks = self._split_text_recursive(text, separators, chunk_size, chunk_overlap)

        # 为每个块附加元数据
        result = []
        base_metadata = (metadata or {}).copy()
        for i, chunk_content in enumerate(chunks):
            if chunk_content.strip():
                chunk_entry = {
                    "content": chunk_content.strip(),
                    "metadata": {
                        **base_metadata,
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        "char_count": len(chunk_content),
                    },
                }
                result.append(chunk_entry)

        self._logger.info("文本分块完成: 共生成%d个块", len(result))
        return result

    def _split_text_recursive(
        self,
        text: str,
        separators: List[str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """递归地按分隔符切分文本。

        尝试用当前分隔符切分。如果某段仍大于chunk_size，
        则使用下一个分隔符继续递归切分。

        Args:
            text: 待切分文本
            separators: 分隔符列表（按优先级排序）
            chunk_size: 目标块大小
            chunk_overlap: 块重叠大小

        Returns:
            List[str]: 切分后的文本片段列表
        """
        # 如果文本本身就在限制内，直接返回
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # 尝试当前分隔符
        separator = separators[0] if separators else ""

        if separator:
            splits = text.split(separator)
        else:
            # 无分隔符可用时按字符切分
            splits = list(text)

        # 合并切分结果，保持每块在 chunk_size 限制内
        final_chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(separator) + len(split) <= chunk_size:
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split
            else:
                # 当前块已满
                if current_chunk:
                    # 检查是否需要递归切分
                    if len(current_chunk) > chunk_size and len(separators) > 1:
                        sub_chunks = self._split_text_recursive(
                            current_chunk,
                            separators[1:],
                            chunk_size,
                            chunk_overlap,
                        )
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(current_chunk)

                # 开始新块，处理split本身是否超过chunk_size
                if len(split) > chunk_size and len(separators) > 1:
                    sub_chunks = self._split_text_recursive(
                        split, separators[1:], chunk_size, chunk_overlap
                    )
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        # 处理最后一块
        if current_chunk:
            if len(current_chunk) > chunk_size and len(separators) > 1:
                sub_chunks = self._split_text_recursive(
                    current_chunk, separators[1:], chunk_size, chunk_overlap
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(current_chunk)

        # 添加重叠（合并相邻块的尾部到下一块的开头）
        if chunk_overlap > 0 and len(final_chunks) > 1:
            overlapped_chunks = [final_chunks[0]]
            for i in range(1, len(final_chunks)):
                prev_tail = final_chunks[i - 1][-chunk_overlap:]
                new_chunk = prev_tail + " " + final_chunks[i]
                overlapped_chunks.append(new_chunk)
            return overlapped_chunks

        return final_chunks

    # ---------- 向量嵌入 ----------

    def get_embedding(self, text: str) -> List[float]:
        """调用OpenAI嵌入API生成文本的向量表示。

        包含速率限制自动重试机制（最多3次，指数退避）。

        Args:
            text: 待嵌入的文本

        Returns:
            List[float]: 嵌入向量（浮点数列表）

        Raises:
            RuntimeError: OpenAI客户端未初始化
            Exception: API调用失败（所有重试耗尽后）
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI客户端未初始化，请设置OPENAI_API_KEY环境变量")

        if not text or not text.strip():
            self._logger.warning("嵌入文本为空，返回零向量")
            return [0.0] * 1536  # text-embedding-3-small 默认维度

        # 截断过长的文本（OpenAI嵌入模型有token限制）
        max_chars = 8000
        if len(text) > max_chars:
            self._logger.debug("文本过长(%d字符)，截断至%d字符", len(text), max_chars)
            text = text[:max_chars]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.settings.OPENAI_EMBEDDING_MODEL,
                    input=text,
                )
                embedding = response.data[0].embedding
                self._logger.debug(
                    "嵌入生成成功: 维度=%d, 模型=%s",
                    len(embedding),
                    self.settings.OPENAI_EMBEDDING_MODEL,
                )
                return embedding

            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                    self._logger.warning(
                        "速率限制，第%d次重试等待%ds: %s",
                        attempt + 1,
                        wait_time,
                        e,
                    )
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"OpenAI嵌入API调用失败（已重试{max_retries}次）: {e}"
                        ) from e
                else:
                    self._logger.error("嵌入生成失败: %s", e)
                    raise

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量。

        对每个文本逐个调用嵌入API。注意：OpenAI的embeddings.create
        也支持传入文本列表进行真正的批量处理，这里为了更好的错误处理和
        进度日志，逐个处理。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        embeddings = []
        total = len(texts)
        for i, text in enumerate(texts):
            try:
                emb = self.get_embedding(text)
                embeddings.append(emb)
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    self._logger.info("嵌入进度: %d/%d", i + 1, total)
            except Exception as e:
                self._logger.error("嵌入第%d个文本时失败: %s", i, e)
                raise
        return embeddings

    # ---------- 文档索引 ----------

    def index_document(self, file_path: str, document_id: str) -> int:
        """执行完整的文档索引流水线：解析→分块→嵌入→存储。

        Args:
            file_path: 文档文件路径
            document_id: 文档唯一标识符

        Returns:
            int: 成功索引的块数量

        Raises:
            ValueError: 解析或分块失败
            RuntimeError: 嵌入或存储失败
        """
        self._logger.info("开始索引文档: document_id=%s, file=%s", document_id, file_path)

        # 步骤1：解析文档
        try:
            parsed = self.parse_file(file_path)
        except Exception as e:
            self._logger.error("文档解析失败: %s", e)
            raise ValueError(f"文档解析失败: {e}") from e

        if not parsed or not parsed[0].get("text", "").strip():
            raise ValueError("文档解析结果为空，无可索引的文本内容")

        text = parsed[0]["text"]
        file_metadata = parsed[0]["metadata"]
        file_metadata["document_id"] = document_id

        # 步骤2：文本分块
        chunks = self.chunk_text(text, metadata=file_metadata)
        if not chunks:
            raise ValueError("文本分块结果为空")

        self._logger.info("分块完成: %d个块", len(chunks))

        # 步骤3：批量生成嵌入
        chunk_texts = [c["content"] for c in chunks]
        try:
            embeddings = self._batch_embed(chunk_texts)
        except Exception as e:
            self._logger.error("批量嵌入生成失败: %s", e)
            raise RuntimeError(f"嵌入生成失败: {e}") from e

        # 步骤4：存储到ChromaDB
        try:
            ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    **c["metadata"],
                    "document_id": document_id,
                }
                for c in chunks
            ]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas,
            )
            self._logger.info(
                "ChromaDB存储完成: %d个向量已写入集合'%s'",
                len(ids),
                self.settings.COLLECTION_NAME,
            )
        except Exception as e:
            self._logger.error("ChromaDB存储失败: %s", e)
            raise RuntimeError(f"向量存储失败: {e}") from e

        return len(chunks)

    # ---------- 语义检索 ----------

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """执行语义检索：将查询嵌入后搜索ChromaDB中最相似的文档块。

        Args:
            query: 查询文本
            top_k: 返回的最大结果数

        Returns:
            List[SearchResult]: 按相似度排序的检索结果列表
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        self._logger.info("执行检索: query='%s', top_k=%d", query[:100], top_k)

        # 生成查询嵌入
        try:
            query_embedding = self.get_embedding(query)
        except Exception as e:
            self._logger.error("查询嵌入生成失败: %s", e)
            raise RuntimeError(f"查询嵌入失败: {e}") from e

        # 在ChromaDB中检索
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            self._logger.error("ChromaDB查询失败: %s", e)
            raise RuntimeError(f"向量检索失败: {e}") from e

        # 构造返回结果
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = (
                    results["distances"][0][i]
                    if results.get("distances")
                    else 0.0
                )
                # ChromaDB的cosine距离在[0, 2]区间，转换为相似度分数
                score = 1.0 - (distance / 2.0) if distance is not None else 0.0

                metadata = (
                    results["metadatas"][0][i]
                    if results.get("metadatas")
                    else {}
                )
                content = (
                    results["documents"][0][i]
                    if results.get("documents")
                    else ""
                )

                search_results.append(
                    SearchResult(
                        content=content,
                        score=round(score, 4),
                        metadata=metadata or {},
                        document_id=metadata.get("document_id", "unknown"),
                    )
                )

        self._logger.info("检索完成: 返回%d条结果", len(search_results))
        return search_results

    # ---------- RAG回答生成 ----------

    def generate_answer(
        self,
        query: str,
        contexts: List[str],
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> str:
        """基于检索到的上下文生成RAG回答。

        构建包含检索上下文和对话历史的提示词，调用OpenAI Chat Completion
        生成流式或非流式的回答。

        Args:
            query: 用户查询文本
            contexts: 检索到的上下文文本列表
            conversation_history: 可选的对话历史记录

        Returns:
            str: LLM生成的回答文本

        Raises:
            RuntimeError: OpenAI客户端未初始化或API调用失败
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI客户端未初始化，请设置OPENAI_API_KEY环境变量")

        # 构建系统提示词（中文，包含RAG指令）
        system_prompt = (
            "你是一个基于检索增强生成（RAG）的专业AI助手。\n\n"
            "请严格遵循以下规则来回答问题：\n"
            "1. 优先使用【提供的上下文信息】来回答问题。\n"
            "2. 如果上下文中包含答案，请基于上下文给出准确、详细的回答。\n"
            "3. 如果上下文中没有足够的信息来回答问题，请明确说明"
            "'根据提供的文档资料，我无法回答这个问题'，"
            "然后基于你自己的知识给出补充说明，并明确标注哪些来自文档、哪些来自常识。\n"
            "4. 不要在回答中编造文档中没有提到的事实。\n"
            "5. 回答时应引用具体的文档片段来支持你的结论。\n"
            "6. 使用简洁、专业的中文进行回答。\n"
            "7. 如果上下文信息之间存在矛盾，请指出矛盾并给出综合分析。"
        )

        # 构建上下文部分
        context_text = ""
        if contexts:
            context_parts = []
            for i, ctx in enumerate(contexts, 1):
                context_parts.append(f"[文档片段 {i}]\n{ctx}")
            context_text = "\n\n".join(context_parts)

        user_content = (
            f"【提供的上下文信息】\n"
            f"{context_text}\n\n"
            f"【用户问题】\n"
            f"{query}\n\n"
            f"请基于以上上下文信息回答用户的问题。"
        )

        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 添加对话历史（如果有）
        if conversation_history:
            for msg in conversation_history[-10:]:  # 最多保留最近10条历史
                messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户查询
        messages.append({"role": "user", "content": user_content})

        # 调用OpenAI Chat Completion
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.settings.OPENAI_CHAT_MODEL,
                    messages=messages,
                    temperature=0.3,  # 较低温度以减少幻觉
                    max_tokens=2048,
                )
                answer = response.choices[0].message.content
                self._logger.info(
                    "回答生成成功: 长度=%d字符, 模型=%s",
                    len(answer or ""),
                    self.settings.OPENAI_CHAT_MODEL,
                )
                return answer or ""

            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = 2 ** (attempt + 1)
                    self._logger.warning(
                        "速率限制，第%d次重试等待%ds",
                        attempt + 1,
                        wait_time,
                    )
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"OpenAI Chat API调用失败（已重试{max_retries}次）: {e}"
                        ) from e
                else:
                    self._logger.error("回答生成失败: %s", e)
                    raise

    def get_document_count(self) -> int:
        """获取当前已索引的唯一文档数量。

        Returns:
            int: 唯一文档ID的数量
        """
        try:
            count = self.collection.count()
            if count == 0:
                return 0
            # 获取所有元数据并统计唯一document_id
            all_data = self.collection.get(include=["metadatas"])
            if all_data["metadatas"]:
                unique_doc_ids = set(
                    m.get("document_id", "")
                    for m in all_data["metadatas"]
                    if m and m.get("document_id")
                )
                return len(unique_doc_ids)
            return 0
        except Exception as e:
            self._logger.error("获取文档计数失败: %s", e)
            return 0


# =============================================================================
# FastAPI 应用初始化
# =============================================================================

app = FastAPI(
    title="RAG API - Phase 03",
    description="从零构建的RAG（检索增强生成）API服务 — 支持文档上传、语义检索、RAG对话生成",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局处理器实例（启动时初始化）
processor: Optional[DocumentProcessor] = None
# 服务启动时间（用于计算uptime）
startup_time: Optional[datetime.datetime] = None


# =============================================================================
# 应用生命周期事件
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """应用启动事件：初始化处理器、创建目录、记录启动信息。"""
    global processor, startup_time

    startup_time = datetime.datetime.now()
    logger.info("=" * 60)
    logger.info("RAG API 服务启动中...")
    logger.info("启动时间: %s", startup_time.isoformat())
    logger.info("=" * 60)

    # 创建必要目录
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

    # 初始化文档处理器
    try:
        processor = DocumentProcessor(settings)
        doc_count = processor.get_document_count()
        logger.info("文档处理器初始化完成，已索引文档数=%d", doc_count)
    except Exception as e:
        logger.error("文档处理器初始化失败: %s", e)
        logger.warning("服务将以降级模式运行")
        processor = None

    logger.info("RAG API 服务就绪，监听 http://0.0.0.0:8000")
    logger.info("API文档: http://localhost:8000/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件：清理临时文件、关闭连接。"""
    logger.info("RAG API 服务正在关闭...")

    # 清理上传目录中的临时文件（仅删除由本会话创建的文件）
    # 保留上传目录和chroma数据目录，以便重启后继续使用
    try:
        temp_dir = Path(settings.UPLOAD_DIR) / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info("临时目录已清理: %s", temp_dir)
    except Exception as e:
        logger.warning("清理临时文件时出错: %s", e)

    logger.info("RAG API 服务已关闭")


# =============================================================================
# 辅助函数
# =============================================================================

def validate_file(file: UploadFile) -> None:
    """验证上传文件的扩展名和大小。

    Args:
        file: FastAPI UploadFile 对象

    Raises:
        HTTPException: 415 (不支持的文件类型) 或 413 (文件过大)
    """
    if not file.filename:
        raise HTTPException(status_code=422, detail="文件名不能为空")

    # 检查扩展名
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"不支持的文件类型: {ext}。"
                f"支持的类型: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
            ),
        )

    # 注意：UploadFile 无法直接检测大小，需要先读取内容
    # 在实际生产环境中建议使用中间件或代理层（如Nginx）限制请求体大小


async def save_upload_file(file: UploadFile, destination: Path) -> int:
    """将上传文件保存到磁盘。

    Args:
        file: FastAPI UploadFile 对象
        destination: 目标文件路径

    Returns:
        int: 写入的字节数

    Raises:
        HTTPException: 文件过大或写入失败
    """
    try:
        total_size = 0
        with open(destination, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 每次读取1MB
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > settings.MAX_FILE_SIZE:
                    # 删除已写入的部分
                    buffer.close()
                    if destination.exists():
                        destination.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"文件大小超过限制 "
                            f"({total_size} > {settings.MAX_FILE_SIZE} 字节)"
                        ),
                    )
                buffer.write(chunk)
        return total_size
    except HTTPException:
        raise
    except Exception as e:
        logger.error("保存文件失败: %s", e)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}") from e


# =============================================================================
# API 端点
# =============================================================================

@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    summary="上传并索引文档",
    description="上传PDF/DOCX/MD/TXT/HTML文档，后台自动解析→分块→嵌入→存储到向量数据库",
    responses={
        200: {"description": "文件上传成功，后台处理中"},
        413: {"description": "文件大小超过10MB限制"},
        415: {"description": "不支持的文件类型"},
        422: {"description": "请求参数验证失败"},
        500: {"description": "服务器内部错误"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="要上传的文档文件"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """上传文档并触发后台索引流水线。

    curl示例:
        curl -X POST http://localhost:8000/documents/upload \\
          -F "file=@example.pdf"
    """
    logger.info("收到文档上传请求: filename=%s", file.filename)

    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=422, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"不支持的文件类型: {ext}。"
                f"支持的类型: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
            ),
        )

    if processor is None:
        raise HTTPException(
            status_code=500,
            detail="文档处理器未初始化，请检查服务日志",
        )

    # 生成文档ID和文件路径
    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}_{file.filename}"
    file_path = Path(settings.UPLOAD_DIR) / safe_filename

    try:
        # 保存上传文件
        file_size = await save_upload_file(file, file_path)
        logger.info(
            "文件已保存: path=%s, size=%d bytes", file_path, file_size
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("保存上传文件失败: %s", e)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}") from e

    # 后台索引任务
    async def _background_index():
        """后台索引任务：解析→分块→嵌入→存储"""
        try:
            logger.info("后台索引开始: document_id=%s", document_id)
            chunks_count = processor.index_document(str(file_path), document_id)
            logger.info(
                "后台索引完成: document_id=%s, chunks=%d",
                document_id,
                chunks_count,
            )
        except Exception as e:
            logger.error(
                "后台索引失败: document_id=%s, error=%s\n%s",
                document_id,
                e,
                traceback.format_exc(),
            )

    background_tasks.add_task(_background_index)

    logger.info(
        "文档上传完成，返回 processing 状态: document_id=%s", document_id
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename or "unknown",
        chunks_count=0,
        status="processing",
        message=(
            f"文档 '{file.filename}' 已上传，正在后台处理中。"
            f"文档ID: {document_id}"
        ),
    )


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="语义检索文档",
    description="对已索引的文档执行语义相似度检索，返回最相关的文档片段",
)
async def search_documents(
    q: str = Query(
        ...,
        min_length=1,
        description="查询文本",
        example="什么是检索增强生成？",
    ),
    top_k: int = Query(
        default=5,
        ge=1,
        le=20,
        description="返回结果数量（1-20）",
    ),
):
    """语义检索已索引的文档片段。

    curl示例:
        curl "http://localhost:8000/search?q=什么是RAG&top_k=5"
    """
    logger.info("收到检索请求: q='%s', top_k=%d", q[:200], top_k)

    if processor is None:
        raise HTTPException(
            status_code=503,
            detail="服务尚未就绪，文档处理器未初始化",
        )

    # 计时开始
    start_time = time.perf_counter()

    try:
        results = processor.search(q, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.error("检索未预期错误: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}") from e

    # 计算耗时
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info("检索完成: 耗时=%.2fms, 结果数=%d", elapsed_ms, len(results))

    return SearchResponse(
        query=q,
        results=results,
        total=len(results),
        search_time_ms=round(elapsed_ms, 2),
    )


@app.post(
    "/chat/completions",
    response_model=ChatResponse,
    summary="RAG对话生成",
    description="基于RAG的对话生成：先检索相关文档，再用LLM基于检索结果生成回答",
)
async def chat_completions(request: ChatRequest):
    """基于RAG的对话生成端点。

    先根据 query 检索相关文档片段，然后将检索结果与对话历史一起
    发送给LLM生成增强回答。

    curl示例:
        curl -X POST http://localhost:8000/chat/completions \\
          -H "Content-Type: application/json" \\
          -d '{"query":"什么是RAG？","top_k":5}'
    """
    logger.info(
        "收到对话请求: query='%s', top_k=%d, history_len=%d",
        request.query[:200],
        request.top_k,
        len(request.conversation_history),
    )

    if processor is None:
        raise HTTPException(
            status_code=503,
            detail="服务尚未就绪，文档处理器未初始化",
        )

    # 计时开始
    start_time = time.perf_counter()

    # 步骤1：检索相关上下文
    try:
        search_results = processor.search(request.query, top_k=request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.error("检索失败: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}") from e

    # 步骤2：提取上下文文本
    contexts = [r.content for r in search_results] if search_results else ["无相关文档"]

    # 步骤3：生成RAG回答
    try:
        answer = processor.generate_answer(
            query=request.query,
            contexts=contexts,
            conversation_history=request.conversation_history,
        )
    except RuntimeError as e:
        logger.error("回答生成失败: %s", e)
        # 即使生成失败也返回检索结果
        answer = f"抱歉，回答生成失败: {str(e)}"
    except Exception as e:
        logger.error("回答生成未预期错误: %s\n%s", e, traceback.format_exc())
        answer = f"抱歉，系统出现内部错误: {str(e)}"

    # 计算耗时
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "对话完成: 耗时=%.2fms, 检索结果数=%d, 回答长度=%d",
        elapsed_ms,
        len(search_results),
        len(answer),
    )

    return ChatResponse(
        answer=answer,
        sources=search_results,
        query=request.query,
        generation_time_ms=round(elapsed_ms, 2),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查服务整体状态，包括OpenAI和ChromaDB的连通性",
)
async def health_check():
    """健康检查端点。

    检查项的判断逻辑：
    - openai_available: 尝试调用一次轻量嵌入API
    - chroma_available: 检查集合是否能正常获取计数
    - status: 仅当所有关键依赖都通过时为 "healthy"

    curl示例:
        curl http://localhost:8000/health
    """
    logger.debug("收到健康检查请求")

    openai_available = False
    chroma_available = False
    documents_count = 0

    # 计算运行时长
    uptime_seconds = 0.0
    if startup_time is not None:
        uptime_seconds = (
            datetime.datetime.now() - startup_time
        ).total_seconds()

    # 检查ChromaDB
    if processor is not None:
        try:
            documents_count = processor.get_document_count()
            chroma_available = True
            logger.debug("ChromaDB健康检查通过，文档数=%d", documents_count)
        except Exception as e:
            logger.warning("ChromaDB健康检查失败: %s", e)

    # 检查OpenAI
    if processor is not None and processor.openai_client is not None:
        try:
            # 用一个短文本测试嵌入API
            test_embedding = processor.get_embedding("health check")
            if test_embedding and len(test_embedding) > 0:
                openai_available = True
                logger.debug("OpenAI健康检查通过")
        except Exception as e:
            logger.warning("OpenAI健康检查失败: %s", e)
    else:
        logger.warning("OpenAI客户端未初始化，无法进行健康检查")

    # 综合判断状态
    if processor is None:
        status = "degraded"
    elif openai_available and chroma_available:
        status = "healthy"
    elif openai_available or chroma_available:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version="1.0.0",
        documents_count=documents_count,
        uptime_seconds=round(uptime_seconds, 2),
        openai_available=openai_available,
        chroma_available=chroma_available,
    )


# =============================================================================
# 全局异常处理
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """处理HTTP异常，返回结构化的ErrorResponse。

    Args:
        request: FastAPI请求对象
        exc: HTTP异常实例

    Returns:
        JSONResponse: 包含结构化错误信息的JSON响应
    """
    logger.warning(
        "HTTP异常: status=%d, detail=%s, path=%s",
        exc.status_code,
        exc.detail,
        request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            detail=str(exc.detail),
            timestamp=datetime.datetime.now().isoformat(),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """处理未预期的通用异常。

    记录完整traceback但仅向客户端返回通用错误消息，
    避免泄露内部实现细节。

    Args:
        request: FastAPI请求对象
        exc: 异常实例

    Returns:
        JSONResponse: 500 Internal Server Error
    """
    logger.error(
        "未处理异常: type=%s, message=%s, path=%s\n%s",
        exc.__class__.__name__,
        str(exc),
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail="服务器内部错误，请查看服务日志获取详细信息",
            timestamp=datetime.datetime.now().isoformat(),
        ).model_dump(),
    )


# =============================================================================
# CLI 入口
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "07-end-to-end-rag-api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
