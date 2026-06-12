#!/usr/bin/env python3
"""
C1-01：LlamaIndex 数据连接器（Data Connectors）
=================================================
LlamaIndex 提供了 160+ 数据连接器，可从 PDF、数据库、API、Notion、
SharePoint、Slack 等 30+ 类数据源加载数据。

本文件演示：
1. SimpleDirectoryReader —— 从本地目录加载
2. PDF 加载器 —— 解析 PDF 文档
3. Web 加载器 —— 从网页抓取内容
4. 数据库加载器 —— 从 SQL 数据库读取
5. 自定义数据连接器 —— 连接任意内部 API

依赖：pip install llama-index llama-index-readers-file llama-index-readers-web beautifulsoup4 pymysql
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# ============================================================================
# 第一部分：SimpleDirectoryReader —— 最常用的万能加载器
# ============================================================================

def demo_simple_directory_reader():
    """从本地目录批量加载文档。支持 .pdf, .txt, .md, .csv, .docx 等格式。"""
    print("=" * 60)
    print("【1】SimpleDirectoryReader —— 目录批量加载")
    print("=" * 60)

    from llama_index.core import SimpleDirectoryReader

    # 假设你有一个 ./data/ 目录，里面放了各种文档
    data_dir = Path("./data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        # 创建示例文件
        (data_dir / "sample.txt").write_text(
            "人工智能正在重塑每一个行业。\nRAG 是当前最热门的企业AI应用模式。",
            encoding="utf-8"
        )
        (data_dir / "readme.md").write_text(
            "# RAG 项目\n\n检索增强生成（Retrieval-Augmented Generation）\n结合了信息检索与文本生成。",
            encoding="utf-8"
        )
        print(f"  [INFO] 已创建示例文件在 {data_dir.absolute()}")

    # 方式1：加载所有文件
    documents = SimpleDirectoryReader(
        input_dir="./data",
        recursive=True,           # 递归子目录
        required_exts=[".txt", ".md"],  # 只加载指定扩展名
        exclude_hidden=True,      # 排除隐藏文件
    ).load_data()

    print(f"  加载文档数: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"  文档 {i+1}: 文件名={doc.metadata.get('file_name', 'unknown')}, "
              f"长度={len(doc.text)} 字符")
        print(f"    内容预览: {doc.text[:80]}...")

    return documents


# ============================================================================
# 第二部分：PDF 加载器
# ============================================================================

def demo_pdf_loader():
    """使用专用 PDF Reader 解析复杂 PDF 文档。"""
    print("\n" + "=" * 60)
    print("【2】PDF 加载器 —— 结构化 PDF 解析")
    print("=" * 60)

    from llama_index.readers.file import PDFReader

    # 创建示例 PDF（使用 reportlab 如果可用）
    pdf_path = Path("./data/sample.pdf")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        c.drawString(100, 750, "企业AI应用白皮书")
        c.drawString(100, 700, "RAG 技术架构：检索增强生成结合了")
        c.drawString(100, 680, "大语言模型与外部知识库，实现精准回答。")
        c.save()
        print(f"  [INFO] 已生成示例 PDF: {pdf_path.absolute()}")
    except ImportError:
        print("  [WARN] reportlab 未安装，跳过 PDF 生成。请用真实 PDF 测试。")
        return []

    parser = PDFReader()
    documents = parser.load_data(file=pdf_path)

    print(f"  加载 PDF 文档数: {len(documents)}")
    for doc in documents:
        print(f"    内容: {doc.text[:120]}")
    return documents


# ============================================================================
# 第三部分：Web 加载器
# ============================================================================

def demo_web_loader():
    """从网页抓取内容并转换为 Document。"""
    print("\n" + "=" * 60)
    print("【3】Web 加载器 —— 网页内容抓取")
    print("=" * 60)

    # BeautifulSoupWebReader 可解析 HTML 并提取正文
    try:
        from llama_index.readers.web import BeautifulSoupWebReader

        reader = BeautifulSoupWebReader()
        # 示例：抓取 LlamaIndex 官方文档
        documents = reader.load_data(
            urls=["https://docs.llamaindex.ai/en/stable/"]
        )
        print(f"  网页文档数: {len(documents)}")
        if documents:
            print(f"    内容预览: {documents[0].text[:200]}...")
        return documents
    except ImportError:
        print("  [WARN] llama-index-readers-web 未安装。"
              "pip install llama-index-readers-web beautifulsoup4")
        return []
    except Exception as e:
        print(f"  [WARN] 网络请求失败（可能是离线环境）: {e}")
        return []


# ============================================================================
# 第四部分：数据库加载器
# ============================================================================

def demo_database_reader():
    """从 SQL 数据库读取结构化数据，每行转为 Document。"""
    print("\n" + "=" * 60)
    print("【4】数据库加载器 —— SQL 数据接入")
    print("=" * 60)

    try:
        from llama_index.readers.database import DatabaseReader
        import sqlite3

        # 创建示例 SQLite 数据库
        db_path = "./data/sample.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                description TEXT,
                category TEXT
            )
        """)
        # 清空旧数据
        cursor.execute("DELETE FROM products")
        cursor.executemany(
            "INSERT INTO products (name, description, category) VALUES (?, ?, ?)",
            [
                ("智能客服系统", "基于 RAG 的 AI 客服，支持多轮对话与知识库检索", "AI产品"),
                ("文档分析平台", "企业级文档智能分析，自动摘要与关键信息提取", "AI产品"),
                ("合规审查助手", "金融合规文档自动审查，监管规则实时更新", "金融科技"),
            ]
        )
        conn.commit()

        reader = DatabaseReader(
            scheme="sqlite",
            uri=db_path,
            query="SELECT name, description, category FROM products",
        )
        documents = reader.load_data()

        print(f"  数据库文档数: {len(documents)}")
        for doc in documents:
            print(f"    文档内容: {doc.text[:120]}...")

        conn.close()
        return documents
    except ImportError:
        print("  [WARN] llama-index-readers-database 未安装。"
              "pip install llama-index-readers-database")
        return []
    except Exception as e:
        print(f"  [ERROR] 数据库加载失败: {e}")
        return []


# ============================================================================
# 第五部分：自定义数据连接器
# ============================================================================

def demo_custom_connector():
    """展示如何编写自定义 Reader，连接任意内部 API 或数据源。"""
    print("\n" + "=" * 60)
    print("【5】自定义数据连接器 —— 连接内部 API")
    print("=" * 60)

    from llama_index.core.readers.base import BaseReader
    from llama_index.core import Document
    import json
    import hashlib
    from datetime import datetime

    class InternalKnowledgeBaseReader(BaseReader):
        """
        自定义 Reader：连接企业内部知识库 API。

        实现要点：
        1. 继承 BaseReader
        2. 实现 lazy_load_data（返回 Iterable[Document]）
        3. 在 metadata 中保留来源信息（用于溯源）
        4. 处理分页、认证、重试
        """

        def __init__(
            self,
            api_base_url: str = "https://internal-api.example.com/v1",
            api_key: Optional[str] = None,
            page_size: int = 100,
        ):
            self.api_base_url = api_base_url
            self.api_key = api_key or os.environ.get("INTERNAL_API_KEY", "")
            self.page_size = page_size

        def _call_api(self, endpoint: str, params: dict) -> List[dict]:
            """模拟 API 调用（生产环境替换为 requests）。"""
            # 实际代码中使用 requests.get()：
            # response = requests.get(
            #     f"{self.api_base_url}/{endpoint}",
            #     headers={"Authorization": f"Bearer {self.api_key}"},
            #     params=params,
            #     timeout=30,
            # )
            # response.raise_for_status()
            # return response.json()["data"]

            # 模拟返回数据（用于演示）
            return [
                {
                    "id": "KB-001",
                    "title": "产品使用手册 v2.3",
                    "content": "本文档描述了智能客服系统的安装、配置和使用方法...",
                    "category": "产品文档",
                    "updated_at": "2025-12-01",
                },
                {
                    "id": "KB-002",
                    "title": "故障排查指南",
                    "content": "常见问题及解决方案：1. 连接超时 → 检查网络配置...",
                    "category": "运维文档",
                    "updated_at": "2025-11-28",
                },
            ]

        def lazy_load_data(self) -> List[Document]:
            """
            延迟加载文档。对于大数据量场景，应使用生成器逐批返回。
            """
            documents = []
            page = 1

            while True:
                params = {"page": page, "page_size": self.page_size}
                items = self._call_api("knowledge-base/articles", params)

                if not items:
                    break  # 无更多数据

                for item in items:
                    # 构建文档 ID（确定性，用于去重）
                    doc_id = hashlib.md5(item["id"].encode()).hexdigest()[:16]

                    doc = Document(
                        text=item["content"],
                        doc_id=doc_id,
                        metadata={
                            "source_type": "internal_kb",
                            "article_id": item["id"],
                            "title": item["title"],
                            "category": item["category"],
                            "updated_at": item["updated_at"],
                            "fetch_timestamp": datetime.now().isoformat(),
                        },
                    )
                    documents.append(doc)

                page += 1

            return documents

    # ---- 使用自定义 Reader ----
    reader = InternalKnowledgeBaseReader(page_size=50)
    documents = reader.lazy_load_data()

    print(f"  自定义加载器返回文档数: {len(documents)}")
    for doc in documents:
        print(f"    文档: {doc.metadata.get('title')} "
              f"[{doc.metadata.get('category')}] "
              f"({len(doc.text)} 字符)")
    return documents


# ============================================================================
# 第六部分：数据连接器最佳实践总结
# ============================================================================

def print_best_practices():
    """打印数据连接器使用的最佳实践。"""
    print("\n" + "=" * 60)
    print("【最佳实践】LlamaIndex 数据连接器")
    print("=" * 60)

    tips = """
    1. 【延迟加载】对于大数据集（>10万条），使用 lazy_load 而非一次性加载
    2. 【增量更新】为每个文档设置 doc_id（基于内容哈希），方便增量更新
    3. 【元数据丰富】metadata 中保留：来源、时间、分类、权限级别
    4. 【错误重试】网络请求务必加上重试机制（tenacity 库）
    5. 【分页处理】API 分页要处理速率限制（rate limiting）
    6. 【字符编码】统一使用 UTF-8，避免中文乱码
    7. 【文档拆分】大文档在 Reader 层面不完全拆分，交给 IngestionPipeline
    8. 【安全隔离】多租户场景在 metadata 中标记 tenant_id
    """
    print(tips)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("LlamaIndex 数据连接器实战演示")
    print("=" * 60)
    print("本文件演示 6 种数据加载方式，按序号逐个执行。\n")

    demo_simple_directory_reader()
    demo_pdf_loader()
    demo_web_loader()
    demo_database_reader()
    demo_custom_connector()
    print_best_practices()

    print("\n[完成] 所有数据连接器演示结束。")
