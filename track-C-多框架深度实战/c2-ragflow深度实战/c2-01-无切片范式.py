#!/usr/bin/env python3
"""
C2-01：RAGFlow 无切片范式（Chunk-Free RAG）
===============================================
RAGFlow 的核心理念是"不要盲目切分文档"。传统 RAG 按照固定
字符数切分文档（chunking），而 RAGFlow 通过深度文档结构理解，
做到"理解文档后再索引"。

本文件演示：
1. RAGFlow 的文档解析流程（DeepDoc）
2. 无切片索引的工作原理
3. 与 LangChain/LlamaIndex 的 chunking 方式对比
4. 何时使用无切片范式

依赖：pip install ragflow-client
注意：需要先启动 RAGFlow 服务端（Docker 方式）
"""

import os
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field


# ============================================================================
# 第一部分：RAGFlow 客户端封装
# ============================================================================

@dataclass
class RAGFlowConfig:
    """RAGFlow 连接配置。"""
    base_url: str = "http://localhost:9380"
    api_key: str = ""
    dataset_id: str = ""


class RAGFlowClient:
    """
    RAGFlow API 客户端封装。

    封装了 RAGFlow 的核心操作：
    - 数据集管理（创建/删除/列表）
    - 文档上传与解析
    - 文档检索
    - 对话/问答
    """

    def __init__(self, config: RAGFlowConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        # 实际使用 ragflow_client SDK：
        # from ragflow import RAGFlow
        # self.client = RAGFlow(api_key=api_key, base_url=base_url)
        print(f"[INFO] RAGFlow 客户端已初始化，目标服务: {self.base_url}")

    def _api_call(self, method: str, endpoint: str, data: dict = None) -> dict:
        """
        模拟 API 调用（生产环境用 requests 库）。

        实际代码：
        response = requests.request(
            method=method,
            url=f"{self.base_url}/api/v1/{endpoint}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=data,
            timeout=30,
        )
        return response.json()
        """
        # 演示用模拟返回
        return {"code": 0, "message": "success", "data": {}}

    def create_dataset(self, name: str, description: str = "",
                       chunk_method: str = "knowledge_graph") -> str:
        """
        创建数据集。

        RAGFlow 支持的 chunk_method:
        - "naive": 通用分块（类似 LangChain 的 RecursiveCharacterTextSplitter）
        - "manual": 手动分块
        - "qa": Q&A 分块（专门为 FAQ/问答对优化）
        - "table": 表格分块（识别和保留表格结构）
        - "paper": 论文分块（保留摘要、章节、引用）
        - "book": 书籍分块（保留章节层次）
        - "laws": 法律分块（保留条款编号）
        - "presentation": PPT 分块
        - "picture": 图片分块
        - "one": 整篇文档不分块（无切片的核心）
        - "knowledge_graph": 知识图谱分块（实体关系提取）
        """
        print(f"  [API] 创建数据集: {name}")
        print(f"    分块策略: {chunk_method}")
        dataset_id = f"ds_{hash(name) % 100000:05d}"
        print(f"    数据集ID: {dataset_id}")
        return dataset_id

    def upload_document(self, dataset_id: str, file_path: str) -> str:
        """
        上传文档到数据集。

        RAGFlow 会调用 DeepDoc 引擎进行深度文档解析：
        1. 识别文档类型（PDF/Word/Excel/PPT/图片/网页）
        2. 提取文档结构（标题、段落、表格、图片、公式）
        3. 根据分块策略进行智能切分
        """
        print(f"  [API] 上传文档: {file_path} -> dataset={dataset_id}")
        print(f"    DeepDoc 解析中...")
        time.sleep(0.1)  # 模拟解析耗时
        doc_id = f"doc_{Path(file_path).stem}"
        print(f"    文档ID: {doc_id}")
        return doc_id

    def search(self, dataset_ids: List[str], query: str,
               top_k: int = 5) -> List[dict]:
        """
        在数据集中检索。

        返回的每个结果包含：
        - content: 文本内容
        - score: 相关度分数
        - document_name: 来源文档名
        - positions: 文档内的位置信息（页码、段落）
        """
        print(f"  [API] 检索: '{query}' (top_k={top_k})")
        return [
            {
                "content": f"关于'{query}'的相关内容...",
                "score": 0.92 - i * 0.05,
                "document_name": f"文档{i+1}.pdf",
                "positions": [f"第{i+1}页", f"第{i+2}段"],
            }
            for i in range(min(top_k, 3))
        ]

    def chat(self, dataset_ids: List[str], question: str,
             conversation_id: str = None) -> dict:
        """对话问答接口。"""
        print(f"  [API] 问答: '{question}'")
        return {
            "answer": f"根据知识库检索结果，关于'{question}'的回答...",
            "reference": [
                {"chunk_id": "abc123", "content": "引用内容...",
                 "document_name": "文档A.pdf"}
            ]
        }


# ============================================================================
# 第二部分：传统分块 vs 无切片对比
# ============================================================================

def compare_chunking_strategies():
    """
    对比传统分块与 RAGFlow 无切片范式的差异。

    这是本节的核心——理解"为什么不要盲目切分"。
    """
    print("=" * 60)
    print("【核心对比】传统分块 vs RAGFlow 无切片范式")
    print("=" * 60)

    # 模拟一篇结构化文档
    sample_doc = """
    第三章 系统架构设计

    3.1 总体架构
    本系统采用微服务架构，分为接入层、业务逻辑层、数据层三个层次。
    接入层负责请求路由与认证，业务逻辑层包含知识检索、对话管理、
    知识图谱构建等核心模块，数据层采用 MySQL + Elasticsearch + Neo4j
    的混合存储方案。

    3.2 关键技术选型
    表3-1 技术选型对比
    ┌──────────────┬───────────────┬────────────────┐
    │   组件       │    选型       │     说明       │
    ├──────────────┼───────────────┼────────────────┤
    │  大语言模型   │  GPT-4o       │  主生成模型    │
    │  嵌入模型     │  BGE-M3       │  中文语义检索  │
    │  向量数据库   │  Milvus       │  十亿级向量    │
    │  图数据库     │  Neo4j        │  知识图谱      │
    │  搜索引擎     │  Elasticsearch│  关键词检索    │
    └──────────────┴───────────────┴────────────────┘

    3.3 部署方案
    系统采用 Kubernetes 集群部署，生产环境配置：
    - GPU 节点: 4 × NVIDIA A100 (用于 LLM 推理)
    - CPU 节点: 8 × 32核64GB (用于微服务)
    - 存储: 分布式 Ceph 集群 100TB
    """

    print("\n  >>> 传统分块方式（chunk_size=200, overlap=50）")

    # 模拟传统分块
    chunk_size = 200
    overlap = 50
    chunks = []
    for i in range(0, len(sample_doc), chunk_size - overlap):
        chunk = sample_doc[i:i + chunk_size]
        chunks.append(chunk)

    print(f"    分块数: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"    块 {i+1}: [{len(chunk)}字] '{chunk[:80]}...'")

    # 展示传统分块的问题
    problems = [
        "❌ 块2切断了表格的连续性（表头和内容分离）",
        "❌ 块3丢失了章节层级信息（3.3 与 3.2 混在一起）",
        "❌ 检索"部署方案需要多少GPU"时，可能只匹配到块3的前半部分",
        "❌ 表格数据被切断后，LLM 无法正确理解技术选型",
        "❌ 固定的 chunk_size 对密集信息区域和稀疏区域不区分对待",
    ]
    print("\n    问题分析：")
    for p in problems:
        print(f"      {p}")

    print("\n  >>> RAGFlow 无切片方式（DeepDoc 文档理解）")

    advantages = [
        "✅ 识别文档类型为"技术文档"，自动采用'章节+表格'混合解析",
        "✅ 3.1 作为一个完整语义单元保留（不被切断）",
        "✅ 表格被完整提取为结构化数据（保留行列关系）",
        "✅ 章节层级被保留（3章 → 3.1 → 内容 → 3.2 → 表格 → 3.3）",
        "✅ 检索"部署的GPU配置"时精确命中 3.3 部署方案",
        "✅ 每个 chunk 都有明确的元数据标注（章节、页码、元素类型）",
    ]
    for a in advantages:
        print(f"      {a}")

    print("\n  【结论】无切片 ≠ 完全不分块，而是"理解后按语义分块"。")


# ============================================================================
# 第三部分：DeepDoc 文档解析演示
# ============================================================================

def demo_deepdoc_parsing():
    """
    模拟 DeepDoc 文档解析引擎的工作流程。

    DeepDoc 的解析步骤：
    1. 文档分类 → 识别文档类型
    2. 结构提取 → 提取标题层级、段落、表格、图片
    3. 内容理解 → 识别关键实体、关系
    4. 智能分块 → 根据文档类型和内容结构进行语义分块
    5. 索引构建 → 为每个块生成向量嵌入和关键词索引
    """
    print("\n" + "=" * 60)
    print("【DeepDoc 解析流程】模拟演示")
    print("=" * 60)

    print("""
    ┌─────────────────────────────────────────────────────┐
    │              DeepDoc 解析流水线                       │
    │                                                       │
    │  输入文档 (PDF/Word/Excel/PPT/图片)                    │
    │    │                                                  │
    │    ▼                                                  │
    │  [1. 文档分类器] ──→ 判断: PDF, 20页, 含表格和图片     │
    │    │                                                  │
    │    ▼                                                  │
    │  [2. 布局分析器] ──→ 检测: 标题×5, 段落×30,            │
    │    │                    表格×3, 图片×8, 页眉页脚        │
    │    ▼                                                  │
    │  [3. OCR引擎(可选)] ──→ 扫描件→可编辑文本               │
    │    │                                                  │
    │    ▼                                                  │
    │  [4. 结构解析器] ──→ 章节树:                            │
    │    │                   Ch1                                      │
    │    │                   ├── 1.1 (段落)                            │
    │    │                   ├── 1.2 (表格 + 段落)                     │
    │    │                   └── 1.3 (图片 + 段落)                     │
    │    │                  Ch2                                         │
    │    │                   ├── 2.1 (段落)                             │
    │    │                   └── 2.2 (表格)                             │
    │    ▼                                                  │
    │  [5. 智能分块器] ──→ 输出语义块:                        │
    │    │                  块1: {类型: 段落, 章节: 1.1, ...}  │
    │    │                  块2: {类型: 表格, 章节: 1.2, ...}  │
    │    │                  块3: {类型: 图片, 章节: 1.3, ...}  │
    │    ▼                                                  │
    │  [6. 嵌入索引器] ──→ 生成向量 + 关键词 + 全文本索引      │
    └─────────────────────────────────────────────────────┘
    """)

    # 模拟 DeepDoc 输出的语义块
    class SemanticChunk:
        def __init__(self, chunk_id: str, content: str, chunk_type: str,
                     section: str, page: int, metadata: dict):
            self.chunk_id = chunk_id
            self.content = content
            self.chunk_type = chunk_type   # paragraph / table / image / formula
            self.section = section          # 所在章节
            self.page = page                # 页码
            self.metadata = metadata

    # 展示无切片后的输出
    semantic_chunks = [
        SemanticChunk("c1", "3.1 总体架构\n本系统采用微服务架构...",
                      "paragraph", "3.1", 12, {"importance": "high"}),
        SemanticChunk("c2",
                      "| 组件 | 选型 | 说明 |\n| LLM | GPT-4o | 主生成模型 |\n"
                      "| 嵌入 | BGE-M3 | 中文语义 |\n| 向量库 | Milvus | 十亿级 |",
                      "table", "3.2", 13, {"table_rows": 5}),
        SemanticChunk("c3", "3.3 部署方案\nGPU节点: 4×A100...",
                      "paragraph", "3.3", 14, {"importance": "high"}),
    ]

    print("  DeepDoc 输出的语义块示例：")
    for chunk in semantic_chunks:
        print(f"\n    块ID: {chunk.chunk_id}")
        print(f"    类型: {chunk.chunk_type}")
        print(f"    章节: {chunk.section}, 页码: {chunk.page}")
        print(f"    内容: {chunk.content[:100]}...")
        print(f"    元数据: {chunk.metadata}")


# ============================================================================
# 第四部分：完整 RAGFlow 工作流
# ============================================================================

def demo_ragflow_workflow():
    """
    完整的 RAGFlow 使用流程：
    1. 启动服务 → 2. 创建数据集 → 3. 上传文档 → 4. 检索 → 5. 问答
    """
    print("\n" + "=" * 60)
    print("【完整工作流】RAGFlow 端到端演示")
    print("=" * 60)

    # 初始化客户端
    config = RAGFlowConfig(
        base_url=os.environ.get("RAGFLOW_URL", "http://localhost:9380"),
        api_key=os.environ.get("RAGFLOW_API_KEY", "demo-key"),
    )
    client = RAGFlowClient(config)

    # 步骤1：创建数据集（使用无切片策略）
    print("\n[步骤1] 创建数据集")
    dataset_id = client.create_dataset(
        name="企业技术文档库",
        description="存储公司内部技术文档、架构设计、部署指南",
        chunk_method="knowledge_graph",  # 知识图谱分块，比 naive 更智能
    )

    # 步骤2：上传文档
    print("\n[步骤2] 上传文档")
    documents = [
        "./data/系统架构设计.pdf",
        "./data/API接口文档.docx",
        "./data/部署运维手册.xlsx",
    ]
    doc_ids = []
    for doc_path in documents:
        # 创建占位文件
        Path(doc_path).parent.mkdir(exist_ok=True)
        if not Path(doc_path).exists():
            Path(doc_path).write_text("示例文档内容", encoding="utf-8")
        doc_id = client.upload_document(dataset_id, doc_path)
        doc_ids.append(doc_id)

    # 步骤3：等待解析完成
    print("\n[步骤3] 等待 DeepDoc 解析...")
    print("  RAGFlow 在后台解析文档结构、构建索引")
    print("  可通过 API GET /api/v1/datasets/{id}/documents 查看解析状态")
    time.sleep(0.5)

    # 步骤4：检索
    print("\n[步骤4] 知识库检索")
    queries = [
        "系统的整体架构是什么？",
        "用到了哪些技术组件？",
        "部署需要多少 GPU？",
    ]
    for q in queries:
        results = client.search([dataset_id], q, top_k=3)
        print(f"\n  查询: {q}")
        for r in results:
            print(f"    [{r['score']:.2f}] {r['content'][:80]}..."
                  f" (来源: {r['document_name']})")

    # 步骤5：问答
    print("\n[步骤5] 对话问答")
    response = client.chat([dataset_id], "总结系统的技术架构")
    print(f"  回答: {response['answer'][:200]}")
    print(f"  引用数: {len(response.get('reference', []))}")

    return dataset_id


# ============================================================================
# 第五部分：何时使用无切片范式
# ============================================================================

def when_to_use_chunk_free():
    """决策指南：什么场景适合无切片范式。"""
    print("\n" + "=" * 60)
    print("【决策指南】何时使用无切片范式")
    print("=" * 60)

    print("""
    ┌─────────────────────────────────────────────────────────┐
    │  适合无切片的场景            不适合无切片的场景          │
    ├─────────────────────────────────────────────────────────┤
    │  ✅ 结构化文档（技术手册、论文） ❌ Twitter/微博短文本    │
    │  ✅ 含表格的文档               ❌ 纯对话记录              │
    │  ✅ 法律/合规文件（需精确引用） ❌ 高度非结构化的数据      │
    │  ✅ 多层级目录结构文档         ❌ 实时流式数据             │
    │  ✅ 需要保留文档位置信息的     ❌ 对延迟极度敏感的场景     │
    │  ✅ 扫描件 PDF（含 OCR 需求）  ❌ 纯数值型数据             │
    │  ✅ 多语言混合文档             ❌ 单个超长 JSON/日志文件   │
    └─────────────────────────────────────────────────────────┘
    """)

    print("\n  【核心判断标准】")
    print("    1. 文档是否具有内在结构？（标题/章节/表格/图片）")
    print("       YES → 无切片范式效果更好")
    print("    2. 回答是否需要追溯到文档的精确位置？")
    print("       YES → 无切片范式（保留页码/段落信息）")
    print("    3. 文档内容是否有逻辑上的"语义边界"？")
    print("       YES → 让 DeepDoc 自动识别，不要手动分块")
    print("    4. 是否涉及跨块信息的推理？")
    print("       YES → 无切片（大块保留更多上下文信息）")


# ============================================================================
# 第六部分：性能注意事项
# ============================================================================

def print_performance_notes():
    """无切片范式的性能特点。"""
    print("\n" + "=" * 60)
    print("【性能注意事项】")
    print("=" * 60)

    print("""
    1. 【解析耗时】DeepDoc 的深度解析比简单分块慢 3-5 倍
       解决方案：异步解析、批量处理、缓存解析结果

    2. 【存储开销】语义块大小不统一，可能差异百倍
       解决方案：设置 max_chunk_size 上限，超限的做二次细分

    3. 【检索延迟】块大小不均可能导致检索延迟抖动
       解决方案：混合检索（小粒度语义 + 大粒度关键词）

    4. 【LLM 上下文】大块内容可能超出 LLM 上下文窗口
       解决方案：配合重排序（reranker）精细化选择

    5. 【增量更新】文档修改后，需重新解析（而不是简单 re-chunk）
       解决方案：文档版本管理 + 增量索引更新

    6. 【成本】解析阶段的 LLM 调用（分类、结构识别）增加成本
       解决方案：缓存文档结构分析结果，仅对新增/修改文档调用
    """)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("RAGFlow 无切片范式实战演示")
    print("=" * 60)
    print("核心思想：理解文档，而非切碎文档\n")

    compare_chunking_strategies()
    demo_deepdoc_parsing()
    demo_ragflow_workflow()
    when_to_use_chunk_free()
    print_performance_notes()

    print("\n[完成] 无切片范式演示结束。")
    print("  [提示] 要运行真实的 RAGFlow，请执行：")
    print("    docker run -p 9380:80 ragflow/ragflow")
