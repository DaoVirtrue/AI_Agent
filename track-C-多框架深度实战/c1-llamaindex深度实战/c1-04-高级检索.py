#!/usr/bin/env python3
"""
C1-04：LlamaIndex 高级检索技术
================================
当基础检索不够用时的进阶方案：

1. 递归检索（Recursive Retrieval）—— 先检索文档摘要，再检索文档内部
2. 句子窗口检索（Sentence Window）—— 检索句子，返回其上下文窗口
3. 自动合并检索（Auto-Merging）—— 分层检索+自动合并相关块
4. 重排序（Re-Ranking）—— 粗筛+精排两阶段检索
5. 查询转换（Query Transformation）—— HyDE、Step-Back、Multi-Query

依赖：pip install llama-index llama-index-postprocessor-cohere
"""

import time
from typing import List, Dict, Any
from pathlib import Path

# ============================================================================
# 准备阶段
# ============================================================================

def get_mock_embed_model():
    """模拟嵌入模型。"""
    from llama_index.core.embeddings import BaseEmbedding

    class MockEmbedding(BaseEmbedding):
        _model_name = "mock-embed-v1"

        def _get_query_embedding(self, query: str) -> List[float]:
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_text_embedding(query)

        def _get_text_embedding(self, text: str) -> List[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [(b / 255.0) * 2 - 1 for b in h[:64]]

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            return [self._get_text_embedding(t) for t in texts]

    return MockEmbedding()


def get_mock_llm():
    """模拟 LLM。"""
    from llama_index.core.llms import BaseLLM
    from llama_index.core.llms import ChatResponse, ChatMessage

    class MockLLM(BaseLLM):
        _model_name = "mock-llm-v1"

        def chat(self, messages, **kwargs):
            user_msg = str(messages[-1]) if messages else "unknown"
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=f"[MockLLM] 对于查询 '{user_msg[:60]}...' 的综合回答。"
                )
            )

        def complete(self, prompt, **kwargs):
            from llama_index.core.llms import CompletionResponse
            return CompletionResponse(text=f"[MockLLM] {prompt[:100]}...")

        @property
        def metadata(self):
            return type("Metadata", (), {"model_name": self._model_name})()

    return MockLLM()


def prepare_long_documents():
    """准备长文档集合（适合演示高级检索）。"""
    from llama_index.core import Document

    # 模拟企业知识库中的多篇长文档
    doc1 = Document(
        text="""
第一部分：公司简介
某某科技有限公司成立于2018年，总部位于北京，是一家专注于企业AI应用的创新企业。
公司已获得C轮融资，累计融资额超过5亿美元。

第二部分：产品线
1. 智能客服平台：基于大语言模型的多轮对话系统，支持50+语言，日处理对话量超百万。
2. 文档分析引擎：自动识别文档类型，提取关键信息，生成结构化摘要。
3. 知识图谱平台：从非结构化文本中自动构建知识图谱，支持复杂关系查询。
4. 合规审查系统：面向金融行业，自动化监管合规检查，减少人工审查成本60%。

第三部分：技术架构
我们的技术栈包括：Python, LangChain, LlamaIndex, PostgreSQL, Redis, Elasticsearch。
采用微服务架构，Kubernetes 部署，日均API调用量 5000 万次。

第四部分：客户案例
- 某大型银行：部署智能客服后，人工坐席工作量减少40%，客户满意度提升15%。
- 某电商平台：商品描述自动生成，提升运营效率300%。
- 某医疗机构：医疗文献自动分析，辅助临床决策。

第五部分：未来规划
2026年计划推出 Agent 开发平台，支持零代码构建企业级 AI Agent。
        """,
        metadata={"title": "公司介绍手册", "category": "企业文档"}
    )

    doc2 = Document(
        text="""
第一章：RAG 系统架构概述
检索增强生成（RAG）是目前最主流的企业AI应用架构。其核心思想是：
在大语言模型生成回答前，先从外部知识库中检索相关信息作为上下文。

第二章：核心组件
2.1 文档加载器（Document Loader）
支持 PDF、HTML、Markdown、数据库、API 等多种数据源。

2.2 文档解析器（Document Parser）
将原始文档分割为适合检索的语义块。常见策略：
- 固定大小分割（chunk_size=512, chunk_overlap=50）
- 句子分割（按句号、换行分割）
- 语义分割（用嵌入模型检测语义边界）
- 层级分割（保留文档的章节结构）

2.3 嵌入模型（Embedding Model）
将文本块转换为向量。主流选择：
- OpenAI text-embedding-3-small: 1536维，$0.02/1M tokens
- Cohere embed-v3: 1024维，$0.10/1M tokens
- BGE-M3（开源）: 1024维，免费
- Jina embeddings-v3: 1024维，支持8K上下文

2.4 向量数据库（Vector Database）
存储和检索嵌入向量：
- Qdrant: 高性能，Rust实现，支持过滤
- Milvus: 分布式，十亿级向量
- Pinecone: 全托管，零运维
- ChromaDB: 轻量级，适合原型

2.5 重排序（Re-Ranker）
对初检结果进行精细排序。常用模型：
- Cohere Rerank v3: 商业API
- BGE-Reranker-v2: 开源
- ColBERT: 延迟交互模型

第三章：高级模式
3.1 父子文档检索（Parent-Child）
小粒度检索（句子/段落）→ 返回大粒度上下文（整个文档/章节）

3.2 递归检索
先检索文档摘要 → 确定相关文档 → 再检索文档内具体段落

3.3 查询增强
在检索前对用户查询进行改写、扩展或分解，提升检索召回率

3.4 自我反思检索
LLM 检查检索结果是否充分，不足时自动触发补充检索
        """,
        metadata={"title": "RAG 技术架构白皮书", "category": "技术文档"}
    )

    doc3 = Document(
        text="""
部署指南 v2.3
=============

阶段一：环境准备
1. 确保 Python >= 3.10
2. 安装 NVIDIA Driver >= 535（GPU 环境）
3. 安装 CUDA >= 12.1

阶段二：依赖安装
pip install llama-index llama-index-llms-openai
pip install llama-index-embeddings-openai
pip install llama-index-vector-stores-qdrant
pip install langchain langchain-openai

阶段三：配置
1. 设置环境变量 OPENAI_API_KEY
2. 配置向量数据库连接：
   QDRANT_URL=http://localhost:6333

阶段四：启动服务
1. 初始化知识库索引
2. 启动 API 服务
3. 配置负载均衡（Nginx/Traefik）
4. 配置监控（Prometheus + Grafana）

阶段五：调优
1. 调整 chunk_size 和 chunk_overlap
2. 选择最优嵌入模型
3. 配置缓存策略
4. 压测并调优并发参数

常见问题：
Q: 索引构建速度慢？
A: 增大 batch_size，使用异步嵌入。

Q: 检索结果不相关？
A: 尝试增大 similarity_top_k，添加重排序步骤。

Q: 回答有幻觉？
A: 降低 temperature，增强 prompt 中的引用要求。
        """,
        metadata={"title": "部署运维指南", "category": "运维文档"}
    )

    return [doc1, doc2, doc3]


# ============================================================================
# 第一部分：递归检索（Recursive Retrieval）
# ============================================================================

def demo_recursive_retrieval(documents):
    """
    递归检索：先粗后细的两阶段检索。

    阶段1：检索文档摘要 → 找到相关文档
    阶段2：在相关文档内部检索具体段落
    """
    print("=" * 60)
    print("【1】递归检索 —— 先粗后细两阶段")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings, Document
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import IndexNode

    Settings.embed_model = get_mock_embed_model()
    Settings.llm = get_mock_llm()

    # ---- 阶段1：构建文档级摘要索引 ----
    # 为每篇文档生成一个摘要节点
    summary_nodes = []
    sub_chunk_nodes = []

    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=30)

    for i, doc in enumerate(documents):
        # 摘要节点：文档的前 300 字作为代表
        summary_text = doc.text[:300]
        summary_node = IndexNode(
            text=summary_text,
            index_id=f"doc_{i}",
            metadata={"title": doc.metadata.get("title", f"文档{i}")},
        )
        summary_nodes.append(summary_node)

        # 子节点：文档被拆分后的所有小块
        chunks = splitter.get_nodes_from_documents([doc])
        for chunk in chunks:
            chunk.metadata["parent_doc_id"] = f"doc_{i}"
        sub_chunk_nodes.extend(chunks)

    # 构建摘要索引（顶层）
    summary_index = VectorStoreIndex(summary_nodes)
    summary_retriever = summary_index.as_retriever(similarity_top_k=2)

    # 构建细粒度索引（底层）
    detail_index = VectorStoreIndex(sub_chunk_nodes)

    # ---- 阶段2：递归检索 ----
    from llama_index.core.retrievers import RecursiveRetriever

    # 建立映射关系：每个摘要节点 → 对应的子检索器
    retriever_map = {}
    for i in range(len(documents)):
        doc_id = f"doc_{i}"
        # 创建一个只检索该文档子块的检索器
        doc_filter = lambda node, did=doc_id: node.metadata.get("parent_doc_id") == did

        child_retriever = detail_index.as_retriever(
            similarity_top_k=3,
            node_postprocessors=[],  # 这里可加过滤器
        )
        retriever_map[doc_id] = child_retriever

    # 创建递归检索器
    recursive_retriever = RecursiveRetriever(
        root_retriever=summary_retriever,
        retriever_dict=retriever_map,
        verbose=False,
    )

    queries = [
        "智能客服平台的功能",
        "如何调优检索结果",
        "核心组件的嵌入模型有哪些选择",
    ]

    for q in queries:
        nodes = recursive_retriever.retrieve(q)
        print(f"\n  查询: {q}")
        print(f"  检索结果数: {len(nodes)}")
        for j, node in enumerate(nodes[:2]):
            print(f"    结果 {j+1}: {node.text[:100]}...")


# ============================================================================
# 第二部分：句子窗口检索（Sentence Window）
# ============================================================================

def demo_sentence_window_retrieval(documents):
    """
    句子窗口检索：以小粒度（句子）检索，返回大粒度（段落）上下文。

    核心思想：
    - 检索时用句子作为索引粒度（更精确）
    - 返回时将句子所在的一段上下文一起返回（给 LLM 更多信息）
    """
    print("\n" + "=" * 60)
    print("【2】句子窗口检索 —— 小粒度检索 + 大粒度返回")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceWindowNodeParser
    from llama_index.core.postprocessor import MetadataReplacementPostProcessor

    Settings.embed_model = get_mock_embed_model()
    Settings.llm = get_mock_llm()

    # 关键：使用 SentenceWindowNodeParser
    # 它以句子为单位切分，同时在 metadata 中保留窗口上下文
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,               # 窗口大小（句子数）
        window_metadata_key="window",  # 上下文的 metadata key
        original_text_metadata_key="original_sentence",  # 原始句子 key
    )

    # 将文档转换为句子窗口节点
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"  句子节点数: {len(nodes)}")
    print(f"  示例节点 metadata keys: {list(nodes[0].metadata.keys())}")

    # 构建索引
    index = VectorStoreIndex(nodes)

    # 检索时，用 MetadataReplacementPostProcessor 将检索到的句子
    # 替换为其窗口上下文（更大的文本块）
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        node_postprocessors=[
            MetadataReplacementPostProcessor(
                target_metadata_key="window"  # 用窗口上下文替换原始文本
            ),
        ],
    )

    query = "RAG 系统的核心组件有哪些？"
    response = query_engine.query(query)
    print(f"\n  查询: {query}")
    print(f"  回答（窗口上下文增强后）: {str(response)[:300]}")

    # 对比：不使用窗口增强
    base_engine = index.as_query_engine(similarity_top_k=3)
    base_response = base_engine.query(query)
    print(f"\n  对比（无窗口增强）: {str(base_response)[:300]}")


# ============================================================================
# 第三部分：自动合并检索（Auto-Merging）
# ============================================================================

def demo_auto_merging_retrieval(documents):
    """
    自动合并检索：层级结构 + 自动合并。

    当检索到的多个小块属于同一个父节点时，自动将它们合并为父节点，
    给 LLM 提供更完整的上下文。
    """
    print("\n" + "=" * 60)
    print("【3】自动合并检索 —— 层级合并上下文")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.node_parser import HierarchicalNodeParser
    from llama_index.core.node_parser import get_leaf_nodes
    from llama_index.core.retrievers import AutoMergingRetriever
    from llama_index.core.storage import StorageContext
    from llama_index.core.storage.docstore import SimpleDocumentStore

    Settings.embed_model = get_mock_embed_model()
    Settings.llm = get_mock_llm()

    # 层级解析器：生成 3 层节点结构
    # Level 0: 大块（parent, chunk_size=512）
    # Level 1: 中块（child, chunk_size=128）
    # Level 2: 小块（leaf, chunk_size=64）
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[512, 128, 64],
        chunk_overlap=20,
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"  层级节点总数: {len(nodes)}")

    # 获取叶子节点（最小粒度，用于检索）
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"  叶子节点数: {len(leaf_nodes)}")

    # 构建 docstore 并添加所有层级的节点
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    storage_context = StorageContext.from_defaults(docstore=docstore)

    # 用叶子节点构建索引
    base_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

    # 自动合并检索器
    base_retriever = base_index.as_retriever(similarity_top_k=6)
    retriever = AutoMergingRetriever(
        vector_retriever=base_retriever,
        storage_context=storage_context,
        verbose=False,
    )

    query = "RAG 系统的嵌入模型选择"
    nodes = retriever.retrieve(query)
    print(f"\n  查询: {query}")
    print(f"  返回节点数（可能已合并）: {len(nodes)}")
    for i, node in enumerate(nodes):
        print(f"    结果 {i+1}: 长度={len(node.text)} 字符 | {node.text[:100]}...")


# ============================================================================
# 第四部分：重排序（Re-Ranking）
# ============================================================================

def demo_reranking(documents):
    """
    重排序：两阶段检索（粗筛+精排）。

    阶段1：向量检索召回 top_k=20 候选
    阶段2：用 Reranker 模型精细排序，取前 5 个
    """
    print("\n" + "=" * 60)
    print("【4】重排序 —— 粗筛 + 精排两阶段")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.postprocessor import SentenceTransformerRerank

    Settings.embed_model = get_mock_embed_model()
    Settings.llm = get_mock_llm()

    index = VectorStoreIndex.from_documents(documents)

    retriever = index.as_retriever(similarity_top_k=8)  # 粗筛 8 个

    # 精排配置
    print("  尝试加载 BGE Reranker（如失败则用基础排序）...")
    try:
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=3,  # 最终返回前 3 个
        )
        print("  BGE Reranker 加载成功")
    except Exception:
        print("  [WARN] BGE Reranker 不可用，使用相似度降序排序")
        reranker = SimilarityPostprocessor(
            similarity_cutoff=0.3,
        )

    from llama_index.core.query_engine import RetrieverQueryEngine

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker],
    )

    query = "如何优化 RAG 系统的检索性能？"
    response = query_engine.query(query)
    print(f"\n  查询: {query}")
    print(f"  回答: {str(response)[:300]}")
    print(f"  精排后来源节点数: {len(response.source_nodes)}")


# ============================================================================
# 第五部分：查询转换（Query Transformation）
# ============================================================================

def demo_query_transformation(documents):
    """
    查询转换：在检索前对查询进行优化处理。

    HyDE：用 LLM 生成一个假设性文档，用该文档的嵌入去检索
    Step-Back：先生成一个更抽象的"后退问题"
    Multi-Query：从不同角度生成多个变体查询
    """
    print("\n" + "=" * 60)
    print("【5】查询转换 —— 优化查询本身")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings

    Settings.embed_model = get_mock_embed_model()
    Settings.llm = get_mock_llm()

    index = VectorStoreIndex.from_documents(documents)

    # ---- 5.1 HyDE 变换 ----
    print("\n  >>> 5.1 HyDE（假设性文档嵌入）")
    from llama_index.core.indices.query.query_transform import HyDEQueryTransform

    hyde_transform = HyDEQueryTransform(include_original=True)
    # HyDE 会：原始查询 → LLM 生成假设文档 → 用假设文档的嵌入检索

    query = "如何部署 RAG 系统？"
    try:
        transformed_bundle = hyde_transform.run(query)
        print(f"  原始查询: {query}")
        print(f"  HyDE 生成的假设文档: {str(transformed_bundle)[:200]}...")
    except Exception as e:
        print(f"  HyDE 变换失败（MockLLM可能不支持）: {e}")

    # ---- 5.2 Step-Back Prompting ----
    print("\n  >>> 5.2 Step-Back 退后提问")
    from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform

    try:
        step_back_transform = StepDecomposeQueryTransform(
            llm=get_mock_llm(),
            verbose=False,
        )
        result = step_back_transform.run(query)
        print(f"  原始查询: {query}")
        print(f"  Step-Back 结果: {str(result)[:200]}...")
    except Exception as e:
        print(f"  Step-Back 变换失败: {e}")

    # ---- 5.3 Multi-Query 多角度查询 ----
    print("\n  >>> 5.3 Multi-Query 多角度查询")
    print("  原理：将一个查询改写为 3-5 个不同角度的问题")
    print("  分别检索 → 合并去重 → 提升召回率")
    print("  示例：'RAG系统部署' →")
    print("     Q1: 'RAG系统的环境配置步骤是什么？'")
    print("     Q2: 'RAG系统部署需要注意哪些问题？'")
    print("     Q3: '如何优化RAG系统的部署性能？'")


# ============================================================================
# 第六部分：高级检索策略对比
# ============================================================================

def print_advanced_retrieval_comparison():
    """高级检索策略对比总表。"""
    print("\n" + "=" * 60)
    print("【高级检索策略对比总表】")
    print("=" * 60)

    table = """
┌─────────────────────┬────────────┬──────────────┬────────────────────────────┐
│ 策略                │ 适用场景   │ 延迟增加     │ 召回提升                  │
├─────────────────────┼────────────┼──────────────┼────────────────────────────┤
│ 递归检索            │ 多文档知识库│ +50%        │ +15-25%（大型文档集）     │
│ 句子窗口            │ 精准定位    │ +10%        │ +10-15%（细粒度答案）      │
│ 自动合并            │ 层级文档    │ +20%        │ +10-20%（结构化文档）      │
│ 重排序              │ 高精度需求  │ +30-100ms   │ +5-15%（几乎所有场景）    │
│ HyDE                │ 查询与文档  │ +1次LLM调用 │ +10-20%（语言风格差异大时）│
│                     │ 风格不同    │             │                           │
│ 多角度查询          │ 复杂问题    │ +3-5次检索  │ +15-30%（多面问题）        │
│ 自反思检索          │ 需要确保    │ +1-2次LLM   │ 大幅减少遗漏              │
│                     │ 完整性      │ 判断        │                           │
└─────────────────────┴────────────┴──────────────┴────────────────────────────┘
    """
    print(table)

    print("\n【组合建议】")
    print("  通用高性能方案：Multi-Query + 混合检索 + 重排序")
    print("  长文档 QA 方案：递归检索 + 句子窗口")
    print("  精准知识库方案：HyDE + 自动合并 + 重排序")
    print("  成本敏感方案：仅重排序（成本最低，效果肉眼可见）")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("LlamaIndex 高级检索技术实战")
    print("=" * 60 + "\n")

    docs = prepare_long_documents()
    print(f"[0] 已准备 {len(docs)} 篇长文档\n")

    demo_recursive_retrieval(docs)
    demo_sentence_window_retrieval(docs)
    demo_auto_merging_retrieval(docs)
    demo_reranking(docs)
    demo_query_transformation(docs)
    print_advanced_retrieval_comparison()

    print("\n[完成] 所有高级检索演示结束。")
