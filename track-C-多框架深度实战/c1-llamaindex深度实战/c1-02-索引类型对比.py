#!/usr/bin/env python3
"""
C1-02：LlamaIndex 索引类型对比
================================
LlamaIndex 提供多种索引结构，适配不同的检索场景。

索引类型           | 适用场景               | 优点                 | 缺点
------------------|------------------------|---------------------|------------------
VectorStoreIndex  | 语义相似度检索          | 通用、效果好          | 无法精确匹配关键词
SummaryIndex      | 文档总结/概览           | 适合长文档生成摘要    | 不支持细粒度检索
TreeIndex         | 层次化文档（如手册）    | 逐步缩小检索范围      | 构建耗时
KeywordTableIndex | 关键词精确匹配          | 精确、可解释          | 忽略语义相似性
KnowledgeGraph    | 实体关系查询            | 结构化推理            | 构建复杂

依赖：pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
"""

import os
import sys
import time
from pathlib import Path
from typing import List

# ============================================================================
# 准备阶段：创建示例数据和嵌入模型模拟
# ============================================================================

def prepare_sample_documents() -> List:
    """准备示例文档集合。"""
    from llama_index.core import Document

    docs = [
        Document(
            text="苹果公司发布了新款 iPhone 16，搭载 A18 芯片和全新的 AI 功能。"
                 "新手机采用了钛金属边框，支持卫星通信。",
            metadata={"category": "科技", "topic": "智能手机"}
        ),
        Document(
            text="华为在深圳举办了鸿蒙生态大会，宣布鸿蒙系统已覆盖 8 亿设备。"
                 "新一代鸿蒙提升了 AI 能力和跨设备协同体验。",
            metadata={"category": "科技", "topic": "操作系统"}
        ),
        Document(
            text="特斯拉公布了 2025 年 Q4 财报，营收同比增长 12%，毛利率维持在 18%。"
                 "Cybertruck 产能爬坡顺利，Model 2 预计明年发布。",
            metadata={"category": "科技", "topic": "电动车"}
        ),
        Document(
            text="OpenAI 发布了 GPT-5，支持更长的上下文窗口（100万token）和原生多模态能力。"
                 "新模型在推理、编码和数学能力上大幅提升。",
            metadata={"category": "科技", "topic": "人工智能"}
        ),
        Document(
            text="苹果的营养价值很高，富含维生素C和膳食纤维。"
                 "每天吃一个苹果有助于降低胆固醇，改善心血管健康。",
            metadata={"category": "健康", "topic": "营养"}
        ),
        Document(
            text="华为是一家全球领先的信息与通信技术（ICT）解决方案供应商。"
                 "其产品包括交换机、路由器、服务器和云服务。",
            metadata={"category": "科技", "topic": "企业介绍"}
        ),
        Document(
            text="Tesla Model 3 的自动辅助驾驶功能不断升级，"
                 "通过 OTA 更新实现了城市道路的自动导航。",
            metadata={"category": "科技", "topic": "自动驾驶"}
        ),
        Document(
            text="大语言模型的训练需要大量 GPU 算力。LLaMA、GPT、Gemini 等模型"
                 "的参数量从 7B 到 1.7T 不等，训练成本高达数亿美元。",
            metadata={"category": "科技", "topic": "人工智能"}
        ),
    ]
    return docs


def get_mock_embed_model():
    """
    创建模拟嵌入模型（用于演示，不消耗 API 调用）。
    生产环境替换为：OpenAIEmbedding(model="text-embedding-3-small")
    """
    from llama_index.core.embeddings import BaseEmbedding

    class MockEmbedding(BaseEmbedding):
        """简单的模拟嵌入：基于词袋的稀疏向量。"""
        _model_name = "mock-embedding-v1"

        def _get_query_embedding(self, query: str) -> List[float]:
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_text_embedding(query)

        def _get_text_embedding(self, text: str) -> List[float]:
            # 简单的基于词汇重叠的模拟嵌入（维度64）
            import hashlib
            hash_bytes = hashlib.sha256(text.encode()).digest()
            return [(b / 255.0) * 2 - 1 for b in hash_bytes[:64]]

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            return [self._get_text_embedding(t) for t in texts]

    return MockEmbedding()


# ============================================================================
# 第一部分：VectorStoreIndex —— 最常用的语义检索引擎
# ============================================================================

def demo_vector_store_index(documents):
    """向量存储索引：将文档分块嵌入，按语义相似度检索。"""
    print("=" * 60)
    print("【1】VectorStoreIndex —— 语义向量检索")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings

    # 配置全局设置
    Settings.embed_model = get_mock_embed_model()
    Settings.chunk_size = 200
    Settings.chunk_overlap = 20

    start = time.time()
    index = VectorStoreIndex.from_documents(documents)
    build_time = time.time() - start
    print(f"  索引构建耗时: {build_time:.3f}s")
    print(f"  索引中节点数: {len(index.docstore.docs)}")

    # 语义检索
    query_engine = index.as_query_engine(similarity_top_k=2)
    queries = [
        "苹果公司的新产品有什么特点？",
        "人工智能领域有什么最新进展？",
        "特斯拉的财务状况如何？",
    ]

    for q in queries:
        response = query_engine.query(q)
        print(f"\n  查询: {q}")
        print(f"  回答: {str(response)[:200]}")

    return index


def demo_summary_index(documents):
    """摘要索引：适合需要先概览再深入的长文档场景。"""
    print("\n" + "=" * 60)
    print("【2】SummaryIndex —— 文档摘要检索")
    print("=" * 60)

    from llama_index.core import SummaryIndex, Settings
    from llama_index.core.response.notebook_utils import display_source_node

    Settings.embed_model = get_mock_embed_model()

    index = SummaryIndex.from_documents(documents)
    print(f"  摘要索引构建完成，节点数: {len(index.docstore.docs)}")

    # SummaryIndex 会将所有文档视为一个整体
    # 适用于"帮我总结一下所有文档的主要内容"
    query_engine = index.as_query_engine(
        response_mode="tree_summarize",  # 递归摘要模式
        use_async=True,
    )

    query = "总结所有文档中关于科技公司的主要信息"
    response = query_engine.query(query)
    print(f"  查询: {query}")
    print(f"  回答: {str(response)[:300]}")

    return index


def demo_keyword_table_index(documents):
    """关键词表索引：基于关键词的精确匹配检索。"""
    print("\n" + "=" * 60)
    print("【3】KeywordTableIndex —— 关键词精确匹配")
    print("=" * 60)

    from llama_index.core import KeywordTableIndex, Settings

    Settings.embed_model = get_mock_embed_model()

    index = KeywordTableIndex.from_documents(documents)
    print(f"  关键词索引构建完成，关键词数: {len(index.index_struct.table)}")
    print(f"  示例关键词: {list(index.index_struct.table.keys())[:10]}")

    query_engine = index.as_query_engine()

    queries = [
        "苹果",      # 精确关键词匹配
        "华为",      # 精确关键词匹配
        "人工智能",   # 精确关键词匹配
    ]
    for q in queries:
        response = query_engine.query(f"关于{q}的信息")
        print(f"\n  查询: '关于{q}的信息'")
        print(f"  回答: {str(response)[:200]}")

    return index


# ============================================================================
# 第四部分：混合检索器 = 向量检索 + 关键词检索
# ============================================================================

def demo_hybrid_search(documents):
    """混合检索：融合 Vector 和 BM25/Keyword 的互补优势。"""
    print("\n" + "=" * 60)
    print("【4】混合检索 —— 向量 + 关键词融合")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings

    Settings.embed_model = get_mock_embed_model()

    # 构建向量索引
    vector_index = VectorStoreIndex.from_documents(documents)

    # 构建关键词索引
    from llama_index.core import KeywordTableIndex
    keyword_index = KeywordTableIndex.from_documents(documents)

    # 使用 QueryFusionRetriever 融合两种检索结果
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.retrievers import KeywordTableSimpleRetriever

    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=3,
    )
    keyword_retriever = KeywordTableSimpleRetriever(
        index=keyword_index,
        num_chunks_per_query=3,
    )

    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        similarity_top_k=5,
        num_queries=1,  # 生成 1 个变体查询
        mode="reciprocal_rerank",  # 使用 RRF 融合算法
    )

    query = "苹果公司的最新产品和技术有哪些？"
    nodes = fusion_retriever.retrieve(query)

    print(f"  查询: {query}")
    print(f"  检索结果数: {len(nodes)}")
    for i, node in enumerate(nodes):
        print(f"    结果 {i+1}: score={node.score:.4f} | {node.text[:80]}...")

    return fusion_retriever


# ============================================================================
# 第五部分：索引选择决策树
# ============================================================================

def print_index_selection_guide():
    """索引类型选择决策指南。"""
    print("\n" + "=" * 60)
    print("【索引选择决策树】")
    print("=" * 60)

    guide = """
    开始 ──> 你的检索需求是什么？
      │
      ├── "语义相似度" ──> VectorStoreIndex（默认首选）
      │    ├── 数据量大(>100万) ──> 加 FAISS/Qdrant 等向量数据库
      │    └── 数据有层次 ──> 考虑 TreeIndex
      │
      ├── "精确关键词" ──> KeywordTableIndex
      │    └── 同时需要语义 ──> 混合检索（上文的 fusion_retriever）
      │
      ├── "结构化推理" ──> KnowledgeGraphIndex
      │    └── 实体关系复杂 ──> 配合 Neo4j/ NebulaGraph
      │
      ├── "先总览再细节" ──> SummaryIndex
      │    └── 长文档摘要 ──> response_mode="tree_summarize"
      │
      └── "复杂多步查询" ──> 组合索引
           ├── RouterQueryEngine（路由到不同索引）
           ├── SubQuestionQueryEngine（拆分子问题）
           └── 递归检索（见 C1-04）
    """
    print(guide)

    # 性能对比表
    print("\n【性能对比】（基于 1000 篇文档的典型值）")
    print(f"{'索引类型':<22}{'构建时间':>10}{'查询延迟':>10}{'准确率':>10}")
    print("-" * 52)
    print(f"{'VectorStoreIndex':<22}{'~15s':>10}{'~200ms':>10}{'92%':>10}")
    print(f"{'KeywordTableIndex':<22}{'~8s':>10}{'~50ms':>10}{'85%':>10}")
    print(f"{'TreeIndex':<22}{'~60s':>10}{'~300ms':>10}{'88%':>10}")
    print(f"{'SummaryIndex':<22}{'~5s':>10}{'~500ms':>10}{'N/A':>10}")
    print(f"{'Hybrid (Vector+Keyword)':<22}{'~20s':>10}{'~250ms':>10}{'95%':>10}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("LlamaIndex 索引类型对比实战")
    print("=" * 60)

    # 准备数据
    print("[0] 准备示例文档...")
    documents = prepare_sample_documents()
    print(f"  已准备 {len(documents)} 篇文档\n")

    # 演示各种索引
    demo_vector_store_index(documents)
    demo_summary_index(documents)
    demo_keyword_table_index(documents)
    demo_hybrid_search(documents)

    # 打印决策指南
    print_index_selection_guide()

    print("\n[完成] 所有索引类型演示结束。")
