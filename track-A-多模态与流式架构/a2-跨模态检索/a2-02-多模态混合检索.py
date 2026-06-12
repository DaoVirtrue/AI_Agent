#!/usr/bin/env python3
"""
A2-02: 多模态混合检索 (Multimodal Hybrid Retrieval)
======================================================
学习目标:
  1. 文本向量 + 图片向量 + 表格向量 → RRF三路融合
  2. 动态权重分配
  3. MultimodalHybridRetriever类设计与实现
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# Section 1: 多模态数据结构
# ============================================================

class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    AUDIO = "audio"


@dataclass
class MultiModalDocument:
    """多模态文档：同一文档的不同模态表示"""
    doc_id: str
    text_content: str = ""            # 文本内容
    text_embedding: Optional[np.ndarray] = None  # 文本向量

    image_paths: List[str] = field(default_factory=list)
    image_embeddings: Optional[np.ndarray] = None  # 图片向量(N_images, dim)

    table_data: List[Dict] = field(default_factory=list)
    # 表格序列化为文本后嵌入
    table_embeddings: Optional[np.ndarray] = None

    metadata: Dict = field(default_factory=dict)


# ============================================================
# Section 2: RRF (Reciprocal Rank Fusion) 实现
# ============================================================

class ReciprocalRankFusion:
    """
    RRF (Reciprocal Rank Fusion) 多模态结果融合
    原理: 对不同模态的结果列表按排名进行倒数加权融合

    RRF_score(d) = sum_over_modalities( 1 / (k + rank_i(d)) )

    其中 k 是平滑常数（默认60），防止单模态排名过高主导结果
    """

    def __init__(self, k: int = 60):
        """
        参数:
            k: 平滑常数。经典值60，来自TREC
                较小k:  排名靠前的结果权重更大
                较大k:  更均衡的融合
        """
        self.k = k

    def fuse(self, ranked_lists: Dict[str, List[Tuple[str, float]]],
             weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        融合多个模态的排序列表

        参数:
            ranked_lists: {
                "text":  [(doc_id, score), ...],
                "image": [(doc_id, score), ...],
                "table": [(doc_id, score), ...],
            }
            weights: {"text": 1.0, "image": 0.8, "table": 0.5}

        返回:
            [{doc_id, rrf_score, per_modality_ranks}, ...]
        """
        if weights is None:
            weights = {mod: 1.0 for mod in ranked_lists}

        # 第一步: 收集所有文档
        all_docs = set()
        for docs in ranked_lists.values():
            for doc_id, _ in docs:
                all_docs.add(doc_id)

        # 第二步: 计算每个文档在每个模态中的排名
        rank_maps = {}
        for modality, docs in ranked_lists.items():
            rank_maps[modality] = {}
            for rank, (doc_id, _) in enumerate(docs, start=1):
                rank_maps[modality][doc_id] = rank

        # 第三步: 计算RRF分数
        scores = {}
        for doc_id in all_docs:
            rrf = 0.0
            per_mod = {}
            for modality, rank_map in rank_maps.items():
                rank = rank_map.get(doc_id, len(rank_map) + 1)
                per_mod[modality] = rank
                w = weights.get(modality, 1.0)
                rrf += w * (1.0 / (self.k + rank))
            scores[doc_id] = {
                "rrf_score": rrf,
                "per_modality_ranks": per_mod,
            }

        # 第四步: 排序
        sorted_results = sorted(
            scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )

        return [
            {
                "doc_id": doc_id,
                "rrf_score": round(info["rrf_score"], 6),
                "per_modality_ranks": info["per_modality_ranks"],
            }
            for doc_id, info in sorted_results
        ]


# ============================================================
# Section 3: 向量索引（简化版FAISS风格）
# ============================================================

class SimpleVectorIndex:
    """简易向量索引（无需FAISS依赖，演示原理）"""

    def __init__(self):
        self.vectors: Dict[str, np.ndarray] = {}  # doc_id -> vector
        self._array: Optional[np.ndarray] = None
        self._ids: List[str] = []

    def add(self, doc_id: str, vector: np.ndarray):
        """添加单个向量"""
        self.vectors[doc_id] = vector

    def build(self):
        """构建索引矩阵"""
        self._ids = list(self.vectors.keys())
        self._array = np.array([self.vectors[i] for i in self._ids])
        # 归一化
        norms = np.linalg.norm(self._array, axis=1, keepdims=True)
        self._array = self._array / (norms + 1e-8)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """余弦相似度检索"""
        if self._array is None:
            self.build()

        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        scores = np.dot(self._array, query_norm)
        indices = np.argsort(scores)[::-1][:top_k]

        return [(self._ids[i], float(scores[i])) for i in indices]

    def __len__(self):
        return len(self.vectors)


# ============================================================
# Section 4: MultimodalHybridRetriever 完整实现
# ============================================================

class MultimodalHybridRetriever:
    """
    多模态混合检索器

    核心设计:
    - 3个独立向量索引: 文本索引、图片索引、表格索引
    - RRF融合: 将3路检索结果合并排序
    - 动态权重: 根据查询意图自动调整模态权重
    """

    def __init__(self, embedding_model_name: str = None):
        """
        参数:
            embedding_model_name: 文本嵌入模型名
                默认使用轻量sentence-transformers模型
        """
        # 三个模态索引
        self.text_index = SimpleVectorIndex()
        self.image_index = SimpleVectorIndex()
        self.table_index = SimpleVectorIndex()

        # 文档存储
        self.documents: Dict[str, MultiModalDocument] = {}

        # 嵌入模型
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                embedding_model_name or "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._has_embedding_model = True
        except ImportError:
            print("  [警告] sentence_transformers未安装，使用随机嵌入(仅供演示架构)")
            self.embedding_model = None
            self._has_embedding_model = False

        # RRF融合器
        self.rrf = ReciprocalRankFusion(k=60)

        # 模态权重（可动态调整）
        self.modality_weights = {
            "text": 1.0,
            "image": 0.8,
            "table": 0.6,
        }

    def _encode(self, text: str) -> np.ndarray:
        """编码文本为向量"""
        if self._has_embedding_model:
            emb = self.embedding_model.encode([text], normalize_embeddings=True)
            return emb[0]
        else:
            # 演示用：生成384维随机向量（可复现）
            np.random.seed(hash(text) % 2**32)
            vec = np.random.randn(384).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-8)

    def add_document(self, doc: MultiModalDocument):
        """添加一个多模态文档"""
        self.documents[doc.doc_id] = doc

        # 文本索引
        if doc.text_content:
            if doc.text_embedding is None:
                doc.text_embedding = self._encode(doc.text_content)
            self.text_index.add(doc.doc_id, doc.text_embedding)

        # 图片索引（多张图片取平均嵌入）
        # 实际应用中应为每张图片创建embedding，这里简化为文档级
        if doc.image_embeddings is not None and len(doc.image_embeddings) > 0:
            avg_img_emb = doc.image_embeddings.mean(axis=0)
            self.image_index.add(f"{doc.doc_id}_img", avg_img_emb)

        # 表格索引
        if doc.table_data:
            table_text = self._serialize_tables(doc.table_data)
            doc.table_embeddings = self._encode(table_text)
            self.table_index.add(doc.doc_id, doc.table_embeddings)

    def _serialize_tables(self, tables: List[Dict]) -> str:
        """将表格数据序列化为文本（用于嵌入）"""
        parts = []
        for i, table in enumerate(tables):
            # 表格标题
            if "title" in table:
                parts.append(f"Table {i+1}: {table['title']}")
            # 列名
            if "columns" in table:
                parts.append(" | ".join(str(c) for c in table["columns"]))
            # 数据行
            if "rows" in table:
                for row in table["rows"]:
                    parts.append(" | ".join(str(c) for c in row))
        return "\n".join(parts)

    def search(self, query: str,
               query_image_path: str = None,
               top_k: int = 10,
               dynamic_weights: bool = True) -> List[Dict]:
        """
        多模态混合检索

        参数:
            query: 文本查询
            query_image_path: 可选的查询图片
            top_k: 返回结果数
            dynamic_weights: 是否启用动态权重

        返回:
            [{doc_id, rrf_score, text_rank, image_rank, table_rank, ...}, ...]
        """
        # 1. 构建索引
        self.text_index.build()
        self.image_index.build()
        self.table_index.build()

        # 2. 对查询编码
        query_emb = self._encode(query)

        # 3. 动态权重调整
        if dynamic_weights:
            weights = self._compute_dynamic_weights(query)
        else:
            weights = self.modality_weights.copy()

        print(f"  [动态权重] text={weights['text']:.2f}, "
              f"image={weights['image']:.2f}, table={weights['table']:.2f}")

        # 4. 各模态独立检索
        ranked_lists = {}

        # 文本检索
        text_results = self.text_index.search(query_emb, top_k=top_k * 2)
        ranked_lists["text"] = text_results

        # 图片检索（用文本查询编码近似，实际应用中用CLIP跨模态检索）
        if len(self.image_index) > 0:
            image_results = self.image_index.search(query_emb, top_k=top_k * 2)
            ranked_lists["image"] = image_results

        # 表格检索
        if len(self.table_index) > 0:
            table_results = self.table_index.search(query_emb, top_k=top_k * 2)
            ranked_lists["table"] = table_results

        # 5. RRF融合
        fused = self.rrf.fuse(ranked_lists, weights)

        return fused[:top_k]

    def _compute_dynamic_weights(self, query: str) -> Dict[str, float]:
        """
        动态权重分配
        根据查询中是否有特定关键词来调整各模态权重

        规则:
        - 包含"图/照片/显示/展示" → 提高图片权重
        - 包含"表/数据/统计/对比" → 提高表格权重
        - 包含"计算/函数/公式" → 提高表格权重
        - 默认均衡
        """
        text_w, img_w, tbl_w = 1.0, 0.8, 0.6

        # 图片相关关键词
        image_keywords = ["图", "图片", "照片", "显示", "展示", "呈现",
                          "image", "photo", "picture", "show", "display",
                          "screenshot", "diagram", "chart"]
        if any(kw in query.lower() for kw in image_keywords):
            img_w = 1.2
            text_w = 0.9

        # 表格相关关键词
        table_keywords = ["表", "表格", "数据", "统计", "对比", "数值",
                          "table", "data", "statistics", "comparison",
                          "rows", "columns", "数值"]
        if any(kw in query.lower() for kw in table_keywords):
            tbl_w = 1.2
            text_w = 0.9

        # 图和表混合查询
        if (
            any(kw in query.lower() for kw in image_keywords) and
            any(kw in query.lower() for kw in table_keywords)
        ):
            img_w = 1.0
            tbl_w = 1.0
            text_w = 0.8

        return {"text": text_w, "image": img_w, "table": tbl_w}


# ============================================================
# Section 5: 评估指标
# ============================================================

class HybridRetrievalMetrics:
    """混合检索评估指标"""

    @staticmethod
    def compute_mrr(results: List[Dict],
                    relevant_doc_ids: List[str]) -> float:
        """Mean Reciprocal Rank (MRR)"""
        for rank, result in enumerate(results, start=1):
            if result["doc_id"] in relevant_doc_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_recall_at_k(results: List[Dict],
                            relevant_doc_ids: List[str],
                            k: int = 10) -> float:
        """Recall@K"""
        retrieved = {r["doc_id"] for r in results[:k]}
        relevant = set(relevant_doc_ids)
        if not relevant:
            return 1.0
        return len(retrieved & relevant) / len(relevant)

    @staticmethod
    def compute_ndcg_at_k(results: List[Dict],
                          relevance_judgments: Dict[str, int],
                          k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain (NDCG@K)"""
        dcg = 0.0
        for rank, result in enumerate(results[:k], start=1):
            relevance = relevance_judgments.get(result["doc_id"], 0)
            dcg += (2 ** relevance - 1) / np.log2(rank + 1)

        # Ideal DCG
        ideal_relevances = sorted(relevance_judgments.values(), reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_relevances[:k], start=1):
            idcg += (2 ** rel - 1) / np.log2(rank + 1)

        return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# Section 6: 主流程
# ============================================================

def main():
    """主函数：演示多模态混合检索全流程"""
    print("=" * 70)
    print("A2-02: 多模态混合检索 — 完整演示")
    print("=" * 70)

    # 1. 初始化检索器
    print("\n[步骤1] 初始化MultimodalHybridRetriever...")
    retriever = MultimodalHybridRetriever()

    # 2. 创建演示文档
    print("\n[步骤2] 构架多模态文档集...")
    docs = [
        MultiModalDocument(
            doc_id="doc_001",
            text_content="2024年第三季度财务报告显示营收增长15%，净利润率达到23.5%。公司核心业务板块表现强劲。",
            table_data=[{
                "title": "Q3财务数据",
                "columns": ["指标", "Q3", "Q2", "环比"],
                "rows": [
                    ["营收", "5.2亿", "4.5亿", "+15.5%"],
                    ["净利润", "1.2亿", "1.0亿", "+20.0%"],
                    ["净利率", "23.5%", "22.2%", "+1.3pp"],
                ],
            }],
            metadata={"date": "2024-10-15", "department": "财务部"},
        ),
        MultiModalDocument(
            doc_id="doc_002",
            text_content="产品A的用户满意度调查结果：整体满意度4.2/5分。用户最满意的是响应速度（4.5分）和界面设计（4.3分）。",
            table_data=[{
                "title": "满意度调查",
                "columns": ["维度", "得分", "同比变化"],
                "rows": [
                    ["整体满意度", "4.2", "+0.3"],
                    ["响应速度", "4.5", "+0.2"],
                    ["界面设计", "4.3", "+0.5"],
                    ["功能完整性", "3.8", "-0.1"],
                ],
            }],
            metadata={"date": "2024-10-20", "department": "产品部"},
        ),
        MultiModalDocument(
            doc_id="doc_003",
            text_content="机器学习模型部署指南：包括模型导出、容器化、API服务构建和性能监控。推荐使用ONNX Runtime或TensorRT进行推理优化。",
            table_data=[{
                "title": "部署方案对比",
                "columns": ["方案", "延迟", "吞吐量", "GPU支持"],
                "rows": [
                    ["ONNX Runtime", "<10ms", "1000 QPS", "是"],
                    ["TensorRT", "<5ms", "2000 QPS", "是"],
                    ["TorchServe", "<20ms", "500 QPS", "是"],
                    ["Ollama", "<50ms", "100 QPS", "否"],
                ],
            }],
            metadata={"date": "2024-11-01", "department": "工程部"},
        ),
        MultiModalDocument(
            doc_id="doc_004",
            text_content="市场分析：竞争对手产品B在2024年Q3市场份额下降至12%。公司在AI领域投资增长40%带来的技术优势开始显现。",
            metadata={"date": "2024-10-25", "department": "市场部"},
        ),
        MultiModalDocument(
            doc_id="doc_005",
            text_content="团队建设活动计划：定于11月15日组织户外拓展训练，地点在城市森林公园。预算15万元，预计参与人数80人。",
            metadata={"date": "2024-11-05", "department": "行政部"},
        ),
    ]

    for doc in docs:
        retriever.add_document(doc)
    print(f"  添加了 {len(docs)} 个多模态文档")

    # 3. 测试各种查询
    print("\n[步骤3] 混合检索测试...")
    test_queries = [
        "财务营收增长数据",        # 应命中doc_001（有文本和表格）
        "用户满意度评分",           # 应命中doc_002
        "模型部署GPU加速方案",     # 应命中doc_003
        "市场竞争对手分析",        # 应命中doc_004
    ]

    for query in test_queries:
        print(f"\n  查询: '{query}'")
        results = retriever.search(query, top_k=3)
        for r in results:
            doc = retriever.documents.get(r['doc_id'])
            meta_prefix = doc.metadata.get('department', '?') if doc else ''
            print(f"    #{doc and doc.metadata.get('department','?'):6s} "
                  f"RRF={r['rrf_score']:.6f} "
                  f"| {doc.text_content[:50] if doc else '?'}...")

    # 4. RRF对比实验
    print("\n[步骤4] RRF融合 vs 单模态对比...")
    print(f"  {'查询':<20s} {'纯文本Top1':<20s} {'RRF融合Top1':<20s}")
    print(f"  {'-'*20} {'-'*20} {'-'*20}")
    for query in test_queries:
        # 纯文本
        query_emb = retriever._encode(query)
        text_results = retriever.text_index.search(query_emb, top_k=1)
        # RRF
        all_results = retriever.search(query, top_k=1, dynamic_weights=True)
        text_top1 = text_results[0][0] if text_results else "?"
        rrf_top1 = all_results[0]['doc_id'] if all_results else "?"
        print(f"  {query:<20s} {text_top1:<20s} {rrf_top1:<20s}")

    # 5. 动态权重演示
    print("\n[步骤5] 动态权重分配演示...")
    weight_queries = [
        ("营收数据表格", "图片表格混合"),
        ("展示财务数据", "图片偏重"),
        ("统计对比分析", "表格偏重"),
        ("财务营收增长", "均衡"),
    ]
    for query, desc in weight_queries:
        weights = retriever._compute_dynamic_weights(query)
        print(f"  '{query}' ({desc}): "
              f"text={weights['text']:.2f}, "
              f"image={weights['image']:.2f}, "
              f"table={weights['table']:.2f}")

    # 6. 评估指标演示
    print("\n[步骤6] 评估指标演示...")
    metrics = HybridRetrievalMetrics()
    sample_results = retriever.search("财务数据", top_k=5)
    relevant = ["doc_001", "doc_002"]  # 财务报告和满意度调查都有表格数据
    mrr = metrics.compute_mrr(sample_results, relevant)
    recall = metrics.compute_recall_at_k(sample_results, relevant, k=5)
    print(f"  MRR: {mrr:.4f}")
    print(f"  Recall@5: {recall:.4f}")

    print("\n" + "=" * 70)
    print("A2-02 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
