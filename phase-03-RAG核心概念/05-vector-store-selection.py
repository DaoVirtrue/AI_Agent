#!/usr/bin/env python3
"""
Phase 03 Module 05: Vector Store Selection & Benchmarking
==========================================================
5种向量数据库的完整实现、基准测试与对比：
- Chroma    (嵌入式, HNSW)           → 原型/开发阶段
- Qdrant    (Docker/Cloud, HNSW)     → 中小生产环境
- Milvus    (Docker/K8s, 多种索引)    → 企业级生产环境
- Pinecone  (SaaS, 托管服务)          → 快速上线
- FAISS     (嵌入式/C++, 多种索引)    → 研究/离线/高吞吐

Prerequisites:
  本地依赖:
    pip install chromadb qdrant-client pymilvus pinecone-client
    pip install sentence-transformers numpy faiss-cpu  # 或 faiss-gpu

  Docker (Qdrant & Milvus):
    # 启动 Qdrant（本地开发）
    docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

    # 启动 Milvus Standalone（docker-compose，见文件底部注释）
    # docker compose up -d  # 使用下方 milvus-docker-compose.yml

  API Keys (Pinecone):
    export PINECONE_API_KEY="pcsk_..."
    # 注册: https://app.pinecone.io

  若某个数据库无法连接，脚本会自动跳过并输出清晰的安装指引。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 优雅降级：尝试导入可选依赖
# ---------------------------------------------------------------------------

# Sentence-Transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Chroma
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
        PayloadSchemaType,
        HnswConfigDiff,
        OptimizersConfigDiff,
        WalConfigDiff,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

# Milvus
try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        MilvusClient,
        connections,
        utility,
    )
    from pymilvus import (
        WeightedRanker,
        AnnSearchRequest,
        RRFRanker,
    )
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False

# Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec, PodSpec  # Pinecone SDK v5+
    HAS_PINECONE = True
except ImportError:
    try:
        import pinecone  # 旧版 SDK
        HAS_PINECONE = True
    except ImportError:
        HAS_PINECONE = False

# FAISS
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ===========================================================================
# 测试数据生成器
# ===========================================================================

def generate_test_vectors(
    n: int,
    dim: int = 768,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    生成 n 个随机归一化向量及对应的元数据载荷。

    Args:
        n: 向量数量
        dim: 向量维度（默认 768 = BGE-base / text-embedding-3-small）
        seed: 随机种子，保证可复现

    Returns:
        (vectors: np.ndarray shape=(n, dim), payloads: list[dict])
    """
    rng = np.random.default_rng(seed)

    # 标准正态分布采样，然后 L2 归一化
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms  # 单位向量

    # 生成元数据载荷（模拟真实文档的元信息）
    categories = ["article", "report", "manual", "faq", "tutorial", "blog", "paper", "spec"]
    languages = ["zh", "en", "ja", "ko", "fr", "de", "es"]
    domains = ["tech", "medical", "finance", "legal", "education", "sports", "entertainment"]
    statuses = ["published", "draft", "review", "archived"]

    payloads: List[Dict[str, Any]] = []
    for i in range(n):
        payloads.append({
            "doc_id": f"doc_{i:08d}",
            "title": f"Document Title {i:08d}",
            "category": str(rng.choice(categories)),
            "language": str(rng.choice(languages)),
            "domain": str(rng.choice(domains)),
            "status": str(rng.choice(statuses)),
            "char_count": int(rng.integers(100, 50000)),
            "page_num": int(rng.integers(1, 200)),
            "year": int(rng.integers(2010, 2025)),
            "score": round(float(rng.random()), 4),
            "tags": ",".join(rng.choice(
                ["python", "ai", "ml", "nlp", "cv", "rl", "devops", "cloud", "database", "security"],
                size=int(rng.integers(1, 4)),
                replace=False,
            ).tolist()),
        })

    return vectors, payloads


def generate_query_vectors(
    n_queries: int,
    dim: int = 768,
    seed: int = 123,
    noise_scale: float = 0.05,
    base_vectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    基于已有向量生成查询向量（模拟真实查询场景）。

    若提供 base_vectors，则从中随机采样并添加小噪声；否则随机生成。

    Args:
        n_queries: 查询向量数量
        dim: 向量维度
        seed: 随机种子
        noise_scale: 噪声标准差（越小越相似）
        base_vectors: 可选，基向量集合

    Returns:
        queries: np.ndarray shape=(n_queries, dim)
    """
    rng = np.random.default_rng(seed)

    if base_vectors is not None:
        # 从基向量中随机选取，添加噪声
        indices = rng.integers(0, len(base_vectors), size=n_queries)
        queries = base_vectors[indices].copy()
        noise = rng.normal(0, noise_scale, size=queries.shape).astype(np.float32)
        queries = queries + noise
    else:
        queries = rng.standard_normal((n_queries, dim), dtype=np.float32)

    # L2 归一化
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / norms
    return queries


# ===========================================================================
# 抽象向量存储接口
# ===========================================================================

class VectorStoreBase(ABC):
    """所有向量数据库实现的抽象基类。"""

    def __init__(self, store_name: str = "base"):
        self.store_name = store_name

    @abstractmethod
    def connect(self) -> bool:
        """建立与向量数据库的连接。返回 True 表示成功。"""

    @abstractmethod
    def create_collection(self, name: str, dim: int) -> bool:
        """创建集合/索引。返回 True 表示成功。"""

    @abstractmethod
    def insert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: List[str],
    ) -> Tuple[int, float]:
        """插入向量及元数据。返回 (插入数量, 耗时秒数)。"""

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """搜索最相似的 top_k 个向量。返回结果列表。"""

    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """删除集合。返回 True 表示成功。"""

    @abstractmethod
    def count(self) -> int:
        """返回集合中的向量总数。"""

    @abstractmethod
    def close(self) -> None:
        """关闭连接，释放资源。"""


# ===========================================================================
# 1. Chroma 实现
# ===========================================================================

class ChromaStore(VectorStoreBase):
    """
    Chroma 向量数据库实现。

    支持：
    - 内存模式（开发/测试） 和 持久化模式（生产）
    - 元数据过滤
    - 完整 CRUD
    - 自动 ID 生成

    依赖: pip install chromadb
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        store_name: str = "Chroma",
    ):
        """
        Args:
            persist_directory: 持久化目录路径。None 表示纯内存模式。
        """
        super().__init__(store_name)
        self.persist_directory = persist_directory
        self._client: Any = None
        self._collection: Any = None

    # ---- connect ----

    def connect(self) -> bool:
        if not HAS_CHROMA:
            print(f"[{self.store_name}] 未安装 chromadb。请执行: pip install chromadb")
            return False

        try:
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                print(f"[{self.store_name}] 已连接（持久化模式，路径={self.persist_directory}）")
            else:
                self._client = chromadb.Client(
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                print(f"[{self.store_name}] 已连接（纯内存模式）")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 连接失败: {e}")
            return False

    # ---- create_collection ----

    def create_collection(self, name: str, dim: int) -> bool:
        if self._client is None:
            print(f"[{self.store_name}] 请先调用 connect()")
            return False

        try:
            # 如已存在则先删除
            try:
                self._client.delete_collection(name)
            except Exception:
                pass

            # Chroma 使用 HNSW 索引 + cosine 距离
            self._collection = self._client.create_collection(
                name=name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "hnsw:search_ef": 100,
                },
            )
            print(f"[{self.store_name}] 集合 '{name}' 创建成功（dim={dim}）")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 创建集合失败: {e}")
            return False

    # ---- insert ----

    def insert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int = 1000,
    ) -> Tuple[int, float]:
        if self._collection is None:
            raise RuntimeError("请先调用 create_collection()")

        n = vectors.shape[0]
        total_time = 0.0
        n_inserted = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_vecs = vectors[start:end].tolist()
            batch_payloads = payloads[start:end]
            batch_ids = ids[start:end]

            t0 = time.perf_counter()
            self._collection.add(
                embeddings=batch_vecs,
                metadatas=batch_payloads,
                ids=batch_ids,
            )
            total_time += time.perf_counter() - t0
            n_inserted += (end - start)

        return n_inserted, total_time

    # ---- search ----

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._collection is None:
            raise RuntimeError("请先调用 create_collection()")

        # Chroma 的 where 过滤语法
        where_filter: Optional[Dict[str, Any]] = None
        if filters:
            where_filter = {}
            for key, value in filters.items():
                if isinstance(value, dict):
                    where_filter[key] = value
                elif isinstance(value, (list, tuple)):
                    where_filter[key] = {"$in": value}
                else:
                    where_filter[key] = value

        results = self._collection.query(
            query_embeddings=query.tolist(),
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances", "documents"],
        )

        # 标准化返回格式
        formatted: List[Dict[str, Any]] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted.append({
                    "id": doc_id,
                    "score": 1.0 - results["distances"][0][i] if results["distances"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })
        return formatted

    # ---- delete_collection ----

    def delete_collection(self, name: str) -> bool:
        if self._client is None:
            return False
        try:
            self._client.delete_collection(name)
            self._collection = None
            print(f"[{self.store_name}] 集合 '{name}' 已删除")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 删除失败: {e}")
            return False

    # ---- count ----

    def count(self) -> int:
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    # ---- close ----

    def close(self) -> None:
        self._collection = None
        self._client = None
        print(f"[{self.store_name}] 连接已关闭")


# ===========================================================================
# 2. Qdrant 实现
# ===========================================================================

class QdrantStore(VectorStoreBase):
    """
    Qdrant 向量数据库实现。

    支持：
    - 本地 Docker / Qdrant Cloud 两种连接方式
    - HNSW 索引配置
    - 丰富的载荷过滤（等值、范围、集合包含等）
    - 批量 upsert 并显示进度

    依赖:
      pip install qdrant-client
      docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant  (本地)
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        store_name: str = "Qdrant",
    ):
        """
        Args:
            url: Qdrant 服务地址（本地默认 http://localhost:6333）
            api_key: Qdrant Cloud API Key（本地部署不需要）
        """
        super().__init__(store_name)
        self.url = url
        self.api_key = api_key
        self._client: Optional[QdrantClient] = None
        self._collection_name: str = ""
        self._dim: int = 0

    # ---- connect ----

    def connect(self) -> bool:
        if not HAS_QDRANT:
            print(f"[{self.store_name}] 未安装 qdrant-client。请执行: pip install qdrant-client")
            return False

        try:
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30.0,
            )
            # 测试连接（获取集群信息）
            self._client.get_collections()
            print(f"[{self.store_name}] 已连接（URL={self.url}）")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 连接失败（URL={self.url}）: {e}")
            print(f"  提示: 请确保 Qdrant 已在 Docker 中运行:")
            print(f"    docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant")
            self._client = None
            return False

    # ---- create_collection ----

    def create_collection(self, name: str, dim: int) -> bool:
        if self._client is None:
            print(f"[{self.store_name}] 请先调用 connect()")
            return False

        try:
            # 如已存在则先删除
            try:
                self._client.delete_collection(name)
            except Exception:
                pass

            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                    on_disk=False,  # 小数据集使用内存
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,                     # 每个节点的最大连接数
                    ef_construct=200,         # 构建时搜索范围
                    full_scan_threshold=10000,
                    max_indexing_threads=0,   # 自动
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    memmap_threshold=20000,
                ),
                wal_config=WalConfigDiff(
                    wal_capacity_mb=32,
                ),
            )
            self._collection_name = name
            self._dim = dim
            print(f"[{self.store_name}] 集合 '{name}' 创建成功（dim={dim}, distance=COSINE）")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 创建集合失败: {e}")
            return False

    # ---- 过滤器构建器 ----

    @staticmethod
    def _build_qdrant_filter(filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """
        将字典过滤器转换为 Qdrant Filter 对象。

        支持的过滤操作：
          {"field": "value"}         → 等值匹配
          {"field": ["a", "b"]}      → 集合包含（$in）
          {"field": {"gte": 1, "lte": 100}}  → 范围
          {"field": {"$in": ["a"]}}  → 显式集合包含
        """
        if not filters or not filters:
            return None

        conditions: List[FieldCondition] = []
        for key, value in filters.items():
            if isinstance(value, dict):
                # 范围过滤：{"gte": ..., "lte": ...}
                range_kwargs = {}
                for op_key in ("gte", "gt", "lte", "lt"):
                    if op_key in value:
                        range_kwargs[op_key] = value[op_key]

                match_keywords = ("$in", "$nin")
                if any(k in value for k in match_keywords):
                    # 集合包含
                    if "$in" in value:
                        conditions.append(FieldCondition(
                            key=key,
                            match=qdrant_models.MatchAny(any=list(value["$in"])),
                        ))
                    elif "$nin" in value:
                        conditions.append(FieldCondition(
                            key=key,
                            match=qdrant_models.MatchExcept(**{"except": list(value["$nin"])}),
                        ))
                elif range_kwargs:
                    conditions.append(FieldCondition(
                        key=key,
                        range=qdrant_models.Range(**range_kwargs),
                    ))
                else:
                    # 嵌套字典视为 match
                    conditions.append(FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=str(value)),
                    ))
            elif isinstance(value, (list, tuple)):
                # 集合包含
                conditions.append(FieldCondition(
                    key=key,
                    match=qdrant_models.MatchAny(any=[str(v) for v in value]),
                ))
            else:
                # 等值匹配
                conditions.append(FieldCondition(
                    key=key,
                    match=qdrant_models.MatchValue(value=str(value)),
                ))

        return Filter(must=conditions) if conditions else None

    # ---- insert ----

    def insert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int = 1000,
    ) -> Tuple[int, float]:
        if self._client is None:
            raise RuntimeError("请先调用 connect() 和 create_collection()")
        if not self._collection_name:
            raise RuntimeError("请先调用 create_collection()")

        n = vectors.shape[0]
        total_time = 0.0
        n_inserted = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_points: List[PointStruct] = []
            for i in range(start, end):
                payload = payloads[i].copy()
                batch_points.append(PointStruct(
                    id=int(ids[i].replace("doc_", "")) if ids[i].startswith("doc_") else hash(ids[i]) % (2**63),
                    vector=vectors[i].tolist(),
                    payload=payload,
                ))

            t0 = time.perf_counter()
            self._client.upsert(
                collection_name=self._collection_name,
                points=batch_points,
                wait=True,
            )
            total_time += time.perf_counter() - t0
            n_inserted += (end - start)

        return n_inserted, total_time

    # ---- search ----

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._client is None or not self._collection_name:
            raise RuntimeError("请先调用 connect() 和 create_collection()")

        q_filter = self._build_qdrant_filter(filters)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query.tolist(),
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
            with_vectors=False,
        )

        formatted: List[Dict[str, Any]] = []
        for hit in results:
            formatted.append({
                "id": str(hit.id),
                "score": hit.score,
                "metadata": hit.payload,
            })
        return formatted

    # ---- delete_collection ----

    def delete_collection(self, name: str) -> bool:
        if self._client is None:
            return False
        try:
            self._client.delete_collection(name)
            self._collection_name = ""
            print(f"[{self.store_name}] 集合 '{name}' 已删除")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 删除失败: {e}")
            return False

    # ---- count ----

    def count(self) -> int:
        if self._client is None or not self._collection_name:
            return 0
        try:
            info = self._client.get_collection(self._collection_name)
            return info.points_count if info else 0
        except Exception:
            return 0

    # ---- close ----

    def close(self) -> None:
        if self._client:
            self._client.close()
        self._client = None
        print(f"[{self.store_name}] 连接已关闭")


# ===========================================================================
# 3. Milvus 实现
# ===========================================================================

class MilvusStore(VectorStoreBase):
    """
    Milvus 向量数据库实现。

    支持：
    - Milvus Standalone (Docker) / Zilliz Cloud
    - 多种索引类型: HNSW, IVF_FLAT, IVF_PQ, IVF_SQ8, DISKANN
    - 分区键支持
    - 标量过滤

    依赖:
      pip install pymilvus
      # Docker Standalone:
      # wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml
      # docker compose up -d

    注意: Milvus 需要在 create_collection 之前先建立 schema，流程为:
      connect → create_schema → create_collection → create_index → load
    """

    INDEX_TYPES = ["HNSW", "IVF_FLAT", "IVF_PQ", "IVF_SQ8", "DISKANN"]

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: Optional[str] = None,       # Zilliz Cloud token
        store_name: str = "Milvus",
        index_type: str = "HNSW",
        nlist: int = 128,                  # IVF 聚类数
        M: int = 16,                       # HNSW M
        efConstruction: int = 200,         # HNSW efConstruction
        metric_type: str = "COSINE",
    ):
        """
        Args:
            uri: Milvus 地址（本地默认 http://localhost:19530）
            token: Zilliz Cloud 认证 token
            index_type: 索引类型（HNSW / IVF_FLAT / IVF_PQ / IVF_SQ8 / DISKANN）
        """
        super().__init__(store_name)
        self.uri = uri
        self.token = token
        self.index_type = index_type.upper()
        self.nlist = nlist
        self.M = M
        self.efConstruction = efConstruction
        self.metric_type = metric_type
        self._alias: str = "default"
        self._collection: Optional[Collection] = None
        self._collection_name: str = ""
        self._dim: int = 0

    # ---- connect ----

    def connect(self) -> bool:
        if not HAS_MILVUS:
            print(f"[{self.store_name}] 未安装 pymilvus。请执行: pip install pymilvus")
            return False

        try:
            if self.token:
                connections.connect(
                    alias=self._alias,
                    uri=self.uri,
                    token=self.token,
                    timeout=30,
                )
            else:
                connections.connect(
                    alias=self._alias,
                    host=self.uri.replace("http://", "").replace("https://", "").split(":")[0],
                    port=self.uri.split(":")[-1] if ":" in self.uri else "19530",
                    timeout=30,
                )
            print(f"[{self.store_name}] 已连接（URI={self.uri}）")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 连接失败（URI={self.uri}）: {e}")
            print(f"  提示: 请确保 Milvus 已通过 Docker 运行:")
            print(f"    # 下载 docker-compose 文件后:")
            print(f"    docker compose up -d")
            print(f"  或使用 Zilliz Cloud: https://cloud.zilliz.com")
            return False

    # ---- create_collection ----

    def create_collection(self, name: str, dim: int) -> bool:
        if not connections.has_connection(self._alias):
            print(f"[{self.store_name}] 请先调用 connect()")
            return False

        try:
            # 先删除同名集合
            if utility.has_collection(name, using=self._alias):
                utility.drop_collection(name, using=self._alias)

            # 定义 Schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="year", dtype=DataType.INT64),
                FieldSchema(name="score", dtype=DataType.FLOAT),
                FieldSchema(name="char_count", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(
                fields=fields,
                description=f"Vector store collection: {name}",
                enable_dynamic_field=True,  # 允许在 payload 中存储额外字段
            )

            self._collection = Collection(
                name=name,
                schema=schema,
                using=self._alias,
                consistency_level="Session",  # 可改为 "Strong" / "Bounded" / "Eventually"
            )
            self._collection_name = name
            self._dim = dim

            # 创建索引
            index_params = self._build_index_params()
            self._collection.create_index(
                field_name="vector",
                index_params=index_params,
            )
            print(f"[{self.store_name}] 索引创建成功（type={self.index_type}）")

            # 加载集合到内存
            self._collection.load()
            print(f"[{self.store_name}] 集合 '{name}' 已加载到内存（dim={dim}）")
            return True

        except Exception as e:
            print(f"[{self.store_name}] 创建集合失败: {e}")
            return False

    def _build_index_params(self) -> Dict[str, Any]:
        """根据 index_type 构建索引参数。"""
        index_type = self.index_type

        if index_type == "HNSW":
            return {
                "index_type": "HNSW",
                "metric_type": self.metric_type,
                "params": {
                    "M": self.M,
                    "efConstruction": self.efConstruction,
                },
            }
        elif index_type == "IVF_FLAT":
            return {
                "index_type": "IVF_FLAT",
                "metric_type": self.metric_type,
                "params": {"nlist": self.nlist},
            }
        elif index_type == "IVF_PQ":
            return {
                "index_type": "IVF_PQ",
                "metric_type": self.metric_type,
                "params": {
                    "nlist": self.nlist,
                    "m": self._dim // 2,  # PQ 子向量数（假设 dim/2）
                    "nbits": 8,
                },
            }
        elif index_type == "IVF_SQ8":
            return {
                "index_type": "IVF_SQ8",
                "metric_type": self.metric_type,
                "params": {"nlist": self.nlist},
            }
        elif index_type == "DISKANN":
            return {
                "index_type": "DISKANN",
                "metric_type": self.metric_type,
            }
        else:
            raise ValueError(f"不支持的索引类型: {index_type}。可选: {self.INDEX_TYPES}")

    # ---- insert ----

    def insert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int = 1000,
    ) -> Tuple[int, float]:
        if self._collection is None:
            raise RuntimeError("请先调用 create_collection()")

        n = vectors.shape[0]
        total_time = 0.0
        n_inserted = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_data: List[List[Any]] = []
            for i in range(start, end):
                p = payloads[i]
                batch_data.append([
                    ids[i],                          # id (VARCHAR)
                    vectors[i].tolist(),             # vector (FLOAT_VECTOR)
                    str(p.get("category", "")),      # category
                    str(p.get("language", "")),      # language
                    str(p.get("domain", "")),        # domain
                    int(p.get("year", 2024)),         # year
                    float(p.get("score", 0.0)),       # score
                    int(p.get("char_count", 0)),      # char_count
                ])

            t0 = time.perf_counter()
            self._collection.insert(batch_data)
            total_time += time.perf_counter() - t0
            n_inserted += (end - start)

        # Milvus 插入后需要 flush 才能被搜索
        self._collection.flush()
        return n_inserted, total_time

    # ---- search ----

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._collection is None:
            raise RuntimeError("请先调用 create_collection()")

        # 构建过滤表达式
        expr: Optional[str] = None
        if filters:
            conditions: List[str] = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                elif isinstance(value, (list, tuple)):
                    quoted = ", ".join(f'"{str(v)}"' for v in value)
                    conditions.append(f'{key} in [{quoted}]')
                elif isinstance(value, dict):
                    for op_key, op_val in value.items():
                        if op_key == "gte":
                            conditions.append(f'{key} >= {op_val}')
                        elif op_key == "lte":
                            conditions.append(f'{key} <= {op_val}')
                        elif op_key == "gt":
                            conditions.append(f'{key} > {op_val}')
                        elif op_key == "lt":
                            conditions.append(f'{key} < {op_val}')
                else:
                    conditions.append(f'{key} == {value}')
            expr = " && ".join(conditions) if conditions else None

        search_params = {"metric_type": self.metric_type, "params": {"ef": 100}}
        if self.index_type.startswith("IVF"):
            search_params["params"] = {"nprobe": 16}

        results = self._collection.search(
            data=[query.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["*"],
        )

        formatted: List[Dict[str, Any]] = []
        if results and results[0]:
            for hit in results[0]:
                formatted.append({
                    "id": str(hit.id),
                    "score": hit.distance if self.metric_type == "IP" else 1.0 - hit.distance,
                    "metadata": hit.entity.to_dict() if hasattr(hit, 'entity') else {},
                })
        return formatted

    # ---- delete_collection ----

    def delete_collection(self, name: str) -> bool:
        try:
            if utility.has_collection(name, using=self._alias):
                utility.drop_collection(name, using=self._alias)
            self._collection = None
            self._collection_name = ""
            print(f"[{self.store_name}] 集合 '{name}' 已删除")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 删除失败: {e}")
            return False

    # ---- count ----

    def count(self) -> int:
        if self._collection is None:
            return 0
        try:
            return self._collection.num_entities
        except Exception:
            return 0

    # ---- close ----

    def close(self) -> None:
        if self._collection:
            try:
                self._collection.release()
            except Exception:
                pass
        try:
            connections.disconnect(self._alias)
        except Exception:
            pass
        self._collection = None
        print(f"[{self.store_name}] 连接已关闭")


# ===========================================================================
# 4. Pinecone 实现
# ===========================================================================

class PineconeStore(VectorStoreBase):
    """
    Pinecone 向量数据库实现。

    支持：
    - Serverless 索引（推荐）
    - Pod 索引（传统）
    - 元数据过滤
    - 批量 upsert

    依赖:
      pip install pinecone-client
      设置环境变量: PINECONE_API_KEY="pcsk_..."
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: str = "us-east-1",
        cloud: str = "aws",
        store_name: str = "Pinecone",
        use_serverless: bool = True,
        pod_type: str = "p1.x1",
    ):
        """
        Args:
            api_key: Pinecone API Key（默认从 PINECONE_API_KEY 环境变量读取）
            environment: 环境区域（仅 Pod 索引需要）
            cloud: 云提供商（aws / gcp / azure），Serverless 专用
            use_serverless: True = Serverless, False = Pod
            pod_type: Pod 类型（仅 Pod 索引）
        """
        super().__init__(store_name)
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        self.environment = environment
        self.cloud = cloud
        self.use_serverless = use_serverless
        self.pod_type = pod_type
        self._client: Any = None
        self._index: Any = None
        self._index_name: str = ""
        self._dim: int = 0

    # ---- connect ----

    def connect(self) -> bool:
        if not HAS_PINECONE:
            print(f"[{self.store_name}] 未安装 pinecone-client。请执行: pip install pinecone-client")
            return False

        if not self.api_key:
            print(f"[{self.store_name}] 缺少 API Key。请设置环境变量 PINECONE_API_KEY")
            print(f"  注册地址: https://app.pinecone.io")
            return False

        try:
            self._client = Pinecone(api_key=self.api_key)
            print(f"[{self.store_name}] 已连接（cloud={self.cloud}）")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 连接失败: {e}")
            return False

    # ---- create_collection ----

    def create_collection(self, name: str, dim: int) -> bool:
        if self._client is None:
            print(f"[{self.store_name}] 请先调用 connect()")
            return False

        try:
            # 检查是否已存在
            existing = self._client.list_indexes()
            if name in [idx.get("name", "") for idx in (existing if isinstance(existing, list) else [])]:
                print(f"[{self.store_name}] 索引 '{name}' 已存在，先删除...")
                self._client.delete_index(name)

            # 创建索引
            spec: Any
            if self.use_serverless:
                spec = ServerlessSpec(cloud=self.cloud, region=self.environment)
            else:
                spec = PodSpec(
                    environment=self.environment,
                    pod_type=self.pod_type,
                )

            self._client.create_index(
                name=name,
                dimension=dim,
                metric="cosine",
                spec=spec,
            )
            self._index_name = name
            self._dim = dim
            print(f"[{self.store_name}] 索引 '{name}' 创建成功（dim={dim}, cosine, serverless={self.use_serverless}）")
            print(f"  注意: Pinecone 索引可能需要 1-2 分钟完成初始化")

            # 等待就绪
            self._wait_for_index_ready(timeout_seconds=120)

            self._index = self._client.Index(name)
            return True

        except Exception as e:
            print(f"[{self.store_name}] 创建索引失败: {e}")
            return False

    def _wait_for_index_ready(self, timeout_seconds: int = 120) -> None:
        """等待 Pinecone 索引就绪。"""
        waited = 0
        interval = 5
        while waited < timeout_seconds:
            try:
                desc = self._client.describe_index(self._index_name)
                if desc and getattr(desc, "status", {}).get("ready", False):
                    print(f"[{self.store_name}] 索引已就绪（等待 {waited}s）")
                    return
            except Exception:
                pass
            time.sleep(interval)
            waited += interval
        print(f"[{self.store_name}] 警告: 等待超时，但继续执行（索引可能仍在初始化）")

    # ---- insert ----

    def insert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int = 100,
    ) -> Tuple[int, float]:
        if self._index is None:
            raise RuntimeError("请先调用 create_collection()")

        n = vectors.shape[0]
        total_time = 0.0
        n_inserted = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_items: List[Dict[str, Any]] = []
            for i in range(start, end):
                batch_items.append({
                    "id": ids[i],
                    "values": vectors[i].tolist(),
                    "metadata": {k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                                 for k, v in payloads[i].items()},
                })

            t0 = time.perf_counter()
            self._index.upsert(vectors=batch_items)
            total_time += time.perf_counter() - t0
            n_inserted += (end - start)

        return n_inserted, total_time

    # ---- search ----

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._index is None:
            raise RuntimeError("请先调用 create_collection()")

        # 构建 Pinecone filter
        pine_filter: Optional[Dict[str, Any]] = None
        if filters:
            pine_filter = {}
            for key, value in filters.items():
                if isinstance(value, dict):
                    pine_filter[key] = value
                elif isinstance(value, (list, tuple)):
                    pine_filter[key] = {"$in": [str(v) for v in value]}
                else:
                    pine_filter[key] = {"$eq": str(value)}

        results = self._index.query(
            vector=query.tolist(),
            top_k=top_k,
            filter=pine_filter,
            include_metadata=True,
        )

        formatted: List[Dict[str, Any]] = []
        if hasattr(results, "matches") and results.matches:
            for match in results.matches:
                formatted.append({
                    "id": match.id,
                    "score": match.score if hasattr(match, "score") else None,
                    "metadata": match.metadata if hasattr(match, "metadata") else {},
                })
        return formatted

    # ---- delete_collection ----

    def delete_collection(self, name: str) -> bool:
        if self._client is None:
            return False
        try:
            self._client.delete_index(name)
            self._index = None
            self._index_name = ""
            print(f"[{self.store_name}] 索引 '{name}' 已删除")
            return True
        except Exception as e:
            print(f"[{self.store_name}] 删除失败: {e}")
            return False

    # ---- count ----

    def count(self) -> int:
        if self._index is None:
            return 0
        try:
            stats = self._index.describe_index_stats()
            return stats.get("total_vector_count", 0) if stats else 0
        except Exception:
            return 0

    # ---- close ----

    def close(self) -> None:
        self._index = None
        self._client = None
        print(f"[{self.store_name}] 连接已关闭")


# ===========================================================================
# 5. FAISS 实现
# ===========================================================================

class FaissStore(VectorStoreBase):
    """
    FAISS 向量数据库实现（纯内存 / 离线）。

    支持的索引类型：
    - IndexFlatIP   : 精确内积搜索（最慢但 100% 召回）
    - IndexHNSW     : 分层可导航小世界图（近似最快）
    - IndexIVFFlat  : 倒排文件 + 精确量化
    - IndexIVFSQ8   : 倒排文件 + 标量量化 8-bit（内存减半）
    - IndexIVFPQ    : 倒排文件 + 乘积量化（极致内存压缩）

    依赖: pip install faiss-cpu  或  faiss-gpu

    注意: FAISS 是纯粹的向量搜索引擎，不内置元数据过滤。
          本实现通过 numpy 数组维护元数据索引，在 ann 结果上再做过滤。
    """

    INDEX_BUILDERS: Dict[str, Callable[[int, int], Any]] = {}

    def __init__(
        self,
        index_type: str = "FLAT",
        store_name: str = "FAISS",
        nlist: int = 100,
        M: int = 32,
        efConstruction: int = 200,
        efSearch: int = 64,
        nbits: int = 8,          # PQ 编码位数
        m: int = 48,             # PQ 子向量数
    ):
        """
        Args:
            index_type: "FLAT" | "HNSW" | "IVF_FLAT" | "IVF_SQ8" | "IVF_PQ"
            nlist: IVF 聚类数（数据集越大，nlist 应越大）
            M: HNSW 图的边数
            efConstruction: HNSW 构建时探索因子
            efSearch: HNSW 搜索时探索因子
            nbits: PQ 量化位数
            m: PQ 将向量切分为 m 个子向量
        """
        super().__init__(store_name)
        self.index_type = index_type
        self.nlist = nlist
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.nbits = nbits
        self.m = m
        self._index: Any = None         # faiss.Index
        self._index_flat: Any = None    # 精确索引（用于计算真实 recall）
        self._metadata: List[Dict[str, Any]] = []
        self._id_map: List[str] = []

    # ---- connect ----

    def connect(self) -> bool:
        if not HAS_FAISS:
            print(f"[{self.store_name}] 未安装 faiss。请执行: pip install faiss-cpu (或 faiss-gpu)")
            return False

        # FAISS 是嵌入式库，无需网络连接
        print(f"[{self.store_name}] FAISS 已就绪（纯内存模式）")
        return True

    # ---- create_collection ----

    def create_collection(self, name: str, dim: int) -> bool:
        """
        创建 FAISS 索引。

        FAISS 的 "collection" 概念等价于一个 Index 对象。
        此方法仅记录名称和维度——真正的索引在 insert 时根据数据构建。
        """
        self._index = None
        self._index_flat = None
        self._metadata = []
        self._id_map = []
        print(f"[{self.store_name}] 集合 '{name}' 准备就绪（dim={dim}, type={self.index_type}）")
        return True

    # ---- insert ----

    def insert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int = 100000,  # FAISS 大批量更高效
    ) -> Tuple[int, float]:
        """构建索引并插入向量。FAISS 一次性构建索引（train + add）。"""
        n, dim = vectors.shape
        vectors_f32 = vectors.astype(np.float32)

        t0 = time.perf_counter()

        # ---- 构建索引 ----
        self._index = self._build_faiss_index(vectors_f32, dim)
        self._id_map = list(ids)
        self._metadata = list(payloads)

        elapsed = time.perf_counter() - t0
        print(f"[{self.store_name}] 索引构建完成: {n} 个向量, {elapsed:.2f}s"

              f", 类型={self.index_type}")
        return n, elapsed

    def _build_faiss_index(self, vectors: np.ndarray, dim: int) -> Any:
        """根据 index_type 构建对应的 FAISS 索引。"""
        index_type = self.index_type.upper()

        if index_type == "FLAT":
            # 精确内积搜索
            index = faiss.IndexFlatIP(dim)

        elif index_type == "HNSW":
            # HNSW（基于内积）
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexHNSWFlat(dim, self.M)
            index.hnsw.efConstruction = self.efConstruction
            index.hnsw.efSearch = self.efSearch

        elif index_type == "IVF_FLAT":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.nprobe = 16  # 搜索时探测的聚类数

        elif index_type == "IVF_SQ8":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFScalarQuantizer(
                quantizer, dim, self.nlist, faiss.ScalarQuantizer.QT_8bit,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.train(vectors)
            index.nprobe = 16

        elif index_type == "IVF_PQ":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(
                quantizer, dim, self.nlist, self.m, self.nbits,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.train(vectors)
            index.nprobe = 16

        else:
            raise ValueError(f"不支持的索引类型: {index_type}。可选: FLAT, HNSW, IVF_FLAT, IVF_SQ8, IVF_PQ")

        # 添加向量
        index.add(vectors)
        return index

    # ---- search ----

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._index is None:
            raise RuntimeError("请先调用 insert() 构建索引")

        query_f32 = query.astype(np.float32).reshape(1, -1)

        # 多取一些结果用于再过滤
        fetch_k = top_k * 5 if filters else top_k
        scores, indices = self._index.search(query_f32, min(fetch_k, len(self._id_map)))

        formatted: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            if len(formatted) >= top_k:
                break

            meta = self._metadata[idx].copy() if idx < len(self._metadata) else {}

            # 元数据过滤（后过滤方式）
            if filters:
                if not self._matches_filter(meta, filters):
                    continue

            formatted.append({
                "id": self._id_map[idx],
                "score": float(score),
                "metadata": meta,
            })

        return formatted

    def _build_flat_index(self, vectors: np.ndarray, dim: int) -> Any:
        """构建精确 FLAT 索引用于 recall 计算。"""
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))
        return index

    @staticmethod
    def _matches_filter(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查元数据是否满足过滤条件。"""
        for key, expected in filters.items():
            actual = metadata.get(key)
            if actual is None:
                return False

            if isinstance(expected, dict):
                if "gte" in expected and (not isinstance(actual, (int, float)) or actual < expected["gte"]):
                    return False
                if "lte" in expected and (not isinstance(actual, (int, float)) or actual > expected["lte"]):
                    return False
                if "gt" in expected and (not isinstance(actual, (int, float)) or actual <= expected["gt"]):
                    return False
                if "lt" in expected and (not isinstance(actual, (int, float)) or actual >= expected["lt"]):
                    return False
            elif isinstance(expected, (list, tuple)):
                if actual not in expected:
                    return False
            else:
                if str(actual) != str(expected):
                    return False
        return True

    # ---- delete_collection ----

    def delete_collection(self, name: str) -> bool:
        self._index = None
        self._index_flat = None
        self._metadata = []
        self._id_map = []
        print(f"[{self.store_name}] 集合 '{name}' 已删除（内存释放）")
        return True

    # ---- count ----

    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    # ---- close ----

    def close(self) -> None:
        self.delete_collection("all")
        print(f"[{self.store_name}] 资源已释放")


# ===========================================================================
# 基准测试框架
# ===========================================================================

@dataclass
class InsertBenchmark:
    """插入操作基准测试结果。"""
    total_vectors: int = 0
    total_time_sec: float = 0.0
    vectors_per_second: float = 0.0
    average_ms_per_vector: float = 0.0
    peak_memory_mb: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class QueryBenchmark:
    """查询操作基准测试结果。"""
    num_queries: int = 0
    top_k: int = 10
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    qps: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class IndexAlgorithmResult:
    """索引算法对比结果。"""
    algorithm: str = ""
    build_time_sec: float = 0.0
    query_p50_ms: float = 0.0
    query_mean_ms: float = 0.0
    recall_at_10: float = 0.0
    index_size_mb: float = 0.0
    memory_per_vector_bytes: float = 0.0


def get_memory_usage_mb() -> float:
    """获取当前进程的内存用量（MB），跨平台。"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # 降级：尝试读取 /proc（Linux）
        try:
            with open(f"/proc/{os.getpid()}/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return -1.0


def benchmark_insert(
    store: VectorStoreBase,
    vectors: np.ndarray,
    payloads: List[Dict[str, Any]],
    ids: List[str],
    batch_size: int = 1000,
    collection_name: str = "bench_collection",
) -> InsertBenchmark:
    """
    测量向量数据库中插入操作的性能。

    Args:
        store: 向量存储实例（已连接）
        vectors: 要插入的向量
        payloads: 元数据
        ids: ID 列表
        batch_size: 批大小
        collection_name: 集合名称

    Returns:
        InsertBenchmark 结果
    """
    n = vectors.shape[0]
    dim = vectors.shape[1]

    # 确保集合已创建
    store.delete_collection(collection_name)
    if not store.create_collection(collection_name, dim):
        return InsertBenchmark(
            total_vectors=n,
            success=False,
            error_message="创建集合失败",
        )

    mem_before = get_memory_usage_mb()

    try:
        t_start = time.perf_counter()
        n_inserted, insert_time = store.insert(vectors, payloads, ids, batch_size=batch_size)
        t_end = time.perf_counter()

        mem_after = get_memory_usage_mb()
        total_time = t_end - t_start

        return InsertBenchmark(
            total_vectors=n_inserted,
            total_time_sec=round(total_time, 4),
            vectors_per_second=round(n_inserted / total_time, 2) if total_time > 0 else 0,
            average_ms_per_vector=round((total_time / n_inserted) * 1000, 4) if n_inserted > 0 else 0,
            peak_memory_mb=round(mem_after - mem_before, 2),
            success=True,
        )
    except Exception as e:
        return InsertBenchmark(
            total_vectors=n,
            success=False,
            error_message=str(e),
        )


def benchmark_query(
    store: VectorStoreBase,
    queries: np.ndarray,
    top_k: int = 10,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> QueryBenchmark:
    """
    测量向量数据库的查询延迟。

    执行 warmup + 多次查询，收集 p50 / p95 / p99 / mean / std。

    Args:
        store: 向量存储实例（已有数据）
        queries: 查询向量矩阵 (n_queries, dim)
        top_k: 返回结果数
        num_runs: 正式测量轮数
        warmup_runs: 预热轮数（不计入统计）

    Returns:
        QueryBenchmark 结果
    """
    n_queries = queries.shape[0]
    latencies: List[float] = []

    if n_queries == 0:
        return QueryBenchmark(
            num_queries=0, top_k=top_k, success=False,
            error_message="查询向量为空",
        )

    try:
        # 预热
        for i in range(min(warmup_runs, n_queries)):
            store.search(queries[i % n_queries], top_k=top_k)

        # 正式测量
        total_search_time = 0.0
        for i in range(num_runs):
            q = queries[i % n_queries]
            t0 = time.perf_counter()
            store.search(q, top_k=top_k)
            elapsed = (time.perf_counter() - t0) * 1000  # 转毫秒
            latencies.append(elapsed)
            total_search_time += elapsed / 1000.0  # 秒

        latencies_sorted = sorted(latencies)
        n_lat = len(latencies_sorted)

        total_time_sec = sum(latencies) / 1000.0
        qps = num_runs / total_time_sec if total_time_sec > 0 else 0

        return QueryBenchmark(
            num_queries=num_runs,
            top_k=top_k,
            p50_ms=round(latencies_sorted[int(n_lat * 0.50)], 4),
            p95_ms=round(latencies_sorted[min(int(n_lat * 0.95), n_lat - 1)], 4),
            p99_ms=round(latencies_sorted[min(int(n_lat * 0.99), n_lat - 1)], 4),
            mean_ms=round(mean(latencies), 4),
            std_ms=round(stdev(latencies), 4) if len(latencies) > 1 else 0.0,
            min_ms=round(min(latencies), 4),
            max_ms=round(max(latencies), 4),
            qps=round(qps, 2),
            success=True,
        )
    except Exception as e:
        return QueryBenchmark(
            num_queries=num_runs, top_k=top_k,
            success=False, error_message=str(e),
        )


def benchmark_at_scales(
    store_class: type,
    dim: int = 768,
    scales: Optional[List[int]] = None,
    store_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    在不同数据规模下对向量存储进行基准测试。

    对每个 scale：
      1. 生成该规模的测试数据
      2. 初始化并连接存储
      3. 测量插入性能
      4. 测量查询性能
      5. 清理集合

    Args:
        store_class: VectorStoreBase 的子类
        dim: 向量维度
        scales: 要测试的规模列表，默认 [1000, 10000, 100000]
        store_kwargs: 传递给 store_class 构造函数的参数

    Returns:
        {scale: {"insert": InsertBenchmark, "query": QueryBenchmark}}
    """
    if scales is None:
        scales = [1000, 10000, 100000]
    if store_kwargs is None:
        store_kwargs = {}

    results: Dict[int, Dict[str, Any]] = {}

    for scale in scales:
        print(f"\n  --- 规模: {scale:,} vectors ---")
        store = store_class(**store_kwargs)

        if not store.connect():
            results[scale] = {
                "insert": InsertBenchmark(total_vectors=scale, success=False, error_message="连接失败"),
                "query": QueryBenchmark(num_queries=0, top_k=10, success=False, error_message="连接失败"),
            }
            continue

        # 生成数据
        vecs, payloads = generate_test_vectors(scale, dim=dim, seed=42)
        ids = [f"doc_{i:08d}" for i in range(scale)]
        queries = generate_query_vectors(50, dim=dim, seed=123, base_vectors=vecs)

        collection_name = f"scale_bench_{scale}"

        # 插入基准测试
        ins = benchmark_insert(store, vecs, payloads, ids, batch_size=min(1000, max(100, scale // 10)), collection_name=collection_name)
        results[scale] = {"insert": ins}

        # 查询基准测试
        if ins.success:
            q = benchmark_query(store, queries, top_k=10, num_runs=min(100, max(10, scale // 10)))
            results[scale]["query"] = q
        else:
            results[scale]["query"] = QueryBenchmark(
                num_queries=0, top_k=10, success=False,
                error_message="插入失败，跳过查询测试",
            )

        # 清理
        store.delete_collection(collection_name)
        store.close()

    return results


# ===========================================================================
# 索引算法对比器（FAISS）
# ===========================================================================

def compare_index_algorithms(
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth_k: int = 10,
    index_configs: Optional[List[Dict[str, Any]]] = None,
) -> List[IndexAlgorithmResult]:
    """
    在相同数据上对比不同的 FAISS 索引算法。

    对比维度:
    - 构建时间
    - 查询 p50 延迟
    - recall@10（以 FLAT 为 ground truth）
    - 索引内存占用

    Args:
        vectors: 数据集向量 (n, dim)
        queries: 查询向量 (m, dim)
        ground_truth_k: ground truth 使用的 K
        index_configs: 索引配置列表。默认对比 FLAT, HNSW, IVF_FLAT, IVF_SQ8, IVF_PQ

    Returns:
        List[IndexAlgorithmResult]
    """
    if not HAS_FAISS:
        print("[compare_index_algorithms] 需要安装 faiss: pip install faiss-cpu")
        return []

    if index_configs is None:
        index_configs = [
            {"type": "FLAT",     "label": "FLAT (精确)"},
            {"type": "HNSW",     "label": "HNSW", "M": 32, "efConstruction": 200, "efSearch": 64},
            {"type": "IVF_FLAT", "label": "IVF_FLAT", "nlist": 100},
            {"type": "IVF_SQ8",  "label": "IVF_SQ8", "nlist": 100},
            {"type": "IVF_PQ",   "label": "IVF_PQ", "nlist": 100, "m": 48, "nbits": 8},
        ]

    n, dim = vectors.shape
    vectors_f32 = vectors.astype(np.float32)
    queries_f32 = queries.astype(np.float32)
    num_queries = min(queries_f32.shape[0], 100)

    # ---- Step 1: 构建 Ground Truth（使用 FLAT）----
    print("\n[compare_index_algorithms] 构建 Ground Truth（FLAT 精确索引）...")
    t0 = time.perf_counter()
    gt_index = faiss.IndexFlatIP(dim)
    gt_index.add(vectors_f32)
    gt_build_time = time.perf_counter() - t0
    gt_dists, gt_indices = gt_index.search(queries_f32[:num_queries], ground_truth_k)
    gt_index_size_mb = _estimate_faiss_index_size(gt_index)
    print(f"  Ground Truth: {gt_build_time:.2f}s, {gt_index_size_mb:.1f}MB")

    results: List[IndexAlgorithmResult] = []

    print(f"\n{'='*80}")
    print(f"{'算法':<16} {'构建时间':>10} {'p50(ms)':>10} {'mean(ms)':>10} {'recall@10':>10} {'大小(MB)':>10} {'字节/向量':>12}")
    print(f"{'='*80}")

    # Ground Truth 行
    gt_result = IndexAlgorithmResult(
        algorithm="FLAT (Ground Truth)",
        build_time_sec=round(gt_build_time, 3),
        query_p50_ms=0,
        query_mean_ms=0,
        recall_at_10=1.0000,
        index_size_mb=round(gt_index_size_mb, 2),
        memory_per_vector_bytes=round(gt_index_size_mb * 1024 * 1024 / n, 2),
    )

    # 测量 FLAT 查询延迟
    latencies_flat: List[float] = []
    for i in range(num_queries):
        t0 = time.perf_counter()
        gt_index.search(queries_f32[i:i+1], ground_truth_k)
        latencies_flat.append((time.perf_counter() - t0) * 1000)
    latencies_flat_sorted = sorted(latencies_flat)
    gt_result.query_p50_ms = round(latencies_flat_sorted[int(len(latencies_flat_sorted) * 0.50)], 3)
    gt_result.query_mean_ms = round(mean(latencies_flat), 3)
    results.append(gt_result)
    _print_result_row(gt_result)

    # ---- Step 2: 测试每种索引 ----
    for config in index_configs:
        if config["type"] == "FLAT":
            continue  # Ground Truth 已测试

        algo_label = config.get("label", config["type"])
        print(f"\n  构建 {algo_label}...")

        try:
            # 构建索引
            t0 = time.perf_counter()

            index_type = config["type"]
            if index_type == "HNSW":
                M = config.get("M", 32)
                index = faiss.IndexHNSWFlat(dim, M)
                index.hnsw.efConstruction = config.get("efConstruction", 200)
                index.hnsw.efSearch = config.get("efSearch", 64)
                index.add(vectors_f32)
            elif index_type == "IVF_FLAT":
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, config.get("nlist", 100), faiss.METRIC_INNER_PRODUCT)
                index.train(vectors_f32)
                index.add(vectors_f32)
                index.nprobe = config.get("nprobe", 16)
            elif index_type == "IVF_SQ8":
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFScalarQuantizer(
                    quantizer, dim, config.get("nlist", 100),
                    faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT,
                )
                index.train(vectors_f32)
                index.add(vectors_f32)
                index.nprobe = config.get("nprobe", 16)
            elif index_type == "IVF_PQ":
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(
                    quantizer, dim, config.get("nlist", 100),
                    config.get("m", dim // 16), config.get("nbits", 8),
                    faiss.METRIC_INNER_PRODUCT,
                )
                index.train(vectors_f32)
                index.add(vectors_f32)
                index.nprobe = config.get("nprobe", 16)
            else:
                print(f"    未知索引类型: {index_type}，跳过")
                continue

            build_time = time.perf_counter() - t0
            index_size_mb = _estimate_faiss_index_size(index)

            # 测量查询延迟
            latencies: List[float] = []
            for i in range(num_queries):
                t0 = time.perf_counter()
                index.search(queries_f32[i:i+1], ground_truth_k)
                latencies.append((time.perf_counter() - t0) * 1000)
            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
            mean_lat = mean(latencies)

            # 测量 recall@10
            dists_approx, indices_approx = index.search(queries_f32[:num_queries], ground_truth_k)

            recall_sum = 0.0
            for q_idx in range(num_queries):
                gt_set = set(gt_indices[q_idx])
                approx_set = set(indices_approx[q_idx])
                overlap = len(gt_set & approx_set)
                recall_sum += overlap / ground_truth_k if ground_truth_k > 0 else 0.0
            recall_at_10 = recall_sum / num_queries

            result = IndexAlgorithmResult(
                algorithm=algo_label,
                build_time_sec=round(build_time, 3),
                query_p50_ms=round(p50, 3),
                query_mean_ms=round(mean_lat, 3),
                recall_at_10=round(recall_at_10, 4),
                index_size_mb=round(index_size_mb, 2),
                memory_per_vector_bytes=round(index_size_mb * 1024 * 1024 / n, 2),
            )
            results.append(result)
            _print_result_row(result)

            # 释放资源
            del index

        except Exception as e:
            print(f"    {algo_label} 失败: {e}")

    print(f"{'='*80}\n")
    return results


def _estimate_faiss_index_size(index: Any) -> float:
    """估算 FAISS 索引的内存占用（MB）。"""
    ntotal = index.ntotal
    dim = index.d
    # 基础向量: n * d * 4 bytes (float32)
    base_bytes = ntotal * dim * 4

    # 尝试获取更精确的索引大小
    try:
        import io
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
            tmp_path = tmp.name
        faiss.write_index(index, tmp_path)
        file_size = os.path.getsize(tmp_path)
        os.unlink(tmp_path)
        return file_size / (1024 * 1024)
    except Exception:
        return base_bytes / (1024 * 1024)


def _print_result_row(result: IndexAlgorithmResult) -> None:
    """打印一行算法对比结果。"""
    print(
        f"{result.algorithm:<16} "
        f"{result.build_time_sec:>8.3f}s "
        f"{result.query_p50_ms:>8.3f} "
        f"{result.query_mean_ms:>8.3f} "
        f"{result.recall_at_10:>8.1%} "
        f"{result.index_size_mb:>8.1f} "
        f"{result.memory_per_vector_bytes:>10.1f}"
    )


# ===========================================================================
# 迁移脚本: Chroma → Qdrant
# ===========================================================================

def migrate_chroma_to_qdrant(
    chroma_path: str,
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: Optional[str] = None,
    collection_name: str = "default",
    batch_size: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    从 Chroma 迁移数据到 Qdrant。

    流程:
      1. 连接 Chroma（持久化目录）
      2. 读取所有 embedding + metadata + id
      3. 连接 Qdrant
      4. 创建同名 collection
      5. 批量写入 Qdrant
      6. 验证：确保两边 count 一致

    Args:
        chroma_path: Chroma 持久化目录路径
        qdrant_url: Qdrant 服务 URL
        qdrant_api_key: Qdrant Cloud API Key（可选）
        collection_name: 要迁移的 collection 名称
        batch_size: 每批写入 Qdrant 的向量数量
        verbose: 是否打印详细进度

    Returns:
        {
            "success": bool,
            "source_count": int,
            "target_count": int,
            "batches": int,
            "total_time_sec": float,
            "errors": list[str],
        }
    """
    errors: List[str] = []
    t_start = time.perf_counter()

    # ---- Step 1: 从 Chroma 读取 ----
    if not HAS_CHROMA:
        return {"success": False, "source_count": 0, "target_count": 0, "batches": 0, "total_time_sec": 0, "errors": ["未安装 chromadb"]}

    if verbose:
        print(f"\n[迁移] 步骤 1/4: 从 Chroma 读取（路径={chroma_path}）...")

    try:
        chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        chroma_collection = chroma_client.get_collection(collection_name)
        source_count = chroma_collection.count()
        if verbose:
            print(f"  源数据: {source_count} 个向量")

        if source_count == 0:
            return {"success": True, "source_count": 0, "target_count": 0, "batches": 0, "total_time_sec": 0, "errors": ["源集合为空"]}

        # 分批读取全部数据
        all_embeddings: List[List[float]] = []
        all_metadatas: List[Dict[str, Any]] = []
        all_ids: List[str] = []

        offset = 0
        while offset < source_count:
            chunk = chroma_collection.get(
                limit=min(batch_size, source_count - offset),
                offset=offset,
                include=["embeddings", "metadatas"],
            )
            if not chunk or not chunk.get("ids"):
                break
            all_ids.extend(chunk["ids"])
            all_embeddings.extend(chunk["embeddings"] if chunk.get("embeddings") else [])
            all_metadatas.extend(chunk["metadatas"] if chunk.get("metadatas") else [{}] * len(chunk["ids"]))
            offset += len(chunk["ids"])

        if verbose:
            print(f"  读取完成: {len(all_ids)} 条记录")

        chroma_client = None  # 释放连接
    except Exception as e:
        errors.append(f"Chroma 读取失败: {e}")
        return {"success": False, "source_count": 0, "target_count": 0, "batches": 0, "total_time_sec": 0, "errors": errors}

    # ---- Step 2: 准备向量数据 ----
    if verbose:
        print(f"[迁移] 步骤 2/4: 准备向量数据...")

    vectors_np = np.array(all_embeddings, dtype=np.float32) if all_embeddings else np.array([[]])

    # ---- Step 3: 写入 Qdrant ----
    if not HAS_QDRANT:
        return {"success": False, "source_count": len(all_ids), "target_count": 0, "batches": 0, "total_time_sec": 0, "errors": ["未安装 qdrant-client"]}

    if verbose:
        print(f"[迁移] 步骤 3/4: 写入 Qdrant（URL={qdrant_url}）...")

    try:
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60.0)

        # 删除同名 collection（如果存在）
        try:
            qdrant_client.delete_collection(collection_name)
        except Exception:
            pass

        dim = vectors_np.shape[1]
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )

        n = len(all_ids)
        n_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points: List[PointStruct] = []
            for i in range(start, end):
                points.append(PointStruct(
                    id=i,
                    vector=vectors_np[i].tolist(),
                    payload=all_metadatas[i],
                ))
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            n_batches += 1
            if verbose and n_batches % 10 == 0:
                print(f"  已写入 {end}/{n} ({end * 100 // n}%)")

        # ---- Step 4: 验证 ----
        if verbose:
            print(f"[迁移] 步骤 4/4: 验证数据完整性...")

        info = qdrant_client.get_collection(collection_name)
        target_count = info.points_count if info else 0

        qdrant_client.close()
    except Exception as e:
        errors.append(f"Qdrant 写入失败: {e}")
        return {"success": False, "source_count": len(all_ids), "target_count": 0, "batches": 0, "total_time_sec": time.perf_counter() - t_start, "errors": errors}

    total_time = time.perf_counter() - t_start
    success = (source_count == target_count)

    # ---- 打印迁移摘要 ----
    if verbose:
        print(f"\n{'='*60}")
        print(f"  迁移摘要")
        print(f"{'='*60}")
        print(f"  源 (Chroma):    {source_count:>8} 个向量")
        print(f"  目标 (Qdrant):  {target_count:>8} 个向量")
        print(f"  一致性检查:     {'通过' if success else '失败'}")
        print(f"  批次数:         {n_batches:>8}")
        print(f"  总耗时:         {total_time:>8.2f}s")
        if not success:
            print(f"  警告: 源和目标数量不一致！")
        print(f"{'='*60}\n")

    return {
        "success": success,
        "source_count": source_count,
        "target_count": target_count,
        "batches": n_batches,
        "total_time_sec": round(total_time, 2),
        "errors": errors,
    }


# ===========================================================================
# 工具函数：打印格式化结果
# ===========================================================================

def print_section_header(title: str, width: int = 70) -> None:
    """打印章节标题。"""
    print(f"\n{'#'*width}")
    print(f"# {title}")
    print(f"{'#'*width}\n")


def print_subsection_header(title: str) -> None:
    """打印子标题。"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_insert_result(label: str, result: InsertBenchmark) -> None:
    """打印插入基准测试结果。"""
    if result.success:
        print(f"  [{label}]")
        print(f"    总计向量: {result.total_vectors:>12,}")
        print(f"    总耗时:   {result.total_time_sec:>12.3f}s")
        print(f"    吞吐量:   {result.vectors_per_second:>12,.0f} vectors/s")
        print(f"    单向量:   {result.average_ms_per_vector:>12.4f}ms")
        print(f"    内存增长: {result.peak_memory_mb:>12.1f} MB")
    else:
        print(f"  [{label}] 失败: {result.error_message}")


def print_query_result(label: str, result: QueryBenchmark) -> None:
    """打印查询基准测试结果。"""
    if result.success:
        print(f"  [{label}]")
        print(f"    查询次数: {result.num_queries:>12}")
        print(f"    Top-K:    {result.top_k:>12}")
        print(f"    P50:      {result.p50_ms:>12.3f}ms")
        print(f"    P95:      {result.p95_ms:>12.3f}ms")
        print(f"    P99:      {result.p99_ms:>12.3f}ms")
        print(f"    均值:     {result.mean_ms:>12.3f}ms")
        print(f"    标准差:   {result.std_ms:>12.3f}ms")
        print(f"    QPS:      {result.qps:>12.1f}")
    else:
        print(f"  [{label}] 失败: {result.error_message}")


def print_summary_table(
    store_results: Dict[str, Dict[str, Any]],
) -> None:
    """打印综合对比表格。"""
    print_section_header("综合对比: 5 种向量数据库")

    header = (
        f"{'数据库':<12} "
        f"{'连接':<6} "
        f"{'插入':>10} "
        f"{'吞吐/s':>10} "
        f"{'P50(ms)':>10} "
        f"{'P95(ms)':>10} "
        f"{'QPS':>10} "
    )
    sep = "-" * len(header)

    print(header)
    print(sep)

    for store_name, data in store_results.items():
        ins = data.get("insert")
        q = data.get("query")

        connected = "Yes" if (hasattr(ins, "success") and ins.success) else "No"
        ins_time = f"{ins.total_time_sec:.2f}s" if (ins and ins.success) else "N/A"
        ins_vps = f"{ins.vectors_per_second:.0f}" if (ins and ins.success) else "N/A"
        p50 = f"{q.p50_ms:.2f}" if (q and q.success) else "N/A"
        p95 = f"{q.p95_ms:.2f}" if (q and q.success) else "N/A"
        q_qps = f"{q.qps:.1f}" if (q and q.success) else "N/A"

        print(
            f"{store_name:<12} "
            f"{connected:<6} "
            f"{ins_time:>10} "
            f"{ins_vps:>10} "
            f"{p50:>10} "
            f"{p95:>10} "
            f"{q_qps:>10} "
        )

    print(sep)
    print()


# ===========================================================================
# __main__: 完整演示
# ===========================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 头部信息
    # ------------------------------------------------------------------
    print("=" * 78)
    print("  Phase 03 Module 05: 向量数据库选型与基准测试")
    print("  对比: Chroma | Qdrant | Milvus | Pinecone | FAISS")
    print("=" * 78)
    print()
    print("Prerequisites:")
    print("  pip install chromadb qdrant-client pymilvus pinecone-client")
    print("  pip install sentence-transformers faiss-cpu numpy")
    print()
    print("  Docker (Qdrant):  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant")
    print("  Docker (Milvus):  docker compose up -d  (见文件底部 docker-compose.yml)")
    print("  Pinecone API Key: export PINECONE_API_KEY=\"pcsk_...\"")
    print()
    print("  若某个数据库无法连接，脚本会自动跳过并输出安装指引。")
    print()

    # ------------------------------------------------------------------
    # 生成测试数据
    # ------------------------------------------------------------------
    TEST_N = 10000     # 1万向量（可根据机器性能调整）
    TEST_DIM = 768     # 与 BGE-base / text-embedding-3-small 一致
    TOP_K = 10
    NUM_QUERY_RUNS = 100

    print_section_header("Step 0: 生成测试数据")
    print(f"  生成 {TEST_N:,} 个 {TEST_DIM}-维归一化向量...")
    vectors, payloads = generate_test_vectors(TEST_N, dim=TEST_DIM)
    ids = [f"doc_{i:08d}" for i in range(TEST_N)]

    # 生成查询向量（从基向量加噪声）
    queries = generate_query_vectors(NUM_QUERY_RUNS, dim=TEST_DIM, seed=123, noise_scale=0.05, base_vectors=vectors)
    print(f"  生成 {NUM_QUERY_RUNS} 个查询向量（添加少量噪声）")
    print(f"  向量内存: {vectors.nbytes / (1024*1024):.1f} MB")

    # 汇总所有存储的测试结果
    all_store_results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Step 1: Chroma
    # ------------------------------------------------------------------
    print_section_header("Step 1: Chroma 基准测试")
    chroma_store = ChromaStore(persist_directory=None)  # 内存模式

    if chroma_store.connect():
        print_subsection_header("插入测试")
        ins_result = benchmark_insert(chroma_store, vectors, payloads, ids, batch_size=1000, collection_name="chroma_bench")
        all_store_results["Chroma"] = {"insert": ins_result, "query": None}
        print_insert_result("Chroma (内存模式)", ins_result)

        if ins_result.success:
            print_subsection_header("查询测试")
            q_result = benchmark_query(chroma_store, queries, top_k=TOP_K, num_runs=NUM_QUERY_RUNS)
            all_store_results["Chroma"]["query"] = q_result
            print_query_result("Chroma", q_result)

        chroma_store.delete_collection("chroma_bench")
        chroma_store.close()
    else:
        all_store_results["Chroma"] = {
            "insert": InsertBenchmark(success=False, error_message="未安装或无法连接"),
            "query": QueryBenchmark(success=False, error_message="未安装或无法连接"),
        }

    # ------------------------------------------------------------------
    # Step 2: Qdrant
    # ------------------------------------------------------------------
    print_section_header("Step 2: Qdrant 基准测试")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", None)
    qdrant_store = QdrantStore(url=qdrant_url, api_key=qdrant_api_key)

    if qdrant_store.connect():
        print_subsection_header("插入测试")
        ins_result = benchmark_insert(qdrant_store, vectors, payloads, ids, batch_size=1000, collection_name="qdrant_bench")
        all_store_results["Qdrant"] = {"insert": ins_result, "query": None}
        print_insert_result("Qdrant", ins_result)

        if ins_result.success:
            print_subsection_header("查询测试")
            q_result = benchmark_query(qdrant_store, queries, top_k=TOP_K, num_runs=NUM_QUERY_RUNS)
            all_store_results["Qdrant"]["query"] = q_result
            print_query_result("Qdrant", q_result)

        qdrant_store.delete_collection("qdrant_bench")
        qdrant_store.close()
    else:
        all_store_results["Qdrant"] = {
            "insert": InsertBenchmark(success=False, error_message="未安装或无法连接"),
            "query": QueryBenchmark(success=False, error_message="未安装或无法连接"),
        }

    # ------------------------------------------------------------------
    # Step 3: Milvus
    # ------------------------------------------------------------------
    print_section_header("Step 3: Milvus 基准测试")
    milvus_uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
    milvus_token = os.environ.get("MILVUS_TOKEN", None)
    milvus_store = MilvusStore(uri=milvus_uri, token=milvus_token, index_type="HNSW")

    if milvus_store.connect():
        print_subsection_header("插入测试")
        ins_result = benchmark_insert(milvus_store, vectors, payloads, ids, batch_size=1000, collection_name="milvus_bench")
        all_store_results["Milvus"] = {"insert": ins_result, "query": None}
        print_insert_result("Milvus (HNSW)", ins_result)

        if ins_result.success:
            print_subsection_header("查询测试")
            q_result = benchmark_query(milvus_store, queries, top_k=TOP_K, num_runs=NUM_QUERY_RUNS)
            all_store_results["Milvus"]["query"] = q_result
            print_query_result("Milvus", q_result)

        milvus_store.delete_collection("milvus_bench")
        milvus_store.close()
    else:
        all_store_results["Milvus"] = {
            "insert": InsertBenchmark(success=False, error_message="未安装或无法连接"),
            "query": QueryBenchmark(success=False, error_message="未安装或无法连接"),
        }

    # ------------------------------------------------------------------
    # Step 4: Pinecone
    # ------------------------------------------------------------------
    print_section_header("Step 4: Pinecone 基准测试")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    pinecone_store = PineconeStore(
        api_key=pinecone_api_key,
        environment=os.environ.get("PINECONE_ENV", "us-east-1"),
        use_serverless=True,
    )

    if pinecone_api_key:
        if pinecone_store.connect():
            print_subsection_header("插入测试")
            ins_result = benchmark_insert(pinecone_store, vectors, payloads, ids, batch_size=100, collection_name="pinecone-bench")
            all_store_results["Pinecone"] = {"insert": ins_result, "query": None}
            print_insert_result("Pinecone (Serverless)", ins_result)

            if ins_result.success:
                print_subsection_header("查询测试")
                q_result = benchmark_query(pinecone_store, queries, top_k=TOP_K, num_runs=min(20, NUM_QUERY_RUNS))
                all_store_results["Pinecone"]["query"] = q_result
                print_query_result("Pinecone", q_result)

            pinecone_store.delete_collection("pinecone-bench")
            pinecone_store.close()
        else:
            all_store_results["Pinecone"] = {
                "insert": InsertBenchmark(success=False, error_message="连接失败"),
                "query": QueryBenchmark(success=False, error_message="连接失败"),
            }
    else:
        print(f"  [Pinecone] 跳过: 未设置 PINECONE_API_KEY 环境变量")
        print(f"    提示: export PINECONE_API_KEY=\"pcsk_...\"")
        print(f"    注册: https://app.pinecone.io")
        all_store_results["Pinecone"] = {
            "insert": InsertBenchmark(success=False, error_message="缺少 API Key"),
            "query": QueryBenchmark(success=False, error_message="缺少 API Key"),
        }

    # ------------------------------------------------------------------
    # Step 5: FAISS
    # ------------------------------------------------------------------
    print_section_header("Step 5: FAISS 基准测试")
    faiss_store = FaissStore(index_type="IVF_FLAT", nlist=100)

    if faiss_store.connect():
        print_subsection_header("插入测试（= 索引构建）")
        ins_result = benchmark_insert(faiss_store, vectors, payloads, ids, batch_size=100000, collection_name="faiss_bench")
        all_store_results["FAISS"] = {"insert": ins_result, "query": None}
        print_insert_result("FAISS (IVF_FLAT)", ins_result)

        if ins_result.success:
            print_subsection_header("查询测试")
            q_result = benchmark_query(faiss_store, queries, top_k=TOP_K, num_runs=NUM_QUERY_RUNS)
            all_store_results["FAISS"]["query"] = q_result
            print_query_result("FAISS", q_result)

        faiss_store.delete_collection("faiss_bench")
        faiss_store.close()
    else:
        all_store_results["FAISS"] = {
            "insert": InsertBenchmark(success=False, error_message="未安装 faiss"),
            "query": QueryBenchmark(success=False, error_message="未安装 faiss"),
        }

    # ------------------------------------------------------------------
    # Step 6: 索引算法对比（FAISS）
    # ------------------------------------------------------------------
    print_section_header("Step 6: FAISS 索引算法对比")
    if HAS_FAISS:
        # 用较小数据集做充分对比
        n_compare = min(5000, TEST_N)
        vecs_compare = vectors[:n_compare]
        q_compare = queries[:50]

        print(f"  使用 {n_compare:,} 向量 + 50 查询进行算法对比...")
        algo_results = compare_index_algorithms(vecs_compare, q_compare, ground_truth_k=10)

        # 打印选型建议
        if algo_results:
            print("  选型建议:")
            sorted_by_speed = sorted(
                [r for r in algo_results if "FLAT" not in r.algorithm],
                key=lambda x: x.query_p50_ms,
            )
            sorted_by_recall = sorted(
                [r for r in algo_results if "FLAT" not in r.algorithm],
                key=lambda x: x.recall_at_10,
                reverse=True,
            )

            if sorted_by_speed:
                print(f"    最快查询:  {sorted_by_speed[0].algorithm} ({sorted_by_speed[0].query_p50_ms:.2f}ms p50)")
            if sorted_by_recall:
                print(f"    最高召回:  {sorted_by_recall[0].algorithm} ({sorted_by_recall[0].recall_at_10:.1%} recall@10)")

            # 最佳平衡
            best_balanced = max(
                [r for r in algo_results if "FLAT" not in r.algorithm],
                key=lambda x: (x.recall_at_10 * 0.6 - x.query_p50_ms / max(1, max(r.query_p50_ms for r in algo_results if "FLAT" not in r.algorithm)) * 0.4),
            )
            print(f"    最佳平衡:  {best_balanced.algorithm} (兼顾速度与召回)")
    else:
        print("  [跳过] FAISS 未安装")

    # ------------------------------------------------------------------
    # Step 7: 综合对比表
    # ------------------------------------------------------------------
    print_summary_table(all_store_results)

    # ------------------------------------------------------------------
    # Step 8: 多规模基准测试（FAISS，因为它不需要外部服务）
    # ------------------------------------------------------------------
    print_section_header("Step 7: FAISS 多规模基准测试")
    if HAS_FAISS:
        scales = [1000, 5000, 10000]
        scale_results = benchmark_at_scales(
            FaissStore,
            dim=TEST_DIM,
            scales=scales,
            store_kwargs={"index_type": "IVF_FLAT", "nlist": 100},
        )

        print(f"\n  {'规模':>12}  {'插入':>10}  {'吞吐/s':>10}  {'P50(ms)':>10}  {'P95(ms)':>10}  {'QPS':>10}")
        print(f"  {'-'*60}")
        for scale in scales:
            data = scale_results.get(scale, {})
            ins = data.get("insert")
            q = data.get("query")

            ins_t = f"{ins.total_time_sec:.2f}s" if (ins and ins.success) else "N/A"
            ins_v = f"{ins.vectors_per_second:.0f}" if (ins and ins.success) else "N/A"
            q_p50 = f"{q.p50_ms:.2f}" if (q and q.success) else "N/A"
            q_p95 = f"{q.p95_ms:.2f}" if (q and q.success) else "N/A"
            q_qps = f"{q.qps:.1f}" if (q and q.success) else "N/A"

            print(f"  {scale:>10,}  {ins_t:>10}  {ins_v:>10}  {q_p50:>10}  {q_p95:>10}  {q_qps:>10}")
    else:
        print("  [跳过] FAISS 未安装")

    # ------------------------------------------------------------------
    # Step 9: 迁移脚本使用示例
    # ------------------------------------------------------------------
    print_section_header("Step 8: 迁移脚本使用示例")

    print("""
  # Chroma → Qdrant 迁移示例:
  #
  # from __main__ import migrate_chroma_to_qdrant
  #
  # result = migrate_chroma_to_qdrant(
  #     chroma_path="./chroma_data",
  #     qdrant_url="http://localhost:6333",
  #     collection_name="my_knowledge_base",
  #     batch_size=100,
  #     verbose=True,
  # )
  #
  # print(f"迁移{'成功' if result['success'] else '失败'}")
  # print(f"  {result['source_count']} → {result['target_count']} 个向量")
  # print(f"  耗时: {result['total_time_sec']}s")

  # 从 FAISS 导出为 JSON（可被其他系统导入）:
  # faiss_index = faiss_store._index
  # faiss.write_index(faiss_index, "my_index.faiss")

  # 各数据库适合的规模:
  #   Chroma    → 原型/开发 (< 1M 向量)
  #   Qdrant    → 中小生产 (< 100M 向量)
  #   Milvus    → 企业级 (> 1B 向量)
  #   Pinecone  → 无需运维 (按需付费)
  #   FAISS     → 离线/嵌入 (无上限，但需自行管理)
""")

    # ------------------------------------------------------------------
    # Docker Compose 示例（注释形式）
    # ------------------------------------------------------------------
    print_section_header("附录: Docker Compose 配置参考")
    print("""
  # --- Milvus Standalone docker-compose.yml ---
  # 文件: milvus-docker-compose.yml
  #
  # version: '3.5'
  #
  # services:
  #   etcd:
  #     container_name: milvus-etcd
  #     image: quay.io/coreos/etcd:v3.5.5
  #     environment:
  #       - ETCD_AUTO_COMPACTION_MODE=revision
  #       - ETCD_AUTO_COMPACTION_RETENTION=1000
  #       - ETCD_QUOTA_BACKEND_BYTES=4294967296
  #       - ETCD_SNAPSHOT_COUNT=50000
  #     volumes:
  #       - ./volumes/etcd:/etcd
  #     command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
  #     healthcheck:
  #       test: ["CMD", "etcdctl", "endpoint", "health"]
  #       interval: 30s
  #       timeout: 20s
  #       retries: 3
  #
  #   minio:
  #     container_name: milvus-minio
  #     image: minio/minio:RELEASE.2023-03-20T20-16-18Z
  #     environment:
  #       MINIO_ACCESS_KEY: minioadmin
  #       MINIO_SECRET_KEY: minioadmin
  #     ports:
  #       - "9001:9001"
  #       - "9000:9000"
  #     volumes:
  #       - ./volumes/minio:/minio_data
  #     command: minio server /minio_data --console-address ":9001"
  #     healthcheck:
  #       test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
  #       interval: 30s
  #       timeout: 20s
  #       retries: 3
  #
  #   standalone:
  #     container_name: milvus-standalone
  #     image: milvusdb/milvus:v2.4.0
  #     command: ["milvus", "run", "standalone"]
  #     security_opt:
  #       - seccomp:unconfined
  #     environment:
  #       ETCD_ENDPOINTS: etcd:2379
  #       MINIO_ADDRESS: minio:9000
  #     volumes:
  #       - ./volumes/milvus:/var/lib/milvus
  #     healthcheck:
  #       test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
  #       interval: 30s
  #       start_period: 90s
  #       timeout: 20s
  #       retries: 3
  #     ports:
  #       - "19530:19530"
  #       - "9091:9091"
  #     depends_on:
  #       - "etcd"
  #       - "minio"
  #
  # networks:
  #   default:
  #     name: milvus
  #
  # --- Qdrant 快速启动（单个 docker 命令）---
  # docker run -d --name qdrant \\
  #   -p 6333:6333 \\
  #   -p 6334:6334 \\
  #   -v $(pwd)/qdrant_storage:/qdrant/storage:z \\
  #   qdrant/qdrant
""")

    # ------------------------------------------------------------------
    # 完成
    # ------------------------------------------------------------------
    print_section_header("测试完成")
    print("  所有基准测试已完成。")
    print("  成功完成的存储会显示具体性能数据；")
    print("  跳过的存储可按照提示安装/启动后重新运行。")
    print()
    print("  下一步: 运行 06-basic-retrieval.ipynb 了解 Dense 检索、Top-K 效应与查询扩展")
    print()

"""
Milvus Standalone Docker Compose (完整 YAML):

将以下内容保存为 milvus-docker-compose.yml，然后执行 docker compose up -d：

version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ./volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ./volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
"""
