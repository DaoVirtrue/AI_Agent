#!/usr/bin/env python3
"""
四级缓存系统
Four-Level Cache System for Production RAG

L1: 精确查询缓存 (Exact Query Cache)
    - 存储: Redis
    - 键: MD5(query)
    - 延迟: <1ms
    - 命中率: ~5%
    - TTL: 1小时

L2: 语义相似度缓存 (Semantic Similarity Cache)
    - 存储: Redis + 内存索引
    - 键: embedding cosine ≥ 0.92 匹配
    - 延迟: ~50ms
    - 命中率: ~15%
    - TTL: 30分钟

L3: 检索结果缓存 (Retrieval Result Cache)
    - 存储: Redis
    - 键: MD5(query + top_k)
    - 值: 检索到的chunk列表
    - 延迟: ~100ms
    - 命中率: ~30%
    - TTL: 1天 (索引更新时失效)

L4: LLM响应缓存 (LLM Response Cache)
    - 存储: Redis / 本地磁盘
    - 键: MD5(prompt)
    - 值: LLM生成的完整响应
    - 延迟: ~200ms (vs 2-5s API调用)
    - 命中率: ~40%
    - TTL: 根据内容类型变化
"""

import json
import time
import hashlib
import threading
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from collections import OrderedDict

import numpy as np


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class CacheLevel(Enum):
    """缓存层级"""
    L1_EXACT_QUERY = 1       # 精确查询缓存
    L2_SEMANTIC_SIMILAR = 2  # 语义相似度缓存
    L3_RETRIEVAL_RESULT = 3  # 检索结果缓存
    L4_LLM_RESPONSE = 4      # LLM响应缓存


class ContentType(Enum):
    """内容类型（用于L4 TTL策略）"""
    FACTUAL = "factual"           # 事实型: TTL 7天
    ANALYTICAL = "analytical"    # 分析型: TTL 1天
    CREATIVE = "creative"        # 创意型: TTL 6小时
    REAL_TIME = "real_time"      # 实时型: TTL 5分钟


@dataclass
class CacheConfig:
    """缓存配置"""
    level: CacheLevel
    ttl_seconds: int
    max_size: int
    backend: str  # "redis" or "memory"

    @staticmethod
    def defaults() -> dict[CacheLevel, "CacheConfig"]:
        return {
            CacheLevel.L1_EXACT_QUERY: CacheConfig(
                level=CacheLevel.L1_EXACT_QUERY,
                ttl_seconds=3600,        # 1小时
                max_size=10000,
                backend="redis",
            ),
            CacheLevel.L2_SEMANTIC_SIMILAR: CacheConfig(
                level=CacheLevel.L2_SEMANTIC_SIMILAR,
                ttl_seconds=1800,        # 30分钟
                max_size=50000,
                backend="redis",
            ),
            CacheLevel.L3_RETRIEVAL_RESULT: CacheConfig(
                level=CacheLevel.L3_RETRIEVAL_RESULT,
                ttl_seconds=86400,       # 1天
                max_size=100000,
                backend="redis",
            ),
            CacheLevel.L4_LLM_RESPONSE: CacheConfig(
                level=CacheLevel.L4_LLM_RESPONSE,
                ttl_seconds=604800,      # 默认7天（事实型）
                max_size=200000,
                backend="redis",
            ),
        }


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    ttl: int
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: dict = field(default_factory=dict)


class CacheStats:
    """缓存统计"""

    def __init__(self):
        self.hits: dict[CacheLevel, int] = {level: 0 for level in CacheLevel}
        self.misses: dict[CacheLevel, int] = {level: 0 for level in CacheLevel}
        self.invalidations: dict[CacheLevel, int] = {level: 0 for level in CacheLevel}
        self.latency_sum: dict[CacheLevel, float] = {level: 0.0 for level in CacheLevel}
        self._lock = threading.RLock()

    def hit(self, level: CacheLevel, latency_ms: float):
        with self._lock:
            self.hits[level] += 1
            self.latency_sum[level] += latency_ms

    def miss(self, level: CacheLevel):
        with self._lock:
            self.misses[level] += 1

    def invalidate(self, level: CacheLevel):
        with self._lock:
            self.invalidations[level] += 1

    def get_hit_rate(self, level: CacheLevel) -> float:
        with self._lock:
            total = self.hits[level] + self.misses[level]
            return self.hits[level] / total if total > 0 else 0.0

    def get_avg_latency(self, level: CacheLevel) -> float:
        with self._lock:
            return self.latency_sum[level] / self.hits[level] if self.hits[level] > 0 else 0.0

    def report(self) -> dict:
        """生成统计报告"""
        report = {}
        for level in CacheLevel:
            total = self.hits[level] + self.misses[level]
            report[level.name] = {
                "hits": self.hits[level],
                "misses": self.misses[level],
                "total": total,
                "hit_rate": f"{self.get_hit_rate(level):.1%}",
                "avg_latency_ms": f"{self.get_avg_latency(level):.2f}",
                "invalidations": self.invalidations[level],
            }
        return report


# ============================================================
# 缓存后端抽象 (Cache Backend Abstraction)
# ============================================================

class CacheBackend(ABC):
    """缓存后端基类"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> bool:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> bool:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass


class MemoryBackend(CacheBackend):
    """内存缓存后端（LRU淘汰）"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() - entry.created_at > entry.ttl:
                self._store.pop(key, None)
                return None
            entry.access_count += 1
            entry.last_accessed = time.time()
            # LRU: 移到末尾
            self._store.move_to_end(key)
            return entry.value

    def set(self, key: str, value: Any, ttl: int) -> bool:
        with self._lock:
            # LRU淘汰
            if len(self._store) >= self.max_size and key not in self._store:
                self._store.popitem(last=False)  # 淘汰最旧的

            self._store[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=ttl,
            )
            self._store.move_to_end(key)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> bool:
        with self._lock:
            self._store.clear()
            return True

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def keys_matching(self, prefix: str) -> list[str]:
        """返回匹配前缀的所有键（用于批量失效）"""
        with self._lock:
            return [k for k in self._store if k.startswith(prefix)]


class RedisBackend(CacheBackend):
    """Redis缓存后端"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(
                    host=self.host, port=self.port, db=self.db,
                    decode_responses=False,
                )
                self._client.ping()
            except ImportError:
                print("redis-py未安装，回退到内存缓存")
                return None
            except Exception as e:
                print(f"Redis连接失败: {e}，回退到内存缓存")
                return None
        return self._client

    def get(self, key: str) -> Optional[Any]:
        client = self.client
        if client is None:
            return None
        try:
            data = client.get(key)
            if data:
                return pickle.loads(data)
        except Exception:
            pass
        return None

    def set(self, key: str, value: Any, ttl: int) -> bool:
        client = self.client
        if client is None:
            return False
        try:
            data = pickle.dumps(value)
            client.setex(key, ttl, data)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        client = self.client
        if client is None:
            return False
        try:
            return client.delete(key) > 0
        except Exception:
            return False

    def clear(self) -> bool:
        client = self.client
        if client is None:
            return False
        try:
            client.flushdb()
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        client = self.client
        if client is None:
            return False
        try:
            return client.exists(key) > 0
        except Exception:
            return False

    def keys_matching(self, prefix: str) -> list[str]:
        """返回匹配前缀的所有键"""
        client = self.client
        if client is None:
            return []
        try:
            return [k.decode() for k in client.keys(f"{prefix}*")]
        except Exception:
            return []


class HybridBackend(CacheBackend):
    """混合后端: Redis + 内存二级缓存"""

    def __init__(self, redis_host="localhost", redis_port=6379, memory_max=5000):
        self.redis = RedisBackend(redis_host, redis_port)
        self.memory = MemoryBackend(memory_max)

    def get(self, key: str) -> Optional[Any]:
        # 先查内存
        value = self.memory.get(key)
        if value is not None:
            return value
        # 再查Redis
        value = self.redis.get(key)
        if value is not None:
            self.memory.set(key, value, 300)  # 内存缓存5分钟
        return value

    def set(self, key: str, value: Any, ttl: int) -> bool:
        self.memory.set(key, value, min(ttl, 300))
        return self.redis.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        self.memory.delete(key)
        return self.redis.delete(key)

    def clear(self) -> bool:
        self.memory.clear()
        return self.redis.clear()

    def exists(self, key: str) -> bool:
        return self.memory.exists(key) or self.redis.exists(key)


# ============================================================
# 四级缓存管理器 (RAG Cache Manager)
# ============================================================

class RAGCacheManager:
    """四级RAG缓存管理器"""

    def __init__(self, use_redis: bool = False, redis_host: str = "localhost", redis_port: int = 6379):
        self.configs = CacheConfig.defaults()
        self.stats = CacheStats()

        # 初始化各层后端
        if use_redis:
            self.backends = {
                CacheLevel.L1_EXACT_QUERY: HybridBackend(redis_host, redis_port, 2000),
                CacheLevel.L2_SEMANTIC_SIMILAR: HybridBackend(redis_host, redis_port, 5000),
                CacheLevel.L3_RETRIEVAL_RESULT: HybridBackend(redis_host, redis_port, 10000),
                CacheLevel.L4_LLM_RESPONSE: HybridBackend(redis_host, redis_port, 20000),
            }
        else:
            # 纯内存模式
            self.backends = {
                CacheLevel.L1_EXACT_QUERY: MemoryBackend(10000),
                CacheLevel.L2_SEMANTIC_SIMILAR: MemoryBackend(50000),
                CacheLevel.L3_RETRIEVAL_RESULT: MemoryBackend(100000),
                CacheLevel.L4_LLM_RESPONSE: MemoryBackend(200000),
            }

        # L2专用: 语义相似度匹配索引
        self._l2_index: dict[str, np.ndarray] = {}  # key -> embedding
        self._l2_index_lock = threading.RLock()

        self._invalidated_version: int = 0  # 索引版本号，用于全局失效

    # ========== L1: 精确查询缓存 ==========

    def l1_get(self, query: str) -> Optional[Any]:
        """L1: 精确查询缓存查找
        键: MD5(query标准化)
        延迟目标: <1ms
        """
        level = CacheLevel.L1_EXACT_QUERY
        key = self._l1_key(query)
        config = self.configs[level]

        t_start = time.time()
        value = self.backends[level].get(key)
        latency_ms = (time.time() - t_start) * 1000

        if value is not None:
            self.stats.hit(level, latency_ms)
            return value
        else:
            self.stats.miss(level)
            return None

    def l1_set(self, query: str, value: Any) -> bool:
        """L1: 存储精确查询结果"""
        level = CacheLevel.L1_EXACT_QUERY
        key = self._l1_key(query)
        config = self.configs[level]
        return self.backends[level].set(key, value, config.ttl_seconds)

    @staticmethod
    def _l1_key(query: str) -> str:
        """L1缓存键: 标准化查询的MD5"""
        normalized = " ".join(query.lower().split())  # 空白字符标准化
        return f"l1:{hashlib.md5(normalized.encode()).hexdigest()}"

    # ========== L2: 语义相似度缓存 ==========

    def l2_get(self, query_embedding: np.ndarray, threshold: float = 0.92) -> Optional[Any]:
        """L2: 语义相似度缓存查找
        通过余弦相似度匹配历史查询
        延迟目标: <50ms
        """
        level = CacheLevel.L2_SEMANTIC_SIMILAR
        t_start = time.time()

        with self._l2_index_lock:
            best_key = None
            best_similarity = threshold

            for key, cached_emb in self._l2_index.items():
                similarity = self._cosine_similarity(query_embedding, cached_emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_key = key

        if best_key:
            value = self.backends[level].get(best_key)
            latency_ms = (time.time() - t_start) * 1000
            if value is not None:
                self.stats.hit(level, latency_ms)
                return value

        self.stats.miss(level)
        return None

    def l2_set(self, query_embedding: np.ndarray, query_text: str, value: Any) -> bool:
        """L2: 存储语义相似度缓存"""
        level = CacheLevel.L2_SEMANTIC_SIMILAR
        key = self._l2_key(query_text)
        config = self.configs[level]

        with self._l2_index_lock:
            self._l2_index[key] = query_embedding.copy()

            # 限制索引大小
            if len(self._l2_index) > config.max_size:
                # 简单FIFO淘汰
                oldest_key = next(iter(self._l2_index))
                del self._l2_index[oldest_key]

        return self.backends[level].set(key, value, config.ttl_seconds)

    @staticmethod
    def _l2_key(query: str) -> str:
        return f"l2:{hashlib.md5(query.encode()).hexdigest()}"

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_norm, b_norm))

    # ========== L3: 检索结果缓存 ==========

    def l3_get(self, query: str, top_k: int) -> Optional[list[dict]]:
        """L3: 检索结果缓存查找
        键: MD5(query + top_k)
        值: 检索到的chunk列表
        延迟目标: <100ms
        """
        level = CacheLevel.L3_RETRIEVAL_RESULT
        key = self._l3_key(query, top_k)

        t_start = time.time()
        value = self.backends[level].get(key)
        latency_ms = (time.time() - t_start) * 1000

        if value is not None:
            self.stats.hit(level, latency_ms)
            return value
        else:
            self.stats.miss(level)
            return None

    def l3_set(self, query: str, top_k: int, chunks: list[dict]) -> bool:
        """L3: 存储检索结果"""
        level = CacheLevel.L3_RETRIEVAL_RESULT
        key = self._l3_key(query, top_k)
        config = self.configs[level]
        return self.backends[level].set(key, chunks, config.ttl_seconds)

    @staticmethod
    def _l3_key(query: str, top_k: int) -> str:
        """L3缓存键: query + top_k 的组合哈希"""
        combined = f"{query}||k={top_k}||v={hashlib.md5(query.encode()).hexdigest()}"
        return f"l3:{hashlib.md5(combined.encode()).hexdigest()}"

    def l3_invalidate_by_index_update(self) -> int:
        """索引更新时批量失效L3缓存
        策略: 增加版本号，旧版本缓存自然过期或被标记失效
        """
        self._invalidated_version += 1
        # 清空所有L3缓存
        count = 0
        backend = self.backends[CacheLevel.L3_RETRIEVAL_RESULT]
        keys = backend.keys_matching("l3:")
        for key in keys:
            if backend.delete(key):
                count += 1
                self.stats.invalidate(CacheLevel.L3_RETRIEVAL_RESULT)
        return count

    # ========== L4: LLM响应缓存 ==========

    def l4_get(self, prompt: str, model: str = "") -> Optional[str]:
        """L4: LLM响应缓存查找
        键: MD5(prompt + model)
        延迟目标: <200ms
        """
        level = CacheLevel.L4_LLM_RESPONSE
        key = self._l4_key(prompt, model)

        t_start = time.time()
        value = self.backends[level].get(key)
        latency_ms = (time.time() - t_start) * 1000

        if value is not None:
            self.stats.hit(level, latency_ms)
            return value
        else:
            self.stats.miss(level)
            return None

    def l4_set(self, prompt: str, response: str, model: str = "",
               content_type: ContentType = ContentType.FACTUAL) -> bool:
        """L4: 存储LLM响应
        TTL根据内容类型动态调整
        """
        level = CacheLevel.L4_LLM_RESPONSE
        key = self._l4_key(prompt, model)

        # 根据内容类型确定TTL
        content_ttl = {
            ContentType.FACTUAL: 604800,     # 7天
            ContentType.ANALYTICAL: 86400,   # 1天
            ContentType.CREATIVE: 21600,     # 6小时
            ContentType.REAL_TIME: 300,      # 5分钟
        }
        ttl = content_ttl.get(content_type, 604800)

        entry = {
            "response": response,
            "content_type": content_type.value,
            "model": model,
            "cached_at": time.time(),
        }
        return self.backends[level].set(key, entry, ttl)

    @staticmethod
    def _l4_key(prompt: str, model: str = "") -> str:
        """L4缓存键: prompt + model 的组合哈希"""
        combined = f"{prompt}||model={model}"
        return f"l4:{hashlib.md5(combined.encode()).hexdigest()}"

    def l4_invalidate_by_content_type(self, content_type: ContentType) -> int:
        """按内容类型失效L4缓存"""
        count = 0
        backend = self.backends[CacheLevel.L4_LLM_RESPONSE]
        keys = backend.keys_matching("l4:")
        for key in keys:
            entry = backend.get(key)
            if entry and isinstance(entry, dict) and entry.get("content_type") == content_type.value:
                if backend.delete(key):
                    count += 1
                    self.stats.invalidate(CacheLevel.L4_LLM_RESPONSE)
        return count

    # ========== 统一的缓存接口 ==========

    def get_or_compute(self, query: str, query_embedding: np.ndarray,
                       top_k: int, compute_fn, level: CacheLevel = CacheLevel.L3_RETRIEVAL_RESULT,
                       model: str = "") -> Any:
        """统一的缓存获取或计算接口"""
        if level == CacheLevel.L1_EXACT_QUERY:
            cached = self.l1_get(query)
            if cached is not None:
                return cached
        elif level == CacheLevel.L2_SEMANTIC_SIMILAR:
            cached = self.l2_get(query_embedding)
            if cached is not None:
                return cached
        elif level == CacheLevel.L3_RETRIEVAL_RESULT:
            cached = self.l3_get(query, top_k)
            if cached is not None:
                return cached
        elif level == CacheLevel.L4_LLM_RESPONSE:
            cached = self.l4_get(query, model)
            if cached is not None:
                return cached

        # 缓存未命中，执行计算
        result = compute_fn()

        # 写入缓存
        if level == CacheLevel.L1_EXACT_QUERY:
            self.l1_set(query, result)
        elif level == CacheLevel.L2_SEMANTIC_SIMILAR:
            self.l2_set(query_embedding, query, result)
        elif level == CacheLevel.L3_RETRIEVAL_RESULT:
            self.l3_set(query, top_k, result)
        elif level == CacheLevel.L4_LLM_RESPONSE:
            self.l4_set(query, result, model)

        return result

    # ========== 缓存预热 (Cache Warming) ==========

    def warm_l1(self, frequent_queries: list[tuple[str, Any]]):
        """预热L1: 预加载高频查询"""
        for query, result in frequent_queries:
            self.l1_set(query, result)
        print(f"L1预热完成: {len(frequent_queries)} 条")

    def warm_l2(self, query_embeddings: list[tuple[np.ndarray, str, Any]]):
        """预热L2: 预加载语义查询"""
        for emb, text, value in query_embeddings:
            self.l2_set(emb, text, value)
        print(f"L2预热完成: {len(query_embeddings)} 条")

    def warm_l3(self, query_results: list[tuple[str, int, list[dict]]]):
        """预热L3: 预加载检索结果"""
        for query, top_k, chunks in query_results:
            self.l3_set(query, top_k, chunks)
        print(f"L3预热完成: {len(query_results)} 条")

    def warm_l4(self, prompt_responses: list[tuple[str, str, ContentType]]):
        """预热L4: 预加载LLM响应"""
        for prompt, response, ctype in prompt_responses:
            self.l4_set(prompt, response, content_type=ctype)
        print(f"L4预热完成: {len(prompt_responses)} 条")

    # ========== 缓存统计与监控 ==========

    def get_report(self) -> dict:
        """获取完整缓存报告"""
        report = self.stats.report()
        report["_meta"] = {
            "cache_levels": 4,
            "total_hits": sum(s["hits"] for s in report.values() if isinstance(s, dict)),
            "total_misses": sum(s["misses"] for s in report.values() if isinstance(s, dict)),
            "index_version": self._invalidated_version,
        }
        # 计算总体命中率
        total_h = report["_meta"]["total_hits"]
        total_m = report["_meta"]["total_misses"]
        report["_meta"]["overall_hit_rate"] = f"{total_h / max(total_h + total_m, 1):.1%}"
        return report

    def clear_all(self):
        """清空所有缓存"""
        for backend in self.backends.values():
            backend.clear()
        with self._l2_index_lock:
            self._l2_index.clear()
        print("所有缓存已清空")


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("四级缓存系统 - 演示运行")
    print("=" * 60)

    # 初始化缓存管理器
    cache = RAGCacheManager(use_redis=False)  # 使用内存模式演示

    # 1. L1 精确查询缓存演示
    print("\n--- L1: 精确查询缓存 ---")
    query = "什么是RAG系统？"
    l1_result = cache.l1_get(query)
    print(f"首次查询 '{query}': {'命中 ✅' if l1_result else '未命中 ❌'}")

    cache.l1_set(query, {"answer": "RAG是检索增强生成系统"})
    l1_result2 = cache.l1_get(query)
    print(f"二次查询 '{query}': {'命中 ✅' if l1_result2 else '未命中 ❌'}")
    print(f"结果: {l1_result2}")

    # 测试空白标准化
    spaced_query = "什么是  RAG   系统？"
    l1_result3 = cache.l1_get(spaced_query)
    print(f"空白标准化查询: {'命中 ✅' if l1_result3 else '未命中 ❌'}")

    # 2. L2 语义相似度缓存演示
    print("\n--- L2: 语义相似度缓存 ---")
    emb1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    query1 = "RAG是什么？"
    cache.l2_set(emb1, query1, {"answer": "RAG解释..."})

    # 相似查询
    emb2 = emb1 + np.random.randn(8) * 0.03  # 微小扰动
    l2_result = cache.l2_get(emb2, threshold=0.90)
    print(f"语义相似查询: {'命中 ✅' if l2_result else '未命中 ❌'}")
    if l2_result:
        print(f"匹配结果: {l2_result}")

    # 不同查询
    emb3 = np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1])
    l2_result2 = cache.l2_get(emb3, threshold=0.90)
    print(f"不同语义查询: {'命中 ✅' if l2_result2 else '未命中 ❌ (正确)'}")

    # 3. L3 检索结果缓存演示
    print("\n--- L3: 检索结果缓存 ---")
    search_query = "向量数据库选型"
    fake_chunks = [
        {"id": "c1", "text": "Faiss是...", "score": 0.95},
        {"id": "c2", "text": "Milvus是...", "score": 0.88},
        {"id": "c3", "text": "ChromaDB是...", "score": 0.82},
    ]

    l3_result = cache.l3_get(search_query, 3)
    print(f"首次检索: {'命中 ✅' if l3_result else '未命中 ❌'}")

    cache.l3_set(search_query, 3, fake_chunks)
    l3_result2 = cache.l3_get(search_query, 3)
    print(f"二次检索: {'命中 ✅' if l3_result2 else '未命中 ❌'}")
    print(f"检索结果: {len(l3_result2)} 条")

    # 不同top_k
    l3_different_k = cache.l3_get(search_query, 5)
    print(f"不同top_k查询: {'命中 ✅' if l3_different_k else '未命中 ❌ (正确)'}")

    # L3失效
    invalidated = cache.l3_invalidate_by_index_update()
    print(f"索引更新后失效: {invalidated} 条")
    l3_after = cache.l3_get(search_query, 3)
    print(f"失效后查询: {'命中 ✅' if l3_after else '未命中 ❌ (正确)'}")

    # 4. L4 LLM响应缓存演示
    print("\n--- L4: LLM响应缓存 ---")
    prompt = "请解释RAG与微调的区别"
    llm_response = "RAG通过检索外部知识增强生成，而微调是调整模型参数..."

    l4_result = cache.l4_get(prompt)
    print(f"首次LLM查询: {'命中 ✅' if l4_result else '未命中 ❌'}")

    cache.l4_set(prompt, llm_response, content_type=ContentType.FACTUAL)
    l4_result2 = cache.l4_get(prompt)
    print(f"二次LLM查询: {'命中 ✅' if l4_result2 else '未命中 ❌'}")
    print(f"缓存响应: {l4_result2['response'][:80]}...")
    print(f"内容类型: {l4_result2['content_type']}")
    print(f"预计TTL: {'7天' if l4_result2['content_type'] == 'factual' else '自定义'}")

    # 5. 统一接口演示
    print("\n--- 统一缓存接口 ---")
    def compute_expensive():
        time.sleep(0.01)  # 模拟耗时计算
        return {"result": "expensive_computation_result"}

    result1 = cache.get_or_compute(
        "测试统一接口", np.random.randn(8), 5, compute_expensive,
        level=CacheLevel.L1_EXACT_QUERY
    )
    print(f"首次计算: {result1}")

    result2 = cache.get_or_compute(
        "测试统一接口", np.random.randn(8), 5, compute_expensive,
        level=CacheLevel.L1_EXACT_QUERY
    )
    print(f"缓存命中: {result2}")

    # 6. 统计报告
    print("\n--- 缓存统计报告 ---")
    report = cache.get_report()
    for level_name, stats in report.items():
        if isinstance(stats, dict) and "hits" in stats:
            print(f"  {level_name}: 命中率={stats['hit_rate']}, 命中={stats['hits']}, 未命中={stats['misses']}")
    print(f"  总体命中率: {report['_meta']['overall_hit_rate']}")

    # 7. 缓存预热演示
    print("\n--- 缓存预热 ---")
    cache.warm_l1([
        ("常见问题1", {"answer": "答案1"}),
        ("常见问题2", {"answer": "答案2"}),
        ("常见问题3", {"answer": "答案3"}),
    ])
    cache.warm_l4([
        ("什么是RAG？", "RAG是检索增强生成系统...", ContentType.FACTUAL),
        ("今天天气怎么样？", "今天晴，25度...", ContentType.REAL_TIME),
    ])

    print("\n" + "=" * 60)
    print("四级缓存系统演示完成！")
    print("=" * 60)
