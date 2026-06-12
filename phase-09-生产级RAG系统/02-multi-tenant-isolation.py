#!/usr/bin/env python3
"""
生产级多租户隔离系统
Production Multi-Tenant Isolation System

三层隔离架构：
  第1层: 独立向量集合 (Separate Collections per Tenant)
  第2层: 元数据级别tenant_id过滤 (Metadata-Level Filtering)
  第3层: API Key认证与授权 (API Key Authentication)

配额管理: FREE / PRO / ENTERPRISE 三级
"""

import hashlib
import secrets
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class TenantTier(Enum):
    """租户等级"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuota:
    """租户配额"""
    max_documents: int
    max_chunks: int
    max_api_calls_per_day: int
    max_collections: int
    vector_dimension: int
    rate_limit_rps: float  # 每秒请求数

    @staticmethod
    def for_tier(tier: TenantTier) -> "TenantQuota":
        """根据等级获取配额"""
        quotas = {
            TenantTier.FREE: TenantQuota(
                max_documents=100,
                max_chunks=10000,
                max_api_calls_per_day=1000,
                max_collections=2,
                vector_dimension=384,
                rate_limit_rps=5.0,
            ),
            TenantTier.PRO: TenantQuota(
                max_documents=10000,
                max_chunks=100000,
                max_api_calls_per_day=100000,
                max_collections=10,
                vector_dimension=768,
                rate_limit_rps=50.0,
            ),
            TenantTier.ENTERPRISE: TenantQuota(
                max_documents=1000000,
                max_chunks=10000000,
                max_api_calls_per_day=10000000,
                max_collections=50,
                vector_dimension=1536,
                rate_limit_rps=500.0,
            ),
        }
        return quotas[tier]


@dataclass
class TenantInfo:
    """租户信息"""
    tenant_id: str
    name: str
    tier: TenantTier
    api_key_hash: str
    created_at: float
    quota: TenantQuota
    active: bool = True
    metadata: dict = field(default_factory=dict)
    usage: dict = field(default_factory=lambda: {
        "documents": 0,
        "chunks": 0,
        "api_calls_today": 0,
        "api_calls_total": 0,
        "last_reset_day": "",
    })


@dataclass
class APIKey:
    """API密钥"""
    key_id: str
    tenant_id: str
    key_prefix: str
    key_hash: str
    created_at: float
    expires_at: Optional[float]
    scopes: list[str]
    active: bool = True


# ============================================================
# 第1层: 独立向量集合隔离 (Collection-Level Isolation)
# ============================================================

class VectorCollectionManager:
    """向量集合管理器 - 每个租户独立集合"""

    def __init__(self):
        self._collections: dict[str, Any] = {}  # tenant_id -> collection
        self._lock = threading.RLock()

    def create_collection(self, tenant_id: str, dimension: int) -> str:
        """为租户创建独立向量集合
        Args:
            tenant_id: 租户ID
            dimension: 向量维度
        Returns:
            collection_name: 集合名称
        """
        with self._lock:
            collection_name = f"tenant_{tenant_id}_vectors"

            if collection_name in self._collections:
                return collection_name

            try:
                import chromadb
                client = chromadb.Client()
                collection = client.create_collection(
                    name=collection_name,
                    metadata={
                        "tenant_id": tenant_id,
                        "dimension": str(dimension),
                        "hnsw:space": "cosine",
                    },
                )
                self._collections[collection_name] = collection
                return collection_name
            except ImportError:
                # 降级: 使用内存字典模拟
                self._collections[collection_name] = _MockCollection(collection_name)
                return collection_name

    def get_collection(self, tenant_id: str) -> Optional[Any]:
        """获取租户的向量集合"""
        collection_name = f"tenant_{tenant_id}_vectors"
        return self._collections.get(collection_name)

    def add_vectors(self, tenant_id: str, ids: list[str], vectors: list[list[float]],
                    documents: list[str], metadata: list[dict]) -> bool:
        """向租户集合添加向量"""
        coll = self.get_collection(tenant_id)
        if coll is None:
            raise ValueError(f"租户 {tenant_id} 的集合不存在")

        # 注入tenant_id到元数据（第2层隔离的基础）
        for meta in metadata:
            meta["tenant_id"] = tenant_id

        try:
            coll.add(
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadata,
            )
            return True
        except Exception as e:
            print(f"添加向量失败: {e}")
            return False

    def search(self, tenant_id: str, query_vector: list[float], k: int = 10,
               filters: Optional[dict] = None) -> list[dict]:
        """在租户集合内搜索
        第1层隔离: 仅搜索该租户的集合
        第2层隔离: 通过tenant_id过滤确保数据隔离
        """
        coll = self.get_collection(tenant_id)
        if coll is None:
            return []

        # 始终附加tenant_id过滤（第2层）
        if filters is None:
            filters = {}
        filters["tenant_id"] = tenant_id

        try:
            results = coll.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=filters,
            )
            return self._format_results(results)
        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def delete_collection(self, tenant_id: str) -> bool:
        """删除租户集合（注销时）"""
        with self._lock:
            collection_name = f"tenant_{tenant_id}_vectors"
            if collection_name in self._collections:
                try:
                    coll = self._collections[collection_name]
                    if hasattr(coll, "delete_collection"):
                        coll.delete_collection()
                    del self._collections[collection_name]
                except Exception:
                    del self._collections[collection_name]
                return True
        return False

    @staticmethod
    def _format_results(results: Any) -> list[dict]:
        """格式化搜索结果"""
        formatted = []
        if hasattr(results, "ids"):
            for i in range(len(results.ids[0])):
                formatted.append({
                    "id": results.ids[0][i],
                    "document": results.documents[0][i] if results.documents else "",
                    "metadata": results.metadatas[0][i] if results.metadatas else {},
                    "distance": results.distances[0][i] if results.distances else 0.0,
                })
        return formatted


class _MockCollection:
    """模拟向量集合（用于chromadb不可用时）"""

    def __init__(self, name: str):
        self.name = name
        self._data: dict[str, dict] = {}
        self._vectors: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict]):
        with self._lock:
            for i, id_ in enumerate(ids):
                self._data[id_] = {
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
                self._vectors[id_] = embeddings[i] if i < len(embeddings) else []

    def query(self, query_embeddings: list[list[float]], n_results: int = 10,
              where: Optional[dict] = None) -> Any:
        """模拟查询"""
        import numpy as np

        query_vec = np.array(query_embeddings[0])
        results = []

        for id_, vec in self._vectors.items():
            if where:
                meta = self._data[id_]["metadata"]
                if not all(meta.get(k) == v for k, v in where.items()):
                    continue
            vec_np = np.array(vec)
            if vec_np.size == 0 or query_vec.size == 0:
                continue
            cosine = np.dot(vec_np, query_vec) / (np.linalg.norm(vec_np) * np.linalg.norm(query_vec) + 1e-8)
            distance = 1.0 - cosine
            results.append((distance, id_))

        results.sort(key=lambda x: x[0])
        top_k = results[:n_results]

        class MockResult:
            pass

        mock = MockResult()
        mock.ids = [[r[1] for r in top_k]]
        mock.documents = [[self._data[r[1]]["document"] for r in top_k]]
        mock.metadatas = [[self._data[r[1]]["metadata"] for r in top_k]]
        mock.distances = [[r[0] for r in top_k]]
        return mock


# ============================================================
# 第2层: 元数据级别过滤 (Metadata-Level Filtering)
# ============================================================

class MetadataIsolationFilter:
    """元数据级别租户隔离过滤器"""

    @staticmethod
    def build_filter(tenant_id: str, additional: Optional[dict] = None) -> dict:
        """构造包含租户过滤的查询条件
        这层确保即使集合级别隔离失效，元数据级别也能防止数据泄漏
        """
        base_filter = {"tenant_id": tenant_id}
        if additional:
            # 使用AND逻辑合并过滤条件
            base_filter.update(additional)
        return base_filter

    @staticmethod
    def validate_chunk_ownership(chunk: dict, tenant_id: str) -> bool:
        """验证块是否属于指定租户"""
        return chunk.get("metadata", {}).get("tenant_id") == tenant_id

    @staticmethod
    def sanitize_metadata(metadata: dict) -> dict:
        """清理元数据，防止租户间数据泄漏"""
        # 移除可能泄漏其他租户信息的字段
        forbidden_keys = {"cross_tenant_references", "shared_keys", "global_secrets"}
        return {k: v for k, v in metadata.items() if k not in forbidden_keys}


# ============================================================
# 第3层: API Key认证 (API Key Authentication)
# ============================================================

class APIKeyManager:
    """API密钥管理器"""

    def __init__(self):
        self._keys: dict[str, APIKey] = {}  # key_hash -> APIKey
        self._tenant_keys: dict[str, list[str]] = {}  # tenant_id -> [key_hashes]
        self._lock = threading.RLock()

    def generate_key(self, tenant_id: str, scopes: Optional[list[str]] = None,
                     ttl_days: int = 365) -> tuple[str, APIKey]:
        """生成新的API密钥
        Returns:
            (raw_key, api_key_info) - 原始密钥(仅返回一次)和密钥信息
        """
        raw_key = f"rag_{secrets.token_hex(16)}_{secrets.token_hex(8)}"
        key_prefix = raw_key[:12]
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = APIKey(
            key_id=secrets.token_hex(8),
            tenant_id=tenant_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
            created_at=time.time(),
            expires_at=time.time() + ttl_days * 86400 if ttl_days else None,
            scopes=scopes or ["read", "write"],
        )

        with self._lock:
            self._keys[key_hash] = api_key
            if tenant_id not in self._tenant_keys:
                self._tenant_keys[tenant_id] = []
            self._tenant_keys[tenant_id].append(key_hash)

        return raw_key, api_key

    def authenticate(self, raw_key: str) -> Optional[TenantInfo]:
        """认证API密钥并返回租户信息
        第3层隔离: 只有持有有效API Key才能访问租户数据
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with self._lock:
            api_key = self._keys.get(key_hash)
            if not api_key:
                return None
            if not api_key.active:
                return None
            if api_key.expires_at and time.time() > api_key.expires_at:
                return None

        return None  # 需要外部TenantManager提供完整信息

    def revoke_key(self, key_hash: str) -> bool:
        """撤销API密钥"""
        with self._lock:
            if key_hash in self._keys:
                self._keys[key_hash].active = False
                return True
        return False

    def rotate_key(self, old_key_hash: str, tenant_id: str) -> Optional[tuple[str, APIKey]]:
        """轮换API密钥（撤销旧密钥并生成新密钥）"""
        if self.revoke_key(old_key_hash):
            return self.generate_key(tenant_id)
        return None


# ============================================================
# 租户管理器 (Tenant Manager)
# ============================================================

class TenantManager:
    """租户管理器 - 整合三层隔离和配额管理"""

    def __init__(self):
        self._tenants: dict[str, TenantInfo] = {}
        self._api_key_manager = APIKeyManager()
        self._collection_manager = VectorCollectionManager()
        self._isolation_filter = MetadataIsolationFilter()
        self._lock = threading.RLock()

    # ========== 租户生命周期管理 ==========

    def provision_tenant(self, name: str, tier: TenantTier,
                         metadata: Optional[dict] = None) -> TenantInfo:
        """创建（开通）租户"""
        tenant_id = f"tenant_{secrets.token_hex(8)}_{int(time.time())}"
        quota = TenantQuota.for_tier(tier)

        tenant = TenantInfo(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            api_key_hash="",  # 稍后生成
            created_at=time.time(),
            quota=quota,
            metadata=metadata or {},
        )

        with self._lock:
            self._tenants[tenant_id] = tenant

        # 创建第1层隔离: 独立向量集合
        self._collection_manager.create_collection(tenant_id, quota.vector_dimension)

        # 生成第3层隔离: API密钥
        raw_key, api_key = self._api_key_manager.generate_key(tenant_id)
        tenant.api_key_hash = api_key.key_hash
        self._tenants[tenant_id] = tenant

        print(f"[Provision] 租户已创建: {name} (ID: {tenant_id}, Tier: {tier.value})")
        print(f"[Provision] API Key: {raw_key} (请安全保存，仅显示一次)")
        print(f"[Provision] 配额: {quota.max_documents} 文档, {quota.max_chunks} 块, {quota.max_api_calls_per_day} API/天")

        return tenant

    def deprovision_tenant(self, tenant_id: str) -> bool:
        """注销租户"""
        with self._lock:
            if tenant_id not in self._tenants:
                return False

            tenant = self._tenants[tenant_id]
            tenant.active = False

            # 撤销所有API密钥
            for key_hash in self._api_key_manager._tenant_keys.get(tenant_id, []):
                self._api_key_manager.revoke_key(key_hash)

            # 删除向量集合
            self._collection_manager.delete_collection(tenant_id)

            # 保留租户记录（软删除）但不删除
            print(f"[Deprovision] 租户已注销: {tenant.name} (ID: {tenant_id})")
            return True

    def get_tenant(self, tenant_id: str) -> Optional[TenantInfo]:
        """获取租户信息"""
        return self._tenants.get(tenant_id)

    def get_tenant_by_api_key(self, raw_key: str) -> Optional[TenantInfo]:
        """通过API Key获取租户（第3层认证）"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with self._lock:
            api_key = self._api_key_manager._keys.get(key_hash)
            if not api_key or not api_key.active:
                return None
            if api_key.expires_at and time.time() > api_key.expires_at:
                return None

            return self._tenants.get(api_key.tenant_id)

    # ========== 配额管理 ==========

    def check_quota(self, tenant_id: str, resource: str, amount: int = 1) -> tuple[bool, str]:
        """检查租户配额
        Returns:
            (allowed, message) - 是否允许及消息
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant or not tenant.active:
            return False, "租户不存在或已停用"

        quota = tenant.quota
        usage = tenant.usage

        # 重置每日计数
        today = time.strftime("%Y-%m-%d")
        if usage["last_reset_day"] != today:
            usage["api_calls_today"] = 0
            usage["last_reset_day"] = today

        if resource == "documents":
            if usage["documents"] + amount > quota.max_documents:
                return False, f"文档配额已满 ({usage['documents']}/{quota.max_documents})"
        elif resource == "chunks":
            if usage["chunks"] + amount > quota.max_chunks:
                return False, f"块配额已满 ({usage['chunks']}/{quota.max_chunks})"
        elif resource == "api_calls":
            if usage["api_calls_today"] + amount > quota.max_api_calls_per_day:
                return False, f"今日API调用配额已满 ({usage['api_calls_today']}/{quota.max_api_calls_per_day})"
        else:
            return False, f"未知资源类型: {resource}"

        return True, "OK"

    def consume_quota(self, tenant_id: str, resource: str, amount: int = 1) -> bool:
        """消耗配额"""
        allowed, msg = self.check_quota(tenant_id, resource, amount)
        if not allowed:
            print(f"[Quota] 配额不足: {msg}")
            return False

        tenant = self.get_tenant(tenant_id)
        if resource == "documents":
            tenant.usage["documents"] += amount
        elif resource == "chunks":
            tenant.usage["chunks"] += amount
        elif resource == "api_calls":
            tenant.usage["api_calls_today"] += amount
            tenant.usage["api_calls_total"] += amount

        return True

    def get_quota_report(self, tenant_id: str) -> dict:
        """获取配额使用报告"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return {}

        # 重置每日计数
        today = time.strftime("%Y-%m-%d")
        if tenant.usage["last_reset_day"] != today:
            tenant.usage["api_calls_today"] = 0
            tenant.usage["last_reset_day"] = today

        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "tier": tenant.tier.value,
            "active": tenant.active,
            "quota": {
                "max_documents": tenant.quota.max_documents,
                "max_chunks": tenant.quota.max_chunks,
                "max_api_calls_per_day": tenant.quota.max_api_calls_per_day,
                "rate_limit_rps": tenant.quota.rate_limit_rps,
            },
            "usage": {
                "documents": tenant.usage["documents"],
                "document_pct": f"{tenant.usage['documents'] / max(tenant.quota.max_documents, 1) * 100:.1f}%",
                "chunks": tenant.usage["chunks"],
                "chunk_pct": f"{tenant.usage['chunks'] / max(tenant.quota.max_chunks, 1) * 100:.1f}%",
                "api_calls_today": tenant.usage["api_calls_today"],
                "api_calls_pct": f"{tenant.usage['api_calls_today'] / max(tenant.quota.max_api_calls_per_day, 1) * 100:.1f}%",
                "api_calls_total": tenant.usage["api_calls_total"],
            },
        }

    # ========== 安全搜索（整合三层隔离） ==========

    def search(self, raw_api_key: str, query_vector: list[float],
               k: int = 10, additional_filters: Optional[dict] = None) -> tuple[bool, list[dict], str]:
        """安全搜索 - 整合三层隔离
        Args:
            raw_api_key: 用户提供的API密钥
            query_vector: 查询向量
            k: 返回结果数量
            additional_filters: 额外过滤条件
        Returns:
            (success, results, message)
        """
        # 第3层: API Key认证
        tenant = self.get_tenant_by_api_key(raw_api_key)
        if not tenant:
            return False, [], "API Key认证失败"

        # 配额检查
        allowed, msg = self.check_quota(tenant.tenant_id, "api_calls")
        if not allowed:
            return False, [], msg

        # 第2层: 构造带租户过滤的条件
        filters = self._isolation_filter.build_filter(
            tenant.tenant_id, additional_filters
        )

        # 第1层: 在租户专属集合中搜索
        results = self._collection_manager.search(
            tenant.tenant_id, query_vector, k, filters
        )

        # 消耗配额
        self.consume_quota(tenant.tenant_id, "api_calls")

        return True, results, "OK"

    # ========== 批量操作 ==========

    def list_tenants(self, active_only: bool = True) -> list[dict]:
        """列出所有租户"""
        tenants = []
        for tid, tenant in self._tenants.items():
            if active_only and not tenant.active:
                continue
            tenants.append({
                "tenant_id": tid,
                "name": tenant.name,
                "tier": tenant.tier.value,
                "active": tenant.active,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tenant.created_at)),
            })
        return tenants

    def upgrade_tenant_tier(self, tenant_id: str, new_tier: TenantTier) -> bool:
        """升级租户等级"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        tenant.tier = new_tier
        tenant.quota = TenantQuota.for_tier(new_tier)
        print(f"[Upgrade] 租户 {tenant.name} 已升级至 {new_tier.value}")
        return True


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("生产级多租户隔离系统 - 演示运行")
    print("=" * 60)

    # 初始化租户管理器
    manager = TenantManager()

    # 1. 创建不同等级的租户
    print("\n--- 租户开通 (Provisioning) ---")
    free_tenant = manager.provision_tenant("个人开发者", TenantTier.FREE)
    pro_tenant = manager.provision_tenant("创业公司", TenantTier.PRO)
    enterprise_tenant = manager.provision_tenant("大型企业", TenantTier.ENTERPRISE)

    print(f"\n当前租户数: {len(manager.list_tenants())}")

    # 2. 为每个租户生成API Key并查看配额
    print("\n--- 配额报告 ---")
    for tenant in [free_tenant, pro_tenant, enterprise_tenant]:
        report = manager.get_quota_report(tenant.tenant_id)
        print(f"\n{report['name']} ({report['tier']}):")
        print(f"  文档: {report['usage']['document_pct']}")
        print(f"  块:   {report['usage']['chunk_pct']}")
        print(f"  API:  {report['usage']['api_calls_pct']}")
        print(f"  限流: {report['quota']['rate_limit_rps']} req/s")

    # 3. 测试配额消耗和限制
    print("\n--- 配额消耗测试 ---")
    print("消费配额...")
    for i in range(5):
        manager.consume_quota(free_tenant.tenant_id, "api_calls")
    report = manager.get_quota_report(free_tenant.tenant_id)
    print(f"已消耗API次数: {report['usage']['api_calls_today']}")

    # 4. 测试第3层API Key认证
    print("\n--- API Key认证测试 ---")
    # 生成单独密钥用于测试
    raw_key, api_key_info = manager._api_key_manager.generate_key(free_tenant.tenant_id)
    print(f"生成的密钥: {raw_key[:20]}...")

    # 认证测试
    authenticated_tenant = manager.get_tenant_by_api_key(raw_key)
    if authenticated_tenant:
        print(f"✅ 认证成功: {authenticated_tenant.name}")
    else:
        print("❌ 认证失败")

    # 错误的密钥
    fake_key = "rag_fakekey_12345678"
    fake_auth = manager.get_tenant_by_api_key(fake_key)
    print(f"{'❌' if fake_auth is None else '⚠️'} 伪造密钥认证: {'失败(正确)' if fake_auth is None else '成功(异常)'}")

    # 5. 模拟三层隔离搜索
    print("\n--- 三层隔离搜索 ---")
    # 为每个租户添加一些数据
    import numpy as np

    for tenant in [free_tenant, pro_tenant, enterprise_tenant]:
        vectors = np.random.randn(3, tenant.quota.vector_dimension).tolist()
        manager._collection_manager.add_vectors(
            tenant.tenant_id,
            ids=[f"{tenant.tenant_id}_doc_{i}" for i in range(3)],
            vectors=vectors,
            documents=[f"{tenant.name} 文档 {i}" for i in range(3)],
            metadata=[{"doc_idx": i, "tenant_id": tenant.tenant_id} for i in range(3)],
        )

    # 搜索（使用free租户的密钥，只能搜到free的数据）
    query_vec = np.random.randn(free_tenant.quota.vector_dimension).tolist()
    success, results, msg = manager.search(raw_key, query_vec, k=5)

    if success:
        print(f"✅ 搜索成功，返回 {len(results)} 条结果")
        for r in results:
            print(f"  - {r['document']} (tenant: {r['metadata'].get('tenant_id', 'N/A')[:12]}...)")
        # 验证没有数据泄漏
        all_from_same_tenant = all(
            r["metadata"].get("tenant_id") == free_tenant.tenant_id
            for r in results
        )
        print(f"数据隔离: {'✅ 正确' if all_from_same_tenant else '❌ 存在泄漏'}")
    else:
        print(f"❌ 搜索失败: {msg}")

    # 6. 租户升级
    print("\n--- 租户升级 ---")
    manager.upgrade_tenant_tier(free_tenant.tenant_id, TenantTier.PRO)
    upgraded_report = manager.get_quota_report(free_tenant.tenant_id)
    print(f"升级后配额: {upgraded_report['quota']['max_documents']} 文档, {upgraded_report['quota']['max_chunks']} 块")

    # 7. 租户注销
    print("\n--- 租户注销 ---")
    manager.deprovision_tenant(free_tenant.tenant_id)
    report_after = manager.get_quota_report(free_tenant.tenant_id)
    print(f"租户状态: {'活跃' if report_after.get('active') else '已注销'}")

    # 8. 最终统计
    print("\n--- 最终统计 ---")
    all_tenants = manager.list_tenants(active_only=False)
    print(f"总租户数: {len(all_tenants)}")
    for t in all_tenants:
        print(f"  [{t['tier']}] {t['name']} - {'活跃' if t['active'] else '已注销'}")

    print("\n" + "=" * 60)
    print("多租户隔离演示完成！")
    print("=" * 60)
