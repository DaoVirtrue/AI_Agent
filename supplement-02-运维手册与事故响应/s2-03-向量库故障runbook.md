# S2-03：向量库故障 Runbook

> **故障类型**：向量数据库不可用、检索返回空/异常结果
> **影响范围**：所有 RAG 检索功能

---

## 一、检测方法

### 自动检测

```yaml
监控指标:
  - vector_db_connection_status = 0 → P0 告警
  - vector_search_latency_p99 > 3s → P1 告警
  - vector_search_empty_rate > 20% → P2 告警（可能索引损坏）
  - vector_db_disk_usage > 90% → P2 告警
```

### 手动检测

```bash
# 测试向量库连通性
curl http://localhost:6333/health

# Milvus 健康检查
curl http://localhost:9091/healthz

# Qdrant 健康检查
curl http://localhost:6333/healthz
```

---

## 二、处理步骤

### Step 1：确认故障范围（2分钟）

```bash
# 检查向量库容器状态
docker ps | grep milvus
# 或
kubectl get pods -n vector-db

# 检查向量库日志
docker logs milvus-standalone --tail 50
```

### Step 2：启用断路器（3分钟）

```python
# 在检索代码中启用断路器
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,    # 连续失败5次断开
    "timeout": 5,              # 单次请求超时5秒
    "recovery_timeout": 30,    # 30秒后尝试恢复
}
```

### Step 3：切到缓存回退（5分钟）

```yaml
缓存回退策略（优先级从高到低）：
1. Redis 热点缓存（最近1万条热门查询的结果缓存）
2. ES 全文检索回退（BM25 关键词匹配）
3. 静态回答回退（返回预定义的 FAQ 回答）
4. 优雅降级（提示：系统繁忙，请稍后再试）
```

```python
class FallbackRetriever:
    """带缓存回退的检索器。"""

    def __init__(self, vector_store, text_index, cache):
        self.vector_store = vector_store
        self.text_index = text_index
        self.cache = cache
        self.vector_db_healthy = True

    def retrieve(self, query: str, top_k: int = 5):
        # 层1：热点缓存
        cached = self.cache.get(f"search:{query}")
        if cached:
            return cached

        # 层2：向量检索（主）
        if self.vector_db_healthy:
            try:
                results = self.vector_store.search(query, top_k)
                self.cache.set(f"search:{query}", results, ttl=3600)
                return results
            except Exception:
                self.vector_db_healthy = False

        # 层3：全文检索回退
        try:
            results = self.text_index.search(query, top_k)
            return results
        except Exception:
            pass

        # 层4：空结果（触发上层 LLM 的"无法回答"逻辑）
        return []
```

### Step 4：索引恢复（30分钟+）

```bash
# 方案A：从备份恢复（最快）
# 1. 停止写入
# 2. 恢复向量索引备份
milvus-backup restore --collection knowledge_base --backup 20260611

# 方案B：从原始文档重建（最彻底）
# 1. 清空损坏的索引
# 2. 重新加载所有文档并嵌入
python scripts/rebuild_vector_index.py --collection knowledge_base

# 方案C：增量修复
# 1. 只重建最近修改的文档
python scripts/rebuild_vector_index.py --collection knowledge_base --modified-since "2026-06-10"
```

### Step 5：验证恢复

```python
# 验证脚本
GOLDEN_QUERIES = [
    ("公司的年假政策是什么？", ["年假", "天", "申请"]),
    ("如何部署RAG系统？", ["部署", "Docker", "Kubernetes"]),
]

def validate_vector_db():
    for query, expected_keywords in GOLDEN_QUERIES:
        results = retriever.retrieve(query, top_k=5)
        # 检查是否有结果
        if not results:
            return False, f"查询'{query}'返回空"
        # 检查内容是否合理
        combined = " ".join(r["content"] for r in results)
        for kw in expected_keywords:
            if kw not in combined:
                return False, f"查询'{query}'缺少关键词'{kw}'"
    return True, "验证通过"
```

---

## 三、常见故障及快速修复

| 症状 | 可能原因 | 快速修复 |
|------|---------|----------|
| 检索返回空 | 索引损坏/Collection 被删 | 从备份恢复或重建索引 |
| 检索极慢(>5s) | 磁盘I/O瓶颈/索引过大 | 增大内存、优化 HNSW 参数 |
| 搜索结果不相关 | 嵌入模型版本不一致 | 统一嵌入模型版本并重建 |
| 连接被拒绝 | 服务未启动/端口被占用 | 重启向量库服务 |
| 磁盘满 | 索引膨胀 | 清理过期数据、扩容磁盘 |
| 写入失败 | 内存不足 | 增加内存或分片 |
