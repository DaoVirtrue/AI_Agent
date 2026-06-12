# Phase 09 检测练习 - 参考答案

---

## 练习1: 数据管道完整性验证

### 解答

```python
# 补充的阶段实现要点

# 1. Web解析器
class WebParser(BaseParser):
    def parse(self, raw: RawDocument) -> ParsedDocument:
        import requests
        resp = requests.get(raw.source, timeout=30)
        # 提取正文内容（去掉script/style标签）
        text = self._extract_text(resp.text)
        return ParsedDocument(
            doc_id=self._generate_doc_id(raw.source, text),
            source=raw.source, source_type=DocumentSource.WEB,
            text=text, structured={"url": raw.source},
            metadata={"url": raw.source},
            raw_hash=hashlib.md5(text.encode()).hexdigest(),
        )

# 2. 补充去重阶段的语义去重
# 超出简单MD5匹配，加入sentence-transformers语义相似度

# 3. 元数据增强（自动+LLM）
class MetadataEnricher:
    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        # 自动元数据: 字数、语言检测、关键词密度
        # LLM增强: 摘要生成、实体标签提取
        pass

# 4. BM25索引（bm25s库）
class BM25IndexBuilder:
    def build(self, chunks: list[Chunk]) -> dict:
        texts = [c.text for c in chunks]
        corpus_tokens = [self._tokenize(t) for t in texts]
        self._index = bm25s.BM25(corpus=corpus_tokens)
        self._index.index(corpus_tokens)

# 5. 进度回调
class ProgressCallback:
    def __init__(self):
        self.callbacks = []
    def register(self, fn):
        self.callbacks.append(fn)
    def notify(self, phase, progress, message):
        for fn in self.callbacks:
            fn(phase, progress, message)

# 6. 错误处理
class ErrorHandler:
    def handle(self, phase, doc_id, error, recoverable=True):
        if not recoverable:
            raise PipelineError(phase, str(error))
        # 记录错误并继续
        self.stats.errors.append({...})

# 7. 质量门控检查（5条）
quality_checks = [
    ("chunks_not_empty", len(chunks) > 0),
    ("hnsw_index_built", hnsw_info["status"] == "success"),
    ("bm25_index_built", bm25_info["status"] == "success"),
    ("vector_coverage", vector_coverage >= 0.8),
    ("text_quality", empty_chunk_ratio < 0.05),
]
```

---

## 练习2: 多租户安全隔离

### 解答

```python
# 三级配额方案
QUOTA_PLANS = {
    "FREE": {
        "max_documents": 100,
        "max_chunks": 10000,
        "max_api_calls_per_day": 1000,
        "rate_limit_rps": 5,
        "vector_dimension": 384,
    },
    "PRO": {
        "max_documents": 10000,
        "max_chunks": 100000,
        "max_api_calls_per_day": 100000,
        "rate_limit_rps": 50,
        "vector_dimension": 768,
    },
    "ENTERPRISE": {
        "max_documents": 1000000,
        "max_chunks": 10000000,
        "max_api_calls_per_day": 10000000,
        "rate_limit_rps": 500,
        "vector_dimension": 1536,
    },
}

# 穿透测试脚本
def penetration_test(tenant_manager):
    vulnerabilities = []
    
    # 测试1: 伪造API Key
    fake_key = "rag_fake_0000000000000000"
    tenant = tenant_manager.get_tenant_by_api_key(fake_key)
    if tenant is not None:
        vulnerabilities.append("VULN-001: 伪造API Key可获取访问权限")
    
    # 测试2: 空tenant_id过滤
    result = search_without_tenant_filter()
    if len(result) > 0 and any(r.get("tenant_id") != current_tenant for r in result):
        vulnerabilities.append("VULN-002: 无tenant_id过滤可跨租户访问")
    
    # 测试3: 缓存跨租户
    cache_key = f"l3:some_query_hash"
    cached = cache.get(cache_key)
    if cached and cached.get("tenant_id") != current_tenant:
        vulnerabilities.append("VULN-003: 缓存键未按租户隔离")
    
    return vulnerabilities

# 租户注销流程
def deprovision_tenant(tenant_id):
    """软删除: 30天内可恢复"""
    tenant.active = False
    tenant.deactivated_at = time.time()
    
    # 1. 撤销所有API密钥
    for key in get_tenant_keys(tenant_id):
        key.active = False
    
    # 2. 冻结向量集合（保留30天）
    freeze_collection(tenant_id)
    
    # 3. 保留数据30天
    schedule_cleanup(tenant_id, after_days=30)
```

---

## 练习3: 缓存命中率优化

### 解答

```python
# 缓存命中率模拟器
def simulate_cache_hits(num_queries=10000, zipf_alpha=1.5):
    """
    模拟幂律分布的查询
    """
    import numpy as np
    
    # Zipf分布: 少数查询占多数请求
    query_popularity = np.random.zipf(zipf_alpha, num_queries)
    
    l1_hits = 0   # 精确匹配
    l2_hits = 0   # 语义相似
    l3_hits = 0   # 检索结果
    l4_hits = 0   # LLM响应
    misses = 0
    
    cache = RAGCacheManager()
    
    for i, qid in enumerate(query_popularity[:1000]):
        query = f"query_{qid}"
        query_emb = get_embedding(query)
        
        # L1: 精确匹配
        if cache.l1_get(query):
            l1_hits += 1
            continue
        
        # L2: 语义相似
        if cache.l2_get(query_emb, threshold=0.92):
            l2_hits += 1
            continue
        
        # 模拟检索执行
        chunks = retrieve(query)
        cache.l3_set(query, 5, chunks)
        
        # L3: 检索结果
        if query_popularity[i] > 5:  # 中等热度
            l3_hits += 1
            continue
        
        misses += 1
    
    total = l1_hits + l2_hits + l3_hits + l4_hits + misses
    print(f"L1命中率: {l1_hits/total:.1%} (精确查询)")
    print(f"L2命中率: {l2_hits/total:.1%} (语义相似)")
    print(f"L3命中率: {l3_hits/total:.1%} (检索结果)")
    print(f"L4命中率: {l4_hits/total:.1%} (LLM响应)")
    print(f"总命中率: {(total-misses)/total:.1%}")
    # 预期: 总命中率 40-60%

# 预热脚本
def warm_cache(cache_manager, query_log):
    """预加载top-100高频查询"""
    top_100 = sorted(query_log, key=lambda x: x["count"], reverse=True)[:100]
    
    for query_info in top_100:
        query = query_info["query"]
        answer = query_info.get("cached_answer")
        chunks = query_info.get("cached_chunks", [])
        
        cache_manager.l1_set(query, answer)  # L1预热
        if chunks:
            cache_manager.l3_set(query, 5, chunks)  # L3预热
    
    print(f"预热完成: {len(top_100)} 条")
```

---

## 练习4: 熔断器配置调优

### 解答

```python
# 问题分析
"""
问题: failure_threshold=3, failure_window=30s, timeout=60s
- 阈值过低: 3次失败在30s内很常见（正常波动）
- 窗口过短: 未考虑突发流量
- 冷却过长: 60s对LLM API来说太久

优化配置:
"""

OPTIMIZED_CONFIGS = {
    "llm_api": CircuitConfig(
        failure_threshold=10,        # 更高阈值（LLM偶尔超时正常）
        failure_window=120,          # 更长窗口
        timeout=15,                  # 较短冷却
        half_open_max_requests=3,    # 允许3个探测请求
        ignore_errors=["429", "rate_limit"],
    ),
    "vector_db": CircuitConfig(
        failure_threshold=3,         # 向量DB应稳定
        failure_window=30,
        timeout=10,
        half_open_max_requests=2,
    ),
    "reranker": CircuitConfig(
        failure_threshold=5,
        failure_window=60,
        timeout=30,
        half_open_max_requests=2,
    ),
}

# 错误分类
def classify_error(error: Exception) -> ErrorCategory:
    error_str = str(error).lower()
    
    # 429速率限制 → 不触发熔断
    if "429" in error_str or "rate limit" in error_str:
        return ErrorCategory.RETRYABLE_RATE_LIMIT
    
    # 超时 → 可能触发（组合判断）
    if "timeout" in error_str:
        return ErrorCategory.RETRYABLE_TIMEOUT
    
    # 500/503 → 应触发
    if any(c in error_str for c in ["500", "503", "502"]):
        return ErrorCategory.CIRCUIT_BREAKER_WORTHY
    
    # 400/401 → 不触发
    if any(c in error_str for c in ["400", "401", "403"]):
        return ErrorCategory.NON_RETRYABLE
    
    return ErrorCategory.CIRCUIT_BREAKER_WORTHY

# 指数退避
def calculate_backoff(open_count: int, base_timeout: float) -> float:
    backoff = base_timeout * (2 ** min(open_count, 4))
    return min(backoff, 300)  # 最大5分钟
```

---

## 练习5: 灾难恢复演练

### 解答

```python
# DR演练完整脚本
def disaster_recovery_drill():
    # 步骤1: 创建模拟数据
    chunks = [generate_chunk(i) for i in range(1000)]
    vectors = np.random.randn(1000, 384).tolist()
    ids = [f"vec_{i:04d}" for i in range(1000)]
    
    # 步骤2: 全量备份
    backup_mgr = BackupManager("./dr_test_backups")
    full = backup_mgr.full_backup(chunks, vectors, ids)
    print(f"全量备份: {full.backup_id} ({full.total_chunks} 块)")
    
    # 步骤3: 3次增量备份
    for i in range(3):
        delta_chunks = [generate_chunk(1000 + i*100 + j) for j in range(100)]
        delta_vectors = np.random.randn(100, 384).tolist()
        delta_ids = [f"vec_inc_{i}_{j:03d}" for j in range(100)]
        
        inc = backup_mgr.incremental_backup(
            delta_chunks, delta_vectors, delta_ids,
            parent_backup_id=(full.backup_id if i == 0 else inc.backup_id)
        )
        print(f"增量{i+1}: {inc.backup_id}")
    
    # 步骤4: 模拟灾难（删除索引）
    print("模拟灾难: 索引已删除!")
    
    # 步骤5: 从最新备份恢复
    t0 = time.time()
    result = backup_mgr.restore(inc.backup_id, verify=True)
    rto = time.time() - t0
    
    # 步骤6: 验证
    assert result.chunks_restored == 1300  # 1000 + 3*100
    assert result.vectors_restored == 1300
    assert result.verification_passed
    assert result.checksum_match
    assert rto < 3600  # RTO target: 1小时
    
    print(f"DR演练完成: RTO={rto:.1f}s, 数据完整性=100%")
    print(f"RTO达标: {'✅' if rto < 3600 else '❌'}")
```

---

## 练习6: 增量更新与回滚

### 解答

```python
# 事务性增量合并
class TransactionalMerger:
    def __init__(self, main_index):
        self.main_index = main_index
        self._snapshot = None
    
    @contextmanager
    def transaction(self):
        """事务上下文管理器"""
        self._snapshot = deepcopy(self.main_index)
        try:
            yield self.main_index
            self._snapshot = None  # 成功，清除快照
        except Exception as e:
            # 失败，回滚
            self.main_index.restore(self._snapshot)
            self._snapshot = None
            raise MergeTransactionError(f"合并失败，已回滚: {e}")
    
    def merge_with_staging(self, delta_chunks, delta_vectors, deleted_ids):
        # 步骤1: 验证增量数据
        is_valid, errors = validate_delta(delta_chunks, delta_vectors)
        if not is_valid:
            return {"skipped": len(delta_chunks), "errors": errors}
        
        # 步骤2: Staging测试
        staging = deepcopy(self.main_index)
        staging.merge(delta_chunks, delta_vectors, deleted_ids)
        
        # 步骤3: 验证Staging
        if not staging.health_check():
            return {"skipped": len(delta_chunks), "errors": ["staging健康检查失败"]}
        
        # 步骤4: 事务性合并
        with self.transaction() as index:
            stats = index.merge(delta_chunks, delta_vectors, deleted_ids)
        
        return stats

# 版本冲突检测
class VersionTracker:
    def __init__(self):
        self._versions = {}  # chunk_id -> version
    
    def check_and_bump(self, chunk_id, expected_version):
        """乐观锁版本检查"""
        current = self._versions.get(chunk_id, 0)
        if expected_version != current:
            raise VersionConflictError(
                f"{chunk_id}: expected v{expected_version}, actual v{current}"
            )
        self._versions[chunk_id] = current + 1
        return current + 1
```

---

## 练习7: 蓝绿索引部署

### 解答

```python
# 指标对比器
class MetricsComparator:
    @staticmethod
    def compare(current, new):
        comparisons = []
        metrics = [
            ("avg_recall_at_10", "Recall@10", True),   # 越高越好
            ("avg_mrr", "MRR", True),
            ("avg_ndcg_at_10", "NDCG@10", True),
            ("avg_query_latency_ms", "平均延迟", False),  # 越低越好
            ("p99_query_latency_ms", "P99延迟", False),
            ("queries_per_second", "QPS", True),
            ("empty_result_rate", "空结果率", False),
            ("index_size_mb", "索引大小", False),
        ]
        for key, name, higher_better in metrics:
            current_val = getattr(current, key)
            new_val = getattr(new, key)
            delta_pct = (new_val - current_val) / current_val * 100
            improved = (new_val > current_val) if higher_better else (new_val < current_val)
            comparisons.append(ComparisonResult(name, current_val, new_val, delta_pct, improved))
        return comparisons
    
    @staticmethod
    def decide(comparisons):
        improved = sum(1 for c in comparisons if c.improved and c.significant)
        degraded = sum(1 for c in comparisons if not c.improved and c.significant)
        
        # 加权评分
        score = sum(
            (1 if c.improved else -1) * abs(c.delta_pct)
            for c in comparisons
        )
        
        if score > 5:
            return RebuildDecision.SWAP, []
        elif score < -10:
            return RebuildDecision.ALERT, [f"严重退化: score={score}"]
        else:
            return RebuildDecision.KEEP_CURRENT, []

# 蓝绿交换
class BlueGreenDeployer:
    def swap(self):
        """原子切换"""
        with self._lock:
            self._active_index, self._staging_index = (
                self._staging_index, self._active_index
            )
            self._active_id, self._staging_id = (
                self._staging_id, self._active_id
            )
```

---

## 练习8: Self-RAG反思循环

### 解答

```python
# 完整Self-RAG流程
def self_rag_pipeline(query: str, retriever, llm, max_iterations=3):
    """
    Self-RAG完整反思循环
    """
    # 步骤1: 检索决策
    decision_prompt = f"查询: {query}\n是否需要检索? 输出 <retrieve> 或 no_retrieve"
    decision = llm.generate(decision_prompt)
    
    if "<retrieve>" not in decision:
        return {"answer": llm.generate(f"回答: {query}"), "retrieved": False}
    
    # 步骤2: 检索
    candidates = retriever.search(query, k=10)
    
    # 步骤3: 相关性判断
    relevant_passages = []
    for doc in candidates:
        relevance_prompt = (
            f"查询: {query}\n文档: {doc['text'][:500]}\n"
            f"是否相关? <relevant> 或 <irrelevant>"
        )
        judgment = llm.generate(relevance_prompt)
        if "<relevant>" in judgment:
            relevant_passages.append(doc)
    
    # 步骤4: 生成 + 反思迭代
    context = "\n\n".join(p["text"] for p in relevant_passages)
    answer = llm.generate(f"基于以下文档回答: {query}\n\n文档: {context}")
    
    for iteration in range(max_iterations):
        # 反思
        reflect_prompt = (
            f"查询: {query}\n文档: {context}\n回答: {answer}\n"
            f"评估回答是否基于文档: "
            f"<supported> / <partially_supported> / <contradictory>"
        )
        reflection = llm.generate(reflect_prompt)
        
        if "<supported>" in reflection:
            break  # 质量合格，停止迭代
        
        # 优化
        refine_prompt = (
            f"原回答: {answer}\n"
            f"反思: {reflection}\n"
            f"请生成改进的回答，仅使用文档中的信息:"
        )
        answer = llm.generate(refine_prompt)
    
    return {"answer": answer, "iterations": iteration + 1}
```

---

## 练习9: CRAG回退策略

### 解答

```python
# CRAG三级置信度评估
def crag_pipeline(query, retriever, llm, web_searcher):
    # 步骤1: 初始检索
    passages = retriever.search(query, k=5)
    
    # 步骤2: 置信度评估
    scores = []
    for p in passages:
        score_prompt = f"查询: {query}\n段落: {p['text'][:300]}\n相关性分数(0-100):"
        score = int(llm.generate(score_prompt).strip())
        scores.append((score, p))
    
    avg_score = sum(s for s, _ in scores) / len(scores)
    
    # 步骤3: 根据置信度分支处理
    if avg_score > 70:
        # CORRECT: 直接生成
        final_passages = [p for s, p in scores if s > 70]
    elif avg_score > 40:
        # AMBIGUOUS: 知识细化
        refined = []
        for s, p in scores:
            if s > 60:
                refined.append(p)
            elif s > 40:
                # 提取相关内容
                extract_prompt = f"查询: {query}\n段落: {p['text']}\n提取相关句子:"
                extracted = llm.generate(extract_prompt)
                if "无关" not in extracted:
                    refined.append({"text": extracted, "source": "refined"})
        final_passages = refined
    else:
        # INCORRECT: Web搜索回退
        web_results = web_searcher.search(query, k=5)
        final_passages = web_results
    
    # 步骤4: 生成最终回答
    context = "\n\n".join(p.get("text", "")[:500] for p in final_passages)
    answer = llm.generate(f"基于以下信息回答: {query}\n\n{context}")
    
    return {
        "answer": answer,
        "confidence": "correct" if avg_score > 70 else "ambiguous" if avg_score > 40 else "incorrect",
        "avg_score": avg_score,
        "web_search_used": avg_score <= 40,
    }

# 测试验证
# 测试1: 知识库内的查询 → 预期CORRECT
# 测试2: 部分相关的查询 → 预期AMBIGUOUS
# 测试3: 完全陌生的查询 → 预期INCORRECT → Web搜索
```

---

## 练习10: 综合集成

### 解答

```text
## 系统架构图（文字描述）

                              ┌─────────────┐
                              │  API Gateway │
                              │  (认证/限流)  │
                              └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼─────┐  ┌──────▼──────┐  ┌─────▼─────┐
              │ RateLimiter│  │CircuitBreaker│  │  Auth     │
              │ (模块06)   │  │  (模块05)    │  │ (模块02)  │
              └─────┬─────┘  └──────┬──────┘  └─────┬─────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Adaptive Router   │
                          │     (模块12)        │
                          │  simple/moderate/   │
                          │     complex         │
                          └──────────┬──────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
    ┌─────▼─────┐            ┌──────▼──────┐           ┌───────▼───────┐
    │ Lightweight│            │  Standard   │           │  Hybrid Fusion│
    │  (模块01)  │            │  (模块01)   │           │  (模块10-13)  │
    │  BM25 Only │            │ Vector+Rerank│          │ Self-RAG/CRAG │
    └─────┬─────┘            └──────┬──────┘           │ Adaptive/KG   │
          │                          │                  └───────┬───────┘
          └──────────────────────────┼──────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Multi-Level Cache │
                          │      (模块03)       │
                          │   L1→L2→L3→L4      │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Tenant Isolation   │
                          │     (模块02)        │
                          │  Col/Metadata/Key   │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼─────┐  ┌──────▼──────┐  ┌─────▼─────┐
              │ Vector DB │  │  BM25 Index │  │Knowledge  │
              │  (HNSW)   │  │  (模块01)   │  │Graph (13) │
              │  (模块01)  │  │             │  │           │
              └─────┬─────┘  └──────┬──────┘  └─────┬─────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Backup/DR System  │
                          │      (模块07)       │
                          │  Full+Incremental   │
                          │  Cross-Region       │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Incremental Update │
                          │      (模块08)       │
                          │  + Index Rebuild    │
                          │      (模块09)       │
                          └─────────────────────┘

## 请求生命周期

1. 客户端发送请求(带API Key)
2. API Gateway 解析请求
3. RateLimiter 检查租户+端点速率(模块06)
   → 超限: 返回429 + Retry-After
4. API Key 认证(模块02, 第3层)
   → 失败: 返回401
5. 配额检查(模块02)
   → 超配额: 返回429
6. Prompt缓存检查(模块04)
   → 命中: 直接返回(节省API成本)
7. 四级缓存查询(模块03)
   L1精确匹配 → L2语义相似 → L3检索结果 → L4 LLM响应
   → 命中: 返回缓存结果
8. CircuitBreaker检查各依赖状态(模块05)
   → OPEN: 拒绝或降级
9. AdaptiveRouter分类查询复杂度(模块12)
   → simple → lightweight(BM25)
   → moderate → standard(vector+rerank)
   → complex → hybrid(decompose+parallel+fusion)
10. 租户隔离的向量+BM25检索(模块02, 第1+2层)
11. 高级RAG处理:
    → Self-RAG(模块10): 反思令牌判断
    → CRAG(模块11): 置信度评估+Web回退
    → KG-RAG(模块13): 图遍历增强
12. 检索结果融合+重排序
13. Context组装 + Prompt构建
14. LLM调用(通过Prompt缓存)
15. 响应后处理+格式验证
16. 返回结果 + 写回L4缓存
17. 审计日志记录

## 运维手册大纲

### 1. 备份与恢复
- 备份调度: 全量(每周日03:00) + 增量(每5分钟)
- 恢复步骤: 选择备份ID → 验证校验和 → 恢复到临时目录 → 验证 → 上线
- RTO目标: < 1小时
- RPO目标: < 5分钟
- DR演练: 每月第一个周一

### 2. 监控指标
- 延迟: P50/P95/P99(每端点)
- 缓存命中率: L1/L2/L3/L4 分别监控
- 熔断器状态: 每个依赖的CLOSED/OPEN/HALF_OPEN
- 速率限制: 每租户的剩余令牌
- 配额使用率: 每租户的文档/块/API使用率
- 索引状态: chunk数、向量数、重建历史
- 备份状态: 最近备份时间、复制延迟

### 3. 扩容操作
- 垂直扩容: 增加向量维度(需重建索引)
- 水平扩容: 增加租户数(集合自动创建)
- 缓存扩容: 增加Redis内存/节点
- 注意事项: 扩容期间冻结增量更新

### 4. 故障处理
- 向量DB不可用 → 熔断器自动拒绝 + 降级到BM25
- LLM API不可用 → 熔断 + 使用缓存结果
- 缓存Redis不可用 → 降级到内存缓存
- 备份损坏 → 使用跨区域副本
- 索引损坏 → 从备份恢复 + 重放增量
- 租户数据泄露 → 立即冻结租户 + 安全审计
```

---

## 评分标准

| 练习 | 分数 | 关键评估点 |
|------|------|-----------|
| 1 | 10分 | 管道完整性、错误处理、质量门控 |
| 2 | 15分 | 三层隔离、穿透测试、配额方案 |
| 3 | 10分 | 四级缓存、命中率计算、缓存预热 |
| 4 | 10分 | 配置优化、错误分类、指数退避 |
| 5 | 10分 | 全量+增量备份、恢复验证、RTO |
| 6 | 10分 | 事务性合并、Staging验证、版本冲突 |
| 7 | 10分 | 指标对比、蓝绿部署、漂移检测 |
| 8 | 10分 | 反思循环、自我优化、令牌解析 |
| 9 | 10分 | 置信度评估、回退策略、准确性对比 |
| 10 | 15分 | 架构设计、请求生命周期、运维手册 |

**总分: 110分** (含10分加分项)

**通过线: 80分**
