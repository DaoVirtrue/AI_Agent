# Pitfall 05: 重排序延迟瓶颈 (Reranker Latency Bottleneck)

## 严重程度：中

## 症状

- 端到端检索延迟从 50ms 飙升到 300-800ms
- Reranker 处理时间占总延迟的 70-85%
- 用户感知到明显的响应延迟
- GPU 利用率波动大（频繁的 CPU-GPU 数据传输）
- 并发请求下延迟呈指数增长

## 根本原因

1. **无批处理**：每个查询单独调用 reranker，每次只处理一个 (query, document) 对。GPU 调用开销（kernel launch + 数据传输）远大于实际计算时间。单个推理可能只需 2ms，但 GPU 调用固定开销就占 5-10ms。

2. **未限制粗检索候选数**：粗检索返回 100+ 个候选全部送入 reranker。Reranker（Cross-Encoder）的复杂度是 O(n * model_cost)，100 个候选意味着 100 次模型推理。

3. **CPU-GPU 数据传输瓶颈**：每个文档从 CPU 内存传到 GPU 显存，处理完传回。频繁的小数据传输导致 PCIe 带宽利用率低下。

4. **Reranker 模型选择不当**：在延迟敏感场景使用大型 reranker（如 BGE-Reranker-v2-m3, 560M 参数）而非轻量级 Cross-Encoder（如 MiniLM-L-6-v2, 22M 参数）。

5. **缺少缓存**：相同或相似查询重复执行 reranking，浪费计算资源。

## 真实场景

某实时客服系统集成 BGE-Reranker-v2-m3：

```python
# 问题代码：每次查询独立调用
def search_with_rerank(query):
    coarse_results = vector_search(query, top_k=50)  # 50 candidates
    reranked = []
    for doc in coarse_results:  # 50 次独立 GPU 调用！
        score = reranker.compute_score([query, doc.text])
        reranked.append((doc, score))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:5]
```

结果：
- 每个查询 50 次 GPU 调用
- 每次调用 ~12ms（kernel launch + 推理 + 传回）
- 总延迟：50 * 12ms = 600ms（仅 reranker）
- 加上向量检索 80ms + 网络 50ms = 总延迟 730ms
- SLA 要求 < 300ms → 未达标

## 解决方案

### 1. 批量 Reranking

```python
class BatchedReranker:
    def __init__(self, model, batch_size=16, max_seq_length=512):
        self.model = model
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if len(documents) <= self.batch_size:
            # Single batch
            pairs = [[query, doc[:self.max_seq_length]] for doc in documents]
            scores = self.model.compute_score(pairs)
            scores = [float(s) for s in scores]
        else:
            # Multi-batch
            scores = []
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                pairs = [[query, doc[:self.max_seq_length]] for doc in batch]
                batch_scores = self.model.compute_score(pairs)
                scores.extend(float(s) for s in batch_scores)

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return indexed
```

延迟改善：50 个候选 / 16 batch = 4 次 GPU 调用，每次 ~15ms（含 batch 开销），总 60ms。相比 600ms，改善 10x。

### 2. 两阶段 Reranking（级联）

```python
class CascadeReranker:
    """Stage 1: Fast Cross-Encoder → Top 10, Stage 2: Heavy BGE Reranker → Top 5"""
    def __init__(self):
        self.fast_reranker = CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.heavy_reranker = BGEReranker('BAAI/bge-reranker-v2-m3')

    def rerank(self, query, documents):
        # Stage 1: Fast filter to Top 15
        stage1 = self.fast_reranker.rerank(query, documents, top_k=15)
        stage1_docs = [documents[i] for i, _ in stage1]

        # Stage 2: Heavy reranker on Top 15
        stage2 = self.heavy_reranker.rerank(query, stage1_docs, top_k=5)

        # Map back to original indices
        stage1_indices = [i for i, _ in stage1]
        result = [(stage1_indices[i], score) for i, score in stage2]
        return result
```

### 3. 粗检索截断策略

```python
def optimal_coarse_top_n(intent: IntentType) -> int:
    """根据查询意图确定粗检索候选数。"""
    return {
        IntentType.FACT: 15,       # 精确查找：候选少
        IntentType.PROCESS: 20,    # 流程查询：中等
        IntentType.COMPARISON: 25, # 对比查询：候选多
        IntentType.TROUBLESHOOT: 20,
        IntentType.REASONING: 25,
        IntentType.CHAT: 10,
    }.get(intent, 20)
```

### 4. 查询缓存

```python
from functools import lru_cache
import hashlib

class CachedReranker:
    def __init__(self, reranker, cache_size=1000):
        self.reranker = reranker
        self.cache = {}

    def rerank(self, query: str, documents: list[str]):
        # 计算文档集合哈希
        doc_hash = hashlib.md5(
            ''.join(d[:100] for d in documents).encode()
        ).hexdigest()
        cache_key = f"{query}_{doc_hash}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self.reranker.rerank(query, documents)
        self.cache[cache_key] = result

        # LRU eviction
        if len(self.cache) > self.cache_max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        return result
```

## 延迟优化决策树

```
Reranker 延迟 > 100ms？
├── YES → 检查粗检索候选数
│   ├── > 30 候选 → 减少到 15-20
│   └── <= 30 → 检查批处理
│       ├── 未批处理 → 启用 batch_size=16
│       └── 已批处理 → 考虑级联策略
│           ├── 轻量 Cross-Encoder (22M) → Top 10
│           └── 重型 BGE Reranker (560M) → Top 5
│
└── 延迟 < 100ms → OK，但继续监控
    └── 增加缓存层以降低重复查询开销
```

## 检查清单

- [ ] 是否使用了批量 reranking（batch_size >= 16）？
- [ ] 粗检索候选数是否控制在 15-25 之间？
- [ ] 是否实施了两阶段级联 reranking？
- [ ] 是否选择了合适的 reranker 模型（延迟 vs 质量）？
- [ ] 是否使用了查询-结果缓存？
- [ ] 是否设置了 reranker 超时（如 500ms）？
- [ ] 是否监控了 reranker 延迟的 P50/P95/P99？
- [ ] GPU 推理是否使用了 FP16 精度？
- [ ] 是否预热了 reranker 模型（首次调用延迟高）？
- [ ] 是否有降级策略（reranker 超时时使用粗检索结果）？
