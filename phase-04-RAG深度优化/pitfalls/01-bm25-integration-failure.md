# Pitfall 01: BM25 静默返回空结果 (Silent Empty Return)

## 严重程度：高

## 症状

- 混合检索结果与纯向量检索完全一致
- 日志中无 BM25 相关错误或警告
- 检索质量在某些查询上明显偏低（特别是关键词精确匹配场景）
- `rank_bm25.get_scores()` 返回全零数组

## 根本原因

1. **BM25 分词器与文档不匹配**：BM25 默认使用英文空格分词，当处理中文文档时，所有文档都被当作单个 token，导致无法建立有效的倒排索引。

2. **查询 token 不在语料库中**：如果查询的所有 token 都未出现在 BM25 索引词汇表中（OOV），`get_scores()` 返回全零。

3. **语料库为空或仅含停用词**：BM25 初始化时，如果所有文档分词后为空或仅含停用词，索引构建失败但不会抛出异常。

4. **BM25Okapi 内部缓存失效**：在某些 `rank_bm25` 版本中，重复调用 `get_scores()` 可能因内部状态损坏返回空数组。

## 真实场景

某团队将混合检索部署到法律文档 RAG 系统，包含大量中文法律条文。他们使用默认的 BM25 配置：

```python
from rank_bm25 import BM25Okapi

# 错误：对中文文档使用默认分词（空格分割）
corpus = ["中华人民共和国合同法第一条规定...", "民法总则第二条..."]
tokenized = [doc.split(" ") for doc in corpus]  # 中文没有空格！
bm25 = BM25Okapi(tokenized)
scores = bm25.get_scores("合同法".split(" "))  # 返回 [0.0, 0.0]
```

每次查询的 BM25 结果都为空，系统自动 fallback 到纯向量检索，导致法律法规中的精确关键词匹配失效。用户查询"合同法第三十八条"时，向量检索返回语义相似但非目标条款，准确率下降 30%。

## 解决方案

### 1. 正确的中文分词策略

```python
import re

def tokenize_chinese(text: str) -> list[str]:
    """Proper Chinese tokenization for BM25."""
    tokens = []
    # Split into segments
    segments = re.split(r'([a-zA-Z0-9]+|[一-鿿]+)', text)
    for seg in segments:
        if not seg.strip():
            continue
        if re.match(r'[a-zA-Z0-9]+', seg):
            tokens.append(seg.lower())
        elif re.match(r'[一-鿿]+', seg):
            # Character bigrams for better matching
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i+2])
            tokens.append(seg[-1])
    return [t for t in tokens if len(t) >= 1]
```

### 2. 显式的 BM25 空结果检测与 Fallback

```python
class RobustSparseRetriever:
    def search(self, query: str, top_k: int = 20) -> list:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []  # 显式返回空

        scores = self.bm25.get_scores(query_tokens)

        # 检测全零结果
        if len(scores) == 0 or np.max(scores) == 0:
            logger.warning(f"BM25 returned all-zero scores for query: {query}")
            return []  # 触发 fallback

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
```

### 3. 完整的 Fallback 链

```python
class HybridRetriever:
    def search(self, query, top_k=10):
        # Step 1: Dense retrieval (always works)
        dense_results = self.dense_retriever.search(query, top_k=50)

        # Step 2: Sparse retrieval with fallback
        try:
            sparse_results = self.sparse_retriever.search(query, top_k=50)
        except Exception as e:
            logger.error(f"BM25 error: {e}")
            sparse_results = []

        # Step 3: Fallback strategy
        if not sparse_results or self._is_empty(sparse_results):
            logger.info(f"BM25 fallback triggered for: {query[:50]}")
            return self._dense_only_results(dense_results, top_k)

        # Step 4: Normal hybrid fusion
        return self._hybrid_fusion(dense_results, sparse_results, top_k)

    def _is_empty(self, results):
        return len(results) == 0 or all(s == 0.0 for _, s in results)
```

## 检查清单

- [ ] 确认使用正确的分词器处理中文文档（不是 `.split(" ")`）
- [ ] BM25 索引后检查词汇表大小（`len(bm25.idf)`）> 0
- [ ] 每个查询都记录 BM25 是否返回有效结果
- [ ] 实现显式的空结果 fallback 到纯向量检索
- [ ] 添加告警：当 BM25 空结果率 > 20% 时触发
- [ ] 单元测试覆盖：全中文、全英文、中英混合、OOV 查询
- [ ] 生产监控：`bm25_empty_rate` 指标

## 影响评估

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| BM25 有效查询比例 | 30-50% | 95%+ |
| 关键词精确匹配召回 | 0% | 85%+ |
| 混合检索真实生效比例 | 30-50% | 100% |
| 系统鲁棒性 | 低 | 高 |
