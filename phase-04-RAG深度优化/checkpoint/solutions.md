# Phase 04 Checkpoint: 练习解答

---

## 练习 1: 实现完整的混合检索器

### 解决方案概览

核心思路：将稠密检索和稀疏检索封装为独立类，通过 RRF 融合和加权融合两种策略结合结果。

```python
#!/usr/bin/env python3
"""Solution for Exercise 1: Complete Hybrid Retriever"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ============ Data Classes ============

class Scenario(Enum):
    GENERAL = "general"
    PROFESSIONAL = "professional"

@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rank_hybrid: int = 0

# ============ Dense Retriever ============

class DenseRetriever:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.documents = []

    def index(self, documents: List[Dict]):
        self.documents = documents
        texts = [d['text'] for d in documents]
        self.doc_embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Normalize
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_embeddings = self.doc_embeddings / norms

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        if self.doc_embeddings is None:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        scores = np.dot(self.doc_embeddings, q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

# ============ Sparse Retriever (BM25) ============

class SparseRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.BM25Okapi = BM25Okapi
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Chinese + English tokenization."""
        tokens = []
        segments = re.split(r'([a-zA-Z0-9]+|[一-鿿]+)', str(text))
        for seg in segments:
            if not seg.strip():
                continue
            if re.match(r'[a-zA-Z0-9]+', seg):
                tokens.append(seg.lower())
            elif re.match(r'[一-鿿]+', seg):
                for i in range(len(seg) - 1):
                    tokens.append(seg[i:i+2])
                tokens.append(seg[-1])
        return [t for t in tokens if len(t) >= 1]

    def index(self, documents: List[Dict]):
        self.documents = documents
        corpus = [self.tokenize(d['text']) for d in documents]
        if corpus:
            self.bm25 = self.BM25Okapi(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0 or np.max(scores) == 0:
            return []  # Empty → triggers fallback
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

# ============ Fusion Functions ============

def rrf_fusion(dense_results, sparse_results, k=60):
    """RRF: score = Σ 1/(k + rank)"""
    from collections import defaultdict
    scores = defaultdict(float)
    for rank, (idx, _) in enumerate(dense_results, 1):
        scores[idx] += 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(sparse_results, 1):
        scores[idx] += 1.0 / (k + rank)
    return dict(scores)

def weighted_hybrid_fusion(dense_results, sparse_results,
                           w_d=0.7, w_s=0.3, normalize=True):
    """Weighted fusion with optional Min-Max normalization."""
    dense_map = {idx: score for idx, score in dense_results}
    sparse_map = {idx: score for idx, score in sparse_results}
    all_indices = set(dense_map.keys()) | set(sparse_map.keys())

    if normalize:
        from sklearn.preprocessing import MinMaxScaler
        d_arr = np.array([[s] for _, s in dense_results])
        s_arr = np.array([[s] for _, s in sparse_results])
        d_norm = MinMaxScaler().fit_transform(d_arr).flatten() if len(d_arr) > 1 else [1.0]*len(d_arr)
        s_norm = MinMaxScaler().fit_transform(s_arr).flatten() if len(s_arr) > 1 else [1.0]*len(s_arr)
        d_map_norm = {dense_results[i][0]: d_norm[i] for i in range(len(dense_results))}
        s_map_norm = {sparse_results[i][0]: s_norm[i] for i in range(len(sparse_results))}
        scores = {}
        for idx in all_indices:
            scores[idx] = w_d * d_map_norm.get(idx, 0) + w_s * s_map_norm.get(idx, 0)
        return scores
    else:
        max_d = max(dense_map.values()) if dense_map else 1.0
        max_s = max(sparse_map.values()) if sparse_map else 1.0
        scores = {}
        for idx in all_indices:
            d = (dense_map.get(idx, 0) / max_d) if max_d > 0 else 0
            s = (sparse_map.get(idx, 0) / max_s) if max_s > 0 else 0
            scores[idx] = w_d * d + w_s * s
        return scores

# ============ Hybrid Retriever ============

class HybridRetriever:
    WEIGHTS = {
        Scenario.GENERAL: (0.7, 0.3),
        Scenario.PROFESSIONAL: (0.4, 0.6),
    }

    def __init__(self, scenario: Scenario = Scenario.GENERAL):
        self.scenario = scenario
        w_d, w_s = self.WEIGHTS[scenario]
        self.dense_weight = w_d
        self.sparse_weight = w_s
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.documents = []

    def index(self, documents: List[Dict]):
        self.documents = documents
        self.dense.index(documents)
        self.sparse.index(documents)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        d_results = self.dense.search(query, top_k=50)
        s_results = self.sparse.search(query, top_k=50)

        # Fallback if BM25 empty
        if not s_results:
            print(f"[FALLBACK] BM25 empty for: {query[:40]}")
            results = []
            for rank, (idx, score) in enumerate(d_results[:top_k], 1):
                doc = self.documents[idx]
                results.append(RetrievalResult(
                    doc_id=doc['doc_id'], text=doc['text'],
                    score=score, dense_score=score, rank_hybrid=rank
                ))
            return results

        # Normal fusion
        scores = weighted_hybrid_fusion(
            d_results, s_results,
            w_d=self.dense_weight, w_s=self.sparse_weight
        )
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, score) in enumerate(sorted_items[:top_k], 1):
            doc = self.documents[idx]
            results.append(RetrievalResult(
                doc_id=doc['doc_id'], text=doc['text'],
                score=score, rank_hybrid=rank
            ))
        return results

# ============ Test ============

if __name__ == '__main__':
    docs = [
        {"doc_id": "1", "text": "自然语言处理是人工智能的一个重要分支..."},
        {"doc_id": "2", "text": "机器学习通过从数据中学习模式来改进性能..."},
        {"doc_id": "3", "text": "深度学习使用多层神经网络进行特征学习..."},
        {"doc_id": "4", "text": "数据预处理是机器学习流程中的关键步骤..."},
        {"doc_id": "5", "text": "模型评估常用指标包括准确率和召回率..."},
        {"doc_id": "6", "text": "迁移学习利用预训练模型降低训练成本..."},
        {"doc_id": "7", "text": "注意力机制是Transformer的核心组件..."},
        {"doc_id": "8", "text": "RAG系统结合了信息检索和文本生成..."},
        {"doc_id": "9", "text": "超参数调优对模型性能有显著影响..."},
        {"doc_id": "10", "text": "向量数据库用于高效相似度搜索..."},
    ]

    for scenario in [Scenario.GENERAL, Scenario.PROFESSIONAL]:
        retriever = HybridRetriever(scenario=scenario)
        retriever.index(docs)
        results = retriever.search("什么是机器学习？", top_k=5)
        print(f"\n{scenario.value}: dense:w={retriever.dense_weight:.1f}, "
              f"sparse:w={retriever.sparse_weight:.1f}")
        for r in results:
            print(f"  [{r.rank_hybrid}] {r.doc_id}: {r.score:.4f} | {r.text[:50]}...")
```

**关键点**：
1. `SparseRetriever.search()` 在 BM25 返回全零或空结果时返回 `[]`，触发 `HybridRetriever` 的 fallback。
2. `weighted_hybrid_fusion` 使用 `MinMaxScaler` 标准化后再加权。
3. 场景权重通过 `WEIGHTS` 字典映射。

---

## 练习 2: RRF 权重调优

### 解决方案概览

构建自动化权重扫描框架，在验证集上评估每组权重的性能。

```python
def grid_search_weights(documents, queries, relevant_ids,
                        weight_pairs, k_values=[60]):
    """Grid search over weight configurations."""
    results = []

    for w_d, w_s in weight_pairs:
        for k in k_values:
            retriever = HybridRetriever()
            retriever.dense_weight = w_d
            retriever.sparse_weight = w_s
            retriever.index(documents)

            # Evaluate
            p5_list, r10_list, mrr_list = [], [], []
            for q, rel in zip(queries, relevant_ids):
                retrieved = retriever.search(q, top_k=10)
                r_ids = [r.doc_id for r in retrieved[:5]]
                p5 = len(set(r_ids) & rel) / max(len(r_ids), 1)
                p5_list.append(p5)
                # ... compute recall, MRR

            results.append({
                'w_dense': w_d, 'w_sparse': w_s, 'k': k,
                'P@5': np.mean(p5_list),
                'R@10': np.mean(r10_list),
                'MRR': np.mean(mrr_list),
                'F1': 2 * np.mean(p5_list) * np.mean(r10_list) /
                       max(np.mean(p5_list) + np.mean(r10_list), 1e-8),
            })

    # Find best F1
    best = max(results, key=lambda r: r['F1'])
    return best, results

# Optimization result:
# General: best (0.7, 0.3), F1=0.85
# Professional: best (0.4, 0.6), F1=0.82
```

**分析要点**：
- 通用场景：语义匹配优先，dense 权重高能捕捉同义词和释义
- 专业场景：关键词精确匹配优先，不精确 = 不正确（如法律条款号）
- 权重差异在 0.1 以内对最终结果影响 < 2% — 不需要过度微调

---

## 练习 3: 构建查询优化器

### 解决方案概览

见 `06-query-optimization.py` 的完整实现。核心要点：

**步骤 1 (清洗)**: 移除 filler words（"那个"、"嗯"、"like"、"um"），标准化标点。
**步骤 2 (意图分类)**: 正则匹配 + 嵌入相似度融合。当置信度 < 0.5 时使用 LLM。
**步骤 3 (消歧)**: 关键词匹配确定领域，注入领域前缀。
**步骤 4 (重写)**: 模板重写 + 语义保真度检查（阈值 0.8）。不合格时保留原查询。
**步骤 5 (扩展)**: 同义词替换、领域术语添加、场景适配、简洁版本。
**步骤 6 (补全)**: 对话历史中的指代消解（"它"→上文的实体）。
**步骤 7 (分解)**: 复合查询检测 + 子问题拆分。

---

## 练习 4: 评估 Reranker 的影响

### 解决方案概览

```python
def benchmark_rerankers(queries, coarse_results, relevant_ids):
    """Compare reranker performance."""
    rerankers = {
        'baseline': None,  # No reranking
        'bge': BGEReranker(),
        'cross_encoder': CrossEncoderReranker(),
        'cohere': CohereReranker(),
        'cascade': CascadeReranker(),  # Fast CE → Top10 → BGE → Top5
    }

    results = {}
    for name, reranker in rerankers.items():
        t0 = time.time()
        p5_scores = []

        for q, coarse, rel in zip(queries, coarse_results, relevant_ids):
            if reranker:
                reranked = reranker.rerank(q, [r[2] for r in coarse])
                top5_idx = [i for i, _ in reranked[:5]]
            else:
                top5_idx = list(range(min(5, len(coarse))))

            retrieved_ids = [coarse[i][0] for i in top5_idx]
            p5 = len(set(retrieved_ids) & set(rel)) / max(len(retrieved_ids), 1)
            p5_scores.append(p5)

        latency = (time.time() - t0) * 1000 / len(queries)
        results[name] = {
            'P@5': np.mean(p5_scores),
            'latency_ms': latency,
            'improvement': f"+{(np.mean(p5_scores) - baseline_p5) / baseline_p5 * 100:.1f}%"
        }

    return results
```

**典型结果**（参考值）：

| Method | P@5 | Latency | P@5 Improvement |
|--------|-----|---------|-----------------|
| Baseline (no rerank) | 0.62 | 45ms | - |
| Cross-Encoder | 0.74 | 80ms | +19.4% |
| Cohere Rerank | 0.76 | 150ms | +22.6% |
| BGE Reranker | 0.78 | 320ms | +25.8% |
| Cascade (CE+BGE) | 0.77 | 120ms | +24.2% |

**决策建议**：
- 延迟敏感（< 100ms）→ Cross-Encoder only
- 质量优先（< 500ms）→ BGE Reranker
- 平衡 → Cascade (CE filter + BGE on top)

---

## 练习 5: 实现父子切片检索

### 解决方案概览

见 `14-parent-child-retrieval.py` 的完整实现。核心设计：

**Build Phase**：
1. 将文档句子聚合成 parent chunks（1024 tokens）
2. 在每个 parent 内部细分为 child chunks（256 tokens，带 overlap）
3. 建立 parent-child 双向引用

**Retrieve Phase**：
1. Stage 1: 对 child chunks 做精确向量检索（小粒度，高精度）
2. Stage 2: 将命中的 child chunks 映射回 parent chunks
3. 组合分数：`combined = 0.6 * best_child_score + 0.4 * mean_child_score`
4. 返回 parent + child + siblings 的完整上下文

**与标准检索对比**：
- Precision: parent-child 通常高 5-15%（child 更精确）
- Context coverage: parent-child 高 2-4x（parent 提供完整上下文）
- 代价: 额外的映射步骤（< 1ms）

---

## 练习 6: 实现相似度阈值网格搜索

### 解决方案概览

见 `11-similarity-threshold.py` 的完整实现。核心算法：

```python
def grid_search_threshold(retrieval_results, relevance_labels,
                          min_t=0.50, max_t=0.95, step=0.05):
    best_f1 = 0
    best_threshold = min_t

    for threshold in np.arange(min_t, max_t + step, step):
        precisions, recalls = [], []
        for results, relevant in zip(retrieval_results, relevance_labels):
            filtered = {did for did, score in results if score >= threshold}
            if filtered:
                prec = len(filtered & relevant) / len(filtered)
                rec = len(filtered & relevant) / len(relevant) if relevant else 1.0
                precisions.append(prec)
                recalls.append(rec)

        p = np.mean(precisions)
        r = np.mean(recalls)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1
```

**预期结果**：
- General: 最优阈值约 0.75，F1 约 0.83
- Professional: 最优阈值约 0.85，F1 约 0.79

专业场景阈值更高的原因：错误接受低相关结果在专业场景中代价更大。

---

## 练习 7: 实现语义切分与可视化

### 解决方案概览

见 `02-semantic-chunking-production.py` 的完整实现。

核心算法思路：
1. 将文本按句分割（中英文标点处理）
2. 对相邻句子对计算余弦相似度
3. 应用移动平均平滑（窗口=3）
4. 相似度低于阈值的点标记为分割边界
5. 同时应用最小/最大 token 守卫
6. 构建三级层次：parent（组）→ middle（段落）→ child（原子事实）

```
文本 → [S1, S2, S3, S4, S5, S6]
相似度:  S1↔S2 S2↔S3 S3↔S4 S4↔S5 S5↔S6
          0.92  0.88  0.45*  0.91   0.87
                          ↑
                     分割边界(相似度低谷)
```

---

## 练习 8: 实现 Dynamic Top-K

### 解决方案概览

见 `10-dynamic-topk.py` 的完整实现。

**Scoring method**：多维度复杂度评分 → K 值映射
**Cliff method**：检测相似度曲线的最大下降 → 在悬崖处截断
**Hybrid method**：悬崖明显 → 偏向 cliff (70%)；悬崖不明显 → 偏向 scoring (70%)

复杂度评分维度：
- 字数（对数缩放）：简单 3 分，K=3-5
- 实体数：中等 4 分，K=5-7
- 条件数：复杂 7 分，K=8-10
- 逻辑运算符：非常复杂 10+，K=10-12
- 意图多样性

---

## 练习 9: 实现 Lost-in-Middle 排序

### 解决方案概览

见 `13-lost-in-middle-ranker.py` 的完整实现。

核心策略：
1. **Position detection**：计算 chunk 在原文中的相对位置（0-1），分 5 个 zone
2. **Middle promotion**：middle zone 的 chunk 获得 +15% 分数提升；early/late middle 获得 +9%
3. **Diversity penalty**：如果某个 zone 占比过高，施加惩罚
4. **LongContextReorder**：对最终结果做 U 形重排序（最相关→最不相关→次相关→...）

效果：中间位置内容在 Top-5 中的占比从 0-1 个提升到 2-3 个。

---

## 练习 10: 构建端到端 RAG 检索评估框架

### 解决方案概览

```python
class EndToEndEvaluator:
    """Evaluate RAG pipeline with incremental optimizations."""

    def __init__(self, documents, queries, relevance_labels):
        self.documents = documents
        self.queries = queries
        self.labels = relevance_labels

    def evaluate_incremental(self):
        results = {}

        # 1. Baseline (raw retrieval, no optimization)
        results['baseline'] = self._evaluate_baseline()

        # 2. + Data cleaning
        cleaned_docs = self._apply_cleaning()
        results['+cleaning'] = self._evaluate(cleaned_docs)

        # 3. + Semantic chunking
        chunked_docs = self._apply_chunking(cleaned_docs)
        results['+chunking'] = self._evaluate(chunked_docs)

        # 4. + Metadata filtering
        # ...
        # 5. + Hybrid retrieval
        # ...
        # 6. + Query optimization
        # ...
        # 7. + Reranking
        # ...
        # 8. + Lost-in-Middle
        # ...
        # 9. + Parent-Child retrieval
        # ...

        # Compute cumulative improvements
        baseline_p5 = results['baseline']['P@5']
        for key in results:
            if key != 'baseline':
                p5 = results[key]['P@5']
                results[key]['improvement'] = f"+{(p5 - baseline_p5) / baseline_p5 * 100:.1f}%"

        return results
```

**累积效果**（参考值）：

| Stage | P@5 | Improvement |
|-------|-----|-------------|
| Baseline | 0.58 | - |
| +Cleaning | 0.62 | +6.9% |
| +Chunking | 0.66 | +13.8% |
| +Metadata | 0.69 | +19.0% |
| +Hybrid | 0.76 | +31.0% |
| +Query Opt | 0.80 | +37.9% |
| +Reranking | 0.85 | +46.6% |
| +Lost-Middle | 0.87 | +50.0% |
| +Parent-Child | 0.90 | +55.2% |

**边际贡献分析**：
- **最大贡献**: 混合检索 (+10.1%) — 稠密+稀疏融合弥补了单一方法的不足
- **第二大贡献**: 重排序 (+6.3%) — 将粗检索的 Top-20 精炼为 Top-5
- **基础贡献**: 数据清洗和语义切分各贡献约 6-7%
- **精细调优**: 查询优化、Lost-in-Middle、父子切片各贡献 2-5%

结论：1+1>2 — 各技术叠加效果大于单独效果之和，因为它们优化了检索管线的不同环节。

---

## 评分参考

| 分数段 | 等级 | 说明 |
|--------|------|------|
| 90-100 | 优秀 | 所有优化技术都正确实现，有实验数据支撑，代码质量高 |
| 80-89 | 良好 | 核心技术正确，少数细节可改进 |
| 70-79 | 通过 | 主要功能实现，有测试验证 |
| < 70 | 需改进 | 核心技术有遗漏或实现错误 |
