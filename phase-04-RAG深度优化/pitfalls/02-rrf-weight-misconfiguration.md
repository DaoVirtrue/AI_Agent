# Pitfall 02: RRF 融合权重配置错误 (RRF Weight Misconfiguration)

## 严重程度：高

## 症状

- 混合检索结果质量低于单一检索（稠密或稀疏）
- 专业场景（法律/医学）使用稠密优先权重，返回语义相关但非精确匹配的结果
- 通用场景（FAQ）使用稀疏优先权重，错过语义相似的优质结果
- 用户反馈"搜索结果没有以前准了"

## 根本原因

RRF（Reciprocal Rank Fusion）的权重配置与场景不匹配：

1. **通用场景误用高稀疏权重**：通用 FAQ/百科问答场景中，用户查询偏口语化、多样化，稠密向量检索能更好捕捉语义。但配置 `dense:sparse = 4:6` 导致系统过度依赖关键词匹配，错过同义词和释义。

2. **专业场景误用高稠密权重**：法律/医学场景中，术语精确匹配至关重要。"刑法第232条"必须精确匹配，不能用语义相似替代。配置 `dense:sparse = 7:3` 导致检索到"类似条款"而非目标条款。

3. **权重的实际效果非线性**：简单的 7:3 不代表稠密贡献 70%。RRF 公式 `1/(k+rank)` 本身有压缩效应——权重差异在排名靠前时影响小，排名靠后时影响大。

4. **未考虑归一化**：稠密得分（0-1 cosine）和 BM25 得分（无界正值）的量纲完全不同，直接加权组合不公平。

## 真实场景

某法律科技公司为合同审查 RAG 系统配置了默认混合检索：

```python
# 错误配置
retriever = HybridRetriever(
    dense_weight=0.7,   # 偏向语义理解
    sparse_weight=0.3,
)
```

用户查询"劳动合同法第四十六条"，期望精确匹配该条款。但系统返回了"劳动合同法第四十七条"和"劳动合同解除的法律依据"等内容——它们在语义上相似，但不是用户需要的精确条款。

根本原因：法律查询需要精确的关键词匹配，但 7:3 的权重让语义相似的结果排在了前面。将权重调整为 3:7 后，精确匹配率从 62% 提升到 91%。

## 解决方案

### 1. 场景自适应权重

```python
from enum import Enum

class Scenario(Enum):
    GENERAL = "general"        # 通用：dense:sparse = 7:3
    PROFESSIONAL = "professional"  # 专业：dense:sparse = 4:6

class HybridRetriever:
    SCENARIO_WEIGHTS = {
        Scenario.GENERAL: (0.7, 0.3),
        Scenario.PROFESSIONAL: (0.4, 0.6),
    }

    def __init__(self, scenario: Scenario = Scenario.GENERAL):
        self.dense_weight, self.sparse_weight = self.SCENARIO_WEIGHTS[scenario]
```

### 2. Min-Max 归一化

```python
def weighted_fusion_with_normalization(dense_results, sparse_results, w_d, w_s):
    """先归一化再融合，确保两种检索的量纲一致。"""
    from sklearn.preprocessing import MinMaxScaler

    dense_scores = np.array([s for _, s in dense_results]).reshape(-1, 1)
    sparse_scores = np.array([s for _, s in sparse_results]).reshape(-1, 1)

    scaler_d = MinMaxScaler().fit(dense_scores)
    scaler_s = MinMaxScaler().fit(sparse_scores)

    dense_norm = scaler_d.transform(dense_scores).flatten()
    sparse_norm = scaler_s.transform(sparse_scores).flatten()

    # 归一化后的加权组合
    combined = {}
    for (idx, _), dn in zip(dense_results, dense_norm):
        combined[idx] = w_d * dn
    for (idx, _), sn in zip(sparse_results, sparse_norm):
        combined[idx] = combined.get(idx, 0) + w_s * sn

    return combined
```

### 3. 权重验证与监控

```python
def validate_weights(queries, relevant_docs, weights_to_test):
    """在验证集上测试不同权重配置。"""
    results = {}
    for w_d, w_s in weights_to_test:
        retriever = HybridRetriever(dense_weight=w_d, sparse_weight=w_s)
        metrics = retriever.benchmark(queries, relevant_docs)
        results[(w_d, w_s)] = metrics['hybrid_rrf'].precision_at_5
    best = max(results, key=results.get)
    return best, results
```

## 权重选择决策树

```
查询意图是什么？
├── 精确查找（法规条款、故障码、药品名）
│   → sparse_weight >= 0.5
│   → 推荐: dense:sparse = 4:6
│
├── 概念理解（定义、原理、概念）
│   → dense_weight >= 0.6
│   → 推荐: dense:sparse = 7:3
│
├── 混合查询（包含步骤、条件）
│   → balanced
│   → 推荐: dense:sparse = 5:5
│
└── 不确定
    → 默认通用场景
    → 推荐: dense:sparse = 7:3
```

## 检查清单

- [ ] 是否为不同场景（通用 vs 专业）配置了不同的权重？
- [ ] 是否在验证集上测试了 3+ 组不同权重配置？
- [ ] 是否启用了 Min-Max 归一化？
- [ ] 是否监控了各场景下的检索准确率变化？
- [ ] 是否记录了权重变更的时间和原因？
- [ ] 是否有 A/B 测试框架验证权重变更效果？
- [ ] 是否检查了 RRF k 值（默认 60，极端场景可能需要调整）？
