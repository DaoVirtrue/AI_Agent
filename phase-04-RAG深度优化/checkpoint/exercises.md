# Phase 04 Checkpoint: 综合练习

> 完成所有 10 个练习以巩固 Phase 04 的核心知识点。

---

## 练习 1: 实现完整的混合检索器

**目标**：从零实现一个包含稠密检索、稀疏检索、RRF 融合的混合检索器。

**要求**：
1. 实现 `DenseRetriever` 类，使用 `sentence-transformers` 进行向量化
2. 实现 `SparseRetriever` 类，使用 `rank_bm25` 进行 BM25 检索
3. 实现 `RRF_fusion()` 函数，k=60
4. 实现 `weighted_hybrid_fusion()` 函数，支持 Min-Max 归一化
5. 为通用和专业场景配置不同的权重（7:3 和 4:6）
6. 实现 BM25 空结果的 fallback 策略

**测试**：
```python
documents = [
    {"doc_id": "1", "text": "自然语言处理是AI的一个重要分支..."},
    {"doc_id": "2", "text": "机器学习通过数据学习模式..."},
    # ... 至少 10 个文档
]
retriever = HybridRetriever(scenario=Scenario.GENERAL)
retriever.index(documents)
results = retriever.search("什么是机器学习？", top_k=5)
assert len(results) == 5
assert results[0].score > 0
```

**评分标准**：
- 稠密检索正确实现: 2 分
- BM25 检索正确实现: 2 分  
- RRF 融合正确: 2 分
- Min-Max 归一化: 1 分
- 场景权重配置: 1 分
- Fallback 策略: 1 分
- BM25 空结果处理: 1 分

---

## 练习 2: RRF 权重调优

**目标**：通过实验找到最优的稠密/稀疏权重组合。

**要求**：
1. 准备一个包含 50+ 文档的测试集和 10+ 标注查询
2. 测试 5 组不同的权重组合：(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)
3. 计算每组的 Precision@5, Recall@10, MRR
4. 绘制权重 vs 性能曲线
5. 分析不同场景下最优权重差异的原因

**输出**：
```python
{
    "general_scenario": {"best_weights": (0.7, 0.3), "best_f1": 0.85},
    "professional_scenario": {"best_weights": (0.4, 0.6), "best_f1": 0.82}
}
```

**评分标准**：
- 测试集构建合理: 2 分
- 完整的评估指标计算: 2 分
- 多组权重对比实验: 2 分
- 可视化权重-性能曲线: 2 分
- 场景差异分析: 2 分

---

## 练习 3: 构建查询优化器

**目标**：实现完整的 7 步查询预处理管线。

**要求**：
1. 实现 `QueryCleaner.clean()` - 处理 5 种以上 filler words
2. 实现 `IntentClassifier.classify()` - 至少 4 种意图类型，准确率 > 75%
3. 实现 `Disambiguator.disambiguate()` - 至少 3 个领域
4. 实现 `QueryRewriter.rewrite()` - 基于模板的重写，含语义保真度检查
5. 实现 `QueryExpander.expand()` - 生成 3-5 个变体
6. 实现 `QueryDecomposer.decompose()` - 复杂查询拆分为子问题
7. 单元测试覆盖每个步骤

**测试**：
```python
pipeline = QueryOptimizationPipeline()
result = pipeline.process("那个就是，机器学习怎么训练啊？")
assert result.cleaned != result.original
assert result.intent in [IntentType.PROCESS, IntentType.FACT]
assert len(result.expansions) >= 1
```

**评分标准**：
- 每个步骤独立可测试: 每个 1 分 (共 7 分)
- 端到端管线串联正确: 1 分
- 边缘情况处理（空输入、超长输入等）: 2 分

---

## 练习 4: 评估 Reranker 的影响

**目标**：量化重排序对检索质量的提升效果。

**要求**：
1. 准备 20 个查询的测试集和相关性标注
2. 实现基准对比：无 Rerank vs BGE Reranker vs Cross-Encoder vs Cohere Rerank(模拟)
3. 计算 Precision@5, Recall@10, MRR, NDCG@10 的提升
4. 测量每种 reranker 的延迟
5. 分析延迟-质量的权衡

**输出**：
```python
{
    "baseline": {"P@5": 0.62, "latency": 45},
    "bge_reranker": {"P@5": 0.78, "latency": 320, "improvement": "+25.8%"},
    "cross_encoder": {"P@5": 0.74, "latency": 80, "improvement": "+19.4%"},
    "cohere_rerank": {"P@5": 0.76, "latency": 150, "improvement": "+22.6%"},
}
```

**评分标准**：
- 测试集和标注准备: 2 分
- 准确率指标计算: 2 分
- 延迟测量: 1 分
- 对比分析: 2 分
- 延迟-质量权衡建议: 1 分
- 代码可复现: 2 分

---

## 练习 5: 实现父子切片检索

**目标**：实现 Two-Stage Parent-Child Retrieval。

**要求**：
1. 实现 `ChunkHierarchyBuilder`：将文档拆分为 parent (1024 tokens) + child (256 tokens) 两层
2. 实现 `ParentChildRetriever`：
   - Stage 1: 用 child chunks 精确检索
   - Stage 2: 映射到 parent chunks 获取完整上下文
3. 实现 `get_full_context()`: 返回 parent 文本 + sibling chunks
4. 与标准检索对比：precision、context coverage
5. 处理边界情况：单 chunk 文档、超短文

**测试**：
```python
builder = ChunkHierarchyBuilder(parent_size=1024, child_size=256)
chunks = builder.build("doc1", long_document)

retriever = ParentChildRetriever(base_retriever)
results = retriever.search("query", final_top_k=3)

assert len(results) <= 3
assert all(r.parent_chunk is not None for r in results)
context = retriever.get_full_context(results[0])
assert len(context) > len(results[0].child_chunk.text)
```

**评分标准**：
- 层级构建正确: 2 分
- 两阶段检索实现: 2 分
- 上下文扩展: 2 分
- 与标准检索对比: 2 分
- 边界情况处理: 2 分

---

## 练习 6: 实现相似度阈值网格搜索

**目标**：找到最优相似度阈值。

**要求**：
1. 准备 30+ 查询和相关性标注的验证集
2. 实现 `grid_search()`：在 [0.50, 0.95] 范围内以 0.05 步长搜索
3. 绘制 Precision-Recall 曲线和 F1-Threshold 曲线
4. 找到 F1-maximizing threshold
5. 测试通用和专业场景的不同最优值

**输出**：
```
General Scenario:
  Best Threshold: 0.75
  Best F1: 0.83

Professional Scenario:
  Best Threshold: 0.85
  Best F1: 0.79
```

**评分标准**：
- 验证集构建: 2 分
- 网格搜索实现: 2 分
- 可视化: 2 分
- 场景对比分析: 2 分
- 代码可复现: 2 分

---

## 练习 7: 实现语义切分与可视化

**目标**：实现并可视化基于相似度低谷的语义切分。

**要求**：
1. 实现 `split_sentences()` - 支持中英文
2. 实现 `compute_sentence_similarities()` - 使用 sentence-transformers
3. 实现 `detect_boundaries()` - 相似度低于阈值的分割点
4. 实现 `build_hierarchy()` - 三级层次（parent/middle/child）
5. 实现 `visualize_chunk_boundaries()` - matplotlib 可视化
6. 测试通用和专业两种配置

**测试**：
```python
chunker = SemanticChunker(scenario=Scenario.GENERAL)
chunks = chunker.chunk(sample_text)

# 验证层级
parents = [c for c in chunks if c.level == ChunkLevel.PARENT]
middles = [c for c in chunks if c.level == ChunkLevel.MIDDLE]
children = [c for c in chunks if c.level == ChunkLevel.CHILD]

assert len(parents) > 0
assert len(middles) > len(parents)
```

**评分标准**：
- 句子分割: 1 分
- 相似度计算: 2 分
- 边界检测: 2 分
- 三级层次构建: 2 分
- 可视化: 2 分
- 双场景支持: 1 分

---

## 练习 8: 实现 Dynamic Top-K

**目标**：实现基于查询复杂度的动态 K 值选择。

**要求**：
1. 实现 `QueryComplexityScorer` - 多维度评分
2. 实现 `TopKMapper` - 复杂度到 K 值的映射
3. 实现 `CliffDetector` - 相似度悬崖检测
4. 实现三种策略（scoring / cliff / hybrid）并能切换
5. 测试简单查询（K=3）到非常复杂查询（K=12）的渐进变化

**测试**：
```python
dtk = DynamicTopK(method="hybrid")

simple_k = dtk.calculate("什么是AI?", scores=[...])
assert 3 <= simple_k.recommended_k <= 5

complex_k = dtk.calculate(
    "为什么模型部署后延迟高且精度下降，该如何诊断和优化？",
    scores=[...]
)
assert 8 <= complex_k.recommended_k <= 12
```

**评分标准**：
- 复杂度评分: 2 分
- K 值映射: 2 分
- 悬崖检测: 2 分
- 三种策略实现: 2 分
- 边界测试: 2 分

---

## 练习 9: 实现 Lost-in-Middle 排序

**目标**：实现位置感知的重排序。

**要求**：
1. 实现 `PositionDetector` - 区分 beginning/early_middle/middle/late_middle/end
2. 实现 `LostInMiddleAdjuster` - 中间位置提升，多样化惩罚
3. 实现 `LongContextReorder` - U 形重排序
4. 对比重排序前后的区域分布
5. 分析对检索多样性的影响

**测试**：
```python
pipeline = LostInMiddlePipeline()
adjusted, stats = pipeline.process(query, chunks, top_k=10)

# 中间区域应该有更好的代表性
assert stats['middle_promoted'] > 0
assert len(stats['zone_distribution']) >= 3  # 至少覆盖 3 个区域
```

**评分标准**：
- 位置检测: 2 分
- 分数调整逻辑: 2 分
- U 形重排序: 2 分
- 前后对比: 2 分
- 多样性分析: 2 分

---

## 练习 10: 构建端到端 RAG 检索评估框架

**目标**：构建一个完整的评估框架，测量所有优化技术叠加后的整体提升。

**要求**：
1. 构建评估数据集（50+ 文档，20+ 标注查询）
2. 实现基线评估（无优化）
3. 逐步叠加优化技术并测量提升：
   - +数据清洗
   - +语义切分
   - +元数据过滤
   - +混合检索
   - +查询优化
   - +重排序
   - +Lost-in-Middle
   - +父子切片
4. 绘制累积提升曲线
5. 分析每个技术的边际贡献

**输出**：
```python
{
    "baseline": {"P@5": 0.58, "MRR": 0.52},
    "+cleaning": {"P@5": 0.62, "improvement": "+6.9%"},
    "+chunking": {"P@5": 0.66, "improvement": "+6.5%"},
    "+metadata": {"P@5": 0.69, "improvement": "+4.5%"},
    "+hybrid": {"P@5": 0.76, "improvement": "+10.1%"},
    "+query_opt": {"P@5": 0.80, "improvement": "+5.3%"},
    "+reranking": {"P@5": 0.85, "improvement": "+6.3%"},
    "+lost_middle": {"P@5": 0.87, "improvement": "+2.4%"},
    "+parent_child": {"P@5": 0.90, "improvement": "+3.4%"},
}
```

**评分标准**：
- 评估数据集构建: 2 分
- 基线评估: 1 分
- 逐步叠加测试: 3 分
- 累积提升分析: 2 分
- 边际贡献分析: 2 分

---

## 评分汇总

| 练习 | 主题 | 分值 |
|------|------|------|
| 1 | 混合检索器实现 | 10 |
| 2 | RRF 权重调优 | 10 |
| 3 | 查询优化器构建 | 10 |
| 4 | Reranker 效果评估 | 10 |
| 5 | 父子切片检索 | 10 |
| 6 | 相似度阈值调优 | 10 |
| 7 | 语义切分与可视化 | 10 |
| 8 | Dynamic Top-K | 10 |
| 9 | Lost-in-Middle 排序 | 10 |
| 10 | 端到端评估框架 | 10 |
| **总计** | | **100** |

**通过标准**：70 分以上
**优秀标准**：85 分以上
