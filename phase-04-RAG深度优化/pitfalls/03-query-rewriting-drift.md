# Pitfall 03: 查询重写导致语义漂移 (Query Rewriting Drift)

## 严重程度：中

## 症状

- 重写后的查询返回与原始意图不相关的结果
- LLM 在补充上下文时添加了原文未提及的信息
- 重写查询包含假设性内容，导致检索偏向特定方向
- 用户说"我问的不是这个"

## 根本原因

1. **LLM 过度扩展**：LLM 将简短的模糊查询扩展时，基于自身知识添加了额外信息，但这些信息并非用户意图。例如"Transformer 性能怎么样？"被重写为"请分析 Transformer 在 GPU TPU 和 CPU 上的性能对比"——但用户可能只想了解基本情况。

2. **缺少语义保真度检查**：没有比较原查询和重写查询的语义相似度。当语义漂移 > 20% 时仍接受了重写结果。

3. **领域术语误替换**：LLM 将用户的口语化术语替换为"规范"术语，但改变了含义。"模型跑不动"被重写为"模型推理速度慢"，但用户可能指的是"模型训练不收敛"。

4. **Few-shot 示例过拟合**：重写提示词中的少样本示例引入了特定领域的偏见，导致 LLM 倾向于按特定模式重写，而不是忠实反映用户意图。

## 真实场景

某电商客服 RAG 系统集成了 LLM 查询重写：

用户查询："这个商品怎么退？"
LLM 重写："请说明商品的退货流程、退款条件、退货地址、运费承担方、退款时效和客服联系方式。"

原始查询只是一个简单的退货流程咨询，但重写后添加了大量用户并未询问的细节（运费、地址、联系方式）。检索返回了包含所有这些内容的综合页面，但用户只想要简单的 3 步退货流程，结果信息过载。

更严重的一例："iPhone 15 好用吗？"
被重写为："请从性能、摄像头、续航、价格、生态系统等五个维度对比分析 iPhone 15 的优劣。"——用户原本只想看一条简单的使用体验总结。

## 解决方案

### 1. 语义保真度检查

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticFidelityChecker:
    def __init__(self, threshold: float = 0.8, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

    def check(self, original: str, rewritten: str) -> tuple[bool, float]:
        embeddings = self.model.encode([original, rewritten])
        similarity = float(cosine_similarity(
            embeddings[0:1], embeddings[1:2]
        )[0][0])

        is_similar = similarity >= self.threshold
        if not is_similar:
            logger.warning(f"Rewrite rejected: similarity={similarity:.3f} < {self.threshold}")
            logger.warning(f"  Original:  {original}")
            logger.warning(f"  Rewritten: {rewritten}")

        return is_similar, similarity
```

### 2. 分层重写策略

```python
class TieredRewriter:
    """根据查询复杂度选择不同的重写策略。"""
    def rewrite(self, query: str) -> str:
        complexity = self._assess_complexity(query)

        if complexity == 'simple':
            # 简单查询：最小化重写，只做清理
            return self._clean_only(query)
        elif complexity == 'moderate':
            # 中等查询：模板化重写
            return self._template_rewrite(query)
        else:
            # 复杂查询：LLM 重写 + 保真度检查
            rewritten = self._llm_rewrite(query)
            is_similar, sim = self.fidelity_checker.check(query, rewritten)
            if not is_similar:
                return self._template_rewrite(query)  # Fallback
            return rewritten
```

### 3. 重写约束提示词

```python
REWRITE_PROMPT = """Rewrite the query for better retrieval.
CRITICAL RULES:
1. DO NOT add information not implied by the original query
2. DO NOT assume user intent beyond what is stated
3. DO NOT replace terms that might change the meaning
4. ONLY complete missing subjects/objects from obvious context
5. Maintain the original scope — don't broaden or narrow

Original: {query}
Rewritten (FAITHFUL to original):"""
```

## 检查清单

- [ ] 是否对所有重写结果执行了语义保真度检查？
- [ ] 保真度阈值是否根据场景调整（通用 0.8，专业 0.85）？
- [ ] 是否记录了被拒绝的重写结果及其原因？
- [ ] 是否监控了重写拒绝率（理想 < 10%）？
- [ ] 简单查询是否跳过了 LLM 重写？
- [ ] Few-shot 示例是否涵盖了用户查询的多样性？
- [ ] 是否保留了原始查询作为 fallback？
