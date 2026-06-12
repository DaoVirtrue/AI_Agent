# Pitfall 04: 意图分类错误导致检索策略偏差 (Intent Classification Errors)

## 严重程度：中

## 症状

- 用户的流程查询（"如何部署模型？"）被误分类为事实查询，检索返回模型定义而非部署步骤
- 精确度不足：错误推断了检索参数（Top-K、权重偏好）
- 部分场景下系统表现良好，但在特定意图类别上持续表现差
- 日志显示大量查询被分到了"未知"或默认类别

## 根本原因

1. **规则匹配的局限性**：基于正则表达式的意图分类只能匹配显式关键词模式，无法理解隐含意图。例如"模型部署需要注意什么"——包含"什么"被分为 FACT，但实际是 PROCESS。

2. **查询简短模糊**：短查询（3-5 字）包含的信息不足以确定意图。"BERT 和 GPT"——是问区别（COMPARISON）、定义（FACT）还是选择建议（PROCESS）？分类器只能猜测。

3. **中英混合查询**：正则规则通常针对单一语言，中英混合查询（"BERT fine-tune 怎么做？"）可能同时触发多个意图模式，导致分类不确定。

4. **领域特有表达方式**：法律领域"如何认定"是 FACT（法律认定标准），但分类规则会将其归为 PROCESS（"如何"）。

5. **缺少多意图处理**：复杂查询包含多个意图（"什么是 X 以及如何用它解决 Y"），但分类器只返回单一意图。

## 真实场景

某技术文档 RAG 系统使用规则分类器处理用户查询：

```python
# 问题：规则过于简单，无法区分意图边界
user_query = "Kubernetes Pod 一直 CrashLoopBackOff 怎么办？"
# 触发: "怎么办" → PROCESS (错误)
# 实际: 故障排查 → TROUBLESHOOT (正确)

# 错误分类导致：
# - Retrieval strategy: top_k=5, prefer_dense=False (PROCESS 策略)
# - 正确策略应该是: top_k=8, prefer_dense=True (TROUBLESHOOT 策略)
# - 结果: 检索到了 K8s Pod 生命周期文档，而非故障排查文档
```

另一个案例：
```python
user_query = "民法典关于合同无效的规定"
# 触发: "关于" → PROCESS (错误)  
# 实际: FACT (查找具体法条)
# 错误分类导致稀疏检索权重过高，返回了大量不相关的合同条款
```

## 解决方案

### 1. 混合分类：规则 + 语义

```python
class HybridIntentClassifier:
    def __init__(self):
        self.rule_classifier = RuleBasedIntentClassifier()
        self.embedding_classifier = EmbeddingIntentClassifier()

    def classify(self, query: str) -> tuple[IntentType, float]:
        # 规则分类作为基线
        rule_intent, rule_conf = self.rule_classifier.classify(query)

        # 语义分类作为补充
        emb_intent, emb_conf = self.semantic_classifier.classify(query)

        # 融合：如果两者一致，高置信度
        if rule_intent == emb_intent:
            return rule_intent, max(rule_conf, emb_conf)

        # 不一致时：选择置信度更高的
        if rule_conf > emb_conf:
            return rule_intent, rule_conf
        else:
            return emb_intent, emb_conf

class EmbeddingIntentClassifier:
    """使用嵌入相似度匹配的意图分类器"""
    INTENT_EXAMPLES = {
        IntentType.FACT: [
            "什么是人工智能？",
            "请解释机器学习的定义",
            "民法典第几条规定了什么？",
        ],
        IntentType.PROCESS: [
            "如何训练模型？",
            "部署步骤是什么？",
            "怎么配置环境？",
        ],
        # ... more examples per intent
    }

    def classify(self, query: str) -> tuple[IntentType, float]:
        query_emb = self.model.encode([query])
        best_intent = None
        best_score = -1

        for intent, examples in self.INTENT_EXAMPLES.items():
            example_embs = self.model.encode(examples)
            sims = cosine_similarity(query_emb, example_embs)[0]
            avg_sim = float(np.mean(sims))
            if avg_sim > best_score:
                best_score = avg_sim
                best_intent = intent

        return best_intent, best_score
```

### 2. 多意图检测与处理

```python
def detect_multi_intent(query: str) -> list[tuple[IntentType, float]]:
    """检测查询中的多个意图，返回排序列表。"""
    intents = []

    # 按分割点拆分查询
    segments = re.split(r'[；;，,]\s*(?:以及|还有|另外|同时)?\s*', query)

    for seg in segments:
        if len(seg) > 3:
            intent, conf = classifier.classify(seg)
            intents.append((intent, conf))

    # 如果拆分后只有一个意图，返回原查询的分类
    if len(intents) <= 1:
        return [(classifier.classify(query))]

    return sorted(intents, key=lambda x: x[1], reverse=True)
```

### 3. 意图混淆矩阵监控

```python
def build_confusion_matrix(predictions, ground_truth):
    """定期用标注数据检查意图分类准确率。"""
    matrix = defaultdict(lambda: defaultdict(int))
    for pred, actual in zip(predictions, ground_truth):
        matrix[actual][pred] += 1

    # 识别高错误率意图对
    for actual in matrix:
        total = sum(matrix[actual].values())
        for pred, count in matrix[actual].items():
            if pred != actual and count / total > 0.15:
                logger.warning(f"High confusion: {actual} → {pred} "
                             f"({count}/{total} = {count/total:.1%})")
```

## 意图分类优先级决策

```
首次查询 → 规则分类 (快速)
  ├── 置信度 > 0.8 → 使用规则结果
  ├── 置信度 0.5-0.8 → 规则 + 语义融合
  └── 置信度 < 0.5 → LLM 分类 (慢但准)

后续查询 → 多轮上下文增强
  ├── 结合历史意图修正
  └── 意图一致性检查 (不应频繁跳变)
```

## 检查清单

- [ ] 是否同时使用了规则和语义分类（融合策略）？
- [ ] 是否为每个意图类型定义了明确的边界和示例？
- [ ] 是否对多意图查询进行了检测和分解？
- [ ] 是否在标注验证集上评估了意图分类准确率（目标 > 85%）？
- [ ] 是否监控了意图分类的混淆矩阵？
- [ ] 是否对低置信度查询使用了更强大的分类方法（如 LLM）？
- [ ] 多轮对话中是否考虑历史意图？
- [ ] 是否正确处理了中英混合查询的分类？
