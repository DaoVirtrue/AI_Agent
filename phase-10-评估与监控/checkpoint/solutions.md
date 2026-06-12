# Phase 10 练习题解答 - 评估与监控

## 练习 1 解答: 六维度评估实践

```python
# solution_ex1.py
from multi_dimension_evaluator import MultiDimensionEvaluator, EvaluationReporter

# 初始化评估器
evaluator = MultiDimensionEvaluator()

# 测试用例
test_cases = [
    {
        'query': '什么是RAG？',
        'response': 'RAG是一种结合检索和生成的AI技术。它从知识库中检索文档，然后基于这些文档生成回答。',
        'reference': 'RAG(检索增强生成)是一种AI架构，通过检索外部知识来增强大语言模型的生成能力。',
        'key_facts': ['RAG结合检索和生成', '从知识库检索文档', '增强语言模型', '减少幻觉'],
        'context': 'RAG是一种AI技术，它将信息检索与文本生成结合起来...',
        'latency': 1.5,
        'cost': 0.003,
    },
    # ... 更多测试用例
]

scores = []
for case in test_cases:
    score = evaluator.evaluate(
        query=case['query'],
        response=case['response'],
        reference=case['reference'],
        key_facts=case['key_facts'],
        context=case['context'],
        latency_seconds=case['latency'],
        cost_dollars=case['cost'],
    )
    scores.append(score)

# 生成报告
report = EvaluationReporter.generate_report(scores, "RAG系统评估")
print(report)

# 分析最低分维度
avg_metrics = {
    'accuracy': sum(s.accuracy for s in scores) / len(scores),
    'relevance': sum(s.relevance for s in scores) / len(scores),
    'faithfulness': sum(s.faithfulness for s in scores) / len(scores),
    'fluency': sum(s.fluency for s in scores) / len(scores),
    'latency': sum(s.latency_score for s in scores) / len(scores),
    'cost': sum(s.cost_score for s in scores) / len(scores),
}
weakest = min(avg_metrics, key=avg_metrics.get)
print(f"\n最需要改进的维度: {weakest} (得分: {avg_metrics[weakest]:.3f})")
print(f"改进建议: ...")
```

**关键点**:
- 参考答案和关键事实是准确度评估的锚点
- 没有参考答案时准确度评估不可靠
- 延迟和成本需要实际测量值

---

## 练习 2 解答: RAGAS指标计算

```python
# solution_ex2.py

# 1. Faithfulness计算
from ragas_simulator import SimFaithfulness

claims = SimFaithfulness.decompose_claims(answer)
# claims = ["向量数据库的检索速度快",
#           "能够存储大量数据",
#           "机器学习模型可以直接使用向量数据库进行训练"]

# 验证每个声明
for claim in claims:
    verified = SimFaithfulness.verify_claim(claim, ' '.join(contexts))
    print(f"  '{claim}': {'✓ 验证通过' if verified else '✗ 无法验证'}")

# 结果:
# "向量数据库的检索速度快" → ✓ (出现在contexts中)
# "能够存储大量数据" → ✗ (contexts未提到"大量数据")
# "机器学习模型可以直接使用向量数据库进行训练" → ✗ (幻觉!)

# faithfulness = 1/3 = 0.333

# 2. AnswerRelevancy
questions_from_answer = SimAnswerRelevancy.generate_questions_from_answer(answer)
# 生成: ["什么是向量数据库？", "向量数据库有什么特点？"]
similarities = [SimAnswerRelevancy.jaccard_similarity(gq, original_question) for gq in questions_from_answer]
# answer_relevancy = mean(similarities)

# 3. ContextPrecision
cp = SimContextPrecision.score(contexts, question, reference)
# 两个上下文中只有一个与问题+参考答案高度相关
# precision ≈ 0.5 (取决于关键词重叠)

# 4. ContextRecall
cr = SimContextRecall.score(contexts, reference)
# 参考答案的句子在上下文中覆盖了多少?
```

**分析**:
- Faithfulness低是因为回答包含了上下文之外的声明(幻觉)
- ContextRecall低表明检索可能遗漏了相关信息
- 回答不完全基于上下文

---

## 练习 3 解答: 检索质量对比

```python
# solution_ex3.py
from retrieval_metrics import *

# 生成测试数据
results_a = TestDataGenerator.generate_random_results(
    num_queries=50, retrieval_quality=0.65
)
results_b = TestDataGenerator.generate_random_results(
    num_queries=50, retrieval_quality=0.85
)

# 计算指标
report_a = RetrievalMetrics.compute_all_metrics(results_a)
report_b = RetrievalMetrics.compute_all_metrics(results_b)

# 统计显著性
# 对HitRate@5进行t-test
a_hr5 = [1 if r.is_relevant(r.retrieved_doc_ids[0]) else 0 for r in results_a]
b_hr5 = [1 if r.is_relevant(r.retrieved_doc_ids[0]) else 0 for r in results_b]

t_stat, p_val, df, mean_a, mean_b = ABTestFramework.welch_t_test(a_hr5, b_hr5)
# 如果p<0.05，系统B显著优于A

# 可视化
MetricsVisualizer.plot_metrics_comparison(
    report_a, "BM25 (基线)",
    report_b, "混合检索 (BM25+向量)",
)
```

---

## 练习 4 解答: 构建金标准数据集

```python
# solution_ex4.py
from golden_dataset_builder import GoldenDatasetBuilder

builder = GoldenDatasetBuilder("my_golden_dataset")

# 1. 人工标注 (10条)
human_annotations = [
    {'query': '...', 'reference_answer': '...', 'key_facts': [...], ...},
    # ... 10条
]
builder.add_human_annotations(human_annotations)

# 2. 合成生成 (10条)
documents = [
    {'doc_id': 'd1', 'title': '...', 'content': '...'},
    # ... 3篇文档
]
builder.generate_synthetic(documents, questions_per_doc=4)

# 3. 生产日志挖掘 (10条)
logs = [
    {'query': '...', 'response': '...', 'feedback_score': 0.9},
    # ... 满足正面反馈条件
]
builder.mine_production_logs(logs)

# 4. 对抗样本 (3条)
easy_queries = [q for q in builder.queries if q.difficulty == DifficultyLevel.EASY]
builder.generate_adversarial_variants(easy_queries[:3], count_per_query=1)

# 构建
dataset = builder.build(target_size=30)
dataset.save("my_golden_dataset.json")
```

---

## 练习 5 解答: A/B测试

```python
# solution_ex5.py
from ab_testing_framework import *

# 1. 样本量计算
n_required = ABTestFramework.calculate_sample_size(
    baseline_mean=0.75, baseline_std=0.08,
    minimum_detectable_effect=0.03,
    significance_level=0.05, power=0.80,
)
print(f"每变体需要 {n_required} 样本")

# 2. 生成模拟数据
import random
random.seed(42)

n = max(n_required, 500)  # 确保足够
baseline_mean = 0.75
effect = 0.04  # 4%提升

group_a = [baseline_mean + random.gauss(0, 0.08) for _ in range(n)]
group_b = [baseline_mean + effect + random.gauss(0, 0.08) for _ in range(n)]

# 3. 执行测试
result = run_ab_test(group_a, group_b, "增强RAG vs 标准RAG")

# 4. 结论
print(f"结论: {result.summary()}")
if result.is_significant():
    if result.relative_lift > 0:
        print("✓ 增强RAG显著优于标准RAG，建议部署")
    else:
        print("✗ 增强RAG表现不如标准RAG")
else:
    print("= 无显著差异")
```

---

## 练习 6 解答: 漂移检测

```python
# solution_ex6.py
from drift_detector import DriftDetector
import random

detector = DriftDetector()

# 设置基线
baseline = [0.75 + random.gauss(0, 0.03) for _ in range(200)]
detector.set_reference("accuracy", baseline)

# 注入漂移数据
recent_normal = [0.75 + random.gauss(0, 0.04) for _ in range(50)]
recent_drifted = [0.55 + random.gauss(0, 0.06) for _ in range(50)]

# 检查正常数据
for val in recent_normal:
    detector.current_window["accuracy"].append(val)

dim_alerts = detector.check_data_drift("accuracy", list(detector.current_window["accuracy"]))
print(f"正常数据: 检测到{len(dim_alerts)}个漂移告警")

# 检查漂移数据
detector.current_window["accuracy"].clear()
for val in recent_drifted:
    detector.current_window["accuracy"].append(val)

dim_alerts = detector.check_data_drift("accuracy", list(detector.current_window["accuracy"]))
print(f"漂移数据: 检测到{len(dim_alerts)}个漂移告警")
for alert in dim_alerts:
    print(f"  - {alert.message}")

# 诊断
# MMD: 高值表明分布差异大
# KS: 低p值表明两个分布显著不同
# z-score: 超过2.576表明概念漂移
```

---

## 练习 7 解答: 在线监控设计

```python
# solution_ex7.py
from online_monitoring import *

registry = MetricsRegistry()
metrics = SkillMetricsManager(registry)

# 导出指标
prometheus_metrics = metrics.get_all_metrics()

# Grafana Dashboard
dashboard = GrafanaDashboardBuilder.build_rag_dashboard_json()

# 告警规则
rules = GrafanaDashboardBuilder.build_alert_rules()

# 告警升级策略:
"""
P0 (服务不可用):
  - 立即触发PagerDuty
  - 5分钟内无响应 → 电话通知
  - 10分钟内无响应 → 通知工程VP

P1 (核心功能受损):
  - PagerDuty + Slack
  - 15分钟内响应

P2 (非核心异常):
  - Slack通知
  - 1小时内处理或降级

P3 (性能下降):
  - 工作日报表
  - 下一个工作日处理

P4 (信息):
  - 仅记录日志
  - 每周汇总审查
"""
```

---

## 练习 8 解答: 持续改进闭环

```python
# solution_ex8.py
from iteration_closed_loop import *

pipeline = ContinuousImprovementPipeline("rag_improvements")
pipeline.set_default_thresholds()

# 模拟评估函数
def evaluate_system(config) -> Dict[str, float]:
    # 这里连接实际的RAG系统评估
    return {
        'weighted_score': 0.78,
        'accuracy': 0.76,
        'relevance': 0.80,
        'faithfulness': 0.82,
        ...
    }

# 三个优化周期
for cycle_num, changes in enumerate([
    ["升级嵌入模型到text-embedding-3-large"],
    ["添加Cross-Encoder重排序"],
    ["优化提示词模板"],
], start=1):
    result = pipeline.run_cycle(
        evaluator_fn=evaluate_system,
        changes_made=changes,
        notes=f"第{cycle_num}轮优化",
    )
    print(f"周期{cycle_num}: {result['decision']}")

# 生成报告
report = pipeline.generate_improvement_report()
print(report)

# 导出历史
history = pipeline.export_history()
```

**检查PASS条件**:
- 所有阈值检查通过 → PASS
- 与基线无显著回归 → PASS
- 有改进趋势 → 更新基线
