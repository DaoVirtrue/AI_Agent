# Phase 10: 评估与监控 (Evaluation & Monitoring)

## 周期: 2周

## 核心理念

从 "it seems to work" 到 "I can PROVE it works with statistical rigor."

评估不是事后行为，而是持续的过程。本阶段构建从离线评估到在线监控的完整体系，确保RAG系统在统计意义上可证明地工作。

## 学习目标

1. 掌握RAG系统的六维度评估框架
2. 能够使用RAGAS等专业工具进行自动化评估
3. 理解并实现检索质量指标
4. 构建金标准测试数据集
5. 实施A/B测试与统计验证
6. 检测数据和概念漂移
7. 集成LangSmith进行全链路追踪
8. 部署Prometheus + Grafana在线监控
9. 建立持续改进闭环

## 文件清单 (9个主要文件 + pitfalls + checkpoint)

| 文件 | 描述 | 技能 |
|------|------|------|
| 01-six-dimension-evaluation.py | 六维度评估框架 (准确度/相关性/忠实度/流畅度/延迟/成本) | 核心评估 |
| 02-ragas-evaluation.ipynb | RAGAS库集成 (Faithfulness, AnswerRelevancy, ContextPrecision等) | 专业工具 |
| 03-retrieval-metrics.py | 检索指标 (Hit Rate@K, MRR, NDCG, Recall, Precision) | 检索评估 |
| 04-golden-dataset-construction.py | 金标准数据集构建 (人工标注/合成生成/生产挖掘) | 数据工程 |
| 05-ab-testing-framework.py | A/B测试框架 (Welch's t-test, Cohen's d, 序贯测试) | 统计验证 |
| 06-drift-detection.py | 漂移检测 (MMD, KS检验, z-score, 健康状态) | 模型监控 |
| 07-langsmith-tracing.ipynb | LangSmith全链路追踪与实验对比 | 可观测性 |
| 08-online-monitoring.py | Prometheus指标 + Grafana面板 + 告警规则 | 生产监控 |
| 09-iteration-closed-loop.py | 持续改进闭环 (评估→分析→门控→记录) | 流程自动化 |

## 前置要求

- Phase 05 (高级RAG) - RAG系统基础
- Phase 08 (高级Agent) - Agent架构理解
- Phase 09 (生产RAG) - 生产环境概念

## 工具与框架

- RAGAS (RAG评估框架)
- trulens-eval (LLM评估)
- LangSmith (可观测性平台)
- Prometheus + Grafana (监控可视化)
- SciPy (统计分析)

## 里程碑

**Week 1**: 完成线下评估体系 (六维度 + RAGAS + 检索指标 + 金标准数据集)
**Week 2**: 完成在线监控 + A/B测试 + 漂移检测 + 持续改进闭环
