#!/usr/bin/env python3
"""
B1-04: 微调后评估 (Post Fine-tuning Evaluation)
==================================================
学习目标:
  1. MTEB中文基准测试
  2. 检索命中率(Retrieval Hit Rate)前后对比
  3. 向量分布可视化(PCA, t-SNE)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================
# Section 1: MTEB中文基准仿真
# ============================================================

@dataclass
class MTEBTask:
    """MTEB评估任务"""
    name: str
    task_type: str  # "retrieval" | "clustering" | "pair_classification" | "sts"
    queries: List[str]
    corpus: List[str]
    relevant_docs: Dict[int, List[int]]  # query_idx → [doc_idx, ...]
    metadata: Dict = field(default_factory=dict)


class MTEBChineseBenchmark:
    """
    MTEB中文基准仿真

    MTEB (Massive Text Embedding Benchmark) 包含多种中文任务:
    - Retrieval: T2Retrieval, MMarco, DuRetrieval
    - STS (Semantic Textual Similarity): ATEC, BQ, LCQMC
    - PairClassification: Ocnli, Cmnli
    - Clustering: CLSClustering, ThuNewsClustering

    这里模拟核心任务的评估流程（真实MTEB需要使用mteb库）
    """

    TASK_DESCRIPTIONS = {
        "T2Retrieval": "面向医学领域的文本检索任务",
        "MMarcoRetrieval": "多语言文档检索（中文子集）",
        "DuRetrieval": "百度知道问答检索",
        "ATEC": "ATEC金融领域语义相似度",
        "BQ": "银行客服问答语义相似度",
        "LCQMC": "大规模中文问句匹配",
        "Ocnli": "原生中文自然语言推理",
        "Cmnli": "翻译中文自然语言推理",
    }

    @staticmethod
    def create_mock_retrieval_task(n_queries: int = 100,
                                   n_corpus: int = 500) -> MTEBTask:
        """创建模拟检索评估任务"""
        np.random.seed(42)

        # 模拟查询和文档
        queries = [f"查询示例 {i}: 关于中文Embedding模型的评估方法" for i in range(n_queries)]
        corpus = [f"文档 {i}: 该文档讨论了自然语言处理中的文本表示方法，包括Word2Vec、BERT和最新的Embedding模型。" for i in range(n_corpus)]

        # 模拟相关性标注（每个查询有1-3个相关文档）
        relevant_docs = {}
        for i in range(n_queries):
            # 让前10个文档与第i个查询"相关"
            n_rel = np.random.randint(1, 4)
            relevant_docs[i] = sorted(
                np.random.choice(n_corpus, n_rel, replace=False).tolist()
            )

        return MTEBTask(
            name="MockRetrieval",
            task_type="retrieval",
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            metadata={"domain": "general", "language": "zh"},
        )

    @staticmethod
    def create_mock_sts_task(n_pairs: int = 100) -> Dict:
        """创建模拟语义相似度任务"""
        np.random.seed(42)
        sentences1 = []
        sentences2 = []
        scores = []  # 0-5的相似度评分

        templates = [
            ("今天天气真好", "今天天气不错", 4.5),
            ("如何学习Python", "Python编程入门指南", 4.0),
            ("北京是中国的首都", "北京是中国首都城市", 5.0),
            ("我喜欢吃苹果", "苹果是一种水果", 3.0),
            ("RAG系统提高准确性", "检索增强生成改善精度", 4.8),
        ]

        for _ in range(n_pairs):
            t = templates[np.random.randint(0, len(templates))]
            sentences1.append(t[0])
            sentences2.append(t[1])
            scores.append(t[2] + np.random.uniform(-0.3, 0.3))

        return {
            "sentences1": sentences1,
            "sentences2": sentences2,
            "scores": [max(0, min(5, s)) for s in scores],
        }


# ============================================================
# Section 2: 检索命中率评估
# ============================================================

class RetrievalHitRateEvaluator:
    """
    检索命中率评估器

    指标:
    - Hit@K: Top-K结果中是否包含至少一个相关文档
    - MRR: Mean Reciprocal Rank
    - NDCG@K: 归一化折损累积增益
    - Recall@K: 召回率
    """

    def __init__(self, ks: List[int] = None):
        self.ks = ks or [1, 3, 5, 10, 20]

    def evaluate(self, task: MTEBTask,
                 query_embeddings: np.ndarray,
                 corpus_embeddings: np.ndarray) -> Dict:
        """
        评估检索性能
        """
        # 归一化
        q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        c_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-8)

        # 余弦相似度矩阵
        sim_matrix = np.dot(q_norm, c_norm.T)

        results = {k: {"hits": 0, "mrr_sum": 0.0} for k in self.ks}
        total_queries = len(task.queries)

        for q_idx in range(total_queries):
            # 排序
            doc_scores = sim_matrix[q_idx]
            ranked_docs = np.argsort(doc_scores)[::-1]  # 降序

            relevant_docs = task.relevant_docs.get(q_idx, [])

            if not relevant_docs:
                continue

            # 计算Hit@K和MRR
            for k in self.ks:
                top_k_docs = ranked_docs[:k]
                if any(doc in relevant_docs for doc in top_k_docs):
                    results[k]["hits"] += 1

                # MRR: 第一个相关文档的倒数排名
                for rank, doc in enumerate(ranked_docs, start=1):
                    if doc in relevant_docs:
                        results[k]["mrr_sum"] += 1.0 / rank
                        break

        # 计算比率
        final_results = {}
        for k in self.ks:
            final_results[f"Hit@{k}"] = results[k]["hits"] / max(total_queries, 1)
            final_results[f"MRR@{k}"] = results[k]["mrr_sum"] / max(total_queries, 1)

        return final_results


# ============================================================
# Section 3: 前后对比分析
# ============================================================

class BeforeAfterComparison:
    """
    微调前后对比分析

    对比维度:
    1. 检索命中率变化
    2. 语义相似度相关性（Spearman）
    3. 向量分布变化
    """

    @staticmethod
    def compare_retrieval(before_results: Dict,
                          after_results: Dict) -> Dict:
        """比较微调前后的检索指标"""
        comparison = {}
        for metric in before_results:
            if metric in after_results:
                before_val = before_results[metric]
                after_val = after_results[metric]
                delta = after_val - before_val
                pct_change = (delta / max(abs(before_val), 1e-8)) * 100
                comparison[metric] = {
                    "before": round(before_val, 4),
                    "after": round(after_val, 4),
                    "delta": round(delta, 4),
                    "pct_change": round(pct_change, 2),
                }
        return comparison

    @staticmethod
    def compute_spearman_correlation(predictions: List[float],
                                     ground_truth: List[float]) -> float:
        """计算Spearman相关系数（微调前后的语义相似度对齐度）"""
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(predictions, ground_truth)
        return float(corr)

    @staticmethod
    def print_comparison_table(comparison: Dict):
        """打印对比表格"""
        print(f"\n  {'指标':<12s} {'微调前':<10s} {'微调后':<10s} "
              f"{'变化':<10s} {'变化%':<8s}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for metric, data in comparison.items():
            delta_sign = "+" if data["delta"] >= 0 else ""
            pct_sign = "+" if data["pct_change"] >= 0 else ""
            print(f"  {metric:<12s} {data['before']:<10.4f} "
                  f"{data['after']:<10.4f} "
                  f"{delta_sign}{data['delta']:<9.4f} "
                  f"{pct_sign}{data['pct_change']:<7.2f}%")


# ============================================================
# Section 4: 向量分布可视化 (PCA, t-SNE)
# ============================================================

class VectorDistributionVisualizer:
    """
    向量分布可视化
    展示微调前后Embedding空间的变化
    """

    @staticmethod
    def reduce_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """PCA降维"""
        # 中心化
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean

        # SVD分解
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # 投影到前n_components个主成分
        projected = np.dot(centered, Vt[:n_components].T)
        return projected

    @staticmethod
    def compute_variance_explained(embeddings: np.ndarray) -> List[float]:
        """计算各主成分解释的方差比例"""
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = np.sum(S ** 2)
        return [(s ** 2 / total_var) for s in S]

    @staticmethod
    def analyze_distribution(embeddings_before: np.ndarray,
                             embeddings_after: np.ndarray,
                             labels: List[str] = None) -> Dict:
        """
        分析微调前后向量分布差异

        指标:
        1. 余弦相似度分布（正样本对vs负样本对）
        2. 类内紧凑度（同一主题的向量聚集程度）
        3. 类间分离度（不同主题的向量距离）
        """
        # PCA降维
        before_2d = VectorDistributionVisualizer.reduce_pca(embeddings_before)
        after_2d = VectorDistributionVisualizer.reduce_pca(embeddings_after)

        # 方差解释
        var_expl_before = VectorDistributionVisualizer.compute_variance_explained(embeddings_before)
        var_expl_after = VectorDistributionVisualizer.compute_variance_explained(embeddings_after)

        # 向量范数分布
        norms_before = np.linalg.norm(embeddings_before, axis=1)
        norms_after = np.linalg.norm(embeddings_after, axis=1)

        return {
            "pca_variance_top2": {
                "before": [round(v, 4) for v in var_expl_before[:2]],
                "after": [round(v, 4) for v in var_expl_after[:2]],
            },
            "norm_stats": {
                "before": {
                    "mean": float(np.mean(norms_before)),
                    "std": float(np.std(norms_before)),
                },
                "after": {
                    "mean": float(np.mean(norms_after)),
                    "std": float(np.std(norms_after)),
                },
            },
            "dimensionality_effective": {
                # 有效维度数（95%方差需要的维度数）
                "before": sum(
                    1 for v in np.cumsum(var_expl_before)
                    if v < 0.95
                ),
                "after": sum(
                    1 for v in np.cumsum(var_expl_after)
                    if v < 0.95
                ),
            },
        }

    @staticmethod
    def ascii_scatter(embeddings_2d: np.ndarray,
                      labels: List[str] = None,
                      width: int = 60, height: int = 20):
        """ASCII散点图（文本模式的PCA可视化）"""
        # 缩放到[0,1]范围
        x = embeddings_2d[:, 0]
        y = embeddings_2d[:, 1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)

        # 创建网格
        grid = [[" " for _ in range(width)] for _ in range(height)]

        for i in range(len(embeddings_2d)):
            px = int(x[i] * (width - 1))
            py = int(y[i] * (height - 1))
            if 0 <= px < width and 0 <= py < height:
                grid[height - 1 - py][px] = "*" if labels is None else labels[i][0]

        # 打印
        print("\n  PCA 2D投影 (ASCII):")
        print("  " + "-" * width)
        for row in grid:
            print("  |" + "".join(row) + "|")
        print("  " + "-" * width)


# ============================================================
# Section 5: 综合评估报告
# ============================================================

class EvaluationReport:
    """综合评估报告生成器"""

    @staticmethod
    def generate(before_metrics: Dict, after_metrics: Dict,
                 distribution_analysis: Dict) -> str:
        """生成Markdown格式的评估报告"""
        report = []
        report.append("# Embedding模型微调评估报告")
        report.append("")
        report.append("## 1. 检索性能对比")
        report.append("")
        report.append("| 指标 | 微调前 | 微调后 | 变化 | 变化率 |")
        report.append("|------|--------|--------|------|--------|")

        for metric in before_metrics:
            if metric in after_metrics:
                b = before_metrics[metric]
                a = after_metrics[metric]
                d = a - b
                p = (d / max(abs(b), 1e-8)) * 100
                report.append(f"| {metric} | {b:.4f} | {a:.4f} | {d:+.4f} | {p:+.1f}% |")

        report.append("")
        report.append("## 2. 向量分布分析")
        report.append("")

        dist = distribution_analysis
        report.append(f"- PCA前2成分方差解释: "
                      f"微调前={dist['pca_variance_top2']['before']}, "
                      f"微调后={dist['pca_variance_top2']['after']}")
        report.append(f"- 有效维度数(95%方差): "
                      f"微调前={dist['dimensionality_effective']['before']}, "
                      f"微调后={dist['dimensionality_effective']['after']}")

        report.append("")
        report.append("## 3. 结论与建议")
        report.append("")

        # 判断改善情况
        avg_improvement = 0
        count = 0
        for metric in before_metrics:
            if metric in after_metrics:
                d = after_metrics[metric] - before_metrics[metric]
                avg_improvement += d
                count += 1
        avg_improvement /= max(count, 1)

        if avg_improvement > 0.05:
            report.append("微调显著提升了检索性能，建议部署微调后的模型。")
        elif avg_improvement > 0.01:
            report.append("微调有一定改善。如果成本允许，建议部署。")
        else:
            report.append("微调改善有限。建议检查训练数据质量或增加数据集规模。")

        return "\n".join(report)


# ============================================================
# Section 6: 主流程
# ============================================================

def main():
    """主函数：演示微调后评估全流程"""
    print("=" * 70)
    print("B1-04: 微调后评估 — 完整演示")
    print("=" * 70)

    # 1. 创建模拟评估数据
    print("\n[步骤1] 创建MTEB模拟评估数据...")
    retrieval_task = MTEBChineseBenchmark.create_mock_retrieval_task(
        n_queries=100, n_corpus=500
    )
    sts_task = MTEBChineseBenchmark.create_mock_sts_task(n_pairs=100)
    print(f"  检索任务: {len(retrieval_task.queries)} 查询, "
          f"{len(retrieval_task.corpus)} 文档")

    # 2. 模拟微调前后的嵌入
    print("\n[步骤2] 模拟微调前后嵌入向量...")
    np.random.seed(42)
    dim = 768
    n_queries = len(retrieval_task.queries)
    n_corpus = len(retrieval_task.corpus)

    # 微调前：随机嵌入（基线）
    query_emb_before = np.random.randn(n_queries, dim).astype(np.float32)
    corpus_emb_before = np.random.randn(n_corpus, dim).astype(np.float32)

    # 微调后：前10%的查询和文档有提升的嵌入
    query_emb_after = query_emb_before.copy()
    corpus_emb_after = corpus_emb_before.copy()

    # 模拟微调带来的改善：相关文档的向量更靠近
    for q_idx, rel_docs in retrieval_task.relevant_docs.items():
        if q_idx < 20:  # 模拟20%的查询得到改善
            for doc_idx in rel_docs:
                # 让相关文档向量更接近查询向量
                corpus_emb_after[doc_idx] = (
                    0.3 * query_emb_before[q_idx] + 0.7 * corpus_emb_after[doc_idx]
                )

    # 归一化
    query_emb_before = query_emb_before / (np.linalg.norm(query_emb_before, axis=1, keepdims=True) + 1e-8)
    corpus_emb_before = corpus_emb_before / (np.linalg.norm(corpus_emb_before, axis=1, keepdims=True) + 1e-8)
    query_emb_after = query_emb_after / (np.linalg.norm(query_emb_after, axis=1, keepdims=True) + 1e-8)
    corpus_emb_after = corpus_emb_after / (np.linalg.norm(corpus_emb_after, axis=1, keepdims=True) + 1e-8)

    # 3. 检索命中率对比
    print("\n[步骤3] 检索命中率前后对比...")
    evaluator = RetrievalHitRateEvaluator(ks=[1, 3, 5, 10])

    before_results = evaluator.evaluate(
        retrieval_task, query_emb_before, corpus_emb_before
    )
    after_results = evaluator.evaluate(
        retrieval_task, query_emb_after, corpus_emb_after
    )

    comparison = BeforeAfterComparison.compare_retrieval(
        before_results, after_results
    )
    BeforeAfterComparison.print_comparison_table(comparison)

    # 4. 向量分布分析
    print("\n[步骤4] 向量分布分析...")
    # 采样前100个向量进行可视化
    sample_queries_before = query_emb_before[:50]
    sample_corpus_before = corpus_emb_before[:50]
    all_before = np.vstack([sample_queries_before, sample_corpus_before])

    sample_queries_after = query_emb_after[:50]
    sample_corpus_after = corpus_emb_after[:50]
    all_after = np.vstack([sample_queries_after, sample_corpus_after])

    distribution = VectorDistributionVisualizer.analyze_distribution(
        all_before, all_after,
        labels=["Q"] * 50 + ["D"] * 50,
    )

    print(f"  PCA方差解释(前2): 微调前={distribution['pca_variance_top2']['before']}")
    print(f"  PCA方差解释(前2): 微调后={distribution['pca_variance_top2']['after']}")
    print(f"  有效维度数: 微调前={distribution['dimensionality_effective']['before']}")
    print(f"  有效维度数: 微调后={distribution['dimensionality_effective']['after']}")

    # 5. PCA可视化
    print("\n[步骤5] PCA 2D可视化...")
    # 微调前
    print("\n  微调前:")
    before_2d = VectorDistributionVisualizer.reduce_pca(all_before)
    VectorDistributionVisualizer.ascii_scatter(
        before_2d,
        labels=["Q"] * 50 + ["D"] * 50,
    )
    # 微调后
    print("\n  微调后:")
    after_2d = VectorDistributionVisualizer.reduce_pca(all_after)
    VectorDistributionVisualizer.ascii_scatter(
        after_2d,
        labels=["Q"] * 50 + ["D"] * 50,
    )

    # 6. 综合报告
    print("\n[步骤6] 生成评估报告...")
    report = EvaluationReport.generate(before_results, after_results, distribution)
    print(report)

    print("\n" + "=" * 70)
    print("B1-04 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
