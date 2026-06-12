#!/usr/bin/env python3
"""
检索指标评估 (Retrieval Metrics)
=================================
从零实现核心检索评估指标:
  - Hit Rate @ K (1, 3, 5, 10)
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG@10)
  - Recall @ K
  - Precision @ K

包含与trulens-eval集成，以及对比可视化。
"""

import math
import json
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class RetrievalResult:
    """单次检索结果"""
    query_id: str
    query: str
    retrieved_doc_ids: List[str]  # 按相关性排序的检索文档ID
    relevant_doc_ids: Set[str]    # 相关文档ID集合
    relevance_scores: Optional[Dict[str, float]] = None  # 文档→相关性分数 (可选)

    def relevance_at(self, doc_id: str) -> float:
        """获取文档相关性

        Returns:
            0.0 (不相关), >0.0 (相关程度)
        """
        if self.relevance_scores and doc_id in self.relevance_scores:
            return self.relevance_scores[doc_id]
        return 1.0 if doc_id in self.relevant_doc_ids else 0.0

    def is_relevant(self, doc_id: str) -> bool:
        """检查文档是否相关"""
        return doc_id in self.relevant_doc_ids


@dataclass
class MetricsReport:
    """指标报告"""
    hit_rate: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: Dict[int, float] = field(default_factory=dict)
    recall: Dict[int, float] = field(default_factory=dict)
    precision: Dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0
    num_queries: int = 0


# ============================================================================
# 核心检索指标
# ============================================================================

class RetrievalMetrics:
    """检索指标计算器

    所有指标从零实现，不依赖外部评估库。
    支持二值相关性 (相关/不相关) 和分级相关性 (0-4)。
    """

    # 标准的K值
    STANDARD_K_VALUES = [1, 3, 5, 10]

    @classmethod
    def hit_rate(
        cls,
        results: List[RetrievalResult],
        k: int
    ) -> float:
        """Hit Rate @ K

        对于每个查询，如果top-K结果中至少有一个相关文档，则为命中。
        HR@K = 命中查询数 / 总查询数

        Args:
            results: 检索结果列表
            k: top-K

        Returns:
            HR@K (0-1)
        """
        if not results:
            return 0.0

        hits = 0
        for result in results:
            top_k = set(result.retrieved_doc_ids[:k])
            if top_k & result.relevant_doc_ids:
                hits += 1

        return hits / len(results)

    @classmethod
    def hit_rate_all_k(
        cls,
        results: List[RetrievalResult],
        k_values: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """计算所有标准K值的Hit Rate

        Args:
            results: 检索结果列表
            k_values: K值列表

        Returns:
            K → Hit Rate 字典
        """
        k_values = k_values or cls.STANDARD_K_VALUES
        return {k: cls.hit_rate(results, k) for k in k_values}

    @classmethod
    def mean_reciprocal_rank(cls, results: List[RetrievalResult]) -> float:
        """Mean Reciprocal Rank (MRR)

        MRR = (1/N) * Σ(1/rank_i)
        其中 rank_i 是第i个查询的第一个相关文档的排名位置

        Args:
            results: 检索结果列表

        Returns:
            MRR值 (0-1)
        """
        if not results:
            return 0.0

        reciprocal_ranks = []
        for result in results:
            for rank, doc_id in enumerate(result.retrieved_doc_ids, start=1):
                if result.is_relevant(doc_id):
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                # 没有相关文档被检索到
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    @classmethod
    def ndcg(
        cls,
        results: List[RetrievalResult],
        k: int,
        use_graded: bool = True
    ) -> float:
        """Normalized Discounted Cumulative Gain @ K

        DCG@K = Σ(i=1 to K) (rel_i / log2(i+1))
        NDCG@K = DCG@K / IDCG@K

        其中 IDCG@K 是理想排序下的DCG

        Args:
            results: 检索结果列表
            k: top-K
            use_graded: 是否使用分级相关性 (否则二值)

        Returns:
            NDCG@K (0-1)
        """
        if not results:
            return 0.0

        ndcg_values = []
        for result in results:
            # 获取top-K的相关性分数
            top_k = result.retrieved_doc_ids[:k]
            relevance_list = []

            for doc_id in top_k:
                if use_graded and result.relevance_scores:
                    relevance_list.append(result.relevance_scores.get(doc_id, 0.0))
                else:
                    relevance_list.append(1.0 if result.is_relevant(doc_id) else 0.0)

            # DCG
            dcg = 0.0
            for i, rel in enumerate(relevance_list):
                if i == 0:
                    dcg += rel
                else:
                    dcg += rel / math.log2(i + 2)  # i+2 因为index从0开始

            # IDCG (理想排序: 所有相关文档排在前面)
            ideal_relevances = []
            if use_graded and result.relevance_scores:
                # 所有相关文档的分数，按降序排列
                ideal_relevances = sorted(
                    [result.relevance_scores.get(d, 0.0) for d in result.relevant_doc_ids],
                    reverse=True
                )
            else:
                ideal_relevances = [1.0] * min(k, len(result.relevant_doc_ids))

            ideal_relevances = ideal_relevances[:k]  # 截断到K

            idcg = 0.0
            for i, rel in enumerate(ideal_relevances):
                if i == 0:
                    idcg += rel
                else:
                    idcg += rel / math.log2(i + 2)

            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

        return sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0

    @classmethod
    def ndcg_all_k(
        cls,
        results: List[RetrievalResult],
        k_values: Optional[List[int]] = None,
        use_graded: bool = True
    ) -> Dict[int, float]:
        """计算所有标准K值的NDCG"""
        k_values = k_values or cls.STANDARD_K_VALUES
        return {k: cls.ndcg(results, k, use_graded) for k in k_values}

    @classmethod
    def recall(cls, results: List[RetrievalResult], k: int) -> float:
        """Recall @ K

        Recall@K = |检索到的top-K相关文档| / |总相关文档|

        Args:
            results: 检索结果列表
            k: top-K

        Returns:
            Recall@K (0-1)
        """
        if not results:
            return 0.0

        recall_values = []
        for result in results:
            if not result.relevant_doc_ids:
                recall_values.append(0.0)
                continue

            top_k_set = set(result.retrieved_doc_ids[:k])
            relevant_in_top_k = top_k_set & result.relevant_doc_ids
            recall_values.append(len(relevant_in_top_k) / len(result.relevant_doc_ids))

        return sum(recall_values) / len(recall_values)

    @classmethod
    def recall_all_k(
        cls,
        results: List[RetrievalResult],
        k_values: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """计算所有标准K值的Recall"""
        k_values = k_values or cls.STANDARD_K_VALUES
        return {k: cls.recall(results, k) for k in k_values}

    @classmethod
    def precision(cls, results: List[RetrievalResult], k: int) -> float:
        """Precision @ K

        Precision@K = |检索到的top-K相关文档| / K

        Args:
            results: 检索结果列表
            k: top-K

        Returns:
            Precision@K (0-1)
        """
        if not results:
            return 0.0

        precision_values = []
        for result in results:
            top_k_set = set(result.retrieved_doc_ids[:k])
            relevant_in_top_k = top_k_set & result.relevant_doc_ids
            precision_values.append(len(relevant_in_top_k) / min(k, len(result.retrieved_doc_ids)) if result.retrieved_doc_ids else 0.0)

        return sum(precision_values) / len(precision_values)

    @classmethod
    def precision_all_k(
        cls,
        results: List[RetrievalResult],
        k_values: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """计算所有标准K值的Precision"""
        k_values = k_values or cls.STANDARD_K_VALUES
        return {k: cls.precision(results, k) for k in k_values}

    @classmethod
    def mean_average_precision(cls, results: List[RetrievalResult]) -> float:
        """Mean Average Precision (MAP)

        对每个查询计算Average Precision，然后取平均。
        只在相关文档被检索到的时候计算Precision，然后对所有这些Precision取平均。

        Args:
            results: 检索结果列表

        Returns:
            MAP值 (0-1)
        """
        if not results:
            return 0.0

        ap_values = []
        for result in results:
            if not result.relevant_doc_ids:
                continue

            precisions = []
            relevant_found = 0

            for i, doc_id in enumerate(result.retrieved_doc_ids, start=1):
                if result.is_relevant(doc_id):
                    relevant_found += 1
                    precision_at_i = relevant_found / i
                    precisions.append(precision_at_i)

            if precisions:
                ap_values.append(sum(precisions) / len(result.relevant_doc_ids))
            else:
                ap_values.append(0.0)

        return sum(ap_values) / len(ap_values) if ap_values else 0.0

    @classmethod
    def compute_all_metrics(
        cls,
        results: List[RetrievalResult],
        k_values: Optional[List[int]] = None
    ) -> MetricsReport:
        """计算所有检索指标

        Args:
            results: 检索结果列表
            k_values: K值列表 (默认[1,3,5,10])

        Returns:
            包含所有指标的MetricsReport
        """
        k_values = k_values or cls.STANDARD_K_VALUES

        report = MetricsReport(num_queries=len(results))
        report.hit_rate = cls.hit_rate_all_k(results, k_values)
        report.mrr = cls.mean_reciprocal_rank(results)
        report.ndcg = cls.ndcg_all_k(results, k_values)
        report.recall = cls.recall_all_k(results, k_values)
        report.precision = cls.precision_all_k(results, k_values)
        report.map_score = cls.mean_average_precision(results)

        return report


# ============================================================================
# 指标可视化
# ============================================================================

class MetricsVisualizer:
    """检索指标可视化工具"""

    @staticmethod
    def plot_metrics_comparison(
        report_a: MetricsReport,
        label_a: str,
        report_b: Optional[MetricsReport] = None,
        label_b: str = "",
        output_path: Optional[str] = None,
    ) -> None:
        """绘制指标对比图

        Args:
            report_a: 系统A的指标报告
            label_a: 系统A标签
            report_b: 系统B的指标报告 (可选)
            label_b: 系统B标签
            output_path: 输出图片路径
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[警告] matplotlib未安装，无法生成图表。")
            print("       请运行: pip install matplotlib")
            return

        k_values = sorted(report_a.hit_rate.keys())
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('检索指标对比 (Retrieval Metrics Comparison)', fontsize=14, fontweight='bold')

        # 1. Hit Rate
        ax = axes[0, 0]
        ax.plot(k_values, [report_a.hit_rate[k] for k in k_values],
                'o-', label=label_a, linewidth=2, markersize=8)
        if report_b:
            ax.plot(k_values, [report_b.hit_rate[k] for k in k_values],
                    's--', label=label_b, linewidth=2, markersize=8)
        ax.set_xlabel('K')
        ax.set_ylabel('Hit Rate')
        ax.set_title('Hit Rate @ K')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. MRR
        ax = axes[0, 1]
        bars_data = [report_a.mrr]
        bars_label = [label_a]
        if report_b:
            bars_data.append(report_b.mrr)
            bars_label.append(label_b)
        bars = ax.bar(bars_label, bars_data, color=['#2196F3', '#FF9800'][:len(bars_label)])
        ax.set_ylabel('MRR')
        ax.set_title('Mean Reciprocal Rank')
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, bars_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        ax.grid(True, alpha=0.3, axis='y')

        # 3. NDCG
        ax = axes[0, 2]
        ax.plot(k_values, [report_a.ndcg[k] for k in k_values],
                'o-', label=label_a, linewidth=2, markersize=8)
        if report_b:
            ax.plot(k_values, [report_b.ndcg[k] for k in k_values],
                    's--', label=label_b, linewidth=2, markersize=8)
        ax.set_xlabel('K')
        ax.set_ylabel('NDCG')
        ax.set_title('NDCG @ K')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Recall
        ax = axes[1, 0]
        ax.plot(k_values, [report_a.recall[k] for k in k_values],
                'o-', label=label_a, linewidth=2, markersize=8)
        if report_b:
            ax.plot(k_values, [report_b.recall[k] for k in k_values],
                    's--', label=label_b, linewidth=2, markersize=8)
        ax.set_xlabel('K')
        ax.set_ylabel('Recall')
        ax.set_title('Recall @ K')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Precision
        ax = axes[1, 1]
        ax.plot(k_values, [report_a.precision[k] for k in k_values],
                'o-', label=label_a, linewidth=2, markersize=8)
        if report_b:
            ax.plot(k_values, [report_b.precision[k] for k in k_values],
                    's--', label=label_b, linewidth=2, markersize=8)
        ax.set_xlabel('K')
        ax.set_ylabel('Precision')
        ax.set_title('Precision @ K')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. MAP + Summary Table
        ax = axes[1, 2]
        ax.axis('off')

        # 创建汇总表
        summary_data = [
            ['Metric', label_a] + ([label_b] if report_b else []),
        ]
        for k in k_values:
            summary_data.append([f'HR@{k}', f'{report_a.hit_rate[k]:.3f}']
                                + ([f'{report_b.hit_rate[k]:.3f}'] if report_b else []))
        summary_data.append(['MRR', f'{report_a.mrr:.3f}']
                            + ([f'{report_b.mrr:.3f}'] if report_b else []))
        summary_data.append(['MAP', f'{report_a.map_score:.3f}']
                            + ([f'{report_b.map_score:.3f}'] if report_b else []))

        table = ax.table(cellText=summary_data, loc='center',
                         cellLoc='center', colWidths=[0.25, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {output_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def print_report(report: MetricsReport, name: str = "检索指标报告") -> None:
        """打印格式化的指标报告"""
        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print(f"{'=' * 50}")
        print(f"  查询数量: {report.num_queries}")
        print(f"  {'─' * 40}")

        k_values = sorted(report.hit_rate.keys())
        for k in k_values:
            print(f"  Hit Rate @{k:>2d}:     {report.hit_rate[k]:.4f}")

        print(f"  {'─' * 40}")
        print(f"  MRR:              {report.mrr:.4f}")

        print(f"  {'─' * 40}")
        for k in k_values:
            print(f"  NDCG @{k:>2d}:        {report.ndcg[k]:.4f}")

        print(f"  {'─' * 40}")
        for k in k_values:
            print(f"  Recall @{k:>2d}:      {report.recall[k]:.4f}")

        print(f"  {'─' * 40}")
        for k in k_values:
            print(f"  Precision @{k:>2d}:   {report.precision[k]:.4f}")

        print(f"  {'─' * 40}")
        print(f"  MAP:              {report.map_score:.4f}")
        print(f"{'=' * 50}")


# ============================================================================
# 测试数据生成器
# ============================================================================

class TestDataGenerator:
    """生成测试用的检索结果数据"""

    @staticmethod
    def generate_random_results(
        num_queries: int = 50,
        num_docs_per_query: int = 20,
        num_relevant_per_query: int = 3,
        retrieval_quality: float = 0.7,  # 0-1, 越高检索质量越好
        seed: int = 42,
    ) -> List[RetrievalResult]:
        """生成随机检索结果用于测试

        Args:
            num_queries: 查询数量
            num_docs_per_query: 每个查询的文档池大小
            num_relevant_per_query: 每个查询的相关文档数量
            retrieval_quality: 检索质量 (高=相关文档容易排前面)
            seed: 随机种子

        Returns:
            检索结果列表
        """
        random.seed(seed)
        results = []

        for qid in range(num_queries):
            # 生成文档ID列表
            doc_ids = [f"doc_{qid}_{d}" for d in range(num_docs_per_query)]

            # 随机选择相关文档
            relevant_ids = set(random.sample(doc_ids, num_relevant_per_query))

            # 模拟检索排序: 相关文档有更高概率排在前面
            # 使用指数分布模拟排名
            all_docs = doc_ids.copy()
            scored_docs = []

            for doc_id in all_docs:
                if doc_id in relevant_ids:
                    # 相关文档: 偏向高分
                    score = random.expovariate(1.0 / (0.9 * retrieval_quality))
                else:
                    # 不相关文档: 偏向低分
                    score = random.expovariate(1.0 / (0.1 * (1 - retrieval_quality)))
                scored_docs.append((doc_id, score))

            # 按分数降序排序 (模拟检索)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            retrieved_ids = [doc_id for doc_id, _ in scored_docs]

            # 生成分级相关性分数
            relevance_scores = {}
            for doc_id in relevant_ids:
                # 相关文档的分级分数 (1-4)
                relevance_scores[doc_id] = random.choice([1, 2, 3, 4])

            results.append(RetrievalResult(
                query_id=f"q_{qid:03d}",
                query=f"测试查询 {qid}",
                retrieved_doc_ids=retrieved_ids,
                relevant_doc_ids=relevant_ids,
                relevance_scores=relevance_scores,
            ))

        return results

    @staticmethod
    def generate_perfect_results(
        num_queries: int = 50,
        num_docs_per_query: int = 20,
        num_relevant_per_query: int = 3,
    ) -> List[RetrievalResult]:
        """生成完美检索结果 (所有相关文档都在前K)"""
        results = []

        for qid in range(num_queries):
            doc_ids = [f"doc_{qid}_{d}" for d in range(num_docs_per_query)]
            relevant_ids = set(doc_ids[:num_relevant_per_query])

            # 相关文档排在最前面
            retrieved_ids = (list(relevant_ids)
                             + [d for d in doc_ids if d not in relevant_ids])

            relevance_scores = {doc_id: 4.0 for doc_id in relevant_ids}

            results.append(RetrievalResult(
                query_id=f"q_{qid:03d}",
                query=f"完美查询 {qid}",
                retrieved_doc_ids=retrieved_ids,
                relevant_doc_ids=relevant_ids,
                relevance_scores=relevance_scores,
            ))

        return results


# ============================================================================
# trulens-eval 集成 (可选)
# ============================================================================

class TruLensIntegration:
    """与trulens-eval库的集成接口

    如果安装了trulens-eval，可以使用其内置的反馈函数。
    """

    @staticmethod
    def check_availability() -> bool:
        """检查trulens-eval是否可用"""
        try:
            import trulens_eval  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def feedback_functions():
        """获取trulens-eval的反馈函数

        Returns:
            反馈函数字典，如果不可用则返回None
        """
        if not TruLensIntegration.check_availability():
            print("[信息] trulens-eval未安装。请运行: pip install trulens-eval")
            return None

        from trulens_eval import Feedback
        from trulens_eval.feedback import Groundedness

        # 创建反馈函数
        groundedness = Groundedness(groundedness_provider=None)
        f_groundedness = (
            Feedback(groundedness.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on_output()
            .on_input()
        )

        return {
            'groundedness': f_groundedness,
        }

    @staticmethod
    def convert_to_trulens_format(
        results: List[RetrievalResult]
    ) -> List[Dict]:
        """将检索结果转换为trulens格式

        Args:
            results: 检索结果列表

        Returns:
            trulens格式的记录列表
        """
        records = []
        for result in results:
            records.append({
                'query_id': result.query_id,
                'query': result.query,
                'retrieved_documents': result.retrieved_doc_ids,
                'relevant_documents': list(result.relevant_doc_ids),
                'num_retrieved': len(result.retrieved_doc_ids),
                'num_relevant': len(result.relevant_doc_ids),
            })
        return records


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  检索指标评估 - 完整演示")
    print("=" * 60)

    # ---- 生成测试数据 ----
    print("\n生成测试数据...")

    # 系统A: 高质量检索 (retrieval_quality=0.85)
    print("  生成系统A数据 (高质量: 0.85)")
    results_a = TestDataGenerator.generate_random_results(
        num_queries=50,
        retrieval_quality=0.85,
    )

    # 系统B: 基线检索 (retrieval_quality=0.55)
    print("  生成系统B数据 (基线: 0.55)")
    results_b = TestDataGenerator.generate_random_results(
        num_queries=50,
        retrieval_quality=0.55,
    )

    # ---- 计算所有指标 ----
    print("\n计算指标...")
    report_a = RetrievalMetrics.compute_all_metrics(results_a)
    report_b = RetrievalMetrics.compute_all_metrics(results_b)

    # ---- 打印报告 ----
    MetricsVisualizer.print_report(report_a, "系统A - 高质量检索")
    MetricsVisualizer.print_report(report_b, "系统B - 基线检索")

    # ---- 对比分析 ----
    print("\n" + "=" * 50)
    print("  系统对比分析")
    print("=" * 50)

    for k in [1, 3, 5, 10]:
        delta_hr = report_a.hit_rate[k] - report_b.hit_rate[k]
        delta_ndcg = report_a.ndcg[k] - report_b.ndcg[k]
        delta_recall = report_a.recall[k] - report_b.recall[k]
        delta_prec = report_a.precision[k] - report_b.precision[k]

        print(f"\n  K={k}:")
        print(f"    Hit Rate:  Δ={delta_hr:+.4f}  (A={report_a.hit_rate[k]:.4f}, B={report_b.hit_rate[k]:.4f})")
        print(f"    NDCG:      Δ={delta_ndcg:+.4f}  (A={report_a.ndcg[k]:.4f}, B={report_b.ndcg[k]:.4f})")
        print(f"    Recall:    Δ={delta_recall:+.4f}  (A={report_a.recall[k]:.4f}, B={report_b.recall[k]:.4f})")
        print(f"    Precision: Δ={delta_prec:+.4f}  (A={report_a.precision[k]:.4f}, B={report_b.precision[k]:.4f})")

    delta_mrr = report_a.mrr - report_b.mrr
    delta_map = report_a.map_score - report_b.map_score
    print(f"\n  MRR: Δ={delta_mrr:+.4f}  (A={report_a.mrr:.4f}, B={report_b.mrr:.4f})")
    print(f"  MAP: Δ={delta_map:+.4f}  (A={report_a.map_score:.4f}, B={report_b.map_score:.4f})")

    # ---- 单一检索演示 ----
    print("\n" + "-" * 40)
    print("  单次检索演示")
    print("-" * 40)

    single_result = RetrievalResult(
        query_id="demo_001",
        query="什么是RAG技术？",
        retrieved_doc_ids=["doc_1", "doc_2", "doc_3", "doc_4", "doc_5",
                           "doc_6", "doc_7", "doc_8", "doc_9", "doc_10"],
        relevant_doc_ids={"doc_1", "doc_3", "doc_7"},
    )

    single_results = [single_result]
    single_report = RetrievalMetrics.compute_all_metrics(single_results)
    MetricsVisualizer.print_report(single_report, "单次检索演示")

    # ---- trulens-eval 检查 ----
    print("\n" + "-" * 40)
    print("  trulens-eval 集成状态")
    print("-" * 40)
    if TruLensIntegration.check_availability():
        print("  trulens-eval: 可用")
        fb_fns = TruLensIntegration.feedback_functions()
        if fb_fns:
            print(f"  可用反馈函数: {list(fb_fns.keys())}")
    else:
        print("  trulens-eval: 未安装")
        print("  安装命令: pip install trulens-eval")

    # ---- 图表输出 ----
    if MATPLOTLIB_AVAILABLE:
        try:
            MetricsVisualizer.plot_metrics_comparison(
                report_a, "高质量检索 (0.85)",
                report_b, "基线检索 (0.55)",
                output_path="retrieval_metrics_comparison.png",
            )
        except Exception as e:
            print(f"\n[注意] 图表生成失败: {e}")
    else:
        print("\n[注意] matplotlib未安装，跳过图表生成。")
        print("安装命令: pip install matplotlib")

    print()
    print("=" * 60)
    print("  检索指标评估完成")
    print("=" * 60)
