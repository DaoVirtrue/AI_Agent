#!/usr/bin/env python3
"""
B1-02: 难负样本挖掘 (Hard Negative Mining)
==============================================
学习目标:
  1. BM25 Top-N检索用于挖掘难负样本
  2. Cross-batch负样本共享
  3. 难负样本比例对训练效果的影响曲线
"""

import os
import json
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================
# Section 1: BM25 难负样本挖掘
# ============================================================

class BM25HardNegativeMiner:
    """
    使用BM25检索挖掘难负样本

    原理:
    - 简单负样本：随机选择的无关文档（太容易区分，训练价值低）
    - 难负样本：BM25检索排名靠前但非相关的文档（在词汇层面相似，语义层面不相关）

    为什么BM25适合挖掘难负样本？
    - BM25基于词汇匹配，会返回词汇相似但语义不同的文档
    - 这些"看起来相关"的文档正是Embedding模型需要学会区分的
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        参数:
            k1: BM25词频饱和参数（经典值1.2-2.0）
            b:  文档长度归一化参数（经典值0.75）
        """
        self.k1 = k1
        self.b = b

        # 索引
        self.documents: List[str] = []         # 文档文本列表
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        # {term: {doc_id: tf}}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.idf: Dict[str, float] = {}

    def build_index(self, documents: List[str]):
        """构建BM25索引"""
        self.documents = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = np.mean(self.doc_lengths)
        total_docs = len(documents)

        # 构建倒排索引
        term_freqs: Dict[str, int] = defaultdict(int)  # term → 文档频率
        for doc_id, doc in enumerate(documents):
            for term in self._tokenize(doc):
                self.inverted_index[term][doc_id] = (
                    self.inverted_index[term].get(doc_id, 0) + 1
                )

        for term, docs in self.inverted_index.items():
            term_freqs[term] = len(docs)

        # 计算IDF
        for term, df in term_freqs.items():
            self.idf[term] = math.log(
                (total_docs - df + 0.5) / (df + 0.5) + 1
            )

    def _tokenize(self, text: str) -> List[str]:
        """简易分词（中文按字符，英文按空格）"""
        tokens = []
        for word in text.split():
            # 简单处理：每个中文单独算
            if any('一' <= c <= '鿿' for c in word):
                tokens.extend(word)
            else:
                tokens.append(word.lower())
        return tokens

    def score(self, query: str, doc_id: int) -> float:
        """计算BM25分数"""
        query_terms = self._tokenize(query)
        doc_length = self.doc_lengths[doc_id]
        score = 0.0

        for term in set(query_terms):
            if term not in self.inverted_index:
                continue

            tf = self.inverted_index[term].get(doc_id, 0)
            if tf == 0:
                continue

            idf = self.idf.get(term, 0)

            # BM25公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 20,
               exclude_ids: List[int] = None) -> List[Tuple[int, float]]:
        """
        BM25检索
        返回top_k个文档的(doc_id, score)
        """
        exclude = set(exclude_ids or [])
        scores = []

        for doc_id in range(len(self.documents)):
            if doc_id in exclude:
                continue
            s = self.score(query, doc_id)
            if s > 0:
                scores.append((doc_id, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def mine_hard_negatives(self, query: str, positive_doc_id: int,
                            top_k: int = 20,
                            rank_range: Tuple[int, int] = (5, 20)
                            ) -> List[Tuple[int, str, float]]:
        """
        挖掘难负样本

        参数:
            query: 查询
            positive_doc_id: 正样本文档ID
            top_k: BM25检索返回的文档数
            rank_range: 选取难负样本的排名范围
                        太靠前(1-4)可能实际相关
                        太靠后(>20)太简单

        返回: [(doc_id, text, bm25_score), ...]
        """
        results = self.search(query, top_k=top_k,
                              exclude_ids=[positive_doc_id])

        hard_negatives = []
        for doc_id, score in results:
            rank = len(hard_negatives) + 1
            if rank_range[0] <= rank <= rank_range[1]:
                hard_negatives.append((doc_id, self.documents[doc_id], score))

        return hard_negatives


# ============================================================
# Section 2: Cross-batch负样本共享
# ============================================================

class CrossBatchNegativeSharing:
    """
    Cross-batch负样本共享

    原理:
    在一个训练batch中，样本i的正样本可以作为样本j的负样本。
    这在对比学习中能显著增加负样本数量。

    数学:
    一个batch N个(query, positive)对 → N个正样本 + N*(N-1)个in-batch负样本

    优点:
    - "免费"获得大量负样本
    - 这些负样本来自同一batch的其他不相关样本
    - 多个样本共享同一批负样本，增大了对比学习的难度

    缺点:
    - 可能引入假负样本（其他样本的正例碰巧也相关）
    - 大batch size需要大显存
    """

    @staticmethod
    def create_in_batch_negatives(
        batch: List[Tuple[str, str]],  # [(query, positive_chunk), ...]
        batch_size: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        为batch创建in-batch负样本矩阵

        参数:
            batch: [("query1", "pos1"), ("query2", "pos2"), ...]

        返回:
            labels: 正样本对的索引（对角线上的元素）
            每个样本使用其他样本的正例作为负例
        """
        n = len(batch) if batch_size is None else batch_size

        # 标签矩阵: labels[i][j] = 1 if i==j else 0
        # 即：query_i 只与 positive_i 配对为正
        labels = np.eye(n, dtype=np.float32)

        return labels

    @staticmethod
    def analyse_negative_distribution(batch_embeddings: np.ndarray,
                                      labels: np.ndarray) -> Dict:
        """
        分析batch内负样本分布
        """
        # 计算余弦相似度矩阵
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        normalized = batch_embeddings / (norms + 1e-8)
        sim_matrix = np.dot(normalized, normalized.T)

        n = len(sim_matrix)

        # 正样本相似度（对角线）
        positive_sims = [sim_matrix[i, i] for i in range(n)]

        # 负样本相似度（非对角线）
        negative_sims = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    negative_sims.append(sim_matrix[i, j])

        # 检测假负样本（相似度过高的in-batch负样本）
        false_negative_threshold = 0.7
        false_negatives = []
        for i in range(n):
            for j in range(n):
                if i != j and sim_matrix[i, j] > false_negative_threshold:
                    false_negatives.append({
                        "query_i": i, "query_j": j,
                        "similarity": float(sim_matrix[i, j]),
                    })

        return {
            "positive_mean_sim": float(np.mean(positive_sims)) if positive_sims else 0,
            "negative_mean_sim": float(np.mean(negative_sims)) if negative_sims else 0,
            "false_negatives_count": len(false_negatives),
            "false_negative_rate": len(false_negatives) / (n * (n-1)) if n > 1 else 0,
            "separability": float(
                (np.mean(positive_sims) - np.mean(negative_sims)) /
                (np.std(negative_sims) + 1e-8)
            ) if negative_sims else 0,
        }


# ============================================================
# Section 3: 难负样本比例影响分析
# ============================================================

class HardNegativeImpactAnalyzer:
    """
    难负样本比例对训练效果的影响分析

    关键发现（文献/经验）:
    - 0% 难负样本: 模型容易过拟合简单模式
    - 10-20%: 性能开始提升
    - 30-50%: 最优区间
    - >60%: 可能过度惩罚合理的语义相关性，反而降低性能
    """

    @staticmethod
    def simulate_impact_curve(
        num_points: int = 20,
        noise_level: float = 0.05,
    ) -> Dict:
        """
        模拟难负样本比例对RAG检索命中率的影响曲线
        （基于公开文献的趋势模拟）

        真实实验中应通过消融实验(ablation)获得此曲线
        """
        ratios = np.linspace(0, 1.0, num_points)

        # 模拟命中率曲线：先上升后下降
        # 基于经验公式: hit_rate = peak * exp(-((ratio-optimal)^2 / 2*sigma^2))
        optimal_ratio = 0.40  # 最优比例约40%
        peak_hit_rate = 0.85  # 最高命中率85%
        sigma = 0.35

        hit_rates = peak_hit_rate * np.exp(
            -((ratios - optimal_ratio) ** 2) / (2 * sigma ** 2)
        )
        # 添加噪声
        noise = np.random.normal(0, noise_level, num_points)
        hit_rates = np.clip(hit_rates + noise, 0.0, 1.0)

        # 计算每段的改善
        improvements = [0.0]
        for i in range(1, len(ratios)):
            imp = hit_rates[i] - hit_rates[0]  # 相对于无难负样本的提升
            improvements.append(imp)

        return {
            "ratios": [round(r, 2) for r in ratios.tolist()],
            "hit_rates": [round(h, 4) for h in hit_rates.tolist()],
            "improvements": [round(imp, 4) for imp in improvements],
            "optimal_ratio": round(optimal_ratio, 2),
            "max_hit_rate": round(float(hit_rates.max()), 4),
            "interpretation": {
                "0-10%": "简单负样本占主导，效果一般",
                "10-30%": "初见提升，模型开始学习区分难例",
                "30-50%": "最优区间，平衡难度与泛化",
                "50-70%": "可能过度区分，边际效益递减",
                "70-100%": "过度惩罚合理关联，性能下降",
            },
        }

    @staticmethod
    def print_impact_table(impact_data: Dict):
        """打印影响分析表格"""
        print("\n  难负样本比例影响分析:")
        print(f"  {'比例':<8s} {'命中率':<10s} {'改善':<8s} {'解释':<s}")
        print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*30}")

        ratios = impact_data["ratios"]
        hit_rates = impact_data["hit_rates"]
        improvements = impact_data["improvements"]

        for r, h, imp in zip(ratios[::3], hit_rates[::3], improvements[::3]):
            for range_key, desc in impact_data["interpretation"].items():
                lo, hi = range_key.split("-")
                if r >= float(lo.replace("%", "")) / 100 and r <= float(hi.replace("%", "")) / 100:
                    print(f"  {r:<8.2f} {h:<10.4f} {imp:+.4f}  {desc}")
                    break
            else:
                print(f"  {r:<8.2f} {h:<10.4f} {imp:+.4f}")


# ============================================================
# Section 4: 完整难负样本挖掘管道
# ============================================================

class HardNegativePipeline:
    """难负样本挖掘完整管道"""

    def __init__(self):
        self.bm25_miner = BM25HardNegativeMiner()
        self.sharing = CrossBatchNegativeSharing()
        self.analyzer = HardNegativeImpactAnalyzer()

    def mine_and_augment(self,
                         queries: List[str],
                         positives: List[str],
                         corpus: List[str],
                         hard_neg_ratio: float = 0.4,
                         ) -> List[Tuple[str, str, str]]:
        """
        完整流程: 挖掘难负样本 + 增强数据集

        返回: [(query, positive, negative), ...]
        """
        # 构建BM25索引
        all_docs = list(set(positives + corpus))
        self.bm25_miner.build_index(all_docs)

        triplets = []
        for i, (query, pos) in enumerate(zip(queries, positives)):
            pos_id = all_docs.index(pos) if pos in all_docs else -1

            # 1. BM25挖掘难负样本
            hard_negs = self.bm25_miner.mine_hard_negatives(
                query, pos_id, top_k=30, rank_range=(5, 25)
            )

            # 2. 随机负样本（简单负样本）
            random_neg_idx = random.randint(0, len(all_docs) - 1)
            while random_neg_idx == pos_id:
                random_neg_idx = random.randint(0, len(all_docs) - 1)
            easy_neg = all_docs[random_neg_idx]

            # 3. 按比例混合
            num_hard = max(1, int(hard_neg_ratio * 5))  # 目标5个负样本
            num_easy = 5 - num_hard

            for neg_id, neg_text, _ in hard_negs[:num_hard]:
                triplets.append((query, pos, neg_text))

            for _ in range(num_easy):
                triplets.append((query, pos, easy_neg))

        return triplets


# ============================================================
# Section 5: 主流程
# ============================================================

def main():
    """主函数：演示难负样本挖掘全流程"""
    print("=" * 70)
    print("B1-02: 难负样本挖掘 — 完整演示")
    print("=" * 70)

    # 1. 构建文档集
    print("\n[步骤1] 构建BM25索引...")
    corpus = [
        "RAG系统通过检索增强生成来提高回答的准确性，核心组件包括Embedding模型和向量数据库。",
        "向量数据库如Milvus和ChromaDB使用ANN算法进行高效的相似度搜索。",
        "机器学习模型的训练需要大量的标注数据，数据质量直接影响模型效果。",
        "Python中的异步编程使用asyncio库，可以显著提高IO密集型任务的性能。",
        "RAG中的检索器负责从知识库中找到与查询最相关的文档片段。",
        "深度学习中的Transformer架构通过自注意力机制实现了序列建模的突破。",
        "Embedding模型的微调可以显著提高特定领域的检索准确率。",
        "数据增强技术通过创建训练样本的变体来提高模型的泛化能力。",
        "多轮对话系统中的上下文管理是保证对话连贯性的关键。",
        "BGE-M3是一个优秀的多语言Embedding模型，支持稠密和稀疏检索。",
    ]

    miner = BM25HardNegativeMiner()
    miner.build_index(corpus)

    # 2. BM25检索演示
    print("\n[步骤2] BM25检索 — 难负样本挖掘...")
    query = "RAG检索准确率优化方法"
    print(f"  查询: '{query}'")

    results = miner.search(query, top_k=10)
    print(f"  BM25 Top10结果:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        text = corpus[doc_id]
        prefix = text[:60]
        print(f"    #{rank}: BM25={score:.2f} | {prefix}...")

    # 3. 挖掘难负样本
    print("\n[步骤3] 挖掘难负样本...")
    hard_negs = miner.mine_hard_negatives(query, positive_doc_id=0,
                                          top_k=20, rank_range=(5, 15))
    for i, (doc_id, text, score) in enumerate(hard_negs):
        print(f"  难负样本 #{i+1}: BM25={score:.2f} | {text[:60]}...")

    # 4. Cross-batch负样本分析
    print("\n[步骤4] Cross-batch负样本共享分析...")
    # 模拟一个batch的嵌入向量
    np.random.seed(123)
    batch_size = 8
    embedding_dim = 384
    batch_embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)
    # 让对角线样本相似度略高（模拟正样本）
    for i in range(batch_size):
        batch_embeddings[i] = batch_embeddings[i] / np.linalg.norm(batch_embeddings[i])

    labels = CrossBatchNegativeSharing.create_in_batch_negatives(
        [("", "")] * batch_size
    )
    analysis = CrossBatchNegativeSharing.analyse_negative_distribution(
        batch_embeddings, labels
    )
    print(f"  正样本平均相似度: {analysis['positive_mean_sim']:.4f}")
    print(f"  负样本平均相似度: {analysis['negative_mean_sim']:.4f}")
    print(f"  可分离性: {analysis['separability']:.4f}")
    print(f"  假负样本率: {analysis['false_negative_rate']:.4f}")

    # 5. 难负样本比例影响曲线
    print("\n[步骤5] 难负样本比例影响分析...")
    impact = HardNegativeImpactAnalyzer.simulate_impact_curve()
    HardNegativeImpactAnalyzer.print_impact_table(impact)
    print(f"\n  最优难负样本比例: {impact['optimal_ratio']}")
    print(f"  对应最高命中率: {impact['max_hit_rate']}")

    # 6. 完整管道演示
    print("\n[步骤6] 完整难负样本挖掘管道...")
    pipeline = HardNegativePipeline()
    queries = [
        "RAG检索准确率优化",
        "向量数据库选型指南",
        "Embedding模型微调方法",
    ]
    positives = [corpus[0], corpus[1], corpus[6]]
    triplets = pipeline.mine_and_augment(queries, positives, corpus,
                                         hard_neg_ratio=0.4)
    print(f"  生成 {len(triplets)} 个三元组 (含{int(0.4*triplets.count)}个难负样本)")
    for i, (q, p, n) in enumerate(triplets[:5]):
        print(f"    #{i+1}: q='{q[:30]}...' | neg='{n[:40]}...'")

    print("\n" + "=" * 70)
    print("B1-02 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
