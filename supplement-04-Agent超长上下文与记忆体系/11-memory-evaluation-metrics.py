#!/usr/bin/env python3
"""
记忆评估指标 (Memory Evaluation Metrics)
=============================================
实现记忆检索质量和摘要压缩质量的完整评估指标体系。

核心指标：
  - 检索质量：Precision@K, Recall@K, MRR, NDCG, Hit Rate
  - 压缩保真度：语义相似度, 关键事实保留率
  - 幻觉检测：摘要中的虚构事实检测
  - 新鲜度评分：记忆时间衰减评估
"""

from __future__ import annotations

import re
import math
import time
import json
import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict, OrderedDict


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class HallucinatedFact:
    """检测到的幻觉事实"""
    fact_text: str                    # 声称的事实
    evidence: Optional[str] = None    # 来源文档中的证据（如果有）
    severity: str = "medium"          # 严重程度: "low", "medium", "high"
    explanation: str = ""             # 为什么判定为幻觉


@dataclass
class MemoryQualityReport:
    """记忆质量综合报告"""
    retrieval_quality: Dict[str, float]
    compression_fidelity: Dict[str, float]
    hallucination_stats: Dict[str, Any]
    freshness_score: float
    overall_score: float
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# 检索质量评估
# ============================================================================

class RetrievalQualityMetrics:
    """
    检索质量指标。

    核心指标：
    - Precision@K: 前K个结果中正确结果的比例
    - Recall@K: 所有正确答案中被检索到的比例
    - MRR (Mean Reciprocal Rank): 第一个正确答案排名的倒数均值
    - NDCG@K: 归一化折损累积增益
    - Hit Rate: 是否至少检索到一个正确答案
    """

    @staticmethod
    def precision_at_k(retrieved_ids: List[str],
                        relevant_ids: Set[str], k: int) -> float:
        """
        Precision@K: 前K个结果中的命中率。

        P@K = |retrieved[:k] ∩ relevant| / k
        """
        if k <= 0:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        hits = len(retrieved_k & relevant_ids)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str],
                     relevant_ids: Set[str], k: int) -> float:
        """
        Recall@K: 前K个结果找到的正确答案比例。

        R@K = |retrieved[:k] ∩ relevant| / |relevant|
        """
        if k <= 0 or not relevant_ids:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        hits = len(retrieved_k & relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def mean_reciprocal_rank(queries_retrieved: List[List[str]],
                              queries_relevant: List[Set[str]]) -> float:
        """
        MRR (Mean Reciprocal Rank): 平均倒数排名。

        MRR = (1/|Q|) * Σ(1/rank_i)
        其中 rank_i 是第一个相关结果的排名位置。
        """
        if not queries_retrieved:
            return 0.0

        reciprocal_ranks = []
        for retrieved, relevant in zip(queries_retrieved, queries_relevant):
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str],
                   relevance_scores: Dict[str, float], k: int) -> float:
        """
        NDCG@K (Normalized Discounted Cumulative Gain)。

        DCG@K = Σ(relevance_i / log2(i+1)), i=1..K
        IDCG@K = DCG of ideal ranking
        NDCG@K = DCG@K / IDCG@K
        """
        if k <= 0 or not retrieved_ids or not relevance_scores:
            return 0.0

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / math.log2(i + 1)

        # IDCG (ideal: sort by relevance descending)
        sorted_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(sorted_rels, start=1):
            idcg += rel / math.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_rate(queries_retrieved: List[List[str]],
                  queries_relevant: List[Set[str]],
                  k: int = 10) -> float:
        """
        Hit Rate@K: 至少包含一个相关结果的比例。

        HitRate@K = count(queries with at least one hit in top-K) / total_queries
        """
        if not queries_retrieved:
            return 0.0

        hits = 0
        for retrieved, relevant in zip(queries_retrieved, queries_relevant):
            if set(retrieved[:k]) & relevant:
                hits += 1

        return hits / len(queries_retrieved)

    @staticmethod
    def evaluate_all(queries_retrieved: List[List[str]],
                      queries_relevant: List[Set[str]],
                      k_values: List[int] = [1, 3, 5, 10]
                      ) -> Dict[str, float]:
        """一站式评估所有检索指标"""
        results = {}

        for k in k_values:
            precisions = [
                RetrievalQualityMetrics.precision_at_k(r, rel, k)
                for r, rel in zip(queries_retrieved, queries_relevant)
            ]
            recalls = [
                RetrievalQualityMetrics.recall_at_k(r, rel, k)
                for r, rel in zip(queries_retrieved, queries_relevant)
            ]
            results[f"Precision@{k}"] = round(
                sum(precisions) / max(len(precisions), 1), 4
            )
            results[f"Recall@{k}"] = round(
                sum(recalls) / max(len(recalls), 1), 4
            )

        results["MRR"] = round(
            RetrievalQualityMetrics.mean_reciprocal_rank(
                queries_retrieved, queries_relevant
            ), 4
        )
        results["HitRate@10"] = round(
            RetrievalQualityMetrics.hit_rate(
                queries_retrieved, queries_relevant, k=10
            ), 4
        )

        return results


# ============================================================================
# 压缩保真度评估
# ============================================================================

class CompressionFidelityMetrics:
    """
    压缩保真度指标。

    评估摘要/压缩后的内容是否保留了原始信息的核心部分。
    """

    @staticmethod
    def semantic_similarity(original: str, summary: str) -> float:
        """
        估算语义相似度（基于关键词Jaccard + 句法结构）。

        实际生产中使用 Sentence-BERT / OpenAI Embeddings。

        Returns:
            相似度 [0, 1]
        """
        # 关键词提取
        def extract_keywords(text: str) -> Set[str]:
            # 去除停用词后的有意义的词
            stopwords = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                '的', '了', '在', '是', '我', '你', '他', '她', '它', '们',
                '这', '那', '吗', '呢', '吧', '啊', '哦', '和', '与', '或',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            }
            words = re.findall(r'[\w一-鿿]{2,}', text.lower())
            return {w for w in words if w not in stopwords}

        orig_kw = extract_keywords(original)
        summ_kw = extract_keywords(summary)

        if not orig_kw:
            return 1.0 if not summ_kw else 0.0

        # Jaccard相似度
        intersection = orig_kw & summ_kw
        union = orig_kw | summ_kw
        jaccard = len(intersection) / len(union) if union else 0.0

        # 覆盖率
        coverage = len(intersection) / len(orig_kw) if orig_kw else 0.0

        # 综合分数
        return 0.6 * jaccard + 0.4 * coverage

    @staticmethod
    def key_fact_retention(original_facts: List[str],
                            summary_facts: List[str]) -> float:
        """
        关键事实保留率。

        fact_retention = |original_facts ∩ summary_facts| / |original_facts|
        """
        if not original_facts:
            return 1.0

        orig_set = {f.strip().lower() for f in original_facts}
        summ_set = {f.strip().lower() for f in summary_facts}

        # 精确匹配 + 包含匹配
        matched = 0
        for fact in orig_set:
            if fact in summ_set:
                matched += 1
            else:
                # 部分匹配
                for summ_fact in summ_set:
                    if len(fact) > 10 and fact[:20] in summ_fact:
                        matched += 0.5
                        break

        return min(matched / len(orig_set), 1.0)

    @staticmethod
    def compression_ratio(original_chars: int,
                           summary_chars: int) -> float:
        """压缩比 = summary_length / original_length"""
        if original_chars == 0:
            return 0.0
        return summary_chars / original_chars

    @staticmethod
    def evaluate_all(original: str, summary: str,
                      original_facts: Optional[List[str]] = None,
                      summary_facts: Optional[List[str]] = None
                      ) -> Dict[str, float]:
        """一站式评估所有压缩保真度指标"""
        results = {
            "semantic_similarity": round(
                CompressionFidelityMetrics.semantic_similarity(original, summary), 4
            ),
            "compression_ratio": round(
                CompressionFidelityMetrics.compression_ratio(
                    len(original), len(summary)
                ), 4
            ),
        }

        if original_facts and summary_facts:
            results["key_fact_retention"] = round(
                CompressionFidelityMetrics.key_fact_retention(
                    original_facts, summary_facts
                ), 4
            )

        return results


# ============================================================================
# 幻觉检测器
# ============================================================================

class HallucinationDetector:
    """
    摘要中的幻觉检测器。

    检测三种幻觉类型：
    1. 事实不一致：摘要中的事实在原文中找不到支持
    2. 数字错误：摘要中的数字与原文不同
    3. 实体幻觉：摘要中出现了原文不存在的实体
    """

    def __init__(self):
        self._detected_hallucinations: List[HallucinatedFact] = []

    def detect_in_summary(self, original: str,
                           summary: str) -> List[HallucinatedFact]:
        """
        检测摘要中的幻觉事实。

        Args:
            original: 原始文本
            summary: 摘要文本

        Returns:
            检测到的幻觉事实列表
        """
        hallucinations = []

        # 提取原始文本中的所有数字
        orig_numbers = set(re.findall(r'\d+\.?\d*', original))
        summ_numbers = set(re.findall(r'\d+\.?\d*', summary))

        # 数字幻觉：摘要中出现了原文没有的数字
        extra_numbers = summ_numbers - orig_numbers
        for num in extra_numbers:
            if int(float(num)) not in [int(float(n)) for n in orig_numbers]:
                # 找到相关的句子
                for sentence in re.split(r'[。！？.!?\n]+', summary):
                    if num in sentence:
                        hallucinations.append(HallucinatedFact(
                            fact_text=sentence.strip()[:100],
                            severity="high" if float(num) > 100 else "medium",
                            explanation=f"摘要中出现了原文不存在的数值: {num}",
                        ))

        # 实体幻觉：检查大写实体词
        orig_entities = set(re.findall(r'[A-Z][a-z]+', original))
        summ_entities = set(re.findall(r'[A-Z][a-z]+', summary))
        extra_entities = summ_entities - orig_entities
        for entity in extra_entities:
            if len(entity) > 3 and entity not in ('The', 'This', 'That', 'These'):
                for sentence in re.split(r'[。！？.!?\n]+', summary):
                    if entity in sentence:
                        hallucinations.append(HallucinatedFact(
                            fact_text=sentence.strip()[:100],
                            severity="medium",
                            explanation=f"摘要中出现了原文不存在的实体: {entity}",
                        ))

        self._detected_hallucinations.extend(hallucinations)
        return hallucinations

    def hallucination_rate(self, summary: str,
                            hallucinations: Optional[List[HallucinatedFact]] = None
                            ) -> float:
        """
        计算幻觉率。

        hallucination_rate = 幻觉句子数 / 总句子数

        Returns:
            幻觉率 [0, 1]
        """
        h_facts = hallucinations or self._detected_hallucinations
        sentences = [s.strip() for s in re.split(r'[。！？.!?\n]+', summary)
                     if len(s.strip()) > 5]

        if not sentences:
            return 0.0

        hallucinated_sentences = set()
        for h in h_facts:
            for sent in sentences:
                if h.fact_text[:30] in sent:
                    hallucinated_sentences.add(sent)

        return len(hallucinated_sentences) / len(sentences)

    def get_stats(self) -> Dict[str, Any]:
        """获取幻觉检测统计"""
        severity_counts = Counter(h.severity for h in self._detected_hallucinations)
        return {
            "total_detected": len(self._detected_hallucinations),
            "by_severity": dict(severity_counts),
            "recent": [
                {"fact": h.fact_text[:50], "severity": h.severity}
                for h in self._detected_hallucinations[-5:]
            ],
        }


# ============================================================================
# 新鲜度评分器
# ============================================================================

class FreshnessScorer:
    """
    记忆新鲜度评分器。

    基于时间衰减的记忆新鲜度评估：
    - 更最近更新的记忆得分更高
    - 不同记忆类型有不同的衰减速度
    - 支持自定义衰减函数
    """

    # 各类型记忆的半衰期（秒）
    TYPE_HALF_LIVES = {
        "fact": 86400 * 365,       # 事实：1年
        "event": 86400 * 30,       # 事件：30天
        "preference": 86400 * 90,  # 偏好：90天
        "general": 86400 * 180,    # 通用：180天
    }

    @staticmethod
    def score(created_at: float, updated_at: float,
               mem_type: str = "general",
               current_time: Optional[float] = None) -> float:
        """
        计算记忆新鲜度分数。

        公式: score = e^(-λ * elapsed)
        其中 λ = ln(2) / half_life

        Args:
            created_at: 创建时间
            updated_at: 最后更新时间
            mem_type: 记忆类型
            current_time: 当前时间（默认使用time.time()）

        Returns:
            新鲜度分数 [0, 1]，1表示最新
        """
        now = current_time or time.time()
        half_life = FreshnessScorer.TYPE_HALF_LIVES.get(
            mem_type, FreshnessScorer.TYPE_HALF_LIVES["general"]
        )

        # 使用最后更新时间计算新鲜度
        elapsed = now - max(updated_at, created_at)
        decay_rate = math.log(2) / half_life

        return math.exp(-decay_rate * elapsed)

    @staticmethod
    def batch_score(memories: List[Dict[str, Any]],
                     current_time: Optional[float] = None) -> List[float]:
        """批量计算新鲜度"""
        now = current_time or time.time()
        scores = []
        for mem in memories:
            s = FreshnessScorer.score(
                created_at=mem.get("created_at", now),
                updated_at=mem.get("updated_at", now),
                mem_type=mem.get("mem_type", "general"),
                current_time=now,
            )
            scores.append(round(s, 4))
        return scores

    @staticmethod
    def get_freshness_curve(half_life_days: float = 30.0,
                             num_points: int = 50) -> List[Tuple[float, float]]:
        """
        生成新鲜度衰减曲线。

        Returns:
            [(days_elapsed, freshness_score), ...]
        """
        half_life = half_life_days * 86400
        decay_rate = math.log(2) / half_life
        points = []
        for i in range(num_points + 1):
            days = (i / num_points) * half_life_days * 3
            score = math.exp(-decay_rate * days * 86400)
            points.append((round(days, 1), round(score, 4)))
        return points


# ============================================================================
# 综合质量报告生成
# ============================================================================

class MemoryQualityReportGenerator:
    """综合质量报告生成器"""

    def __init__(self):
        self.retrieval_metrics = RetrievalQualityMetrics()
        self.fidelity_metrics = CompressionFidelityMetrics()
        self.hallucination_detector = HallucinationDetector()
        self.freshness_scorer = FreshnessScorer()

    def generate_report(self,
                         retrieval_results: Dict[str, float],
                         fidelity_results: Dict[str, float],
                         hallucination_stats: Dict[str, Any],
                         freshness_scores: List[float]
                         ) -> MemoryQualityReport:
        """生成综合质量报告"""
        # 子项评分归一化
        retrieval_avg = sum(retrieval_results.values()) / max(len(retrieval_results), 1)
        fidelity_avg = sum(fidelity_results.values()) / max(len(fidelity_results), 1)
        hallucination_penalty = 1.0 - hallucination_stats.get("total_detected", 0) / 50
        hallucination_penalty = max(hallucination_penalty, 0.0)
        freshness_avg = sum(freshness_scores) / max(len(freshness_scores), 1)

        # 综合评分（加权）
        overall = (
            0.30 * retrieval_avg +
            0.30 * fidelity_avg +
            0.20 * hallucination_penalty +
            0.20 * freshness_avg
        )

        return MemoryQualityReport(
            retrieval_quality=retrieval_results,
            compression_fidelity=fidelity_results,
            hallucination_stats=hallucination_stats,
            freshness_score=round(freshness_avg, 4),
            overall_score=round(overall, 4),
        )


# ============================================================================
# Demo: 20条测试查询
# ============================================================================

def generate_test_data():
    """生成20条测试查询数据"""
    import random
    random.seed(123)

    retrieved = []
    relevant = []

    for i in range(20):
        # 模拟5-15个检索结果
        n_results = random.randint(5, 15)
        query_retrieved = [f"doc-{random.randint(1, 100)}" for _ in range(n_results)]
        # 去重
        query_retrieved = list(OrderedDict.fromkeys(query_retrieved))

        # 模拟1-4个相关文档
        n_relevant = random.randint(1, 4)
        query_relevant = set(random.sample(query_retrieved[:8],
                                           min(n_relevant, len(query_retrieved[:8]))))

        retrieved.append(query_retrieved)
        relevant.append(query_relevant)

    return retrieved, relevant


def demo():
    """完整演示"""
    print("=" * 70)
    print("  记忆评估指标 (Memory Evaluation Metrics) 演示")
    print("=" * 70)

    # ---- 1. 检索质量评估（20条查询） ----
    print("\n【1. 检索质量评估 - 20条测试查询】")
    print("-" * 50)

    retrieved, relevant = generate_test_data()
    print(f"  测试查询数: {len(retrieved)}")
    print(f"  平均检索结果数: {sum(len(r) for r in retrieved) / len(retrieved):.1f}")
    print(f"  平均相关文档数: {sum(len(r) for r in relevant) / len(relevant):.1f}")

    retrieval_results = RetrievalQualityMetrics.evaluate_all(
        retrieved, relevant, k_values=[1, 3, 5, 10]
    )

    for metric, value in retrieval_results.items():
        bar = "█" * int(value * 30)
        print(f"  {metric:15s}: {value:.4f} {bar}")

    # ---- 2. 压缩保真度 ----
    print("\n【2. 压缩保真度评估】")
    print("-" * 50)

    original_text = (
        "在2026年第一季度的销售报告中，公司总收入达到1.5亿元人民币，同比增长23%。"
        "华东地区贡献了45%的收入，华南地区贡献了30%。新产品线AI芯片营收首次突破2000万元。"
        "利润率从去年的18%提升至22%，主要得益于供应链优化和自动化生产线的投产。"
        "CEO张伟在季度会议上宣布，下半年将追加5000万元研发投入。"
        "同时，海外市场拓展取得重大进展，欧洲分公司已签约12家经销商。"
        "客户满意度从92%提升至96%，重复购买率达到78%。"
        "公司员工人数达到1200人，其中研发团队占比35%。"
        "明年计划在东南亚设立新的生产基地，预计投资2亿元。"
    )

    summary_text = (
        "2026年Q1公司营收1.5亿元，同比增长23%。华东占45%收入。"
        "新AI芯片产品营收2000万元。利润率升至22%。CEO宣布追加研发投入。"
        "海外拓展有进展，客户满意度提升至96%。"
    )

    fidelity_results = CompressionFidelityMetrics.evaluate_all(
        original_text, summary_text,
        original_facts=[
            "营收1.5亿元", "同比增长23%", "华东占45%",
            "利润率升至22%", "AI芯片营收2000万元",
        ],
        summary_facts=[
            "营收1.5亿元", "同比增长23%", "华东占45%",
            "利润率升至22%", "AI芯片营收2000万元",
        ],
    )

    for metric, value in fidelity_results.items():
        print(f"  {metric}: {value:.4f}")

    # ---- 3. 幻觉检测 ----
    print("\n【3. 幻觉检测】")
    print("-" * 50)

    detector = HallucinationDetector()

    # 故意在摘要中添加幻觉
    hallucinated_summary = (
        "2026年Q1公司营收2.0亿元（幻觉：实际1.5亿），同比增长30%（幻觉：实际23%）。"
        "与Microsoft（幻觉：原文未提及）达成战略合作。"
        "研发团队超过500人（原文1200*0.35=420，接近但略有偏差）。"
    )

    hallucinations = detector.detect_in_summary(original_text, hallucinated_summary)
    print(f"  检测到 {len(hallucinations)} 个幻觉事实:")
    for h in hallucinations:
        print(f"    [{h.severity}] {h.fact_text[:60]}...")
        print(f"      解释: {h.explanation}")

    rate = detector.hallucination_rate(hallucinated_summary, hallucinations)
    print(f"\n  幻觉率: {rate:.2%}")

    print(f"\n  幻觉检测统计:")
    print(json.dumps(detector.get_stats(), indent=2, ensure_ascii=False))

    # ---- 4. 新鲜度评分 ----
    print("\n【4. 记忆新鲜度评分】")
    print("-" * 50)

    now = time.time()
    test_memories = [
        {"created_at": now - 86400 * 1, "updated_at": now - 3600, "mem_type": "event"},
        {"created_at": now - 86400 * 7, "updated_at": now - 86400 * 7, "mem_type": "fact"},
        {"created_at": now - 86400 * 30, "updated_at": now - 86400 * 30, "mem_type": "preference"},
        {"created_at": now - 86400 * 100, "updated_at": now - 86400 * 90, "mem_type": "general"},
        {"created_at": now - 86400 * 365, "updated_at": now - 86400 * 300, "mem_type": "fact"},
    ]

    labels = ["昨天", "7天前", "30天前", "100天前", "365天前"]
    scores = FreshnessScorer.batch_score(test_memories, current_time=now)
    print("  记忆新鲜度评分:")
    for label, score, mem in zip(labels, scores, test_memories):
        bar = "█" * int(score * 30)
        print(f"    {label:8s} ({mem['mem_type']:10s}): {score:.4f} {bar}")

    # 衰减曲线
    print("\n  新鲜度衰减曲线 (30天半衰期):")
    curve = FreshnessScorer.get_freshness_curve(half_life_days=30, num_points=10)
    for days, score in curve:
        print(f"    {days:>5.0f}天后: {score:.4f}")

    # ---- 5. 综合质量报告 ----
    print("\n【5. 综合质量报告】")
    print("-" * 50)

    generator = MemoryQualityReportGenerator()
    report = generator.generate_report(
        retrieval_results=retrieval_results,
        fidelity_results=fidelity_results,
        hallucination_stats=detector.get_stats(),
        freshness_scores=scores,
    )

    print(f"  检索质量得分: {report.retrieval_quality}")
    print(f"  压缩保真度得分: {report.compression_fidelity}")
    print(f"  幻觉统计: {report.hallucination_stats}")
    print(f"  新鲜度平均分: {report.freshness_score:.4f}")
    print(f"  ╔══════════════════════╗")
    print(f"  ║  综合评分: {report.overall_score:.4f}     ║")
    print(f"  ╚══════════════════════╝")

    # 评级
    if report.overall_score >= 0.8:
        grade = "A (优秀)"
    elif report.overall_score >= 0.6:
        grade = "B (良好)"
    elif report.overall_score >= 0.4:
        grade = "C (一般)"
    else:
        grade = "D (需要改进)"
    print(f"  评级: {grade}")

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
