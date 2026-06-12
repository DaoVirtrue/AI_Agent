#!/usr/bin/env python3
"""
05 - Few-Shot 示例选择器 (Few-Shot Selection)

本模块提供完整的 Few-shot 示例选择系统，包括：
- FewShotSelector: 三种选择策略（静态、嵌入相似度、最大边际相关性）
- ExampleOrderingStrategies: 四种排序策略（难度递增、递减、交错、邻近）
- HardExampleMiner: 自动发现模型失败的困难示例
- FewShotDatabase: 示例的存储、版本管理和检索

所有注释使用中文，代码标识符使用英文。
"""

import os
import json
import random
import hashlib
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """
    单个 Few-shot 示例。

    What: 表示一个完整的（输入，输出）示例对，附带元数据。
    Why: 统一的示例数据结构，便于不同组件之间传递。
    When: 所有涉及 Few-shot 示例的场景。
    """

    id: str                               # 唯一标识符
    input_text: str                       # 用户输入
    output_text: str                      # 期望输出
    category: str = "general"             # 示例类别
    difficulty: float = 0.5               # 难度评分 0.0~1.0
    embedding: list[float] | None = None  # 向量嵌入（可选）
    metadata: dict = field(default_factory=dict)  # 额外元数据
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1                      # 版本号

    def to_prompt_string(self) -> str:
        """
        将示例转换为可直接注入提示的字符串格式。

        Returns:
            格式化的提示示例字符串
        """
        return f"输入: {self.input_text}\n输出: {self.output_text}"

    def to_dict(self) -> dict:
        """序列化为字典（用于存储）"""
        return {
            "id": self.id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "category": self.category,
            "difficulty": self.difficulty,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Example":
        """从字典反序列化"""
        return cls(**data)


# ---------------------------------------------------------------------------
# 嵌入工具（轻量级，不依赖外部嵌入模型）
# ---------------------------------------------------------------------------


class EmbeddingUtil:
    """
    轻量级嵌入工具。

    What: 提供文本到向量的转换能力。
    Why: 嵌入相似度选择策略需要向量表示；提供本地实现避免硬依赖。
    When: 嵌入选择策略或任何需要文本向量化的场景。

    注意：生产环境应使用 OpenAI embeddings API 或本地嵌入模型。
    此处提供基于 TF-IDF 的轻量级实现用于演示。
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}

    def fit(self, texts: list[str]):
        """
        在文本集合上构建词汇表和 IDF。

        Args:
            texts: 文本列表，用于学习词汇分布
        """
        # 统计词频
        doc_count = len(texts)
        df = {}  # 文档频率
        for text in texts:
            tokens = self._tokenize(text)
            seen = set()
            for token in tokens:
                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)
                if token not in self._vocab:
                    self._vocab[token] = len(self._vocab)

        # 计算 IDF
        for token, count in df.items():
            self._idf[token] = max(0.0, math.log((doc_count + 1) / (count + 1)) + 1)

        # 如果词汇表超过 dim，截断到 dim
        if len(self._vocab) > self.dim:
            sorted_vocab = sorted(self._idf.items(), key=lambda x: x[1], reverse=True)
            keep = set(token for token, _ in sorted_vocab[:self.dim])
            new_vocab = {}
            for token in keep:
                new_vocab[token] = len(new_vocab)
            self._vocab = new_vocab

    def encode(self, text: str) -> list[float]:
        """
        将文本编码为向量。

        Args:
            text: 输入文本

        Returns:
            dim 维的浮点数向量
        """
        tokens = self._tokenize(text)
        if not tokens or not self._vocab:
            return [0.0] * self.dim

        # TF-IDF 向量
        vec = [0.0] * self.dim
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        for token, count in tf.items():
            if token in self._vocab and token in self._idf:
                idx = self._vocab[token]
                if idx < self.dim:
                    vec[idx] = count * self._idf[token]

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        简单的中英文分词。

        中文按字符 unigram+bigram，英文按空格和标点分词。
        """
        import re

        tokens = []
        # 分离中英文段落
        segments = re.split(r"([a-zA-Z]+)", text)

        for seg in segments:
            if re.match(r"[a-zA-Z]+", seg):
                # 英文单词
                tokens.append(seg.lower())
            else:
                # 中文按字符处理
                for i, ch in enumerate(seg):
                    if "一" <= ch <= "鿿":
                        tokens.append(ch)
                        if i + 1 < len(seg) and "一" <= seg[i + 1] <= "鿿":
                            tokens.append(ch + seg[i + 1])

        return tokens

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        计算两个向量的余弦相似度。

        Args:
            a: 向量 a
            b: 向量 b

        Returns:
            余弦相似度，范围 [-1, 1]
        """
        if len(a) != len(b):
            raise ValueError(f"向量维度不匹配: {len(a)} vs {len(b)}")

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


import math  # 需要此语句支持 EmbeddingUtil 中的 math 调用


# ---------------------------------------------------------------------------
# 选择策略抽象
# ---------------------------------------------------------------------------


class SelectionStrategy(ABC):
    """
    示例选择策略的抽象基类。

    What: 定义选择策略的统一接口。
    Why: 策略模式 — 允许在运行时切换不同的选择算法。
    When: 需要多种选择策略并统一管理时。
    """

    @abstractmethod
    def select(
        self,
        query: str,
        examples: list[Example],
        k: int,
        **kwargs,
    ) -> list[Example]:
        """
        从示例池中选择 k 个最合适的示例。

        Args:
            query: 用户查询
            examples: 可用示例池
            k: 选择的示例数量

        Returns:
            选中的 k 个示例
        """
        ...


class StaticSelection(SelectionStrategy):
    """
    静态选择策略：始终返回前 k 个示例。

    What: 最简单的选择策略，根据顺序选择。
    Why: 当示例已按优先级预排序时使用（如人工精选的示例集）。
    When: 示例池小且已按重要性排序。
    """

    def select(
        self, query: str, examples: list[Example], k: int, **kwargs
    ) -> list[Example]:
        return examples[:k]


class EmbeddingSimilaritySelection(SelectionStrategy):
    """
    嵌入相似度选择策略：选择与查询最相似的 k 个示例。

    What: 基于向量相似度动态选择最相关的 Few-shot 示例。
    Why: 不同的查询需要不同的示例才能达到最佳效果。
    When: 示例池较大（>10），且查询多样性高。
    """

    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None):
        """
        Args:
            embed_fn: 文本嵌入函数。如不提供，使用内建的 TF-IDF。
        """
        self._embed_util = EmbeddingUtil()
        self._embed_fn = embed_fn

    def select(
        self, query: str, examples: list[Example], k: int, **kwargs
    ) -> list[Example]:
        if not examples:
            return []

        # 确保示例有嵌入向量
        for ex in examples:
            if ex.embedding is None:
                ex.embedding = self._embed(ex.input_text)

        query_emb = self._embed(query)

        # 计算相似度
        scored = []
        for ex in examples:
            sim = EmbeddingUtil.cosine_similarity(query_emb, ex.embedding or [])
            scored.append((sim, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:k]]

    def _embed(self, text: str) -> list[float]:
        if self._embed_fn:
            return self._embed_fn(text)
        return self._embed_util.encode(text)


class MMRSelection(SelectionStrategy):
    """
    最大边际相关性 (Maximal Marginal Relevance, MMR) 选择策略。

    What: 在相似度和多样性之间取得平衡的选择算法。
    Why: 纯相似度选择可能导致选出的示例过于雷同，MMR 通过惩罚与已选示例的相似度来增加多样性。
    When: 需要避免示例冗余，确保覆盖不同角度时。

    公式: MMR = λ * sim(query, ex) - (1-λ) * max(sim(ex, selected_i))
    """

    def __init__(
        self,
        diversity_weight: float = 0.3,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        """
        Args:
            diversity_weight: 多样性权重 (0 = 纯相似度, 1 = 纯多样性)
            embed_fn: 嵌入函数
        """
        self.diversity_weight = diversity_weight
        self._embed_util = EmbeddingUtil()
        self._embed_fn = embed_fn

    def select(
        self, query: str, examples: list[Example], k: int, **kwargs
    ) -> list[Example]:
        if not examples:
            return []

        # 确保嵌入
        for ex in examples:
            if ex.embedding is None:
                ex.embedding = self._embed(ex.input_text)

        query_emb = self._embed(query)
        lambda_val = 1.0 - self.diversity_weight  # 相关性权重

        selected: list[Example] = []
        remaining = list(examples)

        for _ in range(min(k, len(examples))):
            best_score = float("-inf")
            best_example = None

            for ex in remaining:
                # 相关性得分
                relevance = EmbeddingUtil.cosine_similarity(query_emb, ex.embedding or [])

                # 多样性惩罚
                diversity_penalty = 0.0
                if selected:
                    max_sim_to_selected = max(
                        EmbeddingUtil.cosine_similarity(ex.embedding or [], s.embedding or [])
                        for s in selected
                    )
                    diversity_penalty = max_sim_to_selected

                mmr_score = lambda_val * relevance - self.diversity_weight * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_example = ex

            if best_example:
                selected.append(best_example)
                remaining.remove(best_example)

        return selected

    def _embed(self, text: str) -> list[float]:
        if self._embed_fn:
            return self._embed_fn(text)
        return self._embed_util.encode(text)


# ---------------------------------------------------------------------------
# FewShotSelector 主类
# ---------------------------------------------------------------------------


class FewShotSelector:
    """
    Few-shot 示例选择器。

    What: 管理示例池，使用指定的策略为每个查询选择最佳示例。
    Why: 集中管理示例的选择逻辑，支持多种策略和排序。
    When: 构建需要动态 Few-shot 的 LLM 应用时作为核心组件。

    使用示例:
        pool = [Example(...), Example(...), ...]
        selector = FewShotSelector(pool, strategy="embedding")
        selected = selector.select("如何优化数据库查询？", k=3)
    """

    STRATEGIES = {
        "static": StaticSelection,
        "embedding": EmbeddingSimilaritySelection,
        "mmr": MMRSelection,
    }

    def __init__(
        self,
        examples: list[Example],
        strategy: str = "static",
        embed_fn: Callable[[str], list[float]] | None = None,
        **strategy_kwargs,
    ):
        """
        Args:
            examples: 示例池
            strategy: 选择策略名称 ("static" | "embedding" | "mmr")
            embed_fn: 自定义嵌入函数（用于 embedding 和 mmr 策略）
            **strategy_kwargs: 传递给策略构造函数的额外参数
        """
        self.examples = list(examples)  # 防御性拷贝
        self.strategy_name = strategy
        self._embed_fn = embed_fn

        # 初始化策略
        strategy_cls = self.STRATEGIES.get(strategy)
        if strategy_cls is None:
            raise ValueError(f"未知策略 '{strategy}'。可用: {list(self.STRATEGIES.keys())}")

        if strategy in ("embedding", "mmr"):
            self._strategy = strategy_cls(embed_fn=embed_fn, **strategy_kwargs)
        else:
            self._strategy = strategy_cls(**strategy_kwargs)

    def select(
        self,
        query: str,
        k: int = 3,
        category: str | None = None,
    ) -> list[Example]:
        """
        为给定查询选择 k 个最佳示例。

        Args:
            query: 用户查询文本
            k: 返回的示例数量
            category: 可选的类别过滤

        Returns:
            选中的示例列表
        """
        # 类别过滤
        if category:
            pool = [ex for ex in self.examples if ex.category == category]
            if not pool:
                pool = self.examples  # 回退到全部
        else:
            pool = self.examples

        # 执行选择
        selected = self._strategy.select(query, pool, k)
        return selected

    def add_example(self, example: Example):
        """向示例池添加新示例"""
        self.examples.append(example)

    def remove_example(self, example_id: str) -> bool:
        """根据 ID 移除示例"""
        for i, ex in enumerate(self.examples):
            if ex.id == example_id:
                self.examples.pop(i)
                return True
        return False

    def get_pool_stats(self) -> dict:
        """获取示例池的统计信息"""
        categories = {}
        difficulties = []
        for ex in self.examples:
            categories[ex.category] = categories.get(ex.category, 0) + 1
            difficulties.append(ex.difficulty)

        return {
            "total": len(self.examples),
            "categories": categories,
            "avg_difficulty": sum(difficulties) / len(difficulties) if difficulties else 0,
            "strategy": self.strategy_name,
        }


# ---------------------------------------------------------------------------
# 示例排序策略
# ---------------------------------------------------------------------------


class ExampleOrderingStrategies:
    """
    示例排序策略集合。

    What: 提供多种方法对选中的示例进行排序。
    Why: 示例的排列顺序会显著影响模型性能（首因效应和近因效应）。
    When: 选中 k 个示例后，需要决定在提示中的排列顺序。

    已知效应:
    - 首因效应: 模型对列表开头的示例记忆更深
    - 近因效应: 模型对列表末尾的示例记忆更深
    - 将困难示例放在开头或结尾可以更好地引导模型
    """

    @staticmethod
    def ascending_difficulty(examples: list[Example]) -> list[Example]:
        """
        难度递增排序：简单 → 困难。

        What: 从最简单的示例开始，逐渐过渡到困难的示例。
        Why: 渐进式学习 — 模型先理解简单模式，再应对复杂情况。
        When: 示例难度差异大，希望稳定输出质量时。
        """
        return sorted(examples, key=lambda ex: ex.difficulty)

    @staticmethod
    def descending_difficulty(examples: list[Example]) -> list[Example]:
        """
        难度递减排序：困难 → 简单。

        What: 先展示最困难的示例，再展示简单的。
        Why: 利用首因效应 — 模型先看到最复杂的边界情况，更容易形成正确的认知框架。
        When: 边界情况是关键，需要模型首先注意时。
        """
        return sorted(examples, key=lambda ex: ex.difficulty, reverse=True)

    @staticmethod
    def interleaved(examples: list[Example]) -> list[Example]:
        """
        交错排序：简单和困难交替排列。

        What: 排列为 [简单, 困难, 简单, 困难, ...]。
        Why: 平衡首因和近因效应，避免模型过度偏向某类示例。
        When: 需要模型同时关注简单和复杂模式时。
        """
        sorted_ex = sorted(examples, key=lambda ex: ex.difficulty)
        n = len(sorted_ex)
        mid = (n + 1) // 2
        easy = sorted_ex[:mid]
        hard = sorted_ex[mid:]

        result = []
        for i in range(max(len(easy), len(hard))):
            if i < len(easy):
                result.append(easy[i])
            if i < len(hard):
                result.append(hard[i])

        return result

    @staticmethod
    def proximity(
        examples: list[Example], query_embedding: list[float]
    ) -> list[Example]:
        """
        邻近排序：按与查询的相似度排序（最相似的在最前面）。

        What: 与查询最相似的示例排在最前面。
        Why: 把最相关的示例放在模型最注意的位置。
        When: 查询有明确的语义方向时（特定领域或话题）。
        """
        scored = []
        for ex in examples:
            if ex.embedding:
                sim = EmbeddingUtil.cosine_similarity(query_embedding, ex.embedding)
            else:
                sim = 0.0
            scored.append((sim, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored]

    @staticmethod
    def random_shuffle(examples: list[Example]) -> list[Example]:
        """
        随机打乱：用于消融实验的基准。

        What: 完全随机排列示例。
        Why: 作为对照基准，评估其他排序策略的有效性。
        When: 进行 A/B 测试时作为对照组。
        """
        shuffled = list(examples)
        random.shuffle(shuffled)
        return shuffled


# ---------------------------------------------------------------------------
# 困难示例挖掘器
# ---------------------------------------------------------------------------


@dataclass
class HardExampleReport:
    """困难示例挖掘报告"""

    example: Example
    baseline_score: float           # 基线模型的得分
    failure_pattern: str            # 失败模式描述
    suggested_fix: str              # 改进建议
    error_rate: float = 0.0         # 历史错误率
    times_tested: int = 0           # 被测试次数


class HardExampleMiner:
    """
    困难示例挖掘器。

    What: 自动识别模型反复失败的示例，并将其加入训练池。
    Why: 最有价值的 Few-shot 示例往往是基线模型出错的案例 — 这些案例教会模型处理边界情况。
    When: 迭代优化模型性能，尤其是从生产错误日志中学习。

    工作流:
    1. 在测试集上运行基线模型
    2. 收集失败的案例
    3. 分析失败模式（分类错误、格式错误、缺失字段等）
    4. 人工确认后加入示例池
    """

    def __init__(
        self,
        evaluator: Callable[[Example, str], float],
        min_error_rate: float = 0.3,
        max_hard_examples: int = 20,
    ):
        """
        Args:
            evaluator: 评估函数，接受 (Example, model_output) -> score (0.0-1.0)
            min_error_rate: 最小错误率阈值，低于此值不视为困难示例
            max_hard_examples: 最多保留的困难示例数
        """
        self._evaluator = evaluator
        self.min_error_rate = min_error_rate
        self.max_hard_examples = max_hard_examples
        self._hard_examples: list[HardExampleReport] = []
        self._test_records: dict[str, list[float]] = {}  # example_id → scores

    def test(
        self, example: Example, model_output: str
    ) -> HardExampleReport:
        """
        测试单个示例，判断是否为困难示例。

        Args:
            example: 测试示例
            model_output: 模型对该示例的输出

        Returns:
            HardExampleReport
        """
        score = self._evaluator(example, model_output)

        # 记录历史得分
        if example.id not in self._test_records:
            self._test_records[example.id] = []
        self._test_records[example.id].append(score)

        scores = self._test_records[example.id]
        error_rate = 1.0 - (sum(scores) / len(scores))
        times_tested = len(scores)

        # 分析失败模式
        failure_pattern = self._analyze_failure(example, model_output, score)
        suggested_fix = self._suggest_fix(example, model_output, failure_pattern)

        report = HardExampleReport(
            example=example,
            baseline_score=score,
            failure_pattern=failure_pattern,
            suggested_fix=suggested_fix,
            error_rate=error_rate,
            times_tested=times_tested,
        )

        # 如果错误率高，加入困难示例池
        if error_rate >= self.min_error_rate and times_tested >= 2:
            self._add_to_pool(report)

        return report

    def get_hard_examples(self) -> list[Example]:
        """
        获取挖掘到的困难示例（按错误率降序）。

        Returns:
            困难示例列表（已转换为标准的 Example 对象）
        """
        sorted_reports = sorted(
            self._hard_examples, key=lambda r: r.error_rate, reverse=True
        )
        return [r.example for r in sorted_reports]

    def get_report(self) -> str:
        """
        生成困难示例挖掘总结报告。

        Returns:
            格式化的报告文本
        """
        if not self._hard_examples:
            return "暂无困难示例。"

        lines = ["=== 困难示例挖掘报告 ===", f"共发现 {len(self._hard_examples)} 个困难示例", ""]
        for i, report in enumerate(self._hard_examples[:10], 1):
            lines.append(
                f"{i}. [{report.example.category}] {report.example.input_text[:60]}..."
            )
            lines.append(f"   错误率: {report.error_rate:.1%}, "
                         f"基线分: {report.baseline_score:.2f}")
            lines.append(f"   失败模式: {report.failure_pattern}")
            lines.append(f"   改进建议: {report.suggested_fix}")
            lines.append("")

        return "\n".join(lines)

    def _add_to_pool(self, report: HardExampleReport):
        """将报告添加到困难示例池（维护最大数量）"""
        # 避免重复
        existing_ids = {r.example.id for r in self._hard_examples}
        if report.example.id in existing_ids:
            # 更新已有记录
            for i, r in enumerate(self._hard_examples):
                if r.example.id == report.example.id:
                    self._hard_examples[i] = report
                    return

        self._hard_examples.append(report)

        # 保持不超过最大数量
        if len(self._hard_examples) > self.max_hard_examples:
            self._hard_examples.sort(key=lambda r: r.error_rate)
            self._hard_examples = self._hard_examples[-self.max_hard_examples :]

    @staticmethod
    def _analyze_failure(
        example: Example, output: str, score: float
    ) -> str:
        """
        分析失败模式。

        Args:
            example: 原始示例
            output: 模型输出
            score: 评估得分

        Returns:
            失败模式分类
        """
        if score >= 0.8:
            return "基本正确（轻微偏差）"
        elif score >= 0.5:
            return "部分正确（关键信息缺失或不准）"
        elif len(output.strip()) == 0:
            return "空输出（模型未生成任何内容）"
        elif len(output) < len(example.output_text) * 0.3:
            return "输出过短（可能被截断）"
        elif any(kw in output.lower() for kw in ["sorry", "抱歉", "cannot", "无法"]):
            return "模型拒答（遇到无法处理的情况）"
        else:
            return "输出错误（与期望偏差较大）"

    @staticmethod
    def _suggest_fix(
        example: Example, output: str, failure_pattern: str
    ) -> str:
        """
        根据失败模式给出改进建议。

        Args:
            example: 原始示例
            output: 模型错误输出
            failure_pattern: 失败模式

        Returns:
            改进建议
        """
        suggestions = {
            "基本正确（轻微偏差）": "微调示例格式使其更精确，或添加更多约束条件",
            "部分正确（关键信息缺失或不准）": "在示例中明确标注关键信息字段，使用更强的输出格式约束",
            "空输出（模型未生成任何内容）": "检查输入是否包含敏感词或格式错误，考虑降低任务复杂度",
            "输出过短（可能被截断）": "增加 max_tokens 参数，或在提示中明确要求详细输出",
            "模型拒答（遇到无法处理的情况）": "添加类似难度的示例，让模型学习如何处理此类输入",
            "输出错误（与期望偏差较大）": "创建更相似的示例，或考虑将任务分解为子任务",
        }
        return suggestions.get(failure_pattern, "将案例加入训练集，增加类似示例")


# ---------------------------------------------------------------------------
# FewShotDatabase
# ---------------------------------------------------------------------------


class FewShotDatabase:
    """
    Few-shot 示例数据库。

    What: 持久化存储、版本管理和检索 Few-shot 示例。
    Why: 示例是宝贵的知识资产，需要版本控制、搜索和审计能力。
    When: 示例池超过 50 个，或需要跨项目共享示例。

    存储格式:
    - SQLite 数据库（用于元数据和索引）
    - JSON 文件备份（用于可移植性和版本控制）
    """

    def __init__(self, db_path: str = "fewshot_examples.db"):
        """
        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """创建数据库表（如果不存在）"""
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS examples (
                id TEXT PRIMARY KEY,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                difficulty REAL DEFAULT 0.5,
                embedding TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                updated_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                example_id TEXT,
                tag TEXT,
                PRIMARY KEY (example_id, tag),
                FOREIGN KEY (example_id) REFERENCES examples(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_stats (
                example_id TEXT,
                query TEXT,
                score REAL,
                used_at TEXT,
                FOREIGN KEY (example_id) REFERENCES examples(id)
            )
        """)
        self._conn.commit()

    def insert(self, example: Example):
        """
        插入一个新示例。

        Args:
            example: 要插入的示例
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO examples
            (id, input_text, output_text, category, difficulty, embedding,
             metadata, created_at, version, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                example.id,
                example.input_text,
                example.output_text,
                example.category,
                example.difficulty,
                json.dumps(example.embedding) if example.embedding else None,
                json.dumps(example.metadata, ensure_ascii=False),
                example.created_at,
                example.version,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def insert_batch(self, examples: list[Example]):
        """批量插入示例"""
        for ex in examples:
            self.insert(ex)

    def get(self, example_id: str) -> Example | None:
        """
        根据 ID 检索示例。

        Args:
            example_id: 示例 ID

        Returns:
            Example 对象，不存在则返回 None
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM examples WHERE id = ?", (example_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_example(dict(row))

    def search(
        self,
        category: str | None = None,
        min_difficulty: float | None = None,
        max_difficulty: float | None = None,
        tag: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Example]:
        """
        条件搜索示例。

        Args:
            category: 按类别过滤
            min_difficulty: 最低难度
            max_difficulty: 最高难度
            tag: 按标签过滤
            limit: 返回数量限制
            offset: 分页偏移

        Returns:
            符合条件的示例列表
        """
        query = "SELECT DISTINCT e.* FROM examples e"
        conditions = []
        params: list[Any] = []

        if tag:
            query += " JOIN tags t ON e.id = t.example_id"
            conditions.append("t.tag = ?")
            params.append(tag)

        if category:
            conditions.append("e.category = ?")
            params.append(category)

        if min_difficulty is not None:
            conditions.append("e.difficulty >= ?")
            params.append(min_difficulty)

        if max_difficulty is not None:
            conditions.append("e.difficulty <= ?")
            params.append(max_difficulty)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY e.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self._conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_example(dict(row)) for row in rows]

    def count(self, category: str | None = None) -> int:
        """统计示例数量"""
        cursor = self._conn.cursor()
        if category:
            cursor.execute(
                "SELECT COUNT(*) FROM examples WHERE category = ?", (category,)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM examples")
        return cursor.fetchone()[0]

    def add_tag(self, example_id: str, tag: str):
        """为示例添加标签"""
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO tags (example_id, tag) VALUES (?, ?)",
            (example_id, tag),
        )
        self._conn.commit()

    def record_usage(
        self, example_id: str, query: str, score: float
    ):
        """
        记录示例使用情况和效果。

        Args:
            example_id: 示例 ID
            query: 匹配到的查询
            score: 该示例在此查询下的效果得分
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO usage_stats (example_id, query, score, used_at) VALUES (?, ?, ?, ?)",
            (example_id, query, score, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def get_most_effective(
        self, limit: int = 10
    ) -> list[tuple[Example, float]]:
        """
        获取效果最好的示例（按历史平均得分排序）。

        Returns:
            (Example, 平均得分) 列表
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT e.*, AVG(u.score) as avg_score
            FROM examples e
            JOIN usage_stats u ON e.id = u.example_id
            GROUP BY e.id
            ORDER BY avg_score DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()

        results = []
        for row in rows:
            d = dict(row)
            avg = d.pop("avg_score")
            results.append((self._row_to_example(d), avg))
        return results

    def export_json(self, filepath: str):
        """
        导出所有示例为 JSON 文件。

        Args:
            filepath: 输出文件路径
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM examples ORDER BY created_at")
        rows = cursor.fetchall()

        examples_data = [self._row_to_example(dict(row)).to_dict() for row in rows]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(examples_data, f, ensure_ascii=False, indent=2)

    def import_json(self, filepath: str):
        """
        从 JSON 文件导入示例。

        Args:
            filepath: JSON 文件路径
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = [Example.from_dict(item) for item in data]
        self.insert_batch(examples)

    @staticmethod
    def _row_to_example(row: dict) -> Example:
        """将数据库行转换为 Example 对象"""
        return Example(
            id=row["id"],
            input_text=row["input_text"],
            output_text=row["output_text"],
            category=row.get("category", "general"),
            difficulty=row.get("difficulty", 0.5),
            embedding=json.loads(row["embedding"]) if row.get("embedding") else None,
            metadata=json.loads(row.get("metadata", "{}")),
            created_at=row.get("created_at", ""),
            version=row.get("version", 1),
        )

    def close(self):
        """关闭数据库连接"""
        self._conn.close()


# ---------------------------------------------------------------------------
# __main__ 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("Few-Shot 示例选择系统 - 完整演示")
    print("=" * 60)

    # 1. 构建示例池
    print("\n[1] 构建示例池...")
    pool = [
        Example(
            id="ex_001",
            input_text="如何用Python读取CSV文件？",
            output_text="使用pandas的read_csv()函数：pd.read_csv('file.csv')",
            category="python_basic",
            difficulty=0.1,
        ),
        Example(
            id="ex_002",
            input_text="如何在Python中处理大文件时的内存问题？",
            output_text="使用chunksize参数分块读取，或使用dask进行分布式处理",
            category="python_advanced",
            difficulty=0.7,
        ),
        Example(
            id="ex_003",
            input_text="Python列表和元组有什么区别？",
            output_text="列表可变(mutable)，元组不可变(immutable)；列表用[]，元组用()",
            category="python_basic",
            difficulty=0.2,
        ),
        Example(
            id="ex_004",
            input_text="Python装饰器的实现原理是什么？",
            output_text="装饰器本质是闭包，接受函数作为参数并返回新的函数",
            category="python_advanced",
            difficulty=0.6,
        ),
        Example(
            id="ex_005",
            input_text="Python中GIL是什么？如何绕过GIL限制？",
            output_text="GIL是全局解释器锁。绕过方法：使用多进程(multiprocessing)、C扩展、或异步IO",
            category="python_advanced",
            difficulty=0.9,
        ),
        Example(
            id="ex_006",
            input_text="如何用Python发送HTTP请求？",
            output_text="使用requests库：requests.get(url)或requests.post(url, data={})",
            category="python_basic",
            difficulty=0.1,
        ),
    ]
    print(f"示例池大小: {len(pool)}")

    # 2. 测试三种选择策略
    queries = [
        "如何优化Python代码的执行效率？",
        "怎么在Python中做并发编程？",
        "Python数据可视化的常用库有哪些？",
    ]

    for strategy_name in ["static", "embedding", "mmr"]:
        print(f"\n[2] 策略: {strategy_name}")
        selector = FewShotSelector(pool, strategy=strategy_name)

        for query in queries:
            selected = selector.select(query, k=3)
            print(f"\n  查询: {query}")
            for i, ex in enumerate(selected):
                print(f"    [{i + 1}] ({ex.category}, 难度{ex.difficulty:.1f}) {ex.input_text[:40]}...")

    # 3. 测试排序策略
    print("\n[3] 排序策略演示...")
    selected = pool[:5]  # 选取5个示例

    print("  原始顺序:")
    for ex in selected:
        print(f"    - 难度{ex.difficulty:.1f}: {ex.input_text[:40]}...")

    print("\n  难度递增:")
    for ex in ExampleOrderingStrategies.ascending_difficulty(selected):
        print(f"    - 难度{ex.difficulty:.1f}: {ex.input_text[:40]}...")

    print("\n  难度递减:")
    for ex in ExampleOrderingStrategies.descending_difficulty(selected):
        print(f"    - 难度{ex.difficulty:.1f}: {ex.input_text[:40]}...")

    print("\n  交错排序:")
    for ex in ExampleOrderingStrategies.interleaved(selected):
        print(f"    - 难度{ex.difficulty:.1f}: {ex.input_text[:40]}...")

    # 4. 测试困难示例挖掘器
    print("\n[4] 困难示例挖掘器演示...")

    def simple_evaluator(example: Example, output: str) -> float:
        """简单评估器：基于关键词匹配"""
        expected_words = set(example.output_text.lower().split())
        output_words = set(output.lower().split())
        if not expected_words:
            return 0.0
        overlap = len(expected_words & output_words)
        return overlap / len(expected_words)

    miner = HardExampleMiner(
        evaluator=simple_evaluator,
        min_error_rate=0.3,
        max_hard_examples=10,
    )

    # 模拟一些"错误"的模型输出
    test_cases = [
        (pool[0], "使用csv库的reader功能"),  # 部分正确
        (pool[0], "用Excel打开"),             # 错误
        (pool[4], "不知道GIL是什么"),         # 错误
        (pool[4], "听不懂"),                  # 严重错误
    ]

    for example, fake_output in test_cases:
        report = miner.test(example, fake_output)
        print(f"\n  示例: {example.input_text[:40]}...")
        print(f"  得分: {report.baseline_score:.2f}")
        print(f"  失败模式: {report.failure_pattern}")

    print(f"\n{miner.get_report()}")

    # 5. 测试数据库
    print("\n[5] FewShotDatabase 演示...")
    db = FewShotDatabase(":memory:")  # 内存数据库

    db.insert_batch(pool)
    db.add_tag("ex_001", "csv")
    db.add_tag("ex_001", "file-io")
    db.add_tag("ex_004", "decorator")

    print(f"总示例数: {db.count()}")
    print(f"Python基础类别: {db.count('python_basic')}")
    print(f"Python高级类别: {db.count('python_advanced')}")

    # 搜索
    results = db.search(category="python_basic", limit=5)
    print(f"\n  搜索 'python_basic': 找到 {len(results)} 个示例")
    for ex in results:
        print(f"    - {ex.input_text[:40]}...")

    # 导出
    db.export_json("/tmp/fewshot_export_demo.json")
    print(f"\n  已导出到 /tmp/fewshot_export_demo.json")

    db.close()

    print("\n" + "=" * 60)
    print("演示完成！所有组件功能正常。")
    print("=" * 60)
