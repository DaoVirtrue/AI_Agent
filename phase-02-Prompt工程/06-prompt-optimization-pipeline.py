#!/usr/bin/env python3
"""
06 - Prompt 优化管道 (Prompt Optimization Pipeline)

本模块提供完整的提示优化管道，包括：
- TestCase: 测试用例定义
- PromptVariant: 提示变体定义
- EvaluationResult: 评估结果
- BuiltInEvaluator: 内置多层评估器（精确匹配→语义相似度→Jaccard）
- PromptOptimizationPipeline: 完整的优化管道（评估、统计分析、报告生成）

所有注释使用中文，代码标识符使用英文。
"""

import json
import math
import time
import statistics
import concurrent.futures
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """
    单个测试用例。

    What: 表示一个（输入，期望输出）的测试对，附带分类和难度。
    Why: 标准化的测试数据结构，用于统一评估不同提示变体。
    When: 任何需要自动化评估提示效果的场景。
    """

    id: str                              # 唯一标识符
    input: str                           # 用户输入
    expected_output: str                 # 期望输出
    category: str = "general"            # 测试类别
    difficulty: str = "medium"           # 难度: "easy" | "medium" | "hard"
    metadata: dict = field(default_factory=dict)  # 额外元数据

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PromptVariant:
    """
    提示变体：一组（系统提示，用户模板）的组合。

    What: 表示一个具体的提示设计方案。
    Why: 在 A/B 测试中，每个变体是实验组或对照组的提示。
    When: 设计多个提示方案进行对比评估时。
    """

    name: str                            # 变体名称
    system_prompt: str                   # 系统提示文本
    user_template: str                   # 用户提示模板（{input} 作为占位符）
    description: str = ""                # 变体描述
    tags: list[str] = field(default_factory=list)

    def build_user_prompt(self, test_input: str) -> str:
        """
        用测试输入填充用户模板。

        Args:
            test_input: 测试用例的输入

        Returns:
            填充后的用户提示
        """
        return self.user_template.replace("{input}", test_input)


@dataclass
class EvaluationResult:
    """
    单个评估结果。

    What: 记录一个提示变体在一个测试用例上的表现。
    Why: 精细化的评估数据用于统计分析和诊断。
    When: 每次 run_evaluation 产生的每条测试-变体组合。
    """

    variant_name: str                    # 变体名称
    test_case_id: str                    # 测试用例 ID
    output: str                          # 模型实际输出
    score: float                         # 综合得分 0.0~1.0
    latency_ms: float                    # 响应延迟（毫秒）
    tokens_used: int                     # 使用的 token 数
    category: str = "general"            # 测试类别
    difficulty: str = "medium"           # 测试难度
    evaluator_level: str = "unknown"     # 使用的评估层级
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineReport:
    """
    管道执行报告。

    What: 聚合所有变体的评估结果和统计分析。
    Why: 提供决策所需的所有数据（哪个变体最好、好在哪、统计显著性）。
    When: 优化管道执行完毕后。
    """

    timestamp: str
    model: str
    variants: dict[str, PromptVariant] = field(default_factory=dict)
    results: list[EvaluationResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    statistical_tests: dict[str, Any] = field(default_factory=dict)
    stratified_analysis: dict[str, Any] = field(default_factory=dict)

    def to_json(self, filepath: str):
        """导出报告为 JSON 文件"""
        data = {
            "timestamp": self.timestamp,
            "model": self.model,
            "results": [
                {
                    "variant": r.variant_name,
                    "test_case": r.test_case_id,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "tokens_used": r.tokens_used,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "evaluator_level": r.evaluator_level,
                }
                for r in self.results
            ],
            "summary": self.summary,
            "statistical_tests": self.statistical_tests,
            "stratified_analysis": self.stratified_analysis,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 内置评估器
# ---------------------------------------------------------------------------


class BuiltInEvaluator:
    """
    内置多层评估器。

    What: 将模型输出与期望输出进行比较，返回 0.0~1.0 的得分。
    Why: 自动化评估需要可比、可复现的评分标准。
    When: 运行优化管道时的默认评估器。

    评估层级（按优先级递减）:
    1. Exact Match:         输出 == 期望 → 1.0
    2. Semantic Similarity: 基于词重叠的语义相似度
    3. Jaccard Fallback:    最低保证的集合相似度
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
    ):
        """
        Args:
            case_sensitive: 是否区分大小写
            normalize_whitespace: 是否标准化空白字符
        """
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    def evaluate(
        self, expected: str, actual: str
    ) -> tuple[float, str]:
        """
        评估模型输出与期望输出的匹配度。

        Args:
            expected: 期望输出
            actual: 模型实际输出

        Returns:
            (得分 0.0~1.0, 使用的评估层级名称)
        """
        # 预处理
        exp = self._preprocess(expected)
        act = self._preprocess(actual)

        if not act:
            return 0.0, "empty_output"

        # Level 1: 精确匹配
        if exp == act:
            return 1.0, "exact_match"

        # Level 2: 词级别 F1（视为语义相似度的近似）
        f1_score, matched = self._word_f1(exp, act)
        if f1_score >= 0.8:
            return f1_score, "semantic_similarity"

        # Level 3: Jaccard 相似度回退
        jaccard = self._jaccard(exp, act)
        return jaccard, "jaccard_fallback"

    def _preprocess(self, text: str) -> str:
        """文本预处理"""
        if self.normalize_whitespace:
            text = " ".join(text.split())
        if not self.case_sensitive:
            text = text.lower()
        return text.strip()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        简单分词：中文字符作为独立 token，英文按空格和标点分词。
        """
        import re
        tokens = []
        for segment in re.split(r"(\s+)", text):
            if segment.strip():
                # 分离中文和英文
                sub = re.findall(r"[一-鿿]|[a-zA-Z0-9]+|[^\w\s]", segment)
                tokens.extend(sub)
        return tokens

    def _word_f1(self, expected: str, actual: str) -> tuple[float, bool]:
        """
        计算词级别的 F1 分数。

        Returns:
            (F1 分数, 是否有任何匹配)
        """
        exp_tokens = set(self._tokenize(expected))
        act_tokens = set(self._tokenize(actual))

        if not exp_tokens:
            return 0.0, False

        tp = len(exp_tokens & act_tokens)

        if tp == 0:
            return 0.0, False

        precision = tp / len(act_tokens) if act_tokens else 0
        recall = tp / len(exp_tokens)

        if precision + recall == 0:
            return 0.0, False

        f1 = 2 * precision * recall / (precision + recall)
        return f1, True

    @staticmethod
    def _jaccard(expected: str, actual: str) -> float:
        """
        Jaccard 相似度（字符级 bigram）。

        What: 计算两个文本的字符 bigram 集合的 Jaccard 系数。
        Why: 最通用的相似度度量，不依赖语言特性。
        When: 词级别 F1 效果不佳时（如输出格式完全不同）。
        """
        def bigrams(s: str) -> set[str]:
            return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

        exp_bg = bigrams(expected)
        act_bg = bigrams(actual)

        intersection = exp_bg & act_bg
        union = exp_bg | act_bg

        if not union:
            return 0.0
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# 统计分析工具
# ---------------------------------------------------------------------------


class StatisticalAnalyzer:
    """
    统计分析器。

    What: 提供 Welch's t-test 和 Cohen's d 效应量计算。
    Why: 确定两个变体之间的差异是否具有统计显著性，而不仅看均值差异。
    When: 优化管道中 compare_variants() 的核心计算引擎。
    """

    @staticmethod
    def welch_t_test(
        scores_a: list[float], scores_b: list[float]
    ) -> dict[str, Any]:
        """
        Welch's t-test（不等方差 t 检验）。

        What: 比较两组独立样本的均值是否有显著差异，不假设方差相等。
        Why: LLM 评估结果的方差通常不同（某些变体更稳，某些波动大）。
        When: 比较两个提示变体的得分分布时。

        Args:
            scores_a: 变体 A 的得分列表
            scores_b: 变体 B 的得分列表

        Returns:
            {"t_statistic": float, "p_value": float, "significant": bool}
        """
        n_a = len(scores_a)
        n_b = len(scores_b)

        if n_a < 2 or n_b < 2:
            return {
                "t_statistic": None,
                "p_value": None,
                "significant": None,
                "error": "样本量不足（每组至少需要2个样本）",
            }

        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)

        var_a = statistics.variance(scores_a) if n_a > 1 else 0
        var_b = statistics.variance(scores_b) if n_b > 1 else 0

        se_a = var_a / n_a
        se_b = var_b / n_b

        if se_a + se_b == 0:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
            }

        # t 统计量
        t_stat = (mean_a - mean_b) / math.sqrt(se_a + se_b)

        # Welch-Satterthwaite 自由度
        df_num = (se_a + se_b) ** 2
        df_den = (se_a**2) / (n_a - 1) + (se_b**2) / (n_b - 1)
        df = df_num / df_den if df_den > 0 else 0

        # p 值（双侧检验，使用近似）
        p_value = StatisticalAnalyzer._approx_t_pvalue(abs(t_stat), df)

        return {
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "mean_a": round(mean_a, 4),
            "mean_b": round(mean_b, 4),
            "diff": round(mean_a - mean_b, 4),
            "df": round(df, 2),
        }

    @staticmethod
    def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
        """
        Cohen's d 效应量。

        What: 衡量两组数据均值差异的标准化大小（与样本量无关）。
        Why: 统计显著性（p-value）受样本量影响；效应量告诉我们差异的"实际大小"。
        When: 报告结果时应同时报告 p-value 和效应量。

        解释:
        - 0.2: 小效应
        - 0.5: 中等效应
        - 0.8: 大效应
        """
        n_a = len(scores_a)
        n_b = len(scores_b)

        if n_a < 2 or n_b < 2:
            return 0.0

        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)

        var_a = statistics.variance(scores_a)
        var_b = statistics.variance(scores_b)

        # 合并标准差
        pooled_sd = math.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        )

        if pooled_sd == 0:
            return 0.0

        return (mean_a - mean_b) / pooled_sd

    @staticmethod
    def _approx_t_pvalue(t_abs: float, df: float) -> float:
        """
        t 分布的 p 值近似计算（Beta 函数法）。

        What: 在没有 scipy 的情况下计算 t 检验的 p 值。
        Why: 避免对 scipy 的硬依赖，使用数学近似。
        When: 需要 p 值但没有 scipy 时。

        参考: Abramowitz and Stegun 近似公式
        """
        if df <= 0:
            return 1.0

        # 使用 Beta 正则化不完全函数的数值近似
        x = df / (df + t_abs * t_abs)

        # 不完全 Beta 函数近似（用于较大自由度）
        if df < 1:
            return 1.0

        # 简化近似：对中等自由度足够精确
        a = df / 2
        b = 0.5

        # 连分数展开
        if x < (a + 1) / (a + b + 2):
            # 使用连分数
            return 2 * StatisticalAnalyzer._betai_approx(a, b, x)
        else:
            return 2 * (1 - StatisticalAnalyzer._betai_approx(b, a, 1 - x))

    @staticmethod
    def _betai_approx(a: float, b: float, x: float) -> float:
        """
        不完全 Beta 函数的连分数近似。
        """
        if x == 0:
            return 0.0
        if x == 1:
            return 1.0

        # 使用 ln Gamma 计算
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

        # 连分数
        front = math.exp(
            math.log(x) * a + math.log(1 - x) * b - log_beta
        ) / a

        # 简化连分数迭代（Lentz 算法）
        fpm = 1.0
        f = 1.0
        c = 1.0
        d = 0.0

        for _ in range(200):
            m = _ // 2 + 1
            if _ % 2 == 1:
                am = -((a + m - 1) * (a + b + m - 1) * x) / ((a + 2 * m - 2) * (a + 2 * m - 1))
            else:
                am = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))

            d = 1 / (am * d + 1) if d != 0 else 1 / am
            c = am + 1 / c if c != 0 else am
            f *= c * d

            if abs(c * d - 1) < 1e-12:
                break

        return front * (f - 1)


# ---------------------------------------------------------------------------
# Prompt 优化管道
# ---------------------------------------------------------------------------


class PromptOptimizationPipeline:
    """
    Prompt 优化管道。

    What: 自动化评估、比较和优化提示变体的完整管道。
    Why: 手工评估提示效果效率低、主观性强；自动化管道提供可复现的数据驱动决策。
    When: 设计新提示、优化现有提示、或进行 A/B 测试时。

    工作流:
    1. load_test_cases(jsonl)   加载测试用例
    2. add_variant(variant)     添加要测试的提示变体
    3. run_evaluation()         运行评估（可并行）
    4. compare_variants()       统计分析变体差异
    5. stratified_analysis()    按类别和难度分层分析
    6. export_report()          导出完整报告

    使用示例:
        pipeline = PromptOptimizationPipeline(
            llm_call_fn=my_llm_call,
            evaluator=BuiltInEvaluator(),
        )
        pipeline.load_test_cases("test_cases.jsonl")
        pipeline.add_variant(PromptVariant(name="baseline", ...))
        pipeline.add_variant(PromptVariant(name="improved", ...))
        pipeline.run_evaluation(parallel=True)
        report = pipeline.export_report("report.json")
    """

    def __init__(
        self,
        llm_call_fn: Callable[[str, str], tuple[str, float, int]],
        evaluator: BuiltInEvaluator | None = None,
        model: str = "gpt-4o-mini",
    ):
        """
        Args:
            llm_call_fn: LLM 调用函数，签名: (system_prompt, user_prompt) -> (output, latency_ms, tokens)
            evaluator: 评估器实例（默认使用 BuiltInEvaluator）
            model: 模型名称（用于报告）
        """
        self._llm_call = llm_call_fn
        self._evaluator = evaluator or BuiltInEvaluator()
        self.model = model

        self._test_cases: list[TestCase] = []
        self._variants: dict[str, PromptVariant] = {}
        self._results: list[EvaluationResult] = []

    def load_test_cases(self, filepath: str):
        """
        从 JSONL 或 JSON 文件加载测试用例。

        Args:
            filepath: JSONL 文件路径（每行一个 JSON 对象）或 JSON 文件（数组）

        JSONL 格式:
        {"id": "tc_001", "input": "...", "expected_output": "...", "category": "math", "difficulty": "hard"}

        或 JSON 数组格式:
        [{"id": "tc_001", ...}, {"id": "tc_002", ...}]
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content.startswith("["):
            # JSON 数组
            data = json.loads(content)
        else:
            # JSONL（每行一个 JSON）
            data = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        self._test_cases = [TestCase.from_dict(item) for item in data]
        print(f"已加载 {len(self._test_cases)} 个测试用例")

    def add_variant(self, variant: PromptVariant):
        """
        添加提示变体。

        Args:
            variant: PromptVariant 实例
        """
        self._variants[variant.name] = variant
        print(f"已添加变体: {variant.name}")

    def run_evaluation(
        self, parallel: bool = False, max_workers: int = 5
    ) -> list[EvaluationResult]:
        """
        运行评估：对所有变体 × 所有测试用例执行评估。

        Args:
            parallel: 是否并行执行
            max_workers: 最大并行工作线程数

        Returns:
            EvaluationResult 列表
        """
        if not self._test_cases:
            raise ValueError("没有加载测试用例。请先调用 load_test_cases()。")
        if not self._variants:
            raise ValueError("没有添加变体。请先调用 add_variant()。")

        print(f"开始评估: {len(self._variants)} 个变体 × {len(self._test_cases)} 个测试用例")
        self._results = []

        if parallel:
            return self._run_parallel(max_workers)
        else:
            return self._run_sequential()

    def _run_sequential(self) -> list[EvaluationResult]:
        """顺序执行评估"""
        total = len(self._variants) * len(self._test_cases)
        count = 0

        for vname, variant in self._variants.items():
            print(f"\n  评估变体: {vname}")
            for tc in self._test_cases:
                count += 1
                user_prompt = variant.build_user_prompt(tc.input)
                output, latency, tokens = self._llm_call(
                    variant.system_prompt, user_prompt
                )
                score, eval_level = self._evaluator.evaluate(
                    tc.expected_output, output
                )

                result = EvaluationResult(
                    variant_name=vname,
                    test_case_id=tc.id,
                    output=output,
                    score=score,
                    latency_ms=latency,
                    tokens_used=tokens,
                    category=tc.category,
                    difficulty=tc.difficulty,
                    evaluator_level=eval_level,
                )
                self._results.append(result)

                if count % 5 == 0 or count == total:
                    print(f"    进度: {count}/{total} ({count / total:.0%})")

        return self._results

    def _run_parallel(self, max_workers: int) -> list[EvaluationResult]:
        """并行执行评估"""
        tasks = []
        for vname, variant in self._variants.items():
            for tc in self._test_cases:
                tasks.append((vname, variant, tc))

        print(f"  使用 {max_workers} 个工作线程并行执行 {len(tasks)} 个任务...")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for vname, variant, tc in tasks:
                user_prompt = variant.build_user_prompt(tc.input)
                future = executor.submit(
                    self._llm_call, variant.system_prompt, user_prompt
                )
                future_map[future] = (vname, tc)

            for i, future in enumerate(concurrent.futures.as_completed(future_map)):
                vname, tc = future_map[future]
                try:
                    output, latency, tokens = future.result(timeout=60)
                    score, eval_level = self._evaluator.evaluate(
                        tc.expected_output, output
                    )
                    result = EvaluationResult(
                        variant_name=vname,
                        test_case_id=tc.id,
                        output=output,
                        score=score,
                        latency_ms=latency,
                        tokens_used=tokens,
                        category=tc.category,
                        difficulty=tc.difficulty,
                        evaluator_level=eval_level,
                    )
                    results.append(result)
                except Exception as e:
                    results.append(EvaluationResult(
                        variant_name=vname,
                        test_case_id=tc.id,
                        output=f"ERROR: {e}",
                        score=0.0,
                        latency_ms=0,
                        tokens_used=0,
                        category=tc.category,
                        difficulty=tc.difficulty,
                        evaluator_level="error",
                    ))

                if (i + 1) % 5 == 0 or i + 1 == len(tasks):
                    print(f"    进度: {i + 1}/{len(tasks)} ({(i + 1) / len(tasks):.0%})")

        self._results = results
        return results

    def compare_variants(self) -> dict[str, Any]:
        """
        比较所有变体的性能，执行统计检验。

        What: 两两比较所有变体，计算 Welch's t-test 和 Cohen's d。
        Why: 不仅看均值，还要确认差异是否统计显著。
        When: 评估完成后，需要决定哪个变体最好时。

        Returns:
            {
                "variant_means": {变体名: 平均得分},
                "pairwise_tests": {
                    "variant_A vs variant_B": {
                        "t_statistic": ..., "p_value": ..., "significant": ...,
                        "cohens_d": ...
                    }
                },
                "best_variant": str,
                "ranking": [(变体名, 平均得分), ...]
            }
        """
        if not self._results:
            raise ValueError("没有评估结果。请先调用 run_evaluation()。")

        analyzer = StatisticalAnalyzer()

        # 按变体聚合得分
        variant_scores: dict[str, list[float]] = {}
        variant_latencies: dict[str, list[float]] = {}
        for r in self._results:
            variant_scores.setdefault(r.variant_name, []).append(r.score)
            variant_latencies.setdefault(r.variant_name, []).append(r.latency_ms)

        # 计算各变体均值
        means = {}
        for vname, scores in variant_scores.items():
            means[vname] = {
                "mean_score": round(statistics.mean(scores), 4),
                "median_score": round(statistics.median(scores), 4),
                "std_score": round(statistics.stdev(scores) if len(scores) > 1 else 0, 4),
                "mean_latency_ms": round(statistics.mean(variant_latencies[vname]), 1),
                "count": len(scores),
            }

        # 两两 t-test
        variant_names = list(variant_scores.keys())
        pairwise = {}
        for va, vb in combinations(variant_names, 2):
            t_result = analyzer.welch_t_test(variant_scores[va], variant_scores[vb])
            d = analyzer.cohens_d(variant_scores[va], variant_scores[vb])
            key = f"{va} vs {vb}"
            pairwise[key] = {
                **t_result,
                "cohens_d": round(d, 4),
                "cohens_d_interpretation": (
                    "large" if abs(d) >= 0.8 else
                    "medium" if abs(d) >= 0.5 else
                    "small" if abs(d) >= 0.2 else
                    "negligible"
                ),
            }

        # 排序
        ranking = sorted(means.items(), key=lambda x: x[1]["mean_score"], reverse=True)
        best = ranking[0][0] if ranking else None

        return {
            "variant_means": means,
            "pairwise_tests": pairwise,
            "best_variant": best,
            "ranking": ranking,
        }

    def stratified_analysis(self) -> dict[str, Any]:
        """
        分层分析：按类别和难度拆解各变体表现。

        What: 在不同子集上分别计算各变体的平均得分。
        Why: 整体较好可能在某个子集上劣于其他变体 — 分层分析揭示隐藏的弱点。
        When: 需要深入了解变体在不同类型任务上的表现差异时。

        Returns:
            {
                "by_category": {类别: {变体: 平均分}},
                "by_difficulty": {难度: {变体: 平均分}},
                "by_category_difficulty": {类别×难度: {变体: 平均分}},
            }
        """
        if not self._results:
            raise ValueError("没有评估结果。请先调用 run_evaluation()。")

        by_category: dict[str, dict[str, list[float]]] = {}
        by_difficulty: dict[str, dict[str, list[float]]] = {}
        by_combined: dict[str, dict[str, list[float]]] = {}

        for r in self._results:
            # 按类别
            by_category.setdefault(r.category, {}).setdefault(r.variant_name, []).append(r.score)
            # 按难度
            by_difficulty.setdefault(r.difficulty, {}).setdefault(r.variant_name, []).append(r.score)
            # 按类别+难度
            key = f"{r.category}/{r.difficulty}"
            by_combined.setdefault(key, {}).setdefault(r.variant_name, []).append(r.score)

        def aggregate(d: dict) -> dict:
            """将分数列表聚合为统计量"""
            result = {}
            for group_key, variants in d.items():
                result[group_key] = {}
                for vname, scores in variants.items():
                    result[group_key][vname] = {
                        "mean": round(statistics.mean(scores), 4),
                        "count": len(scores),
                        "std": round(statistics.stdev(scores) if len(scores) > 1 else 0, 4),
                    }
            return result

        return {
            "by_category": aggregate(by_category),
            "by_difficulty": aggregate(by_difficulty),
            "by_category_difficulty": aggregate(by_combined),
        }

    def export_report(self, filepath: str | None = None) -> PipelineReport:
        """
        生成并导出完整的评估报告。

        Args:
            filepath: 可选的 JSON 输出路径

        Returns:
            PipelineReport 对象
        """
        if not self._results:
            raise ValueError("没有评估结果。请先调用 run_evaluation()。")

        comparison = self.compare_variants()
        stratified = self.stratified_analysis()

        # 构建汇总
        summary = {
            "total_variants": len(self._variants),
            "total_test_cases": len(self._test_cases),
            "total_evaluations": len(self._results),
            "best_variant": comparison["best_variant"],
            "overall_scores": comparison["variant_means"],
        }

        report = PipelineReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=self.model,
            variants=dict(self._variants),
            results=list(self._results),
            summary=summary,
            statistical_tests=comparison["pairwise_tests"],
            stratified_analysis=stratified,
        )

        if filepath:
            report.to_json(filepath)
            print(f"报告已导出到: {filepath}")

        return report

    def print_summary(self):
        """打印评估摘要到控制台"""
        if not self._results:
            print("暂无评估结果。")
            return

        comparison = self.compare_variants()

        print("\n" + "=" * 60)
        print("Prompt 优化管道 - 评估摘要")
        print("=" * 60)
        print(f"模型: {self.model}")
        print(f"变体数: {len(self._variants)}")
        print(f"测试用例数: {len(self._test_cases)}")
        print(f"总评估次数: {len(self._results)}")

        print("\n--- 各变体表现 ---")
        for vname, stats in comparison["variant_means"].items():
            marker = " ★" if vname == comparison["best_variant"] else ""
            print(
                f"  {vname}{marker}: "
                f"均值={stats['mean_score']:.3f}, "
                f"中位数={stats['median_score']:.3f}, "
                f"标准差={stats['std_score']:.3f}, "
                f"延迟={stats['mean_latency_ms']:.0f}ms, "
                f"N={stats['count']}"
            )

        print("\n--- 统计检验 ---")
        for pair, test in comparison["pairwise_tests"].items():
            sig = "显著" if test.get("significant") else "不显著"
            print(
                f"  {pair}: "
                f"t={test['t_statistic']}, "
                f"p={test['p_value']} ({sig}), "
                f"Cohen's d={test['cohens_d']} ({test['cohens_d_interpretation']})"
            )

        print("=" * 60)


# ---------------------------------------------------------------------------
# __main__ 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("Prompt 优化管道 - 完整演示")
    print("=" * 60)

    # 1. 准备模拟 LLM 调用函数
    def mock_llm_call(system_prompt: str, user_prompt: str) -> tuple[str, float, int]:
        """
        模拟 LLM 调用。

        实际使用中替换为真实的 OpenAI API 调用。
        此处根据提示变体名称模拟不同质量的输出。
        """
        import random
        import time as _time

        _time.sleep(0.05)  # 模拟网络延迟

        # 模拟输出
        if "详细" in system_prompt:
            # "详细"变体：模拟更高质量的回复
            outputs = [
                "Python 的 GIL（全局解释器锁）确保同一时刻只有一个线程执行 Python 字节码。绕过方法包括：1) 使用 multiprocessing 模块启动多个进程；2) 使用 C 扩展释放 GIL；3) 使用 asyncio 进行异步 I/O。",
                "使用 pandas 的 read_csv() 函数：pd.read_csv('file.csv')",
            ]
        elif "简洁" in system_prompt:
            # "简洁"变体：模拟简洁回复
            outputs = [
                "GIL是全局解释器锁。绕过方法：多进程、C扩展、异步IO。",
                "pd.read_csv('file.csv')",
            ]
        else:
            # 基线变体
            outputs = [
                "Python 的 GIL 是全局解释器锁，限制了多线程性能。",
                "使用 pandas 读取：pd.read_csv()",
            ]

        output = outputs[hash(user_prompt) % len(outputs)]
        latency = 200 + random.uniform(-50, 100)
        tokens = len(output) // 2

        return output, latency, tokens

    # 2. 创建优化管道
    pipeline = PromptOptimizationPipeline(
        llm_call_fn=mock_llm_call,
        evaluator=BuiltInEvaluator(),
        model="mock-model",
    )

    # 3. 准备测试用例
    test_cases = [
        TestCase(
            id="tc_001",
            input="Python 中的 GIL 是什么？如何绕过它？",
            expected_output="GIL 是全局解释器锁。绕过方法：多进程、C 扩展、异步 I/O。",
            category="python",
            difficulty="hard",
        ),
        TestCase(
            id="tc_002",
            input="如何在 Python 中读取 CSV 文件？",
            expected_output="使用 pandas 的 read_csv() 函数：pd.read_csv('file.csv')",
            category="python",
            difficulty="easy",
        ),
        TestCase(
            id="tc_003",
            input="Python 装饰器是什么？",
            expected_output="装饰器是接受函数作为参数并返回新函数的闭包，用于在不修改原函数的情况下添加功能。",
            category="python",
            difficulty="medium",
        ),
        TestCase(
            id="tc_004",
            input="Python 中列表和元组的区别？",
            expected_output="列表可变用[]，元组不可变用()。列表适合需要修改的集合，元组适合固定数据。",
            category="python",
            difficulty="easy",
        ),
        TestCase(
            id="tc_005",
            input="什么是 Python 的上下文管理器？",
            expected_output="上下文管理器通过 __enter__ 和 __exit__ 方法实现资源自动管理，使用 with 语句调用。",
            category="python",
            difficulty="medium",
        ),
    ]

    # 写入临时文件（演示 load_test_cases）
    temp_test_file = "/tmp/prompt_test_cases.jsonl"
    with open(temp_test_file, "w", encoding="utf-8") as f:
        for tc in test_cases:
            f.write(json.dumps({
                "id": tc.id,
                "input": tc.input,
                "expected_output": tc.expected_output,
                "category": tc.category,
                "difficulty": tc.difficulty,
            }, ensure_ascii=False) + "\n")

    pipeline.load_test_cases(temp_test_file)

    # 4. 添加提示变体
    pipeline.add_variant(PromptVariant(
        name="baseline",
        system_prompt="你是一个 Python 编程助手。请简洁回答用户的问题。",
        user_template="问题：{input}\n回答：",
        description="基线：通用角色 + 简单指令",
        tags=["baseline", "simple"],
    ))

    pipeline.add_variant(PromptVariant(
        name="detailed_expert",
        system_prompt="你是一个拥有10年经验的资深 Python 开发专家。请提供详细且准确的回答，包含代码示例和性能考量。",
        user_template="请详细回答以下 Python 相关问题：\n{input}\n\n请包含：\n1. 概念解释\n2. 代码示例\n3. 常见陷阱",
        description="详细专家：角色框架 + 结构化输出指令",
        tags=["expert", "structured", "detailed"],
    ))

    pipeline.add_variant(PromptVariant(
        name="concise_bullet",
        system_prompt="你是一个 Python 速查手册。回答必须简洁，使用条列式。",
        user_template="用户问：{input}\n\n请用条列式简洁回答（不超过3点）：",
        description="简洁条列：限制输出格式和长度",
        tags=["concise", "bullet_points"],
    ))

    # 5. 运行评估
    print("\n运行评估...")
    pipeline.run_evaluation(parallel=True, max_workers=3)

    # 6. 比较变体
    comparison = pipeline.compare_variants()
    print(f"\n最佳变体: {comparison['best_variant']}")
    print(f"排名: {comparison['ranking']}")

    # 7. 分层分析
    stratified = pipeline.stratified_analysis()
    print("\n按难度分层:")
    for difficulty, variants in stratified["by_difficulty"].items():
        print(f"  {difficulty}:")
        for vname, stats in variants.items():
            print(f"    {vname}: 均值={stats['mean']:.3f} (N={stats['count']})")

    # 8. 打印摘要
    pipeline.print_summary()

    # 9. 导出报告
    pipeline.export_report("/tmp/prompt_optimization_report.json")

    print("\n" + "=" * 60)
    print("演示完成！")
    print(f"临时文件: {temp_test_file}")
    print(f"报告文件: /tmp/prompt_optimization_report.json")
    print("=" * 60)
