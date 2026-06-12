#!/usr/bin/env python3
"""
S1-02：Prompt 回归测试
=========================
当你修改了 Prompt 模板，效果是变好了还是变差了？
回归测试用统计方法回答这个问题。

核心理念：
- 基线（Baseline）：旧 Prompt 在完整测试集上的表现
- 新版本（New）：新 Prompt 在同一测试集上的表现
- 门禁规则：新版本 < 95% 基线 → 阻断发布

依赖：pip install numpy scipy
"""

import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class TestCase:
    """单个测试用例。"""
    query: str
    expected_keywords: List[str]       # 期望回答中包含的关键词
    expected_sources: List[str] = field(default_factory=list)  # 期望引用的来源
    category: str = "general"          # 分类：factual/reasoning/code/multi_hop
    difficulty: str = "medium"         # easy/medium/hard
    max_acceptable_hallucination: bool = False  # 是否允许部分幻觉

@dataclass
class TestResult:
    """单个用例的测试结果。"""
    query: str
    answer: str
    metrics: Dict[str, float]  # 各项指标得分
    passed: bool
    details: str = ""

@dataclass
class RegressionReport:
    """回归测试报告。"""
    baseline_score: float
    new_version_score: float
    score_change_pct: float
    improved_cases: int
    degraded_cases: int
    degraded_details: List[dict]
    overall_pass: bool
    recommendation: str


# ============================================================================
# PromptRegressionTest 框架
# ============================================================================

class PromptRegressionTest:
    """
    Prompt 回归测试框架。

    使用流程：
    1. 定义测试集（100+ 条用例）
    2. 用当前 Prompt（Baseline）跑一遍，记录各指标
    3. 修改 Prompt 后，用新 Prompt 跑一遍
    4. 对比 → 如果退化 > 5%，阻断发布
    """

    def __init__(self, test_cases: List[TestCase],
                 degradation_threshold: float = 0.95):
        """
        Args:
            test_cases: 测试用例集
            degradation_threshold: 退化阈值（默认 0.95，即允许最多 5% 退化）
        """
        self.test_cases = test_cases
        self.degradation_threshold = degradation_threshold
        self.baseline_results: Dict[str, TestResult] = {}
        self.new_results: Dict[str, TestResult] = {}

    def evaluate_answer(self, query: str, answer: str,
                        test_case: TestCase) -> TestResult:
        """
        评估单个回答的质量。

        评估维度：
        1. 关键词覆盖率（expected_keywords 中命中了多少）
        2. 来源引用率（是否引用了 expected_sources）
        3. 答案长度合理性（不太短也不太长）
        4. 格式完整性
        """
        metrics = {}

        # 1. 关键词覆盖
        keyword_hits = sum(
            1 for kw in test_case.expected_keywords if kw.lower() in answer.lower()
        )
        metrics["keyword_recall"] = (
            keyword_hits / len(test_case.expected_keywords)
            if test_case.expected_keywords else 1.0
        )

        # 2. 来源引用
        if test_case.expected_sources:
            source_hits = sum(
                1 for src in test_case.expected_sources
                if src.lower() in answer.lower()
            )
            metrics["source_citation_rate"] = source_hits / len(test_case.expected_sources)
        else:
            metrics["source_citation_rate"] = 1.0

        # 3. 答案长度
        answer_len = len(answer)
        if answer_len < 20:
            metrics["length_score"] = 0.3  # 太短
        elif answer_len < 50:
            metrics["length_score"] = 0.7
        elif answer_len < 2000:
            metrics["length_score"] = 1.0
        else:
            metrics["length_score"] = 0.8  # 太长但可接受

        # 4. 格式完整性（含有结构标记）
        format_score = 0.0
        if "来源" in answer or "参考" in answer:
            format_score += 0.3
        if "。" in answer or "." in answer:  # 有标点
            format_score += 0.3
        if len(answer.split("\n")) > 1:  # 有段落
            format_score += 0.2
        if "**" in answer or "##" in answer:  # 有格式标记
            format_score += 0.2
        metrics["format_score"] = format_score

        # 综合得分（加权平均）
        composite = (
            metrics["keyword_recall"] * 0.35
            + metrics["source_citation_rate"] * 0.30
            + metrics["length_score"] * 0.15
            + metrics["format_score"] * 0.20
        )

        passed = composite >= 0.70  # 及格线 70%

        return TestResult(
            query=query,
            answer=answer,
            metrics=metrics,
            passed=passed,
            details=f"综合得分: {composite:.2f}"
        )

    def run_baseline(self, answer_fn: callable) -> Dict[str, TestResult]:
        """运行基线测试。"""
        print(f"  [基线测试] 测试 {len(self.test_cases)} 条用例...")
        results = {}
        for tc in self.test_cases:
            answer = answer_fn(tc.query)
            result = self.evaluate_answer(tc.query, answer, tc)
            results[tc.query] = result
        self.baseline_results = results
        return results

    def run_new_version(self, answer_fn: callable) -> Dict[str, TestResult]:
        """运行新版本测试。"""
        print(f"  [新版本测试] 测试 {len(self.test_cases)} 条用例...")
        results = {}
        for tc in self.test_cases:
            answer = answer_fn(tc.query)
            result = self.evaluate_answer(tc.query, answer, tc)
            results[tc.query] = result
        self.new_results = results
        return results

    def compare(self) -> RegressionReport:
        """对比基线和新版本。"""
        if not self.baseline_results or not self.new_results:
            raise ValueError("请先运行 run_baseline() 和 run_new_version()")

        # 计算各问题得分
        baseline_scores = []
        new_scores = []
        improved = 0
        degraded = 0
        degraded_details = []

        for tc in self.test_cases:
            b_result = self.baseline_results[tc.query]
            n_result = self.new_results[tc.query]

            b_score = sum(b_result.metrics.values()) / len(b_result.metrics)
            n_score = sum(n_result.metrics.values()) / len(n_result.metrics)

            baseline_scores.append(b_score)
            new_scores.append(n_score)

            if n_score > b_score + 0.05:
                improved += 1
            elif n_score < b_score - 0.05:
                degraded += 1
                degraded_details.append({
                    "query": tc.query,
                    "baseline_score": round(b_score, 3),
                    "new_score": round(n_score, 3),
                    "degradation": round((b_score - n_score) / b_score * 100, 1) if b_score > 0 else 0,
                })

        avg_baseline = np.mean(baseline_scores)
        avg_new = np.mean(new_scores)

        if avg_baseline > 0:
            change_pct = ((avg_new - avg_baseline) / avg_baseline) * 100
        else:
            change_pct = 0

        # 门禁判断
        if avg_baseline > 0:
            ratio = avg_new / avg_baseline
        else:
            ratio = 1.0

        overall_pass = ratio >= self.degradation_threshold

        if overall_pass:
            recommendation = "✅ 通过 — 新 Prompt 表现满足门禁要求，可以发布。"
        elif ratio >= 0.90:
            recommendation = "⚠️ 警告 — 有退化但仍在可接受范围。建议检查退化的用例。"
        else:
            recommendation = "❌ 阻断 — 严重退化！请修复 Prompt 后重新测试。"

        return RegressionReport(
            baseline_score=round(avg_baseline, 4),
            new_version_score=round(avg_new, 4),
            score_change_pct=round(change_pct, 2),
            improved_cases=improved,
            degraded_cases=degraded,
            degraded_details=degraded_details,
            overall_pass=overall_pass,
            recommendation=recommendation,
        )


# ============================================================================
# 演示
# ============================================================================

def demo_prompt_regression():
    """Prompt 回归测试演示。"""
    print("=" * 60)
    print("Prompt 回归测试演示")
    print("=" * 60)

    # 定义测试集
    test_cases = [
        TestCase(
            query="公司的年假政策是什么？",
            expected_keywords=["年假", "天", "申请"],
            expected_sources=["员工手册"],
            category="factual",
            difficulty="easy",
        ),
        TestCase(
            query="如何申请报销？",
            expected_keywords=["报销", "申请", "流程", "审批"],
            expected_sources=["财务制度"],
            category="factual",
            difficulty="easy",
        ),
        TestCase(
            query="RAG系统和传统搜索有什么区别？",
            expected_keywords=["RAG", "检索", "生成", "区别", "传统"],
            category="reasoning",
            difficulty="medium",
        ),
        TestCase(
            query="分析AI技术对公司业务的影响",
            expected_keywords=["AI", "影响", "业务"],
            category="reasoning",
            difficulty="hard",
        ),
        TestCase(
            query="Python 如何读取 CSV 文件？",
            expected_keywords=["csv", "read", "pandas", "open"],
            category="code",
            difficulty="easy",
        ),
        TestCase(
            query="解释微服务架构和单体架构各自的优缺点",
            expected_keywords=["微服务", "单体", "优缺点", "对比"],
            category="reasoning",
            difficulty="hard",
        ),
    ]

    # 创建回归测试框架
    regression = PromptRegressionTest(
        test_cases=test_cases,
        degradation_threshold=0.95,
    )

    # 模拟基线 Prompt 的回答函数
    def baseline_answer_fn(query: str) -> str:
        """基线 Prompt：详细回答。"""
        answers = {
            "公司的年假政策是什么？":
                "根据《员工手册》第3章，公司年假政策如下：入职满1年享受5天年假，"
                "满3年享受10天，满5年享受15天。申请需提前2周在HR系统中提交。"
                "\n\n📚 来源：《员工手册》2025版 第3章",
            "如何申请报销？":
                "报销申请流程：1) 登录财务系统，2) 填写报销单，3) 上传发票，"
                "4) 提交审批，5) 审批通过后3-5个工作日到账。"
                "\n\n📚 来源：《财务管理制度》v2.1 第5条",
            "RAG系统和传统搜索有什么区别？":
                "RAG（检索增强生成）与传统搜索的核心区别：1) 传统搜索返回链接列表，"
                "RAG返回综合答案；2) RAG能理解上下文并进行多步推理；"
                "3) RAG可以整合多个来源的信息生成连贯回答。"
                "\n\n📚 来源：《RAG技术白皮书》",
            "分析AI技术对公司业务的影响":
                "AI技术将从以下维度影响公司业务：1) 效率提升：自动化重复性工作，"
                "预计节省30%人力；2) 决策支持：数据分析辅助管理层决策；"
                "3) 客户体验：智能客服提升响应速度。"
                "\n\n📚 来源：内部AI战略规划报告",
            "Python 如何读取 CSV 文件？":
                "在 Python 中读取 CSV 文件有多种方式：\n"
                "1) 使用 pandas: `df = pd.read_csv('file.csv')`\n"
                "2) 使用 csv 模块: `reader = csv.reader(open('file.csv'))`\n"
                "3) 使用 numpy: `data = np.loadtxt('file.csv', delimiter=',')`\n"
                "\n推荐使用 pandas，功能最强大。",
            "解释微服务架构和单体架构各自的优缺点":
                "微服务 vs 单体架构对比：\n\n"
                "**微服务优点**：独立部署、技术栈灵活、可扩展性好\n"
                "**微服务缺点**：运维复杂度高、网络延迟、数据一致性挑战\n\n"
                "**单体优点**：开发简单、部署方便、调试容易\n"
                "**单体缺点**：扩展困难、技术栈锁定、代码耦合",
        }
        return answers.get(query, f"[默认回答] 关于'{query}'的回答...")

    # 模拟新 Prompt 的回答函数（有些退化）
    def new_answer_fn(query: str) -> str:
        """新 Prompt：更简洁但有时缺少细节。"""
        answers = {
            "公司的年假政策是什么？":
                "年假为5-15天，按工龄递增。请在HR系统申请。",  # 缺少细节
            "如何申请报销？":
                "报销申请流程：登录系统→填单→上传发票→提交→等待到账。"
                "\n📚 来源：《财务管理制度》",
            "RAG系统和传统搜索有什么区别？":
                "RAG 生成答案，传统搜索返回链接。RAG更智能。",  # 过于简略
            "分析AI技术对公司业务的影响":
                "AI可以提高效率，改善决策，提升客户体验。",  # 缺少深度
            "Python 如何读取 CSV 文件？":
                "使用 pandas 的 read_csv 函数或 csv 模块。",  # 缺少代码
            "解释微服务架构和单体架构各自的优缺点":
                "微服务灵活但复杂，单体简单但不够灵活。",  # 过于简略
        }
        return answers.get(query, f"[新回答] 关于'{query}'的回答...")

    # 运行测试
    print("\n[1] 运行基线测试...")
    baseline = regression.run_baseline(baseline_answer_fn)
    for query, result in baseline.items():
        print(f"  [{result.passed}] {query[:40]}... 得分: "
              f"{sum(result.metrics.values())/len(result.metrics):.3f}")

    print("\n[2] 运行新版本测试...")
    new_results = regression.run_new_version(new_answer_fn)
    for query, result in new_results.items():
        print(f"  [{result.passed}] {query[:40]}... 得分: "
              f"{sum(result.metrics.values())/len(result.metrics):.3f}")

    print("\n[3] 对比分析...")
    report = regression.compare()

    print(f"\n  {'='*50}")
    print(f"  Prompt 回归测试报告")
    print(f"  {'='*50}")
    print(f"  基线得分:     {report.baseline_score:.4f}")
    print(f"  新版本得分:   {report.new_version_score:.4f}")
    print(f"  变化:         {report.score_change_pct:+.2f}%")
    print(f"  改进用例:     {report.improved_cases}")
    print(f"  退化用例:     {report.degraded_cases}")
    print(f"  门禁通过:     {report.overall_pass}")
    print(f"  {'='*50}")
    print(f"  {report.recommendation}")

    if report.degraded_details:
        print(f"\n  退化详情:")
        for d in report.degraded_details:
            print(f"    - {d['query'][:50]}: "
                  f"{d['baseline_score']:.3f} → {d['new_score']:.3f} "
                  f"({d['degradation']}%)")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    demo_prompt_regression()

    print("\n[完成] Prompt 回归测试演示结束。")
    print("  [提示] 生产环境建议:")
    print("    1. 测试集至少 100+ 条，覆盖各种场景")
    print("    2. 每次 Prompt 变更必须运行回归测试")
    print("    3. 退化阈值设为 0.95（允许 5% 以内的退化）")
    print("    4. 用 CI/CD 自动化运行（见 s1-06）")
