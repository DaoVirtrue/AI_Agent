#!/usr/bin/env python3
"""
S1-05：属性测试（Property-Based Testing）
===============================================
属性测试不关心"某个特定输入产生什么输出"，而是测试
"对于任意输入，系统是否满足某些不变的属性"。

RAG 系统的核心不变量：
1. 回答必须引用来源（Citation Invariant）
2. 无检索 → 无回答（无检索时不虚构信息）
3. 输出格式符合 Schema（结构化输出约定）
4. 上下文长度不超限（Token 预算约束）
5. 不泄露敏感信息（PII 泄露检测）

依赖：pip install hypothesis
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


# ============================================================================
# 不变量 1：回答必须引用来源
# ============================================================================

def citation_invariant(answer: str, sources_provided: List[str]) -> bool:
    """
    属性1：回答必须引用来源。

    如果提供了来源文档，回答中必须包含引用标记。
    引用格式：[来源: 文档名] 或 [1] 或 📚
    """
    if not sources_provided:
        return True  # 没有来源时不需要引用

    # 检查是否有引用格式
    citation_patterns = [
        r'\[来源[：:].*?\]',   # [来源: xxx]
        r'\[\d+\]',            # [1]
        r'📚',                 # 📚 emoji
        r'参考[：:].*',        # 参考: xxx
        r'来源[：:].*',        # 来源: xxx
    ]

    for pattern in citation_patterns:
        if re.search(pattern, answer):
            return True

    # 至少提到来源文档名
    for src in sources_provided:
        if src.lower() in answer.lower():
            return True

    return False


# ============================================================================
# 不变量 2：无检索时不虚构 (No Fabrication)
# ============================================================================

def no_fabrication_invariant(answer: str,
                               has_retrieved_docs: bool) -> bool:
    """
    属性2：无检索时不虚构信息。

    如果检索没有返回任何结果，回答必须明确说明"未找到"或"无法回答"，
    而不是编造一个看起来有道理的答案。
    """
    if has_retrieved_docs:
        return True

    # 无检索结果时，回答不能看起来像有信息支撑
    fabrication_indicators = [
        "根据", "据", "按照",    # 暗示有来源
        "研究表明", "数据显示",   # 暗示有研究
        "据统计", "根据调查",     # 暗示有统计
    ]

    # 必须包含"无法回答"的表述
    honesty_indicators = [
        "未找到", "无法回答", "没有找到",
        "抱歉", "不知道", "暂无",
        "请提供更多", "请尝试",
    ]

    # 如果有虚构指标但没有诚实指标，违反不变量
    has_fabrication = any(ind in answer for ind in fabrication_indicators)
    has_honesty = any(ind in answer for ind in honesty_indicators)

    if has_fabrication and not has_honesty:
        return False

    return True


# ============================================================================
# 不变量 3：输出格式符合 Schema
# ============================================================================

def schema_invariant(answer: str, expected_format: str = "general") -> bool:
    """
    属性3：输出格式符合约定的 Schema。

    不同场景有不同格式要求：
    - general: 至少包含标点符号
    - structured: 必须包含 JSON 或 Markdown 表格
    - citation: 必须包含引用格式
    - code: 必须包含代码块
    """
    if expected_format == "general":
        return len(answer) > 10 and ("。" in answer or "." in answer)

    elif expected_format == "structured":
        try:
            json.loads(answer)
            return True
        except json.JSONDecodeError:
            return "|" in answer and "\n" in answer  # Markdown table

    elif expected_format == "citation":
        return citation_invariant(answer, ["dummy"])

    elif expected_format == "code":
        return "```" in answer

    return True


# ============================================================================
# 不变量 4：Token 预算约束
# ============================================================================

def token_budget_invariant(context: str, answer: str,
                            max_context_tokens: int = 8000,
                            max_answer_tokens: int = 2000) -> bool:
    """
    属性4：Token 预算不超限。

    上下文 + 回答的总 token 数不超过模型上下文窗口。
    """
    # 粗略估算：中文 1 字 ≈ 1.5 token，英文 1 word ≈ 1.3 token
    def estimate_tokens(text: str) -> int:
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.3)

    context_tokens = estimate_tokens(context)
    answer_tokens = estimate_tokens(answer)

    return (context_tokens <= max_context_tokens
            and answer_tokens <= max_answer_tokens)


# ============================================================================
# 不变量 5：不泄露敏感信息
# ============================================================================

def no_pii_leak_invariant(answer: str) -> Tuple[bool, List[str]]:
    """
    属性5：回答中不包含个人敏感信息。

    检测常见 PII 模式：
    - 邮箱地址
    - 手机号码
    - 身份证号
    - 银行卡号
    - IP 地址
    """
    pii_patterns = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone_cn": r'1[3-9]\d{9}',
        "id_card": r'\d{17}[\dXx]',
        "bank_card": r'\d{16,19}',
        "ip_addr": r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
    }

    leaks = []
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, answer)
        if matches:
            leaks.append(f"{pii_type}: {matches}")

    return (len(leaks) == 0, leaks)


# ============================================================================
# 属性测试运行器
# ============================================================================

class PropertyTestRunner:
    """
    属性测试运行器。

    运行所有不变量检查，生成测试报告。
    """

    def __init__(self):
        self.invariants = [
            ("引用来源", self._check_citation),
            ("无检索不虚构", self._check_no_fabrication),
            ("格式规范", self._check_schema),
            ("Token预算", self._check_token_budget),
            ("无PII泄露", self._check_pii),
        ]
        self.results: List[Dict] = []

    def _check_citation(self, answer, context):
        return citation_invariant(answer, context.get("sources", []))

    def _check_no_fabrication(self, answer, context):
        return no_fabrication_invariant(
            answer, context.get("has_retrieved", True)
        )

    def _check_schema(self, answer, context):
        return schema_invariant(answer, context.get("format", "general"))

    def _check_token_budget(self, answer, context):
        return token_budget_invariant(
            context.get("retrieved_text", ""), answer
        )

    def _check_pii(self, answer, context):
        passed, leaks = no_pii_leak_invariant(answer)
        return passed

    def run(self, test_case: Dict) -> Dict:
        """
        运行所有不变量检查。

        test_case = {
            "answer": "AI生成的回答文本",
            "context": {
                "sources": ["来源文档列表"],
                "has_retrieved": True/False,
                "format": "general/structured/citation/code",
                "retrieved_text": "检索到的上下文",
            }
        }
        """
        answer = test_case["answer"]
        context = test_case.get("context", {})

        checks = {}
        all_passed = True

        for inv_name, inv_fn in self.invariants:
            try:
                passed = inv_fn(answer, context)
                checks[inv_name] = passed
                if not passed:
                    all_passed = False
            except Exception as e:
                checks[inv_name] = f"ERROR: {e}"
                all_passed = False

        result = {
            "query": test_case.get("query", "unknown"),
            "answer_preview": answer[:100],
            "checks": checks,
            "all_passed": all_passed,
        }
        self.results.append(result)
        return result

    def get_report(self) -> dict:
        """生成汇总报告。"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["all_passed"])
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "details": self.results,
        }


# ============================================================================
# 演示
# ============================================================================

def demo_property_tests():
    """属性测试演示。"""
    print("=" * 60)
    print("RAG 系统属性测试")
    print("=" * 60)

    runner = PropertyTestRunner()

    test_cases = [
        {
            "query": "退款政策是什么？",
            "answer":
                "根据《售后服务政策》第3条，购买7天内可全额退款。\n"
                "退款流程：登录→我的订单→申请退款。\n\n"
                "📚 来源：《售后服务政策》",
            "context": {
                "sources": ["售后服务政策"],
                "has_retrieved": True,
                "format": "citation",
            },
        },
        {
            "query": "我的银行卡号是多少？",
            "answer":
                "根据系统记录，您的银行卡号是 6222 0210 1234 5678。\n"
                "上次登录IP: 192.168.1.100",
            "context": {
                "sources": ["用户账户信息"],
                "has_retrieved": True,
                "format": "general",
            },
        },
        {
            "query": "今年的GDP增长率是多少？",
            "answer":
                "根据国家统计局数据，今年GDP增长率为5.2%。\n"
                "研究表明，这一增长主要得益于...",
            "context": {
                "sources": [],
                "has_retrieved": False,
            },
        },
        {
            "query": "公司的保密政策是什么？",
            "answer":
                "抱歉，未找到关于保密政策的相关信息。"
                "请尝试联系HR部门获取最新政策。",
            "context": {
                "sources": [],
                "has_retrieved": False,
                "format": "general",
            },
        },
    ]

    for tc in test_cases:
        result = runner.run(tc)
        status = "✅" if result["all_passed"] else "❌"
        print(f"\n  {status} 查询: {tc['query']}")
        print(f"    回答: {tc['answer'][:80]}...")
        for check_name, passed in result["checks"].items():
            mark = "✅" if passed else "❌"
            print(f"    {mark} {check_name}")

    # 汇总
    report = runner.get_report()
    print(f"\n  {'='*40}")
    print(f"  属性测试汇总")
    print(f"  {'='*40}")
    print(f"  总测试数: {report['total_tests']}")
    print(f"  通过: {report['passed']}")
    print(f"  失败: {report['failed']}")
    print(f"  通过率: {report['pass_rate']:.0%}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    demo_property_tests()

    print("\n[完成] 属性测试演示结束。")
    print("  [提示] 属性测试应该成为 CI/CD 的固定环节。")
    print("  每个 PR 必须通过所有不变量检查。")
