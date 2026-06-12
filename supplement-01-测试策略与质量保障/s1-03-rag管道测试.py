#!/usr/bin/env python3
"""
S1-03：RAG 管道测试 —— 检索质量 / 生成质量 / E2E 管道
==========================================================
RAG 系统的测试需要覆盖三个层次：
1. 检索质量：检索到的文档是否相关（Hit Rate, MRR）
2. 生成质量：答案是否忠实于来源（Faithfulness, Relevancy）
3. E2E 管道：端到端输入→输出是否满足业务要求

依赖：无需额外安装
"""

import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math


# ============================================================================
# 第一部分：检索质量测试
# ============================================================================

class RetrievalQualityTest:
    """
    检索质量测试。

    核心指标：
    - Hit Rate: 相关文档是否出现在 Top-K 结果中
    - MRR (Mean Reciprocal Rank): 第一个相关结果的排名的倒数平均值
    - Precision@K: Top-K 结果中相关结果的比例
    - Recall@K: 所有相关文档中被检索到的比例
    """

    def __init__(self):
        self.results = []

    def test_batch(
        self,
        queries: List[str],
        ground_truth_doc_ids: List[List[str]],  # 每个查询的真实相关文档ID
        retrieval_fn: callable,                 # 检索函数: (query) → [(doc_id, score), ...]
        top_k: int = 5,
    ) -> dict:
        """批量检索质量测试。"""
        hit_at_k = []
        mrr_list = []
        precision_at_k = []
        recall_at_k = []

        for i, query in enumerate(queries):
            gt_ids = set(ground_truth_doc_ids[i])
            retrieved = retrieval_fn(query)[:top_k]
            retrieved_ids = [r[0] for r in retrieved]

            # Hit Rate
            hits = len(gt_ids & set(retrieved_ids))
            hit_at_k.append(1 if hits > 0 else 0)

            # MRR
            for rank, r_id in enumerate(retrieved_ids, 1):
                if r_id in gt_ids:
                    mrr_list.append(1.0 / rank)
                    break
            else:
                mrr_list.append(0.0)

            # Precision@K
            precision_at_k.append(hits / len(retrieved_ids) if retrieved_ids else 0)

            # Recall@K
            recall_at_k.append(hits / len(gt_ids) if gt_ids else 1.0)

        metrics = {
            "hit_rate@k": sum(hit_at_k) / len(hit_at_k),
            "mrr": sum(mrr_list) / len(mrr_list),
            "precision@k": sum(precision_at_k) / len(precision_at_k),
            "recall@k": sum(recall_at_k) / len(recall_at_k),
            "num_queries": len(queries),
            "top_k": top_k,
        }

        self.results.append(metrics)
        return metrics

    def assert_hit_rate_above(self, threshold: float = 0.85):
        """断言 Hit Rate 高于阈值。"""
        if not self.results:
            raise AssertionError("请先运行 test_batch()")
        hr = self.results[-1]["hit_rate@k"]
        assert hr >= threshold, (
            f"Hit Rate {hr:.2f} 低于阈值 {threshold:.2f}"
        )
        return True

    def assert_mrr_above(self, threshold: float = 0.70):
        """断言 MRR 高于阈值。"""
        if not self.results:
            raise AssertionError("请先运行 test_batch()")
        mrr = self.results[-1]["mrr"]
        assert mrr >= threshold, (
            f"MRR {mrr:.2f} 低于阈值 {threshold:.2f}"
        )
        return True


# ============================================================================
# 第二部分：生成质量测试
# ============================================================================

class GenerationQualityTest:
    """
    生成质量测试。

    核心指标：
    - Faithfulness（忠实度）：答案是否基于检索到的文档
    - Answer Relevancy（回答相关性）：答案是否与问题相关
    - Completeness（完整性）：是否覆盖了所有关键信息点
    - Hallucination Detection（幻觉检测）
    """

    def test_faithfulness(
        self,
        answer: str,
        source_documents: List[str],
    ) -> dict:
        """
        测试答案对来源文档的忠实度。

        逐句检查答案中的陈述是否能在来源文档中找到支撑。
        """
        sentences = [s.strip() for s in answer.replace("！", "。").split("。") if s.strip()]
        if not sentences:
            return {"faithfulness": 0, "details": "空回答"}

        faithful_count = 0
        details = []

        for sent in sentences:
            # 检查句子是否在来源文档中有支撑
            supported = any(
                self._has_support(sent, doc) for doc in source_documents
            )
            if supported:
                faithful_count += 1
            details.append({
                "sentence": sent[:100],
                "faithful": supported,
            })

        faithfulness = faithful_count / len(sentences)
        return {
            "faithfulness": round(faithfulness, 3),
            "total_sentences": len(sentences),
            "faithful_sentences": faithful_count,
            "details": details,
        }

    def _has_support(self, sentence: str, document: str) -> bool:
        """检查句子是否在文档中有支撑（简化版：关键词重叠）。"""
        # 生产环境应用 LLM 做 NLI（Natural Language Inference）
        sent_words = set(sentence)
        doc_words = set(document)
        if not sent_words:
            return False
        overlap = len(sent_words & doc_words) / len(sent_words)
        return overlap > 0.3  # 30% 关键词重叠视为有支撑

    def assert_faithfulness_above(self, threshold: float = 0.90):
        """断言忠实度高于阈值。"""

    def test_relevancy(
        self,
        question: str,
        answer: str,
    ) -> dict:
        """
        测试回答与问题的相关性。

        检查回答是否直接回应了问题，而非跑题。
        """
        # 简化的关键词重叠检查
        q_words = set(question)
        a_words = set(answer)

        if not q_words:
            return {"relevancy": 0}

        overlap = len(q_words & a_words) / len(q_words)

        # 检查是否包含反问/重新表述（表明理解了问题）
        has_reframing = any(
            kw in answer for kw in question.split()[:3]
        )

        return {
            "relevancy": round(min(overlap * (1.2 if has_reframing else 1.0), 1.0), 3),
            "has_reframing": has_reframing,
        }


# ============================================================================
# 第三部分：E2E 管道测试
# ============================================================================

class E2EPipelineTest:
    """
    端到端管道测试。

    测试完整的 RAG 管道：输入 → 检索 → 生成 → 输出。
    """

    def __init__(self, pipeline_fn: callable = None):
        """
        Args:
            pipeline_fn: 管道函数 (query: str) → {"answer": str, "sources": list}
        """
        self.pipeline_fn = pipeline_fn

    def test_e2e(
        self,
        test_cases: List[Dict],
    ) -> dict:
        """
        端到端测试。

        每个测试用例格式：
        {
            "query": "问题文本",
            "expected_answer_contains": ["关键词1", "关键词2"],
            "expected_sources_min": 1,         # 期望的最少来源数
            "expected_response_time_max": 5.0, # 最大响应时间（秒）
            "blocked_keywords": ["不应该", "出现"], # 不应出现的内容
        }
        """
        results = []

        for tc in test_cases:
            import time
            start = time.time()
            output = self.pipeline_fn(tc["query"])
            elapsed = time.time() - start

            checks = {}
            all_passed = True

            # 检查关键词
            if "expected_answer_contains" in tc:
                hits = sum(
                    1 for kw in tc["expected_answer_contains"]
                    if kw in output.get("answer", "")
                )
                checks["keywords"] = {
                    "required": len(tc["expected_answer_contains"]),
                    "found": hits,
                    "passed": hits == len(tc["expected_answer_contains"]),
                }
                all_passed &= checks["keywords"]["passed"]

            # 检查来源数量
            if "expected_sources_min" in tc:
                sources = output.get("sources", [])
                checks["sources"] = {
                    "expected_min": tc["expected_sources_min"],
                    "actual": len(sources),
                    "passed": len(sources) >= tc["expected_sources_min"],
                }
                all_passed &= checks["sources"]["passed"]

            # 检查响应时间
            if "expected_response_time_max" in tc:
                checks["response_time"] = {
                    "expected_max": tc["expected_response_time_max"],
                    "actual": round(elapsed, 3),
                    "passed": elapsed <= tc["expected_response_time_max"],
                }
                all_passed &= checks["response_time"]["passed"]

            # 检查禁用词
            if "blocked_keywords" in tc:
                found_blocked = [
                    kw for kw in tc["blocked_keywords"]
                    if kw in output.get("answer", "")
                ]
                checks["blocked"] = {
                    "found": found_blocked,
                    "passed": len(found_blocked) == 0,
                }
                all_passed &= checks["blocked"]["passed"]

            results.append({
                "query": tc["query"],
                "passed": all_passed,
                "checks": checks,
                "elapsed": round(elapsed, 3),
            })

        pass_count = sum(1 for r in results if r["passed"])
        return {
            "total": len(results),
            "passed": pass_count,
            "failed": len(results) - pass_count,
            "pass_rate": pass_count / len(results) if results else 0,
            "results": results,
        }


# ============================================================================
# 演示
# ============================================================================

def demo_rag_pipeline_tests():
    """RAG 管道测试演示。"""
    print("=" * 60)
    print("RAG 管道测试演示")
    print("=" * 60)

    # ---- 1. 检索质量测试 ----
    print("\n【1】检索质量测试")

    # 模拟知识库
    DOCS = {
        "d1": "RAG 是检索增强生成技术，结合了信息检索和文本生成。",
        "d2": "向量数据库用于存储文本的嵌入向量，支持相似度搜索。",
        "d3": "大语言模型如 GPT-4 可以进行多轮对话和复杂推理。",
        "d4": "嵌入模型将文本转换为向量表示，常用 BGE 和 OpenAI Embeddings。",
        "d5": "Python 是最流行的编程语言之一，广泛用于数据科学和 AI 开发。",
    }

    def mock_retrieval(query: str):
        """模拟检索函数。"""
        results = []
        for doc_id, content in DOCS.items():
            # 简单的关键词匹配
            score = len(set(query) & set(content)) / max(len(set(query)), 1)
            results.append((doc_id, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    retrieval_test = RetrievalQualityTest()

    # 定义测试：query → ground truth doc_ids
    test_queries = [
        ("RAG 技术是什么？", ["d1"]),
        ("向量数据库的作用？", ["d2"]),
        ("什么模型可以做对话？", ["d3"]),
        ("嵌入模型有哪些？", ["d4"]),
        ("Python 用来做什么？", ["d5"]),
    ]

    queries = [t[0] for t in test_queries]
    ground_truth = [t[1] for t in test_queries]

    metrics = retrieval_test.test_batch(queries, ground_truth, mock_retrieval, top_k=3)

    print(f"  Hit Rate@3: {metrics['hit_rate@k']:.2%}")
    print(f"  MRR:        {metrics['mrr']:.3f}")
    print(f"  Precision@3:{metrics['precision@k']:.2%}")
    print(f"  Recall@3:   {metrics['recall@k']:.2%}")

    # 断言
    try:
        retrieval_test.assert_hit_rate_above(0.80)
        print(f"  ✅ 断言通过: Hit Rate >= 0.80")
    except AssertionError as e:
        print(f"  ❌ 断言失败: {e}")

    # ---- 2. 生成质量测试 ----
    print("\n【2】生成质量测试")

    gen_test = GenerationQualityTest()

    # 好答案示例
    good_answer = ("RAG 技术结合了信息检索和文本生成。"
                   "它可以从知识库中检索相关文档，然后由大语言模型生成回答。"
                   "嵌入模型将文本转换为向量，存储在向量数据库中进行相似度搜索。")

    sources = [
        DOCS["d1"], DOCS["d2"], DOCS["d4"]
    ]

    result = gen_test.test_faithfulness(good_answer, sources)
    print(f"  好答案忠实度: {result['faithfulness']:.2%}")
    print(f"    忠实句数: {result['faithful_sentences']}/{result['total_sentences']}")

    # 坏答案示例（含有幻觉）
    bad_answer = ("RAG 技术是由 Google 在 2020 年发明的。"  # 幻觉
                  "它结合了信息检索和文本生成。"
                  "RAG 不需要任何数据库支持。")  # 幻觉

    result = gen_test.test_faithfulness(bad_answer, sources)
    print(f"  坏答案忠实度: {result['faithfulness']:.2%}")
    print(f"    忠实句数: {result['faithful_sentences']}/{result['total_sentences']}")

    # ---- 3. E2E 测试 ----
    print("\n【3】E2E 管道测试")

    def mock_pipeline(query: str) -> dict:
        """模拟完整 RAG 管道。"""
        return {
            "answer": f"根据知识库回答您关于'{query}'的问题。"
                      f"RAG结合了检索和生成技术。"
                      f"来源：《RAG技术白皮书》《向量数据库指南》",
            "sources": ["d1", "d2"],
            "confidence": 0.92,
        }

    e2e_test = E2EPipelineTest(pipeline_fn=mock_pipeline)
    e2e_cases = [
        {
            "query": "RAG 是什么？",
            "expected_answer_contains": ["RAG", "检索", "生成"],
            "expected_sources_min": 1,
        },
        {
            "query": "如何部署 RAG？",
            "expected_answer_contains": ["知识库"],  # 至少提到知识库
            "expected_sources_min": 1,
            "blocked_keywords": ["我不知道", "无法回答"],
        },
    ]

    report = e2e_test.test_e2e(e2e_cases)
    print(f"  总用例: {report['total']}")
    print(f"  通过: {report['passed']}")
    print(f"  失败: {report['failed']}")
    print(f"  通过率: {report['pass_rate']:.0%}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    demo_rag_pipeline_tests()

    print("\n[完成] RAG 管道测试演示结束。")
    print("  [提示] 生产环境建议:")
    print("    1. 检索测试用标准数据集（如 BEIR、MS MARCO）")
    print("    2. 忠实度检测用 LLM 做 NLI（而非关键词匹配）")
    print("    3. E2E 测试应该包含边界条件和异常场景")
    print("    4. 各项指标应该设定 CI/CD 门禁值")
