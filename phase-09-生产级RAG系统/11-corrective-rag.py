#!/usr/bin/env python3
"""
Corrective RAG (CRAG) 实现
Corrective Retrieval-Augmented Generation

检索置信度评分 (3级):
  - correct (正确): 检索结果直接可用于生成
  - ambiguous (模糊): 检索结果部分相关，需要知识优化
  - incorrect (错误): 检索结果不正确，触发Web搜索回退

工作流程:
  1. 检索 → 置信度评估
  2. correct → 直接生成
  3. ambiguous → 知识细化 (Knowledge Refinement)
  4. incorrect → Web搜索回退
  5. 最终生成（使用优化后的上下文）

对比: CRAG vs 标准RAG 准确性分析
"""

import time
import random
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class RetrievalConfidence(Enum):
    """检索置信度"""
    CORRECT = "correct"             # 正确 - 直接可用
    AMBIGUOUS = "ambiguous"         # 模糊 - 需要优化
    INCORRECT = "incorrect"         # 错误 - 需要回退


class KnowledgeAction(Enum):
    """知识动作"""
    PRESERVE = "preserve"           # 保留原知识
    REFINE = "refine"               # 细化知识
    REWRITE = "rewrite"             # 重写知识
    REPLACE = "replace"             # 替换知识（Web回退）


@dataclass
class ConfidenceAssessment:
    """置信度评估结果"""
    confidence: RetrievalConfidence
    score: float                    # 0.0 - 1.0
    reasoning: str
    good_passages: list[dict]       # 高质量段落
    poor_passages: list[dict]       # 低质量段落
    needs_web_search: bool


@dataclass
class KnowledgeRefinement:
    """知识优化结果"""
    action: KnowledgeAction
    refined_passages: list[dict]
    removed_passages: list[str]     # 移除的段落ID
    new_passages_from_web: list[dict]
    refinement_reason: str


@dataclass
class CRAGResult:
    """CRAG结果"""
    query: str
    answer: str
    confidence: RetrievalConfidence
    initial_passages: int
    final_passages: int
    refinement: Optional[KnowledgeRefinement]
    web_search_used: bool
    time_ms: float
    trace: list[dict] = field(default_factory=list)


# ============================================================
# 置信度评估器 (Confidence Evaluator)
# ============================================================

class ConfidenceEvaluator:
    """检索结果置信度评估器"""

    def __init__(self, llm=None):
        """
        Args:
            llm: LLM接口（用于评估）
        """
        self.llm = llm

    def evaluate(self, query: str, passages: list[dict]) -> ConfidenceAssessment:
        """评估检索结果的置信度

        Args:
            query: 用户查询
            passages: 检索到的文档段落

        Returns:
            ConfidenceAssessment: 置信度评估
        """
        if not passages:
            return ConfidenceAssessment(
                confidence=RetrievalConfidence.INCORRECT,
                score=0.0,
                reasoning="无检索结果",
                good_passages=[],
                poor_passages=[],
                needs_web_search=True,
            )

        # 对每个段落进行评分
        scored_passages = []
        for p in passages:
            score = self._score_passage(query, p)
            scored_passages.append((score, p))

        scored_passages.sort(key=lambda x: x[0], reverse=True)

        # 分类
        good = []
        poor = []
        for score, passage in scored_passages:
            if score > 0.7:
                good.append({"score": score, **passage})
            elif score > 0.4:
                poor.append({"score": score, **passage})

        avg_score = sum(s for s, _ in scored_passages) / len(scored_passages) if scored_passages else 0

        # 决策
        if avg_score > 0.7:
            confidence = RetrievalConfidence.CORRECT
            needs_web = False
            reasoning = f"检索结果置信度高 ({avg_score:.2f})，可直接用于生成"
        elif avg_score > 0.4:
            confidence = RetrievalConfidence.AMBIGUOUS
            needs_web = False
            reasoning = f"检索结果置信度中等 ({avg_score:.2f})，需要知识优化"
        else:
            confidence = RetrievalConfidence.INCORRECT
            needs_web = True
            reasoning = f"检索结果置信度低 ({avg_score:.2f})，需要Web搜索回退"

        return ConfidenceAssessment(
            confidence=confidence,
            score=avg_score,
            reasoning=reasoning,
            good_passages=good,
            poor_passages=poor,
            needs_web_search=needs_web,
        )

    def _score_passage(self, query: str, passage: dict) -> float:
        """对单个段落评分"""
        text = passage.get("text", "")
        if not text:
            return 0.0

        # 关键词重叠度
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        keyword_overlap = len(query_words & text_words) / max(len(query_words), 1)

        # 长度合适度（太长或太短扣分）
        text_len = len(text)
        if 100 <= text_len <= 2000:
            length_score = 1.0
        elif text_len < 50:
            length_score = 0.3
        else:
            length_score = max(0.5, 1.0 - (text_len - 2000) / 5000)

        # 语义相关性（如果有LLM则使用LLM评估）
        semantic_score = 0.7  # 默认
        if self.llm:
            semantic_score = self._llm_relevance(query, text)

        score = 0.3 * keyword_overlap + 0.2 * length_score + 0.5 * semantic_score
        return min(1.0, max(0.0, score))

    def _llm_relevance(self, query: str, text: str) -> float:
        """使用LLM评估相关性"""
        try:
            prompt = f"""评分任务: 请评估以下文档段落与查询的相关性，输出0-100的分数。

查询: {query}
段落: {text[:500]}

分数（仅数字）:"""
            result = self.llm.generate(prompt)
            # 尝试提取数字
            import re
            numbers = re.findall(r"\d+", result)
            if numbers:
                return float(numbers[0]) / 100.0
        except Exception:
            pass
        return 0.7


# ============================================================
# 知识优化器 (Knowledge Refinement)
# ============================================================

class KnowledgeRefiner:
    """知识细化器 - 优化模糊的检索结果"""

    def __init__(self, llm=None):
        self.llm = llm

    def refine(self, query: str, good_passages: list[dict],
               poor_passages: list[dict]) -> KnowledgeRefinement:
        """优化知识

        Returns:
            KnowledgeRefinement: 优化结果
        """
        if not poor_passages and good_passages:
            # 没有需要优化的
            return KnowledgeRefinement(
                action=KnowledgeAction.PRESERVE,
                refined_passages=good_passages,
                removed_passages=[],
                new_passages_from_web=[],
                refinement_reason="已有高质量段落，无需优化",
            )

        # 尝试通过LLM从poor_passages中提取有价值部分
        refined = list(good_passages)
        removed_ids = []

        for pp in poor_passages:
            text = pp.get("text", "")
            pid = pp.get("id", "")

            # 如果段落太短或与查询关键词无重叠，移除
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)

            if overlap < 2 and len(text) < 100:
                removed_ids.append(pid)
            else:
                # 尝试提取相关内容
                if self.llm:
                    extracted = self._extract_relevant(query, text)
                    if extracted:
                        refined.append({
                            "id": f"{pid}_refined",
                            "text": extracted,
                            "source": "refined",
                        })
                    else:
                        removed_ids.append(pid)
                else:
                    # 保留但标记
                    refined.append({
                        "id": f"{pid}_kept",
                        "text": text,
                        "source": "ambiguous",
                    })

        return KnowledgeRefinement(
            action=KnowledgeAction.REFINE if removed_ids else KnowledgeAction.PRESERVE,
            refined_passages=refined,
            removed_passages=removed_ids,
            new_passages_from_web=[],
            refinement_reason=f"移除 {len(removed_ids)} 个低质量段落，保留 {len(refined)} 个",
        )

    def _extract_relevant(self, query: str, text: str) -> Optional[str]:
        """从段落中提取与查询相关的部分"""
        try:
            prompt = f"""请从以下段落中提取与查询最相关的1-2句话。

查询: {query}
段落: {text[:1000]}

如果段落与查询完全无关，请输出 NO_RELEVANT_CONTENT。

提取的内容:"""
            result = self.llm.generate(prompt)
            if "NO_RELEVANT_CONTENT" in result:
                return None
            return result[:500]
        except Exception:
            # LLM不可用时返回原文本片段
            return text[:300]


# ============================================================
# Web搜索回退 (Web Search Fallback)
# ============================================================

class WebSearchFallback:
    """Web搜索回退 - 当内部检索不足时"""

    def __init__(self):
        self._mock_web_index: list[dict] = self._build_mock_web_kb()

    def search(self, query: str, k: int = 5) -> list[dict]:
        """执行Web搜索

        生产环境集成:
        - Google Custom Search API
        - Bing Search API
        - SerpAPI
        - Tavily
        """
        print(f"[WebSearch] 搜索: {query[:60]}...")

        # 模拟Web搜索
        results = []
        for doc in self._mock_web_index:
            query_words = set(query.lower().split())
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words & doc_words)
            if overlap >= 2:
                results.append({
                    "id": f"web_{doc['id']}",
                    "title": doc.get("title", ""),
                    "text": doc["text"],
                    "source": "web_search",
                    "url": doc.get("url", ""),
                    "relevance_score": overlap / max(len(query_words), 1),
                })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        web_results = results[:k]

        print(f"[WebSearch] 找到 {len(web_results)} 条结果")
        return web_results

    def _build_mock_web_kb(self) -> list[dict]:
        """构建模拟Web知识库"""
        return [
            {
                "id": "web_001",
                "title": "Understanding RAG Systems",
                "url": "https://example.com/rag-guide",
                "text": "RAG (Retrieval-Augmented Generation) enhances LLMs by retrieving relevant documents from a knowledge base. Key components include a vector database, embedding model, and a retriever-generator pipeline. Modern RAG systems support multi-modal retrieval and hybrid search.",
            },
            {
                "id": "web_002",
                "title": "Circuit Breaker Pattern in Distributed Systems",
                "url": "https://example.com/circuit-breaker",
                "text": "The Circuit Breaker pattern prevents cascading failures in distributed systems. It has three states: CLOSED (normal operation), OPEN (failures exceed threshold), and HALF_OPEN (testing recovery). This pattern is essential for building resilient microservices.",
            },
            {
                "id": "web_003",
                "title": "Vector Database Comparison 2024",
                "url": "https://example.com/vector-db-comparison",
                "text": "Popular vector databases include Pinecone (fully managed), Weaviate (open-source), Milvus (distributed), Qdrant (Rust-based), and ChromaDB (lightweight). HNSW is the most common indexing algorithm. Key metrics: QPS, recall@k, and indexing speed.",
            },
            {
                "id": "web_004",
                "title": "Corrective RAG: Improving Retrieval Quality",
                "url": "https://example.com/crag-paper",
                "text": "Corrective RAG (CRAG) improves retrieval quality by evaluating confidence scores. When retrieval confidence is low, CRAG triggers web search fallback. When confidence is medium, it applies knowledge refinement. This approach significantly improves factual accuracy.",
            },
            {
                "id": "web_005",
                "title": "Token Bucket Rate Limiting Algorithm",
                "url": "https://example.com/rate-limiting",
                "text": "The Token Bucket algorithm allows controlled request rates with burst capability. Tokens are added at a fixed rate; each request consumes tokens. When the bucket is empty, requests are rejected or queued. This provides smoother rate limiting than fixed windows.",
            },
        ]


# ============================================================
# CRAG 核心处理器
# ============================================================

class CRAGProcessor:
    """Corrective RAG 核心处理器"""

    def __init__(self, llm=None, retriever=None, evaluator=None,
                 refiner=None, web_searcher=None):
        self.llm = llm
        self.retriever = retriever
        self.evaluator = evaluator or ConfidenceEvaluator(llm=llm)
        self.refiner = refiner or KnowledgeRefiner(llm=llm)
        self.web_searcher = web_searcher or WebSearchFallback()

        # 统计
        self.stats = {
            "total_queries": 0,
            "correct": 0,
            "ambiguous": 0,
            "incorrect": 0,
            "web_searches": 0,
            "refinements": 0,
        }

    def process(self, query: str, initial_k: int = 5) -> CRAGResult:
        """处理查询 - 完整CRAG流程"""
        t_start = time.time()
        trace = []
        self.stats["total_queries"] += 1

        # 步骤1: 初始检索
        trace.append({"step": 1, "action": "初始检索"})
        initial_passages = self.retriever.search(query, k=initial_k) if self.retriever else []
        trace.append({"step": 1, "passages": len(initial_passages)})

        # 步骤2: 置信度评估
        trace.append({"step": 2, "action": "置信度评估"})
        assessment = self.evaluator.evaluate(query, initial_passages)
        trace.append({"step": 2, "confidence": assessment.confidence.value, "score": assessment.score})

        refinement = None
        final_passages = initial_passages
        web_search_used = False

        if assessment.confidence == RetrievalConfidence.CORRECT:
            # 正确 → 直接使用检索结果
            self.stats["correct"] += 1
            trace.append({"step": 3, "action": "直接生成 (置信度高)"})
            final_passages = assessment.good_passages

        elif assessment.confidence == RetrievalConfidence.AMBIGUOUS:
            # 模糊 → 知识细化
            self.stats["ambiguous"] += 1
            self.stats["refinements"] += 1
            trace.append({"step": 3, "action": "知识细化"})

            refinement = self.refiner.refine(
                query,
                assessment.good_passages,
                assessment.poor_passages,
            )
            final_passages = refinement.refined_passages
            trace.append({"step": 3, "refined": len(final_passages), "removed": len(refinement.removed_passages)})

        elif assessment.confidence == RetrievalConfidence.INCORRECT:
            # 错误 → Web搜索回退
            self.stats["incorrect"] += 1
            self.stats["web_searches"] += 1
            trace.append({"step": 3, "action": "Web搜索回退"})

            web_results = self.web_searcher.search(query, k=5)
            web_search_used = True

            # 合并: 保留好段落 + Web结果
            final_passages = assessment.good_passages + web_results
            refinement = KnowledgeRefinement(
                action=KnowledgeAction.REPLACE,
                refined_passages=final_passages,
                removed_passages=[p["id"] for p in assessment.poor_passages],
                new_passages_from_web=web_results,
                refinement_reason=f"检索置信度低 ({assessment.score:.2f})，使用 {len(web_results)} 条Web结果替代",
            )
            trace.append({"step": 3, "web_results": len(web_results)})

        # 步骤4: 最终生成
        trace.append({"step": 4, "action": "最终生成"})
        answer = self._generate_with_context(query, final_passages)
        trace.append({"step": 4, "answer_length": len(answer)})

        elapsed = (time.time() - t_start) * 1000

        return CRAGResult(
            query=query,
            answer=answer,
            confidence=assessment.confidence,
            initial_passages=len(initial_passages),
            final_passages=len(final_passages),
            refinement=refinement,
            web_search_used=web_search_used,
            time_ms=elapsed,
            trace=trace,
        )

    def _generate_with_context(self, query: str, passages: list[dict]) -> str:
        """基于上下文生成回答"""
        if not passages:
            return "抱歉，未找到与您查询相关的信息。请尝试重新表述您的问题。"

        context = "\n\n".join(
            f"[来源: {p.get('id', 'unknown')}] {p.get('text', '')[:500]}"
            for p in passages[:5]
        )

        if self.llm:
            prompt = f"""基于以下参考信息回答用户问题。请确保回答准确且有依据。

用户问题: {query}

参考信息:
{context[:3000]}

请提供清晰、准确的中文回答："""
            return self.llm.generate(prompt)

        # 无LLM时返回上下文摘要
        return f"基于 {len(passages)} 个参考来源的回答:\n\n{passages[0].get('text', '')[:300]}..."

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = max(self.stats["total_queries"], 1)
        return {
            **self.stats,
            "correct_rate": f"{self.stats['correct'] / total:.1%}",
            "ambiguous_rate": f"{self.stats['ambiguous'] / total:.1%}",
            "incorrect_rate": f"{self.stats['incorrect'] / total:.1%}",
            "web_search_rate": f"{self.stats['web_searches'] / total:.1%}",
            "refinement_rate": f"{self.stats['refinements'] / total:.1%}",
        }


# ============================================================
# CRAG vs Standard RAG 对比评估
# ============================================================

class CRAGComparator:
    """CRAG vs 标准RAG 对比"""

    def __init__(self):
        self.crag_processor: Optional[CRAGProcessor] = None
        self.test_queries: list[dict] = self._load_test_queries()

    def set_crag_processor(self, processor: CRAGProcessor):
        self.crag_processor = processor

    def run_comparison(self) -> dict:
        """运行对比实验"""
        results = {
            "standard_rag": {"total_score": 0, "queries": []},
            "crag": {"total_score": 0, "queries": []},
        }

        for test in self.test_queries:
            query = test["query"]
            expected_keywords = test.get("expected_keywords", [])

            # 标准RAG
            std_passages = self.crag_processor.retriever.search(query, k=3)
            std_context = " ".join(p["text"] for p in std_passages)
            std_score = self._compute_accuracy(std_context, expected_keywords)
            results["standard_rag"]["total_score"] += std_score
            results["standard_rag"]["queries"].append({
                "query": query,
                "score": std_score,
                "passages": len(std_passages),
            })

            # CRAG
            crag_result = self.crag_processor.process(query)
            crag_context = crag_result.answer
            crag_score = self._compute_accuracy(crag_context, expected_keywords)
            results["crag"]["total_score"] += crag_score
            results["crag"]["queries"].append({
                "query": query,
                "score": crag_score,
                "confidence": crag_result.confidence.value,
                "web_search": crag_result.web_search_used,
            })

        n = len(self.test_queries)
        results["standard_rag"]["avg_score"] = results["standard_rag"]["total_score"] / n
        results["crag"]["avg_score"] = results["crag"]["total_score"] / n
        results["improvement"] = (
            results["crag"]["avg_score"] - results["standard_rag"]["avg_score"]
        ) / max(results["standard_rag"]["avg_score"], 0.01) * 100

        return results

    @staticmethod
    def _compute_accuracy(text: str, expected_keywords: list[str]) -> float:
        """基于期望关键词计算准确性"""
        if not expected_keywords:
            return 0.7  # 默认

        text_lower = text.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
        return hits / len(expected_keywords)

    @staticmethod
    def _load_test_queries() -> list[dict]:
        """加载测试查询集"""
        return [
            {
                "query": "什么是RAG系统？",
                "expected_keywords": ["检索", "生成", "知识库", "文档", "增强"],
            },
            {
                "query": "熔断器模式是如何工作的？",
                "expected_keywords": ["熔断器", "CLOSED", "OPEN", "HALF_OPEN", "故障"],
            },
            {
                "query": "向量数据库有哪些选择？",
                "expected_keywords": ["Faiss", "Milvus", "Pinecone", "HNSW", "索引"],
            },
            {
                "query": "什么是令牌桶算法？",
                "expected_keywords": ["令牌", "速率", "限制", "突发", "桶"],
            },
            {
                "query": "CRAG相比标准RAG有什么优势？",
                "expected_keywords": ["置信度", "回退", "Web搜索", "优化", "准确"],
            },
        ]


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Corrective RAG (CRAG) - 演示运行")
    print("=" * 60)

    # 1. 初始化检索器（从Self-RAG模块复用）
    class DemoRetriever:
        def __init__(self):
            self.kb = [
                {"id": "d1", "text": "RAG是检索增强生成技术，通过检索外部知识来增强LLM回答。"},
                {"id": "d2", "text": "熔断器模式用于防止级联故障，有CLOSED/OPEN/HALF_OPEN三种状态。"},
                {"id": "d3", "text": "向量数据库如Faiss和Milvus用于高效相似度搜索，HNSW是常用索引。"},
                {"id": "d4", "text": "令牌桶算法用于速率限制，支持突发流量，令牌按时生成。"},
                {"id": "d5", "text": "CRAG通过置信度评估改进检索质量，低置信度触发Web搜索回退。"},
                {"id": "d6", "text": "今天天气不错，适合出去郊游。"},  # 不相关内容
                {"id": "d7", "text": "Python是最流行的编程语言之一，广泛用于数据科学和AI。"},
            ]

        def search(self, query: str, k: int = 5):
            results = []
            query_words = set(query.lower().split())
            for doc in self.kb:
                text_words = set(doc["text"].lower().split())
                overlap = len(query_words & text_words)
                if overlap > 0:
                    results.append({
                        "id": doc["id"],
                        "text": doc["text"],
                        "score": overlap / max(len(query_words), 1),
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]

    # 2. 初始化各个组件
    class SimpleLLM:
        @staticmethod
        def generate(prompt):
            # 模拟LLM响应
            if "评分" in prompt or "分数" in prompt:
                return "85"
            if "相关" in prompt or "relevant" in prompt:
                return "这段内容与查询相关。"
            if "提取" in prompt:
                return "这是提取的相关内容。"
            # 生成回答
            if "RAG" in prompt:
                return "RAG（检索增强生成）是一种结合信息检索和文本生成的AI技术，其核心思想是从外部知识库检索相关文档来增强大语言模型的回答质量。"
            if "熔断器" in prompt:
                return "熔断器模式通过CLOSED/OPEN/HALF_OPEN三态模型防止分布式系统中的级联故障。当失败次数达到阈值时，熔断器打开并拒绝请求；冷却后进入半开状态进行探测；探测成功后恢复关闭状态。"
            if "向量" in prompt:
                return "主流向量数据库包括Faiss（开源，Meta）、Milvus（分布式，Zilliz）、Pinecone（云服务）、ChromaDB（轻量级），它们都支持HNSW等高效索引算法。"
            return "基于检索到的信息，这是生成的回答。"

    llm = SimpleLLM()
    retriever = DemoRetriever()
    evaluator = ConfidenceEvaluator(llm=llm)
    refiner = KnowledgeRefiner(llm=llm)
    web_searcher = WebSearchFallback()

    # 3. 创建CRAG处理器
    crag = CRAGProcessor(
        llm=llm,
        retriever=retriever,
        evaluator=evaluator,
        refiner=refiner,
        web_searcher=web_searcher,
    )

    # 4. 测试不同置信度场景
    queries = [
        ("查询1 (预期CORRECT)", "什么是RAG系统？"),
        ("查询2 (预期AMBIGUOUS)", "今天有什么推荐？"),
        ("查询3 (预期INCORRECT)", "量子计算在RAG中的应用？"),
    ]

    for label, query in queries:
        print(f"\n{label}: {query}")
        result = crag.process(query)

        print(f"  置信度: {result.confidence.value} (分数: {crag.evaluator.evaluate(query, retriever.search(query)).score:.2f})")
        print(f"  初始段落: {result.initial_passages}")
        print(f"  最终段落: {result.final_passages}")
        print(f"  Web搜索: {'是' if result.web_search_used else '否'}")
        print(f"  耗时: {result.time_ms:.1f}ms")
        print(f"  回答: {result.answer[:120]}...")

    # 5. 统计报告
    print("\n--- CRAG统计报告 ---")
    stats = crag.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 6. CRAG vs 标准RAG对比
    print("\n--- CRAG vs 标准RAG 对比 ---")
    comparator = CRAGComparator()
    comparator.set_crag_processor(crag)
    comparison = comparator.run_comparison()

    print(f"标准RAG 平均准确率: {comparison['standard_rag']['avg_score']:.2%}")
    print(f"CRAG 平均准确率:     {comparison['crag']['avg_score']:.2%}")
    print(f"CRAG 提升:           {comparison['improvement']:+.1f}%")

    print("\n标准RAG 各查询得分:")
    for q in comparison["standard_rag"]["queries"]:
        print(f"  {q['query'][:40]}... → {q['score']:.2f}")

    print("\nCRAG 各查询得分:")
    for q in comparison["crag"]["queries"]:
        print(f"  {q['query'][:40]}... → {q['score']:.2f} ({q['confidence']})")

    # 7. Web搜索回退演示
    print("\n--- Web搜索回退演示 ---")
    web_special = crag.process("最新的向量数据库性能对比2024")
    print(f"  查询: {web_special.query}")
    print(f"  置信度: {web_special.confidence.value}")
    print(f"  Web搜索: {'是 ⚡' if web_special.web_search_used else '否'}")
    if web_special.refinement and web_special.refinement.new_passages_from_web:
        for p in web_special.refinement.new_passages_from_web:
            print(f"  Web结果: [{p['id']}] {p['title']}")

    print("\n" + "=" * 60)
    print("Corrective RAG (CRAG) 演示完成！")
    print("=" * 60)
