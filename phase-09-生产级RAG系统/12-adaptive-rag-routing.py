#!/usr/bin/env python3
"""
Adaptive RAG 路由 (Adaptive RAG Routing)
Query Complexity Classifier + Strategy Router

查询复杂度分类 (3级):
  - simple (简单): 事实型查询，只需单次检索
  - moderate (中等): 需要检索+重排序
  - complex (复杂): 需要分解+并行检索+融合+重排序

策略路由决策树:
  simple → lightweight_retrieval (快速检索)
  moderate → standard_retrieval + rerank (标准检索+重排)
  complex → decomposition + parallel + fusion + rerank (分解+并行+融合+重排)
"""

import time
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Callable


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class QueryComplexity(Enum):
    """查询复杂度"""
    SIMPLE = "simple"         # 简单事实型查询
    MODERATE = "moderate"     # 中等复杂度查询
    COMPLEX = "complex"       # 复杂多步查询


class RetrievalStrategy(Enum):
    """检索策略"""
    LIGHTWEIGHT = "lightweight"                    # 轻量级: BM25 only
    STANDARD = "standard"                          # 标准: 向量检索
    STANDARD_RERANK = "standard_rerank"            # 标准+重排: 向量检索+重排序
    DECOMPOSED_PARALLEL = "decomposed_parallel"    # 分解并行: 查询分解+并行检索
    HYBRID_FUSION = "hybrid_fusion"                # 混合融合: 向量+BM25+重排


@dataclass
class ComplexityClassification:
    """复杂度分类结果"""
    complexity: QueryComplexity
    confidence: float
    features: dict          # 分类特征
    reasoning: str


@dataclass
class SubQuery:
    """子查询"""
    sub_query: str
    weight: float           # 重要性权重
    dependencies: list[str] # 依赖的其他子查询ID


@dataclass
class StrategyResult:
    """策略执行结果"""
    strategy: RetrievalStrategy
    passages: list[dict]
    execution_time_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class AdaptiveResult:
    """Adaptive RAG结果"""
    query: str
    complexity: QueryComplexity
    strategy: RetrievalStrategy
    answer: str
    passages: list[dict]
    sub_results: list[StrategyResult]  # 复杂查询的子结果
    time_ms: float
    trace: list[dict] = field(default_factory=list)


# ============================================================
# 查询复杂度分类器 (Query Complexity Classifier)
# ============================================================

class ComplexityClassifier:
    """查询复杂度分类器"""

    # 复杂度特征定义
    SIMPLE_PATTERNS = [
        r"^(什么|谁|何时|哪里|哪个|how|what|who|when|where|which)",
        r"^(定义|解释|什么是|define|explain)",
    ]

    MODERATE_PATTERNS = [
        r"(比较|对比|区别|vs|compare|difference|versus)",
        r"(优缺点|优势|劣势|advantage|disadvantage)",
        r"(如何|怎么做|how to|步骤|step)",
    ]

    COMPLEX_PATTERNS = [
        r"(分析|评估|总结|综合|analyze|evaluate|synthesize|summarize)",
        r"(和|与|以及|并且|同时|and|also|furthermore|moreover)",
        r"(从.*角度|在.*方面|基于.*的|from.*perspective|in terms of)",
        r"(多个|多种|多项|several|multiple|various)",
        r"\?{2,}",  # 多个问号
    ]

    # 复杂度权重
    FEATURE_WEIGHTS = {
        "query_length": 0.10,
        "concept_count": 0.20,
        "has_comparison": 0.15,
        "has_causality": 0.15,
        "has_multi_step": 0.20,
        "has_synthesis": 0.20,
    }

    def classify(self, query: str) -> ComplexityClassification:
        """分类查询复杂度

        Args:
            query: 用户查询

        Returns:
            ComplexityClassification: 分类结果
        """
        # 提取特征
        features = self._extract_features(query)

        # 基于规则的评分
        simple_score = self._match_score(query, self.SIMPLE_PATTERNS)
        moderate_score = self._match_score(query, self.MODERATE_PATTERNS)
        complex_score = self._match_score(query, self.COMPLEX_PATTERNS)

        # 加权特征评分
        feature_score = (
            features["query_length"] * self.FEATURE_WEIGHTS["query_length"] +
            features["concept_count"] * self.FEATURE_WEIGHTS["concept_count"] +
            features["has_comparison"] * self.FEATURE_WEIGHTS["has_comparison"] +
            features["has_causality"] * self.FEATURE_WEIGHTS["has_causality"] +
            features["has_multi_step"] * self.FEATURE_WEIGHTS["has_multi_step"] +
            features["has_synthesis"] * self.FEATURE_WEIGHTS["has_synthesis"]
        )

        # 综合决策
        total_simple = simple_score * 0.4 + (1.0 - feature_score) * 0.6
        total_moderate = moderate_score * 0.4 + feature_score * 0.6
        total_complex = complex_score * 0.4 + feature_score * 0.6

        # 选择最高分
        scores = {
            QueryComplexity.SIMPLE: total_simple,
            QueryComplexity.MODERATE: total_moderate,
            QueryComplexity.COMPLEX: total_complex,
        }

        best_complexity = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_complexity])

        # 生成推理说明
        reasoning_parts = []
        if features["has_multi_step"] > 0.5:
            reasoning_parts.append("检测到多步骤需求")
        if features["has_synthesis"] > 0.5:
            reasoning_parts.append("检测到综合分析需求")
        if features["has_comparison"] > 0.5:
            reasoning_parts.append("检测到对比需求")
        if features["concept_count"] > 3:
            reasoning_parts.append(f"包含{features['concept_count']}个概念")
        if not reasoning_parts:
            reasoning_parts.append("简单事实型查询")

        return ComplexityClassification(
            complexity=best_complexity,
            confidence=confidence,
            features=features,
            reasoning="; ".join(reasoning_parts),
        )

    def _extract_features(self, query: str) -> dict:
        """提取查询特征"""
        query_lower = query.lower()

        # 查询长度
        query_len = len(query.split())
        length_score = min(1.0, query_len / 30)  # 30词以上=1.0

        # 概念数量（基于关键词）
        # 移除常见停用词
        stop_words = {"的", "是", "在", "和", "与", "了", "the", "is", "in", "and", "of", "to", "a"}
        words = [w for w in re.findall(r"\w+", query_lower) if w not in stop_words]
        concept_count = len(set(words))
        concept_score = min(1.0, concept_count / 8)  # 8个以上概念=1.0

        # 对比特征
        comparison_keywords = {"vs", "比较", "对比", "区别", "不同", "compare", "difference", "versus"}
        has_comparison = 1.0 if any(kw in query_lower for kw in comparison_keywords) else 0.0

        # 因果特征
        causality_keywords = {"因为", "所以", "导致", "影响", "原因", "结果", "because", "therefore", "cause", "effect", "impact"}
        has_causality = 1.0 if any(kw in query_lower for kw in causality_keywords) else 0.0

        # 多步骤特征
        multi_step_keywords = {"首先", "然后", "最后", "步骤", "流程", "first", "then", "finally", "step", "process", "procedure"}
        has_multi_step = 1.0 if any(kw in query_lower for kw in multi_step_keywords) else 0.0
        # 多个问号
        if query.count("?") > 1 or query.count("？") > 1:
            has_multi_step = 1.0

        # 综合分析特征
        synthesis_keywords = {"分析", "评估", "综合", "总结", "概述", "回顾", "analyze", "evaluate", "synthesize", "summarize", "review", "overview"}
        has_synthesis = 1.0 if any(kw in query_lower for kw in synthesis_keywords) else 0.0

        return {
            "query_length": length_score,
            "concept_count": concept_score,
            "has_comparison": has_comparison,
            "has_causality": has_causality,
            "has_multi_step": has_multi_step,
            "has_synthesis": has_synthesis,
        }

    @staticmethod
    def _match_score(query: str, patterns: list[str]) -> float:
        """计算模式匹配得分"""
        matches = sum(1 for p in patterns if re.search(p, query, re.IGNORECASE))
        return min(1.0, matches / max(len(patterns), 1))


# ============================================================
# 查询分解器 (Query Decomposer - for Complex Queries)
# ============================================================

class QueryDecomposer:
    """查询分解器 - 将复杂查询拆分为子查询"""

    def __init__(self, llm=None):
        self.llm = llm

    def decompose(self, query: str) -> list[SubQuery]:
        """将复杂查询分解为子查询

        Args:
            query: 原始复杂查询

        Returns:
            SubQuery列表
        """
        if self.llm:
            return self._llm_decompose(query)
        return self._rule_decompose(query)

    def _llm_decompose(self, query: str) -> list[SubQuery]:
        """使用LLM分解查询"""
        prompt = f"""请将以下复杂查询分解为2-4个独立的子查询，每个子查询应该可以独立检索。

复杂查询: {query}

请按以下JSON格式输出:
[
  {{"sub_query": "子查询1", "weight": 0.5}},
  {{"sub_query": "子查询2", "weight": 0.5}}
]

子查询列表:"""
        # 模拟LLM输出
        return self._rule_decompose(query)

    @staticmethod
    def _rule_decompose(query: str) -> list[SubQuery]:
        """基于规则分解查询"""

        # 按连接词分割
        connectors = ["和", "与", "以及", "同时", "还有", "and", "also", "furthermore"]
        parts = [query]

        for conn in connectors:
            new_parts = []
            for part in parts:
                if conn in part:
                    splits = part.split(conn)
                    new_parts.extend(s.strip() for s in splits if s.strip())
                else:
                    new_parts.append(part)
            parts = new_parts

        # 如果只有一个部分，拆成概念词
        if len(parts) == 1:
            # 提取关键短语
            key_phrases = re.findall(r"[一-鿿\w]+", query)
            mid = len(key_phrases) // 2
            if mid > 2:
                parts = [
                    " ".join(key_phrases[:mid]),
                    " ".join(key_phrases[mid:]),
                ]

        # 如果还是太长或太短
        if len(parts) > 4:
            parts = parts[:4]  # 最多4个子查询
        elif len(parts) < 2:
            parts = [query]

        # 构建SubQuery
        total_parts = len(parts)
        return [
            SubQuery(
                sub_query=part.strip(),
                weight=1.0 / total_parts,
                dependencies=[],
            )
            for i, part in enumerate(parts)
        ]


# ============================================================
# 检索器 (Retriever for Various Strategies)
# ============================================================

class AdaptiveRetriever:
    """自适应检索器 - 支持多种检索策略"""

    def __init__(self, kb: Optional[list[dict]] = None):
        self.kb = kb or self._build_demo_kb()

    def lightweight_search(self, query: str, k: int = 5) -> list[dict]:
        """轻量级BM25搜索"""
        results = []
        query_words = set(query.lower().split())
        for doc in self.kb:
            text_words = set(doc["text"].lower().split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                bm25_score = overlap / max(len(query_words), 1) * 0.8
                results.append({**doc, "score": bm25_score, "method": "bm25"})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def standard_search(self, query: str, k: int = 10) -> list[dict]:
        """标准向量检索"""
        import numpy as np
        results = []
        query_words = set(query.lower().split())
        for doc in self.kb:
            text_words = set(doc["text"].lower().split())
            overlap = len(query_words & text_words)
            semantic_score = overlap / max(len(query_words), 1) * 0.9
            if semantic_score > 0.1:
                results.append({**doc, "score": semantic_score, "method": "vector"})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def rerank(self, query: str, passages: list[dict], top_k: int = 5) -> list[dict]:
        """重排序"""
        # 模拟Cross-encoder重排序
        ranked = []
        for p in passages:
            # 基于更细粒度的相关性评分
            query_words = set(query.lower().split())
            text = p["text"].lower()
            # 完整短语匹配加分
            phrase_bonus = 0.3 if query.lower() in text else 0
            # 精确词匹配
            exact_matches = sum(1 for w in query_words if w in text)
            rerank_score = p.get("score", 0.5) * 0.7 + exact_matches * 0.1 + phrase_bonus
            ranked.append({**p, "score": rerank_score, "reranked": True})

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def parallel_search(self, sub_queries: list[SubQuery], k_per_query: int = 5) -> list[StrategyResult]:
        """并行检索多个子查询"""
        results = []
        for sq in sub_queries:
            t0 = time.time()
            passages = self.standard_search(sq.sub_query, k=k_per_query)
            elapsed = (time.time() - t0) * 1000
            results.append(StrategyResult(
                strategy=RetrievalStrategy.DECOMPOSED_PARALLEL,
                passages=passages,
                execution_time_ms=elapsed,
                metadata={"sub_query": sq.sub_query, "weight": sq.weight},
            ))
        return results

    def hybrid_search(self, query: str, k: int = 10) -> list[dict]:
        """混合搜索: 向量 + BM25"""
        vector_results = self.standard_search(query, k)
        bm25_results = self.lightweight_search(query, k)

        # 融合：RRF (Reciprocal Rank Fusion)
        fused = {}
        for rank, doc in enumerate(vector_results):
            doc_id = doc["id"]
            fused[doc_id] = {**doc, "rrf_score": 1.0 / (rank + 60)}

        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            if doc_id in fused:
                fused[doc_id]["rrf_score"] += 1.0 / (rank + 60)
            else:
                fused[doc_id] = {**doc, "rrf_score": 1.0 / (rank + 60)}

        sorted_results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results[:k]

    def _build_demo_kb(self) -> list[dict]:
        return [
            {"id": "d01", "text": "RAG（检索增强生成）是一种AI技术，通过在生成回答前检索相关文档来提高LLM的准确性和可靠性。核心组件包括文档解析器、向量化引擎、向量数据库和生成器。"},
            {"id": "d02", "text": "向量数据库如ChromaDB和Milvus存储文档的向量表示，支持高效的相似度搜索。HNSW（分层可导航小世界）是最常用的索引算法之一，在召回率和速度之间取得良好平衡。"},
            {"id": "d03", "text": "熔断器模式是微服务架构中的弹性设计模式。它通过CLOSED/OPEN/HALF_OPEN三态模型保护系统免受级联故障影响。"},
            {"id": "d04", "text": "速率限制使用令牌桶算法控制API调用频率。令牌以固定速率生成并存储在桶中，每个请求消耗令牌。当桶为空时，请求被拒绝或排队。"},
            {"id": "d05", "text": "Self-RAG是一种高级RAG架构，通过反思令牌（<retrieve>, <relevant>, <supported>等）实现按需检索和自我评估，提高回答质量。"},
            {"id": "d06", "text": "CRAG（Corrective RAG）通过置信度评估改进检索质量。低置信度触发Web搜索回退，中等置信度触发知识细化。"},
            {"id": "d07", "text": "Adaptive RAG根据查询复杂度动态选择检索策略：简单查询用BM25，中等查询加重排序，复杂查询分解为子查询并行处理。"},
            {"id": "d08", "text": "知识图谱RAG通过实体识别和关系抽取构建图结构，支持基于图遍历的增强检索，可以捕捉文档间的隐含关系。"},
            {"id": "d09", "text": "Prompt缓存可以显著降低LLM调用成本。Anthropic支持显式缓存控制，OpenAI支持自动前缀缓存，DeepSeek支持KV-cache磁盘缓存。"},
            {"id": "d10", "text": "多租户架构需要三层隔离：独立向量集合（第1层）、元数据过滤（第2层）、API Key认证（第3层）。每个租户等级有不同配额。"},
        ]


# ============================================================
# Adaptive RAG 路由器
# ============================================================

class AdaptiveRAGRouter:
    """Adaptive RAG 路由器 - 决策树实现"""

    def __init__(self, llm=None):
        self.classifier = ComplexityClassifier()
        self.decomposer = QueryDecomposer(llm=llm)
        self.retriever = AdaptiveRetriever()
        self.llm = llm

        # 决策树: complexity -> strategy
        self.decision_tree = {
            QueryComplexity.SIMPLE: RetrievalStrategy.LIGHTWEIGHT,
            QueryComplexity.MODERATE: RetrievalStrategy.STANDARD_RERANK,
            QueryComplexity.COMPLEX: RetrievalStrategy.HYBRID_FUSION,
        }

    def route(self, query: str) -> AdaptiveResult:
        """路由查询到合适的策略

        决策树流程:
        ├── 分类查询复杂度
        │   ├── simple → lightweight_retrieval
        │   ├── moderate → standard_retrieval + rerank
        │   └── complex → decomposition + parallel + fusion + rerank
        └── 执行策略 → 生成回答
        """
        t_start = time.time()
        trace = []

        # 步骤1: 分类查询复杂度
        classification = self.classifier.classify(query)
        trace.append({
            "step": "classify",
            "complexity": classification.complexity.value,
            "confidence": classification.confidence,
            "features": classification.features,
        })

        # 步骤2: 决策树路由
        strategy = self.decision_tree[classification.complexity]
        trace.append({"step": "route", "strategy": strategy.value})

        # 步骤3: 执行策略
        strategy_results: list[StrategyResult] = []
        final_passages: list[dict] = []

        if strategy == RetrievalStrategy.LIGHTWEIGHT:
            # 轻量级: BM25 only
            result = self._execute_lightweight(query)
            strategy_results.append(result)
            final_passages = result.passages

        elif strategy == RetrievalStrategy.STANDARD_RERANK:
            # 标准+重排
            result = self._execute_standard_rerank(query)
            strategy_results.append(result)
            final_passages = result.passages

        elif strategy == RetrievalStrategy.HYBRID_FUSION:
            # 混合融合: 分解+并行+融合+重排
            sub_results = self._execute_hybrid_fusion(query)
            strategy_results.extend(sub_results)

            # 融合所有子结果
            all_passages = []
            for sr in sub_results:
                for p in sr.passages:
                    weight = sr.metadata.get("weight", 1.0)
                    all_passages.append({**p, "score": p.get("score", 0) * weight})

            # 去重和排序
            seen_ids = set()
            unique_passages = []
            for p in sorted(all_passages, key=lambda x: x["score"], reverse=True):
                if p["id"] not in seen_ids:
                    seen_ids.add(p["id"])
                    unique_passages.append(p)

            final_passages = self.retriever.rerank(query, unique_passages, top_k=8)

        trace.append({"step": "execute", "results": len(strategy_results), "passages": len(final_passages)})

        # 步骤4: 生成回答
        answer = self._generate_answer(query, final_passages)
        trace.append({"step": "generate", "answer_length": len(answer)})

        elapsed = (time.time() - t_start) * 1000

        return AdaptiveResult(
            query=query,
            complexity=classification.complexity,
            strategy=strategy,
            answer=answer,
            passages=final_passages,
            sub_results=strategy_results,
            time_ms=elapsed,
            trace=trace,
        )

    # ========== 策略执行方法 ==========

    def _execute_lightweight(self, query: str) -> StrategyResult:
        """执行轻量级检索"""
        t0 = time.time()
        passages = self.retriever.lightweight_search(query, k=5)
        elapsed = (time.time() - t0) * 1000

        return StrategyResult(
            strategy=RetrievalStrategy.LIGHTWEIGHT,
            passages=passages,
            execution_time_ms=elapsed,
            metadata={"search_type": "bm25_only"},
        )

    def _execute_standard_rerank(self, query: str) -> StrategyResult:
        """执行标准检索+重排序"""
        t0 = time.time()

        # 先检索更多
        candidates = self.retriever.standard_search(query, k=20)
        # 再重排序
        passages = self.retriever.rerank(query, candidates, top_k=5)

        elapsed = (time.time() - t0) * 1000

        return StrategyResult(
            strategy=RetrievalStrategy.STANDARD_RERANK,
            passages=passages,
            execution_time_ms=elapsed,
            metadata={"search_type": "vector+rerank", "candidates": len(candidates)},
        )

    def _execute_hybrid_fusion(self, query: str) -> list[StrategyResult]:
        """执行混合融合策略"""
        # 步骤1: 分解查询
        sub_queries = self.decomposer.decompose(query)

        # 步骤2: 并行检索
        parallel_results = self.retriever.parallel_search(sub_queries, k_per_query=5)

        # 步骤3: 对所有子结果的段落进行融合
        # 发生在route()方法中

        return parallel_results

    # ========== 回答生成 ==========

    def _generate_answer(self, query: str, passages: list[dict]) -> str:
        """基于检索结果生成回答"""
        if not passages:
            return "抱歉，未找到相关信息。请尝试更具体地描述您的问题。"

        context = "\n\n".join(
            f"[来源 {p.get('id', 'N/A')}] {p.get('text', '')[:400]}"
            for p in passages[:5]
        )

        if self.llm:
            prompt = f"""基于以下参考信息回答用户问题。请确保回答准确、全面。

用户问题: {query}

参考信息:
{context[:3000]}

请提供清晰的中文回答："""
            return self.llm.generate(prompt)

        # 无LLM: 合成上下文
        key_points = []
        for p in passages[:3]:
            key_points.append(p.get("text", "")[:120])
        return "基于检索到的信息：\n\n" + "\n\n".join(f"- {kp}" for kp in key_points)


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Adaptive RAG 路由 - 演示运行")
    print("=" * 60)

    # 1. 初始化路由器
    router = AdaptiveRAGRouter()

    # 2. 定义测试查询
    test_queries = [
        ("简单查询", "什么是RAG？"),
        ("简单查询", "谁发明了熔断器模式？"),
        ("中等查询", "RAG和微调有什么区别？"),
        ("中等查询", "如何优化向量数据库的检索速度？"),
        ("复杂查询", "从性能、成本和可扩展性角度分析Self-RAG、CRAG和Adaptive RAG的优劣"),
        ("复杂查询", "请综合分析向量数据库选型的影响因素，并对比Faiss、Milvus、Pinecone和ChromaDB在RAG系统中的适用场景"),
    ]

    # 3. 运行测试
    for label, query in test_queries:
        print(f"\n{'='*50}")
        print(f"[{label}] {query}")
        print("-" * 40)

        # 分类
        classification = router.classifier.classify(query)
        print(f"复杂度: {classification.complexity.value} (置信度: {classification.confidence:.2f})")
        print(f"推理: {classification.reasoning}")
        print(f"特征: {classification.features}")

        # 路由
        result = router.route(query)
        print(f"策略: {result.strategy.value}")
        print(f"检索段落: {len(result.passages)}")
        print(f"耗时: {result.time_ms:.1f}ms")
        print(f"回答: {result.answer[:150]}...")

    # 4. 决策树可视化
    print("\n\n--- 决策树 ---")
    print("查询复杂度分类 → 策略路由")
    print("├── simple (简单)")
    print("│   └── lightweight_retrieval")
    print("│       策略: BM25 only")
    print("│       目标延迟: <50ms")
    print("│")
    print("├── moderate (中等)")
    print("│   └── standard_rerank")
    print("│       策略: 向量检索(20候选项) → 重排序(top-5)")
    print("│       目标延迟: <200ms")
    print("│")
    print("└── complex (复杂)")
    print("    └── hybrid_fusion")
    print("        策略: 查询分解 → 并行检索 → 结果融合 → 重排序")
    print("        目标延迟: <1000ms")

    # 5. 性能对比
    print("\n--- 各策略性能对比 ---")
    strategies = [
        (RetrievalStrategy.LIGHTWEIGHT, "轻量级", "BM25", "<50ms"),
        (RetrievalStrategy.STANDARD_RERANK, "标准+重排", "Vector+Rerank", "<200ms"),
        (RetrievalStrategy.HYBRID_FUSION, "混合融合", "Decompose+Fusion", "<1000ms"),
    ]

    for strategy, name, methods, target_latency in strategies:
        print(f"  {name:10s} | {methods:20s} | 延迟目标: {target_latency}")

    # 6. 复杂度分类统计
    print("\n--- 复杂度分类测试 ---")
    all_test_queries = [
        "什么是RAG？",
        "RAG的工作原理是什么？",
        "RAG和微调有什么区别和优缺点？",
        "如何在生产环境中部署RAG系统？",
        "请从架构、性能、安全和成本角度综合分析RAG系统的最佳实践",
    ]

    for q in all_test_queries:
        c = router.classifier.classify(q)
        icon = {"simple": "[S]", "moderate": "[M]", "complex": "[C]"}
        print(f"  {icon.get(c.complexity.value, '[?]')} {q[:60]}... → {c.complexity.value} ({c.confidence:.2f})")

    print("\n" + "=" * 60)
    print("Adaptive RAG 路由演示完成！")
    print("=" * 60)
