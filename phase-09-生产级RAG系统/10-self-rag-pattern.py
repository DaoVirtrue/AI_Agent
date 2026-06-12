#!/usr/bin/env python3
"""
Self-RAG 模式实现 (Self-Reflective RAG)
On-Demand Retrieval with Reflection Tokens

反思令牌 (Reflection Tokens):
  - <retrieve>: 触发按需检索
  - <relevant> / <irrelevant>: 检索结果相关性判断
  - <supported> / <partially_supported> / <contradictory>: 生成质量判断

工作流程:
  1. 分析查询 → 决定是否需要检索 → 生成 <retrieve>
  2. 检索相关文档 → 判断相关性 → <relevant>/<irrelevant>
  3. 基于相关文档生成回答 → 自我批评 → <supported>/...
  4. 如需优化，进入新一轮迭代
"""

import re
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Callable


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class RetrieveDecision(Enum):
    """检索决策"""
    RETRIEVE = "retrieve"       # 需要检索
    NO_RETRIEVE = "no_retrieve"  # 不需要检索


class RelevanceJudgment(Enum):
    """相关性判断"""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class SupportJudgment(Enum):
    """支持性判断"""
    FULLY_SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTORY = "contradictory"


class SelfRAGState(Enum):
    """Self-RAG 处理状态"""
    INIT = "init"
    DECIDING = "deciding"
    RETRIEVING = "retrieving"
    JUDGING = "judging"
    GENERATING = "generating"
    REFLECTING = "reflecting"
    REFINING = "refining"
    DONE = "done"


@dataclass
class ReflectionResult:
    """反思结果"""
    retrieve_decision: Optional[RetrieveDecision] = None
    retrieve_reason: str = ""
    relevance_judgments: list[dict] = field(default_factory=list)
    support_judgment: Optional[SupportJudgment] = None
    support_reason: str = ""
    needs_refinement: bool = False
    refinement_suggestions: list[str] = field(default_factory=list)


@dataclass
class SelfRAGResult:
    """Self-RAG 结果"""
    query: str
    answer: str
    passages_used: list[dict]
    reflection: ReflectionResult
    iterations: int
    final_quality_score: float
    trace: list[dict] = field(default_factory=list)


# ============================================================
# 模拟LLM（生产中替换为真实API调用）
# ============================================================

class LLMInterface:
    """LLM接口（可替换为Anthropic/OpenAI等）"""

    def generate(self, prompt: str, stop_tokens: Optional[list[str]] = None) -> str:
        """生成文本"""
        return self._mock_generate(prompt, stop_tokens)

    def _mock_generate(self, prompt: str, stop_tokens: Optional[list[str]] = None) -> str:
        """模拟LLM生成（生产环境替换为真实API调用）"""
        # 基于prompt内容返回模拟的合理回答
        if "是否需要检索" in prompt or "retrieve" in prompt.lower():
            if any(k in prompt for k in ["最新", "今天", "当前", "实时", "specific"]):
                return "<retrieve>\n需要检索外部知识"
            return "no_retrieve"

        if "相关" in prompt or "relevant" in prompt.lower():
            return "<relevant>\n这段内容与查询高度相关"

        if "支持" in prompt or "supported" in prompt.lower():
            if "完整" in prompt:
                return "<fully_supported>\n回答完全基于提供的文档"
            elif "部分" in prompt:
                return "<partially_supported>\n回答部分基于文档，部分来自常识"
            return "<supported>\n回答得到了文档支持"

        # 默认返回一个回答
        return "基于检索到的信息，我为您提供以下回答：这是一个模拟的Self-RAG回答。"

    def generate_with_tokens(self, prompt: str,
                             reflection_tokens: list[str]) -> tuple[str, dict]:
        """生成并提取反思令牌

        Returns:
            (cleaned_text, tokens_found)
        """
        raw_output = self.generate(prompt)
        cleaned, tokens = self._extract_tokens(raw_output, reflection_tokens)
        return cleaned, tokens

    @staticmethod
    def _extract_tokens(text: str, token_types: list[str]) -> tuple[str, dict]:
        """从文本中提取反思令牌"""
        found_tokens = {}
        clean_text = text

        for token in token_types:
            pattern = rf"<{token}>\s*"
            matches = re.findall(pattern, clean_text)
            if matches:
                found_tokens[token] = matches
                clean_text = re.sub(pattern, "", clean_text)

        return clean_text.strip(), found_tokens


# ============================================================
# 检索器接口
# ============================================================

class Retriever:
    """检索器接口"""

    def __init__(self):
        self._knowledge_base: list[dict] = self._build_demo_kb()

    def search(self, query: str, k: int = 5) -> list[dict]:
        """搜索相关文档"""
        # 模拟检索
        results = []
        for doc in self._knowledge_base[:k]:
            score = random.uniform(0.5, 0.99)
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "score": score,
                "title": doc.get("title", ""),
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)

    @staticmethod
    def _build_demo_kb() -> list[dict]:
        """构建演示知识库"""
        return [
            {
                "id": "doc_001",
                "title": "RAG基础",
                "text": "检索增强生成(RAG)是一种结合信息检索和文本生成的技术。它通过从外部知识库检索相关文档来增强LLM的回答质量。",
            },
            {
                "id": "doc_002",
                "title": "Self-RAG架构",
                "text": "Self-RAG通过反思令牌实现自我评估。它使用<retrieve>判断是否需要检索，<relevant>评估检索结果，<supported>检查回答的事实准确性。",
            },
            {
                "id": "doc_003",
                "title": "向量数据库",
                "text": "向量数据库如Faiss和Milvus专为高维向量相似度搜索设计。HNSW索引在大多数场景下提供最佳的查询性能和召回率平衡。",
            },
            {
                "id": "doc_004",
                "title": "熔断器模式",
                "text": "熔断器是一种弹性设计模式。它通过三态模型(CLOSED/OPEN/HALF_OPEN)防止级联故障，并提供指数退避的自动恢复机制。",
            },
            {
                "id": "doc_005",
                "title": "多级缓存",
                "text": "RAG系统的四级缓存包括：L1精确查询缓存、L2语义相似度缓存、L3检索结果缓存和L4 LLM响应缓存。每级缓存有不同的TTL和命中率目标。",
            },
        ]


# ============================================================
# Self-RAG 核心处理器
# ============================================================

class SelfRAGProcessor:
    """Self-RAG 核心处理器"""

    # 反思令牌定义
    REFLECTION_TOKENS = [
        "retrieve",
        "relevant",
        "irrelevant",
        "supported",
        "partially_supported",
        "contradictory",
    ]

    def __init__(self, llm: Optional[LLMInterface] = None, retriever: Optional[Retriever] = None,
                 max_iterations: int = 3, quality_threshold: float = 0.8):
        """
        Args:
            llm: LLM接口
            retriever: 检索器
            max_iterations: 最大自我优化迭代次数
            quality_threshold: 质量阈值
        """
        self.llm = llm or LLMInterface()
        self.retriever = retriever or Retriever()
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    def process(self, query: str) -> SelfRAGResult:
        """处理查询 - 完整的Self-RAG流程"""
        trace = []
        state = SelfRAGState.INIT
        reflection = ReflectionResult()
        retrieved_passages: list[dict] = []
        final_answer = ""

        # 阶段1: 分析查询，决定是否需要检索
        trace.append({"state": state.value, "action": "分析查询"})
        state = SelfRAGState.DECIDING

        retrieve_decision, reason = self._decide_retrieve(query)
        reflection.retrieve_decision = retrieve_decision
        reflection.retrieve_reason = reason
        trace.append({
            "state": state.value,
            "decision": retrieve_decision.value,
            "reason": reason,
        })

        # 阶段2: 如果需要检索，执行检索
        if retrieve_decision == RetrieveDecision.RETRIEVE:
            state = SelfRAGState.RETRIEVING
            trace.append({"state": state.value, "action": "检索文档"})

            candidates = self.retriever.search(query, k=5)
            trace.append({"state": state.value, "results": len(candidates)})

            # 阶段3: 判断检索结果的相关性
            state = SelfRAGState.JUDGING
            trace.append({"state": state.value, "action": "判断相关性"})

            for candidate in candidates:
                judgment = self._judge_relevance(query, candidate["text"])
                candidate["relevance"] = judgment.value
                reflection.relevance_judgments.append({
                    "passage_id": candidate["id"],
                    "judgment": judgment.value,
                })
                if judgment == RelevanceJudgment.RELEVANT:
                    retrieved_passages.append(candidate)

            trace.append({
                "state": state.value,
                "relevant": sum(1 for p in retrieved_passages),
                "irrelevant": len(candidates) - len(retrieved_passages),
            })

        # 阶段4: 基于相关文档生成回答
        state = SelfRAGState.GENERATING
        trace.append({"state": state.value, "action": "生成回答"})

        initial_answer = self._generate_answer(query, retrieved_passages)
        trace.append({"state": state.value, "answer_length": len(initial_answer)})

        # 阶段5: 反思和自我批评
        state = SelfRAGState.REFLECTING
        trace.append({"state": state.value, "action": "反思评估"})

        support_result = self._reflect_on_answer(query, initial_answer, retrieved_passages)
        reflection.support_judgment = support_result["judgment"]
        reflection.support_reason = support_result["reason"]
        reflection.needs_refinement = support_result["needs_refinement"]
        reflection.refinement_suggestions = support_result["suggestions"]

        trace.append({
            "state": state.value,
            "judgment": reflection.support_judgment.value,
            "needs_refinement": reflection.needs_refinement,
        })

        final_answer = initial_answer

        # 阶段6: 如果需要优化，迭代改进（最多N轮）
        iterations = 1
        quality_score = self._calculate_quality(support_result)

        while reflection.needs_refinement and iterations < self.max_iterations:
            state = SelfRAGState.REFINING
            trace.append({"state": state.value, "iteration": iterations})

            # 根据建议改进回答
            refined_answer = self._refine_answer(
                query, final_answer, retrieved_passages,
                reflection.refinement_suggestions
            )

            # 重新评估
            support_result = self._reflect_on_answer(query, refined_answer, retrieved_passages)
            reflection.support_judgment = support_result["judgment"]
            reflection.support_reason = support_result["reason"]
            reflection.needs_refinement = support_result["needs_refinement"]
            reflection.refinement_suggestions = support_result["suggestions"]

            quality_score = self._calculate_quality(support_result)
            final_answer = refined_answer
            iterations += 1

        state = SelfRAGState.DONE
        trace.append({"state": state.value, "final_quality": quality_score})

        return SelfRAGResult(
            query=query,
            answer=final_answer,
            passages_used=retrieved_passages,
            reflection=reflection,
            iterations=iterations,
            final_quality_score=quality_score,
            trace=trace,
        )

    # ========== 阶段1: 检索决策 ==========

    def _decide_retrieve(self, query: str) -> tuple[RetrieveDecision, str]:
        """决定是否需要检索"""
        prompt = f"""你是一个RAG系统的决策模块。请判断以下查询是否需要检索外部知识库。

查询: {query}

分析规则:
- 需要检索: 查询涉及特定事实、最新信息、专业知识、数据查询
- 不需要检索: 简单的寒暄、常识性问题、不需要外部信息即可回答

请输出 <retrieve> 或 no_retrieve，以及简短理由。"""

        output = self.llm.generate(prompt)

        if "<retrieve>" in output.lower() or "retrieve" in output.lower():
            reason_line = output.replace("<retrieve>", "").replace("retrieve", "").strip()
            return RetrieveDecision.RETRIEVE, reason_line[:200]
        else:
            return RetrieveDecision.NO_RETRIEVE, output[:200]

    # ========== 阶段2: 相关性判断 ==========

    def _judge_relevance(self, query: str, passage: str) -> RelevanceJudgment:
        """判断检索结果与查询的相关性"""
        prompt = f"""请判断以下文档段落是否与查询相关。

查询: {query}

文档段落:
{passage[:500]}

请输出:
<relevant> - 如果段落与查询直接相关
<irrelevant> - 如果段落与查询无关"""

        output = self.llm.generate(prompt)

        if "<relevant>" in output.lower():
            return RelevanceJudgment.RELEVANT
        else:
            return RelevanceJudgment.IRRELEVANT

    # ========== 阶段3: 回答生成 ==========

    def _generate_answer(self, query: str, passages: list[dict]) -> str:
        """基于检索结果生成回答"""
        context = "\n\n".join(
            f"[文档 {p['id']}]: {p['text']}" for p in passages
        ) if passages else "（无相关文档）"

        prompt = f"""请基于以下文档内容回答用户问题。如果文档不包含相关信息，请如实说明。

用户问题: {query}

相关文档:
{context[:3000]}

请生成准确、简洁的回答："""

        return self.llm.generate(prompt)

    # ========== 阶段4: 反思评估 ==========

    def _reflect_on_answer(self, query: str, answer: str,
                           passages: list[dict]) -> dict:
        """评估回答质量"""
        context = "\n\n".join(
            f"[{p['id']}]: {p['text']}" for p in passages
        ) if passages else "（无文档）"

        prompt = f"""请严格评估以下回答是否得到文档内容的支持。

查询: {query}

文档内容:
{context[:2000]}

生成的回答:
{answer[:1000]}

请输出以下评估结果之一:
<supported> - 回答完全基于文档内容
<partially_supported> - 回答部分基于文档，部分来自外部知识
<contradictory> - 回答与文档内容矛盾

并说明理由和改进建议。"""

        output = self.llm.generate(prompt)

        # 解析反思令牌
        if "<contradictory>" in output.lower():
            judgment = SupportJudgment.CONTRADICTORY
            needs_refinement = True
        elif "<partially_supported>" in output.lower():
            judgment = SupportJudgment.PARTIALLY_SUPPORTED
            needs_refinement = True
        else:
            judgment = SupportJudgment.FULLY_SUPPORTED
            needs_refinement = False

        # 提取建议
        suggestions = []
        if needs_refinement:
            suggestions = [output.replace(str(judgment.value), "").strip()[:200]]

        return {
            "judgment": judgment,
            "reason": output[:300],
            "needs_refinement": needs_refinement,
            "suggestions": suggestions,
        }

    # ========== 阶段5: 优化迭代 ==========

    def _refine_answer(self, query: str, current_answer: str,
                       passages: list[dict], suggestions: list[str]) -> str:
        """根据反思建议优化回答"""
        context = "\n\n".join(
            f"[{p['id']}]: {p['text']}" for p in passages
        ) if passages else ""

        suggestions_text = "\n".join(f"- {s}" for s in suggestions)

        prompt = f"""请优化以下回答，使其更好地基于文档内容。

查询: {query}

文档:
{context[:2000]}

当前回答:
{current_answer[:1000]}

改进建议:
{suggestions_text}

请生成改进后的回答，确保：
1. 所有陈述都有文档依据
2. 去除与文档无关的推测
3. 引用具体文档ID

优化后的回答："""

        return self.llm.generate(prompt)

    # ========== 质量评分 ==========

    @staticmethod
    def _calculate_quality(reflection_result: dict) -> float:
        """基于反思结果计算质量分数"""
        judgment_scores = {
            SupportJudgment.FULLY_SUPPORTED: 0.95,
            SupportJudgment.PARTIALLY_SUPPORTED: 0.65,
            SupportJudgment.CONTRADICTORY: 0.25,
        }
        base_score = judgment_scores.get(
            reflection_result["judgment"], 0.5
        )
        # 不需要优化则加分
        if not reflection_result["needs_refinement"]:
            base_score = min(1.0, base_score + 0.1)

        return base_score


# ============================================================
# 对比评估器 (Self-RAG vs Standard RAG)
# ============================================================

class RAGComparator:
    """Self-RAG vs 标准RAG 对比"""

    @staticmethod
    def compare_standard_rag(query: str, retriever: Retriever, llm: LLMInterface) -> dict:
        """标准RAG流程（无自我反思）"""
        passages = retriever.search(query, k=3)
        context = "\n\n".join(p["text"] for p in passages)

        prompt = f"""基于以下文档回答用户问题。

文档:
{context[:2000]}

问题: {query}

回答:"""

        answer = llm.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "passages_count": len(passages),
            "has_reflection": False,
            "has_relevance_check": False,
        }

    @staticmethod
    def compare_self_rag(query: str, processor: SelfRAGProcessor) -> SelfRAGResult:
        """Self-RAG流程"""
        return processor.process(query)


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Self-RAG 模式 - 演示运行")
    print("=" * 60)

    # 1. 初始化
    llm = LLMInterface()
    retriever = Retriever()
    processor = SelfRAGProcessor(llm=llm, retriever=retriever, max_iterations=3)

    # 2. 演示反思令牌
    print("\n--- 反思令牌 ---")
    print("Self-RAG使用的反思令牌:")
    tokens = ["<retrieve>", "<relevant>", "<irrelevant>", "<supported>", "<partially_supported>", "<contradictory>"]
    for token in tokens:
        print(f"  {token}", end="")
    print()

    # 3. 处理需要检索的查询
    print("\n--- 查询1: 需要检索的技术查询 ---")
    query1 = "Self-RAG是如何通过反思令牌实现自我评估的？"

    result1 = processor.process(query1)
    print(f"查询: {result1.query}")
    print(f"检索决策: {result1.reflection.retrieve_decision.value}")
    print(f"使用文档: {len(result1.passages_used)} 篇")
    print(f"支持判断: {result1.reflection.support_judgment.value}")
    print(f"需要优化: {result1.reflection.needs_refinement}")
    print(f"迭代次数: {result1.iterations}")
    print(f"质量分数: {result1.final_quality_score:.2f}")
    print(f"回答: {result1.answer[:150]}...")

    # 4. 处理不需要检索的查询
    print("\n--- 查询2: 不需要检索的寒暄 ---")
    query2 = "你好，今天天气怎么样？"

    result2 = processor.process(query2)
    print(f"查询: {result2.query}")
    print(f"检索决策: {result2.reflection.retrieve_decision.value}")
    print(f"使用文档: {len(result2.passages_used)} 篇")

    # 5. 相关性判断演示
    print("\n--- 相关性判断演示 ---")
    test_queries = [
        ("什么是向量数据库？", "向量数据库如Faiss专为高维向量相似度搜索设计，支持HNSW等多种索引。"),
        ("什么是向量数据库？", "今天天气很好，适合出去散步。"),
    ]

    for q, passage in test_queries:
        judgment = processor._judge_relevance(q, passage)
        print(f"  查询: {q}")
        print(f"  段落: {passage[:50]}...")
        print(f"  判断: {judgment.value}")
        print()

    # 6. Self-RAG vs 标准RAG对比
    print("--- Self-RAG vs 标准RAG 对比 ---")

    test_query = "RAG系统中如何使用熔断器？"

    # 标准RAG
    std_result = RAGComparator.compare_standard_rag(test_query, retriever, llm)
    print(f"\n标准RAG:")
    print(f"  检索文档数: {std_result['passages_count']}")
    print(f"  有相关性检查: {std_result['has_relevance_check']}")
    print(f"  有反思评估: {std_result['has_reflection']}")
    print(f"  回答: {std_result['answer'][:100]}...")

    # Self-RAG
    self_result = RAGComparator.compare_self_rag(test_query, processor)
    print(f"\nSelf-RAG:")
    print(f"  检索文档数: {len(self_result.passages_used)}")
    print(f"  有相关性检查: True")
    print(f"  有反思评估: True")
    print(f"  支持判断: {self_result.reflection.support_judgment.value}")
    print(f"  质量分数: {self_result.final_quality_score:.2f}")
    print(f"  回答: {self_result.answer[:100]}...")

    # 7. 处理追踪
    print("\n--- Self-RAG 处理追踪 ---")
    for step in result1.trace:
        step_str = f"  [{step['state']}]"
        for k, v in step.items():
            if k != "state":
                step_str += f" {k}={v}"
        print(step_str)

    print("\n" + "=" * 60)
    print("Self-RAG 演示完成！")
    print("=" * 60)
