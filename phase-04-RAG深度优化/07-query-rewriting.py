#!/usr/bin/env python3
"""
Phase 04 Module 07: LLM-Based Query Rewriting
================================================
- LLM-based rewriting with few-shot examples
- Completing missing subjects and scope constraints
- Converting colloquial to professional phrasing
- Semantic similarity check: reject rewrite if similarity < 0.8 to original
- Generate 3-5 variants per query
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_EMBED = True
except ImportError:
    HAS_EMBED = False


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class RewriteResult:
    """Result of query rewriting."""
    original: str
    rewritten: str
    variants: List[str] = field(default_factory=list)
    is_valid: bool = True
    similarity_score: float = 1.0
    rejection_reason: str = ""
    method: str = "rule"  # "rule", "template", or "llm"
    latency_ms: float = 0.0


@dataclass
class FewShotExample:
    """A few-shot example for query rewriting."""
    original: str
    rewritten: str
    explanation: str = ""


# ===========================================================================
# Few-Shot Examples
# ===========================================================================

FEW_SHOT_EXAMPLES: List[FewShotExample] = [
    FewShotExample(
        original="那个模型怎么训练啊？",
        rewritten="请说明机器学习模型的训练流程，包括数据准备、模型选择、超参数配置和评估方法。",
        explanation="补充缺失的主语（机器学习模型），将口语化表达（怎么训练啊）转为专业表述，添加范围和约束。",
    ),
    FewShotExample(
        original="性能不太好怎么办？",
        rewritten="当机器学习模型性能不佳时，应该如何进行系统性的诊断和优化？请包括数据质量检查、超参数调优、模型架构改进和正则化方法。",
        explanation="将模糊的'不太好'具体化，补充领域上下文（机器学习），增加可操作的建议框架。",
    ),
    FewShotExample(
        original="BERT vs GPT which better?",
        rewritten="Compare BERT and GPT architectures in terms of: (1) model architecture and training objectives, (2) strengths and weaknesses, (3) suitable downstream tasks, and (4) computational requirements. Which model is more appropriate for text classification versus text generation?",
        explanation="将口语化对比转为结构化对比，明确对比维度，添加具体适用场景。",
    ),
    FewShotExample(
        original="它怎么用？",
        rewritten="请说明[上文中提到的工具/框架]的使用方法，包括安装配置、基本用法、常见参数和示例代码。",
        explanation="补充缺失指代对象，添加使用场景上下文，增加具体的使用步骤。",
    ),
    FewShotExample(
        original="RAG系统部署要注意啥？",
        rewritten="在将RAG（检索增强生成）系统部署到生产环境时，需要注意哪些关键方面？请包括：向量数据库选型、检索延迟优化、模型服务化、监控告警、安全与权限控制。",
        explanation="补充领域术语全称，将口语化'注意啥'转为专业表述，结构化关键关注点。",
    ),
    FewShotExample(
        original="how to improve search accuracy?",
        rewritten="What are the proven techniques to improve search accuracy in an enterprise RAG system? Please cover: hybrid retrieval strategies, reranking methods, query expansion, embedding model selection, and evaluation metrics (Precision@K, MRR, NDCG).",
        explanation="限定领域范围（RAG系统），具体化技术和评估指标。",
    ),
]


# ===========================================================================
# Template-Based Rewriter (No LLM required)
# ===========================================================================

class TemplateQueryRewriter:
    """Template-based query rewriting as a baseline / fallback when no LLM.

    This provides a functional rewriter without requiring API access.
    """

    REWRITE_TEMPLATES: Dict[str, str] = {
        'definition': "请提供关于「{q}」的完整定义，包括核心概念、关键特征和应用场景。",
        'process': "请详细说明「{q}」的完整步骤和流程，包括前置条件、每个步骤的具体操作和预期结果。",
        'comparison': "请对比分析「{q}」，从多个维度（如架构、性能、适用场景、优缺点）进行比较，并给出选型建议。",
        'troubleshoot': "针对「{q}」问题，请进行系统性诊断：分析可能的根本原因，提供分步骤的解决方案，并说明如何预防类似问题。",
        'reasoning': "关于「{q}」，请进行逻辑推理分析：明确前提假设，逐步推导，得出有依据的结论，并讨论边界条件。",
        'general': "请回答关于「{q}」的问题，提供全面、准确、结构化的信息。如果涉及专业术语，请给出解释。",
    }

    def __init__(self):
        pass

    def rewrite(self, query: str, intent: str = 'general') -> RewriteResult:
        """Rewrite a query using templates."""
        t0 = time.time()

        # Clean query for template use
        clean_q = self._clean_for_template(query)

        # Determine template
        template = self.REWRITE_TEMPLATES.get(intent, self.REWRITE_TEMPLATES['general'])

        rewritten = template.replace('{q}', clean_q)

        # Generate variants
        variants = self._generate_variants(clean_q, intent)

        elapsed = (time.time() - t0) * 1000

        return RewriteResult(
            original=query,
            rewritten=rewritten,
            variants=variants,
            is_valid=True,
            similarity_score=self._compute_similarity(query, rewritten),
            method="template",
            latency_ms=elapsed,
        )

    def _clean_for_template(self, query: str) -> str:
        """Clean query for use in template."""
        q = query.strip()
        q = re.sub(r'[？?！!。.]$', '', q)
        q = re.sub(r'^(请问|请|帮我|告诉我|那个|这个|就是)\s*', '', q)
        q = re.sub(r'[吗呢啊哦呀]$', '', q)
        return q

    def _generate_variants(self, query: str, intent: str,
                           num_variants: int = 4) -> List[str]:
        """Generate query variants for multi-query retrieval."""
        variants: List[str] = []

        # Variant 1: Concise version
        concise = re.sub(r'^(请|请提供|请说明|请回答|请对比分析)\s*', '', query)
        concise = re.sub(r'\s*。\s*$', '', concise)
        variants.append(concise)

        # Variant 2: Keyword version
        keywords = self._extract_keywords(concise)
        variants.append(' '.join(keywords))

        # Variant 3: English version (if Chinese)
        if re.search(r'[一-鿿]', query):
            variants.append(self._chinese_to_english_keywords(query))

        # Variant 4: Expanded with domain terms
        domain_terms = self._get_domain_terms(intent)
        if domain_terms:
            variants.append(f"{concise} {' '.join(domain_terms[:3])}")

        # Deduplicate
        seen: Set[str] = {query}
        unique: List[str] = []
        for v in variants:
            if v and v not in seen and len(v) > 3:
                seen.add(v)
                unique.append(v)

        return unique[:num_variants]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords for keyword-based query."""
        # Simple keyword extraction
        stopwords = {'的', '是', '在', '了', '和', '与', '或', '也', '就', '都',
                    'the', 'a', 'an', 'is', 'are', 'of', 'in', 'to', 'for',
                    '请', '说明', '提供', '回答', '分析', '如何', '怎么', '什么'}
        words = re.findall(r'[\w一-鿿]+', text)
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 1]
        return keywords

    def _chinese_to_english_keywords(self, text: str) -> str:
        """Convert Chinese query to English keyword query (simplified)."""
        # Mapping of common Chinese tech terms to English
        mapping = {
            '模型': 'model',
            '训练': 'training',
            '部署': 'deployment',
            '优化': 'optimization',
            '性能': 'performance',
            '准确率': 'accuracy',
            '数据': 'data',
            '学习': 'learning',
            '网络': 'network',
            '系统': 'system',
            '检索': 'retrieval',
            '生成': 'generation',
            '注意力': 'attention',
            '架构': 'architecture',
            '参数': 'parameter',
        }
        result = text
        for zh, en in mapping.items():
            if zh in result:
                result = result.replace(zh, en)
        # Remove remaining Chinese chars
        result = re.sub(r'[一-鿿]+', '', result)
        if not result.strip():
            return text  # Keep original if no translation possible
        return f"{result} (English keywords from: {text[:30]}...)"

    def _get_domain_terms(self, intent: str) -> List[str]:
        """Get domain-specific expansion terms."""
        terms = {
            'definition': ['定义', '概念', '特征', '应用', '原理'],
            'process': ['步骤', '操作', '方法', '最佳实践', '注意事项'],
            'comparison': ['对比', '差异', '优缺点', '适用场景', '选型'],
            'troubleshoot': ['问题', '诊断', '解决方案', '预防', '排查'],
            'reasoning': ['前提', '推理', '结论', '边界条件', '假设'],
        }
        return terms.get(intent, ['概述', '总结', '关键信息'])

    def _compute_similarity(self, original: str, rewritten: str) -> float:
        """Compute semantic similarity between original and rewritten query."""
        if HAS_EMBED:
            try:
                model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2'
                )
                emb = model.encode([original, rewritten], convert_to_numpy=True)
                sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
                return float(sim)
            except Exception:
                pass

        # Fallback: Jaccard similarity on character bigrams
        def bigrams(s: str) -> set:
            return set(s[i:i+2] for i in range(len(s)-1))
        o = bigrams(original)
        r = bigrams(rewritten)
        if not o or not r:
            return 0.5
        return len(o & r) / len(o | r)


# ===========================================================================
# LLM-Based Rewriter (with simulated LLM for offline testing)
# ===========================================================================

class LLMQueryRewriter:
    """LLM-based query rewriter.

    In production, this connects to an actual LLM API. For offline testing
    and demonstration, it uses enhanced template patterns with few-shot
    examples to simulate LLM behavior.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        num_variants: int = 4,
        use_llm: bool = False,
        llm_endpoint: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.num_variants = num_variants
        self.use_llm = use_llm
        self.llm_endpoint = llm_endpoint
        self.llm_api_key = llm_api_key

        # Fallback to template rewriter
        self.template_rewriter = TemplateQueryRewriter()
        # Store few-shot examples
        self.examples = FEW_SHOT_EXAMPLES

    def rewrite(self, query: str, intent: str = 'general',
                domain: str = 'general') -> RewriteResult:
        """Rewrite a query using the best available method."""
        t0 = time.time()

        # Try LLM if enabled
        if self.use_llm and self.llm_endpoint:
            result = self._rewrite_with_llm(query, intent, domain)
        else:
            result = self._rewrite_with_templates(query, intent, domain)

        # Validate similarity
        sim = self._check_similarity(query, result.rewritten)
        result.similarity_score = sim

        if sim < self.similarity_threshold:
            result.is_valid = False
            result.rejection_reason = (
                f"Rewrite similarity ({sim:.3f}) below threshold "
                f"({self.similarity_threshold}). Original query preserved."
            )
            result.rewritten = query  # Fallback to original
            result.variants = self._generate_safe_variants(query)

        result.latency_ms = (time.time() - t0) * 1000
        return result

    def rewrite_batch(self, queries: List[Tuple[str, str, str]]
                     ) -> List[RewriteResult]:
        """Rewrite a batch of (query, intent, domain) tuples."""
        return [self.rewrite(q, intent, domain)
                for q, intent, domain in queries]

    def _rewrite_with_templates(self, query: str, intent: str,
                                domain: str) -> RewriteResult:
        """Use template-based rewriting enhanced with domain knowledge."""
        result = self.template_rewriter.rewrite(query, intent)

        # Enhance with domain-specific context
        if domain != 'general':
            domain_prefix = self._get_domain_prefix(domain)
            result.rewritten = f"{domain_prefix}{result.rewritten}"

        return result

    def _rewrite_with_llm(self, query: str, intent: str,
                          domain: str) -> RewriteResult:
        """Rewrite using an actual LLM API. Falls back to templates if
        LLM is unavailable."""
        # Build prompt with few-shot examples
        prompt = self._build_rewrite_prompt(query, intent, domain)

        # Try LLM API call
        try:
            rewritten = self._call_llm_api(prompt)
            variants = self._extract_variants_from_llm(rewritten)
        except Exception:
            # Fallback to templates
            return self._rewrite_with_templates(query, intent, domain)

        return RewriteResult(
            original=query,
            rewritten=rewritten,
            variants=variants[:self.num_variants],
            is_valid=True,
            similarity_score=0.0,  # Will be set by caller
            method="llm",
            latency_ms=0.0,
        )

    def _build_rewrite_prompt(self, query: str, intent: str,
                              domain: str) -> str:
        """Build the LLM prompt for query rewriting with few-shot examples."""
        # Few-shot examples
        examples_text = ""
        for i, ex in enumerate(self.examples[:3], 1):
            examples_text += (
                f"Example {i}:\n"
                f"  Input: {ex.original}\n"
                f"  Output: {ex.rewritten}\n"
                f"  Why: {ex.explanation}\n\n"
            )

        prompt = f"""You are a query rewriting expert for RAG retrieval systems.

Your task: Rewrite the user's query to be more complete, precise, and retrieval-friendly.

Rules:
1. Complete missing subjects and objects
2. Add scope constraints (domain, time, context)
3. Convert colloquial/casual language to professional terminology
4. Make implicit requirements explicit
5. Structure multi-aspect queries
6. Preserve the original intent and meaning
7. Do NOT add information not implied by the original query

Intent: {intent}
Domain: {domain}

Few-shot examples:
{examples_text}

Now, rewrite the following query:
Input: {query}
Output:"""

        return prompt

    def _call_llm_api(self, prompt: str) -> str:
        """Call LLM API (simulated for offline testing).

        In production, replace with actual API call (OpenAI, Anthropic, etc.).
        """
        # Simulated LLM rewriting for demonstration
        # Uses enhanced template with more natural language
        query_match = re.search(r'Input:\s*(.+?)$', prompt, re.MULTILINE)
        if not query_match:
            return prompt

        original = query_match.group(1).strip()

        # Enhanced rewriting that combines templates
        rewrites = [
            f"请提供关于「{original}」的详细信息，包括背景知识、核心概念、关键技术点和实际应用场景。",
            f"请从以下角度分析「{original}」：1) 基本概念与定义 2) 关键组件与原理 3) 常见方法与最佳实践 4) 局限性与未来趋势。",
            f"就「{original}」这一主题，请给出一份全面的技术说明，涵盖理论基础、工程实践和业界案例。",
        ]
        return rewrites[hash(original) % len(rewrites)]

    def _extract_variants_from_llm(self, rewritten: str) -> List[str]:
        """Extract query variants from LLM output."""
        # For simulated version, generate variants
        return self.template_rewriter._generate_variants(rewritten, 'general')

    def _get_domain_prefix(self, domain: str) -> str:
        """Get domain-specific context prefix."""
        prefixes = {
            'legal': '在法律和法规框架下，',
            'medical': '在医学和临床医疗领域内，',
            'financial': '在金融和投资领域内，',
            'technical': '在计算机科学和软件工程领域内，',
        }
        return prefixes.get(domain, '')

    def _check_similarity(self, original: str, rewritten: str) -> float:
        """Check semantic similarity between original and rewritten."""
        return self.template_rewriter._compute_similarity(original, rewritten)

    def _generate_safe_variants(self, query: str) -> List[str]:
        """Generate safe variants (original + minor variations)."""
        variants = [query]
        # Add keyword variant
        keywords = re.findall(r'[\w一-鿿]{2,}', query)
        if keywords:
            variants.append(' '.join(keywords[:5]))
        # Add lowercase
        variants.append(query.lower())
        return list(set(variants))


# ===========================================================================
# __main__: Demo
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Query Rewriting - Self-Test")
    print("=" * 60)

    # Initialize rewriter
    rewriter = LLMQueryRewriter(
        similarity_threshold=0.8,
        num_variants=4,
        use_llm=False,
    )

    test_cases: List[Tuple[str, str, str]] = [
        # (query, intent, domain)
        ("什么是注意力机制？", "definition", "technical"),
        ("模型训练不出来怎么办？", "troubleshoot", "technical"),
        ("Transformer和CNN有什么区别？", "comparison", "technical"),
        ("如何部署RAG系统到生产环境？", "process", "technical"),
        ("如果增加数据量，模型精度会提高吗？", "reasoning", "technical"),
        ("BERT vs GPT哪个好？", "comparison", "technical"),
        ("怎么提高检索准确率？", "process", "technical"),
    ]

    total_valid = 0
    total_rejected = 0

    for qi, (query, intent, domain) in enumerate(test_cases, 1):
        print(f"\n{'─'*50}")
        print(f"Test {qi}: {query}")
        print(f"  Intent: {intent}, Domain: {domain}")

        result = rewriter.rewrite(query, intent, domain)

        print(f"  Method:      {result.method}")
        print(f"  Valid:       {'YES' if result.is_valid else 'NO'}")
        print(f"  Similarity:  {result.similarity_score:.3f}")
        print(f"  Latency:     {result.latency_ms:.1f}ms")

        if not result.is_valid:
            print(f"  Rejected:    {result.rejection_reason}")
            total_rejected += 1
        else:
            total_valid += 1

        print(f"  Original:    {result.original}")
        print(f"  Rewritten:   {result.rewritten[:120]}...")

        if result.variants:
            print(f"  Variants ({len(result.variants)}):")
            for vi, v in enumerate(result.variants, 1):
                print(f"    {vi}. {v[:80]}...")
        else:
            print(f"  Variants: (none)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {total_valid} valid rewrites, {total_rejected} rejected")
    print(f"Rejection threshold: {rewriter.similarity_threshold}")
    print(f"{'='*60}")

    # --- Test batch rewriting ---
    print(f"\n{'='*60}")
    print("Batch Rewriting Test")
    print(f"{'='*60}")
    batch = test_cases[:3]
    results = rewriter.rewrite_batch(batch)
    print(f"Batch of {len(batch)} queries processed in "
          f"{sum(r.latency_ms for r in results):.1f}ms total")

    # --- Test with very different rewrite (should be rejected) ---
    print(f"\n{'='*60}")
    print("Semantic Drift Detection Test")
    print(f"{'='*60}")
    edge_case = ("你好", "chat", "general")
    result_edge = rewriter.rewrite(*edge_case)
    print(f"  Original:  {result_edge.original}")
    print(f"  Rewritten: {result_edge.rewritten[:80]}...")
    print(f"  Similarity: {result_edge.similarity_score:.3f}")
    if result_edge.similarity_score < 0.8:
        print(f"  >> Correctly flagged as potential semantic drift")
    else:
        print(f"  >> Similarity acceptable")

    print(f"\n{'='*60}")
    print("Query Rewriting self-test completed!")
    print(f"{'='*60}")
