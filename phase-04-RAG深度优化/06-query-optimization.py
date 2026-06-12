#!/usr/bin/env python3
"""
Phase 04 Module 06: Query Optimization Pipeline
=================================================
7-step query preprocessing pipeline:
1. Basic cleaning (filler words, invalid punctuation)
2. Intent classification (6 types)
3. Disambiguation (domain context injection)
4. LLM query rewriting (complete subject/scope/constraints)
5. Multi-dimension expansion (synonym + terminology + scenario + concise)
6. Multi-turn context completion (coreference resolution)
7. Complex query decomposition (sub-question splitting)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ===========================================================================
# Enums and Data Classes
# ===========================================================================

class IntentType(Enum):
    FACT = "fact"                   # 事实查询: "什么是X?"
    PROCESS = "process"             # 流程查询: "如何做X?"
    COMPARISON = "comparison"       # 对比查询: "X和Y的区别?"
    TROUBLESHOOT = "troubleshoot"   # 故障排查: "为什么X不工作?"
    REASONING = "reasoning"         # 推理性查询: "如果X那么Y?"
    CHAT = "chat"                   # 闲聊: "你好"

class Domain(Enum):
    GENERAL = "general"
    TECHNICAL = "technical"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"


@dataclass
class OptimizedQuery:
    """Result of query optimization."""
    original: str
    cleaned: str = ""
    intent: IntentType = IntentType.FACT
    domain: Domain = Domain.GENERAL
    disambiguated: str = ""
    rewritten: str = ""
    expansions: List[str] = field(default_factory=list)
    context_completed: str = ""
    sub_questions: List[str] = field(default_factory=list)
    is_complex: bool = False
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Step 1: Basic Query Cleaning
# ===========================================================================

FILLER_WORDS_ZH = {
    '那个', '这个', '就是', '然后', '的话', '比如说', '就是说',
    '那个那个', '嗯', '啊', '呃', '嘛', '吧', '呗',
}

FILLER_WORDS_EN = {
    'um', 'uh', 'like', 'you know', 'i mean', 'sort of',
    'kind of', 'basically', 'actually', 'literally', 'just',
}


class QueryCleaner:
    """Step 1: Clean raw user queries."""

    @staticmethod
    def clean(query: str) -> str:
        """Remove filler words, normalize punctuation, trim whitespace."""
        cleaned = query.strip()

        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Remove leading/trailing punctuation clusters
        cleaned = re.sub(r'^[，。！？,\.!\?\s]+', '', cleaned)
        cleaned = re.sub(r'[，。！？,\.!\?\s]+$', '', cleaned)

        # Normalize Chinese punctuation
        cleaned = cleaned.replace('？', '?').replace('！', '!').replace('，', ',')
        cleaned = cleaned.replace('；', ';').replace('：', ':').replace('。', '.')

        # Remove filler words
        for fw in FILLER_WORDS_ZH:
            cleaned = re.sub(r'\b' + re.escape(fw) + r'\b', '', cleaned)
        for fw in FILLER_WORDS_EN:
            cleaned = re.sub(r'\b' + re.escape(fw) + r'\b', '', cleaned, flags=re.IGNORECASE)

        # Remove repeated punctuation
        cleaned = re.sub(r'([!?.。！？,，;；:：])\1+', r'\1', cleaned)

        # Remove trailing "吗" "呢" "啊" etc when they're the last char on a question
        cleaned = re.sub(r'[吗呢啊哦]$', '', cleaned)

        # Re-collapse spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned


# ===========================================================================
# Step 2: Intent Classification
# ===========================================================================

INTENT_PATTERNS: Dict[IntentType, List[re.Pattern]] = {
    IntentType.FACT: [
        re.compile(r'(什么是|是什么|定义|概念|含义|What\s+is|Define)'),
        re.compile(r'^(什么|谁|哪|何时|哪里|Which|Who|Where|When)'),
        re.compile(r'(介绍|解释|Explain|Describe|Overview)'),
    ],
    IntentType.PROCESS: [
        re.compile(r'(如何|怎么|怎样|步骤|方法|How\s+to|How\s+do)'),
        re.compile(r'(教程|指南|流程|Tutorial|Guide|Procedure|Steps?)'),
        re.compile(r'(怎么做|怎么办|How\s+can\s+I)'),
    ],
    IntentType.COMPARISON: [
        re.compile(r'(区别|对比|比较|差异|vs\.?|versus|Compare|Difference)'),
        re.compile(r'(哪个更好|优缺点|Which\s+is\s+better|Pros?\s+and\s+Cons?)'),
        re.compile(r'(和|与|及|or)\s*.{1,10}\s*(哪个|哪一)', re.DOTALL),
    ],
    IntentType.TROUBLESHOOT: [
        re.compile(r'(为什么|报错|错误|失败|Why|Error|Fail|Bug|Issue|Problem)'),
        re.compile(r'(不工作|不能|无法|出问题|doesn\'t\s+work|not\s+working)'),
        re.compile(r'(怎么修|如何解决|Fix|Solve|Resolve|Troubleshoot)'),
    ],
    IntentType.REASONING: [
        re.compile(r'(如果|假如|假设|那么|If|Suppose|Assume|Then)'),
        re.compile(r'(原因|后果|影响|Cause|Effect|Impact|Consequence)'),
        re.compile(r'(会怎样|会导致|What\s+if|What\s+happens?\s+if)'),
    ],
    IntentType.CHAT: [
        re.compile(r'^(你好|嗨|Hello|Hi|Hey)\b'),
        re.compile(r'^(谢谢|感谢|Thanks?|Thank\s+you)\b'),
        re.compile(r'^(再见|拜拜|Bye|Goodbye)\b'),
    ],
}


class IntentClassifier:
    """Step 2: Classify user query intent."""

    @staticmethod
    def classify(query: str) -> Tuple[IntentType, float]:
        """Classify intent with confidence score."""
        scores: Dict[IntentType, int] = {}
        total_hits = 0

        for intent, patterns in INTENT_PATTERNS.items():
            hits = 0
            for pat in patterns:
                matches = pat.findall(query)
                hits += len(matches)
            scores[intent] = hits
            total_hits += hits

        if total_hits == 0:
            return IntentType.FACT, 0.5  # Default to FACT with low confidence

        # Get the best match
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_score == 0:
            return IntentType.FACT, 0.5

        # Confidence = best_score / (best_score + second_best)
        sorted_scores = sorted(scores.values(), reverse=True)
        second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0
        confidence = best_score / (best_score + second_best) if (best_score + second_best) > 0 else 0.5

        return best_intent, min(confidence, 1.0)

    @staticmethod
    def get_retrieval_strategy(intent: IntentType) -> Dict[str, Any]:
        """Get recommended retrieval strategy based on intent."""
        strategies = {
            IntentType.FACT: {
                'top_k': 5, 'prefer_dense': True, 'min_similarity': 0.75,
                'description': 'Precise fact lookup, leans toward dense retrieval',
            },
            IntentType.PROCESS: {
                'top_k': 8, 'prefer_dense': False, 'min_similarity': 0.70,
                'description': 'Step-by-step retrieval, balanced hybrid',
            },
            IntentType.COMPARISON: {
                'top_k': 10, 'prefer_dense': False, 'min_similarity': 0.70,
                'description': 'Multi-faceted retrieval, keyword-heavy',
            },
            IntentType.TROUBLESHOOT: {
                'top_k': 8, 'prefer_dense': True, 'min_similarity': 0.72,
                'description': 'Error-focused retrieval, semantic understanding',
            },
            IntentType.REASONING: {
                'top_k': 12, 'prefer_dense': False, 'min_similarity': 0.68,
                'description': 'Broad retrieval for reasoning chains',
            },
            IntentType.CHAT: {
                'top_k': 3, 'prefer_dense': True, 'min_similarity': 0.80,
                'description': 'Minimal retrieval, mostly chat',
            },
        }
        return strategies.get(intent, strategies[IntentType.FACT])


# ===========================================================================
# Step 3: Disambiguation
# ===========================================================================

DOMAIN_CONTEXT_MAP: Dict[Domain, str] = {
    Domain.TECHNICAL: "在计算机科学和技术领域中，",
    Domain.LEGAL: "在法律和法规的背景下，",
    Domain.MEDICAL: "在医学和临床医疗领域中，",
    Domain.FINANCIAL: "在金融和投资领域中，",
    Domain.GENERAL: "",
}

DOMAIN_KEYWORDS: Dict[Domain, List[str]] = {
    Domain.TECHNICAL: ['代码', 'API', '算法', '架构', '部署', '服务器', '数据库',
                       'python', 'java', 'docker', 'kubernetes', 'CPU', 'GPU'],
    Domain.LEGAL: ['法律', '法规', '合同', '诉讼', '判决', '条款', '原告', '被告',
                   '知识产权', '仲裁', '赔偿'],
    Domain.MEDICAL: ['患者', '诊断', '治疗', '药物', '手术', '症状', '疾病', '临床',
                     'CT', 'MRI', '化疗', '处方'],
    Domain.FINANCIAL: ['股票', '基金', '投资', '收益', '风险', 'ROI', '市盈率',
                       '财报', '营收', '资产', '负债', '分红'],
}


class Disambiguator:
    """Step 3: Inject domain context to disambiguate queries."""

    @staticmethod
    def detect_domain(query: str) -> Domain:
        """Detect domain from query keywords."""
        scores: Dict[Domain, int] = {}
        query_lower = query.lower()

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw.lower() in query_lower:
                    score += 1
            scores[domain] = score

        scores[Domain.GENERAL] = 1  # Default fallback

        best = max(scores, key=scores.get)
        if scores[best] >= 2:
            return best
        return Domain.GENERAL

    @staticmethod
    def disambiguate(query: str, domain: Optional[Domain] = None) -> str:
        """Inject domain context prefix if domain is detected."""
        if domain is None:
            domain = Disambiguator.detect_domain(query)

        if domain == Domain.GENERAL:
            return query

        prefix = DOMAIN_CONTEXT_MAP.get(domain, "")
        if query.startswith(prefix):
            return query

        return prefix + query


# ===========================================================================
# Step 4: LLM Query Rewriting (simulated)
# ===========================================================================

class QueryRewriter:
    """Step 4: Rewrite queries for completeness and clarity.
    NOTE: In production, this uses an LLM. This is a simulated version with
    templates and rules for offline testing. See 07-query-rewriting.py for
    the full LLM-based implementation.
    """

    @staticmethod
    def rewrite(query: str, intent: IntentType) -> str:
        """Rewrite query to be more complete and specific."""
        # Template-based rewriting based on intent
        templates = {
            IntentType.FACT:
                "请提供关于「{query}」的详细定义、核心概念和关键特征。",
            IntentType.PROCESS:
                "请详细说明「{query}」的操作步骤、所需条件和注意事项。",
            IntentType.COMPARISON:
                "请对比分析「{query}」，包括各自的优缺点、适用场景和关键差异。",
            IntentType.TROUBLESHOOT:
                "针对「{query}」问题，请分析可能的原因并提供解决方案。",
            IntentType.REASONING:
                "关于「{query}」的推理分析，请说明前提条件、逻辑推导过程和结论。",
            IntentType.CHAT:
                "{query}",
        }

        template = templates.get(intent, templates[IntentType.FACT])
        rewritten = template.replace('{query}', query)

        return rewritten

    @staticmethod
    def check_semantic_similarity(original: str, rewritten: str,
                                  threshold: float = 0.8) -> Tuple[bool, float]:
        """Check if rewritten query preserves original meaning.
        (Simplified: keyword overlap check).

        Returns (is_similar, similarity_score).
        """
        # Simple Jaccard similarity on character bigrams as a proxy
        def bigrams(s: str) -> Set[str]:
            return {s[i:i+2] for i in range(len(s)-1)}

        orig_bigrams = bigrams(original)
        rew_bigrams = bigrams(rewritten)

        if not orig_bigrams or not rew_bigrams:
            return True, 1.0

        intersection = orig_bigrams & rew_bigrams
        union = orig_bigrams | rew_bigrams
        similarity = len(intersection) / len(union) if union else 0.0

        return similarity >= threshold, similarity


# ===========================================================================
# Step 5: Multi-Dimension Query Expansion
# ===========================================================================

SYNONYM_MAP: Dict[str, List[str]] = {
    '性能': ['效率', '速度', 'performance'],
    '准确率': ['精度', 'accuracy', '正确率'],
    '训练': ['学习', 'training', '拟合'],
    '模型': ['算法', 'model', '架构'],
    '部署': ['上线', '发布', 'deploy', '运维'],
    '优化': ['调优', '改进', 'optimization', '提升'],
    '数据库': ['存储', 'database', 'DB'],
    '错误': ['异常', 'bug', 'error', '故障'],
    '接口': ['API', '协议', 'interface'],
}


class QueryExpander:
    """Step 5: Multi-dimension query expansion."""

    DIMENSIONS = ['synonym', 'terminology', 'scenario', 'concise']

    @staticmethod
    def expand(query: str, domain: Domain = Domain.GENERAL,
               intent: IntentType = IntentType.FACT) -> List[str]:
        """Generate expanded query variants across multiple dimensions.

        Returns:
            List of expanded query strings (including the expanded original).
        """
        variants: List[str] = []

        # Dimension 1: Synonym expansion
        synonym_variants = QueryExpander._expand_synonyms(query)
        variants.extend(synonym_variants)

        # Dimension 2: Terminology expansion (add domain-specific terms)
        term_variants = QueryExpander._expand_terminology(query, domain)
        variants.extend(term_variants)

        # Dimension 3: Scenario expansion
        scenario_variants = QueryExpander._expand_scenarios(query, intent)
        variants.extend(scenario_variants)

        # Dimension 4: Concise variant (shorter query)
        concise = QueryExpander._make_concise(query)
        if concise and concise != query:
            variants.append(concise)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique: List[str] = []
        for v in variants:
            if v not in seen and v != query:
                seen.add(v)
                unique.append(v)

        return unique[:5]  # Max 5 variants

    @staticmethod
    def _expand_synonyms(query: str) -> List[str]:
        """Generate synonym variants."""
        variants: List[str] = []
        for word, synonyms in SYNONYM_MAP.items():
            if word in query:
                for syn in synonyms:
                    variants.append(query.replace(word, syn))
        return variants

    @staticmethod
    def _expand_terminology(query: str, domain: Domain) -> List[str]:
        """Add domain-specific terms."""
        domain_terms = {
            Domain.TECHNICAL: ['技术实现', '工程方案', '技术架构'],
            Domain.LEGAL: ['法律依据', '法规条款', '司法解释'],
            Domain.MEDICAL: ['临床指南', '诊疗规范', '医学文献'],
            Domain.FINANCIAL: ['财务分析', '投资策略', '市场趋势'],
            Domain.GENERAL: [],
        }
        terms = domain_terms.get(domain, [])
        return [f"{query} {term}" for term in terms]

    @staticmethod
    def _expand_scenarios(query: str, intent: IntentType) -> List[str]:
        """Add scenario-based variants."""
        scenarios = {
            IntentType.FACT: ['基础知识', '入门概念'],
            IntentType.PROCESS: ['实操指南', '最佳实践'],
            IntentType.COMPARISON: ['对比分析', '选型建议'],
            IntentType.TROUBLESHOOT: ['常见问题', '故障排除'],
            IntentType.REASONING: ['分析推理', '逻辑论证'],
            IntentType.CHAT: [],
        }
        scens = scenarios.get(intent, [])
        return [f"{query} {s}" for s in scens]

    @staticmethod
    def _make_concise(query: str) -> str:
        """Create a shorter, keyword-focused version."""
        # Remove question words, keep key terms
        concise = re.sub(r'^(什么是|如何|怎么|怎样|为什么|请问|请|帮我|告诉我)\s*', '', query)
        concise = re.sub(r'[？?！!。.]$', '', concise)
        return concise.strip()


# ===========================================================================
# Step 6: Multi-Turn Context Completion
# ===========================================================================

class MultiTurnCompleter:
    """Step 6: Resolve coreferences from conversation history."""

    # Coreference patterns
    COREF_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r'\b(它|其|这|那|该)\b'), 'ENTITY'),
        (re.compile(r'\b(这个|那个|这些|那些|以上|上述|前面)\b'), 'PREV_TOPIC'),
        (re.compile(r'\b(后者|前者)\b'), 'COMPARISON_REF'),
        (re.compile(r'\b(it|this|that|these|those|the above|the former|the latter)\b',
                     re.IGNORECASE), 'ENTITY_EN'),
    ]

    def __init__(self, history_size: int = 5):
        """Args: history_size = number of recent turns to keep."""
        self.history: List[Dict[str, str]] = []
        self.history_size = history_size

    def add_turn(self, user_query: str, assistant_response: str = "") -> None:
        """Record a conversation turn."""
        self.history.append({
            'user': user_query,
            'assistant': assistant_response,
        })
        # Keep only recent history
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

    def complete(self, query: str) -> str:
        """Complete query by resolving coreferences against history."""
        if not self.history:
            return query

        # Check if query contains coreference markers
        has_coref = False
        for pattern, _ in self.COREF_PATTERNS:
            if pattern.search(query):
                has_coref = True
                break

        if not has_coref:
            return query

        # Get the last topic from history
        last_turn = self.history[-1]
        last_user_query = last_turn.get('user', '')

        # Try to resolve
        resolved = query

        # Replace "它/其/it" with last entity
        entity_patterns = [r'\b它\b', r'\b其\b', r'\bit\b']
        for ep in entity_patterns:
            if re.search(ep, resolved, re.IGNORECASE):
                # Extract the main entity from the last query
                entity = self._extract_main_entity(last_user_query)
                if entity:
                    resolved = re.sub(ep, entity, resolved, flags=re.IGNORECASE)

        # Replace "这个/那个/以上/this/that" with last query summary
        ref_patterns = [r'\b这个\b', r'\b那个\b', r'\b以上\b', r'\b上述\b',
                       r'\bthis\b', r'\bthat\b']
        for rp in ref_patterns:
            if re.search(rp, resolved, re.IGNORECASE):
                resolved = re.sub(rp, f"关于「{last_user_query[:50]}」", resolved,
                                 flags=re.IGNORECASE)

        return resolved

    def _extract_main_entity(self, text: str) -> str:
        """Extract the main noun entity from a query."""
        # Remove question words
        text = re.sub(r'^(什么是|如何|怎么|为什么|请问)\s*', '', text)
        # Take first 3-5 characters as entity (Chinese) or first word (English)
        words = text.split()
        if words:
            return words[0][:8]
        # For Chinese without spaces, take first meaningful segment
        entity = text[:8]
        return entity


# ===========================================================================
# Step 7: Complex Query Decomposition
# ===========================================================================

class QueryDecomposer:
    """Step 7: Detect and decompose complex queries into sub-questions.

    Complex query indicators:
    - Multiple conditions (AND, OR, 和, 及, 与)
    - Multi-step reasoning (先...再...然后)
    - Comparisons requiring two-pass retrieval
    - Nested questions
    """

    COMPLEX_INDICATORS: List[re.Pattern] = [
        re.compile(r'(以及|还有|另外|此外|同时|and\s+also|in\s+addition)', re.IGNORECASE),
        re.compile(r'(先.*再|先.*然后|首先.*其次|first.*then|step\s+\d)', re.IGNORECASE),
        re.compile(r'(对比|区别|差异|分别|vs\.?|compare.*and)', re.IGNORECASE),
        re.compile(r'\?.*\?|？.*？'),  # Multiple question marks
        re.compile(r'[，,;；].{5,}[？?]'),  # Complex sentence structure
    ]

    @staticmethod
    def is_complex(query: str) -> bool:
        """Check if query is complex and needs decomposition."""
        indicator_count = 0
        for pat in QueryDecomposer.COMPLEX_INDICATORS:
            if pat.search(query):
                indicator_count += 1
        return indicator_count >= 1

    @staticmethod
    def decompose(query: str) -> List[str]:
        """Decompose a complex query into independent sub-questions.

        NOTE: In production, this uses LLM decomposition. This implements
        rule-based splitting as a baseline.
        """
        sub_questions: List[str] = []

        # Split by conjunction markers
        split_markers = [
            r'[；;]\s*',           # Semicolons
            r'[，,]\s*(?=以及|还有|另外|此外|同时)',  # Commas before conjunctions
            r'\s+(?:以及|还有|另外|此外)\s+',         # Conjunctions
            r'\s+and\s+(?:also\s+)?',                # English conjunctions
        ]

        # Try to split
        for marker in split_markers:
            parts = re.split(marker, query)
            if len(parts) >= 2:
                # Clean each part
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 5:
                        # Ensure it ends with a question mark
                        if not part.endswith('?') and not part.endswith('？'):
                            part = part + '?'
                        sub_questions.append(part)
                break

        # If no split was possible, return the original
        if not sub_questions:
            return [query]

        return sub_questions


# ===========================================================================
# Full Pipeline
# ===========================================================================

class QueryOptimizationPipeline:
    """Complete 7-step query optimization pipeline."""

    def __init__(
        self,
        domain_hint: Optional[Domain] = None,
        enable_disambiguation: bool = True,
        enable_rewriting: bool = True,
        enable_expansion: bool = True,
        enable_decomposition: bool = True,
        max_expansions: int = 5,
    ):
        self.domain_hint = domain_hint
        self.enable_disambiguation = enable_disambiguation
        self.enable_rewriting = enable_rewriting
        self.enable_expansion = enable_expansion
        self.enable_decomposition = enable_decomposition
        self.max_expansions = max_expansions

        self.cleaner = QueryCleaner()
        self.classifier = IntentClassifier()
        self.disambiguator = Disambiguator()
        self.rewriter = QueryRewriter()
        self.expander = QueryExpander()
        self.completer = MultiTurnCompleter(history_size=5)
        self.decomposer = QueryDecomposer()

    def process(self, raw_query: str) -> OptimizedQuery:
        """Run the full 7-step optimization pipeline.

        Args:
            raw_query: The raw user query string.

        Returns:
            OptimizedQuery with all processing results.
        """
        result = OptimizedQuery(original=raw_query)

        # Step 1: Clean
        result.cleaned = self.cleaner.clean(raw_query)

        # Step 2: Classify intent
        result.intent, confidence = self.classifier.classify(result.cleaned)
        result.metadata['intent_confidence'] = confidence

        # Step 3: Disambiguate
        if self.enable_disambiguation:
            domain = self.domain_hint or self.disambiguator.detect_domain(result.cleaned)
            result.domain = domain
            result.disambiguated = self.disambiguator.disambiguate(result.cleaned, domain)
        else:
            result.domain = Domain.GENERAL
            result.disambiguated = result.cleaned

        # Step 4: Rewrite
        if self.enable_rewriting:
            rewritten = self.rewriter.rewrite(result.disambiguated, result.intent)
            is_similar, sim_score = self.rewriter.check_semantic_similarity(
                result.cleaned, rewritten
            )
            if is_similar:
                result.rewritten = rewritten
            else:
                result.rewritten = result.disambiguated
                result.metadata['rewrite_rejected'] = True
                result.metadata['rewrite_similarity'] = sim_score
        else:
            result.rewritten = result.disambiguated

        # Step 5: Expand (generate query variants)
        if self.enable_expansion:
            result.expansions = self.expander.expand(
                result.rewritten, result.domain, result.intent
            )[:self.max_expansions]

        # Step 6: Context completion (requires history, called separately)

        # Step 7: Detect and decompose complex queries
        result.is_complex = self.decomposer.is_complex(result.cleaned)
        if self.enable_decomposition and result.is_complex:
            result.sub_questions = self.decomposer.decompose(result.cleaned)

        return result

    def process_with_context(self, raw_query: str) -> OptimizedQuery:
        """Process query including multi-turn context completion (Step 6)."""
        result = self.process(raw_query)
        result.context_completed = self.completer.complete(result.rewritten)
        return result

    def add_history(self, user_query: str, assistant_response: str = "") -> None:
        """Add a conversation turn for multi-turn completion."""
        self.completer.add_turn(user_query, assistant_response)


# ===========================================================================
# Retrieval strategy from optimized query
# ===========================================================================

def get_retrieval_params(result: OptimizedQuery) -> Dict[str, Any]:
    """Derive retrieval parameters from an optimized query."""
    strategy = IntentClassifier.get_retrieval_strategy(result.intent)
    return {
        'top_k': strategy['top_k'] + (2 if result.is_complex else 0),
        'prefer_dense': strategy['prefer_dense'],
        'min_similarity': strategy['min_similarity'],
        'query_variants': [result.rewritten] + result.expansions,
        'sub_questions': result.sub_questions if result.is_complex else [],
        'domain': result.domain.value,
        'intent': result.intent.value,
    }


# ===========================================================================
# __main__: Demo
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Query Optimization Pipeline - Self-Test")
    print("=" * 60)

    pipeline = QueryOptimizationPipeline(
        enable_disambiguation=True,
        enable_rewriting=True,
        enable_expansion=True,
        enable_decomposition=True,
    )

    test_queries = [
        # Simple fact query
        "什么是机器学习？",
        # Process query
        "如何训练一个深度学习模型呢？",
        # Comparison query
        "BERT和GPT的区别是什么，哪个更好用？",
        # Troubleshoot query
        "为什么我的模型训练不收敛？",
        # Reasoning query
        "如果增加网络层数，模型性能会怎样变化？",
        # Complex query
        "什么是注意力机制以及它如何提升NLP模型性能，另外在实际部署中需要注意什么？",
        # Chat query
        "你好，能帮我解答一个问题吗？",
        # Query with filler words
        "那个那个，就是我想问一下，嗯，什么是自然语言处理啊？",
    ]

    for qi, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {qi}: {query}")
        print(f"{'='*60}")

        result = pipeline.process(query)

        print(f"  Step 1 (Clean):       {result.cleaned}")
        print(f"  Step 2 (Intent):      {result.intent.value} "
              f"(confidence: {result.metadata.get('intent_confidence', 'N/A'):.2f})")
        print(f"  Step 3 (Disambig):    Domain={result.domain.value}")
        print(f"                       {result.disambiguated[:80]}...")
        print(f"  Step 4 (Rewrite):     {result.rewritten[:100]}...")
        if result.metadata.get('rewrite_rejected'):
            print(f"                       [REJECTED: similarity too low]")
        print(f"  Step 5 (Expansions):  {len(result.expansions)} variants")
        for ei, exp in enumerate(result.expansions, 1):
            print(f"                       {ei}. {exp}")
        print(f"  Step 7 (Complex):     {'YES' if result.is_complex else 'NO'}")
        if result.sub_questions:
            print(f"  Sub-questions ({len(result.sub_questions)}):")
            for si, sq in enumerate(result.sub_questions, 1):
                print(f"                       {si}. {sq}")

        # Derived retrieval params
        params = get_retrieval_params(result)
        print(f"  → Retrieval:          top_k={params['top_k']}, "
              f"prefer_dense={params['prefer_dense']}, "
              f"min_sim={params['min_similarity']}")

    # --- Test multi-turn context completion ---
    print(f"\n{'='*60}")
    print("Multi-Turn Context Completion Test")
    print(f"{'='*60}")

    pipeline2 = QueryOptimizationPipeline()
    pipeline2.add_history("什么是Transformer架构？", "Transformer是一种基于自注意力机制的神经网络架构...")
    pipeline2.add_history("它有什么优点？", "Transformer的主要优点包括并行计算、长距离依赖建模...")

    followup = "它的训练需要多少数据？"
    result_mt = pipeline2.process_with_context(followup)

    print(f"  Original:     {followup}")
    print(f"  Context completed: {result_mt.context_completed}")

    # --- Test decomposition only ---
    print(f"\n{'='*60}")
    print("Complex Query Decomposition Test")
    print(f"{'='*60}")

    decomposer = QueryDecomposer()
    complex_queries = [
        "什么是RAG系统以及它和传统搜索有什么区别？",
        "如何部署模型到生产环境，另外如何监控模型性能？",
        "Compare BERT and GPT architectures and explain when to use each.",
    ]

    for cq in complex_queries:
        is_comp = decomposer.is_complex(cq)
        sub_qs = decomposer.decompose(cq) if is_comp else []
        print(f"\n  {'[COMPLEX]' if is_comp else '[SIMPLE]'} {cq}")
        for si, sq in enumerate(sub_qs, 1):
            print(f"    Sub {si}: {sq}")

    print(f"\n{'='*60}")
    print("Query Optimization Pipeline self-test completed!")
    print(f"{'='*60}")
