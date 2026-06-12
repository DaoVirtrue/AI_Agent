#!/usr/bin/env python3
"""
六维度评估框架 (Six-Dimension Evaluation Framework)
====================================================
评估RAG系统的六个关键维度:
  1. Accuracy (准确度) - 精确匹配 + 关键事实覆盖率 + 语义相似度
  2. Relevance (相关性) - LLM-as-Judge 1-3分制
  3. Faithfulness (忠实度) - 声明分解 + NLI蕴含关系
  4. Fluency (流畅度) - 重复检测 + 长度 + 句式多样性 + 语法
  5. Latency (延迟) - 归一化 (≥10s→0)
  6. Cost (成本) - 归一化 (≥$0.10→0)

每个维度得分0-1，加权组合为最终分数。
"""

import re
import time
import math
import hashlib
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
from enum import Enum

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# 核心数据结构
# ============================================================================

@dataclass
class MultiDimScore:
    """多维度评估得分"""
    accuracy: float = 0.0
    relevance: float = 0.0
    faithfulness: float = 0.0
    fluency: float = 0.0
    latency_score: float = 0.0
    cost_score: float = 0.0

    # 原始指标
    latency_seconds: float = 0.0
    cost_dollars: float = 0.0
    dimension_details: Dict[str, Any] = field(default_factory=dict)

    # 默认权重
    DEFAULT_WEIGHTS = {
        'accuracy': 0.30,
        'relevance': 0.20,
        'faithfulness': 0.25,
        'fluency': 0.10,
        'latency': 0.05,
        'cost': 0.10,
    }

    def weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """计算加权总分

        Args:
            weights: 自定义权重字典，默认使用DEFAULT_WEIGHTS

        Returns:
            0-1之间的加权总分
        """
        if weights is None:
            weights = self.DEFAULT_WEIGHTS

        total = 0.0
        total += self.accuracy * weights.get('accuracy', 0.0)
        total += self.relevance * weights.get('relevance', 0.0)
        total += self.faithfulness * weights.get('faithfulness', 0.0)
        total += self.fluency * weights.get('fluency', 0.0)
        total += self.latency_score * weights.get('latency', 0.0)
        total += self.cost_score * weights.get('cost', 0.0)
        return round(total, 4)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'accuracy': self.accuracy,
            'relevance': self.relevance,
            'faithfulness': self.faithfulness,
            'fluency': self.fluency,
            'latency_score': self.latency_score,
            'cost_score': self.cost_score,
            'latency_seconds': self.latency_seconds,
            'cost_dollars': self.cost_dollars,
            'weighted_total': self.weighted_score(),
            'details': self.dimension_details,
        }

    def __repr__(self) -> str:
        lines = [
            f"MultiDimScore (加权总分: {self.weighted_score():.3f})",
            f"  ├─ Accuracy:      {self.accuracy:.3f}",
            f"  ├─ Relevance:     {self.relevance:.3f}",
            f"  ├─ Faithfulness:  {self.faithfulness:.3f}",
            f"  ├─ Fluency:       {self.fluency:.3f}",
            f"  ├─ Latency:       {self.latency_score:.3f} ({self.latency_seconds:.2f}s)",
            f"  └─ Cost:          {self.cost_score:.3f} (${self.cost_dollars:.4f})",
        ]
        return "\n".join(lines)


@dataclass
class ClaimWithEntailment:
    """声明与其蕴含关系"""
    claim: str
    label: str  # 'entailment', 'contradiction', 'neutral'


# ============================================================================
# 六维度评估器
# ============================================================================

class MultiDimensionEvaluator:
    """RAG系统的完整六维度评估器

    每个维度的评估方法:
      Accuracy    → 精确匹配 + 关键事实覆盖率 + 语义相似度
      Relevance   → LLM-as-Judge 评分 (1-3分，归一化到0-1)
      Faithfulness → 声明分解 + NLI蕴含关系验证
      Fluency     → 重复检测 + 合理长度 + 句式多样性 + 语法检查
      Latency     → 响应时间归一化 (≥10s 得0分)
      Cost        → 成本归一化 (≥$0.10 得0分)
    """

    # 流畅度参数
    REPETITION_NGRAM = 4          # 检测重复的n-gram大小
    REPETITION_MAX_RATIO = 0.3    # 最大允许的重复比例
    IDEAL_LENGTH_MIN = 30         # 理想回答最小字数
    IDEAL_LENGTH_MAX = 500        # 理想回答最大字数
    MAX_LENGTH_FOR_SCORE = 2000   # 最大长度用于评分

    def __init__(self, llm_judge_fn: Optional[callable] = None):
        """初始化评估器

        Args:
            llm_judge_fn: LLM-as-Judge函数，接收 (prompt) 返回 (score, reasoning)
                          如果不提供，使用基于规则的启发式实现
        """
        self.llm_judge = llm_judge_fn or self._heuristic_judge
        self._tfidf_vectorizer = None

    # ========================================================================
    # 1. Accuracy (准确度)
    # ========================================================================

    def evaluate_accuracy(
        self,
        response: str,
        reference: str,
        key_facts: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """评估准确度

        三部分组成:
          1. 精确匹配得分 (5%)
          2. 关键事实覆盖率 (60%)
          3. 语义相似度 (35%)

        Args:
            response: 系统回答
            reference: 参考答案
            key_facts: 关键事实列表 (可选)

        Returns:
            (准确度得分(0-1), 详细信息字典)
        """
        details = {}

        # 1. 精确匹配 (5%)
        exact_match_score = 1.0 if response.strip() == reference.strip() else 0.0
        details['exact_match'] = exact_match_score

        # 2. 关键事实覆盖率 (60%)
        if key_facts:
            fact_scores = []
            for fact in key_facts:
                # 使用模糊匹配检测关键事实
                fact_present = self._fuzzy_fact_match(fact, response)
                fact_scores.append(1.0 if fact_present else 0.0)
            fact_coverage = statistics.mean(fact_scores) if fact_scores else 0.5
        else:
            # 自动提取关键事实: 提取reference中的名词短语和数字
            extracted_facts = self._extract_key_facts(reference)
            fact_scores = []
            for fact in extracted_facts:
                fact_present = self._fuzzy_fact_match(fact, response)
                fact_scores.append(1.0 if fact_present else 0.0)
            fact_coverage = statistics.mean(fact_scores) if fact_scores else 0.5
        details['fact_coverage'] = round(fact_coverage, 4)
        details['facts_checked'] = len(key_facts) if key_facts else len(extracted_facts) if 'extracted_facts' in dir() else 0

        # 3. 语义相似度 (35%)
        semantic_score = self._compute_semantic_similarity(response, reference)
        details['semantic_similarity'] = round(semantic_score, 4)

        # 加权组合
        accuracy = (
            0.05 * exact_match_score
            + 0.60 * fact_coverage
            + 0.35 * semantic_score
        )
        return round(accuracy, 4), details

    def _fuzzy_fact_match(self, fact: str, text: str) -> bool:
        """模糊事实匹配 - 检查事实是否存在于文本中"""
        # 对事实进行分词，检查大部分词是否出现在文本中
        fact_lower = fact.lower()
        text_lower = text.lower()

        # 直接包含检查
        if fact_lower in text_lower:
            return True

        # 单词级别检查 - 需要80%以上的单词匹配
        fact_words = set(re.findall(r'\w+', fact_lower))
        if not fact_words:
            return False

        text_words = set(re.findall(r'\w+', text_lower))
        overlap = fact_words & text_words
        return len(overlap) / len(fact_words) >= 0.8

    def _extract_key_facts(self, text: str) -> List[str]:
        """从文本中提取关键事实 (启发式方法)"""
        facts = []

        # 提取包含数字的句子
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if re.search(r'\d+', sent) and len(sent) > 10:
                facts.append(sent[:100])

        # 提取专有名词 (简化: 大写字母开头的连续词)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        facts.extend(proper_nouns[:10])

        return facts[:20]  # 最多20个事实

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度 (基于TF-IDF的余弦相似度)"""
        if SKLEARN_AVAILABLE:
            try:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(similarity)
            except Exception:
                pass

        # 回退: Jaccard相似度
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    # ========================================================================
    # 2. Relevance (相关性)
    # ========================================================================

    def evaluate_relevance(
        self,
        response: str,
        query: str,
        context: Optional[str] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """使用LLM-as-Judge评估相关性

        Args:
            response: 系统回答
            query: 用户查询
            context: 检索到的上下文 (可选)

        Returns:
            (相关性得分(0-1), 详细信息字典)
        """
        prompt = f"""你是一个严格的评估专家。评估以下回答对于给定问题的相关性。

评分标准:
- 3分: 回答完全针对问题，所有内容都直接相关，没有无关信息
- 2分: 回答基本针对问题，但包含一些不太相关的信息或遗漏了部分问题
- 1分: 回答只有少量内容与问题相关，大部分偏离主题
- 0分: 回答与问题完全无关

问题: {query}

回答: {response}

请给出评分(0-3)和简短理由。格式: "评分: X\n理由: ..."
"""
        score, reasoning = self.llm_judge(prompt)
        # 归一化到0-1
        normalized_score = min(max(score / 3.0, 0.0), 1.0)

        details = {
            'raw_score': score,
            'normalized': round(normalized_score, 4),
            'reasoning': reasoning,
        }
        return normalized_score, details

    def _heuristic_judge(self, prompt: str) -> Tuple[int, str]:
        """启发式评判 (无LLM时的备选方案)

        基于关键词重叠和长度进行简单评分
        """
        # 简单解析prompt提取问题和回答
        query_match = re.search(r'问题:\s*(.+?)(?:\n|$)', prompt)
        response_match = re.search(r'回答:\s*(.+?)(?:\n|$)', prompt)
        if not query_match or not response_match:
            return 2, "无法解析prompt，默认评分2分"

        query = query_match.group(1)
        response = response_match.group(1)

        # 关键词重叠
        query_words = set(re.findall(r'\w+', query.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))
        overlap = query_words & response_words

        if len(query_words) > 0:
            overlap_ratio = len(overlap) / len(query_words)
        else:
            overlap_ratio = 0.0

        # 长度合理性
        response_len = len(response.split())
        length_ok = 10 < response_len < 200

        if overlap_ratio > 0.5 and length_ok:
            score = 3
            reason = "高关键词重叠且长度合理"
        elif overlap_ratio > 0.3:
            score = 2
            reason = "中等关键词重叠"
        elif overlap_ratio > 0.1:
            score = 1
            reason = "低关键词重叠"
        else:
            score = 0
            reason = "几乎不相关"

        return score, reason

    # ========================================================================
    # 3. Faithfulness (忠实度)
    # ========================================================================

    def evaluate_faithfulness(
        self,
        response: str,
        context: str,
        source_doc_ids: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """评估忠实度: 回答是否忠实于提供的上下文

        步骤:
          1. 将回答分解为原子声明 (claim decomposition)
          2. 对每个声明检查与上下文的蕴含关系 (NLI entailment)

        Args:
            response: 系统回答
            context: 提供的上下文
            source_doc_ids: 源文档ID列表 (可选)

        Returns:
            (忠实度得分(0-1), 详细信息字典)
        """
        # 1. 声明分解
        claims = self._decompose_into_claims(response)
        if not claims:
            return 1.0, {'claims': [], 'entailments': [], 'warning': '无声明可分解'}

        # 2. NLI蕴含关系检查 (简化版: 使用启发式方法)
        entailments: List[ClaimWithEntailment] = []
        for claim in claims:
            label = self._check_entailment_heuristic(claim, context)
            entailments.append(ClaimWithEntailment(claim=claim, label=label))

        # 3. 计算忠实度得分
        faithful_count = sum(1 for e in entailments if e.label == 'entailment')
        score = faithful_count / len(claims) if claims else 1.0

        details = {
            'claims': [{'claim': c.claim, 'label': c.label} for c in entailments],
            'claim_count': len(claims),
            'faithful_count': faithful_count,
            'contradiction_count': sum(1 for e in entailments if e.label == 'contradiction'),
            'neutral_count': sum(1 for e in entailments if e.label == 'neutral'),
            'source_doc_count': len(source_doc_ids) if source_doc_ids else 0,
        }
        return round(score, 4), details

    def _decompose_into_claims(self, text: str) -> List[str]:
        """将文本分解为原子声明

        使用句子分割作为声明分解的基本单位
        """
        # 按句号、问号、感叹号分割
        raw_claims = re.split(r'[.!?。！？]+', text)
        claims = []

        for claim in raw_claims:
            claim = claim.strip()
            if len(claim) < 5:  # 过滤太短的片段
                continue
            # 如果声明太长，按逗号/分号进一步分割
            if len(claim) > 150:
                sub_claims = re.split(r'[,;，；、]+', claim)
                for sc in sub_claims:
                    sc = sc.strip()
                    if len(sc) >= 5:
                        claims.append(sc)
            else:
                claims.append(claim)

        return claims

    def _check_entailment_heuristic(self, claim: str, context: str) -> str:
        """启发式蕴含关系检查

        使用关键词匹配和否定词检测来判断:
          - entailment: 声明内容在上下文中找到支持
          - contradiction: 声明内容与上下文明显矛盾
          - neutral: 无法判断

        Returns: 'entailment' | 'contradiction' | 'neutral'
        """
        claim_lower = claim.lower()
        context_lower = context.lower()

        # 获取声明的关键词 (长词更有意义)
        claim_words = [w for w in re.findall(r'\w+', claim_lower) if len(w) > 2]
        if not claim_words:
            return 'neutral'

        context_words = set(re.findall(r'\w+', context_lower))

        # 计算关键词在上下文中的存在比例
        matched = sum(1 for w in claim_words if w in context_words)
        match_ratio = matched / len(claim_words)

        # 检测否定 (简化: 检查 "not", "no", "never", "don't" 等)
        negations_in_claim = re.findall(r'\b(not|no|never|don\'t|doesn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|won\'t|can\'t|cannot)\b', claim_lower)
        negations_in_context = re.findall(r'\b(not|no|never|don\'t|doesn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|won\'t|can\'t|cannot)\b', context_lower)

        # 判断
        if match_ratio >= 0.7:
            return 'entailment'
        elif match_ratio >= 0.4:
            # 检查是否有矛盾
            if negations_in_claim and match_ratio < 0.5:
                return 'contradiction'
            return 'neutral'
        else:
            return 'neutral'

    # ========================================================================
    # 4. Fluency (流畅度)
    # ========================================================================

    def evaluate_fluency(self, response: str) -> Tuple[float, Dict[str, Any]]:
        """评估流畅度

        四个子维度:
          1. 重复检测 (40%) - 检测n-gram重复
          2. 长度合理性 (20%) - 不能太短也不能太长
          3. 句式多样性 (20%) - 不同句长的标准差
          4. 语法检查 (20%) - 基本语法规则

        Args:
            response: 系统回答

        Returns:
            (流畅度得分(0-1), 详细信息字典)
        """
        words = response.split()
        if not words:
            return 0.0, {'error': '空回答'}

        # 1. 重复检测 (40%)
        repetition_score = self._check_repetition(response)
        details = {'repetition_score': round(repetition_score, 4)}

        # 2. 长度合理性 (20%)
        length_score = self._check_length_reasonableness(len(words))
        details['length_score'] = round(length_score, 4)
        details['word_count'] = len(words)

        # 3. 句式多样性 (20%)
        variety_score = self._check_sentence_variety(response)
        details['variety_score'] = round(variety_score, 4)

        # 4. 语法检查 (20%)
        grammar_score = self._check_basic_grammar(response)
        details['grammar_score'] = round(grammar_score, 4)

        # 加权总分
        fluency = (
            0.40 * repetition_score
            + 0.20 * length_score
            + 0.20 * variety_score
            + 0.20 * grammar_score
        )
        return round(fluency, 4), details

    def _check_repetition(self, text: str) -> float:
        """检测文本中的n-gram重复

        返回 1.0 (无重复) 到 0.0 (严重重复)
        """
        words = re.findall(r'\w+', text.lower())
        if len(words) < self.REPETITION_NGRAM:
            return 1.0

        # 生成n-grams
        ngrams = []
        for i in range(len(words) - self.REPETITION_NGRAM + 1):
            ngram = tuple(words[i:i + self.REPETITION_NGRAM])
            ngrams.append(ngram)

        # 计算重复率
        ngram_counts = Counter(ngrams)
        unique_count = len(ngram_counts)
        total_count = len(ngrams)

        if total_count == 0:
            return 1.0

        # 重复n-gram的比率
        repeated_ratio = 1.0 - (unique_count / total_count)

        # 转换为得分 (重复越多分数越低)
        if repeated_ratio <= 0.05:
            return 1.0
        elif repeated_ratio <= self.REPETITION_MAX_RATIO:
            return 1.0 - (repeated_ratio / self.REPETITION_MAX_RATIO) * 0.5
        else:
            return max(0.0, 1.0 - repeated_ratio)

    def _check_length_reasonableness(self, word_count: int) -> float:
        """检查回答长度的合理性"""
        if word_count < self.IDEAL_LENGTH_MIN:
            # 太短
            return max(0.0, word_count / self.IDEAL_LENGTH_MIN)
        elif word_count <= self.IDEAL_LENGTH_MAX:
            # 理想长度
            return 1.0
        elif word_count <= self.MAX_LENGTH_FOR_SCORE:
            # 偏长但可接受
            ratio = (word_count - self.IDEAL_LENGTH_MAX) / (self.MAX_LENGTH_FOR_SCORE - self.IDEAL_LENGTH_MAX)
            return 1.0 - ratio * 0.3
        else:
            # 太长
            return max(0.0, 1.0 - (word_count - self.MAX_LENGTH_FOR_SCORE) / self.MAX_LENGTH_FOR_SCORE)

    def _check_sentence_variety(self, text: str) -> float:
        """检查句式多样性

        基于不同句长的标准差
        """
        sentences = re.split(r'[.!?。！？]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            return 0.5  # 只有一个句子，中等得分

        sent_lengths = [len(s.split()) for s in sentences]

        if len(sent_lengths) >= 2:
            try:
                std_dev = statistics.stdev(sent_lengths)
            except statistics.StatisticsError:
                std_dev = 0

            mean_len = statistics.mean(sent_lengths)

            # 变异系数 (CV = std/mean)，CV在0.3-0.8之间为最佳
            if mean_len > 0:
                cv = std_dev / mean_len
                if 0.3 <= cv <= 0.8:
                    return 1.0
                elif cv < 0.3:
                    return cv / 0.3
                else:
                    return max(0.0, 1.0 - (cv - 0.8) * 0.5)
            else:
                return 0.0
        return 0.5

    def _check_basic_grammar(self, text: str) -> float:
        """基本语法检查 (启发式)

        检查项:
          - 括号匹配
          - 引号匹配
          - 连续标点
          - 大写开头
          - 句末标点
        """
        if not text:
            return 0.0

        score = 1.0
        penalties = 0
        checks = 0

        # 1. 括号匹配
        checks += 1
        open_brackets = text.count('(') + text.count('[') + text.count('{')
        close_brackets = text.count(')') + text.count(']') + text.count('}')
        if open_brackets != close_brackets:
            penalties += 1

        # 2. 连续标点
        checks += 1
        if re.search(r'[.!?。！？]{3,}', text):
            penalties += 0.5
        if re.search(r'[,，]{2,}', text):
            penalties += 0.5

        # 3. 首字母大写 (英文)
        checks += 1
        if re.match(r'^[A-Z]', text.strip()):
            pass  # 正确
        else:
            # 非惩罚：可能以数字或符号开头
            pass

        # 4. 句末有适当标点
        checks += 1
        if re.search(r'[.!?。！？]$', text.strip()):
            pass
        else:
            penalties += 0.5

        if checks > 0:
            score = max(0.0, 1.0 - (penalties / checks))
        return score

    # ========================================================================
    # 5. Latency (延迟)
    # ========================================================================

    def evaluate_latency(self, latency_seconds: float) -> Tuple[float, Dict[str, Any]]:
        """评估延迟

        归一化规则:
          - 0s-1s:   得分 1.0
          - 1s-3s:   得分 0.8-1.0
          - 3s-5s:   得分 0.5-0.8
          - 5s-10s:  得分 0.0-0.5
          - ≥10s:    得分 0.0

        Args:
            latency_seconds: 响应时间(秒)

        Returns:
            (延迟得分(0-1), 详细信息字典)
        """
        if latency_seconds <= 0:
            score = 1.0
        elif latency_seconds <= 1.0:
            score = 1.0
        elif latency_seconds <= 3.0:
            # 线性插值: 1s→1.0, 3s→0.8
            score = 1.0 - (latency_seconds - 1.0) / (3.0 - 1.0) * 0.2
        elif latency_seconds <= 5.0:
            # 线性插值: 3s→0.8, 5s→0.5
            score = 0.8 - (latency_seconds - 3.0) / (5.0 - 3.0) * 0.3
        elif latency_seconds <= 10.0:
            # 线性插值: 5s→0.5, 10s→0.0
            score = 0.5 - (latency_seconds - 5.0) / (10.0 - 5.0) * 0.5
        else:
            score = 0.0

        score = round(max(0.0, min(1.0, score)), 4)

        details = {
            'latency_seconds': latency_seconds,
            'score': score,
            'category': self._latency_category(latency_seconds),
        }
        return score, details

    def _latency_category(self, seconds: float) -> str:
        """延迟分类"""
        if seconds <= 1.0:
            return 'excellent'
        elif seconds <= 3.0:
            return 'good'
        elif seconds <= 5.0:
            return 'acceptable'
        elif seconds <= 10.0:
            return 'slow'
        else:
            return 'unacceptable'

    # ========================================================================
    # 6. Cost (成本)
    # ========================================================================

    def evaluate_cost(self, cost_dollars: float) -> Tuple[float, Dict[str, Any]]:
        """评估成本

        归一化规则:
          - $0.00-$0.01:  得分 1.0
          - $0.01-$0.03:  得分 0.8-1.0
          - $0.03-$0.06:  得分 0.4-0.8
          - $0.06-$0.10:  得分 0.0-0.4
          - ≥$0.10:        得分 0.0

        Args:
            cost_dollars: 单次请求成本(美元)

        Returns:
            (成本得分(0-1), 详细信息字典)
        """
        if cost_dollars <= 0:
            score = 1.0
        elif cost_dollars <= 0.01:
            score = 1.0
        elif cost_dollars <= 0.03:
            score = 1.0 - (cost_dollars - 0.01) / (0.03 - 0.01) * 0.2
        elif cost_dollars <= 0.06:
            score = 0.8 - (cost_dollars - 0.03) / (0.06 - 0.03) * 0.4
        elif cost_dollars <= 0.10:
            score = 0.4 - (cost_dollars - 0.06) / (0.10 - 0.06) * 0.4
        else:
            score = 0.0

        score = round(max(0.0, min(1.0, score)), 4)

        details = {
            'cost_dollars': cost_dollars,
            'score': score,
            'category': self._cost_category(cost_dollars),
        }
        return score, details

    def _cost_category(self, dollars: float) -> str:
        """成本分类"""
        if dollars <= 0.01:
            return 'cheap'
        elif dollars <= 0.03:
            return 'reasonable'
        elif dollars <= 0.06:
            return 'moderate'
        elif dollars <= 0.10:
            return 'expensive'
        else:
            return 'prohibitive'

    # ========================================================================
    # 综合评估
    # ========================================================================

    def evaluate(
        self,
        query: str,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        key_facts: Optional[List[str]] = None,
        latency_seconds: float = 0.0,
        cost_dollars: float = 0.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> MultiDimScore:
        """执行完整的六维度评估

        Args:
            query: 用户查询
            response: 系统回答
            reference: 参考答案 (用于准确度评估)
            context: 检索上下文 (用于忠实度评估)
            key_facts: 关键事实列表
            latency_seconds: 响应延迟
            cost_dollars: 请求成本
            weights: 自定义权重

        Returns:
            MultiDimScore 包含所有评估结果
        """
        score = MultiDimScore(latency_seconds=latency_seconds, cost_dollars=cost_dollars)

        # 1. Accuracy
        if reference:
            score.accuracy, detail = self.evaluate_accuracy(response, reference, key_facts)
            score.dimension_details['accuracy'] = detail
        else:
            score.accuracy = 1.0  # 无参考答案时跳过
            score.dimension_details['accuracy'] = {'skipped': '无参考答案'}

        # 2. Relevance
        score.relevance, detail = self.evaluate_relevance(response, query, context)
        score.dimension_details['relevance'] = detail

        # 3. Faithfulness
        if context:
            score.faithfulness, detail = self.evaluate_faithfulness(response, context)
            score.dimension_details['faithfulness'] = detail
        else:
            score.faithfulness = 1.0
            score.dimension_details['faithfulness'] = {'skipped': '无上下文'}

        # 4. Fluency
        score.fluency, detail = self.evaluate_fluency(response)
        score.dimension_details['fluency'] = detail

        # 5. Latency
        score.latency_score, detail = self.evaluate_latency(latency_seconds)
        score.dimension_details['latency'] = detail

        # 6. Cost
        score.cost_score, detail = self.evaluate_cost(cost_dollars)
        score.dimension_details['cost'] = detail

        return score


# ============================================================================
# 评估结果报告
# ============================================================================

class EvaluationReporter:
    """评估结果报告生成器"""

    @staticmethod
    def generate_report(scores: List[MultiDimScore], name: str = "Evaluation Report") -> str:
        """生成评估报告

        Args:
            scores: 多个MultiDimScore的列表
            name: 报告名称

        Returns:
            格式化的报告字符串
        """
        if not scores:
            return "无评估数据"

        lines = [f"{'='*60}", f"  {name}", f"{'='*60}", ""]

        # 汇总统计
        n = len(scores)
        avg = MultiDimScore()
        avg.accuracy = statistics.mean(s.accuracy for s in scores)
        avg.relevance = statistics.mean(s.relevance for s in scores)
        avg.faithfulness = statistics.mean(s.faithfulness for s in scores)
        avg.fluency = statistics.mean(s.fluency for s in scores)
        avg.latency_score = statistics.mean(s.latency_score for s in scores)
        avg.cost_score = statistics.mean(s.cost_score for s in scores)
        avg.latency_seconds = statistics.mean(s.latency_seconds for s in scores)
        avg.cost_dollars = statistics.mean(s.cost_dollars for s in scores)

        lines.append("📊 汇总统计 (Aggregate Statistics)")
        lines.append("-" * 40)
        lines.append(f"  样本数: {n}")
        lines.append(f"  加权总分 (平均): {avg.weighted_score():.3f}")
        lines.append(f"")
        lines.append(f"  维度得分 (平均):")
        lines.append(f"    Accuracy:      {avg.accuracy:.3f}  ±{EvaluationReporter._std(scores, 'accuracy'):.3f}")
        lines.append(f"    Relevance:     {avg.relevance:.3f}  ±{EvaluationReporter._std(scores, 'relevance'):.3f}")
        lines.append(f"    Faithfulness:  {avg.faithfulness:.3f}  ±{EvaluationReporter._std(scores, 'faithfulness'):.3f}")
        lines.append(f"    Fluency:       {avg.fluency:.3f}  ±{EvaluationReporter._std(scores, 'fluency'):.3f}")
        lines.append(f"    Latency:       {avg.latency_score:.3f}  ±{EvaluationReporter._std(scores, 'latency_score'):.3f}")
        lines.append(f"    Cost:          {avg.cost_score:.3f}  ±{EvaluationReporter._std(scores, 'cost_score'):.3f}")
        lines.append("")
        lines.append(f"  原始指标 (平均):")
        lines.append(f"    延迟: {avg.latency_seconds:.2f}s")
        lines.append(f"    成本: ${avg.cost_dollars:.4f}")
        lines.append("")

        # 等级评定
        final_score = avg.weighted_score()
        grade = EvaluationReporter._calculate_grade(final_score)
        lines.append(f"🏆 综合等级: {grade} (总分: {final_score:.3f})")
        lines.append("")

        # 各维度详细分析
        lines.append("📋 各维度分析")
        lines.append("-" * 40)

        dimensions = [
            ('Accuracy', [s.accuracy for s in scores]),
            ('Relevance', [s.relevance for s in scores]),
            ('Faithfulness', [s.faithfulness for s in scores]),
            ('Fluency', [s.fluency for s in scores]),
            ('Latency', [s.latency_score for s in scores]),
            ('Cost', [s.cost_score for s in scores]),
        ]

        for name, values in dimensions:
            mean_val = statistics.mean(values)
            grade = EvaluationReporter._dimension_grade(mean_val)
            lines.append(f"  {name:15s}: {mean_val:.3f} [{grade}]")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def _std(scores: List[MultiDimScore], attr: str) -> float:
        """计算标准差"""
        values = [getattr(s, attr) for s in scores]
        if len(values) >= 2:
            try:
                return statistics.stdev(values)
            except statistics.StatisticsError:
                return 0.0
        return 0.0

    @staticmethod
    def _calculate_grade(score: float) -> str:
        """计算综合等级"""
        if score >= 0.90:
            return 'A+ (优秀)'
        elif score >= 0.85:
            return 'A (优秀)'
        elif score >= 0.80:
            return 'A- (良好)'
        elif score >= 0.75:
            return 'B+ (良好)'
        elif score >= 0.70:
            return 'B (合格)'
        elif score >= 0.60:
            return 'C (需改进)'
        elif score >= 0.50:
            return 'D (较差)'
        else:
            return 'F (不合格)'

    @staticmethod
    def _dimension_grade(score: float) -> str:
        """单维度等级"""
        if score >= 0.85:
            return '优秀'
        elif score >= 0.75:
            return '良好'
        elif score >= 0.60:
            return '合格'
        else:
            return '需改进'


# ============================================================================
# 定时评估器 (用于性能测试)
# ============================================================================

class TimedEvaluation:
    """带计时的评估包装器"""

    def __init__(self, evaluator: MultiDimensionEvaluator):
        self.evaluator = evaluator

    def evaluate_with_timing(
        self,
        query: str,
        response_fn: callable,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        key_facts: Optional[List[str]] = None,
    ) -> MultiDimScore:
        """执行评估并记录实际延迟

        注意: 成本需要外部计算
        """
        start = time.perf_counter()
        response = response_fn(query)
        end = time.perf_counter()

        actual_latency = end - start
        # 简单成本估算: 假设每token $0.00001, 平均50 tokens
        estimated_tokens = len(response.split()) * 1.3  # 粗略估计
        estimated_cost = estimated_tokens * 0.00001

        return self.evaluator.evaluate(
            query=query,
            response=response,
            reference=reference,
            context=context,
            key_facts=key_facts,
            latency_seconds=actual_latency,
            cost_dollars=estimated_cost,
        )


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  六维度评估框架 - 完整演示")
    print("=" * 60)
    print()

    # 初始化评估器
    evaluator = MultiDimensionEvaluator()

    # ---- 测试数据 ----
    test_cases = [
        {
            'query': '什么是RAG？',
            'response': 'RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的AI技术。'
                       '它先从知识库中检索相关文档，然后将这些文档作为上下文提供给大语言模型来生成更准确的回答。'
                       'RAG可以有效减少幻觉现象，提高回答的事实准确性。',
            'reference': 'RAG（Retrieval-Augmented Generation）是一种AI架构，'
                        '通过先从外部知识库检索相关信息，然后将检索结果注入到大语言模型的提示中，'
                        '以生成更准确、更可靠的回答。它结合了检索系统和生成模型的优势。',
            'key_facts': [
                'RAG是检索增强生成',
                'RAG结合检索和生成',
                'RAG减少幻觉',
                'RAG从知识库检索文档',
                'RAG提高事实准确性',
            ],
            'context': 'RAG (Retrieval-Augmented Generation) 是一种结合信息检索和文本生成的AI技术。'
                      '它从知识库中检索相关文档，然后将文档作为上下文提供给LLM。'
                      '这种方法可以有效减少模型的幻觉现象，提高回答的可靠性。',
            'latency': 1.8,
            'cost': 0.005,
        },
        {
            'query': '深度学习如何工作？',
            'response': '深度学习通过多层神经网络来学习数据的层次化表示。嗯...它使用反向传播算法来调整权重。'
                       '深度学习需要大量数据来训练。通过多层神经网络来学习数据的层次化表示。'
                       '它确实使用反向传播算法来调整权重。深度学习确实需要大量数据来训练。'
                       '深度学习就是深度学习，很强大的技术。非常强大。总之深度学习很厉害。',
            'reference': '深度学习是机器学习的一个子集，使用多层人工神经网络从大量数据中学习表示。'
                        '它通过反向传播算法进行训练，自动发现数据中的模式和特征。'
                        '关键组件包括激活函数、损失函数和优化器。',
            'key_facts': [
                '深度学习使用多层神经网络',
                '深度学习使用反向传播',
                '深度学习需要大量数据',
            ],
            'context': '深度学习使用多层神经网络。训练过程使用反向传播算法。数据量对模型效果有重要影响。',
            'latency': 4.5,
            'cost': 0.025,
        },
        {
            'query': 'Python的优点是什么？',
            'response': 'Python易于学习，语法简洁。它有丰富的库生态系统。Python是解释型语言，'
                       '跨平台兼容性好，社区活跃。在数据科学、Web开发和AI领域都有广泛应用。',
            'reference': 'Python是一种高级编程语言，以其简洁易读的语法而闻名。'
                        '主要优点包括：简单易学、丰富的第三方库、跨平台兼容、活跃的社区支持、'
                        '以及在数据科学和人工智能领域的广泛应用。',
            'key_facts': [
                'Python语法简洁',
                'Python易于学习',
                'Python库丰富',
                'Python跨平台',
                'Python社区活跃',
            ],
            'context': 'Python is known for its simplicity and readability. It has a vast ecosystem of libraries. '
                      'Python runs on multiple platforms and has an active community.',
            'latency': 1.2,
            'cost': 0.003,
        },
    ]

    # ---- 评估所有测试用例 ----
    all_scores: List[MultiDimScore] = []

    for i, case in enumerate(test_cases):
        print(f"{'─' * 60}")
        print(f"测试用例 {i + 1}: {case['query']}")
        print(f"{'─' * 60}")

        score = evaluator.evaluate(
            query=case['query'],
            response=case['response'],
            reference=case.get('reference'),
            context=case.get('context'),
            key_facts=case.get('key_facts'),
            latency_seconds=case.get('latency', 0.0),
            cost_dollars=case.get('cost', 0.0),
        )

        all_scores.append(score)
        print(score)
        print()

    # ---- 生成汇总报告 ----
    print()
    report = EvaluationReporter.generate_report(all_scores, "RAG系统六维度评估报告")
    print(report)

    # ---- 自定义权重示例 ----
    print()
    print("─" * 60)
    print("自定义权重示例 (侧重准确度和忠实度)")
    print("─" * 60)

    custom_weights = {
        'accuracy': 0.40,      # 提高准确度权重
        'relevance': 0.10,
        'faithfulness': 0.35,  # 提高忠实度权重
        'fluency': 0.05,
        'latency': 0.05,
        'cost': 0.05,
    }

    for i, score in enumerate(all_scores):
        standard = score.weighted_score()
        custom = score.weighted_score(custom_weights)
        print(f"  用例{i+1}: 标准权重={standard:.3f}, 自定义权重={custom:.3f}, "
              f"差异={custom - standard:+.3f}")

    # ---- 导出功能演示 ----
    print()
    print("─" * 60)
    print("JSON导出示例 (第一个测试用例)")
    print("─" * 60)

    import json
    json_output = all_scores[0].to_dict()
    print(json.dumps(json_output, ensure_ascii=False, indent=2))

    print()
    print("=" * 60)
    print("  演示完成 - 六维度评估框架就绪")
    print("=" * 60)
