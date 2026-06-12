#!/usr/bin/env python3
"""
金标准数据集构建器 (Golden Dataset Builder)
============================================
三种数据来源构建评估数据集:
  1. 人工标注 (Human Annotation) - 50+条
  2. 合成生成 (Synthetic Generation) - 从文档LLM生成，每文档5个问题
  3. 生产日志挖掘 (Production Log Mining) - 正面反馈过滤

难度分布:
  - Easy (简单): 30%   - 直接事实查询
  - Medium (中等): 40% - 需要推理
  - Hard (困难): 20%   - 多跳/复杂推理
  - Adversarial (对抗): 10% - 刻意构造的挑战性查询

对抗样本生成: 5种变换
  1. 同义词替换 (Synonym)
  2. 误导信息 (Misleading)
  3. 口语化 (Colloquial)
  4. 多问题 (Multi-question)
  5. 拼写错误 (Typos)
"""

import json
import random
import hashlib
import re
import os
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

# 可选依赖
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# 数据结构
# ============================================================================

class DifficultyLevel(Enum):
    """难度级别"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"

    @classmethod
    def distribution(cls) -> Dict[str, float]:
        """标准难度分布"""
        return {
            cls.EASY.value: 0.30,
            cls.MEDIUM.value: 0.40,
            cls.HARD.value: 0.20,
            cls.ADVERSARIAL.value: 0.10,
        }


class DataSource(Enum):
    """数据来源"""
    HUMAN_ANNOTATION = "human_annotation"
    SYNTHETIC = "synthetic"
    PRODUCTION_LOG = "production_log"


@dataclass
class GoldenQuery:
    """金标准查询条目"""
    query_id: str
    query: str
    reference_answer: str
    key_facts: List[str]
    relevant_doc_ids: List[str]
    difficulty: DifficultyLevel
    source: DataSource
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 可选字段
    context: Optional[str] = None
    category: Optional[str] = None
    expected_latency_s: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'query_id': self.query_id,
            'query': self.query,
            'reference_answer': self.reference_answer,
            'key_facts': self.key_facts,
            'relevant_doc_ids': self.relevant_doc_ids,
            'difficulty': self.difficulty.value,
            'source': self.source.value,
            'metadata': self.metadata,
            'context': self.context,
            'category': self.category,
            'tags': self.tags,
        }


@dataclass
class GoldenDataset:
    """金标准数据集"""
    name: str
    version: str = "1.0.0"
    queries: List[GoldenQuery] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    statistics: Dict[str, Any] = field(default_factory=dict)

    def size(self) -> int:
        """数据集大小"""
        return len(self.queries)

    def difficulty_breakdown(self) -> Dict[str, int]:
        """各难度级别的数量分布"""
        breakdown = defaultdict(int)
        for q in self.queries:
            breakdown[q.difficulty.value] += 1
        return dict(breakdown)

    def source_breakdown(self) -> Dict[str, int]:
        """各来源的数量分布"""
        breakdown = defaultdict(int)
        for q in self.queries:
            breakdown[q.source.value] += 1
        return dict(breakdown)

    def validate(self) -> Tuple[bool, List[str]]:
        """验证数据集完整性

        Returns:
            (是否通过, 问题列表)
        """
        issues = []

        # 检查是否有查询
        if not self.queries:
            issues.append("数据集为空")

        # 检查ID唯一性
        ids = [q.query_id for q in self.queries]
        if len(ids) != len(set(ids)):
            issues.append("存在重复的query_id")

        # 检查必填字段
        for q in self.queries:
            if not q.query:
                issues.append(f"{q.query_id}: query为空")
            if not q.reference_answer:
                issues.append(f"{q.query_id}: reference_answer为空")
            if not q.key_facts:
                issues.append(f"{q.query_id}: key_facts为空")

        # 检查难度分布
        breakdown = self.difficulty_breakdown()
        expected_dist = DifficultyLevel.distribution()
        total = self.size()

        if total > 0:
            for level, expected_ratio in expected_dist.items():
                actual_ratio = breakdown.get(level, 0) / total
                # 允许±20%的偏差
                if abs(actual_ratio - expected_ratio) > 0.20:
                    issues.append(
                        f"{level}的比例为{actual_ratio:.1%}，"
                        f"期望{expected_ratio:.1%} (偏差>20%)"
                    )

        return len(issues) == 0, issues

    def save(self, filepath: str) -> None:
        """保存数据集到JSON文件"""
        data = {
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at,
            'description': self.description,
            'total_queries': self.size(),
            'statistics': self.get_statistics(),
            'queries': [q.to_dict() for q in self.queries],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[保存] 数据集已保存到: {filepath} ({self.size()} 条查询)")

    @classmethod
    def load(cls, filepath: str) -> 'GoldenDataset':
        """从JSON文件加载数据集"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = cls(
            name=data['name'],
            version=data.get('version', '1.0.0'),
            created_at=data.get('created_at', ''),
            description=data.get('description', ''),
        )

        for q_data in data['queries']:
            query = GoldenQuery(
                query_id=q_data['query_id'],
                query=q_data['query'],
                reference_answer=q_data['reference_answer'],
                key_facts=q_data['key_facts'],
                relevant_doc_ids=q_data['relevant_doc_ids'],
                difficulty=DifficultyLevel(q_data['difficulty']),
                source=DataSource(q_data['source']),
                metadata=q_data.get('metadata', {}),
                context=q_data.get('context'),
                category=q_data.get('category'),
                tags=q_data.get('tags', []),
            )
            dataset.queries.append(query)

        return dataset

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'total_queries': self.size(),
            'difficulty_breakdown': self.difficulty_breakdown(),
            'source_breakdown': self.source_breakdown(),
            'avg_query_length': round(
                sum(len(q.query) for q in self.queries) / max(1, self.size()), 1
            ),
            'avg_reference_length': round(
                sum(len(q.reference_answer) for q in self.queries) / max(1, self.size()), 1
            ),
            'categories': self._get_categories(),
        }

    def _get_categories(self) -> Dict[str, int]:
        """获取类别分布"""
        cats = defaultdict(int)
        for q in self.queries:
            if q.category:
                cats[q.category] += 1
        return dict(cats)

    def get_subset(
        self,
        difficulty: Optional[DifficultyLevel] = None,
        source: Optional[DataSource] = None,
        n: Optional[int] = None,
    ) -> 'GoldenDataset':
        """获取数据集的子集"""
        subset = [q for q in self.queries]

        if difficulty:
            subset = [q for q in subset if q.difficulty == difficulty]
        if source:
            subset = [q for q in subset if q.source == source]
        if n and n < len(subset):
            random.shuffle(subset)
            subset = subset[:n]

        result = GoldenDataset(
            name=f"{self.name}_subset",
            version=self.version,
            description=f"Subset of {self.name}",
        )
        result.queries = subset
        return result


# ============================================================================
# 金标准数据集构建器
# ============================================================================

class GoldenDatasetBuilder:
    """金标准数据集构建器

    三种来源:
      1. human_annotation - 人工标注
      2. synthetic - LLM从文档自动生成
      3. production_log - 从生产日志中挖掘
    """

    # 难度分类的特征关键词
    EASY_KEYWORDS = ['什么', '什么是', '谁', '哪里', '什么时候', 'what', 'who', 'where', 'when', '定义', 'define']
    MEDIUM_KEYWORDS = ['为什么', '如何', '怎样', '比较', '区别', 'how', 'why', 'compare', 'difference', 'explain']
    HARD_KEYWORDS = ['分析', '评估', '综合', '关系', '影响', '预测', 'analyze', 'evaluate', 'synthesize', 'impact', 'relationship']

    def __init__(
        self,
        dataset_name: str = "golden_dataset",
        llm_generator: Optional[callable] = None,
        seed: int = 42,
    ):
        """初始化构建器

        Args:
            dataset_name: 数据集名称
            llm_generator: LLM生成函数 (可选)，接收(prompt)返回(response)
            seed: 随机种子
        """
        self.dataset_name = dataset_name
        self.llm_generator = llm_generator
        self.seed = seed
        self.queries: List[GoldenQuery] = []
        self._counter = 0

        random.seed(seed)

    def _generate_id(self, prefix: str = "GQ") -> str:
        """生成唯一的查询ID"""
        self._counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_suffix = hashlib.md5(f"{timestamp}_{self._counter}".encode()).hexdigest()[:6]
        return f"{prefix}_{timestamp}_{hash_suffix}"

    # ========================================================================
    # 来源1: 人工标注
    # ========================================================================

    def add_human_annotations(
        self,
        annotations: List[Dict[str, Any]],
        auto_classify_difficulty: bool = True,
    ) -> List[GoldenQuery]:
        """添加人工标注的查询

        预期annotation格式:
        {
            'query': str,
            'reference_answer': str,
            'key_facts': List[str],
            'relevant_doc_ids': List[str],
            'difficulty': Optional[str],  # 'easy'|'medium'|'hard'|'adversarial'
            'category': Optional[str],
            'tags': Optional[List[str]],
        }

        Args:
            annotations: 标注列表
            auto_classify_difficulty: 是否自动分类难度

        Returns:
            添加的GoldenQuery列表
        """
        added = []

        for ann in annotations:
            query_id = self._generate_id("HUMAN")

            # 自动判断难度
            difficulty_str = ann.get('difficulty')
            if auto_classify_difficulty and not difficulty_str:
                difficulty_str = self._auto_classify_difficulty(ann['query'])

            difficulty = DifficultyLevel(difficulty_str or 'medium')

            golden_query = GoldenQuery(
                query_id=query_id,
                query=ann['query'],
                reference_answer=ann['reference_answer'],
                key_facts=ann.get('key_facts', []),
                relevant_doc_ids=ann.get('relevant_doc_ids', []),
                difficulty=difficulty,
                source=DataSource.HUMAN_ANNOTATION,
                category=ann.get('category'),
                tags=ann.get('tags', []),
                metadata={'annotator': ann.get('annotator', 'unknown'), 'annotation_date': ann.get('date')},
            )
            self.queries.append(golden_query)
            added.append(golden_query)

        print(f"[人类标注] 添加了 {len(added)} 条标注查询")
        return added

    def _auto_classify_difficulty(self, query: str) -> str:
        """根据查询内容自动分类难度

        Args:
            query: 查询文本

        Returns:
            难度级别字符串
        """
        query_lower = query.lower()
        query_len = len(query)

        # 计算各类关键词的匹配数
        easy_score = sum(1 for kw in self.EASY_KEYWORDS if kw in query_lower)
        medium_score = sum(1 for kw in self.MEDIUM_KEYWORDS if kw in query_lower)
        hard_score = sum(1 for kw in self.HARD_KEYWORDS if kw in query_lower)

        # 长度因素
        if query_len > 100:
            hard_score += 2
        elif query_len > 50:
            medium_score += 1

        # 问题数量
        question_marks = query.count('?') + query.count('？')
        if question_marks > 2:
            hard_score += 2
        elif question_marks > 1:
            medium_score += 1

        # 判断
        if hard_score > medium_score and hard_score > easy_score:
            return 'hard'
        elif medium_score > easy_score:
            return 'medium'
        else:
            return 'easy'

    # ========================================================================
    # 来源2: 合成生成
    # ========================================================================

    def generate_synthetic(
        self,
        documents: List[Dict[str, Any]],
        questions_per_doc: int = 5,
        difficulty_distribution: Optional[Dict[str, int]] = None,
    ) -> List[GoldenQuery]:
        """从文档使用LLM自动生成问题

        Args:
            documents: 文档列表，每个包含:
                {
                    'doc_id': str,
                    'title': str,
                    'content': str,
                    'category': Optional[str],
                }
            questions_per_doc: 每篇文档生成的问题数
            difficulty_distribution: 难度分布 (如 {'easy': 2, 'medium': 2, 'hard': 1})

        Returns:
            生成的GoldenQuery列表
        """
        if not documents:
            print("[警告] 文档列表为空，无法生成合成数据")
            return []

        if difficulty_distribution is None:
            difficulty_distribution = {
                'easy': max(1, int(questions_per_doc * 0.3)),
                'medium': max(1, int(questions_per_doc * 0.4)),
                'hard': max(1, int(questions_per_doc * 0.2)),
                'adversarial': max(0, int(questions_per_doc * 0.1)),
            }

        # 确保总数正确
        total_assigned = sum(difficulty_distribution.values())
        if total_assigned < questions_per_doc:
            difficulty_distribution['medium'] += (questions_per_doc - total_assigned)
        elif total_assigned > questions_per_doc:
            # 从高到低减少
            for level in ['adversarial', 'hard', 'medium']:
                excess = sum(difficulty_distribution.values()) - questions_per_doc
                if excess <= 0:
                    break
                reduce = min(excess, difficulty_distribution.get(level, 0))
                difficulty_distribution[level] -= reduce

        generated = []
        for doc in documents:
            doc_id = doc['doc_id']
            doc_title = doc.get('title', 'Untitled')
            doc_content = doc['content']
            doc_category = doc.get('category')

            for difficulty, count in difficulty_distribution.items():
                for _ in range(count):
                    questions = self._generate_questions_for_doc(
                        doc_title, doc_content, difficulty, 1
                    )
                    for q_data in questions:
                        query_id = self._generate_id("SYN")
                        golden_query = GoldenQuery(
                            query_id=query_id,
                            query=q_data['query'],
                            reference_answer=q_data['answer'],
                            key_facts=q_data.get('key_facts', []),
                            relevant_doc_ids=[doc_id],
                            difficulty=DifficultyLevel(difficulty),
                            source=DataSource.SYNTHETIC,
                            category=doc_category,
                            tags=['synthetic', difficulty],
                            context=doc_content[:500],  # 前500字符作为上下文
                            metadata={
                                'source_doc_id': doc_id,
                                'source_doc_title': doc_title,
                                'generation_method': 'llm' if self.llm_generator else 'template',
                            },
                        )
                        self.queries.append(golden_query)
                        generated.append(golden_query)

        print(f"[合成生成] 从 {len(documents)} 篇文档生成了 {len(generated)} 条查询")
        return generated

    def _generate_questions_for_doc(
        self,
        title: str,
        content: str,
        difficulty: str,
        count: int,
    ) -> List[Dict[str, Any]]:
        """为文档生成指定难度的问题

        Args:
            title: 文档标题
            content: 文档内容
            difficulty: 难度级别
            count: 生成数量

        Returns:
            问题字典列表 [{'query': ..., 'answer': ..., 'key_facts': [...]}]
        """
        if self.llm_generator:
            return self._generate_with_llm(title, content, difficulty, count)
        else:
            return self._generate_with_template(title, content, difficulty, count)

    def _generate_with_llm(
        self,
        title: str,
        content: str,
        difficulty: str,
        count: int,
    ) -> List[Dict[str, Any]]:
        """使用LLM生成问题"""
        prompt = f"""基于以下文档内容生成{count}个{difficulty}难度的问题。

文档标题: {title}
文档内容: {content[:1500]}

难度要求:
- easy: 直接的事实查询，答案在文档中明确给出
- medium: 需要一定推理或总结的问题
- hard: 多步推理、综合多个位置的信息、或分析性问题
- adversarial: 故意具有挑战性的问题

请以JSON格式返回，每个问题包含:
{{
  "query": "问题文本",
  "answer": "参考答案",
  "key_facts": ["关键事实1", "关键事实2"]
}}

返回JSON数组。
"""
        try:
            response = self.llm_generator(prompt)
            # 尝试解析JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            print(f"[警告] LLM生成失败: {e}，使用模板生成")

        # 回退
        return self._generate_with_template(title, content, difficulty, count)

    def _generate_with_template(
        self,
        title: str,
        content: str,
        difficulty: str,
        count: int,
    ) -> List[Dict[str, Any]]:
        """使用模板生成问题 (无需LLM)

        基于关键信息提取和模板填充
        """
        results = []

        # 提取文档中的关键句子和实体
        sentences = [s.strip() for s in re.split(r'[.!?。！？]+', content) if len(s.strip()) > 15]
        important_sentences = sentences[:min(len(sentences), 10)]

        templates = {
            'easy': [
                ("根据文档，{title}是什么？", lambda s: f"根据文本内容，{title}是{s[:80]}"),
                ("{title}的定义是什么？", lambda s: f"{title}定义为{s[:80]}"),
                ("文档中提到了哪些关键概念？", lambda s: f"文档提到的关键概念包括: {s}"),
            ],
            'medium': [
                ("为什么{title}很重要？", lambda s: f"{title}很重要，因为{s[:100]}"),
                ("{title}有什么特点？", lambda s: f"{title}的特点包括: {s[:100]}"),
                ("如何理解文档中关于{title[:20]}的描述？", lambda s: f"文档中关于{title[:20]}的描述表明{s[:100]}"),
            ],
            'hard': [
                ("{title}与其他概念的关系是什么？", lambda s: f"根据文档，{title}与其他概念的关系在于{s[:100]}"),
                ("分析文档中{title[:20]}的含义及其影响。", lambda s: f"分析显示，{title[:20]}的含义是{s[:100]}"),
                ("从文档中推导关于{title[:15]}的两个结论。", lambda s: f"推导结论: 1) {s[:80]} 2) {s[80:160] if len(s) > 80 else '详见文档'}"),
            ],
        }

        templates_for_difficulty = templates.get(difficulty, templates['medium'])

        for i in range(min(count, len(templates_for_difficulty))):
            template_q, template_a = templates_for_difficulty[i]
            ref_sentence = important_sentences[i % len(important_sentences)] if important_sentences else content[:100]

            query = template_q.format(title=title)
            answer = template_a(ref_sentence)

            # 提取关键事实
            words = re.findall(r'\b[A-Z一-鿿][a-zA-Z一-鿿]{2,}\b', ref_sentence)
            key_facts = words[:3] if words else [ref_sentence[:50]]

            results.append({
                'query': query,
                'answer': answer,
                'key_facts': key_facts,
            })

        return results

    # ========================================================================
    # 来源3: 生产日志挖掘
    # ========================================================================

    def mine_production_logs(
        self,
        logs: List[Dict[str, Any]],
        positive_threshold: float = 0.7,
        max_per_batch: int = 100,
    ) -> List[GoldenQuery]:
        """从生产日志中挖掘高质量查询

        选择标准:
          - 正面反馈 (用户评分高、点击了"有帮助"、对话继续深入)
          - 非空回答
          - 非短查询 (≥10字符)

        Args:
            logs: 生产日志列表，每条包含:
                {
                    'query': str,
                    'response': str,
                    'context': Optional[str],
                    'feedback_score': Optional[float],  # 0-1
                    'user_clicked_helpful': Optional[bool],
                    'conversation_continued': Optional[bool],
                    'retrieved_doc_ids': Optional[List[str]],
                    'timestamp': Optional[str],
                }
            positive_threshold: 正面反馈阈值
            max_per_batch: 最多提取数量

        Returns:
            挖掘的GoldenQuery列表
        """
        mined = []

        for log in logs:
            if len(mined) >= max_per_batch:
                break

            # 过滤条件
            query = log.get('query', '')
            if len(query) < 10:
                continue

            response = log.get('response', '')
            if not response:
                continue

            # 正面反馈检查
            is_positive = self._is_positive_feedback(log, positive_threshold)
            if not is_positive:
                continue

            # 自动提取关键事实
            key_facts = self._extract_facts_from_response(response)

            # 自动判断难度
            difficulty = DifficultyLevel(self._auto_classify_difficulty(query))

            # 使用响应作为参考答案
            query_id = self._generate_id("PROD")
            golden_query = GoldenQuery(
                query_id=query_id,
                query=query,
                reference_answer=response,
                key_facts=key_facts,
                relevant_doc_ids=log.get('retrieved_doc_ids', []),
                difficulty=difficulty,
                source=DataSource.PRODUCTION_LOG,
                context=log.get('context'),
                metadata={
                    'original_timestamp': log.get('timestamp'),
                    'feedback_score': log.get('feedback_score'),
                    'mining_batch': datetime.now().isoformat(),
                },
            )
            self.queries.append(golden_query)
            mined.append(golden_query)

        print(f"[生产日志挖掘] 从 {len(logs)} 条日志中挖掘了 {len(mined)} 条高质查询")
        return mined

    def _is_positive_feedback(self, log: Dict[str, Any], threshold: float) -> bool:
        """检查是否为正面反馈"""
        feedback_score = log.get('feedback_score')
        if feedback_score is not None and feedback_score >= threshold:
            return True

        if log.get('user_clicked_helpful'):
            return True

        if log.get('conversation_continued'):
            return True

        return False

    def _extract_facts_from_response(self, response: str) -> List[str]:
        """从响应中提取关键事实"""
        facts = []
        sentences = re.split(r'[.!?。！？]+', response)
        for sent in sentences:
            sent = sent.strip()
            # 包含具体信息的长句子
            if len(sent) > 20 and (re.search(r'\d+', sent) or len(sent) > 50):
                facts.append(sent[:120])
        return facts[:5]

    # ========================================================================
    # 对抗样本生成
    # ========================================================================

    def generate_adversarial_variants(
        self,
        base_queries: List[GoldenQuery],
        count_per_query: int = 2,
    ) -> List[GoldenQuery]:
        """为已有查询生成对抗变体

        5种对抗变换:
          1. 同义词替换
          2. 误导性信息
          3. 口语化表达
          4. 多问题组合
          5. 拼写错误

        Args:
            base_queries: 基础查询列表
            count_per_query: 每个查询生成的变体数量

        Returns:
            对抗变体列表
        """
        transformations = [
            self._transform_synonym,
            self._transform_misleading,
            self._transform_colloquial,
            self._transform_multi_question,
            self._transform_typo,
        ]

        adversarial_queries = []

        for query in base_queries:
            # 随机选择count_per_query种变换
            selected_transforms = random.sample(
                transformations,
                min(count_per_query, len(transformations))
            )

            for transform_fn in selected_transforms:
                transformed_query = transform_fn(query.query)

                # 保持相同的答案
                query_id = self._generate_id("ADV")
                adv_query = GoldenQuery(
                    query_id=query_id,
                    query=transformed_query,
                    reference_answer=query.reference_answer,
                    key_facts=query.key_facts,
                    relevant_doc_ids=query.relevant_doc_ids,
                    difficulty=DifficultyLevel.ADVERSARIAL,
                    source=DataSource.SYNTHETIC,
                    category=query.category,
                    tags=['adversarial', transform_fn.__name__.replace('_transform_', '')],
                    context=query.context,
                    metadata={
                        'base_query_id': query.query_id,
                        'transformation': transform_fn.__name__,
                    },
                )
                self.queries.append(adv_query)
                adversarial_queries.append(adv_query)

        print(f"[对抗生成] 从 {len(base_queries)} 条查询生成了 {len(adversarial_queries)} 条对抗变体")
        return adversarial_queries

    def _transform_synonym(self, query: str) -> str:
        """变换1: 同义词替换

        使用同义词替换查询中的关键词
        """
        synonym_map = {
            '什么是': '解释一下',
            '如何': '怎样',
            '为什么': '是什么原因导致',
            '优点': '好处',
            '缺点': '不足',
            '影响': '作用',
            '方法': '方式',
            '重要': '关键',
            '使用': '利用',
            '区别': '差异',
        }

        result = query
        for original, replacement in synonym_map.items():
            if original in result:
                result = result.replace(original, replacement, 1)
                break

        if result == query:
            # 如果没有替换，添加同义词变体标记
            result = f"(换句话说) {query}"

        return result

    def _transform_misleading(self, query: str) -> str:
        """变换2: 误导性信息

        在查询中加入误导性前提
        """
        misleading_prefixes = [
            "我听说...（可能不对）",
            "有人告诉我...但我怀疑",
            "某个地方说...是真的吗？",
            "我可能记错了，但是",
            "不确定这对不对——",
        ]
        prefix = random.choice(misleading_prefixes)
        return f"{prefix} {query}"

    def _transform_colloquial(self, query: str) -> str:
        """变换3: 口语化表达

        将正式表达转为口语化
        """
        colloquial_markers = [
            "嘿，问一下哈：",
            "那个...我想问个问题：",
            "嗯...就是",
            "哥们儿，请教一下：",
            "话说，",
        ]
        suffix_markers = [
            "，谢谢啦！",
            "，麻烦你了~",
            "，求解答！",
            "，帮帮忙！",
        ]
        marker = random.choice(colloquial_markers)
        suffix = random.choice(suffix_markers)
        return f"{marker}{query}{suffix}"

    def _transform_multi_question(self, query: str) -> str:
        """变换4: 多问题组合

        在查询中附加额外的问题
        """
        extra_questions = [
            "另外，这个观点有争议吗？",
            "顺便问一下，这和之前提到的有什么关系？",
            "还有，你能举个例子吗？",
            "对了，这个的适用范围是什么？",
            "再问一个，有没有例外情况？",
        ]
        extra = random.choice(extra_questions)
        if query.endswith('?') or query.endswith('？'):
            return f"{query} {extra}"
        else:
            return f"{query}？{extra}"

    def _transform_typo(self, query: str) -> str:
        """变换5: 拼写错误

        故意引入常见拼写错误
        """
        if len(query) < 5:
            return query

        # 对中文: 使用常见错别字
        typo_map = {
            '的': '地',
            '在': '再',
            '什么': '啥么',
            '怎么': '肿么',
            '没有': '木有',
        }

        result = query
        for correct, wrong in typo_map.items():
            if correct in result:
                result = result.replace(correct, wrong, 1)
                break

        if result == query:
            # 对英文: 常见拼写错误
            english_typos = {
                'the': 'teh',
                'what': 'waht',
                'how': 'hwo',
                'where': 'wheer',
                'their': 'thier',
                'there': 'tehre',
                'define': 'defnie',
                'explain': 'expalin',
            }
            for correct, wrong in english_typos.items():
                pattern = re.compile(r'\b' + correct + r'\b', re.IGNORECASE)
                if pattern.search(result):
                    result = pattern.sub(wrong, result, count=1)
                    break

        return result

    # ========================================================================
    # 构建最终数据集
    # ========================================================================

    def build(
        self,
        target_size: int = 100,
        difficulty_distribution: Optional[Dict[str, float]] = None,
        balance_sources: bool = True,
        ensure_diversity: bool = True,
    ) -> GoldenDataset:
        """构建最终的金标准数据集

        执行以下操作:
          1. 检查查询数量
          2. 平衡难度分布
          3. 平衡数据来源
          4. 验证完整性
          5. 生成统计信息

        Args:
            target_size: 目标数据集大小
            difficulty_distribution: 难度分布 (默认为标准分布)
            balance_sources: 是否平衡来源
            ensure_diversity: 是否确保多样性

        Returns:
            GoldenDataset实例
        """
        if not self.queries:
            print("[警告] 没有查询数据，返回空数据集")
            return GoldenDataset(name=self.dataset_name)

        all_queries = self.queries.copy()

        # 1. 去重 (基于查询文本)
        seen_queries = set()
        deduplicated = []
        for q in all_queries:
            normalized = q.query.lower().strip()
            if normalized not in seen_queries:
                seen_queries.add(normalized)
                deduplicated.append(q)
        all_queries = deduplicated
        print(f"[去重] 去重后: {len(all_queries)} 条查询")

        # 2. 按难度采样
        if difficulty_distribution is None:
            difficulty_distribution = DifficultyLevel.distribution()

        balanced_queries = []
        for level_str, ratio in difficulty_distribution.items():
            level = DifficultyLevel(level_str)
            level_queries = [q for q in all_queries if q.difficulty == level]
            target_count = int(target_size * ratio)

            if len(level_queries) >= target_count:
                sampled = random.sample(level_queries, target_count)
            else:
                sampled = level_queries
                if len(level_queries) < target_count:
                    print(f"[提示] {level_str} 级别只有 {len(level_queries)} 条，"
                          f"目标 {target_count} 条")

            balanced_queries.extend(sampled)

        all_queries = balanced_queries
        print(f"[难度平衡] 平衡后: {len(all_queries)} 条查询")

        # 3. 确保多样性 (去除过于相似的查询)
        if ensure_diversity and len(all_queries) > target_size:
            all_queries = self._ensure_diversity(all_queries, target_size)

        # 4. 限制到目标大小
        if len(all_queries) > target_size:
            all_queries = random.sample(all_queries, target_size)

        # 5. 构建数据集
        dataset = GoldenDataset(
            name=self.dataset_name,
            version="1.0.0",
            description=f"金标准评估数据集 (自动生成于 {datetime.now().strftime('%Y-%m-%d')})",
        )
        dataset.queries = all_queries

        # 6. 验证
        passed, issues = dataset.validate()
        if not passed:
            print(f"[验证] 发现问题: {len(issues)}")
            for issue in issues:
                print(f"  - {issue}")

        print(f"\n[构建完成] {dataset.name}")
        print(f"  总查询数: {dataset.size()}")
        print(f"  难度分布: {dataset.difficulty_breakdown()}")
        print(f"  来源分布: {dataset.source_breakdown()}")

        return dataset

    def _ensure_diversity(
        self,
        queries: List[GoldenQuery],
        max_count: int,
    ) -> List[GoldenQuery]:
        """确保查询多样性 (简化版: 按类别采样)"""
        if not queries:
            return queries

        # 按类别分组
        by_category = defaultdict(list)
        for q in queries:
            cat = q.category or 'uncategorized'
            by_category[cat].append(q)

        # 从每个类别均匀采样
        num_categories = len(by_category)
        per_category = max(1, max_count // num_categories)

        diverse = []
        for cat, cat_queries in by_category.items():
            sample_count = min(per_category, len(cat_queries))
            diverse.extend(random.sample(cat_queries, sample_count))

        return diverse[:max_count]


# ============================================================================
# 便捷函数
# ============================================================================

def create_sample_annotations() -> List[Dict[str, Any]]:
    """创建样例人工标注数据"""
    return [
        {
            'query': '什么是RAG技术？',
            'reference_answer': 'RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的人工智能技术。'
                               '它首先从知识库中检索相关文档，然后将这些文档作为上下文提供给大语言模型，'
                               '以生成更准确、更可靠的回答。',
            'key_facts': ['RAG结合检索和生成', '先检索后生成', '提高回答准确性'],
            'relevant_doc_ids': ['doc_001'],
            'difficulty': 'easy',
            'category': 'AI基础',
        },
        {
            'query': '如何评估RAG系统的性能？',
            'reference_answer': '评估RAG系统性能需要多维度考量：1) 检索质量（召回率、精确率、MRR）；'
                               '2) 生成质量（忠实度、相关性、流畅度）；3) 系统效率（延迟、吞吐量）；'
                               '4) 成本效益。常用的评估框架包括RAGAS和trulens-eval。',
            'key_facts': ['多维度评估', '评估检索质量和生成质量', '使用RAGAS框架'],
            'relevant_doc_ids': ['doc_002', 'doc_003'],
            'difficulty': 'medium',
            'category': 'AI评估',
        },
        {
            'query': '分析向量数据库在RAG中的作用及其对检索质量的影响。',
            'reference_answer': '向量数据库是RAG系统的核心组件。它负责存储文档的嵌入向量并支持高效的相似度搜索。'
                               '向量数据库的选择直接影响检索质量：维度和索引类型影响召回率，'
                               '距离度量（余弦、欧几里得、点积）影响排序精度。常用的向量数据库包括'
                               'ChromaDB、Milvus、Pinecone、Weaviate等。',
            'key_facts': ['向量数据库存储嵌入向量', '支持相似度搜索', '影响检索质量'],
            'relevant_doc_ids': ['doc_004', 'doc_005'],
            'difficulty': 'hard',
            'category': 'AI架构',
        },
    ]


def create_sample_documents() -> List[Dict[str, Any]]:
    """创建样例文档数据"""
    return [
        {
            'doc_id': 'doc_001',
            'title': 'RAG技术入门',
            'content': 'RAG（Retrieval-Augmented Generation，检索增强生成）是一种AI架构，它将信息检索系统'
                       '与大语言模型结合。RAG首先从知识库中检索与用户查询相关的文档，然后将这些文档作为'
                       '额外的上下文提供给语言模型，帮助模型生成更准确的回答。',
            'category': 'AI基础',
        },
        {
            'doc_id': 'doc_002',
            'title': 'RAG系统评估方法',
            'content': '评估RAG系统的质量需要从多个维度进行。检索维度关注系统能否找到正确的文档；'
                       '生成维度关注回答是否忠实于检索到的内容。常见的评估指标包括Recall@K、MRR、NDCG等'
                       '检索指标，以及忠实度、相关性和流畅度等生成指标。',
            'category': 'AI评估',
        },
        {
            'doc_id': 'doc_003',
            'title': '向量数据库选型指南',
            'content': '向量数据库是RAG系统中的关键组件。它使用嵌入模型将文档转换为向量表示，并支持'
                       '高效的近似最近邻搜索。选择合适的向量数据库需要考虑性能、扩展性、成本和易用性。'
                       '主流的向量数据库包括ChromaDB、Milvus、Pinecone和Weaviate。',
            'category': 'AI架构',
        },
    ]


def create_sample_production_logs() -> List[Dict[str, Any]]:
    """创建样例生产日志数据"""
    return [
        {
            'query': 'RAG有什么优点？',
            'response': 'RAG的主要优点包括：1）减少幻觉 - 通过检索真实文档来约束生成；'
                       '2）知识更新 - 不需要重新训练模型就可以添加新知识；'
                       '3）可解释性 - 可以溯源到具体的检索文档。',
            'feedback_score': 0.9,
            'user_clicked_helpful': True,
            'retrieved_doc_ids': ['doc_001'],
            'timestamp': '2024-01-15T10:30:00',
        },
        {
            'query': 'ChromaDB和Milvus有什么区别？',
            'response': 'ChromaDB更加轻量，适合原型开发和小规模应用，学习成本低。'
                       'Milvus擅长大规模向量检索，支持分布式部署，性能更优。'
                       '选择时需要根据数据规模、性能要求和运维能力来决定。',
            'feedback_score': 0.85,
            'conversation_continued': True,
            'retrieved_doc_ids': ['doc_003'],
            'timestamp': '2024-01-15T11:00:00',
        },
        {
            'query': '什么是嵌入模型？',
            'response': '嵌入模型是将文本转换为向量表示的模型。它将文本映射到高维空间中的点，'
                       '使得语义相似的文本距离更近。常用的嵌入模型包括OpenAI的text-embedding-3、'
                       'Cohere的embed-v3等。',
            'feedback_score': 0.75,
            'retrieved_doc_ids': ['doc_001', 'doc_003'],
            'timestamp': '2024-01-15T11:30:00',
        },
    ]


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  金标准数据集构建 - 完整演示")
    print("=" * 60)

    # ---- 初始化构建器 ----
    builder = GoldenDatasetBuilder(
        dataset_name="RAG_Golden_Eval_v1",
        seed=42,
    )

    # ---- 来源1: 添加人工标注 ----
    print("\n[步骤1] 添加人工标注...")
    human_annotations = create_sample_annotations()
    builder.add_human_annotations(human_annotations)

    # ---- 来源2: 合成生成 ----
    print("\n[步骤2] 从文档合成生成问题...")
    documents = create_sample_documents()
    builder.generate_synthetic(
        documents=documents,
        questions_per_doc=5,
        difficulty_distribution={'easy': 2, 'medium': 2, 'hard': 1},
    )

    # ---- 来源3: 生产日志挖掘 ----
    print("\n[步骤3] 从生产日志挖掘高质查询...")
    production_logs = create_sample_production_logs()
    builder.mine_production_logs(production_logs)

    # ---- 对抗样本生成 ----
    print("\n[步骤4] 生成对抗样本...")
    # 对easy难度的查询生成对抗变体
    easy_base_queries = [q for q in builder.queries if q.difficulty == DifficultyLevel.EASY]
    if easy_base_queries:
        builder.generate_adversarial_variants(
            easy_base_queries[:3],
            count_per_query=2,
        )

    # ---- 构建最终数据集 ----
    print("\n[步骤5] 构建最终数据集...")
    dataset = builder.build(
        target_size=100,
        ensure_diversity=True,
    )

    # ---- 打印统计 ----
    print("\n" + "=" * 50)
    print("  数据集统计")
    print("=" * 50)

    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # ---- 子集提取 ----
    print("\n[子集提取] 仅获取easy难度的查询")
    easy_subset = dataset.get_subset(difficulty=DifficultyLevel.EASY)
    print(f"  Easy子集大小: {easy_subset.size()}")

    # ---- 保存数据集 ----
    print("\n[保存] 保存数据集...")
    output_path = "golden_dataset_v1.json"
    dataset.save(output_path)

    # ---- 验证 ----
    passed, issues = dataset.validate()
    if passed:
        print("\n[验证] 数据集验证通过!")
    else:
        print(f"\n[验证] 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")

    # ---- 验证加载 ----
    print("\n[验证加载] 从文件重新加载数据集...")
    if os.path.exists(output_path):
        loaded_dataset = GoldenDataset.load(output_path)
        print(f"  加载成功: {loaded_dataset.size()} 条查询")
        assert loaded_dataset.size() == dataset.size(), "加载的数据集大小不一致!"
        print(f"  大小一致，加载验证通过!")

    print()
    print("=" * 60)
    print("  金标准数据集构建完成")
    print("=" * 60)
