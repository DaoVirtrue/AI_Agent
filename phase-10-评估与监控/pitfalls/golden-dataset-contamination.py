#!/usr/bin/env python3
"""
陷阱3: 金标准数据集污染 (Golden Dataset Contamination)
=========================================================
症状: 评估得分随迭代稳步提升，但线上表现没有改善
根因:
  1. 金标准数据泄露到训练数据中
  2. 模型"记住"了评估问题的答案
  3. 评估数据集过时 (不代表当前分布)
  4. 数据标注者偏见 (标注者自身的偏好)

解决方案:
  - 定期刷新金标准数据集
  - 检测数据泄漏
  - 使用隔离的hold-out集
  - 交叉验证标注一致性
"""

import hashlib
import json
import random
import re
from typing import List, Dict, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict


@dataclass
class ContaminationDetector:
    """数据集污染检测器"""

    # 已知的训练数据摘要 (用于检测泄漏)
    training_data_hashes: Set[str] = field(default_factory=set)
    training_data_ngrams: Set[str] = field(default_factory=set)

    def add_training_data(self, texts: List[str]) -> None:
        """注册训练数据"""
        for text in texts:
            # 添加哈希
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            self.training_data_hashes.add(text_hash)

            # 添加n-grams (4-gram)
            words = re.findall(r'\w+', text.lower())
            for i in range(len(words) - 3):
                ngram = ' '.join(words[i:i+4])
                self.training_data_ngrams.add(ngram)

    def check_contamination(
        self,
        eval_sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """检查单个评估样本是否被污染

        Returns:
            污染分析结果
        """
        results = {
            'sample_id': eval_sample.get('query_id', 'unknown'),
            'contaminated': False,
            'findings': [],
        }

        # 检查1: 完全相同匹配
        eval_text = json.dumps(eval_sample, sort_keys=True, ensure_ascii=False)
        eval_hash = hashlib.sha256(eval_text.encode()).hexdigest()
        if eval_hash in self.training_data_hashes:
            results['contaminated'] = True
            results['findings'].append('评估样本与训练数据完全相同!')

        # 检查2: 查询出现在训练数据中
        query = eval_sample.get('query', '')
        query_words = re.findall(r'\w+', query.lower())
        for i in range(len(query_words) - 3):
            ngram = ' '.join(query_words[i:i+4])
            if ngram in self.training_data_ngrams:
                results['contaminated'] = True
                results['findings'].append(f'查询的ngram "{ngram}"出现在训练数据中')
                break

        # 检查3: 参考答案与训练数据高度重叠
        reference = eval_sample.get('reference_answer', '')
        if reference:
            ref_words = set(re.findall(r'\w+', reference.lower()))
            overlap_count = 0
            for ngram in self.training_data_ngrams:
                ngram_words = set(ngram.split())
                if ngram_words.issubset(ref_words):
                    overlap_count += 1

            if overlap_count > 3:
                results['contaminated'] = True
                results['findings'].append(f'参考答案与训练数据有{overlap_count}处n-gram重叠')

        return results

    def audit_dataset(
        self,
        eval_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """审计整个评估数据集

        Returns:
            审计报告
        """
        contaminated_count = 0
        findings_summary = defaultdict(int)

        for sample in eval_samples:
            result = self.check_contamination(sample)
            if result['contaminated']:
                contaminated_count += 1
                for finding in result['findings']:
                    findings_summary[finding[:50]] += 1

        total = len(eval_samples)
        contamination_rate = contaminated_count / max(1, total)

        return {
            'total_samples': total,
            'contaminated_samples': contaminated_count,
            'contamination_rate': round(contamination_rate, 4),
            'severity': (
                'CRITICAL' if contamination_rate > 0.1
                else 'WARNING' if contamination_rate > 0.01
                else 'HEALTHY'
            ),
            'findings': dict(findings_summary),
            'recommendation': (
                '数据集清洁' if contamination_rate == 0
                else f'需要清理{contaminated_count}个污染样本!'
            ),
        }


def check_annotator_bias(
    annotations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """检查标注者偏见

    Args:
        annotations: 标注记录，每条包含 {'annotator_id': ..., 'scores': {...}}

    Returns:
        标注者一致性分析
    """
    if len(annotations) < 2:
        return {'bias_detected': False, 'reason': '标注者不足'}

    # 按标注者分组
    by_annotator = defaultdict(list)
    for ann in annotations:
        annotator_id = ann.get('annotator_id', 'unknown')
        by_annotator[annotator_id].append(ann)

    # 计算每个标注者的平均分
    annotator_means = {}
    for aid, anns in by_annotator.items():
        all_scores = []
        for ann in anns:
            scores = ann.get('scores', {})
            all_scores.extend(scores.values())
        if all_scores:
            annotator_means[aid] = sum(all_scores) / len(all_scores)

    # 检测异常
    if len(annotator_means) >= 2:
        means = list(annotator_means.values())
        overall_mean = sum(means) / len(means)

        biases = []
        for aid, mean in annotator_means.items():
            deviation = mean - overall_mean
            bias_level = (
                'high_bias' if abs(deviation) > 0.15
                else 'medium_bias' if abs(deviation) > 0.08
                else 'normal'
            )
            biases.append({
                'annotator': aid,
                'average_score': round(mean, 4),
                'deviation_from_mean': round(deviation, 4),
                'bias_level': bias_level,
            })

        has_bias = any(b['bias_level'] == 'high_bias' for b in biases)

        return {
            'bias_detected': has_bias,
            'annotator_count': len(annotator_means),
            'overall_mean': round(overall_mean, 4),
            'annotator_biases': biases,
            'recommendation': (
                '标注者一致性良好' if not has_bias
                else '存在标注者偏见，建议校准或交叉验证'
            ),
        }

    return {'bias_detected': False}


def create_freshness_checker() -> callable:
    """创建数据集新鲜度检查器"""

    def check_freshness(
        dataset_created_at: str,
        production_data_distribution: Dict[str, float],
        stale_threshold_days: int = 90,
    ) -> Dict[str, Any]:
        """检查数据集是否过时"""
        created_date = datetime.fromisoformat(dataset_created_at)
        days_old = (datetime.now() - created_date).days

        is_stale = days_old > stale_threshold_days

        return {
            'dataset_age_days': days_old,
            'stale_threshold': stale_threshold_days,
            'is_stale': is_stale,
            'recommendation': (
                '数据集新鲜' if not is_stale
                else f'数据集已{int(days_old)}天未更新，建议刷新!'
            ),
            'refresh_strategies': [
                '每周从生产日志中采样新查询',
                '使用LLM从新文档自动生成评估问题',
                '定期人工审核并补充对抗样本',
            ] if is_stale else [],
        }

    return check_freshness


SOLUTION_CHECKLIST = """
金标准数据集防污染检查清单:

□ 1. 评估集与训练集物理隔离 (不同文件/数据库)
□ 2. 使用哈希检查评估样本是否出现在训练数据中
□ 3. 定期 (每月) 刷新20%的评估样本
□ 4. 多名标注者交叉验证，检查标注者间一致性
□ 5. 保留10%的hold-out数据不用于任何优化
□ 6. 记录数据集的创建时间和来源
□ 7. 每次迭代前检查数据集新鲜度
"""


if __name__ == "__main__":
    print("=" * 60)
    print("  金标准数据集污染检测 - 演示")
    print("=" * 60)

    # 模拟训练数据
    train_texts = [
        "RAG技术结合了信息检索和文本生成，能有效减少幻觉现象。",
        "深度学习使用多层神经网络，通过反向传播算法进行训练。",
        "Python是一种解释型编程语言，在数据科学领域广泛应用。",
        "向量数据库用于存储和检索高维向量，支持近似最近邻搜索。",
    ]

    # 注册训练数据
    detector = ContaminationDetector()
    detector.add_training_data(train_texts)

    # 测试样本
    eval_samples = [
        {
            'query_id': 'clean_001',
            'query': '什么是分布式系统？',
            'reference_answer': '分布式系统是由多台计算机通过网络协作完成任务的系统。',
        },
        {
            'query_id': 'contaminated_001',
            'query': 'RAG技术结合了信息检索和文本生成，能有效减少幻觉现象。',  # 直接来自训练数据!
            'reference_answer': 'RAG结合检索和生成来减少幻觉。',
        },
        {
            'query_id': 'partial_001',
            'query': '深度学习有什么特点？',
            'reference_answer': '深度学习使用多层神经网络，通过反向传播算法进行训练。',  # 参考答案来自训练!
        },
    ]

    # 审计
    for sample in eval_samples:
        result = detector.check_contamination(sample)
        status = '❌ 污染!' if result['contaminated'] else '✓ 清洁'
        print(f"\n  {status} {sample['query_id']}")
        if result['findings']:
            for finding in result['findings']:
                print(f"    → {finding[:80]}")

    # 完整审计
    print("\n" + "-" * 40)
    print("完整审计:")
    audit = detector.audit_dataset(eval_samples)
    print(f"  污染率: {audit['contamination_rate']:.1%}")
    print(f"  严重性: {audit['severity']}")
    print(f"  {audit['recommendation']}")

    # 标注者偏见检查
    print("\n" + "-" * 40)
    print("标注者偏见检查:")

    annotations = [
        {'annotator_id': 'Alice', 'scores': {'accuracy': 0.9, 'relevance': 0.85}},
        {'annotator_id': 'Bob', 'scores': {'accuracy': 0.88, 'relevance': 0.82}},
        {'annotator_id': 'Charlie', 'scores': {'accuracy': 0.72, 'relevance': 0.68}},  # 偏严
        {'annotator_id': 'Diana', 'scores': {'accuracy': 0.95, 'relevance': 0.92}},    # 偏松
    ]

    bias_result = check_annotator_bias(annotations)
    print(f"  偏见检测: {bias_result['bias_detected']}")
    for b in bias_result.get('annotator_biases', []):
        print(f"    {b['annotator']}: avg={b['average_score']:.2f}, "
              f"偏差={b['deviation_from_mean']:+.2f} [{b['bias_level']}]")

    print(f"\n{SOLUTION_CHECKLIST}")
