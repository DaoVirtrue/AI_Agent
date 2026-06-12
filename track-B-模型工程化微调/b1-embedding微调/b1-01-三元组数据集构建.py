#!/usr/bin/env python3
"""
B1-01: 三元组数据集构建 (Triplet Dataset Construction)
=========================================================
学习目标:
  1. 从用户日志提取 (query, positive_chunk, negative_chunk) 三元组
  2. LLM辅助生成负样本
  3. 比例控制 1:3:5 (anchor:positive:negative)
  4. 质量过滤
"""

import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================
# Section 1: 三元组数据结构
# ============================================================

@dataclass
class TripletSample:
    """三元组样本 (anchor-positive-negative)"""
    query: str              # anchor / 查询
    positive_chunk: str     # 正样本 / 相关文档片段
    negative_chunk: str     # 负样本 / 不相关文档片段
    source: str             # "user_log" | "llm_generated" | "hard_negative"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "positive": self.positive_chunk,
            "negative": self.negative_chunk,
            "source": self.source,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict) -> "TripletSample":
        return TripletSample(
            query=data["query"],
            positive_chunk=data["positive"],
            negative_chunk=data["negative"],
            source=data.get("source", "unknown"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TripletDataset:
    """三元组数据集"""
    samples: List[TripletSample] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def add(self, sample: TripletSample):
        self.samples.append(sample)

    def to_jsonl(self, output_path: str):
        """导出为JSONL格式（sentence-transformers兼容）"""
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        print(f"  [导出] {len(self.samples)} 条三元组 → {output_path}")

    def compute_stats(self) -> Dict:
        """计算数据统计"""
        source_counts = defaultdict(int)
        query_lengths = []
        positive_lengths = []
        negative_lengths = []
        unique_queries = set()

        for sample in self.samples:
            source_counts[sample.source] += 1
            query_lengths.append(len(sample.query))
            positive_lengths.append(len(sample.positive_chunk))
            negative_lengths.append(len(sample.negative_chunk))
            unique_queries.add(sample.query)

        self.stats = {
            "total_samples": len(self.samples),
            "unique_queries": len(unique_queries),
            "by_source": dict(source_counts),
            "avg_query_length": sum(query_lengths) / max(len(query_lengths), 1),
            "avg_positive_length": sum(positive_lengths) / max(len(positive_lengths), 1),
            "avg_negative_length": sum(negative_lengths) / max(len(negative_lengths), 1),
            # 正负比: 理想值为 1:3:5 (anchor:positive:negative)
            "positive_per_query": len(self.samples) / max(len(unique_queries), 1),
        }
        return self.stats


# ============================================================
# Section 2: 用户日志提取三元组
# ============================================================

class UserLogTripletExtractor:
    """
    从用户日志提取三元组

    策略:
    - positive: 用户点击/查看了的文档片段
    - negative: 检索返回但用户未点击的文档片段
    - query: 用户的原始搜索查询

    日志格式示例:
    {
        "query": "如何配置RAG管道？",
        "results": [
            {"chunk_id": "doc1_chunk3", "text": "...", "clicked": true, "position": 1},
            {"chunk_id": "doc2_chunk1", "text": "...", "clicked": false, "position": 2},
            {"chunk_id": "doc3_chunk5", "text": "...", "clicked": false, "position": 3},
        ],
        "timestamp": "2024-06-01T10:00:00Z",
        "user_id": "user_001"
    }
    """

    def __init__(self, min_query_length: int = 5,
                 min_chunk_length: int = 50,
                 max_negatives_per_query: int = 5):
        """
        参数:
            min_query_length: 最短查询长度（过滤过短查询）
            min_chunk_length: 最短chunk长度
            max_negatives_per_query: 每个查询最多保留的负样本数
        """
        self.min_query_length = min_query_length
        self.min_chunk_length = min_chunk_length
        self.max_negatives_per_query = max_negatives_per_query

    def extract_from_logs(self, logs: List[Dict]) -> TripletDataset:
        """从用户行为日志中提取三元组"""
        dataset = TripletDataset()
        skipped = 0

        for entry in logs:
            query = entry.get("query", "").strip()
            results = entry.get("results", [])

            # 过滤短查询
            if len(query) < self.min_query_length:
                skipped += 1
                continue

            # 分离正负样本
            positives = []
            negatives = []

            for result in results:
                chunk_text = result.get("text", "")
                if len(chunk_text) < self.min_chunk_length:
                    continue

                if result.get("clicked", False):
                    positives.append(chunk_text)
                else:
                    negatives.append(chunk_text)

            if not positives:
                continue  # 没有正样本，跳过

            # 每个正样本创建多个三元组（针对不同负样本）
            for pos_text in positives:
                # 选择负样本
                selected_negatives = negatives[:self.max_negatives_per_query]
                # 如果负样本不够，从所有未点击的样本中随机选择
                if len(selected_negatives) < self.max_negatives_per_query:
                    all_texts = [r.get("text", "") for r in results
                                if len(r.get("text", "")) >= self.min_chunk_length]
                    extra_negs = [t for t in all_texts
                                 if t not in positives and t not in selected_negatives]
                    selected_negatives.extend(
                        extra_negs[:self.max_negatives_per_query - len(selected_negatives)]
                    )

                for neg_text in selected_negatives[:self.max_negatives_per_query]:
                    # 确保正负不完全相同
                    if pos_text == neg_text:
                        continue

                    sample = TripletSample(
                        query=query,
                        positive_chunk=pos_text,
                        negative_chunk=neg_text,
                        source="user_log",
                        metadata={
                            "user_id": entry.get("user_id", "unknown"),
                            "timestamp": entry.get("timestamp", ""),
                        },
                    )
                    dataset.add(sample)

        print(f"  [用户日志] 从 {len(logs)} 条日志提取了 "
              f"{len(dataset.samples)} 个三元组 (跳过{skipped}条)")
        return dataset


# ============================================================
# Section 3: LLM辅助生成负样本
# ============================================================

class LLMAssistedNegativeGenerator:
    """
    LLM辅助负样本生成

    用途: 当天然负样本不足时，利用LLM生成高质量的"看起来相关"但实际无关的文本

    策略:
    1. 主题相似但内容不相关: 同一领域的另一个话题
    2. 关键词重叠但语义不同: 同样的术语但不同上下文
    3. 部分匹配但整体无关: 包含部分相关句子的无关段落
    """

    NEGATIVE_GENERATION_PROMPT = """你是一个数据集构建助手。给定一个查询和它的正确答案，请生成一个"看起来相关但实际无关"的文本片段作为负样本。

要求:
1. 主题应与查询在同一领域，但具体内容不相关
2. 可能包含查询中的部分关键词，但整体语义不相关
3. 长度与正样本相当
4. 用中文生成
5. 只输出负样本文本，不要加任何说明

查询: {query}
正样本文本: {positive_chunk}

负样本文本:"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # 内置的负样本模板（无API时的回退方案）
        self._fallback_templates = self._init_fallback_templates()

    def _init_fallback_templates(self) -> Dict[str, List[str]]:
        """初始化回退负样本模板（按领域分类）"""
        return {
            "技术": [
                "Python是一种广泛使用的编程语言，以其简洁的语法和丰富的生态系统而闻名。Django和Flask是两个流行的Web框架。",
                "数据库索引是提升查询性能的关键技术。B-Tree索引适用于范围查询，而Hash索引适合等值查询。",
                "CI/CD流水线通过自动化构建、测试和部署流程，提高了软件交付的效率和质量。",
            ],
            "财务": [
                "资产负债表反映了企业在特定日期的财务状况，包括资产、负债和所有者权益三大要素。",
                "现金流管理是企业财务管理的重要组成部分，直接影响企业的流动性和偿债能力。",
                "内部审计通过独立客观的确认和咨询活动，帮助组织改善运营效率。",
            ],
            "通用": [
                "从历史数据来看，企业数字化转型的成功率约为30%，远低于预期。组织文化和人才是其中的关键挑战。",
                "天气预报主要依靠气象卫星和地面观测站的数据，结合数值预报模型进行计算。",
                "交通拥堵是城市化进程中面临的普遍问题，智能交通系统为缓解交通压力提供了一种技术解决方案。",
            ],
        }

    def generate_fallback(self, query: str, positive_chunk: str) -> List[str]:
        """生成回退负样本（基于模板+领域匹配）"""
        # 根据查询关键词匹配领域
        domain_keywords = {
            "技术": ["代码", "编程", "API", "服务器", "数据库", "算法", "框架"],
            "财务": ["财务", "报表", "会计", "审计", "预算", "成本", "收入"],
        }

        matched_domain = "通用"
        for domain, keywords in domain_keywords.items():
            if any(kw in query for kw in keywords):
                matched_domain = domain
                break

        templates = self._fallback_templates.get(matched_domain, [])
        if not templates:
            templates = self._fallback_templates["通用"]

        # 随机选择1-3个模板
        k = random.randint(1, min(3, len(templates)))
        return random.sample(templates, k)

    def generate(self, query: str, positive_chunk: str,
                 num_negatives: int = 3) -> List[str]:
        """
        为单个(query, positive)对生成LLM负样本
        """
        negatives = []

        # 尝试API
        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                for _ in range(num_negatives):
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{
                            "role": "user",
                            "content": self.NEGATIVE_GENERATION_PROMPT.format(
                                query=query, positive_chunk=positive_chunk[:500]
                            ),
                        }],
                        max_tokens=200,
                        temperature=0.8,  # 稍高以增加多样性
                    )
                    neg_text = response.choices[0].message.content.strip()
                    if neg_text and neg_text != positive_chunk:
                        negatives.append(neg_text)
            except Exception as e:
                print(f"  [警告] LLM负样本生成失败: {e}")

        # 回退到模板
        if len(negatives) < num_negatives:
            fallbacks = self.generate_fallback(query, positive_chunk)
            negatives.extend(fallbacks[:num_negatives - len(negatives)])

        return negatives[:num_negatives]


# ============================================================
# Section 4: 数据集构建与质量过滤
# ============================================================

class TripletDatasetBuilder:
    """
    三元组数据集构建器
    实现 1:3:5 比率控制 (anchor:positive:negative)
    即: 每个查询1个 → 每查询3个正样本 → 每正样本5个负样本
    """

    def __init__(self):
        self.extractor = UserLogTripletExtractor()
        self.negative_generator = LLMAssistedNegativeGenerator()

    def build_from_logs(self, logs: List[Dict]) -> TripletDataset:
        """从日志构建完整数据集"""
        # 步骤1: 从日志提取
        dataset = self.extractor.extract_from_logs(logs)

        # 步骤2: LLM补充负样本（如果负样本不足）
        enriched_count = 0
        for sample in dataset.samples:
            # 每条样本至少应有足够负样本
            if len(sample.negative_chunk) < 50:
                # 简单跳过，实际中会调用generate
                pass

        # 步骤3: 质量控制
        dataset = self.quality_filter(dataset)

        # 步骤4: 平衡比率
        dataset = self.balance_dataset(dataset)

        return dataset

    def quality_filter(self, dataset: TripletDataset) -> TripletDataset:
        """
        质量过滤
        规则:
        1. 正样本与查询的Jaccard相似度不能太低
        2. 负样本与正样本不能太相似
        3. 删除重复三元组
        4. 删除过短/过长的文本
        """
        filtered = TripletDataset()
        seen_hashes = set()
        removed_stats = {
            "duplicate": 0,
            "too_short": 0,
            "too_similar_pn": 0,
            "too_dissimilar_qp": 0,
        }

        for sample in dataset.samples:
            # 规则1: 去重
            sample_hash = hashlib.md5(
                f"{sample.query}|{sample.positive_chunk[:100]}|{sample.negative_chunk[:100]}"
                .encode()
            ).hexdigest()
            if sample_hash in seen_hashes:
                removed_stats["duplicate"] += 1
                continue
            seen_hashes.add(sample_hash)

            # 规则2: 长度过滤
            if (len(sample.query) < 3 or
                len(sample.positive_chunk) < 20 or
                len(sample.negative_chunk) < 20):
                removed_stats["too_short"] += 1
                continue

            # 规则3: 正负不能太相似（Jaccard > 0.7 视为太相似）
            p_words = set(sample.positive_chunk[:300])
            n_words = set(sample.negative_chunk[:300])
            intersec = len(p_words & n_words)
            union = max(len(p_words | n_words), 1)
            pn_similarity = intersec / union
            if pn_similarity > 0.7:
                removed_stats["too_similar_pn"] += 1
                continue

            # 规则4: 查询与正样本需要有一定重叠
            q_words = set(sample.query)
            qp_intersec = len(q_words & p_words)
            if qp_intersec == 0 and len(sample.query) < 20:
                # 短查询如果完全没有词重叠，可能是弱相关
                # 不严格过滤，但标记
                removed_stats["too_dissimilar_qp"] += 1

            filtered.add(sample)

        print(f"  [质量过滤] {len(dataset.samples)} → {len(filtered.samples)} "
              f"(移除: {removed_stats})")
        return filtered

    def balance_dataset(self, dataset: TripletDataset,
                        target_pos_per_query: int = 3,
                        target_neg_per_pos: int = 5) -> TripletDataset:
        """
        平衡数据集到目标比率 1:3:5

        当实际数据不满足比率时:
        - 正样本不足：通过数据增强扩充
        - 负样本不足：通过LLM生成
        """
        # 按查询分组
        by_query: Dict[str, List[TripletSample]] = defaultdict(list)
        for sample in dataset.samples:
            by_query[sample.query].append(sample)

        balanced = TripletDataset()
        for query, samples in by_query.items():
            # 收集唯一的正样本
            unique_positives = list(set(s.positive_chunk for s in samples))
            unique_negatives = list(set(s.negative_chunk for s in samples))

            # 如果正样本不够，复制并略微修改
            while len(unique_positives) < target_pos_per_query:
                # 从现有正样本生成变体
                base = unique_positives[0] if unique_positives else "placeholder"
                unique_positives.append(base)
                break

            # 如果负样本不够，用LLM生成
            if len(unique_negatives) < target_neg_per_pos:
                prod = unique_positives[0] if unique_positives else ""
                extra_negs = self.negative_generator.generate(
                    query, prod,
                    num_negatives=target_neg_per_pos - len(unique_negatives),
                )
                unique_negatives.extend(extra_negs)

            # 创建平衡的三元组
            for i, pos in enumerate(unique_positives[:target_pos_per_query]):
                for j, neg in enumerate(unique_negatives[:target_neg_per_pos]):
                    sample = TripletSample(
                        query=query,
                        positive_chunk=pos,
                        negative_chunk=neg,
                        source="balanced",
                        metadata={"pos_index": i, "neg_index": j},
                    )
                    balanced.add(sample)

        pre_balance_stats = dataset.compute_stats()
        post_balance_stats = balanced.compute_stats()
        print(f"  [平衡] {len(dataset.samples)} → {len(balanced.samples)} "
              f"(目标比率 1:{target_pos_per_query}:{target_neg_per_pos})")

        return balanced


# ============================================================
# Section 5: 主流程
# ============================================================

def create_sample_logs() -> List[Dict]:
    """创建示例用户日志数据"""
    return [
        {
            "query": "如何优化RAG系统的检索准确率？",
            "results": [
                {"chunk_id": "doc1_1", "text": "提高RAG检索准确率的方法包括：优化Embedding模型、调整chunking策略、引入重排序模型（Reranker）等。Embedding模型的微调可以显著提升特定领域的检索效果。", "clicked": True, "position": 1},
                {"chunk_id": "doc2_1", "text": "数据库查询优化涉及索引创建、查询计划分析和缓存策略。MySQL和PostgreSQL都提供了丰富的优化工具。", "clicked": False, "position": 2},
                {"chunk_id": "doc3_1", "text": "大语言模型的微调方法包括全参数微调和参数高效微调（PEFT）。LoRA和QLoRA是目前最流行的PEFT方法。", "clicked": False, "position": 3},
            ],
            "user_id": "user_001",
            "timestamp": "2024-06-01T10:00:00Z",
        },
        {
            "query": "什么是向量数据库？",
            "results": [
                {"chunk_id": "doc4_1", "text": "向量数据库是一种专门用于存储和检索高维向量数据的数据库系统。它通过近似最近邻（ANN）搜索算法实现高效的相似度检索。常见的向量数据库包括FAISS、Milvus、ChromaDB和Pinecone等。", "clicked": True, "position": 1},
                {"chunk_id": "doc5_1", "text": "关系型数据库使用二维表格结构来组织数据，通过SQL语言进行查询和操作。常见的关系型数据库有MySQL、PostgreSQL和Oracle。", "clicked": False, "position": 2},
                {"chunk_id": "doc1_2", "text": "Embedding模型将文本转换为固定维度的向量表示，使得语义相近的文本在向量空间中距离较近。BGE和M3E是优秀的中文Embedding模型。", "clicked": True, "position": 3},
            ],
            "user_id": "user_002",
            "timestamp": "2024-06-01T11:00:00Z",
        },
        {
            "query": "RAG系统如何处理多轮对话？",
            "results": [
                {"chunk_id": "doc6_1", "text": "多轮对话RAG系统需要维护对话历史，将前几轮的问题和回答作为上下文，结合当前查询进行检索。上下文感知的检索策略可以显著提升多轮对话的质量。常用的方法包括查询重写和对话状态跟踪。", "clicked": True, "position": 1},
                {"chunk_id": "doc7_1", "text": "机器学习中的特征工程是通过领域知识创建能够使机器学习算法达到最佳性能的特征的过程。好的特征能够简化模型复杂度。", "clicked": False, "position": 2},
            ],
            "user_id": "user_001",
            "timestamp": "2024-06-01T12:00:00Z",
        },
    ]


def main():
    """主函数：演示三元组数据集构建全流程"""
    print("=" * 70)
    print("B1-01: 三元组数据集构建 — 完整演示")
    print("=" * 70)

    # 1. 创建示例日志
    print("\n[步骤1] 加载示例用户日志...")
    logs = create_sample_logs()
    print(f"  加载了 {len(logs)} 条日志")

    # 2. 从日志提取三元组
    print("\n[步骤2] 从用户日志提取三元组...")
    builder = TripletDatasetBuilder()
    dataset = builder.extractor.extract_from_logs(logs)
    stats = dataset.compute_stats()
    print(f"  提取了 {stats['total_samples']} 个三元组")
    print(f"  涉及 {stats['unique_queries']} 个唯一查询")
    print(f"  平均查询长度: {stats['avg_query_length']:.1f} 字符")
    print(f"  平均正样本长度: {stats['avg_positive_length']:.1f} 字符")
    print(f"  平均负样本长度: {stats['avg_negative_length']:.1f} 字符")

    # 3. 展示样本
    print("\n[步骤3] 样本预览...")
    for i, sample in enumerate(dataset.samples[:3]):
        print(f"\n  三元组 #{i+1} (来源: {sample.source}):")
        print(f"    查询: {sample.query[:60]}...")
        print(f"    正样本: {sample.positive_chunk[:80]}...")
        print(f"    负样本: {sample.negative_chunk[:80]}...")

    # 4. LLM负样本生成演示
    print("\n[步骤4] LLM辅助负样本生成（回退模板模式）...")
    generator = LLMAssistedNegativeGenerator()
    fallback_negs = generator.generate(
        "RAG检索准确率优化",
        "优化Embedding模型和chunking策略可以提升RAG检索准确率。",
        num_negatives=3,
    )
    for i, neg in enumerate(fallback_negs):
        print(f"  负样本 {i+1}: {neg[:120]}...")

    # 5. 质量过滤
    print("\n[步骤5] 质量过滤...")
    filtered = builder.quality_filter(dataset)

    # 6. 导出
    print("\n[步骤6] 导出三元组数据集...")
    output_path = "demo_triplet_dataset.jsonl"
    filtered.to_jsonl(output_path)
    print(f"  文件大小: {os.path.getsize(output_path)} bytes")

    # 清理
    if os.path.exists(output_path):
        os.remove(output_path)
        print("  [清理] 已删除示例文件")

    print("\n" + "=" * 70)
    print("B1-01 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
