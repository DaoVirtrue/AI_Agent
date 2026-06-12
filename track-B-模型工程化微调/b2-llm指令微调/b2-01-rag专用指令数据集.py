#!/usr/bin/env python3
"""
B2-01: RAG专用指令数据集 (RAG-Specific Instruction Dataset)
================================================================
学习目标:
  1. 构建RAG专用指令数据（含引用追踪）
  2. 引用追踪训练数据
  3. 拒绝回答训练 ("暂无相关信息")
  4. 数据格式转换 (Alpaca ↔ ShareGPT ↔ ChatML)
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================
# Section 1: RAG指令数据构建器
# ============================================================

@dataclass
class RAGInstructionSample:
    """RAG指令样本"""
    id: str
    system_prompt: str
    user_query: str
    contexts: List[Dict]          # [{"source": ..., "content": ...}]
    assistant_response: str       # 含引用标记的完整回答
    metadata: Dict = field(default_factory=dict)


class RAGInstructionBuilder:
    """
    RAG专用指令数据构建器

    RAG指令数据的特殊要求:
    1. 回答必须引用来源（citation training）
    2. 上下文不足时需要拒绝回答（refusal training）
    3. 多文档综合能力（multi-document synthesis）
    4. 冲突信息处理能力
    """

    # 系统提示模板
    SYSTEM_PROMPTS = {
        "default": (
            "你是一个基于检索增强生成（RAG）的智能助手。"
            "请基于提供的上下文信息回答问题。"
            "每条回答必须引用来源，格式为 [来源: {source_id}]。"
            "如果上下文不足以回答问题，请明确说'根据现有资料无法确定'。"
            "回答使用中文。"
        ),
        "strict_citation": (
            "你是严格引用模式的RAG助手。"
            "要求：\n"
            "1. 每句话都必须标注出处，格式：[来源: {doc_id}]\n"
            "2. 不要添加上下文之外的信息\n"
            "3. 如信息不足，回答'暂无相关信息'\n"
            "4. 如有冲突信息，列出各方说法并标注来源"
        ),
        "multi_doc_synthesis": (
            "你是多文档综合分析助手。"
            "请综合多个来源的信息给出全面回答。"
            "对不同来源的信息进行对比和整合。"
            "每个关键信息点标注其来源。"
        ),
    }

    def __init__(self):
        self.samples: List[RAGInstructionSample] = []

    def build_citation_sample(self) -> List[RAGInstructionSample]:
        """
        构建引用追踪训练数据

        目标: 训练LLM在每个事实后面添加引用标记

        示例:
        User: 什么是RAG？
        Context: [来源: doc1] RAG是检索增强生成系统...
        Assistant: RAG是检索增强生成（Retrieval-Augmented Generation）系统[来源: doc1]。
                  它结合了检索和生成两个步骤[来源: doc1]...
        """
        citation_templates = [
            {
                "query": "什么是RAG系统？",
                "contexts": [
                    {"source": "doc_ai_01", "content": "RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的AI架构。它通过从外部知识库检索相关文档来增强大语言模型的回答质量。"},
                    {"source": "doc_ai_02", "content": "RAG系统的主要组件包括：向量数据库（存储文档嵌入）、检索器（查找相关文档）、重排序器（优化检索结果排序）和生成器（基于检索结果生成回答）。"},
                ],
                "response": (
                    "RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的AI架构[来源: doc_ai_01]。"
                    "它通过从外部知识库检索相关文档来增强大语言模型的回答质量[来源: doc_ai_01]。\n\n"
                    "RAG系统的主要组件包括[来源: doc_ai_02]：\n"
                    "1. 向量数据库 - 存储文档嵌入向量\n"
                    "2. 检索器 - 查找相关文档\n"
                    "3. 重排序器 - 优化检索结果排序\n"
                    "4. 生成器 - 基于检索结果生成回答"
                ),
            },
            {
                "query": "BGE-M3模型有哪些特点？",
                "contexts": [
                    {"source": "doc_embed_01", "content": "BGE-M3是BAAI发布的多语言Embedding模型。M3代表多语言（Multi-Lingual）、多功能（Multi-Functionality）和多粒度（Multi-Granularity）。"},
                    {"source": "doc_embed_02", "content": "BGE-M3支持超过100种语言，在MTEB中文基准测试中取得了优异成绩。它同时支持稠密检索（Dense）和稀疏检索（Sparse）。"},
                ],
                "response": (
                    "BGE-M3是BAAI发布的多语言Embedding模型[来源: doc_embed_01]。"
                    "M3代表三个核心特点[来源: doc_embed_01]：\n"
                    "1. 多语言（Multi-Lingual）- 支持超过100种语言[来源: doc_embed_02]\n"
                    "2. 多功能（Multi-Functionality）- 同时支持稠密检索和稀疏检索[来源: doc_embed_02]\n"
                    "3. 多粒度（Multi-Granularity）- 支持不同粒度的文本表示\n\n"
                    "该模型在MTEB中文基准测试中取得了优异成绩[来源: doc_embed_02]。"
                ),
            },
        ]

        samples = []
        for i, template in enumerate(citation_templates):
            sample = RAGInstructionSample(
                id=f"citation_{i:03d}",
                system_prompt=self.SYSTEM_PROMPTS["strict_citation"],
                user_query=template["query"],
                contexts=template["contexts"],
                assistant_response=template["response"],
                metadata={"type": "citation_training", "num_sources": len(template["contexts"])},
            )
            samples.append(sample)

        return samples

    def build_refusal_sample(self) -> List[RAGInstructionSample]:
        """
        构建拒绝回答训练数据

        目标: 训练LLM在信息不足时拒绝回答，而非编造答案（幻觉减少）

        拒绝策略:
        1. 无上下文: "暂无相关信息，无法回答"
        2. 部分相关: "根据现有资料，只能确认..."
        3. 信息冲突: "不同来源存在矛盾..."

        重要性:
        - 这是RAG系统减少幻觉的关键
        - 没有refusal training的模型倾向于编造内容
        """
        refusal_templates = [
            {
                "query": "2025年比特币的价格走势如何？",
                "contexts": [],  # 空上下文
                "response": "根据现有资料，暂无关于2025年比特币价格走势的相关信息，我无法提供准确回答。建议查阅最新的金融市场报告获取实时数据。",
                "type": "empty_context",
            },
            {
                "query": "小王在哪里工作？",
                "contexts": [
                    {"source": "doc_hr_01", "content": "公司2024年度员工信息表显示，目前在册员工共500人。各部门负责人信息已更新至最新版本。"},
                ],
                "response": "根据当前提供的公司员工信息资料，并未找到关于'小王'的具体工作信息[来源: doc_hr_01]。如需查询特定员工的详细信息，建议提供员工的全名或工号。",
                "type": "partial_info",
            },
            {
                "query": "A产品和B产品哪个更好？",
                "contexts": [
                    {"source": "doc_review_a", "content": "A产品在2024年获得了年度最佳产品奖，用户评分4.8/5。评测机构给出'强烈推荐'评级。"},
                    {"source": "doc_review_b", "content": "B产品在2024年获得了创新技术奖，用户评分4.7/5。评测机构指出其在特定场景下表现更优。"},
                ],
                "response": (
                    "两个产品各有优势，具体情况如下：\n\n"
                    "A产品获得了年度最佳产品奖，用户评分4.8/5[来源: doc_review_a]。\n"
                    "B产品获得了创新技术奖，在特定场景下表现更优[来源: doc_review_b]。\n\n"
                    "由于它们在不同方面各有所长，无法简单判断哪个'更好'。"
                    "建议根据您的具体使用场景和需求来判断哪个产品更适合您。"
                ),
                "type": "conflicting_info",
            },
        ]

        samples = []
        for i, template in enumerate(refusal_templates):
            sample = RAGInstructionSample(
                id=f"refusal_{i:03d}",
                system_prompt=self.SYSTEM_PROMPTS["default"],
                user_query=template["query"],
                contexts=template["contexts"],
                assistant_response=template["response"],
                metadata={"type": f"refusal_training_{template['type']}"},
            )
            samples.append(sample)

        return samples

    def build_multi_doc_sample(self) -> List[RAGInstructionSample]:
        """构建多文档综合训练数据"""
        multi_doc_templates = [
            {
                "query": "比较不同向量数据库的优缺点",
                "contexts": [
                    {"source": "doc_milvus", "content": "Milvus是开源向量数据库，支持十亿级向量检索。优点：高性能、分布式、GPU加速。缺点：部署复杂、资源占用高。"},
                    {"source": "doc_chroma", "content": "ChromaDB是轻量级向量数据库。优点：简单易用、Python原生、适合原型开发。缺点：不适合大规模生产环境。"},
                    {"source": "doc_pinecone", "content": "Pinecone是商业向量数据库服务。优点：零运维、自动扩展、高可用。缺点：成本较高、数据需上传云端。"},
                ],
                "response": (
                    "以下是对三个主流向量数据库的比较：\n\n"
                    "1. Milvus[来源: doc_milvus]：\n"
                    "   - 优点：高性能，支持十亿级向量，分布式架构，GPU加速\n"
                    "   - 缺点：部署复杂，资源占用较高\n"
                    "   - 适用场景：大规模生产环境\n\n"
                    "2. ChromaDB[来源: doc_chroma]：\n"
                    "   - 优点：简单易用，Python原生集成，适合快速原型开发\n"
                    "   - 缺点：不适合大规模生产环境\n"
                    "   - 适用场景：开发测试和小型项目\n\n"
                    "3. Pinecone[来源: doc_pinecone]：\n"
                    "   - 优点：零运维，自动扩展，高可用性\n"
                    "   - 缺点：成本较高，数据需要上传云端\n"
                    "   - 适用场景：需要快速上线的商业项目\n\n"
                    "总结：选择哪个数据库取决于您的具体需求。大型生产系统推荐Milvus或Pinecone，"
                    "快速原型开发推荐ChromaDB。"
                ),
            },
        ]

        samples = []
        for i, template in enumerate(multi_doc_templates):
            sample = RAGInstructionSample(
                id=f"multi_doc_{i:03d}",
                system_prompt=self.SYSTEM_PROMPTS["multi_doc_synthesis"],
                user_query=template["query"],
                contexts=template["contexts"],
                assistant_response=template["response"],
                metadata={"type": "multi_document_synthesis", "num_sources": len(template["contexts"])},
            )
            samples.append(sample)

        return samples

    def build_all(self) -> List[RAGInstructionSample]:
        """构建所有类型的RAG指令数据"""
        self.samples.extend(self.build_citation_sample())
        self.samples.extend(self.build_refusal_sample())
        self.samples.extend(self.build_multi_doc_sample())
        print(f"  [构建] 共生成 {len(self.samples)} 条RAG指令数据")
        print(f"    引用追踪: {sum(1 for s in self.samples if 'citation' in s.metadata.get('type',''))}")
        print(f"    拒绝训练: {sum(1 for s in self.samples if 'refusal' in s.metadata.get('type',''))}")
        print(f"    多文档综合: {sum(1 for s in self.samples if 'multi_doc' in s.metadata.get('type',''))}")
        return self.samples


# ============================================================
# Section 2: 数据格式转换器 (Alpaca ↔ ShareGPT ↔ ChatML)
# ============================================================

class FormatConverter:
    """
    数据格式转换器

    Alpaca格式:
    {
        "instruction": "系统指令 + 用户问题 + 上下文",
        "input": "",
        "output": "回答"
    }

    ShareGPT格式:
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }

    ChatML格式:
    <|im_start|>system
    ...<|im_end|>
    <|im_start|>user
    ...<|im_end|>
    <|im_start|>assistant
    ...<|im_end|>
    """

    @staticmethod
    def to_alpaca(sample: RAGInstructionSample) -> Dict:
        """转换为Alpaca格式"""
        context_text = "\n\n".join(
            f"[来源: {c['source']}]\n{c['content']}"
            for c in sample.contexts
        )
        instruction = (
            f"{sample.system_prompt}\n\n"
            f"上下文信息:\n{context_text}\n\n"
            f"用户问题: {sample.user_query}"
        ) if context_text else (
            f"{sample.system_prompt}\n\n用户问题: {sample.user_query}"
        )

        return {
            "instruction": instruction,
            "input": "",
            "output": sample.assistant_response,
        }

    @staticmethod
    def to_sharegpt(sample: RAGInstructionSample) -> Dict:
        """转换为ShareGPT格式"""
        context_text = "\n\n---\n".join(
            f"[来源: {c['source']}]\n{c['content']}"
            for c in sample.contexts
        )
        user_message = (
            f"上下文:\n{context_text}\n\n问题: {sample.user_query}"
        ) if context_text else sample.user_query

        return {
            "conversations": [
                {"from": "system", "value": sample.system_prompt},
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": sample.assistant_response},
            ],
        }

    @staticmethod
    def to_chatml(sample: RAGInstructionSample) -> str:
        """转换为ChatML格式"""
        context_text = "\n\n---\n".join(
            f"[来源: {c['source']}]\n{c['content']}"
            for c in sample.contexts
        )
        user_message = (
            f"上下文:\n{context_text}\n\n问题: {sample.user_query}"
        ) if context_text else sample.user_query

        return (
            f"<|im_start|>system\n{sample.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n{sample.assistant_response}<|im_end|>"
        )

    @classmethod
    def convert_dataset(cls, samples: List[RAGInstructionSample],
                        target_format: str) -> List:
        """批量转换"""
        if target_format == "alpaca":
            converter = cls.to_alpaca
        elif target_format == "sharegpt":
            converter = cls.to_sharegpt
        elif target_format == "chatml":
            converter = cls.to_chatml
        else:
            raise ValueError(f"不支持的格式: {target_format}")

        return [converter(s) for s in samples]


# ============================================================
# Section 3: 数据集质量控制
# ============================================================

class DatasetQualityChecker:
    """数据集质量控制"""

    @staticmethod
    def check(samples: List[RAGInstructionSample]) -> Dict:
        """检查并报告数据集质量"""
        issues = []
        stats = {
            "total": len(samples),
            "empty_response": 0,
            "missing_citation": 0,
            "short_response": 0,
            "context_too_long": 0,
            "passed": 0,
        }

        for sample in samples:
            # 检查空回答
            if not sample.assistant_response.strip():
                stats["empty_response"] += 1
                issues.append(f"[{sample.id}] 回答为空")
                continue

            # 检查引用（如果有上下文但没有引用标记）
            if sample.contexts and "[来源:" not in sample.assistant_response:
                stats["missing_citation"] += 1
                # 仅在citation类型的样本中强制要求引用
                if "citation" in sample.metadata.get("type", ""):
                    issues.append(f"[{sample.id}] 缺少来源引用")

            # 检查过短回答
            if len(sample.assistant_response) < 20:
                stats["short_response"] += 1
                issues.append(f"[{sample.id}] 回答过短")

            # 检查过长上下文
            total_context_len = sum(len(c["content"]) for c in sample.contexts)
            if total_context_len > 4000:
                stats["context_too_long"] += 1
                issues.append(f"[{sample.id}] 上下文过长: {total_context_len}字符")

            stats["passed"] += 1

        # 打印报告
        print(f"\n  [质量检查] {stats['total']} 条数据:")
        print(f"    通过: {stats['passed']}")
        if stats["empty_response"]:
            print(f"    空回答: {stats['empty_response']}")
        if stats["missing_citation"]:
            print(f"    缺少引用: {stats['missing_citation']}")
        if issues:
            print(f"    问题详情:")
            for issue in issues[:5]:
                print(f"      - {issue}")

        return stats


# ============================================================
# Section 4: 主流程
# ============================================================

def main():
    """主函数：演示RAG专用指令数据集构建全流程"""
    print("=" * 70)
    print("B2-01: RAG专用指令数据集 — 完整演示")
    print("=" * 70)

    # 1. 构建数据集
    print("\n[步骤1] 构建RAG专用指令数据...")
    builder = RAGInstructionBuilder()
    samples = builder.build_all()

    # 2. 展示样本
    print("\n[步骤2] 样本预览...")
    for i, sample in enumerate(samples):
        print(f"\n  --- 样本 {i+1}: {sample.id} ---")
        print(f"  类型: {sample.metadata.get('type', 'unknown')}")
        print(f"  用户查询: {sample.user_query[:60]}...")
        print(f"  上下文数: {len(sample.contexts)}")
        print(f"  回答预览: {sample.assistant_response[:80]}...")
        if i >= 2:
            print(f"\n  ... (共{len(samples)}条，展示前3条)")
            break

    # 3. 格式转换演示
    print("\n[步骤3] 格式转换演示...")
    converter = FormatConverter()
    sample = samples[0]  # 取第一个样本

    print("\n  [Alpaca格式]:")
    alpaca = converter.to_alpaca(sample)
    print(f"  {json.dumps(alpaca, ensure_ascii=False, indent=2)[:300]}...")

    print("\n  [ShareGPT格式]:")
    sharegpt = converter.to_sharegpt(sample)
    print(f"  {json.dumps(sharegpt, ensure_ascii=False, indent=2)[:300]}...")

    print("\n  [ChatML格式]:")
    chatml = converter.to_chatml(sample)
    print(f"  {chatml[:300]}...")

    # 4. 批量转换
    print("\n[步骤4] 批量转换到各格式...")
    for fmt in ["alpaca", "sharegpt", "chatml"]:
        converted = converter.convert_dataset(samples, fmt)
        print(f"  {fmt}: 转换了 {len(converted)} 条数据")

    # 5. 质量检查
    print("\n[步骤5] 数据集质量检查...")
    stats = DatasetQualityChecker.check(samples)

    # 6. 导出示例
    print("\n[步骤6] 导出示例数据集...")
    output_path = "demo_rag_instructions.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            record = {
                "id": sample.id,
                "instruction": sample.system_prompt,
                "input": sample.user_query,
                "output": sample.assistant_response,
                "contexts": sample.contexts,
                "metadata": sample.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  导出: {output_path} ({os.path.getsize(output_path)} bytes)")

    # 清理
    if os.path.exists(output_path):
        os.remove(output_path)

    print("\n" + "=" * 70)
    print("B2-01 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
