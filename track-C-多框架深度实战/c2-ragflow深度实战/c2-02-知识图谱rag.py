#!/usr/bin/env python3
"""
C2-02：RAGFlow 知识图谱增强 RAG（GraphRAG）
===============================================
知识图谱 + RAG = GraphRAG，是解决"多跳推理"和"全局理解"问题的
关键技术路径。RAGFlow 内置了知识图谱构建和查询能力。

本文件演示：
1. 从非结构化文本中自动构建知识图谱
2. 实体提取 + 关系抽取 + 图谱存储
3. 基于知识图谱的多跳推理检索
4. 向量检索 + 图谱检索的混合策略
5. GraphRAG vs 传统 RAG 的互补关系

依赖：pip install ragflow-client networkx
"""

import json
import time
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict


# ============================================================================
# 第一部分：知识图谱数据结构和构建器
# ============================================================================

class KnowledgeGraph:
    """
    简易内存知识图谱（演示用）。

    生产环境使用 Neo4j、NebulaGraph 等专业图数据库。
    RAGFlow 内置了基于 Neo4j 的知识图谱索引。
    """

    def __init__(self):
        self.entities: Dict[str, dict] = {}          # entity_id → {name, type, props}
        self.relations: List[dict] = []               # [{subject, predicate, object, props}]
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # adjacency[entity_id] = [(neighbor_id, relation_type), ...]

    def add_entity(self, entity_id: str, name: str,
                   entity_type: str, properties: dict = None) -> str:
        """添加实体节点。"""
        self.entities[entity_id] = {
            "name": name,
            "type": entity_type,
            "properties": properties or {},
        }
        return entity_id

    def add_relation(self, subject: str, predicate: str,
                     obj: str, properties: dict = None) -> None:
        """添加关系边。"""
        relation = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "properties": properties or {},
        }
        self.relations.append(relation)
        self.adjacency[subject].append((obj, predicate))
        self.adjacency[obj].append((subject, f"被{predicate}"))

    def get_neighbors(self, entity_id: str, depth: int = 1) -> List[dict]:
        """获取实体的邻居（支持多跳）。"""
        visited = set()
        result = []
        queue = [(entity_id, 0)]

        while queue:
            current, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)
            if current != entity_id:
                result.append({"entity_id": current, "depth": d,
                               "entity": self.entities.get(current, {})})
            for neighbor, rel_type in self.adjacency.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, d + 1))
        return result

    def get_entity_by_name(self, name: str) -> Optional[str]:
        """按名称查找实体ID。"""
        for eid, info in self.entities.items():
            if info["name"] == name:
                return eid
        return None

    def get_statistics(self) -> dict:
        """获取图谱统计信息。"""
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entity_types": list(set(e["type"] for e in self.entities.values())),
            "relation_types": list(set(r["predicate"] for r in self.relations)),
        }


class EntityRelationExtractor:
    """
    实体关系提取器（模拟 LLM 驱动的提取过程）。

    实际 RAGFlow 中，这一步由 LLM（如 GPT-4o）完成：
    1. 输入文本 → LLM 识别实体（人名/组织/地点/技术/产品...）
    2. LLM 识别实体间关系（属于/位于/使用/开发/合作...）
    3. 输出结构化三元组 (subject, predicate, object)
    """

    @staticmethod
    def extract_from_text(text: str) -> Tuple[List[dict], List[dict]]:
        """
        从文本中提取实体和关系。

        演示用规则提取，实际使用 LLM 的 function calling：
        {
            "entities": [
                {"name": "某某科技", "type": "公司", "properties": {...}},
                ...
            ],
            "relations": [
                {"subject": "某某科技", "predicate": "开发", "object": "智能客服"},
                ...
            ]
        }
        """
        # 演示数据：模拟 LLM 提取结果
        entities = [
            {"id": "ent_1", "name": "某大型银行", "type": "企业",
             "properties": {"行业": "金融", "规模": "大型"}},
            {"id": "ent_2", "name": "智能客服平台", "type": "产品",
             "properties": {"技术": "RAG+LLM", "部署方式": "私有化"}},
            {"id": "ent_3", "name": "RAGFlow", "type": "技术框架",
             "properties": {"开源": True, "语言": "Python"}},
            {"id": "ent_4", "name": "GPT-4o", "type": "大语言模型",
             "properties": {"提供商": "OpenAI", "参数规模": "未公开"}},
            {"id": "ent_5", "name": "知识图谱引擎", "type": "技术组件",
             "properties": {"底层存储": "Neo4j"}},
            {"id": "ent_6", "name": "某某科技", "type": "企业",
             "properties": {"行业": "AI科技", "融资": "C轮"}},
            {"id": "ent_7", "name": "北京数据中心", "type": "基础设施",
             "properties": {"GPU": "A100×8"}},
            {"id": "ent_8", "name": "金融合规审查", "type": "业务场景",
             "properties": {"领域": "金融", "监管要求": "高"}},
        ]

        relations = [
            {"subject": "ent_1", "predicate": "采购", "object": "ent_2"},
            {"subject": "ent_2", "predicate": "基于", "object": "ent_3"},
            {"subject": "ent_3", "predicate": "调用", "object": "ent_4"},
            {"subject": "ent_3", "predicate": "包含", "object": "ent_5"},
            {"subject": "ent_6", "predicate": "开发", "object": "ent_3"},
            {"subject": "ent_6", "predicate": "拥有", "object": "ent_7"},
            {"subject": "ent_2", "predicate": "服务于", "object": "ent_8"},
            {"subject": "ent_1", "predicate": "面临", "object": "ent_8"},
        ]

        return entities, relations

    @staticmethod
    def extract_from_documents(documents: List[str]) -> Tuple[List[dict], List[dict]]:
        """从多篇文档中提取实体和关系（批量处理）。"""
        all_entities = []
        all_relations = []

        for doc_text in documents:
            entities, relations = EntityRelationExtractor.extract_from_text(doc_text)
            all_entities.extend(entities)
            all_relations.extend(relations)

        # 去重
        seen_entities = {}
        unique_entities = []
        for e in all_entities:
            if e["name"] not in seen_entities:
                seen_entities[e["name"]] = e["id"]
                unique_entities.append(e)

        return unique_entities, all_relations


# ============================================================================
# 第二部分：知识图谱构建演示
# ============================================================================

def demo_graph_construction():
    """模拟从文档构建知识图谱的完整流程。"""
    print("=" * 60)
    print("【1】知识图谱构建 —— 从文本到图谱")
    print("=" * 60)

    # 模拟文档
    documents = [
        """
        某某科技有限公司开发了 RAGFlow 开源框架。
        RAGFlow 是一个检索增强生成框架，内置了知识图谱引擎。
        该框架调用了 GPT-4o 大语言模型进行实体识别和关系抽取。
        某某科技拥有北京数据中心，配备 8 块 A100 GPU。
        """,
        """
        某大型银行采购了智能客服平台，用于金融合规审查场景。
        该智能客服平台基于 RAGFlow 框架构建，采用私有化部署方式。
        智能客服平台服务于金融合规审查，帮助银行降低人工审查成本。
        """,
    ]

    # 步骤1：实体关系提取
    print("\n[步骤1] LLM 驱动的实体关系提取...")
    entities, relations = EntityRelationExtractor.extract_from_documents(documents)

    print(f"  提取到实体: {len(entities)} 个")
    for e in entities:
        print(f"    [{e['type']}] {e['name']} (ID: {e['id']})")

    print(f"\n  提取到关系: {len(relations)} 条")
    for r in relations:
        subj_name = next(e["name"] for e in entities if e["id"] == r["subject"])
        obj_name = next(e["name"] for e in entities if e["id"] == r["object"])
        print(f"    {subj_name} --[{r['predicate']}]--> {obj_name}")

    # 步骤2：构建知识图谱
    print("\n[步骤2] 构建知识图谱...")
    kg = KnowledgeGraph()

    for e in entities:
        kg.add_entity(e["id"], e["name"], e["type"], e.get("properties", {}))

    for r in relations:
        kg.add_relation(r["subject"], r["predicate"], r["object"])

    stats = kg.get_statistics()
    print(f"  图谱统计: {stats}")

    return kg


# ============================================================================
# 第三部分：多跳推理检索
# ============================================================================

def demo_multi_hop_reasoning(kg: KnowledgeGraph):
    """基于知识图谱的多跳推理检索。"""
    print("\n" + "=" * 60)
    print("【2】多跳推理 —— 图遍历检索")
    print("=" * 60)

    print("""
    多跳推理示例：

    问题："智能客服平台调用的大语言模型是什么？"

    推理链：
      [智能客服平台] --[基于]--> [RAGFlow]
      [RAGFlow]      --[调用]--> [GPT-4o]

    答案：GPT-4o (2跳推理完成)
    """)

    # 从实体名称找到实体ID
    start_entity_name = "智能客服平台"
    target_entity_name = "GPT-4o"

    start_id = kg.get_entity_by_name(start_entity_name)
    target_id = kg.get_entity_by_name(target_entity_name)

    if start_id and target_id:
        print(f"  起始实体: {start_entity_name} (ID: {start_id})")
        print(f"  目标实体: {target_entity_name} (ID: {target_id})")

    # 查找路径（BFS）
    print(f"\n  >>> 从 {start_entity_name} 出发的所有邻居（深度2）:")
    neighbors = kg.get_neighbors(start_id, depth=2)
    for n in neighbors:
        e_name = n["entity"].get("name", "unknown")
        print(f"    距离 {n['depth']}: {e_name}")

    # 演示几个实际的多跳查询
    queries = [
        "智能客服平台的底层框架是什么？该框架调用了哪个大模型？",
        "某某科技拥有什么基础设施？该基础设施配置如何？",
        "智能客服平台服务于什么业务场景？这个场景有什么监管要求？",
    ]

    print("\n  >>> 多跳查询示例:")
    for q in queries:
        print(f"\n    问题: {q}")
        # 实际中，这是 GraphRAG 检索器的核心逻辑：
        # 1. 实体链接（找到问题涉及的实体）
        # 2. 图遍历（沿关系边游走 N 跳）
        # 3. 子图提取（收集路径上的所有信息）
        # 4. 文本化（将子图转换为 LLM 可读的文本）
        # 5. 答案生成（LLM 基于子图信息回答）
        print(f"    GraphRAG 回答: (模拟) 根据知识图谱的多跳推理，"
              f"可以确定回答...")


# ============================================================================
# 第四部分：向量检索 + 图谱检索混合
# ============================================================================

def demo_hybrid_vector_graph(kg: KnowledgeGraph):
    """混合检索策略：向量检索（语义相似度） + 图谱检索（结构化推理）。"""
    print("\n" + "=" * 60)
    print("【3】混合检索 —— 向量 + 图谱")
    print("=" * 60)

    print("""
    混合检索的三种策略：

    策略A：并行检索 + 结果融合
      ┌───────────┐    ┌───────────┐
      │ 向量检索   │    │ 图谱检索   │
      │ (语义相似) │    │ (图遍历)   │
      └─────┬─────┘    └─────┬─────┘
            │                │
            └───────┬────────┘
                    │
              [结果融合/重排序]
                    │
                最终答案

    策略B：图谱优先（先实体链接，再向量检索缩小范围）
      查询 → 实体链接 → 子图提取 → 限定范围内的向量检索

    策略C：向量优先（先向量检索，再图谱补充关系上下文）
      查询 → 向量检索 → 获取相关实体 → 图谱扩展关系 → 增强上下文
    """)

    # 演示策略A：并行融合
    print("  >>> 演示策略A：并行检索 + RRF 融合")
    print("    查询: '某某科技开发的框架用于什么业务？'")
    print()
    print("    向量检索结果（语义相似度 Top 3）:")
    print("      1. [0.92] 某大型银行采购了智能客服平台...")
    print("      2. [0.85] 某某科技开发了 RAGFlow 开源框架...")
    print("      3. [0.78] 智能客服平台服务于金融合规审查...")
    print()
    print("    图谱检索结果（2跳邻居）:")
    print("      某某科技 → 开发 → RAGFlow → 基于 → 智能客服平台")
    print("      智能客服平台 → 服务于 → 金融合规审查")
    print()
    print("    RRF 融合后（得分调整）:")
    print("      1. [合并分 0.95] 某某科技 → RAGFlow → 智能客服平台 → 金融合规审查")
    print("      2. [合并分 0.88] 某大型银行采购了智能客服平台...")


# ============================================================================
# 第五部分：GraphRAG 完整管道
# ============================================================================

class GraphRAGPipeline:
    """
    GraphRAG 完整管道。

    集成实体提取 → 图谱构建 → 混合检索 → 答案生成的全流程。
    """

    def __init__(self):
        self.kg = KnowledgeGraph()
        self.entity_extractor = EntityRelationExtractor()
        self.documents_processed = 0

    def ingest_documents(self, documents: List[str]):
        """文档摄入：提取实体关系，构建图谱。"""
        print(f"  [GraphRAG] 摄入 {len(documents)} 篇文档...")

        entities, relations = self.entity_extractor.extract_from_documents(documents)

        for e in entities:
            self.kg.add_entity(e["id"], e["name"], e["type"], e.get("properties", {}))

        for r in relations:
            self.kg.add_relation(r["subject"], r["predicate"], r["object"])

        self.documents_processed += len(documents)
        stats = self.kg.get_statistics()
        print(f"  [GraphRAG] 图谱更新: {stats['entity_count']} 实体, "
              f"{stats['relation_count']} 关系")

    def retrieve(self, query: str, mode: str = "hybrid",
                 graph_depth: int = 2) -> dict:
        """
        检索接口。

        mode 可选值:
        - "vector_only": 仅向量检索
        - "graph_only": 仅图谱检索
        - "hybrid": 向量 + 图谱混合
        """
        print(f"  [GraphRAG] 检索 (mode={mode}): '{query}'")

        graph_context = []
        vector_context = []

        # 图谱检索
        if mode in ("graph_only", "hybrid"):
            # 1. 实体链接（简化：关键词匹配）
            linked_entities = []
            for eid, info in self.kg.entities.items():
                if info["name"] in query:
                    linked_entities.append(eid)

            if not linked_entities:
                # 如果没有精确匹配，用模糊策略
                for eid, info in self.kg.entities.items():
                    if len(info["name"]) >= 2 and info["name"][:2] in query:
                        linked_entities.append(eid)

            # 2. 图遍历
            for eid in linked_entities[:3]:  # 最多追踪3个起始实体
                neighbors = self.kg.get_neighbors(eid, depth=graph_depth)
                for n in neighbors:
                    graph_context.append({
                        "source": "knowledge_graph",
                        "content": f"{self.kg.entities[eid]['name']}"
                                   f" -- 关系 --> "
                                   f"{n['entity'].get('name', 'unknown')}"
                                   f" (距离: {n['depth']})",
                    })

        # 向量检索（模拟）
        if mode in ("vector_only", "hybrid"):
            vector_context = [
                {"source": "vector_search", "content": f"与'{query}'相关的向量检索结果"}
            ]

        return {
            "query": query,
            "graph_context": graph_context,
            "vector_context": vector_context,
            "total_sources": len(graph_context) + len(vector_context),
        }

    def query(self, question: str) -> str:
        """端到端查询。"""
        result = self.retrieve(question, mode="hybrid")

        # 构建综合上下文
        context_parts = []
        for ctx in result["graph_context"]:
            context_parts.append(f"[知识图谱] {ctx['content']}")
        for ctx in result["vector_context"]:
            context_parts.append(f"[文档检索] {ctx['content']}")

        # LLM 生成回答（模拟）
        graph_info_count = len(result["graph_context"])
        vector_info_count = len(result["vector_context"])

        answer = (
            f"[GraphRAG 回答] 基于 {graph_info_count} 条图谱信息 "
            f"和 {vector_info_count} 条文档信息，综合回答：'{question}'"
        )
        return answer


def demo_graphrag_pipeline():
    """GraphRAG 管道端到端演示。"""
    print("\n" + "=" * 60)
    print("【4】GraphRAG 完整管道端到端演示")
    print("=" * 60)

    pipeline = GraphRAGPipeline()

    # 文档摄入
    sample_docs = [
        "某某科技开发了RAGFlow框架。RAGFlow内置知识图谱引擎，调用GPT-4o大模型。",
        "某大型银行采购了智能客服平台，该平台基于RAGFlow构建，用于金融合规审查。",
        "某某科技拥有北京数据中心，配备8块A100 GPU，用于模型训练和推理。",
    ]
    pipeline.ingest_documents(sample_docs)

    # 查询
    questions = [
        "智能客服平台使用了什么技术？",
        "某某科技和某大型银行有什么关系？",
        "RAGFlow框架调用了哪个大语言模型？",
    ]

    for q in questions:
        print(f"\n  问题: {q}")
        answer = pipeline.query(q)
        print(f"  {answer}")

    return pipeline


# ============================================================================
# 第六部分：GraphRAG vs 传统RAG
# ============================================================================

def print_graphrag_vs_traditional():
    """对比 GraphRAG 和传统 RAG。"""
    print("\n" + "=" * 60)
    print("【5】GraphRAG vs 传统 RAG 对比")
    print("=" * 60)

    comparison = """
┌─────────────────────┬──────────────────┬─────────────────────────┐
│ 维度                │ 传统 RAG         │ GraphRAG                │
├─────────────────────┼──────────────────┼─────────────────────────┤
│ 核心数据结构        │ 向量索引          │ 实体关系图 + 向量索引    │
│ 检索方式            │ 语义相似度        │ 图遍历 + 语义相似度     │
│ 单跳查询            │ ✅ 优秀           │ ✅ 优秀                  │
│ 多跳推理            │ ❌ 困难           │ ✅ 擅长                  │
│ 全局总结            │ ❌ 片面           │ ✅ 社区摘要              │
│ 结构化理解          │ ❌ 无             │ ✅ 实体关系显式建模       │
│ 构建成本            │ 低                │ 高（需 LLM 提取）        │
│ 存储需求            │ 向量库            │ 向量库 + 图数据库        │
│ 更新维护            │ 简单              │ 复杂（实体消歧/关系更新）│
│ 适用场景            │ 事实检索/问答      │ 推理/分析/知识发现       │
└─────────────────────┴──────────────────┴─────────────────────────┘
    """
    print(comparison)

    print("\n【互补关系】")
    print("  GraphRAG 不是传统 RAG 的替代品，而是补充。")
    print("  - 事实类问题 → 传统 RAG（快速、低成本）")
    print("  - 推理类问题 → GraphRAG（准确、可解释）")
    print("  - 生产方案 → 两者混合（路由判断问题类型）")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("RAGFlow 知识图谱增强 RAG 实战演示")
    print("=" * 60)
    print("核心思想：用知识图谱补齐传统 RAG 的结构化推理短板\n")

    kg = demo_graph_construction()
    demo_multi_hop_reasoning(kg)
    demo_hybrid_vector_graph(kg)
    pipeline = demo_graphrag_pipeline()
    print_graphrag_vs_traditional()

    print("\n[完成] 知识图谱 RAG 演示结束。")
    print("  [提示] RAGFlow 内置了基于 Neo4j 的 GraphRAG 引擎。")
    print("  生产环境: docker-compose up ragflow neo4j")
