#!/usr/bin/env python3
"""
知识图谱RAG (Knowledge Graph-Enhanced RAG)
Hybrid Retrieval: Vector Search + Graph Traversal

特性:
  - 实体提取: 通过LLM从文档中提取实体
  - 关系提取: 因果/层级/关联/时序关系
  - 图构建: NetworkX (小规模) / Neo4j (生产)
  - 混合检索: 向量检索 + 图遍历
  - 图增强提示: 在上下文中包含实体关系
"""

import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Callable
from collections import defaultdict, deque


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class RelationType(Enum):
    """关系类型"""
    CAUSAL = "causal"             # 因果: A导致B
    HIERARCHICAL = "hierarchical" # 层级: A是B的子类
    ASSOCIATIVE = "associative"   # 关联: A与B相关
    TEMPORAL = "temporal"         # 时序: A先于B
    COMPOSITIONAL = "compositional"  # 组成: A由B组成
    COMPARATIVE = "comparative"   # 对比: A与B对比


@dataclass
class Entity:
    """实体"""
    entity_id: str
    name: str
    type: str                     # 实体类型: PERSON/ORG/TECHNOLOGY/CONCEPT/...
    properties: dict = field(default_factory=dict)
    mentions: list[str] = field(default_factory=list)  # 文档中的提及列表
    embeddings: Optional[list[float]] = None


@dataclass
class Relation:
    """关系"""
    relation_id: str
    source_entity: str            # 源实体ID
    target_entity: str            # 目标实体ID
    relation_type: RelationType
    description: str
    confidence: float             # 0.0-1.0
    source_documents: list[str]   # 来源文档ID列表


@dataclass
class KnowledgeGraph:
    """知识图谱"""
    entities: dict[str, Entity]         # entity_id -> Entity
    relations: dict[str, Relation]       # relation_id -> Relation
    adjacency: dict[str, list[str]]     # entity_id -> [relation_id, ...]
    entity_to_docs: dict[str, list[str]] # entity_id -> [doc_id, ...]


@dataclass
class KGRAGResult:
    """KG-RAG 结果"""
    query: str
    answer: str
    vector_passages: list[dict]         # 向量检索结果
    graph_entities: list[Entity]        # 图遍历发现的实体
    graph_relations: list[Relation]     # 相关关系
    fused_context: str                  # 融合后的上下文
    retrieval_method: str               # "vector_only", "graph_only", "hybrid"


# ============================================================
# 实体提取器 (Entity Extractor)
# ============================================================

class EntityExtractor:
    """实体提取器 - 从文档中提取实体"""

    def __init__(self, llm=None):
        self.llm = llm
        # 基于规则的实体类型模式
        self.entity_patterns = {
            "TECHNOLOGY": [
                r"(RAG|LLM|GPT|BERT|Transformer|Faiss|Milvus|ChromaDB|Pinecone|HNSW|BM25|Redis)",
            ],
            "CONCEPT": [
                r"(检索增强生成|向量数据库|熔断器|速率限制|知识图谱|语义搜索|嵌入模型)",
                r"(retrieval.augmented.generation|vector.database|circuit.breaker|rate.limit)",
            ],
            "PERSON": [
                r"([A-Z][a-z]+ [A-Z][a-z]+)",  # 简单人名模式
            ],
            "ORG": [
                r"(Meta|Google|Microsoft|OpenAI|Anthropic|Zilliz)",
            ],
        }

    def extract_from_document(self, doc_id: str, text: str) -> list[Entity]:
        """从文档中提取实体"""
        entities = []

        # 基于规则提取
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                import re
                matches = re.finditer(pattern, text)
                for match in matches:
                    name = match.group(0)
                    entity_id = self._generate_entity_id(name, entity_type)
                    entities.append(Entity(
                        entity_id=entity_id,
                        name=name,
                        type=entity_type,
                        mentions=[doc_id],
                    ))

        # 使用LLM提取更多实体（如果有）
        if self.llm:
            llm_entities = self._llm_extract(text)
            for ent_data in llm_entities:
                entity_id = self._generate_entity_id(ent_data["name"], ent_data["type"])
                entities.append(Entity(
                    entity_id=entity_id,
                    name=ent_data["name"],
                    type=ent_data.get("type", "CONCEPT"),
                    mentions=[doc_id],
                ))

        # 去重
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.entity_id not in seen:
                seen.add(entity.entity_id)
                unique_entities.append(entity)

        return unique_entities

    def _llm_extract(self, text: str) -> list[dict]:
        """使用LLM提取实体"""
        try:
            prompt = f"""请从以下文本中提取关键实体（技术名称、概念、组织、人物等）。
以JSON格式返回，每个实体包含name和type字段。

文本:
{text[:1500]}

实体列表 (JSON):"""
            result = self.llm.generate(prompt)
            # 尝试解析JSON
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                pass
        except Exception:
            pass
        return []

    @staticmethod
    def _generate_entity_id(name: str, entity_type: str) -> str:
        key = f"{entity_type}:{name.lower()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]


# ============================================================
# 关系提取器 (Relation Extractor)
# ============================================================

class RelationExtractor:
    """关系提取器 - 从文档中提取实体间关系"""

    def __init__(self, llm=None):
        self.llm = llm

        # 基于规则的关系模式
        self.relation_patterns = [
            (r"(\w+)导致(\w+)", RelationType.CAUSAL),
            (r"(\w+)是(\w+)的一种", RelationType.HIERARCHICAL),
            (r"(\w+)与(\w+)相关", RelationType.ASSOCIATIVE),
            (r"(\w+)先于(\w+)", RelationType.TEMPORAL),
            (r"(\w+)由(\w+)组成", RelationType.COMPOSITIONAL),
            (r"(\w+)与(\w+)对比", RelationType.COMPARATIVE),
            # 英文模式
            (r"(\w+) causes? (\w+)", RelationType.CAUSAL),
            (r"(\w+) is a type of (\w+)", RelationType.HIERARCHICAL),
            (r"(\w+) is related to (\w+)", RelationType.ASSOCIATIVE),
            (r"(\w+) consists of (\w+)", RelationType.COMPOSITIONAL),
        ]

    def extract_from_document(self, doc_id: str, text: str,
                              entities: list[Entity]) -> list[Relation]:
        """从文档中提取关系"""
        relations = []
        entity_names = {e.name: e.entity_id for e in entities}

        # 基于规则
        import re
        for pattern, rel_type in self.relation_patterns:
            for match in re.finditer(pattern, text):
                source_name = match.group(1)
                target_name = match.group(2)

                source_id = entity_names.get(source_name)
                target_id = entity_names.get(target_name)

                if source_id and target_id:
                    relation_id = hashlib.md5(
                        f"{source_id}:{target_id}:{rel_type.value}".encode()
                    ).hexdigest()[:16]

                    relations.append(Relation(
                        relation_id=relation_id,
                        source_entity=source_id,
                        target_entity=target_id,
                        relation_type=rel_type,
                        description=f"{source_name} → {target_name}",
                        confidence=0.8,
                        source_documents=[doc_id],
                    ))

        # 使用LLM提取（如果有）
        if self.llm:
            llm_relations = self._llm_extract(text, entities)
            relations.extend(llm_relations)

        return relations

    def _llm_extract(self, text: str, entities: list[Entity]) -> list[Relation]:
        """使用LLM提取关系"""
        entity_list = "\n".join(
            f"- {e.name} ({e.type})" for e in entities[:20]
        )

        try:
            prompt = f"""请从以下文本中提取实体之间的关系。
关系类型: causal(因果), hierarchical(层级), associative(关联), temporal(时序)

已知实体:
{entity_list}

文本:
{text[:1500]}

以JSON格式返回关系列表:
[{{"source": "实体1", "target": "实体2", "type": "关系类型", "confidence": 0.9}}]

关系列表 (JSON):"""
            result = self.llm.generate(prompt)
            try:
                data = json.loads(result)
                relations = []
                entity_map = {e.name: e.entity_id for e in entities}
                for item in data:
                    source_id = entity_map.get(item["source"])
                    target_id = entity_map.get(item["target"])
                    if source_id and target_id:
                        rel_type = RelationType(item.get("type", "associative"))
                        relations.append(Relation(
                            relation_id=hashlib.md5(
                                f"{source_id}:{target_id}:{rel_type.value}".encode()
                            ).hexdigest()[:16],
                            source_entity=source_id,
                            target_entity=target_id,
                            relation_type=rel_type,
                            description=item.get("description", f"{item['source']} → {item['target']}"),
                            confidence=item.get("confidence", 0.7),
                            source_documents=["llm_extracted"],
                        ))
                return relations
            except json.JSONDecodeError:
                pass
        except Exception:
            pass
        return []


# ============================================================
# 知识图谱构建器 (Knowledge Graph Builder)
# ============================================================

class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self, entity_extractor=None, relation_extractor=None, use_neo4j: bool = False):
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.relation_extractor = relation_extractor or RelationExtractor()
        self.use_neo4j = use_neo4j
        self.kg = KnowledgeGraph(
            entities={},
            relations={},
            adjacency={},
            entity_to_docs={},
        )

    def build_from_documents(self, documents: list[dict]) -> KnowledgeGraph:
        """从文档集合构建知识图谱

        Args:
            documents: [{"doc_id": ..., "text": ...}, ...]

        Returns:
            KnowledgeGraph: 构建好的知识图谱
        """
        print(f"[KG Build] 开始构建知识图谱 ({len(documents)} 个文档)")

        for doc in documents:
            doc_id = doc["doc_id"]
            text = doc["text"]

            # 提取实体
            entities = self.entity_extractor.extract_from_document(doc_id, text)

            # 提取关系
            relations = self.relation_extractor.extract_from_document(doc_id, text, entities)

            # 添加到图谱
            for entity in entities:
                if entity.entity_id not in self.kg.entities:
                    self.kg.entities[entity.entity_id] = entity
                else:
                    # 合并提及
                    existing = self.kg.entities[entity.entity_id]
                    existing.mentions.extend(entity.mentions)

                # 更新实体-文档映射
                if entity.entity_id not in self.kg.entity_to_docs:
                    self.kg.entity_to_docs[entity.entity_id] = []
                if doc_id not in self.kg.entity_to_docs[entity.entity_id]:
                    self.kg.entity_to_docs[entity.entity_id].append(doc_id)

            for relation in relations:
                if relation.relation_id not in self.kg.relations:
                    self.kg.relations[relation.relation_id] = relation

                # 更新邻接表
                source = relation.source_entity
                target = relation.target_entity

                if source not in self.kg.adjacency:
                    self.kg.adjacency[source] = []
                self.kg.adjacency[source].append(relation.relation_id)

                if target not in self.kg.adjacency:
                    self.kg.adjacency[target] = []
                self.kg.adjacency[target].append(relation.relation_id)

        print(f"[KG Build] 完成: {len(self.kg.entities)} 实体, {len(self.kg.relations)} 关系")
        return self.kg

    def build_networkx(self) -> Any:
        """构建NetworkX图（用于可视化和小规模分析）"""
        try:
            import networkx as nx

            G = nx.DiGraph()

            for entity_id, entity in self.kg.entities.items():
                G.add_node(entity_id, name=entity.name, type=entity.type)

            for rel_id, relation in self.kg.relations.items():
                G.add_edge(
                    relation.source_entity,
                    relation.target_entity,
                    type=relation.relation_type.value,
                    label=relation.description,
                    confidence=relation.confidence,
                )

            print(f"[NetworkX] 图构建: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
            return G
        except ImportError:
            print("networkx未安装，跳过NetworkX图构建")
            return None

    def visualize(self, output_path: str = "knowledge_graph.html"):
        """可视化知识图谱"""
        try:
            import networkx as nx
            from pyvis.network import Network

            G = self.build_networkx()
            if G is None:
                return

            net = Network(height="700px", width="100%", directed=True)

            for node, attrs in G.nodes(data=True):
                color = {
                    "TECHNOLOGY": "#4CAF50",
                    "CONCEPT": "#2196F3",
                    "PERSON": "#FF9800",
                    "ORG": "#9C27B0",
                }.get(attrs.get("type", ""), "#757575")
                net.add_node(node, label=attrs.get("name", node), color=color)

            for source, target, attrs in G.edges(data=True):
                net.add_edge(source, target, title=attrs.get("label", ""))

            net.save_graph(output_path)
            print(f"[Visualize] 图谱可视化已保存: {output_path}")
        except ImportError:
            print("pyvis未安装，跳过可视化")


# ============================================================
# 图遍历检索器 (Graph Traversal Retriever)
# ============================================================

class GraphRetriever:
    """图遍历检索器 - 基于图结构的检索"""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def traverse_from_entities(self, entity_ids: list[str],
                               max_depth: int = 2,
                               max_entities: int = 20) -> tuple[list[Entity], list[Relation]]:
        """从给定实体出发进行图遍历

        Args:
            entity_ids: 起始实体ID列表
            max_depth: 最大遍历深度
            max_entities: 最大返回实体数

        Returns:
            (发现的实体列表, 遍历到的关系列表)
        """
        visited_entities: set[str] = set()
        visited_relations: set[str] = set()
        queue = deque()

        # 初始化BFS
        for eid in entity_ids:
            if eid in self.kg.entities:
                queue.append((eid, 0))
                visited_entities.add(eid)

        # BFS遍历
        while queue and len(visited_entities) < max_entities:
            current_entity_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # 获取与该实体相关的所有关系
            relation_ids = self.kg.adjacency.get(current_entity_id, [])
            for rel_id in relation_ids:
                if rel_id in visited_relations:
                    continue
                visited_relations.add(rel_id)

                relation = self.kg.relations.get(rel_id)
                if relation is None:
                    continue

                # 探索关系的另一端
                neighbor_id = (
                    relation.target_entity
                    if relation.source_entity == current_entity_id
                    else relation.source_entity
                )

                if neighbor_id not in visited_entities and neighbor_id in self.kg.entities:
                    visited_entities.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        # 收集结果
        entities = [
            self.kg.entities[eid]
            for eid in visited_entities
            if eid in self.kg.entities
        ]
        relations = [
            self.kg.relations[rid]
            for rid in visited_relations
            if rid in self.kg.relations
        ]

        return entities, relations

    def find_path(self, source_entity_id: str, target_entity_id: str,
                  max_depth: int = 4) -> Optional[list[tuple[Entity, Relation, Entity]]]:
        """查找两个实体之间的路径

        Returns:
            路径列表 [(实体, 关系, 实体), ...] 或 None
        """
        if source_entity_id not in self.kg.entities or target_entity_id not in self.kg.entities:
            return None

        # BFS找最短路径
        queue = deque()
        queue.append((source_entity_id, []))
        visited = {source_entity_id}

        while queue:
            current_id, path = queue.popleft()

            if len(path) >= max_depth:
                continue

            for rel_id in self.kg.adjacency.get(current_id, []):
                relation = self.kg.relations.get(rel_id)
                if relation is None:
                    continue

                next_id = (
                    relation.target_entity
                    if relation.source_entity == current_id
                    else relation.source_entity
                )

                if next_id == target_entity_id:
                    # 找到目标
                    full_path = path + [
                        (self.kg.entities[current_id],
                         relation,
                         self.kg.entities[next_id])
                    ]
                    return full_path

                if next_id not in visited and next_id in self.kg.entities:
                    visited.add(next_id)
                    new_path = path + [
                        (self.kg.entities[current_id],
                         relation,
                         self.kg.entities[next_id])
                    ]
                    queue.append((next_id, new_path))

        return None

    def get_entity_relations_graph(self, entity_ids: list[str]) -> dict:
        """获取实体间的关系图数据（用于上下文增强）

        Returns:
            {"entities": [...], "relations": [...], "triplets": [...]}
        """
        entities, relations = self.traverse_from_entities(entity_ids, max_depth=1)

        triplets = []
        for rel in relations:
            source_entity = self.kg.entities.get(rel.source_entity)
            target_entity = self.kg.entities.get(rel.target_entity)
            if source_entity and target_entity:
                triplets.append({
                    "source": source_entity.name,
                    "relation": rel.relation_type.value,
                    "target": target_entity.name,
                    "confidence": rel.confidence,
                })

        return {
            "entities": [{"name": e.name, "type": e.type} for e in entities],
            "relations": [
                {
                    "source": self.kg.entities[r.source_entity].name if r.source_entity in self.kg.entities else "?",
                    "target": self.kg.entities[r.target_entity].name if r.target_entity in self.kg.entities else "?",
                    "type": r.relation_type.value,
                }
                for r in relations
            ],
            "triplets": triplets,
        }


# ============================================================
# KG-RAG 混合检索器 (Hybrid KG-RAG Retriever)
# ============================================================

class KGRAGRetriever:
    """KG-RAG 混合检索器: 向量检索 + 图遍历"""

    def __init__(self, kg: KnowledgeGraph, vector_searcher=None):
        self.kg = kg
        self.graph_retriever = GraphRetriever(kg)
        self.vector_searcher = vector_searcher  # 向量检索器

    def hybrid_retrieve(self, query: str, k_vector: int = 5,
                        k_graph: int = 10, max_graph_depth: int = 2) -> KGRAGResult:
        """混合检索: 向量+图遍历

        Args:
            query: 用户查询
            k_vector: 向量检索返回数量
            k_graph: 图遍历返回实体数
            max_graph_depth: 图遍历深度

        Returns:
            KGRAGResult: 混合检索结果
        """
        # 步骤1: 向量检索
        vector_passages = []
        if self.vector_searcher:
            vector_passages = self.vector_searcher(query, k_vector)
        else:
            vector_passages = self._dummy_vector_search(query, k_vector)

        # 步骤2: 从向量检索结果中识别实体
        extracted_entity_ids = self._extract_entities_from_passages(vector_passages)

        # 步骤3: 图遍历 - 扩展相关实体
        graph_entities, graph_relations = self.graph_retriever.traverse_from_entities(
            extracted_entity_ids,
            max_depth=max_graph_depth,
            max_entities=k_graph,
        )

        # 步骤4: 从图关系找到关联文档
        graph_passages = self._get_passages_for_entities(graph_entities)

        # 步骤5: 融合
        fused_context = self._fuse_contexts(
            query, vector_passages, graph_entities, graph_relations, graph_passages
        )

        return KGRAGResult(
            query=query,
            answer="",  # 稍后由生成器填充
            vector_passages=vector_passages,
            graph_entities=graph_entities,
            graph_relations=graph_relations,
            fused_context=fused_context,
            retrieval_method="hybrid",
        )

    def _extract_entities_from_passages(self, passages: list[dict]) -> list[str]:
        """从向量检索结果中匹配已知实体"""
        found_entity_ids = []

        for passage in passages:
            text = passage.get("text", "")
            for entity_id, entity in self.kg.entities.items():
                if entity.name.lower() in text.lower():
                    if entity_id not in found_entity_ids:
                        found_entity_ids.append(entity_id)

        return found_entity_ids[:5]  # 限制数量

    def _get_passages_for_entities(self, entities: list[Entity]) -> list[dict]:
        """获取与实体相关的文档段落"""
        passages = []
        seen_docs = set()

        for entity in entities:
            doc_ids = self.kg.entity_to_docs.get(entity.entity_id, [])
            for doc_id in doc_ids:
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    passages.append({
                        "doc_id": doc_id,
                        "entity": entity.name,
                        "type": "graph",
                    })

        return passages[:10]

    def _fuse_contexts(self, query: str, vector_passages: list[dict],
                       graph_entities: list[Entity], graph_relations: list[Relation],
                       graph_passages: list[dict]) -> str:
        """融合向量检索和图遍历的上下文"""
        parts = []

        # 向量检索结果
        if vector_passages:
            parts.append("## 直接检索结果\n")
            for i, p in enumerate(vector_passages[:3], 1):
                parts.append(f"{i}. {p.get('text', '')[:300]}\n")

        # 知识图谱关系
        if graph_entities:
            parts.append("\n## 知识图谱实体\n")
            entity_names = [e.name for e in graph_entities[:10]]
            parts.append(f"相关实体: {', '.join(entity_names)}\n")

        if graph_relations:
            parts.append("\n## 实体间关系\n")
            for rel in graph_relations[:10]:
                source_name = self.kg.entities.get(rel.source_entity, Entity("", "", ""))
                target_name = self.kg.entities.get(rel.target_entity, Entity("", "", ""))
                parts.append(
                    f"- {source_name.name} --[{rel.relation_type.value}]--> {target_name.name} "
                    f"(置信度: {rel.confidence:.2f})\n"
                )

        return "\n".join(parts)

    @staticmethod
    def _dummy_vector_search(query: str, k: int) -> list[dict]:
        """模拟向量搜索"""
        demo_results = [
            {"text": "RAG通过检索外部文档增强LLM回答。", "score": 0.9, "id": "d1"},
            {"text": "向量数据库存储高维向量用于相似度搜索。", "score": 0.85, "id": "d2"},
            {"text": "熔断器模式防止分布式系统中的级联故障。", "score": 0.8, "id": "d3"},
        ]
        return demo_results[:k]


# ============================================================
# KG-RAG 生成器
# ============================================================

class KGRAGenerator:
    """KG-RAG 生成器 - 图增强提示"""

    def __init__(self, llm=None):
        self.llm = llm

    def generate(self, result: KGRAGResult) -> str:
        """生成图增强的回答

        Args:
            result: 混合检索结果（包含融合上下文）

        Returns:
            增强的回答
        """
        prompt = self._build_graph_enhanced_prompt(
            result.query,
            result.fused_context,
            result.graph_entities,
            result.graph_relations,
        )

        if self.llm:
            return self.llm.generate(prompt)

        # 无LLM: 从上下文构建回答
        return self._build_basic_answer(result)

    def _build_graph_enhanced_prompt(self, query: str, fused_context: str,
                                     entities: list[Entity],
                                     relations: list[Relation]) -> str:
        """构建包含图信息的增强提示词"""
        prompt_parts = [
            "你是一个知识图谱增强的AI助手。请基于提供的检索内容和知识图谱关系回答用户问题。",
            "",
            f"## 用户问题: {query}",
            "",
            fused_context,
            "",
        ]

        # 添加关系三元组
        if relations:
            prompt_parts.append("## 知识图谱关系三元组")
            for rel in relations[:8]:
                src = self._get_entity_name(entities, rel.source_entity)
                tgt = self._get_entity_name(entities, rel.target_entity)
                prompt_parts.append(
                    f"- ({src}) --[{rel.relation_type.value}]--> ({tgt})"
                )
            prompt_parts.append("")

        prompt_parts.append("请根据以上信息给出准确、全面的中文回答，适当引用实体关系：")

        return "\n".join(prompt_parts)

    def _get_entity_name(self, entities: list[Entity], entity_id: str) -> str:
        for e in entities:
            if e.entity_id == entity_id:
                return e.name
        return entity_id[:8]

    @staticmethod
    def _build_basic_answer(result: KGRAGResult) -> str:
        """当LLM不可用时，构建基础回答"""
        parts = ["基于知识图谱的分析结果：\n"]

        if result.graph_entities:
            parts.append("**相关概念**: ")
            parts.append(", ".join(e.name for e in result.graph_entities[:5]))
            parts.append("\n\n")

        if result.graph_relations:
            parts.append("**概念关系**:\n")
            for rel in result.graph_relations[:5]:
                parts.append(f"- {rel.description}\n")
            parts.append("\n")

        if result.vector_passages:
            parts.append("**相关文档**:\n")
            for p in result.vector_passages[:3]:
                parts.append(f"- {p.get('text', '')[:200]}...\n")

        return "".join(parts)


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("知识图谱RAG (KG-RAG) - 演示运行")
    print("=" * 60)

    # 1. 准备文档
    documents = [
        {
            "doc_id": "doc_001",
            "text": "RAG（检索增强生成）是AI领域的重要技术。RAG使用向量数据库进行高效检索，常用的向量数据库包括Faiss和Milvus。向量数据库通过HNSW索引实现快速相似度搜索。RAG导致LLM回答质量提升。",
        },
        {
            "doc_id": "doc_002",
            "text": "熔断器模式是一种弹性设计模式。熔断器由CLOSED状态、OPEN状态和HALF_OPEN状态组成。熔断器防止级联故障的发生。熔断器与速率限制相关，两者都用于提升系统稳定性。",
        },
        {
            "doc_id": "doc_003",
            "text": "Self-RAG是RAG的一种高级形式。Self-RAG使用反思令牌进行评估。Self-RAG与CRAG对比，前者侧重自我反思，后者侧重检索纠正。知识图谱RAG是另一种高级RAG形式，使用知识图谱增强检索。",
        },
        {
            "doc_id": "doc_004",
            "text": "知识图谱由实体和关系组成。知识图谱RAG使用知识图谱进行增强检索。知识图谱可以构建为图结构。知识图谱与向量数据库相关，两者互补用于混合检索。",
        },
    ]

    # 2. 构建知识图谱
    print("\n--- 知识图谱构建 ---")

    class MockLLM:
        @staticmethod
        def generate(prompt):
            if "实体" in prompt or "entity" in prompt.lower():
                return json.dumps([
                    {"name": "RAG", "type": "TECHNOLOGY"},
                    {"name": "向量数据库", "type": "TECHNOLOGY"},
                ])
            if "关系" in prompt or "relation" in prompt.lower():
                return json.dumps([
                    {"source": "RAG", "target": "向量数据库", "type": "associative", "confidence": 0.95},
                ])
            if "分析" in prompt:
                return "基于知识图谱分析，这些概念之间有密切关联。"
            return "标准回答。"

    llm = MockLLM()
    builder = KnowledgeGraphBuilder(
        entity_extractor=EntityExtractor(llm=llm),
        relation_extractor=RelationExtractor(llm=llm),
    )

    kg = builder.build_from_documents(documents)

    print(f"\n实体列表 ({len(kg.entities)}):")
    for entity_id, entity in kg.entities.items():
        print(f"  [{entity.type}] {entity.name} (提及{len(entity.mentions)}次)")

    print(f"\n关系列表 ({len(kg.relations)}):")
    for rel_id, relation in kg.relations.items():
        src = kg.entities.get(relation.source_entity)
        tgt = kg.entities.get(relation.target_entity)
        if src and tgt:
            print(f"  {src.name} --[{relation.relation_type.value}]--> {tgt.name} (置信度:{relation.confidence:.2f})")

    # 3. 图遍历
    print("\n--- 图遍历 ---")
    retriever = GraphRetriever(kg)

    # 找到RAG实体的ID
    rag_entity_id = None
    for eid, entity in kg.entities.items():
        if entity.name.lower() == "rag":
            rag_entity_id = eid
            break

    if rag_entity_id:
        entities, relations = retriever.traverse_from_entities(
            [rag_entity_id], max_depth=2, max_entities=20
        )
        print(f"从 'RAG' 出发遍历:")
        print(f"  发现 {len(entities)} 个相关实体:")
        for e in entities:
            print(f"    - {e.name} ({e.type})")
        print(f"  发现 {len(relations)} 个关系")

    # 4. 路径查找
    print("\n--- 实体间路径查找 ---")

    # 找 "RAG" 和 "熔断器模式" 的ID
    cb_entity_id = None
    for eid, entity in kg.entities.items():
        if "熔断器" in entity.name:
            cb_entity_id = eid
            break

    if rag_entity_id and cb_entity_id:
        path = retriever.find_path(rag_entity_id, cb_entity_id, max_depth=4)
        if path:
            print(f"RAG → 熔断器模式 路径 (长度{len(path)}):")
            for step in path:
                src_entity, relation, tgt_entity = step
                print(f"  {src_entity.name} --[{relation.relation_type.value}]--> {tgt_entity.name}")
        else:
            print(f"RAG → 熔断器模式: 未找到路径（可能距离太远）")

    # 5. 混合检索
    print("\n--- 混合检索 (向量+图) ---")
    kg_retriever = KGRAGRetriever(kg)

    result = kg_retriever.hybrid_retrieve(
        "RAG系统中如何使用知识图谱？",
        k_vector=5,
        k_graph=10,
        max_graph_depth=2,
    )

    print(f"检索方式: {result.retrieval_method}")
    print(f"向量段落: {len(result.vector_passages)}")
    print(f"图实体:   {len(result.graph_entities)}")
    print(f"图关系:   {len(result.graph_relations)}")

    if result.graph_entities:
        print(f"\n图实体:")
        for e in result.graph_entities[:5]:
            print(f"  - {e.name} ({e.type})")

    if result.graph_relations:
        print(f"\n图关系:")
        for r in result.graph_relations[:5]:
            src = kg.entities.get(r.source_entity)
            tgt = kg.entities.get(r.target_entity)
            if src and tgt:
                print(f"  {src.name} --[{r.relation_type.value}]--> {tgt.name}")

    # 6. 图增强生成
    print("\n--- 图增强生成 ---")
    generator = KGRAGenerator(llm=llm)
    answer = generator.generate(result)
    print(f"回答: {answer[:300]}...")

    # 7. 融合上下文
    print("\n--- 融合上下文 ---")
    print(result.fused_context[:500])
    print("...")

    # 8. NetworkX图构建（可选）
    print("\n--- NetworkX图构建 ---")
    try:
        G = builder.build_networkx()
        if G:
            print(f"节点数: {G.number_of_nodes()}")
            print(f"边数: {G.number_of_edges()}")
    except Exception as e:
        print(f"NetworkX图构建跳过: {e}")

    # 9. 图查询统计
    print("\n--- 统计 ---")
    # 度数最高的实体
    degree_count = {eid: len(kg.adjacency.get(eid, [])) for eid in kg.entities}
    top_entities = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)[:5]
    print("度数最高的实体:")
    for eid, deg in top_entities:
        entity = kg.entities.get(eid)
        if entity:
            print(f"  {entity.name}: {deg} 个关系")

    print("\n" + "=" * 60)
    print("知识图谱RAG (KG-RAG) 演示完成！")
    print("=" * 60)
