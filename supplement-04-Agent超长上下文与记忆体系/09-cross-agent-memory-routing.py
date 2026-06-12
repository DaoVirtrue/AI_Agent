#!/usr/bin/env python3
"""
跨Agent记忆路由 (Cross-Agent Memory Routing)
=================================================
实现多个专业Agent之间的联邦记忆查询，包括查询路由、
结果合并和跨Agent记忆检索。

架构：
  ┌─────────────┐
  │ Query Router │──── 路由分发 ────┬── WeatherAgent
  └─────────────┘                  ├── FinanceAgent
        │                          ├── TravelAgent
  ┌─────▼────────┐                 └── GeneralAgent
  │ Result Merger│──── 结果合并 ────→ 最终响应
  └──────────────┘

核心组件：
  - MemoryFederation: 记忆联邦（Agent注册与管理）
  - QueryRouter: 查询路由器（意图→Agent映射）
  - ResultMerger: 结果合并器（多源结果融合）
  - CrossAgentMemoryRouter: 主路由协调器
"""

from __future__ import annotations

import time
import json
import uuid
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict, OrderedDict


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CrossAgentResult:
    """跨Agent查询结果"""
    agent_id: str
    query: str
    results: List[Dict[str, Any]]
    confidence: float            # 置信度 [0, 1]
    latency_ms: float            # 延迟（毫秒）
    source_type: str             # 来源类型: "exact", "semantic", "fuzzy"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """Agent能力描述"""
    agent_id: str
    name: str
    expertise: List[str]         # 专业领域关键词
    memory_size: int             # 记忆容量
    endpoint: str                # 查询端点
    priority: float = 0.5        # 优先级
    health_status: str = "healthy"


@dataclass
class QueryPlan:
    """查询执行计划"""
    query_id: str
    original_query: str
    routed_agents: List[str]     # 路由到的Agent列表
    sub_queries: Dict[str, str]  # {agent_id: reformatted_query}
    strategy: str                # 执行策略: "parallel", "sequential", "cascade"
    created_at: float = field(default_factory=time.time)


# ============================================================================
# 记忆联邦
# ============================================================================

class MemoryFederation:
    """
    记忆联邦管理器。

    负责：
    - Agent注册与发现
    - 健康检查
    - 联邦查询调度
    """

    def __init__(self):
        self._agents: Dict[str, AgentCapability] = {}
        self._agent_memories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._query_history: List[Dict[str, Any]] = []

    def register_agent(self, capability: AgentCapability):
        """注册一个Agent到联邦"""
        self._agents[capability.agent_id] = capability
        print(f"  [联邦] Agent '{capability.name}' 已注册 "
              f"(专长: {', '.join(capability.expertise[:3])})")

    def add_agent_memory(self, agent_id: str,
                          memory: Dict[str, Any]):
        """向Agent的记忆库添加一条记忆"""
        self._agent_memories[agent_id].append({
            **memory,
            "stored_at": time.time(),
        })

    def federated_query(self, query: str,
                         agent_ids: List[str],
                         top_k: int = 5) -> List[CrossAgentResult]:
        """
        联邦查询：向多个Agent发送查询并收集结果。

        Args:
            query: 查询文本
            agent_ids: 目标Agent列表
            top_k: 每个Agent返回的结果数

        Returns:
            各Agent返回的结果列表
        """
        results = []

        for agent_id in agent_ids:
            agent = self._agents.get(agent_id)
            if not agent or agent.health_status != "healthy":
                continue

            start_time = time.time()

            # 模拟查询Agent记忆库
            mem_results = self._search_agent_memory(agent_id, query, top_k)
            confidence = self._compute_confidence(query, agent.expertise, mem_results)

            latency = (time.time() - start_time) * 1000

            results.append(CrossAgentResult(
                agent_id=agent_id,
                query=query,
                results=mem_results,
                confidence=confidence,
                latency_ms=round(latency, 2),
                source_type="semantic",
            ))

        # 记录查询历史
        self._query_history.append({
            "query": query,
            "agents": agent_ids,
            "results_count": len(results),
            "timestamp": time.time(),
        })

        return results

    def _search_agent_memory(self, agent_id: str, query: str,
                              top_k: int) -> List[Dict[str, Any]]:
        """
        搜索Agent记忆库（模拟语义搜索）。

        实际实现中会使用向量相似度搜索。
        这里使用基于关键词重叠的TF-IDF式评分。
        """
        memories = self._agent_memories.get(agent_id, [])
        if not memories:
            return []

        query_terms = set(query.lower().split())
        scored = []

        for mem in memories:
            mem_text = mem.get("content", "")
            mem_terms = set(mem_text.lower().split())
            overlap = query_terms & mem_terms
            jaccard = len(overlap) / max(len(query_terms | mem_terms), 1)
            tfidf_sim = len(overlap) / max(len(query_terms), 1)

            score = 0.5 * jaccard + 0.5 * tfidf_sim
            scored.append((score, mem))

        # 按分数排序
        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = [{"score": round(s, 4), **mem} for s, mem in scored[:top_k]]
        return top_results

    def _compute_confidence(self, query: str, expertise: List[str],
                             results: List[Dict]) -> float:
        """计算Agent对查询的置信度"""
        query_lower = query.lower()

        # 专长匹配度
        expertise_match = sum(
            1 for exp in expertise if any(t in query_lower for t in exp.lower().split())
        )
        expertise_score = min(expertise_match / max(len(expertise), 1), 1.0)

        # 结果质量分
        result_score = min(len(results) / 5.0, 1.0)  # 最多5个结果满分
        if results:
            avg_relevance = sum(r.get("score", 0.5) for r in results) / len(results)
        else:
            avg_relevance = 0.0

        # 综合置信度
        confidence = 0.3 * expertise_score + 0.3 * result_score + 0.4 * avg_relevance
        return round(confidence, 4)

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有注册的Agent"""
        return [
            {
                "id": a.agent_id,
                "name": a.name,
                "expertise": a.expertise,
                "health": a.health_status,
            }
            for a in self._agents.values()
        ]


# ============================================================================
# 查询路由器
# ============================================================================

class QueryRouter:
    """
    查询路由器。

    根据查询内容和Agent能力，决定将查询路由到哪些Agent。
    """

    def __init__(self, federation: MemoryFederation):
        self.federation = federation
        self._routing_history: List[Dict[str, Any]] = []

    def route(self, query: str) -> List[str]:
        """
        将查询路由到最合适的Agent。

        路由策略：
        1. 关键词匹配Agent专长领域
        2. 路由到1-3个最匹配的Agent
        3. 如果没有匹配，路由到通用Agent

        Args:
            query: 查询文本

        Returns:
            目标Agent ID列表
        """
        query_lower = query.lower()
        scores: Dict[str, float] = {}

        for agent in self.federation._agents.values():
            if agent.health_status != "healthy":
                continue

            score = 0.0
            # 专长关键词匹配
            for exp in agent.expertise:
                exp_keywords = exp.lower().split()
                for kw in exp_keywords:
                    if kw in query_lower:
                        score += 1.0
                # 部分匹配
                for kw_part in exp_keywords:
                    if len(kw_part) > 2:
                        for qw in query_lower.split():
                            if kw_part in qw:
                                score += 0.3

            # 查询长度因子
            score += math.log(len(query) + 1) * 0.1

            scores[agent.agent_id] = score

        # 选择top-N Agent（至少选1个）
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 过滤掉分数为0的
        relevant = [(aid, s) for aid, s in sorted_agents if s > 0]

        if not relevant:
            # 没有匹配的Agent，使用通用Agent
            for agent in self.federation._agents.values():
                if "general" in [e.lower() for e in agent.expertise]:
                    return [agent.agent_id]
            # 返回第一个健康的Agent
            healthy = [a.agent_id for a in self.federation._agents.values()
                       if a.health_status == "healthy"]
            return healthy[:1]

        selected = [aid for aid, _ in relevant[:3]]
        self._routing_history.append({
            "query": query,
            "selected": selected,
            "scores": dict(relevant[:3]),
            "timestamp": time.time(),
        })

        return selected

    def reformulate_query(self, query: str, agent: AgentCapability) -> str:
        """
        为特定Agent重新组织查询内容。

        某些Agent可能需要特定格式的查询（如带日期范围、位置信息等）。

        Returns:
            重新组织后的查询文本
        """
        # 根据Agent专长添加上下文
        if "weather" in [e.lower() for e in agent.expertise]:
            return f"[WEATHER_QUERY] {query} [包含温度、湿度、降水概率]"
        elif "finance" in [e.lower() for e in agent.expertise]:
            return f"[FINANCE_QUERY] {query} [包含实时数据、趋势分析]"
        elif "travel" in [e.lower() for e in agent.expertise]:
            return f"[TRAVEL_QUERY] {query} [包含交通、住宿、景点信息]"
        else:
            return f"[GENERAL_QUERY] {query}"


# ============================================================================
# 结果合并器
# ============================================================================

class ResultMerger:
    """
    结果合并器。

    将来自不同Agent的查询结果合并为统一的响应。
    """

    def merge(self, results: List[CrossAgentResult],
               strategy: str = "confidence_weighted") -> List[Dict[str, Any]]:
        """
        合并多个Agent的结果。

        合并策略：
        - "confidence_weighted": 按置信度加权排序
        - "round_robin": 轮询式交替排列
        - "dedup_first": 去重优先

        Args:
            results: 各Agent的查询结果
            strategy: 合并策略

        Returns:
            合并后的排序结果列表
        """
        if strategy == "confidence_weighted":
            return self._merge_confidence_weighted(results)
        elif strategy == "round_robin":
            return self._merge_round_robin(results)
        elif strategy == "dedup_first":
            return self._merge_dedup_first(results)
        else:
            return self._merge_confidence_weighted(results)

    def _merge_confidence_weighted(self,
                                     results: List[CrossAgentResult]
                                     ) -> List[Dict[str, Any]]:
        """按置信度加权合并"""
        merged = []
        seen_ids = set()

        for r in sorted(results, key=lambda x: x.confidence, reverse=True):
            for item in r.results:
                item_id = item.get("id", item.get("content", str(item)))
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    merged.append({
                        **item,
                        "source_agent": r.agent_id,
                        "agent_confidence": r.confidence,
                        "weighted_score": item.get("score", 0.5) * r.confidence,
                    })

        # 按加权分数排序
        merged.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        return merged

    def _merge_round_robin(self,
                            results: List[CrossAgentResult]
                            ) -> List[Dict[str, Any]]:
        """轮询式合并（公平展示各Agent结果）"""
        all_items = [
            {**item, "source_agent": r.agent_id}
            for r in results
            for item in r.results
        ]

        # 按Agent分组
        by_agent: Dict[str, List[Dict]] = defaultdict(list)
        for item in all_items:
            by_agent[item["source_agent"]].append(item)

        # 轮询取
        merged = []
        agent_list = list(by_agent.keys())
        max_len = max(len(items) for items in by_agent.values())

        for i in range(max_len):
            for agent_id in agent_list:
                if i < len(by_agent[agent_id]):
                    merged.append(by_agent[agent_id][i])

        return merged

    def _merge_dedup_first(self,
                            results: List[CrossAgentResult]
                            ) -> List[Dict[str, Any]]:
        """去重优先合并"""
        seen = OrderedDict()
        for r in sorted(results, key=lambda x: x.confidence, reverse=True):
            for item in r.results:
                content = item.get("content", "")
                # 简单去重：截取前50字符做哈希
                fingerprint = content[:50].lower().strip()
                if fingerprint not in seen:
                    seen[fingerprint] = {
                        **item,
                        "source_agent": r.agent_id,
                    }
        return list(seen.values())


# ============================================================================
# 跨Agent记忆路由器
# ============================================================================

class CrossAgentMemoryRouter:
    """
    跨Agent记忆路由器（协调器）。

    整合联邦管理、查询路由和结果合并的核心协调器。
    """

    def __init__(self):
        self.federation = MemoryFederation()
        self.router = QueryRouter(self.federation)
        self.merger = ResultMerger()
        self._session_memory: Dict[str, Any] = {}
        self._execution_history: List[Dict[str, Any]] = []

    def execute_query(self, query: str,
                       merge_strategy: str = "confidence_weighted",
                       top_k: int = 5) -> Dict[str, Any]:
        """
        执行一次完整的跨Agent查询。

        流程：
        1. 路由查询到合适的Agent
        2. 并行（模拟）查询各Agent记忆
        3. 合并结果
        4. 收集统计

        Args:
            query: 查询文本
            merge_strategy: 结果合并策略
            top_k: 每个Agent返回结果数

        Returns:
            {
                "query": str,
                "routed_agents": [str],
                "results": [...],
                "statistics": {...}
            }
        """
        start = time.time()

        # 路由
        agent_ids = self.router.route(query)
        if not agent_ids:
            # 无匹配Agent，返回所有
            agent_ids = [a.agent_id for a in self.federation._agents.values()
                         if a.health_status == "healthy"]

        # 联邦查询
        raw_results = self.federation.federated_query(query, agent_ids, top_k)

        # 合并结果
        merged = self.merger.merge(raw_results, strategy=merge_strategy)

        elapsed_ms = (time.time() - start) * 1000

        result = {
            "query": query,
            "routed_agents": agent_ids,
            "results": merged,
            "statistics": {
                "total_agents_queried": len(raw_results),
                "total_results": len(merged),
                "total_latency_ms": round(elapsed_ms, 2),
                "merge_strategy": merge_strategy,
                "agent_contributions": {
                    r.agent_id: {
                        "results": len(r.results),
                        "confidence": r.confidence,
                        "latency_ms": r.latency_ms,
                    }
                    for r in raw_results
                },
            },
        }

        self._execution_history.append({
            "query": query,
            "routed": agent_ids,
            "results": len(merged),
            "latency_ms": round(elapsed_ms, 2),
        })

        return result

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        if not self._execution_history:
            return {"total_queries": 0}

        latencies = [e["latency_ms"] for e in self._execution_history]
        return {
            "total_queries": len(self._execution_history),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "avg_results_per_query": round(
                sum(e["results"] for e in self._execution_history) / len(self._execution_history), 1
            ),
        }


# ============================================================================
# Demo: 4 Agent记忆联邦
# ============================================================================

def create_demo_federation() -> CrossAgentMemoryRouter:
    """创建包含4个Agent的演示联邦"""
    router = CrossAgentMemoryRouter()

    # Weather Agent
    router.federation.register_agent(AgentCapability(
        agent_id="weather-agent", name="气象Agent",
        expertise=["weather", "temperature", "rain", "humidity", "forecast",
                    "天气", "温度", "降雨", "湿度", "预报"],
        memory_size=1000, endpoint="weather://local",
    ))

    # Finance Agent
    router.federation.register_agent(AgentCapability(
        agent_id="finance-agent", name="金融Agent",
        expertise=["finance", "stock", "market", "price", "trading",
                    "金融", "股票", "市场", "价格", "交易"],
        memory_size=2000, endpoint="finance://local",
    ))

    # Travel Agent
    router.federation.register_agent(AgentCapability(
        agent_id="travel-agent", name="旅行Agent",
        expertise=["travel", "hotel", "flight", "booking", "tourist",
                    "旅行", "酒店", "航班", "预订", "景点"],
        memory_size=1500, endpoint="travel://local",
    ))

    # General Agent
    router.federation.register_agent(AgentCapability(
        agent_id="general-agent", name="通用Agent",
        expertise=["general", "knowledge", "question", "answer",
                    "通用", "知识", "问答"],
        memory_size=5000, endpoint="general://local",
    ))

    # 添加模拟记忆数据
    # Weather memories
    weather_data = [
        {"id": "w1", "content": "北京今天气温25-32°C，晴转多云，湿度65%"},
        {"id": "w2", "content": "上海明天有暴雨，降水量预计50mm，温度22-28°C"},
        {"id": "w3", "content": "三亚未来三天晴天，适合旅游，温度28-35°C"},
        {"id": "w4", "content": "6月份全国平均气温偏高1-2°C"},
        {"id": "w5", "content": "台风'海燕'预计7月1日登陆福建沿海"},
    ]
    for mem in weather_data:
        router.federation.add_agent_memory("weather-agent", mem)

    # Finance memories
    finance_data = [
        {"id": "f1", "content": "上证指数今日收盘3050点，上涨2.3%"},
        {"id": "f2", "content": "科技板块持续走强，AI概念股集体涨停"},
        {"id": "f3", "content": "央行宣布降准0.25个百分点，释放流动性约5000亿"},
        {"id": "f4", "content": "人民币对美元汇率升至6.85，创年内新高"},
        {"id": "f5", "content": "新能源ETF本周净流入超50亿元"},
    ]
    for mem in finance_data:
        router.federation.add_agent_memory("finance-agent", mem)

    # Travel memories
    travel_data = [
        {"id": "t1", "content": "北京故宫7月门票需提前3天预约"},
        {"id": "t2", "content": "上海至北京高铁最快4小时18分钟"},
        {"id": "t3", "content": "三亚亚特兰蒂斯酒店暑期套房6888元起"},
        {"id": "t4", "content": "国内航班燃油附加费7月1日起下调"},
        {"id": "t5", "content": "暑假旅游保险推荐：安联全球旅行险"},
    ]
    for mem in travel_data:
        router.federation.add_agent_memory("travel-agent", mem)

    # General memories
    general_data = [
        {"id": "g1", "content": "人工智能最新发展：GPT-5预计2026年底发布"},
        {"id": "g2", "content": "全球气温持续升高，各国承诺2050年碳中和"},
        {"id": "g3", "content": "量子计算突破：1000量子比特芯片研发成功"},
        {"id": "g4", "content": "新能源汽车上半年销量同比增长45%"},
        {"id": "g5", "content": "元宇宙概念降温，VR/AR设备销量下滑"},
    ]
    for mem in general_data:
        router.federation.add_agent_memory("general-agent", mem)

    return router


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示"""
    print("=" * 70)
    print("  跨Agent记忆路由 (Cross-Agent Memory Routing) 演示")
    print("=" * 70)

    print("\n【1. 初始化联邦（4个Agent）】")
    print("-" * 50)
    router = create_demo_federation()
    print(f"  注册Agent: {len(router.federation.list_agents())} 个")

    # ---- 2. 单领域查询 ----
    print("\n【2. 单领域查询】")
    print("-" * 50)

    queries = [
        "今天北京天气怎么样",
        "最近股票市场表现如何",
        "三亚旅游有什么推荐",
        "人工智能最新的发展",
    ]

    for query in queries:
        result = router.execute_query(query, merge_strategy="confidence_weighted")
        stats = result["statistics"]
        print(f"\n  查询: '{query}'")
        print(f"  路由到: {result['routed_agents']}")
        print(f"  返回结果: {stats['total_results']} 条 (耗时 {stats['total_latency_ms']:.1f}ms)")
        print(f"  Top 3 结果:")
        for i, item in enumerate(result["results"][:3]):
            src = item.get("source_agent", "unknown")
            content = item.get("content", str(item))[:60]
            print(f"    [{i+1}] [{src}] {content}...")

    # ---- 3. 跨领域查询 ----
    print("\n【3. 跨领域查询】")
    print("-" * 50)

    cross_query = "去三亚旅行需要注意什么，天气和交通方面的建议"
    result = router.execute_query(cross_query)
    stats = result["statistics"]
    print(f"  查询: '{cross_query}'")
    print(f"  路由Agent: {result['routed_agents']}")
    print(f"  各Agent贡献:")
    for agent_id, contrib in stats["agent_contributions"].items():
        print(f"    - {agent_id}: {contrib['results']} 条结果 (置信度: {contrib['confidence']:.2f})")

    # ---- 4. 合并策略对比 ----
    print("\n【4. 合并策略对比】")
    print("-" * 50)

    for strategy in ["confidence_weighted", "round_robin", "dedup_first"]:
        result = router.execute_query("天气 旅游 推荐", merge_strategy=strategy)
        print(f"  [{strategy:20s}] 结果数: {result['statistics']['total_results']}, "
              f"耗时: {result['statistics']['total_latency_ms']:.1f}ms")

    # ---- 5. 执行统计 ----
    print("\n【5. 执行统计】")
    print("-" * 50)
    stats = router.get_execution_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
