#!/usr/bin/env python3
"""
Phase 08 - LangGraph 企业级：03 - 完整的 RAG Agent StateGraph ⭐

将检索增强生成（RAG）实现为一个完整的 StateGraph：
  retrieve → generate → validate → finalize
  条件路由: validate → if "fail" and retry_count < 3 → retrieve (带精炼查询)
            validate → if "pass" → finalize

架构图：
                   ┌─────────────┐
                   │   START     │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │  retrieve   │ ← 搜索知识库获取相关文档
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │  generate   │ ← 基于检索结果生成答案
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │  validate   │ ← 验证答案质量
                   └──┬──────┬──┘
                      │      │
              "pass"  │      │  "fail" (retry_count < 3)
                      │      │
          ┌───────────▼─┐  ┌─▼──────────────┐
          │  finalize   │  │ retrieve (retry)│ ← 循环回去！
          └──────┬──────┘  └────────────────┘
                 │
          ┌──────▼──────┐
          │    END      │
          └─────────────┘
"""

import json
import time
import hashlib
from typing import TypedDict, Literal, Annotated, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


# ============================================================================
# State 定义
# ============================================================================

class RAGAgentState(TypedDict):
    """
    RAG Agent 的全局 State。

    每个节点可以读取和更新这些字段。

    Attributes:
        question: 用户原始问题
        chat_history: 对话历史（list of BaseMessage）
        retrieved_docs: 检索到的文档列表
        refined_query: 重试时使用的精炼查询
        answer: 生成的答案
        validation_result: 验证结果 "pass" | "fail"
        retry_count: 当前重试次数
        max_retries: 最大重试次数
        total_time_ms: 总执行时间（毫秒）
    """
    question: str
    chat_history: Annotated[list, add_messages]
    retrieved_docs: list
    refined_query: str
    answer: str
    validation_result: str  # "pass" or "fail"
    retry_count: int
    max_retries: int
    total_time_ms: float


# ============================================================================
# 模拟知识库
# ============================================================================

# 模拟的文档集合（生产环境中替换为真实的向量数据库检索）
KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "title": "LangGraph 简介",
        "content": "LangGraph 是 LangChain 生态中的有状态多Agent编排框架。"
                   "它基于有向图（StateGraph）概念，支持循环、条件分支、"
                   "状态持久化（Checkpointing）和人机协作（Human-in-the-Loop）。"
                   "核心组件包括 StateGraph、Node、Edge、Checkpointer。",
    },
    {
        "id": "doc_002",
        "title": "RAG 系统最佳实践",
        "content": "检索增强生成（RAG）系统的最佳实践包括：1) 使用混合检索"
                   "（Dense + Sparse）；2) 对检索结果进行重排序（Re-ranking）；"
                   "3) 实现查询重写（Query Rewriting）以提高召回率；"
                   "4) 使用 Self-RAG 或 CRAG 进行自纠正。建议 chunk_size 为 "
                   "512 tokens，overlap 为 50 tokens。",
    },
    {
        "id": "doc_003",
        "title": "StateGraph API 参考",
        "content": "StateGraph 的核心API：builder = StateGraph(StateClass) 创建Builder；"
                   "builder.add_node(name, func) 添加节点；"
                   "builder.set_entry_point(name) 设置入口；"
                   "builder.add_edge(src, dst) 添加固定边；"
                   "builder.add_conditional_edges(src, router, mapping) 添加条件边；"
                   "graph = builder.compile(checkpointer=...) 编译图。"
                   "使用 graph.invoke(state, config) 执行图。",
    },
    {
        "id": "doc_004",
        "title": "Checkpointer 持久化",
        "content": "LangGraph 的 Checkpointer 支持 PostgreSQL 和 SQLite 两种后端。"
                   "使用 PostgresSaver 时，需要先创建数据库表（setup() 方法）。"
                   "每个对话线程（thread）的状态都独立持久化。"
                   "通过 config={'configurable': {'thread_id': '...'}} 来隔离不同会话。"
                   "支持从任意 checkpoint 恢复执行（Time Travel 功能）。",
    },
    {
        "id": "doc_005",
        "title": "Human-in-the-Loop 中断",
        "content": "Human-in-the-Loop（HITL）允许在关键节点暂停图执行，等待人工输入。"
                   "使用 interrupt() 函数在节点内部暂停。"
                   "恢复执行时使用 Command(resume=human_input)。"
                   "典型场景：大额交易审批、敏感内容审查、医疗建议确认。"
                   "前端集成模式：API 端点返回中断状态 → 前端展示审批界面 → 用户提交审批。",
    },
]


def mock_retrieve_knowledge(query: str, top_k: int = 3) -> list[dict]:
    """
    模拟知识库检索。生产环境中替换为真实的向量搜索。
    这里使用简单的关键词匹配。
    """
    query_lower = query.lower()
    scored = []

    for doc in KNOWLEDGE_BASE:
        content_lower = doc["content"].lower()
        title_lower = doc["title"].lower()

        # 简单的关键词匹配得分
        score = 0
        keywords = query_lower.split()
        for kw in keywords:
            if kw in content_lower:
                score += 1
            if kw in title_lower:
                score += 2

        if score > 0:
            scored.append((score, doc))

    # 按得分排序
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ============================================================================
# 节点函数
# ============================================================================

def node_retrieve(state: RAGAgentState) -> dict:
    """
    检索节点：查询知识库获取相关文档。

    如果是重试（retry_count > 0），使用 refined_query 进行检索。
    """
    start = time.time()

    # 决定使用哪个查询
    if state.get('retry_count', 0) > 0 and state.get('refined_query'):
        query = state['refined_query']
        print(f"  [retrieve] 重试模式，使用精炼查询: {query[:80]}...")
    else:
        query = state['question']
        print(f"  [retrieve] 初始检索: {query[:80]}...")

    # 检索
    docs = mock_retrieve_knowledge(query, top_k=3)

    print(f"  [retrieve] 找到 {len(docs)} 篇文档:")
    for doc in docs:
        print(f"    - {doc['title']} (id: {doc['id']})")

    return {
        "retrieved_docs": docs,
        "refined_query": query,
    }


def node_generate(state: RAGAgentState) -> dict:
    """
    生成节点：基于检索到的文档生成答案。

    如果文档不存在，生成一个降级回答。
    """
    docs = state.get('retrieved_docs', [])
    question = state['question']
    retry_count = state.get('retry_count', 0)

    print(f"  [generate] 基于 {len(docs)} 篇文档生成答案 (重试#{retry_count})...")

    if not docs:
        answer = (
            f"抱歉，我在知识库中没有找到与'{question}'相关的信息。\n\n"
            f"建议：\n"
            f"1. 尝试使用不同的关键词重新提问\n"
            f"2. 检查问题的拼写\n"
            f"3. 您的知识库可能缺少相关领域的文档"
        )
    else:
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[文档{i}] {doc['title']}\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # 模拟 LLM 生成答案
        answer = (
            f"## 回答\n\n"
            f"关于'{question}'，根据知识库中的信息：\n\n"
            f"{context[:500]}...\n\n"
            f"### 总结\n综合以上 {len(docs)} 篇文档的信息，可以得出以下结论：\n"
            f"上述文档涵盖了您问题的核心方面。如需更详细的信息，"
            f"请提出更具体的问题。\n\n"
            f"*参考来源: {', '.join(d['title'] for d in docs)}*"
        )

    # 注入重试信息（如果适用）
    if retry_count > 0:
        answer = f"{answer}\n\n*（此回答经过 {retry_count} 次精炼重试）*"

    print(f"  [generate] 生成答案: {len(answer)} 字符")

    return {
        "answer": answer,
        "chat_history": [
            AIMessage(content=answer),
        ],
    }


def node_validate(state: RAGAgentState) -> dict:
    """
    验证节点：评估答案质量，决定 pass 或 fail。

    评估维度：
    1. 是否引用了检索到的文档（避免幻觉）
    2. 答案长度是否合理（不过短也不过长）
    3. 是否包含拒绝回答的标志
    """
    answer = state.get('answer', '')
    docs = state.get('retrieved_docs', [])
    retry_count = state.get('retry_count', 0)

    print(f"  [validate] 验证答案质量...")

    # 检查1：是否有文档基础
    has_doc_basis = any(
        doc['title'].lower() in answer.lower()
        for doc in docs
    ) if docs else False

    # 检查2：答案长度是否合理（至少100字符，最多5000字符）
    length_ok = 100 <= len(answer) <= 5000

    # 检查3：是否包含"抱歉"等否定标志
    is_apologizing = "抱歉" in answer and len(docs) == 0

    # 综合判断
    if has_doc_basis and length_ok:
        result = "pass"
    elif not has_doc_basis and retry_count >= 3:
        result = "pass"  # 重试耗尽，接受不完美的答案
    elif is_apologizing and retry_count >= 2:
        result = "pass"  # 确实没文档，接受道歉回答
    else:
        result = "fail"

    print(f"  [validate] 验证结果: {result}")
    print(f"    文档基础: {'✅' if has_doc_basis else '❌'}")
    print(f"    长度合理: {'✅' if length_ok else '❌'} "
          f"({len(answer)} 字符)")
    print(f"    重试次数: {retry_count}")

    return {
        "validation_result": result,
    }


def node_refine_query(state: RAGAgentState) -> dict:
    """
    查询精炼节点（在失败和重试之间调用）。
    优化查询以在下一次检索中获得更好的结果。
    """
    original_query = state['question']
    retry_count = state.get('retry_count', 0)
    docs = state.get('retrieved_docs', [])

    print(f"  [refine_query] 精炼查询策略 (重试#{retry_count + 1})...")

    # 精炼策略：基于已有的文档关键词扩展查询
    if docs:
        # 提取已有文档中的关键词
        all_keywords = set()
        for doc in docs:
            words = doc['content'].lower().split()
            # 取出现频率高的长词作为关键词
            for w in words:
                if len(w) > 4 and w.isalpha():
                    all_keywords.add(w)

        top_keywords = list(all_keywords)[:5]
        refined = f"{original_query} {' '.join(top_keywords)}"
    else:
        # 没有检索到文档：尝试简化查询
        words = original_query.split()
        if len(words) > 5:
            refined = ' '.join(words[:5])  # 用前5个词
        else:
            refined = original_query + " 概述 入门 基础"

    print(f"  [refine_query] 精炼后: {refined[:100]}...")

    return {
        "refined_query": refined,
        "retry_count": retry_count + 1,
    }


def node_finalize(state: RAGAgentState) -> dict:
    """
    最终化节点：标注最终答案，添加元数据。
    """
    answer = state.get('answer', '')
    docs = state.get('retrieved_docs', [])
    retry_count = state.get('retry_count', 0)

    print(f"  [finalize] 最终化答案...")

    # 添加质量标注
    if state.get('validation_result') == "pass":
        quality_note = "✅ 答案已通过质量验证"
    else:
        quality_note = "⚠️ 答案未通过验证（已使用降级策略）"

    # 构建最终答案
    finalized = (
        f"{answer}\n\n"
        f"---\n"
        f"### 答案元数据\n"
        f"- 质量: {quality_note}\n"
        f"- 重试次数: {retry_count}\n"
        f"- 参考文档数: {len(docs)}\n"
        f"- 生成时间: {state.get('total_time_ms', 0):.0f}ms\n"
    )

    return {
        "answer": finalized,
        "retrieve_status": "final",
    }


# ============================================================================
# 路由函数
# ============================================================================

def route_after_validate(state: RAGAgentState) -> Literal["refine", "finalize"]:
    """
    验证后的路由决策。

    如果验证失败且重试次数未达上限，进入 refine 循环。
    否则进入 finalize。
    """
    if (state.get('validation_result') == "fail"
            and state.get('retry_count', 0) < state.get('max_retries', 3)):
        print(f"  [router] 验证失败 → 进入精炼重试循环")
        return "refine"
    else:
        print(f"  [router] → 进入最终化")
        return "finalize"


# ============================================================================
# 构建 StateGraph
# ============================================================================

def build_rag_graph() -> StateGraph:
    """
    构建完整的 RAG Agent StateGraph。

    图结构:
        START → retrieve → generate → validate → [路由]
                                                   ├── "refine" → refine_query → retrieve (循环)
                                                   └── "finalize" → finalize → END
    """
    builder = StateGraph(RAGAgentState)

    # 添加节点
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("generate", node_generate)
    builder.add_node("validate", node_validate)
    builder.add_node("refine", node_refine_query)
    builder.add_node("finalize", node_finalize)

    # 设置入口
    builder.set_entry_point("retrieve")

    # 添加边
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "validate")

    # 条件路由：validate → refine 或 finalize
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "refine": "refine",
            "finalize": "finalize",
        }
    )

    # refine 循环回 retrieve
    builder.add_edge("refine", "retrieve")

    # finalize 结束
    builder.add_edge("finalize", END)

    # 编译
    graph = builder.compile()

    return graph


# ============================================================================
# 辅助类：RAGAgent 封装
# ============================================================================

class RAGAgent:
    """
    RAG Agent 的高级封装。

    使用方式:
        agent = RAGAgent()
        result = agent.ask("什么是 LangGraph?")
        print(result["answer"])
    """

    def __init__(self, max_retries: int = 3):
        self.graph = build_rag_graph()
        self.max_retries = max_retries
        self.total_calls = 0
        self.total_retries = 0
        self.total_time = 0.0

    def ask(self, question: str, chat_history: list = None) -> dict:
        """
        向 RAG Agent 提问。

        Args:
            question: 用户问题
            chat_history: 可选的历史对话

        Returns:
            包含 answer, retry_count, validation_result 等的字典
        """
        self.total_calls += 1
        start = time.time()

        initial_state = {
            "question": question,
            "chat_history": chat_history or [],
            "retrieved_docs": [],
            "refined_query": "",
            "answer": "",
            "validation_result": "",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "total_time_ms": 0,
        }

        result = self.graph.invoke(initial_state)
        elapsed_ms = (time.time() - start) * 1000

        self.total_time += elapsed_ms
        if result.get("retry_count", 0) > 0:
            self.total_retries += 1

        # 更新时间戳
        result["total_time_ms"] = elapsed_ms

        return result

    def ask_stream(self, question: str) -> dict:
        """
        流式提问（当前版本使用同步invoke）。

        生产环境中可使用 graph.astream() 获得真正的流式输出。
        """
        start = time.time()

        initial_state = {
            "question": question,
            "chat_history": [],
            "retrieved_docs": [],
            "refined_query": "",
            "answer": "",
            "validation_result": "",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "total_time_ms": 0,
        }

        results = []
        for event in self.graph.stream(initial_state):
            results.append(event)
            yield event

    def get_stats(self) -> dict:
        """获取Agent统计信息。"""
        return {
            "total_calls": self.total_calls,
            "total_retries": self.total_retries,
            "total_time_ms": round(self.total_time, 1),
            "avg_time_ms": round(self.total_time / max(1, self.total_calls), 1),
            "retry_rate": round(
                self.total_retries / max(1, self.total_calls), 3
            ),
        }

    def visualize(self) -> str:
        """获取图的 Mermaid 表示。"""
        return self.graph.get_graph().draw_mermaid()


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  LangGraph 企业级 - RAG Agent StateGraph 演示")
    print("=" * 72)

    # 创建 RAG Agent
    agent = RAGAgent(max_retries=3)

    # 可视化
    print("\n📊 图的 Mermaid 表示:")
    print(agent.visualize())

    # ---- Demo 1: 正常查询 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 正常查询（直接 pass）")
    print("=" * 72)

    result1 = agent.ask("什么是 LangGraph 的 StateGraph？")
    print(f"\n{'─' * 72}")
    print(f"  最终答案:")
    print(f"{'─' * 72}")
    print(result1["answer"][:600])

    # ---- Demo 2: 需要重试的查询 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 需要精炼重试的查询")
    print("=" * 72)

    result2 = agent.ask("怎么持久化和恢复对话状态？")
    print(f"\n{'─' * 72}")
    print(f"  最终答案:")
    print(f"{'─' * 72}")
    print(result2["answer"][:600])

    # ---- Demo 3: 知识库外查询 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 知识库外查询（触发降级策略）")
    print("=" * 72)

    result3 = agent.ask("量子计算在密码学中的应用？")
    print(f"\n{'─' * 72}")
    print(f"  最终答案:")
    print(f"{'─' * 72}")
    print(result3["answer"][:600])

    # ---- Demo 4: 流式调用 ----

    print("\n\n" + "=" * 72)
    print("  Demo 4: 流式调用（stream 模式）")
    print("=" * 72)

    print("\n  各节点输出:")
    for i, event in enumerate(agent.ask_stream("RAG 系统最佳实践是什么？")):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        # 截取关键信息
        keys_with_values = {k: v for k, v in node_output.items() if v}
        print(f"    [{i+1}] {node_name}: {keys_with_values}")

    # ---- 统计 ----

    print("\n\n" + "=" * 72)
    print("  📊 Agent 执行统计")
    print("=" * 72)

    stats = agent.get_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print("  RAG Agent StateGraph 演示完成！")
    print("=" * 72)
