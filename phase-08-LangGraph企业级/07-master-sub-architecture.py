#!/usr/bin/env python3
"""
Phase 08 - LangGraph 企业级：07 - Master-Sub 架构（主从图架构）

Master 图：意图分类 → 路由到子图
Sub-graphs：RAG 子图、Tool-calling 子图、Human-approval 子图

State 映射：Master 和 Sub 之间通过 input/output key 配置传递数据。

架构图：
  ┌────────────────────────────────────────────────────────────┐
  │                      Master Graph                           │
  │                                                             │
  │  ┌──────────────┐     ┌──────────────┐                     │
  │  │   classify   │────→│    router    │                     │
  │  │  (意图分类)  │     │  (路由分发)  │                     │
  │  └──────────────┘     └──┬───┬───┬───┘                     │
  │                          │   │   │                          │
  │              ┌───────────┘   │   └───────────┐              │
  │              ▼               ▼               ▼              │
  │  ┌───────────────┐ ┌──────────────┐ ┌──────────────┐      │
  │  │  RAG Sub      │ │  Tool Sub    │ │  Human Sub   │      │
  │  │  Graph        │ │  Graph       │ │  Graph       │      │
  │  │               │ │              │ │              │      │
  │  │ retrieve      │ │ call_tool    │ │ request      │      │
  │  │   ↓           │ │   ↓          │ │   ↓          │      │
  │  │ generate      │ │ validate     │ │ wait_approval│      │
  │  │   ↓           │ │   ↓          │ │   ↓          │      │
  │  │ validate      │ │ finalize     │ │ confirm      │      │
  │  │   ↓           │ │              │ │              │      │
  │  │ finalize      │ │              │ │              │      │
  │  └───────────────┘ └──────────────┘ └──────────────┘      │
  │         │                 │                 │               │
  │         └─────────────────┼─────────────────┘               │
  │                           ▼                                 │
  │                    ┌──────────────┐                         │
  │                    │  response    │                         │
  │                    │  (合成响应)   │                         │
  │                    └──────────────┘                         │
  └────────────────────────────────────────────────────────────┘
"""

import json
import time
from typing import TypedDict, Annotated, Literal, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# ============================================================================
# Master State 定义
# ============================================================================

class MasterState(TypedDict):
    """
    Master 图的全局 State。
    包含所有子图可能需要的信息。
    """
    # 请求信息
    messages: Annotated[list, add_messages]
    user_query: str
    intent: str  # "rag" | "tool" | "human_approval" | "general"

    # RAG 子图需要的字段
    retrieved_docs: list
    rag_answer: str

    # Tool 子图需要的字段
    tool_name: str
    tool_args: dict
    tool_result: str

    # Human 子图需要的字段
    approval_required: bool
    approval_status: str  # "pending" | "approved" | "rejected"

    # 最终响应
    final_response: str
    response_generated: bool


# ============================================================================
# 子图1：RAG Sub-Graph
# ============================================================================

class RAGSubState(TypedDict):
    """RAG子图的内部State。"""
    user_query: str
    retrieved_docs: list
    rag_answer: str
    validation_result: str
    retry_count: int


def rag_retrieve(state: RAGSubState) -> dict:
    """RAG子图：检索节点。"""
    query = state['user_query']
    print(f"    [RAG.retrieve] 检索: {query[:60]}...")

    # 模拟检索
    docs = [
        {"title": "LangGraph 文档", "content": "StateGraph 是 LangGraph 的核心..."},
        {"title": "Checkpointer 指南", "content": "PostgresSaver 用于状态持久化..."},
    ]
    return {"retrieved_docs": docs}


def rag_generate(state: RAGSubState) -> dict:
    """RAG子图：生成节点。"""
    docs = state.get('retrieved_docs', [])
    print(f"    [RAG.generate] 基于 {len(docs)} 篇文档生成...")

    answer = f"RAG答案：基于 {len(docs)} 篇文档的分析结果..."
    return {
        "rag_answer": answer,
        "validation_result": "pass",
    }


def rag_validate(state: RAGSubState) -> dict:
    """RAG子图：验证节点。"""
    retry = state.get('retry_count', 0)
    if retry < 2 and state.get('validation_result') != "pass":
        return {"retry_count": retry + 1, "validation_result": "fail"}
    return {"validation_result": "pass"}


def rag_finalize(state: RAGSubState) -> dict:
    """RAG子图：最终化。"""
    print(f"    [RAG.finalize] 生成最终答案")
    return {"rag_answer": f"✅ {state.get('rag_answer', '')}"}


def rag_route(state: RAGSubState) -> Literal["retrieve", "finalize"]:
    """RAG子图路由。"""
    if state.get('validation_result') == "fail":
        return "retrieve"
    return "finalize"


def build_rag_subgraph() -> StateGraph:
    """构建RAG子图。"""
    builder = StateGraph(RAGSubState)
    builder.add_node("retrieve", rag_retrieve)
    builder.add_node("generate", rag_generate)
    builder.add_node("validate", rag_validate)
    builder.add_node("finalize", rag_finalize)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges("validate", rag_route, {
        "retrieve": "retrieve",
        "finalize": "finalize",
    })
    builder.add_edge("finalize", END)

    return builder.compile()


# ============================================================================
# 子图2：Tool-calling Sub-Graph
# ============================================================================

class ToolSubState(TypedDict):
    """Tool子图的内部State。"""
    user_query: str
    tool_name: str
    tool_args: dict
    tool_result: str
    validation_result: str


def tool_call(state: ToolSubState) -> dict:
    """Tool子图：工具调用节点。"""
    tool = state.get('tool_name', 'unknown')
    args = state.get('tool_args', {})

    print(f"    [Tool.call] 调用工具: {tool}({args})")

    # 模拟工具执行
    if tool == "calculator":
        expr = args.get('expression', '0')
        result = f"计算结果: {expr} = 42 (模拟)"
    elif tool == "web_search":
        query = args.get('query', '')
        result = f"搜索结果: 关于'{query}'的信息..."
    elif tool == "database_query":
        result = "数据库查询结果: 3条记录"
    else:
        result = f"工具 {tool} 执行完成"

    return {"tool_result": result, "validation_result": "pass"}


def tool_validate(state: ToolSubState) -> dict:
    """Tool子图：验证节点。"""
    print(f"    [Tool.validate] 验证工具结果...")
    return {"validation_result": "pass"}


def tool_finalize(state: ToolSubState) -> dict:
    """Tool子图：最终化。"""
    print(f"    [Tool.finalize] 打包工具结果")
    return {"tool_result": f"✅ {state.get('tool_result', '')}"}


def build_tool_subgraph() -> StateGraph:
    """构建Tool子图。"""
    builder = StateGraph(ToolSubState)
    builder.add_node("call_tool", tool_call)
    builder.add_node("validate", tool_validate)
    builder.add_node("finalize", tool_finalize)

    builder.set_entry_point("call_tool")
    builder.add_edge("call_tool", "validate")
    builder.add_edge("validate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# ============================================================================
# 子图3：Human-approval Sub-Graph
# ============================================================================

class HumanSubState(TypedDict):
    """Human-approval子图的内部State。"""
    user_query: str
    approval_required: bool
    approval_status: str
    approval_reason: str


def human_request(state: HumanSubState) -> dict:
    """Human子图：请求审批节点。"""
    print(f"    [Human.request] 创建审批请求...")
    return {"approval_required": True, "approval_status": "pending"}


def human_wait(state: HumanSubState) -> dict:
    """Human子图：等待审批节点。"""
    # 在真实场景中，这里会调用 interrupt()
    # 此处模拟自动批准
    print(f"    [Human.wait] 模拟人工审批...")
    return {
        "approval_status": "approved",
        "approval_reason": "自动审批（演示模式）",
    }


def human_confirm(state: HumanSubState) -> dict:
    """Human子图：确认节点。"""
    print(f"    [Human.confirm] 审批完成: {state.get('approval_status')}")
    return {}


def build_human_subgraph() -> StateGraph:
    """构建Human-approval子图。"""
    builder = StateGraph(HumanSubState)
    builder.add_node("request", human_request)
    builder.add_node("wait", human_wait)
    builder.add_node("confirm", human_confirm)

    builder.set_entry_point("request")
    builder.add_edge("request", "wait")
    builder.add_edge("wait", "confirm")
    builder.add_edge("confirm", END)

    return builder.compile()


# ============================================================================
# Master 图节点
# ============================================================================

def node_classify_intent(state: MasterState) -> dict:
    """
    Master: 意图分类节点。
    分析用户查询，判断应该路由到哪个子图。
    """
    query = state['user_query'].lower()

    if any(w in query for w in ['搜索', '查询', '文档', '知识', 'search', 'find', 'rag']):
        intent = "rag"
    elif any(w in query for w in ['计算', '搜索', '工具', 'tool', 'api', '数据']):
        intent = "tool"
    elif any(w in query for w in ['审批', '批准', 'authorize', 'approve', 'confirm']):
        intent = "human_approval"
    else:
        intent = "general"

    print(f"  [classify] 查询分类: {intent}")
    return {"intent": intent}


def node_route_to_subgraph(state: MasterState) -> Literal["rag_sub", "tool_sub", "human_sub", "general_response"]:
    """
    Master: 路由节点。
    根据意图分发到不同的子图。
    """
    intent = state['intent']
    print(f"  [router] 路由到: {intent}")
    return f"{intent}_sub" if intent != "general" else "general_response"


def node_rag_subgraph_entry(state: MasterState) -> dict:
    """
    Master: RAG子图入口。
    准备子图输入并调用子图。
    """
    print(f"  [rag_sub_entry] 准备RAG子图输入...")

    # 构建子图输入
    sub_input = {
        "user_query": state['user_query'],
        "retrieved_docs": state.get('retrieved_docs', []),
        "rag_answer": "",
        "validation_result": "",
        "retry_count": 0,
    }

    # 执行子图
    sub_graph = build_rag_subgraph()
    sub_result = sub_graph.invoke(sub_input)

    print(f"  [rag_sub_entry] RAG子图完成: {sub_result.get('rag_answer', '')[:60]}...")

    return {
        "retrieved_docs": sub_result.get('retrieved_docs', []),
        "rag_answer": sub_result.get('rag_answer', ''),
    }


def node_tool_subgraph_entry(state: MasterState) -> dict:
    """
    Master: Tool子图入口。
    """
    print(f"  [tool_sub_entry] 准备Tool子图输入...")

    # 根据查询决定使用哪个工具
    query = state['user_query']
    if '计算' in query:
        tool = "calculator"
        args = {"expression": query}
    elif '搜索' in query:
        tool = "web_search"
        args = {"query": query}
    else:
        tool = "database_query"
        args = {"query": query}

    sub_input = {
        "user_query": query,
        "tool_name": tool,
        "tool_args": args,
        "tool_result": "",
        "validation_result": "",
    }

    sub_graph = build_tool_subgraph()
    sub_result = sub_graph.invoke(sub_input)

    print(f"  [tool_sub_entry] Tool子图完成: {sub_result.get('tool_result', '')[:60]}...")

    return {
        "tool_name": tool,
        "tool_args": args,
        "tool_result": sub_result.get('tool_result', ''),
    }


def node_human_subgraph_entry(state: MasterState) -> dict:
    """
    Master: Human-approval子图入口。
    """
    print(f"  [human_sub_entry] 准备Human子图输入...")

    sub_input = {
        "user_query": state['user_query'],
        "approval_required": False,
        "approval_status": "pending",
        "approval_reason": "",
    }

    sub_graph = build_human_subgraph()
    sub_result = sub_graph.invoke(sub_input)

    print(f"  [human_sub_entry] Human子图完成: {sub_result.get('approval_status')}")

    return {
        "approval_required": sub_result.get('approval_required', False),
        "approval_status": sub_result.get('approval_status', 'pending'),
    }


def node_general_response(state: MasterState) -> dict:
    """Master: 通用响应节点（不需要子图）。"""
    print(f"  [general_response] 生成通用响应...")
    return {
        "final_response": f"关于'{state['user_query'][:50]}'，这是一个通用问题。我可以帮您解答...",
        "response_generated": True,
    }


def node_compose_response(state: MasterState) -> dict:
    """
    Master: 响应合成节点。
    根据意图收集子图结果，合成最终响应。
    """
    intent = state.get('intent', 'general')

    if intent == "rag":
        response = state.get('rag_answer', '无RAG结果')
    elif intent == "tool":
        response = state.get('tool_result', '无工具结果')
    elif intent == "human_approval":
        status = state.get('approval_status', 'pending')
        response = f"审批请求处理完成: 状态={status}"
    else:
        response = state.get('final_response', '无响应')

    print(f"  [compose] 合成最终响应: {response[:60]}...")

    return {
        "final_response": response,
        "response_generated": True,
        "messages": [AIMessage(content=response)],
    }


# ============================================================================
# 构建 Master 图
# ============================================================================

def build_master_graph() -> StateGraph:
    """
    构建完整的 Master-Sub 架构图。

    流程:
      classify → router → [rag_sub | tool_sub | human_sub | general_response]
                  → compose_response → END
    """
    builder = StateGraph(MasterState)

    # 添加节点
    builder.add_node("classify", node_classify_intent)
    builder.add_node("rag_sub", node_rag_subgraph_entry)
    builder.add_node("tool_sub", node_tool_subgraph_entry)
    builder.add_node("human_sub", node_human_subgraph_entry)
    builder.add_node("general_response", node_general_response)
    builder.add_node("compose", node_compose_response)

    # 设置入口
    builder.set_entry_point("classify")

    # 条件路由
    builder.add_conditional_edges(
        "classify",
        node_route_to_subgraph,
        {
            "rag_sub": "rag_sub",
            "tool_sub": "tool_sub",
            "human_sub": "human_sub",
            "general_response": "general_response",
        }
    )

    # 所有分支最终到 compose
    builder.add_edge("rag_sub", "compose")
    builder.add_edge("tool_sub", "compose")
    builder.add_edge("human_sub", "compose")
    builder.add_edge("general_response", "compose")

    # compose 结束
    builder.add_edge("compose", END)

    return builder


# ============================================================================
# Master-Sub 架构封装
# ============================================================================

class MasterSubOrchestrator:
    """
    Master-Sub 架构的高级封装。

    使用方式:
        orchestrator = MasterSubOrchestrator()
        result = orchestrator.process("搜索LangGraph的相关文档")
        print(result["final_response"])
    """

    def __init__(self, checkpointer=None):
        self.checkpointer = checkpointer or MemorySaver()
        builder = build_master_graph()
        self.graph = builder.compile(checkpointer=self.checkpointer)

        # 缓存子图
        self._sub_graphs = {
            "rag": build_rag_subgraph(),
            "tool": build_tool_subgraph(),
            "human": build_human_subgraph(),
        }

    def process(self, user_query: str, thread_id: str = None) -> dict:
        """
        处理用户查询。

        Args:
            user_query: 用户问题
            thread_id: 可选线程ID（用于状态持久化）

        Returns:
            完整的处理结果
        """
        if thread_id is None:
            import uuid
            thread_id = f"session_{uuid.uuid4().hex[:8]}"

        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "intent": "",
            "retrieved_docs": [],
            "rag_answer": "",
            "tool_name": "",
            "tool_args": {},
            "tool_result": "",
            "approval_required": False,
            "approval_status": "",
            "final_response": "",
            "response_generated": False,
        }

        start = time.time()
        result = self.graph.invoke(initial_state, config)
        elapsed = (time.time() - start) * 1000

        result["elapsed_ms"] = elapsed
        return result

    def batch_process(self, queries: list[str]) -> list[dict]:
        """批量处理多个查询。"""
        results = []
        for query in queries:
            result = self.process(query)
            results.append({
                "query": query,
                "intent": result.get("intent"),
                "response": result.get("final_response", "")[:100],
                "elapsed_ms": result.get("elapsed_ms"),
            })
        return results

    def get_sub_graph(self, name: str) -> StateGraph:
        """获取已编译的子图。"""
        if name not in self._sub_graphs:
            raise ValueError(f"未知子图: {name}. 可用: {list(self._sub_graphs.keys())}")
        return self._sub_graphs[name]


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  LangGraph 企业级 - Master-Sub 架构演示")
    print("=" * 72)

    orchestrator = MasterSubOrchestrator()

    # ---- 测试查询 ----

    test_queries = [
        ("搜索LangGraph Checkpointer的文档", "期望路由到: RAG Sub"),
        ("帮我计算 1000 * 2.5 的结果", "期望路由到: Tool Sub"),
        ("请审批这笔50000元的转账", "期望路由到: Human Sub"),
        ("今天天气怎么样？", "期望路由到: General Response"),
    ]

    for query, description in test_queries:
        print(f"\n{'─' * 60}")
        print(f"  查询: {query}")
        print(f"  说明: {description}")
        print(f"{'─' * 60}")

        result = orchestrator.process(query)

        print(f"\n  📊 处理结果:")
        print(f"   意图: {result.get('intent')}")
        print(f"   响应: {result.get('final_response', '')[:100]}...")
        print(f"   耗时: {result.get('elapsed_ms', 0):.0f}ms")

    # ---- 批量处理 ----

    print(f"\n\n{'=' * 72}")
    print(f"  批量处理示例")
    print(f"{'=' * 72}")

    batch_queries = [
        "查询RAG系统最佳实践",
        "搜索最新的AI趋势报告",
        "帮我计算项目预算",
    ]

    batch_results = orchestrator.batch_process(batch_queries)
    print(f"\n  📊 批量处理 {len(batch_queries)} 个查询:")
    for br in batch_results:
        print(f"    [{br['intent']}] {br['query'][:40]}... → {br['response'][:60]}...")
        print(f"        耗时: {br['elapsed_ms']:.0f}ms")

    # ---- 图可视化 ----

    print(f"\n\n{'=' * 72}")
    print(f"  Master 图的 Mermaid 表示:")
    print(f"{'=' * 72}")
    print(orchestrator.graph.get_graph().draw_mermaid())

    print("\n" + "=" * 72)
    print("  Master-Sub 架构演示完成！")
    print("=" * 72)
