#!/usr/bin/env python3
"""
Phase 08 - LangGraph 企业级：06 - 时间旅行（Time Travel）

LangGraph 的时间旅行能力：
1. 查看历史状态：get_state_history(config)
2. 从特定检查点回放：invoke(None, config) 恢复到最新
3. "Rewind and Retry"：检测到坏输出 → 回退到前一个状态 → 尝试不同策略

核心概念：
  ┌────────────────────────────────────────────────────┐
  │              时间旅行工作流                          │
  │                                                     │
  │  当前线程 (thread_id="abc")                         │
  │  checkpoint_0 → checkpoint_1 → checkpoint_2 → ...   │
  │       ↑                           ↑                 │
  │       │                           │                 │
  │   create_checkpoint           current_state         │
  │                                                     │
  │  时间旅行:                                          │
  │  get_state_history() → 列出所有checkpoint            │
  │  invoke(None, checkpoint_config) → 从某点重新执行    │
  │                                                     │
  │  Rewind & Retry:                                    │
  │  detect_bad_output() → 找到上一个好checkpoint        │
  │  → 从该点重新invoke（使用不同参数/策略）              │
  └────────────────────────────────────────────────────┘

使用场景：
  1. 调试：查看Agent在每一步的思考和决策
  2. 重试优化：用不同的prompt从中间点重新生成
  3. A/B测试：从同一分叉点尝试不同的下游策略
  4. 审计：完整的状态变更轨迹
"""

import json
import time
import threading
from typing import TypedDict, Annotated, Literal, Optional, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ============================================================================
# State 定义
# ============================================================================

class ArticleState(TypedDict):
    """
    文章生成的 State。
    每个 checkpoint 记录一个完整的生成步骤。
    """
    topic: str
    outline: str
    draft: str
    review_comments: str
    final_version: str
    quality_score: float
    generation_stage: str  # "outline" | "draft" | "review" | "final"
    attempt_count: int


# ============================================================================
# 节点函数
# ============================================================================

def node_create_outline(state: ArticleState) -> dict:
    """节点1：创建大纲。"""
    topic = state['topic']
    attempt = state.get('attempt_count', 1)

    print(f"  [outline] 创建大纲 (尝试#{attempt})...")

    outline = (
        f"# 大纲: {topic}\n\n"
        f"## 1. 引言\n"
        f"## 2. 背景分析\n"
        f"## 3. 核心观点\n"
        f"## 4. 案例研究\n"
        f"## 5. 结论与展望\n\n"
        f"*生成尝试 #{attempt}*"
    )

    return {
        "outline": outline,
        "generation_stage": "outline",
    }


def node_write_draft(state: ArticleState) -> dict:
    """节点2：撰写初稿。"""
    outline = state.get('outline', '')
    attempt = state.get('attempt_count', 1)

    print(f"  [draft] 撰写初稿...")

    draft = (
        f"# {state['topic']}\n\n"
        f"## 引言\n"
        f"本文深入探讨{state['topic']}的各个方面..."
        f"（基于大纲的完整初稿，尝试 #{attempt}）\n\n"
        f"## 背景分析\n"
        f"根据最新行业数据...\n\n"
        f"## 核心观点\n"
        f"经过深入分析...\n\n"
        f"## 案例研究\n"
        f"以下案例展示了...\n\n"
        f"## 结论\n"
        f"综上所述..."
    )

    return {
        "draft": draft,
        "generation_stage": "draft",
    }


def node_review_draft(state: ArticleState) -> dict:
    """节点3：审查初稿。"""
    draft = state.get('draft', '')
    attempt = state.get('attempt_count', 1)

    print(f"  [review] 审查初稿...")

    # 模拟审查结果
    if attempt == 1:
        # 第一次尝试：质量不高
        quality = 0.55
        comments = "初稿质量不足：缺少数据支撑、结构松散、结论不够有力"
        review_result = "needs_revision"
    elif attempt == 2:
        # 第二次尝试：有所改善
        quality = 0.72
        comments = "质量有所改善：数据有所补充，但仍需加强案例部分"
        review_result = "needs_revision"
    else:
        # 第三次及以后：通过
        quality = 0.88
        comments = "质量达标：结构完整、数据充分、逻辑清晰"
        review_result = "approved"

    print(f"  [review] 质量评分: {quality:.2f}, 结论: {review_result}")

    return {
        "review_comments": comments,
        "quality_score": quality,
        "generation_stage": "review",
    }


def node_finalize(state: ArticleState) -> dict:
    """节点4：最终化。"""
    draft = state.get('draft', '')
    quality = state.get('quality_score', 0)

    print(f"  [finalize] 最终化 (质量: {quality:.2f})...")

    finalized = (
        f"{draft}\n\n"
        f"---\n"
        f"*质量评分: {quality:.0%} | 尝试次数: {state.get('attempt_count', 1)}*\n"
        f"*最终版本*"
    )

    return {
        "final_version": finalized,
        "generation_stage": "final",
    }


def node_improve(state: ArticleState) -> dict:
    """节点5：改进策略（在rewind后尝试不同的方法）。"""
    print(f"  [improve] 应用改进策略...")
    attempt = state.get('attempt_count', 1)

    # 根据尝试次数使用不同的改进策略
    if attempt == 1:
        strategy = "增加数据引用"
        improved_outline = state.get('outline', '') + "\n*[改进策略: 增加数据支撑]*"
    elif attempt == 2:
        strategy = "重新组织结构"
        improved_outline = state.get('outline', '') + "\n*[改进策略: 优化章节结构]*"
    else:
        strategy = "专家模板"
        improved_outline = state.get('outline', '') + "\n*[改进策略: 使用专家模板]*"

    print(f"  [improve] 策略: {strategy}")

    return {
        "outline": improved_outline,
        "attempt_count": attempt + 1,
        "generation_stage": "outline",
    }


# ============================================================================
# 路由
# ============================================================================

def route_after_review(state: ArticleState) -> Literal["improve", "finalize"]:
    """审查后的路由。"""
    if (state.get('quality_score', 0) < 0.80
            and state.get('attempt_count', 0) < 3):
        return "improve"
    return "finalize"


# ============================================================================
# 构建图
# ============================================================================

def build_article_graph() -> StateGraph:
    """构建带改进循环的文章生成图。"""
    builder = StateGraph(ArticleState)

    builder.add_node("outline", node_create_outline)
    builder.add_node("draft", node_write_draft)
    builder.add_node("review", node_review_draft)
    builder.add_node("improve", node_improve)
    builder.add_node("finalize", node_finalize)

    builder.set_entry_point("outline")
    builder.add_edge("outline", "draft")
    builder.add_edge("draft", "review")

    builder.add_conditional_edges(
        "review",
        route_after_review,
        {
            "improve": "improve",
            "finalize": "finalize",
        }
    )

    # improve 循环回 outline（形成改进循环）
    builder.add_edge("improve", "outline")
    builder.add_edge("finalize", END)

    return builder


# ============================================================================
# 时间旅行功能
# ============================================================================

class TimeTraveler:
    """
    时间旅行管理器：封装 LangGraph 的时间旅行API。
    """

    def __init__(self, graph, checkpointer):
        self.graph = graph
        self.checkpointer = checkpointer

    def get_history(self, thread_id: str) -> list[dict]:
        """
        获取指定线程的完整状态历史。

        Args:
            thread_id: 对话线程ID

        Returns:
            按时间排序的状态快照列表
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            history = list(self.graph.get_state_history(config))
        except AttributeError:
            # 内存模式下的简化实现
            if hasattr(self.checkpointer, '_storage'):
                checkpoints = self.checkpointer._storage.get(thread_id, [])
                history = []
                for i, cp in enumerate(checkpoints):
                    history.append({
                        "checkpoint_id": str(i),
                        "values": cp[0] if isinstance(cp, tuple) else cp,
                    })
            else:
                history = []

        # 格式化为可读格式
        formatted = []
        for item in history:
            if hasattr(item, 'values') and hasattr(item, 'config'):
                state = item.values
                checkpoint_id = item.config.get("configurable", {}).get("checkpoint_id", "?")
                formatted.append({
                    "checkpoint_id": checkpoint_id,
                    "generation_stage": state.get("generation_stage", "?"),
                    "quality_score": state.get("quality_score", 0),
                    "attempt_count": state.get("attempt_count", 1),
                    "state_keys": list(state.keys()),
                })
            elif isinstance(item, dict):
                formatted.append(item)

        return formatted

    def rewind_to(self, thread_id: str, checkpoint_id: str) -> dict:
        """
        回退到指定的检查点并重新执行。

        Args:
            thread_id: 线程ID
            checkpoint_id: 目标检查点ID

        Returns:
            重新执行后的结果
        """
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

        # 从指定检查点重新 invoke
        result = self.graph.invoke(None, config)
        return result

    def rewind_and_retry(
        self,
        thread_id: str,
        detection_fn,
        new_params: dict = None,
    ) -> dict:
        """
        "Rewind and Retry" 完整流程：
        1. 检测输出是否不好
        2. 找到上一个好的检查点
        3. 从该点重新执行（应用新参数/策略）

        Args:
            thread_id: 线程ID
            detection_fn: 检测函数 (state) -> bool，返回True表示输出不好
            new_params: 重新执行时使用的新参数

        Returns:
            重新执行后的结果
        """
        history = self.get_history(thread_id)
        if not history:
            raise ValueError(f"未找到线程 {thread_id} 的历史状态")

        print(f"\n  🔍 检查历史状态 ({len(history)} 个检查点)...")

        # 从最新的开始往前找
        bad_checkpoint = None
        for i, checkpoint in enumerate(history):
            state = checkpoint.get('values', checkpoint)
            if detection_fn(state):
                bad_checkpoint = checkpoint
                print(f"    ⚠️ 检测到坏输出在检查点 {checkpoint['checkpoint_id']}")
                print(f"       阶段: {checkpoint.get('generation_stage', '?')}")
                print(f"       质量: {checkpoint.get('quality_score', '?')}")
                break

        if bad_checkpoint is None:
            print(f"    ✅ 所有检查点都合格，无需回退")
            return None

        # 找到上一个好的检查点（往前退一步）
        bad_index = history.index(bad_checkpoint) if bad_checkpoint in history else 0
        good_index = min(bad_index + 1, len(history) - 1)

        if good_index == bad_index:
            # 没有更早的检查点，使用第一个
            target = history[0]
        else:
            target = history[good_index]

        print(f"    ⏪ 回退到检查点: {target['checkpoint_id']}")
        print(f"       阶段: {target.get('generation_stage', '?')}")

        # 执行回退
        result = self.rewind_to(thread_id, target['checkpoint_id'])

        if new_params:
            # 如果有新参数，再次 invoke 应用新参数
            result = self.graph.invoke(new_params, {
                "configurable": {"thread_id": thread_id}
            })

        return result

    def print_history_summary(self, thread_id: str):
        """打印历史摘要。"""
        history = self.get_history(thread_id)

        print(f"\n  📜 状态历史 (线程: {thread_id})")
        print(f"  {'检查点':<15} {'阶段':<10} {'质量':<8} {'尝试':<6} {'字段数':<8}")
        print(f"  {'─' * 15} {'─' * 10} {'─' * 8} {'─' * 6} {'─' * 8}")
        for item in history:
            print(
                f"  {item.get('checkpoint_id', '?'):<15} "
                f"{item.get('generation_stage', '?'):<10} "
                f"{item.get('quality_score', 0):.2f}    "
                f"{item.get('attempt_count', '?'):<6} "
                f"{len(item.get('state_keys', [])):<8}"
            )
        print(f"  ───")
        print(f"  共 {len(history)} 个检查点")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  LangGraph 企业级 - 时间旅行（Time Travel）演示")
    print("=" * 72)

    # 构建图
    checkpointer = MemorySaver()
    builder = build_article_graph()
    graph = builder.compile(checkpointer=checkpointer)

    traveler = TimeTraveler(graph, checkpointer)

    # ---- Demo 1: 正常执行并记录历史 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 完整执行并记录历史")
    print("=" * 72)

    thread_id = "article_demo_01"

    result = graph.invoke(
        {
            "topic": "2025年企业AI应用趋势",
            "outline": "",
            "draft": "",
            "review_comments": "",
            "final_version": "",
            "quality_score": 0.0,
            "generation_stage": "start",
            "attempt_count": 1,
        },
        {"configurable": {"thread_id": thread_id}},
    )

    print(f"\n📊 最终结果:")
    print(f"  阶段: {result.get('generation_stage')}")
    print(f"  质量: {result.get('quality_score', 0):.2f}")
    print(f"  尝试: {result.get('attempt_count', 0)}")

    # 查看历史
    traveler.print_history_summary(thread_id)

    # ---- Demo 2: Rewind and Retry ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: Rewind and Retry 流程")
    print("=" * 72)

    # 定义一个检测函数：质量低于0.7就是坏输出
    def is_bad_output(state: dict) -> bool:
        quality = state.get('quality_score', 0)
        stage = state.get('generation_stage', '')
        return quality < 0.70 and stage in ('review', 'draft')

    traveler2 = TimeTraveler(graph, checkpointer)
    traveler2.print_history_summary(thread_id)

    retry_result = traveler2.rewind_and_retry(
        thread_id=thread_id,
        detection_fn=is_bad_output,
        new_params={"topic": "2025年企业AI应用趋势（修订版）"},
    )

    if retry_result:
        print(f"\n  ✅ 重试结果: 质量={retry_result.get('quality_score', 0):.2f}")

    # ---- Demo 3: 并行分支（从同一检查点尝试不同策略） ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 从同一检查点尝试不同策略（A/B测试）")
    print("=" * 72)

    # 创建基础执行
    base_thread_id = "article_ab_test"
    base_result = graph.invoke(
        {
            "topic": "测试主题",
            "outline": "",
            "draft": "",
            "review_comments": "",
            "final_version": "",
            "quality_score": 0.0,
            "generation_stage": "start",
            "attempt_count": 1,
        },
        {"configurable": {"thread_id": base_thread_id}},
    )

    print(f"\n  基础策略: 质量={base_result.get('quality_score', 0):.2f}")

    # 时间旅行回退到 outline 之后
    # 从 outline checkpoint 尝试不同的 draft 策略
    traveler3 = TimeTraveler(graph, checkpointer)
    history = traveler3.get_history(base_thread_id)

    # 找到 outline 阶段的检查点
    outline_checkpoints = [
        h for h in history
        if h.get('generation_stage') == 'outline'
    ]

    if outline_checkpoints:
        print(f"\n  找到 outline 检查点: {outline_checkpoints[0]['checkpoint_id']}")
        print(f"  从此处可以尝试不同的 draft 策略（A/B测试）")

    print("\n" + "=" * 72)
    print("  时间旅行演示完成！")
    print("=" * 72)
