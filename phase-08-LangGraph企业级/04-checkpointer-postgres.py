#!/usr/bin/env python3
"""
Phase 08 - LangGraph 企业级：04 - PostgreSQL Checkpointer

PostgreSQL 状态持久化：跨进程/重启保留对话状态。
Thread 隔离：每个对话线程独立持久化。
断点恢复：服务重启后可从上次状态继续。

核心概念：
  ┌─────────────────────────────────────────────────────┐
  │                 PostgreSQL                          │
  │  ┌──────────────┐  ┌──────────────┐                │
  │  │ checkpoints  │  │ checkpoint_  │                │
  │  │              │  │ writes       │                │
  │  │ thread_id: A │  │ (每个节点的  │                │
  │  │   step 0     │  │  状态快照)   │                │
  │  │   step 1     │  │              │                │
  │  │   step 2     │  │              │                │
  │  ├──────────────┤  │              │                │
  │  │ thread_id: B │  │              │                │
  │  │   step 0     │  │              │                │
  │  │   step 1     │  │              │                │
  │  └──────────────┘  └──────────────┘                │
  └─────────────────────────────────────────────────────┘

使用方法：
  1. 初始化：postgres_saver = PostgresSaver.from_conn_string(conn_string)
  2. 建表：postgres_saver.setup()
  3. 编译：graph = builder.compile(checkpointer=postgres_saver)
  4. 调用：graph.invoke(state, config={"configurable": {"thread_id": "user_123"}})
  5. 恢复：相同 thread_id 再次 invoke，从上次状态继续
"""

import os
import json
import time
import asyncio
from typing import TypedDict, Annotated, Literal, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresSaver


# ============================================================================
# State 定义
# ============================================================================

class ChatState(TypedDict):
    """简单的对话状态（用于演示 Checkpointer）。"""
    messages: Annotated[list, add_messages]
    user_name: str
    turn_count: int
    last_topic: str


# ============================================================================
# 节点函数
# ============================================================================

def node_welcome(state: ChatState) -> dict:
    """欢迎节点。"""
    name = state.get('user_name', '用户')
    turn = state.get('turn_count', 0) + 1
    msg = AIMessage(content=f"欢迎回来，{name}！这是第 {turn} 轮对话。")
    print(f"  [welcome] turn={turn}")
    return {
        "messages": [msg],
        "turn_count": turn,
    }


def node_chat(state: ChatState) -> dict:
    """聊天节点。"""
    # 获取最后一条用户消息
    last_human = None
    for msg in reversed(state.get('messages', [])):
        if hasattr(msg, 'type') and msg.type == 'human':
            last_human = msg.content
            break

    turn = state.get('turn_count', 0)
    topic = f"topic_{turn}"

    response = AIMessage(
        content=(
            f"收到您的消息！当前是第 {turn} 轮对话。\n"
            f"您说: {str(last_human)[:100]}\n"
            f"我已经记住了我们之前的所有对话内容。"
        )
    )
    print(f"  [chat] turn={turn}")
    return {
        "messages": [response],
        "last_topic": topic,
    }


def node_summarize(state: ChatState) -> dict:
    """总结节点。"""
    msg_count = len(state.get('messages', []))
    turns = state.get('turn_count', 0)
    summary = AIMessage(
        content=(
            f"📊 对话总结:\n"
            f"- 总消息数: {msg_count}\n"
            f"- 对话轮数: {turns}\n"
            f"- 最后话题: {state.get('last_topic', 'N/A')}\n"
            f"- 状态已持久化到 PostgreSQL"
        )
    )
    print(f"  [summarize] messages={msg_count}, turns={turns}")
    return {"messages": [summary]}


def route_after_chat(state: ChatState) -> Literal["chat", "summarize"]:
    """路由：3轮后自动总结。"""
    if state.get('turn_count', 0) >= 3:
        return "summarize"
    return "chat"


# ============================================================================
# 构建图
# ============================================================================

def build_chat_graph() -> StateGraph:
    """构建带 Checkpointer 的聊天图。"""
    builder = StateGraph(ChatState)

    builder.add_node("welcome", node_welcome)
    builder.add_node("chat", node_chat)
    builder.add_node("summarize", node_summarize)

    builder.set_entry_point("welcome")
    builder.add_edge("welcome", "chat")

    builder.add_conditional_edges(
        "chat",
        route_after_chat,
        {
            "chat": "chat",        # 循环！
            "summarize": "summarize",
        }
    )

    builder.add_edge("summarize", END)

    return builder


# ============================================================================
# Checkpointer 初始化
# ============================================================================

# SQL 建表脚本（PostgreSQL）
SETUP_SQL = """
-- LangGraph Checkpointer 所需的数据库表
-- 在使用 PostgresSaver 之前执行此脚本

-- 检查点表：每个节点的状态快照
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- 检查点写入表：每个节点写入的通道值
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- 检查点Blobs表：大型数据
CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

-- 索引（加速查询）
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
    ON checkpoints(thread_id);

CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread
    ON checkpoint_writes(thread_id, checkpoint_id);
"""


def get_postgres_connection_string() -> str:
    """
    获取 PostgreSQL 连接字符串。
    从环境变量读取，如果未设置则使用默认值。

    环境变量：
        PGHOST: 主机 (默认: localhost)
        PGPORT: 端口 (默认: 5432)
        PGUSER: 用户 (默认: postgres)
        PGPASSWORD: 密码 (默认: postgres)
        PGDATABASE: 数据库名 (默认: langgraph)
    """
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")
    database = os.getenv("PGDATABASE", "langgraph")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


class CheckpointerManager:
    """
    Checkpointer 管理器：封装 PostgresSaver 的生命周期。

    使用方式:
        async with CheckpointerManager() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            result = graph.invoke(state, config)
    """

    def __init__(self, conn_string: str = None):
        self.conn_string = conn_string or get_postgres_connection_string()
        self._saver: Optional[PostgresSaver] = None

    async def __aenter__(self) -> PostgresSaver:
        self._saver = PostgresSaver.from_conn_string(self.conn_string)
        await self._saver.setup()
        print(f"✅ PostgresSaver 初始化完成")
        print(f"   连接: {self._conn_string_masked()}")
        return self._saver

    async def __aexit__(self, *args):
        if self._saver:
            # PostgresSaver 在使用完后不需要显式关闭
            # 连接池由 asyncpg 自动管理
            pass

    def _conn_string_masked(self) -> str:
        """脱敏的连接字符串。"""
        parts = self.conn_string.split('@')
        if len(parts) == 2:
            return f"***@{parts[1]}"
        return "***"

    @staticmethod
    def get_sql_setup() -> str:
        return SETUP_SQL


# ============================================================================
# 模拟 Checkpointer（用于无PostgreSQL环境的演示）
# ============================================================================

class MemorySaver:
    """
    内存 Checkpointer（用于演示/测试，无需 PostgreSQL）。

    LangGraph 内置的 MemorySaver，这里提供一个兼容接口的简化实现。
    生产环境始终使用 PostgresSaver。
    """

    def __init__(self):
        self._storage: dict[str, list] = {}

    def get_tuple(self, config: dict) -> Optional[tuple]:
        """获取最后保存的状态。"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        if thread_id in self._storage and self._storage[thread_id]:
            return self._storage[thread_id][-1]
        return None

    def put(
        self,
        config: dict,
        checkpoint: tuple,
        metadata: dict = None,
        new_versions: dict = None,
    ) -> dict:
        """保存状态。"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        if thread_id not in self._storage:
            self._storage[thread_id] = []
        self._storage[thread_id].append(checkpoint)
        return {"configurable": {"thread_id": thread_id}}

    def list(self, config: dict, **kwargs) -> list:
        """列出某线程的所有检查点。"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        return self._storage.get(thread_id, [])

    async def asetup(self):
        """异步初始化（兼容接口）。"""
        pass

    async def aget_tuple(self, config: dict) -> Optional[tuple]:
        """异步获取。"""
        return self.get_tuple(config)


# ============================================================================
# 演示函数
# ============================================================================

def demo_thread_isolation(use_real_postgres: bool = False):
    """
    演示 Thread 隔离：
    两个不同的 thread_id 拥有完全独立的对话状态。
    """
    print("=" * 72)
    print("  Demo: Thread 隔离")
    print("=" * 72)

    if use_real_postgres:
        print("  使用 PostgreSQL Checkpointer")
        saver = PostgresSaver.from_conn_string(get_postgres_connection_string())
        asyncio.run(saver.setup())
    else:
        print("  使用内存 Checkpointer（演示模式）")
        from langgraph.checkpoint.memory import MemorySaver as LangGraphMemorySaver
        saver = LangGraphMemorySaver()

    # 编译图
    builder = build_chat_graph()
    graph = builder.compile(checkpointer=saver)

    # 线程A：用户 Alice
    print("\n  ─── 线程 A (user_alice) ───")
    config_a = {"configurable": {"thread_id": "user_alice"}}
    state_a = {"messages": [], "user_name": "Alice", "turn_count": 0, "last_topic": ""}

    result_a1 = graph.invoke(state_a, config_a)
    print(f"    第1轮: turn_count={result_a1.get('turn_count', '?')}")

    result_a2 = graph.invoke(
        {"messages": [HumanMessage(content="我想了解更多关于RAG的信息")]},
        config_a,
    )
    print(f"    第2轮: turn_count={result_a2.get('turn_count', '?')}")

    # 线程B：用户 Bob（完全独立）
    print("\n  ─── 线程 B (user_bob) ───")
    config_b = {"configurable": {"thread_id": "user_bob"}}
    state_b = {"messages": [], "user_name": "Bob", "turn_count": 0, "last_topic": ""}

    result_b1 = graph.invoke(state_b, config_b)
    print(f"    第1轮: turn_count={result_b1.get('turn_count', '?')}")

    # 验证隔离性
    print(f"\n  ✅ Thread 隔离验证:")
    print(f"    Alice 状态: turn_count={result_a2.get('turn_count', '?')} (应该 >= 2)")
    print(f"    Bob 状态:   turn_count={result_b1.get('turn_count', '?')} (应该 = 1)")
    print(f"    隔离正确: {result_b1.get('turn_count', '?') == 1} (Bob不受Alice影响)")


def demo_resume_after_restart(use_real_postgres: bool = False):
    """
    演示断点恢复：
    模拟"服务重启"后使用相同 thread_id 继续对话。
    """
    print("\n" + "=" * 72)
    print("  Demo: 断点恢复（模拟服务重启）")
    print("=" * 72)

    if use_real_postgres:
        saver = PostgresSaver.from_conn_string(get_postgres_connection_string())
        asyncio.run(saver.setup())
    else:
        from langgraph.checkpoint.memory import MemorySaver as LangGraphMemorySaver
        saver = LangGraphMemorySaver()

    # 编译图
    builder = build_chat_graph()
    graph = builder.compile(checkpointer=saver)

    thread_id = "user_restart_demo"

    # 第一次对话
    print("\n  [首次会话]")
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": [], "user_name": "Charlie", "turn_count": 0, "last_topic": ""}

    result1 = graph.invoke(state, config)
    print(f"    turn_count: {result1.get('turn_count', '?')}")

    result2 = graph.invoke(
        {"messages": [HumanMessage(content="你好")]},
        config,
    )
    print(f"    turn_count: {result2.get('turn_count', '?')}")

    # 模拟"服务重启" —— 创建新的 saver 和 graph
    print("\n  [模拟服务重启...]")
    if use_real_postgres:
        saver2 = PostgresSaver.from_conn_string(get_postgres_connection_string())
        asyncio.run(saver2.setup())
    else:
        from langgraph.checkpoint.memory import MemorySaver as LangGraphMemorySaver
        saver2 = LangGraphMemorySaver()

    builder2 = build_chat_graph()
    graph2 = builder2.compile(checkpointer=saver2)

    # 使用相同的 thread_id 恢复！
    print("  [恢复对话]")
    config2 = {"configurable": {"thread_id": thread_id}}
    result3 = graph2.invoke(
        {"messages": [HumanMessage(content="我们刚才聊到哪里了？")]},
        config2,
    )
    print(f"    turn_count: {result3.get('turn_count', '?')}")
    print(f"    消息数: {len(result3.get('messages', []))}")
    print(f"    最后一条AI消息: {result3['messages'][-1].content[:80]}...")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 72)
    print("  LangGraph 企业级 - PostgreSQL Checkpointer 演示")
    print("=" * 72)

    # 检查是否使用真实 PostgreSQL
    use_postgres = "--postgres" in sys.argv

    if use_postgres:
        print("\n  🐘 模式: PostgreSQL Checkpointer")
        print(f"  连接: {get_postgres_connection_string()}")
        print(f"\n  SQL 建表脚本:")
        print(SETUP_SQL)
    else:
        print("\n  💾 模式: 内存 Checkpointer（演示）")
        print("  如需使用 PostgreSQL，请:")
        print("    1. 启动 PostgreSQL: docker run -d --name pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16")
        print("    2. 创建数据库: docker exec pg createdb -U postgres langgraph")
        print("    3. 设置环境变量: export PGPASSWORD=postgres")
        print("    4. 运行: python 04-checkpointer-postgres.py --postgres")

    # Demo 1: Thread 隔离
    demo_thread_isolation(use_real_postgres=use_postgres)

    # 注意：内存模式不支持跨实例恢复，此处仅演示概念
    if use_postgres:
        demo_resume_after_restart(use_real_postgres=True)
    else:
        print("\n  ⚠️ 内存模式下无法演示跨实例恢复，请使用 --postgres 参数")

    print("\n" + "=" * 72)
    print("  PostgreSQL Checkpointer 演示完成！")
    print("=" * 72)
