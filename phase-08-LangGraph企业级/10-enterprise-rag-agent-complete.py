#!/usr/bin/env python3
"""
Phase 08 - LangGraph 企业级：10 - 企业级 RAG Agent 完整方案 (THE CAPSTONE) 🏆

集成从 Phase 01-08 学习的所有核心技术：
  - Master-Sub 架构（意图分类 → 多子图路由）
  - RAG 子图（检索 → 生成 → 验证 → 重试循环）
  - Tool-calling 子图（工具调用 → 验证）
  - Human-approval 子图（审批 + interrupt）
  - PostgreSQL Checkpointer（状态持久化）
  - Redis 分布式锁（防重复执行）
  - RabbitMQ 异步任务队列
  - 环境变量配置管理

架构全景：
  ┌─────────────────────────────────────────────────────────────┐
  │                    Enterprise RAG Agent                      │
  │                                                              │
  │  API Layer (FastAPI / Flask)                                 │
  │  ├── POST /ask  → 202 Accepted → RabbitMQ → 异步处理         │
  │  ├── GET  /status/{id} → 查询结果                             │
  │  └── POST /approve/{id} → Human-in-the-Loop                  │
  │                                                              │
  │  Core Layer                                                  │
  │  ├── MasterGraph (意图分类 + 路由)                            │
  │  │   ├── RAGSubGraph (检索→生成→验证循环)                     │
  │  │   ├── ToolSubGraph (工具调用→验证)                         │
  │  │   └── HumanApprovalSubGraph (审批流程)                     │
  │  └── Checkpointer (PostgreSQL 持久化)                         │
  │                                                              │
  │  Infrastructure Layer                                        │
  │  ├── Redis Lock (防重复执行)                                  │
  │  ├── RabbitMQ (异步任务削峰)                                  │
  │  └── Environment Config (pydantic-settings)                   │
  └─────────────────────────────────────────────────────────────┘

配置方式：所有配置通过环境变量注入。
  APP_ENV=development|staging|production
  OPENAI_API_KEY=sk-...
  PGHOST=localhost
  REDIS_HOST=localhost
  RABBITMQ_HOST=localhost
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import traceback
from typing import TypedDict, Annotated, Literal, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# 配置管理
from pydantic_settings import BaseSettings
from pydantic import Field

# ============================================================================
# 1. 环境配置（pydantic-settings）
# ============================================================================

class AppSettings(BaseSettings):
    """
    企业级配置：所有配置通过环境变量注入。

    用法：
        settings = AppSettings()
        print(settings.llm_model)  # 从环境变量 LLM_MODEL 读取
    """
    # App
    app_env: str = Field(default="development", alias="APP_ENV")
    app_name: str = Field(default="EnterpriseRAGAgent", alias="APP_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # LLM
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    llm_fast_model: str = Field(default="gpt-4o-mini", alias="LLM_FAST_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # RAG
    rag_max_retries: int = Field(default=3, alias="RAG_MAX_RETRIES")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_similarity_threshold: float = Field(default=0.75, alias="RAG_SIMILARITY_THRESHOLD")

    # PostgreSQL Checkpointer
    pg_host: str = Field(default="localhost", alias="PGHOST")
    pg_port: int = Field(default=5432, alias="PGPORT")
    pg_user: str = Field(default="postgres", alias="PGUSER")
    pg_password: str = Field(default="postgres", alias="PGPASSWORD")
    pg_database: str = Field(default="langgraph", alias="PGDATABASE")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_lock_ttl_ms: int = Field(default=30000, alias="REDIS_LOCK_TTL_MS")

    # RabbitMQ
    rabbitmq_host: str = Field(default="localhost", alias="RABBITMQ_HOST")
    rabbitmq_port: int = Field(default=5672, alias="RABBITMQ_PORT")
    rabbitmq_queue: str = Field(default="rag_tasks", alias="RABBITMQ_QUEUE")
    rabbitmq_dlq: str = Field(default="rag_tasks_dlq", alias="RABBITMQ_DLQ")
    mq_max_retries: int = Field(default=3, alias="MQ_MAX_RETRIES")

    # Human-in-the-Loop
    hitl_enabled: bool = Field(default=True, alias="HITL_ENABLED")
    hitl_approval_threshold: float = Field(default=0.70, alias="HITL_APPROVAL_THRESHOLD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def pg_connection_string(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def redis_url(self) -> str:
        pw = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{pw}{self.redis_host}:{self.redis_port}/0"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


# ============================================================================
# 2. 日志配置
# ============================================================================

def setup_logging(settings: AppSettings) -> logging.Logger:
    """配置企业级日志。"""
    logger = logging.getLogger(settings.app_name)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ============================================================================
# 3. Redis 分布式锁（精简版）
# ============================================================================

class SimpleDistributedLock:
    """
    精简的分布式锁实现。
    生产环境使用 08-redis-distributed-lock.py 的完整版本。
    """
    _global_locks: dict[str, tuple[str, float]] = {}  # key → (owner_id, expires_at)
    _global_lock = threading.Lock()

    def __init__(self, lock_key: str, ttl_ms: int = 30000, acquire_timeout_ms: int = 10_000):
        self.lock_key = f"rag-lock:{lock_key}"
        self.ttl_ms = ttl_ms
        self.acquire_timeout_ms = acquire_timeout_ms
        self.owner_id = uuid.uuid4().hex[:12]
        self._acquired = False

    def acquire(self) -> bool:
        start = time.time()
        while True:
            with self._global_lock:
                current = self._global_locks.get(self.lock_key)
                if current is None or time.time() > current[1]:
                    self._global_locks[self.lock_key] = (self.owner_id, time.time() + self.ttl_ms / 1000)
                    self._acquired = True
                    return True
            if (time.time() - start) * 1000 >= self.acquire_timeout_ms:
                return False
            time.sleep(0.05)

    def release(self) -> bool:
        if not self._acquired:
            return False
        with self._global_lock:
            current = self._global_locks.get(self.lock_key)
            if current and current[0] == self.owner_id:
                del self._global_locks[self.lock_key]
        self._acquired = False
        return True

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"获取锁超时: {self.lock_key}")
        return self

    def __exit__(self, *args):
        self.release()
        return False


# ============================================================================
# 4. 消息队列（精简版）
# ============================================================================

class SimpleMessageQueue:
    """精简的消息队列实现。"""
    def __init__(self):
        from collections import deque
        self._queue: deque = deque()
        self._dlq: deque = deque()
        self._results: dict[str, Any] = {}
        self._lock = threading.Lock()

    def publish(self, task_id: str, payload: dict) -> None:
        with self._lock:
            self._queue.append({"id": task_id, "payload": payload, "retries": 0})

    def consume(self) -> Optional[dict]:
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def ack(self, task_id: str, result: Any) -> None:
        with self._lock:
            self._results[task_id] = {"status": "completed", "result": result}

    def nack(self, task: dict, error: str, max_retries: int) -> None:
        with self._lock:
            task["retries"] += 1
            task["last_error"] = error
            if task["retries"] < max_retries:
                self._queue.append(task)
            else:
                self._dlq.append(task)
                self._results[task["id"]] = {"status": "dead_letter", "error": error}

    def get_result(self, task_id: str) -> Optional[dict]:
        return self._results.get(task_id)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "queue_length": len(self._queue),
                "dlq_length": len(self._dlq),
                "completed": len([r for r in self._results.values() if r.get("status") == "completed"]),
                "dead_letter": len([r for r in self._results.values() if r.get("status") == "dead_letter"]),
            }


# ============================================================================
# 5. Master State 和 Sub States
# ============================================================================

class EnterpriseRAGState(TypedDict):
    """
    企业级 RAG Agent 的全局 State。

    此 State 横跨 Master 图和所有子图。
    """
    # 请求信息
    messages: Annotated[list, add_messages]
    user_query: str
    session_id: str

    # 意图分类
    intent: str  # "rag" | "tool" | "human_approval" | "general"

    # RAG 字段
    retrieved_docs: list
    rag_answer: str
    rag_validation: str  # "pass" | "fail"
    rag_retry_count: int

    # Tool 字段
    tool_name: str
    tool_args: dict
    tool_result: str

    # Human Approval 字段
    approval_required: bool
    approval_status: str  # "pending" | "approved" | "rejected"
    approval_reason: str
    risk_score: float

    # 最终响应
    final_response: str
    final_confidence: float
    status: str  # "pending" | "processing" | "completed" | "failed" | "awaiting_approval"
    execution_log: list
    elapsed_ms: float


# ============================================================================
# 6. 知识库和检索（模拟）
# ============================================================================

ENTERPRISE_KNOWLEDGE_BASE = [
    {
        "id": "kb_001",
        "title": "LangGraph StateGraph API",
        "content": "StateGraph 是 LangGraph 的核心抽象。使用 StateGraph(StateClass) "
                   "创建构建器，通过 add_node() 添加节点函数，add_edge() 和 "
                   "add_conditional_edges() 定义路由逻辑。图通过 compile() 编译。",
    },
    {
        "id": "kb_002",
        "title": "Checkpointer 持久化机制",
        "content": "LangGraph 支持 PostgreSQL 和 SQLite 两种 Checkpointer 后端。"
                   "PostgresSaver 需要先执行 setup() 创建表。每个对话线程通过 "
                   "thread_id 隔离。支持 get_state_history() 时间旅行功能。",
    },
    {
        "id": "kb_003",
        "title": "Human-in-the-Loop 中断",
        "content": "HITL 通过 interrupt() 函数在节点内部暂停图执行。"
                   "恢复使用 Command(resume=human_input)。适用于审批流程、"
                   "敏感内容审查、大额交易确认等场景。",
    },
    {
        "id": "kb_004",
        "title": "RAG 系统最佳实践",
        "content": "生产级 RAG 系统需要：1) 混合检索（Dense+Sparse）；"
                   "2) 重排序（Re-ranking）；3) 查询重写；4) Self-RAG 自纠正。"
                   "chunk_size 建议 512 tokens，overlap 50 tokens。"
                   "相似度阈值通常设为 0.75。",
    },
    {
        "id": "kb_005",
        "title": "多Agent协调模式",
        "content": "多Agent系统有7种核心模式：Sequential（串行）、Hierarchical（层级管理）、"
                   "Debate（辩论）、Escalation（升级）、Blackboard（黑板）、"
                   "Ensemble（集成）、Role-Playing（角色扮演）。"
                   "企业场景常用 Hierarchical + Escalation 组合。",
    },
    {
        "id": "kb_006",
        "title": "企业部署架构",
        "content": "企业级部署需要：API网关（限流/认证）、消息队列（削峰填谷）、"
                   "分布式锁（防重复）、状态持久化（Checkpointer）、"
                   "监控告警（Prometheus/Grafana）、日志聚合（ELK）。",
    },
    {
        "id": "kb_007",
        "title": "安全最佳实践",
        "content": "AI系统安全四层防护：1) 输入层（注入检测、内容过滤）；"
                   "2) 模型层（Prompt注入防护、越狱检测）；"
                   "3) 工具层（权限最小化、参数校验）；"
                   "4) 输出层（内容审核、脱敏）。",
    },
]


def enterprise_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """企业级检索（模拟向量搜索）。"""
    query_lower = query.lower()
    scored = []

    for doc in ENTERPRISE_KNOWLEDGE_BASE:
        content_lower = doc["content"].lower()
        score = sum(1 for word in query_lower.split() if word in content_lower)
        score += sum(2 for word in query_lower.split() if word in doc["title"].lower())
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ============================================================================
# 7. 子图实现
# ============================================================================

def rag_retrieve(state: EnterpriseRAGState) -> dict:
    """RAG子图：检索节点。"""
    query = state['user_query']
    docs = enterprise_retrieve(query, top_k=5)
    return {
        "retrieved_docs": docs,
        "execution_log": [f"RAG.retrieve: 找到 {len(docs)} 篇文档"],
    }

def rag_generate(state: EnterpriseRAGState) -> dict:
    """RAG子图：生成节点。"""
    docs = state.get('retrieved_docs', [])
    query = state['user_query']

    if not docs:
        answer = f"知识库中未找到关于'{query[:60]}'的相关信息。请尝试其他关键词。"
    else:
        parts = [f"- [{d['title']}]: {d['content'][:200]}..." for d in docs[:3]]
        answer = (
            f"## 基于知识库的回答\n\n"
            f"关于'{query[:80]}'：\n\n"
            f"{chr(10).join(parts)}\n\n"
            f"**总结**：综合以上 {len(docs)} 篇文档的信息，上述内容应能覆盖您的问题。\n"
            f"*来源：{', '.join(d['title'] for d in docs)}*"
        )

    return {
        "rag_answer": answer,
        "execution_log": [f"RAG.generate: 生成答案 ({len(answer)} 字符)"],
    }

def rag_validate(state: EnterpriseRAGState) -> dict:
    """RAG子图：验证节点。"""
    answer = state.get('rag_answer', '')
    docs = state.get('retrieved_docs', [])

    has_basis = any(d['title'].lower() in answer.lower() for d in docs) if docs else False
    reasonable_length = 50 <= len(answer) <= 10000

    if has_basis and reasonable_length:
        return {"rag_validation": "pass"}
    elif state.get('rag_retry_count', 0) >= 2:
        return {"rag_validation": "pass"}
    return {"rag_validation": "fail", "rag_retry_count": state.get('rag_retry_count', 0) + 1}

def rag_route(state: EnterpriseRAGState) -> Literal["retrieve", "finalize"]:
    if state.get('rag_validation') == "fail" and state.get('rag_retry_count', 0) < 2:
        return "retrieve"
    return "finalize"

def rag_finalize(state: EnterpriseRAGState) -> dict:
    return {
        "final_response": state.get('rag_answer', ''),
        "final_confidence": 0.85,
        "status": "completed",
    }


def tool_call(state: EnterpriseRAGState) -> dict:
    """Tool子图：工具调用节点。"""
    tool = state.get('tool_name', 'unknown')
    args = state.get('tool_args', {})

    if tool == "calculator":
        expr = args.get('expression', '0')
        result = f"计算: {expr} = 42 (模拟)"
    elif tool == "web_search":
        query = args.get('query', '')
        result = f"搜索'{query}': 3条结果 (模拟)"
    elif tool == "database_query":
        result = "数据库查询: 5条记录 (模拟)"
    else:
        result = f"工具 {tool} 执行: 成功 (模拟)"

    return {"tool_result": result, "status": "completed"}

def tool_finalize(state: EnterpriseRAGState) -> dict:
    return {
        "final_response": state.get('tool_result', ''),
        "final_confidence": 0.90,
        "status": "completed",
    }


def human_request(state: EnterpriseRAGState) -> dict:
    """Human子图：请求审批。"""
    risk = 50 if any(w in state['user_query'] for w in ['大额', '删除', '删除全部']) else 20
    needs_approval = risk > 30
    return {"risk_score": risk, "approval_required": needs_approval}

def human_wait(state: EnterpriseRAGState) -> dict:
    """Human子图：等待审批（模拟自动批准）。"""
    return {"approval_status": "approved", "approval_reason": "低风险自动批准（演示模式）"}

def human_confirm(state: EnterpriseRAGState) -> dict:
    return {
        "final_response": f"审批完成: {state.get('approval_status')}",
        "status": "completed",
        "final_confidence": 1.0,
    }


# ============================================================================
# 8. Master 图节点
# ============================================================================

def classify_intent(state: EnterpriseRAGState) -> dict:
    """意图分类：根据用户查询决定路由目标。"""
    query = state['user_query'].lower()

    if any(w in query for w in ['rag', '知识库', '文档', '检索', '查询', '搜索']):
        intent = "rag"
    elif any(w in query for w in ['计算', '工具', 'tool', '数据']):
        intent = "tool"
        if '计算' in query:
            tool = "calculator"
        else:
            tool = "database_query"
        return {"intent": intent, "tool_name": tool, "tool_args": {"query": query}}
    elif any(w in query for w in ['审批', '批准', '大额', '删除']):
        intent = "human_approval"
    else:
        intent = "general"

    return {"intent": intent, "execution_log": [f"Intent: {intent}"]}


def router(state: EnterpriseRAGState) -> Literal["rag_entry", "tool_entry", "human_entry", "general"]:
    return {
        "rag": "rag_entry",
        "tool": "tool_entry",
        "human_approval": "human_entry",
        "general": "general",
    }.get(state['intent'], "general")


def rag_entry(state: EnterpriseRAGState) -> dict:
    """RAG子图入口：编排检索→生成→验证→最终化。"""
    logs = state.get('execution_log', []) + ["→ RAG SubGraph"]

    # Step 1: Retrieve
    r1 = rag_retrieve(state)
    state.update(r1)

    # Step 2: Generate
    r2 = rag_generate(state)
    state.update(r2)

    # Step 3: Validate (with retry loop)
    for retry in range(3):
        r3 = rag_validate(state)
        state.update(r3)
        if state.get('rag_validation') == "pass":
            break
        if retry < 2:
            state.update(rag_retrieve(state))
            state.update(rag_generate(state))

    # Step 4: Finalize
    r4 = rag_finalize(state)
    state.update(r4)
    state['execution_log'] = logs + [f"RAG完成: retries={state.get('rag_retry_count', 0)}"]
    return state


def tool_entry(state: EnterpriseRAGState) -> dict:
    """Tool子图入口。"""
    state.update(tool_call(state))
    state.update(tool_finalize(state))
    state['execution_log'] = state.get('execution_log', []) + ["→ Tool SubGraph 完成"]
    return state


def human_entry(state: EnterpriseRAGState) -> dict:
    """Human子图入口。"""
    state.update(human_request(state))
    state.update(human_wait(state))
    state.update(human_confirm(state))
    state['execution_log'] = state.get('execution_log', []) + [
        f"→ Human Approval: {state.get('approval_status')}"
    ]
    return state


def general_response(state: EnterpriseRAGState) -> dict:
    """通用响应。"""
    return {
        "final_response": f"收到您的问题：'{state['user_query'][:100]}'。"
                          f"这是一个通用问题。如需使用知识库检索，请明确说明。",
        "final_confidence": 0.75,
        "status": "completed",
    }


def compose_response(state: EnterpriseRAGState) -> dict:
    """合成最终响应。"""
    response = state.get('final_response', '处理完成')
    return {
        "messages": [AIMessage(content=response)],
        "execution_log": state.get('execution_log', []) + ["→ 响应合成完成"],
    }


# ============================================================================
# 9. 构建 Enterprise RAG Graph
# ============================================================================

def build_enterprise_rag_graph() -> StateGraph:
    """构建完整的企业级 RAG Agent StateGraph。"""
    builder = StateGraph(EnterpriseRAGState)

    builder.add_node("classify", classify_intent)
    builder.add_node("rag_entry", rag_entry)
    builder.add_node("tool_entry", tool_entry)
    builder.add_node("human_entry", human_entry)
    builder.add_node("general", general_response)
    builder.add_node("compose", compose_response)

    builder.set_entry_point("classify")

    builder.add_conditional_edges(
        "classify", router, {
            "rag_entry": "rag_entry",
            "tool_entry": "tool_entry",
            "human_entry": "human_entry",
            "general": "general",
        }
    )

    for node in ["rag_entry", "tool_entry", "human_entry", "general"]:
        builder.add_edge(node, "compose")

    builder.add_edge("compose", END)

    return builder


# ============================================================================
# 10. Enterprise RAG Agent 类
# ============================================================================

class EnterpriseRAGAgent:
    """
    企业级 RAG Agent：整合所有 Phase 01-08 学习的技术。

    用法：
        agent = EnterpriseRAGAgent()
        result = agent.ask("什么是 LangGraph 的 StateGraph？")
        print(result["final_response"])
    """

    def __init__(self, settings: AppSettings = None):
        self.settings = settings or AppSettings()
        self.logger = setup_logging(self.settings)

        # 构建图
        builder = build_enterprise_rag_graph()

        # Checkpointer
        checkpointer = MemorySaver()  # 生产环境用 PostgresSaver
        self.graph = builder.compile(checkpointer=checkpointer)

        # 消息队列
        self.message_queue = SimpleMessageQueue()
        self._consumer_thread: Optional[threading.Thread] = None
        self._consumer_running = False

        # 统计
        self.total_requests = 0
        self.total_success = 0
        self.total_failures = 0
        self.total_lock_contention = 0

        self.logger.info(f"EnterpriseRAGAgent 已初始化 (env={self.settings.app_env})")

    # ---- 同步接口 ----

    def ask(self, user_query: str, session_id: str = None) -> dict:
        """
        同步提问（阻塞直到完成）。

        Args:
            user_query: 用户问题
            session_id: 会话ID（用于多轮对话状态持久化）

        Returns:
            完整的处理结果
        """
        if session_id is None:
            session_id = uuid.uuid4().hex[:12]

        self.total_requests += 1
        start = time.time()

        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "session_id": session_id,
            "intent": "",
            "retrieved_docs": [],
            "rag_answer": "",
            "rag_validation": "",
            "rag_retry_count": 0,
            "tool_name": "",
            "tool_args": {},
            "tool_result": "",
            "approval_required": False,
            "approval_status": "",
            "approval_reason": "",
            "risk_score": 0.0,
            "final_response": "",
            "final_confidence": 0.0,
            "status": "pending",
            "execution_log": [],
            "elapsed_ms": 0,
        }

        try:
            # 分布式锁保护
            lock = SimpleDistributedLock(
                lock_key=f"rag-ask:{session_id}",
                ttl_ms=self.settings.redis_lock_ttl_ms,
                acquire_timeout_ms=5000,
            )

            with lock:
                result = self.graph.invoke(initial_state, config)
                elapsed = (time.time() - start) * 1000
                result["elapsed_ms"] = elapsed
                result["status"] = "completed"

                if result.get("final_response"):
                    self.total_success += 1
                else:
                    self.total_failures += 1

                self.logger.info(
                    f"请求完成: session={session_id} "
                    f"intent={result.get('intent')} "
                    f"confidence={result.get('final_confidence', 0):.2f} "
                    f"elapsed={elapsed:.0f}ms"
                )

                return result

        except TimeoutError:
            self.total_lock_contention += 1
            return {
                "status": "busy",
                "final_response": "系统繁忙，请稍后重试。",
                "session_id": session_id,
                "elapsed_ms": (time.time() - start) * 1000,
            }
        except Exception as e:
            self.total_failures += 1
            self.logger.error(f"请求失败: {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "final_response": f"处理出错: {str(e)}",
                "session_id": session_id,
                "elapsed_ms": (time.time() - start) * 1000,
            }

    # ---- 异步接口 ----

    def ask_async(self, user_query: str, session_id: str = None) -> dict:
        """
        异步提问（立即返回 202，后台处理）。

        客户端应轮询 GET /status/{session_id} 获取结果。
        """
        if session_id is None:
            session_id = uuid.uuid4().hex[:12]

        # 发布到消息队列
        self.message_queue.publish(session_id, {
            "query": user_query,
            "session_id": session_id,
        })

        # 启动消费者（如果尚未运行）
        self._ensure_consumer_running()

        return {
            "status": "accepted",
            "session_id": session_id,
            "message": "任务已提交，请轮询查询结果。",
            "status_url": f"/status/{session_id}",
        }

    def _ensure_consumer_running(self):
        """确保消费者线程在运行。"""
        if not self._consumer_running:
            self._consumer_running = True
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                daemon=True,
                name="rag-consumer",
            )
            self._consumer_thread.start()

    def _consumer_loop(self):
        """消费者循环：从消息队列拉取任务并处理。"""
        while self._consumer_running:
            task = self.message_queue.consume()
            if task:
                try:
                    result = self.ask(task["payload"]["query"], task["id"])
                    if result.get("status") == "completed":
                        self.message_queue.ack(task["id"], result)
                    else:
                        self.message_queue.nack(task, result.get("final_response", "未知错误"), 3)
                except Exception as e:
                    self.message_queue.nack(task, str(e), 3)
            else:
                time.sleep(0.2)

    def get_async_result(self, session_id: str) -> Optional[dict]:
        """获取异步任务的结果。"""
        return self.message_queue.get_result(session_id)

    # ---- 统计 ----

    def get_stats(self) -> dict:
        """获取 Agent 运行统计。"""
        return {
            "app": self.settings.app_name,
            "env": self.settings.app_env,
            "requests": {
                "total": self.total_requests,
                "success": self.total_success,
                "failures": self.total_failures,
                "lock_contention": self.total_lock_contention,
                "success_rate": round(
                    self.total_success / max(1, self.total_requests), 3
                ),
            },
            "queue": self.message_queue.stats,
            "config": {
                "llm_model": self.settings.llm_model,
                "rag_max_retries": self.settings.rag_max_retries,
                "hitl_enabled": self.settings.hitl_enabled,
                "redis_lock_ttl_ms": self.settings.redis_lock_ttl_ms,
                "mq_max_retries": self.settings.mq_max_retries,
            },
        }

    def health_check(self) -> dict:
        """健康检查。"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "app": self.settings.app_name,
            "version": "1.0.0",
            "uptime_requests": self.total_requests,
        }

    def shutdown(self):
        """优雅关闭。"""
        self._consumer_running = False
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5.0)
        self.logger.info("EnterpriseRAGAgent 已关闭")


# ============================================================================
# 11. 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  🏆 Enterprise RAG Agent - THE CAPSTONE")
    print("=" * 72)

    # 加载配置
    settings = AppSettings()
    print(f"\n  ⚙️ 配置: env={settings.app_env} | model={settings.llm_model}")
    print(f"  RAG: max_retries={settings.rag_max_retries} | top_k={settings.rag_top_k}")
    print(f"  HITL: {'enabled' if settings.hitl_enabled else 'disabled'}")

    # 创建 Agent
    agent = EnterpriseRAGAgent(settings)

    # ---- Demo 1: 同步 RAG 查询 ----

    print("\n\n" + "=" * 72)
    print("  Demo 1: 同步 RAG 查询")
    print("=" * 72)

    result1 = agent.ask("搜索LangGraph StateGraph的相关文档")

    print(f"\n  📊 结果:")
    print(f"  意图: {result1.get('intent')}")
    print(f"  状态: {result1.get('status')}")
    print(f"  置信度: {result1.get('final_confidence', 0):.2f}")
    print(f"  耗时: {result1.get('elapsed_ms', 0):.0f}ms")
    print(f"\n  响应前300字符:")
    print(f"  {result1.get('final_response', 'N/A')[:300]}")

    # ---- Demo 2: Tool 查询 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: Tool 调用查询")
    print("=" * 72)

    result2 = agent.ask("帮我计算 3.14 * 2.71 的结果")

    print(f"\n  📊 结果:")
    print(f"  意图: {result2.get('intent')}")
    print(f"  工具: {result2.get('tool_name')}")
    print(f"  响应: {result2.get('final_response', 'N/A')[:200]}")

    # ---- Demo 3: 异步模式 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 异步提交 + 轮询查询")
    print("=" * 72)

    async_response = agent.ask_async("查询RAG系统最佳实践")
    session_id = async_response["session_id"]
    print(f"\n  提交响应: {json.dumps(async_response, ensure_ascii=False, indent=2)}")

    # 轮询等待结果
    print(f"\n  轮询结果...")
    for _ in range(20):
        result = agent.get_async_result(session_id)
        if result:
            print(f"\n  📊 异步结果:")
            print(f"  状态: {result['status']}")
            if result.get('result'):
                print(f"  响应: {result['result'].get('final_response', 'N/A')[:200]}")
            break
        time.sleep(0.3)
    else:
        print(f"  ⚠️ 未在预期时间内收到结果")

    # ---- Demo 4: 健康检查 ----

    print("\n\n" + "=" * 72)
    print("  Demo 4: 健康检查 + 统计")
    print("=" * 72)

    health = agent.health_check()
    print(f"\n  健康检查: {json.dumps(health, ensure_ascii=False, indent=2)}")

    stats = agent.get_stats()
    print(f"\n  运行统计:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # ---- Demo 5: Human-in-the-Loop ----

    print("\n\n" + "=" * 72)
    print("  Demo 5: Human-in-the-Loop 审批")
    print("=" * 72)

    result5 = agent.ask("请帮我审批一笔大额转账")
    print(f"\n  📊 结果:")
    print(f"  意图: {result5.get('intent')}")
    print(f"  审批状态: {result5.get('approval_status', 'N/A')}")
    print(f"  风险评分: {result5.get('risk_score', 0)}")

    # Clean shutdown
    agent.shutdown()

    print("\n" + "=" * 72)
    print("  🏆 Enterprise RAG Agent 演示完成！")
    print("  This capstone ties together Phases 01-08.")
    print("  From Prompt Engineering → RAG → Agents → LangGraph → Enterprise.")
    print("=" * 72)
