#!/usr/bin/env python3
"""
流式记忆交互 (Streaming Memory Interaction)
===============================================
实现实时流式对话中的记忆积累、检索和写入，支持背压控制和早退检索。

核心组件：
  - StreamingMemoryAccumulator: 流式记忆积累器
  - EarlyExitMemoryRetriever: 早退记忆检索器
  - BackpressureAwareMemoryWriter: 背压感知记忆写入器
  - StreamingTurnTracker: 流式轮次追踪器
  - StreamingMemoryOrchestrator: 流式记忆编排器

流式处理管道：
  用户输入 → Accumulator(积攒chunks) → Committer(提交turn)
           → Retriever(早退检索) → Writer(背压写入) → 下一轮
"""

from __future__ import annotations

import re
import math
import time
import json
import uuid
import heapq
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque


# ============================================================================
# 枚举定义
# ============================================================================

class StreamState(Enum):
    """流式状态"""
    IDLE = auto()            # 空闲
    ACCUMULATING = auto()    # 正在积累chunks
    COMMITTING = auto()      # 正在提交turn
    RETRIEVING = auto()      # 正在检索
    WRITING = auto()         # 正在写入
    BACKPRESSURE = auto()    # 背压等待


class BackpressureLevel(Enum):
    """背压等级"""
    NONE = auto()            # 无背压
    LIGHT = auto()           # 轻微：减慢写入
    MODERATE = auto()        # 中等：暂停写入，排队
    HEAVY = auto()           # 严重：丢弃低优先级写入
    OVERFLOW = auto()        # 溢出：紧急降级


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class StreamChunk:
    """流式数据块"""
    chunk_id: str
    content: str
    sequence_num: int
    is_final: bool = False        # 是否为本turn的最后一个chunk
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommittedTurn:
    """已提交的对话轮次"""
    turn_id: str
    chunks: List[StreamChunk]
    full_text: str
    committed_at: float
    turn_number: int
    role: str = "user"            # "user" or "assistant"


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    memories: List[Dict[str, Any]]
    retrieval_time_ms: float
    early_exit: bool = False      # 是否早退
    confidence: float = 0.5
    retrieval_depth: int = 0      # 检索深度（搜索了多少层）


@dataclass
class WriteResult:
    """写入结果"""
    memory_id: str
    success: bool
    backpressure_delay_ms: float
    queued: bool = False
    error: Optional[str] = None


# ============================================================================
# 流式记忆积累器
# ============================================================================

class StreamingMemoryAccumulator:
    """
    流式记忆积累器。

    负责将实时流式输入碎片收集、重组为完整对话轮次。
    """

    def __init__(self, max_chunks_per_turn: int = 100):
        self._current_chunks: List[StreamChunk] = []
        self._committed_turns: List[CommittedTurn] = []
        self._turn_counter: int = 0
        self._max_chunks_per_turn = max_chunks_per_turn
        self._current_role: str = "user"
        self._state: StreamState = StreamState.IDLE

    def ingest_chunk(self, content: str, is_final: bool = False,
                      role: str = "user",
                      metadata: Optional[Dict] = None) -> Optional[CommittedTurn]:
        """
        接收一个流式数据块。

        Args:
            content: 文本内容
            is_final: 是否是本周的最后一个块
            role: 角色 ("user" / "assistant")
            metadata: 附加元数据

        Returns:
            如果is_final为True，返回提交的CommittedTurn；否则返回None
        """
        self._state = StreamState.ACCUMULATING

        # 角色切换时提交旧turn
        if role != self._current_role and self._current_chunks:
            old_turn = self._commit_current_turn()
            self._current_chunks = []

        self._current_role = role

        # 添加chunk
        chunk = StreamChunk(
            chunk_id=f"chunk-{uuid.uuid4().hex[:8]}",
            content=content,
            sequence_num=len(self._current_chunks),
            is_final=is_final,
            metadata=metadata or {},
        )
        self._current_chunks.append(chunk)

        # 限制chunk数量
        if len(self._current_chunks) > self._max_chunks_per_turn:
            # 强制提交
            return self._commit_current_turn()

        # 如果是最终chunk，提交
        if is_final:
            return self._commit_current_turn()

        return None

    def _commit_current_turn(self) -> CommittedTurn:
        """提交当前积累的所有chunks为一个turn"""
        self._state = StreamState.COMMITTING
        self._turn_counter += 1

        full_text = "".join(c.content for c in self._current_chunks)
        turn = CommittedTurn(
            turn_id=f"turn-{self._turn_counter:04d}",
            chunks=list(self._current_chunks),
            full_text=full_text,
            committed_at=time.time(),
            turn_number=self._turn_counter,
            role=self._current_role,
        )

        self._committed_turns.append(turn)
        self._current_chunks = []
        self._state = StreamState.IDLE

        return turn

    def commit_turn(self) -> Optional[CommittedTurn]:
        """手动提交当前turn（即使未到达is_final）"""
        if self._current_chunks:
            return self._commit_current_turn()
        return None

    def rollback_turn(self) -> Optional[str]:
        """
        回滚最后提交的turn。

        Returns:
            被回滚的turn的完整文本
        """
        if self._committed_turns:
            rolled_back = self._committed_turns.pop()
            self._turn_counter -= 1
            self._state = StreamState.IDLE
            return rolled_back.full_text
        return None

    def get_latest_turn(self) -> Optional[CommittedTurn]:
        """获取最新提交的turn"""
        return self._committed_turns[-1] if self._committed_turns else None

    def get_full_conversation(self) -> str:
        """获取完整对话文本"""
        return "\n".join(
            f"[{t.role}]: {t.full_text}"
            for t in self._committed_turns
        )

    def get_turn_count(self) -> int:
        """获取提交的turn数"""
        return len(self._committed_turns)

    @property
    def state(self) -> StreamState:
        return self._state


# ============================================================================
# 早退记忆检索器
# ============================================================================

class EarlyExitMemoryRetriever:
    """
    早退记忆检索器。

    核心思想：在检索过程中，当置信度已经足够高时提前终止检索，
    节省计算资源。适用于实时流式场景中的低延迟要求。

    早退策略：
    1. Top-1置信度高 → 立即返回
    2. 连续N个结果无相关 → 停止检索
    3. 检索时间超过预算 → 截断返回
    """

    def __init__(self, early_exit_confidence: float = 0.8,
                  max_retrieval_ms: float = 50.0,
                  patience: int = 3):
        """
        Args:
            early_exit_confidence: 触发早退的置信度阈值
            max_retrieval_ms: 最大检索时间(毫秒)
            patience: 连续无结果容忍数
        """
        self.early_exit_confidence = early_exit_confidence
        self.max_retrieval_ms = max_retrieval_ms
        self.patience = patience
        self._retrieval_stats: List[Dict[str, Any]] = []

    def speculative_retrieve(self, query: str,
                              memory_index: List[Dict[str, Any]],
                              query_embedding: Optional[List[float]] = None
                              ) -> RetrievalResult:
        """
        带早退机制的推测性检索。

        支持：
        - 置信度早退
        - 时间预算早退
        - 连续无结果早退

        Args:
            query: 查询文本
            memory_index: 记忆索引列表
            query_embedding: 查询向量（可选）

        Returns:
            RetrievalResult
        """
        start_time = time.time()
        results = []
        consecutive_irrelevant = 0
        retrieval_depth = 0

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # 按分数排序的记忆列表（模拟索引扫描）
        # 实际场景中是向量相似度排序后的列表
        for memory in memory_index:
            retrieval_depth += 1
            content = memory.get("content", "").lower()
            mem_terms = set(content.split())

            # 计算相关性分数
            overlap = query_terms & mem_terms
            if overlap:
                score = len(overlap) / max(len(query_terms), 1)
                if score > 0.1:
                    results.append({**memory, "retrieval_score": round(score, 4)})
                    consecutive_irrelevant = 0
                else:
                    consecutive_irrelevant += 1
            else:
                consecutive_irrelevant += 1

            # 检查早退条件

            # 条件1：Top结果置信度高
            if results and results[0].get("retrieval_score", 0) >= self.early_exit_confidence:
                elapsed_ms = (time.time() - start_time) * 1000
                return RetrievalResult(
                    query=query,
                    memories=results[:5],
                    retrieval_time_ms=round(elapsed_ms, 2),
                    early_exit=True,
                    confidence=results[0]["retrieval_score"],
                    retrieval_depth=retrieval_depth,
                )

            # 条件2：超时
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= self.max_retrieval_ms:
                return RetrievalResult(
                    query=query,
                    memories=results[:5],
                    retrieval_time_ms=round(elapsed_ms, 2),
                    early_exit=True,
                    confidence=results[0]["retrieval_score"] if results else 0.0,
                    retrieval_depth=retrieval_depth,
                )

            # 条件3：连续无结果
            if consecutive_irrelevant >= self.patience and results:
                # 至少有一些结果才早退
                elapsed_ms = (time.time() - start_time) * 1000
                return RetrievalResult(
                    query=query,
                    memories=results[:5],
                    retrieval_time_ms=round(elapsed_ms, 2),
                    early_exit=True,
                    confidence=results[0]["retrieval_score"] if results else 0.0,
                    retrieval_depth=retrieval_depth,
                )

        # 正常完成
        elapsed_ms = (time.time() - start_time) * 1000
        self._retrieval_stats.append({
            "query": query[:30],
            "depth": retrieval_depth,
            "results": len(results),
            "early_exit": False,
            "time_ms": round(elapsed_ms, 2),
        })

        return RetrievalResult(
            query=query,
            memories=results[:5],
            retrieval_time_ms=round(elapsed_ms, 2),
            early_exit=False,
            confidence=results[0]["retrieval_score"] if results else 0.0,
            retrieval_depth=retrieval_depth,
        )

    def refine_results(self, results: List[Dict[str, Any]],
                         query: str) -> List[Dict[str, Any]]:
        """
        对已有检索结果进行二次精炼。

        在获得更多上下文或用户反馈后重新排序。
        """
        if not results:
            return results

        query_lower = query.lower()
        refined = []

        for r in results:
            content = r.get("content", "").lower()

            # 精确匹配加分
            exact_bonus = 1.5 if query_lower in content else 1.0

            # 实体匹配加分
            entity_bonus = 1.0
            entities = re.findall(r'[A-Z][a-z]+|[一-鿿]{2,4}', query)
            for entity in entities:
                if entity.lower() in content:
                    entity_bonus += 0.1

            refined_score = r.get("retrieval_score", 0.5) * exact_bonus * entity_bonus
            refined.append({**r, "refined_score": round(refined_score, 4)})

        refined.sort(key=lambda x: x.get("refined_score", 0), reverse=True)
        return refined

    def get_stats(self) -> Dict[str, Any]:
        """获取检索统计"""
        if not self._retrieval_stats:
            return {"total_searches": 0}

        early_exits = sum(1 for s in self._retrieval_stats if s["early_exit"])
        avg_time = sum(s["time_ms"] for s in self._retrieval_stats) / len(self._retrieval_stats)
        avg_depth = sum(s["depth"] for s in self._retrieval_stats) / len(self._retrieval_stats)

        return {
            "total_searches": len(self._retrieval_stats),
            "early_exit_rate": round(early_exits / len(self._retrieval_stats) * 100, 1),
            "avg_retrieval_time_ms": round(avg_time, 2),
            "avg_search_depth": int(avg_depth),
        }


# ============================================================================
# 背压感知记忆写入器
# ============================================================================

class BackpressureAwareMemoryWriter:
    """
    背压感知的记忆写入器。

    当系统负载过高时（背压），自动调整写入策略：
    - NONE: 正常写入
    - LIGHT: 延迟写入（微小延迟）
    - MODERATE: 排队写入
    - HEAVY: 丢弃低优先级写入
    - OVERFLOW: 紧急降级，仅保留核心记忆
    """

    def __init__(self, max_queue_size: int = 100,
                  write_delay_ms: float = 10.0):
        self.max_queue_size = max_queue_size
        self.write_delay_ms = write_delay_ms
        self._write_queue: deque[Dict[str, Any]] = deque()
        self._written_memories: List[Dict[str, Any]] = []
        self._backpressure_level: BackpressureLevel = BackpressureLevel.NONE
        self._memory_id_counter = 0
        self._dropped_count = 0

    def write_with_backpressure(self, content: str,
                                 importance: float = 0.5,
                                 metadata: Optional[Dict] = None) -> WriteResult:
        """
        在背压感知下写入记忆。

        根据当前背压等级决定写入策略。

        Args:
            content: 记忆内容
            importance: 重要性 [0, 1]
            metadata: 元数据

        Returns:
            WriteResult
        """
        start_time = time.time()

        if self._backpressure_level == BackpressureLevel.NONE:
            # 正常写入
            memory_id = self._do_write(content, importance, metadata)
            return WriteResult(
                memory_id=memory_id,
                success=True,
                backpressure_delay_ms=0.0,
            )

        elif self._backpressure_level == BackpressureLevel.LIGHT:
            # 轻微延迟后写入
            time.sleep(self.write_delay_ms / 1000.0)
            memory_id = self._do_write(content, importance, metadata)
            return WriteResult(
                memory_id=memory_id,
                success=True,
                backpressure_delay_ms=self.write_delay_ms,
            )

        elif self._backpressure_level == BackpressureLevel.MODERATE:
            # 排队写入
            if len(self._write_queue) < self.max_queue_size:
                self._write_queue.append({
                    "content": content,
                    "importance": importance,
                    "metadata": metadata,
                })
                return WriteResult(
                    memory_id="pending",
                    success=True,
                    backpressure_delay_ms=0.0,
                    queued=True,
                )
            else:
                # 队列满，丢弃最低优先级
                min_item = min(self._write_queue,
                              key=lambda x: x.get("importance", 0))
                self._write_queue.remove(min_item)
                self._dropped_count += 1
                self._write_queue.append({
                    "content": content,
                    "importance": importance,
                    "metadata": metadata,
                })
                return WriteResult(
                    memory_id="pending",
                    success=True,
                    backpressure_delay_ms=0.0,
                    queued=True,
                )

        elif self._backpressure_level in (BackpressureLevel.HEAVY,
                                            BackpressureLevel.OVERFLOW):
            # 仅写入高优先级记忆
            if importance > 0.7:
                memory_id = self._do_write(content, importance, metadata)
                return WriteResult(
                    memory_id=memory_id,
                    success=True,
                    backpressure_delay_ms=0.0,
                )
            else:
                self._dropped_count += 1
                return WriteResult(
                    memory_id="dropped",
                    success=False,
                    backpressure_delay_ms=0.0,
                    error="背压: 低优先级记忆被丢弃",
                )

    def _do_write(self, content: str, importance: float,
                    metadata: Optional[Dict]) -> str:
        """执行实际写入操作"""
        self._memory_id_counter += 1
        memory_id = f"smem-{self._memory_id_counter:06d}"

        self._written_memories.append({
            "id": memory_id,
            "content": content,
            "importance": importance,
            "metadata": metadata or {},
            "written_at": time.time(),
        })
        return memory_id

    def flush_queue(self) -> int:
        """清空写入队列，返回写入数量"""
        written = 0
        while self._write_queue:
            item = self._write_queue.popleft()
            self._do_write(item["content"], item["importance"], item["metadata"])
            written += 1
        return written

    def set_backpressure(self, level: BackpressureLevel):
        """设置背压等级"""
        self._backpressure_level = level

    def get_current_backpressure(self) -> BackpressureLevel:
        """获取当前背压等级"""
        return self._backpressure_level

    def get_stats(self) -> Dict[str, Any]:
        """获取写入统计"""
        return {
            "written_count": len(self._written_memories),
            "queue_size": len(self._write_queue),
            "dropped_count": self._dropped_count,
            "backpressure_level": self._backpressure_level.name,
        }


# ============================================================================
# 流式轮次追踪器
# ============================================================================

class StreamingTurnTracker:
    """
    流式轮次追踪器。

    追踪每轮对话的状态和延迟指标。
    """

    def __init__(self):
        self._turn_log: List[Dict[str, Any]] = []
        self._current_turn_start: Optional[float] = None
        self._cumulative_latency_ms: float = 0.0

    def start_turn(self, turn_number: int):
        """开始追踪新一轮"""
        self._current_turn_start = time.time()

    def end_turn(self, turn_number: int, role: str,
                  chunk_count: int, text_length: int):
        """结束追踪当前轮"""
        if self._current_turn_start is None:
            return

        elapsed = (time.time() - self._current_turn_start) * 1000
        self._cumulative_latency_ms += elapsed

        self._turn_log.append({
            "turn": turn_number,
            "role": role,
            "chunks": chunk_count,
            "text_length": text_length,
            "latency_ms": round(elapsed, 2),
        })

        self._current_turn_start = None

    def get_latency_stats(self) -> Dict[str, Any]:
        """获取延迟统计"""
        if not self._turn_log:
            return {"total_turns": 0}

        user_turns = [t for t in self._turn_log if t["role"] == "user"]
        assistant_turns = [t for t in self._turn_log if t["role"] == "assistant"]

        return {
            "total_turns": len(self._turn_log),
            "user_turns": len(user_turns),
            "assistant_turns": len(assistant_turns),
            "avg_user_latency_ms": round(
                sum(t["latency_ms"] for t in user_turns) / max(len(user_turns), 1), 2
            ),
            "avg_assistant_latency_ms": round(
                sum(t["latency_ms"] for t in assistant_turns) / max(len(assistant_turns), 1), 2
            ),
            "total_latency_ms": round(self._cumulative_latency_ms, 2),
        }


# ============================================================================
# 流式记忆编排器
# ============================================================================

class StreamingMemoryOrchestrator:
    """
    流式记忆编排器。

    整合积累器、检索器、写入器和追踪器的总控引擎。
    """

    def __init__(self):
        self.accumulator = StreamingMemoryAccumulator()
        self.retriever = EarlyExitMemoryRetriever()
        self.writer = BackpressureAwareMemoryWriter()
        self.tracker = StreamingTurnTracker()

        # 模拟的记忆索引
        self._memory_index: List[Dict[str, Any]] = []
        self._init_memory_index()

    def _init_memory_index(self):
        """初始化记忆索引"""
        base_memories = [
            "用户偏好：喜欢简洁的回答，不需要冗长的解释",
            "用户上次询问了关于Python装饰器的问题",
            "用户所在时区：UTC+8 (北京时间)",
            "用户工作：软件工程师，主要使用Python和TypeScript",
            "用户最近在做一个RAG系统的项目",
            "用户的订阅到期时间为2026年12月",
            "用户喜欢在回答中包含代码示例",
            "用户之前反馈不喜欢过多的表情符号",
            "用户的团队有5名开发人员",
            "用户项目使用PostgreSQL作为主数据库",
            "用户API密钥的最后4位是ABCD",
            "用户最近的部署是2天前",
            "用户偏好暗色模式UI",
            "用户有过敏史：对花粉过敏",
            "用户代码仓库：github.com/user/project",
        ]
        for i, content in enumerate(base_memories):
            self._memory_index.append({
                "id": f"idx-{i+1:03d}",
                "content": content,
                "mem_type": "preference" if "偏好" in content or "喜欢" in content else "fact",
                "created_at": time.time() - (i * 86400),
                "importance": 0.5 + (i % 5) * 0.1,
            })

    def process_stream(self, content: str, is_final: bool = False,
                        role: str = "user") -> Dict[str, Any]:
        """
        处理一个流式chunk。

        Pipeline: Accumulate → Commit → Retrieve → Write

        Returns:
            处理结果
        """
        result: Dict[str, Any] = {
            "chunk_processed": True,
            "turn_committed": False,
        }

        # 1. 积累
        committed_turn = self.accumulator.ingest_chunk(content, is_final, role)
        if committed_turn:
            result["turn_committed"] = True
            result["turn_id"] = committed_turn.turn_id
            result["full_text"] = committed_turn.full_text[:100]
            self.tracker.end_turn(
                committed_turn.turn_number, role,
                len(committed_turn.chunks), len(committed_turn.full_text)
            )

            # 2. 检索
            retrieval = self.retriever.speculative_retrieve(
                committed_turn.full_text, self._memory_index
            )
            result["retrieval"] = {
                "result_count": len(retrieval.memories),
                "early_exit": retrieval.early_exit,
                "confidence": retrieval.confidence,
                "time_ms": retrieval.retrieval_time_ms,
            }

            # 3. 写入（背压感知）
            importance = self._estimate_importance(committed_turn.full_text)
            write_result = self.writer.write_with_backpressure(
                committed_turn.full_text, importance,
                metadata={"turn_id": committed_turn.turn_id},
            )
            result["write"] = {
                "memory_id": write_result.memory_id,
                "queued": write_result.queued,
                "backpressure_level": self.writer.get_current_backpressure().name,
            }

            # 4. 调整背压
            self._adjust_backpressure()

        return result

    def _estimate_importance(self, text: str) -> float:
        """估算一段文本的重要性"""
        importance = 0.5

        # 包含关键词加分
        important_keywords = ["重要", "必须", "紧急", "confidential", "bug", "error", "错误"]
        for kw in important_keywords:
            if kw.lower() in text.lower():
                importance += 0.1

        # 文本长度加权（太短可能不重要）
        if len(text) > 100:
            importance += 0.1
        if len(text) > 500:
            importance += 0.1

        return min(importance, 1.0)

    def _adjust_backpressure(self):
        """根据系统状态调整背压等级"""
        queue_size = len(self.writer._write_queue)
        if queue_size > 80:
            self.writer.set_backpressure(BackpressureLevel.OVERFLOW)
        elif queue_size > 50:
            self.writer.set_backpressure(BackpressureLevel.HEAVY)
        elif queue_size > 20:
            self.writer.set_backpressure(BackpressureLevel.MODERATE)
        elif queue_size > 5:
            self.writer.set_backpressure(BackpressureLevel.LIGHT)
        else:
            self.writer.set_backpressure(BackpressureLevel.NONE)

    def start_turn(self, turn_number: int):
        """标记开始新的一轮"""
        self.tracker.start_turn(turn_number)

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """获取编排器统计"""
        return {
            "turns": self.accumulator.get_turn_count(),
            "latency": self.tracker.get_latency_stats(),
            "retrieval": self.retriever.get_stats(),
            "writer": self.writer.get_stats(),
        }


# ============================================================================
# Demo: 5轮流式对话
# ============================================================================

def demo():
    """完整演示"""
    print("=" * 70)
    print("  流式记忆交互 (Streaming Memory Interaction) 演示")
    print("=" * 70)

    orchestrator = StreamingMemoryOrchestrator()

    # 模拟5轮流式对话
    conversation = [
        # (role, chunks)  每个chunk是一个(文本, 是否最后)对
        ("user", [
            ("你好", False),
            ("，我想问你", False),
            ("一些关于Python的问题", True),
        ]),
        ("assistant", [
            ("当然可以！", False),
            ("我很乐意帮助你解决Python问题。", True),
        ]),
        ("user", [
            ("请解释", False),
            ("装饰器的工作原理", False),
            ("，最好有代码示例", True),
        ]),
        ("assistant", [
            ("装饰器是Python中的语法糖，", False),
            ("允许你在不修改原函数的情况下", False),
            ("增加额外功能。", False),
            ("这是一个简单示例：@timer装饰器...", True),
        ]),
        ("user", [
            ("谢谢你的解释，", False),
            ("非常清楚明了！", True),
        ]),
    ]

    # ---- 1. 流式处理5轮对话 ----
    print("\n【1. 流式处理5轮对话】")
    print("-" * 50)

    for turn_idx, (role, chunks) in enumerate(conversation, start=1):
        orchestrator.start_turn(turn_idx)
        print(f"\n  --- Turn {turn_idx} ({role}) ---")

        for chunk_text, is_final in chunks:
            result = orchestrator.process_stream(chunk_text, is_final, role)
            preview = chunk_text[:30]
            print(f"    Chunk: '{preview}{'...' if len(chunk_text)>30 else ''}' "
                  f"{'[FINAL]' if is_final else ''}")

            if result.get("turn_committed"):
                retrieval = result.get("retrieval", {})
                write_result = result.get("write", {})
                print(f"      ✓ Turn提交: {result.get('turn_id')}")
                print(f"      ✓ 检索结果: {retrieval.get('result_count', 0)} 条 "
                      f"(早退={'是' if retrieval.get('early_exit') else '否'}, "
                      f"置信度={retrieval.get('confidence', 0):.2f})")
                print(f"      ✓ 写入: {write_result.get('memory_id')} "
                      f"(排队={'是' if write_result.get('queued') else '否'})")

    # ---- 2. 早退检索演示 ----
    print("\n【2. 早退检索详细演示】")
    print("-" * 50)

    retriever = EarlyExitMemoryRetriever(
        early_exit_confidence=0.5,
        max_retrieval_ms=20.0,
        patience=2,
    )

    test_queries = [
        "Python装饰器怎么用",
        "数据库PostgreSQL配置",
        "项目部署相关问题",
    ]

    for query in test_queries:
        result = retriever.speculative_retrieve(
            query, orchestrator._memory_index
        )
        print(f"\n  查询: '{query}'")
        print(f"    结果数: {len(result.memories)}")
        print(f"    早退: {'是' if result.early_exit else '否'}")
        print(f"    检索深度: {result.retrieval_depth}/{len(orchestrator._memory_index)}")
        print(f"    耗时: {result.retrieval_time_ms:.2f}ms")
        for mem in result.memories[:2]:
            content = mem.get("content", "")[:50]
            score = mem.get("retrieval_score", 0)
            print(f"      [{score:.3f}] {content}...")

    # ---- 3. 背压写入演示 ----
    print("\n【3. 背压写入演示】")
    print("-" * 50)

    writer = BackpressureAwareMemoryWriter(max_queue_size=10)
    levels_to_test = [
        BackpressureLevel.NONE,
        BackpressureLevel.LIGHT,
        BackpressureLevel.MODERATE,
        BackpressureLevel.HEAVY,
        BackpressureLevel.OVERFLOW,
    ]

    for level in levels_to_test:
        writer.set_backpressure(level)
        result = writer.write_with_backpressure(
            "测试记忆内容",
            importance=0.3 if level in (BackpressureLevel.HEAVY, BackpressureLevel.OVERFLOW) else 0.6,
        )
        print(f"  [{level.name:10s}]: 重要性=({'高' if 'dropped' not in result.memory_id else '低'})"
              f" → {result.memory_id}"
              f" {'(已排队)' if result.queued else ''}"
              f" {'(已丢弃)' if not result.success else ''}"
              f" 延迟={result.backpressure_delay_ms:.0f}ms")

    # 清空队列
    flushed = writer.flush_queue()
    print(f"\n  清空队列: {flushed} 条写入")

    # ---- 4. 回滚测试 ----
    print("\n【4. 回滚测试】")
    print("-" * 50)

    acc = StreamingMemoryAccumulator()
    acc.ingest_chunk("第一部分", is_final=False)
    acc.ingest_chunk("第二部分", is_final=True)
    print(f"  当前turn数: {acc.get_turn_count()}")
    print(f"  最新turn: {acc.get_latest_turn().full_text if acc.get_latest_turn() else 'N/A'}")

    rolled_back = acc.rollback_turn()
    print(f"  回滚turn: '{rolled_back}'")
    print(f"  回滚后turn数: {acc.get_turn_count()}")

    # ---- 5. 编排器统计 ----
    print("\n【5. 编排器统计】")
    print("-" * 50)

    stats = orchestrator.get_orchestrator_stats()
    print(f"  已提交turns: {stats['turns']}")
    print(f"  延迟统计: {stats['latency']}")
    print(f"  检索统计: {stats['retrieval']}")
    print(f"  写入统计: {stats['writer']}")

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
