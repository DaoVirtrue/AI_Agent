#!/usr/bin/env python3
"""
记忆巩固引擎 (Memory Consolidation Engine)
=============================================
实现基于艾宾浩斯遗忘曲线的记忆管理，模拟人脑的短时记忆(STM)到
长时记忆(LTM)的转换过程。

核心理论：
  - 艾宾浩斯遗忘曲线: R(t) = e^(-t/S)
    其中 R 是记忆保留率，t 是时间，S 是记忆强度(strength)
  - 间隔重复效应: 每次复习增强记忆强度 S *= 1.5
  - 临界复习间隔: t_critical = -S * ln(T)  (T 为保留率阈值)

核心组件：
  - EbbinghausForgettingCurve: 遗忘曲线数学模型
  - MemoryConsolidationEngine: 记忆巩固引擎
  - SleepConsolidationSimulator: 睡眠巩固模拟器
"""

from __future__ import annotations

import math
import time
import random
import json
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque


# ============================================================================
# 记忆等级枚举
# ============================================================================

class MemoryLevel(Enum):
    """记忆存储等级"""
    SENSORY = auto()        # 感知记忆（毫秒级）
    STM = auto()            # 短时记忆 (Short-Term Memory, 秒-分钟级)
    INTERMEDIATE = auto()   # 中间记忆（过渡态）
    LTM = auto()            # 长时记忆 (Long-Term Memory, 小时-年级)
    ARCHIVAL = auto()       # 长期归档记忆

    @property
    def typical_retention(self) -> float:
        """典型保留时间（秒）"""
        return {
            MemoryLevel.SENSORY: 0.5,
            MemoryLevel.STM: 30.0,
            MemoryLevel.INTERMEDIATE: 3600.0,
            MemoryLevel.LTM: 86400.0 * 7,     # 7天
            MemoryLevel.ARCHIVAL: 86400.0 * 365,  # 1年
        }[self]

    @property
    def capacity_hint(self) -> int:
        """容量提示（条数）"""
        return {
            MemoryLevel.SENSORY: 100,
            MemoryLevel.STM: 7,        # 7±2 经典理论
            MemoryLevel.INTERMEDIATE: 50,
            MemoryLevel.LTM: 10000,
            MemoryLevel.ARCHIVAL: 1000000,
        }[self]


# ============================================================================
# 艾宾浩斯遗忘曲线
# ============================================================================

@dataclass
class ForgettingCurvePoint:
    """遗忘曲线上的一个点"""
    time_seconds: float    # 经过的时间
    retention: float       # 保留率 [0, 1]
    memory_strength: float # 当前记忆强度


class EbbinghausForgettingCurve:
    """
    艾宾浩斯遗忘曲线数学模型。

    核心公式（带优化参数的指数衰减）：
        R(t) = e^(-t / S)

    扩展公式（考虑复习次数）：
        R(t, n) = e^(-t / (S * (1 + 0.5 * ln(n + 1))))

    其中：
        R = 记忆保留率 (0-1)
        t = 经过时间（秒）
        S = 当前记忆强度（越大越不容易忘）
        n = 复习次数

    关键时间点（经典艾宾浩斯数据）：
        - 20分钟后: 保留约58%
        - 1小时后: 保留约44%
        - 1天后: 保留约34%
        - 6天后: 保留约25%
        - 31天后: 保留约21%
    """

    # 经典艾宾浩斯数据点 (时间, 保留率)
    EBHAUS_DATA = [
        (0, 1.0),
        (1200, 0.58),        # 20分钟
        (3600, 0.44),        # 1小时
        (86400, 0.34),       # 1天
        (86400 * 2, 0.28),   # 2天
        (86400 * 6, 0.25),   # 6天
        (86400 * 31, 0.21),  # 31天
    ]

    @staticmethod
    def retention(t: float, strength: float = 100.0) -> float:
        """
        计算记忆保留率。

        Args:
            t: 经过的时间（秒）
            strength: 记忆强度（默认100对应约20分钟保留58%）

        Returns:
            保留率 [0, 1]
        """
        if t <= 0:
            return 1.0
        if strength <= 0:
            return 0.0
        return math.exp(-t / strength)

    @staticmethod
    def retention_with_reviews(t: float, strength: float,
                                review_count: int) -> float:
        """
        考虑复习次数的记忆保留率。

        Args:
            t: 经过时间
            strength: 基础记忆强度
            review_count: 复习次数

        Returns:
            调整后的保留率
        """
        if review_count == 0:
            return EbbinghausForgettingCurve.retention(t, strength)

        # 间隔重复效应：每次复习增强有效强度
        boost = 1.0 + 0.5 * math.log(review_count + 1)
        effective_strength = strength * boost

        return EbbinghausForgettingCurve.retention(t, effective_strength)

    @staticmethod
    def critical_review_interval(strength: float,
                                  retention_threshold: float = 0.7) -> float:
        """
        计算临界复习间隔：保留率降至阈值以下前必须复习。

        R(t) = e^(-t/S) = threshold
        => -t/S = ln(threshold)
        => t = -S * ln(threshold)

        Args:
            strength: 记忆强度
            retention_threshold: 保留率阈值

        Returns:
            临界间隔（秒）
        """
        if retention_threshold <= 0 or retention_threshold >= 1:
            return float('inf')
        return -strength * math.log(retention_threshold)

    @staticmethod
    def optimal_review_schedule(initial_strength: float,
                                num_reviews: int = 5,
                                retention_threshold: float = 0.7) -> List[float]:
        """
        生成最优复习时间表（秒）。

        算法：每次复习后记忆强度提升，下次复习间隔相应延长。

        Args:
            initial_strength: 初始记忆强度
            num_reviews: 复习次数
            retention_threshold: 保留率阈值

        Returns:
            复习时间点列表 [t1, t2, t3, ...]（累计秒数）
        """
        schedule = []
        current_strength = initial_strength
        cumulative_time = 0.0

        for i in range(num_reviews):
            interval = EbbinghausForgettingCurve.critical_review_interval(
                current_strength, retention_threshold
            )
            cumulative_time += interval
            schedule.append(cumulative_time)

            # 复习增强记忆强度
            current_strength *= (1.0 + 0.5 * math.log(i + 2))

        return schedule

    @staticmethod
    def strength_from_retention(t: float, observed_retention: float) -> float:
        """
        根据观测保留率反推记忆强度。

        R = e^(-t/S) => ln(R) = -t/S => S = -t / ln(R)

        Args:
            t: 经过时间
            observed_retention: 观测到的保留率 (0-1)

        Returns:
            推算的记忆强度
        """
        if observed_retention <= 0 or observed_retention >= 1:
            return 1.0
        return -t / math.log(observed_retention)

    @staticmethod
    def curve_points(strength: float = 100.0,
                     max_days: float = 30.0,
                     num_points: int = 50) -> List[ForgettingCurvePoint]:
        """生成遗忘曲线数据点集"""
        max_seconds = max_days * 86400.0
        points = []
        for i in range(num_points + 1):
            t = (i / num_points) * max_seconds
            r = EbbinghausForgettingCurve.retention(t, strength)
            points.append(ForgettingCurvePoint(
                time_seconds=t, retention=r, memory_strength=strength,
            ))
        return points


# ============================================================================
# 记忆条目
# ============================================================================

@dataclass
class MemoryEntry:
    """单条记忆"""
    id: str                                    # 唯一标识
    content: str                               # 记忆内容
    level: MemoryLevel = MemoryLevel.STM       # 当前等级
    strength: float = 100.0                    # 记忆强度
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    review_count: int = 0                      # 复习次数
    importance: float = 0.5                    # 重要性 [0-1]
    emotional_valence: float = 0.0             # 情感效价 [-1, 1]
    associations: List[str] = field(default_factory=list)  # 关联记忆ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def current_retention(self) -> float:
        """计算当前保留率"""
        elapsed = time.time() - self.last_accessed_at
        return EbbinghausForgettingCurve.retention_with_reviews(
            elapsed, self.strength, self.review_count
        )

    def is_forgotten(self, threshold: float = 0.3) -> bool:
        """判断该记忆是否已被遗忘"""
        return self.current_retention() < threshold

    def review(self):
        """复习该记忆，增强强度"""
        self.review_count += 1
        self.last_accessed_at = time.time()
        boost = 1.0 + 0.5 * math.log(self.review_count + 1)
        self.strength *= boost


# ============================================================================
# 记忆巩固引擎
# ============================================================================

class MemoryConsolidationEngine:
    """
    记忆巩固引擎。

    模拟人脑的记忆巩固过程：
    1. STM 缓冲区管理（容量限制，约7±2条）
    2. 分级转移 (Graded Transfer): STM → Intermediate → LTM → Archival
    3. 基于重要性、情感效价和复习次数的转移决策
    4. 遗忘模拟（基于艾宾浩斯曲线）
    """

    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}        # 记忆库
        self.stm_buffer: deque[str] = deque(maxlen=7)     # STM缓冲区
        self._transfer_history: List[Dict[str, Any]] = [] # 转移历史
        self._id_counter = 0

    # ---- 记忆创建 ----

    def create_memory(self, content: str,
                      importance: float = 0.5,
                      emotional_valence: float = 0.0,
                      level: MemoryLevel = MemoryLevel.STM,
                      metadata: Optional[Dict] = None) -> MemoryEntry:
        """创建一条新记忆"""
        self._id_counter += 1
        mem_id = f"mem-{self._id_counter:04d}"

        entry = MemoryEntry(
            id=mem_id,
            content=content,
            level=level,
            importance=importance,
            emotional_valence=emotional_valence,
            metadata=metadata or {},
        )
        self.memories[mem_id] = entry

        if level == MemoryLevel.STM:
            if len(self.stm_buffer) >= self.stm_buffer.maxlen:
                # STM满了，淘汰最旧的
                old_id = self.stm_buffer.popleft()
                # 如果仍为STM且不重要，直接丢弃
                old_mem = self.memories.get(old_id)
                if old_mem and old_mem.level == MemoryLevel.STM and old_mem.importance < 0.3:
                    del self.memories[old_id]
            self.stm_buffer.append(mem_id)

        return entry

    # ---- STM → LTM 巩固转移 ----

    def consolidate_stm_to_ltm(self,
                                min_stm_age_seconds: float = 30.0,
                                importance_threshold: float = 0.4) -> List[MemoryEntry]:
        """
        将符合条件的STM记忆转移到LTM。

        转移条件：
        1. 在STM中存在时间 > min_stm_age_seconds
        2. 重要性 > importance_threshold
        3. 最近被访问过（保留率 > 50%）

        Returns:
            成功转移的记忆列表
        """
        promoted = []
        now = time.time()

        for mem_id in list(self.stm_buffer):
            mem = self.memories.get(mem_id)
            if mem is None:
                continue
            if mem.level != MemoryLevel.STM:
                continue

            age = now - mem.created_at
            retention = mem.current_retention()

            if age >= min_stm_age_seconds and \
               mem.importance >= importance_threshold and \
               retention > 0.5:
                # 升级到 INTERMEDIATE
                mem.level = MemoryLevel.INTERMEDIATE
                mem.strength *= 1.5  # 巩固增强
                self.stm_buffer.remove(mem_id)
                promoted.append(mem)

                self._transfer_history.append({
                    "memory_id": mem_id,
                    "from_level": "STM",
                    "to_level": "INTERMEDIATE",
                    "timestamp": now,
                })

        return promoted

    def graded_transfer(self) -> Dict[str, List[MemoryEntry]]:
        """
        分级转移：执行完整的STM→Intermediate→LTM→Archival转移链。

        Returns:
            {"promoted": [...], "demoted": [...]}
        """
        results: Dict[str, List[MemoryEntry]] = {
            "promoted": [],
            "demoted": [],
        }
        now = time.time()

        for mem_id, mem in self.memories.items():
            if mem.level == MemoryLevel.SENSORY:
                # 感知记忆：极短生命，快速升级或丢弃
                if mem.importance > 0.6:
                    mem.level = MemoryLevel.STM
                    mem.strength *= 1.2
                    results["promoted"].append(mem)
                else:
                    results["demoted"].append(mem)

            elif mem.level == MemoryLevel.STM:
                # STM → Intermediate
                if mem.review_count >= 3 or mem.importance > 0.7:
                    mem.level = MemoryLevel.INTERMEDIATE
                    mem.strength *= 1.5
                    results["promoted"].append(mem)
                    if mem_id in self.stm_buffer:
                        self.stm_buffer.remove(mem_id)
                elif mem.is_forgotten(0.2):
                    results["demoted"].append(mem)

            elif mem.level == MemoryLevel.INTERMEDIATE:
                # Intermediate → LTM
                if mem.review_count >= 5 and mem.importance > 0.5:
                    mem.level = MemoryLevel.LTM
                    mem.strength *= 2.0
                    results["promoted"].append(mem)
                elif mem.is_forgotten(0.3):
                    results["demoted"].append(mem)

            elif mem.level == MemoryLevel.LTM:
                # LTM → Archival
                if mem.review_count >= 10 and mem.importance > 0.8:
                    mem.level = MemoryLevel.ARCHIVAL
                    mem.strength *= 1.5
                    results["promoted"].append(mem)
                elif mem.is_forgotten(0.2):
                    results["demoted"].append(mem)

        # 清理降级的记忆
        for mem in results["demoted"]:
            if mem.id in self.memories:
                del self.memories[mem.id]

        return results

    def batch_consolidation(self) -> Dict[str, Any]:
        """
        批量巩固：运行一轮完整的记忆维护。

        包括：
        1. 遗忘检测与清理
        2. STM→LTM 转移
        3. 分级转移

        Returns:
            巩固统计信息
        """
        stats = {
            "before_total": len(self.memories),
            "forgotten": 0,
            "promoted_stm_to_intermediate": 0,
            "promoted_to_ltm": 0,
            "promoted_to_archival": 0,
        }

        # 遗忘检测
        forgotten_ids = []
        for mem_id, mem in self.memories.items():
            if mem.is_forgotten(0.15):
                forgotten_ids.append(mem_id)

        for fid in forgotten_ids:
            del self.memories[fid]
            if fid in self.stm_buffer:
                self.stm_buffer.remove(fid)

        stats["forgotten"] = len(forgotten_ids)

        # STM → Intermediate
        promoted_stm = self.consolidate_stm_to_ltm()
        stats["promoted_stm_to_intermediate"] = len(promoted_stm)

        # 分级转移
        transfer_result = self.graded_transfer()
        stats["promoted_to_ltm"] = len(
            [m for m in transfer_result["promoted"]
             if m.level == MemoryLevel.LTM]
        )
        stats["promoted_to_archival"] = len(
            [m for m in transfer_result["promoted"]
             if m.level == MemoryLevel.ARCHIVAL]
        )
        stats["after_total"] = len(self.memories)

        return stats

    def apply_forgetting(self, elapsed_seconds: float) -> Dict[str, Any]:
        """
        对全体记忆应用时间衰减（模拟经过一段时间后）。

        Args:
            elapsed_seconds: 模拟经过的秒数

        Returns:
            衰减统计
        """
        faded = 0
        forgotten = 0

        for mem_id, mem in list(self.memories.items()):
            # 计算经过时间后的保留率
            current_retention = EbbinghausForgettingCurve.retention_with_reviews(
                elapsed_seconds, mem.strength, mem.review_count
            )

            if current_retention < 0.1:
                del self.memories[mem_id]
                if mem_id in self.stm_buffer:
                    self.stm_buffer.remove(mem_id)
                forgotten += 1
            elif current_retention < 1.0:
                faded += 1

        return {
            "elapsed_seconds": elapsed_seconds,
            "memories_before": len(self.memories) + forgotten,
            "memories_after": len(self.memories),
            "faded": faded,
            "forgotten": forgotten,
        }

    def get_memories_by_level(self, level: MemoryLevel) -> List[MemoryEntry]:
        """按等级获取记忆列表"""
        return [m for m in self.memories.values() if m.level == level]

    def get_retention_stats(self) -> Dict[str, Any]:
        """获取记忆保留率统计"""
        levels: Dict[str, List[float]] = defaultdict(list)
        for mem in self.memories.values():
            levels[mem.level.name].append(mem.current_retention())

        return {
            level_name: {
                "count": len(retentions),
                "avg_retention": round(sum(retentions) / len(retentions), 4)
                if retentions else 0,
                "min_retention": round(min(retentions), 4) if retentions else 0,
            }
            for level_name, retentions in levels.items()
        }


# ============================================================================
# 睡眠巩固模拟器
# ============================================================================

class SleepConsolidationSimulator:
    """
    睡眠巩固模拟器。

    模拟睡眠过程中的记忆巩固：
    - NREM（非快速眼动）阶段：海马体→皮层记忆转移
    - REM（快速眼动）阶段：情感记忆强化，关联记忆整合
    """

    # 典型睡眠周期（分钟）
    SLEEP_CYCLES = [
        {"stage": "NREM-1", "duration_min": 5},    # 浅睡
        {"stage": "NREM-2", "duration_min": 20},   # 中度睡眠
        {"stage": "NREM-3", "duration_min": 30},   # 深度睡眠（慢波）
        {"stage": "REM", "duration_min": 10},       # 快速眼动
    ]

    def __init__(self, engine: MemoryConsolidationEngine):
        self.engine = engine
        self._sleep_logs: List[Dict[str, Any]] = []

    def simulate_sleep(self, num_cycles: int = 5) -> Dict[str, Any]:
        """
        模拟一个完整睡眠周期（默认5个90分钟周期）。

        Returns:
            睡眠巩固统计
        """
        stats = {
            "total_memories_before": len(self.engine.memories),
            "stm_cleared": 0,
            "consolidated_to_ltm": 0,
            "emotionally_boosted": 0,
            "associations_formed": 0,
            "garbage_collected": 0,
        }

        for cycle_num in range(num_cycles):
            for phase in self.SLEEP_CYCLES:
                stage = phase["stage"]
                duration = phase["duration_min"] * 60  # 转秒

                if stage == "NREM-3":
                    # 深度睡眠：STM清空，记忆转移到皮层
                    stm_memories = self.engine.get_memories_by_level(MemoryLevel.STM)
                    for mem in stm_memories:
                        if mem.importance > 0.5:
                            mem.level = MemoryLevel.INTERMEDIATE
                            mem.strength *= 1.8
                            stats["consolidated_to_ltm"] += 1
                        else:
                            # 不重要记忆在深度睡眠中被清除
                            del self.engine.memories[mem.id]
                            stats["garbage_collected"] += 1

                elif stage == "REM":
                    # REM睡眠：情感记忆强化
                    for mem in list(self.engine.memories.values()):
                        if abs(mem.emotional_valence) > 0.5:
                            mem.strength *= (1.0 + abs(mem.emotional_valence) * 0.5)
                            mem.importance += 0.05
                            stats["emotionally_boosted"] += 1

                    # 关联记忆整合
                    self._consolidate_associations(stats)

                # 应用时间衰减（睡眠中的遗忘主要发生在浅睡阶段）
                if stage in ("NREM-1", "NREM-2"):
                    self.engine.apply_forgetting(duration * 0.1)

        stats["total_memories_after"] = len(self.engine.memories)
        self._sleep_logs.append(stats)
        return stats

    def _consolidate_associations(self, stats: Dict[str, Any]):
        """在REM期间巩固关联记忆"""
        memories = list(self.engine.memories.values())
        for i, mem_a in enumerate(memories):
            for mem_b in memories[i + 1:]:
                # 简单的共现关联
                words_a = set(mem_a.content.lower().split())
                words_b = set(mem_b.content.lower().split())
                overlap = words_a & words_b
                if len(overlap) > 2 and overlap != {'is', 'the', 'a', 'of', 'to'}:
                    if mem_b.id not in mem_a.associations:
                        mem_a.associations.append(mem_b.id)
                        stats["associations_formed"] += 1

    def get_sleep_logs(self) -> List[Dict[str, Any]]:
        """获取睡眠日志"""
        return list(self._sleep_logs)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示"""
    print("=" * 70)
    print("  记忆巩固引擎 (Memory Consolidation Engine) 演示")
    print("=" * 70)

    # ---- 1. 艾宾浩斯遗忘曲线 ----
    print("\n【1. 艾宾浩斯遗忘曲线】")
    print("-" * 50)
    eb = EbbinghausForgettingCurve

    test_times = [0, 1200, 3600, 86400, 86400 * 2, 86400 * 7, 86400 * 30]
    for t in test_times:
        r = eb.retention(t, strength=100.0)
        label = {0: "初始", 1200: "20分钟", 3600: "1小时",
                 86400: "1天", 86400*2: "2天", 86400*7: "7天", 86400*30: "30天"}.get(t, "")
        print(f"  {label:>6s} ({t:>8.0f}s): 保留率 {r:.2%}")

    # 最优复习时间表
    print("\n  最优复习时间表 (初始强度=100, 保留率阈值=0.7):")
    schedule = eb.optimal_review_schedule(100.0, num_reviews=5)
    for i, s in enumerate(schedule):
        days = s / 86400.0
        print(f"    第{i+1}次复习: {s:.0f}秒后 ({days:.1f}天)")

    # ---- 2. 记忆创建与巩固 ----
    print("\n【2. 记忆创建与巩固】")
    print("-" * 50)

    engine = MemoryConsolidationEngine()

    # 创建一系列记忆
    test_memories = [
        ("今天的会议时间改到下午3点", 0.8, 0.0),
        ("午餐吃了三明治", 0.2, 0.0),
        ("项目截止日期是下周五", 0.9, 0.1),
        ("收到一封重要邮件", 0.7, 0.0),
        ("路过一家新开的咖啡店", 0.15, 0.3),
        ("老板表扬了团队", 0.6, 0.7),
        ("需要买牛奶", 0.3, 0.0),
        ("昨天看了场好电影", 0.4, 0.6),
        ("客户要修改需求文档", 0.85, -0.2),
        ("电梯在维修", 0.2, 0.0),
    ]

    for content, importance, emotional in test_memories:
        mem = engine.create_memory(content, importance, emotional)
        print(f"  创建记忆 [{mem.id}]: {content[:20]}... (重要性={importance})")

    # STM缓冲区状态
    print(f"\n  STM缓冲区: {list(engine.stm_buffer)}")
    print(f"  当前记忆总数: {len(engine.memories)}")

    # 巩固
    promoted = engine.consolidate_stm_to_ltm(
        min_stm_age_seconds=0.0,  # 立即测试
        importance_threshold=0.5,
    )
    print(f"\n  STM→Intermediate 转移: {len(promoted)} 条")
    for m in promoted:
        print(f"    [{m.id}] {m.content[:30]} → {m.level.name}")

    # 分级转移
    print("\n  执行分级转移...")
    result = engine.graded_transfer()
    print(f"    升级: {len(result['promoted'])} 条")
    print(f"    降级/删除: {len(result['demoted'])} 条")

    # ---- 3. 30天遗忘模拟 ----
    print("\n【3. 30天遗忘模拟】")
    print("-" * 50)

    # 重置引擎
    engine2 = MemoryConsolidationEngine()
    for content, importance, emotional in test_memories:
        engine2.create_memory(content, importance, emotional)

    print(f"  初始记忆数: {len(engine2.memories)}")

    # 模拟30天，每天调用一次
    days = 30
    for day in range(1, days + 1):
        result_forget = engine2.apply_forgetting(86400.0)
        if day % 10 == 0:
            print(f"  第{day:>3}天: 剩余 {result_forget['memories_after']} 条记忆 "
                  f"(遗忘 {result_forget['forgotten']} 条当天)")

        # 每7天批量巩固
        if day % 7 == 0:
            engine2.batch_consolidation()

    # 最终保留率统计
    print(f"\n  30天后:")
    print(f"    剩余记忆: {len(engine2.memories)} 条")
    for level in MemoryLevel:
        count = len(engine2.get_memories_by_level(level))
        if count > 0:
            print(f"    {level.name}: {count} 条")

    # ---- 4. 睡眠巩固模拟 ----
    print("\n【4. 睡眠巩固模拟】")
    print("-" * 50)

    engine3 = MemoryConsolidationEngine()
    for content, importance, emotional in test_memories:
        engine3.create_memory(content, importance, emotional)

    sleep_sim = SleepConsolidationSimulator(engine3)
    sleep_stats = sleep_sim.simulate_sleep(num_cycles=5)

    print(f"  睡眠前记忆数: {sleep_stats['total_memories_before']}")
    print(f"  巩固到LTM: {sleep_stats['consolidated_to_ltm']}")
    print(f"  情感增强: {sleep_stats['emotionally_boosted']}")
    print(f"  关联形成: {sleep_stats['associations_formed']}")
    print(f"  垃圾回收: {sleep_stats['garbage_collected']}")
    print(f"  睡眠后记忆数: {sleep_stats['total_memories_after']}")

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
