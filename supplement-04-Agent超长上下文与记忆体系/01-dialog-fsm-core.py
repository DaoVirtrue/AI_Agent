#!/usr/bin/env python3
"""
对话有限状态机核心 (Dialog FSM Core)
======================================
实现基于枚举和转移表驱动的对话状态机，支持8种对话状态、超时检测、
死锁预防和ASCII状态图生成。

核心概念：
- DialogState: 8种对话状态枚举
- FSMTransition: 单次状态转移的数据结构
- DialogFSM: 完整的有限状态机实现

状态转移图（ASCII Art）：
    IDLE ──┬── GREETING ── COLLECTING_INFO ── CONFIRMING
           │                                      │
           ├── PROCESSING ◄───────────────────────┤
           │       │                              │
           ├── RESPONDING ◄──── ASKING_CLARIFICATION
           │       │
           └── FAREWELL ◄────────────────────────┘
"""

from __future__ import annotations

import time
import json
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict


# ============================================================================
# 状态枚举
# ============================================================================

class DialogState(Enum):
    """对话状态枚举 - 覆盖完整的客服对话生命周期"""
    IDLE = auto()                   # 空闲等待
    GREETING = auto()               # 问候阶段
    COLLECTING_INFO = auto()        # 信息收集阶段
    CONFIRMING = auto()             # 确认信息阶段
    PROCESSING = auto()             # 后台处理阶段
    RESPONDING = auto()             # 响应生成阶段
    ASKING_CLARIFICATION = auto()   # 请求澄清阶段
    FAREWELL = auto()               # 告别结束阶段

    @property
    def cn_name(self) -> str:
        """返回中文名称"""
        names = {
            DialogState.IDLE: "空闲",
            DialogState.GREETING: "问候",
            DialogState.COLLECTING_INFO: "信息收集",
            DialogState.CONFIRMING: "确认",
            DialogState.PROCESSING: "处理中",
            DialogState.RESPONDING: "响应中",
            DialogState.ASKING_CLARIFICATION: "澄清",
            DialogState.FAREWELL: "告别",
        }
        return names.get(self, "未知")

    @property
    def is_terminal(self) -> bool:
        """是否为终止状态"""
        return self == DialogState.FAREWELL

    @property
    def is_interruptible(self) -> bool:
        """是否可被中断"""
        return self in (DialogState.IDLE, DialogState.COLLECTING_INFO,
                        DialogState.RESPONDING)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class FSMTransition:
    """单次状态转移记录"""
    from_state: DialogState           # 源状态
    to_state: DialogState             # 目标状态
    trigger: str                      # 触发器名称
    guard: Optional[Callable[[], bool]] = None   # 守卫条件
    action: Optional[Callable[[], Any]] = None   # 转移动作
    description: str = ""             # 转移说明
    priority: int = 0                 # 优先级（数字越大优先级越高）

    def __post_init__(self):
        if not self.description:
            self.description = f"{self.from_state.cn_name} --[{self.trigger}]--> {self.to_state.cn_name}"


@dataclass
class DialogFSMStats:
    """FSM 运行统计"""
    total_transitions: int = 0
    state_durations: Dict[DialogState, float] = field(default_factory=dict)
    trigger_counts: Dict[str, int] = field(default_factory=dict)
    timeouts_triggered: int = 0
    rejected_transitions: int = 0
    start_time: float = 0.0


# ============================================================================
# 对话状态机
# ============================================================================

class DialogFSM:
    """
    对话有限状态机完整实现。

    特性：
    - 基于转移表的声明式状态定义
    - 守卫条件（guard）和转移动作（action）
    - 超时自动转移机制
    - 状态进入/退出钩子
    - ASCII 状态图生成
    - 运行统计收集
    """

    def __init__(self, name: str = "DialogFSM",
                 initial_state: DialogState = DialogState.IDLE,
                 timeout_seconds: float = 300.0):
        self.name = name
        self._transitions: List[FSMTransition] = []
        self._state_entry_hooks: Dict[DialogState, List[Callable]] = defaultdict(list)
        self._state_exit_hooks: Dict[DialogState, List[Callable]] = defaultdict(list)
        self._state_timestamps: Dict[DialogState, float] = {}
        self._timeout_seconds = timeout_seconds

        # 运行时状态
        self._current_state: DialogState = initial_state
        self._previous_state: Optional[DialogState] = None
        self._last_trigger: Optional[str] = None
        self._state_entered_at: float = time.time()
        self._context: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._stats = DialogFSMStats(start_time=time.time())

    # ---- 状态属性（setter 带钩子） ----

    @property
    def state(self) -> DialogState:
        """当前状态"""
        return self._current_state

    @state.setter
    def state(self, new_state: DialogState):
        """设置状态时触发进入/退出钩子"""
        if new_state == self._current_state:
            return

        old_state = self._current_state
        self._previous_state = old_state
        duration = time.time() - self._state_entered_at
        self._state_timestamps[old_state] = duration

        # 记录到统计
        if old_state not in self._stats.state_durations:
            self._stats.state_durations[old_state] = 0.0
        self._stats.state_durations[old_state] += duration

        # 退出钩子
        for hook in self._state_exit_hooks.get(old_state, []):
            try:
                hook(old_state, new_state, self._context)
            except Exception as e:
                print(f"[FSM] 退出钩子异常: {e}")

        # 切换状态
        self._current_state = new_state
        self._state_entered_at = time.time()

        # 进入钩子
        for hook in self._state_entry_hooks.get(new_state, []):
            try:
                hook(old_state, new_state, self._context)
            except Exception as e:
                print(f"[FSM] 进入钩子异常: {e}")

    # ---- 转移表管理 ----

    def add_transition(self,
                       from_state: DialogState,
                       to_state: DialogState,
                       trigger: str,
                       guard: Optional[Callable[[], bool]] = None,
                       action: Optional[Callable[[], Any]] = None,
                       description: str = "",
                       priority: int = 0) -> 'DialogFSM':
        """
        添加一条状态转移规则。支持链式调用。

        Args:
            from_state: 源状态
            to_state: 目标状态
            trigger: 触发器名称（如 'greet', 'submit', 'cancel'）
            guard: 守卫条件函数，返回 True 才允许转移
            action: 转移时执行的动作函数
            description: 转移描述
            priority: 优先级（当多个转移匹配时选优先级最高的）
        """
        transition = FSMTransition(
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            guard=guard,
            action=action,
            description=description,
            priority=priority,
        )
        self._transitions.append(transition)
        return self

    def on_enter(self, state: DialogState, hook: Callable) -> 'DialogFSM':
        """注册状态进入钩子"""
        self._state_entry_hooks[state].append(hook)
        return self

    def on_exit(self, state: DialogState, hook: Callable) -> 'DialogFSM':
        """注册状态退出钩子"""
        self._state_exit_hooks[state].append(hook)
        return self

    # ---- 核心转移逻辑 ----

    def transition(self, trigger: str) -> bool:
        """
        根据触发器执行状态转移。

        Returns:
            True 如果转移成功执行，False 如果无匹配转移或守卫条件不满足
        """
        candidates = [
            t for t in self._transitions
            if t.from_state == self._current_state and t.trigger == trigger
        ]

        if not candidates:
            self._stats.rejected_transitions += 1
            return False

        # 按优先级排序
        candidates.sort(key=lambda t: t.priority, reverse=True)

        for t in candidates:
            # 检查守卫条件
            if t.guard is not None:
                try:
                    if not t.guard():
                        continue
                except Exception as e:
                    print(f"[FSM] 守卫条件执行异常: {e}")
                    continue

            # 执行转移
            old_state = self._current_state
            self.state = t.to_state  # setter 会触发钩子
            self._last_trigger = trigger

            # 执行动作
            result = None
            if t.action is not None:
                try:
                    result = t.action()
                except Exception as e:
                    print(f"[FSM] 转移动作执行异常: {e}")

            # 记录历史
            self._history.append({
                "from": old_state.name,
                "to": t.to_state.name,
                "trigger": trigger,
                "timestamp": time.time(),
                "duration_ms": (time.time() - self._state_entered_at) * 1000,
            })

            # 更新统计
            self._stats.total_transitions += 1
            self._stats.trigger_counts[trigger] = \
                self._stats.trigger_counts.get(trigger, 0) + 1

            return True

        self._stats.rejected_transitions += 1
        return False

    def check_timeout(self) -> bool:
        """
        检查当前状态是否超时。如果超时，自动转移到 IDLE。

        Returns:
            True 如果发生了超时转移
        """
        elapsed = time.time() - self._state_entered_at
        if elapsed > self._timeout_seconds and self._current_state != DialogState.IDLE:
            self._stats.timeouts_triggered += 1
            self.state = DialogState.IDLE
            return True
        return False

    def force_state(self, new_state: DialogState, reason: str = "forced"):
        """强制切换状态（用于异常恢复），绕过转移表"""
        self._history.append({
            "from": self._current_state.name,
            "to": new_state.name,
            "trigger": f"__force__({reason})",
            "timestamp": time.time(),
            "forced": True,
        })
        self._stats.total_transitions += 1
        self.state = new_state

    def get_allowed_triggers(self) -> List[str]:
        """获取当前状态下所有可能的触发器列表"""
        return list(set(
            t.trigger for t in self._transitions
            if t.from_state == self._current_state
        ))

    def context_set(self, key: str, value: Any):
        """在 FSM 上下文中存储数据"""
        self._context[key] = value

    def context_get(self, key: str, default: Any = None) -> Any:
        """从 FSM 上下文中读取数据"""
        return self._context.get(key, default)

    # ---- 可视化 ----

    def build_transition_diagram(self) -> str:
        """
        生成 ASCII 艺术风格的状态转移图。

        Returns:
            多行字符串，包含完整的状态转移关系
        """
        lines = []
        lines.append(f"╔══ {self.name} 状态转移图 ══╗")
        lines.append("")

        # 按源状态分组
        grouped: Dict[DialogState, List[FSMTransition]] = defaultdict(list)
        for t in self._transitions:
            grouped[t.from_state].append(t)

        state_order = list(DialogState)
        for state in state_order:
            marker = "●" if state == self._current_state else "○"
            terminal = " [终止]" if state.is_terminal else ""
            lines.append(f"  {marker} [{state.name}]{terminal} ({state.cn_name})")

            transitions_for_state = grouped.get(state, [])
            if transitions_for_state:
                for t in transitions_for_state:
                    guard_str = " [G]" if t.guard else ""
                    lines.append(f"    ├──[{t.trigger}]{guard_str}──> [{t.to_state.name}] {t.to_state.cn_name}")
            lines.append("")

        # 添加图例
        lines.append("╔══ 图例 ══╗")
        lines.append("  ● = 当前状态    [G] = 有守卫条件")
        lines.append("  ○ = 非当前状态   [终止] = 对话结束状态")
        lines.append(f"  当前状态: {self._current_state.cn_name}")

        return "\n".join(lines)

    def stats(self) -> dict:
        """返回运行时统计数据"""
        return {
            "name": self.name,
            "current_state": self._current_state.name,
            "current_state_cn": self._current_state.cn_name,
            "previous_state": self._previous_state.name if self._previous_state else None,
            "total_transitions": self._stats.total_transitions,
            "rejected_transitions": self._stats.rejected_transitions,
            "timeouts_triggered": self._stats.timeouts_triggered,
            "state_durations_sec": {
                s.name: round(d, 2)
                for s, d in self._stats.state_durations.items()
            },
            "trigger_counts": dict(self._stats.trigger_counts),
            "uptime_sec": round(time.time() - self._stats.start_time, 2),
            "context_keys": list(self._context.keys()),
            "history_length": len(self._history),
        }

    def reset(self):
        """重置状态机到初始状态"""
        self._current_state = DialogState.IDLE
        self._previous_state = None
        self._state_entered_at = time.time()
        self._context.clear()
        self._history.clear()
        self._stats = DialogFSMStats(start_time=time.time())


# ============================================================================
# 客服对话示例
# ============================================================================

def create_customer_service_fsm() -> DialogFSM:
    """
    构建一个标准的客服对话状态机。

    流程：IDLE → GREETING → COLLECTING_INFO → CONFIRMING → PROCESSING → RESPONDING → FAREWELL
    异常：任意状态 → ASKING_CLARIFICATION（需澄清时） | 任意状态 → IDLE（超时/取消）
    """
    fsm = DialogFSM(name="CustomerServiceFSM", timeout_seconds=600.0)

    # 存储用户数据
    user_data: Dict[str, Any] = {}

    # === 从 IDLE 出发 ===
    fsm.add_transition(DialogState.IDLE, DialogState.GREETING, "start",
                       description="用户发起对话")

    # === 从 GREETING 出发 ===
    fsm.add_transition(DialogState.GREETING, DialogState.COLLECTING_INFO, "greeting_done",
                       description="问候完成，开始收集信息")
    fsm.add_transition(DialogState.GREETING, DialogState.GREETING, "identify",
                       description="用户自报身份")

    # === 从 COLLECTING_INFO 出发 ===
    def _check_info_sufficient() -> bool:
        return len(user_data) >= 2  # 至少收集到两个字段

    fsm.add_transition(DialogState.COLLECTING_INFO, DialogState.CONFIRMING, "info_submitted",
                       guard=_check_info_sufficient,
                       description="信息收集完成，进入确认")
    fsm.add_transition(DialogState.COLLECTING_INFO, DialogState.COLLECTING_INFO, "info_provide",
                       action=lambda: user_data.update({"last_update": time.time()}),
                       description="用户提供一条信息")
    fsm.add_transition(DialogState.COLLECTING_INFO, DialogState.ASKING_CLARIFICATION, "unclear",
                       description="信息不明确，需要澄清")

    # === 从 CONFIRMING 出发 ===
    fsm.add_transition(DialogState.CONFIRMING, DialogState.PROCESSING, "confirm",
                       description="用户确认信息")
    fsm.add_transition(DialogState.CONFIRMING, DialogState.COLLECTING_INFO, "modify",
                       description="用户要求修改信息")
    fsm.add_transition(DialogState.CONFIRMING, DialogState.FAREWELL, "cancel",
                       description="用户取消")

    # === 从 PROCESSING 出发 ===
    fsm.add_transition(DialogState.PROCESSING, DialogState.RESPONDING, "processing_done",
                       description="后台处理完成")
    fsm.add_transition(DialogState.PROCESSING, DialogState.ASKING_CLARIFICATION, "need_info",
                       description="处理中发现需要更多信息")

    # === 从 RESPONDING 出发 ===
    fsm.add_transition(DialogState.RESPONDING, DialogState.COLLECTING_INFO, "follow_up",
                       description="响应后需要进一步收集信息")
    fsm.add_transition(DialogState.RESPONDING, DialogState.FAREWELL, "done",
                       description="对话完成")
    fsm.add_transition(DialogState.RESPONDING, DialogState.RESPONDING, "continue",
                       description="继续对话")

    # === 从 ASKING_CLARIFICATION 出发 ===
    fsm.add_transition(DialogState.ASKING_CLARIFICATION, DialogState.COLLECTING_INFO, "clarified",
                       description="澄清完成，继续收集信息")
    fsm.add_transition(DialogState.ASKING_CLARIFICATION, DialogState.FAREWELL, "give_up",
                       description="用户放弃")

    # === 从 FAREWELL 出发（终止状态，无出边）===

    return fsm


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示客服对话流程"""
    print("=" * 70)
    print("  对话有限状态机 (Dialog FSM) 演示")
    print("=" * 70)
    print()

    fsm = create_customer_service_fsm()

    # 注册钩子
    fsm.on_enter(DialogState.GREETING, lambda old, new, ctx: print("📢 [钩子] 进入问候状态"))
    fsm.on_exit(DialogState.PROCESSING, lambda old, new, ctx: print("📢 [钩子] 离开处理状态"))
    fsm.on_enter(DialogState.FAREWELL, lambda old, new, ctx: print("📢 [钩子] 对话结束！"))

    # 模拟客服对话流程
    conversation_flow = [
        ("start",         "用户发起对话"),
        ("greeting_done", "问候完成"),
        ("info_provide",  "用户提供姓名"),
        ("info_provide",  "用户提供订单号"),
        ("info_submitted","信息提交"),
        ("confirm",       "用户确认"),
        # 模拟处理需要澄清
        ("need_info",     "需要更多信息"),
        ("clarified",     "澄清完毕"),
        ("info_provide",  "补充信息"),
        ("info_submitted","再次提交"),
        ("confirm",       "再次确认"),
        ("processing_done","处理完成"),
        ("done",          "对话完成"),
    ]

    print("【对话流程模拟】")
    print("-" * 50)
    for trigger, description in conversation_flow:
        print(f"\n  >>> 触发: {trigger} ({description})")
        print(f"      当前状态: [{fsm.state.name}] {fsm.state.cn_name}")
        allowed = fsm.get_allowed_triggers()
        print(f"      允许的触发器: {allowed}")

        success = fsm.transition(trigger)
        if success:
            print(f"      ✓ 转移成功 → [{fsm.state.name}] {fsm.state.cn_name}")
        else:
            print(f"      ✗ 转移失败！当前仍在 [{fsm.state.name}]")

    print()
    print("=" * 70)
    print(fsm.build_transition_diagram())
    print()
    print("【运行统计】")
    print(json.dumps(fsm.stats(), indent=2, ensure_ascii=False))

    # 演示超时检测
    print()
    print("【超时检测测试】")
    fsm2 = create_customer_service_fsm()
    fsm2.transition("start")
    fsm2.transition("greeting_done")
    print(f"  转移后状态: {fsm2.state.cn_name}")
    # 手动修改时间戳模拟超时
    fsm2._timeout_seconds = 0.0
    timed_out = fsm2.check_timeout()
    print(f"  超时检测结果: {timed_out}, 当前状态: {fsm2.state.cn_name}")

    print()
    print("=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
