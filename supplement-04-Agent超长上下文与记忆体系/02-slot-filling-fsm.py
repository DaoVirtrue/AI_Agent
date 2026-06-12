#!/usr/bin/env python3
"""
基于FSM的槽位填充系统 (Slot-Filling FSM)
===========================================
继承DialogFSM，实现面向任务的对话槽位填充。支持必填/可选槽位、
自定义验证器、动态提示词生成和缺失槽位自动检测。

核心组件：
- Slot: 命名元组，定义单个槽位的完整元数据
- SlotFillingFSM: 继承DialogFSM，增加槽位管理能力
- 6槽位餐厅预订演示

槽位列表（餐厅预订）:
  1. party_size  - 用餐人数（必填，2-20人）
  2. date        - 预订日期（必填，YYYY-MM-DD）
  3. time        - 预订时间（必填，HH:MM）
  4. dietary     - 饮食要求（可选）
  5. phone       - 联系电话（必填，手机号格式）
  6. name        - 预订人姓名（必填）
"""

from __future__ import annotations

import re
import time
import json
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# 复用 DialogFSM 核心
from sys import path as sys_path
import os
sys_path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接内联导入（避免跨文件依赖问题，教学中可复制代码）
try:
    from dialog_fsm_core import DialogFSM, DialogState
except ImportError:
    # 嵌入式定义，保证文件可独立运行
    from enum import Enum, auto

    class DialogState(Enum):
        IDLE = auto()
        GREETING = auto()
        COLLECTING_INFO = auto()
        CONFIRMING = auto()
        PROCESSING = auto()
        RESPONDING = auto()
        ASKING_CLARIFICATION = auto()
        FAREWELL = auto()

        @property
        def cn_name(self) -> str:
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

    # 内联简化的 DialogFSM（仅包含 SlotFillingFSM 需要的接口）
    import time as _time
    from collections import defaultdict as _dd

    class DialogFSM:
        def __init__(self, name="FSM", initial_state=DialogState.IDLE, timeout_seconds=300.0):
            self.name = name
            self._current_state = initial_state
            self._previous_state = None
            self._transitions = []
            self._state_entered_at = _time.time()
            self._context: Dict[str, Any] = {}
            self._history: List[Dict[str, Any]] = []
            self._state_entry_hooks: Dict[DialogState, List[Callable]] = _dd(list)
            self._state_exit_hooks: Dict[DialogState, List[Callable]] = _dd(list)
            self._timeout_seconds = timeout_seconds

        @property
        def state(self):
            return self._current_state

        @state.setter
        def state(self, new_state):
            if new_state == self._current_state:
                return
            old = self._current_state
            for hook in self._state_exit_hooks.get(old, []):
                try: hook(old, new_state, self._context)
                except: pass
            self._previous_state = old
            self._current_state = new_state
            self._state_entered_at = _time.time()
            for hook in self._state_entry_hooks.get(new_state, []):
                try: hook(old, new_state, self._context)
                except: pass

        def add_transition(self, from_s, to_s, trigger, guard=None, action=None, description="", priority=0):
            self._transitions.append({
                'from': from_s, 'to': to_s, 'trigger': trigger,
                'guard': guard, 'action': action, 'priority': priority,
            })
            return self

        def transition(self, trigger):
            candidates = [t for t in self._transitions
                          if t['from'] == self._current_state and t['trigger'] == trigger]
            candidates.sort(key=lambda t: t['priority'], reverse=True)
            for t in candidates:
                if t['guard'] and not t['guard']():
                    continue
                self.state = t['to']
                if t['action']:
                    try: t['action']()
                    except: pass
                self._history.append({
                    'from': self._previous_state.name if self._previous_state else '',
                    'to': self.state.name, 'trigger': trigger,
                })
                return True
            return False

        def get_allowed_triggers(self):
            return list(set(t['trigger'] for t in self._transitions
                            if t['from'] == self._current_state))

        def on_enter(self, state, hook):
            self._state_entry_hooks[state].append(hook)
            return self

        def on_exit(self, state, hook):
            self._state_exit_hooks[state].append(hook)
            return self

        def context_set(self, k, v):
            self._context[k] = v

        def context_get(self, k, default=None):
            return self._context.get(k, default)

        def force_state(self, new_state, reason="forced"):
            self.state = new_state


# ============================================================================
# 槽位定义
# ============================================================================

# Slot 使用 namedtuple，不可变且轻量
Slot = namedtuple('Slot', [
    'name',          # 槽位名称（英文标识符）
    'type',          # 数据类型 int / str / date
    'required',      # 是否必填
    'prompt',        # 向用户询问时的提示词
    'validator',     # 验证函数: value -> bool
    'default',       # 默认值（可选槽位使用）
])

# 设置默认字段值
Slot.__new__.__defaults__ = (None,) * len(Slot._fields)
# 重新定义以正确设置默认值
Slot = namedtuple('Slot', [
    'name', 'type', 'required', 'prompt', 'validator', 'default'
])
# 为 namedtuple 设置默认值的 Pythonic 方式
_OriginalSlot = Slot

def create_slot(name: str, type: type, required: bool = True,
                prompt: str = "", validator: Optional[Callable] = None,
                default: Any = None) -> Slot:
    """创建槽位定义（工厂函数）"""
    if prompt == "":
        prompt = f"请提供{name}"
    return Slot(name=name, type=type, required=required,
                prompt=prompt, validator=validator, default=default)


# ============================================================================
# 槽位填充状态机
# ============================================================================

class SlotFillingFSM(DialogFSM):
    """
    槽位填充有限状态机。

    在 DialogFSM 基础上增加：
    - 槽位注册与管理
    - 逐个槽位填充逻辑
    - 缺失必填槽位检测
    - 已收集数据的结构化导出
    """

    def __init__(self, name: str = "SlotFillingFSM",
                 timeout_seconds: float = 300.0):
        super().__init__(name=name, initial_state=DialogState.IDLE,
                         timeout_seconds=timeout_seconds)
        self._slots: Dict[str, Slot] = {}           # 槽位定义
        self._slot_order: List[str] = []            # 填充顺序
        self._filled_values: Dict[str, Any] = {}    # 已填充的值
        self._current_slot_index: int = 0           # 当前正在填充的槽位索引
        self._attempts_per_slot: Dict[str, int] = {}# 每个槽位的尝试次数
        self._max_attempts: int = 3                 # 单个槽位最大尝试次数

    # ---- 槽位注册 ----

    def register_slot(self, slot: Slot) -> 'SlotFillingFSM':
        """注册一个槽位。支持链式调用。"""
        self._slots[slot.name] = slot
        self._slot_order.append(slot.name)
        self._filled_values[slot.name] = slot.default
        self._attempts_per_slot[slot.name] = 0
        return self

    def register_slots(self, slots: List[Slot]) -> 'SlotFillingFSM':
        """批量注册槽位"""
        for slot in slots:
            self.register_slot(slot)
        return self

    # ---- 填充逻辑 ----

    def fill_slot(self, slot_name: str, value: Any) -> Tuple[bool, str]:
        """
        尝试填充一个槽位。

        Args:
            slot_name: 槽位名称
            value: 用户提供的值

        Returns:
            (success, message): 是否成功及说明信息
        """
        if slot_name not in self._slots:
            return False, f"未知槽位: {slot_name}"

        slot = self._slots[slot_name]
        self._attempts_per_slot[slot_name] += 1

        # 类型转换
        converted_value = value
        try:
            if slot.type == int:
                converted_value = int(value)
            elif slot.type == float:
                converted_value = float(value)
            elif slot.type == str:
                converted_value = str(value)
        except (ValueError, TypeError):
            return False, f"值 '{value}' 无法转换为 {slot.type.__name__} 类型"

        # 验证器检查
        if slot.validator is not None:
            try:
                if not slot.validator(converted_value):
                    return False, f"值 '{converted_value}' 未通过验证"
            except Exception as e:
                return False, f"验证时出错: {e}"

        # 填充成功
        self._filled_values[slot_name] = converted_value
        return True, f"槽位 [{slot_name}] 填充成功: {converted_value}"

    def get_current_slot(self) -> Optional[Slot]:
        """获取当前正在收集的槽位"""
        if self._current_slot_index < len(self._slot_order):
            slot_name = self._slot_order[self._current_slot_index]
            return self._slots.get(slot_name)
        return None

    def advance_to_next_slot(self) -> bool:
        """
        推进到下一个未填充的必填槽位。

        Returns:
            True 如果找到了下一个槽位，False 如果所有必填槽位已填完
        """
        for i in range(self._current_slot_index, len(self._slot_order)):
            slot_name = self._slot_order[i]
            slot = self._slots[slot_name]
            if slot.required and self._filled_values.get(slot_name) in (None, slot.default):
                self._current_slot_index = i
                return True
        # 所有必填已填完
        self._current_slot_index = len(self._slot_order)
        return False

    def get_missing_required(self) -> List[Slot]:
        """获取所有尚未填充的必填槽位"""
        missing = []
        for slot_name in self._slot_order:
            slot = self._slots[slot_name]
            if not slot.required:
                continue
            current_value = self._filled_values.get(slot_name)
            if current_value is None or current_value == slot.default:
                missing.append(slot)
        return missing

    def all_required_filled(self) -> bool:
        """检查所有必填槽位是否已填充"""
        return len(self.get_missing_required()) == 0

    def get_collected_data(self) -> Dict[str, Any]:
        """获取所有已收集的数据（排除未填充的默认值）"""
        result = {}
        for name, value in self._filled_values.items():
            slot = self._slots[name]
            if value is not None and value != slot.default:
                result[name] = value
        return result

    def get_next_prompt(self) -> Optional[str]:
        """
        获取下一个需要向用户询问的提示词。

        Returns:
            提示词字符串，如果所有必填已填完返回 None
        """
        missing = self.get_missing_required()
        if missing:
            return missing[0].prompt
        return None

    def current_fill_progress(self) -> Dict[str, Any]:
        """返回当前填充进度"""
        total = len(self._slots)
        required = sum(1 for s in self._slots.values() if s.required)
        filled = sum(
            1 for n, v in self._filled_values.items()
            if v is not None and v != self._slots[n].default
        )
        required_filled = sum(
            1 for n, v in self._filled_values.items()
            if self._slots[n].required
            and v is not None and v != self._slots[n].default
        )
        return {
            "total_slots": total,
            "required_slots": required,
            "filled_slots": filled,
            "required_filled": required_filled,
            "all_required_ready": self.all_required_filled(),
            "current_slot_index": self._current_slot_index,
            "filled_values": dict(self._filled_values),
        }

    def reset(self):
        """重置所有槽位"""
        for name in self._slot_order:
            self._filled_values[name] = self._slots[name].default
            self._attempts_per_slot[name] = 0
        self._current_slot_index = 0


# ============================================================================
# 餐厅预订槽位定义
# ============================================================================

def create_restaurant_booking_slots() -> List[Slot]:
    """创建餐厅预订场景的6个槽位"""

    # ---- 验证器 ----
    def validate_party_size(value: int) -> bool:
        return 1 <= value <= 20

    def validate_date(value: str) -> bool:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
            # 不允许预订过去的日期
            return parsed.date() >= datetime.now().date()
        except ValueError:
            return False

    def validate_time(value: str) -> bool:
        if not re.match(r'^\d{2}:\d{2}$', value):
            return False
        try:
            h, m = map(int, value.split(':'))
            return 0 <= h < 24 and 0 <= m < 60
        except ValueError:
            return False

    def validate_dietary(value: str) -> bool:
        # 饮食要求：可为空字符串或常见饮食类型
        valid_choices = ["", "无", "素食", "清真", "无麸质", "过敏-海鲜", "过敏-坚果", "过敏-乳制品"]
        return value in valid_choices or len(value) <= 50

    def validate_phone(value: str) -> bool:
        # 中国大陆手机号格式
        return bool(re.match(r'^1[3-9]\d{9}$', value))

    def validate_name(value: str) -> bool:
        return 1 <= len(value) <= 30 and not value.strip().isdigit()

    slots = [
        create_slot(
            name="party_size", type=int, required=True,
            prompt="请问几位用餐？（1-20人）",
            validator=validate_party_size,
        ),
        create_slot(
            name="date", type=str, required=True,
            prompt="请问预订日期是哪天？（格式：YYYY-MM-DD）",
            validator=validate_date,
        ),
        create_slot(
            name="time", type=str, required=True,
            prompt="请问几点到达？（格式：HH:MM，如 18:30）",
            validator=validate_time,
        ),
        create_slot(
            name="dietary", type=str, required=False,
            prompt="请问有特殊饮食要求吗？（选填，如素食、清真等）",
            validator=validate_dietary,
            default="无",
        ),
        create_slot(
            name="phone", type=str, required=True,
            prompt="请提供联系电话（11位手机号）",
            validator=validate_phone,
        ),
        create_slot(
            name="name", type=str, required=True,
            prompt="请问预订人姓名？",
            validator=validate_name,
        ),
    ]
    return slots


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示槽位填充流程"""

    print("=" * 70)
    print("  槽位填充状态机 (Slot-Filling FSM) 演示")
    print("=" * 70)
    print()

    # 创建 FSM 并注册槽位
    fsm = SlotFillingFSM(name="RestaurantBookingFSM")
    slots = create_restaurant_booking_slots()
    fsm.register_slots(slots)

    # 打印槽位定义
    print("【已注册槽位】")
    print("-" * 50)
    for slot in slots:
        req_mark = "★必填" if slot.required else "○选填"
        print(f"  [{slot.name}] ({slot.type.__name__}) {req_mark}")
        print(f"    提示: {slot.prompt}")
    print()

    # 模拟用户交互
    test_cases = [
        # (slot_name, value, expected_success)
        ("party_size", "4", True),
        ("date", "2026-07-15", True),
        ("time", "18:30", True),
        ("dietary", "素食", True),
        ("phone", "13800138000", True),
        ("name", "张三", True),
        # 测试无效值
        ("party_size", "50", False),    # 超过20人
        ("date", "2020-01-01", False),  # 过去日期
        ("time", "25:00", False),       # 无效时间
        ("phone", "12345", False),      # 无效手机号
    ]

    print("【逐个填充测试】")
    print("-" * 50)
    for slot_name, value, expected in test_cases:
        success, msg = fsm.fill_slot(slot_name, value)
        status = "✓" if success == expected else "✗ 意外"
        print(f"  {status} fill_slot('{slot_name}', '{value}')")
        print(f"    结果: {msg}")

    print()
    print("【进度检查】")
    print(json.dumps(fsm.current_fill_progress(), indent=2, ensure_ascii=False))

    print()
    print("【缺失必填槽位】")
    missing = fsm.get_missing_required()
    if missing:
        for s in missing:
            print(f"  - {s.name}: {s.prompt}")
    else:
        print("  所有必填槽位已填充 ✓")

    print()
    print("【已收集数据】")
    print(json.dumps(fsm.get_collected_data(), indent=2, ensure_ascii=False))

    print()
    print("【下一个提示词】")
    next_prompt = fsm.get_next_prompt()
    print(f"  {next_prompt}")

    print()
    print("=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
