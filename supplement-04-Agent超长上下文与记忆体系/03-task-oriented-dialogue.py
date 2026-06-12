#!/usr/bin/env python3
"""
面向任务的对话Agent (Task-Oriented Dialogue Agent)
====================================================
集成 SlotFillingFSM，模拟完整的餐厅预订对话流程。包含：
- RestaurantBookingAgent: 集成槽位填充和模拟工具的对话Agent
- ConversationSimulator: 支持多场景（happy_path, invalid_input, cancellation）
- 意图分类器和对话流程管理

三个演示场景：
  1. happy_path: 用户顺利提供所有信息，完成预订
  2. invalid_input: 用户多次输入无效值，触发澄清和重试
  3. cancellation: 用户中途取消预订
"""

from __future__ import annotations

import time
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto


# ============================================================================
# 内联依赖（保证文件可独立运行）
# ============================================================================

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
            DialogState.IDLE: "空闲", DialogState.GREETING: "问候",
            DialogState.COLLECTING_INFO: "收集信息", DialogState.CONFIRMING: "确认",
            DialogState.PROCESSING: "处理中", DialogState.RESPONDING: "响应中",
            DialogState.ASKING_CLARIFICATION: "澄清", DialogState.FAREWELL: "告别",
        }
        return names.get(self, "未知")


# 简化版 Slot
from collections import namedtuple
Slot = namedtuple('Slot', ['name', 'type', 'required', 'prompt', 'validator', 'default'])
Slot.__new__.__defaults__ = (None,) * len(Slot._fields)


# 简化版 SlotFillingFSM（核心功能）
class SimpleFSM:
    """简化的 FSM 基础类"""
    def __init__(self, name="FSM"):
        self.name = name
        self._state = DialogState.IDLE
        self._context: Dict[str, Any] = {}
        self._history: List[Dict] = []
        self._transitions: List[Dict] = []

    @property
    def state(self): return self._state

    @state.setter
    def state(self, s): self._state = s

    def add_transition(self, frm, to, trigger, guard=None, action=None, desc="", priority=0):
        self._transitions.append({
            'from': frm, 'to': to, 'trigger': trigger,
            'guard': guard, 'action': action, 'desc': desc, 'priority': priority
        })
        return self

    def transition(self, trigger):
        candidates = [t for t in self._transitions
                      if t['from'] == self._state and t['trigger'] == trigger]
        candidates.sort(key=lambda t: t['priority'], reverse=True)
        for t in candidates:
            if t['guard'] and not t['guard']():
                continue
            self._state = t['to']
            if t['action']:
                try: t['action']()
                except: pass
            return True
        return False


class SlotFillingFSM(SimpleFSM):
    """槽位填充 FSM 核心"""
    def __init__(self, name="SlotFSM"):
        super().__init__(name)
        self._slots: Dict[str, Slot] = {}
        self._slot_order: List[str] = []
        self._filled_values: Dict[str, Any] = {}
        self._attempts_per_slot: Dict[str, int] = {}
        self._current_slot_index = 0

    def register_slot(self, slot: Slot):
        self._slots[slot.name] = slot
        self._slot_order.append(slot.name)
        self._filled_values[slot.name] = slot.default
        self._attempts_per_slot[slot.name] = 0
        return self

    def register_slots(self, slots: List[Slot]):
        for s in slots:
            self.register_slot(s)
        return self

    def fill_slot(self, name, value):
        if name not in self._slots:
            return False, f"未知槽位: {name}"
        slot = self._slots[name]
        self._attempts_per_slot[name] += 1
        try:
            cv = slot.type(value)
        except (ValueError, TypeError):
            return False, f"类型转换失败（期望 {slot.type.__name__}）"
        if slot.validator and not slot.validator(cv):
            return False, f"验证未通过"
        self._filled_values[name] = cv
        return True, f"填充成功"

    def get_missing_required(self):
        return [self._slots[n] for n in self._slot_order
                if self._slots[n].required and self._filled_values.get(n) in (None, self._slots[n].default)]

    def all_required_filled(self):
        return len(self.get_missing_required()) == 0

    def get_collected_data(self):
        return {n: v for n, v in self._filled_values.items()
                if v is not None and v != self._slots[n].default}


# ============================================================================
# 意图枚举
# ============================================================================

class Intent(Enum):
    """用户意图分类"""
    PROVIDE_INFO = auto()       # 提供信息
    ASK_QUESTION = auto()       # 提问
    CONFIRM = auto()            # 确认
    DENY = auto()               # 否认
    CANCEL = auto()             # 取消
    MODIFY = auto()             # 修改
    GREET = auto()              # 问候
    FAREWELL = auto()           # 告别
    UNKNOWN = auto()            # 未知


# ============================================================================
# 模拟工具
# ============================================================================

@dataclass
class ReservationResult:
    """预订结果"""
    success: bool
    reservation_id: str = ""
    message: str = ""
    timestamp: float = 0.0


class RestaurantTools:
    """模拟的餐厅预订后台工具"""

    def __init__(self):
        self._available_tables: Dict[str, Dict[str, List[int]]] = {
            # date -> { time -> [available_sizes] }
        }
        self._reservations: List[Dict[str, Any]] = []
        self._init_mock_data()

    def _init_mock_data(self):
        """初始化模拟数据"""
        from datetime import datetime, timedelta
        base = datetime.now()
        for day_offset in range(30):
            d = (base + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            times = {}
            for hour in range(11, 22):
                t = f"{hour:02d}:00"
                # 随机可用桌位大小
                available = random.sample([2, 2, 4, 4, 4, 6, 6, 8, 10, 12],
                                          k=random.randint(3, 8))
                times[t] = available
            self._available_tables[d] = times

    def check_availability(self, date: str, time: str, party_size: int) -> Dict[str, Any]:
        """
        检查指定日期时间是否有符合人数的桌位。

        Returns:
            {"available": bool, "alternatives": list[str], "message": str}
        """
        if date not in self._available_tables:
            return {"available": False, "alternatives": [], "message": "该日期不可预订"}

        times = self._available_tables[date]

        # 精确匹配
        if time in times:
            available_sizes = times[time]
            can_fit = any(s >= party_size for s in available_sizes)
            if can_fit:
                return {"available": True, "alternatives": [], "message": "有空位！"}

        # 推荐相近时间段
        time_h = int(time.split(":")[0])
        alternatives = []
        for alt_time in sorted(times.keys()):
            alt_h = int(alt_time.split(":")[0])
            if abs(alt_h - time_h) <= 2:
                alt_sizes = times[alt_time]
                if any(s >= party_size for s in alt_sizes):
                    alternatives.append(alt_time)

        return {
            "available": False,
            "alternatives": alternatives[:3],
            "message": "该时段已满" if alternatives else "当天无可选时段",
        }

    def make_reservation(self, data: Dict[str, Any]) -> ReservationResult:
        """
        执行预订操作（模拟）。

        Args:
            data: 包含 party_size, date, time, dietary, phone, name 的字典

        Returns:
            ReservationResult
        """
        required = ["party_size", "date", "time", "phone", "name"]
        missing = [k for k in required if k not in data]
        if missing:
            return ReservationResult(
                success=False,
                message=f"缺少必要信息: {', '.join(missing)}",
            )

        # 再次检查可用性
        availability = self.check_availability(
            data["date"], data["time"], data["party_size"]
        )

        if not availability["available"]:
            alt_text = "可选时段: " + ", ".join(availability["alternatives"]) \
                if availability["alternatives"] else "无替代时段"
            return ReservationResult(
                success=False,
                message=f"预订失败: {availability['message']}。{alt_text}",
            )

        # 生成预订ID
        reservation_id = f"RES-{int(time.time())}-{random.randint(1000, 9999)}"

        record = {
            "id": reservation_id,
            **data,
            "status": "confirmed",
            "created_at": time.time(),
        }
        self._reservations.append(record)

        # 更新可用性缓存
        if data["date"] in self._available_tables:
            if data["time"] in self._available_tables[data["date"]]:
                sizes = self._available_tables[data["date"]][data["time"]]
                # 移除被占用的桌位
                for size_to_remove in [s for s in sizes if s >= data["party_size"]]:
                    if size_to_remove in sizes:
                        sizes.remove(size_to_remove)
                        break

        return ReservationResult(
            success=True,
            reservation_id=reservation_id,
            message=f"预订成功！预订号: {reservation_id}",
            timestamp=time.time(),
        )


# ============================================================================
# 餐厅预订Agent
# ============================================================================

class RestaurantBookingAgent:
    """
    餐厅预订对话Agent。

    集成槽位填充FSM和后台工具，实现完整的：
    - 意图分类
    - 多轮对话管理
    - 确认与修改流程
    - 错误恢复与重试
    """

    def __init__(self):
        self.fsm = SlotFillingFSM(name="RestaurantBookingAgent")
        self.tools = RestaurantTools()
        self._conversation_log: List[Dict[str, str]] = []
        self._init_slots()
        self._init_transitions()
        self._current_reservation_id: Optional[str] = None
        self._greeted = False

    def _init_slots(self):
        """初始化餐厅预订槽位"""

        def v_party_size(v): return isinstance(v, int) and 1 <= v <= 20
        def v_date(v):
            try:
                from datetime import datetime
                d = datetime.strptime(v, "%Y-%m-%d")
                return d.date() >= datetime.now().date()
            except: return False
        def v_time(v): return bool(re.match(r'^\d{2}:\d{2}$', v))
        def v_phone(v): return bool(re.match(r'^1[3-9]\d{9}$', v))
        def v_name(v): return isinstance(v, str) and 1 <= len(v) <= 30

        self.fsm.register_slots([
            Slot("party_size", int, True, "请问几位用餐？", v_party_size, None),
            Slot("date", str, True, "请问预订日期？(YYYY-MM-DD)", v_date, None),
            Slot("time", str, True, "请问几点到达？(HH:MM)", v_time, None),
            Slot("dietary", str, False, "有无特殊饮食要求？", lambda v: True, "无"),
            Slot("phone", str, True, "请提供联系电话", v_phone, None),
            Slot("name", str, True, "请问预订人姓名？", v_name, None),
        ])

    def _init_transitions(self):
        """设置状态转移"""
        f = self.fsm

        f.add_transition(DialogState.IDLE, DialogState.GREETING, "greet")
        f.add_transition(DialogState.GREETING, DialogState.COLLECTING_INFO, "start_booking")
        f.add_transition(DialogState.COLLECTING_INFO, DialogState.CONFIRMING, "all_info_collected",
                         guard=lambda: f.all_required_filled())
        f.add_transition(DialogState.COLLECTING_INFO, DialogState.COLLECTING_INFO, "info_provided")
        f.add_transition(DialogState.COLLECTING_INFO, DialogState.ASKING_CLARIFICATION, "need_clarify")
        f.add_transition(DialogState.COLLECTING_INFO, DialogState.FAREWELL, "cancel")

        f.add_transition(DialogState.CONFIRMING, DialogState.PROCESSING, "confirmed")
        f.add_transition(DialogState.CONFIRMING, DialogState.COLLECTING_INFO, "modify")
        f.add_transition(DialogState.CONFIRMING, DialogState.FAREWELL, "cancel")

        f.add_transition(DialogState.PROCESSING, DialogState.RESPONDING, "processing_done")
        f.add_transition(DialogState.PROCESSING, DialogState.ASKING_CLARIFICATION, "need_clarify")

        f.add_transition(DialogState.RESPONDING, DialogState.FAREWELL, "done")
        f.add_transition(DialogState.RESPONDING, DialogState.COLLECTING_INFO, "new_booking")
        f.add_transition(DialogState.RESPONDING, DialogState.RESPONDING, "ask_more")

        f.add_transition(DialogState.ASKING_CLARIFICATION, DialogState.COLLECTING_INFO, "clarified")
        f.add_transition(DialogState.ASKING_CLARIFICATION, DialogState.FAREWELL, "give_up")

    # ---- 意图分类 ----

    def _classify_intent(self, message: str) -> Intent:
        """
        基于关键词的意图分类器（实际项目中可由LLM完成）。

        Args:
            message: 用户输入文本

        Returns:
            分类后的用户意图
        """
        msg_lower = message.lower().strip()

        # 取消相关
        if any(w in msg_lower for w in ["取消", "算了", "不要了", "cancel", "quit"]):
            return Intent.CANCEL

        # 确认相关
        if any(w in msg_lower for w in ["确认", "是的", "没错", "对的", "可以", "好", "行", "ok", "yes", "嗯"]):
            return Intent.CONFIRM

        # 否认相关
        if any(w in msg_lower for w in ["不对", "不是", "错了", "no", "不"]):
            return Intent.DENY

        # 修改相关
        if any(w in msg_lower for w in ["修改", "改", "换", "变更", "调整"]):
            return Intent.MODIFY

        # 问候相关
        if any(w in msg_lower for w in ["你好", "hi", "hello", "嗨", "在吗"]):
            return Intent.GREET

        # 告别相关
        if any(w in msg_lower for w in ["再见", "拜拜", "bye", "谢谢"]):
            return Intent.FAREWELL

        # 提问相关
        if "?" in msg_lower or "?" in msg_lower or any(w in msg_lower for w in ["什么", "怎么", "如何", "为什么"]):
            return Intent.ASK_QUESTION

        # 默认：提供信息
        return Intent.PROVIDE_INFO

    # ---- 信息提取 ----

    def _extract_info(self, message: str) -> Dict[str, Any]:
        """
        从用户消息中尝试提取槽位信息。
        简单规则匹配，实际项目可由 NLU 模型完成。

        Returns:
            {slot_name: value} 提取到的键值对
        """
        extracted = {}

        # 提取人数
        m = re.search(r'(\d+)\s*[位个人]', message)
        if m:
            extracted["party_size"] = m.group(1)

        # 提取日期
        m = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', message)
        if m:
            extracted["date"] = m.group(1).replace("/", "-")

        # 提取时间
        m = re.search(r'(\d{1,2}[:：]\d{2})', message)
        if m:
            extracted["time"] = m.group(1).replace("：", ":").zfill(5)

        # 提取手机号
        m = re.search(r'(1[3-9]\d{9})', message)
        if m:
            extracted["phone"] = m.group(1)

        # 提取姓名（简单策略：2-3个汉字）
        m = re.search(r'[我姓是叫]?([一-龥]{2,3})[，,。.!！]', message)
        if m:
            extracted["name"] = m.group(1)

        return extracted

    # ---- 核心处理 ----

    def handle_message(self, message: str) -> str:
        """
        处理用户消息并返回Agent响应。这是对话Agent的主入口。

        Args:
            message: 用户输入的原始文本

        Returns:
            Agent的响应文本
        """
        self._conversation_log.append({
            "role": "user",
            "content": message,
            "timestamp": time.time(),
        })

        intent = self._classify_intent(message)
        current_state = self.fsm.state
        response = ""

        # === 状态路由 ===

        if current_state == DialogState.IDLE:
            response = self._handle_idle(intent, message)

        elif current_state == DialogState.GREETING:
            response = self._handle_greeting(intent, message)

        elif current_state == DialogState.COLLECTING_INFO:
            response = self._handle_collecting_info(intent, message)

        elif current_state == DialogState.CONFIRMING:
            response = self._handle_confirming(intent, message)

        elif current_state == DialogState.PROCESSING:
            response = self._handle_processing(intent, message)

        elif current_state == DialogState.RESPONDING:
            response = self._handle_responding(intent, message)

        elif current_state == DialogState.ASKING_CLARIFICATION:
            response = self._handle_clarification(intent, message)

        elif current_state == DialogState.FAREWELL:
            response = self._handle_farewell(intent, message)

        self._conversation_log.append({
            "role": "agent",
            "content": response,
            "timestamp": time.time(),
            "state": current_state.name,
        })

        return response

    def _handle_idle(self, intent: Intent, msg: str) -> str:
        if intent == Intent.GREET:
            self.fsm.transition("greet")
            return "您好！欢迎使用餐厅预订服务。请问有什么可以帮助您的？"
        self.fsm.transition("greet")
        return "您好！我是餐厅预订助手，请问需要预订餐位吗？"

    def _handle_greeting(self, intent: Intent, msg: str) -> str:
        if intent == Intent.GREET:
            return "您好！请问需要预订餐位吗？"
        self.fsm.transition("start_booking")
        missing = self.fsm.get_missing_required()
        if missing:
            return "好的，我来帮您预订餐位。" + missing[0].prompt
        return "请提供预订信息。"

    def _handle_collecting_info(self, intent: Intent, msg: str) -> str:
        if intent == Intent.CANCEL:
            self.fsm.transition("cancel")
            return "好的，已取消当前预订流程。再见！"

        # 尝试提取信息
        extracted = self._extract_info(msg)
        if extracted:
            for slot_name, value in extracted.items():
                success, result_msg = self.fsm.fill_slot(slot_name, value)
                if not success:
                    return f"抱歉，{slot_name} 的信息不正确: {result_msg}。请重新提供。"

        # 检查是否所有必填已填完
        if self.fsm.all_required_filled():
            self.fsm.transition("all_info_collected")
            return self._build_confirmation_message()
        else:
            # 询问下一个缺失槽位
            self.fsm.transition("info_provided")
            missing = self.fsm.get_missing_required()
            if missing:
                return f"收到。{missing[0].prompt}"
            return "请继续提供预订信息。"

    def _handle_confirming(self, intent: Intent, msg: str) -> str:
        if intent == Intent.CONFIRM:
            self.fsm.transition("confirmed")
            return "正在为您处理预订，请稍候..."

        if intent == Intent.DENY or intent == Intent.MODIFY:
            self.fsm.transition("modify")
            return "好的，请问需要修改哪一项信息？请直接告诉我新的内容。"

        if intent == Intent.CANCEL:
            self.fsm.transition("cancel")
            return "好的，已取消预订。再见！"

        return "请确认预订信息是否正确（是/否）。"

    def _handle_processing(self, intent: Intent, msg: str) -> str:
        # 执行预订
        data = self.fsm.get_collected_data()
        result = self.tools.make_reservation(data)

        if result.success:
            self._current_reservation_id = result.reservation_id
            self.fsm.transition("processing_done")
            return f"预订成功！您的预订号为 {result.reservation_id}。{self._build_summary()}"

        # 预订失败
        if self.tools.check_availability(data.get("date", ""), data.get("time", ""), data.get("party_size", 1)).get("alternatives"):
            self.fsm.transition("need_clarify")
            return result.message + "\n需要我帮您尝试其他时段吗？"
        else:
            self.fsm.transition("need_clarify")
            return result.message + "\n需要调整预订条件吗？"

    def _handle_responding(self, intent: Intent, msg: str) -> str:
        if intent == Intent.FAREWELL or intent == Intent.CANCEL:
            self.fsm.transition("done")
            return "感谢您的预订，祝您用餐愉快！再见！"
        if intent == Intent.ASK_QUESTION:
            return "请问有什么我可以帮助您的？"
        self.fsm.transition("new_booking")
        return "需要再预订其他时间的餐位吗？"

    def _handle_clarification(self, intent: Intent, msg: str) -> str:
        if intent == Intent.CANCEL:
            self.fsm.transition("give_up")
            return "好的，已退出。如有需要请随时联系。"
        self.fsm.transition("clarified")
        missing = self.fsm.get_missing_required()
        if missing:
            return f"明白了。{missing[0].prompt}"
        return "请提供需要的信息。"

    def _handle_farewell(self, intent: Intent, msg: str) -> str:
        return "感谢您的光临，再见！"

    def _build_confirmation_message(self) -> str:
        """构建确认消息"""
        data = self.fsm.get_collected_data()
        parts = [
            "请确认以下预订信息：",
            f"  用餐人数: {data.get('party_size', 'N/A')} 位",
            f"  日期: {data.get('date', 'N/A')}",
            f"  时间: {data.get('time', 'N/A')}",
            f"  饮食要求: {data.get('dietary', '无')}",
            f"  联系电话: {data.get('phone', 'N/A')}",
            f"  预订人: {data.get('name', 'N/A')}",
            "",
            "确认无误请回复"是"，如需修改请回复"修改"。",
        ]
        return "\n".join(parts)

    def _build_summary(self) -> str:
        """构建预订摘要"""
        data = self.fsm.get_collected_data()
        return (f"预订摘要：{data.get('name')}，{data.get('party_size')}位，"
                f"{data.get('date')} {data.get('time')}")

    def reset(self):
        """重置Agent状态"""
        self.fsm.__init__(name="RestaurantBookingAgent")
        self._init_slots()
        self._init_transitions()
        self._conversation_log.clear()
        self._current_reservation_id = None
        self._greeted = False


# ============================================================================
# 对话模拟器
# ============================================================================

@dataclass
class ConversationScenario:
    """对话场景定义"""
    name: str
    description: str
    messages: List[str]  # 用户消息序列


class ConversationSimulator:
    """对话模拟器：运行预设场景并输出完整对话记录"""

    def __init__(self, agent: RestaurantBookingAgent):
        self.agent = agent

    def run_scenario(self, scenario: ConversationScenario) -> List[Dict[str, str]]:
        """
        运行一个对话场景。

        Returns:
            完整的对话记录 [(role, content), ...]
        """
        log: List[Dict[str, str]] = []
        self.agent.reset()

        for i, msg in enumerate(scenario.messages):
            log.append({"turn": str(i + 1), "role": "👤 用户", "content": msg})
            response = self.agent.handle_message(msg)
            log.append({"turn": str(i + 1), "role": "🤖 Agent", "content": response})

        return log

    def print_scenario(self, scenario: ConversationScenario):
        """运行并美化打印一个场景"""
        print(f"\n{'='*60}")
        print(f"  场景: {scenario.name}")
        print(f"  描述: {scenario.description}")
        print(f"{'='*60}")

        log = self.run_scenario(scenario)
        for entry in log:
            print(f"\n  [{entry['turn']}] {entry['role']}:")
            # 缩进多行内容
            for line in entry['content'].split('\n'):
                print(f"    │ {line}")

        print(f"\n  --- 场景结束 ---")


# ============================================================================
# 预定义场景
# ============================================================================

def create_scenarios() -> List[ConversationScenario]:
    """创建三个测试场景"""

    # 场景1：顺利预订
    happy_path = ConversationScenario(
        name="happy_path",
        description="用户顺利提供所有信息，完成预订",
        messages=[
            "你好，我想预订餐位",
            "4个人",
            "下周六，2026-07-18",
            "晚上6点半",
            "没有特殊要求",
            "13800138000",
            "我叫张三",
            "是的，确认",
        ],
    )

    # 场景2：无效输入
    invalid_input = ConversationScenario(
        name="invalid_input",
        description="用户多次输入无效值，Agent引导纠正",
        messages=[
            "我要预订",
            "100个人",      # 无效：超过20人
            "好的，4个人",   # 修正
            "昨天",          # 无效：过去日期
            "2026-07-20",   # 修正
            "下午3点",       # 非标准格式
            "15:00",        # 修正
            "13800138000",
            "李四",
            "确认",
        ],
    )

    # 场景3：中途取消
    cancellation = ConversationScenario(
        name="cancellation",
        description="用户在信息收集中途取消预订",
        messages=[
            "你好",
            "预订餐位",
            "6个人",
            "2026-08-01",
            "算了，不订了",
        ],
    )

    return [happy_path, invalid_input, cancellation]


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示"""
    print("=" * 70)
    print("  面向任务的对话Agent (Task-Oriented Dialogue) 演示")
    print("=" * 70)

    agent = RestaurantBookingAgent()
    simulator = ConversationSimulator(agent)
    scenarios = create_scenarios()

    for scenario in scenarios:
        simulator.print_scenario(scenario)

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
