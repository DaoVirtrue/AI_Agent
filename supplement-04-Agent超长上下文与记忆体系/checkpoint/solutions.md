# Supplement 04 检验点：参考解答

---

## 练习 1：客服对话 FSM

```python
from enum import Enum
from dataclasses import dataclass
from collections import deque
import time

class DialogState(Enum):
    GREETING = "greeting"
    COLLECTING_ISSUE = "collecting_issue"
    TROUBLESHOOTING = "troubleshooting"
    CONFIRMING_RESOLVED = "confirming_resolved"
    COLLECTING_FEEDBACK = "collecting_feedback"
    FAREWELL = "farewell"
    HANDOFF = "handoff"

@dataclass
class FSMTransition:
    from_state: DialogState
    to_state: DialogState
    trigger: str
    priority: int = 0

class SupportFSM:
    def __init__(self):
        self.state = DialogState.GREETING
        self.timeout_map = {DialogState.TROUBLESHOOTING: 120}
        self._last_transition = time.monotonic()
        self._setup_transitions()
    
    def _setup_transitions(self):
        self.transitions = [
            FSMTransition(DialogState.GREETING, DialogState.COLLECTING_ISSUE, "ask_issue"),
            FSMTransition(DialogState.COLLECTING_ISSUE, DialogState.TROUBLESHOOTING, "issue_collected"),
            FSMTransition(DialogState.TROUBLESHOOTING, DialogState.CONFIRMING_RESOLVED, "resolved"),
            FSMTransition(DialogState.CONFIRMING_RESOLVED, DialogState.COLLECTING_FEEDBACK, "confirmed"),
            FSMTransition(DialogState.COLLECTING_FEEDBACK, DialogState.FAREWELL, "done"),
            FSMTransition(DialogState.FAREWELL, DialogState.GREETING, "new_session"),
            # 逃生：任意状态转人工
            *(FSMTransition(s, DialogState.HANDOFF, "human", 100) for s in DialogState),
            # 回退：故障排查中用户要求重新描述问题
            FSMTransition(DialogState.TROUBLESHOOTING, DialogState.COLLECTING_ISSUE, "restart"),
        ]
    
    def transition(self, trigger: str) -> str:
        applicable = [t for t in self.transitions 
                      if t.from_state == self.state and t.trigger == trigger]
        if not applicable:
            return f"无匹配迁移: {trigger} in {self.state.value}"
        best = max(applicable, key=lambda t: t.priority)
        old = self.state
        self.state = best.to_state
        self._last_transition = time.monotonic()
        return f"{old.value} --{trigger}--> {self.state.value}"
    
    def check_timeout(self) -> bool:
        limit = self.timeout_map.get(self.state)
        if limit and (time.monotonic() - self._last_transition) > limit:
            self.state = DialogState.HANDOFF
            return True
        return False

# ---- demo ----
if __name__ == "__main__":
    fsm = SupportFSM()
    print("初始状态:", fsm.state.value)
    print(fsm.transition("ask_issue"))
    print(fsm.transition("issue_collected"))
    print(fsm.transition("resolved"))
    print(fsm.transition("confirmed"))
    print(fsm.transition("done"))
    print(fsm.transition("human"))  # 逃生到人工
```

---

## 练习 2：航班订票槽位填充

```python
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass
class Slot:
    name: str
    type: type
    required: bool = True
    prompt: str = ""
    validator: Optional[Callable] = None
    default: Any = None
    format_hint: str = ""
    max_retries: int = 3

class FlightBookingFSM:
    def __init__(self):
        self.slots = {}
        self.filled = {}
        self._retries = {}
        self._register_slots()
    
    def _register_slots(self):
        slots = [
            Slot("departure", str, True, "请输入出发城市", 
                 validator=lambda v: len(v) >= 2, format_hint="城市名至少2个字"),
            Slot("destination", str, True, "请输入目的城市",
                 validator=lambda v: len(v) >= 2 and v != self.filled.get("departure",""), 
                 format_hint="不能与出发城市相同"),
            Slot("date", str, True, "请输入出发日期",
                 validator=lambda v: bool(re.match(r'\d{4}-\d{2}-\d{2}', v)),
                 format_hint="YYYY-MM-DD 格式", default="2025-01-01"),
            Slot("return_date", str, False, "请输入返程日期（可选）", default=None),
            Slot("passengers", int, True, "请输入乘客人数",
                 validator=lambda v: 1 <= v <= 9, format_hint="1-9之间"),
            Slot("cabin", str, False, "请输入舱位（经济/商务/头等，可选）",
                 validator=lambda v: v in ("经济","商务","头等"), default="经济"),
            Slot("phone", str, True, "请输入手机号",
                 validator=lambda v: bool(re.match(r'^1[3-9]\d{9}$', v.replace('-','').replace(' ',''))),
                 format_hint="11位手机号"),
            Slot("email", str, False, "请输入邮箱（可选）",
                 validator=lambda v: "@" in v and "." in v, default=""),
        ]
        for s in slots:
            self.slots[s.name] = s
    
    def fill(self, name: str, value: Any) -> tuple[bool, str]:
        slot = self.slots[name]
        self._retries.setdefault(name, 0)
        try:
            # 归一化
            if slot.type == int:
                value = int(value)
            elif isinstance(value, str):
                value = value.replace('-', '').replace(' ', '')
            
            if slot.validator and not slot.validator(value):
                raise ValueError(f"格式不符合要求（期望：{slot.format_hint}）")
            self.filled[name] = value
            return True, f"✅ {name} 已填写"
        except Exception as e:
            self._retries[name] += 1
            if self._retries[name] >= slot.max_retries:
                self.filled[name] = slot.default
                return True, f"⚠️ {name} 已达最大重试次数，使用默认值"
            return False, f"❌ {e}（剩余重试: {slot.max_retries - self._retries[name]}）"
    
    def missing_required(self) -> list[str]:
        return [n for n, s in self.slots.items() if s.required and n not in self.filled]
    
    def progress(self) -> float:
        total = len([s for s in self.slots.values() if s.required])
        filled = total - len(self.missing_required())
        return filled / max(total, 1)

# ---- demo ----
if __name__ == "__main__":
    fsm = FlightBookingFSM()
    tests = [("departure","北京"), ("destination","上海"), ("date","2025-06-15"),
             ("passengers","2"), ("cabin","经济"), ("phone","13912345678")]
    for name, val in tests:
        ok, msg = fsm.fill(name, val)
        print(msg)
    print(f"\n进度: {fsm.progress():.0%}, 已完成: {list(fsm.filled.keys())}, 缺失: {fsm.missing_required()}")
```

---

## 练习 3-10 核心方案

### 练习 3：递归摘要压缩
```python
import tiktoken

def recursive_summarize(messages: list[str], max_tokens: int = 500) -> dict:
    enc = tiktoken.get_encoding("cl100k_base")
    total = sum(len(enc.encode(m)) for m in messages)
    if total <= max_tokens:
        return {"summary": "\n".join(messages), "ratio": 1.0}
    
    # 分段：每段大约 max_tokens 大小
    segment = []
    segments = []
    seg_tokens = 0
    for msg in messages:
        t = len(enc.encode(msg))
        if seg_tokens + t > max_tokens and segment:
            segments.append(" ".join(segment))
            segment, seg_tokens = [], 0
        segment.append(msg)
        seg_tokens += t
    if segment:
        segments.append(" ".join(segment))
    
    # 递归摘要每段（这里用启发式：取每段中心句作为摘要）
    sub_summaries = [s[:max_tokens//2] + "..." for s in segments]
    result = recursive_summarize(sub_summaries, max_tokens)
    return {"summary": result["summary"], "ratio": max_tokens / total}
```

### 练习 4：艾宾浩斯淘汰
```python
import math, time

class EbbinghausMemoryStore:
    def __init__(self, half_life_days=7):
        self.memories = {}  # id -> {content, strength, created_at}
        self.half_life = half_life_days
    
    def retention(self, days: float, strength: float) -> float:
        return math.exp(-days / (strength * self.half_life))
    
    def apply_forgetting(self, threshold=0.3) -> int:
        now = time.time()
        removed = 0
        for mid in list(self.memories.keys()):
            m = self.memories[mid]
            days = (now - m["created_at"]) / 86400
            if self.retention(days, m["strength"]) < threshold:
                del self.memories[mid]
                removed += 1
        return removed
```

### 练习 5-10
核心模式详见 supplement-04 对应模块文件：
- 练习 5 → `06-ultra-long-context-strategies.py`
- 练习 6 → `08-shared-memory-blackboard.py`
- 练习 7 → `10-persistent-memory-store.py`
- 练习 8 → `11-memory-evaluation-metrics.py`
- 练习 9 → `13-agent-memory-reflection.py`
- 练习 10 → `14-streaming-memory-interaction.py`

每个文件均包含完整的 `__main__` 演示代码，可直接运行参考。
