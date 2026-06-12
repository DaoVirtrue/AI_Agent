# 坑点 #5：对话状态机卡死

## 症状
用户反复输入手机号，Agent 反复说"请提供您的手机号码"，陷入死循环。用户最终愤怒退出。

## 根因
1. **无超时机制**：`COLLECTING_INFO` 状态没有超时限制，用户输入无效值就一直卡着
2. **验证过于严格**：正则要求 "1[3-9]\\d{9}"，用户输入 "139-1234-5678"（带横线）被拒绝，但没有提示格式要求
3. **无逃生通道**：用户说"跳过"、"算了"、"不要了"，FSM 没有对应迁移规则
4. **重试无上限**：同一槽位失败后无限重试

## 解决方案

### 1. 超时自动推进
```python
class DialogFSM:
    def __init__(self):
        self.timeout_map = {
            DialogState.COLLECTING_INFO: 60,   # 60秒无响应就推进
            DialogState.CONFIRMING: 30,
            DialogState.PROCESSING: 120,
        }
    
    def check_and_handle_timeout(self):
        if self.check_timeout():
            if self.state == DialogState.COLLECTING_INFO:
                self.force_state(DialogState.ASKING_CLARIFICATION)
                return "我注意到您可能遇到困难，可以换种方式提供信息吗？"
            elif self.state == DialogState.CONFIRMING:
                self.force_state(DialogState.PROCESSING)
                return "已超时，我将使用已确认的信息继续处理。"
```

### 2. 逃生触发器
```python
ESCAPE_TRIGGERS = {
    "skip": "跳过当前步骤",
    "cancel": "取消整个流程",
    "help": "获取帮助",
    "agent": "转人工客服",
    "restart": "重新开始",
}

# 在 FSM 中为每个状态添加逃生迁移
for state in DialogState:
    fsm.add_transition(state, DialogState.FAREWELL, "cancel", priority=100)
    fsm.add_transition(state, DialogState.RESPONDING, "help", priority=90)
```

### 3. 槽位验证放宽 + 提示
```python
class SlotFillingFSM:
    MAX_RETRIES = 3  # 每个槽位最多重试3次
    
    def fill_slot(self, slot_name, value):
        slot = self.slots[slot_name]
        self._retry_count[slot_name] = self._retry_count.get(slot_name, 0)
        
        try:
            normalized = self._normalize(value, slot.type)  # 去除横线、空格
            if slot.validator and not slot.validator(normalized):
                raise ValueError(f"格式不正确，期望格式：{slot.format_hint}")
            self.filled_slots[slot_name] = normalized
            return True
        except Exception as e:
            self._retry_count[slot_name] += 1
            if self._retry_count[slot_name] >= self.MAX_RETRIES:
                # 超过重试次数，跳过该槽位
                self.filled_slots[slot_name] = slot.default
                return True
            raise  # 继续重试
```

### 4. 优雅降级
```python
def graceful_degradation(self):
    """收集到部分信息就继续，不卡死"""
    missing = self.get_missing_required()
    if len(missing) <= 1:  # 只差1个就放过
        for slot_name in missing:
            self.filled_slots[slot_name] = self.slots[slot_name].default
        self.transition("confirm")
        return True
    return False
```

## 检查清单
- [ ] 每个状态是否配置了超时？
- [ ] 是否有全局逃生触发器（skip/cancel/help）？
- [ ] 槽位填充是否有最大重试次数？
- [ ] 是否支持部分信息就继续（优雅降级）？
- [ ] 是否在验证失败时给出了明确的格式提示？
- [ ] FSM 状态转换是否被记录到日志？
