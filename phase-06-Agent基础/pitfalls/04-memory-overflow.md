# 记忆溢出 (Memory Overflow)

> 陷阱编号: #04 | 难度: 中等 | 影响: 高

---

## 1. 症状 (Symptoms)

Agent运行时出现以下现象：

- **记忆无限增长**: Agent的对话上下文随时间持续膨胀，内存占用不断攀升
- **响应质量下降**: 随着上下文变长，Agent开始"忘记"早期指令，回答变得不准确
- **响应速度变慢**: 处理超长上下文的推理和生成速度显著下降
- **最终OOM崩溃**: 达到系统内存上限，进程被杀死或抛出 `OutOfMemoryError`
- **模型截断**: 超过模型上下文窗口限制，最旧的消息被静默截断，导致关键信息丢失

### 典型日志表现

```
WARNING: Context size exceeded 8000 tokens. Some messages truncated.
ERROR: Memory allocation failed: requested 1.2GB, max available 512MB
WARNING: Agent response quality score dropped from 0.85 to 0.42 over session
```

---

## 2. 根本原因 (Root Cause)

### 2.1 滑动窗口过大

```python
# 反模式：无限制的上下文积累
class BadAgent:
    def __init__(self):
        self.conversation_history = []  # 无限增长！

    def chat(self, message):
        self.conversation_history.append(message)
        # 每次都将完整历史发送给LLM
        response = llm.generate(self.conversation_history)
        self.conversation_history.append(response)
        return response
```

### 2.2 没有压缩机制

- 旧消息没有被摘要或压缩，保留了所有冗余信息
- 大量"你好"、"谢谢"、"好的"等低价值消息占据上下文空间
- 早期系统指令被后续的冗余对话挤出上下文窗口

### 2.3 没有淘汰策略

- 所有记忆一视同仁，无法区分重要和次要信息
- 没有基于重要性、recency或frequency的淘汰机制
- 长期记忆和短期记忆没有分离，全部混在一个存储中

---

## 3. 真实场景 (Real-World Scenario)

### 场景: 50+轮长对话

```
轮次 1-5:  用户设置任务目标，Agent理解需求           [~500 tokens]
轮次 6-15: Agent执行搜索、查询工具                    [~2000 tokens]
轮次 16-25: 分析结果、生成报告                         [~3000 tokens]
轮次 26-35: 用户要求修改、Agent重新执行                 [~4000 tokens]
轮次 36-45: 进一步优化、细节调整                       [~5000 tokens]
轮次 46-50: Agent开始出现"遗忘"——                      
            - 忘记了最初的系统指令
            - 重复执行已完成的操作
            - 回答质量从精确变为模糊
轮次 50+:  上下文超出模型token限制 (如8000 tokens)
            早期指令被截断，Agent行为完全偏离
```

### 后果

1. **遗忘早期指令**: "请使用正式语气"在30轮后失效
2. **重复工作**: Agent忘记已经调用过某个工具，再次调用
3. **用户困惑**: Agent给出前后矛盾的答案
4. **资源浪费**: 重复的工具调用消耗不必要的API费用

---

## 4. 解决方案 (Solutions)

### 方案1: Token计数 + 硬上限截断

最直接的方法：设定token上限，超过时截断最旧消息。

```python
class HardLimitMemory:
    """带硬上限的短期记忆"""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.messages = deque()
        self.current_tokens = 0
    
    def add(self, message: dict) -> None:
        msg_tokens = self._estimate_tokens(str(message.get("content", "")))
        
        # 超过上限？截断最旧消息
        while self.current_tokens + msg_tokens > self.max_tokens and self.messages:
            old_msg = self.messages.popleft()
            self.current_tokens -= self._estimate_tokens(
                str(old_msg.get("content", ""))
            )
        
        self.messages.append(message)
        self.current_tokens += msg_tokens
    
    def _estimate_tokens(self, text: str) -> int:
        # 简单估算：中文 ~1.5 token/字，英文 ~0.25 token/字符
        count = 0
        for char in text:
            if '一' <= char <= '鿿':
                count += 1.5
            else:
                count += 0.25
        return max(1, int(count))
```

**优点**: 简单、确定性强、不会OOM  
**缺点**: 可能丢失关键信息，截断位置不智能

---

### 方案2: 摘要压缩

保留最近N条完整消息，旧消息压缩为摘要。

```python
class SummarizationMemory:
    """带摘要压缩的短期记忆"""
    
    def __init__(self, max_tokens=8000, compression_threshold=0.8):
        self.max_tokens = max_tokens
        self.messages = deque()
        self.current_tokens = 0
        self.compression_threshold = compression_threshold
        self.compress_count = 0
    
    def add(self, message):
        # ... token估算和添加逻辑 ...
        
        # 达到80%阈值时触发压缩
        if self.current_tokens > self.max_tokens * self.compression_threshold:
            self._compress()
    
    def _compress(self):
        """压缩：保留最近50%消息，摘要旧消息"""
        total = len(self.messages)
        keep = max(5, total // 2)
        
        old_messages = list(self.messages)[:total - keep]
        recent = list(self.messages)[total - keep:]
        
        # 生成摘要（生产环境用LLM）
        summary = self._generate_summary(old_messages)
        
        # 重建
        self.messages.clear()
        self.current_tokens = 0
        
        # 先放摘要
        self.add({
            "role": "system",
            "content": f"[对话摘要 v{self.compress_count}] {summary}"
        })
        
        # 再放最近消息
        for msg in recent:
            self.add(msg)
    
    def _generate_summary(self, messages):
        """简单启发式摘要"""
        # 统计消息数量、角色分布、关键词
        roles = {}
        keywords = []
        for msg in messages:
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
        
        parts = [f"共{len(messages)}条历史消息"]
        for role, count in roles.items():
            parts.append(f"{role}: {count}条")
        
        return " | ".join(parts)
```

**优点**: 保留语义信息，不会完全丢失上下文  
**缺点**: 摘要可能丢失细节，需要额外LLM调用

---

### 方案3: 基于重要性的淘汰

为每条记忆评分，淘汰低分记忆，保留高分记忆。

```python
class ImportanceEvictionMemory:
    """基于重要性评分的淘汰策略"""
    
    def __init__(self, max_tokens=8000):
        self.max_tokens = max_tokens
        self.messages = OrderedDict()
        self.current_tokens = 0
    
    def add(self, message_id, message, importance=1.0):
        """importance: 0.0-10.0, 越高越重要"""
        # 系统消息默认高重要性
        if message.get("role") == "system":
            importance = max(importance, 8.0)
        
        # 包含关键信息的消息提高重要性
        content = str(message.get("content", ""))
        if any(kw in content for kw in ["preference", "requirement", "rule"]):
            importance = max(importance, 7.0)
        
        msg_tokens = self._estimate_tokens(content)
        
        while self.current_tokens + msg_tokens > self.max_tokens:
            # 找到最低重要性条目并淘汰
            if not self.messages:
                break
            worst_key = min(
                self.messages.keys(),
                key=lambda k: self.messages[k]["importance"]
            )
            removed = self.messages.pop(worst_key)
            self.current_tokens -= removed["tokens"]
        
        self.messages[message_id] = {
            "message": message,
            "importance": importance,
            "tokens": msg_tokens,
            "timestamp": time.time(),
        }
        self.current_tokens += msg_tokens
```

**优点**: 智能保留重要信息，丢弃无关内容  
**缺点**: 需要准确的重要性判定，简单启发式可能不准确

---

### 方案4: 分层记忆架构 (推荐)

结合以上方案的优点，构建三层记忆体系。

```
┌──────────────────────────────────────────────────┐
│              分层记忆架构 (Hierarchical Memory)     │
├──────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────────────────────┐                  │
│  │   短期记忆 (Short-Term)       │                  │
│  │   - 最近N轮完整对话           │                  │
│  │   - 滑动窗口: 最近20条消息    │                  │
│  │   - 容量: ~4000 tokens       │                  │
│  └─────────────────────────────┘                  │
│              │ 压缩（摘要）                         │
│              ▼                                    │
│  ┌─────────────────────────────┐                  │
│  │   工作记忆 (Working)          │                  │
│  │   - 压缩的早期对话摘要        │                  │
│  │   - 当前任务状态              │                  │
│  │   - 容量: ~2000 tokens       │                  │
│  └─────────────────────────────┘                  │
│              │ 提取关键信息                         │
│              ▼                                    │
│  ┌─────────────────────────────┐                  │
│  │   长期记忆 (Long-Term)        │                  │
│  │   - 用户偏好、关键事实        │                  │
│  │   - 语义检索获取               │                  │
│  │   - 无token限制（向量存储）    │                  │
│  └─────────────────────────────┘                  │
│                                                   │
└──────────────────────────────────────────────────┘
```

完整实现示例：

```python
class HierarchicalMemory:
    """分层记忆系统 - 方案4完整实现"""
    
    def __init__(self):
        self.short_term = ShortTermMemory(max_tokens=4000)
        self.working = WorkingMemory(max_tokens=2000)
        self.long_term = LongTermMemory()  # 向量存储，无token限制
    
    def add_message(self, role, content):
        """添加新消息到记忆系统"""
        msg = {"role": role, "content": content, "timestamp": time.time()}
        
        # 1. 添加到短期记忆
        self.short_term.add(msg)
        
        # 2. 检查短期记忆压缩状态
        stats = self.short_term.stats()
        if stats["usage_percent"] > 80:
            # 触发压缩：旧消息摘要移入工作记忆
            compressed = self.short_term.compress()
            if compressed:
                self.working.add(compressed)
        
        # 3. 提取关键信息存入长期记忆
        if self._is_important(content):
            self.long_term.add(
                content=f"{role}: {content}",
                metadata={"timestamp": time.time(), "role": role}
            )
    
    def get_context(self):
        """组装完整上下文"""
        context = []
        
        # 从长期记忆检索相关知识
        if self.long_term.memories:
            recent_topic = self._get_recent_topic()
            relevant = self.long_term.retrieve(recent_topic, top_k=3)
            if relevant:
                context.append({
                    "role": "system",
                    "content": f"相关知识: {relevant}"
                })
        
        # 添加工作记忆（压缩的历史摘要）
        context.extend(self.working.get_all())
        
        # 添加短期记忆（最近对话）
        context.extend(self.short_term.get_context())
        
        return context
    
    def _is_important(self, content):
        """判断内容是否值得存入长期记忆"""
        important_keywords = [
            "记住", "偏好", "要求", "规则", "重要",
            "联系方式", "地址", "名字", "习惯",
        ]
        return any(kw in content for kw in important_keywords)
```

---

## 5. 检查清单 (Checklist)

在部署Agent到生产环境前，请确认：

- [ ] **Token预算**: 是否设置了合理的 `max_tokens` 限制？（建议4000-8000）
- [ ] **压缩阈值**: 是否在80%容量时触发压缩？（而非100%才处理）
- [ ] **系统消息保护**: 关键的系统指令是否被标记为高重要性，防止被淘汰？
- [ ] **压缩后验证**: 压缩后的上下文是否仍然包含足够的信息完成任务？
- [ ] **监控告警**: 是否监控了上下文大小、压缩次数、OOM事件？
- [ ] **分层设计**: 是否分离了短期记忆、工作记忆和长期记忆？

---

## 6. 前后对比 (Before/After)

### Before: 无记忆管理的Agent

```python
class AgentBefore:
    def __init__(self):
        self.history = []  # 永远增长，永不清理
    
    def step(self, observation):
        self.history.append(observation)
        # 把所有历史都发给模型 --- 迟早爆炸
        return llm(self.history)  # 💥 第50轮后OOM
```

### After: 带分层记忆的Agent

```python
class AgentAfter:
    def __init__(self):
        self.memory = HierarchicalMemory()
    
    def step(self, observation):
        self.memory.add_message("user", observation)
        
        # 获取结构化上下文（自动控制大小）
        context = self.memory.get_context()
        
        # 上下文始终在预算内
        response = llm(context)
        
        self.memory.add_message("assistant", response)
        return response
    
    def get_stats(self):
        return {
            "short_term": self.memory.short_term.stats(),
            "working": self.memory.working.stats(),
            "long_term": self.memory.long_term.stats(),
        }
```

---

## 关键要点

1. **永远不要信任无限增长的列表** -- 每个数据结构都需要容量上限
2. **80%是魔法数字** -- 在80%容量时压缩，留20%缓冲
3. **分层是关键** -- 近期详细 + 中期摘要 + 长期事实
4. **监控必不可少** -- 没有可见性就无法发现问题
5. **压缩不是完美的** -- 总有信息损失，需要在简洁性和完整性之间权衡
