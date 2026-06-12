# Phase 06 - Agent基础 练习题

> 完成以下8个练习，巩固Agent基础知识

---

## 练习1: 从零实现ReAct Agent

**难度**: ★★★★ | **预计时间**: 60分钟

从零实现一个ReAct Agent，支持至少3个工具，完成一个多步骤推理任务。

### 要求

1. 实现ReAct循环: Thought → Action → Observation → ... → Final Answer
2. 支持至少3个不同类别的工具（如：搜索、计算、文件操作）
3. Agent能根据Observation自动决定下一步Action
4. 包含最大步数限制（防止无限循环）
5. 完整的多步骤推理任务演示（至少3步）

### 提示

```python
class ReActAgent:
    def __init__(self, tools, llm, max_steps=10):
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps
    
    def run(self, task: str) -> str:
        # 实现ReAct循环
        pass
```

### 验收标准

- Agent能正确解析 Thought/Action/Observation 格式
- 多步骤任务中每一步都有合理的推理
- 达到最大步数时优雅退出而非崩溃
- 有完整的演示代码

---

## 练习2: 构建ToolRegistry

**难度**: ★★★ | **预计时间**: 45分钟

构建一个完整的ToolRegistry，注册5个工具，实现超时和错误分类。

### 要求

1. 实现 ToolRegistry 类，支持 register/get/unregister/search 操作
2. 注册5个不同功能的工具（天气、计算、搜索、翻译、文件读写）
3. 实现跨平台超时控制（使用 threading）
4. 实现错误分类：SUCCESS / TIMEOUT / RETRYABLE_ERROR / FATAL_ERROR / INVALID_ARGS
5. 包含依赖检查：注册时验证工具的依赖是否满足
6. 完整的演示代码

### 提示

```python
class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._history = []
    
    def register(self, tool_def): ...
    def execute(self, name, **kwargs): ...
    def _execute_with_timeout(self, func, timeout, **kwargs): ...
    def _categorize_error(self, error): ...
```

### 验收标准

- 5个工具全部注册成功且有实际功能
- 超时工具能正确返回TIMEOUT状态
- 错误工具能正确分类为RETRYABLE或FATAL
- 依赖检查能正确阻止/允许注册

---

## 练习3: 为Agent添加ShortTermMemory

**难度**: ★★★ | **预计时间**: 45分钟

实现ShortTermMemory类，提供滑动窗口和压缩功能，并集成到Agent中。

### 要求

1. 实现滑动窗口（deque），支持 max_tokens 限制
2. 实现Token估算（中文字符加权）
3. 80%容量时触发压缩
4. 压缩策略：保留最近50%消息，前50%摘要为系统消息
5. 提供 stats() 方法返回统计信息
6. 将ShortTermMemory集成到练习1的Agent中

### 提示

```python
class ShortTermMemory:
    def __init__(self, max_tokens=8000, compression_threshold=0.8):
        self.messages = deque()
        self.current_tokens = 0
        ...
    
    def add(self, message): ...
    def _compress(self): ...
    def _summarize(self, messages): ...
    def get_context(self): ...
    def stats(self): ...
```

### 验收标准

- Token计数接近实际值（不要求精确）
- 超过阈值时自动触发压缩
- 压缩后消息数减少但保留关键信息
- 统计数据准确

---

## 练习4: 实现EpisodicMemory

**难度**: ★★★ | **预计时间**: 45分钟

实现EpisodicMemory，记录任务轨迹，查找相似历史情景。

### 要求

1. 实现 Episode 数据类（ID、任务、轨迹、结果、成功标志、经验教训、标签）
2. 实现 record() 方法记录完整任务过程
3. 实现 find_similar() 基于嵌入相似度查找
4. 实现 get_lessons() 提取相关经验
5. 使用字符n-gram + 哈希生成嵌入（不依赖外部模型）
6. 支持 max_episodes 限制和自动淘汰

### 提示

```python
@dataclass
class Episode:
    episode_id: str
    task: str
    trajectory: list
    result: Any
    success: bool
    timestamp: float
    lessons_learned: str
    tags: list
```

### 验收标准

- 能正确记录和检索情景
- 相似度计算合理（相同任务得分高，无关任务得分低）
- 超过 max_episodes 时旧情景被淘汰
- 经验教训能正确关联到新任务

---

## 练习5: 使用LangChain create_react_agent构建Agent

**难度**: ★★★ | **预计时间**: 60分钟

使用LangChain的 create_react_agent 构建Agent，处理解析错误。

### 要求

1. 使用 `from langgraph.prebuilt import create_react_agent` 创建Agent
2. 传递 ChatOpenAI 模型 + tools列表
3. 配置 AgentExecutor 参数：
   - max_iterations: 防止无限循环
   - max_execution_time: 时间限制
   - return_intermediate_steps: 调试用
   - handle_parsing_errors: True/函数
4. 创建3个LangChain风格的 @tool 装饰器工具
5. 实现解析错误的自定义处理函数
6. 完整的演示

### 提示

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

@tool
def my_tool(param: str) -> str:
    """工具描述"""
    ...

model = ChatOpenAI(model="gpt-4")
agent = create_react_agent(model, [my_tool, ...])
```

### 验收标准

- Agent能正确调用工具并完成多步骤任务
- 解析错误被正确处理（不会中断执行）
- max_iterations 限制生效
- 中间步骤可查看

---

## 练习6: 实现MemoryManager的LRU+重要性评分淘汰

**难度**: ★★★★ | **预计时间**: 60分钟

实现完整的MemoryManager，支持LRU + 重要性评分淘汰策略。

### 要求

1. 支持多种记忆类型：short_term, long_term, episodic, working
2. 每种类型独立的 max_entries 限制
3. 实现淘汰评分: score = recency * 0.6 + frequency * 0.4
4. recency使用指数衰减：R(t) = (1/2)^(t/30days)
5. 实现持久化：pickle序列化和JSON导出
6. 实现 query() 方法支持多条件过滤
7. 完整的统计信息

### 提示

```python
class EvictionPolicy:
    HALF_LIFE_DAYS = 30.0
    
    @classmethod
    def calculate_recency(cls, entry, current_time):
        days = (current_time - entry.last_accessed) / 86400
        return math.pow(0.5, days / cls.HALF_LIFE_DAYS)
    
    @classmethod
    def calculate_score(cls, entry, max_freq, current_time):
        recency = cls.calculate_recency(entry, current_time)
        freq = entry.access_count / max(1, max_freq)
        return recency * 0.6 + freq * 0.4
```

### 验收标准

- 超过容量限制时最低分条目被淘汰
- recency衰减计算正确（30天后分数减半）
- 持久化后能正确恢复
- query() 支持关键词、重要性、时间过滤

---

## 练习7: 为Agent添加工具调用验证

**难度**: ★★★ | **预计时间**: 45分钟

实现完整的工具调用验证系统，防止幻觉工具调用。

### 要求

1. 实现工具名白名单验证
2. 实现模糊匹配（编辑距离/子串匹配）
3. 实现参数名验证（与函数签名对比）
4. 实现参数类型验证（Pydantic或手动检查）
5. 实现清晰的错误信息返回（帮助LLM自我纠正）
6. 实现常见别名映射（如 "calc" → "calculator"）

### 提示

```python
class ToolCallValidator:
    def __init__(self, registry):
        self.registry = registry
        self.alias_map = {
            'calc': 'calculator',
            'compute': 'calculator',
            'get_weather': 'weather',
            ...
        }
    
    def validate(self, tool_name, **kwargs) -> ValidationResult:
        ...
```

### 验收标准

- 不存在的工具名能返回清晰错误和相似建议
- 错误的参数名能被检测并提示正确参数
- 常见别名能自动映射到正确工具
- 错误信息对LLM友好（能帮助自我纠正）

---

## 练习8: 综合Agent设计

**难度**: ★★★★★ | **预计时间**: 90分钟

设计并实现一个完整的Agent，综合运用所有Phase 06知识。

### 要求

1. 实现完整的 ToolRegistry，注册5+个工具
2. 实现 ShortTermMemory（滑动窗口+压缩）
3. 实现 LongTermMemory（向量存储+语义检索）
4. 实现 EpisodicMemory（情景记录+相似查找）
5. 实现 MemoryManager（统一管理+淘汰策略）
6. 实现工具调用验证（防幻觉）
7. 实现 ReAct 循环
8. 至少一个多步骤真实任务演示（如：研究某个主题，查询天气，计算结果，总结文本）

### 提示

```python
class FullAgent:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.tool_validator = ToolCallValidator(self.tool_registry)
        self.memory_manager = MemoryManager()
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.em = EpisodicMemory()
    
    def run(self, task: str) -> str:
        # 1. 查询历史经验
        # 2. 查询相关知识
        # 3. ReAct循环执行
        # 4. 更新记忆
        pass
```

### 验收标准

- 所有组件正确集成
- 多步骤任务能成功完成
- 记忆系统在整个过程中正确更新
- 工具调用经过验证
- 代码结构清晰，有适当的中文注释
