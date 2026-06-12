# Phase 05 关卡练习：LangChain LCEL 生产级 RAG 链

> **关卡目标**：通过 8 个递进式练习，掌握 LCEL 核心概念、RAG 链构建、检索器定制、工具调用、流式输出、查询路由等生产级技能。

---

## 练习 1：使用 RunnableParallel 同时获取天气和新闻，格式化输出结果

### 目标
使用 `RunnableParallel` 构建并行执行链，同时获取天气信息和新闻资讯，并将两个结果合并为格式化的自然语言输出。

### 要求
1. 编写两个模拟异步函数：
   - `fetch_weather(city: str)`：模拟天气 API，返回 dict 格式 `{"city": str, "temperature": int, "condition": str, "humidity": int}`
   - `fetch_news(topic: str)`：模拟新闻 API，返回 dict 格式 `{"topic": str, "articles": list[dict], "total_count": int}`
2. 使用 `RunnableLambda` 将两个函数包装为 Runnable
3. 使用 `RunnableParallel` 同时执行两个分支，键名分别为 `"weather"` 和 `"news"`
4. 在并行分支后连接一个格式化链，将两个结果合并为一段通顺的中文文本
5. 添加随机延迟（`time.sleep(random.uniform(0.5, 1.5))`）模拟网络请求，验证并行执行确实比串行快
6. 计时对比串行 vs 并行的总耗时

### 输入 / 输出示例

| 项目 | 示例 |
|------|------|
| **输入** | `{"city": "北京", "topic": "人工智能"}` |
| **输出（天气部分）** | `{"city": "北京", "temperature": 25, "condition": "晴", "humidity": 45}` |
| **输出（新闻部分）** | `{"topic": "人工智能", "articles": [...], "total_count": 3}` |
| **最终格式化输出** | `"北京今日天气：晴，温度 25°C，湿度 45%。关于'人工智能'的最新资讯：共找到 3 条相关新闻。头条：OpenAI 发布 GPT-5 模型..."` |
| **性能输出** | `串行耗时: 2.34s | 并行耗时: 1.28s | 加速比: 1.83x` |

### 关键知识点
- `RunnableParallel` 自动并发执行各分支
- `RunnableLambda` 将普通函数转为 Runnable
- 管道操作符 `|` 连接各步骤
- `RunnablePassthrough.assign()` 用于在字典中追加新字段

### 预计用时
2-3 小时

---

## 练习 2：构建带源引用的 LCEL RAG 链，在回答中标注信息来源

### 目标
构建一个完整的 LCEL RAG 链，不仅生成答案，还能在答案中标注每个事实陈述的来源（文档名、页码或段落编号）。

### 要求
1. 准备至少 8 篇模拟文档，内容涵盖不同领域的知识（科技、历史、地理、生物等）
2. 每篇文档的元数据必须包含：`source`（来源名称）、`page`（页码/段落号）、`author`（作者）
3. 使用 `Chroma` 向量存储（in-memory 模式）存储文档
4. 构建包含以下步骤的 LCEL RAG 链：
   - **检索步骤**：从向量库中检索 top-4 相关文档
   - **格式化步骤**：将检索到的文档格式化为上下文，**在上下文中明确标注来源信息**（如 `[来源: XXX, 页码: Y]`）
   - **提示词步骤**：设计 prompt，要求模型在回答中引用来源，格式为 `【来源: XXX, 页码: Y】`
   - **生成步骤**：调用 LLM 生成带来源引用的回答
5. 测试 3 个不同领域的查询，验证来源标注是否正确
6. 额外：实现来源验证函数，检查回答中引用的来源是否确实存在于检索结果中

### 输入 / 输出示例

| 项目 | 示例 |
|------|------|
| **输入查询** | `"什么是光合作用？"` |
| **检索结果** | `[Document(metadata={"source": "生物学基础", "page": 42, "author": "李教授"}, ...), ...]` |
| **格式化上下文** | `"[来源: 生物学基础, 页码: 42, 作者: 李教授] 光合作用是植物利用光能...\n\n[来源: 植物学入门, 页码: 108, 作者: 王老师] 光合作用分为光反应和暗反应..."` |
| **最终输出** | `"光合作用是植物利用光能将二氧化碳和水转化为有机物并释放氧气的过程。【来源: 生物学基础, 页码: 42】\n\n光合作用分为光反应和暗反应两个阶段。【来源: 植物学入门, 页码: 108】"` |
| **来源验证** | `验证通过: 2/2 个来源引用均存在于检索结果中` |

### 关键知识点
- 文档元数据的传递与保留
- 上下文格式化函数中的来源标注
- Prompt 工程中引导模型引用来源
- `RunnablePassthrough.assign()` 保留原始检索结果用于来源验证

### 预计用时
3-4 小时

---

## 练习 3：继承 BaseRetriever 实现自定义检索器（基于关键词匹配）

### 目标
继承 `langchain_core.retrievers.BaseRetriever` 实现一个基于关键词匹配的自定义检索器，理解检索器的内部接口和评分机制。

### 要求
1. 继承 `BaseRetriever` 实现 `KeywordRetriever` 类，必须实现以下方法：
   - `_get_relevant_documents(query: str, *, run_manager=None) -> List[Document]`
2. 关键词匹配评分策略（至少两种）：
   - **简单关键词匹配**：统计查询词在文档中出现的次数作为分数
   - **TF-IDF 加权匹配**：使用 `sklearn.feature_extraction.text.TfidfVectorizer` 计算相似度
3. 支持以下参数：
   - `k: int`（返回多少个文档，默认 4）
   - `match_threshold: float`（最低匹配分数阈值，低于此分数的文档被过滤）
   - `strategy: str`（"simple" 或 "tfidf"，选择匹配策略）
4. 实现中文分词（使用 `jieba` 分词库），确保中文查询能正确匹配
5. 准备至少 15 篇不同类型的中文文档
6. 与标准的向量检索器（Chroma + OpenAIEmbeddings）对比：
   - 使用相同的 5 个查询，对比两种检索器的 top-3 结果
   - 计算检索结果的重叠率（Jaccard 相似度）
   - 分析关键词检索在哪些场景优于/劣于向量检索
7. 测试边界情况：空查询、仅停用词的查询、超长查询

### 输入 / 输出示例

| 项目 | 示例 |
|------|------|
| **输入查询** | `"人工智能在医疗领域的应用"` |
| **关键词检索结果（simple）** | `[Document("AI医疗影像诊断系统...", score=3), Document("机器学习辅助药物研发...", score=2), ...]` |
| **关键词检索结果（tfidf）** | `[Document("AI医疗影像诊断系统...", score=0.85), Document("深度学习在病理分析...", score=0.72), ...]` |
| **向量检索结果** | `[Document("人工智能医疗应用概述...", score=0.92), Document("计算机辅助诊断技术...", score=0.88), ...]` |
| **重叠率** | `Jaccard相似度 = 0.50 (3个结果中有2个重叠)` |
| **边界测试** | 空查询 → 返回空列表 + 日志警告；仅停用词 → 返回空列表 |

### 关键知识点
- `BaseRetriever` 抽象类的接口规范
- 关键词匹配 vs 语义匹配的优劣场景
- 中文文本预处理（分词、去停用词）
- TF-IDF 向量化与余弦相似度计算

### 预计用时
4-5 小时

---

## 练习 4：构建 MultiQueryRetriever 并与单查询检索对比结果数量和相关性

### 目标
实现 MultiQueryRetriever（多查询检索器），通过 LLM 从原始查询生成多个不同角度的查询变体，提高检索召回率。与单查询检索进行系统对比分析。

### 要求
1. 准备一个包含 15+ 篇文档的测试集，文档主题多样但有一定重叠
2. 准备 5 个"模糊查询"（查询意图不明确、表述简略的查询，如"那个学习方法"、"最新的那个技术"）
3. 实现 MultiQueryRetriever（若无法使用真实 LLM，可手动编写查询改写函数作为替代）：
   - 从原始查询生成 3-5 个不同角度的查询变体
   - 每个变体独立检索
   - 合并所有结果并去重（按文档 ID）
   - 按最高分排序
4. 对比指标（对每个查询分别计算）：
   - **结果数量**：单查询 vs 多查询各返回多少不重复文档
   - **新增文档数**：多查询新增了多少单查询未检索到的文档
   - **重叠文档数**：两种方法共同检索到的文档数
   - **召回率提升**：以合并结果池为全集，计算各自的 recall
   - **精确率变化**：多查询是否引入了不相关文档
5. 汇总对比表格，分析多查询检索的适用场景和成本考量

### 输入 / 输出示例

| 项目 | 示例 |
|------|------|
| **输入查询** | `"那个很火的学习方法"` |
| **LLM 生成变体** | `["费曼学习法是什么", "高效学习方法有哪些", "最流行的学习技巧", "如何快速掌握新知识"]` |
| **单查询 top-5** | `[Doc1(费曼技巧), Doc2(学习方法论), Doc3(记忆术), Doc4(速读技巧), Doc5(番茄工作法)]` |
| **多查询合并 top-5** | `[Doc1(费曼技巧), Doc2(学习方法论), Doc6(间隔重复), Doc3(记忆术), Doc7(主动回忆)]` |
| **对比数据** | `单查询5个 | 多查询5个 | 重叠3个 | 新增2个(间隔重复、主动回忆)` |
| **召回率** | `单查询: 3/7=43% | 多查询: 5/7=71% | 召回率提升: +28%` |

### 关键知识点
- MultiQueryRetriever 的工作原理
- 查询改写对召回率的影响
- 合并去重策略（按 ID 去重、按分数排序）
- 检索结果评估指标的实践应用
- 多查询的成本分析（LLM 调用次数 × 检索次数）

### 预计用时
3-4 小时

---

## 练习 5：使用 @tool 装饰器创建工具，绑定到模型，展示错误处理

### 目标
使用 `@tool` 装饰器和 `bind_tools()` 方法创建工具并绑定到 LLM，重点展示工具调用中的错误处理机制。

### 要求
1. 使用 `@tool` 装饰器创建至少 4 个工具：
   - `calculator(expression: str)`：数学表达式计算，安全限制：
     - 只允许数字、四则运算符、括号、小数点
     - 拒绝任何函数调用（如 `__import__`、`eval` 嵌套等）
     - 除零错误 → 抛出 `ToolException("除数不能为零")`
     - 非法表达式 → 抛出 `ToolException(f"非法的数学表达式: {expression}")`
   - `text_translator(text: str, target_language: str)`：文本翻译（模拟）：
     - 支持 "中文" 和 "英文" 之间的互译
     - 不支持的语言 → 抛出 `ToolException(f"不支持的目标语言: {target_language}")`
     - 空文本 → 抛出 `ToolException("待翻译文本不能为空")`
   - `code_runner(code: str)`：模拟代码执行（安全沙箱）：
     - 只允许包含 `print()` 的简单语句
     - 检测到 `import`、`__`、`os`、`subprocess`、`eval`、`exec`、`open` 等危险关键词 → 抛出 `ToolException("检测到不安全的代码操作")`
     - 有效代码 → 返回模拟执行结果
   - `knowledge_search(query: str, top_k: int = 3)`：知识库搜索（模拟）：
     - 查询为空 → 抛出 `ToolException("搜索查询不能为空")`
     - 正常返回模拟搜索结果
     - top_k 非法（≤0 或 >10）→ 抛出 `ToolException(f"top_k 必须在 1-10 之间，当前值: {top_k}")`
2. 使用 `llm.bind_tools([calculator, text_translator, code_runner, knowledge_search])` 绑定工具
3. 展示以下测试场景及其输出：
   - **正常场景**：`"帮我计算 (15 + 25) * 3 / 2 的结果"` → 正确返回 60.0
   - **除零错误**：`"计算 100 除以 0"` → ToolException → 优雅的错误消息
   - **不安全代码**：`"执行代码: import os; os.system('ls')"` → ToolException → 安全警告
   - **不支持的语言**：`"把'你好'翻译成法语"` → ToolException → 提示支持的语言列表
4. 展示两种错误处理模式：
   - `handle_tool_errors=True`：捕获错误并返回格式化错误消息
   - `handle_tool_errors=False`：让 ToolException 原样抛出

### 输入 / 输出示例

| 场景 | 输入 | 输出 |
|------|------|------|
| **正常计算** | `计算 (15+25)*3/2` | `结果: 60.0` |
| **除零错误** | `计算 100/0` | `错误: 除数不能为零。请输入有效的数学表达式。` |
| **不安全代码** | `执行 import os; os.system('ls')` | `错误: 检测到不安全的代码操作: 'import'。仅允许 print() 语句。` |
| **不支持的语言** | `翻译'你好'到法语` | `错误: 不支持的目标语言 '法语'。当前支持: 中文, 英文` |
| **绑定工具** | `llm.bind_tools([...])` | LLM 响应中的 `tool_calls` 字段包含工具调用信息 |

### 关键知识点
- `@tool` 装饰器的用法和参数
- `ToolException` 异常处理机制
- `bind_tools()` 将工具绑定到模型
- `handle_tool_errors` 参数控制错误行为
- 安全沙箱的基本原则

### 预计用时
3-4 小时

---

## 练习 6：构建流式 RAG 链，使用自定义回调跟踪 token 使用和成本

### 目标
构建一个支持流式输出的完整 RAG 链，同时实现自定义回调处理器跟踪每次 LLM 调用的 token 使用量和成本，最后展示汇总统计。

### 要求
1. 构建标准 RAG 链：检索 → 格式化 → 提示词 → LLM → 输出解析
2. 实现两个自定义回调处理器：

   **TokenUsageCallback(BaseCallbackHandler)**：
   - `on_llm_start()`：记录每次 LLM 调用开始时间
   - `on_llm_end()`：从 `response.llm_output` 中提取 `token_usage`（prompt_tokens, completion_tokens, total_tokens）
   - 维护 `usage_history: list` 记录每次调用的详细信息
   - 提供 `total_tokens` 属性返回累计 token 数
   - 提供 `report()` 方法输出详细统计

   **CostEstimateCallback(BaseCallbackHandler)**：
   - 基于 `token_usage` 和模型定价计算费用
   - 预设价格表（RMB/1M tokens）：
     - gpt-4o: 输入 72元, 输出 108元
     - gpt-4o-mini: 输入 1.1元, 输出 4.4元
     - gpt-4-turbo: 输入 72元, 输出 216元
   - 维护 `total_cost: float`
   - 提供 `report()` 方法输出费用明细

3. 使用 `.stream()` 方法进行流式输出，展示逐 token 打印效果
4. 实现 `StreamingTokenCounter`：在流式模式下，通过 `on_llm_new_token()` 回调统计 completion token 数（因为 `.stream()` 时 `llm_output` 中的 `token_usage` 可能不可用）
5. 调用完成后输出汇总统计：
   - 总 prompt tokens / 总 completion tokens / 总 tokens
   - 估算总费用（分模型输出）
   - 总处理时间（包含检索 + 格式化 + LLM 调用）
   - 平均 tokens/秒

### 输入 / 输出示例

| 项目 | 示例 |
|------|------|
| **输入查询** | `"什么是向量数据库？它和传统数据库有什么区别？"` |
| **流式输出** | `向量数据库是一种专门用于存储和检索高维向量的数据库系统。` （逐字打印） |
| **Token 统计** | `Prompt: 245 tokens | Completion: 156 tokens | Total: 401 tokens` |
| **成本估算** | `模型: gpt-4o-mini | 输入: ¥0.00027 | 输出: ¥0.00069 | 总计: ¥0.00096` |
| **性能统计** | `总耗时: 2.34s | LLM耗时: 1.87s | 速度: 83.4 tokens/s` |

### 关键知识点
- `BaseCallbackHandler` 的生命周期方法
- `on_llm_start` / `on_llm_end` / `on_llm_new_token` 的触发时机
- 流式模式下 token 统计的特殊处理
- `.stream()` vs `.invoke()` 在回调行为上的差异
- 成本估算与模型定价

### 预计用时
3-4 小时

---

## 练习 7：实现 SelfQueryRetriever，定义至少 3 个元数据属性并测试过滤查询

### 目标
实现一个 SelfQueryRetriever，能够从自然语言查询中自动提取语义查询字符串和元数据过滤条件，使用结构化过滤器提高检索精度。

### 要求
1. 准备一个包含丰富元数据的文档集（至少 12 条文档）：
   - 元数据字段定义（至少 4 个）：
     - `category: str`：文档分类（如 "技术", "科学", "历史", "文学", "商业"）
     - `year: int`：发布年份（如 2020-2025）
     - `rating: float`：评分（1.0-5.0）
     - `author: str`：作者姓名
2. 使用 `AttributeInfo` 定义每个元数据字段的描述、类型和取值范围：
   ```python
   from langchain.chains.query_constructor.base import AttributeInfo
   metadata_field_info = [
       AttributeInfo(name="category", description="文档分类", type="string"),
       AttributeInfo(name="year", description="发布年份", type="integer"),
       AttributeInfo(name="rating", description="评分(1-5)", type="float"),
       AttributeInfo(name="author", description="作者姓名", type="string"),
   ]
   ```
3. 实现 SelfQueryRetriever（若无法使用真实 LLM 进行查询解析，需手动实现一个查询解析器）：
   - 解析自然语言查询，提取两部分：
     - **语义查询字符串**（用于向量相似度检索）
     - **结构化过滤条件**（用于元数据过滤）
4. 测试以下自然语言查询（至少 5 个），验证过滤器是否正确提取：

   | 查询 | 期望提取的 filter |
   |------|------------------|
   | `"2023年评分高于4星的技术类文章"` | `category="技术" AND year=2023 AND rating>=4.0` |
   | `"张三写的2022年以后的文档"` | `author="张三" AND year>=2022` |
   | `"评分最高的3篇文章"` | `(无filter，仅语义查询，top_k=3)` |
   | `"2024年关于人工智能的技术文档"` | `category="技术" AND year=2024`（语义: "人工智能"） |
   | `"李四或王五写的评分低于3分的文章"` | `author IN ["李四","王五"] AND rating<3.0` |

5. 对每个查询输出：
   - 提取出的语义查询字符串
   - 提取出的结构化过滤条件（dict 格式）
   - 过滤后的检索结果（文档列表 + 元数据）
   - 最终检索到的文档数量
6. 对比：分别展示不过滤和带过滤的检索结果差异

### 输入 / 输出示例

| 项目 | 示例 |
|------|------|
| **原始查询** | `"2023年评分高于4星的技术类文章"` |
| **提取的语义查询** | `"技术类文章"` |
| **提取的过滤条件** | `{"op": "and", "filters": [{"key": "category", "op": "eq", "value": "技术"}, {"key": "year", "op": "eq", "value": 2023}, {"key": "rating", "op": "gte", "value": 4.0}]}` |
| **不过滤结果数** | `12 篇文档` |
| **过滤后结果数** | `3 篇文档`（都满足 category=技术, year=2023, rating>=4.0） |

### 关键知识点
- SelfQueryRetriever 的两阶段查询解析
- `AttributeInfo` 元数据定义
- 结构化过滤条件的表达（Comparison, Operation, AND/OR 组合）
- 向量检索与元数据过滤的结合

### 预计用时
4-5 小时

---

## 练习 8：使用 RunnableBranch 构建路由链，根据查询类型路由到不同检索器

### 目标
使用 `RunnableBranch` 构建一个智能路由系统，根据用户查询的语义类型自动选择最合适的检索器和处理链，实现多路分发。

### 要求
1. 准备 3 个不同类型的文档集和对应的检索器：
   - **技术文档检索器**：包含编程、框架、API 等技术内容（10+ 篇）
   - **新闻资讯检索器**：包含时事新闻、事件报道等内容（10+ 篇）
   - **通用知识检索器**：包含百科知识、常识等内容（10+ 篇）
2. 使用 `RunnableBranch` 构建路由逻辑，根据查询类型选择处理分支：
   - **技术查询**：包含代码/编程/框架/API 等关键词 → 技术文档检索器 + 技术风格 prompt
   - **新闻查询**：包含时间/事件/新闻等关键词 → 新闻检索器 + 新闻风格 prompt
   - **通用查询**：其他所有情况 → 通用知识检索器 + 通用 prompt
3. 每个分支使用不同的 prompt 模板（体现区分度）：
   - 技术分支 prompt：强调代码示例和技术细节
   - 新闻分支 prompt：强调时间线和事件经过
   - 通用分支 prompt：强调简明扼要的解释
4. 路由条件使用 `RunnableLambda` 实现分类函数，返回分支标识（如 `"tech"`, `"news"`, `"general"`）
5. 扩展要求（进阶）：
   - 实现"置信度不足时合并多个检索器结果"的逻辑
   - 当分类函数置信度低于阈值时，同时查询两个相关检索器并合并结果
6. 测试至少 8 个不同类型的查询，展示：
   - 每个查询的分类结果
   - 路由到的分支
   - 返回的答案（体现不同 prompt 风格）
   - 置信度得分（进阶部分）

### 输入 / 输出示例

| 输入查询 | 分类结果 | 路由分支 | 回答风格 |
|----------|---------|----------|---------|
| `"Python 中如何使用 asyncio 进行异步编程？"` | tech | 技术检索器 | 含代码示例 + 技术解释 |
| `"2024年诺贝尔奖获得者是谁？"` | news | 新闻检索器 | 含时间线 + 事件背景 |
| `"什么是光合作用？"` | general | 通用检索器 | 简明扼要解释 |
| `"Rust 语言的异步运行时 Tokio 最新版本有什么新特性？"` | tech (置信度 0.6) → tech+news | 合并检索 | 合并两个检索器的结果 |
| `"最近 AI 领域有什么重大突破？"` | news | 新闻检索器 | 含时间戳 + 事件描述 |
| `"介绍一下机器学习的基本概念？"` | general | 通用检索器 | 入门级解释 |

### 关键知识点
- `RunnableBranch` 的条件路由机制
- `(condition, runnable)` 元组的匹配优先级
- 默认分支（`RunnableBranch` 的最后一个参数）
- 查询分类的特征工程
- 多检索器结果合并策略

### 预计用时
4-5 小时

---

## 提交清单

完成本关卡后，请提交以下文件：

```
phase-05-checkpoint/
├── exercise_01_parallel.py              # 练习 1: RunnableParallel 并行
├── exercise_02_sourced_rag.py           # 练习 2: 带源引用的 RAG 链
├── exercise_03_keyword_retriever.py     # 练习 3: 自定义关键词检索器
├── exercise_04_multiquery_compare.py    # 练习 4: MultiQueryRetriever 对比
├── exercise_05_tool_error_handling.py   # 练习 5: 工具创建与错误处理
├── exercise_06_streaming_tokens.py      # 练习 6: 流式 RAG + Token 追踪
├── exercise_07_self_query.py            # 练习 7: SelfQueryRetriever
├── exercise_08_routing.py              # 练习 8: RunnableBranch 路由
└── README.md                            # 简要说明运行方法和依赖
```

所有代码文件必须可直接 `python <filename>` 运行。依赖在各自文件顶部以注释列出。

---

*Phase 05 关卡 · LangChain LCEL 生产级 RAG 链 · v1.0*
