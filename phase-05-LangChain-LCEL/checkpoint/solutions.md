# Phase 05 关卡练习解答：LangChain LCEL 生产级 RAG 链

---

## 练习 1 解答：RunnableParallel 并行获取天气和新闻

```python
"""
练习1: 使用 RunnableParallel 同时获取天气和新闻，格式化输出结果

核心思路:
- 两个模拟函数各自含有随机延迟，模拟网络请求
- RunnableLambda 将普通函数包装为 Runnable
- RunnableParallel 自动并发执行所有分支
- 计时对比验证并行加速效果
"""

import time
import random
from typing import Dict, List
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough


# === 步骤1: 模拟异步 API 函数 ===

def fetch_weather(city: str) -> Dict:
    """
    模拟天气 API 调用。
    
    Args:
        city: 城市名称
    
    Returns:
        dict: 包含城市、温度、天气状况、湿度的字典
    """
    # 模拟网络延迟
    delay = random.uniform(0.5, 1.5)
    time.sleep(delay)
    
    weather_data = {
        "北京": {"temperature": 25, "condition": "晴", "humidity": 45},
        "上海": {"temperature": 28, "condition": "多云", "humidity": 65},
        "广州": {"temperature": 32, "condition": "雷阵雨", "humidity": 80},
        "深圳": {"temperature": 30, "condition": "晴间多云", "humidity": 55},
        "杭州": {"temperature": 26, "condition": "小雨", "humidity": 70},
    }
    
    default = {"temperature": 22, "condition": "晴", "humidity": 50}
    weather = weather_data.get(city, default)
    
    return {
        "city": city,
        "temperature": weather["temperature"],
        "condition": weather["condition"],
        "humidity": weather["humidity"],
        "fetch_time": round(delay, 2),
    }


def fetch_news(topic: str) -> Dict:
    """
    模拟新闻 API 调用。
    
    Args:
        topic: 新闻主题
    
    Returns:
        dict: 包含主题、文章列表、总数的字典
    """
    delay = random.uniform(0.5, 1.5)
    time.sleep(delay)
    
    news_data = {
        "人工智能": {
            "articles": [
                {"title": "OpenAI 发布 GPT-5 模型", "source": "科技日报"},
                {"title": "AI 在医疗诊断领域取得突破", "source": "健康报"},
                {"title": "全球AI投资突破万亿美元", "source": "经济观察"},
            ],
            "total_count": 3,
        },
        "科技": {
            "articles": [
                {"title": "新型芯片性能提升300%", "source": "电子时报"},
                {"title": "量子计算机实现新里程碑", "source": "科学杂志"},
            ],
            "total_count": 2,
        },
        "金融": {
            "articles": [
                {"title": "央行宣布降准0.5个百分点", "source": "金融时报"},
                {"title": "A股市场迎来反弹", "source": "证券日报"},
                {"title": "数字人民币试点扩大", "source": "经济日报"},
                {"title": "外资持续流入中国债市", "source": "国际金融报"},
            ],
            "total_count": 4,
        },
    }
    
    default = {
        "articles": [
            {"title": f"关于'{topic}'的最新动态", "source": "综合新闻"},
            {"title": f"{topic}领域发展前景分析", "source": "行业观察"},
        ],
        "total_count": 2,
    }
    
    news = news_data.get(topic, default)
    return {
        "topic": topic,
        "articles": news["articles"],
        "total_count": news["total_count"],
        "fetch_time": round(delay, 2),
    }


# === 步骤2: 包装为 Runnable ===

def weather_runnable_fn(inputs: Dict) -> Dict:
    """从输入字典提取 city 并获取天气"""
    return fetch_weather(inputs["city"])


def news_runnable_fn(inputs: Dict) -> Dict:
    """从输入字典提取 topic 并获取新闻"""
    return fetch_news(inputs["topic"])


weather_runnable = RunnableLambda(weather_runnable_fn)
news_runnable = RunnableLambda(news_runnable_fn)


# === 步骤3: RunnableParallel 并行执行 ===

parallel_fetch = RunnableParallel(
    weather=weather_runnable,
    news=news_runnable,
)


# === 步骤4: 格式化输出 ===

def format_output(data: Dict) -> str:
    """
    将并行获取的天气和新闻数据格式化为可读文本。
    """
    weather = data.get("weather", {})
    news = data.get("news", {})
    
    city = weather.get("city", "未知")
    temp = weather.get("temperature", "?")
    condition = weather.get("condition", "未知")
    humidity = weather.get("humidity", "?")
    
    topic = news.get("topic", "未知")
    articles = news.get("articles", [])
    total = news.get("total_count", 0)
    headline = articles[0]["title"] if articles else "无相关新闻"
    
    lines = [
        "=" * 60,
        f"           综合信息报告",
        "=" * 60,
        "",
        f"【天气】 {city}今日天气：{condition}，温度 {temp}°C，湿度 {humidity}%。",
        "",
        f"【新闻】 关于'{topic}'的最新资讯：共找到 {total} 条相关新闻。",
        f"        头条：{headline}",
        "",
    ]
    
    # 如果有更多文章，列出
    if len(articles) > 1:
        for i, art in enumerate(articles[1:], 2):
            lines.append(f"         {i}. {art['title']} (来源: {art['source']})")
    
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"数据获取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(lines)


format_runnable = RunnableLambda(format_output)

# 完整链
full_chain = parallel_fetch | format_runnable


# === 步骤5: 串行 vs 并行计时对比 ===

def run_serial(inputs: Dict) -> tuple:
    """串行执行：先天气，后新闻"""
    start = time.time()
    weather_result = weather_runnable.invoke(inputs)
    news_result = news_runnable.invoke(inputs)
    elapsed = time.time() - start
    combined = {"weather": weather_result, "news": news_result}
    return combined, elapsed


def run_parallel(inputs: Dict) -> tuple:
    """并行执行"""
    start = time.time()
    combined = parallel_fetch.invoke(inputs)
    elapsed = time.time() - start
    return combined, elapsed


# === 步骤6: 测试 ===

if __name__ == "__main__":
    test_input = {"city": "北京", "topic": "人工智能"}
    
    print("=" * 60)
    print("RunnableParallel 并行执行测试")
    print("=" * 60)
    
    # 预热（消除首次运行开销影响）
    _ = full_chain.invoke(test_input)
    
    # 多次测量取平均
    n_trials = 3
    serial_times = []
    parallel_times = []
    
    for i in range(n_trials):
        _, st = run_serial(test_input)
        _, pt = run_parallel(test_input)
        serial_times.append(st)
        parallel_times.append(pt)
    
    avg_serial = sum(serial_times) / n_trials
    avg_parallel = sum(parallel_times) / n_trials
    speedup = avg_serial / avg_parallel
    
    print(f"\n性能对比 (平均 {n_trials} 次):")
    print(f"  串行耗时: {avg_serial:.2f}s")
    print(f"  并行耗时: {avg_parallel:.2f}s")
    print(f"  加速比:   {speedup:.2f}x")
    
    print(f"\n输出示例:")
    result = format_runnable.invoke(run_parallel(test_input)[0])
    print(result)
```

---

## 练习 2 解答：带来源引用的 RAG 链

```python
"""
练习2: 构建带源引用的 LCEL RAG 链，在回答中标注信息来源

核心思路:
- 文档元数据(source, page, author)在检索和格式化过程中保留
- 格式化函数在上下文中标注来源
- Prompt 引导 LLM 引用来源
- 额外实现来源验证函数
"""

import tempfile
import re
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma
from operator import itemgetter


# === 步骤1: 准备 8+ 篇跨领域文档 ===

documents = [
    Document(
        page_content="光合作用是绿色植物利用光能，将二氧化碳和水转化为有机物（主要是葡萄糖）"
                     "并释放氧气的过程。该过程主要发生在叶绿体中，包含光反应和暗反应两个阶段。",
        metadata={"source": "生物学基础", "page": 42, "author": "李教授"}
    ),
    Document(
        page_content="光反应发生在类囊体薄膜上，将光能转化为化学能，产生ATP和NADPH。"
                     "暗反应（卡尔文循环）发生在叶绿体基质中，利用ATP和NADPH固定二氧化碳。",
        metadata={"source": "植物学入门", "page": 108, "author": "王老师"}
    ),
    Document(
        page_content="明代长城东起鸭绿江，西至嘉峪关，全长约8851.8公里。"
                     "长城始建于春秋战国时期，明朝是最后一个大规模修筑长城的朝代。",
        metadata={"source": "中国历史纲要", "page": 215, "author": "张教授"}
    ),
    Document(
        page_content="秦始皇统一六国后，将原有的长城连接并扩建。"
                     "汉代进一步向西延伸长城，以保护丝绸之路的安全。",
        metadata={"source": "中华文明史", "page": 89, "author": "刘研究员"}
    ),
    Document(
        page_content="Python是一种解释型、面向对象的高级编程语言，由Guido van Rossum于1991年首次发布。"
                     "Python的设计哲学强调代码的可读性和简洁的语法。",
        metadata={"source": "Python编程指南", "page": 3, "author": "陈工程师"}
    ),
    Document(
        page_content="DNA（脱氧核糖核酸）是生物遗传信息的载体。"
                     "DNA双螺旋结构由沃森和克里克于1953年发现，这是分子生物学史上的里程碑。",
        metadata={"source": "分子生物学", "page": 156, "author": "赵博士"}
    ),
    Document(
        page_content="相对论是爱因斯坦提出的物理学理论，包括狭义相对论（1905年）和广义相对论（1915年）。"
                     "狭义相对论基于两个基本假设：物理定律在所有惯性系中相同，光速在真空中恒定。",
        metadata={"source": "物理学原理", "page": 312, "author": "钱教授"}
    ),
    Document(
        page_content="机器学习是人工智能的一个子领域，通过算法使计算机从数据中学习。"
                     "主要分为监督学习、无监督学习和强化学习三大类。",
        metadata={"source": "AI入门教程", "page": 18, "author": "孙博士"}
    ),
    Document(
        page_content="地中海气候的特点是夏季炎热干燥，冬季温和多雨。"
                     "这种气候主要分布在地中海沿岸、加利福尼亚、智利中部、南非西南部和澳大利亚西南部。",
        metadata={"source": "世界地理", "page": 201, "author": "周教授"}
    ),
    Document(
        page_content="第一次世界大战（1914-1918）的导火索是萨拉热窝事件。"
                     "同盟国（德国、奥匈帝国等）与协约国（英国、法国、俄国等）展开了长达四年的战争。",
        metadata={"source": "世界现代史", "page": 45, "author": "刘研究员"}
    ),
]


# === 步骤2: 创建向量存储 ===

persist_dir = tempfile.mkdtemp(prefix="cited_rag_")
embeddings = FakeEmbeddings(size=384)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_dir,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# === 步骤3: 上下文格式化（标注来源） ===

def format_docs_with_citations(docs: list) -> str:
    """
    将检索到的文档格式化为带来源标注的上下文字符串。
    
    格式:
    [来源: {source}, 页码: {page}, 作者: {author}]
    {content}
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source_line = (
            f"[来源{ i }: {meta.get('source', '未知')}, "
            f"页码: {meta.get('page', '?')}, "
            f"作者: {meta.get('author', '未知')}]"
        )
        parts.append(f"{source_line}\n{doc.page_content}")
    return "\n\n".join(parts)


def extract_source_ids(docs: list) -> set:
    """提取检索结果中的所有 (source, page) 元组用于验证"""
    return {
        (doc.metadata.get("source", ""), str(doc.metadata.get("page", "")))
        for doc in docs
    }


# === 步骤4: 提示词模板 ===

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个严谨的知识助手。请严格基于以下提供的文档内容回答问题。\n\n"
        "重要规则:\n"
        "1. 每个事实陈述后必须标注来源，格式为 【来源: XXX, 页码: Y】\n"
        "2. 来源名称和页码必须与文档中标注的完全一致\n"
        "3. 如果文档中没有相关信息，请明确回答'提供的文档不包含此信息'\n"
        "4. 不要编造或推断文档中没有的内容\n\n"
        "提供的文档:\n"
        "{context}"
    ),
    (
        "human",
        "{question}"
    ),
])


# === 步骤5: RAG 链 ===

rag_chain = (
    {
        "context": itemgetter("question") | retriever | RunnableLambda(format_docs_with_citations),
        "question": itemgetter("question"),
        "raw_docs": itemgetter("question") | retriever,  # 保留原文档用于验证
    }
    | RunnablePassthrough.assign(
        answer=(
            RunnableLambda(
                lambda d: ChatPromptTemplate.from_messages([
                    ("system", d["context"]),
                    ("human", d["question"]),
                ])
            )
            # | llm          # 实际使用时替换为真实 LLM
            # | StrOutputParser()
        )
    )
)


# === 步骤6: 来源验证函数 ===

def verify_citations(answer: str, raw_docs: list) -> Dict:
    """
    验证回答中的来源引用是否确实存在于检索结果中。
    
    Returns:
        dict: {"total": int, "valid": int, "invalid": list, "missing": list}
    """
    # 提取回答中的引用
    citation_pattern = re.compile(r'【来源:\s*([^,，]+)[,，]\s*页码:\s*(\d+)】')
    cited = citation_pattern.findall(answer)
    
    # 提取文档中的来源
    available = extract_source_ids(raw_docs)
    
    valid = []
    invalid = []
    for source, page in cited:
        if (source.strip(), page.strip()) in available:
            valid.append((source, page))
        else:
            invalid.append((source, page))
    
    # 检查哪些文档未被引用
    all_sources = extract_source_ids(raw_docs)
    cited_set = {(s.strip(), p.strip()) for s, p in cited}
    missing = all_sources - cited_set
    
    return {
        "total_citations": len(cited),
        "valid_citations": len(valid),
        "invalid_citations": invalid,
        "cited_sources": valid,
        "uncited_sources": list(missing),
    }


# === 步骤7: 模拟测试 ===

def simulate_rag_with_verification(query: str):
    """
    模拟 RAG 链执行，展示所有中间步骤和验证结果。
    """
    print(f"\n{'='*70}")
    print(f"查询: {query}")
    print(f"{'='*70}")
    
    # 检索
    docs = retriever.invoke(query)
    print(f"\n[检索] 返回 {len(docs)} 个文档:")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.metadata['source']} p.{doc.metadata['page']}")
    
    # 格式化
    formatted = format_docs_with_citations(docs)
    print(f"\n[格式化上下文]:")
    print(formatted[:400] + "...")
    
    # 模拟 LLM 回答
    simulated_answer = (
        "光合作用是绿色植物利用光能将二氧化碳和水转化为有机物并释放氧气的过程。"
        "【来源: 生物学基础, 页码: 42】"
        "该过程包含光反应和暗反应两个阶段。"
        "【来源: 植物学入门, 页码: 108】"
    )
    
    print(f"\n[LLM回答]:\n{simulated_answer}")
    
    # 来源验证
    verification = verify_citations(simulated_answer, docs)
    print(f"\n[来源验证]:")
    print(f"  总引用数: {verification['total_citations']}")
    print(f"  有效引用: {verification['valid_citations']}")
    if verification['invalid_citations']:
        print(f"  无效引用: {verification['invalid_citations']}")
    if verification['uncited_sources']:
        print(f"  未引用源: {verification['uncited_sources']}")
    
    if verification['total_citations'] == verification['valid_citations']:
        print(f"  验证通过: {verification['valid_citations']}/{verification['total_citations']} 个来源引用均存在于检索结果中")


if __name__ == "__main__":
    simulate_rag_with_verification("什么是光合作用？")
    simulate_rag_with_verification("长城的建造历史")
    simulate_rag_with_verification("爱因斯坦的贡献")
```

---

## 练习 3 解答：自定义关键词检索器

```python
"""
练习3: 继承 BaseRetriever 实现基于关键词匹配的自定义检索器

核心思路:
- 实现 KeywordRetriever，支持 simple 和 tfidf 两种策略
- 使用 jieba 进行中文分词
- 与向量检索器对比 Jaccard 相似度
- 边界情况处理
"""

import re
import warnings
from typing import List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_chroma import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile


# 尝试导入 jieba
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("提示: jieba 未安装，将使用简单字符分割。请运行: pip install jieba")


# 中文停用词表
STOPWORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如何", "为什么", "吗", "呢", "吧", "啊", "哦", "嗯",
    "这个", "那个", "可以", "还是", "或者", "但", "因为", "所以", "如果",
    "虽然", "然而", "而且", "并且", "关于", "对于", "以及", "通过", "进行",
    "以下", "以上", "之后", "之前", "其中", "其他", "所有", "一些", "各种",
    ":" , "：" , "。" , "，" , "、" , "；" , "！" , "？", "\n", "\t", " ",
])


def tokenize_chinese(text: str) -> List[str]:
    """中文分词，过滤停用词"""
    if JIEBA_AVAILABLE:
        words = jieba.lcut(text)
    else:
        # 简单字符级分词
        words = list(text)
    
    return [
        w.strip() for w in words
        if w.strip() and w.strip() not in STOPWORDS and len(w.strip()) > 1
    ]


class KeywordRetriever(BaseRetriever):
    """
    基于关键词匹配的自定义检索器。
    
    支持两种匹配策略:
    - "simple": 统计查询词在文档中出现的次数
    - "tfidf": 使用 TF-IDF 向量化 + 余弦相似度
    """
    
    documents: List[Document]
    k: int = 4
    match_threshold: float = 0.0
    strategy: str = "simple"  # "simple" 或 "tfidf"
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        """
        核心检索方法。
        """
        # 边界检查：空查询
        if not query or not query.strip():
            warnings.warn("KeywordRetriever: 查询为空，返回空结果")
            return []
        
        # 检查是否为仅停用词的查询
        query_tokens = tokenize_chinese(query)
        if not query_tokens:
            warnings.warn("KeywordRetriever: 查询仅包含停用词，返回空结果")
            return []
        
        if self.strategy == "tfidf":
            return self._retrieve_tfidf(query, query_tokens)
        else:
            return self._retrieve_simple(query, query_tokens)
    
    def _retrieve_simple(self, query: str, query_tokens: List[str]) -> List[Document]:
        """简单关键词计数匹配"""
        scored = []
        for doc in self.documents:
            doc_tokens = tokenize_chinese(doc.page_content)
            # 统计查询词在文档中的出现次数
            score = sum(
                1 for qt in query_tokens
                for dt in doc_tokens
                if qt in dt
            )
            if score >= self.match_threshold:
                scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc, score in scored[:self.k]:
            doc.metadata["_keyword_score"] = score
            results.append(doc)
        return results
    
    def _retrieve_tfidf(self, query: str, query_tokens: List[str]) -> List[Document]:
        """TF-IDF 向量化匹配"""
        # 准备语料库
        corpus = [doc.page_content for doc in self.documents]
        if not corpus:
            return []
        
        # TF-IDF 向量化
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_chinese,
            token_pattern=None,  # 使用自定义分词器时禁用默认模式
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        except ValueError:
            # 如果所有 tokens 都是停用词
            return []
        
        scored = []
        for i, doc in enumerate(self.documents):
            score = float(similarities[i])
            if score >= self.match_threshold:
                scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc, score in scored[:self.k]:
            doc.metadata["_tfidf_score"] = round(score, 4)
            results.append(doc)
        return results


# === 准备测试文档 ===

test_documents = [
    Document(page_content="人工智能在医疗领域的应用日益广泛，AI影像诊断系统可以辅助医生识别病灶，"
                         "提高诊断准确率。深度学习模型在病理分析中表现出色。", metadata={"id": "doc1"}),
    Document(page_content="机器学习算法在药物研发中发挥重要作用，通过分析大量化合物数据，"
                         "加速新药发现过程。强化学习用于优化药物剂量方案。", metadata={"id": "doc2"}),
    Document(page_content="自然语言处理技术用于分析电子病历，提取关键医疗信息。"
                         "医疗聊天机器人可以为患者提供初步健康咨询。", metadata={"id": "doc3"}),
    Document(page_content="Python是数据科学领域最流行的编程语言，拥有丰富的科学计算库。"
                         "NumPy和Pandas是数据处理的基础工具。", metadata={"id": "doc4"}),
    Document(page_content="深度学习框架TensorFlow和PyTorch的对比研究。"
                         "PyTorch在学术研究领域更受欢迎，TensorFlow在工业部署中应用广泛。", metadata={"id": "doc5"}),
    Document(page_content="中国长城是世界上最长的防御工事，总长度超过两万公里。"
                         "长城不仅是一道城墙，还包括烽火台、关城等军事设施。", metadata={"id": "doc6"}),
    Document(page_content="明朝是中国历史上最后一个由汉族建立的大一统王朝。"
                         "朱元璋建立明朝后定都南京，后朱棣迁都北京并修建紫禁城。", metadata={"id": "doc7"}),
    Document(page_content="光合作用是地球上最重要的生化过程之一。"
                         "植物通过叶绿体中的叶绿素捕获光能，将其转化为化学能储存于有机物中。", metadata={"id": "doc8"}),
    Document(page_content="气候变化导致全球气温升高，极端天气事件频发。"
                         "减少碳排放、发展可再生能源是应对气候变化的关键措施。", metadata={"id": "doc9"}),
    Document(page_content="Python异步编程使用asyncio库实现协程，"
                         "通过事件循环管理并发任务。async和await关键字是异步编程的核心语法。", metadata={"id": "doc10"}),
    Document(page_content="癌症早期诊断对于提高患者生存率至关重要。"
                         "AI辅助诊断系统可以在CT、MRI等影像中检测微小的肿瘤。", metadata={"id": "doc11"}),
    Document(page_content="唐代是中国历史上文化最繁荣的时期之一，诗歌、绘画、音乐都达到了高峰。"
                         "李白、杜甫是唐代最著名的诗人。", metadata={"id": "doc12"}),
    Document(page_content="计算机视觉技术在自动驾驶中扮演关键角色。"
                         "通过摄像头和激光雷达感知环境，实现车道检测、行人识别等功能。", metadata={"id": "doc13"}),
    Document(page_content="基因编辑技术CRISPR-Cas9能够精确修改DNA序列，"
                         "为遗传病治疗提供了新的可能。但也引发了伦理争议。", metadata={"id": "doc14"}),
    Document(page_content="云计算通过互联网提供按需计算资源，包括服务器、存储、数据库等。"
                         "AWS、Azure和阿里云是全球主要的云服务提供商。", metadata={"id": "doc15"}),
]


# === 与向量检索器对比 ===

def jaccard_similarity(list1: List[Document], list2: List[Document]) -> float:
    """计算两个检索结果列表的 Jaccard 相似度"""
    ids1 = {doc.metadata.get("id", doc.page_content[:30]) for doc in list1}
    ids2 = {doc.metadata.get("id", doc.page_content[:30]) for doc in list2}
    
    intersection = len(ids1 & ids2)
    union = len(ids1 | ids2)
    
    return intersection / union if union > 0 else 0.0


def compare_retrievers():
    """对比关键词检索器与向量检索器"""
    import tempfile
    
    # 创建向量检索器
    persist_dir = tempfile.mkdtemp(prefix="compare_")
    embeddings = FakeEmbeddings(size=384)
    vectorstore = Chroma.from_documents(
        test_documents, embeddings, persist_directory=persist_dir
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 创建关键词检索器（两种策略）
    keyword_simple = KeywordRetriever(
        documents=test_documents, k=3, strategy="simple", match_threshold=1
    )
    keyword_tfidf = KeywordRetriever(
        documents=test_documents, k=3, strategy="tfidf", match_threshold=0.01
    )
    
    queries = [
        "人工智能在医疗领域的应用",
        "Python编程",
        "中国历史文化遗产",
        "气候变化与环境保护",
        "基因编辑技术",
    ]
    
    print("=" * 80)
    print("检索器对比分析")
    print("=" * 80)
    
    for query in queries:
        print(f"\n查询: '{query}'")
        print("-" * 60)
        
        try:
            vec_results = vector_retriever.invoke(query)
            sim_results = keyword_simple.invoke(query)
            tfidf_results = keyword_tfidf.invoke(query)
        except Exception as e:
            print(f"  检索错误: {e}")
            continue
        
        # 打印结果
        for label, results in [
            ("向量检索", vec_results),
            ("关键词(simple)", sim_results),
            ("关键词(TF-IDF)", tfidf_results),
        ]:
            ids = [doc.metadata.get("id", "?") for doc in results]
            print(f"  {label}: {ids}")
        
        # 计算 Jaccard 相似度
        jac_vs_sim = jaccard_similarity(vec_results, sim_results)
        jac_vs_tfidf = jaccard_similarity(vec_results, tfidf_results)
        jac_sim_vs_tfidf = jaccard_similarity(sim_results, tfidf_results)
        print(f"  Jaccard: 向量vs简单={jac_vs_sim:.2f}, 向量vsTFIDF={jac_vs_tfidf:.2f}, "
              f"简单vsTFIDF={jac_sim_vs_tfidf:.2f}")
    
    # 边界测试
    print(f"\n{'='*60}")
    print("边界测试")
    print(f"{'='*60}")
    
    edge_cases = [
        ("", "空查询"),
        ("的 了 在 是", "仅停用词"),
        ("人工智能在医疗领域的应用" * 50, "超长查询"),
    ]
    
    for query, desc in edge_cases:
        print(f"\n测试: {desc} ('{query[:30]}...')")
        results = keyword_simple.invoke(query)
        print(f"  关键词检索(simple): {len(results)} 个结果")
        results = keyword_tfidf.invoke(query)
        print(f"  关键词检索(tfidf): {len(results)} 个结果")


if __name__ == "__main__":
    compare_retrievers()
```

---

## 练习 4 解答：MultiQueryRetriever 对比分析

```python
"""
练习4: 构建 MultiQueryRetriever 并与单查询检索对比

核心思路:
- 手动实现查询变体生成（模拟 LLM）
- 系统对比单查询 vs 多查询
- 计算召回率、精确率等评估指标
"""

from typing import List, Set, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_chroma import Chroma
import tempfile


# === 步骤1: 准备15+篇文档 ===

all_documents = [
    Document(page_content="费曼学习法是一种通过教学来学习的技巧。核心思想是：如果你不能用简单的语言解释清楚，"
                         "说明你还没有真正理解。这个方法包括概念选择、教学模拟、发现盲点和简化表达四个步骤。",
             metadata={"id": "doc1", "domain": "学习方法"}),
    Document(page_content="间隔重复是一种高效记忆技术。通过在即将遗忘的时间点复习，最大化记忆保持率。"
                         "Anki是使用间隔重复原理的知名记忆软件。",
             metadata={"id": "doc2", "domain": "学习方法"}),
    Document(page_content="主动回忆是指在不看材料的情况下，主动提取记忆中的信息。"
                         "研究表明，这种方法比被动重读效果更好。",
             metadata={"id": "doc3", "domain": "学习方法"}),
    Document(page_content="番茄工作法将工作时间划分为25分钟的工作段和5分钟的休息段。"
                         "每四个番茄钟后进行一次较长的休息（15-30分钟）。",
             metadata={"id": "doc4", "domain": "学习方法"}),
    Document(page_content="思维导图是一种可视化的思维工具，由中心主题向外扩展分支。"
                         "通过颜色、图像和关键词帮助理解和记忆。",
             metadata={"id": "doc5", "domain": "学习方法"}),
    Document(page_content="Python列表推导式是创建列表的简洁语法：[x for x in range(10)]。"
                         "它比传统的for循环更高效且更Pythonic。",
             metadata={"id": "doc6", "domain": "编程"}),
    Document(page_content="Docker容器技术允许将应用及其依赖打包到轻量级容器中。"
                         "容器与宿主机共享操作系统内核，比虚拟机更高效。",
             metadata={"id": "doc7", "domain": "编程"}),
    Document(page_content="LangChain是一个用于构建LLM应用的框架。"
                         "2024年推出了LangGraph用于构建有状态的AI代理。",
             metadata={"id": "doc8", "domain": "AI技术"}),
    Document(page_content="Transformer架构基于自注意力机制，是GPT、BERT等模型的基础。"
                         "注意力机制使模型能够捕捉序列中的长距离依赖关系。",
             metadata={"id": "doc9", "domain": "AI技术"}),
    Document(page_content="RAG（检索增强生成）结合了信息检索和文本生成。"
                         "它先从知识库检索相关文档，再将文档作为上下文提供给生成模型。",
             metadata={"id": "doc10", "domain": "AI技术"}),
    Document(page_content="向量数据库专门用于存储和检索高维向量嵌入。"
                         "Chroma、Qdrant、Pinecone是常用的向量数据库。",
             metadata={"id": "doc11", "domain": "AI技术"}),
    Document(page_content="Prompt Engineering是设计和优化提示词以引导LLM生成期望输出的技术。"
                         "包括零样本提示、少样本提示、思维链提示等策略。",
             metadata={"id": "doc12", "domain": "AI技术"}),
    Document(page_content="Kubernetes（K8s）是一个开源的容器编排平台，用于自动化部署、扩展和管理容器化应用。"
                         "它提供了服务发现、负载均衡、自动伸缩等功能。",
             metadata={"id": "doc13", "domain": "编程"}),
    Document(page_content="敏捷开发是一种迭代式的软件开发方法论，强调快速交付、持续反馈和团队协作。"
                         "Scrum是最流行的敏捷框架之一。",
             metadata={"id": "doc14", "domain": "编程"}),
    Document(page_content="机器学习模型部署涉及将训练好的模型集成到生产环境中。"
                         "MLOps实践包括模型版本控制、持续监控和自动重新训练。",
             metadata={"id": "doc15", "domain": "AI技术"}),
    Document(page_content="Git是分布式版本控制系统，由Linus Torvalds创建。"
                         "GitHub是基于Git的代码托管平台，提供协作开发功能。",
             metadata={"id": "doc16", "domain": "编程"}),
]


# === 步骤2: 创建向量检索器 ===

persist_dir = tempfile.mkdtemp(prefix="multiquery_")
embeddings = FakeEmbeddings(size=384)
vectorstore = Chroma.from_documents(all_documents, embeddings, persist_directory=persist_dir)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# === 步骤3: 查询变体生成（模拟 LLM） ===

QUERY_EXPANSIONS = {
    "学习": ["学习方法", "学习技巧", "高效学习", "学习策略"],
    "技术": ["新技术", "技术发展", "技术趋势", "科技"],
    "编程": ["编程语言", "写代码", "软件开发", "程序设计"],
    "AI": ["人工智能", "机器学习", "深度学习", "大模型"],
    "部署": ["上线", "发布", "运维", "DevOps"],
    "记忆": ["记忆技巧", "记忆力", "背诵", "记住"],
    "方法": ["方法论", "技巧", "策略", "方式"],
}


def generate_query_variations(original: str, n: int = 4) -> List[str]:
    """使用规则模拟 LLM 生成查询变体"""
    variations = [original]
    
    # 模板改写
    templates = [
        lambda q: f"{q}是什么",
        lambda q: f"如何理解{q}",
        lambda q: f"{q}的核心要点",
        lambda q: f"{q}的详细解释",
        lambda q: f"关于{q}的一切",
    ]
    
    for tpl in templates[:n]:
        variation = tpl(original)
        if variation not in variations:
            variations.append(variation)
    
    # 关键词扩展
    for keyword, alternatives in QUERY_EXPANSIONS.items():
        if keyword in original:
            for alt in alternatives[:2]:
                new_q = original.replace(keyword, alt)
                if new_q not in variations:
                    variations.append(new_q)
    
    return list(dict.fromkeys(variations))[:n + 1]


# === 步骤4: 多查询检索 ===

def multi_query_retrieve(query: str, k: int = 5) -> Tuple[List[Document], List[str]]:
    """
    多查询检索: 生成变体 -> 独立检索 -> 合并去重。
    
    Returns:
        (merged_results, variations_used)
    """
    variations = generate_query_variations(query, n=4)
    
    seen_ids = set()
    all_docs = []
    
    for var in variations:
        docs = base_retriever.invoke(var)
        for doc in docs:
            doc_id = doc.metadata.get("id", doc.page_content[:40])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
    
    return all_docs[:k], variations


# === 步骤5: 对比分析 ===

def evaluate_retrieval(query: str) -> Dict:
    """对比单查询 vs 多查询"""
    
    # 单查询
    single_docs = base_retriever.invoke(query)
    single_ids = {d.metadata.get("id", "") for d in single_docs}
    
    # 多查询
    multi_docs, variations = multi_query_retrieve(query, k=5)
    multi_ids = {d.metadata.get("id", "") for d in multi_docs}
    
    # 合并结果池（视为"相关文档全集"的近似）
    pool_ids = single_ids | multi_ids
    pool_size = len(pool_ids)
    
    # 评估指标
    overlap = len(single_ids & multi_ids)
    only_single = len(single_ids - multi_ids)
    only_multi = len(multi_ids - single_ids)
    
    # 以合并结果池为全集计算召回率
    single_recall = len(single_ids) / pool_size if pool_size > 0 else 1.0
    multi_recall = len(multi_ids) / pool_size if pool_size > 0 else 1.0
    
    return {
        "query": query,
        "variations": variations,
        "single_count": len(single_docs),
        "multi_count": len(multi_docs),
        "pool_size": pool_size,
        "overlap": overlap,
        "only_single": only_single,
        "only_multi": only_multi,
        "single_recall": round(single_recall, 3),
        "multi_recall": round(multi_recall, 3),
        "recall_improvement": round(multi_recall - single_recall, 3),
        "single_docs": single_docs,
        "multi_docs": multi_docs,
    }


# === 步骤6: 测试 ===

if __name__ == "__main__":
    # 模糊查询测试集
    fuzzy_queries = [
        "那个很火的学习方法",
        "最新的那个技术",
        "怎么更好地写代码",
        "那个关于记忆的东西",
        "AI那个新框架",
    ]
    
    all_results = []
    
    for query in fuzzy_queries:
        result = evaluate_retrieval(query)
        all_results.append(result)
        
        print(f"\n{'='*70}")
        print(f"查询: '{query}'")
        print(f"{'='*70}")
        print(f"LLM生成变体: {result['variations'][1:]}")
        print(f"\n单查询 top-5: {[d.metadata['id'] for d in result['single_docs']]}")
        print(f"多查询 top-5: {[d.metadata['id'] for d in result['multi_docs']]}")
        print(f"\n对比数据: 单查询{result['single_count']}个 | "
              f"多查询{result['multi_count']}个 | "
              f"重叠{result['overlap']}个 | "
              f"新增{result['only_multi']}个")
        print(f"召回率: 单查询={result['single_recall']:.0%} | "
              f"多查询={result['multi_recall']:.0%} | "
              f"提升: +{result['recall_improvement']:.0%}")
    
    # 汇总表格
    print(f"\n{'='*70}")
    print(f"汇总对比表格")
    print(f"{'='*70}")
    print(f"{'查询':<25} {'单查询':<8} {'多查询':<8} {'重叠':<6} {'新增':<6} {'召回提升':<10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['query']:<25} {r['single_count']:<8} {r['multi_count']:<8} "
              f"{r['overlap']:<6} {r['only_multi']:<6} {r['recall_improvement']:>+8.0%}")
```

---

## 练习 5 解答：@tool + bind_tools + 错误处理

```python
"""
练习5: 使用 @tool 装饰器创建工具，绑定到模型，展示错误处理

核心思路:
- 4个工具各有不同的校验逻辑和错误场景
- ToolException 用于结构化错误
- handle_tool_errors 参数控制错误行为
"""

import re
from typing import Optional
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field


# === 工具1: 计算器 ===

class CalculatorInput(BaseModel):
    expression: str = Field(
        description="数学表达式，如 '(15+25)*3/2'。仅支持数字、四则运算符、括号、小数点。"
    )


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    安全执行数学表达式计算。

    限制: 仅允许数字、+、-、*、/、()、.和空格。
    禁止任何函数调用或变量引用。
    """
    # 安全检查：只允许安全字符
    allowed_pattern = re.compile(r'^[\d+\-*/(). ]+$')
    if not allowed_pattern.match(expression):
        raise ToolException(
            f"非法的数学表达式: {expression}。"
            f"仅允许数字、四则运算符(+-*/)、括号和小数点。"
        )
    
    # 检查危险模式
    dangerous = ["__", "import", "eval", "exec", "open", "os", "sys", "lambda"]
    expr_lower = expression.lower()
    for keyword in dangerous:
        if keyword in expr_lower:
            raise ToolException(f"表达式包含不允许的关键词: '{keyword}'")
    
    try:
        # 安全计算
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except ZeroDivisionError:
        raise ToolException("除数不能为零。请输入有效的数学表达式。")
    except SyntaxError:
        raise ToolException(f"表达式语法错误: {expression}")
    except Exception as e:
        raise ToolException(f"计算错误: {type(e).__name__} - {str(e)}")


# === 工具2: 文本翻译器 ===

class TranslatorInput(BaseModel):
    text: str = Field(description="要翻译的文本")
    target_language: str = Field(description="目标语言: '中文' 或 '英文'")


@tool(args_schema=TranslatorInput)
def text_translator(text: str, target_language: str) -> str:
    """
    文本翻译工具。
    
    支持中文和英文之间的互译。自动检测源语言。
    """
    # 空文本检查
    if not text or not text.strip():
        raise ToolException("待翻译文本不能为空")
    
    # 语言支持检查
    supported = ["中文", "英文"]
    if target_language not in supported:
        raise ToolException(
            f"不支持的目标语言: '{target_language}'。"
            f"当前支持: {', '.join(supported)}"
        )
    
    # 检测源语言
    has_chinese = any('一' <= c <= '鿿' for c in text)
    source_lang = "中文" if has_chinese else "英文"
    
    # 同语言检查
    if source_lang == target_language:
        return f"翻译结果: 源语言与目标语言相同，无需翻译。原文: {text}"
    
    # 模拟翻译
    if source_lang == "中文" and target_language == "英文":
        return f"翻译结果 (中文→英文): [Translated] {text}"
    else:
        return f"翻译结果 (英文→中文): [已翻译] {text}"


# === 工具3: 代码执行器 ===

class CodeRunnerInput(BaseModel):
    code: str = Field(description="要执行的Python代码")


@tool(args_schema=CodeRunnerInput)
def code_runner(code: str) -> str:
    """
    在安全沙箱中执行Python代码。
    
    安全限制: 仅允许简单的 print 语句。
    禁止: import、文件操作、系统调用、eval/exec 等。
    """
    code_lower = code.lower()
    
    # 危险关键词检测
    dangerous_keywords = [
        "import", "__", "os.", "os ", "subprocess", "eval(", "exec(",
        "open(", "file(", "socket", "shutil", "sys.", "compile",
        "globals", "locals", "getattr", "setattr", "delattr",
    ]
    
    for keyword in dangerous_keywords:
        if keyword.lower() in code_lower:
            raise ToolException(
                f"检测到不安全的代码操作: '{keyword}'。仅允许 print() 语句。"
            )
    
    # 白名单检查
    stripped = code.strip()
    if not stripped.startswith("print(") and not stripped.startswith("print ("):
        raise ToolException("仅允许 print() 语句。其他操作被禁止。")
    
    # 模拟执行
    import io
    import sys
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    
    try:
        safe_builtins = {"print": print, "True": True, "False": False, "None": None}
        exec(code, {"__builtins__": safe_builtins}, {})
        output = captured.getvalue().strip()
        return f"执行成功:\n{output}" if output else "执行成功（无输出）"
    except Exception as e:
        return f"执行错误: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout


# === 工具4: 知识库搜索 ===

class SearchInput(BaseModel):
    query: str = Field(description="搜索查询关键词")
    top_k: int = Field(default=3, ge=1, le=10, description="返回结果数量 (1-10)")


@tool(args_schema=SearchInput)
def knowledge_search(query: str, top_k: int = 3) -> str:
    """
    在知识库中搜索相关信息。
    """
    # 空查询检查
    if not query or not query.strip():
        raise ToolException("搜索查询不能为空")
    
    # top_k 范围检查
    if top_k < 1 or top_k > 10:
        raise ToolException(f"top_k 必须在 1-10 之间，当前值: {top_k}")
    
    # 模拟知识库搜索结果
    knowledge_base = {
        "langchain": "LangChain 是一个用于开发LLM应用的开源框架...",
        "python": "Python 是一种高级编程语言，以简洁易读著称...",
        "rag": "RAG（检索增强生成）结合了信息检索和文本生成...",
        "docker": "Docker 是开源的容器化平台...",
    }
    
    results = []
    for key, content in knowledge_base.items():
        if key in query.lower():
            results.append(f"[{key}] {content}")
    
    if not results:
        results.append(f"未找到关于'{query}'的精确匹配结果。"
                       f"建议尝试其他关键词搜索。")
    
    return "\n\n".join(results[:top_k])


# === 测试 ===

if __name__ == "__main__":
    all_tools = [calculator, text_translator, code_runner, knowledge_search]
    
    print("工具列表:")
    for t in all_tools:
        print(f"  - {t.name}: {t.description[:60]}...")
    
    # === 正常场景 ===
    print(f"\n{'='*60}")
    print("正常场景测试")
    print(f"{'='*60}")
    
    result = calculator.invoke({"expression": "(15 + 25) * 3 / 2"})
    print(f"  calculator: {result}")
    
    result = text_translator.invoke({"text": "你好世界", "target_language": "英文"})
    print(f"  translator: {result}")
    
    result = code_runner.invoke({"code": "print('Hello, Sandbox!')"})
    print(f"  code_runner: {result}")
    
    result = knowledge_search.invoke({"query": "python编程", "top_k": 2})
    print(f"  search: {result}")
    
    # === 错误场景 ===
    print(f"\n{'='*60}")
    print("错误场景测试")
    print(f"{'='*60}")
    
    # 除零
    result = calculator.invoke({"expression": "100 / 0"})
    print(f"  除零: {result}")
    
    # 不安全代码
    result = code_runner.invoke({"code": "import os; os.system('ls')"})
    print(f"  不安全代码: {result}")
    
    # 不支持的语言
    result = text_translator.invoke({"text": "你好", "target_language": "法语"})
    print(f"  不支持的语言: {result}")
    
    # 空查询
    result = knowledge_search.invoke({"query": "", "top_k": 3})
    print(f"  空查询: {result}")
    
    # 非法表达式
    result = calculator.invoke({"expression": "__import__('os').system('ls')"})
    print(f"  非法表达式: {result}")


# === bind_tools 使用示例 ===
"""
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# 绑定工具
llm_with_tools = llm.bind_tools(
    [calculator, text_translator, code_runner, knowledge_search],
    tool_choice="auto",
)

# 模型会根据查询自动选择合适的工具
from langchain_core.messages import HumanMessage

response = llm_with_tools.invoke([
    HumanMessage(content="帮我计算 (15+25)*3/2 的结果")
])

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"模型决定调用: {tc['name']}({tc['args']})")
"""
```

---

## 练习 6 解答：流式 RAG + Token/成本回调

```python
"""
练习6: 构建流式 RAG 链，使用自定义回调跟踪 token 使用和成本

核心思路:
- TokenUsageCallback 记录每次调用的 token 详情
- CostEstimateCallback 基于模型定价计算费用
- .stream() 流式输出
- 最终输出汇总统计
"""

import time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.embeddings import FakeEmbeddings
from langchain_chroma import Chroma
from operator import itemgetter
import tempfile


# === 回调1: Token 使用量追踪 ===

class TokenUsageCallback(BaseCallbackHandler):
    """
    记录每次 LLM 调用的详细 token 使用情况。
    
    从 response.llm_output["token_usage"] 中提取:
    - prompt_tokens: 输入 token 数
    - completion_tokens: 输出 token 数
    - total_tokens: 总 token 数
    """
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_records: List[Dict] = []
        self._call_start_time: Optional[float] = None
    
    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        """记录调用开始时间"""
        self._call_start_time = time.time()
        self.call_count += 1
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """提取 token 使用量"""
        elapsed = time.time() - self._call_start_time if self._call_start_time else 0
        
        # 从 response.llm_output 提取
        token_usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
        
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total
        
        self.call_records.append({
            "call_number": self.call_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "latency_seconds": round(elapsed, 3),
        })
    
    def get_summary(self) -> Dict:
        """获取汇总统计"""
        return {
            "总调用次数": self.call_count,
            "总输入Token": self.total_prompt_tokens,
            "总输出Token": self.total_completion_tokens,
            "总Token": self.total_tokens,
            "平均输入Token/次": round(
                self.total_prompt_tokens / max(self.call_count, 1), 1
            ),
            "平均输出Token/次": round(
                self.total_completion_tokens / max(self.call_count, 1), 1
            ),
            "调用明细": self.call_records,
        }
    
    def reset(self):
        """重置所有计数器"""
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_records = []
        self._call_start_time = None


# === 回调2: 成本估算 ===

class CostEstimateCallback(BaseCallbackHandler):
    """
    基于 token 使用量估算 API 调用成本。
    
    定价表 (每1K tokens, USD, 2024年参考):
    - gpt-4o: $0.005 prompt / $0.015 completion
    - gpt-4o-mini: $0.00015 prompt / $0.0006 completion
    - gpt-4-turbo: $0.01 prompt / $0.03 completion
    """
    
    PRICING = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    }
    
    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__()
        self.model_name = model_name
        self.total_cost_usd = 0.0
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._session_start_time: Optional[float] = None
    
    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        if self._session_start_time is None:
            self._session_start_time = time.time()
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.call_count += 1
        pricing = self.PRICING.get(
            self.model_name,
            {"prompt": 0.001, "completion": 0.002}
        )
        
        token_usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
        
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]
        
        self.total_cost_usd += prompt_cost + completion_cost
    
    def get_summary(self) -> Dict:
        """获取成本摘要"""
        elapsed = time.time() - self._session_start_time if self._session_start_time else 0
        return {
            "模型": self.model_name,
            "调用次数": self.call_count,
            "总输入Token": self.total_prompt_tokens,
            "总输出Token": self.total_completion_tokens,
            "总成本(USD)": round(self.total_cost_usd, 6),
            "总成本(CNY, 汇率7.25)": round(self.total_cost_usd * 7.25, 4),
            "处理时间(秒)": round(elapsed, 2),
        }
    
    def reset(self):
        self.total_cost_usd = 0.0
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._session_start_time = None


# === RAG 链构建 ===

documents = [
    Document(page_content="LangChain使用LCEL（LangChain Expression Language）语法构建链。", metadata={"id": "1"}),
    Document(page_content="流式输出通过.stream()方法实现逐token生成。", metadata={"id": "2"}),
    Document(page_content="RAG结合检索和生成，提高答案准确性。", metadata={"id": "3"}),
    Document(page_content="向量数据库存储文档嵌入向量用于语义搜索。", metadata={"id": "4"}),
    Document(page_content="回调处理器可以在链执行的各阶段插入自定义逻辑。", metadata={"id": "5"}),
]

persist_dir = tempfile.mkdtemp(prefix="stream_rag_")
embeddings = FakeEmbeddings(size=384)
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs: list) -> str:
    return "\n\n".join(d.page_content for d in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", "基于上下文简洁回答:\n\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {
        "context": itemgetter("question") | retriever | RunnableLambda(format_docs),
        "question": itemgetter("question"),
    }
    | prompt
    # | llm           # 替换为真实 LLM
    | StrOutputParser()
)


# === 模拟流式演示 ===

def demo_streaming_rag(query: str):
    """演示流式 RAG 链 + 回调追踪"""
    
    token_cb = TokenUsageCallback()
    cost_cb = CostEstimateCallback(model_name="gpt-4o-mini")
    
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"{'='*60}")
    
    # 检索
    docs = retriever.invoke(query)
    formatted = format_docs(docs)
    print(f"\n[检索] 获取 {len(docs)} 个文档 ({len(formatted)} 字符)")
    
    # 模拟流式输出
    simulated = (
        "根据上下文，RAG（检索增强生成）通过结合文档检索和文本生成来提高答案准确性。"
        "LangChain的LCEL语法使用管道运算符构建处理链，并支持.stream()方法实现流式输出。"
    )
    
    print(f"\n[流式输出]")
    for char in simulated:
        print(char, end="", flush=True)
        time.sleep(0.02)
    print()
    
    # 模拟 token 数据
    mock_prompt = len(formatted) // 2
    mock_completion = len(simulated) // 2
    
    token_cb.call_count = 1
    token_cb.total_prompt_tokens = mock_prompt
    token_cb.total_completion_tokens = mock_completion
    token_cb.total_tokens = mock_prompt + mock_completion
    
    cost_cb.call_count = 1
    cost_cb.total_prompt_tokens = mock_prompt
    cost_cb.total_completion_tokens = mock_completion
    pricing = CostEstimateCallback.PRICING["gpt-4o-mini"]
    cost_cb.total_cost_usd = (
        (mock_prompt / 1000) * pricing["prompt"] +
        (mock_completion / 1000) * pricing["completion"]
    )
    
    # 输出统计
    print(f"\n{'='*60}")
    print("Token 统计")
    print(f"{'='*60}")
    for k, v in token_cb.get_summary().items():
        if k != "调用明细":
            print(f"  {k}: {v}")
    
    print(f"\n{'='*60}")
    print("成本统计")
    print(f"{'='*60}")
    for k, v in cost_cb.get_summary().items():
        print(f"  {k}: {v}")
    
    print(f"\n实际使用方式:")
    print(f"""
token_cb = TokenUsageCallback()
cost_cb = CostEstimateCallback(model_name="gpt-4o-mini")

for chunk in chain.stream(
    {{"question": query}},
    config={{"callbacks": [token_cb, cost_cb]}}
):
    print(chunk, end="", flush=True)

print(f"\\\\n总Token: {{token_cb.total_tokens}}")
print(f"总成本: ${{cost_cb.total_cost_usd:.6f}}")
""")


if __name__ == "__main__":
    demo_streaming_rag("什么是RAG和流式处理？")
```

---

## 练习 7 解答：SelfQueryRetriever

```python
"""
练习7: 实现 SelfQueryRetriever 并测试多属性过滤

核心思路:
- 定义3个 AttributeInfo 描述元数据字段
- 手动规则解析模拟 LLM 的查询解析能力
- 展示每个查询提取的 semantic_query 和 filter
- 验证过滤准确性
"""

import re
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
import tempfile


# === 步骤1: 准备10+篇含丰富元数据的文档 ===

documents = [
    Document(page_content="Transformer架构详解：自注意力机制、多头注意力和位置编码。"
                         "2024年最新整理的版本，包含大量代码示例。",
             metadata={"category": "技术", "year": 2024, "rating": 4.8, "author": "张三"}),
    Document(page_content="Python高级编程：装饰器、生成器、上下文管理器和元编程技术。",
             metadata={"category": "技术", "year": 2023, "rating": 4.5, "author": "李四"}),
    Document(page_content="UI/UX设计原则：一致性、反馈、容错、易学性的深入分析。",
             metadata={"category": "设计", "year": 2024, "rating": 4.2, "author": "王五"}),
    Document(page_content="敏捷项目管理实战：Scrum框架、Sprint规划、回顾会议。",
             metadata={"category": "管理", "year": 2023, "rating": 3.9, "author": "张三"}),
    Document(page_content="深度学习入门：CNN、RNN和Transformer的PyTorch实现。"
                         "适合有一定Python基础的读者。",
             metadata={"category": "技术", "year": 2022, "rating": 4.6, "author": "赵六"}),
    Document(page_content="用户体验研究：用户访谈、可用性测试和数据分析方法。",
             metadata={"category": "设计", "year": 2024, "rating": 4.7, "author": "王五"}),
    Document(page_content="团队领导力：激励、沟通、冲突解决和绩效管理。",
             metadata={"category": "管理", "year": 2024, "rating": 4.0, "author": "李四"}),
    Document(page_content="AI伦理与安全：偏见检测、隐私保护和负责任AI开发指南。",
             metadata={"category": "技术", "year": 2023, "rating": 4.9, "author": "赵六"}),
    Document(page_content="交互设计模式大全：导航、搜索、表单和反馈设计最佳实践。",
             metadata={"category": "设计", "year": 2022, "rating": 4.1, "author": "张三"}),
    Document(page_content="远程团队管理：分布式团队的沟通工具、流程和绩效考核。",
             metadata={"category": "管理", "year": 2023, "rating": 3.8, "author": "王五"}),
    Document(page_content="LLM应用开发：Prompt Engineering、RAG和Agent开发全解。",
             metadata={"category": "技术", "year": 2024, "rating": 4.9, "author": "李四"}),
    Document(page_content="数据可视化设计：图表选择、配色和信息层级设计指南。",
             metadata={"category": "设计", "year": 2023, "rating": 4.3, "author": "赵六"}),
]


# === 步骤2: AttributeInfo 定义 ===

metadata_field_info = [
    AttributeInfo(
        name="category",
        description="文档分类: '技术', '设计', 或 '管理'",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="发布年份，如 2022, 2023, 2024",
        type="integer",
    ),
    AttributeInfo(
        name="rating",
        description="评分 (1.0-5.0)，如 4.5",
        type="float",
    ),
    AttributeInfo(
        name="author",
        description="作者姓名: '张三', '李四', '王五', 或 '赵六'",
        type="string",
    ),
]

document_content_description = "技术、设计和管理领域的文章"


# === 步骤3: 查询解析器（模拟 LLM） ===

def parse_query(query: str) -> Tuple[str, Dict]:
    """
    模拟 SelfQuery LLM 的查询解析。
    
    从自然语言中提取:
    1. semantic_query: 用于向量相似度搜索
    2. filter: 元数据过滤条件
    
    Returns:
        (semantic_query_str, filter_dict)
    """
    semantic = query
    filters = {}
    
    # 解析年份
    year_patterns = [
        (r'(\d{4})年(?:以后|之后|以来)', "$gte"),
        (r'(\d{4})年(?:以后|之后|以来)', "$gte"),
        (r'(\d{4})年(?:以前|之前)', "$lte"),
        (r'(\d{4})年', "$eq"),
    ]
    for pattern, op in year_patterns:
        m = re.search(pattern, query)
        if m:
            year = int(m.group(1))
            filters["year"] = {op: year}
            semantic = semantic.replace(m.group(0), "").strip()
            break
    
    # 解析评分
    rating_patterns = [
        (r'评分.*?高于\s*([\d.]+)', "$gte"),
        (r'评分.*?超过\s*([\d.]+)', "$gte"),
        (r'评分.*?低于\s*([\d.]+)', "$lte"),
        (r'([\d.]+)分以上', "$gte"),
        (r'([\d.]+)星', "$gte"),
    ]
    for pattern, op in rating_patterns:
        m = re.search(pattern, query)
        if m:
            rating = float(m.group(1))
            filters["rating"] = {op: rating}
            break
    
    # 解析分类
    for cat in ["技术", "设计", "管理"]:
        if cat in query:
            filters["category"] = cat
            semantic = semantic.replace(f"{cat}类", "").replace(cat, "").strip()
            break
    
    # 解析作者
    for author in ["张三", "李四", "王五", "赵六"]:
        if author in query:
            filters["author"] = author
            semantic = semantic.replace(f"{author}写的", "").replace(f"{author}撰写", "").replace(author, "").strip()
            break
    
    # 清理语义查询
    semantic = re.sub(r'的[，。]*$', '', semantic)
    semantic = re.sub(r'[，。！？、]$', '', semantic)
    semantic = semantic.replace("评分", "").strip()
    
    return semantic if semantic else query, filters


# === 步骤4: SelfQueryRetriever 模拟 ===

class SimulatedSelfQueryRetriever:
    """模拟 SelfQueryRetriever 行为"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def invoke(self, query: str) -> List[Document]:
        semantic_query, filter_dict = parse_query(query)
        
        print(f"\n  语义查询: '{semantic_query}'")
        print(f"  过滤器: {filter_dict}")
        
        if filter_dict:
            results = self.vectorstore.similarity_search(
                semantic_query, k=10,
                filter=filter_dict if isinstance(filter_dict, dict) else None,
            )
        else:
            results = self.vectorstore.similarity_search(semantic_query, k=10)
        
        # 限制数量
        limit_match = re.search(r'(\d+)篇|前(\d+)', query)
        limit = int(limit_match.group(1) or limit_match.group(2)) if limit_match else 5
        return results[:limit]


# === 步骤5: 测试 ===

if __name__ == "__main__":
    persist_dir = tempfile.mkdtemp(prefix="selfquery_")
    embeddings = FakeEmbeddings(size=384)
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)
    retriever = SimulatedSelfQueryRetriever(vectorstore)
    
    test_queries = [
        "2024年评分高于4.5的技术类文章",
        "张三写的2022年以后的文档",
        "评分最高的3篇文章",
        "李四写的技术类文章",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: '{query}'")
        results = retriever.invoke(query)
        
        print(f"  结果 ({len(results)} 个):")
        for i, doc in enumerate(results, 1):
            m = doc.metadata
            print(f"    [{i}] [{m['category']}] {m['author']} "
                  f"({m['year']}, 评分:{m['rating']}): {doc.page_content[:50]}...")
```

---

## 练习 8 解答：RunnableBranch 路由系统

```python
"""
练习8: 构建 RunnableBranch 路由系统，将不同查询发送到不同检索器

核心思路:
- 查询类型检测: "tech", "news", "general"
- RunnableBranch 根据类型路由
- 低置信度时合并多检索器结果
- 展示路由决策过程
"""

from typing import List, Dict, Tuple, Optional
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document


# === 步骤1: 三个检索器 ===

tech_docs = [
    Document(page_content="Python 3.12 泛型语法简化，性能提升显著", metadata={"source": "tech"}),
    Document(page_content="React Server Components 成为新默认渲染模式", metadata={"source": "tech"}),
    Document(page_content="Kubernetes 1.30 新增弹性伸缩和成本优化功能", metadata={"source": "tech"}),
    Document(page_content="Rust在系统编程领域份额持续增长", metadata={"source": "tech"}),
    Document(page_content="GraphQL在API设计中越来越流行", metadata={"source": "tech"}),
]

news_docs = [
    Document(page_content="2024巴黎奥运会筹备工作进入最后阶段", metadata={"source": "news"}),
    Document(page_content="美联储维持利率不变，市场反应积极", metadata={"source": "news"}),
    Document(page_content="科技巨头最新财报显示AI业务大幅增长", metadata={"source": "news"}),
    Document(page_content="全球气候峰会达成新的减排协议", metadata={"source": "news"}),
    Document(page_content="新能源汽车销量首次超过燃油车", metadata={"source": "news"}),
]

general_docs = [
    Document(page_content="水在标准大气压下100°C沸腾", metadata={"source": "general"}),
    Document(page_content="地球是太阳系第三颗行星", metadata={"source": "general"}),
    Document(page_content="莎士比亚创作了37部戏剧和154首十四行诗", metadata={"source": "general"}),
    Document(page_content="相对论由爱因斯坦于1905年提出", metadata={"source": "general"}),
    Document(page_content="光合作用将光能转化为化学能", metadata={"source": "general"}),
]


def make_retriever(docs: List[Document]) -> RunnableLambda:
    return RunnableLambda(lambda q: docs)


tech_retriever = make_retriever(tech_docs)
news_retriever = make_retriever(news_docs)
general_retriever = make_retriever(general_docs)


# === 步骤2: 查询类型检测 ===

TECH_KEYWORDS = [
    "代码", "编程", "python", "javascript", "react", "vue", "api",
    "数据库", "docker", "kubernetes", "微服务", "算法", "架构",
    "rust", "golang", "typescript", "http", "前端", "后端", "部署",
    "graphql", "devops", "容器", "框架", "库", "函数",
]

NEWS_KEYWORDS = [
    "新闻", "最新", "报道", "宣布", "发布", "事件", "峰会", "协议",
    "政策", "利率", "股市", "奥运", "气候", "选举", "财报",
    "收购", "融资", "上市", "市场", "经济", "销量", "数据",
    "阶段", "成果", "趋势", "发展",
]


def detect_query_type(query: str) -> Tuple[str, float]:
    """
    检测查询类型并返回置信度。
    
    Returns:
        (type, confidence): type为"tech"/"news"/"general", confidence为0.0-1.0
    """
    query_lower = query.lower()
    
    tech_score = sum(1 for kw in TECH_KEYWORDS if kw in query_lower)
    news_score = sum(1 for kw in NEWS_KEYWORDS if kw in query_lower)
    
    max_score = max(tech_score, news_score, 1)
    
    if tech_score > news_score:
        return "tech", min(tech_score / max_score, 1.0)
    elif news_score > tech_score:
        return "news", min(news_score / max_score, 1.0)
    else:
        return "general", 0.5


# === 步骤3: RunnableBranch 路由 ===

branch = RunnableBranch(
    (lambda q: detect_query_type(q)[0] == "tech", tech_retriever),
    (lambda q: detect_query_type(q)[0] == "news", news_retriever),
    general_retriever,
)


# === 步骤4: 带置信度的路由 ===

def route_with_merge(query: str) -> Dict:
    """
    路由查询，低置信度时合并多检索器结果。
    """
    qtype, confidence = detect_query_type(query)
    
    # 高置信度 -> 单一检索器
    if confidence >= 0.4:
        docs = branch.invoke(query)
        return {
            "query": query,
            "type": qtype,
            "confidence": confidence,
            "docs": docs,
            "merged": False,
        }
    
    # 低置信度 -> 合并
    all_docs = []
    seen = set()
    for retriever in [tech_retriever, news_retriever, general_retriever]:
        for doc in retriever.invoke(query):
            key = doc.page_content[:30]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
    
    return {
        "query": query,
        "type": "merged",
        "confidence": confidence,
        "docs": all_docs[:8],
        "merged": True,
    }


# === 步骤5: 测试 ===

if __name__ == "__main__":
    test_queries = [
        "Python代码优化",
        "最新科技公司财报",
        "今天天气怎么样",
        "最近国际大事",
        "Kubernetes部署方案",
        "莎士比亚有什么作品",
        "AI技术最新进展",
        "奥运会筹备情况",
    ]
    
    type_labels = {"tech": "技术检索器", "news": "新闻检索器", 
                   "general": "通用检索器", "merged": "合并检索器"}
    
    results_summary = []
    
    for query in test_queries:
        result = route_with_merge(query)
        results_summary.append(result)
        
        qtype = result["type"]
        conf = result["confidence"]
        docs = result["docs"]
        merged = result["merged"]
        
        merge_tag = " [低置信度合并]" if merged else ""
        label = type_labels.get(qtype, qtype)
        
        print(f"查询: '{query}'")
        print(f"  类型: {label}{merge_tag} (置信度: {conf:.2f})")
        print(f"  结果 ({len(docs)}):")
        for doc in docs:
            src = doc.metadata.get("source", "?")
            print(f"    [{src}] {doc.page_content[:50]}...")
        print()
    
    # 统计
    print(f"\n{'='*50}")
    print("路由统计")
    counts = {"tech": 0, "news": 0, "general": 0, "merged": 0}
    for r in results_summary:
        counts[r["type"]] += 1
    for t, c in counts.items():
        print(f"  {type_labels.get(t, t)}: {c} 次")
```

---

所有 8 个练习解答均提供完整可运行代码。实际使用时需将模拟 LLM、FakeEmbeddings 等替换为真实的 API 调用组件。
