#!/usr/bin/env python3
"""
C1-03：LlamaIndex 查询引擎（Query Engine）
=============================================
查询引擎是 LlamaIndex 的核心抽象，负责"接收查询 → 检索 → 合成 → 返回答案"。

本文件演示六种查询引擎模式：
1. 基础查询引擎 —— 直接检索+回答
2. RouterQueryEngine —— 根据查询意图路由到不同索引
3. SubQuestionQueryEngine —— 拆分子问题分别检索再合并
4. SQL 查询引擎 —— 自然语言转 SQL 查询结构化数据
5. 自定义查询引擎 —— 继承 BaseQueryEngine 实现业务逻辑
6. 响应模式对比 —— compact/refine/tree_summarize/simple_summarize

依赖：pip install llama-index
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# ============================================================================
# 准备阶段
# ============================================================================

def get_mock_llm():
    """创建模拟 LLM（演示用，不调用真实 API）。"""
    from llama_index.core.llms import BaseLLM
    from llama_index.core.llms import ChatMessage, ChatResponse, MessageTypes

    class MockLLM(BaseLLM):
        """模拟 LLM：根据查询内容返回预设回答。"""
        _model_name = "mock-llm-v1"

        def chat(self, messages, **kwargs):
            from llama_index.core.llms import ChatMessage, ChatResponse

            # 提取用户最后一条消息
            user_msg = ""
            for m in messages:
                if hasattr(m, 'role') and m.role == "user":
                    user_msg = m.content if hasattr(m, 'content') else str(m)
                elif isinstance(m, dict) and m.get("role") == "user":
                    user_msg = m.get("content", "")

            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=f"[MockLLM 回答] 基于检索到的上下文，针对问题"
                            f"'{user_msg[:50]}...'的综合回答。"
                )
            )

        def complete(self, prompt, **kwargs):
            from llama_index.core.llms import CompletionResponse
            return CompletionResponse(
                text=f"[MockLLM 回答] {prompt[:100]}..."
            )

        @property
        def metadata(self):
            return type("Metadata", (), {"model_name": self._model_name})()

    return MockLLM()


def get_mock_embed_model():
    """创建模拟嵌入模型。"""
    from llama_index.core.embeddings import BaseEmbedding

    class MockEmbedding(BaseEmbedding):
        _model_name = "mock-embed-v1"

        def _get_query_embedding(self, query: str) -> List[float]:
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_text_embedding(query)

        def _get_text_embedding(self, text: str) -> List[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [(b / 255.0) * 2 - 1 for b in h[:64]]

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            return [self._get_text_embedding(t) for t in texts]

    return MockEmbedding()


def prepare_documents():
    """准备分类文档集合。"""
    from llama_index.core import Document

    tech_docs = [
        Document(
            text="全新 iPhone 16 Pro 搭载 A18 Pro 芯片，支持光追。"
                 "屏幕尺寸 6.3 英寸，支持 120Hz ProMotion。起步价 999 美元。",
            metadata={"category": "科技", "subcategory": "产品发布"}
        ),
        Document(
            text="Python 3.13 正式发布，引入了新的 JIT 编译器，"
                 "性能提升 30%。同时改进了 asyncio 和类型系统。",
            metadata={"category": "科技", "subcategory": "技术更新"}
        ),
    ]

    finance_docs = [
        Document(
            text="央行宣布降准 0.5 个百分点，释放长期资金约 1 万亿元。"
                 "此举旨在支持实体经济，降低企业融资成本。",
            metadata={"category": "金融", "subcategory": "货币政策"}
        ),
        Document(
            text="A股三大指数今日集体收涨，沪指涨 1.2%，深成指涨 1.8%，"
                 "创业板指涨 2.3%。成交额突破万亿。",
            metadata={"category": "金融", "subcategory": "市场行情"}
        ),
    ]

    health_docs = [
        Document(
            text="WHO 发布最新健康指南：成年人每周应进行至少 150 分钟的"
                 "中等强度有氧运动。久坐会增加心血管疾病风险。",
            metadata={"category": "健康", "subcategory": "运动指南"}
        ),
        Document(
            text="研究发现每天饮用 2-3 杯咖啡可降低 15% 的心血管疾病风险。"
                 "但过量饮用（>6杯）可能增加焦虑和失眠风险。",
            metadata={"category": "健康", "subcategory": "营养研究"}
        ),
    ]

    return tech_docs + finance_docs + health_docs


# ============================================================================
# 第一部分：基础查询引擎
# ============================================================================

def demo_basic_query_engine(documents):
    """基础查询引擎：最简化的 RAG 流程。"""
    print("=" * 60)
    print("【1】基础查询引擎 —— retrieve → synthesize → answer")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings

    Settings.llm = get_mock_llm()
    Settings.embed_model = get_mock_embed_model()

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(
        similarity_top_k=3,           # 检索最相关的 3 个节点
        response_mode="compact",      # 折叠上下文到提示词（默认）
        streaming=False,              # 是否流式输出
        verbose=False,                # 是否打印调试信息
    )

    queries = [
        "iPhone 16 有什么新特性？",
        "最近有什么货币政策变化？",
        "怎样保持心血管健康？",
    ]

    for q in queries:
        start = time.time()
        response = query_engine.query(q)
        elapsed = time.time() - start
        print(f"\n  查询: {q}")
        print(f"  延迟: {elapsed*1000:.0f}ms")
        print(f"  回答: {str(response)[:200]}")
        print(f"  来源节点数: {len(response.source_nodes)}")


# ============================================================================
# 第二部分：RouterQueryEngine —— 智能路由
# ============================================================================

def demo_router_query_engine(documents):
    """路由查询引擎：根据查询意图自动选择最合适的索引。"""
    print("\n" + "=" * 60)
    print("【2】RouterQueryEngine —— 查询意图路由")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings, SummaryIndex
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.selectors import LLMSingleSelector

    Settings.llm = get_mock_llm()
    Settings.embed_model = get_mock_embed_model()

    # 步骤1：按类别分别构建索引
    categories = {}
    for doc in documents:
        cat = doc.metadata.get("category", "通用")
        categories.setdefault(cat, []).append(doc)

    index_map = {}
    for cat, cat_docs in categories.items():
        index_map[cat] = VectorStoreIndex.from_documents(cat_docs)
        print(f"  已构建索引: {cat} ({len(cat_docs)} 文档)")

    # 步骤2：将每个索引用 QueryEngineTool 包装
    tools = []
    for cat, idx in index_map.items():
        tool = QueryEngineTool(
            query_engine=idx.as_query_engine(similarity_top_k=2),
            metadata=ToolMetadata(
                name=f"{cat}_tool",
                description=f"用于查询{cat}相关的问题。"
                            f"当用户问题涉及{cat}领域时使用此工具。",
            ),
        )
        tools.append(tool)

    # 步骤3：创建路由器（LLM 根据 tool description 自动选工具）
    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=tools,
        verbose=False,
    )

    test_queries = [
        ("iPhone 16 的价格是多少？", "应路由到科技"),
        ("央行最近有什么政策？", "应路由到金融"),
        ("每天应该运动多长时间？", "应路由到健康"),
    ]

    for q, expected in test_queries:
        response = router.query(q)
        print(f"\n  查询: {q}")
        print(f"  预期路由: {expected}")
        print(f"  回答: {str(response)[:150]}")

    return router


# ============================================================================
# 第三部分：SubQuestionQueryEngine —— 子问题分解
# ============================================================================

def demo_sub_question_engine(documents):
    """子问题查询引擎：将复杂查询拆分为子问题逐一回答后合并。"""
    print("\n" + "=" * 60)
    print("【3】SubQuestionQueryEngine —— 复杂问题分解")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.core.query_engine import SubQuestionQueryEngine

    Settings.llm = get_mock_llm()
    Settings.embed_model = get_mock_embed_model()

    # 构建全量索引，并用工具包装
    index = VectorStoreIndex.from_documents(documents)
    base_engine = index.as_query_engine(similarity_top_k=3)

    # SubQuestionQueryEngine 需要一个"能回答一切"的基础工具
    query_tool = QueryEngineTool(
        query_engine=base_engine,
        metadata=ToolMetadata(
            name="general_knowledge",
            description="通用的知识库，包含科技、金融、健康等领域的信息。",
        ),
    )

    # 创建子问题引擎
    # LLM 会自动将复杂问题分解为子问题，逐一用 query_tool 回答，再合并
    sub_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[query_tool],
        verbose=False,
    )

    complex_queries = [
        "比较 iPhone 16 的新特性与央行最新货币政策的影响",
        "健康生活方式对心血管的好处，以及最新的科技产品如何帮助监测健康",
    ]

    for q in complex_queries:
        response = sub_engine.query(q)
        print(f"\n  复杂查询: {q}")
        print(f"  综合回答: {str(response)[:250]}")


# ============================================================================
# 第四部分：SQL 查询引擎
# ============================================================================

def demo_sql_query_engine():
    """SQL 查询引擎：自然语言转 SQL，查询结构化数据库。"""
    print("\n" + "=" * 60)
    print("【4】SQL 查询引擎 —— 自然语言 → SQL")
    print("=" * 60)

    import sqlite3
    from llama_index.core import SQLDatabase, Settings
    from llama_index.core.query_engine import NLSQLTableQueryEngine

    # 创建示例数据库
    db_path = "./data/company.db"
    Path("./data").mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 建表
    cursor.executescript("""
        DROP TABLE IF EXISTS employees;
        DROP TABLE IF EXISTS departments;
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT,
            location TEXT
        );
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department_id INTEGER,
            salary REAL,
            hire_date TEXT,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );
        INSERT INTO departments VALUES (1, '技术部', '北京');
        INSERT INTO departments VALUES (2, '市场部', '上海');
        INSERT INTO departments VALUES (3, '销售部', '深圳');
        INSERT INTO employees VALUES (1, '张三', 1, 35000, '2023-01-15');
        INSERT INTO employees VALUES (2, '李四', 1, 42000, '2022-06-01');
        INSERT INTO employees VALUES (3, '王五', 2, 28000, '2023-09-10');
        INSERT INTO employees VALUES (4, '赵六', 2, 31000, '2023-03-20');
        INSERT INTO employees VALUES (5, '钱七', 3, 38000, '2021-11-05');
        INSERT INTO employees VALUES (6, '孙八', 3, 45000, '2020-08-15');
    """)
    conn.commit()

    Settings.llm = get_mock_llm()

    # 创建 SQL Database 对象
    sql_database = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    # 创建 NL-to-SQL 查询引擎
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["employees", "departments"],
    )

    queries = [
        "哪个部门的平均工资最高？",
        "技术部有多少名员工？",
        "列出工资超过35000的员工姓名和部门",
    ]

    for q in queries:
        try:
            response = query_engine.query(q)
            print(f"\n  自然语言: {q}")
            print(f"  查询结果: {str(response)[:200]}")
        except Exception as e:
            print(f"  查询失败: {e}")

    conn.close()


# ============================================================================
# 第五部分：自定义查询引擎
# ============================================================================

def demo_custom_query_engine(documents):
    """自定义查询引擎：实现特定业务逻辑的查询管道。"""
    print("\n" + "=" * 60)
    print("【5】自定义查询引擎 —— 业务逻辑定制")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.response.schema import Response
    from llama_index.core.callbacks import CallbackManager
    from typing import Optional

    Settings.llm = get_mock_llm()
    Settings.embed_model = get_mock_embed_model()

    class AuditQueryEngine(BaseQueryEngine):
        """
        带审计日志的查询引擎。

        在标准 RAG 管道基础上增加：
        1. 查询日志记录
        2. 敏感词过滤
        3. 回答置信度评分
        4. 审计追踪
        """

        def __init__(
            self,
            index: VectorStoreIndex,
            sensitive_words: Optional[List[str]] = None,
            audit_log_path: Optional[str] = None,
        ):
            super().__init__(callback_manager=CallbackManager([]))
            self.index = index
            self.retriever = index.as_retriever(similarity_top_k=3)
            self.sensitive_words = sensitive_words or []
            self.audit_log_path = audit_log_path or "./data/audit.log"
            self.query_count = 0

        def _log_audit(self, query: str, result: str, metadata: Dict):
            """记录审计日志。"""
            import json
            from datetime import datetime
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "result_preview": result[:100],
                "metadata": metadata,
            }
            Path(self.audit_log_path).parent.mkdir(exist_ok=True)
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        def _check_sensitive(self, query: str) -> Optional[str]:
            """敏感词检查。"""
            for word in self.sensitive_words:
                if word in query:
                    return f"查询包含敏感词，已被拦截"
            return None

        def _estimate_confidence(self, nodes) -> float:
            """基于检索节点相似度估算置信度。"""
            if not nodes:
                return 0.0
            scores = [n.score or 0.5 for n in nodes]
            return sum(scores) / len(scores)

        def custom_query(self, query_str: str) -> Response:
            # 1. 敏感词检查
            block_msg = self._check_sensitive(query_str)
            if block_msg:
                return Response(response=block_msg)

            # 2. 检索
            nodes = self.retriever.retrieve(query_str)
            confidence = self._estimate_confidence(nodes)

            # 3. 合成回答（简化：拼接检索内容）
            context = "\n".join([n.text[:150] for n in nodes[:3]])
            answer = f"[审计引擎] 基于 {len(nodes)} 篇文档（置信度 {confidence:.2f}）\n{context}"

            # 4. 审计日志
            self.query_count += 1
            self._log_audit(query_str, answer, {
                "node_count": len(nodes),
                "confidence": confidence,
                "query_id": self.query_count,
            })

            return Response(response=answer, metadata={"confidence": confidence})

        def _query(self, query_bundle):
            # 适配 LlamaIndex 内部接口
            return self.custom_query(query_bundle.query_str)

        async def _aquery(self, query_bundle):
            return self._query(query_bundle)

    # 使用自定义引擎
    index = VectorStoreIndex.from_documents(documents)
    engine = AuditQueryEngine(
        index=index,
        sensitive_words=["攻击", "黑客"],  # 敏感词示例
    )

    queries = [
        "iPhone 16 有什么特点？",
        "如何进行网络攻击？",  # 应被拦截
    ]
    for q in queries:
        response = engine.query(q)
        print(f"\n  查询: {q}")
        print(f"  回答: {str(response)[:200]}")


# ============================================================================
# 第六部分：响应模式对比
# ============================================================================

def demo_response_modes(documents):
    """对比不同响应合成模式的差异。"""
    print("\n" + "=" * 60)
    print("【6】响应模式对比 —— compact/refine/tree_summarize/simple")
    print("=" * 60)

    from llama_index.core import VectorStoreIndex, Settings

    Settings.llm = get_mock_llm()
    Settings.embed_model = get_mock_embed_model()

    index = VectorStoreIndex.from_documents(documents)

    modes = {
        "compact": "先将检索结果压缩到上下文窗口，再一次性生成回答。适合答案分散在多个文档的场景。",
        "refine": "逐个处理检索到的文本块，依次精炼回答。适合需要逐步推演的场景。",
        "tree_summarize": "将文本块作为叶子节点，递归向上摘要。适合超长文档。",
        "simple_summarize": "截断上下文到窗口大小，简单拼接后生成。最快但可能丢失信息。",
        "accumulate": "逐个处理上下文块，累积答案。适合需要完整覆盖的场景。",
        "compact_accumulate": "compact + accumulate 的混合模式。",
    }

    for mode, desc in modes.items():
        print(f"\n  模式: {mode}")
        print(f"    说明: {desc}")
        try:
            engine = index.as_query_engine(
                response_mode=mode,
                similarity_top_k=2,
            )
            response = engine.query("总结文档的主要内容")
            print(f"    输出长度: {len(str(response))} 字符")
        except Exception as e:
            print(f"    [不适用] {e}")


# ============================================================================
# 第七部分：查询引擎最佳实践
# ============================================================================

def print_best_practices():
    """查询引擎使用最佳实践。"""
    print("\n" + "=" * 60)
    print("【查询引擎最佳实践】")
    print("=" * 60)

    tips = """
    1. 【默认选择】90% 的场景用 VectorStoreIndex + compact 模式即可
    2. 【路由优先】多领域知识库优先使用 RouterQueryEngine，比单一大索引效果好
    3. 【子问题分解】问题明显包含"比较"、"分别"、"和"等连接词时，启用 SubQuestion
    4. 【SQL 场景】结构化数据用 NLSQLTableQueryEngine，不要硬塞到文本索引
    5. 【流式输出】用户面向场景务必启用 streaming=True
    6. 【置信度】在 response.metadata 中附带置信度信息，供上游决策
    7. 【缓存】对高频查询启用缓存（ingestion_cache、query_cache）
    8. 【回调】使用 CallbackManager 监控每次查询的 token 消耗和延迟
    """
    print(tips)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("LlamaIndex 查询引擎实战演示")
    print("=" * 60 + "\n")

    docs = prepare_documents()

    demo_basic_query_engine(docs)
    demo_router_query_engine(docs)
    demo_sub_question_engine(docs)
    demo_sql_query_engine()
    demo_custom_query_engine(docs)
    demo_response_modes(docs)
    print_best_practices()

    print("\n[完成] 所有查询引擎演示结束。")
