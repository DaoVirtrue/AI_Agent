#!/usr/bin/env python3
"""
C3-04：Dify API 与应用集成
=============================
Dify 提供完整的 REST API，可以将知识库、工作流、Agent 集成到自己的应用中。

本文件演示：
1. Dify API 认证与基础调用
2. 知识库管理 API
3. 对话/聊天 API（流式与非流式）
4. 工作流 API
5. 上传文件到知识库
6. 反馈与标注 API
7. 将 Dify 集成到 FastAPI 后端

依赖：pip install requests
"""

import os
import json
import time
from typing import Optional, Dict, List, Generator
from pathlib import Path


# ============================================================================
# 第一部分：Dify API 客户端封装
# ============================================================================

class DifyClient:
    """
    Dify API 客户端。

    封装 Dify 平台的核心 API，支持：
    - 知识库管理（创建/上传/检索）
    - 对话/聊天（流式/非流式）
    - 工作流执行
    - 反馈提交
    """

    def __init__(self, base_url: str, api_key: str):
        """
        初始化客户端。

        Args:
            base_url: Dify 服务地址，如 https://api.dify.ai/v1
            api_key: API 密钥（在 Dify 应用设置中生成）
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = self._create_session()

    def _create_session(self):
        """创建带认证的 HTTP 会话。"""
        import requests
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        return session

    # ---- 知识库 API ----

    def create_knowledge_base(self, name: str, description: str = "") -> dict:
        """
        创建知识库。

        实际 API: POST /v1/datasets
        """
        print(f"  [API] 创建知识库: '{name}'")
        # 实际调用：
        # resp = self.session.post(
        #     f"{self.base_url}/datasets",
        #     json={"name": name, "description": description}
        # )
        # return resp.json()
        return {"id": f"kb_{hash(name) % 100000:05d}", "name": name}

    def upload_document(self, kb_id: str, file_path: str,
                        indexing_technique: str = "high_quality") -> str:
        """
        上传文档到知识库。

        Args:
            kb_id: 知识库 ID
            file_path: 文件路径
            indexing_technique: 索引方式 (high_quality / economy)
        """
        print(f"  [API] 上传文档: {file_path} -> kb={kb_id}")
        # 实际调用：
        # with open(file_path, "rb") as f:
        #     resp = self.session.post(
        #         f"{self.base_url}/datasets/{kb_id}/document/create-by-file",
        #         files={"file": f},
        #         data={"indexing_technique": indexing_technique}
        #     )
        # return resp.json()["document"]["id"]
        return f"doc_{Path(file_path).stem}"

    def retrieve(self, kb_id: str, query: str, top_k: int = 5) -> List[dict]:
        """
        从知识库检索。

        实际 API: POST /v1/datasets/{kb_id}/retrieve
        """
        print(f"  [API] 检索: '{query}' (top_k={top_k})")
        # 实际调用：
        # resp = self.session.post(
        #     f"{self.base_url}/datasets/{kb_id}/retrieve",
        #     json={"query": query, "retrieval_model": {
        #         "search_method": "hybrid_search",
        #         "top_k": top_k,
        #         "score_threshold": 0.5,
        #     }}
        # )
        # return resp.json()["records"]
        return [
            {
                "content": f"关于'{query}'的检索结果 {i+1}",
                "score": 0.95 - i * 0.05,
                "document_name": f"文档{i+1}.pdf",
                "segment_position": i + 1,
            }
            for i in range(min(top_k, 3))
        ]

    # ---- 对话/聊天 API ----

    def chat(self, query: str, conversation_id: str = "",
             user: str = "default", inputs: dict = None) -> dict:
        """
        非流式对话。

        实际 API: POST /v1/chat-messages
        """
        print(f"  [API] 对话: '{query}'")
        # 实际调用：
        # resp = self.session.post(
        #     f"{self.base_url}/chat-messages",
        #     json={
        #         "query": query,
        #         "conversation_id": conversation_id,
        #         "user": user,
        #         "inputs": inputs or {},
        #         "response_mode": "blocking",
        #     }
        # )
        # return resp.json()
        return {
            "answer": f"关于'{query}'的回答...",
            "conversation_id": conversation_id or "conv_001",
            "message_id": "msg_001",
        }

    def chat_stream(self, query: str, conversation_id: str = "",
                    user: str = "default", inputs: dict = None) -> Generator:
        """
        流式对话（SSE 流）。

        实际 API: POST /v1/chat-messages (response_mode=streaming)
        """
        print(f"  [API] 流式对话: '{query}'")

        # 实际调用：
        # resp = self.session.post(
        #     f"{self.base_url}/chat-messages",
        #     json={
        #         "query": query,
        #         "conversation_id": conversation_id,
        #         "user": user,
        #         "inputs": inputs or {},
        #         "response_mode": "streaming",
        #     },
        #     stream=True,
        # )
        # for line in resp.iter_lines():
        #     if line.startswith(b"data:"):
        #         yield json.loads(line[5:])

        # 模拟流式输出
        simulated_chunks = [
            "基于", "知识库", "的", "检索", "结果", "，",
            f"关于'{query}'", "的回答", "已经", "生成。"
        ]
        for chunk in simulated_chunks:
            time.sleep(0.05)
            yield {"event": "message", "answer": chunk}

    # ---- 工作流 API ----

    def run_workflow(self, inputs: dict, user: str = "default") -> dict:
        """
        执行工作流（非流式）。

        实际 API: POST /v1/workflows/run
        """
        print(f"  [API] 执行工作流: inputs={inputs}")
        # 实际调用：
        # resp = self.session.post(
        #     f"{self.base_url}/workflows/run",
        #     json={"inputs": inputs, "user": user, "response_mode": "blocking"}
        # )
        # return resp.json()
        return {
            "workflow_run_id": "wfr_001",
            "outputs": {"result": f"工作流执行完成，输入: {inputs}"},
            "elapsed_time": 1.5,
        }

    # ---- 反馈 API ----

    def submit_feedback(self, message_id: str, rating: str,
                        content: str = "") -> dict:
        """
        提交用户反馈（点赞/点踩）。

        rating: "like" / "dislike"

        实际 API: POST /v1/messages/{message_id}/feedbacks
        """
        print(f"  [API] 反馈: message={message_id}, rating={rating}")
        return {"result": "success"}


# ============================================================================
# 第二部分：知识库集成场景
# ============================================================================

def demo_kb_integration():
    """演示知识库管理的完整 API 流程。"""
    print("=" * 60)
    print("【1】知识库管理 API 集成")
    print("=" * 60)

    client = DifyClient(
        base_url=os.environ.get("DIFY_BASE_URL", "https://api.dify.ai/v1"),
        api_key=os.environ.get("DIFY_API_KEY", "app-demo-key"),
    )

    # 创建知识库
    kb = client.create_knowledge_base(
        name="产品技术文档库",
        description="存储产品使用手册、API文档、技术白皮书",
    )
    kb_id = kb["id"]
    print(f"  知识库创建成功: ID={kb_id}")

    # 上传文档
    sample_files = [
        "./data/产品使用手册.pdf",
        "./data/API接口文档.docx",
    ]
    doc_ids = []
    for fp in sample_files:
        Path(fp).parent.mkdir(exist_ok=True)
        if not Path(fp).exists():
            Path(fp).write_text("示例文档内容", encoding="utf-8")
        doc_id = client.upload_document(kb_id, fp)
        doc_ids.append(doc_id)

    # 检索测试
    print("\n  检索测试:")
    queries = [
        "如何配置API接口？",
        "产品支持的部署方式有哪些？",
        "系统的性能参数是什么？",
    ]
    for q in queries:
        results = client.retrieve(kb_id, q, top_k=3)
        print(f"\n    查询: {q}")
        for r in results:
            print(f"      [{r['score']:.2f}] {r['content'][:60]}...")


# ============================================================================
# 第三部分：对话 API 集成
# ============================================================================

def demo_chat_integration():
    """演示对话/聊天 API 的集成方式。"""
    print("\n" + "=" * 60)
    print("【2】对话 API 集成")
    print("=" * 60)

    client = DifyClient(
        base_url=os.environ.get("DIFY_BASE_URL", "https://api.dify.ai/v1"),
        api_key=os.environ.get("DIFY_API_KEY", "app-demo-key"),
    )

    # ---- 非流式对话 ----
    print("\n  >>> 非流式对话模式")
    conversation_id = ""  # 新会话

    questions = [
        "你好，我想了解产品的退款政策",
        "退款需要什么材料？",
        "大概多久能到账？",
    ]

    for q in questions:
        response = client.chat(
            query=q,
            conversation_id=conversation_id,
            user="user_12345",
            inputs={"language": "zh-CN"},
        )
        conversation_id = response["conversation_id"]
        print(f"  用户: {q}")
        print(f"  AI: {response['answer'][:120]}...")
        print(f"  会话ID: {conversation_id}")

    # ---- 流式对话 ----
    print("\n  >>> 流式对话模式")
    print("  用户: 总结一下产品的核心功能")
    print("  AI: ", end="", flush=True)

    for chunk in client.chat_stream(
        query="总结一下产品的核心功能",
        conversation_id=conversation_id,
        user="user_12345",
    ):
        if chunk.get("event") == "message":
            print(chunk.get("answer", ""), end="", flush=True)
    print()


# ============================================================================
# 第四部分：工作流 API 集成
# ============================================================================

def demo_workflow_integration():
    """演示工作流 API 的调用方式。"""
    print("\n" + "=" * 60)
    print("【3】工作流 API 集成")
    print("=" * 60)

    client = DifyClient(
        base_url=os.environ.get("DIFY_BASE_URL", "https://api.dify.ai/v1"),
        api_key=os.environ.get("DIFY_API_KEY", "app-demo-key"),
    )

    # 场景：内容审核工作流
    print("\n  场景：文章内容审核工作流")

    test_articles = [
        {"title": "产品功能介绍", "content": "我们的产品具有以下功能...",
         "author": "张三", "category": "技术"},
        {"title": "优惠活动通知", "content": "限时折扣，全场5折...",
         "author": "李四", "category": "营销"},
    ]

    for article in test_articles:
        result = client.run_workflow(
            inputs={
                "title": article["title"],
                "content": article["content"],
                "author": article["author"],
                "category": article["category"],
                "check_dimensions": ["合规性", "准确性", "可读性"],
            },
            user="editor",
        )
        print(f"\n  待审文章: {article['title']}")
        print(f"  审核结果: {result['outputs'].get('result', 'N/A')[:100]}")
        print(f"  耗时: {result.get('elapsed_time', 0)}s")


# ============================================================================
# 第五部分：集成到 FastAPI 后端
# ============================================================================

def demo_fastapi_integration():
    """展示如何将 Dify 集成到自己的 FastAPI 后端服务。"""
    print("\n" + "=" * 60)
    print("【4】集成到 FastAPI 后端（代码模板）")
    print("=" * 60)

    code_template = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dify_client import DifyClient  # 你封装的客户端

app = FastAPI(title="内部 AI 中台")

# 初始化 Dify 客户端
dify = DifyClient(
    base_url=os.environ["DIFY_BASE_URL"],
    api_key=os.environ["DIFY_API_KEY"],
)

class ChatRequest(BaseModel):
    query: str
    conversation_id: str = ""
    user_id: str = "anonymous"
    app_type: str = "customer_service"  # 路由到不同的 Dify 应用

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list = []

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """统一对话接口"""
    # 根据 app_type 路由到不同的 Dify 应用
    dify_api_key = get_api_key_for_app(request.app_type)

    client = DifyClient(
        base_url=os.environ["DIFY_BASE_URL"],
        api_key=dify_api_key,
    )

    response = client.chat(
        query=request.query,
        conversation_id=request.conversation_id,
        user=request.user_id,
    )

    return ChatResponse(
        answer=response["answer"],
        conversation_id=response["conversation_id"],
        sources=response.get("retriever_resources", []),
    )

@app.post("/api/feedback")
async def feedback(message_id: str, rating: str):
    """用户反馈接口（点赞/点踩）"""
    if rating not in ("like", "dislike"):
        raise HTTPException(400, "rating must be like or dislike")

    dify.submit_feedback(message_id, rating)
    return {"status": "ok"}

@app.post("/api/kb/search")
async def search_knowledge_base(query: str, kb_name: str = "default"):
    """知识库检索接口"""
    kb_id = get_kb_id_by_name(kb_name)
    results = dify.retrieve(kb_id, query, top_k=5)
    return {"results": results}

def get_api_key_for_app(app_type: str) -> str:
    """根据应用类型返回对应的 API Key"""
    keys = {
        "customer_service": os.environ.get("DIFY_CS_KEY"),
        "knowledge_qa": os.environ.get("DIFY_QA_KEY"),
        "content_review": os.environ.get("DIFY_REVIEW_KEY"),
    }
    return keys.get(app_type, os.environ["DIFY_API_KEY"])

def get_kb_id_by_name(name: str) -> str:
    """根据知识库名称获取ID（实际应缓存）"""
    kb_map = {
        "default": "kb_00001",
        "products": "kb_00002",
        "internal": "kb_00003",
    }
    return kb_map.get(name, kb_map["default"])
'''
    print(code_template)


# ============================================================================
# 第六部分：反馈循环集成
# ============================================================================

def demo_feedback_loop():
    """演示如何通过用户反馈持续改进 AI 效果。"""
    print("\n" + "=" * 60)
    print("【5】反馈循环 —— 持续改进 AI")
    print("=" * 60)

    client = DifyClient(
        base_url=os.environ.get("DIFY_BASE_URL", "https://api.dify.ai/v1"),
        api_key=os.environ.get("DIFY_API_KEY", "app-demo-key"),
    )

    print("""
    反馈循环工作流：

    1. 用户提问 → Dify 返回回答（附带 message_id）
    2. 用户反馈 → 调用 feedback API（like / dislike）
    3. Dify 记录反馈 → 在「标注」页面可查看
    4. 运营人员分析：
       - 被点踩的回答 → 人工修正 → 添加到知识库
       - 被点赞的回答 → 作为好的示例
    5. 持续迭代 → 提升整体效果

    代码实现：
    """)

    # 模拟一轮交互
    response = client.chat(query="产品如何收费？", user="test_user")
    message_id = response["message_id"]
    print(f"  用户提问: 产品如何收费？")
    print(f"  AI回答: {response['answer'][:80]}...")
    print(f"  message_id: {message_id}")

    # 用户反馈
    client.submit_feedback(message_id, "like", "回答准确")
    print(f"  用户反馈: 赞")

    # 在 Dify 后台可查看：
    print(f"\n  在 Dify 后台 → 标注 → 可查看反馈历史")
    print(f"  通过反馈数据，发现知识库盲区，持续改进")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("Dify API 与应用集成实战")
    print("=" * 60)

    # 设置演示环境变量
    os.environ.setdefault("DIFY_BASE_URL", "https://api.dify.ai/v1")
    os.environ.setdefault("DIFY_API_KEY", "app-demo-key-12345")

    demo_kb_integration()
    demo_chat_integration()
    demo_workflow_integration()
    demo_fastapi_integration()
    demo_feedback_loop()

    print("\n[完成] Dify API 集成演示结束。")
    print("  提示：替换 DIFY_BASE_URL 和 DIFY_API_KEY 为真实值即可运行。")
