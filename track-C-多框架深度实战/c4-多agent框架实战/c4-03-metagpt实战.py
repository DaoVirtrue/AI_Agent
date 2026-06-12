#!/usr/bin/env python3
"""
C4-03：MetaGPT 实战 —— SOP 驱动的多 Agent 框架
==================================================
MetaGPT 核心理念：模拟软件公司的标准化作业流程（SOP）。
以"产品经理 → 架构师 → 工程师 → QA"的流水线方式协作。

核心概念：
1. Role：角色的标准化行为模板
2. Action：角色可以执行的具体操作
3. Message：角色之间的通信协议
4. Environment：共享工作空间
5. SOP：Standard Operating Procedure（标准作业程序）

与 CrewAI/AutoGen 的区别：
- CrewAI: 灵活角色 + 任务级编排
- AutoGen: 对话驱动 + 自由交流
- MetaGPT: SOP 驱动 + 严格流程 + 结构化输出

依赖：pip install metagpt
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime


# ============================================================================
# 第一部分：MetaGPT 核心概念
# ============================================================================

def explain_metagpt_concepts():
    """解释 MetaGPT 的核心概念和 SOP 模型。"""
    print("=" * 60)
    print("【1】MetaGPT 核心概念 —— SOP 驱动的软件公司")
    print("=" * 60)

    print("""
    MetaGPT 模拟了一个软件公司的组织架构：

    ┌─────────────────────────────────────────────────┐
    │              MetaGPT 软件公司                     │
    │                                                   │
    │  [产品经理 PM]                                    │
    │   ↓ 输出: PRD (产品需求文档)                      │
    │                                                   │
    │  [架构师 Architect]                               │
    │   ↓ 输出: 系统设计文档 + 技术选型                  │
    │                                                   │
    │  [工程师 Engineer]                                │
    │   ↓ 输出: 源代码 + API 文档                       │
    │                                                   │
    │  [QA 测试]                                        │
    │   ↓ 输出: 测试报告 + Bug 列表                     │
    │                                                   │
    │  SOP 流程:                                        │
    │  需求 → 设计 → 编码 → 测试 → 交付                │
    └─────────────────────────────────────────────────┘

    关键设计理念：

    1. 【SOP 标准化】每个角色有预定义的行为模板
       - PM: 需求分析 → 竞品调研 → 撰写 PRD → 评审
       - Architect: 理解需求 → 系统设计 → 技术选型 → 输出设计文档
       - Engineer: 理解设计 → 编写代码 → 自测 → 提交
       - QA: 理解需求+设计 → 编写测试用例 → 执行测试 → 报告

    2. 【结构化输出】每个阶段的输出是结构化的文档
       - PRD 有固定格式（项目名、需求、功能列表、优先级...）
       - 设计文档有固定格式（架构图、模块划分、接口定义...）
       - 代码有规范（命名、注释、错误处理...）

    3. 【消息驱动】角色之间通过 Message 对象通信
       - Message(content, role, cause_by, ...)
       - 每个 Role 有 _watch() 方法监听感兴趣的 Message
       - 消息传递形成信息流

    4. 【环境共享】所有角色共享一个 Environment
       - 共享记忆 (Memory)
       - 共享知识 (Knowledge)
       - 历史消息可追溯
    """)


# ============================================================================
# 第二部分：MetaGPT Role 架构
# ============================================================================

class SimulatedRole:
    """模拟 MetaGPT Role 的简化实现，帮助理解其原理。"""

    def __init__(self, name: str, profile: str, goal: str):
        self.name = name
        self.profile = profile
        self.goal = goal
        self.inbox: List[dict] = []   # 接收的消息
        self.outbox: List[dict] = []  # 发送的消息
        self.memory: List[dict] = []   # 记忆

    def _watch(self, message_types: List[str]):
        """注册感兴趣的消息类型（MetaGPT 核心机制）。"""
        self._watched_types = message_types

    def _observe(self, message: dict):
        """观察环境中的消息，过滤感兴趣的类型。"""
        if message.get("type") in getattr(self, "_watched_types", []):
            self.inbox.append(message)
            self.memory.append(message)

    def _think(self) -> Optional[dict]:
        """思考：根据收到的消息决定下一步行动。"""
        # MetaGPT 中，这一步由 LLM 完成
        if not self.inbox:
            return None
        latest_msg = self.inbox[-1]
        return {"action": f"分析: {latest_msg.get('content', '')[:50]}"}

    def _act(self) -> Optional[dict]:
        """行动：执行具体的工作。"""
        thought = self._think()
        if not thought:
            return None
        # 在 MetaGPT 中，Action 是预定义的类
        result = {
            "type": "output",
            "from": self.name,
            "content": f"[{self.name}] 基于输入完成的工作成果",
            "timestamp": datetime.now().isoformat(),
        }
        self.outbox.append(result)
        self.inbox.clear()
        return result

    def run(self, initial_message: dict = None):
        """运行一次 think → act 循环。"""
        if initial_message:
            self.inbox.append(initial_message)
        return self._act()


def demo_role_architecture():
    """演示 MetaGPT Role 的工作机制。"""
    print("\n" + "=" * 60)
    print("【2】MetaGPT Role 架构模拟")
    print("=" * 60)

    # 创建角色
    pm = SimulatedRole(
        name="产品经理",
        profile="经验丰富的产品经理，擅长需求分析和竞品研究",
        goal="输出一份完整的 RAG 智能客服系统 PRD",
    )
    pm._watch(["requirement", "feedback"])

    architect = SimulatedRole(
        name="架构师",
        profile="资深系统架构师，精通分布式系统和 AI 架构",
        goal="根据 PRD 设计技术架构方案",
    )
    architect._watch(["PRD"])

    # 模拟 SOP 流程
    print("\n  === 第1步：需求输入 ===")
    # 老板提出需求
    requirement = {
        "type": "requirement",
        "content": "我们需要构建一个智能客服系统，支持多渠道接入、"
                   "知识库检索、多轮对话、人工转接。预计日活1万用户。",
        "from": "老板",
    }

    # PM 接收需求，开始工作
    prd_output = pm.run(requirement)
    if prd_output:
        print(f"  {pm.name} 输出: {prd_output['content'][:100]}...")

    print("\n  === 第2步：PRD 传递给架构师 ===")
    prd_message = {
        "type": "PRD",
        "content": prd_output["content"] if prd_output else "PRD 内容...",
        "from": pm.name,
    }
    design_output = architect.run(prd_message)
    if design_output:
        print(f"  {architect.name} 输出: {design_output['content'][:100]}...")

    print("\n  这就是 MetaGPT 的核心流程：")
    print("  消息被发送到 Environment → Role._observe() 过滤 → "
          "Role._think() 思考 → Role._act() 执行 → 输出新消息")


# ============================================================================
# 第三部分：完整 SOP Pipeline 模拟
# ============================================================================

def demo_sop_pipeline():
    """模拟 MetaGPT 完整的 SOP 流水线：PM → Architect → Engineer → QA。"""
    print("\n" + "=" * 60)
    print("【3】完整 SOP Pipeline 模拟")
    print("=" * 60)

    # 定义标准化的输出模板
    PRD_TEMPLATE = """
    # 产品需求文档 (PRD)
    ## 项目名称
    {project_name}
    ## 需求概述
    {overview}
    ## 功能列表
    {features}
    ## 优先级
    {priority}
    ## 约束条件
    {constraints}
    """

    DESIGN_TEMPLATE = """
    # 系统设计文档
    ## 架构概述
    {architecture}
    ## 技术选型
    {tech_stack}
    ## 模块划分
    {modules}
    ## 接口定义
    {interfaces}
    ## 数据流
    {data_flow}
    """

    CODE_TEMPLATE = """
    # {module_name} 模块
    ## 功能说明
    {description}
    ## 核心代码
    ```python
    {code}
    ```
    ## API 文档
    {api_doc}
    """

    TEST_TEMPLATE = """
    # 测试报告
    ## 测试范围
    {scope}
    ## 测试用例
    {test_cases}
    ## 通过率
    {pass_rate}
    ## 发现的问题
    {issues}
    """

    # 模拟整个 Pipeline
    print("\n  >>> 项目启动：智能客服 RAG 系统")

    # Step 1: PM → PRD
    print("\n  [Step 1] 产品经理 → 撰写 PRD")
    prd = PRD_TEMPLATE.format(
        project_name="智能客服 RAG 系统",
        overview="基于 RAG 技术的多渠道智能客服系统，支持知识库检索、多轮对话",
        features="""1. 多渠道接入（Web/微信/API）
2. 知识库管理（上传/检索/更新）
3. 多轮对话引擎
4. 意图识别与路由
5. 人工转接
6. 数据统计看板""",
        priority="P0: 知识库检索、多轮对话 | P1: 多渠道接入、意图识别 | P2: 数据看板",
        constraints="需要支持1000并发，P99延迟<3秒，支持私有化部署",
    )
    print(f"    输出: PRD ({len(prd)} 字符)")

    # Step 2: Architect → Design
    print("\n  [Step 2] 架构师 → 系统设计")
    design = DESIGN_TEMPLATE.format(
        architecture="微服务架构：API Gateway → 对话管理 → 检索服务 → LLM 调用",
        tech_stack="""- LLM: GPT-4o (主) + DeepSeek-V3 (降级)
- 嵌入: BGE-M3
- 向量库: Milvus
- 搜索引擎: Elasticsearch
- 缓存: Redis
- 消息队列: RabbitMQ
- 数据库: PostgreSQL""",
        modules="""1. gateway-service (API 网关/认证/限流)
2. dialog-service (对话管理/上下文)
3. retrieval-service (知识库检索/重排序)
4. llm-service (LLM 调用/缓存)
5. knowledge-service (知识库管理)
6. analytics-service (数据统计)""",
        interfaces="REST API (内部) / WebSocket (实时对话) / gRPC (服务间通信)",
        data_flow="用户消息 → Gateway → Dialog → Retrieval → LLM → Response",
    )
    print(f"    输出: 设计文档 ({len(design)} 字符)")

    # Step 3: Engineer → Code
    print("\n  [Step 3] 工程师 → 编码实现")
    code = CODE_TEMPLATE.format(
        module_name="retrieval-service",
        description="知识库检索服务，支持向量检索+全文检索混合",
        code="""
class RetrievalService:
    def __init__(self, vector_store, text_index, reranker):
        self.vector_store = vector_store
        self.text_index = text_index
        self.reranker = reranker

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        # 向量检索
        vector_results = self.vector_store.search(query, top_k=20)
        # 全文检索
        text_results = self.text_index.search(query, top_k=20)
        # RRF 融合
        merged = self.rrf_merge(vector_results, text_results)
        # 重排序
        return self.reranker.rerank(query, merged)[:top_k]
        """,
        api_doc="POST /api/v1/retrieval/search  {query, top_k, filters}",
    )
    print(f"    输出: 代码 ({len(code)} 字符)")

    # Step 4: QA → Test
    print("\n  [Step 4] QA → 测试报告")
    test_report = TEST_TEMPLATE.format(
        scope="检索服务的单元测试、集成测试、性能测试",
        test_cases="""1. 单关键词检索 (通过)
2. 多关键词混合检索 (通过)
3. 空查询边界测试 (通过)
4. 大数据量压测 (通过，10000文档P99<500ms)
5. 并发检索测试 (通过，100并发无错误)""",
        pass_rate="95% (19/20 通过)",
        issues="1. [中] 超长文本查询时延迟增加至800ms，需优化",
    )
    print(f"    输出: 测试报告 ({len(test_report)} 字符)")

    # 汇总
    print("\n  === SOP Pipeline 完成 ===")
    print(f"  总产出: PRD + 设计文档 + 代码 + 测试报告")
    print(f"  总字符数: {len(prd) + len(design) + len(code) + len(test_report)}")


# ============================================================================
# 第四部分：MetaGPT 代码示例（真实框架调用）
# ============================================================================

def demo_real_metagpt_code():
    """展示真实的 MetaGPT 代码调用方式。"""
    print("\n" + "=" * 60)
    print("【4】MetaGPT 真实调用代码")
    print("=" * 60)

    code_example = '''
from metagpt.roles import (
    Architect,
    Engineer,
    ProductManager,
    QaEngineer,
)
from metagpt.team import Team
from metagpt.actions import WritePRD, WriteDesign, WriteCode
import asyncio

async def build_rag_system():
    """
    用 MetaGPT 构建一个 RAG 系统的完整流程。
    """
    # 定义团队
    team = Team()

    # 添加角色（SOP 驱动的预定义角色）
    team.hire([
        ProductManager(
            name="Alice",
            profile="资深产品经理，专注企业AI产品",
            goal="输出高质量的 RAG 系统 PRD",
        ),
        Architect(
            name="Bob",
            profile="资深架构师，精通分布式和AI架构",
            goal="设计高性能、可扩展的 RAG 系统架构",
        ),
        Engineer(
            name="Charlie",
            profile="全栈工程师，Python/Go 专家",
            goal="实现高质量的 RAG 系统代码",
        ),
        QaEngineer(
            name="Diana",
            profile="资深 QA，注重细节和边界条件",
            goal="确保系统质量达到生产标准",
        ),
    ])

    # 投入项目需求
    team.invest(idea="构建一个企业级 RAG 知识库问答系统")

    # 按 SOP 顺序执行项目
    await team.run_project()

    # 获取产出物
    prd = team.get_product("PRD")
    design = team.get_product("SystemDesign")
    code = team.get_product("Code")
    tests = team.get_product("Tests")

    return prd, design, code, tests

# MetaGPT 内部执行流程（简化）：
# 1. PM._think() → 需要写PRD → _act() → WritePRD.run()
# 2. PM 发布 PRD Message → Environment
# 3. Architect._observe(PRD) → _think() → _act() → WriteDesign.run()
# 4. Architect 发布 Design Message → Environment
# 5. Engineer._observe(Design) → _think() → _act() → WriteCode.run()
# 6. QA._observe(All) → _think() → _act() → WriteTest.run()

if __name__ == "__main__":
    asyncio.run(build_rag_system())
'''
    print(code_example)


# ============================================================================
# 第五部分：MetaGPT 的优缺点
# ============================================================================

def print_pros_cons():
    """MetaGPT 的优缺点分析。"""
    print("\n" + "=" * 60)
    print("【5】MetaGPT 优缺点分析")
    print("=" * 60)

    print("""
    【优点】
    ✅ SOP 驱动：流程可预测、可控，输出质量稳定
    ✅ 结构化输出：每个阶段的产出物格式固定，便于集成
    ✅ 角色专业化：每个角色有深度领域知识（通过 system prompt）
    ✅ 从需求到代码：真正实现"一句话生成项目"
    ✅ 适合结构化工程任务：API 开发、脚本编写、测试用例生成

    【缺点】
    ❌ 灵活性差：不适合开放式、探索性任务
    ❌ 流程固定：难以处理需求变更和迭代
    ❌ 成本高：每个角色都是 LLM 调用，4 角色 × N 轮 = 大量 Token
    ❌ 学习曲线：概念较多（Role/Action/Message/Environment/SOP）
    ❌ 生态较小：社区不如 CrewAI 和 AutoGen 活跃
    ❌ 输出不可控：LLM 生成的代码质量不稳定，仍需人工审查

    【适用场景】
    ✅ 从零开始的项目原型生成
    ✅ API 接口的快速实现
    ✅ 标准化测试用例生成
    ✅ 技术文档自动生成
    ✅ 代码脚手架搭建

    【不适用场景】
    ❌ 已有大型代码库的增量开发
    ❌ 需要人机频繁协作的任务
    ❌ 对代码质量有极高要求的场景
    ❌ 需要实时响应的生产系统
    ❌ 非结构化或创意性任务
    """)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("MetaGPT 多 Agent 框架实战")
    print("=" * 60)
    print("核心理念: SOP 驱动的软件公司模拟\n")

    explain_metagpt_concepts()
    demo_role_architecture()
    demo_sop_pipeline()
    demo_real_metagpt_code()
    print_pros_cons()

    print("\n[完成] MetaGPT 实战演示结束。")
    print("  提示: pip install metagpt 安装框架。")
    print("  运行: export OPENAI_API_KEY='sk-xxx' && python 本文件")
