#!/usr/bin/env python3
"""
C4-02：AutoGen 实战 —— 对话驱动多 Agent 框架
================================================
AutoGen（Microsoft）核心理念：ConversableAgent（可对话代理）
+ GroupChat（群聊管理）+ UserProxyAgent（人机协同）。

与 CrewAI 的区别：
- CrewAI: 角色驱动、任务级编排（Role/Goal/Task）
- AutoGen: 对话驱动、Agent 间自由对话（ConversableAgent）

本文件演示：
1. 基础 ConversableAgent：双人对话模式
2. GroupChat：多人协作群聊
3. UserProxyAgent：人机协同（HITL）
4. 工具集成：注册函数为 Agent 工具
5. RAG 场景：检索 Agent + 生成 Agent 协作

依赖：pip install autogen-agentchat autogen-ext[openai]
"""

import os
from typing import List, Dict, Any


# ============================================================================
# 第一部分：环境检查
# ============================================================================

def check_environment():
    """检查 AutoGen 环境。"""
    try:
        import autogen
        print(f"[OK] AutoGen 可用")
        return True
    except ImportError:
        print("[WARN] AutoGen 未安装。pip install autogen-agentchat autogen-ext[openai]")
        return False


# ============================================================================
# 第二部分：基础 ConversableAgent —— 双人对话
# ============================================================================

def demo_two_agent_chat():
    """
    双 Agent 对话模式。

    AutoGen 的核心是 ConversableAgent，每个 Agent 都可以：
    1. 接收消息 (receive)
    2. 生成回复 (generate_reply)
    3. 发送消息 (send)

    两个 Agent 可以自主对话，直到任务完成或达到最大轮次。
    """
    print("=" * 60)
    print("【1】基础双 Agent 对话")
    print("=" * 60)

    if not check_environment():
        return

    from autogen import ConversableAgent

    # 配置 LLM
    llm_config = {
        "config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY", "")}],
        "temperature": 0.1,
        "timeout": 60,
    }

    # 创建两个 Agent
    assistant = ConversableAgent(
        name="技术助手",
        system_message="你是一个专业的 RAG 技术专家。"
                       "用中文回答，给出具体的技术建议。"
                       "每次回答后，询问用户是否有进一步的问题。",
        llm_config=llm_config,
        human_input_mode="NEVER",  # 不需要人类输入
    )

    user_proxy = ConversableAgent(
        name="用户代理",
        system_message="你代表一个正在学习 RAG 技术的开发者。"
                       "提出具体的、有深度的问题。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print(f"  Agent '{assistant.name}' 已创建")
    print(f"  Agent '{user_proxy.name}' 已创建")
    print(f"  对话模式: 双向自由对话")

    # 发起对话
    if os.environ.get("OPENAI_API_KEY"):
        print("\n  >>> 开始对话...")
        chat_result = user_proxy.initiate_chat(
            assistant,
            message="你好！我想了解 RAG 系统中如何评估检索质量，"
                    "有哪些常用的评估指标？",
            max_turns=3,  # 最多 3 轮对话
        )
        print(f"\n  对话结束。共 {len(chat_result.chat_history)} 条消息。")
    else:
        print("\n  [INFO] 设置 OPENAI_API_KEY 执行真实对话。")


# ============================================================================
# 第三部分：GroupChat —— 多人协作群聊
# ============================================================================

def demo_group_chat():
    """
    GroupChat：多个 Agent 在群聊中协作。

    GroupChatManager 负责管理对话流程：
    - 决定下一个发言的 Agent 是谁
    - 选择发言人策略："auto"（自动）、"round_robin"（轮询）、"random"（随机）
    """
    print("\n" + "=" * 60)
    print("【2】GroupChat 多人协作群聊")
    print("=" * 60)

    if not check_environment():
        return

    from autogen import ConversableAgent, GroupChat, GroupChatManager

    llm_config = {
        "config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY", "")}],
        "temperature": 0.1,
    }

    # 创建专家团队
    product_manager = ConversableAgent(
        name="产品经理",
        system_message="你是产品经理。你负责定义产品需求，"
                       "确保技术方案符合业务目标。"
                       "发言时关注用户价值和商业可行性。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    architect = ConversableAgent(
        name="架构师",
        system_message="你是系统架构师。你负责设计技术方案，"
                       "评估技术可行性和扩展性。"
                       "发言时关注系统设计、技术选型和架构决策。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    engineer = ConversableAgent(
        name="工程师",
        system_message="你是后端工程师。你负责评估实现细节，"
                       "包括 API 设计、数据库选型、性能优化。"
                       "发言时关注可实现性和技术细节。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    reviewer = ConversableAgent(
        name="质量审核员",
        system_message="你是质量审核员。你负责审查技术方案，"
                       "提出风险和问题。"
                       "发言时关注安全性、可靠性和可维护性。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 创建群聊
    group_chat = GroupChat(
        agents=[product_manager, architect, engineer, reviewer],
        messages=[],
        max_round=6,
        speaker_selection_method="auto",  # 自动选择下一个发言人
        allow_repeat_speaker=False,       # 不允许同一个 Agent 连续发言
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    print(f"  群聊成员: {[a.name for a in group_chat.agents]}")
    print(f"  最大轮次: {group_chat.max_round}")
    print(f"  发言人选择: 自动")

    if os.environ.get("OPENAI_API_KEY"):
        print("\n  >>> 开始群聊...")
        user_proxy = ConversableAgent(
            name="老板",
            system_message="你代表公司高层，提出一个 AI 平台建设需求。",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        user_proxy.initiate_chat(
            manager,
            message="我们需要构建一个企业级 RAG 知识库平台，"
                    "服务 5000 名员工，预算 200 万。请给出技术方案。",
        )
    else:
        print("\n  [INFO] 设置 OPENAI_API_KEY 执行真实群聊。")


# ============================================================================
# 第四部分：UserProxyAgent —— 人机协同（HITL）
# ============================================================================

def demo_human_in_the_loop():
    """
    UserProxyAgent：在关键决策点引入人类判断。

    HITL 的典型场景：
    - AI 生成初稿 → 人类审核通过/修改
    - AI 检测到高风险操作 → 请求人类确认
    - AI 遇到不确定的情况 → 向人类求助
    """
    print("\n" + "=" * 60)
    print("【3】UserProxyAgent —— 人机协同（HITL）")
    print("=" * 60)

    print("""
    AutoGen 的 HITL 机制：

    1. human_input_mode 参数控制人机交互：
       - "NEVER": 完全自主，不询问人类
       - "TERMINATE": 只在终止时询问
       - "ALWAYS": 每次都询问人类

    2. 实际代码模式：

    from autogen import UserProxyAgent

    # 创建人类代理
    user_proxy = UserProxyAgent(
        name="人类审核员",
        human_input_mode="TERMINATE",  # 在关键决策点暂停
        max_consecutive_auto_reply=3,  # 自动回复最多3轮
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
        },
    )

    # AI 助手在需要人类决策时会暂停
    assistant = ConversableAgent(
        name="AI助手",
        system_message="你是 AI 助手。在以下情况请求人类决策：
                        1. 金额超过 1000 元的操作
                        2. 涉及用户隐私数据的访问
                        3. 不确定的技术决策",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 典型交互流程：
    # 1. AI: "我建议删除旧的用户数据，确认吗？(TERMINATE)"
    # 2. 人类: "确认，但先备份" 或 "取消"
    # 3. AI: 根据人类反馈继续执行

    HITL 最佳实践：
    - 将人类判断点放在真正需要决策的位置
    - 给人类提供充分的上文信息做决策
    - 设置合理的超时时间
    - 提供"全部批准"的快捷选项（对批量操作）
    """)


# ============================================================================
# 第五部分：工具集成 —— 注册函数为 Agent 工具
# ============================================================================

def demo_tool_integration():
    """
    将 Python 函数注册为 Agent 的工具。

    AutoGen 原生支持将函数注册为工具：
    - @user_proxy.register_for_execution()  # 注册到执行者
    - @assistant.register_for_llm()         # 注册到 LLM 的描述
    """
    print("\n" + "=" * 60)
    print("【4】工具集成 —— 注册函数为 Agent 工具")
    print("=" * 60)

    # 定义工具函数（模拟数据库操作）
    def search_knowledge_base(query: str, top_k: int = 3) -> str:
        """
        搜索知识库。

        Args:
            query: 搜索关键词
            top_k: 返回结果数
        """
        import json
        results = [
            {"title": f"RAG 架构设计指南", "score": 0.95,
             "content": "RAG 系统包括文档加载、分段、嵌入..."},
            {"title": f"向量数据库选型对比", "score": 0.87,
             "content": "Milvus、Qdrant、Pinecone 的对比..."},
            {"title": f"RAG 性能优化实践", "score": 0.82,
             "content": "通过缓存、重排序、查询改写提升 RAG 性能..."},
        ]
        return json.dumps(results[:top_k], ensure_ascii=False)

    def calculate_cost(model: str, queries_per_day: int) -> str:
        """
        计算 LLM 使用成本。

        Args:
            model: 模型名称 (gpt-4o / gpt-4o-mini / claude-sonnet)
            queries_per_day: 日均查询次数
        """
        prices = {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.0006,
            "claude-sonnet": 0.003,
        }
        unit_price = prices.get(model, 0.005)
        daily = unit_price * queries_per_day
        return f"日成本: ${daily:.2f}, 月成本: ${daily*30:.2f}"

    def get_system_status() -> str:
        """获取系统运行状态。"""
        return "系统状态: 正常 | GPU使用率: 45% | 队列长度: 12 | 平均延迟: 230ms"

    # 展示工具注册方式
    print("""
    工具注册模式：

    from autogen import ConversableAgent, register_function

    # 创建 Agent
    assistant = ConversableAgent(
        name="技术助手",
        system_message="你是技术助手，可以使用工具回答用户问题。",
        llm_config=llm_config,
    )

    user_proxy = ConversableAgent(
        name="用户",
        human_input_mode="NEVER",
    )

    # 注册工具
    register_function(
        search_knowledge_base,
        caller=assistant,      # 谁可以调用（LLM 看到工具描述）
        executor=user_proxy,   # 谁执行（实际运行代码）
        name="search_kb",      # 工具名称
        description="搜索知识库，获取技术文档信息",  # 工具描述
    )

    register_function(
        calculate_cost,
        caller=assistant,
        executor=user_proxy,
        name="calc_cost",
        description="计算 LLM API 使用成本",
    )

    register_function(
        get_system_status,
        caller=assistant,
        executor=user_proxy,
        name="sys_status",
        description="获取系统运行状态",
    )
    """)

    # 直接演示工具调用
    print("\n  >>> 本地演示工具调用（无需 API Key）:")
    print(f"  知识库搜索: {search_knowledge_base('RAG 架构', top_k=2)[:120]}")
    print(f"  成本计算: {calculate_cost('gpt-4o', 5000)}")
    print(f"  系统状态: {get_system_status()}")


# ============================================================================
# 第六部分：RAG 场景：检索 Agent + 生成 Agent
# ============================================================================

def demo_rag_two_agent():
    """双 Agent RAG：检索专家 + 回答专家。"""
    print("\n" + "=" * 60)
    print("【5】RAG 双 Agent 协作 —— 检索 + 生成")
    print("=" * 60)

    # 模拟知识库
    KNOWLEDGE_BASE = {
        "RAG架构": "RAG 由文档加载器、嵌入模型、向量数据库、检索器、生成器五部分组成。",
        "嵌入模型": "常用嵌入模型：OpenAI text-embedding-3-small (1536维)、"
                    "BGE-M3 (1024维)、Cohere embed-v3 (1024维)。",
        "向量数据库": "Qdrant (Rust,高性能)、Milvus (分布式)、Pinecone (全托管)。",
        "重排序": "用 Cohere Rerank 或 BGE-Reranker 对初检结果精细排序。",
        "评估指标": "Hit Rate, MRR, NDCG, Faithfulness, Answer Relevancy。",
    }

    class RetrievalAgent:
        """检索 Agent：负责从知识库中找到相关信息。"""

        def retrieve(self, query: str) -> List[Dict]:
            results = []
            for key, content in KNOWLEDGE_BASE.items():
                if any(word in query for word in key):
                    results.append({"source": key, "content": content, "score": 0.9})
                elif any(word in query.lower() for word in content.lower()[:20]):
                    results.append({"source": key, "content": content, "score": 0.7})
            # 按分数降序
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:3]

    class GenerationAgent:
        """生成 Agent：基于检索结果生成回答。"""

        def generate(self, query: str, context: List[Dict]) -> str:
            if not context:
                return "抱歉，未找到与您问题相关的信息。请尝试换个说法。"

            # 构建上下文
            ctx = "\n".join(
                f"[来源: {c['source']}] {c['content']}"
                for c in context
            )

            # 模拟 LLM 生成
            answer = f"【基于 {len(context)} 条信息来源】\n\n"
            answer += f"关于您的问题'{query}'：\n\n"
            answer += ctx
            answer += f"\n\n---\n以上信息均来自知识库，请参考来源标注。"
            return answer

    # 演示协作
    retriever = RetrievalAgent()
    generator = GenerationAgent()

    queries = [
        "RAG架构的主要组成部分",
        "如何选择合适的嵌入模型",
        "怎样评估检索质量",
    ]

    for q in queries:
        print(f"\n  用户: {q}")
        context = retriever.retrieve(q)
        print(f"  [检索Agent] 找到 {len(context)} 条相关信息")
        answer = generator.generate(q, context)
        print(f"  [生成Agent] {answer[:200]}...")


# ============================================================================
# 第七部分：AutoGen 最佳实践
# ============================================================================

def print_best_practices():
    """AutoGen 使用最佳实践。"""
    print("\n" + "=" * 60)
    print("【AutoGen 最佳实践】")
    print("=" * 60)

    tips = """
    1. 【Agent 设计】
       - system_message 要详细（定义角色、行为、边界）
       - 不同 Agent 的 system_message 不要重叠
       - 在 system_message 中指定停止条件

    2. 【对话管理】
       - max_turns 设置合理的上限（防止无限循环）
       - speaker_selection_method="auto" 适合开放式讨论
       - speaker_selection_method="round_robin" 适合流程化讨论
       - 设置 is_termination_msg 判断对话结束

    3. 【工具注册】
       - 函数必须有清晰的 docstring（AutoGen 用它生成工具描述）
       - 参数类型提示要完整（用于 schema 生成）
       - caller=assistant（LLM 可见）, executor=user_proxy（执行）

    4. 【HITL 策略】
       - 只在关键决策点启用 human_input_mode="ALWAYS"
       - 在 system_message 中明确何时需要人类决策
       - 给人类提供足够的上下文和选项

    5. 【成本控制】
       - 使用 max_consecutive_auto_reply 限制自动回复次数
       - 用 gpt-4o-mini 做简单 Agent，gpt-4o 做关键 Agent
       - 缓存重复查询的对话结果

    6. 【与 CrewAI 的选择】
       - 需要结构化任务流程 → CrewAI
       - 需要自由对话/头脑风暴 → AutoGen GroupChat
       - 需要强人机协同 → AutoGen
       - 两者可以混合使用
    """
    print(tips)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("AutoGen 多 Agent 框架实战")
    print("=" * 60)

    demo_two_agent_chat()
    demo_group_chat()
    demo_human_in_the_loop()
    demo_tool_integration()
    demo_rag_two_agent()
    print_best_practices()

    print("\n[完成] AutoGen 实战演示结束。")
    print("  提示: export OPENAI_API_KEY='sk-xxx' 后重新运行。")
