#!/usr/bin/env python3
"""
C4-01：CrewAI 实战 —— 多 Agent 协作框架
===========================================
CrewAI 核心理念：Role（角色）→ Goal（目标）→ Backstory（背景故事）
→ Task（任务）→ Crew（团队）。角色驱动、任务级编排。

本文件演示：
1. 基础 CrewAI：Role/Goal/Backstory → Task → Crew
2. 工具集成：Agent 调用自定义工具
3. 顺序执行 vs 层级执行
4. 完整多 Agent 写作团队实战

依赖：pip install crewai crewai-tools
"""

import os
from typing import List


# ============================================================================
# 第一部分：确保环境可用
# ============================================================================

def check_environment():
    """检查 CrewAI 环境。"""
    try:
        import crewai
        print(f"[OK] CrewAI 版本: {crewai.__version__}")
    except ImportError:
        print("[WARN] CrewAI 未安装。pip install crewai")
        return False
    return True


# ============================================================================
# 第二部分：CrewAI 核心概念演示
# ============================================================================

def demo_crewai_basics():
    """
    CrewAI 基础：15 行代码启动一个多 Agent 团队。

    核心元素：
    1. Agent: Role + Goal + Backstory
    2. Task: description + expected_output + agent
    3. Crew: agents + tasks + process
    """
    print("=" * 60)
    print("【1】CrewAI 基础 —— 15行启动多Agent团队")
    print("=" * 60)

    if not check_environment():
        print("  [跳过] 环境未就绪")
        return

    from crewai import Agent, Task, Crew, Process

    # ---- 定义 Agent ----
    researcher = Agent(
        role="资深技术研究员",
        goal="深入研究技术主题，提供全面、准确、有深度的分析报告",
        backstory="你是一名拥有15年经验的技术研究员，曾在顶级科技公司工作。"
                  "你擅长从多个维度分析技术问题，总能发现别人忽略的关键细节。",
        verbose=True,
        allow_delegation=False,
        llm=os.environ.get("OPENAI_API_KEY") and "gpt-4o" or None,
    )

    writer = Agent(
        role="技术文档撰写专家",
        goal="将复杂的技术分析转化为清晰、易懂、结构化的技术文档",
        backstory="你是一名资深技术文档工程师，擅长将晦涩的技术概念"
                  "转化为通俗易懂的表达。你的文档总是条理清晰、重点突出。",
        verbose=True,
        allow_delegation=False,
        llm=os.environ.get("OPENAI_API_KEY") and "gpt-4o" or None,
    )

    reviewer = Agent(
        role="技术审核专家",
        goal="审核技术文档的准确性、完整性和可读性，提出改进建议",
        backstory="你是一名严谨的技术审核专家，曾在顶级学术期刊担任审稿人。"
                  "你对技术细节有极强的敏感性，能够发现最微小的错误。",
        verbose=True,
        allow_delegation=True,
        llm=os.environ.get("OPENAI_API_KEY") and "gpt-4o" or None,
    )

    # ---- 定义 Task ----
    research_task = Task(
        description="""研究 RAG（检索增强生成）技术的当前发展状况。
        请涵盖以下方面：
        1. RAG 的核心原理和演进历程
        2. 2025-2026 年的最新进展（GraphRAG、Agentic RAG、多模态 RAG）
        3. 主流框架对比（LangChain、LlamaIndex、RAGFlow、Dify）
        4. 企业落地的关键挑战和解决方案""",
        expected_output="一份结构化的研究报告，包含核心原理、最新进展、"
                        "框架对比表和企业落地建议，总计约 800-1000 字。",
        agent=researcher,
    )

    writing_task = Task(
        description="""基于研究员提供的研究报告，撰写一篇面向技术团队的
        技术博客文章。文章应当：
        1. 开头引人入胜，点明 RAG 的重要性
        2. 主体部分清晰易读，配有对比表格
        3. 结尾给出实用建议
        4. 语言通俗但不失专业性""",
        expected_output="一篇约 1000-1200 字的技术博客，格式规范，内容完整。",
        agent=writer,
        context=[research_task],  # 依赖研究任务的输出
    )

    review_task = Task(
        description="""审核技术博客文章的质量：
        1. 技术准确性：是否有事实错误
        2. 逻辑完整性：是否覆盖了所有要点
        3. 可读性：是否易于目标读者理解
        4. 给出具体的改进建议""",
        expected_output="审核报告，包含：评分（1-10分）、具体问题和改进建议。",
        agent=reviewer,
        context=[writing_task],  # 依赖写作任务的输出
    )

    # ---- 定义 Crew（团队）----
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,  # 顺序执行
        verbose=True,
    )

    print(f"\n  Crew 已创建:")
    print(f"    Agent 数: {len(crew.agents)}")
    print(f"    Task 数: {len(crew.tasks)}")
    print(f"    执行模式: {crew.process}")

    # ---- 执行（如果有 API Key 才真正调用）----
    if os.environ.get("OPENAI_API_KEY"):
        print("\n  >>> 开始执行 CrewAI 团队任务...")
        result = crew.kickoff()
        print(f"\n  === 最终结果 ===")
        print(str(result)[:500])
    else:
        print("\n  [INFO] 设置 OPENAI_API_KEY 环境变量以执行真实调用。")
        print("  当前为演示模式，展示了 CrewAI 的完整定义。")


# ============================================================================
# 第三部分：带工具的 Agent
# ============================================================================

def demo_crewai_with_tools():
    """Agent 集成自定义工具。"""
    print("\n" + "=" * 60)
    print("【2】CrewAI Agent + 工具集成")
    print("=" * 60)

    if not check_environment():
        return

    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool

    # 自定义搜索工具（模拟）
    class SearchKnowledgeBaseTool(BaseTool):
        name: str = "知识库搜索"
        description: str = "搜索企业内部知识库，获取技术文档信息。输入搜索关键词。"

        def _run(self, query: str) -> str:
            """模拟知识库搜索。"""
            kb = {
                "RAG": "RAG（检索增强生成）技术架构包含：文档加载、分段、嵌入、"
                       "向量存储、检索、答案生成六个核心环节。",
                "向量数据库": "主流向量数据库：Milvus（分布式）、Qdrant（高性能）、"
                            "Pinecone（全托管）、ChromaDB（轻量级）。",
                "GraphRAG": "GraphRAG 将知识图谱与 RAG 结合，解决多跳推理问题。"
                           "核心步骤：实体提取→关系构建→图存储→图遍历检索。",
            }
            for key, value in kb.items():
                if key.lower() in query.lower():
                    return value
            return f"未找到关于'{query}'的相关信息。"

    # 自定义计算工具
    class CostCalculatorTool(BaseTool):
        name: str = "成本计算器"
        description: str = "计算 RAG 系统的运行成本。输入格式：'日查询量,模型名称'"

        def _run(self, params: str) -> str:
            parts = params.split(",")
            queries = int(parts[0].strip()) if parts else 1000
            model = parts[1].strip() if len(parts) > 1 else "gpt-4o"

            cost_per_1k = {"gpt-4o": 0.005, "gpt-4o-mini": 0.0006,
                          "claude": 0.008, "deepseek": 0.001}
            unit_cost = cost_per_1k.get(model, 0.005)
            total = queries * unit_cost

            return (f"模型: {model}\n"
                    f"日查询量: {queries}\n"
                    f"单次成本: ${unit_cost:.4f}\n"
                    f"日成本: ${total:.2f}\n"
                    f"月成本(30天): ${total*30:.2f}")

    # 带工具的 Agent
    analyst = Agent(
        role="技术成本分析师",
        goal="分析 AI 技术方案的成本效益，为企业决策提供数据支持",
        backstory="你是一名经验丰富的技术成本分析师，"
                  "擅长评估技术方案的投资回报率。",
        tools=[SearchKnowledgeBaseTool(), CostCalculatorTool()],
        verbose=True,
        allow_delegation=False,
        llm=os.environ.get("OPENAI_API_KEY") and "gpt-4o" or None,
    )

    analysis_task = Task(
        description="""分析在日均10000次查询的场景下，
        使用 RAG 技术的成本效益。请：
        1. 搜索 RAG 技术的核心架构
        2. 计算使用 gpt-4o 模型的日成本和月成本
        3. 评估主要成本来源
        4. 给出成本优化建议""",
        expected_output="一份成本分析报告，包含技术概述、成本计算和优化建议。",
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=True,
    )

    print(f"\n  Agent '{analyst.role}' 已配置 {len(analyst.tools)} 个工具:")
    for tool in analyst.tools:
        print(f"    - {tool.name}: {tool.description}")

    if os.environ.get("OPENAI_API_KEY"):
        result = crew.kickoff()
        print(f"\n  === 结果 ===\n{str(result)[:500]}")
    else:
        print("\n  [INFO] 设置 OPENAI_API_KEY 执行真实调用。")


# ============================================================================
# 第四部分：层级执行模式
# ============================================================================

def demo_hierarchical_process():
    """层级执行：管理者 Agent 分配任务给执行 Agent。"""
    print("\n" + "=" * 60)
    print("【3】层级执行模式 —— Manager 分配任务")
    print("=" * 60)

    print("""
    CrewAI 支持两种执行模式：

    模式 A：Sequential（顺序执行）
      Task 1 → Task 2 → Task 3
      适合：步骤明确、前后依赖的任务

    模式 B：Hierarchical（层级执行）
              [Manager Agent]
                 /  |  \\
           [Worker A] [Worker B] [Worker C]
      适合：任务需要动态分配、Manager 根据结果调整策略

    层级模式的特点：
    - Manager LLM 负责决策任务分配
    - Worker LLM 负责执行具体任务
    - Manager 审查 Worker 输出，决定是否需要重做
    - 适合复杂、不确定性的任务

    配置方式：
    crew = Crew(
        agents=[manager, worker1, worker2],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(model="gpt-4o"),  # Manager 用更强的模型
    )
    """)


# ============================================================================
# 第五部分：完整实战 —— 技术写作团队
# ============================================================================

def demo_full_writing_crew():
    """完整演示：用 CrewAI 构建技术内容创作团队。"""
    print("\n" + "=" * 60)
    print("【4】完整实战 —— 技术内容创作团队")
    print("=" * 60)

    if not check_environment():
        return

    from crewai import Agent, Task, Crew, Process

    # 定义团队
    agents_config = [
        {
            "role": "技术选题策划师",
            "goal": "根据目标读者和行业趋势，策划有价值的技术文章选题",
            "backstory": "资深技术编辑，对技术趋势有敏锐嗅觉",
        },
        {
            "role": "技术内容研究员",
            "goal": "深入收集技术资料，确保内容技术准确",
            "backstory": "前软件工程师，现在是技术研究者",
        },
        {
            "role": "技术文章撰稿人",
            "goal": "将研究资料转化为引人入胜的技术文章",
            "backstory": "技术博主，擅长用通俗语言解释技术",
        },
        {
            "role": "内容质量审核员",
            "goal": "确保文章质量达到发布标准",
            "backstory": "前出版社编辑，对文字质量要求极高",
        },
    ]

    agents = []
    for cfg in agents_config:
        agent = Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=os.environ.get("OPENAI_API_KEY") and "gpt-4o" or None,
        )
        agents.append(agent)

    tasks = [
        Task(
            description="策划一篇关于 AI Agent 技术的文章选题",
            expected_output="选题方案：标题、大纲、目标读者、核心观点",
            agent=agents[0],
        ),
        Task(
            description="根据选题方案收集技术资料和案例",
            expected_output="研究资料汇总：技术原理、行业案例、数据支持",
            agent=agents[1],
            context=[],
        ),
        Task(
            description="撰写完整的技术文章",
            expected_output="完整文章：引言、正文、案例、结论",
            agent=agents[2],
            context=[],
        ),
        Task(
            description="审核文章质量",
            expected_output="审核报告：评分、修改建议",
            agent=agents[3],
            context=[],
        ),
    ]

    # 设置任务依赖
    tasks[1].context = [tasks[0]]
    tasks[2].context = [tasks[1]]
    tasks[3].context = [tasks[2]]

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    print(f"\n  团队规模: {len(agents)} 个 Agent")
    print(f"  任务数: {len(tasks)} 个 Task")
    print(f"  执行模式: Sequential（任务依赖链）")
    print(f"\n  执行流程:")
    print(f"    {agents[0].role} → {agents[1].role} → "
          f"{agents[2].role} → {agents[3].role}")

    if os.environ.get("OPENAI_API_KEY"):
        print("\n  >>> 开始执行...")
        result = crew.kickoff()
        print(f"\n  === 最终文章 ===")
        print(str(result)[:800])
    else:
        print("\n  [INFO] 设置 OPENAI_API_KEY 执行真实调用。")


# ============================================================================
# 第六部分：CrewAI 最佳实践
# ============================================================================

def print_best_practices():
    """CrewAI 使用最佳实践。"""
    print("\n" + "=" * 60)
    print("【CrewAI 最佳实践】")
    print("=" * 60)

    tips = """
    1. 【角色设计】
       - Role 要具体（"Python专家"比"程序员"好）
       - Backstory 要详细（给 LLM 足够的行为指导）
       - 不同 Agent 的 Role 要有清晰边界，避免重叠

    2. 【任务设计】
       - description 要具体明确
       - expected_output 要给出格式和长度要求
       - 利用 context 参数建立任务依赖链

    3. 【执行模式选择】
       - 任务步骤明确 → sequential
       - 需要动态决策 → hierarchical
       - Manager Agent 要用更强的模型（如 GPT-4o）

    4. 【成本控制】
       - 测试阶段用 GPT-4o-mini
       - 生产环境关键 Agent 用 GPT-4o/Claude
       - 设置 max_iter 限制循环次数

    5. 【工具设计】
       - 工具描述要包含使用场景（"当你需要X时使用此工具"）
       - 工具返回值要结构化（方便 Agent 解析）
       - 工具需要错误处理（返回友好错误信息）

    6. 【调试技巧】
       - verbose=True 查看完整执行过程
       - 先用 1-2 个 Agent 调试，再扩展到多 Agent
       - 检查 Task 的 context 是否正确传递

    7. 【生产部署】
       - 使用环境变量管理 API Key
       - 记录每次 Crew 执行的日志
       - 设置超时和重试机制
    """
    print(tips)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("CrewAI 多 Agent 框架实战")
    print("=" * 60)

    demo_crewai_basics()
    demo_crewai_with_tools()
    demo_hierarchical_process()
    demo_full_writing_crew()
    print_best_practices()

    print("\n[完成] CrewAI 实战演示结束。")
    print("  提示: export OPENAI_API_KEY='sk-xxx' 后重新运行以执行真实调用。")
