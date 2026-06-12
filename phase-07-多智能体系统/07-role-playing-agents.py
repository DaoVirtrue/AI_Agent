#!/usr/bin/env python3
"""
Phase 07 - 多智能体系统：07 - 角色扮演式Agent（Role-Playing Agents）

Persona 定义（个性特征、专业技能、沟通风格）。
角色特定的工具访问权限。
示例：软件开发团队（PM、Developer、QA、DevOps）协作完成项目。

架构图：
  ┌─────────────────────────────────────────────────────────┐
  │                    项目协作空间                            │
  │                                                         │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
  │  │    PM    │  │ Developer│  │    QA    │  │  DevOps  ││
  │  │ (产品经理)│  │ (开发者) │  │ (测试)   │  │ (运维)   ││
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘│
  │       │             │             │             │       │
  │       └─────────────┼─────────────┼─────────────┘       │
  │                     │             │                      │
  │         ┌───────────▼─────────────▼───────────┐          │
  │         │         共享项目上下文 (Story)        │          │
  │         │  - requirements: [...]              │          │
  │         │  - code: {...}                      │          │
  │         │  - test_results: [...]              │          │
  │         │  - deployment_status: ...           │          │
  │         └────────────────────────────────────┘          │
  └─────────────────────────────────────────────────────────┘
"""

import json
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================================
# Persona 定义
# ============================================================================

@dataclass
class Persona:
    """
    角色人设定义。

    Attributes:
        name: 角色名称
        role: 职位/角色
        personality_traits: 个性特征列表
        expertise: 专业领域
        communication_style: 沟通风格描述
        catchphrases: 口头禅
        decision_style: 决策风格 (analytical/intuitive/collaborative/cautious)
        tone: 语气 (formal/casual/technical/empathetic)
    """
    name: str
    role: str
    personality_traits: list[str] = field(default_factory=list)
    expertise: list[str] = field(default_factory=list)
    communication_style: str = "专业、直接"
    catchphrases: list[str] = field(default_factory=list)
    decision_style: str = "analytical"
    tone: str = "professional"

    def generate_system_prompt(self) -> str:
        """根据Persona生成系统提示词。"""
        return (
            f"你是 {self.name}，担任 {self.role}。\n\n"
            f"## 个性特征\n"
            f"{chr(10).join(f'- {trait}' for trait in self.personality_traits)}\n\n"
            f"## 专业领域\n"
            f"{chr(10).join(f'- {exp}' for exp in self.expertise)}\n\n"
            f"## 沟通风格\n"
            f"{self.communication_style}\n\n"
            f"## 决策风格\n"
            f"{self.decision_style}\n\n"
            f"## 沟通指南\n"
            f"- 始终以 {self.name}（{self.role}）的身份发言\n"
            f"- 使用 {self.tone} 的语气\n"
            f"- 在你的专业领域内给出具体、可操作的建议\n"
            f"- 如果超出你的专业范围，明确说明并建议找对应角色\n"
        )


# ============================================================================
# 工具定义
# ============================================================================

@dataclass
class Tool:
    """
    Agent可用的工具定义。

    Attributes:
        name: 工具名称
        description: 工具描述
        params_schema: 参数JSON Schema
        access_roles: 哪些角色可以使用此工具
    """
    name: str
    description: str
    params_schema: dict
    access_roles: list[str]  # 允许使用的角色列表


# ---- 预定义工具 ----

def create_standard_tools() -> list[Tool]:
    """创建软件开发团队的标准化工具集。"""
    return [
        Tool(
            name="create_requirement",
            description="创建新的需求文档",
            params_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "需求标题"},
                    "description": {"type": "string", "description": "需求详细描述"},
                    "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "description", "priority"],
            },
            access_roles=["PM"],
        ),
        Tool(
            name="write_code",
            description="编写或修改代码",
            params_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "code": {"type": "string"},
                    "language": {"type": "string"},
                    "commit_message": {"type": "string"},
                },
                "required": ["code", "language"],
            },
            access_roles=["Developer"],
        ),
        Tool(
            name="create_test",
            description="创建测试用例",
            params_schema={
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "test_type": {"type": "string", "enum": ["unit", "integration", "e2e"]},
                    "test_code": {"type": "string"},
                    "expected_result": {"type": "string"},
                },
                "required": ["test_name", "test_type", "expected_result"],
            },
            access_roles=["QA"],
        ),
        Tool(
            name="run_tests",
            description="运行测试套件",
            params_schema={
                "type": "object",
                "properties": {
                    "test_suite": {"type": "string"},
                    "environment": {"type": "string", "enum": ["dev", "staging", "production"]},
                },
                "required": ["test_suite"],
            },
            access_roles=["QA", "DevOps"],
        ),
        Tool(
            name="deploy_service",
            description="部署服务到指定环境",
            params_schema={
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "version": {"type": "string"},
                    "environment": {"type": "string", "enum": ["dev", "staging", "production"]},
                    "strategy": {"type": "string", "enum": ["blue-green", "rolling", "canary"]},
                },
                "required": ["service_name", "version", "environment"],
            },
            access_roles=["DevOps"],
        ),
        Tool(
            name="review_code",
            description="审查代码并提交审阅意见",
            params_schema={
                "type": "object",
                "properties": {
                    "pr_number": {"type": "string"},
                    "review_comments": {"type": "array", "items": {"type": "string"}},
                    "decision": {"type": "string", "enum": ["approve", "request_changes", "comment"]},
                },
                "required": ["pr_number", "decision"],
            },
            access_roles=["Developer", "QA"],
        ),
        Tool(
            name="monitor_service",
            description="监控服务运行状态",
            params_schema={
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "time_range": {"type": "string"},
                },
                "required": ["service_name"],
            },
            access_roles=["DevOps"],
        ),
    ]


class ToolRegistry:
    """
    工具注册表：管理工具定义和权限检查。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具。"""
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        """批量注册工具。"""
        for tool in tools:
            self.register(tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具定义。"""
        return self._tools.get(name)

    def list_tools_for_role(self, role: str) -> list[Tool]:
        """列出某角色可以使用的所有工具。"""
        return [
            tool for tool in self._tools.values()
            if role in tool.access_roles
        ]

    def can_access(self, tool_name: str, role: str) -> bool:
        """检查角色是否有权使用某工具。"""
        tool = self._tools.get(tool_name)
        return tool is not None and role in tool.access_roles


# ============================================================================
# 团队Agent
# ============================================================================

class TeamAgent(ABC):
    """
    团队Agent基类。
    封装了Persona + 工具访问 + 协作逻辑。
    """

    def __init__(self, persona: Persona, tool_registry: ToolRegistry):
        self.persona = persona
        self.tool_registry = tool_registry
        self.my_tools = tool_registry.list_tools_for_role(persona.role)
        self.conversation_history: list[dict] = []
        self.completed_tasks: list[str] = []

    @property
    def role(self) -> str:
        return self.persona.role

    @property
    def name(self) -> str:
        return self.persona.name

    def use_tool(self, tool_name: str, **params) -> dict:
        """
        使用工具（带权限检查）。

        Args:
            tool_name: 工具名称
            **params: 工具参数

        Returns:
            工具执行结果

        Raises:
            PermissionError: 如果角色无权使用此工具
        """
        if not self.tool_registry.can_access(tool_name, self.role):
            raise PermissionError(
                f"{self.name}({self.role}) 无权使用工具 '{tool_name}'。"
                f"可用工具: {[t.name for t in self.my_tools]}"
            )

        # 模拟工具执行
        return self._execute_tool(tool_name, params)

    def _execute_tool(self, tool_name: str, params: dict) -> dict:
        """模拟工具执行（生产环境替换为真实工具）。"""
        # 模拟执行延迟
        time.sleep(random.uniform(0.05, 0.2))

        tool_handlers = {
            "create_requirement": self._tool_create_requirement,
            "write_code": self._tool_write_code,
            "create_test": self._tool_create_test,
            "run_tests": self._tool_run_tests,
            "deploy_service": self._tool_deploy_service,
            "review_code": self._tool_review_code,
            "monitor_service": self._tool_monitor_service,
        }

        handler = tool_handlers.get(tool_name, self._tool_default)
        result = handler(params)
        return {"tool": tool_name, "params": params, "result": result, "executor": self.name}

    @abstractmethod
    def respond_to(self, message: str, context: dict = None) -> str:
        """
        响应用户消息或其他Agent的消息。

        Args:
            message: 收到的消息
            context: 项目上下文

        Returns:
            响应内容
        """
        ...

    # ---- 工具模拟实现 ----

    def _tool_create_requirement(self, params: dict) -> str:
        return f"已创建需求: '{params.get('title', 'Untitled')}' [优先级: {params.get('priority', 'P2')}]"

    def _tool_write_code(self, params: dict) -> str:
        return f"已编写 {params.get('language', 'python')} 代码 ({len(params.get('code', ''))} 字符)"

    def _tool_create_test(self, params: dict) -> str:
        return f"已创建 {params.get('test_type', 'unit')} 测试: '{params.get('test_name', 'Unnamed')}'"

    def _tool_run_tests(self, params: dict) -> str:
        passed = random.randint(7, 10)
        total = 10
        return f"测试结果: {passed}/{total} 通过 ({params.get('test_suite', 'all')})"

    def _tool_deploy_service(self, params: dict) -> str:
        return f"已部署 {params.get('service_name', 'unknown')} v{params.get('version', '1.0')} → {params.get('environment', 'dev')}"

    def _tool_review_code(self, params: dict) -> str:
        return f"PR #{params.get('pr_number', '?')} 审查完成: {params.get('decision', 'comment')}"

    def _tool_monitor_service(self, params: dict) -> str:
        return f"{params.get('service_name')} 运行正常 (CPU: 45%, Mem: 62%, Latency P99: 120ms)"

    def _tool_default(self, params: dict) -> str:
        return f"已执行工具操作"


class PMAgent(TeamAgent):
    """产品经理Agent。"""

    def respond_to(self, message: str, context: dict = None) -> str:
        self.conversation_history.append({"role": "user", "content": message})

        if "需求" in message or "requirement" in message:
            result = self.use_tool(
                "create_requirement",
                title="用户提出的新功能需求",
                description=message[:200],
                priority="P1",
                acceptance_criteria=["功能完整可用", "用户体验流畅", "性能达标"],
            )
            response = (
                f"[{self.name} - PM] 收到需求！我已经创建了需求文档。\n"
                f"  📋 {result['result']}\n"
                f"  💡 按我的经验，这个需求的核心价值在于用户体验的提升。"
                f"建议开发团队在实现时关注以下三点：\n"
                f"  1. 最小可行产品（MVP）优先\n"
                f"  2. 用户反馈闭环\n"
                f"  3. 可衡量的成功指标"
            )
        elif "进度" in message or "status" in message:
            response = (
                f"[{self.name} - PM] 当前项目进度概览：\n"
                f"  ✓ 需求阶段：已完成\n"
                f"  ✓ 设计评审：已完成\n"
                f"  → 开发阶段：进行中（预计还需3天）\n"
                f"  ○ 测试阶段：待开始\n"
                f"  ○ 部署发布：待开始\n\n"
                f"  ⚠️ 风险项：开发团队反馈第三方API集成可能延期1天，已纳入缓冲时间。"
            )
        else:
            response = (
                f"[{self.name} - PM] 收到。作为产品经理，我从用户价值角度分析这个问题：\n"
                f"  这个需求的本质是解决用户的_____痛点。\n"
                f"  建议我们先做用户调研，确认优先级后再排入Sprint。\n"
                f"  #{random.choice(self.persona.catchphrases or ['关注用户价值'])}"
            )

        self.conversation_history.append({"role": "assistant", "content": response})
        return response


class DeveloperAgent(TeamAgent):
    """开发者Agent。"""

    def respond_to(self, message: str, context: dict = None) -> str:
        self.conversation_history.append({"role": "user", "content": message})

        if "代码" in message or "实现" in message or "code" in message.lower():
            sample_code = (
                "class FeatureService:\n"
                "    def __init__(self, config: Config):\n"
                "        self.config = config\n"
                "        self.cache = RedisClient(config.redis_url)\n\n"
                "    async def process(self, data: dict) -> Result:\n"
                "        # 输入验证\n"
                "        validated = self._validate(data)\n"
                "        # 业务逻辑\n"
                "        result = await self._execute(validated)\n"
                "        # 缓存结果\n"
                "        await self.cache.set(f'result:{result.id}', result, ttl=300)\n"
                "        return result"
            )
            result = self.use_tool(
                "write_code",
                file_path="src/services/feature_service.py",
                code=sample_code,
                language="python",
                commit_message="feat: 实现FeatureService核心逻辑",
            )
            response = (
                f"[{self.name} - Developer] 代码已实现。\n"
                f"  💻 {result['result']}\n"
                f"  技术选型说明：使用 async/await 异步模式，Redis缓存降低数据库压力。\n"
                f"  预估性能：QPS > 1000，P99 < 200ms。\n"
                f"  ⚠️ 注意：请在配置文件中设置 REDIS_URL 环境变量。"
            )
        elif "bug" in message.lower() or "修" in message or "fix" in message.lower():
            response = (
                f"[{self.name} - Developer] Bug分析中...\n"
                f"  🔍 根因定位：空指针异常，输入未做nil检查（第47行）\n"
                f"  🔧 修复方案：添加 guard clause + 单元测试\n"
                f"  ✅ 已提交修复 PR #1287\n"
                f"  💬 建议QA补充边界值测试用例。"
            )
        elif "审查" in message or "review" in message.lower():
            result = self.use_tool(
                "review_code",
                pr_number="1287",
                review_comments=["✅ 代码逻辑正确", "⚠️ 建议添加类型注解", "📝 文档字符串可以更详细"],
                decision="approve",
            )
            response = (
                f"[{self.name} - Developer] 代码审查完成。\n"
                f"  📝 {result['result']}\n"
                f"  总体评价：代码质量良好，通过了所有检查项。"
            )
        else:
            response = (
                f"[{self.name} - Developer] 从技术实现角度看：\n"
                f"  技术可行性：✅ 可行\n"
                f"  预估工作量：3-5人天\n"
                f"  技术风险：中等（第三方API集成存在不确定性）\n"
                f"  建议技术栈：Python 3.12 + FastAPI + PostgreSQL + Redis\n"
                f"  #{random.choice(self.persona.catchphrases or ['代码即文档'])}"
            )

        self.conversation_history.append({"role": "assistant", "content": response})
        return response


class QAAgent(TeamAgent):
    """测试工程师Agent。"""

    def respond_to(self, message: str, context: dict = None) -> str:
        self.conversation_history.append({"role": "user", "content": message})

        if "测试" in message or "test" in message.lower():
            test_result = self.use_tool(
                "create_test",
                test_name="FeatureService_Integration_Test",
                test_type="integration",
                test_code="...",
                expected_result="所有API端点返回200 OK",
            )
            run_result = self.use_tool(
                "run_tests",
                test_suite="FeatureService",
                environment="dev",
            )
            response = (
                f"[{self.name} - QA] 测试执行完成。\n"
                f"  🧪 {test_result['result']}\n"
                f"  📊 {run_result['result']}\n\n"
                f"  测试总结：\n"
                f"  - 单元测试：85/85 ✅\n"
                f"  - 集成测试：12/12 ✅\n"
                f"  - 端到端测试：5/5 ✅\n"
                f"  - 性能测试：P99 < 200ms ✅\n\n"
                f"  📋 质量门禁：全部通过，可以进入发布阶段！\n"
                f"  💡 建议：补充一个边缘场景测试（超大输入数据）"
            )
        elif "bug" in message.lower() or "缺陷" in message:
            response = (
                f"[{self.name} - QA] 缺陷跟踪：\n"
                f"  🐛 BUG-001: 空输入导致500错误（严重度：High）→ 已分配Developer\n"
                f"  🐛 BUG-002: 日期格式不一致（严重度：Medium）→ 待分配\n"
                f"  📊 当前缺陷密度: 0.8 bug/KLOC（行业平均水平为1.0）\n"
                f"  📈 修复率: 85%（23/27已修复）"
            )
        else:
            response = (
                f"[{self.name} - QA] 从质量保障角度：\n"
                f"  建议在以下场景补充测试覆盖：\n"
                f"  1. 边界条件测试\n"
                f"  2. 并发压力测试\n"
                f"  3. 安全渗透测试\n"
                f"  4. 用户验收测试（UAT）\n"
                f"  #{random.choice(self.persona.catchphrases or ['质量是设计出来的'])}"
            )

        self.conversation_history.append({"role": "assistant", "content": response})
        return response


class DevOpsAgent(TeamAgent):
    """运维工程师Agent。"""

    def respond_to(self, message: str, context: dict = None) -> str:
        self.conversation_history.append({"role": "user", "content": message})

        if "部署" in message or "deploy" in message.lower():
            result = self.use_tool(
                "deploy_service",
                service_name="feature-service",
                version="2.1.0",
                environment="staging",
                strategy="blue-green",
            )
            monitor = self.use_tool(
                "monitor_service",
                service_name="feature-service",
                metrics=["cpu", "memory", "latency", "error_rate"],
                time_range="last_15m",
            )
            response = (
                f"[{self.name} - DevOps] 部署完成。\n"
                f"  🚀 {result['result']}\n"
                f"  📊 {monitor['result']}\n\n"
                f"  部署详情：\n"
                f"  - 策略：蓝绿部署（零停机时间）\n"
                f"  - 新版本流量：10%（灰度验证中）\n"
                f"  - 自动回滚条件：错误率 > 1% 或 P99 > 500ms\n"
                f"  - 健康检查：所有探针通过 ✅"
            )
        elif "监控" in message or "告警" in message or "monitor" in message.lower():
            response = (
                f"[{self.name} - DevOps] 系统监控面板：\n"
                f"  📊 CPU: 45% (正常范围 < 70%)\n"
                f"  💾 内存: 62% (正常范围 < 80%)\n"
                f"  ⏱️ 延迟P99: 120ms (SLO < 200ms) ✅\n"
                f"  ⚠️ 错误率: 0.15% (接近告警阈值 0.2%)\n"
                f"  📈 QPS: 850 (峰值容量 2000)\n\n"
                f"  建议：关注错误率趋势，如果持续上升需要回滚。"
            )
        elif "回滚" in message or "rollback" in message.lower():
            response = (
                f"[{self.name} - DevOps] 回滚操作：\n"
                f"  ⏪ 已将 feature-service 从 v2.1.0 回滚至 v2.0.5\n"
                f"  ✅ 回滚完成，无流量中断\n"
                f"  📋 事后复盘建议：\n"
                f"  1. 增强预发布环境测试\n"
                f"  2. 添加金丝雀发布阶段（1%→10%→50%→100%）\n"
                f"  3. 完善告警规则，降低错误率告警阈值至0.1%"
            )
        else:
            response = (
                f"[{self.name} - DevOps] 从运维角度评估：\n"
                f"  ✅ 基础设施：满足需求（Kubernetes集群可用资源充足）\n"
                f"  ✅ CI/CD：流水线配置就绪\n"
                f"  ✅ 监控：Prometheus + Grafana 已覆盖\n"
                f"  ⚠️ 注意：日志保留策略需要调整（当前7天→建议30天）\n"
                f"  #{random.choice(self.persona.catchphrases or ['稳定压倒一切'])}"
            )

        self.conversation_history.append({"role": "assistant", "content": response})
        return response


# ============================================================================
# 团队协作调度器
# ============================================================================

class TeamOrchestrator:
    """
    团队协作调度器：管理角色扮演Agent的交互。
    """

    def __init__(self, name: str = "DevTeam"):
        self.name = name
        self.agents: dict[str, TeamAgent] = {}
        self.project_story: dict = {
            "requirements": [],
            "code_changes": [],
            "test_results": [],
            "deployments": [],
            "conversation": [],
        }

    def add_agent(self, agent: TeamAgent) -> "TeamOrchestrator":
        """添加团队成员。"""
        self.agents[agent.role] = agent
        return self

    def get_agent(self, role: str) -> Optional[TeamAgent]:
        """按角色获取Agent。"""
        return self.agents.get(role)

    def list_agents(self) -> list[str]:
        """列出所有成员。"""
        return [
            f"{agent.name} ({agent.role}) - 专长: {', '.join(agent.persona.expertise[:3])}"
            for agent in self.agents.values()
        ]

    def broadcast(
        self,
        message: str,
        sender_role: str = None,
        verbose: bool = True,
    ) -> dict[str, str]:
        """
        向所有团队成员广播消息。

        Args:
            message: 消息内容
            sender_role: 发送者角色（可选）
            verbose: 是否打印日志

        Returns:
            所有Agent的响应字典
        """
        responses = {}

        sender_name = self.agents[sender_role].name if sender_role else "系统"

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  📢 [{sender_name}] 广播消息: {message[:100]}...")
            print(f"{'─' * 60}")

        for role, agent in self.agents.items():
            if role == sender_role:
                continue  # 不回复自己

            response = agent.respond_to(message, self.project_story)
            responses[role] = response

            if verbose:
                print(f"  [{agent.name} - {role}]")
                for line in response.split('\n')[:5]:
                    print(f"    {line}")
                if len(response.split('\n')) > 5:
                    print(f"    ...")

        self.project_story["conversation"].append({
            "sender": sender_role or "system",
            "message": message,
            "responses": responses,
            "timestamp": time.time(),
        })

        return responses

    def direct_message(
        self,
        from_role: str,
        to_role: str,
        message: str,
        verbose: bool = True,
    ) -> str:
        """
        发送定向消息给特定角色。

        Args:
            from_role: 发送者角色
            to_role: 接收者角色
            message: 消息内容
            verbose: 是否打印日志

        Returns:
            接收者的响应
        """
        sender = self.agents.get(from_role)
        receiver = self.agents.get(to_role)

        if not receiver:
            return f"错误：找不到角色 '{to_role}'"

        sender_name = sender.name if sender else "系统"

        if verbose:
            print(f"  💬 [{sender_name} → {receiver.name}({to_role})] {message[:80]}...")

        response = receiver.respond_to(message, self.project_story)

        if verbose:
            preview = response[:200].replace('\n', ' | ')
            print(f"    ← {preview}...")

        return response

    def get_project_summary(self) -> dict:
        """获取项目摘要。"""
        return {
            "team": self.name,
            "members": self.list_agents(),
            "conversation_turns": len(self.project_story["conversation"]),
            "total_requirements": len(self.project_story["requirements"]),
            "total_code_changes": len(self.project_story["code_changes"]),
            "total_test_sessions": len(self.project_story["test_results"]),
            "total_deployments": len(self.project_story["deployments"]),
        }


# ============================================================================
# 预定义团队
# ============================================================================

def create_dev_team(tool_registry: ToolRegistry) -> TeamOrchestrator:
    """创建标准软件开发团队。"""
    team = TeamOrchestrator(name="AlphaDevTeam")

    # PM
    team.add_agent(PMAgent(
        persona=Persona(
            name="张明",
            role="PM",
            personality_traits=["注重用户价值", "数据驱动", "善于沟通", "组织能力强"],
            expertise=["需求分析", "产品战略", "用户研究", "敏捷管理"],
            communication_style="清晰有条理，善用数据和用户故事",
            catchphrases=["用户才是最终裁判", "数据不会说谎", "MVP优先"],
            decision_style="collaborative",
            tone="professional",
        ),
        tool_registry=tool_registry,
    ))

    # Developer
    team.add_agent(DeveloperAgent(
        persona=Persona(
            name="李华",
            role="Developer",
            personality_traits=["技术精湛", "追求代码质量", "乐于助人", "专注细节"],
            expertise=["Python", "Go", "微服务", "系统设计", "性能优化"],
            communication_style="技术化但易懂，喜欢用代码示例说话",
            catchphrases=["代码即文档", "先写测试", "Keep It Simple"],
            decision_style="analytical",
            tone="technical",
        ),
        tool_registry=tool_registry,
    ))

    # QA
    team.add_agent(QAAgent(
        persona=Persona(
            name="王芳",
            role="QA",
            personality_traits=["细致严谨", "善于发现边缘案例", "系统性思维", "追求零缺陷"],
            expertise=["自动化测试", "性能测试", "安全测试", "测试策略"],
            communication_style="条理清晰，以数据和证据说话",
            catchphrases=["质量是设计出来的", "没有测试的代码就是遗留代码"],
            decision_style="cautious",
            tone="professional",
        ),
        tool_registry=tool_registry,
    ))

    # DevOps
    team.add_agent(DevOpsAgent(
        persona=Persona(
            name="陈刚",
            role="DevOps",
            personality_traits=["冷静沉稳", "系统思维", "注重稳定性", "快速响应"],
            expertise=["Kubernetes", "CI/CD", "监控告警", "云原生"],
            communication_style="直接明了，关注风险和影响范围",
            catchphrases=["稳定压倒一切", "自动化一切", "测量才能管理"],
            decision_style="cautious",
            tone="technical",
        ),
        tool_registry=tool_registry,
    ))

    return team


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  多智能体系统 - 角色扮演式Agent演示")
    print("=" * 72)

    # ---- 初始化 ----

    # 创建工具注册表
    tools = create_standard_tools()
    tool_registry = ToolRegistry()
    tool_registry.register_all(tools)

    print(f"\n📋 已注册 {len(tools)} 个工具：")
    for tool in tools:
        roles_str = ", ".join(tool.access_roles)
        print(f"  🔧 {tool.name} [{roles_str}]")

    # 创建团队
    team = create_dev_team(tool_registry)

    print(f"\n👥 团队 '{team.name}' 已组建：")
    for member in team.list_agents():
        print(f"  {member}")

    # ---- Demo 1: 完整工作流（模拟Sprint周期） ----

    print("\n" + "=" * 72)
    print("  Demo 1: 完整Sprint工作流模拟")
    print("=" * 72)

    # Step 1: PM宣布新需求
    print("\n  ─── Step 1: PM宣布新需求 ───")
    responses = team.broadcast(
        "我们收到了一个重要需求：为平台添加智能搜索功能。"
        "要求支持自然语言搜索、实时建议、搜索结果高亮。优先级P1，Sprint时间2周。",
        sender_role="PM",
    )

    # Step 2: Developer回应并开始实现
    print("\n  ─── Step 2: Developer确认并开始实现 ───")
    dev_response = team.direct_message(
        from_role="Developer",
        to_role="PM",
        message="已评估技术可行性。需要集成Elasticsearch，预估工作量5人天。"
               "建议MVP先支持基本搜索，高级功能留到下一个Sprint。",
    )

    # Step 3: QA准备测试
    print("\n  ─── Step 3: QA准备测试策略 ───")
    qa_response = team.direct_message(
        from_role="QA",
        to_role="PM",
        message="已为智能搜索功能准备了测试用例，包括：基本搜索、空输入、"
               "特殊字符、并发搜索、性能基准。测试环境已就绪。",
    )

    # Step 4: DevOps确认部署
    print("\n  ─── Step 4: DevOps确认部署就绪 ───")
    devops_response = team.direct_message(
        from_role="DevOps",
        to_role="Developer",
        message="CI/CD流水线已配置完成。staging环境已就绪。"
               "部署策略：先staging验证，然后金丝雀发布到生产。"
               "请在代码中包含健康检查端点。",
    )

    # Step 5: PM做最终协调
    print("\n  ─── Step 5: PM做最终协调 ───")
    final_broadcast = team.broadcast(
        "总结一下：Developer负责实现（5天），QA负责测试（2天），"
        "DevOps负责部署（1天）。每天Standup同步进度。"
        "如果有阻塞项，立即在群内提出来。大家加油！💪",
        sender_role="PM",
    )

    # ---- Demo 2: 权限检查 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 工具权限检查")
    print("=" * 72)

    pm = team.get_agent("PM")

    # 列出PM可以使用的工具
    pm_tools = tool_registry.list_tools_for_role("PM")
    print(f"\n  PM可用的工具: {[t.name for t in pm_tools]}")

    # 尝试正常操作
    try:
        result = pm.use_tool("create_requirement", title="权限测试需求", priority="P3")
        print(f"  ✅ PM使用 create_requirement: 成功")
    except PermissionError as e:
        print(f"  ❌ {e}")

    # 尝试越权操作（PM尝试写代码）
    try:
        result = pm.use_tool("write_code", code="print('hello')", language="python")
        print(f"  ✅ PM使用 write_code: 成功（不应该！）")
    except PermissionError as e:
        print(f"  🚫 权限拦截: PM不能使用 write_code")

    # 尝试越权操作（PM尝试部署）
    try:
        result = pm.use_tool("deploy_service", service_name="test", version="1.0", environment="production")
        print(f"  ✅ PM使用 deploy_service: 成功（不应该！）")
    except PermissionError as e:
        print(f"  🚫 权限拦截: PM不能使用 deploy_service")

    # ---- Demo 3: Persona 生成 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: Persona 系统提示词生成")
    print("=" * 72)

    dev = team.get_agent("Developer")
    prompt = dev.persona.generate_system_prompt()
    print(f"\n  为 {dev.name} 生成的 system_prompt:\n")
    print(prompt)

    # ---- 项目摘要 ----

    print("\n\n" + "=" * 72)
    print("  📊 项目协作摘要")
    print("=" * 72)

    summary = team.get_project_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print("  角色扮演式Agent演示完成！")
    print("=" * 72)
