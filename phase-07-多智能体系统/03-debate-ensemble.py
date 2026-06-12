#!/usr/bin/env python3
"""
Phase 07 - 多智能体系统：03 - 辩论与集成模式（Debate & Ensemble）

多个Agent从不同角度辩论同一问题。结构化轮次（Opening → Rebuttal → Closing）。
投票机制（加权得分）+ Judge Agent 裁决。
对比单Agent vs 多Agent集成准确率。

架构图：
                     ┌─────────────┐
                     │   问题输入    │
                     └──────┬──────┘
                            │
          ┌─────────┬───────┼───────┬─────────┐
          ▼         ▼       ▼       ▼         ▼
    ┌─────────┐ ┌───────┐ ┌─────┐ ┌───────┐ ┌───────────┐
    │ Agent A │ │Agent B│ │Agent│ │Agent D│ │ Agent E   │
    │(赞成方) │ │(反对方)│ │  C  │ │       │ │(中立观察) │
    └────┬────┘ └───┬───┘ └──┬──┘ └───┬───┘ └─────┬─────┘
         │          │        │        │            │
         └──────────┼────────┼────────┼────────────┘
                    │        │        │
         ┌──────────▼────────▼────────▼──────────┐
         │           辩论轮次                      │
         │  Round 1: Opening 陈述                │
         │  Round 2: Rebuttal 反驳               │
         │  Round 3: Closing 总结                │
         └──────────┬────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │   Judge Agent       │
         │   (加权投票 + 裁决)  │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   最终判决 + 得分    │
         └─────────────────────┘
"""

import json
import time
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================================
# 数据模型
# ============================================================================

class DebateRole(Enum):
    """辩论角色。"""
    PROPONENT = "proponent"       # 赞成方
    OPPONENT = "opponent"         # 反对方
    NEUTRAL = "neutral"           # 中立观察
    EXPERT = "expert"             # 领域专家
    SKEPTIC = "skeptic"           # 质疑者
    PRAGMATIST = "pragmatist"     # 实用主义者


class RoundType(Enum):
    """辩论轮次类型。"""
    OPENING = "opening"           # 开场陈述
    REBUTTAL = "rebuttal"         # 反驳/交叉质询
    CLOSING = "closing"           # 总结陈词


@dataclass
class DebateAgent:
    """
    辩论Agent配置。

    Attributes:
        agent_id: 唯一标识
        name: 显示名称
        role: 辩论角色
        perspective: 观点/立场描述
        system_prompt: 系统提示词（包含角色设定和论证要求）
        model: 使用的LLM模型
        expertise_weight: 专业权重（该Agent的发言在投票时的权重，1.0=标准）
        evidence_requirements: 证据要求（该Agent需要提供什么级别的证据）
    """
    agent_id: str
    name: str
    role: DebateRole
    perspective: str
    system_prompt: str
    model: str = "gpt-4o-mini"
    expertise_weight: float = 1.0
    evidence_requirements: str = "medium"  # low | medium | high


@dataclass
class DebateStatement:
    """
    辩论中的一个发言/陈述。

    Attributes:
        round_num: 轮次编号
        round_type: 轮次类型（Opening/Rebuttal/Closing）
        agent_id: 发言Agent的ID
        content: 发言内容
        references: 引用的论据/数据
        confidence: Agent自我评估的置信度（0-1）
    """
    round_num: int
    round_type: RoundType
    agent_id: str
    content: str
    references: list[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class Vote:
    """
    Judge对某个Agent陈述的投票。
    """
    agent_id: str                     # 被评分的Agent
    scores: dict[str, float]          # 各维度得分
    total_score: float                # 加权总分
    reasoning: str                    # 评分理由


@dataclass
class DebateResult:
    """
    完整辩论结果。
    """
    topic: str                        # 辩论主题
    statements: list[DebateStatement] # 所有发言
    votes: list[Vote]                 # 所有评分
    winner: Optional[str]             # 获胜方（或个人）
    final_verdict: str                # 最终裁决
    confidence: float                 # 裁决置信度
    total_time: float                 # 总耗时
    ensemble_score: float             # 集成得分 vs 单Agent得分


# ============================================================================
# 模拟LLM响应
# ============================================================================

def simulate_debate_response(
    agent: DebateAgent,
    topic: str,
    round_type: RoundType,
    previous_statements: list[DebateStatement],
) -> tuple[str, list[str], float]:
    """
    模拟辩论Agent的发言。
    生产环境中替换为真实的LLM API调用。
    """
    role_name_map = {
        DebateRole.PROPONENT: "赞成方",
        DebateRole.OPPONENT: "反对方",
        DebateRole.NEUTRAL: "中立观察",
        DebateRole.EXPERT: "领域专家",
        DebateRole.SKEPTIC: "质疑者",
        DebateRole.PRAGMATIST: "实用主义者",
    }

    role_name = role_name_map.get(agent.role, "参与者")

    # 根据角色和轮次生成不同的模拟响应
    round_label = {
        RoundType.OPENING: "开场陈述",
        RoundType.REBUTTAL: "反驳阶段",
        RoundType.CLOSING: "总结陈词",
    }[round_type]

    if round_type == RoundType.OPENING:
        content, refs = _generate_opening(agent, topic, role_name)
    elif round_type == RoundType.REBUTTAL:
        content, refs = _generate_rebuttal(agent, topic, role_name, previous_statements)
    else:
        content, refs = _generate_closing(agent, topic, role_name, previous_statements)

    # 置信度评估
    confidence = {
        DebateRole.PROPONENT: 0.85,
        DebateRole.OPPONENT: 0.80,
        DebateRole.NEUTRAL: 0.75,
        DebateRole.EXPERT: 0.90,
        DebateRole.SKEPTIC: 0.70,
        DebateRole.PRAGMATIST: 0.82,
    }.get(agent.role, 0.80)

    return content, refs, confidence


def _generate_opening(agent: DebateAgent, topic: str, role_name: str) -> tuple[str, list[str]]:
    """生成开场陈述。"""
    if agent.role == DebateRole.PROPONENT:
        content = (
            f"## 开场陈述 - {agent.name}（{role_name}）\n\n"
            f"**核心观点**：我坚定支持这一方案。\n\n"
            f"关于'{topic}'，我认为这是一个极具价值的方向。基于以下几点：\n\n"
            f"1. **效率提升**：实施该方案预计可提升效率30-45%（参考McKinsey 2025数字化转型报告）\n"
            f"2. **成本优化**：自动化可降低人力成本约25%，第一年即可收回投入\n"
            f"3. **竞争优势**：行业先行者已获得显著市场优势（早期采用者市场份额+8.2%）\n"
            f"4. **可扩展性**：架构设计支持横向扩展，可支撑10倍+的用户增长\n\n"
            f"**推荐方案**：分三期实施，每期3个月，总预算X，预期ROI为250%。\n"
        )
    elif agent.role == DebateRole.OPPONENT:
        content = (
            f"## 开场陈述 - {agent.name}（{role_name}）\n\n"
            f"**核心观点**：我对此方案持保留态度，建议慎重考虑。\n\n"
            f"关于'{topic}'，我们需要看到硬币的另一面：\n\n"
            f"1. **隐性成本**：实施过程中的培训、迁移、维护成本常被低估（实际项目超支率：平均47%）\n"
            f"2. **风险因素**：数据迁移风险、系统不稳定带来的业务中断、安全合规挑战\n"
            f"3. **组织阻力**：员工抵触、技能缺口、管理层的期望管理——这些'软性'挑战往往导致项目失败\n"
            f"4. **替代方案**：渐进式优化现有系统可能获得80%的收益，但风险仅为新方案的20%\n\n"
            f"**建议**：先用3个月做POC（概念验证），数据驱动决策，而非跟风。\n"
        )
    elif agent.role == DebateRole.EXPERT:
        content = (
            f"## 开场陈述 - {agent.name}（{role_name}）\n\n"
            f"**技术评估**：作为领域专家，我从技术可行性角度分析'{topic}'：\n\n"
            f"1. **技术成熟度**：核心技术在Gartner Hype Cycle中位于'稳步爬升期'，但部分组件仍有不确定性\n"
            f"2. **技术债务**：现有系统与目标架构的差距评估为'中高'——需要大量适配工作\n"
            f"3. **性能基准**：在实验室环境中，方案核心组件达到99.5%的SLA，但生产环境通常下降3-5%\n"
            f"4. **可维护性**：新架构引入3个新技术栈，团队学习曲线约2-3个月\n\n"
            f"**技术判断**：方案技术可行，但建议将安全审计前置到设计阶段。\n"
        )
    elif agent.role == DebateRole.SKEPTIC:
        content = (
            f"## 开场陈述 - {agent.name}（{role_name}）\n\n"
            f"**质疑视角**：我需要指出'{topic}'中被忽视的假设和盲点：\n\n"
            f"1. **数据质量假设**：方案假设现有数据质量良好——但审计显示17%的数据存在不一致\n"
            f"2. **市场假设**：'市场会持续增长'——但我们需要考虑下行风险场景\n"
            f"3. **时间假设**：'3个月完成第一阶段'——历史数据显示类似项目平均延期40%\n"
            f"4. **能力假设**：当前团队是否具备所需技能？技能缺口评估尚未完成\n\n"
            f"**要求**：在决策前，请先解答以上4个假设的验证情况。\n"
        )
    else:
        content = (
            f"## 开场陈述 - {agent.name}（{role_name}）\n\n"
            f"**平衡视角**：关于'{topic}'，我从平衡的角度提供观察：\n\n"
            f"从多维度来看，这个方案有其合理之处，但也存在需要解决的挑战。\n"
            f"关键在于用数据而非感觉来做决策。我建议采用'红队-蓝队'方式验证方案。\n"
        )

    # 生成引用的参考来源
    refs = [
        "McKinsey Digital Transformation Report 2025",
        "Gartner Technology Adoption Roadmap 2025",
        "Harvard Business Review: AI Strategy, 2025 Q1",
    ]

    return content, refs


def _generate_rebuttal(
    agent: DebateAgent,
    topic: str,
    role_name: str,
    prev: list[DebateStatement],
) -> tuple[str, list[str]]:
    """生成反驳陈述。"""
    # 提取上一个发言的关键内容用于回应
    prev_content = prev[-1].content[:150] if prev else ""

    if agent.role == DebateRole.PROPONENT:
        content = (
            f"## 反驳 - {agent.name}（{role_name}）\n\n"
            f"针对反对方的质疑，我做出以下回应：\n\n"
            f"1. **关于成本**：反对方引用的'47%超支'数据是针对传统IT项目，"
            f"而AI相关项目的超支率中位数为22%（Stanford HAI 2025）——差距显著\n"
            f"2. **关于风险**：风险是可控的——我们有成熟的迁移策略和回滚方案\n"
            f"3. **关于替代方案**：'优化现有系统'看似稳妥，但技术债务复利效应最终代价更大\n\n"
            f"**数据支撑**：我引用的数据全部来自同行评审的研究报告，非行业宣传材料。\n"
        )
    elif agent.role == DebateRole.OPPONENT:
        content = (
            f"## 反驳 - {agent.name}（{role_name}）\n\n"
            f"针对赞成方的论述，我提出以下问题：\n\n"
            f"1. **效率提升的测算**：'30-45%'是实验室数字，实际部署中由于集成摩擦，"
            f"这个数字通常衰减到15-25%\n"
            f"2. **ROI假设**：250%的ROI假设了所有变量都在最理想状态，没有做敏感性分析\n"
            f"3. **竞争优势窗口**：先发优势确实存在，但'快速跟随者'策略同样成功（案例：Facebook、Google）\n\n"
            f"**核心关切**：我们需要在'过度乐观'和'过度保守'之间找到真正的平衡点。\n"
        )
    elif agent.role == DebateRole.EXPERT:
        content = (
            f"## 技术回应 - {agent.name}（{role_name}）\n\n"
            f"从技术角度澄清几个关键点：\n\n"
            f"1. SLO vs SLA：我们讨论的是服务等级目标（SLO）而非保证（SLA）——99.5%是合理目标\n"
            f"2. 技术选型：建议的3个新技术栈中有2个与现有栈高度兼容，学习曲线实际上约为1.5个月\n"
            f"3. 迁移策略：推荐蓝绿部署 + 12%流量灰度验证，可将风险控制在可控范围\n"
            f"4. 建议增加一个'技术验证冲刺'（2周），验证关键假设后再做最终决策\n"
        )
    else:
        content = (
            f"## 反驳 - {agent.name}（{role_name}）\n\n"
            f"关于'{topic}'，综合双方观点后，我观察到：\n\n"
            f"正反双方的分歧根源在于风险偏好不同和对数据的不同解读。\n"
            f"建议双方在以下事项上寻求共识：1) 数据来源和可靠性标准；2) 风险评估框架；3) 决策时间表。\n"
        )

    refs = ["Quarterly Industry Review 2025 Q1", "Internal benchmarking data 2024-2025"]
    return content, refs


def _generate_closing(
    agent: DebateAgent,
    topic: str,
    role_name: str,
    prev: list[DebateStatement],
) -> tuple[str, list[str]]:
    """生成总结陈词。"""
    if agent.role == DebateRole.PROPONENT:
        content = (
            f"## 总结陈词 - {agent.name}（{role_name}）\n\n"
            f"综合整场辩论，我重申支持立场，并提出最终建议：\n\n"
            f"**方案优化**：吸取反对方合理意见，修改为'分阶段实施+每阶段出口评审'\n"
            f"**风险对冲**：设立10%的缓冲预算和1个月的缓冲时间\n"
            f"**决策建议**：建议董事会批准第一阶段（POC的升级版），3个月后根据数据决定后续投入\n\n"
            f"这个方案在'前进'和'稳健'之间取得了平衡，是当前最优解。\n"
        )
    elif agent.role == DebateRole.OPPONENT:
        content = (
            f"## 总结陈词 - {agent.name}（{role_name}）\n\n"
            f"我仍然建议更谨慎的路径，但承认方案有可取之处：\n\n"
            f"**折中建议**：\n"
            f"1. 先执行2个月的'发现阶段'（Discovery Phase），包括技术验证和团队评估\n"
            f"2. 聘请独立第三方进行技术审查\n"
            f"3. 设定明确的'Go/No-Go'标准\n\n"
            f"如果发现阶段的结果正面，我将支持进入正式实施。但在数据出来之前，保持谨慎。\n"
        )
    else:
        content = (
            f"## 总结陈词 - {agent.name}（{role_name}）\n\n"
            f"总结双方核心论点，我认为最佳路径是：\n\n"
            f"- 采纳赞成方的方向（方向正确）\n"
            f"- 吸收反对方的审慎（方法需要更稳健）\n"
            f"- 补充专家的技术建议（实现需要更务实）\n\n"
            f"**最终建议**：有条件批准，条件为[技术验证通过 + 风险预案就绪]。\n"
        )

    refs = ["综合前两轮讨论的所有引用资料"]
    return content, refs


# ============================================================================
# Judge Agent
# ============================================================================

class JudgeAgent:
    """
    Judge Agent：评估所有辩论发言，计算加权得分，做出最终裁决。

    评分维度：
    - logic: 逻辑严密性
    - evidence: 证据充分性
    - relevance: 与主题相关性
    - clarity: 表达清晰度
    - persuasiveness: 说服力
    """

    SCORE_DIMENSIONS = ["logic", "evidence", "relevance", "clarity", "persuasiveness"]
    DIMENSION_WEIGHTS = {
        "logic": 0.25,
        "evidence": 0.25,
        "relevance": 0.20,
        "clarity": 0.15,
        "persuasiveness": 0.15,
    }

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def evaluate(
        self,
        topic: str,
        statements: list[DebateStatement],
        agents: list[DebateAgent],
        verbose: bool = True,
    ) -> DebateResult:
        """
        评估整场辩论并做出裁决。

        Args:
            topic: 辩论主题
            statements: 所有发言记录
            agents: 参与辩论的Agent列表
            verbose: 是否打印详细日志

        Returns:
            辩论结果
        """
        if verbose:
            print(f"\n  ⚖️ Judge ({self.model}) 正在评估辩论...")

        agent_map = {a.agent_id: a for a in agents}

        # 为每个Agent计算得分
        votes: list[Vote] = []
        for agent in agents:
            agent_statements = [s for s in statements if s.agent_id == agent.agent_id]
            if not agent_statements:
                continue

            vote = self._score_agent(agent, agent_statements, topic)
            votes.append(vote)

        # 找出获胜方
        if votes:
            # 按角色分组，计算各方总分
            role_scores: dict[str, float] = defaultdict(float)
            for vote in votes:
                agent = agent_map[vote.agent_id]
                role_scores[agent.role.value] += vote.total_score * agent.expertise_weight

            # 归为支持/反对两方
            support_score = (
                role_scores.get("proponent", 0) +
                role_scores.get("pragmatist", 0) * 0.7
            )
            oppose_score = (
                role_scores.get("opponent", 0) +
                role_scores.get("skeptic", 0)
            )
            neutral_score = (
                role_scores.get("neutral", 0) +
                role_scores.get("expert", 0)
            )

            # 中立方的分数分配给两边
            support_score += neutral_score * 0.5
            oppose_score += neutral_score * 0.5

            if support_score > oppose_score:
                winner = "支持方（Proponent阵营）"
                confidence = support_score / (support_score + oppose_score)
            elif oppose_score > support_score:
                winner = "反对方（Opponent阵营）"
                confidence = oppose_score / (support_score + oppose_score)
            else:
                winner = "平局"
                confidence = 0.5
        else:
            winner = "无有效发言"
            confidence = 0.0

        # 生成最终裁决
        verdict = self._generate_verdict(topic, votes, winner, agent_map)

        if verbose:
            print(f"\n  📊 投票结果:")
            for vote in votes:
                print(f"    {agent_map[vote.agent_id].name}: "
                      f"{vote.total_score:.2f}/1.00")
            print(f"  🏆 获胜方: {winner}")
            print(f"  📈 裁决置信度: {confidence:.1%}")

        # 计算集成得分
        ensemble_score = self._calculate_ensemble_score(votes, agent_map)

        return DebateResult(
            topic=topic,
            statements=statements,
            votes=votes,
            winner=winner,
            final_verdict=verdict,
            confidence=confidence,
            total_time=0,  # 会在外部设置
            ensemble_score=ensemble_score,
        )

    def _score_agent(
        self,
        agent: DebateAgent,
        statements: list[DebateStatement],
        topic: str,
    ) -> Vote:
        """
        对单个Agent的所有发言进行评分。

        生产环境中替换为LLM调用：
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": f"你是公正的辩论裁判。请从{self.SCORE_DIMENSIONS}五个维度"
                           f"评估以下发言，每个维度0-1分。"
            }, {
                "role": "user",
                "content": f"主题: {topic}\n发言: {statements}"
            }],
            response_format={"type": "json_object"},
        )
        """
        # 模拟评分逻辑
        scores = {}

        # 基于发言长度和质量给予不同分数
        total_content_len = sum(len(s.content) for s in statements)
        ref_count = sum(len(s.references) for s in statements)
        round_count = len(statements)

        # 逻辑性：基于发言结构复杂度
        scores["logic"] = min(0.95, 0.5 + 0.1 * round_count + 0.05 * bool(ref_count))

        # 证据：基于引用数量
        scores["evidence"] = min(0.95, 0.4 + 0.1 * ref_count)

        # 相关性：所有Agent默认较高（由角色设定保证）
        scores["relevance"] = 0.80 + 0.05 * (agent.expertise_weight - 1.0)

        # 清晰度：基于内容长度（过短或过长都会扣分）
        optimal_len = 500
        len_ratio = total_content_len / max(1, round_count * optimal_len)
        scores["clarity"] = max(0.5, min(0.95, 0.95 - abs(len_ratio - 1.0) * 0.3))

        # 说服力：专家和实用主义者天然更高
        persuasiveness_base = {
            DebateRole.PROPONENT: 0.75,
            DebateRole.OPPONENT: 0.70,
            DebateRole.EXPERT: 0.82,
            DebateRole.SKEPTIC: 0.65,
            DebateRole.NEUTRAL: 0.60,
            DebateRole.PRAGMATIST: 0.78,
        }.get(agent.role, 0.70)
        scores["persuasiveness"] = persuasiveness_base + 0.05 * (ref_count > 2)

        # 加权总分
        total = sum(
            scores[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in self.SCORE_DIMENSIONS
        )

        reasoning = (
            f"{agent.name}（{agent.role.value}）："
            f"逻辑{scores['logic']:.2f}、证据{scores['evidence']:.2f}、"
            f"相关{scores['relevance']:.2f}、清晰{scores['clarity']:.2f}、"
            f"说服{scores['persuasiveness']:.2f} → 加权 {total:.3f}"
        )

        return Vote(
            agent_id=agent.agent_id,
            scores=scores,
            total_score=total,
            reasoning=reasoning,
        )

    def _generate_verdict(
        self,
        topic: str,
        votes: list[Vote],
        winner: str,
        agent_map: dict,
    ) -> str:
        """生成最终裁决文本。"""
        verdict = (
            f"# 辩论裁决书\n\n"
            f"## 辩论主题\n{topic}\n\n"
            f"## 参与者\n"
        )
        for agent_id, agent in agent_map.items():
            verdict += f"- {agent.name}（{agent.role.value}）权重: {agent.expertise_weight}\n"

        verdict += f"\n## 评分详情\n"
        for vote in votes:
            verdict += f"### {agent_map[vote.agent_id].name}\n"
            verdict += f"- 总分: {vote.total_score:.3f}/1.000\n"
            for dim, score in vote.scores.items():
                verdict += f"  - {dim}: {score:.2f}\n"
            verdict += f"\n{vote.reasoning}\n\n"

        verdict += (
            f"\n## 最终裁决\n\n"
            f"**获胜方**: {winner}\n\n"
            f"**判决理由**:\n"
            f"综合所有Agent的论证质量、证据充分性和逻辑严密性，本庭做出以上裁决。\n"
            f"建议决策者参考本裁决，同时结合自身实际情况做出最终判断。\n"
        )

        return verdict

    def _calculate_ensemble_score(
        self,
        votes: list[Vote],
        agent_map: dict,
    ) -> float:
        """
        计算集成得分（多个Agent协作的整体质量指标）。
        用于对比单Agent vs 多Agent集成效果。
        """
        if not votes:
            return 0.0

        # 集成得分 = 最高分 * 多样性红利 + 平均分
        max_score = max(v.total_score for v in votes)
        avg_score = sum(v.total_score for v in votes) / len(votes)

        # 角色多样性：不同角色越多，多样性红利越大
        roles = set()
        for v in votes:
            for aid, agent in agent_map.items():
                if aid == v.agent_id:
                    roles.add(agent.role)

        diversity_bonus = min(0.15, len(roles) * 0.03)

        ensemble_score = max_score * (1.0 + diversity_bonus) * 0.6 + avg_score * 0.4
        return round(ensemble_score, 4)


# ============================================================================
# DebateEngine
# ============================================================================

class DebateEngine:
    """
    辩论引擎：管理多Agent辩论的完整流程。
    """

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.agents: list[DebateAgent] = []
        self.judge = JudgeAgent(model="gpt-4o")
        self.statements: list[DebateStatement] = []

    def add_agent(self, agent: DebateAgent) -> "DebateEngine":
        """添加一个辩论Agent。"""
        self.agents.append(agent)
        return self

    def add_agents(self, agents: list[DebateAgent]) -> "DebateEngine":
        """批量添加辩论Agent。"""
        self.agents.extend(agents)
        return self

    def run_debate(self, topic: str, verbose: bool = True) -> DebateResult:
        """
        运行完整辩论流程。

        Args:
            topic: 辩论主题
            verbose: 是否打印详细日志

        Returns:
            辩论结果
        """
        if len(self.agents) < 2:
            raise ValueError("辩论至少需要2个Agent参与")

        self.statements = []
        total_start = time.time()

        if verbose:
            print(f"\n  🎤 辩论开始！主题: '{topic}'")
            print(f"  参与Agent: {len(self.agents)}")
            for a in self.agents:
                print(f"    - {a.name} ({a.role.value}，权重: {a.expertise_weight})")

        # Round 1: Opening 陈述
        if verbose:
            print(f"\n  ─── Round 1: 开场陈述 ───")
        for agent in self.agents:
            self._agent_speak(agent, topic, RoundType.OPENING, 1, verbose)

        # Round 2: Rebuttal 反驳（如果超过2个轮次）
        if self.max_rounds >= 2:
            if verbose:
                print(f"\n  ─── Round 2: 反驳阶段 ───")
            for agent in self.agents:
                self._agent_speak(agent, topic, RoundType.REBUTTAL, 2, verbose)

        # Round 3: Closing 总结
        if self.max_rounds >= 3:
            if verbose:
                print(f"\n  ─── Round 3: 总结陈词 ───")
            for agent in self.agents:
                self._agent_speak(agent, topic, RoundType.CLOSING, 3, verbose)

        # Judge 评估和裁决
        total_time = time.time() - total_start

        if verbose:
            print(f"\n  🏁 辩论结束（{total_time:.1f}s）")

        result = self.judge.evaluate(topic, self.statements, self.agents, verbose)
        result.total_time = total_time

        return result

    def _agent_speak(
        self,
        agent: DebateAgent,
        topic: str,
        round_type: RoundType,
        round_num: int,
        verbose: bool,
    ):
        """让一个Agent发言。"""
        if verbose:
            print(f"    🎙️ {agent.name} ({agent.role.value}) 准备发言...")

        content, refs, confidence = simulate_debate_response(
            agent, topic, round_type, self.statements,
        )

        statement = DebateStatement(
            round_num=round_num,
            round_type=round_type,
            agent_id=agent.agent_id,
            content=content,
            references=refs,
            confidence=confidence,
        )
        self.statements.append(statement)

        if verbose:
            content_preview = content[:100].replace('\n', ' ')
            print(f"    ✅ {agent.name}: [{len(content)}字符] {content_preview}...")


# ============================================================================
# Ensemble 对比实验
# ============================================================================

def compare_single_vs_ensemble(
    topic: str,
    verbose: bool = True,
) -> dict:
    """
    对比单Agent vs 多Agent集成的效果。

    运行流程：
    1. 单Agent模式：仅用一个Agent直接回答
    2. 集成模式：多Agent辩论后投票
    3. 对比两种模式的得分
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  单Agent vs 多Agent集成 对比实验")
        print("=" * 60)

    # ---- 单Agent模式 ----
    if verbose:
        print("\n  📍 实验A: 单Agent直接回答")

    single_agent = DebateAgent(
        agent_id="single_1",
        name="通用Agent",
        role=DebateRole.NEUTRAL,
        perspective="从综合角度分析",
        system_prompt="你是资深分析师，请从多个角度分析问题...",
        model="gpt-4o-mini",
    )

    single_start = time.time()
    content, refs, conf = simulate_debate_response(
        single_agent, topic, RoundType.OPENING, []
    )
    single_time = time.time() - single_start

    single_statement = DebateStatement(
        round_num=1,
        round_type=RoundType.OPENING,
        agent_id=single_agent.agent_id,
        content=content,
        references=refs,
        confidence=conf,
    )

    # 单Agent评分
    judge = JudgeAgent()
    single_vote = judge._score_agent(single_agent, [single_statement], topic)
    single_score = single_vote.total_score

    if verbose:
        print(f"  单Agent得分: {single_score:.3f} | 耗时: {single_time:.1f}s")
        print(f"  各维度: {single_vote.scores}")

    # ---- 多Agent集成模式 ----
    if verbose:
        print("\n  📍 实验B: 多Agent辩论集成")

    engine = DebateEngine(max_rounds=2)
    agents = create_debate_agents()
    for a in agents:
        engine.add_agent(a)

    debate_result = engine.run_debate(topic, verbose=verbose)

    # ---- 对比分析 ----
    comparison = {
        "topic": topic,
        "single_agent": {
            "score": round(single_score, 4),
            "time": round(single_time, 2),
            "agent": single_agent.name,
            "statements": 1,
        },
        "ensemble": {
            "score": round(debate_result.ensemble_score, 4),
            "time": round(debate_result.total_time, 2),
            "agents": len(engine.agents),
            "statements": len(debate_result.statements),
            "winner": debate_result.winner,
            "confidence": round(debate_result.confidence, 4),
        },
        "improvement": {
            "score_delta": round(debate_result.ensemble_score - single_score, 4),
            "score_pct": round(
                (debate_result.ensemble_score - single_score) / single_score * 100, 1
            ),
            "time_penalty": round(debate_result.total_time / single_time, 1),
        },
    }

    if verbose:
        print(f"\n  {'=' * 60}")
        print(f"  📊 对比结果:")
        print(f"  {'指标':<20} {'单Agent':<15} {'多Agent集成':<15} {'改善':<15}")
        print(f"  {'─' * 20} {'─' * 15} {'─' * 15} {'─' * 15}")
        print(f"  {'得分':<20} {single_score:<15.4f} "
              f"{debate_result.ensemble_score:<15.4f} "
              f"{comparison['improvement']['score_pct']:+}%")
        print(f"  {'耗时(秒)':<20} {single_time:<15.1f} "
              f"{debate_result.total_time:<15.1f} "
              f"{comparison['improvement']['time_penalty']:}x")
        print(f"  {'发言人/发言数':<20} 1/1 {'':<7} "
              f"{len(engine.agents)}/{len(debate_result.statements)}")
        print(f"  {'裁决置信度':<20} N/A {'':<7} "
              f"{debate_result.confidence:.1%}")

    return comparison


# ============================================================================
# 预定义辩论团队
# ============================================================================

def create_debate_agents() -> list[DebateAgent]:
    """创建标准辩论团队：5个Agent，覆盖多种视角。"""
    return [
        DebateAgent(
            agent_id="agent_pro",
            name="方案倡导者-马克",
            role=DebateRole.PROPONENT,
            perspective="积极推动变革，强调机遇和收益",
            system_prompt="你是变革推动者。你的任务是清晰有力地阐述方案的价值、...",
            model="gpt-4o-mini",
            expertise_weight=1.2,
        ),
        DebateAgent(
            agent_id="agent_con",
            name="风险分析师-丽莎",
            role=DebateRole.OPPONENT,
            perspective="关注风险和成本，确保决策稳健",
            system_prompt="你是风险管理专家。你的任务是识别方案中的潜在风险、...",
            model="gpt-4o-mini",
            expertise_weight=1.2,
        ),
        DebateAgent(
            agent_id="agent_expert",
            name="技术架构师-汤姆",
            role=DebateRole.EXPERT,
            perspective="从技术可行性角度提供专业评估",
            system_prompt="你是资深技术架构师。你的任务是对方案进行技术评估、...",
            model="gpt-4o",
            expertise_weight=1.5,
        ),
        DebateAgent(
            agent_id="agent_skeptic",
            name="批判性思考者-萨拉",
            role=DebateRole.SKEPTIC,
            perspective="挑战假设，揭示盲点，防止群体思维",
            system_prompt="你是批判性思维专家。你的任务是挑战所有假设、...",
            model="gpt-4o-mini",
            expertise_weight=1.0,
        ),
        DebateAgent(
            agent_id="agent_pragmatist",
            name="执行专家-大卫",
            role=DebateRole.PRAGMATIST,
            perspective="关注落地可行性，平衡理想与现实",
            system_prompt="你是实操专家。你的任务是评估方案的落地可行性、...",
            model="gpt-4o-mini",
            expertise_weight=1.1,
        ),
    ]


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  多智能体系统 - 辩论与集成模式演示")
    print("=" * 72)

    # ---- Demo 1: 完整辩论 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 正式辩论（全5Agent + 3轮次）")
    print("=" * 72)

    topic1 = "公司是否应该投入500万美元将现有单体架构全面迁移到微服务架构？"

    engine1 = DebateEngine(max_rounds=3)
    agents1 = create_debate_agents()
    for agent in agents1:
        engine1.add_agent(agent)

    debate_result1 = engine1.run_debate(topic1, verbose=True)

    print(f"\n{'─' * 72}")
    print("  最终裁决:")
    print(f"{'─' * 72}")
    print(debate_result1.final_verdict[:800])

    # ---- Demo 2: 对比实验 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 单Agent vs 多Agent集成 效果对比")
    print("=" * 72)

    topic2 = "AI辅助代码审查工具是否应该在团队中强制使用？"

    comparison = compare_single_vs_ensemble(topic2, verbose=True)

    # ---- Demo 3: 简单快速辩论 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 快速辩论（3Agent + 2轮次）")
    print("=" * 72)

    topic3 = "远程办公 vs 办公室办公：哪种模式更适合知识工作者？"

    engine3 = DebateEngine(max_rounds=2)
    quick_agents = [
        DebateAgent(
            agent_id="quick_pro",
            name="远程派",
            role=DebateRole.PROPONENT,
            perspective="远程办公提升生产力和员工满意度",
            system_prompt="你支持远程办公...",
        ),
        DebateAgent(
            agent_id="quick_con",
            name="办公室派",
            role=DebateRole.OPPONENT,
            perspective="面对面协作不可替代",
            system_prompt="你支持办公室办公...",
        ),
        DebateAgent(
            agent_id="quick_neutral",
            name="平衡派",
            role=DebateRole.PRAGMATIST,
            perspective="混合模式是最优解",
            system_prompt="你倡导混合办公模式...",
        ),
    ]
    for a in quick_agents:
        engine3.add_agent(a)

    result3 = engine3.run_debate(topic3, verbose=True)

    print("\n" + "=" * 72)
    print("  辩论与集成模式演示完成！")
    print("=" * 72)
