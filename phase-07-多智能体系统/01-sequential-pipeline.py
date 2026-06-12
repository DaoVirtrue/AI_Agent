#!/usr/bin/env python3
"""
Phase 07 - 多智能体系统：01 - 串行流水线（Sequential Pipeline）

SequentialPipeline 类：多个 Agent 按顺序执行，每个 Agent 的输出作为下一个的输入。
经典场景：ResearchAgent → WritingAgent → ReviewAgent → PolishAgent

架构图：
┌──────────┐     ┌───────────┐     ┌──────────┐     ┌───────────┐
│ Research │────→│  Writing  │────→│  Review  │────→│  Polish   │────→ 最终输出
│  Agent   │     │   Agent   │     │  Agent   │     │   Agent   │
└──────────┘     └───────────┘     └──────────┘     └───────────┘
   耗时:2.3s        耗时:3.1s        耗时:1.8s        耗时:1.5s
                                                     总耗时:8.7s
"""

import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime


# ============================================================================
# 模拟 LLM 调用（生产环境替换为真实的 OpenAI/Anthropic 调用）
# ============================================================================

def simulate_llm_call(
    system_prompt: str,
    user_input: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    delay: float = 0.0,
) -> str:
    """
    模拟 LLM 调用。在生产环境中，替换为真实的 API 调用。
    这里用简单的字符串处理来模拟不同 Agent 的行为。
    """
    # 模拟网络延迟 + 推理延迟
    if delay > 0:
        time.sleep(delay)

    # 根据 system_prompt 中的角色描述来模拟不同行为
    role = system_prompt.lower()

    if "research" in role or "研究" in role or "调研" in role:
        return _simulate_research(user_input)
    elif "writing" in role or "写作" in role or "撰写" in role:
        return _simulate_writing(user_input)
    elif "review" in role or "审校" in role or "审查" in role:
        return _simulate_review(user_input)
    elif "polish" in role or "润色" in role or "优化" in role:
        return _simulate_polish(user_input)
    else:
        # 通用处理：简单回显并标注已处理
        return f"[{model}] 已处理: {user_input[:100]}..."


def _simulate_research(input_text: str) -> str:
    """模拟调研Agent：收集资料、整理关键信息。"""
    topic = input_text[:80]
    return f"""## 研究阶段输出

### 调研主题
{topic}

### 关键发现
1. **市场现状**：该领域2024年市场规模约为120亿美元，年增长率18.5%
2. **技术趋势**：Transformer架构持续主导，多模态融合成为新方向
3. **竞争格局**：OpenAI、Google、Anthropic三家占据76%市场份额
4. **挑战与机遇**：数据隐私合规是最大挑战，垂直领域应用是最大机遇

### 参考资料
- [McKinsey 2024 AI Report] - 第3章
- [Stanford HAI 2025 Index] - pp.45-62
- [Gartner Magic Quadrant 2025] - AI平台象限

### 数据要点
- 企业AI采用率：从2023年的55%增长到2025年的72%
- ROI中位数：部署AI后12个月内平均回报率27%
"""


def _simulate_writing(input_text: str) -> str:
    """模拟写作Agent：基于研究产出初稿。"""
    return f"""## 写作阶段输出

### 标题
人工智能技术演进与企业应用：2025年全景分析

### 正文初稿

{input_text[:200] if input_text.startswith('##') else ''}

人工智能（AI）正以前所未有的速度重塑全球商业格局。根据最新研究数据，2024年全球AI市场规模已突破1200亿美元，预计到2027年将超过3000亿美元。这一增长由三大核心驱动力支撑：大语言模型（LLM）的能力飞跃、企业数字化转型的加速，以及各国政府对AI技术的高度重视。

在企业应用层面，AI的渗透率正在从"早期采用者"阶段向"主流采纳"阶段过渡。调查显示，72%的企业已在至少一个业务环节中部署了AI解决方案。最常见的应用场景包括：智能客服（38%）、文档自动化处理（31%）、数据分析与洞察（27%）、代码生成与审查（22%）。

然而，AI在企业中的落地并非一帆风顺。数据质量不足、人才短缺、合规风险，以及技术选型困惑，仍然是阻碍AI价值实现的主要瓶颈。

### 章节结构
1. 引言：AI正在改变世界
2. 技术现状：从GPT到多模态
3. 企业应用：场景、案例与ROI
4. 挑战与对策：数据、人才、合规
5. 未来展望：AGI还有多远？
"""


def _simulate_review(input_text: str) -> str:
    """模拟审校Agent：检查事实准确性、逻辑连贯性、格式规范。"""
    line_count = input_text.count('\n') + 1
    word_count = len(input_text)

    return f"""## 审校阶段输出

### 审校报告

**总体评分**：7.5/10

**优点**：
- 结构清晰，章节划分合理
- 数据引用充分，有具体数字支撑
- 语言流畅，专业性与可读性平衡得当

**需要改进**：
1. 第2段"1200亿美元"在不同章节中表述不一致，请统一为"1200亿美元（约合8700亿人民币）"
2. 第3段"72%的企业..."缺少引用来源年份，建议补充为"(McKinsey, 2025)"
3. "AGI还有多远"这一节标题过于口语化，建议改为"通向通用人工智能的路径"
4. 建议在第4章增加一个关于"负责任AI(Responsible AI)"的子节

**统计数据**：
- 总字数：约{word_count}字
- 总行数：{line_count}行
- 需修改项：4项
- 建议项：2项

---

{input_text[:200]}...

[以上为审校建议，请根据建议修改原文]
"""


def _simulate_polish(input_text: str) -> str:
    """模拟润色Agent：语言优化、格式统一、最终打磨。"""
    return f"""## 润色阶段输出

### 最终版本

> *以下为经过调研→写作→审校→润色四阶段处理后的最终输出*

人工智能技术演进与企业应用：2025年全景分析
==============================================

**摘要**：本文全面分析了2024-2025年全球人工智能技术发展态势与企业应用现状。
研究基于McKinsey、Stanford HAI、Gartner等权威机构的最新数据，系统梳理了
技术演进轨迹、企业采纳模式、关键挑战和未来趋势。

---

（正文经过语言润色，已消除所有审校阶段指出的问题）

关键改进：
- ✅ 数据口径统一（全部折算为2024年美元不变价）
- ✅ 补充了引用的年份和来源
- ✅ 章节标题规范化
- ✅ 增加了"负责任AI"小节
- ✅ 整体语言风格统一为学术报告体
- ✅ 补充了3处数据可视化建议

---

**最终字数**：约4,200字（含摘要和参考文献）
**生成时间**：全部四个阶段总耗时记录在Pipeline日志中
**置信度**：高（所有关键数据均有可追溯来源）
"""


# ============================================================================
# Pipeline 核心数据结构
# ============================================================================

@dataclass
class AgentStage:
    """流水线中的一个Agent阶段。"""
    name: str                          # Agent名称，如 "ResearchAgent"
    role: str                          # 角色描述，如 "研究员"
    system_prompt: str                 # 系统提示词
    model: str = "gpt-4o-mini"         # 使用的模型
    temperature: float = 0.7           # 温度参数
    simulate_delay: float = 0.0        # 模拟延迟（秒），仅Demo使用

    def __post_init__(self):
        """验证阶段配置。"""
        if not self.name.strip():
            raise ValueError("AgentStage name 不能为空")
        if not self.system_prompt.strip():
            raise ValueError(f"Agent {self.name} 的 system_prompt 不能为空")


@dataclass
class PipelineResult:
    """流水线执行结果。"""
    stage_name: str                    # 阶段名称
    input_preview: str                 # 输入摘要（前200字符）
    output: str                        # 完整输出
    start_time: float                  # 开始时间（time.time()）
    end_time: float                    # 结束时间
    elapsed_seconds: float             # 耗时（秒）
    model: str                         # 使用的模型
    error: Optional[str] = None        # 错误信息（如果有）
    token_estimate: int = 0            # 估算的Token数

    @property
    def elapsed_formatted(self) -> str:
        """格式化的耗时显示。"""
        if self.elapsed_seconds < 1:
            return f"{self.elapsed_seconds * 1000:.0f}ms"
        elif self.elapsed_seconds < 60:
            return f"{self.elapsed_seconds:.1f}s"
        else:
            minutes = int(self.elapsed_seconds // 60)
            seconds = self.elapsed_seconds % 60
            return f"{minutes}m {seconds:.0f}s"

    @property
    def input_hash(self) -> str:
        """输入的SHA256哈希（用于追踪）。"""
        return hashlib.sha256(self.input_preview.encode()).hexdigest()[:12]


# ============================================================================
# SequentialPipeline 核心类
# ============================================================================

class SequentialPipeline:
    """
    串行多Agent流水线。

    使用方法：
        pipeline = SequentialPipeline(name="内容生成流水线")
        pipeline.add_agent(
            name="ResearchAgent",
            role="研究员",
            system_prompt="你是资深研究员，负责收集和分析信息...",
            model="gpt-4o-mini"
        )
        pipeline.add_agent(
            name="WritingAgent",
            role="撰稿人",
            system_prompt="你是专业撰稿人，负责撰写高质量内容...",
            model="gpt-4o"
        )
        result = pipeline.run("请写一篇关于AI发展趋势的文章")
    """

    def __init__(self, name: str = "Sequential Pipeline"):
        self.name = name
        self.stages: list[AgentStage] = []
        self.results: list[PipelineResult] = []
        self._on_stage_start: Optional[Callable] = None
        self._on_stage_end: Optional[Callable] = None
        self._llm_call_fn: Callable = simulate_llm_call

    # ---- Agent管理 ----

    def add_agent(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        simulate_delay: float = 0.0,
    ) -> "SequentialPipeline":
        """
        向流水线添加一个Agent阶段。

        Args:
            name: Agent名称（如 "ResearchAgent"）
            role: 角色描述（中文，如 "研究员"）
            system_prompt: 系统提示词
            model: 使用的模型ID
            temperature: 生成温度（0-2）
            simulate_delay: 模拟延迟（Demo用）

        Returns:
            self，支持链式调用
        """
        stage = AgentStage(
            name=name,
            role=role,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            simulate_delay=simulate_delay,
        )
        self.stages.append(stage)
        return self

    def remove_agent(self, name: str) -> bool:
        """按名称移除Agent。返回是否成功。"""
        for i, stage in enumerate(self.stages):
            if stage.name == name:
                self.stages.pop(i)
                return True
        return False

    # ---- 钩子函数 ----

    def on_stage_start(self, fn: Callable) -> Callable:
        """装饰器：注册阶段开始前的回调。"""
        self._on_stage_start = fn
        return fn

    def on_stage_end(self, fn: Callable) -> Callable:
        """装饰器：注册阶段结束后的回调。"""
        self._on_stage_end = fn
        return fn

    # ---- 执行 ----

    def run(
        self,
        initial_input: str,
        verbose: bool = True,
    ) -> list[PipelineResult]:
        """
        运行整个流水线。

        Args:
            initial_input: 初始输入文本
            verbose: 是否打印详细日志

        Returns:
            每个阶段的执行结果列表
        """
        if not self.stages:
            raise ValueError("流水线中没有Agent阶段，请先调用 add_agent()")

        self.results = []
        current_input = initial_input
        total_start = time.time()

        if verbose:
            print("=" * 72)
            print(f"  Pipeline: {self.name}")
            print(f"  阶段数: {len(self.stages)} | 开始时间: {datetime.now().isoformat()}")
            print("=" * 72)

        for i, stage in enumerate(self.stages):
            stage_start = time.time()

            if verbose:
                print(f"\n{'─' * 48}")
                print(f"  [{i + 1}/{len(self.stages)}] {stage.name} ({stage.role})")
                print(f"  模型: {stage.model} | 温度: {stage.temperature}")
                print(f"  输入长度: {len(current_input)} 字符")
                print(f"{'─' * 48}")

            # 触发阶段开始钩子
            if self._on_stage_start:
                self._on_stage_start(stage, i, current_input)

            # 执行LLM调用
            error = None
            output = ""
            try:
                output = self._llm_call_fn(
                    system_prompt=stage.system_prompt,
                    user_input=current_input,
                    model=stage.model,
                    temperature=stage.temperature,
                    delay=stage.simulate_delay,
                )
            except Exception as e:
                error = str(e)
                output = f"[ERROR] {stage.name} 执行失败: {error}"
                if verbose:
                    print(f"  ❌ 错误: {error}")
                # 出错时仍然继续（将错误输出传递给下一个阶段）
                # 生产环境中可以改为 raise 或使用降级策略

            stage_end = time.time()
            elapsed = stage_end - stage_start

            # 估算Token数（简单按字符数/4估算，中英文混合约3-4字符/token）
            token_estimate = len(current_input) // 4 + len(output) // 3

            result = PipelineResult(
                stage_name=stage.name,
                input_preview=current_input[:200],
                output=output,
                start_time=stage_start,
                end_time=stage_end,
                elapsed_seconds=elapsed,
                model=stage.model,
                error=error,
                token_estimate=token_estimate,
            )
            self.results.append(result)

            if verbose:
                status = "✅" if error is None else "⚠️"
                print(f"  {status} 完成 | 耗时: {result.elapsed_formatted} | "
                      f"输出: {len(output)} 字符 | 估算 ~{token_estimate} tokens")
                # 显示输出摘要
                output_preview = output[:150].replace('\n', ' ')
                print(f"  输出摘要: {output_preview}...")

            # 将当前阶段的输出作为下一阶段的输入
            current_input = output

        total_elapsed = time.time() - total_start

        if verbose:
            print(f"\n{'=' * 72}")
            print(f"  流水线完成 | 总耗时: {total_elapsed:.1f}s")
            self._print_timing_summary()
            print(f"{'=' * 72}")

        return self.results

    def _print_timing_summary(self):
        """打印各阶段耗时分布。"""
        if not self.results:
            return
        total = sum(r.elapsed_seconds for r in self.results)
        print(f"\n  📊 各阶段耗时分布：")
        print(f"  {'阶段':<25} {'耗时':<12} {'占比':<10} {'模型':<18} {'Token估':<10}")
        print(f"  {'─' * 25} {'─' * 12} {'─' * 10} {'─' * 18} {'─' * 10}")
        for r in self.results:
            pct = (r.elapsed_seconds / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {r.stage_name:<25} {r.elapsed_formatted:<12} "
                  f"{pct:5.1f}% {bar:<4} {r.model:<18} ~{r.token_estimate:<10}")

    # ---- 结果查询 ----

    def get_final_output(self) -> Optional[str]:
        """获取流水线的最终输出（最后一个阶段的结果）。"""
        if self.results:
            return self.results[-1].output
        return None

    def get_stage_result(self, stage_name: str) -> Optional[PipelineResult]:
        """按阶段名称获取结果。"""
        for r in self.results:
            if r.stage_name == stage_name:
                return r
        return None

    def to_dict(self) -> dict:
        """将流水线结果序列化为字典（用于日志/存储）。"""
        return {
            "pipeline_name": self.name,
            "total_stages": len(self.stages),
            "total_elapsed": sum(r.elapsed_seconds for r in self.results) if self.results else 0,
            "stages": [
                {
                    "name": r.stage_name,
                    "model": r.model,
                    "elapsed_s": round(r.elapsed_seconds, 3),
                    "output_length": len(r.output),
                    "error": r.error,
                    "input_hash": r.input_hash,
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """序列化为JSON字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # ---- 流水线检查 ----

    def validate(self) -> list[str]:
        """
        验证流水线配置的有效性。
        返回问题列表（空列表表示配置有效）。
        """
        issues = []
        if len(self.stages) < 2:
            issues.append("流水线至少需要2个Agent阶段才能体现多Agent协作")

        models_used = set()
        for stage in self.stages:
            models_used.add(stage.model)
            if not stage.system_prompt.strip():
                issues.append(f"{stage.name} 的 system_prompt 为空")
            if len(stage.system_prompt) < 20:
                issues.append(f"{stage.name} 的 system_prompt 过短（<20字符），可能效果不佳")

        return issues


# ============================================================================
# 预定义流水线工厂
# ============================================================================

def create_content_creation_pipeline() -> SequentialPipeline:
    """
    创建"内容创作"预定义流水线：
    ResearchAgent → WritingAgent → ReviewAgent → PolishAgent
    """
    return (
        SequentialPipeline(name="内容创作四阶段流水线")
        .add_agent(
            name="ResearchAgent",
            role="研究员",
            system_prompt=(
                "你是资深研究分析师，拥有15年行业研究经验。你的任务是：\n"
                "1. 深入理解用户提出的主题和需求\n"
                "2. 收集和组织关键信息、数据、趋势\n"
                "3. 列出重要的事实、引用和统计数字\n"
                "4. 识别不同观点和争议点\n"
                "5. 输出结构化的研究简报，为后续写作提供素材\n\n"
                "输出格式要求：使用Markdown，包含'关键发现'、'数据要点'和'参考资料'三个部分。"
            ),
            model="gpt-4o-mini",
        )
        .add_agent(
            name="WritingAgent",
            role="撰稿人",
            system_prompt=(
                "你是资深科技撰稿人，拥有10年专业写作经验。你的任务是：\n"
                "1. 基于研究报告撰写引人入胜的原创内容\n"
                "2. 使用清晰、流畅、准确的语言\n"
                "3. 合理组织章节结构（引言→正文→结论）\n"
                "4. 正确引用前序研究中的所有数据和事实\n"
                "5. 保持专业但不过于学术化的写作风格\n\n"
                "输出格式要求：使用Markdown，包含明确的标题层级和段落结构。"
            ),
            model="gpt-4o",
        )
        .add_agent(
            name="ReviewAgent",
            role="审校员",
            system_prompt=(
                "你是资深编辑和事实核查员。你的任务是：\n"
                "1. 逐段检查文章的准确性和完整性\n"
                "2. 验证数据一致性（同一数字在不同位置不应矛盾）\n"
                "3. 检查逻辑链条是否完整\n"
                "4. 识别需要补充或修改的内容\n"
                "5. 给出具体、可执行的修改建议\n\n"
                "输出格式：先总结评分，再逐条列出问题和建议。不要重写全文。"
            ),
            model="gpt-4o-mini",
        )
        .add_agent(
            name="PolishAgent",
            role="润色师",
            system_prompt=(
                "你是资深文字编辑和排版专家。你的任务是：\n"
                "1. 根据审校建议修正所有问题\n"
                "2. 优化语言表达，消除冗余和歧义\n"
                "3. 统一全文风格和术语\n"
                "4. 确保所有引用格式正确\n"
                "5. 输出可以直接发布的高质量终稿\n\n"
                "输出格式：完整的最终版本，包含所有章节。"
            ),
            model="gpt-4o-mini",
        )
    )


def create_code_review_pipeline() -> SequentialPipeline:
    """
    创建"代码审查"预定义流水线：
    StaticAnalyzer → SecurityChecker → PerformanceAuditor → FinalReviewer
    """
    return (
        SequentialPipeline(name="代码审查四阶段流水线")
        .add_agent(
            name="StaticAnalyzer",
            role="静态分析器",
            system_prompt=(
                "你是资深代码审查专家。你的任务是：\n"
                "1. 检查代码风格是否符合PEP 8规范\n"
                "2. 识别潜在的逻辑错误和边界条件\n"
                "3. 检查类型注解是否完整和正确\n"
                "4. 评估代码可读性和可维护性\n\n"
                "输出：按严重程度（Critical/Major/Minor/Suggestion）分类的问题列表。"
            ),
            model="gpt-4o-mini",
        )
        .add_agent(
            name="SecurityChecker",
            role="安全检查器",
            system_prompt=(
                "你是资深应用安全专家（CISSP）。你的任务是：\n"
                "1. 检查输入验证和输出编码\n"
                "2. 识别SQL注入、XSS、CSRF等常见漏洞\n"
                "3. 审核密钥管理和敏感数据处理\n"
                "4. 验证依赖库安全性\n\n"
                "输出：按OWASP Top 10分类的安全问题及修复建议。"
            ),
            model="gpt-4o-mini",
        )
        .add_agent(
            name="PerformanceAuditor",
            role="性能审计师",
            system_prompt=(
                "你是资深性能工程师。你的任务是：\n"
                "1. 分析算法复杂度（时间/空间）\n"
                "2. 识别N+1查询和低效循环\n"
                "3. 检查异步/并发使用是否合理\n"
                "4. 建议缓存策略和数据库索引优化\n\n"
                "输出：性能瓶颈列表及优化建议（附估算改善幅度）。"
            ),
            model="gpt-4o-mini",
        )
        .add_agent(
            name="FinalReviewer",
            role="最终审查员",
            system_prompt=(
                "你是资深技术主管。你的任务是：\n"
                "1. 汇总前三阶段的所有发现\n"
                "2. 按优先级排序所有问题\n"
                "3. 对整体代码质量给出评分（1-10）\n"
                "4. 输出最终审查报告\n\n"
                "输出格式：结构化的代码审查报告（Markdown）。"
            ),
            model="gpt-4o",
        )
    )


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  多智能体系统 - 串行流水线（Sequential Pipeline）演示")
    print("=" * 72)

    # ---- Demo 1: 内容创作流水线 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 内容创作流水线（Research → Writing → Review → Polish）")
    print("=" * 72)

    pipeline1 = create_content_creation_pipeline()

    # 验证流水线配置
    issues = pipeline1.validate()
    if issues:
        print("\n  ⚠️ 流水线配置问题：")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  ✅ 流水线配置验证通过")

    # 运行
    results1 = pipeline1.run(
        initial_input="请撰写一篇题为'2025年企业AI应用趋势'的深度分析文章，"
                      "要求包含数据支撑和行业案例，面向企业决策者。",
        verbose=True,
    )

    # 输出最终结果
    print(f"\n{'─' * 72}")
    print("  最终输出（前500字符）:")
    print(f"{'─' * 72}")
    final = pipeline1.get_final_output()
    if final:
        print(final[:500])
        print(f"\n  ...（共 {len(final)} 字符）")

    # 导出JSON
    print(f"\n{'─' * 72}")
    print("  流水线执行摘要（JSON）:")
    print(f"{'─' * 72}")
    print(pipeline1.to_json())

    # ---- Demo 2: 代码审查流水线 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 代码审查流水线（Static → Security → Performance → Final）")
    print("=" * 72)

    sample_code = '''
import os
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

class UserService:
    def __init__(self):
        self.api_key = "sk-abc123def456ghi789"

    def fetch_all_users(self):
        users = []
        for i in range(1000):
            user = get_user(i)
            users.append(user)
        return users
'''

    pipeline2 = create_code_review_pipeline()
    results2 = pipeline2.run(
        initial_input=f"请审查以下Python代码：\n\n```python\n{sample_code}\n```",
        verbose=True,
    )

    final2 = pipeline2.get_final_output()
    if final2:
        print(f"\n  审查报告（前400字符）:")
        print(f"  {final2[:400]}...")

    print("\n" + "=" * 72)
    print("  所有流水线演示完成！")
    print("=" * 72)
