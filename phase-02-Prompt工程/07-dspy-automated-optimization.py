#!/usr/bin/env python3
"""
07 - DSPy 自动优化 (DSPy Automated Optimization)

本模块演示使用 DSPy 编译器自动优化提示词和 Few-shot 示例。
DSPy 将 LLM 调用抽象为可编程模块，通过"编译"自动优化提示和 Few-shot 选择。

核心概念:
- Signature: 定义 LLM 的输入/输出字段（类似函数签名）
- Module: 一个或多个 Signatures 编排而成的处理单元
- Optimizer: 编译时优化器，自动选择最佳 Few-shot 示例和指令
- Evaluate: 在 devset 上评估编译后程序的性能
- CompiledProgram: 优化后的最终程序，可直接保存和加载

本模块包含:
1. DSPy 环境初始化（LM 配置）
2. 自定义 Signatures: Summarize, Classify, QA
3. Modules: RAGModule, MultiHopQA
4. Optimizers: BootstrapFewShot, MIPROv2, BootstrapFewShotWithRandomSearch
5. 评估与对比: 基线 vs 优化后
6. 模型的保存与加载

所有注释使用中文，代码标识符使用英文。

运行前安装: pip install dspy-ai openai
"""

import os
import json
import sys
from typing import Any
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# DSPy 安装检查
# ---------------------------------------------------------------------------

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠ DSPy 未安装。请运行: pip install dspy-ai")
    print("  本模块的大部分功能需要 DSPy 支持。")
    print("  部分基础演示仍可运行。")


# ---------------------------------------------------------------------------
# 1. DSPy 环境初始化
# ---------------------------------------------------------------------------

def init_dspy(model: str = "gpt-4o-mini", api_key: str | None = None):
    """
    初始化 DSPy 环境，配置语言模型。

    What: 设置 DSPy 使用的 LLM 后端。
    Why: DSPy 需要 LM 实例来执行生成和优化操作。
    When: 在任何 DSPy 操作之前调用。

    Args:
        model: OpenAI 模型名称
        api_key: OpenAI API Key（默认从环境变量读取）
    """
    if not DSPY_AVAILABLE:
        raise RuntimeError("DSPy 未安装。请运行: pip install dspy-ai")

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    # 配置 LM
    lm = dspy.LM(
        model=f"openai/{model}",
        api_key=api_key,
        api_base=os.environ.get("OPENAI_BASE_URL", None),
        temperature=0.0,
        max_tokens=1024,
    )

    dspy.configure(lm=lm)

    print(f"DSPy 已初始化: model={model}")
    return lm


# ---------------------------------------------------------------------------
# 2. 定义 Signatures
# ---------------------------------------------------------------------------

def create_signatures():
    """
    创建 DSPy Signatures：定义 LLM 的输入/输出规范。

    What: 每个 Signature 定义了一个 LLM 任务 — 输入什么字段，输出什么字段。
    Why: Signature 是 DSPy 的核心抽象，编译器知道如何优化任何 Signature。
    When: DSPy 项目的第一步 — 定义你的任务。

    Returns:
        一个包含所有 Signature 定义的模块
    """

    class Summarize(dspy.Signature):
        """
        摘要任务 Signature。

        输入: text (原文)
        输出: summary (3句话中文摘要), key_points (3-5个关键点)
        """
        text: str = dspy.InputField(desc="原文内容")
        summary: str = dspy.OutputField(desc="3句话的中文摘要，保留核心信息")
        key_points: list[str] = dspy.OutputField(desc="3-5个关键信息点，每点一行")

    class Classify(dspy.Signature):
        """
        分类任务 Signature。

        输入: text (待分类文本), categories (可选类别列表)
        输出: category (分类结果), confidence (置信度 0-1), reason (分类理由)
        """
        text: str = dspy.InputField(desc="待分类的文本")
        categories: str = dspy.InputField(desc="可选类别，逗号分隔，如 '科技,体育,娱乐'")
        category: str = dspy.OutputField(desc="分类结果，必须从 categories 中选择")
        confidence: float = dspy.OutputField(desc="置信度，0.0 到 1.0")
        reason: str = dspy.OutputField(desc="分类理由，一句话说明")

    class QA(dspy.Signature):
        """
        问答任务 Signature。

        输入: question (问题), context (参考上下文)
        输出: answer (答案), evidence (引用原文中的证据)
        """
        question: str = dspy.InputField(desc="用户的问题")
        context: str = dspy.InputField(desc="参考上下文，用于查找答案")
        answer: str = dspy.OutputField(desc="问题的答案，精确简洁")
        evidence: str = dspy.OutputField(desc="从 context 中引用支持答案的原文")

    class CodeReview(dspy.Signature):
        """
        代码审查任务 Signature。

        输入: code (源代码), language (编程语言)
        输出: issues (问题列表), severity (严重程度), suggestions (改进建议)
        """
        code: str = dspy.InputField(desc="源代码")
        language: str = dspy.InputField(desc="编程语言")
        issues: list[str] = dspy.OutputField(desc="发现的问题列表")
        severity: str = dspy.OutputField(desc="总体严重程度: low/medium/high/critical")
        suggestions: list[str] = dspy.OutputField(desc="改进建议列表")

    print("Signatures 已定义: Summarize, Classify, QA, CodeReview")
    return Summarize, Classify, QA, CodeReview


# ---------------------------------------------------------------------------
# 3. 构建 DSPy Modules
# ---------------------------------------------------------------------------

class RAGModule(dspy.Module):
    """
    RAG (Retrieval-Augmented Generation) 模块。

    What: 实现标准的 RAG 管道：检索 → 上下文合并 → 生成答案。
    Why: 展示 DSPy 如何编排多个 LLM 调用来完成复杂任务。
    When: 构建问答系统、客服机器人、知识库查询等场景。

    工作流:
    User Query → 检索相关上下文 → [查询+上下文] → QA Signature → 答案
    """

    def __init__(self, passages: list[str] | None = None):
        """
        Args:
            passages: 知识库段落列表（如果为 None，使用内置示例）
        """
        super().__init__()

        # 知识库
        self.passages = passages or [
            "大语言模型（LLM）基于 Transformer 架构，通过自注意力机制处理文本序列。GPT 系列使用 Decoder-only 架构，通过自回归方式生成文本。",
            "Python 3.12 引入了多项性能改进，包括更快的解释器和改进的错误消息。f-string 语法更加灵活，支持多行表达式。",
            "Docker 是一个容器化平台，通过镜像和容器实现应用的封装和隔离。Docker Compose 允许定义和运行多容器应用。",
            "机器学习模型部署的常见挑战包括：模型版本管理、延迟要求、资源消耗、监控和回滚策略。MLOps 的最佳实践包括 CI/CD 管道和 A/B 测试。",
            "微服务架构将单体应用拆分为小型、独立的服务。每个服务有自己的数据库、部署周期和扩缩容策略。服务间通过 API 或消息队列通信。",
        ]

        # 定义检索器（简化版：基于关键词的相似度匹配）
        self.retrieve = dspy.ChainOfThought("query -> top_k_passages: str")

        # 定义 QA 生成器
        self.generate_answer = dspy.ChainOfThought("context, question -> answer, evidence")

    def forward(self, question: str) -> dspy.Prediction:
        """
        处理用户问题。

        Args:
            question: 用户问题

        Returns:
            dspy.Prediction: 包含 answer 和 evidence
        """
        # 步骤 1: 检索相关上下文（使用 DSPy 的检索提示）
        # 将知识库暴露给检索步骤
        kb_text = "\n---\n".join(
            f"[{i}] {p}" for i, p in enumerate(self.passages)
        )

        retrieved = self.retrieve(
            query=f"问题: {question}\n\n知识库:\n{kb_text}\n\n请选出与问题最相关的2-3个段落编号和内容。"
        )

        # 使用检索到的内容或全部知识库
        context = retrieved.top_k_passages if hasattr(retrieved, 'top_k_passages') else kb_text[:2000]

        # 步骤 2: 基于上下文生成答案
        result = self.generate_answer(
            context=context if isinstance(context, str) else str(context),
            question=question,
        )

        return dspy.Prediction(
            answer=getattr(result, 'answer', ''),
            evidence=getattr(result, 'evidence', ''),
        )


class MultiHopQA(dspy.Module):
    """
    多跳问答模块。

    What: 实现需要多个推理步骤的问答 — 将复杂问题分解为子问题，逐步解答。
    Why: 某些问题需要先回答子问题才能得出最终答案（如"谁更大：A的CEO还是B的CEO？"）
    When: 问题需要跨越多个事实才能回答时。
    """

    def __init__(self, knowledge_base: dict[str, str] | None = None):
        super().__init__()
        self.kb = knowledge_base or {}

        # 子问题分解
        self.decompose = dspy.ChainOfThought(
            "question, knowledge -> sub_questions: list[str]"
        )

        # 单步回答
        self.answer_step = dspy.ChainOfThought(
            "question, context -> answer"
        )

        # 最终综合
        self.synthesize = dspy.ChainOfThought(
            "original_question, sub_answers -> final_answer, reasoning"
        )

    def forward(self, question: str) -> dspy.Prediction:
        """
        多跳问答处理。

        Args:
            question: 复杂问题

        Returns:
            dspy.Prediction: 包含 final_answer 和 reasoning
        """
        kb_text = "\n".join(f"- {k}: {v}" for k, v in self.kb.items())

        # 步骤 1: 分解问题
        decomp = self.decompose(
            question=question,
            knowledge=kb_text,
        )
        sub_questions = getattr(decomp, 'sub_questions', [question])

        # 步骤 2: 逐一回答子问题
        sub_answers = {}
        for sq in sub_questions:
            ans = self.answer_step(question=sq, context=kb_text)
            sub_answers[sq] = getattr(ans, 'answer', '')

        # 步骤 3: 综合最终答案
        summary = "\n".join(f"Q: {q}\nA: {a}" for q, a in sub_answers.items())
        final = self.synthesize(
            original_question=question,
            sub_answers=summary,
        )

        return dspy.Prediction(
            final_answer=getattr(final, 'final_answer', ''),
            reasoning=getattr(final, 'reasoning', ''),
            sub_answers=str(sub_answers),
        )


# ---------------------------------------------------------------------------
# 4. 评估指标
# ---------------------------------------------------------------------------

def evaluate_answer_similarity(example, prediction, trace=None) -> float:
    """
    答案相似度评估指标。

    What: 计算预测答案与参考答案的语义相似度。
    Why: 数字化的评估分数，让优化器知道哪些 Few-shot 示例有效。
    When: 作为 dspy.Evaluate 的 metric 参数。

    Returns:
        0.0 到 1.0 的相似度得分
    """
    # 获取预测文本
    if hasattr(prediction, 'answer'):
        pred_text = prediction.answer
    elif hasattr(prediction, 'final_answer'):
        pred_text = prediction.final_answer
    else:
        pred_text = str(prediction)

    # 获取参考答案
    ref_text = getattr(example, 'answer', '') or getattr(example, 'expected', '')

    if not ref_text:
        return 0.0

    # 简单的词重叠相似度（生产环境使用 embedding 相似度或 LLM-as-Judge）
    pred_words = set(pred_text.lower().split())
    ref_words = set(ref_text.lower().split())

    if not ref_words:
        return 0.0

    overlap = len(pred_words & ref_words)
    return overlap / len(ref_words)


def evaluate_classification_accuracy(example, prediction, trace=None) -> float:
    """
    分类准确率指标。

    Returns:
        1.0 如果分类正确，否则 0.0
    """
    pred_category = getattr(prediction, 'category', '').strip().lower()
    expected_category = getattr(example, 'category', '').strip().lower()

    if not expected_category:
        return 0.0

    return 1.0 if pred_category == expected_category else 0.0


# ---------------------------------------------------------------------------
# 5. 准备数据集
# ---------------------------------------------------------------------------

def create_sample_datasets():
    """
    创建示例训练集和开发集。

    What: 为 DSPy 编译器提供标注数据。
    Why: 优化器需要示例来学习有效的 Few-shot 和指令。
    When: 编译 Module 之前。

    Returns:
        (trainset, devset)
    """

    # 摘要数据
    summarize_trainset = [
        dspy.Example(
            text="Python 3.12 发布了多项新特性，包括改进的错误消息、更快的解释器性能、以及对类型提示的增强支持。PEP 701 引入了更灵活的 f-string 语法。",
            summary="Python 3.12 引入了改进的错误消息、更快的解释器性能、增强的类型提示和更灵活的 f-string 语法。",
            key_points=["改进的错误消息", "更快的解释器性能", "增强的类型提示", "更灵活的 f-string 语法"],
        ).with_inputs("text"),
        dspy.Example(
            text="Docker Compose 允许用户通过 YAML 文件定义多容器应用。使用 docker-compose up 命令可以一键启动所有服务。容器间通过服务名互相访问。",
            summary="Docker Compose 通过 YAML 文件定义多容器应用，使用 docker-compose up 一键启动。",
            key_points=["YAML 文件定义多容器应用", "docker-compose up 一键启动", "容器通过服务名互访"],
        ).with_inputs("text"),
    ]

    # 分类数据
    classify_trainset = [
        dspy.Example(
            text="全新发布的智能手机采用了折叠屏设计，支持5G网络和卫星通信。",
            categories="科技,体育,娱乐,财经",
            category="科技",
            confidence=0.95,
            reason="涉及智能手机、5G和卫星通信等技术话题",
        ).with_inputs("text", "categories"),
        dspy.Example(
            text="中国队在奥运会上再次夺得金牌，刷新了世界纪录。",
            categories="科技,体育,娱乐,财经",
            category="体育",
            confidence=0.98,
            reason="涉及奥运会、金牌和世界纪录等体育话题",
        ).with_inputs("text", "categories"),
        dspy.Example(
            text="央行宣布降息0.25个百分点，股市应声上涨，房地产板块表现活跃。",
            categories="科技,体育,娱乐,财经",
            category="财经",
            confidence=0.93,
            reason="涉及央行、降息、股市和房地产等金融话题",
        ).with_inputs("text", "categories"),
    ]

    # QA 数据
    qa_trainset = [
        dspy.Example(
            question="Python 3.12 引入了什么新特性？",
            answer="Python 3.12 引入了改进的错误消息、更快的解释器性能和更灵活的 f-string 语法。",
        ).with_inputs("question"),
        dspy.Example(
            question="Docker Compose 的作用是什么？",
            answer="Docker Compose 允许通过 YAML 文件定义和运行多容器 Docker 应用。",
        ).with_inputs("question"),
    ]

    # 合并
    all_trainset = summarize_trainset + classify_trainset + qa_trainset
    devset = summarize_trainset[:1] + classify_trainset[:1] + qa_trainset[:1]

    print(f"训练集: {len(all_trainset)} 个示例")
    print(f"开发集: {len(devset)} 个示例")

    return all_trainset, devset


# ---------------------------------------------------------------------------
# 6. 优化器演示
# ---------------------------------------------------------------------------

def demo_bootstrap_fewshot(SummarizeSignature, trainset, devset):
    """
    演示 BootstrapFewShot 优化器。

    What: 自动选择最佳的 Few-shot 示例注入到提示中。
    Why: 手工挑选 Few-shot 示例费时且不精确；BootstrapFewShot 自动发现最有效的示例。
    When: 有少量标注数据（10-50条），想自动找到最佳示例编排。

    Args:
        SummarizeSignature: 摘要任务 Signature
        trainset: 训练集
        devset: 开发集

    Returns:
        编译后的程序
    """
    print("\n=== BootstrapFewShot 优化演示 ===")

    # 创建基线模块
    baseline = dspy.ChainOfThought(SummarizeSignature)
    print(f"基线模块: ChainOfThought({SummarizeSignature.__name__})")

    # 设置优化器（max_labeled_demos=4 限制 Few-shot 数量）
    optimizer = dspy.BootstrapFewShot(
        metric=evaluate_answer_similarity,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        max_rounds=1,
    )

    print("优化配置: max_demos=4, max_rounds=1")

    # 编译
    compiled = optimizer.compile(
        student=baseline,
        trainset=[ex for ex in trainset if hasattr(ex, 'text')],  # 只选摘要相关
    )

    print("编译完成! Few-shot 示例已自动选择。")
    return compiled


def demo_mipro_v2(SummarizeSignature, trainset, devset):
    """
    演示 MIPROv2 优化器（自动模式）。

    What: MIPROv2 (Multi-Instruction Prompt Optimization v2) 同时优化指令和 Few-shot 示例。
    Why: 不仅选示例，还优化提示词本身 — 比纯 Few-shot 优化更强大。
    When: 对提示质量要求高，且愿意投入更多编译时间的场景。

    Args:
        SummarizeSignature: 摘要任务 Signature
        trainset: 训练集
        devset: 开发集

    Returns:
        编译后的程序
    """
    print("\n=== MIPROv2 优化演示 ===")

    baseline = dspy.ChainOfThought(SummarizeSignature)

    # MIPROv2 在 "light" 模式下运行更快
    try:
        optimizer = dspy.MIPROv2(
            metric=evaluate_answer_similarity,
            auto="light",  # 轻量模式：更快但覆盖略少
            num_threads=2,
        )

        print("优化配置: auto='light', num_threads=2")

        compiled = optimizer.compile(
            student=baseline,
            trainset=trainset,
            max_bootstrapped_demos=2,
            max_labeled_demos=3,
            requires_permission_to_run=False,
        )

        print("MIPROv2 编译完成!")
        return compiled

    except Exception as e:
        print(f"MIPROv2 优化失败（可能是 API 限制或版本问题）: {e}")
        print("回退到 BootstrapFewShot...")
        return demo_bootstrap_fewshot(SummarizeSignature, trainset, devset)


def demo_random_search(SummarizeSignature, trainset):
    """
    演示 BootstrapFewShotWithRandomSearch 优化器。

    What: 通过随机搜索寻找最佳 Few-shot 组合。
    Why: 比全量搜索快，比单一引导更可能找到全局最优。
    When: 示例池较大，希望探索更多组合但时间有限。

    Args:
        SummarizeSignature: 摘要任务 Signature
        trainset: 训练集

    Returns:
        编译后的程序
    """
    print("\n=== BootstrapFewShotWithRandomSearch 演示 ===")

    baseline = dspy.ChainOfThought(SummarizeSignature)

    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=evaluate_answer_similarity,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        num_candidate_programs=4,
        num_threads=2,
    )

    print("优化配置: max_demos=3, num_candidates=4, num_threads=2")

    compiled = optimizer.compile(
        student=baseline,
        trainset=trainset,
    )

    print("随机搜索编译完成!")
    return compiled


# ---------------------------------------------------------------------------
# 7. 评估与对比
# ---------------------------------------------------------------------------

def compare_baseline_vs_compiled(
    SummarizeSignature,
    trainset,
    devset,
):
    """
    对比基线模型和编译后模型的性能。

    What: 使用 dspy.Evaluate 在 devset 上评估两个模型的性能差异。
    Why: 量化优化效果，验证编译是否确实提升了性能。
    When: 优化完成后，需要验证效果时。

    Args:
        SummarizeSignature: 摘要任务 Signature
        trainset: 训练集（用于编译）
        devset: 开发集（用于评估）

    Returns:
        包含基线分和优化分的字典
    """
    print("\n=== 基线 vs 编译后对比 ===")

    # 基线模型
    baseline = dspy.ChainOfThought(SummarizeSignature)

    # 编译模型
    try:
        optimizer = dspy.BootstrapFewShot(
            metric=evaluate_answer_similarity,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
        )
        compiled = optimizer.compile(
            student=baseline,
            trainset=trainset,
        )
    except Exception as e:
        print(f"编译失败: {e}")
        compiled = baseline

    # 评估基线
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=evaluate_answer_similarity,
        num_threads=2,
        display_progress=True,
        display_table=False,
    )

    print("\n评估基线模型...")
    try:
        baseline_score = evaluator(baseline)
        print(f"基线得分: {baseline_score:.3f}")
    except Exception as e:
        print(f"基线评估失败: {e}")
        baseline_score = 0.0

    # 评估编译后
    print("\n评估编译后模型...")
    try:
        compiled_score = evaluator(compiled)
        print(f"编译后得分: {compiled_score:.3f}")
    except Exception as e:
        print(f"编译后评估失败: {e}")
        compiled_score = 0.0

    # 对比
    improvement = compiled_score - baseline_score
    print(f"\n--- 对比结果 ---")
    print(f"基线得分:   {baseline_score:.4f}")
    print(f"编译后得分: {compiled_score:.4f}")
    print(f"提升:       {improvement:+.4f} ({improvement / max(baseline_score, 0.001) * 100:+.1f}%)")

    return {
        "baseline_score": baseline_score,
        "compiled_score": compiled_score,
        "improvement": improvement,
    }


# ---------------------------------------------------------------------------
# 8. 保存与加载
# ---------------------------------------------------------------------------

def save_compiled_program(program, filepath: str):
    """
    保存编译后的 DSPy 程序到文件。

    What: 将优化后的 Module（含 Few-shot 示例和指令）序列化到磁盘。
    Why: 编译是耗时的（需要多次 LLM 调用），缓存结果避免重复编译。
    When: 优化完成后，需要在生产环境中直接加载使用。

    Args:
        program: 编译后的 DSPy 模块
        filepath: 保存路径（JSON 格式）
    """
    try:
        program.save(filepath)
        print(f"程序已保存到: {filepath}")
    except Exception as e:
        print(f"保存失败: {e}")
        # 备用方案：手动序列化
        try:
            state = {
                "type": type(program).__name__,
                # DSPy 内部状态可通过 program.dump_state() 获取
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            print(f"程序状态已保存（备用方案）: {filepath}")
        except Exception as e2:
            print(f"备用保存也失败: {e2}")


def load_compiled_program(filepath: str, module_class=None):
    """
    加载之前保存的编译程序。

    What: 从磁盘反序列化 DSPy 程序。
    Why: 在生产环境中直接加载优化后的程序，避免重新编译。
    When: 启动服务时需要加载已优化的模型。

    Args:
        filepath: 程序文件路径
        module_class: 模块类（如 ChainOfThought）

    Returns:
        加载的 DSPy 模块
    """
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return None

    try:
        if module_class is None:
            module_class = dspy.ChainOfThought

        program = module_class.__new__(module_class)
        program.load(filepath)
        print(f"程序已从 {filepath} 加载")
        return program
    except Exception as e:
        print(f"加载失败: {e}")
        return None


# ---------------------------------------------------------------------------
# 9. 无需 DSPy 的备用演示（当 DSPy 不可用时）
# ---------------------------------------------------------------------------

def fallback_demo():
    """
    当 DSPy 不可用时的备用演示。

    What: 展示 DSPy 的核心概念和代码结构，无需实际运行 DSPy。
    Why: 确保 notebook/课程在没有安装 DSPy 时仍可教学。
    When: DSPY_AVAILABLE == False 时自动调用。
    """
    print("=" * 60)
    print("DSPy 备用演示（概念讲解模式）")
    print("=" * 60)

    print("""
DSPy 核心理念：将 LLM 调用编程化

1. Signature（签名）
   定义 LLM 的输入/输出接口，类似函数签名。
   示例：
   class Summarize(dspy.Signature):
       text: str = dspy.InputField(desc="原文")
       summary: str = dspy.OutputField(desc="摘要")

2. Module（模块）
   编排多个 LLM 调用，构建复杂管道。
   示例：
   - RAGModule: 检索 → 上下文 → 生成答案
   - MultiHopQA: 问题分解 → 子问题回答 → 综合

3. Optimizer（优化器）
   自动编译 Module，选择最佳 Few-shot 和指令。
   - BootstrapFewShot: 自动选择和排序 Few-shot 示例
   - MIPROv2: 同时优化指令和示例（贝叶斯优化）
   - BootstrapFewShotWithRandomSearch: 随机搜索最优组合

4. Evaluate（评估）
   在 devset 上评估优化效果。
   - 支持自定义 metric（相似度、准确率等）
   - 多线程并行评估

5. 工作流
   Signature 定义 → Module 构建 → 准备数据 → 选择 Optimizer
   → 编译 → 评估 → 保存 → 生产部署

安装 DSPy:
   pip install dspy-ai openai

最小运行示例:
   import dspy

   # 配置
   lm = dspy.LM('openai/gpt-4o-mini', api_key='...')
   dspy.configure(lm=lm)

   # 定义任务
   class Translate(dspy.Signature):
       text: str = dspy.InputField()
       target_language: str = dspy.InputField()
       translation: str = dspy.OutputField()

   # 使用
   translator = dspy.ChainOfThought(Translate)
   result = translator(
       text="Hello, world!",
       target_language="中文"
   )
   print(result.translation)  # "你好，世界！"
""")


# ===========================================================================
# __main__ 完整演示
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DSPy 自动优化 - 完整演示")
    print("=" * 60)

    if not DSPY_AVAILABLE:
        print("\n⚠ DSPy 未安装。运行备用演示...\n")
        fallback_demo()
        print("\n提示: 安装 DSPy 后运行本脚本可看到完整的自动优化流程。")
        print("  pip install dspy-ai openai")
        sys.exit(0)

    # 1. 初始化 DSPy
    print("\n[1] 初始化 DSPy...")
    try:
        init_dspy(model="gpt-4o-mini")
    except Exception as e:
        print(f"初始化失败（可能需要有效的 API Key）: {e}")
        print("继续以演示模式运行...")

    # 2. 创建 Signatures
    print("\n[2] 创建 Signatures...")
    Summarize, Classify, QA, CodeReview = create_signatures()

    # 3. 构建 Module
    print("\n[3] 构建 RAG Module...")
    rag_module = RAGModule()

    print("\n构建 MultiHopQA Module...")
    sample_kb = {
        "Python创始人": "Guido van Rossum 于 1991 年创建了 Python。",
        "Java创始人": "James Gosling 于 1995 年创建了 Java。",
        "Python发布时间": "Python 1.0 于 1994 年 1 月发布。",
        "Java发布时间": "Java 1.0 于 1996 年 1 月发布。",
    }
    multihop_module = MultiHopQA(knowledge_base=sample_kb)

    # 4. 准备数据集
    print("\n[4] 准备数据集...")
    trainset, devset = create_sample_datasets()

    # 先过滤出与特定 Signature 类型相关的示例
    summarize_trainset = [ex for ex in trainset if hasattr(ex, 'text') and not hasattr(ex, 'categories')]
    if not summarize_trainset:
        # 如果没有纯摘要示例，使用全部训练集
        summarize_trainset = trainset

    # 5. 演示优化器
    print("\n[5] 优化器演示...")
    try:
        compiled_bfs = demo_bootstrap_fewshot(Summarize, summarize_trainset, devset)

        # 6. 对比评估
        print("\n[6] 基线 vs 优化后对比...")
        comparison = compare_baseline_vs_compiled(Summarize, summarize_trainset, devset)

        # 7. 保存编译后程序
        if compiled_bfs:
            print("\n[7] 保存编译后程序...")
            save_path = "/tmp/dspy_compiled_program.json"
            save_compiled_program(compiled_bfs, save_path)

            # 8. 加载编译后程序
            print("\n[8] 加载编译后程序...")
            loaded = load_compiled_program(save_path)
            if loaded:
                print("程序加载成功，可直接用于推理。")
    except Exception as e:
        print(f"\n完整演示中遇到错误: {e}")
        print("这可能是由于 API Key 未配置或网络问题。")
        print("将运行概念演示...\n")
        fallback_demo()

    print("\n" + "=" * 60)
    print("DSPy 自动优化演示完成！")
    print("=" * 60)
