# 陷阱 01：Naive RAG 的幻觉问题

> 当检索到的文档被 LLM 无视时，RAG 退化为纯生成 —— 幻觉由此诞生。

---

## 1. 症状（Symptoms）

在 Naive RAG 中，你可能会观察到以下现象：

| 症状 | 表现 | 紧急程度 |
|------|------|----------|
| **答案与检索文档矛盾** | LLM 输出的结论与检索到的文档内容完全相反 | 严重 |
| **凭空捏造数据** | 答案中出现检索文档不存在的数字、日期、人名 | 严重 |
| **过度自信的错误** | 用非常肯定的语气给出错误信息，用户难以察觉 | 严重 |
| **上下文被部分忽略** | 只采纳了检索文档中的几个词，其余全部忽略 | 中等 |
| **Source 标注形同虚设** | 虽然引用了文档编号，但内容实际并非来自该文档 | 中等 |

### 真实案例演示

```python
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# ============================================================
# 模拟 Naive RAG 幻觉场景
# ============================================================

@dataclass
class Document:
    """模拟一个检索到的文档"""
    doc_id: str
    content: str
    source: str
    retrieval_score: float = 0.0

# 场景：用户询问中国民法典关于诉讼时效的规定
query = "中华人民共和国民法典规定的普通诉讼时效是几年？"

# 检索到的文档（正确信息）
retrieved_docs = [
    Document(
        doc_id="civil_code_188",
        content="《中华人民共和国民法典》第一百八十八条：向人民法院请求保护民事权利的诉讼时效期间为三年。"
                "法律另有规定的，依照其规定。诉讼时效期间自权利人知道或者应当知道权利受到损害以及义务人之日起计算。",
        source="民法典第一编第九章",
        retrieval_score=0.92
    ),
    Document(
        doc_id="civil_code_189",
        content="《中华人民共和国民法典》第一百八十九条：当事人约定同一债务分期履行的，"
                "诉讼时效期间自最后一期履行期限届满之日起计算。",
        source="民法典第一编第九章",
        retrieval_score=0.87
    ),
]

# 模拟 Naive RAG 的 LLM 调用（这里用函数模拟实际 LLM 行为）
def simulate_naive_rag_llm(
    query: str,
    documents: List[Document],
    hallucinate: bool = False
) -> Dict:
    """
    模拟 Naive RAG 的 LLM 调用。

    参数 hallucinate=True 时模拟 LLM 忽略检索文档、
    依赖自身参数化知识（可能过时或错误）的情况。
    """
    context = "\n\n".join([f"[{d.doc_id}] {d.content}" for d in documents])

    # 构建 prompt（Naive RAG 的标准做法）
    prompt = f"""基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{query}

请基于以上文档回答。如果文档中没有相关信息，请说明。"""

    if not hallucinate:
        # 正常情况：LLM 遵循指令，基于文档回答
        answer = (
            "根据《中华人民共和国民法典》第一百八十八条，"
            "普通诉讼时效期间为三年。"
        )
        grounded = True
    else:
        # 幻觉情况：LLM 忽略了检索文档，使用了过时的内部知识
        # （民法总则时期的旧规定是两年，民法典已改为三年）
        answer = (
            "根据中国法律规定，普通诉讼时效为两年。"
            "根据《民法通则》第一百三十五条的规定，"
            "向人民法院请求保护民事权利的诉讼时效期间为二年。"
        )
        grounded = False

    return {
        "prompt": prompt,
        "answer": answer,
        "grounded": grounded,
        "retrieved_doc_ids": [d.doc_id for d in documents],
    }

# 运行演示
print("=" * 60)
print("Naive RAG 幻觉演示")
print("=" * 60)

result_normal = simulate_naive_rag_llm(query, retrieved_docs, hallucinate=False)
result_hallucinated = simulate_naive_rag_llm(query, retrieved_docs, hallucinate=True)

print("\n【正常 RAG 输出】")
print(f"答案：{result_normal['answer']}")
print(f"是否基于文档：{'是' if result_normal['grounded'] else '否'}")

print("\n【幻觉 RAG 输出】")
print(f"答案：{result_hallucinated['answer']}")
print(f"是否基于文档：{'是' if result_hallucinated['grounded'] else '否'}")
```

---

## 2. 根因（Root Cause）

幻觉在 Naive RAG 中的根因是多层次的，下面逐层分析。

### 2.1 检索层根因：检索失败

```python
import numpy as np
from collections import Counter

def analyze_retrieval_failure_modes():
    """
    分析检索失败的几种典型模式及其对幻觉的影响。
    """
    failure_modes = {
        "语义漂移（Semantic Drift）": {
            "description": (
                "用户查询的语义意图与文档的实际语义不一致。"
                "例如用户问的是'诉讼时效'的法律概念，但检索系统"
                "匹配到了'诉讼时效'在新闻报道中的用法。"
            ),
            "root_mechanics": (
                "Embedding 模型将 query 和文档映射到向量空间时，"
                "基于分布相似性而非精确语义匹配。在法律、医学等"
                "专业领域，同一个词在不同语境下的含义差异极大。"
            ),
            "detection_signal": "top-k 文档的 retrieval score 普遍低于 0.75",
            "impact_on_hallucination": (
                "LLM 收到不相关文档后，如果 Confidence 很高，"
                "它会忽略这些'噪声'文档，转而使用内部参数化知识，"
                "从而产生幻觉。"
            ),
        },
        "覆盖率不足（Coverage Gap）": {
            "description": (
                "知识库中根本没有包含回答该问题所需的信息。"
                "检索系统仍然返回了 top-k 最'相似'的结果，"
                "但这些结果都无法回答用户问题。"
            ),
            "root_mechanics": (
                "向量检索的本质是'近似最近邻'（ANN），它总是"
                "返回 top-k 结果，无论这些结果是否真正相关。"
                "系统缺少'拒答'机制或相关性阈值过滤。"
            ),
            "detection_signal": "top-1 文档的 retrieval score 低于 0.60",
            "impact_on_hallucination": (
                "最危险的情况：LLM 收到了文档（所以它认为应该基于文档回答），"
                "但文档内容无法回答问题。LLM 被迫'脑补'答案，"
                "产生事实性幻觉。"
            ),
        },
        "多跳检索断裂（Multi-hop Breakdown）": {
            "description": (
                "问题需要多份文档的信息组合才能回答，"
                "但检索系统只返回了部分信息。"
            ),
            "root_mechanics": (
                "单次检索只能捕获 query 的直接语义邻居。"
                "对于需要'A 导致 B，B 导致 C，问 A 与 C 的关系'这类"
                "多跳推理问题，单次检索无法覆盖完整的推理链。"
            ),
            "detection_signal": "答案质量对检索数量 k 极度敏感",
            "impact_on_hallucination": (
                "LLM 拥有部分信息后，容易在缺失的推理环节上"
                "'自行补全'，产生不易察觉的逻辑幻觉。"
            ),
        },
    }

    for mode_name, details in failure_modes.items():
        print(f"\n{'─' * 50}")
        print(f"失败模式：{mode_name}")
        print(f"描述：{details['description']}")
        print(f"机制：{details['root_mechanics']}")
        print(f"检测信号：{details['detection_signal']}")
        print(f"对幻觉的影响：{details['impact_on_hallucination']}")

    return failure_modes

failure_modes = analyze_retrieval_failure_modes()
```

### 2.2 LLM 层根因：上下文冲突与知识过时

```python
def analyze_llm_context_conflict():
    """
    分析 LLM 在处理检索上下文时的冲突决策过程。
    """
    conflict_scenarios = [
        {
            "scenario": "参数化知识 vs 检索上下文冲突",
            "example": (
                "LLM 在预训练时学到的信息（如'诉讼时效为两年'—民法通则时期）"
                "与检索到的文档（'诉讼时效为三年'—民法典）发生冲突。"
                "LLM 在内部存在两种冲突的知识。"
            ),
            "why_happens": (
                "LLM 的参数化知识是在训练截止日期前固定的。"
                "当检索到的文档与参数化知识不一致时，LLM 需要决定信任哪一个。"
                "Naive RAG 的 prompt 设计没有明确指示 LLM 优先信任检索文档，"
                "导致 LLM 可能选择自己'更确信'的参数化知识。"
            ),
            "fix_direction": (
                "在 prompt 中明确优先级：'如果参考文档与你的知识冲突，"
                "以参考文档为准。参考文档是权威来源。'"
            ),
        },
        {
            "scenario": "上下文窗口内的信息冲突",
            "example": (
                "检索返回的多份文档之间存在矛盾（例如新旧法律条文并存），"
                "LLM 需要在这些矛盾信息中做出选择或调和。"
            ),
            "why_happens": (
                "检索系统没有对文档进行'信息一致性'过滤。"
                "知识库中同时存在过时文档和最新文档。"
                "LLM 在面对上下文内冲突时，可能选择错误的一方，"
                "或者尝试'调和'导致产生不存在的折中事实。"
            ),
            "fix_direction": (
                "添加文档时效性权重，在检索结果中标注文档时间，"
                "prompt 中要求优先采纳最新来源。"
            ),
        },
        {
            "scenario": "上下文注意力稀释（Attention Dilution）",
            "example": (
                "检索返回了 10 个文档块，总长度超过 4000 tokens。"
                "关键信息藏在第 7 个文档的末尾。"
                "LLM 的注意力在长上下文中被稀释，忽略了关键细节。"
            ),
            "why_happens": (
                "Transformer 的注意力机制在处理长序列时，"
                "对中间和末尾信息的关注度分布不均匀。"
                "研究表明，LLM 倾向于更关注上下文开头和结尾的信息，"
                "中间的信息容易被'遗忘'（Lost in the Middle 现象）。"
            ),
            "fix_direction": (
                "减少检索文档数量（3-5 个），或者使用 re-ranking "
                "确保最重要的文档排在最前面。"
            ),
        },
    ]

    for s in conflict_scenarios:
        print(f"\n{'─' * 50}")
        print(f"场景：{s['scenario']}")
        print(f"实例：{s['example']}")
        print(f"原因：{s['why_happens']}")
        print(f"修复方向：{s['fix_direction']}")

    return conflict_scenarios

conflict_analysis = analyze_llm_context_conflict()
```

---

## 3. 真实场景（Real-world Scenario）

### 场景：法律咨询 RAG 系统

某律师事务所开发了一个内部 RAG 系统，用于快速查询法规条文。知识库包含全部中国法律法规。

**时间线：**

```
08:45  律师查询："新公司法下，有限责任公司的最低注册资本是多少？"
08:46  RAG 检索到 5 篇文档，其中 3 篇是关于 2018 年旧公司法的
        （最低注册资本 3 万元），2 篇是关于 2024 年新公司法的
        （取消了最低注册资本限制）
08:47  LLM 回答："有限责任公司的最低注册资本为人民币三万元。"
        —— 幻觉！因为 LLM 在 5 篇混合文档中，参数化知识偏向旧法，
        且 prompt 中没有指示优先信任新文档
08:50  律师基于此回答起草了法律意见书
09:30  合伙人审阅时发现问题，几乎造成严重后果
```

### 检测代码：比对答案与源文档

```python
import re
from typing import List, Tuple

class HallucinationDetector:
    """
    检测 LLM 输出是否基于检索文档的工具类。

    核心思路：
    1. N-gram 覆盖率：答案中的 n-gram 有多少能在源文档中找到
    2. 实体一致性：答案中提到的实体（数字、日期、人名）是否与文档一致
    3. 矛盾检测：答案是否包含与源文档直接矛盾的陈述
    """

    def __init__(self, ngram_range: Tuple[int, int] = (3, 5)):
        self.ngram_range = ngram_range

    def extract_ngrams(self, text: str, n: int) -> set:
        """提取 n-gram，用于计算覆盖率。"""
        # 中文按字符提取 n-gram
        cleaned = re.sub(r'[^一-鿿\w]', '', text)
        if len(cleaned) < n:
            return set()
        return set(cleaned[i:i+n] for i in range(len(cleaned) - n + 1))

    def compute_grounding_score(
        self,
        answer: str,
        source_documents: List[str]
    ) -> Dict:
        """
        计算答案的'接地分数'——答案有多大比例能在源文档中找到依据。

        返回：
        - overall_score: 0-1 之间的分数，1 表示完全基于源文档
        - ngram_details: 各 n-gram 级别的详细分数
        - uncovered_claims: 无法在源文档中验证的断言
        """
        combined_source = " ".join(source_documents)
        ngram_details = {}

        total_weighted_score = 0.0
        total_weight = 0

        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            answer_ngrams = self.extract_ngrams(answer, n)
            source_ngrams = self.extract_ngrams(combined_source, n)

            if len(answer_ngrams) == 0:
                ngram_details[f"{n}-gram"] = {"coverage": 1.0, "total": 0}
                continue

            matched = answer_ngrams & source_ngrams
            coverage = len(matched) / len(answer_ngrams) if answer_ngrams else 1.0

            weight = n  # Longer n-grams are more meaningful
            total_weighted_score += coverage * weight
            total_weight += weight

            ngram_details[f"{n}-gram"] = {
                "coverage": round(coverage, 4),
                "matched": len(matched),
                "total": len(answer_ngrams),
            }

        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        # 提取无法验证的数字断言
        uncovered_claims = self.detect_uncovered_numeric_claims(
            answer, combined_source
        )

        return {
            "overall_score": round(overall_score, 4),
            "ngram_details": ngram_details,
            "uncovered_claims": uncovered_claims,
            "verdict": (
                "grounded" if overall_score > 0.7
                else "partially_grounded" if overall_score > 0.4
                else "hallucinated"
            ),
        }

    def detect_uncovered_numeric_claims(
        self,
        answer: str,
        source: str
    ) -> List[str]:
        """
        检测答案中在源文档里找不到的数字断言。
        数字是最容易产生幻觉的事实类型。
        """
        # 匹配中文文本中的数字模式
        number_patterns = [
            r'\d+\.?\d*',                     # 阿拉伯数字
            r'[零一二三四五六七八九十百千万亿]+',  # 中文数字
        ]

        uncovered = []
        for pattern in number_patterns:
            for match in re.finditer(pattern, answer):
                num_text = match.group()
                # 获取数字周围的上下文
                start = max(0, match.start() - 10)
                end = min(len(answer), match.end() + 10)
                context = answer[start:end]

                if num_text not in source:
                    uncovered.append({
                        "claim": context.strip(),
                        "number": num_text,
                        "position": match.start(),
                    })

        return uncovered

    def detect_contradiction(
        self,
        answer: str,
        source_documents: List[str]
    ) -> List[Dict]:
        """
        检测答案是否包含与源文档直接矛盾的陈述。
        使用否定词 + 数字的组合模式检测。
        """
        contradictions = []

        # 中文否定模式
        negation_patterns = [
            r'(?:不是|并非|没有|不应当|不可以|禁止|不得)\s*.{0,20}',
        ]

        combined_source = " ".join(source_documents)

        for pattern in negation_patterns:
            for match in re.finditer(pattern, combined_source):
                negated_clause = match.group()
                # 检查答案中是否有对同一主题的肯定陈述
                # 这是一个启发式方法
                key_terms = re.findall(r'[一-鿿]{2,}', negated_clause)
                for term in key_terms:
                    if term in answer and len(term) >= 3:
                        contradictions.append({
                            "source_clause": negated_clause,
                            "conflicting_term": term,
                            "severity": "medium",
                        })

        return contradictions


# ============================================================
# 使用演示
# ============================================================

def demo_hallucination_detection():
    """演示幻觉检测的完整工作流程。"""

    detector = HallucinationDetector(ngram_range=(2, 5))

    # 场景：正确答案（基于文档）和幻觉答案的对比
    source_docs = [
        "《中华人民共和国民法典》第一百八十八条：向人民法院请求保护民事权利的诉讼时效期间为三年。"
        "法律另有规定的，依照其规定。诉讼时效期间自权利人知道或者应当知道权利受到损害以及义务人之日起计算。",
        "但是，自权利受到损害之日起超过二十年的，人民法院不予保护。"
        "有特殊情况的，人民法院可以根据权利人的申请决定延长。",
    ]

    # 基于文档的正确答案
    correct_answer = (
        "中华人民共和国民法典规定的普通诉讼时效期间为三年。"
        "该期间自权利人知道或者应当知道权利受到损害以及义务人之日起计算。"
        "但是，自权利受到损害之日起超过二十年的，人民法院不予保护。"
    )

    # 幻觉答案（使用了旧的民法通则数据）
    hallucinated_answer = (
        "根据中国法律规定，普通诉讼时效为两年。"
        "特殊情况下可以延长至五年。"
    )

    print("=" * 60)
    print("幻觉检测演示")
    print("=" * 60)

    # 检测正确答案
    result_correct = detector.compute_grounding_score(correct_answer, source_docs)
    print("\n【正确答案的接地检测】")
    print(f"综合接地分数: {result_correct['overall_score']}")
    print(f"判定: {result_correct['verdict']}")
    print(f"N-gram 详情: {result_correct['ngram_details']}")

    # 检测幻觉答案
    result_hallucinated = detector.compute_grounding_score(
        hallucinated_answer, source_docs
    )
    print("\n【幻觉答案的接地检测】")
    print(f"综合接地分数: {result_hallucinated['overall_score']}")
    print(f"判定: {result_hallucinated['verdict']}")
    print(f"N-gram 详情: {result_hallucinated['ngram_details']}")
    print(f"无法验证的断言: {result_hallucinated['uncovered_claims']}")

    # 矛盾检测
    contradictions = detector.detect_contradiction(
        hallucinated_answer, source_docs
    )
    print(f"\n矛盾检测结果: {contradictions}")

    return {
        "correct": result_correct,
        "hallucinated": result_hallucinated,
        "contradictions": contradictions,
    }

detection_results = demo_hallucination_detection()
```

---

## 4. 修复方案（Fix with Code）

### 4.1 策略一：强化 Prompt 设计 —— 指示 LLM 信任检索文档

```python
from typing import List, Optional

class GroundedPromptBuilder:
    """
    构建能有效防止幻觉的 RAG Prompt。

    核心原则：
    1. 明确指定检索文档的权威性
    2. 要求 LLM 逐条引用文档内容
    3. 对不确定的情况要求 LLM 明确表示
    4. 结构化输出，便于后续验证
    """

    @staticmethod
    def build_grounded_prompt(
        query: str,
        documents: List[Document],
        strict_mode: bool = True,
        language: str = "zh"
    ) -> str:
        """
        构建防幻觉的 RAG prompt。

        参数：
        - strict_mode: True 时强制 LLM 逐句标注来源
        """
        if language == "zh":
            return GroundedPromptBuilder._build_chinese_prompt(
                query, documents, strict_mode
            )
        else:
            return GroundedPromptBuilder._build_english_prompt(
                query, documents, strict_mode
            )

    @staticmethod
    def _build_chinese_prompt(
        query: str,
        documents: List[Document],
        strict_mode: bool
    ) -> str:
        """构建中文防幻觉 prompt。"""

        # 格式化文档，带编号
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            doc_texts.append(
                f"【文档 {i}】（来源：{doc.source}，检索相关度：{doc.retrieval_score:.2f}）\n"
                f"{doc.content}"
            )
        formatted_docs = "\n\n---\n\n".join(doc_texts)

        if strict_mode:
            prompt = f"""你是一个严格基于参考文档回答问题的助手。请遵守以下规则：

## 核心规则（必须遵守）
1. **仅基于参考文档回答**：你的每一个论断都必须能在参考文档中找到原文依据。
2. **逐句标注来源**：回答中的每个事实陈述后面，用方括号标注来源文档编号，例如 [文档1]。
3. **区分事实与推理**：如果你的回答包含基于文档的推理，请在推理前加 [推理] 标记。
4. **不确定就说不知道**：如果参考文档中没有足够信息回答问题，请明确说"根据提供的参考文档，无法确定……"，不要猜测。
5. **尊重文档优先级**：如果参考文档之间存在矛盾，以最新发布的文档为准。
6. **不要依赖内部知识**：即使你预训练时学到的知识与参考文档不同，也必须以参考文档为准。

## 参考文档
{formatted_docs}

## 用户问题
{query}

## 请按以下格式回答：
**回答**：[你的回答，逐句标注来源]
**置信度**：[高/中/低]，基于参考文档覆盖问题的程度
**引用文档**：[列出实际使用了哪些文档]"""
        else:
            prompt = f"""请基于以下参考文档回答用户问题。请尽量引用文档中的具体内容。

参考文档：
{formatted_docs}

用户问题：{query}

请回答（请标注信息来源）："""

        return prompt

    @staticmethod
    def _build_english_prompt(
        query: str,
        documents: List[Document],
        strict_mode: bool
    ) -> str:
        """Build English anti-hallucination prompt."""
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            doc_texts.append(f"[Document {i}] (Source: {doc.source})\n{doc.content}")
        formatted_docs = "\n\n---\n\n".join(doc_texts)

        return f"""You are a retrieval-grounded assistant. Follow these rules:

1. Base your answer ONLY on the provided reference documents.
2. Cite the source document number for each factual claim, e.g., [Doc 1].
3. If the documents lack sufficient information, explicitly state so.
4. If the documents conflict, prefer the most recent one.
5. Do NOT rely on your pre-training knowledge if it contradicts the documents.

## Reference Documents
{formatted_docs}

## Question
{query}

## Answer (with citations):"""


# 演示改进后的 prompt
builder = GroundedPromptBuilder()
improved_prompt = builder.build_grounded_prompt(
    query=query,
    documents=retrieved_docs,
    strict_mode=True,
)

print("=" * 60)
print("改进后的防幻觉 Prompt")
print("=" * 60)
print(improved_prompt)
```

### 4.2 策略二：相关性阈值过滤 —— 不传无意义文档

```python
import numpy as np
from typing import List, Tuple

class RelevanceFilter:
    """
    在将检索结果送入 LLM 之前，过滤掉低相关度文档。

    这解决了'覆盖率不足'导致的幻觉：不传无关文档，
    LLM 就不会被误导去'脑补'无关信息。
    """

    def __init__(
        self,
        similarity_threshold: float = 0.70,
        max_documents: int = 5,
        diversity_penalty: float = 0.0,
    ):
        """
        参数：
        - similarity_threshold: 相似度阈值，低于此值的文档被丢弃
        - max_documents: 最多保留几篇文档
        - diversity_penalty: 去重惩罚（0-1），越高则排斥内容重复的文档
        """
        self.similarity_threshold = similarity_threshold
        self.max_documents = max_documents
        self.diversity_penalty = diversity_penalty

    def filter(
        self,
        documents: List[Document],
        return_rejected: bool = False
    ) -> List[Document]:
        """
        过滤检索结果。

        返回：
        - 如果 return_rejected=False：仅返回通过过滤的文档
        - 如果 return_rejected=True：返回 (accepted, rejected) 元组
        """
        accepted = []
        rejected = []

        # Step 1: 相似度阈值过滤
        for doc in documents:
            if doc.retrieval_score >= self.similarity_threshold:
                accepted.append(doc)
            else:
                rejected.append(doc)

        # Step 2: 数量限制 + 去重
        if self.diversity_penalty > 0 and len(accepted) > 1:
            accepted = self._apply_diversity_filter(accepted)

        # Step 3: 截断
        accepted = accepted[:self.max_documents]

        if return_rejected:
            return accepted, rejected
        return accepted

    def _apply_diversity_filter(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        基于内容多样性的去重过滤。

        使用 Jaccard 相似度检测内容高度重叠的文档，
        保留分数更高的那个。
        """
        if len(documents) <= 1:
            return documents

        def char_bigrams(text: str) -> set:
            """提取字符级 bigram 用于 Jaccard 计算。"""
            cleaned = text.replace(" ", "").replace("\n", "")
            return set(cleaned[i:i+2] for i in range(len(cleaned) - 1))

        filtered = [documents[0]]  # 保留最高分的
        for doc in documents[1:]:
            doc_bigrams = char_bigrams(doc.content)
            is_duplicate = False

            for existing in filtered:
                existing_bigrams = char_bigrams(existing.content)
                if not existing_bigrams:
                    continue
                intersection = len(doc_bigrams & existing_bigrams)
                union = len(doc_bigrams | existing_bigrams)
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard > (1.0 - self.diversity_penalty):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(doc)

        return filtered

    def should_decline_to_answer(
        self,
        accepted_docs: List[Document]
    ) -> Tuple[bool, str]:
        """
        判断是否应该拒绝回答。

        当所有通过过滤的文档的检索分数仍然很低时，
        说明知识库可能不包含相关信息，应该拒绝回答
        而不是让 LLM 猜测。
        """
        if not accepted_docs:
            return True, "所有检索文档均未达到相关性阈值，知识库可能不包含相关信息。"

        avg_score = np.mean([d.retrieval_score for d in accepted_docs])
        if avg_score < self.similarity_threshold:
            return True, (
                f"文档平均相关度 ({avg_score:.2f}) 低于阈值 "
                f"({self.similarity_threshold})，回答可能不可靠。"
            )

        return False, ""


# 使用演示
relevance_filter = RelevanceFilter(
    similarity_threshold=0.70,
    max_documents=3,
    diversity_penalty=0.3,
)

# 模拟一批检索结果（混入低相关文档）
mixed_results = [
    Document("d1", "民法典第一百八十八条：诉讼时效为三年。", "民法典", 0.92),
    Document("d2", "民法典第一百八十九条：分期债务诉讼时效。", "民法典", 0.87),
    Document("d3", "今天天气很好，适合出门散步。", "无关文档", 0.35),  # 低相关
    Document("d4", "明天可能有小雨转多云。", "无关文档", 0.28),  # 低相关
    Document("d5", "民法典第一百九十条：无民事行为能力人。", "民法典", 0.81),
]

accepted, rejected = relevance_filter.filter(mixed_results, return_rejected=True)

print("\n" + "=" * 60)
print("相关性过滤结果")
print("=" * 60)
print(f"\n接受的文档 ({len(accepted)} 篇):")
for doc in accepted:
    print(f"  [{doc.doc_id}] score={doc.retrieval_score:.2f} - {doc.content[:50]}...")

print(f"\n拒绝的文档 ({len(rejected)} 篇):")
for doc in rejected:
    print(f"  [{doc.doc_id}] score={doc.retrieval_score:.2f} - {doc.content[:50]}...")

should_decline, reason = relevance_filter.should_decline_to_answer(accepted)
print(f"\n是否应拒绝回答: {should_decline}")
if should_decline:
    print(f"原因: {reason}")
```

### 4.3 策略三：事实一致性校验 —— 后处理验证

```python
import hashlib
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


class FactualConsistencyValidator:
    """
    后处理阶段对 LLM 输出进行事实一致性校验。

    使用多层检测策略：
    1. 精确匹配：答案中的短句是否在源文档中出现
    2. 语义蕴涵：使用 NLI 模型判断文档是否支持答案
    3. 矛盾检测：答案中的断言是否与文档直接矛盾
    """

    def __init__(self, use_nli: bool = False):
        self.use_nli = use_nli

    def validate(
        self,
        answer: str,
        source_documents: List[str]
    ) -> Dict[str, Any]:
        """
        对答案进行完整的事实一致性校验。
        """
        combined_source = "\n".join(source_documents)

        # Layer 1: 精确匹配检查
        exact_match_report = self._exact_match_check(answer, source_documents)

        # Layer 2: 数字事实检查
        numeric_report = self._numeric_fact_check(answer, combined_source)

        # Layer 3: 句子级蕴涵检查
        sentence_report = self._sentence_entailment_check(answer, source_documents)

        # 汇总
        all_issues = (
            exact_match_report["issues"]
            + numeric_report["issues"]
            + sentence_report["issues"]
        )

        overall_consistency = (
            "consistent" if len(all_issues) == 0
            else "minor_issues" if len(all_issues) <= 2
            else "major_issues"
        )

        return {
            "overall_consistency": overall_consistency,
            "total_issues": len(all_issues),
            "exact_match_report": exact_match_report,
            "numeric_report": numeric_report,
            "sentence_report": sentence_report,
            "issues": all_issues,
        }

    def _exact_match_check(
        self,
        answer: str,
        source_documents: List[str]
    ) -> Dict:
        """
        检查答案中的关键短句（8-20 字）是否能在源文档中找到精确匹配。
        """
        issues = []

        # 将答案按标点分割为短句
        sentences = re.split(r'[，。；：、！？\n]', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 8]

        combined_source = " ".join(source_documents)

        matched_count = 0
        unmatched_sentences = []

        for sentence in sentences:
            # 使用滑动窗口在源文档中查找
            if sentence in combined_source:
                matched_count += 1
            else:
                # 对于较长的句子，尝试找到部分匹配
                partial_match = any(
                    len(sentence) >= 8 and sentence[:8] in combined_source
                    for _ in [1]  # ensure single evaluation
                )
                # 更好的部分匹配：检查是否有 70% 的 2-gram 匹配
                sent_bigrams = set(
                    sentence[i:i+2] for i in range(len(sentence) - 1)
                )
                source_bigrams = set(
                    combined_source[j:j+2]
                    for j in range(len(combined_source) - 1)
                )
                if sent_bigrams:
                    overlap_ratio = (
                        len(sent_bigrams & source_bigrams) / len(sent_bigrams)
                    )
                else:
                    overlap_ratio = 0.0

                if overlap_ratio < 0.5:
                    unmatched_sentences.append({
                        "sentence": sentence,
                        "bigram_overlap": round(overlap_ratio, 2),
                        "severity": "high" if overlap_ratio < 0.3 else "medium",
                    })

        issues.extend(unmatched_sentences)

        return {
            "total_sentences": len(sentences),
            "matched_sentences": matched_count,
            "match_ratio": matched_count / len(sentences) if sentences else 1.0,
            "issues": unmatched_sentences,
        }

    def _numeric_fact_check(
        self,
        answer: str,
        combined_source: str
    ) -> Dict:
        """
        专门检查数字事实。
        数字是 RAG 幻觉中最常见也最危险的错误类型。
        """
        issues = []

        # 提取答案中的所有数字及其上下文
        number_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(年|月|日|天|元|万元|亿|%|％|条|款|项|岁|人|次|倍|个|件)'
        )
        answer_numbers = number_pattern.findall(answer)

        for num_value, unit in answer_numbers:
            full_pattern = f"{num_value}{unit}"
            if full_pattern not in combined_source:
                issues.append({
                    "type": "numeric_mismatch",
                    "value": full_pattern,
                    "severity": "high",
                    "message": (
                        f"答案中的数字 '{full_pattern}' 在源文档中未找到。"
                        f"这可能是幻觉。"
                    ),
                })

        return {
            "total_numbers_checked": len(answer_numbers),
            "mismatched_numbers": len(issues),
            "issues": issues,
        }

    def _sentence_entailment_check(
        self,
        answer: str,
        source_documents: List[str]
    ) -> Dict:
        """
        句子级蕴涵检查。

        将答案拆分为独立断言，对每个断言在源文档中进行语义搜索。
        这里使用简化的关键词重叠方法作为 NLI 的轻量级替代。

        注意：完整的 NLI 实现需要加载专门的模型（如 BERT NLI），
        这里展示的是可在无 GPU 环境下运行的启发式方法。
        """
        issues = []

        # 拆分答案为独立断言
        assertions = re.split(r'[。；\n]', answer)
        assertions = [a.strip() for a in assertions if len(a.strip()) >= 15]

        combined_source = " ".join(source_documents)

        for assertion in assertions:
            # 提取关键词（2-4 字的中文词）
            keywords = re.findall(r'[一-鿿]{2,4}', assertion)
            if not keywords:
                continue

            # 检查每个关键词是否在源文档中出现
            keyword_hits = sum(1 for kw in keywords if kw in combined_source)
            keyword_ratio = keyword_hits / len(keywords) if keywords else 0.0

            if keyword_ratio < 0.4:
                issues.append({
                    "type": "unsupported_assertion",
                    "assertion": assertion[:100],
                    "keyword_support_ratio": round(keyword_ratio, 2),
                    "severity": "high" if keyword_ratio < 0.2 else "medium",
                })

        return {
            "total_assertions": len(assertions),
            "unsupported_assertions": len(issues),
            "issues": issues,
        }


# 完整修复方案集成
class RobustRAGPipeline:
    """
    整合了所有防幻觉策略的完整 RAG Pipeline。
    """

    def __init__(
        self,
        similarity_threshold: float = 0.70,
        max_documents: int = 5,
        strict_prompt: bool = True,
        enable_validation: bool = True,
    ):
        self.relevance_filter = RelevanceFilter(
            similarity_threshold=similarity_threshold,
            max_documents=max_documents,
            diversity_penalty=0.3,
        )
        self.prompt_builder = GroundedPromptBuilder()
        self.validator = FactualConsistencyValidator() if enable_validation else None
        self.strict_prompt = strict_prompt

    def run(
        self,
        query: str,
        retrieved_documents: List[Document],
        llm_call_fn=None,  # 在实际使用中注入 LLM 调用函数
    ) -> Dict[str, Any]:
        """
        执行完整的防幻觉 RAG 流程。

        流程：
        1. 相关性过滤 → 2. 构建防幻觉 Prompt → 3. LLM 生成
        → 4. 事实一致性验证 → 5. 如果不通过，重试或告警
        """
        result = {
            "query": query,
            "pipeline_steps": {},
        }

        # Step 1: 相关性过滤
        accepted_docs, rejected_docs = self.relevance_filter.filter(
            retrieved_documents, return_rejected=True
        )
        result["pipeline_steps"]["filtering"] = {
            "accepted_count": len(accepted_docs),
            "rejected_count": len(rejected_docs),
            "rejected_ids": [d.doc_id for d in rejected_docs],
        }

        # 判断是否应该拒答
        should_decline, decline_reason = (
            self.relevance_filter.should_decline_to_answer(accepted_docs)
        )
        if should_decline:
            result["answer"] = (
                f"抱歉，根据现有资料无法准确回答此问题。{decline_reason}"
            )
            result["declined"] = True
            return result

        # Step 2: 构建防幻觉 Prompt
        prompt = self.prompt_builder.build_grounded_prompt(
            query=query,
            documents=accepted_docs,
            strict_mode=self.strict_prompt,
        )
        result["pipeline_steps"]["prompt"] = {
            "prompt_length": len(prompt),
            "documents_used": [d.doc_id for d in accepted_docs],
        }

        # Step 3: LLM 生成（示例中使用模拟）
        # 在实际代码中，这里调用真实的 LLM API
        if llm_call_fn:
            raw_answer = llm_call_fn(prompt)
        else:
            raw_answer = (
                "【文档1】根据《中华人民共和国民法典》第一百八十八条，"
                "普通诉讼时效期间为三年。[文档1]"
            )

        result["pipeline_steps"]["generation"] = {
            "raw_answer": raw_answer,
        }

        # Step 4: 事实一致性验证
        if self.validator:
            source_texts = [d.content for d in accepted_docs]
            validation = self.validator.validate(raw_answer, source_texts)
            result["pipeline_steps"]["validation"] = validation

            # Step 5: 根据验证结果决定是否需要重试
            if validation["overall_consistency"] == "major_issues":
                result["warning"] = (
                    "检测到严重的事实不一致问题，建议人工审核。"
                    f"问题数量: {validation['total_issues']}"
                )

        result["answer"] = raw_answer
        result["declined"] = False
        return result


# 演示完整 Pipeline
print("\n" + "=" * 60)
print("完整防幻觉 RAG Pipeline 演示")
print("=" * 60)

pipeline = RobustRAGPipeline(
    similarity_threshold=0.70,
    max_documents=3,
    strict_prompt=True,
    enable_validation=True,
)

pipeline_result = pipeline.run(
    query="中华人民共和国民法典规定的普通诉讼时效是几年？",
    retrieved_documents=mixed_results,
)

print(f"\nPipeline 结果:")
print(f"  是否拒答: {pipeline_result['declined']}")
print(f"  过滤阶段: {pipeline_result['pipeline_steps']['filtering']}")
print(f"  答案: {pipeline_result['answer']}")
if 'warning' in pipeline_result:
    print(f"  警告: {pipeline_result['warning']}")
```

---

## 5. 检查清单（Checklist）

在部署 RAG 系统之前，逐项检查以下内容：

### 检索层

- [ ] 是否有检索相关性阈值？低于阈值的文档不会被送入 LLM？
- [ ] 是否对 top-k 的 k 值做了调优？k 太小可能遗漏信息，太大可能引入噪声？
- [ ] 是否有文档时效性排序机制？（新文档优先于旧文档）
- [ ] 是否测试了"知识库不包含相关信息"时的拒答行为？
- [ ] 是否使用了 Re-ranking 来提升检索精度？

### Prompt 层

- [ ] Prompt 中是否明确要求 LLM 仅基于参考文档回答？
- [ ] 是否有"如果参考文档与内部知识冲突，以参考文档为准"的指令？
- [ ] 是否要求 LLM 逐句标注来源文档编号？
- [ ] 是否要求 LLM 在不确定时明确表示"不知道"？
- [ ] 是否对长上下文做了截断或重要性排序？

### 后处理层

- [ ] 是否有答案与源文档的事实一致性校验？
- [ ] 是否检查了答案中的数字与源文档的一致性？
- [ ] 对于高风险场景（法律、医疗），是否有二次人工审核流程？
- [ ] 是否记录了每次查询的检索文档和生成答案的完整链路？
- [ ] 是否有基于用户反馈的幻觉检测机制？

### 监控层

- [ ] 是否监控了"拒答率"指标？拒答率突然升高可能表示检索质量下降？
- [ ] 是否监控了"接地分数"（grounding score）的分布？
- [ ] 是否对高频查询建立了专用的评估集？
- [ ] 是否定期（每周/每月）进行人工抽检？

---

> **核心教训**：幻觉不是 LLM 的 bug，而是 Naive RAG 设计的必然结果。
> 对抗幻觉需要在检索、Prompt、后处理三个层面同时设防。
> 永远不要假设"检索到了文档，LLM 就一定基于它回答"。
