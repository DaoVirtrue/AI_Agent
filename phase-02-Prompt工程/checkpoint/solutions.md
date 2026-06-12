# Phase 02 Checkpoint：Prompt 工程 - 参考答案

本文档提供 8 个练习的完整 Python 实现参考。

---

## 练习 1：Zero-shot 分类器

```python
"""
Exercise 1: Zero-shot 分类器
将用户消息分为 refund / shipping / product_info / complaint 四类。
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClassifyResult:
    """分类结果"""
    category: str
    confidence: float
    raw_response: str


def zero_shot_classifier(
    message: str,
    categories: List[str],
    llm_call,
    model: str = "gpt-4o-mini",
) -> ClassifyResult:
    """
    Zero-shot 分类：不提供任何示例，仅通过类别描述让LLM分类。

    Args:
        message: 用户消息
        categories: 类别列表，如 ["refund", "shipping", "product_info", "complaint"]
        llm_call: LLM调用函数，签名为 (messages, model) -> str

    Returns:
        ClassifyResult
    """
    cat_list = "\n".join(f"- {c}" for c in categories)

    system_prompt = f"""你是一个文本分类器。请将用户消息精确分类到以下类别之一：

{cat_list}

分类规则：
- refund: 涉及退款、退货、赔偿的请求
- shipping: 涉及物流、快递、配送、包裹跟踪的问题
- product_info: 涉及商品规格、价格、库存、使用方法的咨询
- complaint: 涉及投诉、不满、差评表达

你必须严格输出以下JSON格式：
{{"category": "<类别名>", "confidence": <0.0-1.0的置信度>}}

不要输出任何其他文字。只输出JSON。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    raw_response = llm_call(messages, model=model).strip()

    # 解析响应（使用鲁棒JSON解析）
    import json
    try:
        data = json.loads(raw_response)
        category = data.get("category", "unknown")
        confidence = float(data.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError, KeyError):
        # 降级：直接查找类别关键词
        raw_lower = raw_response.lower()
        for cat in categories:
            if cat.lower() in raw_lower:
                return ClassifyResult(
                    category=cat,
                    confidence=0.3,
                    raw_response=raw_response,
                )
        return ClassifyResult(
            category=categories[0],
            confidence=0.1,
            raw_response=raw_response,
        )

    # 校验类别名
    if category not in categories:
        for cat in categories:
            if cat.lower() in category.lower():
                category = cat
                break
        else:
            category = categories[0]
            confidence = 0.2

    return ClassifyResult(
        category=category,
        confidence=min(max(confidence, 0.0), 1.0),
        raw_response=raw_response,
    )


# ===== 模拟LLM调用 =====
def mock_llm_call(messages, model="gpt-4o-mini") -> str:
    """模拟LLM（实际使用时替换为真实API调用）"""
    user_msg = messages[-1]["content"]
    if "包裹" in user_msg and "没到" in user_msg:
        return '{"category": "shipping", "confidence": 0.95}'
    elif "退款" in user_msg:
        return '{"category": "refund", "confidence": 0.92}'
    elif "怎么用" in user_msg or "规格" in user_msg:
        return '{"category": "product_info", "confidence": 0.88}'
    else:
        return '{"category": "complaint", "confidence": 0.85}'


# ===== 测试 =====
if __name__ == "__main__":
    categories = ["refund", "shipping", "product_info", "complaint"]

    msg1 = "我的包裹已经一个星期了还没到，能帮我查一下吗？"
    result1 = zero_shot_classifier(msg1, categories, mock_llm_call)
    print(f"Test 1: {result1}")

    msg2 = "这个商品质量太差了，用了两天就坏了，我要退款！"
    result2 = zero_shot_classifier(msg2, categories, mock_llm_call)
    print(f"Test 2: {result2}")

    msg3 = "你们这款手机支持5G吗？电池容量多大？"
    result3 = zero_shot_classifier(msg3, categories, mock_llm_call)
    print(f"Test 3: {result3}")
```

---

## 练习 2：Few-shot 命名实体识别（NER）

```python
"""
Exercise 2: Few-shot 命名实体识别（NER）
从技术文档中提取软件名称、版本号和公司名称。
"""

import json
import re
from typing import List, Dict, Any


# ===== Few-shot 示例 =====
FEW_SHOT_EXAMPLES = [
    {
        "input": "Docker 24.0.5 由 Docker Inc. 发布，支持 Kubernetes 集成。",
        "output": {
            "entities": [
                {"text": "Docker", "type": "SOFTWARE", "start": 0, "end": 6},
                {"text": "24.0.5", "type": "VERSION", "start": 7, "end": 13},
                {"text": "Docker Inc.", "type": "COMPANY", "start": 16, "end": 27},
                {"text": "Kubernetes", "type": "SOFTWARE", "start": 36, "end": 46},
            ]
        }
    },
    {
        "input": "Python 3.12 新增了更快的错误提示功能。",
        "output": {
            "entities": [
                {"text": "Python", "type": "SOFTWARE", "start": 0, "end": 6},
                {"text": "3.12", "type": "VERSION", "start": 7, "end": 11},
            ]
        }
    },
    {
        "input": "今天的天气不错，适合出去散步。",
        "output": {
            "entities": []
        }
    },
]


def build_few_shot_prompt(examples: List[Dict]) -> str:
    """构建Few-shot提示的示例部分"""
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Input: {ex['input']}")
        parts.append(f"Output: {json.dumps(ex['output'], ensure_ascii=False)}")
        parts.append("")
    return "\n".join(parts)


def few_shot_ner_extractor(
    text: str,
    llm_call,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    使用Few-shot示例引导LLM进行NER提取。

    Args:
        text: 待提取的文本
        llm_call: LLM调用函数
        model: 模型名

    Returns:
        {"entities": [...]}
    """
    few_shot_text = build_few_shot_prompt(FEW_SHOT_EXAMPLES)

    system_prompt = f"""你是一个命名实体识别（NER）系统。从技术文档中提取以下类型的实体：
- SOFTWARE: 软件名称
- VERSION: 版本号
- COMPANY: 公司名称

以下是几个示例：

{few_shot_text}

你必须严格输出JSON格式，不要添加任何解释或markdown标记。
每个实体包含 text, type, start, end 字段。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请从以下文本中提取实体：\n\n{text}"},
    ]

    raw_response = llm_call(messages, model=model).strip()

    # 使用鲁棒解析（简化版）
    return parse_ner_output(raw_response, text)


def parse_ner_output(raw: str, original_text: str) -> Dict[str, Any]:
    """鲁棒解析NER输出"""
    # Level 1: 直接json.loads
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Level 2: 提取markdown代码块
    md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Level 3: 提取大括号块
    brace_match = re.search(r'\{.*"entities".*\}', raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # 全部失败
    return {"entities": [], "_error": "Failed to parse", "_raw": raw}


# ===== 模拟LLM调用 =====
def mock_llm_call_ner(messages, model="gpt-4o-mini") -> str:
    """模拟NER的LLM调用"""
    text = messages[-1]["content"]
    result = {"entities": []}

    # 简单的规则匹配（实际使用LLM）
    sw_patterns = [
        ("Apache Hadoop", "SOFTWARE"),
        ("HDFS", "SOFTWARE"),
        ("YARN", "SOFTWARE"),
        ("Apache Spark", "SOFTWARE"),
    ]
    for name, etype in sw_patterns:
        idx = text.find(name)
        if idx >= 0:
            result["entities"].append({
                "text": name, "type": etype,
                "start": idx, "end": idx + len(name),
            })

    ver_match = re.search(r'\d+\.\d+(?:\.\d+)?', text)
    if ver_match:
        result["entities"].append({
            "text": ver_match.group(0), "type": "VERSION",
            "start": ver_match.start(), "end": ver_match.end(),
        })

    company_match = re.search(r'(Apache Software Foundation|Microsoft|Google|Docker Inc\.)', text)
    if company_match:
        result["entities"].append({
            "text": company_match.group(0), "type": "COMPANY",
            "start": company_match.start(), "end": company_match.end(),
        })

    return json.dumps(result, ensure_ascii=False)


# ===== 测试 =====
if __name__ == "__main__":
    text = "Apache Hadoop 3.3.6 由 Apache Software Foundation 发布，包含 HDFS 和 YARN 组件。"
    result = few_shot_ner_extractor(text, mock_llm_call_ner)
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

## 练习 3：Chain-of-Thought 推理

```python
"""
Exercise 3: Chain-of-Thought 推理
使用 [REASONING] 和 [ANSWER] 标记系统实现多步推理。
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class CoTResult:
    """Chain-of-Thought 推理结果"""
    reasoning_steps: List[str]
    final_answer: str
    raw_output: str
    success: bool
    error: str = ""


def cot_reasoner(
    question: str,
    llm_call,
    model: str = "gpt-4o",
    custom_steps: List[str] = None,
) -> CoTResult:
    """
    Chain-of-Thought 推理器。

    Args:
        question: 用户问题
        llm_call: LLM调用函数
        model: 模型名
        custom_steps: 自定义推理步骤（可选）

    Returns:
        CoTResult
    """
    if custom_steps:
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(custom_steps))
        step_guide = f"请按照以下步骤进行推理：\n{steps_text}"
    else:
        step_guide = "请一步一步地进行推理，展示完整的思考过程。"

    system_prompt = f"""你是一个擅长逻辑推理的助手。对于每个问题，你必须：

1. 先进行详细的逐步推理
2. 使用 [REASONING] 标签包裹推理过程
3. 然后使用 [ANSWER] 标签给出最终答案

{step_guide}

输出格式示例：

[REASONING]
第一步：理解问题...
第二步：分析条件...
第三步：得出结论...
[/REASONING]

[ANSWER]
最终答案
[/ANSWER]

请严格遵守此格式，不要省略任何标签。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"问题：{question}"},
    ]

    raw = llm_call(messages, model=model)
    return parse_cot_output(raw)


def parse_cot_output(raw_output: str) -> CoTResult:
    """解析CoT输出，提取推理步骤和答案"""
    raw = raw_output.strip()

    # 提取 [REASONING] 块
    reasoning_steps = []
    reason_match = re.search(
        r'\[REASONING\]\s*\n?(.*?)\n?\s*\[/REASONING\]',
        raw, re.DOTALL | re.IGNORECASE
    )

    if reason_match:
        reasoning_text = reason_match.group(1).strip()
        # 按数字编号分割步骤
        step_lines = re.split(r'(?:\d+[.、．]\s*)', reasoning_text)
        reasoning_steps = [s.strip() for s in step_lines if s.strip()]

        # 如果按编号分割失败，尝试按换行分割
        if len(reasoning_steps) <= 1:
            reasoning_steps = [
                s.strip() for s in reasoning_text.split('\n')
                if s.strip() and len(s.strip()) > 5
            ]
    else:
        # 回退：查找包含"步骤"或"step"的行
        for line in raw.split('\n'):
            if re.search(r'步骤|step|第[一二三四五六七八九十\d]+', line, re.IGNORECASE):
                reasoning_steps.append(line.strip())

    # 提取 [ANSWER] 块
    answer_match = re.search(
        r'\[ANSWER\]\s*\n?(.*?)\n?\s*\[/ANSWER\]',
        raw, re.DOTALL | re.IGNORECASE
    )

    final_answer = ""
    if answer_match:
        final_answer = answer_match.group(1).strip()
    else:
        # 回退：取最后一个段落
        paragraphs = [p.strip() for p in raw.split('\n\n') if p.strip()]
        if paragraphs:
            final_answer = paragraphs[-1]

    success = len(reasoning_steps) > 0 and len(final_answer) > 0
    error = "" if success else "Failed to parse reasoning or answer from output"

    return CoTResult(
        reasoning_steps=reasoning_steps,
        final_answer=final_answer,
        raw_output=raw,
        success=success,
        error=error,
    )


# ===== 模拟LLM调用 =====
def mock_llm_call_cot(messages, model="gpt-4o") -> str:
    """模拟CoT推理的LLM调用"""
    question = messages[-1]["content"]

    if "120件" in question:
        return """[REASONING]
第一步：理解问题。周一卖出120件商品。
第二步：计算周二销售量。周二比周一多卖15%，即120 * (1 + 0.15) = 120 * 1.15 = 138件。
第三步：计算周三销售量。周三比周二少卖10%，即138 * (1 - 0.10) = 138 * 0.90 = 124.2件，取124件。
第四步：计算总销售量。120 + 138 + 124 = 382件。
第五步：验证。周一120件，周二138件，周三约124件，总计382件。计算正确。
[/REASONING]

[ANSWER]
三天总共卖出382件商品。
[/ANSWER]"""

    return """[REASONING]
分析问题后得出结论。
[/REASONING]

[ANSWER]
无法确定。
[/ANSWER]"""


# ===== 测试 =====
if __name__ == "__main__":
    question = "一个商店周一卖出120件商品，周二比周一多卖15%，周三比周二少卖10%。三天总共卖出多少件？"
    result = cot_reasoner(question, mock_llm_call_cot)
    print(f"Success: {result.success}")
    print(f"推理步骤 ({len(result.reasoning_steps)}步):")
    for i, step in enumerate(result.reasoning_steps, 1):
        print(f"  {i}. {step}")
    print(f"最终答案: {result.final_answer}")
```

---

## 练习 4：Token 预算控制器

```python
"""
Exercise 4: Token 预算控制器
管理上下文窗口的token分配，确保不超过限制。
"""

import re
from typing import List, Dict, Any, Optional


class TokenBudget:
    """
    Token预算控制器。

    管理LLM输入中各部分的token分配：
    - System Prompt
    - 对话历史
    - 用户查询
    - 检索文档（动态计算剩余配额）
    """

    def __init__(
        self,
        max_input_tokens: int = 8000,
        system_reserve: int = 1000,
        history_reserve: int = 1500,
        output_reserve: int = 2000,
    ):
        self.max_input_tokens = max_input_tokens
        self.system_reserve = system_reserve
        self.history_reserve = history_reserve
        self.output_reserve = output_reserve

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        粗略token计数。
        中文：1字符约等于1 token
        英文/数字：约4字符为1 token
        """
        if not text:
            return 0
        chinese = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿')
        other = len(text) - chinese
        return chinese + max(1, other // 4)

    def allocate(
        self,
        system_prompt: str,
        query: str,
        history: Optional[List[str]] = None,
        num_docs: int = 0,
    ) -> Dict[str, Any]:
        """
        计算各部分token分配。

        Returns:
            {
                "system_tokens": allocated,
                "history_tokens": allocated,
                "query_tokens": allocated,
                "docs_budget": available_for_docs,
                "is_over_budget": bool,
                "warnings": [str, ...],
            }
        """
        warnings = []

        system_tokens = min(self.count_tokens(system_prompt), self.system_reserve)
        if self.count_tokens(system_prompt) > self.system_reserve:
            warnings.append(
                f"System prompt ({self.count_tokens(system_prompt)}t) "
                f"exceeds reserve ({self.system_reserve}t). Truncated."
            )

        history_text = "\n".join(history or [])
        history_tokens = min(self.count_tokens(history_text), self.history_reserve)
        if self.count_tokens(history_text) > self.history_reserve:
            warnings.append(f"History exceeds reserve. Truncated.")

        query_tokens = self.count_tokens(query)

        used = system_tokens + history_tokens + query_tokens + self.output_reserve
        docs_budget = max(0, self.max_input_tokens - used)
        is_over_budget = used > self.max_input_tokens

        if is_over_budget:
            warnings.append(
                f"Over budget even without docs: used {used}t > max {self.max_input_tokens}t"
            )

        return {
            "system_tokens": system_tokens,
            "history_tokens": history_tokens,
            "query_tokens": query_tokens,
            "docs_budget": docs_budget,
            "output_reserve": self.output_reserve,
            "is_over_budget": is_over_budget,
            "warnings": warnings,
        }

    def truncate_docs_to_budget(
        self, documents: List[str], budget_tokens: int
    ) -> Dict[str, Any]:
        """
        按token预算截断文档列表。

        策略：
        1. 逐文档添加，直到超出预算
        2. 最后一份文档截断填充剩余空间（在句子边界处断开）
        3. 保证至少保留一篇文档

        Returns:
            {
                "kept_docs": [str, ...],
                "stats": {
                    "original_count": int,
                    "kept_count": int,
                    "truncated_chars": int,
                    "tokens_used": int,
                }
            }
        """
        if not documents:
            return {
                "kept_docs": [],
                "stats": {
                    "original_count": 0, "kept_count": 0,
                    "truncated_chars": 0, "tokens_used": 0,
                }
            }

        kept = []
        used = 0
        truncated_chars = 0

        for doc in documents:
            doc_tokens = self.count_tokens(doc)
            if used + doc_tokens <= budget_tokens:
                kept.append(doc)
                used += doc_tokens
            else:
                remaining = budget_tokens - used
                if remaining > 30 and not kept:
                    # 第一篇就需截断：在句子边界处截断
                    truncated = self._truncate_at_sentence(doc, remaining)
                    kept.append(truncated)
                    used += self.count_tokens(truncated)
                    truncated_chars += len(doc) - len(truncated)
                elif remaining > 50 and kept:
                    truncated = self._truncate_at_sentence(doc, remaining)
                    kept.append(truncated)
                    used += self.count_tokens(truncated)
                    truncated_chars += len(doc) - len(truncated)
                break

        return {
            "kept_docs": kept,
            "stats": {
                "original_count": len(documents),
                "kept_count": len(kept),
                "truncated_chars": truncated_chars,
                "tokens_used": used,
            }
        }

    def _truncate_at_sentence(self, text: str, max_tokens: int) -> str:
        """在句子边界处截断文本"""
        max_chars = max_tokens * 3  # 粗略字符-token转换

        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]

        # 在最后一个句子边界处截断
        sentence_end = max(
            truncated.rfind('。'),
            truncated.rfind('！'),
            truncated.rfind('？'),
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? '),
            truncated.rfind('\n'),
        )

        if sentence_end > max_chars * 0.3:
            return truncated[:sentence_end + 1] + " ..."

        # 句子边界太靠前，回退到词边界
        return truncated[:max_chars - 10] + " ..."


# ===== 测试 =====
if __name__ == "__main__":
    budget = TokenBudget(max_input_tokens=500)

    docs = [
        "文档1内容。" * 50,   # ~250 chars
        "文档2内容。文档2内容。" * 100,  # ~800 chars
        "文档3内容。文档3内容。文档3内容。" * 150,  # ~1800 chars
    ]

    system_prompt = "你是客服助手。请回答用户问题。" * 5
    query = "用户查询"

    allocation = budget.allocate(system_prompt, query, None, len(docs))
    print(f"Allocation: {allocation}")
    print(f"Docs budget: {allocation['docs_budget']} tokens")

    result = budget.truncate_docs_to_budget(docs, allocation["docs_budget"])
    print(f"Result stats: {result['stats']}")
    print(f"Kept {result['stats']['kept_count']}/{result['stats']['original_count']} docs")
```

---

## 练习 5：Jinja2 模板化 Prompt 引擎

```python
"""
Exercise 5: Jinja2 模板化 Prompt 引擎
基于Jinja2的prompt模板管理系统。
"""

from jinja2 import Environment, DictLoader, TemplateNotFound, TemplateSyntaxError
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    """单个prompt模板"""
    name: str
    template_string: str
    parent: Optional[str] = None
    variables: List[str] = field(default_factory=list)


class PromptTemplateEngine:
    """
    基于Jinja2的Prompt模板引擎。

    功能：
    - 注册模板（支持继承）
    - 条件渲染（{% if %}...{% endif %}）
    - 循环渲染（{% for %}...{% endfor %}）
    - 变量替换（{{ variable }}）
    """

    def __init__(self):
        self._templates: Dict[str, str] = {}
        self._env = Environment(
            loader=DictLoader(self._templates),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # 注册内置基础模板
        self.register_template("base_template", """{# Base template #}
{% block system %}{% endblock %}

{% block user %}{% endblock %}

{% block few_shot %}{% endblock %}
""")

    def register_template(self, name: str, template_string: str) -> None:
        """
        注册一个模板。

        Args:
            name: 模板名称
            template_string: Jinja2模板字符串
        """
        self._templates[name] = template_string
        # 重新创建Environment以反映变更
        self._env = Environment(
            loader=DictLoader(self._templates),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(
        self,
        template_name: str,
        **variables,
    ) -> Dict[str, str]:
        """
        渲染模板并返回完整的prompt字典。

        Args:
            template_name: 要渲染的模板名
            **variables: 模板变量

        Returns:
            {"system": "...", "user": "..."}

        Raises:
            ValueError: 模板未找到或缺少必要变量
        """
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self._templates.keys())}")

        try:
            template = self._env.get_template(template_name)
            rendered = template.render(**variables)
        except TemplateNotFound as e:
            raise ValueError(f"Template '{template_name}' not found: {e}")
        except TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error in '{template_name}': {e}")
        except Exception as e:
            raise ValueError(f"Render error in '{template_name}': {e}")

        # 解析渲染结果，分割 system 和 user 部分
        return self._parse_rendered(rendered)

    def _parse_rendered(self, rendered: str) -> Dict[str, str]:
        """将渲染后的文本解析为 system/user 部分"""
        result = {"system": "", "user": ""}

        # 按双换行分割主要部分
        parts = rendered.strip().split('\n\n\n')

        # 简单启发式：第一个非空块作为system
        non_empty = [p.strip() for p in parts if p.strip()]

        if len(non_empty) >= 2:
            result["system"] = non_empty[0]
            result["user"] = '\n\n'.join(non_empty[1:])
        elif len(non_empty) == 1:
            result["user"] = non_empty[0]

        return result

    def list_templates(self) -> List[str]:
        """列出所有已注册的模板"""
        return list(self._templates.keys())


# ===== 测试 =====
if __name__ == "__main__":
    engine = PromptTemplateEngine()

    # 注册子模板（继承base_template）
    engine.register_template("customer_service", """{% extends "base_template" %}

{% block system %}
你是一个{{ role }}。请用{{ language }}回答用户问题。

规则：
{% for rule in rules %}
- {{ rule }}
{% endfor %}
{% endblock %}

{% block user %}
用户问题：{{ query }}

{% if docs %}
参考以下文档：
{% for doc in docs %}
---
标题：{{ doc.title }}
内容：{{ doc.snippet }}
---
{% endfor %}
{% endif %}
{% endblock %}
""")

    # 渲染测试
    result = engine.render(
        "customer_service",
        role="电商客服助手",
        language="中文",
        rules=["态度友好", "回答简洁", "不透露内部信息"],
        query="我的订单什么时候发货？",
        docs=[
            {"title": "发货政策", "snippet": "订单付款后24小时内发货。"},
            {"title": "物流查询", "snippet": "可在'我的订单'页面查询物流信息。"},
        ],
    )

    print("=== SYSTEM ===")
    print(result["system"])
    print("\n=== USER ===")
    print(result["user"])
```

---

## 练习 6：RobustJSONParser（完整实现）

```python
"""
Exercise 6: RobustJSONParser 完整五级回退实现
"""

import json
import re
import ast
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ParseLevel(Enum):
    LEVEL_1_DIRECT = 1
    LEVEL_2_MARKDOWN = 2
    LEVEL_3_BRACE = 3
    LEVEL_4_REPAIR = 4
    LEVEL_5_AST = 5


@dataclass
class ParseResult:
    success: bool
    data: Any = None
    level: Optional[ParseLevel] = None
    error: Optional[str] = None
    repair_log: List[str] = field(default_factory=list)
    original_output: str = ""
    attempts: int = 0


class RobustJSONParser:
    """五级回退JSON解析器"""

    def __init__(self):
        self.stats: Dict[str, int] = {f"level_{i}": 0 for i in range(1, 6)}
        self.stats["total"] = 0
        self.stats["failed"] = 0

    def parse(self, text: str) -> ParseResult:
        self.stats["total"] += 1
        repair_log = []

        # Level 1: 直接 json.loads
        result = self._level_1(text)
        if result.success:
            self.stats["level_1"] += 1
            return result
        repair_log.append(f"L1 failed: {result.error}")

        # Level 2: Markdown 提取
        result = self._level_2(text)
        if result.success:
            self.stats["level_2"] += 1
            result.repair_log = repair_log
            return result
        repair_log.append("L2 failed")

        # Level 3: 大括号提取
        result = self._level_3(text)
        if result.success:
            self.stats["level_3"] += 1
            result.repair_log = repair_log
            return result
        repair_log.append("L3 failed")

        # Level 4: 错误修复
        result = self._level_4(text)
        if result.success:
            self.stats["level_4"] += 1
            result.repair_log = repair_log
            return result
        repair_log.append("L4 failed")

        # Level 5: ast.literal_eval
        result = self._level_5(text)
        if result.success:
            self.stats["level_5"] += 1
            result.repair_log = repair_log
            return result

        self.stats["failed"] += 1
        return ParseResult(
            success=False, level=None,
            error="All 5 levels failed",
            repair_log=repair_log,
            original_output=text, attempts=5,
        )

    def _level_1(self, text: str) -> ParseResult:
        try:
            data = json.loads(text.strip())
            return ParseResult(success=True, data=data,
                             level=ParseLevel.LEVEL_1_DIRECT,
                             original_output=text, attempts=1)
        except json.JSONDecodeError as e:
            return ParseResult(success=False, error=str(e),
                             original_output=text, attempts=1)

    def _level_2(self, text: str) -> ParseResult:
        patterns = [
            r'```(?:json)?\s*\n?(.*?)\n?```',
            r'```\s*\n?(.*?)\n?```',
            r'~~~(?:json)?\s*\n?(.*?)\n?~~~',
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text, re.DOTALL):
                try:
                    data = json.loads(match.strip())
                    return ParseResult(success=True, data=data,
                                     level=ParseLevel.LEVEL_2_MARKDOWN,
                                     original_output=text, attempts=2)
                except json.JSONDecodeError:
                    continue
        return ParseResult(success=False, error="No valid JSON in code blocks",
                         original_output=text, attempts=2)

    def _level_3(self, text: str) -> ParseResult:
        for open_ch in ['{', '[']:
            start = text.find(open_ch)
            if start == -1:
                continue
            close_ch = '}' if open_ch == '{' else ']'
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False; continue
                if ch == '\\':
                    escape = True; continue
                if ch == '"':
                    in_string = not in_string; continue
                if in_string:
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(text[start:i+1])
                            return ParseResult(success=True, data=data,
                                             level=ParseLevel.LEVEL_3_BRACE,
                                             original_output=text, attempts=3)
                        except json.JSONDecodeError:
                            break
        return ParseResult(success=False, error="No balanced braces found",
                         original_output=text, attempts=3)

    def _level_4(self, text: str) -> ParseResult:
        # 提取候选JSON
        extracted = text
        for open_ch in ['{', '[']:
            s = text.find(open_ch)
            if s >= 0:
                close_ch = '}' if open_ch == '{' else ']'
                depth = 0
                in_s = False
                esc = False
                for i in range(s, len(text)):
                    c = text[i]
                    if esc:
                        esc = False; continue
                    if c == '\\':
                        esc = True; continue
                    if c == '"':
                        in_s = not in_s; continue
                    if in_s:
                        continue
                    if c == open_ch:
                        depth += 1
                    elif c == close_ch:
                        depth -= 1
                        if depth == 0:
                            extracted = text[s:i+1]
                            break
                if extracted != text:
                    break

        repair_log = []

        # 修复1: 移除注释
        fixed = re.sub(r'/\*.*?\*/', '', extracted, flags=re.DOTALL)
        lines = []
        for line in fixed.split('\n'):
            in_s = False
            for i, c in enumerate(line):
                if c == '"' and (i == 0 or line[i-1] != '\\'):
                    in_s = not in_s
                if not in_s and c == '/' and i+1 < len(line) and line[i+1] == '/':
                    line = line[:i]
                    break
            lines.append(line)
        fixed = '\n'.join(lines)

        # 修复2: 尾部逗号
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)

        # 修复3: 单引号转双引号
        result_chars = []
        in_double = False
        esc = False
        for c in fixed:
            if esc:
                esc = False; result_chars.append(c); continue
            if c == '\\':
                esc = True; result_chars.append(c); continue
            if c == '"':
                in_double = not in_double; result_chars.append(c)
            elif c == "'" and not in_double:
                result_chars.append('"')
            else:
                result_chars.append(c)
        fixed = ''.join(result_chars)

        # 修复4: 无引号键名
        fixed = re.sub(r'(?<=[\{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', fixed)

        # 修复5: NaN/Infinity
        for kw in ['NaN', 'Infinity', '-Infinity']:
            fixed = re.sub(r'\b' + kw + r'\b', 'null', fixed)

        # 修复6: 补缺失闭合
        counts = {'{': 0, '[': 0}
        pairs = {'{': '}', '[': ']'}
        in_s = False
        esc = False
        for c in fixed:
            if esc:
                esc = False; continue
            if c == '\\':
                esc = True; continue
            if c == '"':
                in_s = not in_s; continue
            if in_s:
                continue
            if c in counts:
                counts[c] += 1
            elif c in ('}', ']'):
                for oc, cc in pairs.items():
                    if c == cc and counts.get(oc, 0) > 0:
                        counts[oc] -= 1
        for ch in ('[', '{'):
            fixed += pairs[ch] * counts.get(ch, 0)

        try:
            data = json.loads(fixed)
            return ParseResult(success=True, data=data,
                             level=ParseLevel.LEVEL_4_REPAIR,
                             repair_log=repair_log,
                             original_output=text, attempts=4)
        except json.JSONDecodeError as e:
            return ParseResult(success=False, error=str(e),
                             original_output=text, attempts=4)

    def _level_5(self, text: str) -> ParseResult:
        for open_ch in ['{', '[']:
            start = text.find(open_ch)
            if start == -1:
                continue
            close_ch = '}' if open_ch == '{' else ']'
            depth = 0
            in_s = False
            esc = False
            for i in range(start, len(text)):
                c = text[i]
                if esc:
                    esc = False; continue
                if c == '\\':
                    esc = True; continue
                if c == '"':
                    in_s = not in_s; continue
                if in_s:
                    continue
                if c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            candidate = candidate.replace('null', 'None')
                            candidate = candidate.replace('true', 'True')
                            candidate = candidate.replace('false', 'False')
                            data = ast.literal_eval(candidate)
                            return ParseResult(success=True, data=data,
                                             level=ParseLevel.LEVEL_5_AST,
                                             original_output=text, attempts=5)
                        except (ValueError, SyntaxError):
                            break
        return ParseResult(success=False, error="ast.literal_eval failed",
                         original_output=text, attempts=5)

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["total"]
        if total == 0:
            return self.stats
        return {
            **self.stats,
            "success_rate": round((total - self.stats["failed"]) / total * 100, 2),
        }


# ===== 测试 =====
if __name__ == "__main__":
    parser = RobustJSONParser()

    tests = [
        '{"name": "Alice", "age": 30}',                   # L1
        '```json\n{"name": "Bob"}\n```',                  # L2
        '结果是：\n{"items": [1,2,3]}\n完成',             # L3
        "{'name': 'Charlie', 'age': 25,}",                 # L4
        '```\n{"tags": ["a", "b"],}\n```',                # L2 + 修复
    ]

    for i, t in enumerate(tests, 1):
        r = parser.parse(t)
        level_name = r.level.name if r.level else "FAILED"
        print(f"Test {i}: {level_name} | success={r.success} | data={r.data}")

    print(f"\nStats: {parser.get_stats()}")
```

---

## 练习 7：A/B 测试流水线

```python
"""
Exercise 7: A/B 测试流水线
比较两个Prompt变体的效果，使用LLM-as-Judge。
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from jinja2 import Template
import statistics


@dataclass
class ABScore:
    """单个评测分数"""
    accuracy: int       # 1-5
    completeness: int   # 1-5
    conciseness: int    # 1-5

    @property
    def total(self) -> int:
        return self.accuracy + self.completeness + self.conciseness


@dataclass
class ABReport:
    """A/B测试报告"""
    total_tests: int
    a_wins: int
    b_wins: int
    ties: int
    a_avg_score: float
    b_avg_score: float
    a_scores: List[ABScore] = field(default_factory=list)
    b_scores: List[ABScore] = field(default_factory=list)
    detailed_results: List[Dict] = field(default_factory=list)


class ABTestPipeline:
    """
    A/B测试流水线。

    对同一组测试数据，分别用两个prompt变体生成答案，
    然后用LLM评判器打分，统计胜出情况。
    """

    def __init__(
        self,
        template_a: str,
        template_b: str,
        test_data: List[Dict[str, str]],
        llm_call,
        judge_llm_call,
    ):
        """
        Args:
            template_a: 变体A的Jinja2模板
            template_b: 变体B的Jinja2模板
            test_data: [{"question": "...", "reference": "..."}, ...]
            llm_call: 用于生成回答的LLM函数
            judge_llm_call: 用于评判的LLM函数（可相同）
        """
        self.template_a = Template(template_a)
        self.template_b = Template(template_b)
        self.test_data = test_data
        self.llm_call = llm_call
        self.judge_llm_call = judge_llm_call

    def run(self) -> ABReport:
        """执行A/B测试并返回报告"""
        a_scores = []
        b_scores = []
        detailed = []

        for i, sample in enumerate(self.test_data):
            question = sample["question"]
            reference = sample.get("reference", "")

            # 生成两个变体的回答
            prompt_a = self.template_a.render(question=question, **sample)
            prompt_b = self.template_b.render(question=question, **sample)

            answer_a = self._generate(prompt_a)
            answer_b = self._generate(prompt_b)

            # LLM评判
            score_a = self._judge(question, answer_a, reference)
            score_b = self._judge(question, answer_b, reference)

            a_scores.append(score_a)
            b_scores.append(score_b)

            detailed.append({
                "question": question,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "score_a": score_a.total,
                "score_b": score_b.total,
            })

        # 统计
        a_wins = sum(
            1 for d in detailed if d["score_a"] > d["score_b"]
        )
        b_wins = sum(
            1 for d in detailed if d["score_b"] > d["score_a"]
        )
        ties = len(detailed) - a_wins - b_wins

        return ABReport(
            total_tests=len(self.test_data),
            a_wins=a_wins,
            b_wins=b_wins,
            ties=ties,
            a_avg_score=statistics.mean(s.total for s in a_scores),
            b_avg_score=statistics.mean(s.total for s in b_scores),
            a_scores=a_scores,
            b_scores=b_scores,
            detailed_results=detailed,
        )

    def _generate(self, prompt: str) -> str:
        """调用LLM生成回答"""
        messages = [
            {"role": "system", "content": "你是一个问答助手。"},
            {"role": "user", "content": prompt},
        ]
        return self.llm_call(messages)

    def _judge(
        self, question: str, answer: str, reference: str
    ) -> ABScore:
        """LLM评判器：对回答质量打分"""
        judge_prompt = f"""请从以下三个维度评估这个回答的质量（1-5分）：

问题：{question}

参考回答：{reference}

待评估回答：{answer}

评分标准：
- 准确性（accuracy）：回答与参考回答的事实一致性。完全一致=5，完全矛盾=1。
- 完整性（completeness）：回答覆盖了参考回答的关键信息点。完全覆盖=5，遗漏重要信息=1。
- 简洁性（conciseness）：回答没有冗余信息，表达精炼。非常简洁=5，大量无关内容=1。

严格输出以下JSON格式：
{{"accuracy": 分数, "completeness": 分数, "conciseness": 分数}}"""

        messages = [
            {"role": "system", "content": "你是一个客观的评测者。只输出JSON。"},
            {"role": "user", "content": judge_prompt},
        ]

        raw = self.judge_llm_call(messages)

        try:
            data = json.loads(raw)
            return ABScore(
                accuracy=int(data["accuracy"]),
                completeness=int(data["completeness"]),
                conciseness=int(data["conciseness"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return ABScore(accuracy=3, completeness=3, conciseness=3)

    def print_report(self, report: ABReport) -> None:
        """打印测试报告"""
        print("=" * 50)
        print("A/B TEST REPORT")
        print("=" * 50)
        print(f"Total tests: {report.total_tests}")
        print(f"Variant A wins: {report.a_wins} ({report.a_wins/report.total_tests*100:.1f}%)")
        print(f"Variant B wins: {report.b_wins} ({report.b_wins/report.total_tests*100:.1f}%)")
        print(f"Ties: {report.ties} ({report.ties/report.total_tests*100:.1f}%)")
        print(f"Variant A avg score: {report.a_avg_score:.2f}/15")
        print(f"Variant B avg score: {report.b_avg_score:.2f}/15")
        print(f"Difference: {report.b_avg_score - report.a_avg_score:+.2f}")
        print("-" * 50)


# ===== 模拟LLM调用 =====
import json

def mock_generate(messages) -> str:
    return "这是一个模拟的回答。包含关键信息点1、2、3。"

def mock_judge(messages) -> str:
    import random
    random.seed(hash(messages[1]["content"]) % 1000)
    acc = random.randint(3, 5)
    comp = random.randint(3, 5)
    conc = random.randint(3, 5)
    return json.dumps({"accuracy": acc, "completeness": comp, "conciseness": conc})


# ===== 测试 =====
if __name__ == "__main__":
    test_data = [
        {"question": "什么是机器学习？",
         "reference": "机器学习是AI的一个分支，使计算机能从数据中学习模式而无需显式编程。"},
        {"question": "Python中list和tuple的区别？",
         "reference": "list可变，可以修改元素；tuple不可变，创建后不能修改。list用方括号，tuple用圆括号。"},
        {"question": "什么是REST API？",
         "reference": "REST API是一种基于HTTP协议的API设计风格，使用GET/POST/PUT/DELETE操作资源。"},
    ]

    template_a = "请简洁回答：{{ question }}"
    template_b = "请详细回答：{{ question }}。请从定义、特点、示例三个方面进行说明。"

    pipeline = ABTestPipeline(
        template_a, template_b, test_data,
        mock_generate, mock_judge,
    )

    report = pipeline.run()
    pipeline.print_report(report)
```

---

## 练习 8：DSPy 自动优化器

```python
"""
Exercise 8: DSPy 自动优化器
使用 DSPy 框架自动优化 Few-shot 示例。

当 DSPy 可用时使用真实实现，不可用时提供模拟实现展示工作原理。
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


# =====================================================
# 模拟实现（当 DSPy 不可用时）
# =====================================================

@dataclass
class OptimizeResult:
    """优化结果"""
    optimized_examples: List[Dict[str, str]]
    baseline_accuracy: float
    optimized_accuracy: float
    improvement: float
    num_examples_selected: int


class MockDSPyOptimizer:
    """
    模拟DSPy的BootstrapFewShot优化器。

    工作流程：
    1. 用训练集评估各种Few-shot组合
    2. 选择在训练集上表现最好的示例组合
    3. 在验证集上评估
    """

    def __init__(
        self,
        train_data: List[Dict[str, str]],
        val_data: List[Dict[str, str]],
        eval_fn,
        max_examples: int = 5,
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.eval_fn = eval_fn
        self.max_examples = max_examples

    def optimize(self) -> OptimizeResult:
        """
        模拟BootstrapFewShot的优化过程。

        真实DSPy中，BootstrapFewShot会：
        1. 用teacher模型生成推理链
        2. 选择最有效的示例
        3. 组合到prompt中
        """
        # 基线：不使用Few-shot
        baseline_correct = 0
        for sample in self.val_data:
            prediction = self._predict(sample["question"], [])
            if self._is_correct(prediction, sample["answer"]):
                baseline_correct += 1
        baseline_acc = baseline_correct / len(self.val_data)

        # 模拟优化：从训练集中贪心选择最佳示例
        # 真实场景中，DSPy 会使用更复杂的搜索策略
        candidate_pool = list(self.train_data)
        selected = []

        for _ in range(min(self.max_examples, len(candidate_pool))):
            best_example = None
            best_score = -1

            for candidate in candidate_pool:
                if candidate in selected:
                    continue
                temp_prompt = selected + [candidate]
                score = self._evaluate_prompt(temp_prompt)
                if score > best_score:
                    best_score = score
                    best_example = candidate

            if best_example and best_score > 0:
                selected.append(best_example)
            else:
                break

        # 优化后验证
        opt_correct = 0
        for sample in self.val_data:
            prediction = self._predict(sample["question"], selected)
            if self._is_correct(prediction, sample["answer"]):
                opt_correct += 1
        opt_acc = opt_correct / len(self.val_data)

        return OptimizeResult(
            optimized_examples=selected,
            baseline_accuracy=round(baseline_acc, 4),
            optimized_accuracy=round(opt_acc, 4),
            improvement=round(opt_acc - baseline_acc, 4),
            num_examples_selected=len(selected),
        )

    def _predict(self, question: str, examples: List[Dict]) -> str:
        """模拟模型预测（含Few-shot prompt）"""
        # 构建prompt
        prompt_parts = ["回答以下问题。只输出答案，不要解释。\n"]

        for ex in examples:
            prompt_parts.append(f"问题：{ex['question']}")
            prompt_parts.append(f"答案：{ex['answer']}\n")

        prompt_parts.append(f"问题：{question}")
        prompt_parts.append("答案：")

        return self.eval_fn("\n".join(prompt_parts))

    def _is_correct(self, prediction: str, expected: str) -> bool:
        """判断预测是否正确（模糊匹配）"""
        return expected.lower().strip() in prediction.lower().strip()

    def _evaluate_prompt(self, examples: List[Dict]) -> float:
        """在训练集上评估一组示例的效果"""
        correct = 0
        for sample in self.train_data:
            if sample in examples:
                continue  # 跳过作为示例的数据
            pred = self._predict(sample["question"], examples)
            if self._is_correct(pred, sample["answer"]):
                correct += 1

        eval_samples = [s for s in self.train_data if s not in examples]
        if not eval_samples:
            return 0.0
        return correct / len(eval_samples)


# =====================================================
# 真实DSPy实现（当DSPy可用时）
# =====================================================

def dspy_optimize_real(
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]],
    llm_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    真实DSPy优化实现。

    使用方法：
    1. pip install dspy
    2. 设置 OPENAI_API_KEY 环境变量
    """
    try:
        import dspy
    except ImportError:
        raise ImportError(
            "DSPy not installed. Install with: pip install dspy\n"
            "Or use the mock implementation above."
        )

    # 配置DSPy
    lm = dspy.LM(model=llm_model)
    dspy.configure(lm=lm)

    # 定义签名
    class QA(dspy.Signature):
        """问答签名"""
        question = dspy.InputField(desc="用户问题")
        answer = dspy.OutputField(desc="问题的答案")

    # 创建训练/验证集
    trainset = [
        dspy.Example(question=s["question"], answer=s["answer"]).with_inputs("question")
        for s in train_data
    ]
    valset = [
        dspy.Example(question=s["question"], answer=s["answer"]).with_inputs("question")
        for s in val_data
    ]

    # 定义评估指标
    def metric(example, pred, trace=None):
        return example.answer.lower().strip() in pred.answer.lower().strip()

    # 基线：零样本
    baseline_program = dspy.ChainOfThought(QA)
    baseline_eval = dspy.Evaluate(
        devset=valset,
        metric=metric,
        num_threads=1,
        display_progress=False,
    )
    baseline_score = baseline_eval(baseline_program)

    # 优化：BootstrapFewShot
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=5,
        max_labeled_demos=3,
    )
    optimized_program = optimizer.compile(
        student=baseline_program,
        trainset=trainset,
    )

    # 优化后评估
    opt_eval = dspy.Evaluate(
        devset=valset,
        metric=metric,
        num_threads=1,
        display_progress=False,
    )
    opt_score = opt_eval(optimized_program)

    return {
        "baseline_accuracy": round(baseline_score * 100, 2),
        "optimized_accuracy": round(opt_score * 100, 2),
        "improvement": round((opt_score - baseline_score) * 100, 2),
        "num_demos": len(optimized_program.demos) if hasattr(optimized_program, 'demos') else 0,
    }


# =====================================================
# 统一入口
# =====================================================

def run_optimization(
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]],
    eval_fn,
    use_real_dspy: bool = False,
) -> Dict[str, Any]:
    """统一的优化入口"""

    if use_real_dspy:
        try:
            return dspy_optimize_real(train_data, val_data)
        except ImportError as e:
            print(f"DSPy not available: {e}")
            print("Falling back to mock implementation...")

    # 使用模拟实现
    optimizer = MockDSPyOptimizer(
        train_data=train_data,
        val_data=val_data,
        eval_fn=eval_fn,
        max_examples=5,
    )
    result = optimizer.optimize()

    return {
        "baseline_accuracy": result.baseline_accuracy,
        "optimized_accuracy": result.optimized_accuracy,
        "improvement": result.improvement,
        "num_examples_selected": result.num_examples_selected,
        "examples": result.optimized_examples,
    }


# =====================================================
# 测试
# =====================================================

def mock_eval_fn(prompt: str) -> str:
    """模拟模型回答"""
    import random
    random.seed(hash(prompt) % 10000)

    # 简单的规则匹配
    questions = {
        "法国的首都是": "巴黎",
        "2+2等于": "4",
        "水的化学式是": "H2O",
        "太阳系最大的行星是": "木星",
        "中国的首都是": "北京",
    }
    for q, a in questions.items():
        if q in prompt:
            return a if random.random() > 0.3 else "不知道"

    return "未知"


if __name__ == "__main__":
    # 准备数据
    train_data = [
        {"question": "法国的首都是什么？", "answer": "巴黎"},
        {"question": "2+2等于多少？", "answer": "4"},
        {"question": "水的化学式是什么？", "answer": "H2O"},
        {"question": "太阳系最大的行星是什么？", "answer": "木星"},
        {"question": "一年有多少天？", "answer": "365"},
    ]

    val_data = [
        {"question": "中国的首都是什么？", "answer": "北京"},
        {"question": "法国首都是什么？", "answer": "巴黎"},
        {"question": "2+2是多少？", "answer": "4"},
    ]

    result = run_optimization(
        train_data, val_data, mock_eval_fn, use_real_dspy=False
    )

    print("=" * 50)
    print("DSPy OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Baseline accuracy:   {result['baseline_accuracy']*100:.1f}%")
    print(f"Optimized accuracy:  {result['optimized_accuracy']*100:.1f}%")
    print(f"Improvement:         {result['improvement']*100:+.1f}%")
    print(f"Examples selected:   {result['num_examples_selected']}")
    if "examples" in result:
        print("\nOptimized Few-shot Examples:")
        for i, ex in enumerate(result["examples"], 1):
            print(f"  {i}. Q: {ex['question']} -> A: {ex['answer']}")
```

---

## 总结

以上8个练习覆盖了Phase 02 Prompt工程的核心技能：

1. **Zero-shot** -- 基础提示，零示例分类
2. **Few-shot** -- 示例引导，稳定输出格式
3. **Chain-of-Thought** -- 多步推理，标记化输出
4. **TokenBudget** -- 上下文管理，预算控制
5. **Jinja2模板** -- 提示工程化，复用管理
6. **RobustJSON** -- 五级回退，容错解析
7. **A/B测试** -- 科学对比，LLM评判
8. **DSPy优化** -- 自动优化，示例选择

每个练习都是独立可运行的模块，可以作为生产系统的基础组件。
