# JSON 输出格式不稳定问题与鲁棒解析

## 一、问题本质

当LLM被要求输出结构化JSON时，它的输出实际上是一个**概率采样过程**，而非确定性程序。以下情况极其常见：

- 多输了一个逗号（trailing comma）
- 在JSON前后加了markdown代码块标记
- 缺失闭合括号或大括号
- JSON中嵌入了自然语言解释
- 键名用了单引号而非双引号
- 嵌套层级过深导致中途截断
- 数字字段变成了字符串（类型不匹配）

核心矛盾：**LLM是语言模型，不是JSON序列化器**。

---

## 二、症状与影响

| 症状 | 生产环境表现 | 影响 |
|------|-------------|------|
| `json.JSONDecodeError` | API返回500 | 用户体验断崖式下降 |
| 字段类型漂移 | Pydantic校验失败 | 下游管道中断 |
| 截断JSON | 解析成功但数据不完整 | 静默数据丢失（更危险） |
| Markdown包裹 | 正则预处理不够健壮 | 间歇性故障 |
| 多余文本 | 提取逻辑失败 | 需要人工介入 |

---

## 三、五级降级解析器（RobustJSONParser）

设计原则：**从最严格到最宽松逐级降级，每级尝试不同的修复策略**。

```
Level 1: json.loads() 直接解析          <- 最快路径，90%的情况
Level 2: Markdown代码块提取后解析       <- 去掉代码块包裹
Level 3: 大括号/方括号提取后解析        <- 从混有文本的输出中提取JSON片段
Level 4: 启发式错误修复后解析           <- 自动修复常见JSON语法错误
Level 5: LLM重试（self-correction）    <- 最慢但最强，让模型自己修正
```

### 完整实现

```python
"""
RobustJSONParser -- 五级降级JSON解析器

每一级失败后自动降级到下一级，确保在LLM输出不稳定的情况下
也能最大概率地提取出合法的JSON数据结构。
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ParseLevel(Enum):
    """解析成功的级别"""
    LEVEL_1_DIRECT = 1       # json.loads 直接成功
    LEVEL_2_MARKDOWN = 2     # 去除markdown包裹后成功
    LEVEL_3_BRACE = 3        # 大括号提取后成功
    LEVEL_4_REPAIR = 4       # 启发式修复后成功
    LEVEL_5_LLM_RETRY = 5    # LLM自我修正后成功


@dataclass
class ParseResult:
    """解析结果"""
    success: bool
    data: Any
    level: Optional[ParseLevel] = None
    error: Optional[str] = None
    repair_log: List[str] = field(default_factory=list)
    original_output: str = ""
    attempts: int = 0


class RobustJSONParser:
    """
    五级降级JSON解析器。

    每一级解析失败后自动降级到下一级更宽松的策略。
    提供完整的修复日志，方便调试和监控。
    """

    def __init__(
        self,
        llm_retry_fn: Optional[Callable[[str, str], str]] = None,
        enable_llm_retry: bool = True,
        max_repair_attempts: int = 5,
        strict_types: bool = False,
    ):
        self.llm_retry_fn = llm_retry_fn
        self.enable_llm_retry = enable_llm_retry
        self.max_repair_attempts = max_repair_attempts
        self.strict_types = strict_types

        self.stats: Dict[str, int] = {
            "total": 0, "level_1": 0, "level_2": 0,
            "level_3": 0, "level_4": 0, "level_5": 0, "failed": 0,
        }

    # ----------------------------------------------------------
    # 公有API
    # ----------------------------------------------------------

    def parse(self, text: str, expected_schema: Optional[Dict] = None) -> ParseResult:
        """主解析入口。尝试五级降级策略解析JSON。"""
        self.stats["total"] += 1
        repair_log = []

        # Level 1: 直接json.loads
        result = self._level_1_direct(text)
        if result.success:
            self.stats["level_1"] += 1
            return result
        repair_log.append(f"Level 1 failed: {result.error}")

        # Level 2: 去除markdown代码块包裹
        result = self._level_2_markdown_extraction(text)
        if result.success:
            self.stats["level_2"] += 1
            result.repair_log = repair_log
            return result
        repair_log.append("Level 2 (markdown extraction) failed")

        # Level 3: 大括号/方括号提取
        result = self._level_3_brace_extraction(text)
        if result.success:
            self.stats["level_3"] += 1
            result.repair_log = repair_log
            return result
        repair_log.append("Level 3 (brace extraction) failed")

        # Level 4: 启发式错误修复
        result = self._level_4_error_repair(text)
        if result.success:
            self.stats["level_4"] += 1
            result.repair_log = repair_log
            return result
        repair_log.append("Level 4 (error repair) failed")

        # Level 5: LLM自我修正
        if self.enable_llm_retry and self.llm_retry_fn:
            result = self._level_5_llm_retry(text, repair_log)
            if result.success:
                self.stats["level_5"] += 1
                return result

        self.stats["failed"] += 1
        return ParseResult(
            success=False, data=None, level=None,
            error=f"All 5 levels failed.", repair_log=repair_log,
            original_output=text, attempts=5,
        )

    # ----------------------------------------------------------
    # Level 1: 直接解析
    # ----------------------------------------------------------

    def _level_1_direct(self, text: str) -> ParseResult:
        cleaned = text.strip()
        try:
            data = json.loads(cleaned)
            return ParseResult(
                success=True, data=data, level=ParseLevel.LEVEL_1_DIRECT,
                original_output=text, attempts=1,
            )
        except json.JSONDecodeError as e:
            return ParseResult(
                success=False, data=None, level=None,
                error=str(e), original_output=text, attempts=1,
            )

    # ----------------------------------------------------------
    # Level 2: Markdown代码块提取
    # ----------------------------------------------------------

    def _level_2_markdown_extraction(self, text: str) -> ParseResult:
        """提取被markdown代码块包裹的JSON。"""
        patterns = [
            r'```(?:json|JSON)?\s*\n?(.*?)\n?```',
            r'```\s*\n?(.*?)\n?```',
            r'`([^`]*)`',
            r'~~~(?:json|JSON)?\s*\n?(.*?)\n?~~~',
        ]

        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text, re.DOTALL)
            for j, extracted in enumerate(matches):
                extracted = extracted.strip()
                if not extracted:
                    continue
                try:
                    data = json.loads(extracted)
                    return ParseResult(
                        success=True, data=data, level=ParseLevel.LEVEL_2_MARKDOWN,
                        original_output=text, attempts=2,
                        repair_log=[f"Extracted via markdown pattern {i+1}, match {j+1}"],
                    )
                except json.JSONDecodeError:
                    continue

        return ParseResult(
            success=False, data=None, level=None,
            error="No valid JSON found inside markdown code blocks",
            original_output=text, attempts=2,
        )

    # ----------------------------------------------------------
    # Level 3: 大括号/方括号提取
    # ----------------------------------------------------------

    def _level_3_brace_extraction(self, text: str) -> ParseResult:
        """从混有自然语言文本的输出中提取最外层的JSON对象或数组。"""
        start_indices = []
        for ch in ['{', '[']:
            idx = text.find(ch)
            if idx != -1:
                start_indices.append((ch, idx))

        if not start_indices:
            return ParseResult(
                success=False, data=None, level=None,
                error="No JSON starting character ({ or [) found",
                original_output=text, attempts=3,
            )

        start_indices.sort(key=lambda x: x[1])

        for open_char, start_idx in start_indices:
            close_char = '}' if open_char == '{' else ']'
            depth = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == open_char:
                    depth += 1
                elif ch == close_char:
                    depth -= 1
                    if depth == 0:
                        json_str = text[start_idx:i + 1]
                        try:
                            data = json.loads(json_str)
                            return ParseResult(
                                success=True, data=data, level=ParseLevel.LEVEL_3_BRACE,
                                original_output=text, attempts=3,
                                repair_log=[f"Extracted from pos {start_idx} to {i+1}"],
                            )
                        except json.JSONDecodeError:
                            break

        return ParseResult(
            success=False, data=None, level=None,
            error="Found JSON-like structures but none parsed",
            original_output=text, attempts=3,
        )

    # ----------------------------------------------------------
    # Level 4: 启发式错误修复
    # ----------------------------------------------------------

    def _level_4_error_repair(self, text: str) -> ParseResult:
        """
        尝试修复常见的JSON语法错误。
        修复: 注释, 尾部逗号, 单引号, 无引号键名, NaN/Infinity, 缺失闭合
        """
        extracted = text
        repair_log = []

        json_str = self._extract_json_candidate(text)
        if json_str:
            extracted = json_str
            repair_log.append("Extracted JSON candidate from text")

        # 4.1: 移除注释
        cleaned = self._remove_json_comments(extracted)
        if cleaned != extracted:
            repair_log.append("Removed JSON comments")
            extracted = cleaned

        # 4.2: 移除尾部逗号
        cleaned = re.sub(r',(\s*[}\]])', r'\1', extracted)
        if cleaned != extracted:
            repair_log.append("Removed trailing commas")
            extracted = cleaned

        # 4.3: 单引号转双引号
        cleaned = self._normalize_quotes(extracted)
        if cleaned != extracted:
            repair_log.append("Normalized quotes")
            extracted = cleaned

        # 4.4: 无引号键名加引号
        cleaned = re.sub(
            r'(?<=[\{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
            r'"\1":', extracted
        )
        if cleaned != extracted:
            repair_log.append("Quoted unquoted keys")
            extracted = cleaned

        # 4.5: NaN/Infinity 转 null
        for kw in ['NaN', 'Infinity', '-Infinity']:
            if kw in extracted:
                extracted = re.sub(r'\b' + kw + r'\b', 'null', extracted)
                repair_log.append(f"Replaced {kw} with null")

        # 4.6: 补缺失闭合括号
        cleaned = self._close_open_structures(extracted)
        if cleaned != extracted:
            repair_log.append("Closed unclosed brackets/braces")
            extracted = cleaned

        try:
            data = json.loads(extracted)
            return ParseResult(
                success=True, data=data, level=ParseLevel.LEVEL_4_REPAIR,
                original_output=text, attempts=4, repair_log=repair_log,
            )
        except json.JSONDecodeError as e:
            return ParseResult(
                success=False, data=None, level=None,
                error=f"Repair failed: {e}", original_output=text,
                attempts=4, repair_log=repair_log,
            )

    def _extract_json_candidate(self, text: str) -> Optional[str]:
        """找到最外层的 { } 或 [ ] 块"""
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
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        return None

    def _remove_json_comments(self, text: str) -> str:
        """移除 // 和 /* */ 注释"""
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        lines = []
        for line in text.split('\n'):
            in_str = False
            for i, ch in enumerate(line):
                if ch == '"' and (i == 0 or line[i-1] != '\\'):
                    in_str = not in_str
                if not in_str and ch == '/' and i+1 < len(line) and line[i+1] == '/':
                    line = line[:i]
                    break
            lines.append(line)
        return '\n'.join(lines)

    def _normalize_quotes(self, text: str) -> str:
        """单引号转双引号（简化状态机）"""
        result = []
        in_double = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                result.append(ch)
                continue
            if ch == '\\':
                escape = True
                result.append(ch)
                continue
            if ch == '"':
                in_double = not in_double
                result.append(ch)
            elif ch == "'" and not in_double:
                result.append('"')
            else:
                result.append(ch)
        return ''.join(result)

    def _close_open_structures(self, text: str) -> str:
        """补充缺失的闭合括号"""
        counts = {'{': 0, '[': 0}
        pairs = {'{': '}', '[': ']'}
        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in counts:
                counts[ch] += 1
            elif ch in ('}', ']'):
                for open_ch, close_ch in pairs.items():
                    if ch == close_ch and counts.get(open_ch, 0) > 0:
                        counts[open_ch] -= 1
        suffix = ''.join(pairs[ch] * counts[ch] for ch in ('[', '{'))
        return text + suffix

    # ----------------------------------------------------------
    # Level 5: LLM自我修正
    # ----------------------------------------------------------

    def _level_5_llm_retry(
        self, text: str, repair_log: List[str]
    ) -> ParseResult:
        """让LLM自己修正格式错误的JSON输出。"""
        if not self.llm_retry_fn:
            return ParseResult(
                success=False, data=None, level=None,
                error="LLM retry function not configured",
                original_output=text, attempts=5,
            )

        error_context = "\n".join(repair_log[-3:])

        correction_prompt = (
            f"The following text was supposed to be valid JSON but parsing failed.\n\n"
            f"Original output:\n```\n{text[:2000]}\n```\n\n"
            f"Errors:\n{error_context}\n\n"
            f"Please output ONLY the corrected valid JSON. No markdown, no explanation."
        )

        try:
            corrected = self.llm_retry_fn(correction_prompt, error_context)
        except Exception as e:
            return ParseResult(
                success=False, data=None, level=None,
                error=f"LLM retry call failed: {e}",
                original_output=text, attempts=5, repair_log=repair_log,
            )

        if not corrected or not corrected.strip():
            return ParseResult(
                success=False, data=None, level=None,
                error="LLM retry returned empty response",
                original_output=text, attempts=5,
            )

        result = self._level_1_direct(corrected)
        if not result.success:
            result = self._level_2_markdown_extraction(corrected)

        if result.success:
            result.level = ParseLevel.LEVEL_5_LLM_RETRY
            result.attempts = 5
            result.repair_log = repair_log + ["LLM retry succeeded"]
            return result

        return ParseResult(
            success=False, data=None, level=None,
            error=f"LLM retry output also failed: {result.error}",
            original_output=text, attempts=5, repair_log=repair_log,
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取解析统计信息"""
        total = self.stats["total"]
        if total == 0:
            return self.stats
        return {
            **self.stats,
            "success_rate": round((total - self.stats["failed"]) / total * 100, 2),
            "level_distribution": {
                f"level_{i}": round(self.stats[f"level_{i}"] / total * 100, 2)
                for i in range(1, 6) if self.stats[f"level_{i}"] > 0
            },
        }

    def reset_stats(self):
        """重置统计计数器"""
        for key in self.stats:
            self.stats[key] = 0


# ============================================================
# 便捷函数与使用示例
# ============================================================

def robust_json_parse(text: str, **kwargs) -> ParseResult:
    """一行调用：鲁棒解析JSON"""
    parser = RobustJSONParser(**kwargs)
    return parser.parse(text)


if __name__ == "__main__":
    parser = RobustJSONParser()

    test_cases = [
        '{"name": "Alice", "age": 30}',
        '```json\n{"name": "Bob"}\n```',
        'Here is the result:\n{"items": [1, 2, 3]}\nHope this helps!',
        "{'name': 'Charlie', 'age': 25,}",
        '{name: "David", score: 95}',
    ]

    for i, case in enumerate(test_cases, 1):
        result = parser.parse(case)
        print(f"Test {i}: level={result.level.name if result.level else 'FAILED'}")
        print(f"  Success: {result.success} | Data: {result.data}")
```

---

## 四、与Pydantic的集成

在生产环境中，建议将RobustJSONParser与Pydantic的`model_validate`结合使用：

```python
from pydantic import BaseModel, ValidationError

class UserProfile(BaseModel):
    name: str
    age: int
    email: str

def parse_and_validate(text: str, model_cls):
    parser = RobustJSONParser()
    result = parser.parse(text)
    if not result.success:
        raise ValueError(f"JSON parsing failed: {result.error}")
    try:
        return model_cls.model_validate(result.data)
    except ValidationError as e:
        raise ValueError(f"Schema mismatch at {result.level.name}: {e}")
```

---

## 五、关键设计决策

1. **为什么不用json_repair库？** -- json_repair是好库，但Level 4是自包含实现，适合零依赖场景。如果环境允许，可将Level 4替换为`json_repair.repair_json()`。

2. **LLM重试的成本** -- Level 5消耗额外token。建议生产环境设置速率限制，并监控Level 5调用次数。

3. **性能** -- Level 1~3在微秒级，Level 4在毫秒级，Level 5取决于模型延迟。95%+在Level 1或2成功。

---

## 六、使用建议

- 对关键业务路径，temperature设置为0或使用`response_format`原生约束
- 对非关键路径，使用五级回退解析器兜底，确保不影响用户体验
- 解析失败时，返回优雅的降级结果而非直接抛500
- 记录解析失败的原始输出到日志，用于持续改进prompt

---

*上一篇：[01 Prompt Injection 攻击](01-prompt-injection.md)*  
*下一篇：[03 位置偏差 Lost in the Middle](03-position-bias.md)*
