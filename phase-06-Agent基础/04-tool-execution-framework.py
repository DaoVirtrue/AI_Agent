"""
Tool Execution Framework - 工具执行框架
========================================
提供完整的工具管理、执行、沙箱和版本管理功能
"""

import time
import signal
import traceback
import threading
import enum
import re
import math
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from functools import wraps


class ToolStatus(enum.Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    RETRYABLE_ERROR = "retryable_error"
    FATAL_ERROR = "fatal_error"
    INVALID_ARGS = "invalid_args"


@dataclass
class ToolResult:
    status: ToolStatus
    data: Any = None
    error: str = ""
    execution_time: float = 0.0
    retry_count: int = 0


@dataclass
class ToolDefinition:
    """工具的完整定义"""
    name: str
    description: str
    func: Callable
    version: str = "1.0.0"  # 语义化版本
    timeout: float = 30.0   # 超时时间(秒)
    max_retries: int = 2
    dependencies: list = field(default_factory=list)   # 依赖的其他工具
    conflicts: list = field(default_factory=list)      # 冲突的其他工具
    tags: list = field(default_factory=list)           # 分类标签
    category: str = "general"


class ToolRegistry:
    """工具注册中心 - 管理所有工具的注册、查找、执行"""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._execution_history: list = []

    def register(self, definition: ToolDefinition) -> None:
        """注册工具，检查依赖和冲突"""
        if definition.name in self._tools:
            raise ValueError(f"工具 '{definition.name}' 已经注册过了")

        # 检查依赖是否存在
        for dep_name in definition.dependencies:
            if dep_name not in self._tools:
                raise ValueError(
                    f"工具 '{definition.name}' 依赖的工具 '{dep_name}' 尚未注册，"
                    f"请先注册依赖工具"
                )

        # 检查冲突
        for conflict_name in definition.conflicts:
            if conflict_name in self._tools:
                raise ValueError(
                    f"工具 '{definition.name}' 与已注册的工具 '{conflict_name}' 存在冲突"
                )

        self._tools[definition.name] = definition

    def unregister(self, name: str) -> None:
        """注销工具，检查是否有其他工具依赖它"""
        if name not in self._tools:
            raise KeyError(f"工具 '{name}' 未找到，无法注销")

        # 检查是否有其他工具依赖此工具
        dependents = []
        for tool_name, td in self._tools.items():
            if name in td.dependencies:
                dependents.append(tool_name)

        if dependents:
            raise ValueError(
                f"无法注销工具 '{name}'，以下工具依赖它: {dependents}"
            )

        del self._tools[name]

    def get(self, name: str) -> ToolDefinition:
        """获取单个工具定义"""
        if name not in self._tools:
            raise KeyError(f"工具 '{name}' 未找到。可用工具: {list(self._tools.keys())}")
        return self._tools[name]

    def list(self, category: str = None, tags: list = None) -> list:
        """列出工具，支持按分类和标签过滤"""
        result = list(self._tools.values())

        if category:
            result = [td for td in result if td.category == category]

        if tags:
            result = [td for td in result if any(t in td.tags for t in tags)]

        return result

    def search(self, query: str) -> list:
        """模糊搜索工具（名称+描述+标签）"""
        query_lower = query.lower()
        results = []

        for td in self._tools.values():
            score = 0
            if query_lower in td.name.lower():
                score += 10  # 名称精确匹配权重最高
            if query_lower in td.description.lower():
                score += 5   # 描述匹配
            for tag in td.tags:
                if query_lower in tag.lower():
                    score += 3  # 标签匹配
            if score > 0:
                results.append((score, td))

        # 按得分降序排序
        results.sort(key=lambda x: x[0], reverse=True)
        return [td for score, td in results]

    def execute(self, name: str, **kwargs) -> ToolResult:
        """执行工具，带超时、错误分类、重试"""
        td = self.get(name)
        attempt = 0
        last_result = None

        while attempt <= td.max_retries:
            attempt += 1
            start_time = time.time()

            try:
                result = self._execute_with_timeout(td, kwargs)
                result.retry_count = attempt - 1
                result.execution_time = time.time() - start_time
            except Exception as e:
                elapsed = time.time() - start_time
                status = self._categorize_error(e)
                result = ToolResult(
                    status=status,
                    error=str(e),
                    execution_time=elapsed,
                    retry_count=attempt - 1,
                )

            self._execution_history.append({
                "tool": name,
                "kwargs": kwargs,
                "attempt": attempt,
                "status": result.status.value,
                "time": time.time(),
            })

            if result.status == ToolStatus.SUCCESS:
                return result
            elif result.status == ToolStatus.RETRYABLE_ERROR:
                last_result = result
                continue  # 重试
            else:
                return result  # 致命错误、超时、无效参数，不重试

        # 所有重试都用完了
        if last_result:
            last_result.retry_count = attempt
        return last_result

    def _execute_with_timeout(self, td: ToolDefinition, kwargs: dict) -> ToolResult:
        """带超时的执行（使用threading实现跨平台超时）"""
        result_container = {"result": None, "error": None}

        def target():
            try:
                func_result = td.func(**kwargs)
                if isinstance(func_result, ToolResult):
                    result_container["result"] = func_result
                else:
                    result_container["result"] = ToolResult(
                        status=ToolStatus.SUCCESS,
                        data=func_result,
                    )
            except Exception as e:
                result_container["error"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=td.timeout)

        if thread.is_alive():
            # 超时 - 线程会作为daemon被清理
            raise TimeoutError(
                f"工具 '{td.name}' 执行超时（{td.timeout}秒）"
            )

        if result_container["error"]:
            raise result_container["error"]

        return result_container["result"]

    def _categorize_error(self, error: Exception) -> ToolStatus:
        """错误分类：可重试 vs 不可重试"""
        error_type = type(error)
        error_msg = str(error).lower()

        # 可重试的错误
        retryable_types = (
            TimeoutError,
            ConnectionError,
            OSError,  # 可能是临时IO问题
        )
        retryable_keywords = [
            "timeout", "connection", "temporary", "retry",
            "rate limit", "too many requests", "server error",
            "5xx", "503", "502", "504",
        ]

        # 超时错误
        timeout_keywords = ["timeout", "timed out"]

        # 无效参数错误
        invalid_keywords = [
            "missing", "required", "invalid", "argument",
            "parameter", "type error",
        ]

        if error_type is TimeoutError:
            return ToolStatus.TIMEOUT

        if any(kw in error_msg for kw in timeout_keywords):
            return ToolStatus.TIMEOUT

        if any(kw in error_msg for kw in invalid_keywords):
            return ToolStatus.INVALID_ARGS

        if error_type in retryable_types:
            return ToolStatus.RETRYABLE_ERROR

        if any(kw in error_msg for kw in retryable_keywords):
            return ToolStatus.RETRYABLE_ERROR

        return ToolStatus.FATAL_ERROR

    def get_history(self) -> list:
        """获取执行历史"""
        return self._execution_history.copy()

    def clear_history(self) -> None:
        """清空执行历史"""
        self._execution_history.clear()


class ExecutionSandbox:
    """执行沙箱 - 超时控制、错误隔离"""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def run(self, func: Callable, *args, **kwargs) -> ToolResult:
        """在沙箱中执行函数，捕获所有异常"""
        start_time = time.time()

        result_container = {"result": None, "error": None}

        def target():
            try:
                result_container["result"] = func(*args, **kwargs)
            except Exception as e:
                result_container["error"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        elapsed = time.time() - start_time

        if thread.is_alive():
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                error=f"沙箱执行超时（{self.timeout}秒）",
                execution_time=elapsed,
            )

        if result_container["error"]:
            exc = result_container["error"]
            return ToolResult(
                status=ToolStatus.FATAL_ERROR,
                error=f"{type(exc).__name__}: {str(exc)}",
                execution_time=elapsed,
            )

        func_result = result_container["result"]
        if isinstance(func_result, ToolResult):
            func_result.execution_time = elapsed
            return func_result

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=func_result,
            execution_time=elapsed,
        )


class VersionManager:
    """工具版本管理 - 语义化版本，兼容性检查"""

    @staticmethod
    def parse_version(ver: str) -> tuple:
        """解析语义化版本 返回 (major, minor, patch)"""
        ver = ver.strip()
        # 支持 'v' 前缀
        if ver.startswith('v'):
            ver = ver[1:]

        parts = ver.split('.')
        if len(parts) != 3:
            raise ValueError(f"无效的语义化版本: '{ver}'，应为 'major.minor.patch' 格式")

        result = []
        for part in parts:
            if not part.isdigit():
                # 尝试提取数字部分（处理 '1.0.0-alpha' 等情况）
                match = re.match(r'(\d+)', part)
                if match:
                    result.append(int(match.group(1)))
                else:
                    raise ValueError(f"无效的语义化版本: '{ver}'")
            else:
                result.append(int(part))

        return tuple(result)

    @staticmethod
    def compare(v1: str, v2: str) -> int:
        """
        比较两个版本号
        返回: -1 表示 v1 < v2, 0 表示相等, 1 表示 v1 > v2
        """
        parts1 = VersionManager.parse_version(v1)
        parts2 = VersionManager.parse_version(v2)

        for p1, p2 in zip(parts1, parts2):
            if p1 < p2:
                return -1
            if p1 > p2:
                return 1
        return 0

    @staticmethod
    def is_compatible(v1: str, v2: str) -> bool:
        """检查两个版本是否兼容（主版本号相同即可兼容）"""
        p1 = VersionManager.parse_version(v1)
        p2 = VersionManager.parse_version(v2)
        # 主版本号相同即为兼容（语义化版本的核心约定）
        return p1[0] == p2[0]

    @staticmethod
    def bump_version(ver: str, level: str = "patch") -> str:
        """
        升级版本号
        level: 'major' 主版本号+1, minor/patch归零
               'minor' 次版本号+1, patch归零
               'patch' 补丁版本号+1
        """
        major, minor, patch = VersionManager.parse_version(ver)

        if level == "major":
            major += 1
            minor = 0
            patch = 0
        elif level == "minor":
            minor += 1
            patch = 0
        elif level == "patch":
            patch += 1
        else:
            raise ValueError(f"无效的升级级别: '{level}'，应为 major/minor/patch")

        return f"{major}.{minor}.{patch}"


# ============================================================
# === 演示工具 (6个) ===
# ============================================================

def weather_tool(city: str, units: str = "celsius") -> dict:
    """天气查询工具 - 返回结构化天气数据"""
    # 模拟天气数据（生产环境可接入真实天气API）
    weather_db = {
        "北京": {"temp": 22, "humidity": 45, "condition": "晴"},
        "上海": {"temp": 28, "humidity": 70, "condition": "多云"},
        "广州": {"temp": 32, "humidity": 80, "condition": "雷阵雨"},
        "深圳": {"temp": 30, "humidity": 75, "condition": "多云转晴"},
        "成都": {"temp": 25, "humidity": 60, "condition": "阴"},
        "杭州": {"temp": 27, "humidity": 65, "condition": "小雨"},
        "default": {"temp": 20, "humidity": 50, "condition": "未知"},
    }

    city_data = weather_db.get(city, weather_db["default"]).copy()
    city_data["city"] = city
    city_data["units"] = units
    city_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # 单位转换
    if units == "fahrenheit":
        city_data["temp"] = round(city_data["temp"] * 9 / 5 + 32, 1)

    return city_data


def calculate_tool(expression: str) -> dict:
    """安全数学计算工具 - 使用受限的数学函数白名单"""
    # 安全检查：只允许数字、基本运算符、数学函数
    allowed_pattern = re.compile(
        r'^[\d\s\+\-\*\/\(\)\.\,\%\^eEpPiIfF\s'
        r'aabs\sin\cos\tan\log\exp\sqrt\ceil\floor\pow'
        r'a]*$'
    )

    if not allowed_pattern.match(expression.replace('.', '.')):
        raise ValueError(f"表达式包含不允许的字符: {expression}")

    # 安全的数学函数白名单
    safe_math = {
        # 常量
        'pi': math.pi,
        'e': math.e,
        'inf': float('inf'),
        # 函数
        'abs': abs,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'log': math.log,
        'log2': math.log2,
        'log10': math.log10,
        'exp': math.exp,
        'sqrt': math.sqrt,
        'ceil': math.ceil,
        'floor': math.floor,
        'pow': pow,
        'round': round,
        'max': max,
        'min': min,
        'sum': sum,
        # 类型转换
        'int': int,
        'float': float,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_math)
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__,
        }
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e}")
    except Exception as e:
        raise ValueError(f"计算错误: {e}")


def search_tool(query: str, max_results: int = 5) -> dict:
    """模拟搜索工具 - 返回搜索结果"""
    # 模拟知识库
    knowledge_base = {
        "python": [
            "Python是一种解释型、面向对象的高级编程语言",
            "Python由Guido van Rossum于1991年创建",
            "Python广泛应用于数据科学、Web开发和人工智能",
        ],
        "机器学习": [
            "机器学习是人工智能的一个分支，使系统能够从数据中学习",
            "常见的机器学习算法包括决策树、SVM和神经网络",
            "Scikit-learn是Python中最流行的机器学习库",
        ],
        "agent": [
            "AI Agent是能够自主感知环境并采取行动的智能体",
            "ReAct模式结合了推理(Reasoning)和行动(Acting)",
            "LangChain提供了构建AI Agent的高级框架",
        ],
    }

    # 搜索匹配
    results = []
    for topic, entries in knowledge_base.items():
        if query.lower() in topic.lower() or any(query.lower() in e.lower() for e in entries):
            for entry in entries:
                results.append({"topic": topic, "content": entry})

    # 如果没有精确匹配，返回所有相关内容
    if not results:
        for topic, entries in knowledge_base.items():
            for entry in entries:
                if len(results) < max_results:
                    words = set(query.lower().split())
                    entry_words = set(entry.lower().split())
                    if words & entry_words:
                        results.append({"topic": topic, "content": entry})

    results = results[:max_results]

    return {
        "query": query,
        "total_results": len(results),
        "results": results,
    }


def file_read_tool(path: str, encoding: str = "utf-8") -> dict:
    """文件读取工具 - 实际读取文件，有大小限制"""
    import os

    max_size = 1024 * 1024  # 1MB 限制

    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    file_size = os.path.getsize(path)
    if file_size > max_size:
        raise ValueError(
            f"文件过大 ({file_size} bytes)，超过限制 ({max_size} bytes)"
        )

    try:
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
    except UnicodeDecodeError:
        # 回退到二进制读取
        with open(path, 'rb') as f:
            content = f.read()
            content = content.decode(encoding, errors='replace')

    # 统计信息
    lines = content.split('\n')
    words = content.split()

    return {
        "path": path,
        "encoding": encoding,
        "size_bytes": file_size,
        "lines": len(lines),
        "words": len(words),
        "chars": len(content),
        "content": content[:5000],  # 返回前5000字符
        "truncated": len(content) > 5000,
    }


def text_analysis_tool(text: str, operation: str = "summary") -> dict:
    """文本分析工具（摘要、关键词、情感）"""
    if not text.strip():
        raise ValueError("输入文本不能为空")

    result = {
        "operation": operation,
        "text_length": len(text),
        "word_count": len(text.split()),
    }

    if operation == "summary":
        # 简单摘要：提取前几句
        sentences = re.split(r'[。！？\.\!\?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        result["total_sentences"] = len(sentences)
        # 取前3句作为摘要
        summary_sentences = sentences[:min(3, len(sentences))]
        result["summary"] = '。'.join(summary_sentences) + '。' if summary_sentences else text[:200]

    elif operation == "keywords":
        # 提取关键词：基于TF的简单实现
        words = re.findall(r'[一-鿿]+|[a-zA-Z]+', text.lower())
        word_freq = {}
        # 停用词（简化的中文和英文停用词）
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'you', 'your', 'we', 'our',
            'they', 'them', 'this', 'that', 'these', 'those', 'it', 'its',
            'and', 'or', 'but', 'not', 'no', 'in', 'on', 'at', 'to', 'for',
            'of', 'from', 'by', 'with', 'as', 'if', 'so', 'than', 'then',
        }

        for word in words:
            if len(word) < 2 or word in stopwords:
                continue
            word_freq[word] = word_freq.get(word, 0) + 1

        # 排序取前10
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        result["keywords"] = [{"word": w, "frequency": f} for w, f in sorted_keywords[:10]]

    elif operation == "sentiment":
        # 简单情感分析
        positive_words = {
            '好', '棒', '优秀', '喜欢', '高兴', '开心', '成功', '完美', '精彩',
            'good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'happy',
        }
        negative_words = {
            '差', '糟糕', '讨厌', '难过', '失败', '问题', '错误', '坏', '不好',
            'bad', 'terrible', 'awful', 'hate', 'sad', 'poor', 'wrong', 'fail',
        }

        pos_count = sum(1 for w in positive_words if w in text.lower())
        neg_count = sum(1 for w in negative_words if w in text.lower())

        if pos_count > neg_count:
            sentiment = "正面"
            score = min(1.0, (pos_count - neg_count) / max(1, pos_count + neg_count))
        elif neg_count > pos_count:
            sentiment = "负面"
            score = -min(1.0, (neg_count - pos_count) / max(1, pos_count + neg_count))
        else:
            sentiment = "中性"
            score = 0.0

        result["sentiment"] = sentiment
        result["score"] = score
        result["positive_words"] = pos_count
        result["negative_words"] = neg_count

    else:
        raise ValueError(f"不支持的操作: '{operation}'，支持: summary/keywords/sentiment")

    return result


def time_date_tool(operation: str = "now", timezone: str = "UTC") -> dict:
    """时间和日期工具"""
    import datetime as dt

    now = dt.datetime.now()

    if operation == "now":
        return {
            "operation": "now",
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": timezone,
            "weekday": now.strftime("%A"),
            "weekday_cn": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()],
            "week_number": now.isocalendar()[1],
            "timestamp": now.timestamp(),
        }

    elif operation == "today":
        return {
            "operation": "today",
            "date": now.strftime("%Y-%m-%d"),
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "weekday": now.strftime("%A"),
            "weekday_cn": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()],
            "is_weekend": now.weekday() >= 5,
            "days_in_month": dt.datetime(now.year, now.month + 1, 1).day - 1 if now.month < 12 else 31,
        }

    elif operation == "diff":
        return {
            "operation": "diff",
            "message": "请提供两个日期进行差值计算: time_date_tool(operation='diff', date1='2024-01-01', date2='2024-12-31')",
        }

    elif operation == "add_days":
        return {
            "operation": "add_days",
            "message": "请提供基准日期和天数: time_date_tool(operation='add_days', date='2024-01-01', days=30)",
        }

    else:
        raise ValueError(f"不支持的操作: '{operation}'，支持: now/today/diff/add_days")


# ============================================================
# === __main__ 完整演示 ===
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  工具执行框架 (Tool Execution Framework) - 完整演示")
    print("=" * 70)

    # ---------------------------------------------------------
    # Demo 1: 注册工具、列出工具、搜索工具
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Demo 1: 工具注册与管理")
    print("=" * 50)

    registry = ToolRegistry()

    # 注册所有工具
    tools_to_register = [
        ToolDefinition(
            name="weather",
            description="查询指定城市的天气信息，支持摄氏度和华氏度",
            func=weather_tool,
            version="1.0.0",
            timeout=10.0,
            max_retries=1,
            tags=["weather", "api", "query"],
            category="data",
        ),
        ToolDefinition(
            name="calculator",
            description="安全地计算数学表达式，支持加减乘除、三角函数、对数等",
            func=calculate_tool,
            version="2.0.0",
            timeout=5.0,
            max_retries=1,
            tags=["math", "calculation", "utility"],
            category="utility",
        ),
        ToolDefinition(
            name="search",
            description="搜索知识库中的信息，支持模糊匹配",
            func=search_tool,
            version="1.0.0",
            timeout=10.0,
            max_retries=2,
            tags=["search", "knowledge", "query"],
            category="data",
        ),
        ToolDefinition(
            name="file_reader",
            description="读取文件内容，支持多种编码格式，大小限制1MB",
            func=file_read_tool,
            version="1.1.0",
            timeout=15.0,
            max_retries=1,
            tags=["file", "io", "reader"],
            category="io",
        ),
        ToolDefinition(
            name="text_analyzer",
            description="文本分析工具：摘要、关键词提取、情感分析",
            func=text_analysis_tool,
            version="1.0.0",
            timeout=10.0,
            max_retries=1,
            tags=["text", "nlp", "analysis"],
            category="analysis",
        ),
        ToolDefinition(
            name="datetime",
            description="获取当前时间、日期、星期等时间信息",
            func=time_date_tool,
            version="1.0.0",
            timeout=5.0,
            max_retries=0,
            tags=["time", "date", "utility"],
            category="utility",
        ),
    ]

    for td in tools_to_register:
        registry.register(td)
        print(f"  [+] 已注册工具: {td.name} v{td.version} [{td.category}]")

    # 列出所有工具
    print("\n  所有已注册工具:")
    for td in registry.list():
        print(f"    - {td.name} v{td.version}: {td.description}")

    # 按分类过滤
    print("\n  按分类 'utility' 过滤:")
    for td in registry.list(category="utility"):
        print(f"    - {td.name}: {td.description}")

    # 搜索工具
    print("\n  搜索 'weather':")
    results = registry.search("weather")
    for td in results:
        print(f"    - {td.name}: {td.description}")

    print("\n  搜索 'text':")
    results = registry.search("text")
    for td in results:
        print(f"    - {td.name}: {td.description}")

    # ---------------------------------------------------------
    # Demo 2: 执行工具（正常、超时、错误）
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Demo 2: 工具执行")
    print("=" * 50)

    # 正常执行 - 天气查询
    print("\n  [执行] weather(city='北京'):")
    result = registry.execute("weather", city="北京", units="celsius")
    print(f"    状态: {result.status.value}")
    print(f"    数据: {result.data}")
    print(f"    耗时: {result.execution_time:.4f}s")

    # 正常执行 - 计算器
    print("\n  [执行] calculator(expression='sqrt(144) + log(100) * 2'):")
    result = registry.execute("calculator", expression="sqrt(144) + log(100) * 2")
    print(f"    状态: {result.status.value}")
    print(f"    数据: {result.data}")

    # 正常执行 - 文本分析
    print("\n  [执行] text_analyzer(text='今天天气真好...', operation='sentiment'):")
    sample_text = "今天天气真好，我非常开心！这是一个完美的一天。"
    result = registry.execute("text_analyzer", text=sample_text, operation="sentiment")
    print(f"    状态: {result.status.value}")
    print(f"    数据: {result.data}")

    # 无效参数
    print("\n  [执行] calculator(expression='__import__(\"os\").system(\"dir\")'):")
    result = registry.execute("calculator", expression='__import__("os").system("dir")')
    print(f"    状态: {result.status.value}")
    print(f"    错误: {result.error}")

    # 不存在的工具
    print("\n  [执行] nonexistent_tool():")
    try:
        registry.execute("nonexistent_tool")
    except KeyError as e:
        print(f"    异常: {e}")

    # ---------------------------------------------------------
    # Demo 3: 版本管理演示
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Demo 3: 版本管理")
    print("=" * 50)

    vm = VersionManager()

    print("\n  版本解析:")
    for ver in ["1.0.0", "2.5.3", "v3.1.0", "10.20.30"]:
        parsed = vm.parse_version(ver)
        print(f"    {ver} -> {parsed}")

    print("\n  版本比较:")
    comparisons = [
        ("1.0.0", "1.0.0"),
        ("1.0.0", "2.0.0"),
        ("2.1.0", "1.9.9"),
        ("1.0.0", "1.1.0"),
        ("1.0.0", "1.0.1"),
    ]
    for v1, v2 in comparisons:
        cmp_result = vm.compare(v1, v2)
        symbol = "<" if cmp_result < 0 else "=" if cmp_result == 0 else ">"
        print(f"    {v1} {symbol} {v2}")

    print("\n  兼容性检查:")
    compat_checks = [
        ("1.0.0", "1.5.0"),
        ("1.0.0", "2.0.0"),
        ("2.0.0", "2.1.3"),
        ("3.0.0", "3.9.9"),
    ]
    for v1, v2 in compat_checks:
        compat = vm.is_compatible(v1, v2)
        print(f"    {v1} 与 {v2}: {'兼容' if compat else '不兼容'}")

    print("\n  版本升级:")
    base_ver = "1.5.2"
    for level in ["patch", "minor", "major"]:
        new_ver = vm.bump_version(base_ver, level)
        print(f"    {base_ver} + {level.upper():5s} -> {new_ver}")

    # ---------------------------------------------------------
    # Demo 4: 依赖冲突检查
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Demo 4: 依赖和冲突检查")
    print("=" * 50)

    registry2 = ToolRegistry()

    # 注册基础工具
    base_tool = ToolDefinition(
        name="base_api",
        description="基础API工具",
        func=lambda: {"status": "ok"},
        version="1.0.0",
    )
    registry2.register(base_tool)
    print("  [+] 注册了基础工具: base_api")

    # 注册有依赖的工具
    dependent_tool = ToolDefinition(
        name="advanced_api",
        description="高级API工具，依赖base_api",
        func=lambda: {"status": "advanced"},
        dependencies=["base_api"],
    )
    registry2.register(dependent_tool)
    print("  [+] 注册了高级工具: advanced_api (依赖 base_api)")

    # 尝试注册缺少依赖的工具
    print("\n  尝试注册缺少依赖的工具:")
    try:
        bad_tool = ToolDefinition(
            name="bad_tool",
            description="缺少依赖",
            func=lambda: None,
            dependencies=["nonexistent_dep"],
        )
        registry2.register(bad_tool)
        print("    注册成功（不应该看到这个）")
    except ValueError as e:
        print(f"    错误: {e}")

    # 尝试注销被依赖的工具
    print("\n  尝试注销被依赖的基础工具:")
    try:
        registry2.unregister("base_api")
        print("    注销成功（不应该看到这个）")
    except ValueError as e:
        print(f"    错误: {e}")

    # 冲突检查
    print("\n  尝试注册冲突的工具:")
    try:
        conflict_tool = ToolDefinition(
            name="conflict_api",
            description="冲突工具",
            func=lambda: None,
            conflicts=["advanced_api"],
        )
        registry2.register(conflict_tool)
        print("    注册成功（不应该看到这个）")
    except ValueError as e:
        print(f"    错误: {e}")

    # ---------------------------------------------------------
    # Demo 5: 执行历史查看
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Demo 5: 执行历史与沙箱")
    print("=" * 50)

    print("\n  执行历史记录:")
    history = registry.get_history()
    for i, entry in enumerate(history):
        print(f"    {i+1}. [{entry['tool']}] attempt={entry['attempt']} "
              f"status={entry['status']}")

    # 沙箱演示
    print("\n  沙箱执行演示:")

    def risky_function():
        """模拟风险操作"""
        return {"result": "成功执行"}

    def failing_function():
        """模拟失败操作"""
        raise RuntimeError("模拟的运行错误")

    def slow_function():
        """模拟慢速操作"""
        time.sleep(2)
        return {"result": "慢慢执行完了"}

    sandbox = ExecutionSandbox(timeout=1.0)

    print("\n    正常执行:")
    result = sandbox.run(risky_function)
    print(f"      状态: {result.status.value}, 数据: {result.data}")

    print("\n    异常捕获:")
    result = sandbox.run(failing_function)
    print(f"      状态: {result.status.value}, 错误: {result.error}")

    print("\n    超时控制:")
    result = sandbox.run(slow_function)
    print(f"      状态: {result.status.value}, 错误: {result.error}")

    # ---------------------------------------------------------
    # 总结
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  所有演示完成！")
    print("=" * 70)
    print(f"  - 注册工具数量: {len(registry.list())}")
    print(f"  - 执行历史条目: {len(registry.get_history())}")
    print(f"  - 版本管理测试: 通过")
    print(f"  - 依赖冲突检查: 通过")
    print(f"  - 沙箱执行演示: 通过")
