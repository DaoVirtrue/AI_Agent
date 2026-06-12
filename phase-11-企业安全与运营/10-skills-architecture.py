#!/usr/bin/env python3
"""
技能系统架构 (Skills Architecture)
=====================================
4层可扩展技能系统:

Layer 1: AIAppRouter - 意图识别 → 路由到技能/Agent/RAG
Layer 2: SkillRegistry - 技能注册中心 (热加载 + 版本管理)
Layer 3: SkillExecutor - 技能执行器 (熔断器 + 降级)
Layer 4: Shared Resources - 共享资源 (LLM池, 向量DB, 缓存, 消息队列)

示例技能:
  - translation (翻译)
  - summarization (摘要)
  - code_review (代码审查)
  - customer_support (客服)
"""

import json
import time
import uuid
import hashlib
import threading
from typing import List, Dict, Optional, Any, Callable, Set, Tuple, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
from abc import ABC, abstractmethod


# ============================================================================
# Layer 1: AI 应用路由器
# ============================================================================

class IntentType(Enum):
    """意图类型"""
    QA = "qa"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CODE_REVIEW = "code_review"
    CUSTOMER_SUPPORT = "customer_support"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: IntentType
    confidence: float
    extracted_entities: Dict[str, str] = field(default_factory=dict)
    suggested_skill: str = ""
    sub_queries: List[str] = field(default_factory=list)


class AIAppRouter:
    """Layer 1: AI应用路由器

    负责:
      1. 意图识别
      2. 路由决策 (技能/Agent/RAG)
      3. 请求预处理
    """

    # 意图→技能的映射
    INTENT_SKILL_MAP = {
        IntentType.QA: "rag_qa",
        IntentType.TRANSLATION: "translation",
        IntentType.SUMMARIZATION: "summarization",
        IntentType.CODE_REVIEW: "code_review",
        IntentType.CUSTOMER_SUPPORT: "customer_support",
        IntentType.CHITCHAT: "chitchat",
        IntentType.UNKNOWN: "rag_qa",  # 默认回退
    }

    def __init__(self, skill_registry: 'SkillRegistry'):
        self.registry = skill_registry
        self.routing_history: deque = deque(maxlen=500)

    def route(self, user_input: str, user_id: str = "", session_id: str = "") -> Tuple[str, IntentResult]:
        """路由用户请求到对应技能

        Args:
            user_input: 用户输入
            user_id: 用户ID
            session_id: 会话ID

        Returns:
            (技能名称, 意图结果)
        """
        # 意图识别
        intent = self._classify_intent(user_input)
        skill_name = self.INTENT_SKILL_MAP.get(intent.intent, "rag_qa")

        # 检查技能是否可用
        skill = self.registry.get_skill(skill_name)
        if skill is None:
            skill_name = "rag_qa"  # 回退到默认

        # 路由记录
        self.routing_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'session_id': session_id,
            'intent': intent.intent.value,
            'confidence': intent.confidence,
            'routed_to': skill_name,
        })

        return skill_name, intent

    def _classify_intent(self, user_input: str) -> IntentResult:
        """意图分类 (基于规则的快速分类)"""
        text_upper = user_input.upper()
        text_lower = user_input.lower()

        # 翻译意图
        translate_keywords = ['TRANSLATE', '翻译', 'TRANSLATION']
        if any(kw in text_upper for kw in translate_keywords):
            return IntentResult(
                intent=IntentType.TRANSLATION,
                confidence=0.9,
                suggested_skill="translation",
            )

        # 摘要意图
        summary_keywords = ['SUMMARIZE', 'SUMMARY', '摘要', '总结', '概括']
        if any(kw in text_upper for kw in summary_keywords):
            return IntentResult(
                intent=IntentType.SUMMARIZATION,
                confidence=0.9,
                suggested_skill="summarization",
            )

        # 代码审查意图
        code_keywords = ['CODE', 'REVIEW', '代码', 'FUNCTION', 'DEBUG', 'BUG']
        if any(kw in text_upper for kw in code_keywords):
            return IntentResult(
                intent=IntentType.CODE_REVIEW,
                confidence=0.8,
                suggested_skill="code_review",
            )

        # 客服意图
        support_keywords = ['HELP', 'PROBLEM', 'ISSUE', '投诉', '帮助', '问题', '退款']
        if any(kw in text_upper for kw in support_keywords):
            return IntentResult(
                intent=IntentType.CUSTOMER_SUPPORT,
                confidence=0.8,
                suggested_skill="customer_support",
            )

        # 闲聊意图
        chitchat_keywords = ['HI', 'HELLO', 'HEY', '你好', '嗨', '谢谢']
        if len(text_lower) < 20 and any(kw in text_lower for kw in chitchat_keywords):
            return IntentResult(
                intent=IntentType.CHITCHAT,
                confidence=0.9,
                suggested_skill="chitchat",
            )

        # 默认: QA
        return IntentResult(
            intent=IntentType.QA,
            confidence=0.7,
            suggested_skill="rag_qa",
        )


# ============================================================================
# Layer 2: 技能注册中心
# ============================================================================

class SkillStatus(Enum):
    """技能状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    CANARY = "canary"  # 金丝雀部署中


@dataclass
class SkillDefinition:
    """技能定义"""
    name: str
    version: str
    description: str
    execute_fn: Callable
    status: SkillStatus = SkillStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SkillRegistry:
    """Layer 2: 技能注册中心

    功能:
      - 技能注册和发现
      - 热加载 (hot-reload)
      - 版本管理
    """

    def __init__(self):
        self._skills: Dict[str, Dict[str, SkillDefinition]] = defaultdict(dict)
        #                       skill_name → {version → definition}

        self._active_version: Dict[str, str] = {}  # skill_name → active_version
        self._lock = threading.RLock()

    def register(self, definition: SkillDefinition) -> bool:
        """注册技能

        Args:
            definition: 技能定义

        Returns:
            是否成功
        """
        with self._lock:
            name = definition.name
            version = definition.version

            self._skills[name][version] = definition

            # 如果是第一个版本，自动激活
            if name not in self._active_version:
                self._active_version[name] = version

            print(f"[注册] 技能 '{name}' v{version} 已注册")
            return True

    def get_skill(self, name: str, version: Optional[str] = None) -> Optional[SkillDefinition]:
        """获取技能

        Args:
            name: 技能名称
            version: 版本号 (None=使用活跃版本)

        Returns:
            技能定义或None
        """
        with self._lock:
            if name not in self._skills:
                return None

            if version:
                return self._skills[name].get(version)
            else:
                active_ver = self._active_version.get(name)
                if active_ver:
                    return self._skills[name].get(active_ver)
        return None

    def set_active_version(self, name: str, version: str) -> bool:
        """设置活跃版本 (热切换)"""
        with self._lock:
            if name in self._skills and version in self._skills[name]:
                self._active_version[name] = version
                print(f"[版本切换] 技能 '{name}' 已切换到 v{version}")
                return True
        return False

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有技能"""
        with self._lock:
            result = []
            for name, versions in self._skills.items():
                active_ver = self._active_version.get(name)
                result.append({
                    'name': name,
                    'active_version': active_ver,
                    'versions': list(versions.keys()),
                    'status': versions[active_ver].status.value if active_ver else 'none',
                })
            return result

    def hot_reload(self, name: str, new_definition: SkillDefinition):
        """热加载: 注册新版本并立即激活"""
        self.register(new_definition)
        self.set_active_version(name, new_definition.version)


# ============================================================================
# Layer 3: 技能执行器 (带熔断)
# ============================================================================

class CircuitState(Enum):
    CLOSED = "closed"           # 正常工作
    OPEN = "open"               # 熔断打开 (拒绝请求)
    HALF_OPEN = "half_open"     # 半开 (尝试恢复)


@dataclass
class CircuitBreaker:
    """熔断器"""
    failure_threshold: int = 5          # 连续失败N次触发熔断
    recovery_timeout_seconds: int = 30   # 熔断后的恢复时间
    half_open_max_requests: int = 3      # 半开状态下允许的试探请求

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """通过熔断器调用函数

        Returns:
            函数返回值

        Raises:
            CircuitBreakerOpen: 熔断器打开
        """
        if self.state == CircuitState.OPEN:
            # 检查是否可以进入半开状态
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = datetime.now()
                else:
                    raise RuntimeError(f"熔断器打开 ({elapsed:.0f}s / {self.recovery_timeout_seconds}s)")

        if self.state == CircuitState.HALF_OPEN and self.failure_count >= self.half_open_max_requests:
            raise RuntimeError("熔断器半开 - 已达试探请求上限")

        try:
            result = fn(*args, **kwargs)
            # 成功: 恢复
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.last_state_change = datetime.now()
            else:
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.last_state_change = datetime.now()
                raise RuntimeError(f"熔断器触发! (连续{self.failure_count}次失败)") from e

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.last_state_change = datetime.now()

            raise


class SkillExecutor:
    """Layer 3: 技能执行器

    功能:
      - 执行技能 (带熔断器)
      - 服务降级回退
      - 执行统计
    """

    def __init__(self, registry: SkillRegistry, degradation_manager=None):
        self.registry = registry
        self.degradation_manager = degradation_manager
        self.circuit_breakers: Dict[str, CircuitBreaker] = defaultdict(CircuitBreaker)
        self.stats: Dict[str, Dict] = defaultdict(lambda: {'calls': 0, 'failures': 0, 'total_latency': 0.0})

    def execute(
        self,
        skill_name: str,
        user_input: str,
        context: Optional[Dict] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行技能

        Args:
            skill_name: 技能名称
            user_input: 用户输入
            context: 上下文
            version: 技能版本

        Returns:
            执行结果
        """
        # 1. 获取技能定义
        skill = self.registry.get_skill(skill_name, version)
        if not skill:
            return {'success': False, 'error': f'技能 "{skill_name}" 未找到'}

        if skill.status == SkillStatus.INACTIVE:
            return {'success': False, 'error': f'技能 "{skill_name}" 已停用'}

        # 2. 检查降级状态
        if self.degradation_manager:
            dm = self.degradation_manager
            if dm.is_feature_disabled(skill_name):
                return {
                    'success': False,
                    'error': f'技能 "{skill_name}" 因服务降级暂时不可用',
                    'degraded': True,
                    'degradation_level': dm.current_level.value,
                }

        # 3. 通过熔断器执行
        breaker = self.circuit_breakers[skill_name]
        start_time = time.time()

        try:
            result = breaker.call(
                skill.execute_fn,
                user_input=user_input,
                context=context or {},
            )
            self.stats[skill_name]['calls'] += 1
            self.stats[skill_name]['total_latency'] += (time.time() - start_time)
            return {'success': True, 'result': result, 'skill': skill_name, 'version': skill.version}
        except RuntimeError as e:
            self.stats[skill_name]['failures'] += 1
            return {'success': False, 'error': str(e), 'circuit_open': True}
        except Exception as e:
            self.stats[skill_name]['failures'] += 1
            return {'success': False, 'error': str(e)}


# ============================================================================
# Layer 4: 共享资源
# ============================================================================

class SharedResources:
    """Layer 4: 共享资源层

    管理:
      - LLM连接池
      - 向量数据库客户端
      - 缓存
      - 消息队列
    """

    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self._pools: Dict[str, List] = {}
        self._lock = threading.RLock()

    def register_resource(self, name: str, resource: Any):
        """注册共享资源"""
        with self._lock:
            self._resources[name] = resource

    def get_resource(self, name: str) -> Optional[Any]:
        """获取共享资源"""
        return self._resources.get(name)

    def create_pool(self, name: str, factory: Callable, pool_size: int = 5):
        """创建资源池"""
        with self._lock:
            self._pools[name] = [factory() for _ in range(pool_size)]

    def acquire(self, pool_name: str, timeout: float = 5.0) -> Optional[Any]:
        """从池中获取资源"""
        pool = self._pools.get(pool_name)
        if pool:
            return pool.pop() if pool else None
        return None

    def release(self, pool_name: str, resource: Any):
        """释放资源回池"""
        if pool_name in self._pools:
            self._pools[pool_name].append(resource)


# ============================================================================
# 示例技能定义
# ============================================================================

def create_sample_skills() -> List[SkillDefinition]:
    """创建示例技能"""
    return [
        SkillDefinition(
            name="translation",
            version="1.0.0",
            description="多语言翻译",
            execute_fn=lambda user_input, context: {
                'translated_text': f"[翻译结果] {user_input[:50]}...",
                'source_lang': 'auto',
                'target_lang': 'en',
            },
        ),
        SkillDefinition(
            name="summarization",
            version="1.0.0",
            description="文本摘要",
            execute_fn=lambda user_input, context: {
                'summary': f"[摘要: {len(user_input)}字符 → 关键要点]",
                'original_length': len(user_input),
            },
        ),
        SkillDefinition(
            name="code_review",
            version="1.0.0",
            description="代码审查",
            execute_fn=lambda user_input, context: {
                'review': "[代码审查结果] 发现2个问题，3个建议",
                'issues': ['未处理的异常', '缺少参数验证'],
            },
        ),
        SkillDefinition(
            name="customer_support",
            version="1.0.0",
            description="客服支持",
            execute_fn=lambda user_input, context: {
                'response': '[客服回复] 已了解您的问题，正在处理中...',
                'ticket_id': str(uuid.uuid4())[:8],
                'category': 'general',
            },
        ),
        SkillDefinition(
            name="rag_qa",
            version="1.0.0",
            description="RAG问答",
            execute_fn=lambda user_input, context: {
                'answer': f'[RAG回答] 关于 "{user_input[:30]}..." 的回答',
                'sources': 3,
            },
        ),
        SkillDefinition(
            name="chitchat",
            version="1.0.0",
            description="闲聊",
            execute_fn=lambda user_input, context: {
                'response': '你好！有什么我可以帮你的吗？',
            },
        ),
    ]


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  技能系统架构 - 完整演示")
    print("=" * 60)

    # 初始化
    print("\n[初始化] 4层技能架构")

    # Layer 4: 共享资源
    resources = SharedResources()
    resources.register_resource("vector_db", {"type": "chromadb", "collection": "rag_kb"})
    resources.register_resource("cache", {"type": "redis", "host": "localhost"})
    print("  Layer 4: 共享资源就绪")

    # Layer 2: 技能注册
    registry = SkillRegistry()
    for skill in create_sample_skills():
        registry.register(skill)
    print("  Layer 2: 技能注册就绪")

    # Layer 3: 技能执行器
    executor = SkillExecutor(registry)
    print("  Layer 3: 技能执行器就绪")

    # Layer 1: 路由器
    router = AIAppRouter(registry)
    print("  Layer 1: 路由器就绪")

    # 测试路由和执行
    print(f"\n{'─' * 60}")
    print("测试请求路由")
    print(f"{'─' * 60}")

    test_queries = [
        "请帮我翻译这段文字: Hello World",
        "总结一下这篇文章的主要内容",
        "Review this Python code for bugs",
        "我的订单为什么还没发货？",
        "什么是RAG技术？",
        "嗨，你好！",
    ]

    for query in test_queries:
        # 路由
        skill_name, intent = router.route(query)

        # 执行
        result = executor.execute(skill_name, query)

        print(f"\n  查询: {query[:50]}...")
        print(f"  意图: {intent.intent.value} ({intent.confidence:.0%})")
        print(f"  路由: {skill_name}")
        print(f"  结果: {result.get('result', result.get('error', ''))}")

    # 列出技能
    print(f"\n{'─' * 60}")
    print("技能列表")
    print(f"{'─' * 60}")

    for skill_info in registry.list_skills():
        print(f"  {skill_info['name']:25s} v{skill_info['active_version']} "
              f"[{skill_info['status']}] (版本: {', '.join(skill_info['versions'])})")

    # 热加载演示
    print(f"\n{'─' * 60}")
    print("热加载演示")
    print(f"{'─' * 60}")

    new_skill = SkillDefinition(
        name="translation",
        version="2.0.0",
        description="多语言翻译 v2 (增强版)",
        execute_fn=lambda user_input, context: {
            'translated_text': f"[v2 增强翻译] {user_input[:30]}...",
        },
    )
    registry.hot_reload("translation", new_skill)

    # 执行新版本
    result = executor.execute("translation", "Hello World 翻译")
    print(f"  新版本结果: {result}")

    # 熔断器演示
    print(f"\n{'─' * 60}")
    print("熔断器演示")
    print(f"{'─' * 60}")

    # 创建一个会失败的技能
    failing_skill = SkillDefinition(
        name="failing_skill",
        version="1.0.0",
        description="测试熔断器的技能",
        execute_fn=lambda user_input, context: 1 / 0,  # 总是失败
    )
    registry.register(failing_skill)

    for i in range(7):
        try:
            result = executor.execute("failing_skill", "test")
            print(f"  尝试{i+1}: 成功")
        except Exception as e:
            print(f"  尝试{i+1}: {str(e)[:60]}")

    print(f"\n  技能统计:")
    for name, stats in executor.stats.items():
        if stats['calls'] > 0 or stats['failures'] > 0:
            print(f"    {name}: {stats['calls']}次调用, {stats['failures']}次失败")

    print()
    print("=" * 60)
    print("  技能系统架构演示完成")
    print("=" * 60)
