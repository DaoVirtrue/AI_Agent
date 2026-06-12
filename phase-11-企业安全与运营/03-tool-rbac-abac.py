#!/usr/bin/env python3
"""
工具权限管理: RBAC + ABAC (Role-Based + Attribute-Based Access Control)
========================================================================
完整的企业级访问控制:

RBAC (基于角色):
  - 角色: admin, developer, viewer, auditor
  - 权限: 每个角色拥有不同的工具访问权限
  - 角色继承: admin 继承 developer 的所有权限

ABAC (基于属性):
  - 动态属性: 时间窗口, 地理位置, 数据敏感度, 请求来源
  - 策略决策: 属性组合决定访问权限
  - 上下文感知: 根据请求上下文动态调整

每个工具调用都被权限检查包装，所有访问决策都被审计。
"""

import json
import time
from typing import List, Dict, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time as dt_time
from collections import defaultdict


# ============================================================================
# 核心数据结构
# ============================================================================

class Permission(Enum):
    """权限枚举"""
    # 检索
    SEARCH_READ = "search.read"
    SEARCH_WRITE = "search.write"
    SEARCH_ADMIN = "search.admin"

    # 生成
    GENERATE_TEXT = "generate.text"
    GENERATE_CODE = "generate.code"
    GENERATE_ADMIN = "generate.admin"

    # 数据
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_EXPORT = "data.export"
    DATA_ADMIN = "data.admin"

    # 系统
    SYSTEM_VIEW = "system.view"
    SYSTEM_CONFIG = "system.config"
    SYSTEM_ADMIN = "system.admin"

    # 用户
    USER_VIEW = "user.view"
    USER_MANAGE = "user.manage"

    # 审计
    AUDIT_VIEW = "audit.view"
    AUDIT_EXPORT = "audit.export"


@dataclass
class Role:
    """角色定义"""
    name: str
    permissions: Set[Permission]
    inherits_from: Set[str] = field(default_factory=set)  # 继承的角色名
    description: str = ""

    def get_all_permissions(self, all_roles: Dict[str, 'Role']) -> Set[Permission]:
        """获取所有权限 (包括继承的)"""
        perms = set(self.permissions)
        for parent_name in self.inherits_from:
            if parent_name in all_roles:
                perms.update(all_roles[parent_name].get_all_permissions(all_roles))
        return perms


@dataclass
class ABACContext:
    """ABAC上下文属性"""
    user_id: str
    roles: List[str]
    request_time: datetime = field(default_factory=datetime.now)
    source_ip: str = ""
    data_sensitivity: str = "public"  # public, internal, confidential, restricted
    request_origin: str = "internal"  # internal, vpn, external
    session_duration_minutes: float = 0.0
    previous_violations: int = 0
    device_trust_level: str = "unknown"  # trusted, managed, untrusted

    def is_business_hours(self) -> bool:
        """是否在工作时间 (9:00-18:00)"""
        now = self.request_time.time()
        start = dt_time(9, 0)
        end = dt_time(18, 0)
        return start <= now <= end

    def is_internal_network(self) -> bool:
        """是否在内网"""
        return self.request_origin in ("internal", "vpn")


@dataclass
class AccessDecision:
    """访问决策"""
    allowed: bool
    reason: str
    decision_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# RBAC 管理器
# ============================================================================

class RBACManager:
    """基于角色的访问控制管理器"""

    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, List[str]] = defaultdict(list)
        self._init_default_roles()

    def _init_default_roles(self):
        """初始化默认角色"""
        # Admin - 拥有所有权限
        self.add_role(Role(
            name="admin",
            permissions={
                Permission.SEARCH_ADMIN,
                Permission.GENERATE_ADMIN,
                Permission.DATA_ADMIN,
                Permission.SYSTEM_ADMIN,
                Permission.USER_MANAGE,
                Permission.AUDIT_VIEW,
                Permission.AUDIT_EXPORT,
            },
            description="系统管理员 - 完全访问",
        ))

        # Developer - 开发和调试
        self.add_role(Role(
            name="developer",
            permissions={
                Permission.SEARCH_READ, Permission.SEARCH_WRITE,
                Permission.GENERATE_TEXT, Permission.GENERATE_CODE,
                Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_EXPORT,
                Permission.SYSTEM_VIEW,
                Permission.AUDIT_VIEW,
            },
            description="开发者 - 读写访问 + 调试",
        ))

        # Viewer - 只读
        self.add_role(Role(
            name="viewer",
            permissions={
                Permission.SEARCH_READ,
                Permission.GENERATE_TEXT,
                Permission.DATA_READ,
                Permission.SYSTEM_VIEW,
            },
            description="查看者 - 只读访问",
        ))

        # Auditor - 审计
        self.add_role(Role(
            name="auditor",
            permissions={
                Permission.AUDIT_VIEW,
                Permission.AUDIT_EXPORT,
            },
            description="审计员 - 仅审计访问",
        ))

    def add_role(self, role: Role):
        """添加角色"""
        self.roles[role.name] = role

    def assign_role(self, user_id: str, role_name: str):
        """为用户分配角色"""
        if role_name not in self.roles:
            raise ValueError(f"角色 '{role_name}' 不存在")
        if role_name not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role_name)

    def remove_role(self, user_id: str, role_name: str):
        """移除用户角色"""
        if role_name in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role_name)

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """获取用户的所有权限 (包含继承)"""
        perms: Set[Permission] = set()
        for role_name in self.user_roles.get(user_id, []):
            role = self.roles.get(role_name)
            if role:
                perms.update(role.get_all_permissions(self.roles))
        return perms

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """检查用户是否有指定权限"""
        return permission in self.get_user_permissions(user_id)


# ============================================================================
# ABAC 管理器
# ============================================================================

class ABACPolicy:
    """ABAC策略"""
    def __init__(self, name: str, condition_fn: Callable, action: str = "allow"):
        self.name = name
        self.condition = condition_fn
        self.action = action  # 'allow' | 'deny'

    def evaluate(self, context: ABACContext) -> bool:
        """评估策略"""
        try:
            return self.condition(context)
        except Exception:
            return False


class ABACManager:
    """基于属性的访问控制管理器"""

    def __init__(self):
        self.policies: List[ABACPolicy] = []
        self._init_default_policies()

    def _init_default_policies(self):
        """初始化默认ABAC策略"""
        # 策略1: 敏感数据只能在内网访问
        self.policies.append(ABACPolicy(
            name="sensitive_data_internal_only",
            condition_fn=lambda ctx: (
                ctx.data_sensitivity in ("confidential", "restricted")
                and not ctx.is_internal_network()
            ),
            action="deny",
        ))

        # 策略2: 管理操作只能在工作时间
        self.policies.append(ABACPolicy(
            name="admin_ops_business_hours",
            condition_fn=lambda ctx: (
                "admin" in ctx.roles
                and not ctx.is_business_hours()
                and ctx.data_sensitivity == "restricted"
            ),
            action="deny",
        ))

        # 策略3: 违规记录过多的用户加强限制
        self.policies.append(ABACPolicy(
            name="violation_threshold",
            condition_fn=lambda ctx: (
                ctx.previous_violations > 5
                and ctx.data_sensitivity != "public"
            ),
            action="deny",
        ))

        # 策略4: 非受信任设备限制
        self.policies.append(ABACPolicy(
            name="untrusted_device_restriction",
            condition_fn=lambda ctx: (
                ctx.device_trust_level == "untrusted"
                and ctx.data_sensitivity in ("confidential", "restricted")
            ),
            action="deny",
        ))

        # 策略5: 外部来源 + 敏感数据 → 需要VPN
        self.policies.append(ABACPolicy(
            name="external_sensitive_vpn_required",
            condition_fn=lambda ctx: (
                ctx.request_origin == "external"
                and ctx.data_sensitivity in ("confidential", "restricted")
            ),
            action="deny",
        ))

    def evaluate(self, context: ABACContext) -> AccessDecision:
        """评估所有ABAC策略

        Deny优先级高于Allow: 只要有一个deny策略匹配，就拒绝。
        """
        for policy in self.policies:
            if policy.evaluate(context):
                return AccessDecision(
                    allowed=False,
                    reason=f"ABAC策略 '{policy.name}' 拒绝访问",
                    decision_id=f"abac_{int(time.time())}",
                    details={'policy': policy.name, 'context': str(context)},
                )

        return AccessDecision(
            allowed=True,
            reason="所有ABAC策略允许",
            decision_id=f"abac_{int(time.time())}",
        )


# ============================================================================
# 统一权限检查
# ============================================================================

class AccessController:
    """统一访问控制器 - 集成RBAC和ABAC"""

    def __init__(self):
        self.rbac = RBACManager()
        self.abac = ABACManager()
        self.audit_log: List[Dict[str, Any]] = []
        self.decision_id_counter = 0

    def check_access(
        self,
        user_id: str,
        required_permission: Permission,
        abac_context: Optional[ABACContext] = None,
        tool_name: str = "",
        parameters: Optional[Dict] = None,
    ) -> AccessDecision:
        """执行完整的访问检查

        流程:
          1. RBAC检查 (用户是否有此权限?)
          2. ABAC检查 (上下文是否允许?)
          3. 审计记录

        Args:
            user_id: 用户ID
            required_permission: 所需权限
            abac_context: ABAC上下文
            tool_name: 工具名称 (用于审计)
            parameters: 工具参数 (用于审计)

        Returns:
            AccessDecision
        """
        self.decision_id_counter += 1
        decision_id = f"ACC_{int(time.time())}_{self.decision_id_counter:06d}"

        # 1. RBAC检查
        if not self.rbac.check_permission(user_id, required_permission):
            decision = AccessDecision(
                allowed=False,
                reason=f"RBAC: 用户 '{user_id}' 缺少权限 '{required_permission.value}'",
                decision_id=decision_id,
                details={
                    'user_roles': self.rbac.user_roles.get(user_id, []),
                    'required_permission': required_permission.value,
                    'user_permissions': [p.value for p in self.rbac.get_user_permissions(user_id)],
                },
            )
            self._audit(decision, user_id, tool_name, parameters)
            return decision

        # 2. ABAC检查
        if abac_context:
            abac_decision = self.abac.evaluate(abac_context)
            if not abac_decision.allowed:
                self._audit(abac_decision, user_id, tool_name, parameters)
                return abac_decision

        # 3. 通过
        decision = AccessDecision(
            allowed=True,
            reason="RBAC和ABAC均通过",
            decision_id=decision_id,
            details={
                'rbac': 'passed',
                'abac': 'passed',
                'permission': required_permission.value,
            },
        )

        self._audit(decision, user_id, tool_name, parameters)
        return decision

    def _audit(
        self,
        decision: AccessDecision,
        user_id: str,
        tool_name: str = "",
        parameters: Optional[Dict] = None,
    ):
        """记录所有访问决策"""
        record = {
            'decision_id': decision.decision_id,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'allowed': decision.allowed,
            'reason': decision.reason,
            'tool': tool_name,
            'parameters_snapshot': json.dumps(parameters)[:200] if parameters else "",
            'details': decision.details,
        }
        self.audit_log.append(record)

    def get_audit_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取审计报告"""
        now = datetime.now()
        cutoff = now.timestamp() - hours * 3600

        recent = [
            r for r in self.audit_log
            if datetime.fromisoformat(r['timestamp']).timestamp() > cutoff
        ]

        allowed = sum(1 for r in recent if r['allowed'])
        denied = sum(1 for r in recent if not r['allowed'])

        # 按用户统计
        by_user = defaultdict(lambda: {'allowed': 0, 'denied': 0})
        for r in recent:
            key = 'allowed' if r['allowed'] else 'denied'
            by_user[r['user_id']][key] += 1

        # 按工具统计
        by_tool = defaultdict(lambda: {'allowed': 0, 'denied': 0})
        for r in recent:
            if r['tool']:
                key = 'allowed' if r['allowed'] else 'denied'
                by_tool[r['tool']][key] += 1

        return {
            'period_hours': hours,
            'total_decisions': len(recent),
            'allowed': allowed,
            'denied': denied,
            'deny_rate': round(denied / max(1, len(recent)), 4),
            'by_user': dict(by_user),
            'by_tool': dict(by_tool),
        }


# ============================================================================
# 工具包装器
# ============================================================================

class ToolWrapper:
    """工具包装器 - 在工具调用前后进行权限检查"""

    def __init__(self, access_controller: AccessController):
        self.ac = access_controller

    def execute(
        self,
        tool_fn: Callable,
        user_id: str,
        required_permission: Permission,
        abac_context: Optional[ABACContext] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """执行工具调用 (带权限检查)

        Args:
            tool_fn: 工具函数
            user_id: 用户ID
            required_permission: 需要的权限
            abac_context: ABAC上下文
            **kwargs: 工具参数

        Returns:
            工具执行结果或权限拒绝信息
        """
        tool_name = tool_fn.__name__ if hasattr(tool_fn, '__name__') else "unknown"

        # 权限检查
        decision = self.ac.check_access(
            user_id=user_id,
            required_permission=required_permission,
            abac_context=abac_context,
            tool_name=tool_name,
            parameters=kwargs,
        )

        if not decision.allowed:
            return {
                'success': False,
                'error': f"访问被拒绝: {decision.reason}",
                'decision_id': decision.decision_id,
                'access_denied': True,
            }

        # 执行工具
        try:
            result = tool_fn(**kwargs)
            return {
                'success': True,
                'result': result,
                'decision_id': decision.decision_id,
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'decision_id': decision.decision_id,
            }


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  RBAC + ABAC 权限管理 - 完整演示")
    print("=" * 60)

    controller = AccessController()

    # ---- 用户角色分配 ----
    print("\n[1] 角色分配")
    print("-" * 40)

    controller.rbac.assign_role("alice", "admin")
    controller.rbac.assign_role("bob", "developer")
    controller.rbac.assign_role("charlie", "viewer")
    controller.rbac.assign_role("diana", "auditor")

    for user in ["alice", "bob", "charlie", "diana"]:
        roles = controller.rbac.user_roles[user]
        perms = controller.rbac.get_user_permissions(user)
        print(f"  {user}: 角色={roles}, 权限数={len(perms)}")

    # ---- RBAC检查 ----
    print("\n[2] RBAC 权限检查")
    print("-" * 40)

    rbac_tests = [
        ("alice", Permission.SYSTEM_ADMIN, "管理员系统管理"),
        ("bob", Permission.DATA_WRITE, "开发者数据写入"),
        ("bob", Permission.SYSTEM_ADMIN, "开发者尝试系统管理"),
        ("charlie", Permission.DATA_EXPORT, "查看者尝试导出"),
        ("charlie", Permission.SEARCH_READ, "查看者阅读搜索"),
    ]

    for user, permission, description in rbac_tests:
        has_perm = controller.rbac.check_permission(user, permission)
        status = "✓ 允许" if has_perm else "✗ 拒绝"
        print(f"  [{status}] {description}: {user} {permission.value}")

    # ---- ABAC检查 ----
    print("\n[3] ABAC 属性检查")
    print("-" * 40)

    abac_tests = [
        ("内部+工作时间", ABACContext(
            user_id="alice", roles=["admin"],
            source_ip="10.0.0.1", data_sensitivity="confidential",
            request_origin="internal", device_trust_level="trusted",
            request_time=datetime(2024, 1, 15, 10, 0),
        ), True),
        ("外部+敏感数据", ABACContext(
            user_id="alice", roles=["admin"],
            source_ip="203.0.113.1", data_sensitivity="confidential",
            request_origin="external", device_trust_level="unknown",
        ), False),
        ("非受信设备+机密", ABACContext(
            user_id="bob", roles=["developer"],
            device_trust_level="untrusted", data_sensitivity="confidential",
            request_origin="internal",
        ), False),
        ("多次违规用户", ABACContext(
            user_id="charlie", roles=["viewer"],
            previous_violations=10, data_sensitivity="internal",
        ), False),
    ]

    for label, ctx, expected in abac_tests:
        decision = controller.abac.evaluate(ctx)
        status = "✓ 允许" if decision.allowed == expected else "✗ 与预期不符"
        print(f"  [{status}] {label}: {decision.allowed} ({decision.reason[:60]})")

    # ---- 统一访问控制 ----
    print("\n[4] 统一访问控制 (RBAC + ABAC)")
    print("-" * 40)

    wrapper = ToolWrapper(controller)

    def search_documents(query: str, top_k: int = 5) -> Dict:
        return {"query": query, "results": ["doc1", "doc2"], "count": 2}

    def generate_report(report_type: str) -> Dict:
        return {"type": report_type, "content": "report content"}

    # 模拟工具调用
    test_calls = [
        ("alice", Permission.SEARCH_READ, ABACContext(
            user_id="alice", roles=["admin"],
            data_sensitivity="public", request_origin="internal",
        ), search_documents, {"query": "RAG技术"}),

        ("charlie", Permission.DATA_EXPORT, ABACContext(
            user_id="charlie", roles=["viewer"],
            data_sensitivity="internal",
        ), generate_report, {"report_type": "analytics"}),

        ("bob", Permission.GENERATE_CODE, ABACContext(
            user_id="bob", roles=["developer"],
            request_origin="external", data_sensitivity="confidential",
        ), lambda: "code generated", {}),
    ]

    for user, perm, ctx, fn, kwargs in test_calls:
        result = wrapper.execute(fn, user, perm, ctx, **kwargs)
        allowed = not result.get('access_denied', False)
        status = "✓" if allowed else "✗"
        print(f"  [{status}] {user} 调用 {fn.__name__}: {allowed}")

    # ---- 审计报告 ----
    print("\n[5] 审计报告")
    print("-" * 40)

    report = controller.get_audit_report(hours=24)
    print(f"  总决策数: {report['total_decisions']}")
    print(f"  允许: {report['allowed']}, 拒绝: {report['denied']}")
    print(f"  拒绝率: {report['deny_rate']:.1%}")

    print(f"\n  按用户:")
    for user, stats in report['by_user'].items():
        print(f"    {user}: 允许={stats['allowed']}, 拒绝={stats['denied']}")

    print()
    print("=" * 60)
    print("  RBAC + ABAC 演示完成")
    print("=" * 60)
