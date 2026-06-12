#!/usr/bin/env python3
"""
C5-03：MCP 安全认证
======================
当 MCP Server 暴露在网络上时，安全认证是必须要解决的问题。

本文件演示四种认证策略：
1. OAuth 2.0 Bearer Token —— 标准企业认证
2. API Key Header —— 简单 API 密钥认证
3. 会话级认证 —— Session Token 管理
4. 工具级权限 —— 细粒度的 Tool 访问控制

MCP 协议在 2025 年引入了 Auth 扩展（RFC 兼容）。

依赖：pip install mcp httpx
"""

import os
import json
import hashlib
import time
import asyncio
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# ============================================================================
# 第一部分：安全模型概述
# ============================================================================

def explain_security_model():
    """MCP 安全模型概述。"""
    print("=" * 60)
    print("【MCP 安全模型】")
    print("=" * 60)

    print("""
    MCP 安全分层模型：

    ┌──────────────────────────────────────────────────┐
    │              第1层：传输安全 (TLS)                │
    │  HTTPS 加密所有通信，防止中间人攻击               │
    ├──────────────────────────────────────────────────┤
    │           第2层：身份认证 (Authentication)        │
    │  验证"你是谁"：OAuth2 / API Key / mTLS           │
    ├──────────────────────────────────────────────────┤
    │          第3层：授权 (Authorization)              │
    │  决定"你能做什么"：工具级/资源级权限              │
    ├──────────────────────────────────────────────────┤
    │           第4层：审计 (Audit)                     │
    │  记录所有操作：谁、何时、做了什么、结果           │
    ├──────────────────────────────────────────────────┤
    │          第5层：输入验证 (Input Validation)       │
    │  防止注入攻击、参数篡改、越权访问                 │
    └──────────────────────────────────────────────────┘

    关键原则：
    - 深度防御：不依赖单一安全层
    - 最小权限：只授予完成任务所需的最小权限
    - 零信任：不信任任何来源，始终验证
    - 审计完整：所有操作可追溯
    """)


# ============================================================================
# 第二部分：OAuth 2.0 Bearer Token 认证
# ============================================================================

class OAuth2AuthProvider:
    """
    OAuth 2.0 Bearer Token 认证。

    流程：
    1. 客户端向 Auth Server 获取 Token
    2. 请求 MCP Server 时携带 Authorization: Bearer <token>
    3. MCP Server 验证 Token 有效性
    4. Token 过期后客户端刷新
    """

    def __init__(self, issuer_url: str = "https://auth.example.com"):
        self.issuer_url = issuer_url
        self._tokens: Dict[str, dict] = {}  # token → {user, scope, expires}
        self._refresh_tokens: Dict[str, str] = {}  # refresh_token → access_token

    def issue_token(
        self,
        user_id: str,
        scopes: List[str],
        expires_in: int = 3600,
    ) -> dict:
        """
        签发访问令牌。

        Args:
            user_id: 用户ID
            scopes: 权限范围，如 ["tools:read", "tools:execute", "resources:read"]
            expires_in: 过期时间（秒）
        """
        access_token = f"mcp_at_{hashlib.sha256(f'{user_id}{time.time()}'.encode()).hexdigest()[:32]}"
        refresh_token = f"mcp_rt_{hashlib.sha256(f'{user_id}{time.time()}refresh'.encode()).hexdigest()[:32]}"

        token_data = {
            "user_id": user_id,
            "scopes": scopes,
            "expires_at": datetime.now() + timedelta(seconds=expires_in),
            "issued_at": datetime.now().isoformat(),
        }

        self._tokens[access_token] = token_data
        self._refresh_tokens[refresh_token] = access_token

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "scopes": scopes,
        }

    def validate_token(self, token: str) -> Optional[dict]:
        """验证 Token 并返回用户信息。"""
        token_data = self._tokens.get(token)
        if not token_data:
            return None
        if datetime.now() > token_data["expires_at"]:
            del self._tokens[token]
            return None
        return token_data

    def refresh_access_token(self, refresh_token: str) -> Optional[dict]:
        """用 refresh_token 刷新 access_token。"""
        old_token = self._refresh_tokens.get(refresh_token)
        if not old_token or old_token not in self._tokens:
            return None

        # 删除旧 token，签发新的
        user_data = self._tokens.pop(old_token)
        del self._refresh_tokens[refresh_token]

        return self.issue_token(
            user_id=user_data["user_id"],
            scopes=user_data["scopes"],
        )


def demo_oauth2_auth():
    """OAuth2 认证演示。"""
    print("\n" + "=" * 60)
    print("【1】OAuth 2.0 Bearer Token 认证")
    print("=" * 60)

    auth = OAuth2AuthProvider()

    # 签发 Token
    token_response = auth.issue_token(
        user_id="user_12345",
        scopes=["tools:read", "tools:execute", "resources:read"],
    )
    print(f"  用户: user_12345")
    print(f"  Access Token: {token_response['access_token'][:20]}...")
    print(f"  权限范围: {token_response['scopes']}")
    print(f"  过期时间: {token_response['expires_in']}秒")

    # 验证 Token
    validated = auth.validate_token(token_response["access_token"])
    if validated:
        print(f"\n  [验证通过] 用户: {validated['user_id']}")
        print(f"  权限: {validated['scopes']}")

    # 刷新 Token
    new_token = auth.refresh_access_token(token_response["refresh_token"])
    if new_token:
        print(f"\n  [刷新成功] 新 Token: {new_token['access_token'][:20]}...")

    # MCP Server 中集成 OAuth2 的代码模式
    print("\n  MCP Server 集成代码:")
    code = '''
# 在 MCP Server 中验证 OAuth Token
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class OAuth2Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        token = auth_header[7:]
        user_info = auth_provider.validate_token(token)
        if not user_info:
            return JSONResponse({"error": "token_expired"}, status_code=401)

        request.state.user = user_info
        return await call_next(request)
'''
    print(code)


# ============================================================================
# 第三部分：API Key 认证
# ============================================================================

class ApiKeyAuthProvider:
    """
    API Key 认证（适用于服务间调用）。

    与 OAuth 的区别：
    - OAuth 适合用户授权场景（有人参与）
    - API Key 适合服务间调用（无人参与）
    """

    def __init__(self):
        self._api_keys: Dict[str, dict] = {}

    def create_api_key(
        self,
        name: str,
        permissions: List[str],
        rate_limit: int = 1000,
    ) -> str:
        """创建 API Key。"""
        api_key = f"mcp_sk_{hashlib.sha256(f'{name}{time.time()}'.encode()).hexdigest()[:40]}"
        self._api_keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "call_count": 0,
        }
        return api_key

    def validate_api_key(self, api_key: str) -> Optional[dict]:
        """验证 API Key。"""
        return self._api_keys.get(api_key)

    def check_permission(self, api_key: str, required: str) -> bool:
        """检查是否有指定权限。"""
        key_info = self._api_keys.get(api_key)
        if not key_info:
            return False
        return required in key_info["permissions"]

    def revoke_api_key(self, api_key: str) -> bool:
        """吊销 API Key。"""
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            return True
        return False


def demo_api_key_auth():
    """API Key 认证演示。"""
    print("\n" + "=" * 60)
    print("【2】API Key Header 认证")
    print("=" * 60)

    auth = ApiKeyAuthProvider()

    # 创建不同权限级别的 API Key
    admin_key = auth.create_api_key(
        name="admin-service",
        permissions=["tools:*", "resources:*", "prompts:*"],
    )
    readonly_key = auth.create_api_key(
        name="readonly-service",
        permissions=["tools:read", "resources:read"],
    )

    print(f"  Admin API Key: {admin_key[:30]}...")
    print(f"    权限: tools:*, resources:*, prompts:*")

    print(f"\n  ReadOnly API Key: {readonly_key[:30]}...")
    print(f"    权限: tools:read, resources:read")

    # 验证权限
    print(f"\n  Admin 能否执行 tools:write? {auth.check_permission(admin_key, 'tools:write')}")
    print(f"  ReadOnly 能否执行 tools:write? {auth.check_permission(readonly_key, 'tools:write')}")

    # MCP Server 中通过 X-API-Key header 接收
    print("\n  MCP Server 集成代码:")
    code = '''
# 通过 HTTP Header 传递 API Key
# 客户端:
headers = {"X-API-Key": "mcp_sk_xxxx"}

# 服务端验证:
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key or not auth_provider.validate_api_key(api_key):
        return JSONResponse({"error": "invalid_api_key"}, 401)

    if request.url.path.startswith("/tools/"):
        tool_name = request.url.path.split("/")[-1]
        if not auth_provider.check_permission(api_key, f"tools:{tool_name}"):
            return JSONResponse({"error": "forbidden"}, 403)

    return await call_next(request)
'''
    print(code)


# ============================================================================
# 第四部分：会话级认证
# ============================================================================

class SessionAuthProvider:
    """
    会话级认证：每个 MCP 连接维护独立的会话上下文。

    特点：
    - 会话隔离：不同会话看到不同的数据
    - 超时管理：会话超时自动失效
    - 多租户支持：会话关联租户和用户
    """

    def __init__(self, session_timeout: int = 1800):
        self.session_timeout = session_timeout
        self._sessions: Dict[str, dict] = {}

    def create_session(self, user_id: str, tenant_id: str,
                       permissions: List[str]) -> str:
        """创建新会话。"""
        session_id = f"sess_{hashlib.sha256(f'{user_id}{tenant_id}{time.time()}'.encode()).hexdigest()[:24]}"
        self._sessions[session_id] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "permissions": permissions,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "tool_calls": [],
            "context": {},  # 会话上下文（如对话历史）
        }
        return session_id

    def validate_session(self, session_id: str) -> Optional[dict]:
        """验证会话有效性。"""
        session = self._sessions.get(session_id)
        if not session:
            return None

        elapsed = (datetime.now() - session["last_activity"]).total_seconds()
        if elapsed > self.session_timeout:
            del self._sessions[session_id]
            return None

        session["last_activity"] = datetime.now()
        return session

    def log_tool_call(self, session_id: str, tool_name: str, params: dict):
        """记录工具调用（审计）。"""
        session = self._sessions.get(session_id)
        if session:
            session["tool_calls"].append({
                "tool": tool_name,
                "params": str(params)[:200],
                "timestamp": datetime.now().isoformat(),
            })


def demo_session_auth():
    """会话级认证演示。"""
    print("\n" + "=" * 60)
    print("【3】会话级认证 —— Session Token + 多租户")
    print("=" * 60)

    auth = SessionAuthProvider(session_timeout=1800)

    # 创建两个不同租户的会话
    session_a = auth.create_session(
        user_id="alice",
        tenant_id="company_a",
        permissions=["tools:search", "tools:create_ticket"],
    )
    session_b = auth.create_session(
        user_id="bob",
        tenant_id="company_b",
        permissions=["tools:search"],
    )

    print(f"  Alice 会话: {session_a} (company_a)")
    print(f"  Bob 会话: {session_b} (company_b)")

    # 模拟工具调用
    auth.log_tool_call(session_a, "search_knowledge_base",
                       {"query": "RAG 架构", "tenant_id": "company_a"})
    auth.log_tool_call(session_b, "search_knowledge_base",
                       {"query": "RAG 架构", "tenant_id": "company_b"})

    # 验证会话隔离
    session_a_data = auth.validate_session(session_a)
    session_b_data = auth.validate_session(session_b)
    print(f"\n  Alice 会话调用数: {len(session_a_data['tool_calls'])}")
    print(f"  Alice 租户: {session_a_data['tenant_id']}")
    print(f"  Bob 租户: {session_b_data['tenant_id']}")
    print(f"  数据隔离: Alice 看不到 Bob 的数据 ✅")


# ============================================================================
# 第五部分：工具级权限控制
# ============================================================================

@dataclass
class ToolPermission:
    """工具权限定义。"""
    tool_name: str
    required_permission: str
    rate_limit_per_user: int = 100   # 每用户每分钟
    rate_limit_per_session: int = 20 # 每会话每分钟
    sensitive: bool = False          # 是否需要额外审计
    confirmation_required: bool = False  # 是否需要用户确认


class ToolPermissionManager:
    """工具级权限管理器。"""

    def __init__(self):
        self._permissions: Dict[str, ToolPermission] = {}
        self._rate_limits: Dict[str, List[float]] = {}

    def register_tool_permission(self, perm: ToolPermission):
        """注册工具的权限配置。"""
        self._permissions[perm.tool_name] = perm

    def check_access(
        self,
        tool_name: str,
        user_permissions: List[str],
        user_id: str,
        session_id: str,
    ) -> tuple[bool, str]:
        """
        检查用户是否有权调用指定工具。

        Returns:
            (allowed, reason)
        """
        perm = self._permissions.get(tool_name)
        if not perm:
            return False, f"未知工具: {tool_name}"

        # 权限检查
        if perm.required_permission not in user_permissions:
            return False, f"缺少权限: {perm.required_permission}"

        # 速率限制检查
        rate_key = f"{user_id}:{tool_name}"
        now = time.time()
        calls = self._rate_limits.get(rate_key, [])
        calls = [t for t in calls if now - t < 60]  # 保留最近1分钟

        if len(calls) >= perm.rate_limit_per_user:
            return False, f"超过速率限制 ({perm.rate_limit_per_user}/分钟)"

        calls.append(now)
        self._rate_limits[rate_key] = calls

        # 敏感操作需要额外审计
        if perm.sensitive:
            print(f"  [审计] 敏感工具调用: {tool_name} by user={user_id}")

        return True, "OK"


def demo_tool_permissions():
    """工具级权限控制演示。"""
    print("\n" + "=" * 60)
    print("【4】工具级权限 —— 细粒度访问控制")
    print("=" * 60)

    mgr = ToolPermissionManager()

    # 注册工具权限
    mgr.register_tool_permission(ToolPermission(
        tool_name="search_knowledge_base",
        required_permission="tools:read",
        rate_limit_per_user=60,
        sensitive=False,
    ))
    mgr.register_tool_permission(ToolPermission(
        tool_name="create_support_ticket",
        required_permission="tools:write",
        rate_limit_per_user=20,
        sensitive=False,
    ))
    mgr.register_tool_permission(ToolPermission(
        tool_name="delete_document",
        required_permission="tools:admin",
        rate_limit_per_user=5,
        sensitive=True,
        confirmation_required=True,
    ))
    mgr.register_tool_permission(ToolPermission(
        tool_name="export_user_data",
        required_permission="tools:admin",
        rate_limit_per_user=2,
        sensitive=True,
        confirmation_required=True,
    ))

    # 模拟不同权限级别的用户访问
    test_cases = [
        ("search_knowledge_base", ["tools:read"], "user_readonly"),
        ("search_knowledge_base", ["tools:*"], "user_admin"),
        ("create_support_ticket", ["tools:read"], "user_readonly"),
        ("create_support_ticket", ["tools:write"], "user_editor"),
        ("delete_document", ["tools:write"], "user_editor"),
        ("delete_document", ["tools:admin"], "user_admin"),
    ]

    print("\n  权限矩阵测试:")
    for tool, perms, user in test_cases:
        allowed, reason = mgr.check_access(tool, perms, user, f"sess_{user}")
        status = "✅ 通过" if allowed else f"❌ 拒绝 ({reason})"
        print(f"    {user} ({', '.join(perms)}) → {tool}: {status}")

    print("\n  最佳实践:")
    print("    - 默认拒绝（deny by default）")
    print("    - 最小权限原则（least privilege）")
    print("    - 敏感操作二次确认")
    print("    - 定期审查权限配置")
    print("    - 异常行为告警（如短时间大量删除操作）")


# ============================================================================
# 第六部分：完整安全集成示例
# ============================================================================

def print_full_security_integration():
    """展示完整的 MCP Server 安全集成。"""
    print("\n" + "=" * 60)
    print("【5】完整安全集成 —— 多层防护")
    print("=" * 60)

    code = '''
# MCP Server 完整安全集成模板

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# 1. 传输层安全
middleware = [
    Middleware(HTTPSRedirectMiddleware),           # 强制 HTTPS
    Middleware(TrustedHostMiddleware, allowed_hosts=["*.example.com"]),
    Middleware(CORSMiddleware, allow_origins=["https://app.example.com"]),
]

# 2. 认证层
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # 检查多种认证方式
        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")
        session_token = request.headers.get("X-Session-Token", "")

        user_info = None

        if auth_header.startswith("Bearer "):
            user_info = oauth_provider.validate_token(auth_header[7:])
        elif api_key:
            user_info = api_key_provider.validate_api_key(api_key)
        elif session_token:
            user_info = session_provider.validate_session(session_token)

        if not user_info:
            return JSONResponse({"error": "unauthorized"}, 401)

        request.state.user = user_info
        return await call_next(request)

# 3. 授权层（在 Tool 函数内部）
async def delete_document(document_id: str, session: Session):
    # 权限检查
    if not permission_manager.check_access(
        "delete_document",
        session.user_permissions,
        session.user_id,
        session.session_id,
    ):
        raise PermissionError("无权执行此操作")

    # 敏感操作审计
    audit_log.log(
        action="delete_document",
        user=session.user_id,
        target=document_id,
        timestamp=datetime.now().isoformat(),
    )

    # 执行操作
    return await document_service.delete(document_id)

# 4. 审计层
class AuditLogger:
    def log(self, **kwargs):
        # 写入审计日志（数据库/文件/ELK）
        pass

# 5. 输入验证
@server.tool()
async def search(query: str, top_k: int = 5):
    # 参数校验
    if len(query) > 500:
        raise ValueError("查询文本过长")
    if not (1 <= top_k <= 20):
        raise ValueError("top_k 必须在 1-20 之间")
    # SQL 注入防护（如果查询包含 SQL）
    # XSS 防护（如果结果包含 HTML）
    ...
'''
    print(code)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("MCP 安全认证实战")
    print("=" * 60)

    explain_security_model()
    demo_oauth2_auth()
    demo_api_key_auth()
    demo_session_auth()
    demo_tool_permissions()
    print_full_security_integration()

    print("\n[完成] MCP 安全认证演示结束。")
    print("  [提醒] 生产环境务必配置所有安全层，不要跳过任何一层。")
