#!/usr/bin/env python3
"""
陷阱2: 工具权限提升 (Tool Permission Escalation)
===================================================
症状: 普通用户通过LLM工具调用执行了管理员操作
根因: 工具调用没有进行独立的权限验证，仅依赖LLM不会调用

解决: 每个工具独立权限检查 + 最小权限原则 + 审计所有工具调用
"""

class ToolPermissionChecker:
    """工具权限检查器"""
    def __init__(self):
        self.tool_permissions = {
            'read_document': {'roles': ['admin', 'developer', 'viewer']},
            'delete_document': {'roles': ['admin']},
            'update_config': {'roles': ['admin']},
            'search': {'roles': ['admin', 'developer', 'viewer']},
            'export_data': {'roles': ['admin', 'developer']},
        }

    def check(self, tool_name: str, user_roles: List[str]) -> bool:
        allowed = self.tool_permissions.get(tool_name, {})
        return any(r in allowed.get('roles', []) for r in user_roles)

    def secure_wrapper(self, tool_fn, tool_name: str, user_roles: List[str], **kwargs):
        if not self.check(tool_name, user_roles):
            return {'error': f'权限不足: {tool_name}', 'access_denied': True}
        return tool_fn(**kwargs)

if __name__ == "__main__":
    checker = ToolPermissionChecker()
    print(f"viewer执行delete_document: {checker.check('delete_document', ['viewer'])}")
    print(f"admin执行delete_document: {checker.check('delete_document', ['admin'])}")
    print(f"developer执行search: {checker.check('search', ['developer'])}")
