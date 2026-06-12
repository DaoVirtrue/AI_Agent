#!/usr/bin/env python3
"""
多租户数据隔离验证 (Data Isolation Verification)
==================================================
自动化渗透测试验证多租户边界:

测试:
  1. 跨租户查询测试 (应FAIL - 不能访问其他租户数据)
  2. 权限提升测试 (应FAIL - 不能越权操作)
  3. 隔离边界测试 (应PASS - 同租户正常访问)

定期调度 + 违反告警机制。
"""

import json
import time
import uuid
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass
class IsolationTestCase:
    """隔离测试用例"""
    test_id: str
    name: str
    description: str
    expected_to_pass: bool  # True=应通过(安全的), False=应失败(检测到隔离)

    # 测试参数
    tenant_id: str = ""
    target_tenant_id: str = ""  # 试图访问的租户
    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    # 结果
    result: Optional[TestResult] = None
    actual_output: str = ""
    error_message: str = ""
    execution_time_ms: float = 0.0
    timestamp: str = ""


@dataclass
class PenetrationTestReport:
    """渗透测试报告"""
    report_id: str
    timestamp: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    critical_findings: List[str] = field(default_factory=list)
    test_results: List[IsolationTestCase] = field(default_factory=list)

    @property
    def is_secure(self) -> bool:
        """判断系统是否安全"""
        # 应通过的测试都通过了，应失败的测试都失败了
        for test in self.test_results:
            if test.expected_to_pass and test.result != TestResult.PASS:
                return False
            if not test.expected_to_pass and test.result == TestResult.PASS:
                return False
        return True


# ============================================================================
# 多租户模拟
# ============================================================================

class MultiTenantSimulator:
    """多租户环境模拟器"""

    def __init__(self):
        # 每个租户的数据
        self.tenant_data: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        self.tenant_users: Dict[str, List[str]] = defaultdict(list)
        self.user_roles: Dict[str, List[str]] = defaultdict(list)

        # 初始化: 3个租户
        self._init_tenants()

    def _init_tenants(self):
        """初始化租户"""
        # Tenant A: 金融公司
        self.tenant_data['tenant_a'] = {
            'report_001': {'title': 'Q4 Financial Report', 'content': 'Revenue: $10M', 'sensitivity': 'high'},
            'report_002': {'title': 'Employee Salaries', 'content': 'Confidential data', 'sensitivity': 'critical'},
            'doc_001': {'title': 'Company Policy', 'content': 'Remote work policy', 'sensitivity': 'low'},
        }
        self.tenant_users['tenant_a'] = ['alice', 'bob']
        self.user_roles['alice'] = ['admin', 'tenant_a']
        self.user_roles['bob'] = ['viewer', 'tenant_a']

        # Tenant B: 医疗公司
        self.tenant_data['tenant_b'] = {
            'report_003': {'title': 'Patient Records', 'content': 'PHI data - DO NOT SHARE', 'sensitivity': 'critical'},
            'doc_002': {'title': 'Doctor Schedule', 'content': 'Weekly schedule', 'sensitivity': 'medium'},
        }
        self.tenant_users['tenant_b'] = ['charlie']
        self.user_roles['charlie'] = ['admin', 'tenant_b']

        # Tenant C: 教育机构
        self.tenant_data['tenant_c'] = {
            'doc_003': {'title': 'Student Grades', 'content': 'Semester results', 'sensitivity': 'medium'},
            'doc_004': {'title': 'Course Catalog', 'content': 'Available courses', 'sensitivity': 'low'},
        }
        self.tenant_users['tenant_c'] = ['diana']
        self.user_roles['diana'] = ['viewer', 'tenant_c']

    def query_documents(
        self,
        user_id: str,
        query: str,
        tenant_context: str = "",
    ) -> Dict[str, Any]:
        """模拟文档查询 (带租户隔离)

        Args:
            user_id: 用户ID
            query: 查询文本
            tenant_context: 租户上下文 (模拟API请求中的租户header)

        Returns:
            查询结果
        """
        # 用户所属租户
        user_tenants = [r for r in self.user_roles.get(user_id, []) if r.startswith('tenant_')]

        if not user_tenants:
            return {'success': False, 'error': '用户无租户关联'}

        user_primary_tenant = user_tenants[0]  # 用户的主租户

        # 关键安全检查: 用户只能访问自己的租户数据
        # 如果tenant_context不同且有越权嫌疑
        if tenant_context and tenant_context != user_primary_tenant:
            # 如果用户属于被请求的租户
            if tenant_context in user_tenants:
                effective_tenant = tenant_context
            else:
                return {
                    'success': False,
                    'error': f'访问被拒绝: 用户 {user_id} 不属于租户 {tenant_context}',
                    'isolation_violation': True,
                }
        else:
            effective_tenant = user_primary_tenant

        # 在有效租户内搜索
        tenant_docs = self.tenant_data.get(effective_tenant, {})
        results = []
        for doc_id, doc in tenant_docs.items():
            if query.lower() in doc['title'].lower() or query.lower() in doc['content'].lower():
                results.append({
                    'doc_id': doc_id,
                    'title': doc['title'],
                    'tenant': effective_tenant,
                    'sensitivity': doc['sensitivity'],
                })

        return {
            'success': True,
            'results': results,
            'count': len(results),
            'tenant': effective_tenant,
        }

    def admin_action(self, user_id: str, action: str) -> Dict[str, Any]:
        """模拟管理员操作 (带权限检查)"""
        roles = self.user_roles.get(user_id, [])
        if 'admin' not in roles:
            return {
                'success': False,
                'error': '权限不足: 需要admin角色',
                'privilege_escalation_attempt': True,
            }
        return {'success': True, 'action': action, 'message': f'{action} executed'}


# ============================================================================
# 自动渗透测试
# ============================================================================

class IsolationPenetrationTester:
    """多租户隔离自动化渗透测试器"""

    def __init__(self, target_system: MultiTenantSimulator):
        self.target = target_system
        self.test_cases: List[IsolationTestCase] = []

    def generate_test_suite(self) -> List[IsolationTestCase]:
        """生成测试套件"""
        tests = []

        # === 跨租户查询测试 (应FAIL) ===
        tests.append(IsolationTestCase(
            test_id="ISOLATION_001",
            name="跨租户直接查询 (无权限)",
            description="用户bob(tenant_a)试图直接查询tenant_b的文档",
            expected_to_pass=False,  # 应被拒绝
            tenant_id="tenant_a",
            target_tenant_id="tenant_b",
            action="query_documents",
            parameters={'user_id': 'bob', 'query': 'patient', 'tenant_context': 'tenant_b'},
        ))

        tests.append(IsolationTestCase(
            test_id="ISOLATION_002",
            name="跨租户敏感数据访问",
            description="用户charlie(tenant_b)试图查询tenant_a的高敏感度文档",
            expected_to_pass=False,
            tenant_id="tenant_b",
            target_tenant_id="tenant_a",
            action="query_documents",
            parameters={'user_id': 'charlie', 'query': 'salary', 'tenant_context': 'tenant_a'},
        ))

        # === 权限提升测试 (应FAIL) ===
        tests.append(IsolationTestCase(
            test_id="PRIV_ESC_001",
            name="普通用户执行管理员操作",
            description="viewer角色用户diana试图执行管理员操作",
            expected_to_pass=False,
            tenant_id="tenant_c",
            action="admin_action",
            parameters={'user_id': 'diana', 'action': 'delete_all_data'},
        ))

        tests.append(IsolationTestCase(
            test_id="PRIV_ESC_002",
            name="越权修改其他租户数据",
            description="用户bob(tenant_a viewer)试图修改tenant_a的管理设置",
            expected_to_pass=False,
            tenant_id="tenant_a",
            action="admin_action",
            parameters={'user_id': 'bob', 'action': 'modify_tenant_config'},
        ))

        # === 正常操作测试 (应PASS) ===
        tests.append(IsolationTestCase(
            test_id="NORMAL_001",
            name="同租户正常查询",
            description="用户alice(tenant_a admin)查询自己的文档",
            expected_to_pass=True,
            tenant_id="tenant_a",
            action="query_documents",
            parameters={'user_id': 'alice', 'query': 'report', 'tenant_context': 'tenant_a'},
        ))

        tests.append(IsolationTestCase(
            test_id="NORMAL_002",
            name="管理员在自己的租户执行操作",
            description="用户alice在自己的租户内执行管理操作",
            expected_to_pass=True,
            tenant_id="tenant_a",
            action="admin_action",
            parameters={'user_id': 'alice', 'action': 'update_system_config'},
        ))

        tests.append(IsolationTestCase(
            test_id="NORMAL_003",
            name="用户在自己的租户查询非敏感数据",
            description="用户diana查询自己的课程目录",
            expected_to_pass=True,
            tenant_id="tenant_c",
            action="query_documents",
            parameters={'user_id': 'diana', 'query': 'course', 'tenant_context': 'tenant_c'},
        ))

        # === 边界测试 ===
        tests.append(IsolationTestCase(
            test_id="EDGE_001",
            name="空租户上下文",
            description="用户不提供租户context时的行为",
            expected_to_pass=True,  # 应允许但限制在自己的租户
            tenant_id="tenant_a",
            action="query_documents",
            parameters={'user_id': 'alice', 'query': 'report', 'tenant_context': ''},
        ))

        self.test_cases = tests
        return tests

    def run_all_tests(self) -> PenetrationTestReport:
        """执行所有测试"""
        report = PenetrationTestReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            total_tests=len(self.test_cases),
        )

        for test in self.test_cases:
            test.timestamp = datetime.now().isoformat()
            start = time.time()

            try:
                # 执行测试
                if test.action == "query_documents":
                    result = self.target.query_documents(**test.parameters)
                elif test.action == "admin_action":
                    result = self.target.admin_action(**test.parameters)
                else:
                    result = {'success': False, 'error': f'未知操作: {test.action}'}

                execution_time = (time.time() - start) * 1000
                test.execution_time_ms = execution_time

                # 评估结果
                if test.expected_to_pass:
                    # 期望操作成功
                    test.result = TestResult.PASS if result.get('success') else TestResult.FAIL
                else:
                    # 期望操作失败 (隔离应阻止)
                    test.result = TestResult.PASS if not result.get('success') else TestResult.FAIL

                test.actual_output = json.dumps(result, default=str)[:200]
                test.error_message = result.get('error', '')

            except Exception as e:
                execution_time = (time.time() - start) * 1000
                test.execution_time_ms = execution_time
                test.result = TestResult.ERROR
                test.error_message = str(e)

            report.test_results.append(test)

            # 统计
            if test.result == TestResult.PASS:
                report.passed += 1
            elif test.result == TestResult.FAIL:
                report.failed += 1
                if not test.expected_to_pass and test.result == TestResult.FAIL:
                    # 这是好结果: 隔离阻止了越权
                    pass
                elif test.expected_to_pass and test.result == TestResult.FAIL:
                    # 严重: 正常操作被阻止
                    report.critical_findings.append(
                        f"正常操作被阻止: {test.name} - {test.error_message}"
                    )
            else:
                report.errors += 1
                report.critical_findings.append(f"测试错误: {test.name} - {test.error_message}")

        return report

    def schedule_with_alert(
        self,
        interval_hours: int = 24,
        alert_callback: Optional[callable] = None,
    ) -> None:
        """定期调度测试并在违规时告警

        实际部署中使用cron或调度器调用。
        这里是演示实现。
        """
        print(f"[调度] 每 {interval_hours} 小时执行一次隔离测试")

        report = self.run_all_tests()

        if not report.is_secure:
            alert_msg = (
                f"⚠ 多租户隔离违规! "
                f"测试{report.total_tests}项, "
                f"通过{report.passed}项, "
                f"失败{report.failed}项"
            )
            if alert_callback:
                alert_callback(alert_msg, report)
            else:
                print(f"  {alert_msg}")
        else:
            print(f"  ✓ 隔离测试全部通过 ({report.total_tests} 项)")

        self.print_report(report)

    @staticmethod
    def print_report(report: PenetrationTestReport):
        """打印报告"""
        print(f"\n{'=' * 60}")
        print(f"  渗透测试报告: {report.report_id[:8]}")
        print(f"{'=' * 60}")
        print(f"  时间: {report.timestamp[:19]}")
        print(f"  总计: {report.total_tests} | 通过: {report.passed} | "
              f"失败: {report.failed} | 错误: {report.errors}")
        print(f"  安全状态: {'✓ 安全' if report.is_secure else '🚫 存在漏洞!'}")

        print(f"\n  详细结果:")
        for test in report.test_results:
            icon = {'PASS': '✓', 'FAIL': '✗', 'ERROR': '⚠'}.get(test.result.value, '?')
            expected = '应通过' if test.expected_to_pass else '应阻止'
            print(f"  {icon} [{test.result.value}] {test.name}")
            print(f"       {expected} | 耗时: {test.execution_time_ms:.1f}ms")
            if test.error_message:
                print(f"       消息: {test.error_message[:80]}")

        if report.critical_findings:
            print(f"\n  🚫 关键发现:")
            for finding in report.critical_findings:
                print(f"    - {finding}")

        print(f"{'=' * 60}")


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  多租户数据隔离验证 - 渗透测试演示")
    print("=" * 60)

    # 初始化
    system = MultiTenantSimulator()
    tester = IsolationPenetrationTester(system)

    # 生成测试套件
    tester.generate_test_suite()

    # 执行所有测试
    report = tester.run_all_tests()

    # 打印完整报告
    tester.print_report(report)

    # 告警验证
    print(f"\n[告警] 调度测试 (模拟)...")
    def alert_handler(msg, rpt):
        print(f"  🚨 ALERT: {msg}")

    tester.schedule_with_alert(interval_hours=24, alert_callback=alert_handler)

    print()
    print("=" * 60)
    print("  数据隔离验证演示完成")
    print("=" * 60)
