#!/usr/bin/env python3
"""
陷阱1: 隐蔽式安全 (Security Through Obscurity)
=================================================
症状: "我们的系统很安全，因为攻击者不知道内部实现"
根因: 依赖隐蔽性而非真正的安全机制

解决: 纵深防御 + 独立安全审计 + 定期渗透测试
"""

import hashlib, json, re
from typing import List, Dict, Any

class SecurityAuditor:
    """安全审计器 - 检测隐蔽式安全的问题"""

    CHECKS = [
        ("硬编码密钥", r'(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*["\'][^"\']+["\']'),
        ("弱哈希算法", r'(MD5|SHA1)\s*\('),
        ("明文传输", r'http://'),
        ("缺少认证检查", r'def\s+\w+\(.*\).*:\s*\n\s*#.*TODO.*auth'),
        ("调试模式开启", r'(DEBUG|debug)\s*=\s*(True|1)'),
    ]

    def audit_code(self, code: str) -> Dict[str, Any]:
        findings = []
        for name, pattern in self.CHECKS:
            matches = re.findall(pattern, code)
            if matches:
                findings.append({'issue': name, 'matches': len(matches), 'severity': 'HIGH'})
        return {'total_issues': len(findings), 'findings': findings, 'recommendation': '不要依赖隐蔽性，实施显式安全控制'}

if __name__ == "__main__":
    auditor = SecurityAuditor()
    test_code = 'API_KEY = "sk-1234567890abcdef"\nDEBUG = True\nresponse = requests.get("http://api.example.com")'
    result = auditor.audit_code(test_code)
    print(f"发现 {result['total_issues']} 个安全问题:")
    for f in result['findings']:
        print(f"  - {f['issue']} [{f['severity']}]")
