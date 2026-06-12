#!/usr/bin/env python3
"""
陷阱2: 生产数据意外 (Production Data Surprises)
================================================
症状: 测试环境完美，生产环境各种意外输入
根因: 测试数据不代表真实分布

解决: 生产影子流量 + 数据验证 + 灰度发布
"""

class ProductionDataValidator:
    def validate(self, data: str) -> dict:
        issues = []
        if len(data) > 10000: issues.append("过长输入")
        if '\x00' in data: issues.append("Null字节")
        if data.count('{') != data.count('}'): issues.append("括号不匹配")
        return {'valid': len(issues) == 0, 'issues': issues}

if __name__ == "__main__":
    v = ProductionDataValidator()
    for test in ["Normal query", "\x00Bad", "A" * 20000]:
        result = v.validate(test)
        print(f"'{test[:30]}': {'✓' if result['valid'] else '✗'} {result['issues']}")
