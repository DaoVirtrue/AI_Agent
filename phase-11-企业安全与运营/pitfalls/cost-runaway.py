#!/usr/bin/env python3
"""
陷阱3: 成本失控 (Cost Runaway)
===============================
症状: 月底收到惊人的LLM API账单
根因: 无预算限制 + 无成本监控 + 无使用限制

解决: 每日/每月预算 + 自动降级 + 成本归因 + 使用配额
"""

class CostGuard:
    def __init__(self, daily_budget=50.0, monthly_budget=1000.0):
        self.limits = {'daily': daily_budget, 'monthly': monthly_budget}
        self.spent = {'daily': 0.0, 'monthly': 0.0}
        self.per_user_quota = 100  # 每日每用户最大请求数

    def check(self, cost: float, user_id: str) -> bool:
        self.spent['daily'] += cost
        self.spent['monthly'] += cost
        if self.spent['daily'] > self.limits['daily'] * 0.8:
            print(f"⚠ 预算告警: 已用{self.spent['daily']:.0f}/{self.limits['daily']}")
        if self.spent['daily'] > self.limits['daily']:
            return False
        return True

if __name__ == "__main__":
    guard = CostGuard(daily_budget=10)
    for i in range(5):
        ok = guard.check(2.5, f"user_{i}")
        print(f"请求{i+1}: {'允许' if ok else '拒绝'} (已用${guard.spent['daily']:.1f})")
