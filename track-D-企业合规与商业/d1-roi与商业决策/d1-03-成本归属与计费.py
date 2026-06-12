#!/usr/bin/env python3
"""
D1-03：成本归属与计费
========================
当你的 AI 平台服务多个租户、多个功能、多个模型时，
如何准确地归属成本、内部计费、设置预算告警？

本文件演示：
1. 多租户成本归属模型（按租户/按功能/按模型）
2. 内部计费系统实现
3. 预算告警机制
4. 成本分析报告

依赖：无需额外安装
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


# ============================================================================
# 数据模型
# ============================================================================

class ResourceType(Enum):
    LLM_INPUT_TOKEN = "llm_input_token"
    LLM_OUTPUT_TOKEN = "llm_output_token"
    EMBEDDING_TOKEN = "embedding_token"
    VECTOR_STORAGE_GB = "vector_storage_gb"
    SEARCH_QUERY = "search_query"
    API_CALL = "api_call"


@dataclass
class PricingTier:
    """阶梯定价。"""
    tier_name: str
    min_units: float
    max_units: float      # float('inf') for unlimited
    price_per_unit: float


@dataclass
class ResourcePricing:
    """资源定价配置。"""
    resource_type: ResourceType
    unit_name: str         # "1K tokens", "1 query", "1 GB-month"
    pricing_tiers: List[PricingTier]
    base_rate: float = 0   # 基础费率


@dataclass
class UsageRecord:
    """使用记录。"""
    timestamp: datetime
    tenant_id: str
    feature_id: str        # "knowledge_qa", "agent_chat", "document_analysis"
    model_name: str        # "gpt-4o", "gpt-4o-mini", "deepseek-v3"
    resource_type: ResourceType
    quantity: float        # token数、查询次数、存储量
    cost: float = 0        # 计算后的成本


@dataclass
class TenantBudget:
    """租户预算。"""
    tenant_id: str
    monthly_budget: float          # 月度预算总额
    alert_threshold_pct: float = 0.80  # 告警阈值（80%）
    hard_limit: bool = False       # 是否强制限制
    current_usage: float = 0
    alert_sent: bool = False
    blocked: bool = False


# ============================================================================
# 定价表配置
# ============================================================================

class PricingTable:
    """多模型、多资源类型的定价表。"""

    def __init__(self):
        self.pricing: Dict[str, Dict[ResourceType, ResourcePricing]] = {}
        self._init_default_pricing()

    def _init_default_pricing(self):
        """初始化默认定价表（参考 2026 年市场价）。"""
        # GPT-4o 定价
        self.pricing["gpt-4o"] = {
            ResourceType.LLM_INPUT_TOKEN: ResourcePricing(
                resource_type=ResourceType.LLM_INPUT_TOKEN,
                unit_name="百万输入token",
                base_rate=2.50,
                pricing_tiers=[
                    PricingTier("标准", 0, 1_000_000, 2.50),
                    PricingTier("批量", 1_000_001, float("inf"), 1.25),
                ],
            ),
            ResourceType.LLM_OUTPUT_TOKEN: ResourcePricing(
                resource_type=ResourceType.LLM_OUTPUT_TOKEN,
                unit_name="百万输出token",
                base_rate=10.00,
                pricing_tiers=[
                    PricingTier("标准", 0, 1_000_000, 10.00),
                    PricingTier("批量", 1_000_001, float("inf"), 5.00),
                ],
            ),
        }

        # GPT-4o-mini 定价
        self.pricing["gpt-4o-mini"] = {
            ResourceType.LLM_INPUT_TOKEN: ResourcePricing(
                resource_type=ResourceType.LLM_INPUT_TOKEN,
                unit_name="百万输入token",
                base_rate=0.15,
                pricing_tiers=[PricingTier("标准", 0, float("inf"), 0.15)],
            ),
            ResourceType.LLM_OUTPUT_TOKEN: ResourcePricing(
                resource_type=ResourceType.LLM_OUTPUT_TOKEN,
                unit_name="百万输出token",
                base_rate=0.60,
                pricing_tiers=[PricingTier("标准", 0, float("inf"), 0.60)],
            ),
        }

        # DeepSeek-V3 定价
        self.pricing["deepseek-v3"] = {
            ResourceType.LLM_INPUT_TOKEN: ResourcePricing(
                resource_type=ResourceType.LLM_INPUT_TOKEN,
                unit_name="百万输入token",
                base_rate=0.27,
                pricing_tiers=[PricingTier("标准", 0, float("inf"), 0.27)],
            ),
            ResourceType.LLM_OUTPUT_TOKEN: ResourcePricing(
                resource_type=ResourceType.LLM_OUTPUT_TOKEN,
                unit_name="百万输出token",
                base_rate=1.10,
                pricing_tiers=[PricingTier("标准", 0, float("inf"), 1.10)],
            ),
        }

        # 嵌入模型定价
        self.pricing["text-embedding-3-small"] = {
            ResourceType.EMBEDDING_TOKEN: ResourcePricing(
                resource_type=ResourceType.EMBEDDING_TOKEN,
                unit_name="百万token",
                base_rate=0.02,
                pricing_tiers=[PricingTier("标准", 0, float("inf"), 0.02)],
            ),
        }

        # 基础设施统一定价
        self.pricing["infrastructure"] = {
            ResourceType.VECTOR_STORAGE_GB: ResourcePricing(
                resource_type=ResourceType.VECTOR_STORAGE_GB,
                unit_name="GB-月",
                base_rate=0.50,
                pricing_tiers=[PricingTier("标准", 0, float("inf"), 0.50)],
            ),
            ResourceType.SEARCH_QUERY: ResourcePricing(
                resource_type=ResourceType.SEARCH_QUERY,
                unit_name="千次查询",
                base_rate=0.10,
                pricing_tiers=[PricingTier("标准", 0, float("inf"), 0.10)],
            ),
        }

    def get_price(self, model: str, resource_type: ResourceType,
                  quantity: float) -> float:
        """计算给定模型、资源类型、用量的成本。"""
        model_pricing = self.pricing.get(model, {})
        rp = model_pricing.get(resource_type)
        if not rp:
            return 0

        # 查阶梯定价
        for tier in rp.pricing_tiers:
            if tier.min_units <= quantity <= tier.max_units:
                return quantity * tier.price_per_unit

        return quantity * rp.base_rate


# ============================================================================
# 成本归属引擎
# ============================================================================

class CostAttributionEngine:
    """
    成本归属引擎。

    将每次 LLM/Embedding/基础设施的使用按以下维度归属：
    1. 租户（tenant_id）—— 谁用的
    2. 功能（feature_id）—— 用来做什么
    3. 模型（model_name）—— 用了哪个模型
    """

    def __init__(self, pricing: PricingTable):
        self.pricing = pricing
        self.usage_records: List[UsageRecord] = []
        self.tenant_budgets: Dict[str, TenantBudget] = {}

    def record_usage(
        self,
        tenant_id: str,
        feature_id: str,
        model_name: str,
        resource_type: ResourceType,
        quantity: float,
    ) -> UsageRecord:
        """记录一次使用并计算成本。"""
        cost = self.pricing.get_price(model_name, resource_type, quantity)

        record = UsageRecord(
            timestamp=datetime.now(),
            tenant_id=tenant_id,
            feature_id=feature_id,
            model_name=model_name,
            resource_type=resource_type,
            quantity=quantity,
            cost=cost,
        )
        self.usage_records.append(record)

        # 更新租户预算
        self._update_tenant_usage(tenant_id, cost)

        return record

    def _update_tenant_usage(self, tenant_id: str, cost: float):
        """更新租户的累计使用量并检查预算。"""
        if tenant_id not in self.tenant_budgets:
            return

        budget = self.tenant_budgets[tenant_id]
        budget.current_usage += cost

        # 检查告警阈值
        usage_pct = budget.current_usage / budget.monthly_budget
        if usage_pct >= budget.alert_threshold_pct and not budget.alert_sent:
            budget.alert_sent = True
            print(f"  [预算告警] 租户 {tenant_id}: "
                  f"已使用 {usage_pct:.1%} (${budget.current_usage:.2f} / "
                  f"${budget.monthly_budget:.2f})")

        # 检查硬限制
        if budget.hard_limit and budget.current_usage >= budget.monthly_budget:
            budget.blocked = True
            print(f"  [预算阻断] 租户 {tenant_id}: 预算已耗尽，服务暂停")

    def set_tenant_budget(self, tenant_id: str, monthly_budget: float,
                          alert_pct: float = 0.80, hard_limit: bool = False):
        """设置租户预算。"""
        self.tenant_budgets[tenant_id] = TenantBudget(
            tenant_id=tenant_id,
            monthly_budget=monthly_budget,
            alert_threshold_pct=alert_pct,
            hard_limit=hard_limit,
        )

    # ---- 多维度成本报告 ----

    def get_cost_by_tenant(self, start_date: datetime = None,
                           end_date: datetime = None) -> Dict[str, float]:
        """按租户汇总成本。"""
        costs = defaultdict(float)
        for r in self._filter_by_date(start_date, end_date):
            costs[r.tenant_id] += r.cost
        return dict(costs)

    def get_cost_by_feature(self, start_date: datetime = None,
                            end_date: datetime = None) -> Dict[str, float]:
        """按功能汇总成本。"""
        costs = defaultdict(float)
        for r in self._filter_by_date(start_date, end_date):
            costs[r.feature_id] += r.cost
        return dict(costs)

    def get_cost_by_model(self, start_date: datetime = None,
                          end_date: datetime = None) -> Dict[str, float]:
        """按模型汇总成本。"""
        costs = defaultdict(float)
        for r in self._filter_by_date(start_date, end_date):
            costs[r.model_name] += r.cost
        return dict(costs)

    def get_cost_breakdown(self, tenant_id: str,
                           start_date: datetime = None,
                           end_date: datetime = None) -> Dict[str, dict]:
        """获取某个租户的详细成本分解。"""
        result = {
            "by_feature": defaultdict(float),
            "by_model": defaultdict(float),
            "by_resource": defaultdict(float),
        }
        for r in self._filter_by_date(start_date, end_date):
            if r.tenant_id == tenant_id:
                result["by_feature"][r.feature_id] += r.cost
                result["by_model"][r.model_name] += r.cost
                result["by_resource"][r.resource_type.value] += r.cost
        return {k: dict(v) for k, v in result.items()}

    def _filter_by_date(self, start: datetime, end: datetime) -> List[UsageRecord]:
        """按日期范围过滤记录。"""
        if not start and not end:
            return self.usage_records
        return [
            r for r in self.usage_records
            if (not start or r.timestamp >= start)
            and (not end or r.timestamp <= end)
        ]


# ============================================================================
# 内部计费单生成
# ============================================================================

class BillingGenerator:
    """内部计费单生成器。"""

    def __init__(self, engine: CostAttributionEngine):
        self.engine = engine

    def generate_tenant_invoice(self, tenant_id: str,
                                 billing_month: str = None) -> dict:
        """
        为指定租户生成月度账单。

        Returns:
            账单字典，包含费用明细和汇总
        """
        if not billing_month:
            billing_month = datetime.now().strftime("%Y-%m")

        year, month = map(int, billing_month.split("-"))
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)

        cost_breakdown = self.engine.get_cost_breakdown(tenant_id, start, end)

        total = sum(cost_breakdown["by_feature"].values())

        invoice = {
            "invoice_id": f"INV-{tenant_id}-{billing_month}",
            "tenant_id": tenant_id,
            "billing_period": billing_month,
            "generated_at": datetime.now().isoformat(),
            "line_items": [],
            "subtotal": total,
            "tax": total * 0.06,      # 6% 税
            "total": total * 1.06,
            "currency": "USD",
        }

        # 按功能分行
        for feature, amount in sorted(
            cost_breakdown["by_feature"].items(),
            key=lambda x: x[1], reverse=True
        ):
            invoice["line_items"].append({
                "description": f"功能: {feature}",
                "amount": round(amount, 2),
            })

        # 按模型分行
        for model, amount in sorted(
            cost_breakdown["by_model"].items(),
            key=lambda x: x[1], reverse=True
        ):
            invoice["line_items"].append({
                "description": f"模型: {model}",
                "amount": round(amount, 2),
            })

        return invoice


# ============================================================================
# 预算告警系统
# ============================================================================

class BudgetAlertSystem:
    """
    预算告警系统。

    支持多级告警：
    - 50%: 信息通知
    - 80%: 警告
    - 95%: 严重警告
    - 100%: 紧急（如开启硬限制，则阻断）
    """

    def __init__(self, engine: CostAttributionEngine):
        self.engine = engine
        self.alert_handlers: List[callable] = []

    def register_alert_handler(self, handler: callable):
        """注册告警处理器（可接邮件/短信/Webhook）。"""
        self.alert_handlers.append(handler)

    def send_alert(self, tenant_id: str, level: str, message: str):
        """发送告警。"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "level": level,
            "message": message,
        }
        for handler in self.alert_handlers:
            handler(alert)

    def check_all_budgets(self):
        """检查所有租户的预算状态。"""
        alerts = []
        for tenant_id, budget in self.engine.tenant_budgets.items():
            usage_pct = budget.current_usage / budget.monthly_budget

            if usage_pct >= 1.0:
                level = "critical"
                msg = f"预算已耗尽: {usage_pct:.1%}"
            elif usage_pct >= 0.95:
                level = "severe"
                msg = f"预算严重告警: {usage_pct:.1%}"
            elif usage_pct >= 0.80:
                level = "warning"
                msg = f"预算告警: {usage_pct:.1%}"
            elif usage_pct >= 0.50:
                level = "info"
                msg = f"预算使用过半: {usage_pct:.1%}"
            else:
                continue

            alerts.append((tenant_id, level, msg))

        return alerts


# ============================================================================
# 演示
# ============================================================================

def demo_cost_attribution():
    """演示成本归属与计费系统。"""
    print("=" * 60)
    print("【成本归属与计费系统演示】")
    print("=" * 60)

    # 初始化
    pricing = PricingTable()
    engine = CostAttributionEngine(pricing)

    # 设置租户预算
    engine.set_tenant_budget("company_a", monthly_budget=5000, alert_pct=0.80)
    engine.set_tenant_budget("company_b", monthly_budget=2000, alert_pct=0.80,
                             hard_limit=True)
    engine.set_tenant_budget("company_c", monthly_budget=10000, alert_pct=0.90)

    print("\n  [1] 模拟多租户使用场景...")

    # 模拟使用
    scenarios = [
        # (tenant, feature, model, resource, quantity)
        ("company_a", "knowledge_qa", "gpt-4o",
         ResourceType.LLM_INPUT_TOKEN, 500_000),
        ("company_a", "knowledge_qa", "gpt-4o",
         ResourceType.LLM_OUTPUT_TOKEN, 150_000),
        ("company_a", "agent_chat", "deepseek-v3",
         ResourceType.LLM_INPUT_TOKEN, 1_000_000),
        ("company_b", "knowledge_qa", "gpt-4o-mini",
         ResourceType.LLM_INPUT_TOKEN, 2_000_000),
        ("company_b", "document_analysis", "deepseek-v3",
         ResourceType.LLM_INPUT_TOKEN, 3_000_000),
        ("company_b", "document_analysis", "deepseek-v3",
         ResourceType.LLM_OUTPUT_TOKEN, 4_000_000),  # 大用量触发 budget limit
        ("company_c", "knowledge_qa", "gpt-4o",
         ResourceType.LLM_INPUT_TOKEN, 800_000),
    ]

    for tenant, feature, model, resource, qty in scenarios:
        record = engine.record_usage(tenant, feature, model, resource, qty)
        status = ""
        if tenant in engine.tenant_budgets:
            budget = engine.tenant_budgets[tenant]
            if budget.blocked:
                status = " [已被阻断]"
        print(f"    {tenant}/{feature}/{model}: "
              f"{qty/1000:.0f}K tokens → ${record.cost:.4f}{status}")

    # 成本报告
    print("\n  [2] 按租户成本汇总:")
    for tenant, cost in sorted(engine.get_cost_by_tenant().items()):
        budget = engine.tenant_budgets.get(tenant)
        budget_info = ""
        if budget:
            pct = budget.current_usage / budget.monthly_budget * 100
            budget_info = f" (预算使用率: {pct:.1f}%)"
        print(f"    {tenant}: ${cost:.2f}{budget_info}")

    print("\n  [3] 按功能成本汇总:")
    for feature, cost in sorted(engine.get_cost_by_feature().items()):
        print(f"    {feature}: ${cost:.2f}")

    print("\n  [4] 按模型成本汇总:")
    for model, cost in sorted(engine.get_cost_by_model().items()):
        print(f"    {model}: ${cost:.2f}")

    # 生成账单
    print("\n  [5] 生成月度账单:")
    billing = BillingGenerator(engine)
    invoice = billing.generate_tenant_invoice("company_a")
    print(f"    账单ID: {invoice['invoice_id']}")
    print(f"    期间: {invoice['billing_period']}")
    print(f"    小计: ${invoice['subtotal']:.2f}")
    print(f"    总计: ${invoice['total']:.2f}")
    print(f"    明细项数: {len(invoice['line_items'])}")

    # 预算告警检查
    print("\n  [6] 预算告警检查:")
    alert_system = BudgetAlertSystem(engine)
    alerts = alert_system.check_all_budgets()
    if alerts:
        for tenant, level, msg in alerts:
            print(f"    [{level.upper()}] {tenant}: {msg}")
    else:
        print("    无告警")

    return engine


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("成本归属与计费系统")
    print("=" * 60)

    engine = demo_cost_attribution()

    print("\n[完成] 成本归属与计费演示结束。")
    print("  [提示] 生产环境建议:")
    print("    1. 使用时间序列数据库存储 UsageRecord（如 ClickHouse）")
    print("    2. 异步记录 usage，不阻塞主流程")
    print("    3. 预算检查放在网关层（API Gateway middleware）")
    print("    4. 账单数据定期导出到财务系统")
