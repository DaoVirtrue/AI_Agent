#!/usr/bin/env python3
"""
D1-01：AI 功能 ROI 计算器
=============================
企业引入 AI 功能前，必须回答的核心问题：
"投入多少钱，能省多少钱，多久回本？"

本计算器覆盖三个维度：
1. 成本侧：LLM + Embedding + 基础设施成本
2. 收益侧A：人力替代节省（Labor Replacement Savings）
3. 收益侧B：准确性提升带来的收入影响（Accuracy Revenue Impact）

依赖：无需额外安装（纯标准库）
"""

import json
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class LLMCostConfig:
    """LLM 成本配置。"""
    model_name: str = "gpt-4o"
    input_price_per_1m_tokens: float = 2.50    # 美元/百万输入token
    output_price_per_1m_tokens: float = 10.00   # 美元/百万输出token
    avg_input_tokens_per_query: int = 500
    avg_output_tokens_per_query: int = 300
    cache_hit_rate: float = 0.30               # 缓存命中率（节省输入token）


@dataclass
class EmbeddingCostConfig:
    """嵌入模型成本配置。"""
    model_name: str = "text-embedding-3-small"
    price_per_1m_tokens: float = 0.02          # 美元/百万token
    avg_tokens_per_doc: int = 800
    docs_ingested_per_day: int = 100
    reindex_full_every_n_days: int = 30         # 全量重索引周期


@dataclass
class InfrastructureCostConfig:
    """基础设施成本配置。"""
    vector_db_monthly: float = 200     # 向量数据库月费
    search_engine_monthly: float = 150 # 搜索引擎月费
    api_gateway_monthly: float = 100   # API 网关月费
    monitoring_monthly: float = 100    # 监控月费
    compute_monthly: float = 300       # 计算资源月费
    storage_monthly: float = 50        # 存储月费


@dataclass
class LaborCostConfig:
    """人力成本配置。"""
    avg_annual_salary_per_employee: float = 150000  # CNY
    employees_replaced: float = 3.0                  # 被替代的人力（FTE）
    replacement_ratio: float = 0.40                  # 替代比例（AI 替代 40% 工作量）
    retraining_cost_per_employee: float = 5000      # 每人再培训成本
    transition_period_months: int = 3                # 过渡期


@dataclass
class AccuracyImpactConfig:
    """准确率提升带来的收入影响。"""
    current_accuracy: float = 0.85                   # 当前准确率
    target_accuracy: float = 0.95                    # 目标准确率
    error_cost_per_incident: float = 500             # 每次错误的成本（CNY）
    queries_per_day: int = 5000                      # 日查询量
    revenue_per_correct_answer: float = 2.0          # 每次正确回答的收益
    customer_satisfaction_impact: float = 0.05       # 满意度提升 → 收入提升比例


@dataclass
class ROIResult:
    """ROI 计算结果。"""
    # 成本
    monthly_llm_cost: float = 0
    monthly_embedding_cost: float = 0
    monthly_infrastructure_cost: float = 0
    monthly_labor_cost: float = 0
    total_monthly_cost: float = 0
    total_annual_cost: float = 0

    # 节省
    monthly_labor_savings: float = 0
    monthly_accuracy_savings: float = 0
    total_monthly_savings: float = 0
    total_annual_savings: float = 0

    # 指标
    monthly_net_benefit: float = 0
    annual_net_benefit: float = 0
    roi_percentage: float = 0
    payback_months: float = 0
    break_even_queries_per_day: float = 0


# ============================================================================
# ROI 计算器
# ============================================================================

class AIROICalculator:
    """
    AI 功能 ROI 计算器。

    计算公式：
    总成本 = LLM成本 + 嵌入成本 + 基础设施成本 + 人力成本
    总节省 = 人力替代节省 + 准确率提升收益
    ROI = (总节省 - 总成本) / 总成本 × 100%
    """

    def __init__(
        self,
        llm_config: LLMCostConfig = None,
        embed_config: EmbeddingCostConfig = None,
        infra_config: InfrastructureCostConfig = None,
        labor_config: LaborCostConfig = None,
        accuracy_config: AccuracyImpactConfig = None,
    ):
        self.llm = llm_config or LLMCostConfig()
        self.embed = embed_config or EmbeddingCostConfig()
        self.infra = infra_config or InfrastructureCostConfig()
        self.labor = labor_config or LaborCostConfig()
        self.accuracy = accuracy_config or AccuracyImpactConfig()

    # ---- LLM 成本计算 ----

    def calc_llm_cost_per_query(self) -> float:
        """计算单次查询的 LLM 成本（美元）。"""
        # 有效输入token = 输入token × (1 - 缓存命中率)
        effective_input = self.llm.avg_input_tokens_per_query * (
            1 - self.llm.cache_hit_rate
        )
        input_cost = (effective_input / 1_000_000) * self.llm.input_price_per_1m_tokens
        output_cost = (
            self.llm.avg_output_tokens_per_query / 1_000_000
        ) * self.llm.output_price_per_1m_tokens
        return input_cost + output_cost

    def calc_monthly_llm_cost(self, queries_per_day: int) -> float:
        """计算月 LLM 成本（美元）。"""
        return self.calc_llm_cost_per_query() * queries_per_day * 30

    # ---- 嵌入成本计算 ----

    def calc_embedding_cost_per_doc(self) -> float:
        """计算单篇文档的嵌入成本（美元）。"""
        return (
            self.embed.avg_tokens_per_doc / 1_000_000
        ) * self.embed.price_per_1m_tokens

    def calc_monthly_embedding_cost(self) -> float:
        """计算月嵌入成本（美元）。"""
        # 每日新增嵌入
        daily_new = (
            self.embed.docs_ingested_per_day
            * self.calc_embedding_cost_per_doc()
        )
        # 定期全量重索引的日均摊
        daily_reindex = (
            self.embed.docs_ingested_per_day
            * self.calc_embedding_cost_per_doc()
        ) / self.embed.reindex_full_every_n_days
        return (daily_new + daily_reindex) * 30

    # ---- 基础设施成本 ----

    def calc_monthly_infrastructure_cost(self) -> float:
        """计算月基础设施成本（美元）。"""
        return (
            self.infra.vector_db_monthly
            + self.infra.search_engine_monthly
            + self.infra.api_gateway_monthly
            + self.infra.monitoring_monthly
            + self.infra.compute_monthly
            + self.infra.storage_monthly
        )

    # ---- 总成本 ----

    def calc_total_monthly_cost(self, queries_per_day: int) -> float:
        """计算月总成本（美元）。"""
        return (
            self.calc_monthly_llm_cost(queries_per_day)
            + self.calc_monthly_embedding_cost()
            + self.calc_monthly_infrastructure_cost()
            + (self.labor.avg_annual_salary_per_employee / 12)
            * self.labor.employees_replaced
            * (1 - self.labor.replacement_ratio)
        )

    # ---- 人力节省 ----

    def calc_monthly_labor_savings(self) -> float:
        """计算月人力节省（CNY → 转为美元按 7.2 汇率）。"""
        saved_per_employee = (
            self.labor.avg_annual_salary_per_employee
            * self.labor.replacement_ratio
        )
        total_savings = saved_per_employee * self.labor.employees_replaced
        return total_savings / 12  # 月节省

    # ---- 准确率提升收益 ----

    def calc_monthly_accuracy_savings(self, queries_per_day: int) -> float:
        """计算月准确率提升带来的收益（CNY → USD）。"""
        current_errors = int(
            queries_per_day * (1 - self.accuracy.current_accuracy)
        )
        target_errors = int(
            queries_per_day * (1 - self.accuracy.target_accuracy)
        )
        errors_reduced = current_errors - target_errors

        # 减少错误节省的成本
        error_savings = errors_reduced * self.accuracy.error_cost_per_incident

        # 满意度提升带来的收入增长
        satisfaction_revenue = (
            queries_per_day
            * self.accuracy.revenue_per_correct_answer
            * self.accuracy.current_accuracy
            * self.accuracy.customer_satisfaction_impact
        )

        return (error_savings + satisfaction_revenue) * 30

    # ---- 综合计算 ----

    def calculate(self, queries_per_day: int = 5000) -> ROIResult:
        """执行完整的 ROI 计算。"""
        result = ROIResult()

        # 成本
        result.monthly_llm_cost = self.calc_monthly_llm_cost(queries_per_day)
        result.monthly_embedding_cost = self.calc_monthly_embedding_cost()
        result.monthly_infrastructure_cost = self.calc_monthly_infrastructure_cost()
        result.monthly_labor_cost = (
            (self.labor.avg_annual_salary_per_employee / 12)
            * self.labor.employees_replaced
            * (1 - self.labor.replacement_ratio)
        ) / 7.2  # CNY → USD

        result.total_monthly_cost = (
            result.monthly_llm_cost
            + result.monthly_embedding_cost
            + result.monthly_infrastructure_cost
            + result.monthly_labor_cost
        )
        result.total_annual_cost = result.total_monthly_cost * 12

        # 节省（CNY → USD）
        CNY_TO_USD = 7.2
        result.monthly_labor_savings = (
            self.calc_monthly_labor_savings() / CNY_TO_USD
        )
        result.monthly_accuracy_savings = (
            self.calc_monthly_accuracy_savings(queries_per_day) / CNY_TO_USD
        )

        result.total_monthly_savings = (
            result.monthly_labor_savings
            + result.monthly_accuracy_savings
        )
        result.total_annual_savings = result.total_monthly_savings * 12

        # ROI
        result.monthly_net_benefit = (
            result.total_monthly_savings - result.total_monthly_cost
        )
        result.annual_net_benefit = result.monthly_net_benefit * 12

        if result.total_annual_cost > 0:
            result.roi_percentage = (
                result.annual_net_benefit / result.total_annual_cost * 100
            )
        else:
            result.roi_percentage = float("inf")

        # 投资回收期
        # 初始投资 = 过渡期人力成本 + 培训成本 + 3个月基础设施
        initial_investment = (
            self.labor.retraining_cost_per_employee
            * self.labor.employees_replaced
            / CNY_TO_USD
            + result.total_monthly_cost
            * self.labor.transition_period_months
        )

        if result.monthly_net_benefit > 0:
            result.payback_months = (
                initial_investment / result.monthly_net_benefit
            )
        else:
            result.payback_months = float("inf")

        # 盈亏平衡点（每日查询量）
        # 解方程：成本 = 节省
        if self.calc_llm_cost_per_query() > 0:
            # 固定月成本（不含 LLM 变量）
            fixed_monthly = (
                result.monthly_embedding_cost
                + result.monthly_infrastructure_cost
                + result.monthly_labor_cost
            )
            savings_per_query = result.monthly_accuracy_savings / queries_per_day
            cost_per_query = self.calc_llm_cost_per_query() * 30

            if savings_per_query > cost_per_query:
                # 即使0查询，也能盈利？（不太可能）
                result.break_even_queries_per_day = 0
            else:
                result.break_even_queries_per_day = (
                    fixed_monthly / (savings_per_query - cost_per_query)
                )
        else:
            result.break_even_queries_per_day = 0

        return result


# ============================================================================
# 报告生成
# ============================================================================

def print_roi_report(result: ROIResult, calculator: AIROICalculator):
    """打印格式化的 ROI 报告。"""
    print("\n" + "=" * 60)
    print("               AI 功能 ROI 分析报告")
    print("=" * 60)
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # 成本明细
    print("  ┌─────────────────────────────────────────┐")
    print("  │           月度成本明细 (USD)             │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │ LLM API 调用      │ ${result.monthly_llm_cost:>10,.2f}         │")
    print(f"  │ 嵌入模型           │ ${result.monthly_embedding_cost:>10,.2f}         │")
    print(f"  │ 基础设施           │ ${result.monthly_infrastructure_cost:>10,.2f}         │")
    print(f"  │ 人力(保留部分)     │ ${result.monthly_labor_cost:>10,.2f}         │")
    print(f"  │────────────────────┼────────────────────│")
    print(f"  │ 月度总成本         │ ${result.total_monthly_cost:>10,.2f}         │")
    print(f"  │ 年度总成本         │ ${result.total_annual_cost:>10,.2f}         │")
    print("  └─────────────────────────────────────────┘")

    # 节省明细
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │           月度节省明细 (USD)             │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │ 人力替代节省       │ ${result.monthly_labor_savings:>10,.2f}         │")
    print(f"  │ 准确率提升收益     │ ${result.monthly_accuracy_savings:>10,.2f}         │")
    print(f"  │────────────────────┼────────────────────│")
    print(f"  │ 月度总节省         │ ${result.total_monthly_savings:>10,.2f}         │")
    print(f"  │ 年度总节省         │ ${result.total_annual_savings:>10,.2f}         │")
    print("  └─────────────────────────────────────────┘")

    # 关键指标
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │              关键指标                    │")
    print("  ├─────────────────────────────────────────┤")
    net_label = "净收益" if result.monthly_net_benefit > 0 else "净亏损"
    print(f"  │ 月度{net_label}        │ ${result.monthly_net_benefit:>10,.2f}         │")
    print(f"  │ 年度{net_label}        │ ${result.annual_net_benefit:>10,.2f}         │")

    roi_str = f"{result.roi_percentage:.1f}%"
    if result.roi_percentage > 100:
        roi_str += " 🟢 优秀"
    elif result.roi_percentage > 0:
        roi_str += " 🟡 正向"
    else:
        roi_str += " 🔴 亏损"
    print(f"  │ 年化 ROI            │ {roi_str:>22} │")

    if result.payback_months == float("inf"):
        pb_str = "永不回本"
    else:
        pb_str = f"{result.payback_months:.1f} 个月"
    print(f"  │ 投资回收期          │ {pb_str:>22} │")

    be_str = f"{result.break_even_queries_per_day:,.0f} 次/天"
    print(f"  │ 盈亏平衡点          │ {be_str:>22} │")

    # 单次查询成本
    cost_per_q = calculator.calc_llm_cost_per_query()
    print(f"  │ 单次查询 LLM 成本   │ ${cost_per_q:>10,.4f}         │")
    print("  └─────────────────────────────────────────┘")

    # 结论
    print("\n  【结论】")
    if result.annual_net_benefit > 0 and result.payback_months < 12:
        print(f"  ✅ 项目可行。预计 {result.payback_months:.1f} 个月回本，"
              f"年化 ROI {result.roi_percentage:.1f}%。建议推进。")
    elif result.annual_net_benefit > 0:
        print(f"  ⚠️ 项目可行但回本周期较长（{result.payback_months:.1f} 个月）。"
              f"建议优化成本结构或寻找更多应用场景。")
    else:
        print(f"  ❌ 项目当前不可行。月度净亏损 ${abs(result.monthly_net_benefit):,.2f}。"
              f"建议：降低模型成本 / 提升准确率 / 提高替代比例。")


# ============================================================================
# 灵敏度分析
# ============================================================================

def sensitivity_analysis(calculator: AIROICalculator):
    """
    灵敏度分析：关键参数变化对 ROI 的影响。

    分析变量：
    1. 查询量变化（±50%）
    2. LLM 模型切换（GPT-4o → GPT-4o-mini → Claude → DeepSeek）
    3. 缓存命中率变化
    4. 准确率变化
    """
    print("\n" + "=" * 60)
    print("              灵敏度分析")
    print("=" * 60)

    # 1. 查询量变化
    print("\n  【场景1】查询量对月成本的影响")
    query_levels = [1000, 3000, 5000, 10000, 50000, 100000]
    print(f"  {'日查询量':>12} | {'月LLM成本':>12} | {'月总成本':>12} | {'年ROI':>10}")
    print("  " + "-" * 52)
    for q in query_levels:
        result = calculator.calculate(queries_per_day=q)
        print(f"  {q:>10,} | ${result.monthly_llm_cost:>10,.2f} | "
              f"${result.total_monthly_cost:>10,.2f} | {result.roi_percentage:>9.1f}%")

    # 2. LLM 模型切换
    print("\n  【场景2】LLM 模型对月成本的影响（5000次/天）")
    models = [
        ("GPT-4o", 2.50, 10.00),
        ("GPT-4o-mini", 0.15, 0.60),
        ("Claude Sonnet 4", 3.00, 15.00),
        ("DeepSeek-V3", 0.27, 1.10),
        ("Qwen-Max", 0.40, 1.60),
    ]
    print(f"  {'模型':<18} | {'月LLM成本':>12} | {'年化ROI':>10}")
    print("  " + "-" * 44)
    for name, in_price, out_price in models:
        calc = AIROICalculator(
            llm_config=LLMCostConfig(
                model_name=name,
                input_price_per_1m_tokens=in_price,
                output_price_per_1m_tokens=out_price,
            )
        )
        result = calc.calculate(queries_per_day=5000)
        print(f"  {name:<18} | ${result.monthly_llm_cost:>10,.2f} | "
              f"{result.roi_percentage:>9.1f}%")

    # 3. 缓存命中率影响
    print("\n  【场景3】缓存命中率对 LLM 成本的影响（GPT-4o, 5000次/天）")
    cache_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    print(f"  {'缓存命中率':>12} | {'月LLM成本':>12} | {'节省比例':>10}")
    print("  " + "-" * 38)
    base_config = LLMCostConfig()
    for rate in cache_rates:
        calc = AIROICalculator(
            llm_config=LLMCostConfig(cache_hit_rate=rate)
        )
        result = calc.calculate(queries_per_day=5000)
        base_result = AIROICalculator().calculate(queries_per_day=5000)
        saving_pct = (1 - result.monthly_llm_cost / base_result.monthly_llm_cost) * 100
        print(f"  {rate:>10.0%} | ${result.monthly_llm_cost:>10,.2f} | "
              f"{saving_pct:>9.1f}%")


# ============================================================================
# 生成 CSV 报告
# ============================================================================

def export_csv_report(result: ROIResult, filepath: str = "./roi_report.csv"):
    """导出 ROI 结果为 CSV 文件。"""
    import csv
    from pathlib import Path

    Path(filepath).parent.mkdir(exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "值 (USD)", "备注"])
        writer.writerow(["月LLM成本", f"${result.monthly_llm_cost:,.2f}",
                         "包括输入+输出token"])
        writer.writerow(["月嵌入成本", f"${result.monthly_embedding_cost:,.2f}",
                         "每日新增嵌入+定期重索引"])
        writer.writerow(["月基础设施成本", f"${result.monthly_infrastructure_cost:,.2f}",
                         "向量库+ES+网关+监控+计算+存储"])
        writer.writerow(["月人力节省", f"${result.monthly_labor_savings:,.2f}",
                         "替代工作量对应的薪资节省"])
        writer.writerow(["月准确率提升收益", f"${result.monthly_accuracy_savings:,.2f}",
                         "减少错误+客户满意度提升"])
        writer.writerow(["月净收益", f"${result.monthly_net_benefit:,.2f}", ""])
        writer.writerow(["年化ROI", f"{result.roi_percentage:.1f}%", ""])
        writer.writerow(["投资回收期", f"{result.payback_months:.1f}月", ""])

    print(f"\n  [导出] ROI 报告已保存到: {filepath}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("AI 功能 ROI 计算器")
    print("=" * 60)
    print("基于当前配置计算 AI 项目的投资回报率。\n")

    # 默认配置
    calculator = AIROICalculator()

    # 执行计算
    result = calculator.calculate(queries_per_day=5000)

    # 打印报告
    print_roi_report(result, calculator)

    # 灵敏度分析
    sensitivity_analysis(calculator)

    # 导出 CSV
    export_csv_report(result)

    print("\n[完成] ROI 分析结束。")
    print("  提示：修改 AIROICalculator 的配置参数以适应你的实际情况。")
    print("  关键参数：queries_per_day, llm_config.model_name,")
    print("           labor.employees_replaced, accuracy.target_accuracy")
