#!/usr/bin/env python3
"""
B2-05: 微调成本收益分析 (Fine-tuning Cost-Benefit Analysis)
================================================================
学习目标:
  1. GPU训练成本 vs API调用成本的盈亏平衡计算
  2. 7B vs 14B vs 70B ROI对比
  3. 敏感性分析 (乐观/基准/悲观三种情景)
  4. 图表可视化
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================
# Section 1: 成本模型
# ============================================================

@dataclass
class TrainingCost:
    """训练成本"""
    model_size: str          # "7B" | "14B" | "70B"
    method: str              # "LoRA" | "QLoRA" | "Full"

    # GPU租赁
    gpu_type: str            # "A100-80GB" | "A10-24GB" | "RTX4090"
    gpu_hourly_rate: float   # 每小时GPU租赁费 (USD)
    training_hours: float    # 训练总时长

    # 数据
    dataset_size: int        # 训练样本数
    labeling_cost_per_sample: float  # 每个样本的标注成本 (USD)

    # 人力
    engineer_hours: float    # 工程时间
    engineer_hourly_rate: float = 50.0  # 工程师时薪

    @property
    def gpu_cost(self) -> float:
        return self.gpu_hourly_rate * self.training_hours

    @property
    def data_cost(self) -> float:
        return self.dataset_size * self.labeling_cost_per_sample

    @property
    def labor_cost(self) -> float:
        return self.engineer_hours * self.engineer_hourly_rate

    @property
    def total_cost(self) -> float:
        return self.gpu_cost + self.data_cost + self.labor_cost


@dataclass
class InferenceCost:
    """推理成本"""
    # API调用模式
    api_price_per_1k_tokens: float  # 每千token价格 (USD)
    tokens_per_query: float         # 每次查询的token数 (输入+输出)
    queries_per_month: int          # 每月查询量

    # 自部署模式
    inference_gpu_hourly: float     # 推理GPU每小时
    queries_per_second: int         # GPU吞吐量 (queries/sec)

    @property
    def api_monthly_cost(self) -> float:
        """API模式月度成本"""
        tokens_per_month = self.tokens_per_query * self.queries_per_month
        return tokens_per_month * self.api_price_per_1k_tokens / 1000

    @property
    def self_hosted_monthly_cost(self) -> float:
        """自部署月度成本"""
        seconds_needed = self.queries_per_month / max(self.queries_per_second, 1)
        hours_needed = seconds_needed / 3600
        return hours_needed * self.inference_gpu_hourly


# ============================================================
# Section 2: 盈亏平衡分析
# ============================================================

class BreakEvenAnalyzer:
    """
    盈亏平衡分析器

    回答: "自己微调部署 vs 直接调用API，哪个更划算？"

    核心公式:
    训练成本 / (API月度成本 - 自部署月度成本) = 盈亏平衡月数

    如果 盈亏平衡月数 < 预期使用月数 → 微调部署更划算
    如果 盈亏平衡月数 > 预期使用月数 → API调用更划算
    """

    GPU_RATES = {
        "A100-80GB": 2.50,   # $2.50/hour
        "A10-24GB":  0.75,   # $0.75/hour
        "RTX4090":   1.20,   # $1.20/hour (consumer GPU in cloud)
    }

    API_RATES = {
        "gpt-4o":      {"input": 2.50, "output": 10.00},  # per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet": {"input": 3.00, "output": 15.00},
        "qwen-max":    {"input": 0.50, "output": 2.00},
        "deepseek-v3": {"input": 0.27, "output": 0.80},
    }

    def __init__(self, training: TrainingCost,
                 inference: InferenceCost):
        self.training = training
        self.inference = inference

    def compute_break_even_months(self) -> Dict:
        """
        计算盈亏平衡月数

        返回值:
        - break_even_months: 盈亏平衡需要的月数
        - savings_12months: 12个月的总节省
        - roi_12months: 12个月的投资回报率
        """
        monthly_api = self.inference.api_monthly_cost
        monthly_self = self.inference.self_hosted_monthly_cost
        monthly_savings = monthly_api - monthly_self

        if monthly_savings <= 0:
            return {
                "break_even_months": float("inf"),
                "monthly_api_cost": round(monthly_api, 2),
                "monthly_self_hosted_cost": round(monthly_self, 2),
                "monthly_savings": round(monthly_savings, 2),
                "verdict": "自部署不如API划算",
            }

        break_even = self.training.total_cost / monthly_savings

        # 12个月总节省
        savings_12m = monthly_savings * 12 - self.training.total_cost

        # ROI
        roi = (savings_12m / self.training.total_cost) * 100

        return {
            "break_even_months": round(break_even, 1),
            "monthly_api_cost": round(monthly_api, 2),
            "monthly_self_hosted_cost": round(monthly_self, 2),
            "monthly_savings": round(monthly_savings, 2),
            "training_total_cost": round(self.training.total_cost, 2),
            "savings_12months": round(savings_12m, 2),
            "roi_12months": round(roi, 1),
            "verdict": (
                "强烈推荐微调部署" if break_even < 3 else
                "推荐微调部署" if break_even < 6 else
                "可考虑微调部署" if break_even < 12 else
                "建议使用API"
            ),
        }

    @staticmethod
    def total_cost_breakdown(training: TrainingCost) -> Dict:
        """成本分解"""
        return {
            "gpu_cost": round(training.gpu_cost, 2),
            "gpu_pct": round(training.gpu_cost / max(training.total_cost, 1) * 100, 1),
            "data_cost": round(training.data_cost, 2),
            "data_pct": round(training.data_cost / max(training.total_cost, 1) * 100, 1),
            "labor_cost": round(training.labor_cost, 2),
            "labor_pct": round(training.labor_cost / max(training.total_cost, 1) * 100, 1),
            "total": round(training.total_cost, 2),
        }


# ============================================================
# Section 3: 多模型ROI对比
# ============================================================

class ModelROIComparator:
    """
    7B vs 14B vs 70B 多模型ROI对比

    不同规模模型的权衡:
    - 小模型(7B): 训练快、推理便宜、响应快，但质量可能不足
    - 中模型(14B): 性能和成本的折中点
    - 大模型(70B): 质量最高，但训练和推理成本都高
    """

    @staticmethod
    def create_scenarios() -> List[TrainingCost]:
        """创建不同模型规模的训练成本场景"""
        return [
            TrainingCost(
                model_size="7B",
                method="LoRA",
                gpu_type="A10-24GB",
                gpu_hourly_rate=0.75,
                training_hours=8,
                dataset_size=5000,
                labeling_cost_per_sample=0.10,
                engineer_hours=20,
            ),
            TrainingCost(
                model_size="14B",
                method="QLoRA",
                gpu_type="A100-80GB",
                gpu_hourly_rate=2.50,
                training_hours=12,
                dataset_size=5000,
                labeling_cost_per_sample=0.10,
                engineer_hours=25,
            ),
            TrainingCost(
                model_size="70B",
                method="QLoRA",
                gpu_type="A100-80GB",
                gpu_hourly_rate=2.50,
                training_hours=36,
                dataset_size=10000,
                labeling_cost_per_sample=0.15,
                engineer_hours=40,
            ),
        ]

    @staticmethod
    def compare_roi(
        scenarios: List[TrainingCost],
        queries_per_month: int = 100_000,
        tokens_per_query: float = 500,
    ) -> List[Dict]:
        """
        对比不同模型规模的ROI

        假设:
        - 对比的API是 GPT-4o-mini ($0.15/1M input, $0.60/1M output)
        - 平均每查询500 tokens (300 input + 200 output)
        """
        # API成本基准 (GPT-4o-mini)
        input_price = 0.15 / 1_000_000  # per token
        output_price = 0.60 / 1_000_000

        avg_input_tokens = tokens_per_query * 0.6
        avg_output_tokens = tokens_per_query * 0.4
        avg_cost_per_query = avg_input_tokens * input_price + avg_output_tokens * output_price
        avg_cost_per_1k = avg_cost_per_query * 1000

        results = []
        for scenario in scenarios:
            # 推理吞吐量（模型越大越慢）
            qps_map = {"7B": 50, "14B": 25, "70B": 5}
            qps = qps_map.get(scenario.model_size, 10)

            # 推理GPU配置
            inference_gpu_map = {"7B": "RTX4090", "14B": "A10-24GB", "70B": "A100-80GB"}
            inference_hourly_map = {"RTX4090": 1.20, "A10-24GB": 0.75, "A100-80GB": 2.50}

            gpu_type = inference_gpu_map.get(scenario.model_size, "A10-24GB")
            hourly = inference_hourly_map.get(gpu_type, 1.00)

            inference = InferenceCost(
                api_price_per_1k_tokens=avg_cost_per_1k,
                tokens_per_query=tokens_per_query,
                queries_per_month=queries_per_month,
                inference_gpu_hourly=hourly,
                queries_per_second=qps,
            )

            analyzer = BreakEvenAnalyzer(scenario, inference)
            be_result = analyzer.compute_break_even_months()
            cost_breakdown = analyzer.total_cost_breakdown(scenario)

            results.append({
                "model_size": scenario.model_size,
                "method": scenario.method,
                "training_cost": round(scenario.total_cost, 2),
                "training_hours": scenario.training_hours,
                "inference_qps": qps,
                "monthly_api_cost": be_result["monthly_api_cost"],
                "monthly_self_cost": be_result["monthly_self_hosted_cost"],
                "break_even_months": be_result["break_even_months"],
                "savings_12months": be_result["savings_12months"],
                "roi_12months": be_result["roi_12months"],
                "verdict": be_result["verdict"],
                "cost_breakdown": cost_breakdown,
            })

        return results

    @staticmethod
    def print_comparison_table(results: List[Dict]):
        """打印多模型对比表"""
        print(f"\n  {'模型':<8s} {'方法':<8s} {'训练成本':<10s} "
              f"{'训练时长':<10s} {'推理QPS':<10s} {'月API费':<10s} "
              f"{'月自部署':<10s}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for r in results:
            print(f"  {r['model_size']:<8s} {r['method']:<8s} "
                  f"${r['training_cost']:<9.2f} {r['training_hours']:<9.1f}h "
                  f"{r['inference_qps']:<10d} ${r['monthly_api_cost']:<9.2f} "
                  f"${r['monthly_self_cost']:<9.2f}")

        print(f"\n  {'模型':<8s} {'盈亏平衡':<12s} {'12月节省':<12s} "
              f"{'12月ROI':<10s} {'建议':<s}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*20}")

        for r in results:
            be = f"{r['break_even_months']}月" if r['break_even_months'] != float('inf') else "永不"
            print(f"  {r['model_size']:<8s} {be:<12s} "
                  f"${r['savings_12months']:<11.2f} "
                  f"{r['roi_12months']:<9.1f}% "
                  f"{r['verdict']:<s}")


# ============================================================
# Section 4: 敏感性分析
# ============================================================

class SensitivityAnalyzer:
    """
    敏感性分析 (Sensitivity Analysis)

    分析关键变量的变化如何影响盈亏平衡:

    情景:
    - 乐观(Optimistic): 高质量数据集+经验丰富的团队
    - 基准(Base): 标准数据集+标准团队
    - 悲观(Pessimistic): 数据质量差+需要多轮迭代
    """

    SCENARIOS = {
        "optimistic": {
            "label": "乐观情景",
            "desc": "高质量标注数据, 团队经验丰富, 一次训练成功",
            "training_hours_mult": 0.7,
            "data_cost_mult": 0.5,
            "labor_hours_mult": 0.6,
        },
        "base": {
            "label": "基准情景",
            "desc": "标准数据质量, 标准团队, 正常迭代",
            "training_hours_mult": 1.0,
            "data_cost_mult": 1.0,
            "labor_hours_mult": 1.0,
        },
        "pessimistic": {
            "label": "悲观情景",
            "desc": "数据质量差, 需要多轮迭代, 团队需学习摸索",
            "training_hours_mult": 2.0,
            "data_cost_mult": 2.0,
            "labor_hours_mult": 2.5,
        },
    }

    def __init__(self, base_training: TrainingCost,
                 base_inference: InferenceCost):
        self.base_training = base_training
        self.base_inference = base_inference

    def analyze(self) -> Dict:
        """执行三情景敏感性分析"""
        results = {}

        for scenario_name, params in self.SCENARIOS.items():
            # 创建变体
            variant = TrainingCost(
                model_size=self.base_training.model_size,
                method=self.base_training.method,
                gpu_type=self.base_training.gpu_type,
                gpu_hourly_rate=self.base_training.gpu_hourly_rate,
                training_hours=self.base_training.training_hours * params["training_hours_mult"],
                dataset_size=self.base_training.dataset_size,
                labeling_cost_per_sample=self.base_training.labeling_cost_per_sample * params["data_cost_mult"],
                engineer_hours=self.base_training.engineer_hours * params["labor_hours_mult"],
                engineer_hourly_rate=self.base_training.engineer_hourly_rate,
            )

            analyzer = BreakEvenAnalyzer(variant, self.base_inference)
            be = analyzer.compute_break_even_months()

            results[scenario_name] = {
                "label": params["label"],
                "description": params["desc"],
                "training_cost": round(variant.total_cost, 2),
                "break_even_months": be["break_even_months"],
                "savings_12months": be["savings_12months"],
                "roi_12months": be["roi_12months"],
                "verdict": be["verdict"],
            }

        return results

    @staticmethod
    def print_sensitivity_report(results: Dict):
        """打印敏感性分析报告"""
        print(f"\n  敏感性分析报告")
        print(f"  {'='*50}")
        print(f"  {'情景':<12s} {'训练成本':<12s} {'盈亏平衡':<12s} "
              f"{'12月节省':<12s} {'ROI':<8s}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

        for key, r in results.items():
            be = f"{r['break_even_months']}月" if r['break_even_months'] != float('inf') else '永不'
            print(f"  {r['label']:<12s} ${r['training_cost']:<11.2f} "
                  f"{be:<12s} ${r['savings_12months']:<11.2f} "
                  f"{r['roi_12months']:<7.1f}%")

        print(f"\n  情景说明:")
        for key, r in results.items():
            print(f"    {r['label']}: {r['description']}")


# ============================================================
# Section 5: 图表可视化 (ASCII Chart)
# ============================================================

class ASCIIChatter:
    """ASCII图表生成器"""

    @staticmethod
    def bar_chart(data: Dict[str, float], title: str = "",
                  width: int = 40, max_label_width: int = 15):
        """ASCII柱状图"""
        print(f"\n  {title}")
        print(f"  {'='*60}")

        max_val = max(data.values()) if data else 1
        for label, value in data.items():
            bar_len = int(value / max_val * width)
            bar = "█" * bar_len
            print(f"  {label:<{max_label_width}s} |{bar} {value:.1f}")

    @staticmethod
    def break_even_curve(monthly_savings: float,
                         training_cost: float,
                         months: int = 24):
        """盈亏平衡曲线（ASCII版）"""
        print(f"\n  盈亏平衡曲线 (训练成本=${training_cost:.0f}, "
              f"月节省=${monthly_savings:.0f})")
        print(f"  {'='*60}")

        cumulative_savings = []
        for m in range(months + 1):
            cumulative_savings.append(monthly_savings * m - training_cost)

        # Y轴: 累计节省, X轴: 月份
        max_savings = max(cumulative_savings[-1], training_cost)
        min_savings = -training_cost
        height = 15
        width = months

        grid = [[" " for _ in range(width + 1)] for _ in range(height + 1)]

        # 绘制Y轴和零线
        zero_y = int((-min_savings) / (max_savings - min_savings) * height) if (max_savings - min_savings) > 0 else height
        for y in range(height + 1):
            grid[y][0] = "│"
        for x in range(width + 1):
            if 0 <= zero_y <= height:
                grid[zero_y][x] = "─"

        # 绘制曲线
        for m in range(months + 1):
            val = cumulative_savings[m]
            y = int((val - min_savings) / (max_savings - min_savings) * height) if (max_savings - min_savings) > 0 else height // 2
            y = max(0, min(height, y))
            if 0 <= y <= height and 0 <= m <= width:
                grid[height - y][m] = "*"

        # 标注盈亏平衡点
        be_month_label = ""
        for m in range(1, months + 1):
            if cumulative_savings[m] >= 0 and cumulative_savings[m-1] < 0:
                be_month_label = f" ← 盈亏平衡: {m}月"

        for row in grid:
            print(f"  {''.join(row)}")

        if be_month_label:
            print(f"  {be_month_label}")


# ============================================================
# Section 6: 主流程
# ============================================================

def main():
    """主函数：演示微调成本收益分析全流程"""
    print("=" * 70)
    print("B2-05: 微调成本收益分析 — 完整演示")
    print("=" * 70)

    # 1. 单模型盈亏平衡分析
    print("\n[演示1] 单模型盈亏平衡分析 (7B LoRA)...")

    training = TrainingCost(
        model_size="7B",
        method="LoRA",
        gpu_type="A10-24GB",
        gpu_hourly_rate=0.75,
        training_hours=10,
        dataset_size=5000,
        labeling_cost_per_sample=0.10,
        engineer_hours=20,
    )

    inference = InferenceCost(
        api_price_per_1k_tokens=0.0006,  # GPT-4o-mini级别 (简化)
        tokens_per_query=500,
        queries_per_month=100_000,
        inference_gpu_hourly=1.20,
        queries_per_second=50,
    )

    analyzer = BreakEvenAnalyzer(training, inference)
    result = analyzer.compute_break_even_months()

    print(f"  训练总成本: ${result['training_total_cost']:.2f}")
    print(f"  API月度成本: ${result['monthly_api_cost']:.2f}")
    print(f"  自部署月度成本: ${result['monthly_self_hosted_cost']:.2f}")
    print(f"  月度节省: ${result['monthly_savings']:.2f}")
    print(f"  盈亏平衡: {result['break_even_months']} 个月")
    print(f"  12月ROI: {result['roi_12months']}%")
    print(f"  结论: {result['verdict']}")

    # 2. 成本分解
    print("\n[演示2] 训练成本分解...")
    breakdown = analyzer.total_cost_breakdown(training)
    print(f"  GPU成本: ${breakdown['gpu_cost']:.2f} ({breakdown['gpu_pct']}%)")
    print(f"  数据成本: ${breakdown['data_cost']:.2f} ({breakdown['data_pct']}%)")
    print(f"  人力成本: ${breakdown['labor_cost']:.2f} ({breakdown['labor_pct']}%)")
    print(f"  总成本: ${breakdown['total']:.2f}")

    # ASCII饼图
    ASCIIChatter.bar_chart(
        {
            "GPU": breakdown["gpu_cost"],
            "数据标注": breakdown["data_cost"],
            "人力": breakdown["labor_cost"],
        },
        title="训练成本构成",
        width=30,
    )

    # 3. 多模型ROI对比
    print("\n[演示3] 7B vs 14B vs 70B ROI对比...")
    scenarios = ModelROIComparator.create_scenarios()
    comparison = ModelROIComparator.compare_roi(
        scenarios,
        queries_per_month=100_000,
        tokens_per_query=500,
    )
    ModelROIComparator.print_comparison_table(comparison)

    # 4. 敏感性分析
    print("\n[演示4] 敏感性分析 (乐观/基准/悲观)...")
    sensitivity = SensitivityAnalyzer(training, inference)
    sens_results = sensitivity.analyze()
    SensitivityAnalyzer.print_sensitivity_report(sens_results)

    # 5. 盈亏平衡曲线
    print("\n[演示5] 盈亏平衡曲线...")
    ASCIIChatter.break_even_curve(
        monthly_savings=result["monthly_savings"],
        training_cost=result["training_total_cost"],
        months=24,
    )

    # 6. 查询量敏感度
    print("\n[演示6] 查询量对盈亏平衡的影响...")
    query_volumes = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
    be_by_volume = {}
    for qpm in query_volumes:
        inf_variant = InferenceCost(
            api_price_per_1k_tokens=0.0006,
            tokens_per_query=500,
            queries_per_month=qpm,
            inference_gpu_hourly=1.20,
            queries_per_second=50,
        )
        analyzer_v = BreakEvenAnalyzer(training, inf_variant)
        be = analyzer_v.compute_break_even_months()
        be_by_volume[f"{qpm/1000:.0f}K/月"] = be["break_even_months"]

    ASCIIChatter.bar_chart(
        be_by_volume,
        title="查询量 vs 盈亏平衡月数 (越低越好)",
        width=35,
        max_label_width=8,
    )

    print("\n" + "=" * 70)
    print("B2-05 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
