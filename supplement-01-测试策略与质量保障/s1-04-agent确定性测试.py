#!/usr/bin/env python3
"""
S1-04：Agent 确定性测试 —— 固定种子 / 轨迹验证
===================================================
Agent 的非确定性是测试的最大挑战。本文件提供策略：
1. 固定随机种子 → 可复现的测试
2. 确定性工具链 → 可控的 Agent 行为
3. 轨迹验证 → 验证 Agent 的推理路径
4. AgentTrajectoryTest 框架

依赖：无需额外安装
"""

import random
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================================
# 确定性 Agent 组件
# ============================================================================

class DeterministicLLM:
    """
    确定性 LLM：基于查询内容哈希的固定输出。

    用于测试场景，确保相同输入总是得到相同输出。
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._response_map: Dict[str, str] = {}
        self.call_log: List[Dict] = []

    def add_response(self, query_contains: str, response: str):
        """注册关键词 → 固定回答的映射。"""
        self._response_map[query_contains] = response

    def complete(self, prompt: str, **kwargs) -> str:
        """固定的、可预测的输出。"""
        self.call_log.append({"prompt": prompt[:200], "timestamp": datetime.now()})

        # 查找匹配的固定回答
        for keyword, response in self._response_map.items():
            if keyword in prompt:
                return response

        # 基于哈希的确定但不同的输出
        h = hashlib.sha256(f"{self.seed}:{prompt}".encode()).hexdigest()[:8]
        return f"[确定LLM-{h}] 对 '{prompt[:50]}...' 的回答"


# ============================================================================
# Agent 轨迹定义
# ============================================================================

@dataclass
class TrajectoryStep:
    """
    轨迹中的一个步骤。

    包含：Agent 思考了什么、调用了什么工具、看到了什么结果。
    """
    step_number: int
    thought: str          # Agent 的思考内容
    action: str           # 采取的行动（tool_name 或 "final_answer"）
    action_input: dict    # 行动的参数
    observation: str      # 行动的结果
    expected_next: Optional[str] = None  # 预期的下一步


@dataclass
class ExpectedTrajectory:
    """
    预期的完整轨迹。

    在测试中，我们预先定义 Agent 应该如何执行任务，
    然后与实际执行轨迹进行对比。
    """
    task: str
    steps: List[TrajectoryStep]
    expected_final_answer_contains: List[str] = field(default_factory=list)

    def verify(self, actual_steps: List[TrajectoryStep]) -> Tuple[bool, str]:
        """
        验证实际轨迹是否与预期一致。

        Returns:
            (是否通过, 差异描述)
        """
        if len(actual_steps) < len(self.steps):
            return False, f"步骤数不足: 预期{len(self.steps)}步，实际{len(actual_steps)}步"

        for i, expected in enumerate(self.steps):
            if i >= len(actual_steps):
                return False, f"缺少第{i+1}步"

            actual = actual_steps[i]

            # 检查行动类型是否一致
            if expected.action != actual.action:
                return False, (
                    f"第{i+1}步行动不一致: "
                    f"预期={expected.action}, 实际={actual.action}"
                )

            # 检查行动参数是否一致
            for key, val in expected.action_input.items():
                if key not in actual.action_input:
                    return False, f"第{i+1}步缺少参数: {key}"
                if actual.action_input[key] != val:
                    return False, (
                        f"第{i+1}步参数'{key}'不一致: "
                        f"预期={val}, 实际={actual.action_input[key]}"
                    )

        return True, "轨迹验证通过"


# ============================================================================
# AgentTrajectoryTest 框架
# ============================================================================

class AgentTrajectoryTest:
    """
    Agent 轨迹测试框架。

    使用方式：
    1. 定义任务和预期轨迹
    2. 用确定性 LLM + 确定性工具运行 Agent
    3. 收集实际轨迹
    4. 对比预期与实际
    """

    def __init__(self):
        self.test_cases: List[Dict] = []

    def add_test_case(self, task: str,
                      expected_trajectory: ExpectedTrajectory):
        """添加测试用例。"""
        self.test_cases.append({
            "task": task,
            "expected": expected_trajectory,
        })

    def run_with_agent(self, agent_fn: callable) -> List[Dict]:
        """
        用给定的 Agent 运行所有测试用例。

        agent_fn: (task) → (List[TrajectoryStep], final_answer)
        """
        results = []

        for tc in self.test_cases:
            task = tc["task"]
            expected = tc["expected"]

            # 运行 Agent
            actual_steps, final_answer = agent_fn(task)

            # 验证轨迹
            passed, message = expected.verify(actual_steps)

            # 验证最终回答
            answer_check = True
            if expected.expected_final_answer_contains:
                answer_check = all(
                    kw in final_answer
                    for kw in expected.expected_final_answer_contains
                )

            results.append({
                "task": task,
                "trajectory_pass": passed,
                "trajectory_message": message,
                "answer_pass": answer_check,
                "overall_pass": passed and answer_check,
                "actual_steps_count": len(actual_steps),
                "expected_steps_count": len(expected.steps),
            })

        return results


# ============================================================================
# 演示：客服 Agent 轨迹测试
# ============================================================================

def demo_agent_trajectory_test():
    """演示 Agent 确定性轨迹测试。"""
    print("=" * 60)
    print("Agent 确定性测试 —— 轨迹验证")
    print("=" * 60)

    # 定义预期轨迹：用户询问退款 → Agent 查询政策 → 回答
    expected_trajectory = ExpectedTrajectory(
        task="用户想退一款买了3天的产品",
        steps=[
            TrajectoryStep(
                step_number=1,
                thought="用户想退款，需要先查询退款政策了解条件和流程",
                action="search_knowledge_base",
                action_input={"query": "退款政策 条件 流程"},
                observation="退款政策：购买7天内可全额退款，需保留完整包装。"
                           "退款流程：登录→我的订单→申请退款→填写原因→提交。",
            ),
            TrajectoryStep(
                step_number=2,
                thought="用户购买了3天，在7天期限内，满足退款条件。"
                       "可以告知用户退款流程。",
                action="final_answer",
                action_input={},
                observation="",
            ),
        ],
        expected_final_answer_contains=["退款", "7天", "申请"],
    )

    # 创建测试框架
    test = AgentTrajectoryTest()
    test.add_test_case(
        task="用户咨询退款：购买3天的产品能否退款？",
        expected_trajectory=expected_trajectory,
    )

    print(f"\n  预期轨迹: 2步")
    for step in expected_trajectory.steps:
        print(f"    Step {step.step_number}:")
        print(f"      Thought: {step.thought[:60]}...")
        print(f"      Action: {step.action}({step.action_input})")

    # 模拟 "好" Agent 的实际执行
    def good_agent(task: str):
        """正确执行了预期轨迹的 Agent。"""
        steps = [
            TrajectoryStep(
                step_number=1,
                thought="用户想退款，需要先查询退款政策",
                action="search_knowledge_base",
                action_input={"query": "退款政策 条件 流程"},
                observation="退款政策：购买7天内可全额退款...",
            ),
            TrajectoryStep(
                step_number=2,
                thought="满足退款条件，告知用户流程",
                action="final_answer",
                action_input={},
                observation="",
            ),
        ]
        final_answer = ("您好！根据退款政策，购买7天内可全额退款。"
                        "请在APP中提交退款申请。")
        return steps, final_answer

    # 模拟 "坏" Agent 的实际执行（跳过了检索）
    def bad_agent(task: str):
        """跳过了必要的检索步骤的 Agent。"""
        steps = [
            TrajectoryStep(
                step_number=1,
                thought="我直接回答就行",
                action="final_answer",
                action_input={},
                observation="",
            ),
        ]
        final_answer = "可以退款，请自行操作。"
        return steps, final_answer

    # 运行测试
    print("\n  [测试1] 好 Agent（遵循预期轨迹）")
    results = test.run_with_agent(good_agent)
    for r in results:
        status = "✅" if r["overall_pass"] else "❌"
        print(f"    {status} 轨迹通过: {r['trajectory_pass']} | "
              f"回答通过: {r['answer_pass']} | "
              f"步骤: {r['actual_steps_count']}/{r['expected_steps_count']}")

    print("\n  [测试2] 坏 Agent（跳过检索步骤）")
    results = test.run_with_agent(bad_agent)
    for r in results:
        status = "✅" if r["overall_pass"] else "❌"
        print(f"    {status} 轨迹通过: {r['trajectory_pass']} | "
              f"回答通过: {r['answer_pass']} | "
              f"步骤: {r['actual_steps_count']}/{r['expected_steps_count']}")

    # 确定性 LLM 演示
    print("\n  [演示] 确定性 LLM 的固定输出")
    dllm = DeterministicLLM(seed=42)
    dllm.add_response("退款", "【退款政策】购买7天内可全额退款...")

    r1 = dllm.complete("用户想退款怎么办？")
    r2 = dllm.complete("用户想退款怎么办？")
    print(f"    首次调用: {r1[:60]}...")
    print(f"    再次调用: {r2[:60]}...")
    print(f"    确定性验证: {'✅ 输出相同' if r1 == r2 else '❌ 输出不同'}")
    print(f"    调用日志数: {len(dllm.call_log)}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    demo_agent_trajectory_test()

    print("\n[完成] Agent 确定性测试演示结束。")
    print("  [提示] 生产环境建议:")
    print("    1. 为每种场景定义预期轨迹（退款、查询、投诉...）")
    print("    2. 用确定性 LLM 在 CI 环境中运行轨迹验证")
    print("    3. 新功能上线前必须通过所有轨迹测试")
    print("    4. 如果 LLM 升级，审查并更新预期轨迹")
