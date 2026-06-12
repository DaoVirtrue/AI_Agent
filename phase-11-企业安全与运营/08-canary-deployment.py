#!/usr/bin/env python3
"""
金丝雀部署 (Canary Deployment)
===============================
渐进式发布新版本技能:

阶段:
  5%  → 60分钟观察 → 健康检查
  25% → 120分钟观察 → 健康检查
  50% → 180分钟观察 → 健康检查
  100% → 手动确认

自动回滚触发条件:
  - 错误率 > 2x 基线
  - P95延迟 > 3x 基线

MD5哈希分桶确定用户路由。
每分钟健康检查。
"""

import time
import hashlib
import json
import threading
from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict


# ============================================================================
# 核心数据结构
# ============================================================================

class DeployPhase(Enum):
    """部署阶段"""
    PREPARING = "preparing"
    CANARY_5 = "canary_5_percent"
    CANARY_25 = "canary_25_percent"
    CANARY_50 = "canary_50_percent"
    FULL_100 = "full_100_percent"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class DeployPhaseConfig:
    """部署阶段配置"""
    phase: DeployPhase
    traffic_percent: float
    observation_minutes: int
    manual_approval_required: bool = False


@dataclass
class HealthMetrics:
    """健康指标"""
    error_rate: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    success_rate: float = 1.0

    # 基线指标 (部署前的正常运行值)
    baseline_error_rate: float = 0.01
    baseline_p95_latency: float = 500.0

    def is_healthy(self) -> Tuple[bool, str]:
        """检查是否健康 (与基线比较)"""
        issues = []

        # 错误率检查
        if self.error_rate > self.baseline_error_rate * 2.0:
            issues.append(
                f"错误率({self.error_rate:.1%}) > 2x 基线({self.baseline_error_rate:.1%})"
            )

        # 延迟检查
        if self.p95_latency_ms > self.baseline_p95_latency * 3.0:
            issues.append(
                f"P95延迟({self.p95_latency_ms:.0f}ms) > 3x 基线({self.baseline_p95_latency:.0f}ms)"
            )

        # 整体判断
        if len(issues) >= 2:
            return False, "多个健康指标异常: " + "; ".join(issues)
        elif len(issues) == 1:
            return False, issues[0]

        return True, "健康"


@dataclass
class DeploymentState:
    """部署状态"""
    skill_name: str
    new_version: str
    current_phase: DeployPhase
    started_at: str
    phase_started_at: str
    traffic_routed_to_new: int = 0
    traffic_routed_to_old: int = 0
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    rollback_reason: str = ""


# ============================================================================
# 金丝雀部署器
# ============================================================================

class SkillCanaryDeployer:
    """技能金丝雀部署器"""

    # 部署阶段配置
    PHASES_CONFIG = [
        DeployPhaseConfig(DeployPhase.CANARY_5, 0.05, 60, False),
        DeployPhaseConfig(DeployPhase.CANARY_25, 0.25, 120, False),
        DeployPhaseConfig(DeployPhase.CANARY_50, 0.50, 180, False),
        DeployPhaseConfig(DeployPhase.FULL_100, 1.00, 0, True),  # 手动确认
    ]

    def __init__(self, health_check_interval_seconds: int = 60):
        self.health_interval = health_check_interval_seconds

        # 当前部署状态
        self.deployment: Optional[DeploymentState] = None

        # 健康指标滑动窗口
        self._recent_errors: deque = deque(maxlen=1000)
        self._recent_latencies: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

        # 部署历史
        self.deployment_history: List[Dict] = []

        # 回调
        self.on_phase_change: Optional[Callable] = None
        self.on_health_alert: Optional[Callable] = None
        self.on_rollback: Optional[Callable] = None

    # ========================================================================
    # 哈希分桶
    # ========================================================================

    @staticmethod
    def get_traffic_bucket(user_id: str, skill_name: str, version: str) -> int:
        """MD5哈希分桶

        确定性: 同一个user_id总是被分配到同一个桶
        平衡性: 均匀分布在0-9999之间
        """
        key = f"{user_id}:{skill_name}:{version}"
        hash_hex = hashlib.md5(key.encode()).hexdigest()
        return int(hash_hex, 16) % 10000

    @staticmethod
    def should_route_to_new_version(
        user_id: str,
        skill_name: str,
        new_version: str,
        traffic_percent: float,
    ) -> bool:
        """判断该用户是否应路由到新版本

        Args:
            user_id: 用户ID
            skill_name: 技能名称
            new_version: 新版本号
            traffic_percent: 当前阶段的流量百分比 (0-1)

        Returns:
            True=路由到新版本
        """
        bucket = SkillCanaryDeployer.get_traffic_bucket(user_id, skill_name, new_version)
        threshold = int(traffic_percent * 10000)
        return bucket < threshold

    # ========================================================================
    # 部署流程
    # ========================================================================

    def start_deployment(
        self,
        skill_name: str,
        new_version: str,
        baseline_error_rate: float = 0.01,
        baseline_p95_latency: float = 500.0,
    ) -> DeploymentState:
        """启动金丝雀部署

        Args:
            skill_name: 技能名称
            new_version: 新版本号
            baseline_error_rate: 基线错误率
            baseline_p95_latency: 基线P95延迟

        Returns:
            初始部署状态
        """
        now = datetime.now().isoformat()

        self.deployment = DeploymentState(
            skill_name=skill_name,
            new_version=new_version,
            current_phase=DeployPhase.PREPARING,
            started_at=now,
            phase_started_at=now,
        )

        # 设置基线
        self.baseline_error_rate = baseline_error_rate
        self.baseline_p95_latency = baseline_p95_latency

        print(f"[部署] 金丝雀部署启动: {skill_name} v{new_version}")
        print(f"[部署] 基线: 错误率={baseline_error_rate:.1%}, P95={baseline_p95_latency}ms")

        # 自动推进到第一个阶段
        self._advance_phase()

        return self.deployment

    def _advance_phase(self):
        """推进到下一个部署阶段"""
        if not self.deployment:
            return

        current_phase = self.deployment.current_phase

        # 找到下一个阶段
        phase_index = -1
        for i, config in enumerate(self.PHASES_CONFIG):
            if config.phase == current_phase:
                phase_index = i
                break

        next_index = phase_index + 1
        if next_index >= len(self.PHASES_CONFIG):
            # 已经完成了所有阶段
            self.deployment.current_phase = DeployPhase.COMPLETED
            print(f"[部署] 部署完成!")
            return

        # 如果有手动确认要求
        current_config = self.PHASES_CONFIG[next_index - 1] if next_index > 0 else None
        if current_config and current_config.manual_approval_required:
            print(f"[部署] 需要手动确认才能推进到 {self.PHASES_CONFIG[next_index].phase.value}")
            return

        # 推进
        self.deployment.current_phase = self.PHASES_CONFIG[next_index].phase
        self.deployment.phase_started_at = datetime.now().isoformat()

        traffic = self.PHASES_CONFIG[next_index].traffic_percent
        observation = self.PHASES_CONFIG[next_index].observation_minutes

        print(f"[部署] 阶段变更: → {self.deployment.current_phase.value} "
              f"(流量={traffic:.0%}, 观察={observation}分钟)")

        if self.on_phase_change:
            self.on_phase_change(self.deployment)

        # 如果需要观察，启动观察线程
        if observation > 0:
            threading.Thread(
                target=self._observe_phase,
                args=(observation,),
                daemon=True,
            ).start()

    def _observe_phase(self, duration_minutes: int):
        """观察当前阶段"""
        end_time = time.time() + duration_minutes * 60
        last_health_check = time.time()

        while time.time() < end_time:
            if time.time() - last_health_check >= self.health_interval:
                self._perform_health_check()
                last_health_check = time.time()

                # 检查是否健康
                metrics = self._get_current_health()
                is_healthy, reason = metrics.is_healthy()

                if not is_healthy:
                    if self.deployment and self.deployment.current_phase not in (
                        DeployPhase.ROLLING_BACK, DeployPhase.ROLLED_BACK
                    ):
                        self._trigger_rollback(reason)
                    return

            time.sleep(min(5, self.health_interval))

        # 观察期结束，健康检查通过
        print(f"[部署] {self.deployment.current_phase.value} 观察期通过，推进到下一阶段")
        self.deployment.health_checks_passed += 1
        self._advance_phase()

    def _perform_health_check(self):
        """执行健康检查"""
        metrics = self._get_current_health()
        is_healthy, reason = metrics.is_healthy()

        status = "✓" if is_healthy else "✗"
        print(f"  [健康检查] {status} 新版本: err={metrics.error_rate:.3%}, "
              f"P95={metrics.p95_latency_ms:.0f}ms | {reason}")

        if not is_healthy:
            self.deployment.health_checks_failed += 1
            if self.on_health_alert:
                self.on_health_alert(metrics, reason)
        else:
            self.deployment.health_checks_passed += 1

    def _get_current_health(self) -> HealthMetrics:
        """获取当前健康指标"""
        with self._lock:
            errors = list(self._recent_errors)
            latencies = list(self._recent_latencies)

        if not errors:
            error_rate = 0.0
        else:
            error_rate = sum(errors) / len(errors)

        if latencies:
            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[len(latencies_sorted) // 2]
            p95_idx = int(len(latencies_sorted) * 0.95)
            p99_idx = int(len(latencies_sorted) * 0.99)
            p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]
            p99 = latencies_sorted[min(p99_idx, len(latencies_sorted) - 1)]
        else:
            p50 = p95 = p99 = 0.0

        return HealthMetrics(
            error_rate=error_rate,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            baseline_error_rate=getattr(self, 'baseline_error_rate', 0.01),
            baseline_p95_latency=getattr(self, 'baseline_p95_latency', 500.0),
        )

    # ========================================================================
    # 记录指标
    # ========================================================================

    def record_request(
        self,
        user_id: str,
        skill_name: str,
        version: str,
        is_error: bool,
        latency_ms: float,
    ):
        """记录一次请求的指标

        在金丝雀部署中使用，用于追踪新版本的性能。

        Args:
            user_id: 用户ID
            skill_name: 技能名称
            version: 技能版本
            is_error: 是否是错误
            latency_ms: 延迟(毫秒)
        """
        if not self.deployment or self.deployment.current_phase in (
            DeployPhase.ROLLED_BACK, DeployPhase.COMPLETED
        ):
            return

        # 只收集新版本的指标
        if version == self.deployment.new_version:
            with self._lock:
                self._recent_errors.append(1.0 if is_error else 0.0)
                self._recent_latencies.append(latency_ms)

                if version == self.deployment.new_version:
                    self.deployment.traffic_routed_to_new += 1
                else:
                    self.deployment.traffic_routed_to_old += 1

    # ========================================================================
    # 回滚
    # ========================================================================

    def _trigger_rollback(self, reason: str):
        """触发自动回滚"""
        if not self.deployment:
            return

        print(f"[回滚] 自动回滚触发! 原因: {reason}")
        print(f"[回滚] 从 {self.deployment.current_phase.value} 回滚到旧版本")

        self.deployment.current_phase = DeployPhase.ROLLING_BACK
        self.deployment.rollback_reason = reason

        # 设置流量全部回到旧版本
        time.sleep(1)  # 模拟回滚操作

        self.deployment.current_phase = DeployPhase.ROLLED_BACK

        # 记录
        self.deployment_history.append({
            'skill': self.deployment.skill_name,
            'version': self.deployment.new_version,
            'result': 'rolled_back',
            'phase_reached': self.deployment.current_phase.value,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        })

        if self.on_rollback:
            self.on_rollback(self.deployment, reason)

        print(f"[回滚] 回滚完成")

    def manual_approve(self):
        """手动批准推进 (用于100%阶段)"""
        if not self.deployment:
            print("[部署] 无活动部署")
            return

        if self.deployment.current_phase in (DeployPhase.FULL_100, DeployPhase.COMPLETED):
            self._advance_phase()

    # ========================================================================
    # 查询
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """获取当前部署状态"""
        if not self.deployment:
            return {'status': 'no_active_deployment'}

        return {
            'skill': self.deployment.skill_name,
            'version': self.deployment.new_version,
            'phase': self.deployment.current_phase.value,
            'started': self.deployment.started_at,
            'phase_started': self.deployment.phase_started_at,
            'traffic_to_new': self.deployment.traffic_routed_to_new,
            'traffic_to_old': self.deployment.traffic_routed_to_old,
            'health_checks_passed': self.deployment.health_checks_passed,
            'health_checks_failed': self.deployment.health_checks_failed,
            'rollback_reason': self.deployment.rollback_reason,
        }


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  金丝雀部署 - 完整演示")
    print("=" * 60)

    deployer = SkillCanaryDeployer(health_check_interval_seconds=5)

    # 启动部署
    print("\n[启动部署]")
    deployer.start_deployment(
        skill_name="customer_support",
        new_version="v2.1.0",
        baseline_error_rate=0.02,
        baseline_p95_latency=800.0,
    )

    # 模拟流量 (演示分桶)
    print("\n[流量模拟] MD5哈希分桶")
    test_users = [f"user_{i:04d}" for i in range(20)]
    for user_id in test_users:
        should_route = SkillCanaryDeployer.should_route_to_new_version(
            user_id, "customer_support", "v2.1.0", 0.05  # 5%阶段
        )
        print(f"  {user_id} → {'新版本' if should_route else '旧版本'}")

    # 模拟健康数据
    print("\n[模拟健康数据]")
    import random
    random.seed(42)

    for i in range(30):
        is_ok = random.random() < 0.94  # 6%错误率 (2% * 3x over baseline)
        latency = random.expovariate(1.0 / 600)  # 平均600ms
        deployer.record_request(
            f"user_{i:04d}", "customer_support", "v2.1.0",
            is_error=not is_ok, latency_ms=latency * 1000,
        )

    # 健康检查
    health = deployer._get_current_health()
    is_healthy, reason = health.is_healthy()
    print(f"  错误率: {health.error_rate:.2%}")
    print(f"  P95延迟: {health.p95_latency_ms:.0f}ms")
    print(f"  健康: {is_healthy} ({reason})")

    # 状态查询
    print(f"\n[部署状态]")
    status = deployer.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 模拟自动回滚场景
    print(f"\n{'─' * 40}")
    print("模拟高错误率回滚场景")

    deployer2 = SkillCanaryDeployer(health_check_interval_seconds=3)
    deployer2.start_deployment(
        skill_name="code_review",
        new_version="v3.0.0",
        baseline_error_rate=0.01,
        baseline_p95_latency=300.0,
    )

    # 注入高错误率数据
    for i in range(20):
        deployer2.record_request(
            f"user_{i:04d}", "code_review", "v3.0.0",
            is_error=(i < 4),  # 前4个请求错误 = 20%错误率
            latency_ms=random.expovariate(1.0 / 0.5) * 1000,
        )

    health2 = deployer2._get_current_health()
    is_healthy2, reason2 = health2.is_healthy()
    print(f"  错误率: {health2.error_rate:.2%}")
    print(f"  健康: {is_healthy2} ({reason2})")

    if not is_healthy2:
        deployer2._trigger_rollback(reason2)

    print()
    print("=" * 60)
    print("  金丝雀部署演示完成")
    print("=" * 60)
