#!/usr/bin/env python3
"""
陷阱5: 监控告警疲劳 (Monitoring Alert Fatigue)
==================================================
症状: 运维团队忽略告警，真正的故障被遗漏
根因:
  1. 告警过多、过于频繁
  2. 低价值告警 (自愈的、非关键的)
  3. 告警缺少优先级和上下文
  4. 没有告警抑制和聚合

解决方案:
  - 告警分级 (P0-P4) + 优先级路由
  - 告警抑制 (1小时内部重复合并)
  - 告警聚合 (同类型合并为一个通知)
  - 告警静默期 (维护窗口)
  - 告警升级 (无人响应时升级到更高优先级)
"""

import time
import json
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque


class AlertSeverity(Enum):
    """告警严重级别"""
    P0_CRITICAL = "P0"  # 服务完全中断
    P1_HIGH = "P1"      # 核心功能受损
    P2_MEDIUM = "P2"    # 非核心功能异常
    P3_LOW = "P3"       # 性能下降
    P4_INFO = "P4"      # 信息通知


@dataclass
class Alert:
    """告警"""
    alert_id: str
    name: str
    severity: AlertSeverity
    description: str
    source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[str] = None

    def fingerprint(self) -> str:
        """计算告警指纹 (用于去重和聚合)"""
        import hashlib
        key = f"{self.name}:{self.source}:{sorted(self.labels.items())}"
        return hashlib.md5(key.encode()).hexdigest()[:16]


@dataclass
class AlertManager:
    """告警管理器 - 减少告警疲劳"""

    # 配置
    suppress_duration_minutes: int = 60  # 相同告警60分钟内不重复发送
    aggregation_window_minutes: int = 5  # 5分钟内的同类告警聚合
    max_alerts_per_window: int = 10      # 每个窗口最多10条告警

    # 状态
    sent_alerts: deque = field(default_factory=lambda: deque(maxlen=1000))
    suppressed_alerts: Dict[str, datetime] = field(default_factory=dict)
    aggregated_alerts: Dict[str, List[Alert]] = field(default_factory=lambda: defaultdict(list))

    def process_alert(self, alert: Alert) -> Dict[str, Any]:
        """处理告警 (可能抑制、聚合或发送)

        Returns:
            处理结果
        """
        fp = alert.fingerprint()
        now = datetime.now()

        # 检查1: 是否在抑制期?
        if fp in self.suppressed_alerts:
            last_sent = self.suppressed_alerts[fp]
            if (now - last_sent).total_seconds() < self.suppress_duration_minutes * 60:
                return {
                    'action': 'suppressed',
                    'reason': f'在{self.suppress_duration_minutes}分钟内已发送过',
                    'last_sent': last_sent.isoformat(),
                }

        # 检查2: 是否应该聚合?
        self.aggregated_alerts[fp].append(alert)
        # 清理过期聚合
        self._cleanup_aggregations(now)

        # 更新抑制记录
        self.suppressed_alerts[fp] = now

        # 检查聚合组的大小
        if len(self.aggregated_alerts[fp]) > 1:
            return {
                'action': 'aggregated',
                'count': len(self.aggregated_alerts[fp]),
                'reason': '同类告警已聚合',
            }

        # 发送告警
        self.sent_alerts.append(alert)
        return {
            'action': 'sent',
            'severity': alert.severity.value,
            'notification_channel': self._get_notification_channel(alert.severity),
        }

    def _cleanup_aggregations(self, now: datetime) -> None:
        """清理过期的聚合组"""
        cutoff = now - timedelta(minutes=self.aggregation_window_minutes)
        expired = []
        for fp, alerts in self.aggregated_alerts.items():
            alerts[:] = [a for a in alerts
                        if datetime.fromisoformat(a.timestamp) > cutoff]
            if not alerts:
                expired.append(fp)
        for fp in expired:
            del self.aggregated_alerts[fp]

    def _get_notification_channel(self, severity: AlertSeverity) -> List[str]:
        """根据严重级别确定通知渠道"""
        channels = {
            AlertSeverity.P0_CRITICAL: ['pagerduty', 'phone_call', 'slack'],
            AlertSeverity.P1_HIGH: ['pagerduty', 'slack'],
            AlertSeverity.P2_MEDIUM: ['slack', 'email'],
            AlertSeverity.P3_LOW: ['slack'],
            AlertSeverity.P4_INFO: ['email'],
        }
        return channels.get(severity, ['email'])

    def get_fatigue_score(self) -> Dict[str, Any]:
        """计算告警疲劳度"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        # 过去1小时和24小时的告警数
        last_hour = sum(1 for a in self.sent_alerts
                       if datetime.fromisoformat(a.timestamp) > hour_ago)
        last_day = sum(1 for a in self.sent_alerts
                      if datetime.fromisoformat(a.timestamp) > day_ago)

        # 疲劳度评级
        if last_hour > 50:
            fatigue = 'CRITICAL'
        elif last_hour > 20:
            fatigue = 'HIGH'
        elif last_hour > 5:
            fatigue = 'MODERATE'
        else:
            fatigue = 'LOW'

        return {
            'fatigue_level': fatigue,
            'alerts_last_hour': last_hour,
            'alerts_last_24h': last_day,
            'alerts_per_hour_avg': round(last_day / max(1, 24), 1),
            'suppressed_count': len(self.suppressed_alerts),
            'recommendation': self._get_fatigue_recommendation(fatigue),
        }

    def _get_fatigue_recommendation(self, fatigue: str) -> str:
        """获取降低疲劳度的建议"""
        recommendations = {
            'CRITICAL': '紧急: 检查告警规则，移除噪声告警，增加抑制规则',
            'HIGH': '建议: 审查P3/P4级别告警，合并重复规则',
            'MODERATE': '注意: 监控趋势，定期审查告警有效性',
            'LOW': '健康: 告警量在合理范围内',
        }
        return recommendations.get(fatigue, '')


def create_alert_rules_best_practices() -> Dict[str, Any]:
    """创建告警规则最佳实践配置"""
    return {
        'symptom_based': {
            'description': '基于症状而非原因设置告警',
            'good': 'CPU使用率>90%持续5分钟',
            'bad': '某个pod重启',
        },
        'actionable': {
            'description': '每个告警必须有可执行的操作',
            'good': '包含runbook链接和操作步骤',
            'bad': '仅通知"某个指标异常"',
        },
        'severity_triage': {
            'P0': '服务完全不可用 - 立即响应 (5分钟)',
            'P1': '核心功能受损 - 15分钟内响应',
            'P2': '非核心功能异常 - 1小时内响应',
            'P3': '性能下降 - 工作日响应',
            'P4': '信息通知 - 无需响应',
        },
        'suppression_rules': [
            '维护窗口静默',
            '已知问题的重复告警抑制',
            '自愈事件 (自动恢复) 不告警',
            '依赖服务故障不重复告警',
        ],
    }


SOLUTION_CHECKLIST = """
告警疲劳避免清单:

□ 1. 每个告警都有明确的严重级别 (P0-P4)
□ 2. 告警必须有runbook链接
□ 3. 相同告警在60分钟内不重复通知
□ 4. 同类告警在5分钟内聚合为一个通知
□ 5. 维护窗口自动静默
□ 6. 每周审查告警有效性
□ 7. 告警升级: 15分钟无人响应→升级
□ 8. 告警必须可操作 (actionable)
□ 9. 区分告警和通知 (alert vs notification)
□ 10. 定期删除不再有效的告警规则
"""


if __name__ == "__main__":
    print("=" * 60)
    print("  告警疲劳管理 - 演示")
    print("=" * 60)

    manager = AlertManager(
        suppress_duration_minutes=60,
        aggregation_window_minutes=5,
    )

    print("\n[1] 正常告警流程")
    for i in range(3):
        alert = Alert(
            alert_id=f"alert_{i:03d}",
            name=f"High Latency - API {i+1}",
            severity=AlertSeverity.P2_MEDIUM,
            description=f"P95延迟超过5秒 - API {i+1}",
            source="prometheus",
            labels={"service": f"api_{i+1}", "region": "us-east-1"},
        )

        result = manager.process_alert(alert)
        print(f"  {alert.alert_id}: action={result['action']}, "
              f"channel={result.get('notification_channel', [])}")

    # 模拟重复告警 (应被抑制)
    print("\n[2] 重复告警抑制")
    for i in range(5):
        alert = Alert(
            alert_id=f"repeat_{i}",
            name="High Latency - API 1",  # 相同名称
            severity=AlertSeverity.P2_MEDIUM,
            description="P95延迟超过5秒",
            source="prometheus",
            labels={"service": "api_1", "region": "us-east-1"},
        )

        result = manager.process_alert(alert)
        print(f"  {alert.alert_id}: action={result['action']}")

    # 不同严重级别的通知渠道
    print("\n[3] 通知渠道路由")
    for severity in AlertSeverity:
        alert = Alert(
            alert_id=f"sev_{severity.value}",
            name=f"Test - {severity.value}",
            severity=severity,
            description="测试告警",
        )
        channels = manager._get_notification_channel(severity)
        print(f"  {severity.value}: {', '.join(channels)}")

    # 获取疲劳度
    print("\n[4] 告警疲劳度评估")
    fatigue = manager.get_fatigue_score()
    print(f"  疲劳等级: {fatigue['fatigue_level']}")
    print(f"  过去1小时: {fatigue['alerts_last_hour']} 条")
    print(f"  过去24小时: {fatigue['alerts_last_24h']} 条")
    print(f"  抑制数: {fatigue['suppressed_count']}")
    print(f"  建议: {fatigue['recommendation']}")

    # 最佳实践
    print("\n[5] 告警规则最佳实践")
    practices = create_alert_rules_best_practices()
    for key, value in practices.items():
        print(f"  {key}: {value.get('description', value)}")

    print(f"\n{SOLUTION_CHECKLIST}")
