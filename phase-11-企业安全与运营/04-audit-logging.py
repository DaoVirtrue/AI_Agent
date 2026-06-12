#!/usr/bin/env python3
"""
审计日志系统 (Audit Logging System)
=====================================
企业级审计日志，具备:
  - 结构化审计日志schema (JSON)
  - 不可变存储 (append-only)
  - 合规保留 (compliance retention policies)
  - 防篡改检测 (tamper detection via hash chain)
  - SIEM集成接口
  - 高量日志采样 (log sampling for high volume)
"""

import json
import os
import time
import hashlib
import shutil
import uuid
from typing import List, Dict, Optional, Any, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


# ============================================================================
# 数据结构
# ============================================================================

class AuditEventType(Enum):
    """审计事件类型"""
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"
    TOOL_EXECUTED = "tool.executed"
    DATA_READ = "data.read"
    DATA_MODIFIED = "data.modified"
    DATA_EXPORTED = "data.exported"
    CONFIG_CHANGED = "config.changed"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    SECURITY_ALERT = "security.alert"
    SYSTEM_ERROR = "system.error"


class ComplianceStandard(Enum):
    """合规标准"""
    SOC2 = "SOC2"
    ISO27001 = "ISO27001"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI-DSS"


@dataclass
class RetentionPolicy:
    """保留策略"""
    standard: ComplianceStandard
    retention_days: int
    min_required_fields: List[str]
    encryption_required: bool = True
    tamper_proof_required: bool = True


@dataclass
class AuditEntry:
    """单条审计日志条目"""
    event_id: str
    event_type: AuditEventType
    timestamp: str
    actor_id: str
    actor_ip: str = ""
    resource: str = ""
    action: str = ""
    result: str = ""  # 'success', 'failure', 'error'
    details: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    trace_id: str = ""

    # 完整性
    previous_hash: str = ""
    current_hash: str = ""

    def to_json(self) -> str:
        """序列化为JSON"""
        data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'actor_id': self.actor_id,
            'actor_ip': self.actor_ip,
            'resource': self.resource,
            'action': self.action,
            'result': self.result,
            'details': self.details,
            'session_id': self.session_id,
            'trace_id': self.trace_id,
            'previous_hash': self.previous_hash,
            'current_hash': self.current_hash,
        }
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'AuditEntry':
        """从JSON反序列化"""
        data = json.loads(json_str)
        return cls(
            event_id=data['event_id'],
            event_type=AuditEventType(data['event_type']),
            timestamp=data['timestamp'],
            actor_id=data['actor_id'],
            actor_ip=data.get('actor_ip', ''),
            resource=data.get('resource', ''),
            action=data.get('action', ''),
            result=data.get('result', ''),
            details=data.get('details', {}),
            session_id=data.get('session_id', ''),
            trace_id=data.get('trace_id', ''),
            previous_hash=data.get('previous_hash', ''),
            current_hash=data.get('current_hash', ''),
        )

    def compute_hash(self, previous_hash: str = "") -> str:
        """计算日志条目的哈希 (用于防篡改链)"""
        content = (
            f"{self.event_id}|{self.timestamp}|{self.actor_id}|"
            f"{self.resource}|{self.action}|{self.result}|"
            f"{json.dumps(self.details, sort_keys=True)}|"
            f"{previous_hash}"
        )
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


# ============================================================================
# 审计日志存储
# ============================================================================

class AuditLogStore:
    """不可变审计日志存储

    使用追加写入 + 哈希链实现防篡改。
    支持按时间范围、事件类型、用户等查询。
    """

    # 保留策略 (按合规标准)
    RETENTION_POLICIES = {
        ComplianceStandard.SOC2: RetentionPolicy(ComplianceStandard.SOC2, 365,
                                                   ['event_id', 'timestamp', 'actor_id', 'event_type', 'result']),
        ComplianceStandard.GDPR: RetentionPolicy(ComplianceStandard.GDPR, 730,
                                                  ['event_id', 'timestamp', 'actor_id', 'event_type', 'result']),
        ComplianceStandard.HIPAA: RetentionPolicy(ComplianceStandard.HIPAA, 2190,  # 6年
                                                   ['event_id', 'timestamp', 'actor_id', 'event_type', 'result', 'resource']),
        ComplianceStandard.PCI_DSS: RetentionPolicy(ComplianceStandard.PCI_DSS, 365,
                                                     ['event_id', 'timestamp', 'actor_id', 'event_type']),
        ComplianceStandard.ISO27001: RetentionPolicy(ComplianceStandard.ISO27001, 1095,
                                                      ['event_id', 'timestamp', 'actor_id', 'event_type']),
    }

    def __init__(
        self,
        log_directory: str = "./audit_logs",
        compliance_standard: ComplianceStandard = ComplianceStandard.SOC2,
        max_log_file_size_mb: int = 100,
        sampling_enabled: bool = False,
        sampling_rate: float = 0.1,  # 采样率10%
    ):
        self.log_dir = log_directory
        self.compliance = compliance_standard
        self.max_file_size = max_log_file_size_mb * 1024 * 1024  # 转字节
        self.sampling_enabled = sampling_enabled
        self.sampling_rate = sampling_rate

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 当前日志文件
        self._current_file: Optional[str] = None
        self._entry_count_in_file: int = 0
        self._last_hash: str = "0" * 64  # 初始哈希
        self._total_entries: int = 0
        self._sample_counter: int = 0

        # 初始化
        self._init_log_file()

    def _init_log_file(self):
        """初始化或找到当前日志文件"""
        today = datetime.now().strftime("%Y%m%d")
        existing = sorted([
            f for f in os.listdir(self.log_dir)
            if f.startswith(f"audit_{today}") and f.endswith(".log")
        ])

        if existing:
            self._current_file = os.path.join(self.log_dir, existing[-1])
            # 读取最后一个条目的哈希
            self._recompute_last_hash()
        else:
            self._rotate_file()

    def _rotate_file(self):
        """轮转日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{timestamp}_{uuid.uuid4().hex[:8]}.log"
        self._current_file = os.path.join(self.log_dir, filename)
        self._entry_count_in_file = 0

    def _recompute_last_hash(self):
        """从现有文件重新计算最后的哈希值"""
        if not self._current_file or not os.path.exists(self._current_file):
            self._last_hash = "0" * 64
            return

        last_entry = None
        with open(self._current_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    last_entry = line

        if last_entry:
            try:
                entry_data = json.loads(last_entry)
                self._last_hash = entry_data.get('current_hash', "0" * 64)
            except json.JSONDecodeError:
                self._last_hash = "0" * 64
        else:
            self._last_hash = "0" * 64

    def append(self, entry: AuditEntry) -> str:
        """追加一条审计日志

        Args:
            entry: 审计条目

        Returns:
            条目的event_id
        """
        # 生成ID
        if not entry.event_id:
            entry.event_id = str(uuid.uuid4())

        if not entry.timestamp:
            entry.timestamp = datetime.now().isoformat()

        # 采样 (高量情况下)
        if self.sampling_enabled:
            self._sample_counter += 1
            # 确定性采样比例
            if hash(entry.event_id) % 100 > self.sampling_rate * 100:
                return entry.event_id  # 被采样跳过

        # 计算哈希链
        entry.previous_hash = self._last_hash
        entry.current_hash = entry.compute_hash(entry.previous_hash)

        # 检查是否需要轮转
        if self._current_file and os.path.exists(self._current_file):
            if os.path.getsize(self._current_file) > self.max_file_size:
                self._rotate_file()

        if not self._current_file:
            self._rotate_file()

        # 追加写入
        json_line = entry.to_json()
        with open(self._current_file, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')

        # 更新状态
        self._last_hash = entry.current_hash
        self._entry_count_in_file += 1
        self._total_entries += 1

        return entry.event_id

    def verify_integrity(self) -> Dict[str, Any]:
        """验证日志完整性 (哈希链验证)

        Returns:
            验证结果
        """
        violations = []
        total_checked = 0

        for log_file in sorted(os.listdir(self.log_dir)):
            if not log_file.endswith('.log'):
                continue

            filepath = os.path.join(self.log_dir, log_file)
            prev_hash = "0" * 64

            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry_data = json.loads(line)
                    except json.JSONDecodeError as e:
                        violations.append({
                            'file': log_file,
                            'line': line_num,
                            'error': f"JSON解析失败: {e}",
                        })
                        continue

                    # 验证前一个哈希
                    if entry_data.get('previous_hash') != prev_hash:
                        violations.append({
                            'file': log_file,
                            'line': line_num,
                            'error': f"哈希链断裂: 期望 {prev_hash[:16]}..., 实际 {entry_data.get('previous_hash', 'N/A')[:16]}...",
                        })
                        continue

                    total_checked += 1
                    prev_hash = entry_data.get('current_hash', prev_hash)

        return {
            'integrity_verified': len(violations) == 0,
            'total_checked': total_checked,
            'violations': violations,
            'tamper_detected': len(violations) > 0,
        }

    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        actor_id: Optional[str] = None,
        max_results: int = 1000,
    ) -> List[AuditEntry]:
        """查询审计日志

        Args:
            start_time: 开始时间
            end_time: 结束时间
            event_type: 事件类型过滤
            actor_id: 用户过滤
            max_results: 最大结果数

        Returns:
            匹配的审计条目列表
        """
        results = []

        log_files = sorted([
            f for f in os.listdir(self.log_dir) if f.endswith('.log')
        ], reverse=True)  # 最近的文件先查

        for log_file in log_files:
            if len(results) >= max_results:
                break

            filepath = os.path.join(self.log_dir, log_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(results) >= max_results:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry_data = json.loads(line)
                        entry = AuditEntry.from_json(line)

                        # 时间过滤
                        if start_time:
                            entry_time = datetime.fromisoformat(entry.timestamp)
                            if entry_time < start_time:
                                continue
                        if end_time:
                            entry_time = datetime.fromisoformat(entry.timestamp)
                            if entry_time > end_time:
                                continue
                        if event_type and entry.event_type != event_type:
                            continue
                        if actor_id and entry.actor_id != actor_id:
                            continue

                        results.append(entry)
                    except (json.JSONDecodeError, Exception):
                        continue

        return results

    def enforce_retention(self) -> Dict[str, Any]:
        """执行保留策略 - 删除过期日志"""
        policy = self.RETENTION_POLICIES.get(self.compliance)
        if not policy:
            return {'status': 'no_policy', 'message': f'无合规标准 {self.compliance.value} 的保留策略'}

        cutoff = datetime.now() - timedelta(days=policy.retention_days)
        removed = 0

        for log_file in os.listdir(self.log_dir):
            if not log_file.endswith('.log'):
                continue

            filepath = os.path.join(self.log_dir, log_file)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

            if mtime < cutoff:
                os.remove(filepath)
                removed += 1

        return {
            'status': 'completed',
            'policy': policy.standard.value,
            'retention_days': policy.retention_days,
            'cutoff_date': cutoff.isoformat(),
            'removed_files': removed,
        }


# ============================================================================
# SIEM集成
# ============================================================================

class SIEMExporter:
    """SIEM (安全信息和事件管理) 集成"""

    @staticmethod
    def to_cef(entry: AuditEntry, vendor: str = "RAGPlatform", product: str = "AIAssistant") -> str:
        """转换为CEF (Common Event Format) 格式

        Syslog兼容的SIEM格式
        """
        cef_header = f"CEF:0|{vendor}|{product}|1.0|{entry.event_type.value}|{entry.action}|{entry.result}|"

        extensions = {
            'suser': entry.actor_id,
            'src': entry.actor_ip,
            'cs1': entry.resource,
            'cs2': entry.trace_id,
            'cs3': entry.session_id,
            'rt': entry.timestamp,
        }
        ext_str = ' '.join(f"{k}={v}" for k, v in extensions.items() if v)

        return f"{cef_header}{ext_str}"

    @staticmethod
    def to_syslog(entry: AuditEntry, facility: int = 4, severity: int = 6) -> str:
        """转换为Syslog格式 (RFC 5424)"""
        priority = facility * 8 + severity
        timestamp = entry.timestamp
        hostname = "rag-platform"
        app_name = "rag-audit"
        procid = "-"
        msgid = entry.event_id

        structured_data = (
            f'[audit event_type="{entry.event_type.value}" '
            f'actor="{entry.actor_id}" resource="{entry.resource}" '
            f'result="{entry.result}"]'
        )

        message = f"{entry.action} by {entry.actor_id}"
        if entry.details:
            message += f" {json.dumps(entry.details)}"

        return f"<{priority}>1 {timestamp} {hostname} {app_name} {procid} {msgid} {structured_data} {message}"


# ============================================================================
# 审计日志管理器
# ============================================================================

class AuditManager:
    """审计日志管理器 - 统一入口"""

    def __init__(
        self,
        log_dir: str = "./audit_logs",
        compliance: ComplianceStandard = ComplianceStandard.SOC2,
    ):
        self.store = AuditLogStore(log_dir, compliance)
        self.exporter = SIEMExporter()

    def log_event(
        self,
        event_type: AuditEventType,
        actor_id: str,
        resource: str = "",
        action: str = "",
        result: str = "success",
        details: Dict[str, Any] = None,
        actor_ip: str = "",
        session_id: str = "",
        trace_id: str = "",
    ) -> str:
        """记录一条审计事件

        Args:
            event_type: 事件类型
            actor_id: 操作者ID
            resource: 资源标识
            action: 具体操作
            result: 结果
            details: 详细信息
            actor_ip: 操作者IP
            session_id: 会话ID
            trace_id: 追踪ID

        Returns:
            事件ID
        """
        entry = AuditEntry(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            actor_id=actor_id,
            actor_ip=actor_ip,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            session_id=session_id,
            trace_id=trace_id,
        )

        event_id = self.store.append(entry)
        return event_id

    def verify(self) -> Dict[str, Any]:
        """验证日志完整性"""
        return self.store.verify_integrity()

    def query(
        self,
        **kwargs,
    ) -> List[AuditEntry]:
        """查询审计日志"""
        return self.store.query(**kwargs)

    def cleanup(self) -> Dict[str, Any]:
        """执行合规保留策略"""
        return self.store.enforce_retention()


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  审计日志系统 - 完整演示")
    print("=" * 60)

    # 初始化
    manager = AuditManager(log_dir="./demo_audit_logs")

    # ---- 记录事件 ----
    print("\n[1] 记录审计事件")
    print("-" * 40)

    events = [
        {"type": AuditEventType.USER_LOGIN, "actor": "alice", "action": "login",
         "result": "success", "details": {"auth_method": "oauth2"}},
        {"type": AuditEventType.TOOL_EXECUTED, "actor": "alice", "action": "search",
         "result": "success", "details": {"tool": "vector_search", "query_len": 45}},
        {"type": AuditEventType.ACCESS_DENIED, "actor": "bob", "action": "admin_config",
         "result": "failure", "details": {"permission": "system.admin", "roles": ["developer"]}},
        {"type": AuditEventType.DATA_EXPORTED, "actor": "alice", "action": "export_report",
         "result": "success", "details": {"format": "csv", "rows": 1500}},
        {"type": AuditEventType.CONFIG_CHANGED, "actor": "alice", "action": "update_prompt",
         "result": "success", "details": {"version": "v2.3.1"}},
        {"type": AuditEventType.SECURITY_ALERT, "actor": "system", "action": "injection_detected",
         "result": "alert", "details": {"user": "unknown", "risk_score": 75}},
    ]

    for evt in events:
        eid = manager.log_event(**evt)
        print(f"  [{evt['type'].value}] {evt['actor']}: {evt['action']} → {evt['result']}")

    # ---- 查询 ----
    print("\n[2] 查询审计日志")
    print("-" * 40)

    alice_events = manager.query(actor_id="alice", max_results=10)
    denied_events = manager.query(event_type=AuditEventType.ACCESS_DENIED)

    print(f"  alice的事件: {len(alice_events)} 条")
    for e in alice_events:
        print(f"    {e.timestamp[:19]} [{e.event_type.value}] {e.action} → {e.result}")

    print(f"\n  访问拒绝事件: {len(denied_events)} 条")

    # ---- 完整性验证 ----
    print("\n[3] 完整性验证 (哈希链)")
    print("-" * 40)

    integrity = manager.verify()
    print(f"  完整性: {'✓ 通过' if integrity['integrity_verified'] else '✗ 被篡改!'}")
    print(f"  检查条目: {integrity['total_checked']}")
    if integrity['violations']:
        for v in integrity['violations']:
            print(f"    违规: {v['error']}")

    # ---- SIEM导出 ----
    print("\n[4] SIEM格式导出")
    print("-" * 40)

    if alice_events:
        cef = SIEMExporter.to_cef(alice_events[0])
        print(f"  CEF格式: {cef[:120]}...")

        syslog = SIEMExporter.to_syslog(alice_events[0])
        print(f"  Syslog格式: {syslog[:120]}...")

    # ---- 清理 (演示用) ----
    shutil.rmtree("./demo_audit_logs", ignore_errors=True)

    print()
    print("=" * 60)
    print("  审计日志系统演示完成")
    print("=" * 60)
