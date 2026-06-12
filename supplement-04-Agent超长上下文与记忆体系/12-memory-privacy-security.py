#!/usr/bin/env python3
"""
记忆隐私与安全 (Memory Privacy & Security)
==============================================
实现完整的内存隐私保护体系：PII检测与脱敏、保留策略管理、
访问控制和审计日志哈希链。

9种PII类型检测：
  1. 中文姓名  2. 手机号  3. 身份证号  4. 邮箱地址
  5. IP地址   6. 银行卡号  7. 统一社会信用代码  8. 家庭地址
  9. 车牌号

核心组件：
  - PIIDetector: 9种PII正则检测 + 分级脱敏
  - MemorySanitizer: 记忆清理器
  - RetentionPolicyManager: 保留策略管理
  - AccessControlManager: 访问控制管理
  - AuditLogger: 审计日志（哈希链完整性保护）
"""

from __future__ import annotations

import re
import time
import json
import hashlib
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict, OrderedDict


# ============================================================================
# 枚举定义
# ============================================================================

class PIIType(Enum):
    """PII 类型枚举"""
    CHINESE_NAME = auto()              # 中文姓名
    PHONE_NUMBER = auto()             # 手机号
    ID_CARD = auto()                  # 身份证号
    EMAIL = auto()                    # 邮箱地址
    IP_ADDRESS = auto()               # IP地址
    BANK_CARD = auto()                # 银行卡号
    CREDIT_CODE = auto()              # 统一社会信用代码
    HOME_ADDRESS = auto()             # 家庭地址
    LICENSE_PLATE = auto()            # 车牌号

    @property
    def risk_level(self) -> str:
        """PII风险等级"""
        risk_map = {
            PIIType.CHINESE_NAME: "medium",
            PIIType.PHONE_NUMBER: "high",
            PIIType.ID_CARD: "critical",
            PIIType.EMAIL: "medium",
            PIIType.IP_ADDRESS: "low",
            PIIType.BANK_CARD: "critical",
            PIIType.CREDIT_CODE: "low",
            PIIType.HOME_ADDRESS: "high",
            PIIType.LICENSE_PLATE: "medium",
        }
        return risk_map.get(self, "low")


class RetentionLevel(Enum):
    """数据保留等级"""
    TRANSIENT = auto()    # 临时：会话结束后删除
    SHORT_TERM = auto()   # 短期：7天
    MEDIUM_TERM = auto()  # 中期：90天
    LONG_TERM = auto()    # 长期：365天
    PERMANENT = auto()    # 永久：不自动删除


class AccessLevel(Enum):
    """访问等级"""
    NO_ACCESS = auto()
    READ_ONLY = auto()
    READ_WRITE = auto()
    ADMIN = auto()


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class PIIMatch:
    """PII 检测匹配结果"""
    pii_type: PIIType
    matched_text: str
    start_pos: int
    end_pos: int
    risk_level: str
    masked_text: str = ""


@dataclass
class SanitizedEntry:
    """经过脱敏处理的记忆条目"""
    original: str
    sanitized: str
    pii_matches: List[PIIMatch] = field(default_factory=list)
    sanitization_level: str = "none"   # none, masked, removed
    timestamp: float = field(default_factory=time.time)


@dataclass
class AuditEntry:
    """审计日志条目"""
    entry_id: str
    operation: str           # READ, WRITE, DELETE, ACCESS_GRANTED, etc.
    user_id: str
    resource: str            # 操作对象
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    previous_hash: str = ""  # 链式哈希（防篡改）


@dataclass
class RetentionPolicy:
    """数据保留策略"""
    mem_type: str
    retention_level: RetentionLevel
    retention_days: int
    auto_delete: bool = True
    requires_approval: bool = False


# ============================================================================
# PII 检测器（9种PII正则）
# ============================================================================

class PIIDetector:
    """
    PII 检测器。

    支持9种个人身份信息类型的检测和脱敏。
    """

    # 9种PII类型的正则表达式
    PII_PATTERNS: Dict[PIIType, str] = {
        # 1. 中文姓名：2-4个汉字
        PIIType.CHINESE_NAME: (
            r'(?:姓名[：:]\s*|姓[：:]\s*|名[：:]\s*|'
            r'联系人[：:]\s*|负责人[：:]\s*|客户[：:]\s*)?'
            r'[一-鿿]{2,4}(?=[\s，,。.]|$)'
        ),

        # 2. 手机号：1开头的11位数字
        PIIType.PHONE_NUMBER: r'1[3-9]\d{9}',

        # 3. 身份证号：18位（17数字+1数字或X）
        PIIType.ID_CARD: r'[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]',

        # 4. 邮箱地址
        PIIType.EMAIL: r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',

        # 5. IP地址
        PIIType.IP_ADDRESS: r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',

        # 6. 银行卡号：16-19位数字
        PIIType.BANK_CARD: r'\b[1-9]\d{15,18}\b',

        # 7. 统一社会信用代码：18位
        PIIType.CREDIT_CODE: r'[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}',

        # 8. 家庭地址（简单模式）
        PIIType.HOME_ADDRESS: (
            r'(?:地址[：:]\s*)?'
            r'(?:[一-鿿]{2,}(?:省|市|区|县|镇|乡|村|路|街|巷|号|楼|室|单元|栋|幢)'
            r'(?:\d+[号楼室单元栋幢号]?){1,5})'
        ),

        # 9. 车牌号
        PIIType.LICENSE_PLATE: r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警]',
    }

    def __init__(self):
        self._compiled_patterns: Dict[PIIType, re.Pattern] = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            self._compiled_patterns[pii_type] = re.compile(pattern)
        self._detection_history: List[Dict[str, Any]] = []

    def detect(self, text: str) -> List[PIIMatch]:
        """
        检测文本中的所有PII。

        Args:
            text: 待检测文本

        Returns:
            PII匹配列表
        """
        matches = []

        for pii_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                pii_match = PIIMatch(
                    pii_type=pii_type,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    risk_level=pii_type.risk_level,
                )
                matches.append(pii_match)

        # 按位置排序
        matches.sort(key=lambda m: (m.start_pos, -m.end_pos))

        # 去除重叠匹配（保留更长的）
        non_overlapping = self._remove_overlaps(matches)

        self._detection_history.append({
            "text_length": len(text),
            "matches_count": len(non_overlapping),
            "timestamp": time.time(),
        })

        return non_overlapping

    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """去除重叠的PII匹配"""
        if not matches:
            return []
        result = [matches[0]]
        for current in matches[1:]:
            last = result[-1]
            if current.start_pos >= last.end_pos:
                result.append(current)
            else:
                # 保留更长的匹配
                if (current.end_pos - current.start_pos) > (last.end_pos - last.start_pos):
                    result[-1] = current
        return result

    def mask(self, text: str, matches: Optional[List[PIIMatch]] = None) -> str:
        """
        对检测到的PII进行脱敏处理。

        脱敏策略：
        - 手机号: 138****1234
        - 身份证: 3201**********1234
        - 邮箱: a***@domain.com
        - 姓名: 张*
        - 银行卡: 6222****1234

        Args:
            text: 原始文本
            matches: PII匹配列表（None时自动检测）

        Returns:
            脱敏后的文本
        """
        if matches is None:
            matches = self.detect(text)

        # 按位置倒序处理（避免索引偏移）
        chars = list(text)
        for match in reversed(matches):
            masked = self._mask_value(match)
            match.masked_text = masked
            # 替换
            chars[match.start_pos:match.end_pos] = list(masked)

        return "".join(chars)

    def _mask_value(self, match: PIIMatch) -> str:
        """根据PII类型生成脱敏后的值"""
        value = match.matched_text
        pii_type = match.pii_type

        if pii_type == PIIType.PHONE_NUMBER:
            # 138****1234
            return value[:3] + "****" + value[-4:]

        elif pii_type == PIIType.ID_CARD:
            # 3201**********1234
            return value[:4] + "**********" + value[-4:]

        elif pii_type == PIIType.EMAIL:
            # a***@domain.com
            parts = value.split("@")
            if len(parts) == 2:
                local = parts[0]
                if len(local) <= 2:
                    return local[0] + "***@" + parts[1]
                return local[0] + "***" + local[-1] + "@" + parts[1]
            return "***@***"

        elif pii_type == PIIType.CHINESE_NAME:
            # 张三 → 张*
            if len(value) >= 2:
                # 去除前缀标签
                name_part = re.sub(r'[姓名联系人负责人客户][：:]\s*', '', value)
                if len(name_part) >= 2:
                    return name_part[0] + "*" * (len(name_part) - 1)
            return "*" * len(value)

        elif pii_type == PIIType.BANK_CARD:
            # 6222****1234
            return value[:4] + "****" + value[-4:]

        elif pii_type == PIIType.IP_ADDRESS:
            # 192.168.***.***
            parts = value.split(".")
            return parts[0] + "." + parts[1] + ".***.***" if len(parts) >= 2 else "***.***.***.***"

        else:
            # 默认：一半掩码
            half = len(value) // 2
            return value[:half] + "*" * (len(value) - half)

    def risk_level(self, matches: List[PIIMatch]) -> str:
        """
        评估整体PII风险等级。

        Returns:
            "critical", "high", "medium", "low", "none"
        """
        if not matches:
            return "none"

        levels = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        max_level = max(levels.get(m.risk_level, 1) for m in matches)

        if max_level >= 4:
            return "critical"
        elif max_level >= 3:
            return "high"
        elif max_level >= 2:
            return "medium"
        return "low"

    def get_stats(self) -> Dict[str, Any]:
        """获取检测统计"""
        if not self._detection_history:
            return {"total_scans": 0}

        return {
            "total_scans": len(self._detection_history),
            "total_matches": sum(h["matches_count"] for h in self._detection_history),
            "avg_matches_per_scan": round(
                sum(h["matches_count"] for h in self._detection_history) / len(self._detection_history), 1
            ),
        }


# ============================================================================
# 记忆清理器
# ============================================================================

class MemorySanitizer:
    """
    记忆清理器。

    在记忆存入之前进行PII检测和脱敏处理。
    """

    def __init__(self, pii_detector: Optional[PIIDetector] = None):
        self.detector = pii_detector or PIIDetector()
        self._sanitized_entries: List[SanitizedEntry] = []

    def sanitize(self, text: str,
                  level: str = "masked") -> SanitizedEntry:
        """
        对记忆文本进行清理。

        Args:
            text: 原始文本
            level: 清理级别
                - "none": 不做处理
                - "masked": 脱敏（替换为掩码）
                - "removed": 完全移除PII

        Returns:
            SanitizedEntry
        """
        if level == "none":
            entry = SanitizedEntry(
                original=text,
                sanitized=text,
                pii_matches=[],
                sanitization_level="none",
            )
        else:
            matches = self.detector.detect(text)

            if level == "masked":
                sanitized_text = self.detector.mask(text, matches)
            elif level == "removed":
                # 移除所有匹配到的PII
                sanitized_text = text
                for match in reversed(matches):
                    sanitized_text = (
                        sanitized_text[:match.start_pos] +
                        sanitized_text[match.end_pos:]
                    )

            entry = SanitizedEntry(
                original=text,
                sanitized=sanitized_text,
                pii_matches=matches,
                sanitization_level=level,
            )

        self._sanitized_entries.append(entry)
        return entry

    def get_stats(self) -> Dict[str, Any]:
        """获取清理统计"""
        total = len(self._sanitized_entries)
        if total == 0:
            return {"total_sanitized": 0}

        with_pii = sum(1 for e in self._sanitized_entries if e.pii_matches)
        return {
            "total_sanitized": total,
            "with_pii": with_pii,
            "pii_rate": round(with_pii / total * 100, 1),
            "avg_pii_per_entry": round(
                sum(len(e.pii_matches) for e in self._sanitized_entries) / total, 1
            ),
        }


# ============================================================================
# 保留策略管理器
# ============================================================================

class RetentionPolicyManager:
    """
    保留策略管理器。

    管理不同类型记忆的保留策略和过期检查。
    """

    def __init__(self):
        self._policies: Dict[str, RetentionPolicy] = {}
        self._deletion_requests: List[Dict[str, Any]] = []
        self._init_default_policies()

    def _init_default_policies(self):
        """初始化默认保留策略"""
        defaults = [
            ("event", RetentionLevel.SHORT_TERM, 7),
            ("preference", RetentionLevel.MEDIUM_TERM, 90),
            ("fact", RetentionLevel.LONG_TERM, 365),
            ("general", RetentionLevel.MEDIUM_TERM, 90),
            ("system", RetentionLevel.PERMANENT, -1),
        ]
        for mem_type, level, days in defaults:
            self._policies[mem_type] = RetentionPolicy(
                mem_type=mem_type,
                retention_level=level,
                retention_days=days,
                auto_delete=level != RetentionLevel.PERMANENT,
            )

    def check_expired(self, mem_type: str,
                       created_at: float,
                       current_time: Optional[float] = None) -> bool:
        """
        检查记忆是否已过期。

        Args:
            mem_type: 记忆类型
            created_at: 创建时间戳
            current_time: 当前时间

        Returns:
            True 如果已过期
        """
        now = current_time or time.time()
        policy = self._policies.get(mem_type)

        if not policy or policy.retention_level == RetentionLevel.PERMANENT:
            return False

        elapsed_days = (now - created_at) / 86400.0
        return elapsed_days > policy.retention_days

    def apply_retention(self, memories: List[Dict[str, Any]],
                         current_time: Optional[float] = None) -> Tuple[
                             List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        应用保留策略，分离保留和待删除的记忆。

        Returns:
            (to_keep, to_delete)
        """
        now = current_time or time.time()
        to_keep = []
        to_delete = []

        for mem in memories:
            mem_type = mem.get("mem_type", "general")
            created_at = mem.get("created_at", now)

            if self.check_expired(mem_type, created_at, now):
                to_delete.append(mem)
            else:
                to_keep.append(mem)

        return to_keep, to_delete

    def user_deletion_request(self, memory_id: str, user_id: str,
                               reason: str = "") -> str:
        """
        处理用户删除请求。

        Returns:
            请求ID
        """
        request_id = f"del-{uuid.uuid4().hex[:12]}"
        self._deletion_requests.append({
            "request_id": request_id,
            "memory_id": memory_id,
            "user_id": user_id,
            "reason": reason,
            "timestamp": time.time(),
            "status": "pending",
        })
        return request_id

    def get_policies(self) -> List[Dict[str, Any]]:
        """获取所有保留策略"""
        return [
            {
                "mem_type": p.mem_type,
                "level": p.retention_level.name,
                "days": p.retention_days,
                "auto_delete": p.auto_delete,
            }
            for p in self._policies.values()
        ]


# ============================================================================
# 访问控制管理器
# ============================================================================

class AccessControlManager:
    """
    访问控制管理器。

    管理用户/Agent对记忆资源的访问权限。
    """

    def __init__(self):
        self._permissions: Dict[str, Dict[str, AccessLevel]] = defaultdict(dict)
        self._access_log: List[Dict[str, Any]] = []

    def grant_access(self, user_id: str, resource: str,
                      level: AccessLevel):
        """授予访问权限"""
        self._permissions[user_id][resource] = level

    def revoke_access(self, user_id: str, resource: str):
        """撤销访问权限"""
        if user_id in self._permissions:
            self._permissions[user_id].pop(resource, None)

    def check_permission(self, user_id: str, resource: str,
                          required_level: AccessLevel) -> bool:
        """
        检查用户是否有足够的访问权限。

        Args:
            user_id: 用户ID
            resource: 资源标识
            required_level: 需要的权限等级

        Returns:
            True 如果有足够权限
        """
        user_perms = self._permissions.get(user_id, {})
        actual_level = user_perms.get(
            resource,
            user_perms.get("*", AccessLevel.NO_ACCESS)
        )

        has_access = actual_level.value >= required_level.value

        # 记录访问日志
        self._access_log.append({
            "user_id": user_id,
            "resource": resource,
            "required": required_level.name,
            "granted": has_access,
            "timestamp": time.time(),
        })

        return has_access

    def get_user_permissions(self, user_id: str) -> Dict[str, str]:
        """获取用户的所有权限"""
        return {
            resource: level.name
            for resource, level in self._permissions.get(user_id, {}).items()
        }


# ============================================================================
# 审计日志（哈希链）
# ============================================================================

class AuditLogger:
    """
    带哈希链完整性的审计日志。

    每个日志条目包含前一条的哈希值，形成防篡改链：
    Entry[N].previous_hash = SHA256(Entry[N-1])

    任何对历史日志的修改都会导致后续哈希链断裂。
    """

    def __init__(self, log_id: str = "memory-audit"):
        self.log_id = log_id
        self._entries: List[AuditEntry] = []
        self._last_hash: str = "0000000000000000000000000000000000000000000000000000000000000000"

    def log(self, operation: str, user_id: str, resource: str,
             details: Optional[Dict[str, Any]] = None) -> AuditEntry:
        """
        记录一条审计日志。

        Args:
            operation: 操作类型
            user_id: 操作用户
            resource: 操作对象
            details: 操作详情

        Returns:
            创建的审计条目
        """
        entry_id = f"audit-{len(self._entries)+1:06d}"

        entry = AuditEntry(
            entry_id=entry_id,
            operation=operation,
            user_id=user_id,
            resource=resource,
            details=details or {},
            previous_hash=self._last_hash,
        )

        # 计算当前条目的哈希
        current_hash = self._compute_hash(entry)
        self._last_hash = current_hash
        self._entries.append(entry)

        return entry

    def _compute_hash(self, entry: AuditEntry) -> str:
        """计算审计条目的哈希"""
        data = json.dumps({
            "entry_id": entry.entry_id,
            "operation": entry.operation,
            "user_id": entry.user_id,
            "resource": entry.resource,
            "details": entry.details,
            "timestamp": entry.timestamp,
            "previous_hash": entry.previous_hash,
        }, sort_keys=True, ensure_ascii=False)

        return hashlib.sha256(data.encode()).hexdigest()

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """
        验证哈希链完整性。

        Returns:
            (is_valid, first_corrupted_index)
            - is_valid: 链是否完整
            - first_corrupted_index: 第一个被破坏的条目索引（如果完整则为None）
        """
        expected_hash = "0000000000000000000000000000000000000000000000000000000000000000"

        for i, entry in enumerate(self._entries):
            # 验证 previous_hash
            if entry.previous_hash != expected_hash:
                return False, i

            # 重新计算当前哈希作为下一次的期望
            data = json.dumps({
                "entry_id": entry.entry_id,
                "operation": entry.operation,
                "user_id": entry.user_id,
                "resource": entry.resource,
                "details": entry.details,
                "timestamp": entry.timestamp,
                "previous_hash": entry.previous_hash,
            }, sort_keys=True, ensure_ascii=False)
            expected_hash = hashlib.sha256(data.encode()).hexdigest()

        return True, None

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取审计日志"""
        return [
            {
                "id": e.entry_id,
                "operation": e.operation,
                "user_id": e.user_id,
                "resource": e.resource,
                "previous_hash": e.previous_hash[:16] + "...",
            }
            for e in self._entries[-limit:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取审计日志统计"""
        is_valid, corrupt_index = self.verify_chain()
        return {
            "log_id": self.log_id,
            "total_entries": len(self._entries),
            "chain_valid": is_valid,
            "corrupt_index": corrupt_index,
            "last_entry_time": self._entries[-1].timestamp if self._entries else None,
        }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示"""
    print("=" * 70)
    print("  记忆隐私与安全 (Memory Privacy & Security) 演示")
    print("=" * 70)

    # ---- 1. PII 检测与脱敏 ----
    print("\n【1. PII 检测与脱敏（9种PII类型）】")
    print("-" * 50)

    detector = PIIDetector()

    # 包含多种PII的测试文本
    test_text = (
        "客户张三的联系电话是13812345678，身份证号320102199001011234，"
        "邮箱zhangsan@example.com，银行卡号6222021234567890，"
        "地址：江苏省南京市鼓楼区中山路100号12栋305室，"
        "公司统一社会信用代码91320100MA1WXXXXXX，"
        "车辆苏A12345，IP地址192.168.1.100。"
    )

    print(f"  原始文本: {test_text[:100]}...")

    # 检测
    matches = detector.detect(test_text)
    print(f"\n  检测到 {len(matches)} 个PII:")
    for m in matches:
        print(f"    [{m.risk_level:8s}] {m.pii_type.name:20s}: {m.matched_text}")

    # 脱敏
    masked_text = detector.mask(test_text, matches)
    print(f"\n  脱敏后: {masked_text}")

    # 风险评估
    risk = detector.risk_level(matches)
    print(f"\n  整体PII风险等级: {risk}")

    # 单独测试每种PII
    print(f"\n  9种PII正则测试:")
    pii_test_cases = [
        ("手机号: 13900001111", PIIType.PHONE_NUMBER),
        ("身份证: 320102199001011234", PIIType.ID_CARD),
        ("邮箱: test@company.com", PIIType.EMAIL),
        ("IP: 10.0.0.1", PIIType.IP_ADDRESS),
        ("银行卡: 6222021234567890123", PIIType.BANK_CARD),
        ("信用代码: 91320100MA1WXXXXXX", PIIType.CREDIT_CODE),
        ("地址: 北京市海淀区中关村大街1号", PIIType.HOME_ADDRESS),
        ("车牌: 京A88888", PIIType.LICENSE_PLATE),
        ("姓名: 王五", PIIType.CHINESE_NAME),
    ]
    for text, pii_type in pii_test_cases:
        matches = detector.detect(text)
        if matches:
            print(f"    ✓ {pii_type.name}: 检测到 '{matches[0].matched_text}'")
        else:
            print(f"    ✗ {pii_type.name}: 未检测到（在 '{text}' 中）")

    # ---- 2. 记忆清理器 ----
    print("\n【2. 记忆清理器（Sanitizer）】")
    print("-" * 50)

    sanitizer = MemorySanitizer(detector)

    memory_texts = [
        "用户李四的订单已确认，联系方式：13900001111",
        "会议记录：与客户赵六讨论合作，邮箱：zhao@company.com",
        "系统日志：用户192.168.1.1登录成功",
        "预订信息：张三，13812345678，身份证320102199001011234",
    ]

    for text in memory_texts:
        entry = sanitizer.sanitize(text, level="masked")
        print(f"  原始: {text[:50]}...")
        print(f"  清理: {entry.sanitized[:50]}...")
        print(f"    PII数: {len(entry.pii_matches)}")
        print()

    print(f"  清理统计: {json.dumps(sanitizer.get_stats(), indent=2)}")

    # ---- 3. 保留策略 ----
    print("\n【3. 保留策略管理】")
    print("-" * 50)

    retention_mgr = RetentionPolicyManager()

    print(f"  默认保留策略:")
    for policy in retention_mgr.get_policies():
        print(f"    {policy['mem_type']:12s}: {policy['level']:12s} ({policy['days']}天)"
              f" 自动删除={'是' if policy['auto_delete'] else '否'}")

    # 过期检查
    now = time.time()
    test_memories = [
        {"id": "m1", "mem_type": "event", "created_at": now - 86400 * 10, "content": "旧事件"},
        {"id": "m2", "mem_type": "fact", "created_at": now - 86400 * 100, "content": "事实"},
        {"id": "m3", "mem_type": "system", "created_at": now - 86400 * 500, "content": "系统"},
    ]

    keep, delete = retention_mgr.apply_retention(test_memories)
    print(f"\n  保留策略应用结果:")
    print(f"    保留: {len(keep)} 条")
    print(f"    删除: {len(delete)} 条")
    for m in delete:
        print(f"      删除: [{m['id']}] {m['mem_type']} - {m['content']}")

    # 用户删除请求
    req_id = retention_mgr.user_deletion_request("m1", "user_001", "隐私保护要求")
    print(f"\n  用户删除请求: {req_id}")

    # ---- 4. 访问控制 ----
    print("\n【4. 访问控制】")
    print("-" * 50)

    acm = AccessControlManager()

    # 配置权限
    acm.grant_access("admin", "*", AccessLevel.ADMIN)
    acm.grant_access("agent_1", "memories:*", AccessLevel.READ_WRITE)
    acm.grant_access("agent_2", "memories:public", AccessLevel.READ_ONLY)
    acm.grant_access("guest", "memories:public", AccessLevel.READ_ONLY)

    # 权限检查
    test_accesses = [
        ("admin", "memories:private", AccessLevel.ADMIN),
        ("agent_1", "memories:public", AccessLevel.READ_ONLY),
        ("agent_1", "memories:private", AccessLevel.READ_WRITE),
        ("agent_2", "memories:public", AccessLevel.READ_ONLY),
        ("agent_2", "memories:private", AccessLevel.READ_WRITE),
        ("guest", "memories:private", AccessLevel.READ_ONLY),
    ]

    for user, resource, required in test_accesses:
        result = acm.check_permission(user, resource, required)
        status = "✓ 允许" if result else "✗ 拒绝"
        print(f"    {user:10s} → {resource:20s} [{required.name:10s}] {status}")

    # ---- 5. 审计日志（哈希链） ----
    print("\n【5. 审计日志与哈希链完整性】")
    print("-" * 50)

    audit = AuditLogger(log_id="memory-security-audit")

    # 记录一些操作
    operations = [
        ("MEMORY_WRITE", "agent_1", "memory_001", {"action": "create"}),
        ("MEMORY_READ", "agent_2", "memory_001", {"action": "query"}),
        ("PII_SCAN", "sanitizer", "system", {"matches": 3}),
        ("ACCESS_DENIED", "guest", "memory_001", {"reason": "no_permission"}),
        ("MEMORY_DELETE", "admin", "memory_002", {"reason": "retention_expired"}),
        ("POLICY_UPDATE", "admin", "retention_policy", {"change": "event_retention_7d"}),
    ]

    for op, user, resource, details in operations:
        entry = audit.log(op, user, resource, details)
        print(f"  [{entry.entry_id}] {op} by {user} on {resource}")

    # 验证哈希链
    is_valid, corrupt_idx = audit.verify_chain()
    print(f"\n  哈希链验证: {'✓ 完整' if is_valid else f'✗ 在索引{corrupt_idx}处断裂'}")

    # 模拟篡改
    print("\n  [模拟篡改检测]")
    # 修改某个历史条目（仅演示逻辑）
    if audit._entries:
        original_entry = audit._entries[2]
        print(f"    修改条目 {original_entry.entry_id} 的operation...")
        original_entry.operation = "TAMPERED"

    is_valid_after, corrupt_idx_after = audit.verify_chain()
    print(f"  篡改后哈希链验证: {'✓ 完整' if is_valid_after else f'✗ 在索引{corrupt_idx_after}处断裂'}")

    # 审计日志概览
    print(f"\n  审计日志统计:")
    print(json.dumps(audit.get_stats(), indent=2))

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
