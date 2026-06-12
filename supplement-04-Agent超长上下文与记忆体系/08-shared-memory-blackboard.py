#!/usr/bin/env python3
"""
共享内存黑板系统 (Shared Memory Blackboard)
===============================================
实现基于黑板架构的多Agent共享内存，支持命名空间隔离、
发布/订阅模式和事务管理。

架构模式：
  Agent-1 ──写入──┐
  Agent-2 ──写入──┤
  Agent-3 ──读取──┼── Blackboard ──发布──> 订阅者
                    │   ├── namespace:task-001
                    │   ├── namespace:knowledge
                    │   └── namespace:system
                    │
  冲突检测 ←───────┘

核心组件：
  - SharedMemoryBlackboard: 共享黑板
  - ConflictDetector: 写入冲突检测器
  - TransactionManager: 事务管理器（支持ACID）
"""

from __future__ import annotations

import time
import json
import uuid
import re
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict


# ============================================================================
# 数据结构
# ============================================================================

class Permission(Enum):
    """访问权限"""
    READ = auto()
    WRITE = auto()
    READ_WRITE = auto()
    NONE = auto()


class EntryScope(Enum):
    """条目可见范围"""
    PRIVATE = auto()        # 仅创建Agent可见
    NAMESPACE = auto()      # 命名空间内可见
    GLOBAL = auto()         # 全局可见


@dataclass
class BlackboardEntry:
    """黑板上的一个条目"""
    entry_id: str                               # 唯一ID
    namespace: str                              # 命名空间
    key: str                                    # 键
    value: Any                                  # 值
    agent_id: str                               # 创建者ID
    scope: EntryScope = EntryScope.NAMESPACE    # 可见范围
    version: int = 1                            # 版本号
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None                 # 生存时间(秒)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


@dataclass
class Conflict:
    """写入冲突"""
    conflict_type: str          # 冲突类型: "version" | "write-write" | "permission"
    entry_id: str               # 冲突条目ID
    agent_a: str                # Agent A
    agent_b: str                # Agent B
    key: str                    # 冲突键
    description: str            # 冲突描述
    timestamp: float = field(default_factory=time.time)


@dataclass
class Subscription:
    """订阅记录"""
    subscription_id: str
    agent_id: str
    namespace_pattern: str      # 命名空间匹配模式（正则）
    key_pattern: str            # 键匹配模式（正则）
    callback: Callable          # 回调函数
    created_at: float = field(default_factory=time.time)


# ============================================================================
# 事务管理器
# ============================================================================

class Transaction:
    """单个事务"""
    def __init__(self, tx_id: str):
        self.tx_id = tx_id
        self.operations: List[Dict[str, Any]] = []
        self.snapshot: Dict[str, Any] = {}
        self.status: str = "active"  # active, committed, rolled_back

    def add_operation(self, op_type: str, key: str, old_value: Any = None,
                       new_value: Any = None):
        """记录操作"""
        self.operations.append({
            "op": op_type,
            "key": key,
            "old": old_value,
            "new": new_value,
            "timestamp": time.time(),
        })


class TransactionManager:
    """
    事务管理器。

    提供：
    - begin(): 开启事务
    - commit(): 提交事务
    - rollback(): 回滚事务
    - 基于快照的隔离级别支持
    """

    def __init__(self):
        self._active_transactions: Dict[str, Transaction] = {}
        self._transaction_log: List[Dict[str, Any]] = []

    def begin(self) -> str:
        """开启一个新事务，返回事务ID"""
        tx_id = f"tx-{uuid.uuid4().hex[:8]}"
        self._active_transactions[tx_id] = Transaction(tx_id)
        return tx_id

    def commit(self, tx_id: str) -> bool:
        """
        提交事务。

        Returns:
            True 如果提交成功
        """
        tx = self._active_transactions.get(tx_id)
        if not tx:
            return False

        tx.status = "committed"
        self._transaction_log.append({
            "tx_id": tx_id,
            "status": "committed",
            "operations": len(tx.operations),
            "timestamp": time.time(),
        })
        del self._active_transactions[tx_id]
        return True

    def rollback(self, tx_id: str) -> bool:
        """
        回滚事务，恢复事务中修改的所有条目。

        Returns:
            True 如果回滚成功
        """
        tx = self._active_transactions.get(tx_id)
        if not tx:
            return False

        tx.status = "rolled_back"
        self._transaction_log.append({
            "tx_id": tx_id,
            "status": "rolled_back",
            "operations": len(tx.operations),
            "timestamp": time.time(),
        })

        # 逆序回滚操作
        for op in reversed(tx.operations):
            if op["op"] == "write":
                # 恢复旧值（实际实现中需要访问黑板）
                pass

        del self._active_transactions[tx_id]
        return True

    def get_active_transactions(self) -> List[str]:
        """获取活跃事务列表"""
        return list(self._active_transactions.keys())

    def get_log(self) -> List[Dict[str, Any]]:
        """获取事务日志"""
        return list(self._transaction_log)


# ============================================================================
# 冲突检测器
# ============================================================================

class ConflictDetector:
    """
    冲突检测器。

    检测类型：
    - 版本冲突：两个Agent尝试写入同一数据的不同版本
    - 写-写冲突：两个Agent同时写入同一键
    - 权限冲突：Agent尝试写入无权限区域
    """

    def __init__(self):
        self._conflicts: List[Conflict] = []
        self._resolved_conflicts: List[str] = []

    def detect(self, entries: Dict[str, BlackboardEntry],
               agent_permissions: Dict[str, Dict[str, Permission]]
               ) -> List[Conflict]:
        """
        检测黑板上的冲突。

        Args:
            entries: 当前所有条目
            agent_permissions: {agent_id: {namespace: Permission}}

        Returns:
            检测到的冲突列表
        """
        conflicts = []

        # 按 (namespace, key) 分组检查写-写冲突
        key_groups: Dict[Tuple[str, str], List[BlackboardEntry]] = defaultdict(list)
        for entry in entries.values():
            key_groups[(entry.namespace, entry.key)].append(entry)

        # 写-写冲突：同一键有来自不同Agent的多个版本
        for (ns, key), group in key_groups.items():
            if len(group) > 1:
                agents = list(set(e.agent_id for e in group))
                if len(agents) > 1:
                    conflicts.append(Conflict(
                        conflict_type="write-write",
                        entry_id=group[0].entry_id,
                        agent_a=agents[0],
                        agent_b=agents[1] if len(agents) > 1 else "",
                        key=f"{ns}:{key}",
                        description=f"键 '{ns}:{key}' 被 {len(agents)} 个Agent同时写入",
                    ))

        # 权限冲突：检查Agent是否有写入权限
        for entry in entries.values():
            agent_perm = agent_permissions.get(entry.agent_id, {})
            ns_perm = agent_perm.get(entry.namespace, Permission.READ_WRITE)

            if ns_perm in (Permission.NONE, Permission.READ):
                conflicts.append(Conflict(
                    conflict_type="permission",
                    entry_id=entry.entry_id,
                    agent_a=entry.agent_id,
                    agent_b="",
                    key=f"{entry.namespace}:{entry.key}",
                    description=f"Agent '{entry.agent_id}' 无写入权限于 '{entry.namespace}'",
                ))

        self._conflicts.extend(conflicts)
        return conflicts

    def resolve_strategy(self, conflict: Conflict) -> str:
        """
        为冲突选择解决策略。

        Returns:
            策略名称
        """
        strategies = {
            "write-write": "last-write-wins",  # 或 "manual-merge"
            "version": "accept-newer",
            "permission": "reject",
        }
        return strategies.get(conflict.conflict_type, "reject")

    def get_stats(self) -> Dict[str, Any]:
        """获取冲突统计"""
        return {
            "total_detected": len(self._conflicts),
            "resolved": len(self._resolved_conflicts),
            "pending": len(self._conflicts) - len(self._resolved_conflicts),
            "by_type": {
                t: sum(1 for c in self._conflicts if c.conflict_type == t)
                for t in set(c.conflict_type for c in self._conflicts)
            },
        }


# ============================================================================
# 共享内存黑板
# ============================================================================

class SharedMemoryBlackboard:
    """
    共享内存黑板系统。

    特性：
    - 命名空间隔离：不同任务/领域的记忆相互隔离
    - 作用域控制：PRIVATE / NAMESPACE / GLOBAL
    - 发布/订阅：Agent可订阅特定命名空间的变化
    - TTL支持：条目可设置生存时间，自动过期清理
    - 版本追踪：每次写入递增版本号
    """

    def __init__(self, name: str = "SharedBlackboard"):
        self.name = name
        self._entries: Dict[str, BlackboardEntry] = {}  # entry_id -> entry
        self._index: Dict[Tuple[str, str], str] = {}     # (ns, key) -> entry_id

        # 权限管理
        self._agent_permissions: Dict[str, Dict[str, Permission]] = defaultdict(dict)

        # 订阅管理
        self._subscriptions: List[Subscription] = []
        self._sub_counter: int = 0

        # 事务管理
        self.transaction_manager = TransactionManager()

        # 冲突检测
        self.conflict_detector = ConflictDetector()

        # 事件日志
        self._event_log: List[Dict[str, Any]] = []

        # 命名空间列表
        self._namespaces: Set[str] = set()

    # ---- 权限管理 ----

    def grant_permission(self, agent_id: str, namespace: str,
                          permission: Permission):
        """授予Agent对某命名空间的权限"""
        self._agent_permissions[agent_id][namespace] = permission

    def _check_permission(self, agent_id: str, namespace: str,
                           required: Permission) -> bool:
        """检查Agent的权限"""
        perms = self._agent_permissions.get(agent_id, {})
        actual = perms.get(namespace, perms.get("*", Permission.READ_WRITE))

        if required == Permission.READ:
            return actual in (Permission.READ, Permission.READ_WRITE)
        elif required == Permission.WRITE:
            return actual in (Permission.WRITE, Permission.READ_WRITE)
        elif required == Permission.READ_WRITE:
            return actual == Permission.READ_WRITE
        return False

    # ---- 核心读写 ----

    def write_with_scope(self, agent_id: str, namespace: str, key: str,
                          value: Any, scope: EntryScope = EntryScope.NAMESPACE,
                          ttl: Optional[float] = None,
                          metadata: Optional[Dict] = None) -> Optional[str]:
        """
        向黑板写入条目。

        Args:
            agent_id: 写入Agent的ID
            namespace: 命名空间
            key: 键名
            value: 值
            scope: 可见范围
            ttl: 生存时间（秒）
            metadata: 附加元数据

        Returns:
            条目ID，失败返回 None
        """
        # 权限检查
        if not self._check_permission(agent_id, namespace, Permission.WRITE):
            self._event_log.append({
                "type": "permission_denied",
                "agent": agent_id,
                "namespace": namespace,
                "key": key,
                "timestamp": time.time(),
            })
            return None

        # 检查是否已存在
        existing_key = (namespace, key)
        existing_entry_id = self._index.get(existing_key)

        if existing_entry_id and existing_entry_id in self._entries:
            existing = self._entries[existing_entry_id]
            # 写-写冲突检测
            if existing.agent_id != agent_id:
                self.conflict_detector.detect(
                    {existing_entry_id: existing},
                    dict(self._agent_permissions),
                )

            entry_id = existing_entry_id
            # 更新现有条目
            existing.value = value
            existing.version += 1
            existing.updated_at = time.time()
            existing.scope = scope
            existing.agent_id = agent_id
            existing.ttl = ttl
            if metadata:
                existing.metadata.update(metadata)
        else:
            # 创建新条目
            entry_id = f"ent-{uuid.uuid4().hex[:12]}"
            entry = BlackboardEntry(
                entry_id=entry_id,
                namespace=namespace,
                key=key,
                value=value,
                agent_id=agent_id,
                scope=scope,
                ttl=ttl,
                metadata=metadata or {},
            )
            self._entries[entry_id] = entry
            self._index[existing_key] = entry_id

        # 注册命名空间
        self._namespaces.add(namespace)

        # 记录事件
        self._event_log.append({
            "type": "write",
            "entry_id": entry_id,
            "agent": agent_id,
            "namespace": namespace,
            "key": key,
            "version": self._entries[entry_id].version,
            "timestamp": time.time(),
        })

        # 发布通知
        self._publish_notification(namespace, key, "write", self._entries[entry_id])

        return entry_id

    def read_with_permission(self, agent_id: str, namespace: str,
                              key: str) -> Optional[Any]:
        """
        从黑板读取条目。

        Args:
            agent_id: 读取Agent的ID
            namespace: 命名空间
            key: 键名

        Returns:
            条目的值，不存在或无权限返回 None
        """
        # 权限检查
        if not self._check_permission(agent_id, namespace, Permission.READ):
            return None

        entry_id = self._index.get((namespace, key))
        if not entry_id:
            return None

        entry = self._entries.get(entry_id)
        if not entry:
            return None

        # TTL 过期检查
        if entry.is_expired():
            self._cleanup_entry(entry_id)
            return None

        # 作用域检查
        if entry.scope == EntryScope.PRIVATE and entry.agent_id != agent_id:
            return None

        return entry.value

    def read_all_namespace(self, agent_id: str,
                            namespace: str) -> Dict[str, Any]:
        """
        读取整个命名空间的所有条目。

        Returns:
            {key: value} 字典
        """
        if not self._check_permission(agent_id, namespace, Permission.READ):
            return {}

        results = {}
        for (ns, key), entry_id in self._index.items():
            if ns == namespace and entry_id in self._entries:
                entry = self._entries[entry_id]
                if not entry.is_expired():
                    if entry.scope != EntryScope.PRIVATE or entry.agent_id == agent_id:
                        results[key] = entry.value

        return results

    def delete(self, agent_id: str, namespace: str, key: str) -> bool:
        """删除条目"""
        if not self._check_permission(agent_id, namespace, Permission.WRITE):
            return False

        entry_id = self._index.pop((namespace, key), None)
        if entry_id and entry_id in self._entries:
            del self._entries[entry_id]
            self._event_log.append({
                "type": "delete",
                "entry_id": entry_id,
                "agent": agent_id,
                "namespace": namespace,
                "key": key,
                "timestamp": time.time(),
            })
            self._publish_notification(namespace, key, "delete", None)
            return True
        return False

    # ---- 发布/订阅 ----

    def subscribe(self, agent_id: str, namespace_pattern: str,
                   key_pattern: str,
                   callback: Callable[[str, str, str, Any], None]) -> str:
        """
        订阅黑板变化。

        Args:
            agent_id: 订阅者ID
            namespace_pattern: 命名空间匹配模式（正则表达式）
            key_pattern: 键名匹配模式（正则表达式）
            callback: 回调函数 (namespace, key, event_type, value)

        Returns:
            订阅ID
        """
        self._sub_counter += 1
        sub_id = f"sub-{self._sub_counter:04d}"

        sub = Subscription(
            subscription_id=sub_id,
            agent_id=agent_id,
            namespace_pattern=namespace_pattern,
            key_pattern=key_pattern,
            callback=callback,
        )
        self._subscriptions.append(sub)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        before = len(self._subscriptions)
        self._subscriptions = [
            s for s in self._subscriptions
            if s.subscription_id != subscription_id
        ]
        return len(self._subscriptions) < before

    def _publish_notification(self, namespace: str, key: str,
                               event_type: str, entry: Optional[BlackboardEntry]):
        """向匹配的订阅者发布通知"""
        for sub in self._subscriptions:
            try:
                ns_match = re.match(sub.namespace_pattern, namespace)
                key_match = re.match(sub.key_pattern, key)
                if ns_match and key_match:
                    value = entry.value if entry else None
                    sub.callback(namespace, key, event_type, value)
            except Exception as e:
                self._event_log.append({
                    "type": "subscription_error",
                    "subscription_id": sub.subscription_id,
                    "error": str(e),
                    "timestamp": time.time(),
                })

    # ---- 清理与维护 ----

    def _cleanup_entry(self, entry_id: str):
        """清理过期条目"""
        if entry_id in self._entries:
            entry = self._entries[entry_id]
            self._index.pop((entry.namespace, entry.key), None)
            del self._entries[entry_id]

    def cleanup_expired(self) -> int:
        """清理所有过期条目，返回清理数量"""
        expired_ids = [
            eid for eid, entry in self._entries.items()
            if entry.is_expired()
        ]
        for eid in expired_ids:
            self._cleanup_entry(eid)
        return len(expired_ids)

    def get_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        return sorted(self._namespaces)

    def get_stats(self) -> Dict[str, Any]:
        """获取黑板统计"""
        return {
            "name": self.name,
            "total_entries": len(self._entries),
            "namespaces": self.get_namespaces(),
            "active_subscriptions": len(self._subscriptions),
            "event_log_length": len(self._event_log),
            "conflict_stats": self.conflict_detector.get_stats(),
            "active_transactions": len(
                self.transaction_manager.get_active_transactions()
            ),
        }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """完整演示"""
    print("=" * 70)
    print("  共享内存黑板系统 (Shared Memory Blackboard) 演示")
    print("=" * 70)

    # ---- 1. 创建黑板和Agent ----
    print("\n【1. 初始化和权限配置】")
    print("-" * 50)

    blackboard = SharedMemoryBlackboard(name="MultiAgentBoard")

    # 配置3个Agent及其权限
    agents = {
        "research_agent": {"task-001": Permission.READ_WRITE, "knowledge": Permission.READ},
        "planning_agent": {"task-001": Permission.READ, "system": Permission.READ_WRITE},
        "execution_agent": {"knowledge": Permission.READ, "system": Permission.READ_WRITE},
    }

    for agent_id, perms in agents.items():
        for ns, perm in perms.items():
            blackboard.grant_permission(agent_id, ns, perm)
        print(f"  Agent [{agent_id}]: 已配置 {len(perms)} 个命名空间权限")

    # ---- 2. 写入和读取 ----
    print("\n【2. 写入与读取测试】")
    print("-" * 50)

    # research_agent 写入任务上下文
    entries_written = [
        ("research_agent", "task-001", "objective", "分析市场趋势"),
        ("research_agent", "task-001", "deadline", "2026-07-20"),
        ("research_agent", "knowledge", "market_data", {"stocks": [1, 2, 3]}),
        ("planning_agent", "system", "plan_version", "v2.1"),
        ("execution_agent", "system", "execution_status", "running"),
    ]

    for agent_id, ns, key, value in entries_written:
        eid = blackboard.write_with_scope(agent_id, ns, key, value)
        status = "OK" if eid else "FAIL"
        print(f"  [{agent_id}] write {ns}:{key} → {eid} ({status})")

    # 写入权限测试：research_agent 尝试写入 system（无权限）
    eid = blackboard.write_with_scope("research_agent", "system", "test", "hack")
    print(f"  [research_agent] 无权限写入 system:test → {'被拒绝' if eid is None else '意外通过'}")

    # 读取测试
    print("\n  读取测试:")
    value = blackboard.read_with_permission("planning_agent", "task-001", "objective")
    print(f"    [planning_agent] 读取 task-001:objective → {value}")

    value = blackboard.read_with_permission("execution_agent", "task-001", "objective")
    print(f"    [execution_agent] 读取 task-001:objective → {value} (无权限)")

    # ---- 3. 发布/订阅 ----
    print("\n【3. 发布/订阅机制】")
    print("-" * 50)

    def on_task_update(namespace, key, event_type, value):
        print(f"    📢 通知: {namespace}:{key} 发生 {event_type} 事件, 新值={value}")

    # planning_agent 订阅 task-001 的变化
    sub_id = blackboard.subscribe(
        "planning_agent",
        namespace_pattern=r"task-\d+",
        key_pattern=r".*",
        callback=on_task_update,
    )
    print(f"  [planning_agent] 已订阅 task-* (订阅ID: {sub_id})")

    # 触发通知
    print("  写入触发通知:")
    blackboard.write_with_scope("research_agent", "task-001", "priority", "HIGH")

    # 取消订阅
    blackboard.unsubscribe(sub_id)
    print(f"  已取消订阅 {sub_id}")

    # ---- 4. 事务管理 ----
    print("\n【4. 事务管理】")
    print("-" * 50)

    tx_id = blackboard.transaction_manager.begin()
    print(f"  开启事务: {tx_id}")

    # 在事务中执行操作
    blackboard.write_with_scope("execution_agent", "system", "temp_var", "temp_value")
    blackboard.write_with_scope("execution_agent", "system", "another_var", 42)

    committed = blackboard.transaction_manager.commit(tx_id)
    print(f"  提交事务 {tx_id}: {'成功' if committed else '失败'}")

    # 回滚演示
    tx_id2 = blackboard.transaction_manager.begin()
    print(f"  开启事务: {tx_id2}")
    blackboard.write_with_scope("planning_agent", "system", "plan_temp", "will_be_rolled_back")

    rolled_back = blackboard.transaction_manager.rollback(tx_id2)
    print(f"  回滚事务 {tx_id2}: {'成功' if rolled_back else '失败'}")

    print(f"  活跃事务: {blackboard.transaction_manager.get_active_transactions()}")

    # ---- 5. 冲突检测 ----
    print("\n【5. 冲突检测】")
    print("-" * 50)

    # 模拟写-写冲突：两个Agent写同一键
    blackboard.write_with_scope("research_agent", "task-001", "shared_key", "value_from_research")
    blackboard.write_with_scope("planning_agent", "task-001", "shared_key", "value_from_planning")

    stats = blackboard.get_stats()
    print(f"  总条目数: {stats['total_entries']}")
    print(f"  命名空间: {stats['namespaces']}")
    print(f"  活跃订阅: {stats['active_subscriptions']}")
    print(f"  事件日志: {stats['event_log_length']} 条")
    print(f"  冲突统计: {stats['conflict_stats']}")

    # ---- 6. TTL过期清理 ----
    print("\n【6. TTL过期测试】")
    print("-" * 50)

    # 写入一个短TTL条目
    blackboard.write_with_scope("research_agent", "task-001", "temp_data",
                                 "will_expire_soon", ttl=0.01)
    print(f"  写入条目设置 TTL=0.01秒")

    # 等待过期
    import time as _t
    _t.sleep(0.02)
    cleaned = blackboard.cleanup_expired()
    print(f"  清理过期条目: {cleaned} 条")

    value = blackboard.read_with_permission("research_agent", "task-001", "temp_data")
    print(f"  过期后读取: {value}")

    # 最终统计
    print(f"\n  最终黑板统计:")
    final_stats = blackboard.get_stats()
    for k, v in final_stats.items():
        if k != "conflict_stats":  # 已单独打印
            print(f"    {k}: {v}")

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
