"""
Memory Manager - 记忆管理器
============================
统一的记忆管理、淘汰策略、持久化
"""

import time
import math
import json
import os
import pickle
from typing import Any, Optional
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class MemoryEntry:
    """记忆条目"""
    key: str
    content: Any
    importance: float = 1.0  # 重要性分数 0.0-10.0
    access_count: int = 0   # 访问次数
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class EvictionPolicy:
    """
    淘汰策略：LRU + 重要性评分

    评分公式: score = recency * 0.6 + frequency * 0.4
    - recency: 指数衰减，30天半衰期
    - frequency: 归一化访问次数
    """

    HALF_LIFE_DAYS = 30.0  # 半衰期(天)

    @classmethod
    def calculate_recency(cls, entry: MemoryEntry, current_time: float) -> float:
        """
        指数衰减计算recency
        R(t) = R_0 * (1/2)^(t / half_life)
        其中 t 是自上次访问以来的时间(天)
        """
        time_diff_days = (current_time - entry.last_accessed) / (24 * 3600)
        decay = math.pow(0.5, time_diff_days / cls.HALF_LIFE_DAYS)
        return decay

    @classmethod
    def calculate_score(cls, entry: MemoryEntry, max_frequency: int,
                        current_time: float) -> float:
        """
        综合评分
        score = recency * 0.6 + frequency_normalized * 0.4
        """
        recency = cls.calculate_recency(entry, current_time)
        freq_normalized = entry.access_count / max(1, max_frequency)
        score = recency * 0.6 + freq_normalized * 0.4
        entry.importance = score  # 更新importance为当前计算值
        return score


class MemoryManager:
    """
    统一的记忆管理器

    功能：
    - 支持多种记忆类型（短期、长期、情景、工作记忆）
    - 每个类型独立的max_entries限制
    - LRU + 重要性评分淘汰
    - 序列化持久化（pickle + JSON）
    - 备份和恢复
    - 统计和监控
    """

    def __init__(self, config: dict = None):
        """
        config = {
            "short_term": {"max_entries": 50},
            "long_term": {"max_entries": 500},
            "episodic": {"max_entries": 100},
            "working": {"max_entries": 10},
        }
        """
        self.config = config or {
            "short_term": {"max_entries": 50},
            "long_term": {"max_entries": 500},
            "episodic": {"max_entries": 100},
            "working": {"max_entries": 10},
        }
        self.stores: dict[str, OrderedDict] = {
            "short_term": OrderedDict(),
            "long_term": OrderedDict(),
            "episodic": OrderedDict(),
            "working": OrderedDict(),
        }
        self._access_stats: dict = {}
        self._total_puts: int = 0
        self._total_gets: int = 0
        self._total_evictions: int = 0

    # ============================================================
    # === CRUD 操作 ===
    # ============================================================

    def put(self, store: str, key: str, content: Any,
            importance: float = 1.0, metadata: dict = None) -> str:
        """
        存储记忆条目
        - 自动淘汰如果超过max_entries
        - 返回存储的key
        """
        if store not in self.stores:
            raise ValueError(f"不支持的存储类型: '{store}'，支持: {list(self.stores.keys())}")

        current_time = time.time()

        # 如果key已存在，更新
        if key in self.stores[store]:
            entry = self.stores[store][key]
            entry.content = content
            entry.last_accessed = current_time
            if metadata:
                entry.metadata.update(metadata)
            # 移动到末尾（最近使用）
            self.stores[store].move_to_end(key)
        else:
            entry = MemoryEntry(
                key=key,
                content=content,
                importance=importance,
                access_count=0,
                created_at=current_time,
                last_accessed=current_time,
                metadata=metadata or {},
            )
            self.stores[store][key] = entry

        self._total_puts += 1

        # 检查容量，自动淘汰
        self._ensure_capacity(store)

        return key

    def get(self, store: str, key: str) -> Optional[MemoryEntry]:
        """获取记忆条目，更新访问统计"""
        if store not in self.stores:
            raise ValueError(f"不支持的存储类型: '{store}'")

        if key not in self.stores[store]:
            return None

        entry = self.stores[store][key]
        entry.access_count += 1
        entry.last_accessed = time.time()

        # 更新访问统计
        self._access_stats[key] = self._access_stats.get(key, 0) + 1
        self._total_gets += 1

        # 移动到末尾（LRU）
        self.stores[store].move_to_end(key)

        return entry

    def delete(self, store: str, key: str) -> bool:
        """删除记忆条目"""
        if store not in self.stores:
            raise ValueError(f"不支持的存储类型: '{store}'")

        if key not in self.stores[store]:
            return False

        del self.stores[store][key]
        self._access_stats.pop(key, None)
        return True

    def query(self, store: str,
              keyword: str = None,
              min_importance: float = None,
              since: float = None,
              limit: int = 10) -> list:
        """查询记忆条目，支持多条件过滤"""
        if store not in self.stores:
            raise ValueError(f"不支持的存储类型: '{store}'")

        results = []
        current_time = time.time()

        for entry in self.stores[store].values():
            matches = True

            # 关键词过滤
            if keyword:
                content_str = str(entry.content).lower()
                meta_str = str(entry.metadata).lower()
                kw_lower = keyword.lower()
                if kw_lower not in content_str and kw_lower not in meta_str:
                    matches = False

            # 重要性过滤
            if min_importance is not None and matches:
                if entry.importance < min_importance:
                    matches = False

            # 时间过滤
            if since is not None and matches:
                if entry.created_at < since:
                    matches = False

            if matches:
                results.append(entry)

        # 按重要性降序排序
        results.sort(key=lambda e: e.importance, reverse=True)

        return results[:limit]

    # ============================================================
    # === 淘汰管理 ===
    # ============================================================

    def evict(self, store: str, count: int = 1) -> list:
        """
        淘汰最低分条目
        1. 计算所有条目的score
        2. 排序找出最低分
        3. 删除并返回被淘汰的key列表
        """
        if store not in self.stores:
            raise ValueError(f"不支持的存储类型: '{store}'")

        store_data = self.stores[store]
        if len(store_data) <= count:
            count = min(count, len(store_data))

        if count == 0:
            return []

        current_time = time.time()

        # 计算所有条目的最大访问频率
        max_freq = 1
        for entry in store_data.values():
            if entry.access_count > max_freq:
                max_freq = entry.access_count

        # 计算每个条目的分数
        scored = []
        for key, entry in store_data.items():
            score = EvictionPolicy.calculate_score(entry, max_freq, current_time)
            scored.append((score, key))

        # 按分数升序排序（最低分在前）
        scored.sort(key=lambda x: x[0])

        # 淘汰最低分的count个条目
        evicted_keys = []
        for score, key in scored[:count]:
            del store_data[key]
            self._access_stats.pop(key, None)
            evicted_keys.append(key)
            self._total_evictions += 1

        return evicted_keys

    def _ensure_capacity(self, store: str) -> None:
        """确保不超过容量限制，必要时自动淘汰"""
        max_entries = self._get_max_entries(store)
        current_count = len(self.stores[store])

        if current_count > max_entries:
            excess = current_count - max_entries
            self.evict(store, count=excess)

    def _get_max_entries(self, store: str) -> int:
        """获取某类型记忆的最大条目数"""
        store_config = self.config.get(store, {})
        return store_config.get("max_entries", 100)

    # ============================================================
    # === 持久化 ===
    # ============================================================

    def save(self, filepath: str) -> None:
        """序列化保存到文件（使用pickle）"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        save_data = {
            "config": self.config,
            "stores": {},
            "access_stats": self._access_stats,
            "total_puts": self._total_puts,
            "total_gets": self._total_gets,
            "total_evictions": self._total_evictions,
            "timestamp": time.time(),
        }

        for store_name, store_data in self.stores.items():
            save_data["stores"][store_name] = list(store_data.items())

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str) -> bool:
        """从文件加载记忆"""
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            self.config = save_data.get("config", self.config)
            self._access_stats = save_data.get("access_stats", {})
            self._total_puts = save_data.get("total_puts", 0)
            self._total_gets = save_data.get("total_gets", 0)
            self._total_evictions = save_data.get("total_evictions", 0)

            # 恢复各存储
            for store_name, items in save_data.get("stores", {}).items():
                if store_name in self.stores:
                    self.stores[store_name] = OrderedDict(items)

            return True
        except Exception as e:
            print(f"加载失败: {e}")
            return False

    def export_json(self, filepath: str) -> None:
        """导出为JSON格式（仅内容，不含运行时状态）"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        export_data = {
            "config": self.config,
            "stores": {},
            "timestamp": time.time(),
            "export_format": "json_content_only",
        }

        for store_name, store_data in self.stores.items():
            entries = []
            for key, entry in store_data.items():
                entries.append({
                    "key": key,
                    "content": str(entry.content),
                    "importance": entry.importance,
                    "access_count": entry.access_count,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "metadata": entry.metadata,
                })
            export_data["stores"][store_name] = entries

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def import_json(self, filepath: str) -> None:
        """从JSON导入"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            import_data = json.load(f)

        for store_name, entries in import_data.get("stores", {}).items():
            if store_name not in self.stores:
                continue

            self.stores[store_name].clear()
            for entry_data in entries:
                key = entry_data["key"]
                entry = MemoryEntry(
                    key=key,
                    content=entry_data["content"],
                    importance=entry_data.get("importance", 1.0),
                    access_count=entry_data.get("access_count", 0),
                    created_at=entry_data.get("created_at", time.time()),
                    last_accessed=entry_data.get("last_accessed", time.time()),
                    metadata=entry_data.get("metadata", {}),
                )
                self.stores[store_name][key] = entry

    # ============================================================
    # === 统计 ===
    # ============================================================

    def stats(self) -> dict:
        """全面的统计信息"""
        store_stats = {}
        for store_name, store_data in self.stores.items():
            entries = list(store_data.values())
            store_stats[store_name] = {
                "count": len(entries),
                "max_entries": self._get_max_entries(store_name),
                "usage_percent": round(
                    len(entries) / max(1, self._get_max_entries(store_name)) * 100, 1
                ),
                "avg_importance": round(
                    sum(e.importance for e in entries) / max(1, len(entries)), 2
                ),
                "total_accesses": sum(e.access_count for e in entries),
                "oldest_age_days": round(
                    (time.time() - min((e.created_at for e in entries), default=time.time()))
                    / 86400, 1
                ),
                "newest_age_days": round(
                    (time.time() - max((e.created_at for e in entries), default=time.time()))
                    / 86400, 1
                ),
            }

        total_entries = sum(s["count"] for s in store_stats.values())
        total_capacity = sum(s["max_entries"] for s in store_stats.values())

        return {
            "stores": store_stats,
            "summary": {
                "total_entries": total_entries,
                "total_capacity": total_capacity,
                "overall_usage": round(
                    total_entries / max(1, total_capacity) * 100, 1
                ),
                "total_puts": self._total_puts,
                "total_gets": self._total_gets,
                "total_evictions": self._total_evictions,
                "hit_rate": round(
                    self._total_gets / max(1, self._total_puts) * 100, 1
                ),
            },
        }

    def get_store_keys(self, store: str) -> list:
        """获取某个存储的所有key"""
        if store not in self.stores:
            raise ValueError(f"不支持的存储类型: '{store}'")
        return list(self.stores[store].keys())


# ============================================================
# === 演示示例 ===
# ============================================================
def demo_memory_manager():
    """完整演示记忆管理器的所有功能"""
    print("=" * 70)
    print("  记忆管理器 (Memory Manager) - 完整演示")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. 初始化
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. 初始化记忆管理器")
    print("=" * 50)

    mm = MemoryManager({
        "short_term": {"max_entries": 20},
        "long_term": {"max_entries": 50},
        "episodic": {"max_entries": 10},
        "working": {"max_entries": 5},
    })

    print(f"  配置: {json.dumps(mm.config, ensure_ascii=False, indent=2)}")

    # ---------------------------------------------------------
    # 2. CRUD 操作演示
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. CRUD 操作演示")
    print("=" * 50)

    # Put - 添加长期记忆
    print("\n  [PUT] 添加长期记忆条目...")
    knowledge_items = [
        ("python_intro", "Python是一种解释型高级编程语言", 8.0),
        ("ml_basics", "机器学习是AI的重要分支", 7.0),
        ("dl_intro", "深度学习使用多层神经网络", 8.5),
        ("nlp_overview", "自然语言处理研究计算机与人类语言的交互", 7.5),
        ("rl_basics", "强化学习通过奖励机制训练智能体", 6.0),
        ("cv_basics", "计算机视觉让机器理解和处理图像", 7.0),
        ("transformer", "Transformer架构通过自注意力机制处理序列", 9.0),
        ("gpt_model", "GPT是自回归语言模型基于Transformer架构", 8.5),
        ("agent_concept", "AI Agent能自主感知环境并执行任务", 9.5),
        ("rag_system", "RAG结合检索和生成提升LLM回答质量", 9.0),
    ]

    for key, content, importance in knowledge_items:
        mm.put("long_term", key, content, importance=importance)
        print(f"    [+] 长期记忆: {key} (重要性: {importance})")

    # Put - 添加工作记忆
    print("\n  [PUT] 添加工作记忆...")
    mm.put("working", "current_task", "正在演示记忆管理器功能", importance=5.0)
    mm.put("working", "focus", "CRUD操作", importance=8.0)
    mm.put("working", "state", "正常", importance=3.0)
    print(f"    当前工作记忆keys: {mm.get_store_keys('working')}")

    # Get
    print("\n  [GET] 获取记忆条目...")
    entry = mm.get("long_term", "agent_concept")
    if entry:
        print(f"    agent_concept: {entry.content}")
        print(f"    重要性: {entry.importance}, 访问次数: {entry.access_count}")

    # 再次获取，验证访问计数
    entry = mm.get("long_term", "agent_concept")
    print(f"    再次获取后访问次数: {entry.access_count}")

    # 获取不存在的key
    entry = mm.get("long_term", "nonexistent")
    print(f"    获取不存在的key: {entry}")

    # Query - 关键字查询
    print("\n  [QUERY] 关键字查询...")
    results = mm.query("long_term", keyword="学习", limit=5)
    print(f"    查询'学习'的结果({len(results)}条):")
    for e in results:
        print(f"      - {e.key}: {str(e.content)[:50]}...")

    # 重要性过滤
    print("\n  [QUERY] 重要性过滤 (>= 8.0)...")
    results = mm.query("long_term", min_importance=8.0, limit=10)
    print(f"    高重要性记忆({len(results)}条):")
    for e in results:
        print(f"      - {e.key} (重要性: {e.importance}): {str(e.content)[:50]}...")

    # Delete
    print("\n  [DELETE] 删除记忆条目...")
    deleted = mm.delete("working", "state")
    print(f"    删除 'state': {'成功' if deleted else '失败'}")
    print(f"    工作记忆剩余keys: {mm.get_store_keys('working')}")

    # ---------------------------------------------------------
    # 3. 淘汰策略演示
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. 淘汰策略演示 (LRU + 重要性评分)")
    print("=" * 50)

    # 创建一个容量很小的存储用于演示淘汰
    mm_small = MemoryManager({
        "test": {"max_entries": 5},
    })

    print("\n  添加6个条目到容量为5的存储中...")
    for i in range(6):
        mm_small.put("test", f"item_{i}", f"内容_{i}", importance=float(i + 1))

    print(f"  当前条目: {mm_small.get_store_keys('test')}")

    # 模拟访问以改变评分
    print("\n  多次访问 item_0 和 item_1 以提高它们的评分...")
    for _ in range(10):
        mm_small.get("test", "item_0")
    for _ in range(5):
        mm_small.get("test", "item_1")
    # item_2 item_3 item_4 保持低访问

    # 添加新条目触发淘汰
    print("\n  添加新条目触发淘汰...")
    mm_small.put("test", "item_new", "新内容", importance=3.0)
    print(f"  淘汰后条目: {mm_small.get_store_keys('test')}")

    # 手动淘汰最差的2个
    print("\n  手动淘汰2个最低分条目...")
    evicted = mm_small.evict("test", count=2)
    print(f"  被淘汰: {evicted}")
    print(f"  剩余条目: {mm_small.get_store_keys('test')}")

    # ---------------------------------------------------------
    # 4. 指数衰减演示
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("4. 指数衰减计算演示")
    print("=" * 50)

    now = time.time()
    # 模拟不同时间前的访问
    for days_ago, label in [(0, "刚刚"), (1, "1天前"), (7, "1周前"), (30, "30天前"), (90, "3个月前")]:
        entry = MemoryEntry(
            key="test",
            content="test",
            created_at=now - days_ago * 86400,
            last_accessed=now - days_ago * 86400,
            access_count=5,
        )
        recency = EvictionPolicy.calculate_recency(entry, now)
        score = EvictionPolicy.calculate_score(entry, max_frequency=10, current_time=now)
        print(f"    {label:8s}: recency={recency:.4f}, score={score:.4f}")

    # ---------------------------------------------------------
    # 5. 持久化演示
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("5. 持久化演示 (保存/加载/导出JSON)")
    print("=" * 50)

    save_path = "memory_manager_demo.pkl"
    json_path = "memory_manager_demo.json"

    # 清空测试文件
    for p in [save_path, json_path]:
        if os.path.exists(p):
            os.remove(p)

    # 保存
    print(f"\n  保存到文件: {save_path}")
    mm.save(save_path)
    file_size = os.path.getsize(save_path)
    print(f"    文件大小: {file_size} bytes")

    # 加载到新实例
    print(f"\n  从文件加载: {save_path}")
    mm_loaded = MemoryManager()
    success = mm_loaded.load(save_path)
    if success:
        print(f"    加载成功!")
        loaded_stats = mm_loaded.stats()
        print(f"    加载后条目总数: {loaded_stats['summary']['total_entries']}")
        print(f"    长期记忆数: {len(mm_loaded.get_store_keys('long_term'))}")

    # JSON导出
    print(f"\n  导出JSON: {json_path}")
    mm.export_json(json_path)
    print(f"    JSON文件大小: {os.path.getsize(json_path)} bytes")

    # JSON导入
    print(f"\n  从JSON导入:")
    mm_imported = MemoryManager()
    mm_imported.import_json(json_path)
    imported_stats = mm_imported.stats()
    print(f"    导入后条目总数: {imported_stats['summary']['total_entries']}")

    # 清理
    for p in [save_path, json_path]:
        if os.path.exists(p):
            os.remove(p)

    print(f"\n  清理临时文件: {save_path}, {json_path}")

    # ---------------------------------------------------------
    # 6. 统计信息演示
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("6. 完整统计信息")
    print("=" * 50)

    stats = mm.stats()

    print(f"\n  各存储统计:")
    for store_name, store_stat in stats["stores"].items():
        print(f"    [{store_name}]")
        for k, v in store_stat.items():
            print(f"      {k}: {v}")

    print(f"\n  总体统计:")
    for k, v in stats["summary"].items():
        print(f"    {k}: {v}")

    # ---------------------------------------------------------
    # 7. 综合场景演示
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("7. 综合场景: 模拟Agent会话记忆管理")
    print("=" * 50)

    agent_mm = MemoryManager({
        "short_term": {"max_entries": 15},
        "long_term": {"max_entries": 30},
        "episodic": {"max_entries": 5},
        "working": {"max_entries": 3},
    })

    # 模拟多轮对话
    print("\n  模拟Agent多轮对话...")
    conversation = [
        ("user", "帮我查询北京天气"),
        ("assistant", "正在查询北京天气... 结果: 晴，22°C"),
        ("user", "那上海的天气呢？"),
        ("assistant", "上海天气: 多云，28°C"),
        ("user", "分析一下这两天的温差"),
        ("assistant", "北京和上海温差6°C，上海更暖和"),
        ("user", "帮我记录：以后每天都要查询上海天气"),
        ("assistant", "已记录用户偏好：每日查询上海天气"),
    ]

    for i, (role, content) in enumerate(conversation):
        agent_mm.put(
            "short_term",
            f"msg_{i:02d}",
            {"role": role, "content": content},
            importance=5.0,
            metadata={"turn": i, "role": role},
        )

        # 提取重要信息存入长期记忆
        if "记录" in content or "偏好" in content:
            agent_mm.put(
                "long_term",
                f"preference_user_{i}",
                f"用户偏好: {content}",
                importance=9.0,
                metadata={"type": "user_preference", "turn": i},
            )

    # 添加情景记忆
    agent_mm.put(
        "episodic",
        "session_001",
        {
            "task": "天气查询和偏好设置",
            "turns": len(conversation),
            "tools_used": ["weather_query"],
            "success": True,
        },
        importance=7.0,
        metadata={"session_id": "session_001", "duration": "5分钟"},
    )

    print(f"    对话轮次: {len(conversation)}")
    print(f"    短期记忆条目: {len(agent_mm.get_store_keys('short_term'))}")
    print(f"    长期记忆条目: {len(agent_mm.get_store_keys('long_term'))}")
    print(f"    情景记忆条目: {len(agent_mm.get_store_keys('episodic'))}")

    # 查询用户偏好
    results = agent_mm.query("long_term", keyword="偏好", limit=5)
    print(f"\n  查询用户偏好:")
    for e in results:
        print(f"     [{e.key}] {e.content}")

    # 最终统计
    final_stats = agent_mm.stats()
    print(f"\n  最终统计:")
    print(f"    总条目: {final_stats['summary']['total_entries']}")
    print(f"    总容量: {final_stats['summary']['total_capacity']}")
    print(f"    使用率: {final_stats['summary']['overall_usage']}%")
    print(f"    命中率: {final_stats['summary']['hit_rate']}%")

    print("\n" + "=" * 70)
    print("  记忆管理器演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo_memory_manager()
