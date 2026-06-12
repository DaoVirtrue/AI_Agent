#!/usr/bin/env python3
"""
Phase 07 - 多智能体系统：05 - 黑板模式（Blackboard Pattern）

共享黑板（Blackboard）数据结构，多个Agent独立读写。
线程安全（threading.Lock），冲突检测与解决。
示例：旅行规划（航班Agent、酒店Agent、活动Agent → 共享行程表）

架构图：
  ┌──────────────────────────────────────────────┐
  │                 Blackboard                     │
  │  ┌─────────────────────────────────────────┐  │
  │  │  Shared Data (Thread-Safe Dict)         │  │
  │  │  - itinerary: {flights, hotels, ...}    │  │
  │  │  - conflicts: []                        │  │
  │  │  - constraints: {budget, dates, ...}    │  │
  │  │  - status: {flight: found, hotel: ...}  │  │
  │  └──────────┬──────────────────────────────┘  │
  │             │ Lock                              │
  └──────┬──────┴──────┬──────┬──────────────────┘
         │             │      │
    ┌────▼────┐   ┌───▼───┐ ┌▼──────────┐
    │ Flight  │   │Hotel  │ │Activity   │
    │ Agent   │   │Agent  │ │Agent      │
    │ 搜索航班 │   │搜索酒店│ │推荐活动   │
    └─────────┘   └───────┘ └───────────┘
         写入          写入         写入
         黑板          黑板         黑板
"""

import json
import time
import uuid
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock, RLock
from typing import Any, Callable, Optional, Union


# ============================================================================
# 黑板数据结构
# ============================================================================

@dataclass
class BlackboardEntry:
    """
    黑板上的一个数据条目。

    Attributes:
        key: 数据键
        value: 数据值
        agent_id: 写入此数据的Agent ID
        timestamp: 写入时间戳
        version: 版本号（用于冲突检测）
        tags: 标签
    """
    key: str
    value: Any
    agent_id: str
    timestamp: float = field(default_factory=time.time)
    version: int = 1
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "tags": self.tags,
        }


@dataclass
class ConflictRecord:
    """
    冲突记录。

    Attributes:
        conflict_id: 冲突ID
        key: 冲突的键
        entries: 冲突的条目列表
        resolvable: 是否可自动解决
        resolution: 解决方案描述
        resolved_by: 解决者（AgentID）
        resolved_at: 解决时间
    """
    conflict_id: str
    key: str
    entries: list[BlackboardEntry]
    resolvable: bool = True
    resolution: str = ""
    resolved_by: str = ""
    resolved_at: float = 0.0

    @property
    def is_resolved(self) -> bool:
        return self.resolved_at > 0


class Blackboard:
    """
    线程安全的黑板数据结构。

    特性：
    - 多Agent并发读写（RLock）
    - 写时版本号递增
    - 冲突检测（同key多Agent写入）
    - 条目历史追踪
    - 观察者通知
    """

    def __init__(self, name: str = "DefaultBlackboard"):
        self.name = name
        self._data: dict[str, BlackboardEntry] = {}
        self._history: list[BlackboardEntry] = []
        self._conflicts: list[ConflictRecord] = []
        self._observers: list[Callable] = []
        self._lock = RLock()  # 可重入锁

    # ---- 基本读写 ----

    def write(
        self,
        key: str,
        value: Any,
        agent_id: str,
        tags: list[str] = None,
    ) -> BlackboardEntry:
        """
        写入数据到黑板。

        Args:
            key: 数据键
            value: 数据值
            agent_id: 写入的Agent标识
            tags: 标签

        Returns:
            创建的条目
        """
        with self._lock:
            # 检查冲突：是否已有其他Agent写过同名key
            existing = self._data.get(key)
            if existing and existing.agent_id != agent_id:
                conflict = ConflictRecord(
                    conflict_id=f"CF-{uuid.uuid4().hex[:8]}",
                    key=key,
                    entries=[existing],
                )
                self._conflicts.append(conflict)

            # 计算新版本号
            version = (existing.version + 1) if existing else 1

            entry = BlackboardEntry(
                key=key,
                value=value,
                agent_id=agent_id,
                version=version,
                tags=tags or [],
            )

            self._data[key] = entry
            self._history.append(entry)

            # 通知观察者
            self._notify("write", entry)

            return entry

    def read(self, key: str) -> Optional[BlackboardEntry]:
        """
        读取黑板上的数据。

        Args:
            key: 数据键

        Returns:
            条目，如果不存在则返回None
        """
        with self._lock:
            entry = self._data.get(key)
            if entry:
                self._notify("read", entry)
            return entry

    def read_all(self, agent_id: str = None) -> dict[str, BlackboardEntry]:
        """
        读取黑板上的所有数据。
        可选按agent_id过滤。

        Args:
            agent_id: 过滤条件（可选）

        Returns:
            所有匹配的条目字典
        """
        with self._lock:
            if agent_id:
                return {
                    k: v for k, v in self._data.items()
                    if v.agent_id == agent_id
                }
            return dict(self._data)

    def delete(self, key: str, agent_id: str) -> bool:
        """
        删除黑板上的数据（仅允许写入者删除自己的数据）。

        Args:
            key: 数据键
            agent_id: 请求删除的Agent（必须与写入者相同）

        Returns:
            是否成功删除
        """
        with self._lock:
            entry = self._data.get(key)
            if entry and entry.agent_id == agent_id:
                del self._data[key]
                self._notify("delete", entry)
                return True
            return False

    # ---- 冲突管理 ----

    def get_conflicts(self, resolved: bool = None) -> list[ConflictRecord]:
        """获取冲突列表。"""
        with self._lock:
            if resolved is None:
                return list(self._conflicts)
            return [
                c for c in self._conflicts
                if c.is_resolved == resolved
            ]

    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        resolved_by: str,
        winning_entry_key: str = None,
    ) -> bool:
        """
        解决冲突。

        Args:
            conflict_id: 冲突ID
            resolution: 解决方案描述
            resolved_by: 解决者AgentID
            winning_entry_key: 如果选择一个胜者，指定其键

        Returns:
            是否成功解决
        """
        with self._lock:
            for conflict in self._conflicts:
                if conflict.conflict_id == conflict_id and not conflict.is_resolved:
                    conflict.resolution = resolution
                    conflict.resolved_by = resolved_by
                    conflict.resolved_at = time.time()

                    # 如果指定了胜者，删除其他条目
                    if winning_entry_key:
                        keys_to_remove = [
                            e.key for e in conflict.entries
                            if e.key != winning_entry_key
                        ]
                        for k in keys_to_remove:
                            if k in self._data:
                                del self._data[k]

                    return True
            return False

    # ---- 观察者模式 ----

    def observe(self, callback: Callable) -> Callable:
        """注册观察者回调函数。"""
        with self._lock:
            self._observers.append(callback)
        return callback

    def _notify(self, operation: str, entry: BlackboardEntry):
        """通知所有观察者。"""
        for observer in self._observers:
            try:
                observer(operation, entry)
            except Exception:
                pass  # 观察者异常不影响主流程

    # ---- 查询与统计 ----

    def get_history(self, limit: int = 50) -> list[BlackboardEntry]:
        """获取写入历史。"""
        with self._lock:
            return self._history[-limit:]

    def get_stats(self) -> dict:
        """获取黑板统计信息。"""
        with self._lock:
            agents = set(e.agent_id for e in self._data.values())
            return {
                "name": self.name,
                "total_entries": len(self._data),
                "total_history": len(self._history),
                "active_agents": list(agents),
                "conflicts_total": len(self._conflicts),
                "conflicts_unresolved": sum(
                    1 for c in self._conflicts if not c.is_resolved
                ),
                "keys": list(self._data.keys()),
            }

    def to_dict(self) -> dict:
        """导出完整状态为字典。"""
        with self._lock:
            return {
                "name": self.name,
                "data": {k: v.to_dict() for k, v in self._data.items()},
                "conflicts": [
                    {
                        "id": c.conflict_id,
                        "key": c.key,
                        "resolved": c.is_resolved,
                        "resolution": c.resolution,
                    }
                    for c in self._conflicts
                ],
            }

    def to_json(self, indent: int = 2) -> str:
        """导出为JSON字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


# ============================================================================
# BlackboardAgent 基类
# ============================================================================

class BlackboardAgent(ABC):
    """
    黑板Agent基类。

    继承此类并实现 execute() 方法。
    """

    def __init__(self, agent_id: str, blackboard: Blackboard):
        self.agent_id = agent_id
        self.blackboard = blackboard
        self.execution_count: int = 0
        self.total_time: float = 0.0

    @abstractmethod
    def execute(self, context: dict = None) -> dict:
        """
        执行Agent任务，读写黑板。

        Args:
            context: 额外上下文

        Returns:
            执行结果摘要
        """
        ...

    def write(self, key: str, value: Any, tags: list[str] = None) -> BlackboardEntry:
        """写入黑板（封装）。"""
        return self.blackboard.write(key, value, self.agent_id, tags)

    def read(self, key: str) -> Optional[BlackboardEntry]:
        """读取黑板（封装）。"""
        return self.blackboard.read(key)

    def read_all(self) -> dict[str, BlackboardEntry]:
        """读取所有数据（封装）。"""
        return self.blackboard.read_all()

    def _simulate_work(self, min_s: float = 0.05, max_s: float = 0.3):
        """模拟工作延迟。"""
        time.sleep(random.uniform(min_s, max_s))


# ============================================================================
# 旅行规划专用Agent
# ============================================================================

class FlightAgent(BlackboardAgent):
    """
    航班搜索Agent：搜索并写入可选航班信息到黑板。
    """

    def __init__(self, blackboard: Blackboard):
        super().__init__("flight_agent", blackboard)
        self.airlines = ["中国国际航空", "东方航空", "南方航空", "海南航空", "春秋航空"]

    def execute(self, context: dict = None) -> dict:
        """搜索航班。"""
        self._simulate_work(0.1, 0.3)
        self.execution_count += 1

        # 从黑板读取约束条件
        constraints = self.read("constraints")
        budget = constraints.value.get("budget", 5000) if constraints else 5000
        origin = constraints.value.get("origin", "北京") if constraints else "北京"
        destination = constraints.value.get("destination", "上海") if constraints else "上海"

        # 模拟搜索航班
        flights = []
        for i, airline in enumerate(self.airlines):
            price = random.randint(800, 2500)
            if price <= budget * 0.4:  # 航班预算不超过总预算40%
                flight = {
                    "airline": airline,
                    "flight_no": f"{airline[:2].upper()}{random.randint(100, 999)}",
                    "from": origin,
                    "to": destination,
                    "departure": "08:00" if i < 2 else "14:30",
                    "arrival": "10:30" if i < 2 else "17:00",
                    "price": price,
                    "currency": "CNY",
                    "stops": 0 if i < 3 else 1,
                }
                flights.append(flight)

        # 写入黑板
        self.write(
            "flights",
            flights,
            tags=["transportation", "flight", "search_result"],
        )

        return {
            "agent": self.agent_id,
            "flights_found": len(flights),
            "price_range": f"{min(f['price'] for f in flights)}-{max(f['price'] for f in flights)}"
            if flights else "N/A",
        }


class HotelAgent(BlackboardAgent):
    """
    酒店搜索Agent：搜索并写入酒店信息到黑板。
    """

    def __init__(self, blackboard: Blackboard):
        super().__init__("hotel_agent", blackboard)
        self.hotel_chains = ["万豪", "希尔顿", "洲际", "凯悦", "如家精选", "亚朵"]

    def execute(self, context: dict = None) -> dict:
        """搜索酒店。"""
        self._simulate_work(0.1, 0.3)
        self.execution_count += 1

        constraints = self.read("constraints")
        budget = constraints.value.get("budget", 5000) if constraints else 5000
        destination = constraints.value.get("destination", "上海") if constraints else "上海"
        checkin = constraints.value.get("checkin", "2025-06-15") if constraints else "2025-06-15"

        # 模拟搜索酒店
        hotels = []
        for chain in self.hotel_chains:
            price_per_night = random.randint(300, 1200)
            if price_per_night <= budget * 0.3:  # 酒店预算不超过总预算30%
                hotel = {
                    "name": f"{chain}酒店（{destination}中心店）",
                    "chain": chain,
                    "location": f"{destination}市中心",
                    "price_per_night": price_per_night,
                    "rating": round(random.uniform(4.0, 5.0), 1),
                    "amenities": random.sample(
                        ["WiFi", "早餐", "健身房", "游泳池", "商务中心", "洗衣服务"],
                        k=random.randint(3, 6),
                    ),
                    "distance_to_center_km": round(random.uniform(0.5, 5.0), 1),
                }
                hotels.append(hotel)

        # 写入黑板
        self.write(
            "hotels",
            hotels,
            tags=["accommodation", "hotel", "search_result"],
        )

        return {
            "agent": self.agent_id,
            "hotels_found": len(hotels),
            "price_range": f"{min(h['price_per_night'] for h in hotels)}-{max(h['price_per_night'] for h in hotels)}/晚"
            if hotels else "N/A",
        }


class ActivityAgent(BlackboardAgent):
    """
    活动/景点推荐Agent：推荐目的地活动。
    """

    def __init__(self, blackboard: Blackboard):
        super().__init__("activity_agent", blackboard)

    def execute(self, context: dict = None) -> dict:
        """推荐活动。"""
        self._simulate_work(0.05, 0.2)
        self.execution_count += 1

        constraints = self.read("constraints")
        destination = constraints.value.get("destination", "上海") if constraints else "上海"
        interests = constraints.value.get("interests", ["文化", "美食"]) if constraints else ["文化", "美食"]

        # 活动库
        all_activities = {
            "上海": [
                {"name": "外滩观景", "type": "观光", "price": 0, "duration": "2小时", "rating": 4.8},
                {"name": "豫园游览", "type": "文化", "price": 40, "duration": "2-3小时", "rating": 4.6},
                {"name": "上海博物馆", "type": "文化", "price": 0, "duration": "3小时", "rating": 4.7},
                {"name": "迪士尼乐园", "type": "娱乐", "price": 475, "duration": "全天", "rating": 4.5},
                {"name": "新天地美食漫步", "type": "美食", "price": 200, "duration": "3小时", "rating": 4.4},
                {"name": "田子坊文艺街区", "type": "文化", "price": 0, "duration": "2小时", "rating": 4.3},
                {"name": "浦江夜游", "type": "观光", "price": 180, "duration": "1.5小时", "rating": 4.6},
                {"name": "本帮菜体验", "type": "美食", "price": 150, "duration": "2小时", "rating": 4.5},
            ],
            "北京": [
                {"name": "故宫博物院", "type": "文化", "price": 60, "duration": "4小时", "rating": 4.9},
                {"name": "长城（八达岭）", "type": "观光", "price": 40, "duration": "半天", "rating": 4.8},
                {"name": "颐和园", "type": "观光", "price": 30, "duration": "3小时", "rating": 4.7},
                {"name": "簋街美食", "type": "美食", "price": 100, "duration": "2小时", "rating": 4.4},
            ],
        }

        city_activities = all_activities.get(destination, all_activities["上海"])

        # 根据兴趣过滤
        filtered = [
            a for a in city_activities
            if any(interest in a["type"] for interest in interests)
        ]
        if not filtered:
            filtered = city_activities

        # 写入黑板
        self.write(
            "activities",
            filtered,
            tags=["activities", "sightseeing", "recommendation"],
        )

        return {
            "agent": self.agent_id,
            "activities_found": len(filtered),
            "types": list(set(a["type"] for a in filtered)),
        }


class ItineraryPlanner(BlackboardAgent):
    """
    行程规划Agent：读取所有搜索结果，生成综合行程计划。
    """

    def __init__(self, blackboard: Blackboard):
        super().__init__("itinerary_planner", blackboard)

    def execute(self, context: dict = None) -> dict:
        """综合规划行程。"""
        self._simulate_work(0.1, 0.2)
        self.execution_count += 1

        # 读取所有Agent的结果
        all_data = self.read_all()

        flights = all_data.get("flights")
        hotels = all_data.get("hotels")
        activities = all_data.get("activities")
        constraints = all_data.get("constraints")

        # 检查数据完整性
        missing = []
        if not flights:
            missing.append("航班")
        if not hotels:
            missing.append("酒店")
        if not activities:
            missing.append("活动")

        if missing:
            self.write(
                "itinerary",
                {"status": "incomplete", "missing": missing},
                tags=["itinerary", "incomplete"],
            )
            return {"agent": self.agent_id, "status": "incomplete", "missing": missing}

        # 选择最优方案
        flights_list = flights.value if isinstance(flights.value, list) else []
        hotels_list = hotels.value if isinstance(hotels.value, list) else []
        activities_list = activities.value if isinstance(activities.value, list) else []

        best_flight = min(flights_list, key=lambda x: x["price"]) if flights_list else None
        best_hotel = max(hotels_list, key=lambda x: x["rating"]) if hotels_list else None
        top_activities = sorted(activities_list, key=lambda x: x["rating"], reverse=True)[:3]

        # 计算总费用
        total_cost = 0
        if best_flight:
            total_cost += best_flight["price"] * 2  # 往返
        if best_hotel:
            total_cost += best_hotel["price_per_night"] * 3  # 3晚
        if top_activities:
            total_cost += sum(a["price"] for a in top_activities)

        budget = constraints.value.get("budget", 5000) if constraints else 5000

        itinerary = {
            "status": "complete",
            "budget": budget,
            "total_estimated_cost": total_cost,
            "within_budget": total_cost <= budget,
            "flight": best_flight,
            "hotel": best_hotel,
            "top_activities": top_activities,
            "summary": self._generate_summary(
                best_flight, best_hotel, top_activities, total_cost, budget
            ),
            "generated_at": datetime.now().isoformat(),
        }

        self.write(
            "itinerary",
            itinerary,
            tags=["itinerary", "final", "complete"],
        )

        return {
            "agent": self.agent_id,
            "status": "complete",
            "total_cost": total_cost,
            "within_budget": total_cost <= budget,
        }

    def _generate_summary(
        self,
        flight: dict,
        hotel: dict,
        activities: list,
        total_cost: float,
        budget: float,
    ) -> str:
        """生成行程摘要文本。"""
        lines = [
            "=== 旅行行程规划 ===",
            f"预算: ¥{budget} | 预估总费用: ¥{total_cost} "
            f"{'✅ 在预算内' if total_cost <= budget else '⚠️ 超出预算'}",
            "",
        ]
        if flight:
            lines.append(
                f"✈️ 航班: {flight['airline']} {flight['flight_no']} "
                f"({flight['from']}→{flight['to']}) "
                f"¥{flight['price']}"
            )
        if hotel:
            lines.append(
                f"🏨 酒店: {hotel['name']} "
                f"评分{hotel['rating']} ¥{hotel['price_per_night']}/晚"
            )
        if activities:
            lines.append("🎯 推荐活动:")
            for a in activities:
                lines.append(f"  - {a['name']} ({a['type']}) ¥{a['price']}")

        lines.append(f"\n📅 生成时间: {datetime.now().isoformat()}")
        return "\n".join(lines)


# ============================================================================
# 协作调度器
# ============================================================================

class BlackboardOrchestrator:
    """
    黑板模式调度器：管理多个Agent在黑板上的协作。
    """

    def __init__(self, blackboard: Blackboard = None):
        self.blackboard = blackboard or Blackboard()
        self.agents: list[BlackboardAgent] = []

    def register(self, agent: BlackboardAgent) -> "BlackboardOrchestrator":
        """注册Agent。"""
        self.agents.append(agent)
        return self

    def register_all(self, agents: list[BlackboardAgent]) -> "BlackboardOrchestrator":
        """批量注册。"""
        self.agents.extend(agents)
        return self

    def run_parallel(
        self,
        context: dict = None,
        max_workers: int = 5,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        并行运行所有Agent。
        所有Agent并发读写黑板。

        Args:
            context: 传递给Agent的上下文
            max_workers: 最大并发数
            verbose: 是否打印日志

        Returns:
            所有Agent的执行结果
        """
        if verbose:
            print(f"\n  🎬 黑板模式启动: {len(self.agents)} 个Agent")
            print(f"  黑板: {self.blackboard.name}")
            print(f"  并发数: {max_workers}")

        results = {}
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_agent = {
                executor.submit(agent.execute, context): agent
                for agent in self.agents
            }

            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent.agent_id] = result
                    if verbose:
                        print(f"  ✅ {agent.agent_id}: {result}")
                except Exception as e:
                    results[agent.agent_id] = {"error": str(e)}
                    if verbose:
                        print(f"  ❌ {agent.agent_id}: {e}")

        elapsed = time.time() - start_time

        # 检测冲突
        conflicts = self.blackboard.get_conflicts(resolved=False)
        if conflicts and verbose:
            print(f"\n  ⚠️ 检测到 {len(conflicts)} 个冲突：")
            for c in conflicts:
                print(f"    - {c.key}: Agents {[e.agent_id for e in c.entries]} 竞争写入")

        if verbose:
            print(f"\n  ⏱️ 总耗时: {elapsed:.1f}s")
            print(f"  📊 黑板统计: {self.blackboard.get_stats()}")

        return results

    def run_sequential(
        self,
        context: dict = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        顺序运行所有Agent（适用于有依赖关系的场景）。
        """
        results = {}
        start_time = time.time()

        for agent in self.agents:
            try:
                result = agent.execute(context)
                results[agent.agent_id] = result
                if verbose:
                    print(f"  ✅ {agent.agent_id}: {result}")
            except Exception as e:
                results[agent.agent_id] = {"error": str(e)}

        elapsed = time.time() - start_time
        if verbose:
            print(f"  ⏱️ 总耗时: {elapsed:.1f}s")

        return results


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  多智能体系统 - 黑板模式演示")
    print("=" * 72)

    # ---- Demo 1: 旅行规划 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 旅行规划（航班+酒店+活动 → 共享行程表）")
    print("=" * 72)

    # 创建黑板
    trips_blackboard = Blackboard(name="TripPlanningBoard")

    # 设置行程约束（写入初始数据）
    trips_blackboard.write(
        "constraints",
        {
            "origin": "北京",
            "destination": "上海",
            "checkin": "2025-06-15",
            "checkout": "2025-06-18",
            "budget": 5000,
            "travelers": 1,
            "interests": ["文化", "美食", "观光"],
        },
        "orchestrator",
        tags=["constraints", "initial"],
    )

    # 创建Agent
    flight_agent = FlightAgent(trips_blackboard)
    hotel_agent = HotelAgent(trips_blackboard)
    activity_agent = ActivityAgent(trips_blackboard)
    planner = ItineraryPlanner(trips_blackboard)

    # 创建调度器
    orchestrator = BlackboardOrchestrator(trips_blackboard)
    orchestrator.register_all([
        flight_agent,
        hotel_agent,
        activity_agent,
        planner,
    ])

    # 执行
    results = orchestrator.run_parallel(max_workers=4, verbose=True)

    # 查看最终行程
    itinerary = trips_blackboard.read("itinerary")
    if itinerary:
        print(f"\n{'─' * 72}")
        print("  📋 最终行程:")
        print(f"{'─' * 72}")
        if isinstance(itinerary.value, dict):
            print(itinerary.value.get("summary", itinerary.value))
            print(f"\n  详细数据 (JSON):")
            print(json.dumps(itinerary.value, ensure_ascii=False, indent=2, default=str))

    # ---- Demo 2: 黑板冲突检测 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 冲突检测与解决")
    print("=" * 72)

    conflict_blackboard = Blackboard(name="ConflictDemoBoard")

    # 模拟两个Agent写入同一key
    conflict_blackboard.write(
        "recommendation",
        "方案A：优先考虑成本优化",
        "agent_alpha",
    )
    print("  Agent Alpha 写入 'recommendation' → 方案A")

    conflict_blackboard.write(
        "recommendation",
        "方案B：优先考虑用户体验",
        "agent_beta",
    )
    print("  Agent Beta 写入 'recommendation' → 方案B")

    conflicts = conflict_blackboard.get_conflicts(resolved=False)
    print(f"\n  ⚠️ 检测到 {len(conflicts)} 个冲突")
    for c in conflicts:
        print(f"    Key: {c.key}")
        for e in c.entries:
            print(f"      来自 {e.agent_id}: {e.value} (v{e.version})")

    # 解决冲突
    resolved = conflict_blackboard.resolve_conflict(
        conflict_id=conflicts[0].conflict_id,
        resolution="经过综合评估，采用方案B（用户体验优先）作为主要方向，"
                    "但在实施中纳入方案A中的成本控制建议",
        resolved_by="conflict_resolver",
        winning_entry_key="recommendation",
    )
    print(f"\n  🔧 冲突已解决: {resolved}")

    # ---- Demo 3: 观察者通知 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 观察者模式 - 实时日志")
    print("=" * 72)

    observer_blackboard = Blackboard(name="ObserverDemoBoard")

    # 注册观察者（记录所有操作）
    def log_observer(operation: str, entry: BlackboardEntry):
        print(f"    [观察者] {operation.upper()} | {entry.key} ← {entry.agent_id}")

    observer_blackboard.observe(log_observer)

    # 模拟Agent操作
    agent_x = FlightAgent(observer_blackboard)
    agent_x.agent_id = "agent_x"
    agent_x.write("data_1", "航班信息A")
    agent_x.write("data_2", "航班信息B")
    _ = observer_blackboard.read("data_1")
    observer_blackboard.delete("data_2", "agent_x")

    # 黑板统计
    print(f"\n  黑板最终状态:")
    print(json.dumps(trips_blackboard.get_stats(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print("  黑板模式演示完成！")
    print("=" * 72)
