#!/usr/bin/env python3
"""
Phase 07 - 多智能体系统：02 - 层级管理-Worker模式（Hierarchical Manager-Worker）

Manager Agent 分解任务 → 分配给专业化 Worker → 合成结果。
支持依赖感知调度、ThreadPoolExecutor 并行执行、死锁检测。

架构图：
                         ┌─────────────┐
                         │   Manager   │  (gpt-4o: 任务分解 + 结果合成)
                         │   Agent     │
                         └──┬───┬───┬──┘
                            │   │   │
                   ┌────────┘   │   └────────┐
                   ▼            ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │ Worker A │ │ Worker B │ │ Worker C │  (gpt-4o-mini: 专业化执行)
            │ (数据分析)│ │ (技术写作)│ │ (可视化) │
            └──────────┘ └──────────┘ └──────────┘
                   │            │            │
                   └────────────┼────────────┘
                                ▼
                         ┌─────────────┐
                         │   Manager   │  合成结果
                         │   Agent     │
                         └─────────────┘
"""

import json
import time
import hashlib
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Optional, Union


# ============================================================================
# 数据模型
# ============================================================================

class TaskStatus(Enum):
    """任务状态枚举。"""
    PENDING = "pending"          # 等待执行
    RUNNING = "running"          # 执行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    BLOCKED = "blocked"          # 被阻塞（等待依赖）

    def is_terminal(self) -> bool:
        """是否为终态。"""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED)


@dataclass
class Subtask:
    """
    Manager 分解出的子任务。

    Attributes:
        task_id: 唯一标识
        description: 任务描述（会传给Worker作为输入）
        worker_type: 需要哪种Worker处理（如 "data_analyst"）
        depends_on: 依赖的任务ID列表（这些任务完成后才能执行）
        status: 当前状态
        result: 执行结果（完成后填充）
        error: 错误信息（失败时填充）
        assigned_worker: 分配的Worker名称
    """
    task_id: str
    description: str
    worker_type: str
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    assigned_worker: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def elapsed(self) -> float:
        """执行耗时（秒）。"""
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return 0.0


@dataclass
class WorkerProfile:
    """
    Worker 专业化配置。

    Attributes:
        name: Worker名称
        worker_type: Worker类型（与Subtask.worker_type匹配）
        system_prompt: 专业化系统提示词
        model: 使用的模型ID
        capabilities: 能力标签列表
    """
    name: str
    worker_type: str
    system_prompt: str
    model: str = "gpt-4o-mini"
    capabilities: list[str] = field(default_factory=list)


# ============================================================================
# Worker 基类和具体实现
# ============================================================================

class BaseWorker(ABC):
    """Worker 基类。所有专业Worker继承此类。"""

    def __init__(self, profile: WorkerProfile):
        self.profile = profile
        self.tasks_completed: int = 0
        self.tasks_failed: int = 0
        self.total_time: float = 0.0

    @abstractmethod
    def execute(self, task_description: str, context: dict = None) -> str:
        """
        执行任务。

        Args:
            task_description: 任务描述
            context: 额外上下文（依赖任务的结果等）

        Returns:
            执行结果字符串
        """
        ...

    def __repr__(self) -> str:
        return f"<Worker {self.profile.name} ({self.profile.worker_type})>"


class SimulatedLLMWorker(BaseWorker):
    """
    基于模拟LLM的Worker实现。
    生产环境中替换为真实的LLM API调用。
    """

    _EXPERTISE_RESPONSES = {
        "data_analyst": lambda desc: (
            f"## 数据分析报告\n\n"
            f"### 分析任务\n{desc[:200]}\n\n"
            f"### 数据洞察\n"
            f"1. **关键指标**：根据初步分析，目标指标同比增长23.7%\n"
            f"2. **异常检测**：发现3处数据异常点，建议进一步验证（时间点：Q2-01, Q2-15, Q3-07）\n"
            f"3. **相关性发现**：用户活跃度与新功能使用率呈强正相关（r=0.82, p<0.01）\n"
            f"4. **趋势预测**：基于ARIMA模型，预计下季度增长12-15%\n\n"
            f"### 建议\n"
            f"- 优先跟进3处数据异常点\n"
            f"- 考虑在新功能推广上加大投入\n"
            f"- 建议建立实时监控Dashboard\n"
        ),
        "technical_writer": lambda desc: (
            f"## 技术文档\n\n"
            f"### 文档概述\n针对以下需求：{desc[:200]}\n\n"
            f"### 1. 系统架构说明\n"
            f"本系统采用微服务架构，包含以下核心模块：\n"
            f"- **API Gateway**：统一入口，负责路由、限流和认证\n"
            f"- **Service Layer**：业务逻辑层，每个服务独立部署\n"
            f"- **Data Layer**：PostgreSQL主库 + Redis缓存 + Elasticsearch搜索\n\n"
            f"### 2. API文档\n```\nPOST /api/v1/analyze\nContent-Type: application/json\n\n{{\"query\": \"...\", \"filters\": {{...}}}}\n→ {{\"result\": ..., \"confidence\": 0.92}}\n```\n\n"
            f"### 3. 部署说明\n- 使用Docker Compose编排\n- CI/CD via GitHub Actions\n- 蓝绿部署策略\n"
        ),
        "visualization_expert": lambda desc: (
            f"## 可视化方案\n\n"
            f"### 需求分析\n{desc[:200]}\n\n"
            f"### 推荐图表\n"
            f"1. **趋势图（Line Chart）**：展示时间序列变化\n"
            f"   - X轴：时间（按天/周/月）\n"
            f"   - Y轴：指标值\n"
            f"   - 工具：ECharts / Plotly\n\n"
            f"2. **热力图（Heatmap）**：展示多维度交叉密度\n"
            f"   - 行：用户分组\n"
            f"   - 列：时间窗口\n"
            f"   - 配色：蓝→白→红渐变\n\n"
            f"3. **雷达图（Radar Chart）**：多指标对比评估\n"
            f"   - 5-8个指标轴\n"
            f"   - 多个数据集叠加对比\n\n"
            f"### 实现建议\n- 使用Plotly Dash构建交互式仪表盘\n- 数据刷新频率：每5分钟\n- 移动端适配：使用响应式布局\n"
        ),
        "code_developer": lambda desc: (
            f"## 代码实现\n\n"
            f"### 需求\n{desc[:200]}\n\n"
            f"### 方案设计\n采用工厂模式 + 策略模式，实现可扩展的数据处理管道。\n\n"
            f"### 核心代码\n```python\n"
            f"from abc import ABC, abstractmethod\n"
            f"from typing import Any, Iterator\n\n"
            f"class DataProcessor(ABC):\n"
            f"    @abstractmethod\n"
            f"    def process(self, data: Any) -> Any: ...\n\n"
            f"class Pipeline:\n"
            f"    def __init__(self, processors: list[DataProcessor]):\n"
            f"        self._processors = processors\n"
            f"    \n"
            f"    def run(self, data: Any) -> Any:\n"
            f"        result = data\n"
            f"        for p in self._processors:\n"
            f"            result = p.process(result)\n"
            f"        return result\n"
            f"```\n\n"
            f"### 测试建议\n- 单元测试覆盖率目标：>85%\n"
            f"- 集成测试：pipeline端到端测试\n"
            f"- 性能测试：单次处理延迟 < 100ms (P99)\n"
        ),
        "qa_tester": lambda desc: (
            f"## QA测试报告\n\n"
            f"### 测试范围\n{desc[:200]}\n\n"
            f"### 测试用例\n"
            f"| ID | 描述 | 期望结果 | 实际结果 | 状态 |\n"
            f"|----|------|----------|----------|------|\n"
            f"| TC-01 | 正常输入返回200 | 200 OK | 200 OK | ✅ |\n"
            f"| TC-02 | 空输入返回400 | 400 Bad Request | 400 Bad Request | ✅ |\n"
            f"| TC-03 | SQL注入防护 | 查询被参数化 | 使用PreparedStatement | ✅ |\n"
            f"| TC-04 | 并发100请求 | P99<200ms | P99=187ms | ✅ |\n"
            f"| TC-05 | 超长输入截断 | 截断至1000字符 | 截断至1024字符 | ⚠️ |\n\n"
            f"### 发现Bug\n"
            f"1. TC-05: 截断长度应为1000，实际截断至1024，需修复\n"
            f"2. N+1查询：get_users_with_orders()存在N+1问题\n\n"
            f"### 通过率：96.7% (29/30)\n"
        ),
    }

    def execute(self, task_description: str, context: dict = None) -> str:
        """使用worker类型对应的专业知识处理任务。"""
        # 模拟处理延迟
        time.sleep(0.1)

        handler = self._EXPERTISE_RESPONSES.get(
            self.profile.worker_type,
            lambda desc: f"[{self.profile.name}] 已处理: {desc[:200]}",
        )

        result = handler(task_description)

        # 如果有依赖任务的结果作为上下文，附加相关信息
        if context:
            result = (
                f"{result}\n\n---\n"
                f"### 上下文参考（来自依赖任务）\n"
                f"{json.dumps({k: v[:100] + '...' if len(v) > 100 else v for k, v in context.items()}, ensure_ascii=False, indent=2)}"
            )

        return result


# ============================================================================
# Manager Agent
# ============================================================================

class ManagerAgent:
    """
    Manager Agent：负责任务分解、调度和结果合成。

    工作流程：
    1. 接收复杂任务
    2. 分解为子任务（Subtask），定义依赖关系
    3. 管理Worker池，按依赖关系调度
    4. 收集所有Worker结果
    5. 合成最终答案
    """

    # Manager 分解任务的模板
    TASK_DECOMPOSITION_PROMPT = (
        "你是资深项目经理。请将以下复杂任务分解为可并行执行的子任务。\n\n"
        "要求：\n"
        "1. 每个子任务指定需要的 Worker 类型（data_analyst / technical_writer / "
        "visualization_expert / code_developer / qa_tester）\n"
        "2. 明确子任务之间的依赖关系（depends_on 字段）\n"
        "3. 确保依赖关系不形成环（无死锁）\n"
        "4. 每个子任务描述应具体、可执行\n\n"
        "输出JSON格式：\n"
        "[{{\"task_id\": \"...\", \"description\": \"...\", \"worker_type\": \"...\", "
        "\"depends_on\": [\"task_id...\"]}}]"
    )

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.workers: dict[str, BaseWorker] = {}
        self._lock = Lock()

    def register_worker(self, worker: BaseWorker) -> None:
        """注册一个Worker到管理池中。"""
        with self._lock:
            self.workers[worker.profile.worker_type] = worker

    def register_workers(self, workers: list[BaseWorker]) -> None:
        """批量注册Worker。"""
        for w in workers:
            self.register_worker(w)

    def get_worker(self, worker_type: str) -> Optional[BaseWorker]:
        """获取指定类型的Worker。"""
        return self.workers.get(worker_type)

    def list_worker_types(self) -> list[str]:
        """列出所有可用的Worker类型。"""
        return list(self.workers.keys())

    # ---- 任务分解 ----

    def decompose(self, complex_task: str, verbose: bool = True) -> list[Subtask]:
        """
        将复杂任务分解为子任务列表。

        生产环境中，这里会调用 gpt-4o 进行任务分解。
        这里使用基于规则的方法作为示例。

        Args:
            complex_task: 复杂任务描述
            verbose: 是否打印分解日志

        Returns:
            子任务列表
        """
        if verbose:
            print(f"\n  🔍 Manager ({self.model}) 正在分解任务...")
            print(f"  任务内容: {complex_task[:100]}...")

        # 模拟 AI 分解过程
        subtasks = self._rule_based_decompose(complex_task)

        if verbose:
            print(f"  分解完成：{len(subtasks)} 个子任务")
            for subtask in subtasks:
                deps = f"，依赖: {subtask.depends_on}" if subtask.depends_on else ""
                print(f"    [{subtask.task_id}] {subtask.worker_type}: "
                      f"{subtask.description[:60]}...{deps}")

        return subtasks

    def _rule_based_decompose(self, task: str) -> list[Subtask]:
        """
        基于规则的模拟任务分解。
        在真实场景中替换为 LLM 调用：

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.TASK_DECOMPOSITION_PROMPT},
                {"role": "user", "content": task},
            ],
            response_format={"type": "json_object"},
        )
        subtasks_json = json.loads(response.choices[0].message.content)
        """

        task_lower = task.lower()

        # 根据任务关键词选择Worker类型组合
        if "报告" in task or "report" in task_lower or "分析" in task:
            return [
                Subtask(
                    task_id="T1",
                    description="收集和整理数据，进行统计分析，输出关键指标和趋势",
                    worker_type="data_analyst",
                ),
                Subtask(
                    task_id="T2",
                    description="基于数据分析结果，撰写结构化的技术报告",
                    worker_type="technical_writer",
                    depends_on=["T1"],  # T2依赖T1完成
                ),
                Subtask(
                    task_id="T3",
                    description="设计可视化图表方案，包括趋势图、分布图、热力图",
                    worker_type="visualization_expert",
                    depends_on=["T1"],
                ),
                Subtask(
                    task_id="T4",
                    description="最终整合：将T2的报告和T3的图表整合为完整交付物",
                    worker_type="technical_writer",
                    depends_on=["T2", "T3"],
                ),
            ]
        elif "代码" in task or "code" in task_lower or "开发" in task:
            return [
                Subtask(
                    task_id="T1",
                    description="设计系统架构和模块划分，输出技术方案文档",
                    worker_type="technical_writer",
                ),
                Subtask(
                    task_id="T2",
                    description="根据技术方案实现核心功能代码",
                    worker_type="code_developer",
                    depends_on=["T1"],
                ),
                Subtask(
                    task_id="T3",
                    description="编写单元测试、集成测试和端到端测试",
                    worker_type="qa_tester",
                    depends_on=["T2"],
                ),
            ]
        elif "数据" in task or "data" in task_lower or "可视化" in task:
            return [
                Subtask(
                    task_id="T1",
                    description="执行数据清洗、预处理和探索性数据分析（EDA）",
                    worker_type="data_analyst",
                ),
                Subtask(
                    task_id="T2",
                    description="设计并实现交互式数据可视化仪表盘",
                    worker_type="visualization_expert",
                    depends_on=["T1"],
                ),
            ]
        else:
            # 通用分解
            return [
                Subtask(
                    task_id="T1",
                    description=f"分析任务要求，收集必要信息：{task[:80]}",
                    worker_type="data_analyst",
                ),
                Subtask(
                    task_id="T2",
                    description=f"基于分析结果，生成最终输出",
                    worker_type="technical_writer",
                    depends_on=["T1"],
                ),
            ]

    # ---- 死锁检测 ----

    def detect_deadlock(self, subtasks: list[Subtask]) -> Optional[list[str]]:
        """
        检测子任务依赖图中是否存在死锁（循环依赖）。

        使用DFS检测有向图中的环。

        Args:
            subtasks: 子任务列表

        Returns:
            如果存在环，返回环中的任务ID列表；否则返回None
        """
        task_ids = {s.task_id for s in subtasks}
        adjacency = {s.task_id: s.depends_on for s in subtasks}

        # 三种颜色：WHITE=未访问, GRAY=访问中, BLACK=已完成
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in task_ids}

        cycle = []

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in adjacency.get(node, []):
                if neighbor not in color:
                    # 依赖不存在的task_id → 不是死锁但应警告
                    continue
                if color[neighbor] == GRAY:
                    # 发现环
                    cycle.append(node)
                    return True
                if color[neighbor] == WHITE:
                    if dfs(neighbor):
                        cycle.append(node)
                        return True
            color[node] = BLACK
            return False

        for tid in task_ids:
            if color[tid] == WHITE:
                if dfs(tid):
                    cycle.reverse()
                    return cycle

        return None

    # ---- 调度执行 ----

    def execute(
        self,
        complex_task: str,
        max_workers: int = 5,
        verbose: bool = True,
    ) -> tuple[list[Subtask], str]:
        """
        完整的Manager-Worker执行流程。

        Args:
            complex_task: 复杂任务描述
            max_workers: 最大并行Worker数
            verbose: 是否打印详细日志

        Returns:
            (所有子任务, 最终合成结果)
        """
        # 步骤1：任务分解
        subtasks = self.decompose(complex_task, verbose=verbose)

        # 步骤2：死锁检测
        cycle = self.detect_deadlock(subtasks)
        if cycle:
            error_msg = f"[错误] 检测到任务循环依赖（死锁）: {' → '.join(cycle)}"
            if verbose:
                print(f"\n  ❌ {error_msg}")
            raise ValueError(error_msg)

        # 步骤3：依赖感知的并行调度
        if verbose:
            print(f"\n  ⚙️ 开始并行调度（最多 {max_workers} 个并发Worker）...")

        task_map = {s.task_id: s for s in subtasks}
        completed_task_ids: set[str] = set()
        failed_task_ids: set[str] = set()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task: dict[Future, Subtask] = {}
            submitted: set[str] = set()

            while len(completed_task_ids) + len(failed_task_ids) < len(subtasks):
                # 找出可以提交的任务（依赖都已完成，且未提交，且未失败）
                for subtask in subtasks:
                    if subtask.task_id in submitted:
                        continue
                    if subtask.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                        continue

                    # 检查依赖是否全部完成
                    deps_ready = all(
                        dep in completed_task_ids
                        for dep in subtask.depends_on
                    )
                    # 检查是否存在失败的依赖
                    deps_failed = any(
                        dep in failed_task_ids
                        for dep in subtask.depends_on
                    )

                    if deps_failed:
                        subtask.status = TaskStatus.BLOCKED
                        subtask.error = "依赖任务失败，无法执行"
                        failed_task_ids.add(subtask.task_id)
                        if verbose:
                            print(f"  🚫 [{subtask.task_id}] 被阻塞：依赖任务失败")
                        continue

                    if deps_ready:
                        worker = self.get_worker(subtask.worker_type)
                        if worker is None:
                            subtask.status = TaskStatus.FAILED
                            subtask.error = f"找不到 {subtask.worker_type} 类型的Worker"
                            failed_task_ids.add(subtask.task_id)
                            if verbose:
                                print(f"  ❌ [{subtask.task_id}] 找不到Worker: {subtask.worker_type}")
                            continue

                        subtask.status = TaskStatus.RUNNING
                        subtask.assigned_worker = worker.profile.name
                        subtask.started_at = time.time()
                        submitted.add(subtask.task_id)

                        # 构建上下文（依赖任务的结果）
                        context = {
                            dep_id: task_map[dep_id].result
                            for dep_id in subtask.depends_on
                            if dep_id in completed_task_ids
                        }

                        future = executor.submit(
                            self._execute_single_task,
                            worker,
                            subtask,
                            context,
                            verbose,
                        )
                        future_to_task[future] = subtask

                # 检查完成的任务
                done_futures = set()
                for future, subtask_ref in future_to_task.items():
                    if future.done():
                        done_futures.add(future)
                        try:
                            result = future.result()
                            subtask_ref.result = result
                            subtask_ref.status = TaskStatus.COMPLETED
                            subtask_ref.finished_at = time.time()
                            completed_task_ids.add(subtask_ref.task_id)
                            if verbose:
                                print(f"  ✅ [{subtask_ref.task_id}] {subtask_ref.worker_type} "
                                      f"完成 ({subtask_ref.elapsed:.1f}s)")
                        except Exception as e:
                            subtask_ref.status = TaskStatus.FAILED
                            subtask_ref.error = str(e)
                            subtask_ref.finished_at = time.time()
                            failed_task_ids.add(subtask_ref.task_id)
                            if verbose:
                                print(f"  ❌ [{subtask_ref.task_id}] 执行失败: {e}")

                for f in done_futures:
                    del future_to_task[f]

                # 短暂休眠避免忙等待
                if len(done_futures) == 0 and len(future_to_task) < len(subtasks):
                    time.sleep(0.05)

        # 步骤4：结果合成
        if verbose:
            print(f"\n  📊 调度完成: {len(completed_task_ids)} 成功, {len(failed_task_ids)} 失败")

        final_answer = self.synthesize(subtasks, verbose=verbose)

        return subtasks, final_answer

    def _execute_single_task(
        self,
        worker: BaseWorker,
        subtask: Subtask,
        context: dict,
        verbose: bool,
    ) -> str:
        """单个任务的执行包装器（在ThreadPoolExecutor中执行）。"""
        return worker.execute(subtask.description, context)

    # ---- 结果合成 ----

    def synthesize(self, subtasks: list[Subtask], verbose: bool = True) -> str:
        """
        收集所有子任务结果，合成为最终答案。

        生产环境中，这里会调用 gpt-4o：
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是高级分析师，负责整合多个专家的输出..."},
                {"role": "user", "content": f"请整合以下子任务结果：\n{subtask_results}"},
            ],
        )
        """
        if verbose:
            print(f"\n  🧠 Manager ({self.model}) 正在合成最终结果...")

        # 构建合成报告
        sections = []
        for subtask in subtasks:
            status_icon = "✅" if subtask.status == TaskStatus.COMPLETED else "❌"
            sections.append(
                f"### {status_icon} {subtask.task_id}: {subtask.description[:80]}\n"
                f"- Worker: {subtask.worker_type} ({subtask.assigned_worker})\n"
                f"- 状态: {subtask.status.value} | 耗时: {subtask.elapsed:.1f}s\n\n"
                f"{subtask.result if subtask.result else f'[错误] {subtask.error}'}\n"
                f"---\n"
            )

        timeline = self._generate_timeline(subtasks)

        final_report = (
            f"# 任务执行报告\n\n"
            f"## 执行摘要\n"
            f"- 总子任务数: {len(subtasks)}\n"
            f"- 成功: {sum(1 for s in subtasks if s.status == TaskStatus.COMPLETED)}\n"
            f"- 失败: {sum(1 for s in subtasks if s.status == TaskStatus.FAILED)}\n"
            f"- 总耗时: {sum(s.elapsed for s in subtasks):.1f}s\n\n"
            f"## 执行时间线\n{timeline}\n\n"
            f"## 各子任务详细结果\n\n"
            f"{''.join(sections)}\n\n"
            f"---\n"
            f"*本报告由 Manager Agent ({self.model}) 自动合成*\n"
        )

        if verbose:
            print(f"  合成完成：{len(final_report)} 字符")

        return final_report

    def _generate_timeline(self, subtasks: list[Subtask]) -> str:
        """生成执行时间线（Mermaid格式）。"""
        lines = ["```mermaid", "gantt", "    title 任务执行时间线", "    dateFormat X"]
        for s in sorted(subtasks, key=lambda x: x.started_at or 0):
            if s.started_at:
                start = s.started_at - min(t.started_at or s.started_at for t in subtasks)
                duration = max(s.elapsed or 0.1, 0.1)
                status = "done" if s.status == TaskStatus.COMPLETED else "crit"
                lines.append(f"    {s.task_id} ({s.worker_type}): {status}, {start:.0f}, {duration:.0f}")
        lines.append("```")
        return "\n".join(lines)


# ============================================================================
# 工厂函数
# ============================================================================

def create_standard_worker_pool() -> list[BaseWorker]:
    """创建标准Worker池。"""
    profiles = [
        WorkerProfile(
            name="数据分析师-Bob",
            worker_type="data_analyst",
            system_prompt="你是资深数据分析师，擅长统计分析、SQL查询、Python数据分析...",
            model="gpt-4o-mini",
            capabilities=["statistics", "SQL", "Python", "EDA", "ML-analysis"],
        ),
        WorkerProfile(
            name="技术撰稿人-Alice",
            worker_type="technical_writer",
            system_prompt="你是资深技术撰稿人，擅长技术文档、API文档、架构设计文档...",
            model="gpt-4o-mini",
            capabilities=["documentation", "architecture", "API-docs", "technical-writing"],
        ),
        WorkerProfile(
            name="可视化专家-Carol",
            worker_type="visualization_expert",
            system_prompt="你是资深数据可视化专家，擅长ECharts、D3.js、Plotly...",
            model="gpt-4o-mini",
            capabilities=["charts", "dashboards", "ECharts", "D3.js", "Plotly"],
        ),
        WorkerProfile(
            name="开发者-Dave",
            worker_type="code_developer",
            system_prompt="你是资深后端开发者，擅长Python、Go、微服务架构...",
            model="gpt-4o-mini",
            capabilities=["Python", "Go", "microservices", "APIs"],
        ),
        WorkerProfile(
            name="测试工程师-Eve",
            worker_type="qa_tester",
            system_prompt="你是资深QA工程师，擅长自动化测试、性能测试、安全测试...",
            model="gpt-4o-mini",
            capabilities=["testing", "automation", "CI/CD", "coverage"],
        ),
    ]
    return [SimulatedLLMWorker(p) for p in profiles]


# ============================================================================
# 运行统计
# ============================================================================

@dataclass
class ExecutionStats:
    """一次Manager-Worker执行的统计信息。"""
    task_name: str
    total_subtasks: int
    completed: int
    failed: int
    blocked: int
    total_wall_time: float
    max_concurrency: int
    worker_utilization: dict[str, float]


def compute_stats(
    task_name: str,
    subtasks: list[Subtask],
    wall_time: float,
    max_concurrency: int,
) -> ExecutionStats:
    """计算执行统计。"""
    worker_counts: dict[str, int] = {}
    for s in subtasks:
        key = s.worker_type
        worker_counts[key] = worker_counts.get(key, 0) + 1

    return ExecutionStats(
        task_name=task_name,
        total_subtasks=len(subtasks),
        completed=sum(1 for s in subtasks if s.status == TaskStatus.COMPLETED),
        failed=sum(1 for s in subtasks if s.status == TaskStatus.FAILED),
        blocked=sum(1 for s in subtasks if s.status == TaskStatus.BLOCKED),
        total_wall_time=wall_time,
        max_concurrency=max_concurrency,
        worker_utilization={
            k: v / len(subtasks) for k, v in worker_counts.items()
        },
    )


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  多智能体系统 - 层级管理Worker模式演示")
    print("=" * 72)

    # ---- 初始化 ----

    # 创建Worker池
    workers = create_standard_worker_pool()
    print(f"\n📋 已注册 {len(workers)} 个专业Worker：")
    for w in workers:
        print(f"  - {w.profile.name} ({w.profile.worker_type}) [{w.profile.model}]")

    # 创建Manager
    manager = ManagerAgent(model="gpt-4o")
    manager.register_workers(workers)

    # ---- Demo 1: 报告生成任务 ----

    print("\n" + "=" * 72)
    print("  Demo 1: 生成企业AI应用分析报告")
    print("=" * 72)

    task1 = (
        "请生成一份关于'2025年企业AI应用现状'的综合分析报告。"
        "需要包含数据分析、趋势图表和可操作建议。"
        "目标读者是CIO和CTO级别的决策者。"
    )

    start_time = time.time()
    subtasks1, report1 = manager.execute(task1, max_workers=3, verbose=True)
    wall_time1 = time.time() - start_time

    stats1 = compute_stats("Demo 1 - 报告生成", subtasks1, wall_time1, 3)
    print(f"\n📊 执行统计:")
    print(f"  子任务: {stats1.completed}/{stats1.total_subtasks} 成功")
    print(f"  总耗时: {stats1.total_wall_time:.1f}s")
    print(f"  Worker利用率: {stats1.worker_utilization}")

    print(f"\n{'─' * 72}")
    print("  最终合成报告（前500字符）:")
    print(f"{'─' * 72}")
    print(report1[:500])
    print(f"\n  ...（共 {len(report1)} 字符）")

    # ---- Demo 2: 软件开发任务 ----

    print("\n\n" + "=" * 72)
    print("  Demo 2: 开发数据处理API微服务")
    print("=" * 72)

    task2 = (
        "请开发一个数据处理API微服务。"
        "需要支持：数据清洗、格式转换（JSON/CSV/XML）、数据验证。"
        "要求有完整的单元测试和API文档。"
    )

    start_time2 = time.time()
    subtasks2, report2 = manager.execute(task2, max_workers=3, verbose=True)
    wall_time2 = time.time() - start_time2

    stats2 = compute_stats("Demo 2 - 软件开发", subtasks2, wall_time2, 3)
    print(f"\n📊 执行统计:")
    print(f"  子任务: {stats2.completed}/{stats2.total_subtasks} 成功")
    print(f"  总耗时: {stats2.total_wall_time:.1f}s")

    # ---- Demo 3: 死锁检测 ----

    print("\n\n" + "=" * 72)
    print("  Demo 3: 死锁检测演示")
    print("=" * 72)

    # 创建一组存在循环依赖的子任务
    deadlock_subtasks = [
        Subtask(task_id="A", description="任务A", worker_type="data_analyst",
                depends_on=["C"]),  # A依赖C
        Subtask(task_id="B", description="任务B", worker_type="technical_writer",
                depends_on=["A"]),
        Subtask(task_id="C", description="任务C", worker_type="data_analyst",
                depends_on=["B"]),  # C依赖B → A→C→B→A 形成环！
    ]

    print("\n  测试子任务依赖关系:")
    for s in deadlock_subtasks:
        print(f"    {s.task_id} → depends_on: {s.depends_on}")

    cycle = manager.detect_deadlock(deadlock_subtasks)
    if cycle:
        print(f"\n  ✅ 成功检测到死锁!")
        print(f"  循环路径: {' → '.join(cycle)}")
    else:
        print(f"\n  ❌ 未检测到死锁（可能检测算法有误）")

    # 验证正常任务无死锁
    cycle2 = manager.detect_deadlock(subtasks1)
    print(f"\n  正常任务死锁检测: {'❌ 发现死锁' if cycle2 else '✅ 无死锁'}")

    print("\n" + "=" * 72)
    print("  Manager-Worker 模式演示完成！")
    print("=" * 72)
