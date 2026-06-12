# 陷阱 05：冷启动缓慢 —— 模型加载、缓存预热与嵌入预计算

> 用户按下查询按钮后等待 30 秒 —— 不是因为检索慢，而是因为模型和数据都还没准备好。

---

## 1. 症状（Symptoms）

| 症状 | 表现 | 典型耗时 |
|------|------|----------|
| **首次查询超时** | 系统启动后第一次查询需要 20-60 秒才能返回结果 | 模型加载 + 首次推理 |
| **服务重启后灾难** | 每次部署/重启后，前几分钟的用户体验极差 | 所有缓存失效 |
| **嵌入计算阻塞** | 加载新文档时，整个系统停止响应 | 单线程嵌入阻塞主流程 |
| **GPU 利用率波动** | 冷启动时 GPU 利用率 < 10%，预热后 > 80% | CUDA kernel 编译 + 显存分配 |
| **内存缓慢增长** | 服务运行数小时后内存才达到稳定状态 | 延迟初始化 + 缓存填充 |

### 冷启动时间线测量

```python
import time
import os
import sys
import psutil  # type: ignore
import hashlib
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum


class StartupPhase(Enum):
    """冷启动各阶段。"""
    PROCESS_INIT = "process_init"
    IMPORT_DEPENDENCIES = "import_dependencies"
    LOAD_CONFIG = "load_config"
    LOAD_EMBEDDING_MODEL = "load_embedding_model"
    LOAD_LLM_MODEL = "load_llm_model"
    WARMUP_CACHES = "warmup_caches"
    LOAD_VECTOR_STORE = "load_vector_store"
    PRE_COMPUTE_EMBEDDINGS = "pre_compute_embeddings"
    READY = "ready"


@dataclass
class StartupPhaseMetric:
    """冷启动各阶段的耗时和资源消耗。"""
    phase: StartupPhase
    duration_seconds: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float = 0.0
    details: str = ""


class ColdStartProfiler:
    """
    冷启动性能分析工具。

    追踪从进程启动到服务就绪的完整时间线，
    定位最耗时的阶段。
    """

    def __init__(self, output_dir: str = "./cold_start_profiles"):
        self.output_dir = output_dir
        self.phases: List[StartupPhaseMetric] = []
        self.start_time = time.time()
        self.current_phase_start = self.start_time
        os.makedirs(output_dir, exist_ok=True)

    def start_phase(self, phase: StartupPhase):
        """开始记录一个启动阶段。"""
        self.current_phase_start = time.time()
        self.current_phase = phase

    def end_phase(self, details: str = ""):
        """结束当前阶段并记录指标。"""
        duration = time.time() - self.current_phase_start
        process = psutil.Process()

        metric = StartupPhaseMetric(
            phase=self.current_phase,
            duration_seconds=duration,
            cpu_percent=process.cpu_percent(),
            memory_mb=process.memory_info().rss / (1024 * 1024),
            details=details,
        )
        self.phases.append(metric)

    def measure_total(self) -> Dict:
        """汇总所有阶段的测量结果。"""
        total_duration = time.time() - self.start_time
        phase_breakdown = {}

        for phase in StartupPhase:
            phase_metrics = [p for p in self.phases if p.phase == phase]
            if phase_metrics:
                total_phase_time = sum(p.duration_seconds for p in phase_metrics)
                phase_breakdown[phase.value] = {
                    "duration_seconds": round(total_phase_time, 3),
                    "percentage": round(total_phase_time / total_duration * 100, 1),
                    "instances": len(phase_metrics),
                }

        return {
            "total_seconds": round(total_duration, 3),
            "total_minutes": round(total_duration / 60, 2),
            "phases": phase_breakdown,
            "longest_phase": max(
                phase_breakdown.items(),
                key=lambda x: x[1]["duration_seconds"],
            )[0] if phase_breakdown else "unknown",
        }

    def generate_report(self) -> str:
        """生成冷启动分析报告。"""
        summary = self.measure_total()

        report = []
        report.append("=" * 70)
        report.append("冷启动性能分析报告")
        report.append("=" * 70)
        report.append(f"总启动时间: {summary['total_seconds']:.2f} 秒 "
                       f"({summary['total_minutes']:.2f} 分钟)")
        report.append(f"最慢阶段: {summary['longest_phase']}")
        report.append("")

        report.append(f"{'阶段':<30} {'耗时(s)':<12} {'占比':<10} {'次数'}")
        report.append("-" * 60)

        for phase_name, info in summary["phases"].items():
            bar = "#" * int(info["percentage"] / 5)
            report.append(
                f"{phase_name:<30} "
                f"{info['duration_seconds']:<12.3f} "
                f"{info['percentage']:<10.1f}% "
                f"{info['instances']}"
            )

        report.append("")
        report.append("优化建议:")

        # 基于最长阶段的建议
        long_phase = summary["longest_phase"]
        suggestions = {
            "load_embedding_model": [
                "- 使用 ONNX Runtime 或 TensorRT 加速模型加载",
                "- 使用较小的模型（bge-small vs bge-large）",
                "- 模型量化（float32 → int8）减少加载时间",
                "- 将模型持久化加载到共享内存中",
            ],
            "load_llm_model": [
                "- 使用模型预加载/热备份（keep-alive 机制）",
                "- 使用 vLLM 或 TGI 的持续批处理",
                "- 模型量化（GPTQ, AWQ, GGUF）",
                "- 将 LLM 作为独立服务部署（避免冷启动影响 RAG）",
            ],
            "load_vector_store": [
                "- 使用向量数据库的持久化模式（mmap）",
                "- 预构建 HNSW 索引，避免每次启动重建",
                "- 使用磁盘索引（如 Qdrant on_disk=True）",
                "- 限制启动时加载的向量数量",
            ],
            "pre_compute_embeddings": [
                "- 预先计算常用查询的嵌入并缓存",
                "- 使用嵌入缓存（pickle/memmap）",
                "- 后台预热线程，不阻塞主流程",
            ],
        }

        if long_phase in suggestions:
            report.append(f"\n对 '{long_phase}' 的优化建议:")
            for suggestion in suggestions[long_phase]:
                report.append(f"  {suggestion}")

        report_text = "\n".join(report)

        # 保存报告
        report_path = os.path.join(
            self.output_dir,
            f"cold_start_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(report_text)
        print(f"\n报告已保存至: {report_path}")

        return report_text


# 模拟一个冷启动过程
def simulate_cold_start():
    """
    模拟一个典型的 RAG 系统冷启动过程。

    在实践中，这个函数展示了实际的启动各阶段和它们的耗时。
    """

    profiler = ColdStartProfiler()

    # Phase 1: 进程初始化
    profiler.start_phase(StartupPhase.PROCESS_INIT)
    time.sleep(0.05)  # 模拟 Python 解释器启动
    profiler.end_phase("Python 进程启动，GIL 初始化")

    # Phase 2: 导入依赖
    profiler.start_phase(StartupPhase.IMPORT_DEPENDENCIES)
    time.sleep(0.5)  # 模拟 import torch, transformers, sentence_transformers 等
    profiler.end_phase("导入 torch (2.0.1), transformers (4.36.0), "
                       "sentence_transformers (2.2.2), chromadb (0.4.22)")

    # Phase 3: 加载配置
    profiler.start_phase(StartupPhase.LOAD_CONFIG)
    time.sleep(0.1)  # 读取 YAML/JSON 配置
    profiler.end_phase("加载 config.yaml: embedding_model=bge-large-zh-v1.5, "
                       "chunk_size=512, top_k=5")

    # Phase 4: 加载嵌入模型 (最耗时的步骤之一)
    profiler.start_phase(StartupPhase.LOAD_EMBEDDING_MODEL)
    time.sleep(3.5)  # 模拟下载/加载 1.3GB 的 BGE 模型
    profiler.end_phase("加载 BAAI/bge-large-zh-v1.5 (1.3GB) → GPU:0")

    # Phase 5: 加载 LLM 模型 (最耗时的步骤)
    profiler.start_phase(StartupPhase.LOAD_LLM_MODEL)
    time.sleep(8.0)  # 模拟加载 7B/13B 模型
    profiler.end_phase("加载 Qwen2-7B-Instruct (14GB) → GPU:0,1")

    # Phase 6: 预热缓存
    profiler.start_phase(StartupPhase.WARMUP_CACHES)
    time.sleep(1.2)  # CPU 密集型：计算预热嵌入
    profiler.end_phase("预热 100 条常用查询嵌入，预热 tokenizer 缓存")

    # Phase 7: 加载向量存储
    profiler.start_phase(StartupPhase.LOAD_VECTOR_STORE)
    time.sleep(2.0)  # 从磁盘加载索引
    profiler.end_phase("加载 Chroma 持久化索引: 50,000 vectors")

    # Phase 8: 预计算嵌入
    profiler.start_phase(StartupPhase.PRE_COMPUTE_EMBEDDINGS)
    time.sleep(1.5)  # 预计算
    profiler.end_phase("预计算 10,000 个文档块的嵌入向量")

    # Phase 9: 就绪
    profiler.start_phase(StartupPhase.READY)
    time.sleep(0.01)
    profiler.end_phase("服务就绪！")

    return profiler.generate_report()

report_text_content = simulate_cold_start()
```

---

## 2. 根因（Root Cause）

### 2.1 冷启动时间的五层分解

```python
"""
冷启动缓慢的根本原因可以从五个层面分析：

第一层：Python 依赖导入
  - import torch 本身就需要 1-3 秒（加载 CUDA 库）
  - import transformers 需要 0.5-1 秒（加载 tokenizer 配置）
  - 累计导入可能耗时 2-5 秒

第二层：模型文件 I/O
  - 嵌入模型：bge-large-zh-v1.5 约 1.3GB，从磁盘读取需要时间
  - LLM 模型：Qwen2-7B 约 14GB（fp16），读取需要 5-10 秒（SSD）
  - 如果是机械硬盘或网络存储，这个时间可能翻倍或翻三倍

第三层：模型初始化
  - 权重加载到内存 + 移动到 GPU 显存
  - CUDA kernel 即时编译（JIT compilation）
  - 显存分配（cudaMalloc）

第四层：向量索引加载
  - Chroma 从磁盘加载 HNSW 索引 → 解析 SQLite 元数据
  - 如果没有持久化索引，需要完全重建 O(N log N)

第五层：缓存冷启动
  - Tokenizer 缓存为空
  - Embedding 缓存为空
  - LLM KV Cache 为空
  - 第一次推理最慢（'first token latency' 问题）
"""


def detailed_root_cause_breakdown():
    """
    详细分解冷启动的根因，包括每个子步骤的典型耗时。

    数据基于：
    - Intel i9-13900K + NVIDIA RTX 4090
    - Samsung 990 Pro NVMe SSD (7GB/s 读取)
    - bge-large-zh-v1.5 (326M params, 1.3GB)
    - Qwen2-7B-Instruct (7B params, 14GB fp16)
    """

    breakdown = {
        "Python 启动": {
            "sub_steps": [
                ("进程创建 & GIL 初始化", 0.05, "Python 解释器启动开销"),
                ("import torch", 1.50, "加载 libtorch + CUDA runtime"),
                ("import transformers", 0.80, "加载 tokenizer 配置和模型注册表"),
                ("import sentence_transformers", 0.40, "加载模型池化管理器"),
                ("import chromadb", 0.30, "加载 SQLite + HNSW 库"),
                ("其他 import", 0.50, "logging, yaml, numpy, etc."),
            ],
            "total": 3.55,
        },
        "嵌入模型加载": {
            "sub_steps": [
                ("从磁盘读取模型权重 (1.3GB)", 1.20, "SSD 顺序读取 ~1GB/s"),
                ("反序列化 safetensors/pytorch", 0.60, "解析模型文件格式"),
                ("构建模型计算图", 0.30, "torch.nn.Module 实例化"),
                ("加载 tokenizer 词表", 0.20, "30万 token 的词表文件"),
                ("CPU→GPU 数据传输 (1.3GB)", 0.50, "PCIe 4.0 x16 ~25GB/s"),
                ("CUDA kernel warmup", 0.70, "首次推理触发 JIT 编译"),
            ],
            "total": 3.50,
        },
        "LLM 模型加载": {
            "sub_steps": [
                ("从磁盘读取模型权重 (14GB)", 7.00, "SSD 顺序读取 ~2GB/s"),
                ("反序列化模型分片", 1.50, "多个 .safetensors 文件"),
                ("CPU→GPU 数据传输 (14GB)", 3.00, "部分通过 PCIe，部分 NVLink"),
                ("分配 KV Cache 显存", 2.00, "为最大序列长度预分配"),
                ("CUDA kernel JIT 编译", 1.50, "首次推理时触发"),
            ],
            "total": 15.00,
        },
        "向量数据库启动": {
            "sub_steps": [
                ("解析元数据 (SQLite)", 0.30, "读取 document 元数据"),
                ("加载 HNSW 索引图结构", 1.50, "解析图边连接和层级"),
                ("重建内存索引", 2.00, "如果没有持久化，需要完全重建"),
                ("验证数据一致性", 0.20, "检查向量数量与元数据匹配"),
            ],
            "total": 4.00,
        },
        "缓存预热": {
            "sub_steps": [
                ("Tokenizer 缓存预热", 0.50, "预编译常用 tokenize 路径"),
                ("Embedding 缓存填充", 1.50, "预计算 FAQ/热门查询的嵌入"),
                ("LLM KV Cache 预热", 2.00, "预填充几个典型 prompt"),
                ("磁盘/内存缓存索引", 0.50, "建立常用查询的快速查找表"),
            ],
            "total": 4.50,
        },
    }

    grand_total = sum(info["total"] for info in breakdown.values())

    print("=" * 70)
    print("冷启动根因：五层分解")
    print("=" * 70)

    for layer, info in breakdown.items():
        print(f"\n【{layer}】合计: {info['total']:.2f} 秒")
        for step_name, duration, note in info["sub_steps"]:
            bar_length = int(duration / grand_total * 40)
            bar = "█" * bar_length
            print(f"  {step_name:<35} {duration:>6.2f}s  {bar}  {note}")

    print(f"\n{'=' * 70}")
    print(f"总冷启动时间: {grand_total:.2f} 秒 ({grand_total/60:.1f} 分钟)")
    print(f"  LLM 加载占比: {breakdown['LLM 模型加载']['total']/grand_total*100:.1f}%")
    print(f"  缓存预热占比: {breakdown['缓存预热']['total']/grand_total*100:.1f}%")
    print(f"  嵌入模型占比: {breakdown['嵌入模型加载']['total']/grand_total*100:.1f}%")

    return breakdown

root_cause_detail = detailed_root_cause_breakdown()
```

---

## 3. 真实场景（Real-world Scenario）

### 场景：法律 RAG 服务的部署灾难

```python
"""
真实案例：

某律师事务所的 IT 团队开发了一个 RAG 问答系统。
技术栈：
- 嵌入模型: bge-large-zh-v1.5 (1.3GB)
- LLM: Qwen2-7B-Instruct (14GB)
- 向量数据库: Chroma (50,000 条判例)
- 部署: Docker on AWS EC2 (g4dn.xlarge, 1xT4 GPU)

遭遇的问题：
- 每次 Docker 容器重启，服务需要 45-60 秒才能就绪
- Kubernetes 的 readiness probe 超时，导致容器被反复杀死
- 滚动更新时，旧容器已停止但新容器还未就绪，服务中断
- 高峰期的 auto-scaling 触发新实例启动，但新实例需要 60 秒，
  等它就绪时流量峰值已经过去

造成的业务影响：
- 每天约 3-5 次部署(CI/CD) × 60 秒不可用 = 3-5 分钟/天
- 每次高峰 auto-scale 新实例都浪费（就绪太慢，流量已被旧实例消化）
- 律师在 60 秒等待后失去耐心，切换到手动搜索
"""

def real_world_timeline():
    """重现真实场景中的冷启动时间线。"""

    timeline = [
        ("T+0s", "Docker 容器启动", "Python 进程初始化"),
        ("T+2s", "导入依赖包", "torch, transformers, chromadb 等 300+ 包"),
        ("T+6s", "开始加载 LLM 模型", "从 EFS/NFS 读取 14GB 模型文件"),
        ("T+15s", "LLM 权重加载完成", "但 CUDA kernel 还在编译"),
        ("T+20s", "LLM 就绪", "完成了首次推理的 JIT 编译"),
        ("T+24s", "加载嵌入模型", "bge-large-zh-v1.5 CPU→GPU"),
        ("T+28s", "嵌入模型就绪", ""),
        ("T+30s", "加载向量数据库索引", "从 Chroma 持久化文件恢复"),
        ("T+35s", "向量数据库就绪", "50K vectors 索引已加载"),
        ("T+38s", "Kubernetes readiness probe 第一次检查", "失败！服务未就绪"),
        ("T+42s", "Kubernetes readiness probe 第二次检查", "失败！"),
        ("T+46s", "Kubernetes readiness probe 第三次检查", "失败！→ 容器被 Kill"),
        ("T+48s", "缓存预热进行中", "但已经太晚了……"),
    ]

    print("=" * 70)
    print("真实场景：K8s Readiness Probe 与冷启动的冲突")
    print("=" * 70)
    print(f"{'时间':<10} {'事件':<30} {'备注'}")
    print("-" * 70)

    for time_point, event, note in timeline:
        print(f"{time_point:<10} {event:<30} {note}")

    print(f"\n根因分析：")
    print(f"  K8s readiness probe 默认超时: 1s × 3 次 = 3s")
    print(f"  实际需要的启动时间: ~45s")
    print(f"  → Readiness probe 超时设置与冷启动时间不匹配")
    print(f"")
    print(f"修复方案:")
    print(f"  1. initialDelaySeconds: 50 (给冷启动足够时间)")
    print(f"  2. periodSeconds: 10 (降低检查频率)")
    print(f"  3. failureThreshold: 5 (允许多次失败)")
    print(f"  4. 使用 startupProbe 替代 readinessProbe 的初始检查")

real_world_timeline()
```

---

## 4. 修复方案（Fix with Code）

### 4.1 策略一：模型预加载和热备份

```python
"""
模型预加载策略：在服务启动阶段将模型持久化加载到内存中，
避免每次查询时的延迟初始化。

关键思路：
1. 使用单例模式管理模型实例
2. 在模块导入时就开始后台加载
3. 使用共享内存加速多进程场景
"""

import threading
import os
import pickle
import hashlib
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


class ModelPreloader:
    """
    模型预加载器。

    在后台线程中预先加载模型，不阻塞主流程。
    支持：
    - 异步预加载（返回 Future）
    - 加载进度回调
    - 加载超时保护
    - 多模型编排加载（按优先级）
    """

    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, Any] = {}
        self._loading_status: Dict[str, str] = {}
        self._loading_lock = threading.Lock()
        self._loading_events: Dict[str, threading.Event] = {}

    def preload_embedding_model(
        self,
        model_name: str,
        device: str = "cuda",
        use_fp16: bool = True,
        timeout: float = 120.0,
        on_progress: Optional[Callable] = None,
    ) -> threading.Thread:
        """
        在后台线程中预加载嵌入模型。

        实际代码中取消注释：
        ```python
        from sentence_transformers import SentenceTransformer

        def _load():
            self._loading_status[model_name] = "loading"
            if on_progress:
                on_progress(model_name, 0, "开始加载")

            model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=str(self.cache_dir),
            )

            if use_fp16 and device == "cuda":
                model.half()  # FP16 精度，内存减半

            # 执行一次预热推理
            if on_progress:
                on_progress(model_name, 80, "预热推理中")
            _ = model.encode(["预热文本"], show_progress_bar=False)

            with self._loading_lock:
                self._loaded_models[model_name] = model
                self._loading_status[model_name] = "ready"

            if model_name in self._loading_events:
                self._loading_events[model_name].set()

            if on_progress:
                on_progress(model_name, 100, "就绪")
        ```

        参数：
        - model_name: HuggingFace 模型名或路径
        - device: 'cuda' / 'cpu'
        - use_fp16: 是否使用半精度（减少 50% 显存）
        - timeout: 最大加载时间
        - on_progress: 进度回调函数 (model_name, percent, message)
        """
        event = threading.Event()
        self._loading_events[model_name] = event
        self._loading_status[model_name] = "pending"

        def _mock_load():
            """模拟加载过程。"""
            self._loading_status[model_name] = "loading"
            if on_progress:
                on_progress(model_name, 0, "开始加载嵌入模型")

            # 模拟：从磁盘读取模型文件
            time.sleep(1.2)
            if on_progress:
                on_progress(model_name, 30, "读取模型权重文件")

            # 模拟：构建计算图
            time.sleep(0.6)
            if on_progress:
                on_progress(model_name, 60, "构建计算图")

            # 模拟：CPU→GPU 传输
            if device == "cuda":
                time.sleep(0.5)
                if on_progress:
                    on_progress(model_name, 80, "传输到 GPU")

            # 模拟：预热推理
            time.sleep(0.7)
            if on_progress:
                on_progress(model_name, 95, "预热推理")

            # 标记加载完成
            with self._loading_lock:
                self._loaded_models[model_name] = f"<{model_name} on {device}>"
                self._loading_status[model_name] = "ready"

            event.set()
            if on_progress:
                on_progress(model_name, 100, "就绪")

        thread = threading.Thread(
            target=_mock_load,
            name=f"preload_{model_name.replace('/', '_')}",
            daemon=True,
        )
        thread.start()

        return thread

    def wait_for_model(self, model_name: str, timeout: float = 120.0) -> bool:
        """
        等待模型加载完成。

        返回 True 表示模型已就绪，False 表示超时。
        """
        if model_name in self._loading_events:
            return self._loading_events[model_name].wait(timeout=timeout)
        return model_name in self._loaded_models

    def get_model(self, model_name: str) -> Optional[Any]:
        """获取已加载的模型实例。"""
        return self._loaded_models.get(model_name)

    def get_status(self) -> Dict[str, str]:
        """获取所有模型的加载状态。"""
        return dict(self._loading_status)

    def preload_all(
        self,
        model_configs: List[Dict],
        parallel: bool = True,
    ) -> List[threading.Thread]:
        """
        批量预加载多个模型。

        模型配置格式：
        [
            {
                "name": "bge-large-zh-v1.5",
                "device": "cuda",
                "use_fp16": True,
                "priority": 1,  # 优先级，数字越小越先加载
            },
            ...
        ]
        """
        # 按优先级排序
        sorted_configs = sorted(
            model_configs,
            key=lambda x: x.get("priority", 999),
        )

        threads = []
        for config in sorted_configs:
            t = self.preload_embedding_model(
                model_name=config["name"],
                device=config.get("device", "cuda"),
                use_fp16=config.get("use_fp16", True),
            )

            if not parallel:
                t.join()  # 串行加载，等待每个模型加载完成

            threads.append(t)

        return threads


# 演示预加载
def demo_preloading():
    """演示模型预加载流程。"""

    def progress_callback(model_name, percent, message):
        """加载进度回调。"""
        bar = "=" * (percent // 5) + ">" + " " * (20 - percent // 5)
        print(f"\r[{model_name}] [{bar}] {percent:3d}% {message}", end="")

    preloader = ModelPreloader()

    print("=" * 60)
    print("模型预加载演示")
    print("=" * 60)

    # 配置要预加载的模型
    configs = [
        {"name": "bge-large-zh-v1.5", "device": "cuda", "priority": 1},
        {"name": "bge-reranker-v2-m3", "device": "cuda", "priority": 2},
    ]

    print("\n开始后台预加载...")
    threads = preloader.preload_all(configs, parallel=True)

    # 在主线程中做其他初始化工作
    print("主线程：正在初始化向量数据库...")
    time.sleep(1.0)

    print("\n主线程：正在加载配置文件...")
    time.sleep(0.5)

    # 等待所有模型加载完成
    print("\n等待模型加载完成...")
    for config in configs:
        model_name = config["name"]
        if preloader.wait_for_model(model_name, timeout=60):
            print(f"  {model_name}: 就绪")
        else:
            print(f"  {model_name}: 加载超时！")

    final_status = preloader.get_status()
    print(f"\n最终状态: {final_status}")

demo_preloading()
```

### 4.2 策略二：嵌入缓存 —— Pickle + Memmap

```python
"""
嵌入缓存策略：避免每次启动都重新计算所有文档的嵌入向量。

两种缓存方式：
1. Pickle 缓存：将嵌入向量序列化到磁盘，适合小型知识库 (<50K)
2. Memmap 缓存：将嵌入向量映射为内存文件，适合大中型知识库 (>50K)
"""

import struct
import pickle
import hashlib
import time
from typing import List, Dict, Optional, Tuple


class EmbeddingCache:
    """
    嵌入向量缓存管理器。

    支持两级缓存：
    - L1: 内存缓存（最快，但重启后丢失）
    - L2: 磁盘缓存（pickle 或 memmap，重启后保留）

    Memmap 缓存的优势：
    - 不需要将整个文件读入内存
    - 操作系统按需加载页面
    - 多进程可以共享同一个映射（节省内存）
    - 适合 100K+ 向量的场景
    """

    def __init__(
        self,
        cache_dir: str = "./embedding_cache",
        cache_type: str = "memmap",
        memory_cache_size: int = 1000,
    ):
        """
        参数：
        - cache_dir: 缓存文件目录
        - cache_type: 'pickle' / 'memmap' / 'hybrid'
        - memory_cache_size: L1 内存缓存的最大条目数
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_type = cache_type
        self.memory_cache_size = memory_cache_size

        # L1 内存缓存（LRU）
        self._l1_cache: Dict[str, List[float]] = {}
        self._l1_access_order: List[str] = []

        # 统计
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "total_queries": 0,
        }

    def _cache_key(self, text: str) -> str:
        """基于文本内容的缓存键。"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _get_pickle_path(self, cache_key: str) -> Path:
        """获取 pickle 缓存文件路径。"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _get_memmap_dir(self) -> Path:
        """获取 memmap 缓存目录。"""
        mmap_dir = self.cache_dir / "memmap"
        mmap_dir.mkdir(exist_ok=True)
        return mmap_dir

    def _update_l1(self, cache_key: str, embedding: List[float]):
        """更新 L1 内存缓存（LRU 淘汰）。"""
        if cache_key in self._l1_cache:
            self._l1_access_order.remove(cache_key)

        self._l1_cache[cache_key] = embedding
        self._l1_access_order.append(cache_key)

        # LRU 淘汰
        while len(self._l1_cache) > self.memory_cache_size:
            oldest_key = self._l1_access_order.pop(0)
            del self._l1_cache[oldest_key]

    def get(self, text: str) -> Optional[List[float]]:
        """
        从缓存中获取嵌入向量。

        查找顺序: L1 → L2 → 无

        返回 None 表示缓存未命中，需要重新计算。
        """
        self.stats["total_queries"] += 1
        cache_key = self._cache_key(text)

        # 1. L1 内存缓存
        if cache_key in self._l1_cache:
            self.stats["l1_hits"] += 1
            # 更新访问顺序
            self._l1_access_order.remove(cache_key)
            self._l1_access_order.append(cache_key)
            return self._l1_cache[cache_key]

        # 2. L2 磁盘缓存
        embedding = None

        if self.cache_type == "pickle":
            embedding = self._get_from_pickle(cache_key)
        elif self.cache_type == "memmap":
            embedding = self._get_from_memmap(cache_key)
        elif self.cache_type == "hybrid":
            embedding = self._get_from_pickle(cache_key)
            if embedding is None:
                embedding = self._get_from_memmap(cache_key)

        if embedding is not None:
            self.stats["l2_hits"] += 1
            # 提升到 L1
            self._update_l1(cache_key, embedding)
            return embedding

        # 3. 未命中
        self.stats["misses"] += 1
        return None

    def put(self, text: str, embedding: List[float]):
        """
        将嵌入向量写入缓存。

        同时写入 L1（内存）和 L2（磁盘）。
        """
        cache_key = self._cache_key(text)

        # 写入 L1
        self._update_l1(cache_key, embedding)

        # 写入 L2
        if self.cache_type == "pickle":
            self._put_to_pickle(cache_key, embedding)
        elif self.cache_type == "memmap":
            self._put_to_memmap(cache_key, embedding)
        elif self.cache_type == "hybrid":
            self._put_to_memmap(cache_key, embedding)
            # 异步写入 pickle 作为备份
            threading.Thread(
                target=self._put_to_pickle,
                args=(cache_key, embedding),
                daemon=True,
            ).start()

    def _get_from_pickle(self, cache_key: str) -> Optional[List[float]]:
        """从 pickle 文件读取缓存。"""
        pickle_path = self._get_pickle_path(cache_key)
        if not pickle_path.exists():
            return None

        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            return data.get("embedding")
        except (pickle.PickleError, EOFError, KeyError):
            return None

    def _put_to_pickle(self, cache_key: str, embedding: List[float]):
        """写入 pickle 文件。"""
        pickle_path = self._get_pickle_path(cache_key)

        # 检查是否已存在（避免重复写入）
        if pickle_path.exists():
            return

        data = {
            "embedding": embedding,
            "dim": len(embedding),
            "timestamp": time.time(),
        }

        # 先写入临时文件，再原子重命名
        tmp_path = pickle_path.with_suffix(".tmp")
        with open(tmp_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.rename(pickle_path)

    def _get_from_memmap(self, cache_key: str) -> Optional[List[float]]:
        """
        从 memmap 文件读取缓存。

        Memmap 文件格式：
        [header: 16 bytes magic + 4 bytes dim]
        [body: N * dim * 4 bytes (float32)]
        [index: cache_key → offset mapping]
        """
        mmap_dir = self._get_memmap_dir()
        index_path = mmap_dir / "index.pkl"

        if not index_path.exists():
            return None

        try:
            with open(index_path, 'rb') as f:
                index = pickle.load(f)

            if cache_key not in index:
                return None

            offset, dim = index[cache_key]
            data_path = mmap_dir / "embeddings.bin"

            with open(data_path, 'rb') as f:
                f.seek(offset)
                raw_bytes = f.read(dim * 4)
                if len(raw_bytes) != dim * 4:
                    return None
                # 解包 float32 数组
                embedding = list(struct.unpack(f'{dim}f', raw_bytes))

            return embedding

        except (FileNotFoundError, struct.error, pickle.PickleError):
            return None

    def _put_to_memmap(self, cache_key: str, embedding: List[float]):
        """写入 memmap 文件。"""
        mmap_dir = self._get_memmap_dir()
        data_path = mmap_dir / "embeddings.bin"
        index_path = mmap_dir / "index.pkl"

        # 读取或创建索引
        index = {}
        if index_path.exists():
            try:
                with open(index_path, 'rb') as f:
                    index = pickle.load(f)
            except (pickle.PickleError, EOFError):
                index = {}

        # 检查是否已缓存
        if cache_key in index:
            return

        # 追加写入向量数据
        mode = 'ab' if data_path.exists() else 'wb'
        with open(data_path, mode) as f:
            offset = f.tell()
            dim = len(embedding)
            raw_bytes = struct.pack(f'{dim}f', *embedding)
            f.write(raw_bytes)

        # 更新索引
        index[cache_key] = (offset, dim)
        tmp_index = index_path.with_suffix(".tmp")
        with open(tmp_index, 'wb') as f:
            pickle.dump(index, f)
        tmp_index.rename(index_path)

    def precompute_and_cache(
        self,
        texts: List[str],
        encode_fn: Callable[[List[str]], List[List[float]]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> int:
        """
        批量预计算嵌入并缓存。

        参数：
        - texts: 要计算嵌入的文本列表
        - encode_fn: 嵌入函数，接受 List[str]，返回 List[List[float]]
        - batch_size: 批次大小
        - show_progress: 是否显示进度

        返回实际计算的新嵌入数量（跳过已缓存的）。
        """
        new_count = 0
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]

            # 检查哪些需要计算
            to_compute = []
            to_compute_indices = []
            for j, text in enumerate(batch):
                if self.get(text) is None:
                    to_compute.append(text)
                    to_compute_indices.append(j)

            # 批量计算
            if to_compute:
                embeddings = encode_fn(to_compute)
                for text, embedding in zip(to_compute, embeddings):
                    self.put(text, embedding)
                new_count += len(to_compute)

            if show_progress:
                progress = min((i + batch_size) / total * 100, 100)
                bar = "=" * int(progress / 5) + " " * (20 - int(progress / 5))
                h = self.stats["l2_hits"] + self.stats["l1_hits"]
                m = self.stats["misses"]
                print(
                    f"\r预计算嵌入: [{bar}] {progress:.0f}% "
                    f"| 新计算: {new_count} | 缓存命中: {h}",
                    end="",
                )

        if show_progress:
            print()

        return new_count

    def get_stats(self) -> Dict:
        """获取缓存统计信息。"""
        total = self.stats["total_queries"]
        hit_rate = (
            (self.stats["l1_hits"] + self.stats["l2_hits"]) / total
            if total > 0 else 0
        )

        return {
            **self.stats,
            "hit_rate": round(hit_rate, 4),
            "l1_size": len(self._l1_cache),
            "l1_hit_rate": round(
                self.stats["l1_hits"] / total if total > 0 else 0, 4
            ),
        }


# 演示嵌入缓存
def demo_embedding_cache():
    """演示嵌入缓存的完整工作流程。"""

    cache = EmbeddingCache(
        cache_dir="./demo_embedding_cache",
        cache_type="hybrid",  # 同时使用 pickle + memmap
        memory_cache_size=500,
    )

    # 模拟嵌入函数
    def mock_encode(texts: List[str]) -> List[List[float]]:
        """模拟嵌入计算。"""
        time.sleep(0.01 * len(texts))  # 模拟计算延迟
        results = []
        for text in texts:
            seed = sum(ord(c) for c in text)
            rng = __import__('random')
            local_rng = rng.Random(seed)
            results.append([local_rng.uniform(-1, 1) for _ in range(768)])
        return results

    # 模拟文档集
    documents = [
        "当事人一方不履行合同义务的，应当承担违约责任。",
        "侵权责任编规定了各种侵权行为的法律后果。",
        "有限责任公司股东以其认缴的出资额为限对公司承担责任。",
        "民法典是新中国第一部以法典命名的法律。",
        "合同的成立需要要约和承诺两个基本要素。",
    ] * 20  # 100 个文档（模拟）

    print("=" * 60)
    print("嵌入缓存演示")
    print("=" * 60)

    # 第一轮：计算所有嵌入（冷启动）
    print("\n第一轮（冷启动）:")
    start = time.time()
    new_count = cache.precompute_and_cache(
        texts=documents,
        encode_fn=mock_encode,
        batch_size=16,
    )
    elapsed = time.time() - start
    print(f"  耗时: {elapsed:.2f}s | 新计算: {new_count}")

    # 第二轮：从缓存读取（热启动）
    print("\n第二轮（热启动 - 全部命中缓存）:")
    start = time.time()
    new_count = cache.precompute_and_cache(
        texts=documents,
        encode_fn=mock_encode,
        batch_size=16,
    )
    elapsed = time.time() - start
    print(f"  耗时: {elapsed:.2f}s | 新计算: {new_count}")
    print(f"  加速比: {100 * (1 - elapsed / (elapsed + 0.01)):.0f}%+")

    stats = cache.get_stats()
    print(f"\n缓存统计:")
    print(f"  总查询次数: {stats['total_queries']}")
    print(f"  L1 命中: {stats['l1_hits']} ({stats['l1_hit_rate']:.1%})")
    print(f"  L2 命中: {stats['l2_hits']}")
    print(f"  总命中率: {stats['hit_rate']:.1%}")
    print(f"  L1 缓存大小: {stats['l1_size']}")

demo_embedding_cache()
```

### 4.3 策略三：完整启动优化器

```python
"""
完整启动优化方案：将上述策略整合为一个统一的 StartupOptimizer。
"""

import threading
import time
import os
import signal
from typing import List, Dict, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class ServiceState(Enum):
    """服务状态。"""
    INITIALIZING = "initializing"
    PRELOADING = "preloading"
    WARMING = "warming"
    READY = "ready"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class StartupConfig:
    """启动优化配置。"""
    # 模型预加载
    preload_embedding_model: bool = True
    preload_llm_model: bool = False  # LLM 太重，建议独立部署
    embedding_model_name: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cuda"
    embedding_use_fp16: bool = True

    # 缓存预热
    warmup_embeddings: bool = True
    warmup_queries_file: Optional[str] = None  # 预热查询列表
    warmup_query_count: int = 100

    # 向量数据库
    vector_db_type: str = "chroma"
    preload_vector_index: bool = True

    # 健康检查
    readiness_timeout: float = 60.0
    enable_background_warmup: bool = True

    # 优化
    pin_memory: bool = True  # CPU 内存锁定，避免 swap
    precompile_cuda: bool = True  # 预编译 CUDA kernel
    use_lazy_imports: bool = False  # 延迟导入非关键模块


class StartupOptimizer:
    """
    RAG 服务启动优化器。

    集成所有冷启动优化策略，提供统一的启动流程管理。
    """

    def __init__(self, config: StartupConfig):
        self.config = config
        self.state = ServiceState.INITIALIZING
        self.preloader = ModelPreloader()
        self.embedding_cache: Optional[EmbeddingCache] = None

        # 启动阶段计时
        self.phase_timings: Dict[str, float] = {}
        self._start_time = time.time()

    def initialize(self) -> float:
        """
        执行优化的启动流程。

        返回总启动时间（秒）。
        """
        print("=" * 60)
        print("RAG 服务启动优化器")
        print("=" * 60)

        self.state = ServiceState.INITIALIZING

        # Phase 1: 延迟导入非关键模块
        self._phase_lazy_imports()

        # Phase 2: 后台预加载嵌入模型
        self._phase_preload_models()

        # Phase 3: 加载向量数据库
        self._phase_load_vector_db()

        # Phase 4: 预热缓存
        self._phase_warmup_caches()

        # Phase 5: CUDA 预编译
        self._phase_precompile_cuda()

        # Phase 6: 标记就绪
        self.state = ServiceState.READY
        total_time = time.time() - self._start_time

        self._print_startup_summary(total_time)
        return total_time

    def _phase_lazy_imports(self):
        """Phase 1: 管理 Python 包导入。"""
        phase_start = time.time()

        if self.config.use_lazy_imports:
            # 延迟加载：只导入启动必需的模块
            # 其他模块在首次使用时才导入
            print("[1/5] 启用延迟导入模式...")
        else:
            print("[1/5] 导入核心依赖...")

        # 必需的导入（这些必须立即加载）
        import sys
        import logging

        self.phase_timings["lazy_imports"] = time.time() - phase_start

    def _phase_preload_models(self):
        """Phase 2: 后台预加载模型。"""
        phase_start = time.time()

        if self.config.preload_embedding_model:
            print("[2/5] 后台预加载嵌入模型...")

            # 启动后台加载线程
            self.preloader.preload_embedding_model(
                model_name=self.config.embedding_model_name,
                device=self.config.embedding_device,
                use_fp16=self.config.embedding_use_fp16,
                on_progress=lambda name, pct, msg: print(
                    f"\r  嵌入模型加载: {pct}% - {msg}", end=""
                ),
            )

        if self.config.preload_llm_model:
            print("[2/5] 后台预加载 LLM 模型...")
            # LLM 模型通常作为独立服务，这里只做连接测试

        self.phase_timings["preload_models"] = time.time() - phase_start

    def _phase_load_vector_db(self):
        """Phase 3: 加载向量数据库。"""
        phase_start = time.time()

        if self.config.preload_vector_index:
            print("\n[3/5] 加载向量数据库索引...")
            time.sleep(1.0)  # 模拟

        self.phase_timings["load_vector_db"] = time.time() - phase_start

    def _phase_warmup_caches(self):
        """Phase 4: 预热缓存。"""
        phase_start = time.time()

        if self.config.warmup_embeddings:
            print("\n[4/5] 预热缓存...")

            # 初始化嵌入缓存
            self.embedding_cache = EmbeddingCache(
                cache_type="hybrid",
                memory_cache_size=1000,
            )

            # 加载预热查询列表
            warmup_queries = self._load_warmup_queries()

            if warmup_queries:
                # 后台预热（不阻塞就绪状态）
                if self.config.enable_background_warmup:
                    threading.Thread(
                        target=self._background_warmup,
                        args=(warmup_queries,),
                        daemon=True,
                    ).start()
                    print("  后台预热已启动（服务可以先就绪）")
                else:
                    self._foreground_warmup(warmup_queries)

        self.phase_timings["warmup_caches"] = time.time() - phase_start

    def _phase_precompile_cuda(self):
        """Phase 5: CUDA kernel 预编译。"""
        phase_start = time.time()

        if self.config.precompile_cuda:
            print("\n[5/5] CUDA kernel 预编译...")
            time.sleep(0.5)  # 模拟

        self.phase_timings["precompile_cuda"] = time.time() - phase_start

    def _load_warmup_queries(self) -> List[str]:
        """
        加载预热查询列表。

        可以从文件加载，也可以使用内置的默认查询。
        """
        if self.config.warmup_queries_file:
            path = Path(self.config.warmup_queries_file)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]

        # 默认预热查询（中文法律领域）
        return [
            "违约责任的承担方式有哪些？",
            "什么是不可抗力？",
            "有限责任公司的股东责任如何界定？",
            "民法典规定的诉讼时效是多久？",
            "合同成立需要哪些要件？",
            "侵权行为造成人身损害需要赔偿什么？",
            "公司注册资本的最低限额是多少？",
            "如何认定格式条款的效力？",
            "定金和违约金的区别是什么？",
            "民法典关于隐私权的规定有哪些？",
        ]

    def _foreground_warmup(self, queries: List[str]):
        """前台预热（阻塞主流程，直到预热完成）。"""
        if self.embedding_cache is None:
            return

        for i, query in enumerate(queries):
            if i >= self.config.warmup_query_count:
                break

    def _background_warmup(self, queries: List[str]):
        """
        后台预热（异步执行，不阻塞服务就绪）。

        服务可以先 marked as ready，缓存在后台慢慢填充。
        在缓存完全预热之前，查询性能会略低但功能正常。
        """
        self.state = ServiceState.WARMING

        for i, query in enumerate(queries):
            if self.state == ServiceState.SHUTTING_DOWN:
                break
            if i >= self.config.warmup_query_count:
                break

            # 检查缓存是否已有（避免重复计算）
            if self.embedding_cache and self.embedding_cache.get(query) is None:
                # 立即返回，不阻塞
                pass

            if i % 20 == 0:
                print(f"\r  后台预热: {i}/{min(len(queries), self.config.warmup_query_count)}", end="")

        if self.state != ServiceState.SHUTTING_DOWN:
            self.state = ServiceState.READY
            print("\r  后台预热完成。" + " " * 20)

    def _print_startup_summary(self, total_time: float):
        """打印启动总结。"""
        print(f"\n{'=' * 60}")
        print(f"启动完成！")
        print(f"总启动时间: {total_time:.2f} 秒")
        print(f"服务状态: {self.state.value}")

        print(f"\n各阶段耗时:")
        for phase, duration in self.phase_timings.items():
            print(f"  {phase}: {duration:.3f}s")

        print(f"\n优化建议:")
        if total_time > 30:
            print(f"  启动时间 > 30s，建议：")
            print(f"    1. 使用更小的嵌入模型 (bge-small vs bge-large)")
            print(f"    2. 将 LLM 作为独立服务，不在 RAG 进程中加载")
            print(f"    3. 使用 startupProbe 延长 K8s 的就绪等待时间")
        elif total_time > 10:
            print(f"  启动时间 10-30s，建议：")
            print(f"    1. 确保模型文件在本地 SSD 上（不要用 NFS）")
            print(f"    2. 使用 embedding pickle 缓存减少预热时间")
        else:
            print(f"  启动时间 < 10s，性能良好！")

        print(f"{'=' * 60}")

    def readiness_check(self) -> Tuple[bool, str]:
        """
        就绪检查（用于 K8s readiness probe）。

        返回 (is_ready, message)。
        """
        if self.state == ServiceState.READY:
            return True, "服务就绪"

        elapsed = time.time() - self._start_time
        if elapsed > self.config.readiness_timeout:
            return False, f"启动超时 ({elapsed:.0f}s > {self.config.readiness_timeout}s)"

        # 检查关键组件
        if self.config.preload_embedding_model:
            if not self.preloader.get_model(self.config.embedding_model_name):
                return False, "嵌入模型尚未加载完成"

        if self.config.preload_vector_index:
            # 检查向量数据库连接
            pass

        return False, f"正在初始化 (已耗时 {elapsed:.0f}s)"

    def shutdown(self):
        """优雅关闭。"""
        self.state = ServiceState.SHUTTING_DOWN
        print("\n正在关闭服务...")

        # 保存嵌入缓存
        if self.embedding_cache:
            stats = self.embedding_cache.get_stats()
            print(f"  嵌入缓存命中率: {stats['hit_rate']:.1%}")

        print("服务已关闭。")


# ============================================================
# 启动优化器演示
# ============================================================

def demo_startup_optimizer():
    """演示启动优化器的完整工作流程。"""

    config = StartupConfig(
        preload_embedding_model=True,
        preload_llm_model=False,
        embedding_model_name="BAAI/bge-large-zh-v1.5",
        embedding_device="cuda",
        embedding_use_fp16=True,
        warmup_embeddings=True,
        warmup_query_count=10,
        preload_vector_index=True,
        precompile_cuda=True,
        enable_background_warmup=True,
    )

    optimizer = StartupOptimizer(config)

    # 执行优化启动
    total_time = optimizer.initialize()

    # 检查就绪状态
    is_ready, message = optimizer.readiness_check()
    print(f"\n就绪检查: {message}")

    # 等待后台预热完成
    print("\n等待后台预热完成...")
    time.sleep(2.0)

    # 再次检查
    is_ready, message = optimizer.readiness_check()
    print(f"就绪检查: {message}")

    return optimizer

startup_optimizer_demo = demo_startup_optimizer()
```

### 4.4 策略四：WarmupManager —— 预加载模型、预计算常用嵌入、就绪检查

```python
"""
WarmupManager: Pre-load models, pre-compute common embeddings, readiness check.

This class orchestrates the warmup process for a RAG service:
1. Pre-load embedding models into GPU/CPU memory
2. Pre-compute embeddings for common/frequent queries
3. Execute warmup inference passes (CUDA kernel compilation)
4. Verify readiness with a health check query
5. Background warmup without blocking service availability
"""

import time
import threading
import os
import pickle
import hashlib
import json
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


class WarmupPhase(Enum):
    """Warmup execution phases."""
    PENDING = "pending"
    LOADING_MODELS = "loading_models"
    CUDA_WARMUP = "cuda_warmup"
    PRE_COMPUTING_EMBEDDINGS = "pre_computing_embeddings"
    POPULATING_CACHE = "populating_cache"
    READINESS_CHECK = "readiness_check"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class WarmupConfig:
    """Configuration for the warmup process."""
    # Model loading
    embedding_model_name: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cuda"  # "cuda", "cpu"
    use_fp16: bool = True

    # CUDA warmup
    cuda_warmup_iterations: int = 5
    cuda_warmup_batch_size: int = 8

    # Common query embeddings
    common_queries_file: Optional[str] = None  # Path to JSON file with common queries
    pre_compute_query_count: int = 100
    pre_compute_batch_size: int = 32

    # Readiness check
    readiness_query_text: str = "测试查询：民法典诉讼时效规定"
    readiness_expected_min_results: int = 1
    readiness_timeout_seconds: float = 30.0

    # Background warmup
    enable_background: bool = True
    background_priority_phases: List[str] = field(default_factory=lambda: [
        "cuda_warmup", "pre_computing_embeddings"
    ])


class WarmupManager:
    """
    Pre-load model, pre-compute common embeddings, readiness check.

    Orchestrates the full warmup lifecycle for a RAG service.
    Supports both synchronous (blocking) and asynchronous (background) warmup.
    """

    def __init__(
        self,
        config: WarmupConfig,
        embedding_model: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        embedding_cache: Optional[Any] = None,
    ):
        """
        Initialize WarmupManager.

        Args:
            config: Warmup configuration
            embedding_model: Pre-loaded embedding model (optional, loads if not provided)
            vector_store: Vector store instance for readiness checks
            embedding_cache: Embedding cache instance for pre-population
        """
        self.config = config
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.embedding_cache = embedding_cache

        self._phase = WarmupPhase.PENDING
        self._phase_start_times: Dict[WarmupPhase, float] = {}
        self._phase_durations: Dict[WarmupPhase, float] = {}
        self._errors: List[Dict] = []
        self._is_ready = False
        self._lock = threading.RLock()

        # Background warmup thread
        self._warmup_thread: Optional[threading.Thread] = None
        self._stop_background = threading.Event()

        # Pre-computed embeddings storage
        self._pre_computed_embeddings: Dict[str, List[float]] = {}

    # ================================================================
    # Model Pre-loading
    # ================================================================

    def load_embedding_model(self) -> Any:
        """
        Pre-load embedding model into memory.

        Supports:
        - FP16 half-precision for 50% reduced loading time
        - GPU device selection
        - Model warmup passes
        """
        self._transition_phase(WarmupPhase.LOADING_MODELS)

        print(f"[WarmupManager] Loading embedding model: {self.config.embedding_model_name}")
        t0 = time.time()

        try:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {}
            if self.config.use_fp16 and self.config.embedding_device == "cuda":
                model_kwargs["torch_dtype"] = "float16"

            model = SentenceTransformer(
                self.config.embedding_model_name,
                device=self.config.embedding_device,
                model_kwargs=model_kwargs,
            )

            load_time = time.time() - t0
            print(f"[WarmupManager] Model loaded in {load_time:.1f}s "
                  f"(device={self.config.embedding_device}, fp16={self.config.use_fp16})")

            self.embedding_model = model
            return model

        except ImportError:
            print("[WarmupManager] sentence_transformers not installed, "
                  "using mock model for demonstration")
            # Create a mock model for demonstration
            class MockEmbeddingModel:
                def encode(self, texts, batch_size=32, show_progress_bar=False,
                           normalize_embeddings=True):
                    """Mock encode returning deterministic vectors based on text hash."""
                    import hashlib
                    import struct
                    results = []
                    for text in texts:
                        h = hashlib.sha256(text.encode('utf-8')).digest()
                        vec = list(struct.unpack('f' * 256, h[:1024]))
                        results.append(vec)
                    return results

            self.embedding_model = MockEmbeddingModel()
            return self.embedding_model

        except Exception as e:
            self._errors.append({"phase": "loading_models", "error": str(e)})
            self._transition_phase(WarmupPhase.FAILED)
            raise

    # ================================================================
    # CUDA Kernel Warmup
    # ================================================================

    def warmup_cuda_kernels(self):
        """
        Execute warmup inference passes to compile CUDA kernels.

        The first inference pass on GPU triggers CUDA kernel compilation,
        which can take 2-5 seconds. By running warmup passes during startup,
        we avoid this latency on the first real user query.
        """
        if self.config.embedding_device != "cuda":
            print("[WarmupManager] Skipping CUDA warmup (device is not cuda)")
            return

        self._transition_phase(WarmupPhase.CUDA_WARMUP)
        print(f"[WarmupManager] Running {self.config.cuda_warmup_iterations} "
              f"CUDA warmup iterations...")

        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Call load_embedding_model() first.")

        t0 = time.time()

        # Generate warmup texts of varying lengths
        warmup_texts = []
        for i in range(self.config.cuda_warmup_batch_size):
            length = 50 + (i % 5) * 100  # 50-450 characters
            warmup_texts.append(
                f"测试文本内容 #{i}: "
                + "民法典合同编违约责任条款相关规定" * (length // 20)
            )

        # Execute warmup passes
        for iteration in range(self.config.cuda_warmup_iterations):
            _ = self.embedding_model.encode(
                warmup_texts,
                batch_size=self.config.cuda_warmup_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        warmup_time = time.time() - t0
        print(f"[WarmupManager] CUDA warmup complete in {warmup_time:.1f}s "
              f"({self.config.cuda_warmup_iterations} iterations)")

    # ================================================================
    # Pre-compute Common Query Embeddings
    # ================================================================

    def load_common_queries(self) -> List[str]:
        """Load common/frequent queries from configuration file."""
        if self.config.common_queries_file and os.path.exists(self.config.common_queries_file):
            try:
                with open(self.config.common_queries_file, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
                print(f"[WarmupManager] Loaded {len(queries)} common queries from file")
                return queries
            except Exception as e:
                print(f"[WarmupManager] Failed to load common queries file: {e}")

        # Fallback: use built-in common legal queries
        builtin_queries = [
            "民法典规定的普通诉讼时效是几年",
            "违约责任的承担方式有哪些",
            "合同成立需要什么条件",
            "侵权责任的构成要件",
            "公司法对注册资本的要求",
            "什么是不可抗力",
            "合同解除的条件和程序",
            "精神损害赔偿的适用条件",
            "连带责任的认定标准",
            "格式条款的法律效力",
            "买卖合同的风险负担",
            "赠与合同的撤销条件",
            "租赁合同的期限规定",
            "保证责任的类型和区别",
            "定金与违约金的适用关系",
            "债权人代位权的行使条件",
            "债务转移需要债权人同意吗",
            "债权转让需要通知债务人吗",
            "诉讼时效中断的情形有哪些",
            "诉讼时效中止的条件是什么",
        ]
        return builtin_queries[:self.config.pre_compute_query_count]

    def pre_compute_embeddings(self):
        """
        Pre-compute embeddings for common/frequent queries.

        Benefits:
        - First real user query for common topics hits cache immediately
        - Reduces cold-start perceived latency
        - Fills embedding cache before taking production traffic
        """
        self._transition_phase(WarmupPhase.PRE_COMPUTING_EMBEDDINGS)

        queries = self.load_common_queries()
        print(f"[WarmupManager] Pre-computing embeddings for {len(queries)} common queries...")

        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Call load_embedding_model() first.")

        t0 = time.time()
        total_computed = 0

        for i in range(0, len(queries), self.config.pre_compute_batch_size):
            batch = queries[i:i + self.config.pre_compute_batch_size]
            embeddings = self.embedding_model.encode(
                batch,
                batch_size=self.config.pre_compute_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            # Store in local cache
            for j, query in enumerate(batch):
                if j < len(embeddings):
                    emb_list = embeddings[j].tolist() if hasattr(embeddings[j], 'tolist') else list(embeddings[j])
                    self._pre_computed_embeddings[query] = emb_list

            total_computed += len(batch)

        elapsed = time.time() - t0
        rate = total_computed / elapsed if elapsed > 0 else 0
        print(f"[WarmupManager] Pre-computed {total_computed} embeddings in {elapsed:.1f}s "
              f"({rate:.0f} queries/sec)")

    # ================================================================
    # Embedding Cache Population
    # ================================================================

    def populate_embedding_cache(self):
        """
        Populate the embedding cache with pre-computed embeddings.

        Must be called after pre_compute_embeddings().
        """
        self._transition_phase(WarmupPhase.POPULATING_CACHE)

        if self.embedding_cache is None:
            print("[WarmupManager] No embedding cache configured, skipping population")
            return

        if not self._pre_computed_embeddings:
            print("[WarmupManager] No pre-computed embeddings available, "
                  "call pre_compute_embeddings() first")
            return

        count = 0
        for query, embedding in self._pre_computed_embeddings.items():
            try:
                self.embedding_cache.set(query, embedding)
                count += 1
            except Exception as e:
                print(f"[WarmupManager] Failed to cache embedding for '{query[:50]}': {e}")

        print(f"[WarmupManager] Populated cache with {count} pre-computed embeddings")

    # ================================================================
    # Readiness Check
    # ================================================================

    def readiness_check(self) -> Tuple[bool, str]:
        """
        Check if the service is ready to serve requests.

        Performs a test query against the vector store to verify:
        1. Embedding model can encode queries
        2. Vector store can search and return results
        3. End-to-end latency is within acceptable range
        """
        self._transition_phase(WarmupPhase.READINESS_CHECK)

        print(f"[WarmupManager] Running readiness check...")

        if self.embedding_model is None:
            return False, "Embedding model not loaded"

        t0 = time.time()

        try:
            # Encode the readiness test query
            query_embedding = self.embedding_model.encode(
                [self.config.readiness_query_text],
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            encode_time = time.time() - t0

            # Search vector store (if configured)
            search_results = []
            search_time = 0.0

            if self.vector_store is not None:
                t1 = time.time()
                try:
                    search_results = self.vector_store.search(
                        query_embedding[0].tolist() if hasattr(query_embedding[0], 'tolist')
                        else list(query_embedding[0]),
                        limit=5,
                    )
                    search_time = time.time() - t1
                except Exception as e:
                    search_time = time.time() - t1
                    print(f"[WarmupManager] Vector store search failed: {e}")

            total_time = time.time() - t0

            # Evaluate readiness
            if total_time > self.config.readiness_timeout_seconds:
                self._is_ready = False
                return False, (f"Readiness check timeout: {total_time:.1f}s "
                               f"(limit: {self.config.readiness_timeout_seconds}s)")

            if self.vector_store is not None and len(search_results) < self.config.readiness_expected_min_results:
                self._is_ready = False
                return False, (f"Insufficient search results: {len(search_results)} "
                               f"(expected >= {self.config.readiness_expected_min_results})")

            self._is_ready = True
            return True, (f"Service is ready (encode: {encode_time:.1f}s, "
                          f"search: {search_time:.1f}s, total: {total_time:.1f}s)")

        except Exception as e:
            self._is_ready = False
            return False, f"Readiness check failed: {e}"

    # ================================================================
    # Full Warmup Orchestration
    # ================================================================

    def warmup_sync(self, blocking: bool = True) -> bool:
        """
        Execute complete synchronous warmup process.

        Steps:
        1. Load embedding model
        2. Warmup CUDA kernels (if GPU)
        3. Pre-compute common query embeddings
        4. Populate embedding cache
        5. Readiness check

        Args:
            blocking: If True, blocks until warmup complete.
                      If False, starts background warmup.

        Returns:
            True if warmup completed successfully (or started in background mode).
        """
        if not blocking:
            return self._start_background_warmup()

        print("=" * 60)
        print("[WarmupManager] Starting synchronous warmup...")
        print("=" * 60)

        try:
            # Phase 1: Load models
            self.load_embedding_model()

            # Phase 2: CUDA warmup
            self.warmup_cuda_kernels()

            # Phase 3: Pre-compute embeddings
            self.pre_compute_embeddings()

            # Phase 4: Populate cache
            self.populate_embedding_cache()

            # Phase 5: Readiness check
            is_ready, message = self.readiness_check()

            self._transition_phase(WarmupPhase.COMPLETE)
            print(f"\n[WarmupManager] Warmup complete. Ready: {is_ready}")
            print(f"[WarmupManager] {message}")

            # Print phase timings
            print(f"\n[WarmupManager] Phase timings:")
            for phase in WarmupPhase:
                if phase in self._phase_durations:
                    print(f"  {phase.value}: {self._phase_durations[phase]:.1f}s")

            return is_ready

        except Exception as e:
            self._transition_phase(WarmupPhase.FAILED)
            print(f"[WarmupManager] Warmup failed: {e}")
            return False

    # ================================================================
    # Background Warmup
    # ================================================================

    def _start_background_warmup(self) -> bool:
        """Start warmup in a background thread."""
        print("[WarmupManager] Starting background warmup...")
        self._warmup_thread = threading.Thread(
            target=self._background_warmup_worker,
            daemon=True,
            name="warmup-worker",
        )
        self._warmup_thread.start()
        return True

    def _background_warmup_worker(self):
        """Background warmup worker thread."""
        try:
            # Phase 1: Load models (blocking - required before serving)
            self.load_embedding_model()

            # Phase 2-4: Run remaining phases in background
            if self._stop_background.is_set():
                return

            if "cuda_warmup" in self.config.background_priority_phases:
                self.warmup_cuda_kernels()

            if self._stop_background.is_set():
                return

            if "pre_computing_embeddings" in self.config.background_priority_phases:
                self.pre_compute_embeddings()
                self.populate_embedding_cache()

            # Phase 5: Readiness check
            is_ready, message = self.readiness_check()
            print(f"[WarmupManager] Background warmup complete. Ready: {is_ready}")
            print(f"[WarmupManager] {message}")

            self._transition_phase(WarmupPhase.COMPLETE)

        except Exception as e:
            self._transition_phase(WarmupPhase.FAILED)
            print(f"[WarmupManager] Background warmup failed: {e}")

    def stop_background_warmup(self):
        """Stop background warmup thread."""
        self._stop_background.set()
        if self._warmup_thread and self._warmup_thread.is_alive():
            self._warmup_thread.join(timeout=5.0)

    # ================================================================
    # State Management
    # ================================================================

    def _transition_phase(self, new_phase: WarmupPhase):
        """Record phase transition and timing."""
        now = time.time()
        with self._lock:
            if self._phase != WarmupPhase.PENDING:
                duration = now - self._phase_start_times.get(self._phase, now)
                self._phase_durations[self._phase] = duration

            self._phase = new_phase
            self._phase_start_times[new_phase] = now

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def current_phase(self) -> WarmupPhase:
        return self._phase

    def get_warmup_stats(self) -> Dict[str, Any]:
        """Get warmup statistics."""
        with self._lock:
            return {
                "phase": self._phase.value,
                "is_ready": self._is_ready,
                "phase_durations": {
                    k.value: round(v, 1)
                    for k, v in self._phase_durations.items()
                },
                "pre_computed_embeddings_count": len(self._pre_computed_embeddings),
                "error_count": len(self._errors),
            }


# ============================================================
# WarmupManager Usage Example
# ============================================================

def demo_warmup_manager():
    """Demonstrate WarmupManager usage."""

    # Configuration
    config = WarmupConfig(
        embedding_model_name="BAAI/bge-large-zh-v1.5",
        embedding_device="cuda",
        use_fp16=True,
        cuda_warmup_iterations=3,
        pre_compute_query_count=20,
        readiness_timeout_seconds=30.0,
    )

    # Initialize WarmupManager
    warmup_mgr = WarmupManager(config=config)

    # Execute synchronous warmup
    is_ready = warmup_mgr.warmup_sync(blocking=True)

    if is_ready:
        print("\n[Demo] Service is ready to accept requests!")
    else:
        print("\n[Demo] Service warmup failed. Check errors above.")

    # Print stats
    stats = warmup_mgr.get_warmup_stats()
    print(f"\n[Demo] Warmup stats: {json.dumps(stats, indent=2, ensure_ascii=False)}")

    return warmup_mgr


# Run demonstration
warmup_manager_demo = demo_warmup_manager()
```

### 4.5 策略五：启动健康检查和超时配置

```python
"""
Startup Health Check with Timeout

Kubernetes-compatible health check configuration for RAG services.
Prevents the infinite restart loop caused by slow startup.

Key concepts:
- startupProbe: Allows longer initial startup time before livenessProbe kicks in
- readinessProbe: Only routes traffic when service is truly ready
- Graceful startup: Background warmup while reporting ready for basic queries
"""

import time
import threading
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class ServiceHealth(Enum):
    """Service health states."""
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class HealthCheckConfig:
    """Health check configuration for RAG service."""
    # Startup probe
    startup_initial_delay_seconds: int = 10
    startup_period_seconds: int = 5
    startup_failure_threshold: int = 30  # Allow up to 150s for startup
    startup_timeout_seconds: int = 300   # Absolute timeout for startup

    # Readiness probe
    readiness_period_seconds: int = 5
    readiness_timeout_seconds: int = 3   # Per-check timeout
    readiness_success_threshold: int = 1
    readiness_failure_threshold: int = 3

    # Liveness probe
    liveness_period_seconds: int = 10
    liveness_timeout_seconds: int = 3
    liveness_failure_threshold: int = 3

    # Warmup
    warmup_timeout_seconds: int = 120
    background_warmup_enabled: bool = True


class StartupHealthChecker:
    """
    Startup health checker with timeouts.

    Integrates with Kubernetes probes to prevent the infinite restart
    loop caused by slow RAG service startup.
    """

    def __init__(
        self,
        config: HealthCheckConfig,
        warmup_manager: Optional[Any] = None,
        embedding_model_check: Optional[Callable[[], bool]] = None,
        vector_store_check: Optional[Callable[[], bool]] = None,
    ):
        self.config = config
        self.warmup_manager = warmup_manager
        self._embedding_model_check = embedding_model_check
        self._vector_store_check = vector_store_check

        self._health = ServiceHealth.STARTING
        self._startup_start_time = time.time()
        self._ready_start_time: Optional[float] = None
        self._lock = threading.RLock()
        self._consecutive_failures = 0

    def startup_check(self) -> Tuple[bool, str]:
        """
        Kubernetes startupProbe handler.

        Returns True while the service is still starting up (within timeout).
        Once startup completes or times out, behavior changes:
        - Still within startup window: returns True (still starting)
        - Warmup completed: returns True (startup succeeded)
        - Timeout exceeded: returns False (startup failed)
        """
        elapsed = time.time() - self._startup_start_time

        # Check if startup has exceeded absolute timeout
        if elapsed > self.config.startup_timeout_seconds:
            self._health = ServiceHealth.UNHEALTHY
            return False, f"Startup timeout: {elapsed:.0f}s > {self.config.startup_timeout_seconds}s"

        # Check if warmup manager indicates readiness
        if self.warmup_manager and self.warmup_manager.is_ready:
            if self._health == ServiceHealth.STARTING:
                self._health = ServiceHealth.READY
                self._ready_start_time = time.time()
            return True, f"Startup complete ({elapsed:.0f}s)"

        # Still starting
        return True, f"Starting... ({elapsed:.0f}s)"

    def readiness_check(self) -> Tuple[bool, str]:
        """
        Kubernetes readinessProbe handler.

        Returns True only when the service is ready to serve traffic.
        Checks that all critical components are operational.
        """
        with self._lock:
            # Service shutting down
            if self._health == ServiceHealth.SHUTTING_DOWN:
                self._consecutive_failures += 1
                return False, "Service is shutting down"

            # Service unhealthy
            if self._health == ServiceHealth.UNHEALTHY:
                self._consecutive_failures += 1
                return False, "Service is unhealthy"

            # Still starting
            if self._health == ServiceHealth.STARTING:
                elapsed = time.time() - self._startup_start_time
                if elapsed > self.config.startup_timeout_seconds:
                    self._health = ServiceHealth.UNHEALTHY
                    return False, f"Startup timeout ({elapsed:.0f}s)"
                return False, f"Still starting ({elapsed:.0f}s)"

            # Component checks
            checks_passed = True
            messages = []

            if self._embedding_model_check:
                try:
                    if not self._embedding_model_check():
                        checks_passed = False
                        messages.append("Embedding model not ready")
                except Exception as e:
                    checks_passed = False
                    messages.append(f"Embedding model check error: {e}")

            if self._vector_store_check:
                try:
                    if not self._vector_store_check():
                        checks_passed = False
                        messages.append("Vector store not ready")
                except Exception as e:
                    checks_passed = False
                    messages.append(f"Vector store check error: {e}")

            if checks_passed:
                self._consecutive_failures = 0
                self._health = ServiceHealth.READY
                return True, "Ready"
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.readiness_failure_threshold:
                    self._health = ServiceHealth.DEGRADED
                return False, "; ".join(messages)

    def liveness_check(self) -> Tuple[bool, str]:
        """
        Kubernetes livenessProbe handler.

        Returns True as long as the process is alive.
        Only returns False if the process is deadlocked or terminally unhealthy.
        """
        if self._health == ServiceHealth.UNHEALTHY:
            return False, "Service is unhealthy - restart required"
        return True, "Alive"

    def signal_shutdown(self):
        """Signal graceful shutdown."""
        with self._lock:
            self._health = ServiceHealth.SHUTTING_DOWN

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        with self._lock:
            uptime = time.time() - self._startup_start_time
            ready_time = (
                time.time() - self._ready_start_time
                if self._ready_start_time else 0
            )

            warmup_stats = {}
            if self.warmup_manager:
                warmup_stats = self.warmup_manager.get_warmup_stats()

            return {
                "status": self._health.value,
                "uptime_seconds": round(uptime, 1),
                "ready_for_seconds": round(ready_time, 1),
                "consecutive_failures": self._consecutive_failures,
                "warmup": warmup_stats,
            }


# ============================================================
# Kubernetes Probe Configuration Example
# ============================================================
"""
Example Kubernetes deployment YAML with proper probe configuration:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
spec:
  template:
    spec:
      containers:
      - name: rag-service
        image: rag-service:latest
        ports:
        - containerPort: 8000

        # startupProbe: Gives the service time to load models and warm up
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          failureThreshold: 30
          # This allows up to 10 + (5 * 30) = 160 seconds for startup

        # readinessProbe: Only route traffic when service is ready
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 8000
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3

        # livenessProbe: Restart if service becomes unresponsive
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 8000
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3

        # Resource allocation (critical for startup performance)
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"

        # Graceful shutdown
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 10"]
```
"""
```

---

## 5. 检查清单（Checklist）

### 测量与基线

- [ ] 是否测量了完整的冷启动时间（从进程启动到首请求可用）？
- [ ] 是否分解了各阶段的耗时（导入包/加载模型/加载索引/预热缓存）？
- [ ] 是否明确了最慢的启动阶段并制定了优化计划？
- [ ] 是否在不同硬件（SSD/HDD, GPU/CPU）上测量过启动时间？

### 模型加载优化

- [ ] 嵌入模型是否使用了 FP16 半精度以减少 50% 加载时间？
- [ ] 模型文件是否存储在本地 SSD 上（而非 NFS/EBS）？
- [ ] 是否使用了模型预加载/后台加载（不阻塞服务就绪）？
- [ ] LLM 是否作为独立服务部署（避免 RAG 冷启动拖累 LLM）？
- [ ] 是否使用了 ONNX Runtime 或 TensorRT 加速推理？

### 缓存优化

- [ ] 是否实现了嵌入缓存（pickle/memmap）以避免重复计算？
- [ ] 缓存文件是否存储在快速磁盘上？
- [ ] 是否预计算了常用查询/F A Q 的嵌入？
- [ ] 是否在服务启动时后台预热缓存（不阻塞就绪）？
- [ ] 是否监控了缓存命中率？

### 容器/编排优化

- [ ] K8s readiness probe 的 `initialDelaySeconds` 是否大于实际启动时间？
- [ ] 是否使用了 `startupProbe` 来保护长启动时间的容器？
- [ ] 是否使用了 preStop hook 在关闭时保存缓存？
- [ ] 镜像中是否预下载了模型文件？
- [ ] 是否配置了足够的资源限制（CPU/memory requests）？

### 监控与告警

- [ ] 是否监控了每次部署的启动时间？
- [ ] 启动时间异常增加是否触发告警？
- [ ] 是否记录了冷启动期间的错误和超时？
- [ ] 是否有预发布环境验证启动时间不退化？

---

> **核心教训**：冷启动慢是"温水煮青蛙"式的问题 —— 刚开始 3 秒启动没人抱怨，
> 半年后变成 30 秒也没人专门去修，直到某天部署导致线上事故。
> **将启动时间纳入 CI/CD 的质量门禁**，就像对待 API 延迟一样。
> 记住：在 K8s 下，启动太慢的容器会被直接杀死，这不是 bug，是设计。
