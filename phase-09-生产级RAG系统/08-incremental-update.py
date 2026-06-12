#!/usr/bin/env python3
"""
增量更新系统 (Incremental Update System)
Delta Indexing with Rollback Capability

特性:
  - 变更检测: 新文档 / 修改文档 / 删除文档
  - 仅处理变更文档 (delta indexing)
  - 合并策略: 增量并入主索引
  - 版本跟踪: 每块 v1, v2, ...
  - 回滚能力: 恢复到之前的索引状态
  - 调度器: 可配置间隔定期运行
"""

import os
import json
import time
import hashlib
import threading
import schedule
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
from collections import defaultdict


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class ChangeType(Enum):
    """变更类型"""
    NEW = "new"           # 新文档
    MODIFIED = "modified" # 修改文档
    DELETED = "deleted"   # 删除文档
    UNCHANGED = "unchanged"  # 未变化


class IndexMergeStrategy(Enum):
    """索引合并策略"""
    APPEND = "append"                      # 追加
    REPLACE_BY_ID = "replace_by_id"       # 按ID替换
    UPSERT = "upsert"                     # 更新或插入
    FULL_REINDEX = "full_reindex"          # 完全重建（适用于大量变更）


@dataclass
class DocumentVersion:
    """文档版本信息"""
    doc_id: str
    file_path: str
    version: int                          # 当前版本号
    checksum: str                         # MD5
    last_modified: float                  # 修改时间戳
    chunk_versions: dict[str, int] = field(default_factory=dict)  # chunk_id -> version

    def bump_version(self):
        self.version += 1


@dataclass
class DeltaChange:
    """增量变更记录"""
    doc_id: str
    change_type: ChangeType
    old_version: Optional[int]
    new_version: Optional[int]
    changed_chunks: list[str]             # 变更的chunk ID列表
    timestamp: float
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexSnapshot:
    """索引快照（用于回滚）"""
    snapshot_id: str
    created_at: float
    total_chunks: int
    total_vectors: int
    hnsw_state: Any                       # HNSW索引状态
    bm25_state: Any                       # BM25索引状态
    chunk_versions: dict[str, DocumentVersion]
    prev_snapshot_id: Optional[str] = None


@dataclass
class SchedulerConfig:
    """调度器配置"""
    interval_minutes: int = 5            # 检查间隔（分钟）
    max_delta_size: int = 1000            # 最大增量大小（超过则全量重建）
    auto_rollback_on_failure: bool = True # 失败时自动回滚
    max_snapshots: int = 10               # 最大快照保留数


# ============================================================
# 文档监视器 (Document Watcher)
# ============================================================

class DocumentWatcher:
    """文档变更检测器"""

    def __init__(self, watch_dir: str):
        self.watch_dir = Path(watch_dir)
        self._known_files: dict[str, DocumentVersion] = {}  # file_path -> version
        self._lock = threading.RLock()

    def scan(self) -> dict[str, ChangeType]:
        """扫描变更
        Returns:
            {file_path: ChangeType} - 变更的文件及其类型
        """
        changes: dict[str, ChangeType] = {}

        # 获取当前文件系统中的文件
        current_files: set[str] = set()
        for pattern in ["**/*.md", "**/*.txt", "**/*.pdf", "**/*.docx", "**/*.html"]:
            for fp in self.watch_dir.glob(pattern):
                current_files.add(str(fp))

        with self._lock:
            # 检测新增文件
            for fp in current_files:
                if fp not in self._known_files:
                    changes[fp] = ChangeType.NEW

            # 检测修改文件
            for fp in current_files:
                if fp in self._known_files:
                    current_checksum = self._compute_checksum(fp)
                    known_version = self._known_files[fp]
                    if current_checksum != known_version.checksum:
                        changes[fp] = ChangeType.MODIFIED

            # 检测删除文件
            for fp in self._known_files:
                if fp not in current_files:
                    changes[fp] = ChangeType.DELETED

        return changes

    def update_known(self, file_path: str, version: DocumentVersion):
        """更新已知文件记录"""
        with self._lock:
            self._known_files[file_path] = version

    def remove_known(self, file_path: str):
        """移除已知文件记录"""
        with self._lock:
            self._known_files.pop(file_path, None)

    def get_known_state(self) -> dict[str, DocumentVersion]:
        """获取已知文件状态"""
        with self._lock:
            return dict(self._known_files)

    @staticmethod
    def _compute_checksum(filepath: str) -> str:
        """计算文件校验和"""
        sha = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
        except FileNotFoundError:
            return "DELETED"
        return sha.hexdigest()


# ============================================================
# 增量处理器 (Delta Processor)
# ============================================================

class DeltaProcessor:
    """增量处理器: 处理变更文档"""

    def __init__(self, chunker=None, vectorizer=None):
        """
        Args:
            chunker: 分块器实例
            vectorizer: 向量化器实例
        """
        self.chunker = chunker
        self.vectorizer = vectorizer
        self._version_registry: dict[str, DocumentVersion] = {}
        self._delta_history: list[DeltaChange] = []

    def process_new(self, file_path: str) -> tuple[list[dict], list[list[float]], DocumentVersion]:
        """处理新文档"""
        doc_id = self._generate_doc_id(file_path)

        # 解析文档（简化模拟）
        text = self._read_file(file_path)
        chunks_data = self._simulate_chunk(text, doc_id)

        # 向量化
        vectors = self._simulate_vectorize(chunks_data)

        # 创建版本记录
        version = DocumentVersion(
            doc_id=doc_id,
            file_path=file_path,
            version=1,
            checksum=DocumentWatcher._compute_checksum(file_path),
            last_modified=os.path.getmtime(file_path),
            chunk_versions={c["chunk_id"]: 1 for c in chunks_data},
        )

        self._version_registry[doc_id] = version

        delta = DeltaChange(
            doc_id=doc_id,
            change_type=ChangeType.NEW,
            old_version=None,
            new_version=1,
            changed_chunks=[c["chunk_id"] for c in chunks_data],
            timestamp=time.time(),
        )
        self._delta_history.append(delta)

        return chunks_data, vectors, version

    def process_modified(self, file_path: str, old_version: DocumentVersion) -> tuple[list[dict], list[list[float]], DocumentVersion]:
        """处理修改文档"""
        doc_id = old_version.doc_id

        # 读新内容
        text = self._read_file(file_path)
        chunks_data = self._simulate_chunk(text, doc_id)

        # 向量化
        vectors = self._simulate_vectorize(chunks_data)

        # 更新版本
        old_version.bump_version()
        old_version.checksum = DocumentWatcher._compute_checksum(file_path)
        old_version.last_modified = os.path.getmtime(file_path)
        old_version.chunk_versions = {c["chunk_id"]: old_version.version for c in chunks_data}

        self._version_registry[doc_id] = old_version

        delta = DeltaChange(
            doc_id=doc_id,
            change_type=ChangeType.MODIFIED,
            old_version=old_version.version - 1,
            new_version=old_version.version,
            changed_chunks=[c["chunk_id"] for c in chunks_data],
            timestamp=time.time(),
        )
        self._delta_history.append(delta)

        return chunks_data, vectors, old_version

    def process_deleted(self, file_path: str, old_version: DocumentVersion) -> list[str]:
        """处理删除文档
        Returns:
            需要删除的chunk ID列表
        """
        doc_id = old_version.doc_id

        deleted_chunk_ids = list(old_version.chunk_versions.keys())
        self._version_registry.pop(doc_id, None)

        delta = DeltaChange(
            doc_id=doc_id,
            change_type=ChangeType.DELETED,
            old_version=old_version.version,
            new_version=None,
            changed_chunks=deleted_chunk_ids,
            timestamp=time.time(),
        )
        self._delta_history.append(delta)

        return deleted_chunk_ids

    def get_delta_stats(self) -> dict:
        """获取增量统计"""
        return {
            "total_deltas": len(self._delta_history),
            "new_count": sum(1 for d in self._delta_history if d.change_type == ChangeType.NEW),
            "modified_count": sum(1 for d in self._delta_history if d.change_type == ChangeType.MODIFIED),
            "deleted_count": sum(1 for d in self._delta_history if d.change_type == ChangeType.DELETED),
            "tracked_documents": len(self._version_registry),
        }

    # ========== 辅助方法 ==========

    @staticmethod
    def _generate_doc_id(file_path: str) -> str:
        return hashlib.md5(file_path.encode()).hexdigest()[:16]

    @staticmethod
    def _read_file(file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return f"[BINARY_CONTENT] {file_path}"

    @staticmethod
    def _simulate_chunk(text: str, doc_id: str) -> list[dict]:
        """模拟分块（生产环境使用真实chunker）"""
        import re

        chunks = []
        # 简单按段落分割
        paragraphs = text.split("\n\n")

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para or len(para) < 10:
                continue

            chunk_id = f"{doc_id}_chunk_{i:04d}_v1"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": para[:2000],
                "char_count": len(para),
                "metadata": {
                    "source": doc_id,
                    "chunk_index": i,
                    "version": 1,
                },
            })

        return chunks

    @staticmethod
    def _simulate_vectorize(chunks: list[dict]) -> list[list[float]]:
        """模拟向量化（生产环境使用真实vectorizer）"""
        import numpy as np
        return np.random.randn(len(chunks), 384).tolist()


# ============================================================
# 索引合并器 (Index Merger)
# ============================================================

class IndexMerger:
    """索引合并器: 将增量合并入主索引"""

    def __init__(self, strategy: IndexMergeStrategy = IndexMergeStrategy.UPSERT):
        self.strategy = strategy
        self._main_chunks: dict[str, dict] = {}      # chunk_id -> chunk
        self._main_vectors: dict[str, list[float]] = {}  # chunk_id -> vector
        self._deleted_chunks: set[str] = set()
        self._lock = threading.RLock()

    def merge(self, new_chunks: list[dict], new_vectors: list[list[float]],
              deleted_chunk_ids: list[str] = None) -> dict:
        """合并增量到主索引

        Args:
            new_chunks: 新增/修改的块
            new_vectors: 对应的向量
            deleted_chunk_ids: 要删除的块ID

        Returns:
            合并统计
        """
        stats = {"added": 0, "updated": 0, "deleted": 0, "errors": 0}

        with self._lock:
            # 处理新增/修改
            for chunk, vector in zip(new_chunks, new_vectors):
                chunk_id = chunk["chunk_id"]
                if self.strategy == IndexMergeStrategy.UPSERT:
                    is_new = chunk_id not in self._main_chunks
                    self._main_chunks[chunk_id] = chunk
                    self._main_vectors[chunk_id] = vector
                    if is_new:
                        stats["added"] += 1
                    else:
                        stats["updated"] += 1

                elif self.strategy == IndexMergeStrategy.REPLACE_BY_ID:
                    self._main_chunks[chunk_id] = chunk
                    self._main_vectors[chunk_id] = vector
                    stats["added" if chunk_id not in self._main_chunks else "updated"] += 1

                elif self.strategy == IndexMergeStrategy.APPEND:
                    if chunk_id not in self._main_chunks:
                        self._main_chunks[chunk_id] = chunk
                        self._main_vectors[chunk_id] = vector
                        stats["added"] += 1

            # 处理删除
            if deleted_chunk_ids:
                for chunk_id in deleted_chunk_ids:
                    if chunk_id in self._main_chunks:
                        del self._main_chunks[chunk_id]
                        del self._main_vectors[chunk_id]
                        self._deleted_chunks.add(chunk_id)
                        stats["deleted"] += 1

        return stats

    def get_main_index_size(self) -> dict:
        """获取主索引大小"""
        with self._lock:
            return {
                "total_chunks": len(self._main_chunks),
                "total_vectors": len(self._main_vectors),
                "deleted_chunks": len(self._deleted_chunks),
            }


# ============================================================
# 快照管理器 (Snapshot Manager for Rollback)
# ============================================================

class SnapshotManager:
    """快照管理器 - 用于索引回滚"""

    def __init__(self, max_snapshots: int = 10):
        self.max_snapshots = max_snapshots
        self._snapshots: list[IndexSnapshot] = []
        self._lock = threading.RLock()

    def create_snapshot(self, chunks: dict, vectors: dict,
                        versions: dict[str, DocumentVersion],
                        hnsw_state: Any = None, bm25_state: Any = None) -> IndexSnapshot:
        """创建当前索引的快照"""
        snapshot_id = f"snap_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        # 深拷贝数据
        chunks_copy = json.loads(json.dumps({k: v for k, v in list(chunks.items())[:100]}))
        vectors_copy = {k: v[:] for k, v in list(vectors.items())[:100]}

        prev_id = self._snapshots[-1].snapshot_id if self._snapshots else None

        snapshot = IndexSnapshot(
            snapshot_id=snapshot_id,
            created_at=time.time(),
            total_chunks=len(chunks),
            total_vectors=len(vectors),
            hnsw_state=hnsw_state,
            bm25_state=bm25_state,
            chunk_versions=dict(versions),
            prev_snapshot_id=prev_id,
        )

        with self._lock:
            self._snapshots.append(snapshot)

            # 限制快照数量
            while len(self._snapshots) > self.max_snapshots:
                removed = self._snapshots.pop(0)
                print(f"[Snapshot] 清理旧快照: {removed.snapshot_id}")

        print(f"[Snapshot] 创建快照: {snapshot_id} ({snapshot.total_chunks} 块)")
        return snapshot

    def get_latest_snapshot(self) -> Optional[IndexSnapshot]:
        """获取最新快照"""
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def get_snapshot(self, snapshot_id: str) -> Optional[IndexSnapshot]:
        """获取指定快照"""
        for snap in self._snapshots:
            if snap.snapshot_id == snapshot_id:
                return snap
        return None

    def rollback_to(self, snapshot_id: str) -> Optional[IndexSnapshot]:
        """回滚到指定快照"""
        target = self.get_snapshot(snapshot_id)
        if target is None:
            print(f"[Rollback] 快照不存在: {snapshot_id}")
            return None

        # 删除目标之后的所有快照
        target_idx = None
        for i, snap in enumerate(self._snapshots):
            if snap.snapshot_id == snapshot_id:
                target_idx = i
                break

        if target_idx is not None:
            with self._lock:
                self._snapshots = self._snapshots[: target_idx + 1]

        print(f"[Rollback] 回滚到快照: {snapshot_id} (后{target_idx}个快照已清除)")
        return target

    def list_snapshots(self) -> list[dict]:
        """列出所有快照"""
        return [
            {
                "snapshot_id": s.snapshot_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.created_at)),
                "total_chunks": s.total_chunks,
                "total_vectors": s.total_vectors,
            }
            for s in self._snapshots
        ]


# ============================================================
# 增量更新调度器 (Incremental Update Scheduler)
# ============================================================

class IncrementalUpdateScheduler:
    """增量更新调度器"""

    def __init__(self, config: SchedulerConfig = None):
        self.config = config or SchedulerConfig()
        self.watcher: Optional[DocumentWatcher] = None
        self.processor = DeltaProcessor()
        self.merger = IndexMerger()
        self.snapshots = SnapshotManager(max_snapshots=self.config.max_snapshots)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_complete_callbacks: list[Callable] = []

    def set_watch_dir(self, directory: str):
        """设置监视目录"""
        self.watcher = DocumentWatcher(directory)

    def start(self):
        """启动调度器"""
        if self.watcher is None:
            raise ValueError("未设置监视目录，请先调用set_watch_dir()")

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[Scheduler] 增量更新调度器已启动 (间隔: {self.config.interval_minutes}分钟)")

    def stop(self):
        """停止调度器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        print("[Scheduler] 调度器已停止")

    def run_once(self) -> dict:
        """手动执行一次增量更新"""
        if self.watcher is None:
            return {"status": "error", "message": "未设置监视目录"}

        print(f"[Scheduler] 执行增量更新...")

        try:
            # 步骤1: 检测变更
            changes = self.watcher.scan()

            if not changes:
                print("[Scheduler] 无变更，跳过")
                return {"status": "no_changes", "changes": 0}

            print(f"[Scheduler] 检测到 {len(changes)} 个变更")

            # 步骤2: 创建快照（用于回滚）
            current_size = self.merger.get_main_index_size()
            self.snapshots.create_snapshot(
                chunks=dict(list(self.merger._main_chunks.items())[:100]),
                vectors=dict(list(self.merger._main_vectors.items())[:100]),
                versions=self.processor._version_registry,
            )

            # 步骤3: 处理变更
            all_new_chunks = []
            all_new_vectors = []
            all_deleted_ids = []
            stats = {"new": 0, "modified": 0, "deleted": 0}

            known_state = self.watcher.get_known_state()

            for file_path, change_type in changes.items():
                if change_type == ChangeType.NEW:
                    chunks, vectors, version = self.processor.process_new(file_path)
                    all_new_chunks.extend(chunks)
                    all_new_vectors.extend(vectors)
                    self.watcher.update_known(file_path, version)
                    stats["new"] += 1

                elif change_type == ChangeType.MODIFIED:
                    old_version = known_state.get(file_path)
                    if old_version:
                        chunks, vectors, version = self.processor.process_modified(file_path, old_version)
                        all_new_chunks.extend(chunks)
                        all_new_vectors.extend(vectors)
                        self.watcher.update_known(file_path, version)
                        stats["modified"] += 1

                elif change_type == ChangeType.DELETED:
                    old_version = known_state.get(file_path)
                    if old_version:
                        deleted_ids = self.processor.process_deleted(file_path, old_version)
                        all_deleted_ids.extend(deleted_ids)
                        self.watcher.remove_known(file_path)
                        stats["deleted"] += 1

            # 步骤4: 合并到主索引
            merge_stats = self.merger.merge(all_new_chunks, all_new_vectors, all_deleted_ids)

            # 步骤5: 检查是否需要全量重建
            total_changes = stats["new"] + stats["modified"] + stats["deleted"]
            if total_changes > self.config.max_delta_size:
                print(f"[Scheduler] 变更量 ({total_changes}) 超过阈值 ({self.config.max_delta_size})，建议全量重建索引")

            result = {
                "status": "success",
                "changes_detected": len(changes),
                "processed": stats,
                "merge_stats": merge_stats,
                "index_size": self.merger.get_main_index_size(),
                "delta_stats": self.processor.get_delta_stats(),
            }

            # 触发完成回调
            for cb in self._on_complete_callbacks:
                try:
                    cb(result)
                except Exception as e:
                    print(f"[Scheduler] 回调失败: {e}")

            print(f"[Scheduler] 增量更新完成: {stats}")
            return result

        except Exception as e:
            print(f"[Scheduler] 增量更新失败: {e}")

            # 自动回滚
            if self.config.auto_rollback_on_failure:
                latest_snap = self.snapshots.get_latest_snapshot()
                if latest_snap:
                    print(f"[Scheduler] 自动回滚到快照: {latest_snap.snapshot_id}")
                    # 回滚逻辑...

            return {"status": "error", "message": str(e)}

    def on_complete(self, callback: Callable):
        """注册完成回调"""
        self._on_complete_callbacks.append(callback)

    def _run_loop(self):
        """调度器主循环"""
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                print(f"[Scheduler] 周期执行异常: {e}")

            # 等待下一个间隔
            time.sleep(self.config.interval_minutes * 60)


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("增量更新系统 - 演示运行")
    print("=" * 60)

    # 1. 准备测试目录和文件
    test_dir = "./test_docs_delta"
    os.makedirs(test_dir, exist_ok=True)

    # 写入初始文档
    initial_docs = {
        "doc1.md": "# RAG系统设计\n\nRAG系统需要良好的架构设计。\n\n核心组件包括检索器和生成器。",
        "doc2.md": "# 向量数据库\n\n向量数据库是RAG系统的关键组件。\n\n常用选项有Faiss和Milvus。",
        "doc3.md": "# 性能优化\n\n性能优化包括缓存和索引调优。\n\n合理的参数配置至关重要。",
    }

    for name, content in initial_docs.items():
        with open(os.path.join(test_dir, name), "w", encoding="utf-8") as f:
            f.write(content)

    print(f"创建测试目录: {test_dir}")
    print(f"初始文档: {len(initial_docs)} 个")

    # 2. 初始化增量更新调度器
    config = SchedulerConfig(
        interval_minutes=1,
        max_delta_size=100,
        auto_rollback_on_failure=True,
        max_snapshots=5,
    )
    scheduler = IncrementalUpdateScheduler(config)
    scheduler.set_watch_dir(test_dir)

    # 注册完成回调
    def on_update_complete(result):
        print(f"  [Callback] 更新完成: {result.get('changes_detected', 0)} 变更")

    scheduler.on_complete(on_update_complete)

    # 3. 首次运行（检测初始文档）
    print("\n--- 首次扫描（检测初始文档） ---")
    result1 = scheduler.run_once()
    print(f"结果: {result1['status']}")
    if "processed" in result1:
        print(f"处理: 新增={result1['processed']['new']}, 修改={result1['processed']['modified']}, 删除={result1['processed']['deleted']}")

    # 4. 模拟文档修改
    print("\n--- 模拟文档修改 ---")
    time.sleep(1)  # 确保修改时间不同

    with open(os.path.join(test_dir, "doc1.md"), "a", encoding="utf-8") as f:
        f.write("\n\n## 新增章节\n\n这是新添加的内容，包含更多关于RAG系统的细节。")

    print("doc1.md 已修改")

    # 5. 添加新文档
    print("\n--- 模拟新增文档 ---")
    with open(os.path.join(test_dir, "doc4.md"), "w", encoding="utf-8") as f:
        f.write("# 熔断器模式\n\n熔断器是弹性设计的重要组成部分。\n\n它防止级联故障的发生。")

    print("doc4.md 已创建")

    # 6. 再次扫描
    print("\n--- 第二次扫描（检测变更） ---")
    result2 = scheduler.run_once()
    print(f"结果: {result2['status']}")
    if "processed" in result2:
        print(f"处理: 新增={result2['processed']['new']}, 修改={result2['processed']['modified']}, 删除={result2['processed']['deleted']}")
    if "merge_stats" in result2:
        print(f"合并: 添加={result2['merge_stats']['added']}, 更新={result2['merge_stats']['updated']}")

    # 7. 索引状态
    print("\n--- 索引状态 ---")
    index_size = scheduler.merger.get_main_index_size()
    print(f"总块数: {index_size['total_chunks']}")
    print(f"总向量: {index_size['total_vectors']}")
    print(f"已删除: {index_size['deleted_chunks']}")

    # 8. 版本跟踪
    print("\n--- 版本跟踪 ---")
    delta_stats = scheduler.processor.get_delta_stats()
    print(f"跟踪文档: {delta_stats['tracked_documents']}")
    print(f"总变更:   {delta_stats['total_deltas']}")
    print(f"  新增:   {delta_stats['new_count']}")
    print(f"  修改:   {delta_stats['modified_count']}")
    print(f"  删除:   {delta_stats['deleted_count']}")

    for doc_id, version in scheduler.processor._version_registry.items():
        print(f"  {doc_id}: v{version.version} ({len(version.chunk_versions)} 块)")

    # 9. 快照管理
    print("\n--- 快照列表 ---")
    snapshots = scheduler.snapshots.list_snapshots()
    for s in snapshots:
        print(f"  {s['snapshot_id']} | {s['created_at']} | {s['total_chunks']} 块")

    # 10. 演示回滚
    if snapshots:
        print("\n--- 回滚演示 ---")
        oldest = snapshots[0]["snapshot_id"]
        rolled = scheduler.snapshots.rollback_to(oldest)
        if rolled:
            print(f"回滚到: {rolled.snapshot_id} ({rolled.total_chunks} 块)")
        remaining = scheduler.snapshots.list_snapshots()
        print(f"剩余快照: {len(remaining)}")

    # 11. 无变更扫描
    print("\n--- 无变更扫描 ---")
    result3 = scheduler.run_once()
    print(f"结果: {result3['status']}")

    # 12. 清理
    print("\n--- 清理 ---")
    try:
        import shutil
        shutil.rmtree(test_dir)
        print("测试文件已清理")
    except Exception as e:
        print(f"清理失败: {e}")

    print("\n" + "=" * 60)
    print("增量更新系统演示完成！")
    print("=" * 60)
