#!/usr/bin/env python3
"""
灾难恢复系统 (Disaster Recovery System)
Production Disaster Recovery for Vector Index

特性:
  - 向量索引备份: 导出到Parquet/JSONL
  - 增量备份: 自上次全量备份以来的增量
  - 跨区域复制: 多区域备份存储
  - 恢复过程: 带验证的完整恢复
  - RTO/RPO 文档化
  - DR演练自动化
"""

import os
import json
import time
import shutil
import hashlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import tempfile

import numpy as np


# ============================================================
# 数据类型定义 (Data Types)
# ============================================================

class BackupType(Enum):
    """备份类型"""
    FULL = "full"               # 全量备份
    INCREMENTAL = "incremental" # 增量备份


class BackupStatus(Enum):
    """备份状态"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


class Region(Enum):
    """区域"""
    PRIMARY = "us-east-1"
    SECONDARY = "us-west-2"
    TERTIARY = "eu-west-1"
    ASIA = "ap-southeast-1"


@dataclass
class BackupManifest:
    """备份清单"""
    backup_id: str
    backup_type: BackupType
    created_at: str                    # ISO 格式时间戳
    total_chunks: int
    total_vectors: int
    vector_dimension: int
    index_type: str                    # "hnsw" / "bm25"
    file_paths: list[str]              # 备份文件路径列表
    checksums: dict[str, str]          # 文件路径 -> SHA256
    parent_backup_id: Optional[str] = None  # 增量备份的父备份ID
    status: BackupStatus = BackupStatus.RUNNING
    metadata: dict = field(default_factory=dict)


@dataclass
class RestoreResult:
    """恢复结果"""
    success: bool
    backup_id: str
    chunks_restored: int
    vectors_restored: int
    verification_passed: bool
    errors: list[str]
    duration_seconds: float
    checksum_match: bool

    @property
    def rto_seconds(self) -> float:
        """RTO: 恢复时间目标 (Recovery Time Objective)"""
        return self.duration_seconds


@dataclass
class DRPlan:
    """灾难恢复计划"""
    rto_target: float = 3600.0      # 目标RTO: 1小时
    rpo_target: float = 300.0       # 目标RPO: 5分钟
    backup_interval: float = 3600.0  # 备份间隔: 1小时
    full_backup_interval: float = 86400.0  # 全量备份间隔: 24小时
    retention_days: int = 30        # 保留天数
    replication_regions: list[Region] = field(default_factory=lambda: [
        Region.SECONDARY,
    ])
    max_incremental_chain: int = 24  # 最大增量链长度（24小时后必须全量）


# ============================================================
# 备份格式实现
# ============================================================

class ParquetBackupFormat:
    """Parquet备份格式"""

    @staticmethod
    def export_chunks(chunks: list[dict], filepath: str) -> str:
        """导出块数据到Parquet"""
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq

            # 展平元数据
            rows = []
            for c in chunks:
                row = {
                    "chunk_id": c.get("chunk_id", ""),
                    "doc_id": c.get("doc_id", ""),
                    "text": c.get("text", ""),
                    "char_count": len(c.get("text", "")),
                }
                # 展平metadata中的字段
                meta = c.get("metadata", {})
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        row[f"meta_{k}"] = v
                    elif isinstance(v, list):
                        row[f"meta_{k}"] = json.dumps(v, ensure_ascii=False)
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_parquet(filepath, index=False)
            return filepath
        except ImportError:
            return JSONLBackupFormat.export_chunks(chunks, filepath.replace(".parquet", ".jsonl"))

    @staticmethod
    def export_vectors(vectors: list[list[float]], ids: list[str], filepath: str) -> str:
        """导出向量到Parquet"""
        try:
            import pandas as pd
            import pyarrow.parquet as pq

            rows = []
            for i, (vid, vec) in enumerate(zip(ids, vectors)):
                rows.append({
                    "vector_id": vid,
                    "dimension": len(vec),
                    "values": vec,  # pyarrow支持list类型
                })

            df = pd.DataFrame(rows)
            df.to_parquet(filepath, index=False)
            return filepath
        except ImportError:
            return JSONLBackupFormat.export_vectors(vectors, ids, filepath.replace(".parquet", ".jsonl"))

    @staticmethod
    def import_chunks(filepath: str) -> list[dict]:
        """从Parquet导入块"""
        try:
            import pandas as pd

            df = pd.read_parquet(filepath)
            chunks = []
            for _, row in df.iterrows():
                chunk = {
                    "chunk_id": row.get("chunk_id", ""),
                    "doc_id": row.get("doc_id", ""),
                    "text": row.get("text", ""),
                    "metadata": {},
                }
                for col in df.columns:
                    if col.startswith("meta_"):
                        key = col[5:]
                        val = row[col]
                        try:
                            chunk["metadata"][key] = json.loads(val)
                        except (TypeError, json.JSONDecodeError):
                            chunk["metadata"][key] = val
                chunks.append(chunk)
            return chunks
        except ImportError:
            return JSONLBackupFormat.import_chunks(filepath.replace(".parquet", ".jsonl"))


class JSONLBackupFormat:
    """JSONL备份格式（降级方案）"""

    @staticmethod
    def export_chunks(chunks: list[dict], filepath: str) -> str:
        with open(filepath, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        return filepath

    @staticmethod
    def export_vectors(vectors: list[list[float]], ids: list[str], filepath: str) -> str:
        with open(filepath, "w", encoding="utf-8") as f:
            for vid, vec in zip(ids, vectors):
                record = {"vector_id": vid, "dimension": len(vec), "values": vec}
                f.write(json.dumps(record) + "\n")
        return filepath

    @staticmethod
    def import_chunks(filepath: str) -> list[dict]:
        chunks = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks


# ============================================================
# 备份管理器
# ============================================================

class BackupManager:
    """备份管理器"""

    def __init__(self, backup_dir: str = "./backups", plan: Optional[DRPlan] = None):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.plan = plan or DRPlan()
        self._lock = threading.RLock()

        # 备份历史
        self._manifests: list[BackupManifest] = []
        self._load_manifest_index()

    # ========== 全量备份 ==========

    def full_backup(self, chunks: list[dict], vectors: list[list[float]],
                    vector_ids: list[str], index_type: str = "hnsw",
                    dimension: int = 384) -> BackupManifest:
        """执行全量备份

        Args:
            chunks: 块数据列表
            vectors: 向量列表
            vector_ids: 向量ID列表
            index_type: 索引类型
            dimension: 向量维度

        Returns:
            BackupManifest: 备份清单
        """
        backup_id = self._generate_backup_id()
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)

        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.FULL,
            created_at=datetime.now().isoformat(),
            total_chunks=len(chunks),
            total_vectors=len(vectors),
            vector_dimension=dimension,
            index_type=index_type,
            file_paths=[],
            checksums={},
            status=BackupStatus.RUNNING,
        )

        print(f"[Backup] 开始全量备份: {backup_id}")
        print(f"[Backup] 块数: {len(chunks)}, 向量数: {len(vectors)}")

        try:
            # 导出块数据
            chunks_path = backup_path / "chunks.parquet"
            ParquetBackupFormat.export_chunks(chunks, str(chunks_path))
            manifest.file_paths.append(str(chunks_path))
            manifest.checksums[str(chunks_path)] = self._compute_checksum(str(chunks_path))

            # 导出向量数据
            vectors_path = backup_path / "vectors.parquet"
            ParquetBackupFormat.export_vectors(vectors, vector_ids, str(vectors_path))
            manifest.file_paths.append(str(vectors_path))
            manifest.checksums[str(vectors_path)] = self._compute_checksum(str(vectors_path))

            # 写入清单
            manifest_path = backup_path / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest_to_dict(manifest), f, indent=2, ensure_ascii=False)

            manifest.status = BackupStatus.COMPLETED
            self._manifests.append(manifest)
            self._save_manifest_index()

            print(f"[Backup] 全量备份完成: {backup_id}")
            print(f"[Backup] 文件: {manifest.file_paths}")

        except Exception as e:
            manifest.status = BackupStatus.FAILED
            manifest.metadata["error"] = str(e)
            print(f"[Backup] 全量备份失败: {e}")
            raise

        return manifest

    # ========== 增量备份 ==========

    def incremental_backup(self, delta_chunks: list[dict],
                           delta_vectors: list[list[float]],
                           delta_vector_ids: list[str],
                           parent_backup_id: str) -> BackupManifest:
        """执行增量备份

        Args:
            delta_chunks: 自上次备份以来变更的块
            delta_vectors: 变更的向量
            delta_vector_ids: 变更向量ID
            parent_backup_id: 父备份ID
        """
        backup_id = self._generate_backup_id()
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)

        parent = self._find_manifest(parent_backup_id)
        if parent is None:
            raise ValueError(f"父备份不存在: {parent_backup_id}")

        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.INCREMENTAL,
            created_at=datetime.now().isoformat(),
            total_chunks=len(delta_chunks),
            total_vectors=len(delta_vectors),
            vector_dimension=parent.vector_dimension,
            index_type=parent.index_type,
            file_paths=[],
            checksums={},
            parent_backup_id=parent_backup_id,
            status=BackupStatus.RUNNING,
        )

        print(f"[Backup] 开始增量备份: {backup_id} (父: {parent_backup_id})")

        try:
            # 导出增量块
            chunks_path = backup_path / "delta_chunks.jsonl"
            JSONLBackupFormat.export_chunks(delta_chunks, str(chunks_path))
            manifest.file_paths.append(str(chunks_path))
            manifest.checksums[str(chunks_path)] = self._compute_checksum(str(chunks_path))

            # 导出增量向量
            vectors_path = backup_path / "delta_vectors.jsonl"
            JSONLBackupFormat.export_vectors(delta_vectors, delta_vector_ids, str(vectors_path))
            manifest.file_paths.append(str(vectors_path))
            manifest.checksums[str(vectors_path)] = self._compute_checksum(str(vectors_path))

            # 写入manifest
            manifest_path = backup_path / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest_to_dict(manifest), f, indent=2, ensure_ascii=False)

            manifest.status = BackupStatus.COMPLETED
            self._manifests.append(manifest)
            self._save_manifest_index()

            print(f"[Backup] 增量备份完成: {backup_id}")

        except Exception as e:
            manifest.status = BackupStatus.FAILED
            manifest.metadata["error"] = str(e)
            print(f"[Backup] 增量备份失败: {e}")
            raise

        return manifest

    # ========== 恢复过程 ==========

    def restore(self, backup_id: str, target_dir: Optional[str] = None,
                verify: bool = True) -> RestoreResult:
        """从备份恢复

        Args:
            backup_id: 备份ID
            target_dir: 恢复目标目录
            verify: 是否验证恢复结果

        Returns:
            RestoreResult: 恢复结果
        """
        t_start = time.time()
        errors = []

        manifest = self._find_manifest(backup_id)
        if manifest is None:
            return RestoreResult(
                success=False, backup_id=backup_id,
                chunks_restored=0, vectors_restored=0,
                verification_passed=False,
                errors=[f"备份不存在: {backup_id}"],
                duration_seconds=0, checksum_match=False,
            )

        print(f"[Restore] 开始恢复: {backup_id}")

        # 如果是指向增量备份，需要还原父备份链
        chain = self._get_restore_chain(manifest)
        print(f"[Restore] 恢复链长度: {len(chain)}")

        all_chunks = []
        all_vectors = []
        all_vector_ids = []
        checksum_ok = True

        for m in reversed(chain):  # 从最老的全量备份开始
            print(f"[Restore] 恢复: {m.backup_id} ({m.backup_type.value})")
            for fp in m.file_paths:
                if not os.path.exists(fp):
                    errors.append(f"文件缺失: {fp}")
                    continue

                # 验证校验和
                if verify:
                    expected = m.checksums.get(fp)
                    if expected:
                        actual = self._compute_checksum(fp)
                        if actual != expected:
                            errors.append(f"校验和不匹配: {fp}")
                            checksum_ok = False

                # 导入数据
                if "chunks" in fp or "delta_chunks" in fp:
                    try:
                        chunks = JSONLBackupFormat.import_chunks(fp)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        errors.append(f"导入块失败 {fp}: {e}")
                elif "vectors" in fp or "delta_vectors" in fp:
                    try:
                        with open(fp, "r") as f:
                            for line in f:
                                record = json.loads(line)
                                all_vector_ids.append(record["vector_id"])
                                all_vectors.append(record["values"])
                    except Exception as e:
                        errors.append(f"导入向量失败 {fp}: {e}")

        duration = time.time() - t_start

        result = RestoreResult(
            success=len(errors) == 0,
            backup_id=backup_id,
            chunks_restored=len(all_chunks),
            vectors_restored=len(all_vectors),
            verification_passed=checksum_ok and len(errors) == 0,
            errors=errors,
            duration_seconds=duration,
            checksum_match=checksum_ok,
        )

        print(f"[Restore] 恢复完成: {result.chunks_restored} 块, "
              f"{result.vectors_restored} 向量, "
              f"耗时 {duration:.1f}s, "
              f"验证: {'通过' if result.verification_passed else '失败'}")

        return result

    # ========== 跨区域复制 ==========

    def replicate(self, backup_id: str, target_region: Region) -> bool:
        """将备份复制到另一个区域

        Args:
            backup_id: 备份ID
            target_region: 目标区域
        """
        manifest = self._find_manifest(backup_id)
        if manifest is None:
            print(f"[Replicate] 备份不存在: {backup_id}")
            return False

        region_dir = self.backup_dir / target_region.value
        region_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Replicate] 复制 {backup_id} → {target_region.value}")

        try:
            source_dir = self.backup_dir / backup_id
            target_subdir = region_dir / backup_id

            if source_dir.exists():
                if target_subdir.exists():
                    shutil.rmtree(str(target_subdir))
                shutil.copytree(str(source_dir), str(target_subdir))
                print(f"[Replicate] 复制完成: {target_subdir}")
                return True
        except Exception as e:
            print(f"[Replicate] 复制失败: {e}")
            return False

        return False

    # ========== DR 演练 ==========

    def dr_drill(self, backup_id: str, restore_to_temp: bool = True) -> RestoreResult:
        """DR演练: 执行测试恢复

        Args:
            backup_id: 要测试恢复的备份ID
            restore_to_temp: 恢复到临时目录
        """
        print(f"[DR Drill] 开始灾难恢复演练: {backup_id}")
        print(f"[DR Drill] RTO目标: {self.plan.rto_target:.0f}s, RPO目标: {self.plan.rpo_target:.0f}s")

        target = tempfile.mkdtemp(prefix="dr_drill_") if restore_to_temp else None
        result = self.restore(backup_id, target_dir=target, verify=True)

        # RTO检查
        rto_met = result.rto_seconds <= self.plan.rto_target
        print(f"[DR Drill] RTO: {result.rto_seconds:.1f}s "
              f"({'✅' if rto_met else '❌'} 目标: {self.plan.rto_target:.0f}s)")

        # 清理临时文件
        if restore_to_temp and target:
            try:
                shutil.rmtree(target)
            except Exception:
                pass

        return result

    # ========== 备份清理 ==========

    def cleanup_old_backups(self):
        """清理超过保留期的备份"""
        now = time.time()
        cutoff = now - self.plan.retention_days * 86400

        for manifest in list(self._manifests):
            created = datetime.fromisoformat(manifest.created_at).timestamp()
            if created < cutoff and manifest.backup_type == BackupType.FULL:
                # 清理全量备份时确保不破坏增量链
                has_dependents = any(
                    m.parent_backup_id == manifest.backup_id
                    for m in self._manifests
                )
                if not has_dependents:
                    self._delete_backup(manifest.backup_id)

    def _delete_backup(self, backup_id: str):
        """删除备份"""
        backup_path = self.backup_dir / backup_id
        if backup_path.exists():
            shutil.rmtree(str(backup_path))
            print(f"[Cleanup] 删除旧备份: {backup_id}")
        self._manifests = [m for m in self._manifests if m.backup_id != backup_id]
        self._save_manifest_index()

    # ========== 辅助方法 ==========

    def _generate_backup_id(self) -> str:
        """生成备份ID"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"backup_{ts}_{rand}"

    def _compute_checksum(self, filepath: str) -> str:
        """计算文件SHA256"""
        sha = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
        except FileNotFoundError:
            return "FILE_NOT_FOUND"
        return sha.hexdigest()

    def _find_manifest(self, backup_id: str) -> Optional[BackupManifest]:
        """查找备份清单"""
        for m in self._manifests:
            if m.backup_id == backup_id:
                return m
        return None

    def _get_restore_chain(self, manifest: BackupManifest) -> list[BackupManifest]:
        """获取恢复链（从当前到最早全量备份）"""
        chain = [manifest]
        current = manifest

        while current.parent_backup_id:
            parent = self._find_manifest(current.parent_backup_id)
            if parent is None:
                break
            chain.append(parent)
            current = parent

        return chain

    def _manifest_to_dict(self, manifest: BackupManifest) -> dict:
        return {
            "backup_id": manifest.backup_id,
            "backup_type": manifest.backup_type.value,
            "created_at": manifest.created_at,
            "total_chunks": manifest.total_chunks,
            "total_vectors": manifest.total_vectors,
            "vector_dimension": manifest.vector_dimension,
            "index_type": manifest.index_type,
            "file_paths": manifest.file_paths,
            "checksums": manifest.checksums,
            "parent_backup_id": manifest.parent_backup_id,
            "status": manifest.status.value,
            "metadata": manifest.metadata,
        }

    def _save_manifest_index(self):
        """保存清单索引"""
        index_path = self.backup_dir / "backup_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            manifests_data = [self._manifest_to_dict(m) for m in self._manifests]
            json.dump(manifests_data, f, indent=2, ensure_ascii=False)

    def _load_manifest_index(self):
        """加载清单索引"""
        index_path = self.backup_dir / "backup_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    manifest = BackupManifest(
                        backup_id=item["backup_id"],
                        backup_type=BackupType(item["backup_type"]),
                        created_at=item["created_at"],
                        total_chunks=item["total_chunks"],
                        total_vectors=item["total_vectors"],
                        vector_dimension=item.get("vector_dimension", 384),
                        index_type=item.get("index_type", "hnsw"),
                        file_paths=item["file_paths"],
                        checksums=item.get("checksums", {}),
                        parent_backup_id=item.get("parent_backup_id"),
                        status=BackupStatus(item.get("status", "completed")),
                        metadata=item.get("metadata", {}),
                    )
                    self._manifests.append(manifest)
            except Exception as e:
                print(f"加载备份索引失败: {e}")

    def list_backups(self) -> list[dict]:
        """列出所有备份"""
        return [self._manifest_to_dict(m) for m in self._manifests]


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("灾难恢复系统 - 演示运行")
    print("=" * 60)

    # 1. 初始化
    plan = DRPlan(
        rto_target=3600.0,
        rpo_target=300.0,
        retention_days=30,
    )
    manager = BackupManager(backup_dir="./test_backups", plan=plan)

    # 2. 准备测试数据
    print("\n--- 准备测试数据 ---")
    test_chunks = [
        {
            "chunk_id": f"chunk_{i:04d}",
            "doc_id": f"doc_{(i // 10):04d}",
            "text": f"这是第{i}个测试块的内容，包含RAG系统的相关说明。" * (i + 1),
            "metadata": {
                "source": f"doc_{(i // 10):04d}.pdf",
                "chapter": i // 20,
                "language": "zh",
                "tenant_id": "tenant_001",
            },
        }
        for i in range(50)
    ]

    test_vectors = np.random.randn(50, 384).tolist()
    test_vector_ids = [f"vec_{i:04d}" for i in range(50)]

    print(f"测试块: {len(test_chunks)}")
    print(f"测试向量: {len(test_vectors)} x 384")

    # 3. 全量备份
    print("\n--- 全量备份 ---")
    full_manifest = manager.full_backup(
        test_chunks, test_vectors, test_vector_ids,
        index_type="hnsw", dimension=384,
    )
    print(f"备份ID: {full_manifest.backup_id}")
    print(f"状态: {full_manifest.status.value}")

    # 4. 增量备份
    print("\n--- 增量备份 ---")

    # 模拟新增/修改的块
    delta_chunks = [
        {
            "chunk_id": "chunk_new_001",
            "doc_id": "doc_0999",
            "text": "这是一个新添加的文档块，包含关于熔断器模式的内容。",
            "metadata": {"source": "new_doc.pdf", "tenant_id": "tenant_001"},
        },
        {
            "chunk_id": "chunk_new_002",
            "doc_id": "doc_0999",
            "text": "这是另一个新添加的文档块，包含关于速率限制的内容。",
            "metadata": {"source": "new_doc.pdf", "tenant_id": "tenant_001"},
        },
    ]
    delta_vectors = np.random.randn(2, 384).tolist()
    delta_vids = ["vec_new_001", "vec_new_002"]

    inc_manifest = manager.incremental_backup(
        delta_chunks, delta_vectors, delta_vids,
        parent_backup_id=full_manifest.backup_id,
    )
    print(f"增量备份ID: {inc_manifest.backup_id}")
    print(f"父备份: {inc_manifest.parent_backup_id}")
    print(f"增量块: {len(delta_chunks)}")

    # 5. 列出所有备份
    print("\n--- 备份列表 ---")
    for b in manager.list_backups():
        print(f"  [{b['backup_type']}] {b['backup_id']} | "
              f"块:{b['total_chunks']} | 向量:{b['total_vectors']} | "
              f"状态:{b['status']}")

    # 6. 跨区域复制
    print("\n--- 跨区域复制 ---")
    replicated = manager.replicate(full_manifest.backup_id, Region.SECONDARY)
    print(f"复制到 {Region.SECONDARY.value}: {'成功 ✅' if replicated else '失败 ❌'}")

    # 7. 恢复验证
    print("\n--- 恢复测试 ---")
    restore_result = manager.restore(full_manifest.backup_id, verify=True)
    print(f"恢复: {'成功 ✅' if restore_result.success else '失败 ❌'}")
    print(f"块数: {restore_result.chunks_restored}")
    print(f"向量数: {restore_result.vectors_restored}")
    print(f"验证: {'通过 ✅' if restore_result.verification_passed else '失败 ❌'}")
    print(f"耗时: {restore_result.duration_seconds:.2f}s")
    print(f"校验和: {'匹配 ✅' if restore_result.checksum_match else '不匹配 ❌'}")

    # 8. RTO/RPO 报告
    print("\n--- RTO/RPO 文档 ---")
    print(f"RTO 目标:  {plan.rto_target:.0f}s (实际: {restore_result.rto_seconds:.2f}s)")
    print(f"RPO 目标:  {plan.rpo_target:.0f}s (备份间隔: {plan.backup_interval:.0f}s)")
    print(f"保留天数:  {plan.retention_days} 天")
    print(f"复制区域:  {[r.value for r in plan.replication_regions]}")
    print(f"最大增量链: {plan.max_incremental_chain} 个")

    rto_met = restore_result.rto_seconds <= plan.rto_target
    print(f"\nRTO达标: {'✅' if rto_met else '❌'}")

    # 9. DR演练
    print("\n--- DR演练 ---")
    drill_result = manager.dr_drill(full_manifest.backup_id, restore_to_temp=True)
    print(f"DR演练: {'通过 ✅' if drill_result.success else '失败 ❌'}")
    print(f"评估: 可以{'正常' if drill_result.success else '不'}从备份恢复")

    # 10. 清理
    print("\n--- 清理 ---")
    try:
        shutil.rmtree("./test_backups")
        print("测试备份文件已清理")
    except Exception as e:
        print(f"清理失败: {e}")

    print("\n" + "=" * 60)
    print("灾难恢复系统演示完成！")
    print("=" * 60)
