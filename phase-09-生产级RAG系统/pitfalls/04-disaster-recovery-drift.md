# Pitfall 04: 灾难恢复漂移 (Backup Stale, Restore Fails)

## 症状 (Symptoms)

- 备份文件存在但恢复时发现数据不完整
- 增量备份链断裂：第N个增量备份引用的父备份已被清理
- 备份校验和不匹配（静默数据损坏）
- 跨区域复制的备份比主区域落后数小时
- DR演练时发现恢复后的索引无法正常查询
- 备份数据格式与当前代码版本不兼容

## 根本原因 (Root Cause)

1. **备份未验证**: 创建备份后未验证其可恢复性
2. **清理策略不当**: 删除旧全量备份时未检查增量依赖链
3. **跨区域复制延迟**: 没有监控复制延迟和完整性
4. **Schema变更未向后兼容**: 代码更新后旧备份格式无法解析
5. **恢复过程未自动化**: 手工恢复步骤多，容易出错

## 真实场景 (Real Scenario)

某金融科技公司每周做全量备份，每天做增量备份。保留策略是30天。
- 第20天：运维删除了"旧的"全量备份（第1天）
- 第21天：磁盘故障，需要恢复
- 增量备份链：全量(D1) → 增量(D2) → ... → 增量(D20)
- 但D1已被删除！所有增量备份失效
- RTO从1小时变为72小时（必须从更早的离线备份恢复）
- 损失数百万美元

## 完整解决方案 (Complete Solution)

```python
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class ValidatedBackupManager:
    """带验证的备份管理器"""
    
    def __init__(self, backup_dir: str):
        self.backup_dir = Path(backup_dir)
        self._manifest_index: dict = {}
        self._schema_version = "2.1"
    
    def create_full_backup(self, chunks: list, vectors: list, ids: list) -> str:
        """创建全量备份（带完整性验证）"""
        backup_id = self._generate_id()
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True)
        
        # 1. 写入数据
        self._write_backup_data(backup_path, chunks, vectors, ids)
        
        # 2. 立即验证
        if not self._verify_backup(backup_path):
            raise BackupVerificationError(f"备份验证失败: {backup_id}")
        
        # 3. 跨区域复制（异步触发）
        self._schedule_replication(backup_id)
        
        # 4. 试恢复验证（在临时目录）
        self._test_restore(backup_path)
        
        return backup_id
    
    def _verify_backup(self, backup_path: Path) -> bool:
        """验证备份完整性"""
        checks = []
        
        # 检查1: 所有文件存在且非空
        for expected_file in ["chunks.parquet", "vectors.parquet", "manifest.json"]:
            fp = backup_path / expected_file
            if not fp.exists() or fp.stat().st_size == 0:
                checks.append(False)
                print(f"  ❌ 文件缺失或为空: {expected_file}")
            else:
                checks.append(True)
        
        # 检查2: 校验和验证
        manifest = self._load_manifest(backup_path)
        for file_path, expected_checksum in manifest.get("checksums", {}).items():
            actual_checksum = self._compute_checksum(file_path)
            if actual_checksum != expected_checksum:
                checks.append(False)
                print(f"  ❌ 校验和不匹配: {file_path}")
            else:
                checks.append(True)
        
        # 检查3: 数据计数一致
        if manifest.get("total_chunks", 0) == 0:
            checks.append(False)
            print(f"  ❌ 空备份: 0个块")
        else:
            checks.append(True)
        
        # 检查4: Schema版本兼容
        backup_schema = manifest.get("schema_version", "unknown")
        if backup_schema != self._schema_version:
            checks.append(False)
            print(f"  ❌ Schema版本不匹配: {backup_schema} vs {self._schema_version}")
        else:
            checks.append(True)
        
        return all(checks)
    
    def _test_restore(self, backup_path: Path) -> bool:
        """试恢复测试"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # 尝试加载数据
                chunks = self._load_chunks(backup_path / "chunks.parquet")
                vectors = self._load_vectors(backup_path / "vectors.parquet")
                
                # 验证数据可查询
                if len(chunks) > 0 and len(vectors) == len(chunks):
                    print(f"  ✅ 试恢复成功: {len(chunks)} 块")
                    return True
                else:
                    print(f"  ❌ 试恢复: 块数({len(chunks)}) != 向量数({len(vectors)})")
                    return False
            except Exception as e:
                print(f"  ❌ 试恢复失败: {e}")
                return False
    
    def safe_cleanup(self, retention_days: int = 30):
        """安全清理旧备份（不破坏增量链）"""
        cutoff = time.time() - retention_days * 86400
        
        # 建立依赖图
        dependency_graph = self._build_dependency_graph()
        
        # 标记不可删除的备份（被增量备份引用）
        protected = set()
        for inc_backup_id, parent_id in dependency_graph.items():
            # 收集整个链
            current = parent_id
            while current:
                protected.add(current)
                current = dependency_graph.get(current)
        
        # 清理
        for manifest in self._get_manifests_older_than(cutoff):
            backup_id = manifest["backup_id"]
            if backup_id not in protected:
                self._delete_backup(backup_id)
            else:
                print(f"  保留(被增量链引用): {backup_id}")
    
    def monitor_replication_lag(self, backup_id: str) -> dict:
        """监控跨区域复制延迟"""
        primary_ts = self._get_backup_timestamp(backup_id, "primary")
        secondary_ts = self._get_backup_timestamp(backup_id, "secondary")
        
        lag = (primary_ts - secondary_ts) if primary_ts and secondary_ts else float("inf")
        
        return {
            "backup_id": backup_id,
            "replication_lag_seconds": lag,
            "status": "healthy" if lag < 300 else "degraded",  # 5分钟阈值
        }
    
    def run_dr_drill(self, backup_id: str) -> dict:
        """自动化DR演练"""
        results = {
            "backup_id": backup_id,
            "started_at": time.time(),
            "steps": [],
        }
        
        # 步骤1: 验证备份存在
        step1 = self._verify_backup_exists(backup_id)
        results["steps"].append({"step": "verify_exists", "passed": step1})
        
        # 步骤2: 校验和验证
        step2 = self._verify_backup(self.backup_dir / backup_id)
        results["steps"].append({"step": "verify_checksums", "passed": step2})
        
        # 步骤3: 试恢复
        step3 = self._test_restore(self.backup_dir / backup_id)
        results["steps"].append({"step": "test_restore", "passed": step3})
        
        # 步骤4: 验证查询能力
        step4 = self._verify_query_ability(backup_id)
        results["steps"].append({"step": "verify_queries", "passed": step4})
        
        # 步骤5: RTO评估
        results["rto_seconds"] = time.time() - results["started_at"]
        results["passed"] = all(s["passed"] for s in results["steps"])
        
        return results


def rto_rpo_documentation():
    """RTO/RPO文档模板"""
    return {
        "rto": {
            "target": "1小时内",
            "measured_last_drill": "12分钟",
            "breakdown": {
                "backup_discovery": "2分钟",
                "data_transfer": "5分钟",
                "index_rebuild": "3分钟",
                "validation": "2分钟",
            },
        },
        "rpo": {
            "target": "5分钟",
            "achieved": "增量备份每5分钟执行",
            "data_loss_risk": "最多5分钟的增量数据",
        },
        "backup_schedule": {
            "full": "每周日 03:00 UTC",
            "incremental": "每5分钟",
            "retention": "30天",
            "replication": "同步到3个区域",
        },
        "drill_schedule": "每月第一个周一 10:00 自动执行",
    }
```

## 检查清单 (Checklist)

- [ ] 每次备份后自动验证完整性
- [ ] 定期试恢复（至少每周一次）
- [ ] 清理旧备份前检查增量依赖链
- [ ] 跨区域复制延迟监控（阈值 < 5分钟）
- [ ] Schema版本管理（备份数据包含版本号）
- [ ] DR演练自动化（每月至少一次）
- [ ] 备份元数据索引（快速查找备份）
- [ ] 备份数据加密（静态加密 + 传输加密）
- [ ] RTO/RPO 监控仪表板
- [ ] 备份失败告警（PagerDuty/钉钉/邮件）
