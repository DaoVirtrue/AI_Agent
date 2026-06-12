# Pitfall 05: 增量更新索引损坏 (Delta Breaks Main Index)

## 症状 (Symptoms)

- 增量更新后，主索引查询返回空结果或错误结果
- 某些chunk在索引中重复出现（旧版本未删除）
- 向量维度不匹配错误（新chunk向量维度与索引不一致）
- 部分文档的chunk无法被检索到
- 索引文件大小异常增长（未清理已删除的chunk）
- 版本号跳跃（chunk直接从v1跳到v5）

## 根本原因 (Root Cause)

1. **合并前未验证delta质量**: 新增/修改的chunk格式不正确就合并入主索引
2. **删除操作不完整**: 标记删除但索引中仍保留旧向量
3. **版本冲突**: 同时修改同一文档导致版本覆盖
4. **无staging验证**: 直接修改生产索引，没有先在staging环境验证
5. **事务性缺失**: 部分合并操作成功、部分失败，索引处于不一致状态

## 真实场景 (Real Scenario)

某电商平台的产品知识库RAG系统：
- 批量更新1000个产品文档
- 增量更新器处理了800个后因网络问题崩溃
- 400个chunk已写入主索引，400个未写入
- 200个旧chunk已删除
- 结果：主索引状态为"400+200+200"，即新旧混合
- 用户搜索产品时：有的显示新版信息，有的显示旧版，有的显示"产品不存在"

## 完整解决方案 (Complete Solution)

```python
import copy
import threading
from contextlib import contextmanager

class StagingIndexValidator:
    """Staging索引验证器 - 在合并前验证增量数据"""
    
    def __init__(self):
        self._checks = []
    
    def validate_delta(self, delta_chunks: list, delta_vectors: list,
                       delta_ids: list, main_index_dimension: int) -> tuple[bool, list[str]]:
        """
        验证增量数据质量
        Returns: (is_valid, error_messages)
        """
        errors = []
        
        # 检查1: 数量一致性
        if len(delta_chunks) != len(delta_vectors) != len(delta_ids):
            errors.append(f"数量不一致: chunks={len(delta_chunks)}, vectors={len(delta_vectors)}, ids={len(delta_ids)}")
        
        # 检查2: 向量维度
        for i, vec in enumerate(delta_vectors):
            if len(vec) != main_index_dimension:
                errors.append(f"维度不匹配: chunk[{i}] dim={len(vec)} vs index={main_index_dimension}")
        
        # 检查3: 非空chunk
        for i, chunk in enumerate(delta_chunks):
            if not chunk.get("text", "").strip():
                errors.append(f"空chunk: idx={i}, id={chunk.get('chunk_id', 'unknown')}")
        
        # 检查4: chunk_id唯一性
        chunk_ids = [c["chunk_id"] for c in delta_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            duplicates = [cid for cid in chunk_ids if chunk_ids.count(cid) > 1]
            errors.append(f"重复chunk_id: {list(set(duplicates))}")
        
        # 检查5: 必需的元数据字段
        required_meta = ["source"]
        for i, chunk in enumerate(delta_chunks):
            missing = [k for k in required_meta if k not in chunk.get("metadata", {})]
            if missing:
                errors.append(f"缺少元数据: chunk[{i}] missing={missing}")
        
        return len(errors) == 0, errors


class TransactionalIndexMerger:
    """事务性索引合并器 - 保证原子性"""
    
    def __init__(self, main_index):
        self.main_index = main_index
        self._lock = threading.RLock()
        self._backup = None
    
    @contextmanager
    def transactional_merge(self):
        """事务性合并上下文管理器"""
        with self._lock:
            # 创建快照
            self._backup = copy.deepcopy(self.main_index)
            try:
                yield self.main_index
                # 成功：保持修改
                self._backup = None
            except Exception as e:
                # 失败：回滚
                print(f"[Transactional] 合并失败，回滚: {e}")
                self.main_index.update(self._backup)
                self._backup = None
                raise
    
    def merge_with_staging(self, delta_chunks: list, delta_vectors: list,
                           deleted_ids: list) -> dict:
        """带Staging验证的事务性合并"""
        stats = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0, "errors": 0}
        
        # 步骤1: 验证增量数据
        validator = StagingIndexValidator()
        is_valid, errors = validator.validate_delta(
            delta_chunks, delta_vectors,
            [c["chunk_id"] for c in delta_chunks],
            self.main_index.dimension,
        )
        
        if not is_valid:
            print(f"[Staging] 增量数据验证失败:")
            for err in errors:
                print(f"  - {err}")
            return {"skipped": len(delta_chunks), "errors": len(errors)}
        
        # 步骤2: 在Staging环境中测试合并
        staging_index = copy.deepcopy(self.main_index)
        try:
            staging_index.add(delta_chunks, delta_vectors, deleted_ids)
            
            # 步骤3: 验证Staging索引
            staging_health = staging_index.health_check()
            if not staging_health["healthy"]:
                print(f"[Staging] 合并后健康检查失败: {staging_health}")
                return {"skipped": len(delta_chunks), "errors": 1}
            
            # 步骤4: 运行验证查询
            test_queries = self._get_test_queries()
            test_results = staging_index.run_queries(test_queries)
            if not test_results["passed"]:
                print(f"[Staging] 验证查询失败: {test_results}")
                return {"skipped": len(delta_chunks), "errors": 1}
            
        except Exception as e:
            print(f"[Staging] Staging合并异常: {e}")
            return {"skipped": len(delta_chunks), "errors": 1}
        
        # 步骤5: 事务性合并到主索引
        with self.transactional_merge() as index:
            stats = index.add(delta_chunks, delta_vectors, deleted_ids)
        
        return stats
    
    def _get_test_queries(self) -> list[str]:
        """获取验证查询集"""
        return [
            "什么是RAG？",
            "向量数据库",
            "性能优化",
        ]


def version_conflict_resolution():
    """版本冲突解决策略"""
    
    class VersionTracker:
        def __init__(self):
            self._versions: dict[str, int] = {}  # chunk_id -> version
            self._lock = threading.RLock()
        
        def check_and_bump(self, chunk_id: str, expected_version: int) -> int:
            """检查版本并递增（乐观锁）"""
            with self._lock:
                current = self._versions.get(chunk_id, 0)
                
                if expected_version != current:
                    # 版本冲突！
                    raise VersionConflictError(
                        f"版本冲突: {chunk_id} expected=v{expected_version} actual=v{current}"
                    )
                
                new_version = current + 1
                self._versions[chunk_id] = new_version
                return new_version
        
        def force_update(self, chunk_id: str, new_version: int):
            """强制更新（冲突时使用）"""
            with self._lock:
                self._versions[chunk_id] = new_version
    
    class VersionConflictError(Exception):
        pass


def index_consistency_check(index) -> dict:
    """索引一致性检查"""
    checks = {
        "chunk_vector_count_match": len(index.chunks) == len(index.vectors),
        "no_orphan_vectors": all(vid in index.chunks for vid in index.vector_ids),
        "no_deleted_in_index": all(cid not in index.deleted_ids for cid in index.chunks),
        "hnsw_index_size_match": index.hnsw_size == len(index.vectors),
        "bm25_doc_count_match": index.bm25_doc_count == len(index.chunks),
    }
    
    checks["healthy"] = all(checks.values())
    return checks
```

## 检查清单 (Checklist)

- [ ] 增量数据合并前通过Staging环境验证
- [ ] 使用事务性合并（全部成功或全部回滚）
- [ ] 实现乐观锁版本检查（防并发冲突）
- [ ] 每次合并后执行索引一致性检查
- [ ] 合并前创建可回滚快照
- [ ] 限制单次增量大小（超过阈值触发全量重建）
- [ ] 向量维度强制匹配（不匹配时自动拒绝）
- [ ] 监控合并失败率和索引碎片率
- [ ] 区分软删除和硬删除（延迟物理删除）
