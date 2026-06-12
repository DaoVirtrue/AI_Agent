# Pitfall 02: 缓存失效问题 (Stale Cache After Knowledge Update)

## 症状 (Symptoms)

- 文档更新后，用户查询仍返回旧版本的内容
- 新添加的文档在搜索中不可见（缓存未失效）
- 已删除文档的内容仍然出现在检索结果中
- L3检索结果缓存返回的chunk ID在索引中已不存在
- 用户反馈"信息过时"的投诉增加

## 根本原因 (Root Cause)

缓存失效机制不完善：

1. **L3缓存未与索引更新联动**: 索引重建/增量更新后未清理L3检索结果缓存
2. **L2语义缓存过于宽泛**: 相似查询命中了旧内容的缓存
3. **L4 LLM响应缓存TTL过长**: 事实性内容应该按内容类型分TTL
4. **缺少事件驱动的失效**: 文档变更后未主动通知缓存层

## 真实场景 (Real Scenario)

某新闻聚合RAG系统每天增量更新500篇新闻。某天发布了一条重要的更正新闻，但由于：
- L3缓存的TTL设置为24小时
- 增量更新后未触发缓存失效
- 更正新闻在用户查询中完全不可见

结果导致用户持续看到旧信息超过12小时，引发公关危机。

## 完整解决方案 (Complete Solution)

```python
import time
from enum import Enum
from collections import defaultdict

class CacheInvalidationStrategy(Enum):
    """缓存失效策略"""
    TTL_ONLY = "ttl_only"                    # 仅依赖TTL
    WRITE_THROUGH = "write_through"          # 写穿透（更新时同时更新缓存）
    WRITE_INVALIDATE = "write_invalidate"    # 写失效（更新时删除缓存）
    EVENT_DRIVEN = "event_driven"            # 事件驱动（发布订阅）


class EventDrivenInvalidator:
    """事件驱动的缓存失效器"""
    
    def __init__(self):
        self._subscribers: dict[str, list[callable]] = defaultdict(list)
        self._version: int = 0
    
    def subscribe(self, event_type: str, callback):
        """订阅事件"""
        self._subscribers[event_type].append(callback)
    
    def publish(self, event_type: str, **kwargs):
        """发布事件"""
        self._version += 1
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(**kwargs)
            except Exception as e:
                print(f"失效回调失败: {e}")


class CacheInvalidationManager:
    """缓存失效管理器"""
    
    def __init__(self, cache_manager, event_bus):
        self.cache = cache_manager
        self.events = event_bus
        self._setup_listeners()
    
    def _setup_listeners(self):
        """设置事件监听"""
        # 索引更新事件 → 失效L3
        self.events.subscribe("index.updated", self._on_index_updated)
        # 文档更新事件 → 失效L3中相关条目
        self.events.subscribe("document.updated", self._on_document_updated)
        # 文档删除事件 → 失效所有层
        self.events.subscribe("document.deleted", self._on_document_deleted)
        # LLM模型更新 → 失效L4
        self.events.subscribe("model.updated", self._on_model_updated)
    
    def _on_index_updated(self, **kwargs):
        """索引更新: 失效所有L3缓存"""
        count = self.cache.l3_invalidate_by_index_update()
        print(f"[Invalidation] 索引更新，失效L3缓存: {count} 条")
    
    def _on_document_updated(self, doc_id: str, **kwargs):
        """文档更新: 失效包含该文档的L3缓存"""
        # 查找包含该doc_id的所有L3缓存条目并删除
        count = self._invalidate_l3_by_doc_id(doc_id)
        print(f"[Invalidation] 文档更新 {doc_id}，失效L3缓存: {count} 条")
    
    def _on_document_deleted(self, doc_id: str, **kwargs):
        """文档删除: 失效所有相关缓存"""
        l3_count = self._invalidate_l3_by_doc_id(doc_id)
        # 同时失效L1和L2（因为查询结果可能包含已删除文档）
        l1_count = self._invalidate_l1_l2_related(doc_id)
        print(f"[Invalidation] 文档删除 {doc_id}，失效: L1/L2={l1_count}, L3={l3_count}")
    
    def _on_model_updated(self, model_name: str, **kwargs):
        """模型更新: 失效L4缓存"""
        count = self.cache.l4_invalidate_by_content_type("factual")
        print(f"[Invalidation] 模型更新 {model_name}，失效L4缓存: {count} 条")
    
    def _invalidate_l3_by_doc_id(self, doc_id: str) -> int:
        """按文档ID失效L3缓存"""
        # 生产中需要维护 doc_id → cache_keys 的倒排索引
        return 0
    
    def _invalidate_l1_l2_related(self, doc_id: str) -> int:
        """失效与文档相关的L1/L2缓存"""
        return 0


def correct_cache_configuration():
    """正确的缓存配置"""
    return {
        "L1": {
            "ttl": 3600,  # 1小时
            "invalidation": "exact_query_change",
            "max_entries": 10000,
        },
        "L2": {
            "ttl": 1800,  # 30分钟
            "invalidation": "index_update",
            "similarity_threshold": 0.92,
        },
        "L3": {
            "ttl": 86400,  # 1天（但索引更新时强制失效）
            "invalidation": "event_driven + index_version",
            "strategy": "write_invalidate",
        },
        "L4": {
            "ttl": {
                "factual": 604800,    # 7天
                "analytical": 86400,  # 1天
                "creative": 21600,    # 6小时
                "real_time": 300,     # 5分钟
            },
            "invalidation": "content_type_based",
        },
    }
```

## 检查清单 (Checklist)

- [ ] 索引更新/重建时自动失效L3全部缓存
- [ ] 文档变更事件触发增量缓存失效
- [ ] L4缓存按内容类型设置不同TTL
- [ ] 缓存键包含版本号（index_version）
- [ ] 维护 doc_id → cache_key 倒排索引用于精准失效
- [ ] 监控缓存命中率骤降（可能表示缓存污染）
- [ ] 提供手动缓存清理API用于紧急情况
- [ ] 写入后立即验证（read-after-write consistency check）
