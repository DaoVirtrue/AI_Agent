# 坑点 #2：记忆无限增长导致系统崩溃

## 症状
运行 3 个月后：检索延迟从 50ms 飙升到 5s，存储从 100MB 涨到 10GB，API 月费从 $100 涨到 $3000。

## 根因
1. **无保留策略**：所有对话永久存储，从未清理
2. **无 TTL**：3 个月前的临时聊天记录仍然占着存储
3. **无分级存储**：重要记忆和不重要的闲聊存在同一张表
4. **索引膨胀**：HNSW 索引随数据量增大，性能指数级下降

## 解决方案

### 1. 艾宾浩斯遗忘曲线
```python
class EbbinghausRetention:
    def retention(self, elapsed_days: float, strength: float = 1.0) -> float:
        """R = e^(-t / (s * 7))  7天为基准半衰期"""
        import math
        return math.exp(-elapsed_days / (strength * 7))
    
    def should_evict(self, memory_age_days: float, importance: float, threshold=0.3):
        retention = self.retention(memory_age_days, importance)
        return retention < threshold

# 示例：重要性 0.5 的记忆，30 天后保留率仅 0.04% → 淘汰
```

### 2. TTL 按类型分级
```python
MEMORY_TTL = {
    "user_preference": None,      # 永不过期
    "task_result": 90,            # 90 天
    "conversation_turn": 30,      # 30 天
    "casual_chat": 7,             # 7 天
    "tool_call_log": 14,          # 14 天
}

def clean_expired(store):
    for entry in store.all():
        ttl = MEMORY_TTL.get(entry['type'], 30)
        if ttl and (now() - entry['created_at']).days > ttl:
            store.delete(entry['id'])
```

### 3. 分层存储
```python
# 热存储（Redis，<1ms）：本周的活跃对话
# 温存储（SQLite，<10ms）：30天内的历史
# 冷存储（S3/文件，>100ms）：归档数据，按需加载
```

### 4. 定期压缩
```python
def weekly_vacuum(store):
    store.vacuum()                    # 回收空间
    store.rebuild_index()             # 重建 HNSW
    store.merge_duplicates(threshold=0.95)  # 合并相似记忆
```

## 检查清单
- [ ] 每种记忆类型是否配置了 TTL？
- [ ] 是否实现了艾宾浩斯衰减或等效的淘汰策略？
- [ ] 是否设置了存储上限（按容量和条目数）？
- [ ] 是否定期执行 vacuum 和索引重建？
- [ ] 是否有存储增长监控告警？
- [ ] 冷数据是否归档到低成本存储？
