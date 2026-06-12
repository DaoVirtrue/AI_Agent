# 坑点 #1：Agent 中途遗忘关键上下文

## 症状
用户："我之前说过我是素食者，你怎么又推荐牛排馆？"  
Agent 在第 8 轮对话中推荐了完全不合适的餐厅，仿佛前 7 轮对话从未发生。

## 根因
1. **STM 窗口太小**：`max_tokens=4000` 只保留最近 3-4 轮，关键信息在第 5 轮被丢弃
2. **无 LTM 提升**：重要信息（用户偏好、任务目标）没有被提升到长期记忆
3. **无重要性评分**：所有消息同等对待，"我是素食者"和"今天天气不错"权重一样

## 解决方案

### 1. 重要性感知的消息保留
```python
class ImportanceAwareMemory:
    KEY_PATTERNS = [
        (r"我是|我的是|我的偏好", 0.9),     # 用户偏好声明
        (r"不要|禁止|千万别|绝对不能", 0.85), # 强约束
        (r"电话号码|邮箱|地址.*是", 0.8),     # 联系信息
        (r"目标|任务|要做|需要完成", 0.7),    # 任务目标
    ]
    
    def score_importance(self, message: str) -> float:
        score = 0.3  # 基础分
        for pattern, weight in self.KEY_PATTERNS:
            if re.search(pattern, message):
                score = max(score, weight)
        return score
    
    def evict(self, messages, budget):
        """淘汰时保留高重要性消息"""
        scored = [(self.score_importance(m['content']), m) for m in messages]
        scored.sort(key=lambda x: x[0], reverse=True)
        # 优先淘汰低分消息
        ...
```

### 2. STM → LTM 自动提升
```python
def promote_to_ltm(self, stm_entry, importance_threshold=0.7):
    if self.score_importance(stm_entry['content']) >= importance_threshold:
        self.ltm.store(
            content=stm_entry['content'],
            metadata={"type": "user_preference", "source": "stm_promotion"}
        )
```

### 3. 上下文自我检查
每次响应前，Agent 扫描 LTM 中是否有相关偏好：
```python
def check_memory(self, current_task):
    relevant = self.ltm.retrieve(current_task, top_k=5)
    constraints = [m for m in relevant if m['metadata']['type'] == 'user_preference']
    return f"已知用户约束：{constraints}" if constraints else ""
```

## 检查清单
- [ ] STM 是否根据消息重要性和时间双重排序？
- [ ] 用户偏好/约束是否自动提升到 LTM？
- [ ] 每次响应前是否检查 LTM 中的相关约束？
- [ ] 是否设置了 `max_tokens` 和压缩阈值（80%）？
- [ ] 是否在日志中记录了内存淘汰决策？
- [ ] 是否有监控告警当关键信息被淘汰？
