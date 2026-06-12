# Phase 11 练习题解答 - 企业安全与运营

## 练习 1 解答: 四层安全防御

```python
# 关键发现:
# Layer 1 阻止: 提示词注入, SQL注入, 越狱
# Layer 2 警告: PII检测
# Layer 4 检查: 内容策略

coordinator = SecurityCoordinator()
coordinator.initialize_canary_tokens()
# 每个场景测试见演示代码
```

## 练习 2-6 解答 (代码示例)
```python
# 新增注入模式示例:
(r'(?i)act\s+as\s+if\s+you\s+have\s+no\s+(knowledge|memory)\s+of',
 'memory_wipe', '记忆消除尝试'),
# ...更多模式
```

## 练习 7 解答: 综合安全方案
```python
# 关键架构决策:
# 1. 四层纵深防御 (每层独立，层间协作)
# 2. RBAC+ABAC 双层权限控制
# 3. 哈希链不可变审计日志
# 4. 金丝雀部署 + 自动回滚
# 5. 五级成本优化 + 预算告警
# 6. 6级服务降级保障可用性
```
