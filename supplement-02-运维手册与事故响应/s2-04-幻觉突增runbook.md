# S2-04：幻觉突增 Runbook

> **故障类型**：RAG 回答质量突然下降，幻觉率显著上升
> **影响范围**：用户信任度和满意度

---

## 一、检测方法

### 自动检测

```yaml
监控指标:
  - hallucination_rate > 5% → P2 告警
  - hallucination_rate > 10% → P1 告警
  - user_dislike_rate > 15% → P2 告警
  - answer_confidence_avg < 0.7 → P2 告警
```

### 手动检测

```
查看用户反馈：
- Dify/RAGFlow 后台 → 标注 → 点踩记录
- 客服工单中"回答不准确"的投诉趋势
```

---

## 二、处理步骤

### Step 1：确认退化程度（5分钟）

```python
# 用黄金标准测试集验证
GOLDEN_DATASET = [
    # (问题, 期望出现在答案中的关键词)
    ("公司的年假政策是什么？", ["年假", "天", "HR系统"]),
    ("如何申请报销？", ["报销", "发票", "审批"]),
    ("退款需要什么条件？", ["退款", "7天", "包装"]),
    ("产品支持哪些部署方式？", ["私有化", "SaaS", "混合云"]),
    ("如何联系技术支持？", ["工单", "电话", "邮件"]),
]

def run_golden_regression():
    failures = []
    for query, expected_keywords in GOLDEN_DATASET:
        answer = rag_system.query(query)
        missing = [kw for kw in expected_keywords if kw not in answer]
        if missing:
            failures.append({
                "query": query,
                "missing_keywords": missing,
                "answer_preview": answer[:200],
            })
    return failures
```

### Step 2：回滚索引（如果最近更新了索引）

```bash
# 回滚到上一个已知良好的索引版本
milvus-backup restore --collection knowledge_base --backup 20260610_stable

# 验证回滚效果
python scripts/run_golden_regression.py
```

### Step 3：检查嵌入模型（10分钟）

```bash
# 确认嵌入模型版本是否变更
# BGE-M3 版本变化会导致检索结果偏移

# 检查嵌入模型的配置
grep "EMBEDDING_MODEL" .env config/*.yaml

# 如果是自部署模型，检查：
# - 模型权重是否被意外更新
# - GPU 驱动是否有兼容性问题
```

### Step 4：检查数据源（15分钟）

```bash
# 检查最近上传的文档
# 是否有大量低质量/错误文档被导入？

# 查看最近添加的文档列表
SELECT title, uploaded_at, file_size
FROM documents
WHERE uploaded_at > NOW() - INTERVAL '7 days'
ORDER BY uploaded_at DESC;

# 如果有问题文档，批量移除
python scripts/remove_documents.py --uploaded-after "2026-06-04"
```

### Step 5：黄金数据集全量回归（30分钟）

```python
def full_golden_regression():
    """
    用完整的黄金数据集（200+题）进行回归。

    黄金数据集应覆盖：
    - 各知识领域（产品/技术/制度/FAQ）
    - 各难度级别（简单/中/难）
    - 各问题类型（事实/推理/对比/流程）
    - 边界条件（空搜索、超长问题、恶意输入）
    """
    results = {
        "total": len(GOLDEN_DATASET_FULL),
        "passed": 0,
        "failed": 0,
        "hallucination_score": 0,
    }
    # ... 运行完整评估
    return results
```

### Step 6：根因分析

```
常见幻觉突增原因：

1. 索引质量下降（40%）
   - 文档更新引入了错误内容
   - 索引构建时遗漏了部分文档
   - 解决：回滚索引 + 审查新增文档

2. Prompt 变更（30%）
   - 最近修改了 Prompt 模板
   - 修改后的 Prompt 降低了引用要求
   - 解决：回滚 Prompt + 运行回归测试

3. 模型行为变化（20%）
   - LLM 供应商悄然更新了模型
   - temperature 参数被错误修改
   - 解决：固定模型版本 + 锁定参数

4. 检索策略变更（10%）
   - 修改了 top_k、相似度阈值等参数
   - 解决：恢复默认参数
```

---

## 三、预防措施

```yaml
防护策略（多层防御）:

第1层：检索质量保证
  - 相似度阈值过滤（score < 0.5 丢弃）
  - 多样性检查（确保检索结果覆盖不同文档）
  - 时效性检查（确保检索到最新版本）

第2层：Prompt 约束
  - 明确要求"只基于文档回答"
  - "不知道就说不知道"
  - 强制要求引用来源

第3层：后处理校验
  - 对每个回答做 faithfulness 检测
  - 低置信度回答自动标记
  - 敏感领域强制人工审核

第4层：持续监控
  - 每日运行黄金数据集回归
  - 监控用户反馈趋势
  - 每周人工抽查 100 条回答
```
