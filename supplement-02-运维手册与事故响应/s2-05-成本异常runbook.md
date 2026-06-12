# S2-05：成本异常 Runbook

> **故障类型**：LLM API 成本突发性飙涨（50%+）
> **影响范围**：财务预算、服务可持续性

---

## 一、检测方法

### 自动检测

```yaml
监控指标:
  - llm_daily_cost > baseline × 1.5 → P2 告警（成本飙升50%+）
  - llm_daily_cost > baseline × 2.0 → P1 告警（翻倍）
  - llm_daily_cost > budget_limit → P0 告警（超出预算上限）
  - avg_tokens_per_query > baseline × 1.5 → P2 告警
  - cache_hit_rate < 20% → P2 告警（缓存失效）
```

### 手动检测

```bash
# 查看 OpenAI 使用量
# https://platform.openai.com/usage

# 查看当天成本趋势
python scripts/cost_report.py --today --by-model --by-feature
```

---

## 二、处理步骤

### Step 1：快速止血 —— 检查 Agent 循环（5分钟）

```python
# Agent 无限循环是最常见的成本杀手

# 症状诊断：
# 1. 同一个会话 ID 发起了 50+ 次 LLM 调用
# 2. 工具调用链无进展但持续调用
# 3. 每次调用的 prompt 几乎相同

def detect_agent_loops():
    """检测 Agent 是否陷入死循环。"""
    from collections import Counter

    recent_calls = fetch_recent_llm_calls(minutes=30)

    # 按会话ID统计调用次数
    session_counts = Counter(
        c["session_id"] for c in recent_calls
    )

    # 找出异常高的会话
    suspicious = []
    for session_id, count in session_counts.items():
        if count > 20:  # 单会话30分钟内超过20次调用
            calls = [c for c in recent_calls if c["session_id"] == session_id]
            suspicious.append({
                "session_id": session_id,
                "call_count": count,
                "estimated_cost": sum(c["cost"] for c in calls),
            })

    return suspicious

# 紧急措施：
# 1. 终止可疑会话
# 2. 限制单会话最大迭代次数（max_iterations=10）
# 3. 对 Agent 禁用无需的工具
```

### Step 2：检查缓存命中率（10分钟）

```bash
# 检查 Redis 缓存健康度
redis-cli INFO stats | grep -E "keyspace_hits|keyspace_misses"

# 缓存命中率 = hits / (hits + misses)
# 如果 < 30%，说明缓存策略需要调整
```

```python
# 缓存健康检查
def check_cache_health():
    metrics = redis_client.info("stats")
    hits = metrics.get("keyspace_hits", 0)
    misses = metrics.get("keyspace_misses", 0)
    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0

    if hit_rate < 0.30:
        print(f"⚠️ 缓存命中率异常: {hit_rate:.1%}")
        print("  可能原因: 缓存过期/容量不足/缓存策略变更")

    return hit_rate
```

### Step 3：检查模型路由（15分钟）

```bash
# 是否有大量流量被路由到了昂贵模型？
# 检查模型使用分布
python scripts/cost_report.py --by-model --last-hour

# 理想分布:
#   gpt-4o-mini: 70% (日常简单问答)
#   gpt-4o: 20% (复杂分析)
#   claude-sonnet: 8% (代码/长文本)
#   deepseek-v3: 2% (降级)

# 异常分布（需要排查）:
#   gpt-4o: 90% ← 路由策略可能有问题
```

```python
# 强制降级路由（紧急情况）
# 将所有流量临时切换到便宜模型
KUBE_CMD = """
kubectl set env deployment/rag-api \\
  LLM_DEFAULT_MODEL=gpt-4o-mini \\
  LLM_FALLBACK_MODEL=deepseek-v3
"""
```

### Step 4：启用限流（20分钟）

```yaml
# API Gateway 层面的速率限制
# Kong/Traefik/Nginx 配置

限流策略（紧急模式）：
  - 每个用户: 20次/分钟
  - 每个 IP: 50次/分钟
  - 全局: 500次/分钟

恢复策略（逐步放开）：
  1. 先放开全局限制到 1000次/分钟
  2. 观察 30 分钟
  3. 逐步恢复到正常水平
```

### Step 5：根因分析与复盘

```yaml
常见成本异常原因及频率：

1. Agent 无限循环 (35%)
   根因: max_iterations 设置过高 / 工具描述不清晰导致重复调用
   修复: 限制 max_iterations ≤ 10 / 添加循环检测 / 设置 max_cost_per_session

2. 缓存失效 (25%)
   根因: Redis OOM / 缓存策略变更 / 缓存过期时间过短
   修复: 增加 Redis 内存 / 恢复缓存策略 / 延长 TTL

3. 模型路由偏差 (20%)
   根因: 路由逻辑 Bug / 便宜模型不可用
   修复: 修正路由规则 / 恢复备用模型

4. 批量任务误触发 (10%)
   根因: 定时任务配置错误 / 数据重处理意外触发
   修复: 增加批量任务审批流程

5. Prompt 膨胀 (10%)
   根因: Prompt 模板中上下文无限增长
   修复: 限制上下文大小 / 启用滑动窗口
```
