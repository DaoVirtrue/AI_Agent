# S2-02：LLM API 故障 Runbook

> **故障类型**：LLM API 不可用、超时、返回异常
> **影响范围**：所有 AI 生成功能

---

## 一、检测方法

### 自动检测

```yaml
监控指标:
  - llm_api_success_rate < 95% → P2 告警
  - llm_api_success_rate < 80% → P1 告警
  - llm_api_success_rate < 50% → P0 告警
  - llm_api_p99_latency > 10s → P2 告警
  - llm_api_error_rate_5xx > 5% → P1 告警

检测端点:
  - 健康检查: GET /api/health (每30秒)
  - 主动探测: 每分钟发送一个已知查询
```

### 手动检测

```bash
# 测试 LLM API 连通性
curl -X POST https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}]}'

# 检查供应商状态页
# OpenAI: https://status.openai.com
# Anthropic: https://status.anthropic.com
```

---

## 二、处理步骤

### Step 1：确认故障范围（2分钟）

```bash
# 1. 确认是单个模型还是全部模型
grep "model=" /var/log/rag/llm_calls.log | tail -100

# 2. 确认是单个供应商还是全部供应商
# 分别测试每个供应商
```

### Step 2：切换到备用供应商（5分钟）

```python
# 自动降级链配置（应在代码中预置）
FALLBACK_CHAIN = {
    "gpt-4o": [
        ("anthropic", "claude-sonnet-4-20250514"),  # 第一备用
        ("deepseek", "deepseek-v3"),                 # 第二备用
        ("openai", "gpt-4o-mini"),                   # 最后降级
    ],
}

# 手动切换（如果自动降级未触发）
# 更新配置中心/环境变量
export LLM_PRIMARY_PROVIDER=anthropic
export LLM_PRIMARY_MODEL=claude-sonnet-4-20250514

# 重启服务（优雅重启，不丢连接）
kubectl rollout restart deployment/rag-api
```

### Step 3：用户通知（10分钟）

```
状态页更新：
"由于上游 LLM 服务提供商出现故障，我们已将服务切换到备用模型。
您可能会注意到回答风格的细微变化。功能不受影响。

当前状态: 🔶 降级运行中
备用模型: Claude Sonnet 4
预计恢复: 正在等待上游服务恢复"
```

### Step 4：故障恢复（持续监控）

```bash
# 持续监控主供应商状态
watch -n 30 'curl -s https://status.openai.com | grep -i "operational"'

# 主供应商恢复后，逐步切回流量
# 1. 先切 5% 流量验证
# 2. 观察 10 分钟
# 3. 逐步提升到 100%
```

### Step 5：事后复盘

见 postmortem-template.md

---

## 三、多供应商配置

```python
# LLM 供应商配置（应在代码中实现）
class LLMRouter:
    """
    智能 LLM 路由器：自动故障切换 + 负载均衡。
    """

    def __init__(self):
        self.providers = {
            "openai": {"healthy": True, "priority": 1},
            "anthropic": {"healthy": True, "priority": 2},
            "deepseek": {"healthy": True, "priority": 3},
        }
        self.circuit_breaker_threshold = 5  # 连续失败5次断开
        self.failure_counts = {}

    async def call_llm(self, messages, **kwargs):
        """按优先级尝试各个供应商。"""
        errors = []

        for provider, config in sorted(
            self.providers.items(),
            key=lambda x: x[1]["priority"]
        ):
            if not config["healthy"]:
                continue

            # 断路器检查
            failures = self.failure_counts.get(provider, 0)
            if failures >= self.circuit_breaker_threshold:
                config["healthy"] = False
                print(f"  [断路器] {provider} 已断开")
                continue

            try:
                result = await self._call_provider(provider, messages, **kwargs)
                self.failure_counts[provider] = 0  # 重置失败计数
                return result
            except Exception as e:
                self.failure_counts[provider] = failures + 1
                errors.append(f"{provider}: {e}")
                continue

        raise RuntimeError(f"所有 LLM 供应商调用失败: {errors}")
```

---

## 四、常见故障及快速修复

| 症状 | 可能原因 | 快速修复 |
|------|---------|----------|
| 429 Too Many Requests | 超过速率限制 | 降低并发、启用退避重试 |
| 503 Service Unavailable | 供应商宕机 | 切换备用供应商 |
| Timeout (>30s) | 请求太复杂/供应商过载 | 减小上下文、切换模型 |
| 401 Unauthorized | API Key 过期 | 更新 API Key |
| 返回空内容 | Token 超限 | 限制上下文长度 |
| 返回乱码 | 编码问题 | 检查 prompt 编码 |
