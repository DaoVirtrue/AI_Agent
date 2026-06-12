# 生产运维手册 (Production Runbook)

## 1. 事件严重性分级

| 级别 | 名称 | 描述 | 响应SLA | 升级时间 |
|------|------|------|---------|---------|
| P0 | 紧急 | 服务完全不可用，影响所有用户 | 5分钟响应 | 15分钟→工程VP |
| P1 | 严重 | 核心功能不可用，影响>50%用户 | 15分钟响应 | 1小时→Tech Lead |
| P2 | 中等 | 非核心功能异常，影响<50%用户 | 1小时响应 | 4小时→Team Lead |
| P3 | 一般 | 性能下降，用户可见但功能正常 | 工作日响应 | 24小时 |
| P4 | 低 | 小问题，不影响用户体验 | 下个迭代 | 无 |

## 2. On-Call轮值

- 主On-Call: 7天轮换
- 副On-Call: 7天轮换（偏移3.5天）
- 时区覆盖: APAC→EMEA→AMER
- 工具: PagerDuty (路由→电话→Slack→邮件)

## 3. 告警→事件映射

| 告警 | 严重性 | Runbook |
|------|--------|---------|
| `service_down > 5m` | P0 | LLM Outage |
| `error_rate > 5% for 5m` | P1 | Hallucination Spike |
| `p99_latency > 10s for 5m` | P2 | Performance Degradation |
| `vector_db_unavailable > 1m` | P1 | Vector DB Failure |
| `cost_rate > budget for 15m` | P2 | Cost Anomaly |
| `suspicious_injection_detected` | P1 | Security Incident |

## 4. Runbooks

### Runbook 1: LLM API Outage (P0)
**症状**: 所有API调用返回错误，错误率>50%
**诊断**:
1. 检查: `/health/readiness` → `llm_api: down`
2. 检查: API状态页面 (status.openai.com / status.anthropic.com)
3. 检查: 是否有API配额或速率限制

**恢复步骤**:
1. 如果外部API宕机: 启用降级模式 → 使用缓存响应
2. 如果是配额问题: 联系账户经理增加配额
3. 如果是速率限制: 降低并发，增加退避时间
4. 回退到备用LLM提供商 (如果配置了)

**回滚**: 无需回滚（外部服务问题）
**预防**: 配置多个LLM提供商故障转移

### Runbook 2: Vector DB Failure (P1)
**症状**: 检索返回0结果，回答质量下降
**诊断**:
1. `kubectl get pods -l app=milvus`
2. `docker logs milvus-container --tail 100`
3. 检查磁盘空间: `df -h`

**恢复步骤**:
1. 如果Pod挂掉: `kubectl rollout restart deployment/milvus`
2. 如果磁盘满: 清理旧日志和索引
3. 如果集群分裂: 重建etcd集群
4. 紧急回退: 使用BM25关键词搜索作为后备

### Runbook 3: Hallucination Spike (P1)
**症状**: 用户报告大量不准确回答，忠实度评分下降
**诊断**:
1. 检查最近部署 (是否有新版本)
2. 检查是否有数据漂移 (MMD/KS检验)
3. 抽查最近的Trace (LangSmith)

**恢复步骤**:
1. 如果最近部署导致: 立即回滚到上一个版本
2. 如果数据漂移: 重新索引知识库
3. 如果提示词问题: 恢复为已验证的提示词版本

### Runbook 4: Cost Anomaly (P2)
**症状**: 每小时成本超过预期的200%
**诊断**:
1. 检查模型路由日志 (是否有异常大量请求用高价模型)
2. 检查是否有循环调用 (Agent无限重试)
3. 检查是否有恶意用户 (异常请求量)

**恢复步骤**:
1. 限制高价模型的调用 → 强制使用便宜模型
2. 增加Rate Limiting
3. 封禁异常用户
4. 启用L1降级 (成本保护模式)

## 5. 事后分析模板 (Post-Mortem)

```markdown
# 事件事后分析: [事件名称]

## 事件概述
- 日期: YYYY-MM-DD
- 持续时间: X小时Y分钟
- 严重性: P0/P1/P2/P3
- 影响: [描述对用户的影响]

## 时间线
| 时间 (UTC) | 事件 |
|------------|------|
| HH:MM | 开始 |
| HH:MM | 告警触发 |
| HH:MM | On-call确认 |
| HH:MM | 开始处理 |
| HH:MM | 恢复 |

## 根因分析
[5-Why分析]

## 修复措施
- [ ] 短期: 
- [ ] 长期: 

## 预防措施
- [ ] 监控告警改进
- [ ] 自动化恢复
- [ ] 流程改进

## 经验教训
- 
```
