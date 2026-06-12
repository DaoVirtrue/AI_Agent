# Phase 12: 毕业项目 (Capstone Integration)

## 周期: 3-4周

## 核心理念

将Phase 01-11的所有知识整合为一个完整的、企业级的生产就绪系统。这不是另一个练习 -- 这是你在前面11个阶段所学一切的终极合成。

## 学习目标

1. 设计完整的系统架构文档
2. 构建企业级RAG Agent系统 (~2000行)
3. 编写生产部署指南
4. 执行负载测试和瓶颈分析
5. 实施混沌工程和故障注入
6. 编写生产运维手册
7. 为技术面试做好准备

## 文件清单 (7个主要文件 + pitfalls + checkpoint)

| 文件 | 描述 | 行数 |
|------|------|------|
| 01-system-architecture.md | 完整架构文档 (系统上下文图/容器图/部署图/数据流/安全覆盖) | ~500 |
| 02-enterprise-rag-agent-system.py | 终极综合系统 (多租户/LangGraph/4级缓存/熔断器/评估/安全/技能) | ~2000 |
| 03-deployment-guide.md | Docker Compose + 环境变量 + 健康检查 + TLS + 备份 + Prometheus | ~300 |
| 04-load-testing.py | Locust脚本 + 4种用户场景 + 100并发目标 | ~300 |
| 05-failure-injection-testing.py | 混沌工程 (杀PostgreSQL/Redis/RabbitMQ/LLM API/网络分区) | ~300 |
| 06-production-runbook.md | 事件分级(P0-P4) + Runbooks + 告警映射 + 事后分析模板 | ~300 |
| 07-interview-preparation.md | 50道面试题 (RAG 15 + Agent 15 + Prompt 10 + LangChain/LangGraph 10) | ~300 |

## 前置要求

- 所有 Phase 01-11
- 具备独立设计和实现的能力
- 理解分布式系统概念

## 毕业标准

- 系统架构文档可以通过架构评审
- 综合系统代码可运行且覆盖所有关键功能
- 部署配置可在Docker环境中启动
- 负载测试达到100并发 P95<2s目标
- 故障注入测试覆盖5种故障场景
- 运维手册可直接用于值班
- 面试准备覆盖50个核心知识点

## 里程碑

**Week 1-2**: 架构设计 + 综合系统实现
**Week 3**: 部署 + 测试 + 运维
**Week 4**: 面试准备 + 最终审查
