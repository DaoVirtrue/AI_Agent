# 面试准备 - 50道核心面试题

## RAG (15题)

**Q1: 什么是RAG？它与传统LLM使用方式有什么区别？**
A: RAG(Retrieval-Augmented Generation)是一种架构模式，在LLM生成回答前先从外部知识库检索相关文档，然后基于检索结果生成回答。区别：传统LLM仅依赖训练时学到的知识，RAG结合了实时检索的能力，可以有效减少幻觉、支持知识更新、提供可溯源性。

**Q2: RAG的核心组件有哪些？**
A: 1)文档加载器 2)文本分割器 3)嵌入模型 4)向量数据库 5)检索器 6)生成器/LLM。可选：重排序器、查询改写器。

**Q3: 如何评估RAG系统的质量？**
A: 多维度评估：检索质量(Recall@K, MRR, NDCG)、生成质量(Faithfulness, Relevance)、系统效率(Latency, Cost)、用户体验。常用框架：RAGAS, trulens-eval。 (Phase 10)

**Q4-15**: 涵盖检索策略、嵌入模型选择、分块策略、高级RAG技术、混合检索、多模态RAG、生产部署、缓存策略、成本优化等。

---

## Agent (15题)

**Q16: 什么是AI Agent？与普通LLM调用的区别？**
A: Agent是能够自主感知环境、做出决策、使用工具、规划并执行多步任务的AI系统。区别：LLM调用是单次请求-响应；Agent具有循环(感知-思考-行动)、工具使用、记忆和规划能力。

**Q17: LangGraph的核心概念是什么？**
A: LangGraph是有状态的图框架。核心概念：State(状态)是图的持久化数据结构；Nodes是处理步骤(LLM, Tool)；Edges定义了流程但不限制数据流；Graph是有向循环图支持Agent循环。(Phase 08)

**Q18-30**: 涵盖Agent架构、工具定义、ReAct模式、Plan-and-Execute、Multi-Agent、Human-in-the-loop、记忆管理、错误恢复、Agent评估、安全考虑等。

---

## Prompt Engineering (10题)

**Q31: 什么是最佳提示词设计原则？**
A: 1)明确角色和任务 2)提供格式要求 3)使用示例(Few-shot) 4)分步骤引导 5)给出约束条件 6)要求推理过程(Chain-of-Thought)。(Phase 02)

**Q32-40**: 涵盖Few-shot策略、CoT变体、提示词注入防御、提示词版本管理、A/B测试提示词、多语言提示、角色设定、安全约束、提示词模板化、动态提示构建。

---

## LangChain/LangGraph (10题)

**Q41: LangChain的核心抽象是什么？**
A: 1)Prompt Templates 2)LLMs/Chat Models 3)Chains (LCEL) 4)Retrievers 5)Tools 6)Agents 7)Memory。(Phase 03)

**Q42: LCEL (LangChain Expression Language) 有什么优势？**
A: LCEL使用 `|` 管道操作符组合Runnable组件。优势：声明式、自动并行化、流式支持、中间结果访问、输入输出schema验证。

**Q43-50**: 涵盖Chain vs Agent、Memory类型、流式处理、回调系统、自定义工具、LangServe部署、LangSmith追踪、性能优化、错误处理、生产最佳实践。

---

## 系统设计 (3道附加题)

**SD1: 设计一个企业级RAG系统**
需求：100万文档、1000并发用户、99.9%可用性、多租户隔离、P95<2s
考虑：架构分层、技术选型、扩展策略、故障恢复、监控告警

**SD2: 设计一个Multi-Agent客服系统**
考虑：Agent分解、协作机制、人机交互、知识管理、SLA保证

**SD3: 如何保证AI系统的安全性？**
考虑：四层防御(输入/模型/工具/输出)、RBAC+ABAC、审计日志、提示词注入防御、数据脱敏

---

## 常见追问

1. "详细解释你项目中的缓存层次设计"
2. "如何处理RAG检索返回不相关文档的情况？"
3. "Agent中了死循环怎么办？如何检测和恢复？"
4. "如何评估一个提示词写得好不好？"
5. "数据隔离在多租户系统中如何实现？"

## 课程阶段参考

| 面试领域 | 课程阶段 |
|---------|---------|
| RAG基础 | Phase 04 |
| RAG高级 | Phase 06 |
| RAG评估 | Phase 10 |
| RAG生产 | Phase 09 |
| Prompt | Phase 02 |
| LangChain | Phase 03 |
| Agent | Phase 08 |
| 安全 | Phase 11 |
| 部署 | Phase 12 |
