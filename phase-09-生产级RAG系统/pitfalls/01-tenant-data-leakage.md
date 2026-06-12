# Pitfall 01: 跨租户数据泄露 (Cross-Tenant Data Leakage)

## 症状 (Symptoms)

- 用户A的查询返回了属于用户B的文档片段
- API响应中包含其他租户的chunk_id或元数据
- 相同查询在不同租户间返回相同结果（即使知识库不同）
- 数据隔离审计发现跨租户的向量检索结果

## 根本原因 (Root Cause)

多租户隔离不完整，存在以下漏洞：

1. **集合级别未隔离**: 所有租户共享同一个向量集合，仅依赖元数据过滤
2. **元数据过滤遗漏**: 查询时未强制带tenant_id过滤条件
3. **API Key验证不严格**: 未在每个请求前验证API Key有效性
4. **缓存跨租户**: L3检索结果缓存未区分租户，相同查询返回了其他租户的缓存结果

## 真实场景 (Real Scenario)

某SaaS公司部署了RAG系统供多个企业客户使用。由于开发阶段仅实现了元数据级别的tenant_id过滤，未创建独立集合，导致一次代码变更中查询过滤器被意外移除。结果：
- 5个客户的敏感文档被其他客户检索到
- 涉及GDPR违规
- 客户信任严重受损

## 完整解决方案 (Complete Solution)

```python
def secure_multitenant_search(tenant_id: str, api_key: str, query_vector: list, top_k: int = 10) -> tuple[bool, list, str]:
    """
    安全的多租户检索 - 强制三层隔离
    """
    # 第3层: API Key 认证（每个请求强制验证）
    if not verify_api_key(api_key, tenant_id):
        return False, [], "API Key认证失败"
    
    # 第2层: 强制tenant_id过滤（代码层面强制注入）
    mandatory_filter = {"tenant_id": {"$eq": tenant_id}}
    
    # 第1层: 在租户专属集合中搜索（物理隔离）
    collection = get_tenant_collection(tenant_id)  # 获取租户专属集合
    if not collection:
        return False, [], "租户集合不存在"
    
    # 执行搜索 - 集合级别隔离 + 元数据过滤双重保险
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=mandatory_filter,  # 第2层
    )
    
    # 验证所有结果确实属于该租户
    validated_results = []
    for r in results:
        if r.get("metadata", {}).get("tenant_id") == tenant_id:
            # 清理元数据中的敏感字段
            r["metadata"] = sanitize_metadata(r["metadata"])
            validated_results.append(r)
    
    # 审计日志
    audit_log(tenant_id, "search", len(validated_results))
    
    return True, validated_results, "OK"


def sanitize_metadata(metadata: dict) -> dict:
    """清理可能泄露其他租户信息的字段"""
    allowed_keys = {"source", "chunk_index", "language", "version", "tenant_id"}
    return {k: v for k, v in metadata.items() if k in allowed_keys}


def penetration_test(tenant_manager) -> list[str]:
    """
    穿透测试: 尝试跨租户数据访问
    Returns: 发现的安全漏洞列表
    """
    vulnerabilities = []
    
    # 测试1: 伪造tenant_id
    try:
        results = search_with_fake_tenant_id()
        if results:
            vulnerabilities.append("伪造tenant_id可获取其他租户数据")
    except:
        pass
    
    # 测试2: 空元数据过滤
    try:
        results = search_without_tenant_filter()
        if results:
            vulnerabilities.append("无tenant_id过滤可获取全量数据")
    except:
        pass
    
    # 测试3: 无效API Key
    try:
        results = search_with_invalid_api_key()
        if results:
            vulnerabilities.append("无效API Key仍可获取数据")
    except:
        pass
    
    # 测试4: 缓存跨租户
    try:
        results = cache_poisoning_test()
        if results:
            vulnerabilities.append("缓存未隔离导致跨租户数据泄露")
    except:
        pass
    
    return vulnerabilities
```

## 检查清单 (Checklist)

- [ ] 每个租户使用独立向量集合（Chromadb Collection / Milvus Partition）
- [ ] 所有查询强制注入tenant_id过滤条件
- [ ] API Key在每个请求中验证（不缓存验证结果超过1秒）
- [ ] 缓存键包含tenant_id（L1-L4所有层级）
- [ ] 元数据返回前进行安全清理
- [ ] 定期执行穿透测试（至少每月一次）
- [ ] 审计日志记录所有跨租户访问尝试
- [ ] 实现配额隔离（一个租户的超限不影响其他租户）
