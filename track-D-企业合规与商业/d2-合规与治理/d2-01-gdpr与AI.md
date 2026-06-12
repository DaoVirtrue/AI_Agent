# D2-01：GDPR 与 AI —— RAG 系统的 GDPR 合规指南

> **核心问题**：RAG 系统存储了大量用户数据和文档，如何满足 GDPR 要求？
> **适用场景**：处理欧盟用户数据的任何 AI 系统

---

## 一、GDPR 与 RAG 的交叉点

### 1.1 核心权利与 RAG 影响

| GDPR 条款 | 权利 | 对 RAG 的影响 |
|-----------|------|--------------|
| **第17条** | 被遗忘权（Right to Erasure） | 必须能从向量索引中删除用户数据 |
| **第15条** | 访问权（Right of Access） | 用户可查询系统存储了哪些个人信息 |
| **第20条** | 数据可携带权 | 需要提供结构化数据导出 |
| **第5条** | 数据最小化 | 只索引必要的信息，不过度收集 |
| **第25条** | 数据保护设计（PbD） | 从架构设计就考虑隐私 |
| **第35条** | DPIA（数据保护影响评估） | AI 系统上线前必须完成 DPIA |
| **第44-49条** | 跨境数据传输 | 数据存储和处理的地理位置约束 |

### 1.2 RAG 系统中的个人数据位置

```
RAG 系统中可能存储个人数据的地方：

┌─────────────────────────────────────────┐
│ 1. 原始文档库                           │
│    - 上传的 PDF/DOCX 含姓名/邮箱/地址    │
│    - 聊天记录含用户个人信息              │
│                                          │
│ 2. 向量索引                             │
│    - 文本分块后的向量表示                │
│    - 可能包含个人数据的嵌入              │
│    ⚠️ 向量本身不能直接"删除"，需重建    │
│                                          │
│ 3. 对话历史                             │
│    - 用户问题历史                        │
│    - AI 回答历史                        │
│    - 会话元数据                          │
│                                          │
│ 4. 日志和监控                           │
│    - API 调用日志                       │
│    - 错误日志中含用户输入               │
│    - 审计日志                            │
└─────────────────────────────────────────┘
```

---

## 二、被遗忘权的技术实现

### 2.1 删除流程

```
用户请求删除数据 →
  │
  ├── 步骤1：定位数据
  │   在所有存储层中搜索用户标识：
  │   - 元数据库（PostgreSQL/MySQL）：用户ID → 关联的文档ID
  │   - 文档存储（MinIO/S3）：文件名 → 用户ID
  │   - 对话存储：session_id → user_id → 历史消息
  │   - 日志系统：user_id → 日志条目
  │
  ├── 步骤2：删除原始数据
  │   - 删除原始文档文件
  │   - 删除对话历史
  │   - 删除用户元数据
  │
  ├── 步骤3：重建向量索引（关键难点）
  │   向量索引不像数据库，不能"删除一条记录"后自动清理。
  │   方案A：标记删除 + 定期重建（适合小规模）
  │   方案B：增量删除（Milvus/Qdrant 支持 delete by filter）
  │   方案C：维护独立的用户专属索引（物理隔离）
  │
  ├── 步骤4：日志清理
  │   - 日志中的用户数据脱敏或删除
  │   - 审计日志保留但匿名化
  │
  └── 步骤5：确认完成
      向用户发送删除确认
      保留删除记录（但不含数据内容）
```

### 2.2 向量索引删除的实现

```python
# 方案A：标记删除 + 定期清理

class GDPRCompliantVectorStore:
    """
    GDPR 合规的向量存储封装。

    关键设计：
    1. 每个文档关联 user_id（用于定位）
    2. 删除时标记 deleted=True + 立即从活跃索引移除
    3. 定期后台完全清理已标记的向量
    """

    def __init__(self, vector_db, metadata_db):
        self.vector_db = vector_db   # Qdrant / Milvus
        self.metadata_db = metadata_db  # PostgreSQL

    def delete_user_data(self, user_id: str) -> dict:
        """根据 GDPR 第17条删除用户数据。"""
        # 1. 找到该用户的所有文档
        docs = self.metadata_db.query(
            "SELECT doc_id, vector_ids FROM documents WHERE user_id = ?",
            [user_id]
        )

        deleted_count = 0
        for doc in docs:
            # 2. 从向量库中删除向量
            for vec_id in doc["vector_ids"]:
                self.vector_db.delete(vector_id=vec_id)
                deleted_count += 1

            # 3. 删除原始文档
            self.metadata_db.execute(
                "UPDATE documents SET deleted=1, deleted_at=NOW() "
                "WHERE doc_id=?", [doc["doc_id"]]
            )

        # 4. 删除对话历史
        convs_deleted = self.metadata_db.execute(
            "DELETE FROM conversations WHERE user_id=?", [user_id]
        )

        # 5. 匿名化日志
        self.metadata_db.execute(
            "UPDATE audit_logs SET user_id='ANONYMIZED', "
            "query_text='[REDACTED]' WHERE user_id=?", [user_id]
        )

        return {
            "vectors_deleted": deleted_count,
            "documents_marked": len(docs),
            "conversations_deleted": convs_deleted,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }


# 方案B：独立用户索引（物理隔离）

class TenantIsolatedVectorStore:
    """
    每个用户/租户独立的向量集合。

    优点：删除时直接 drop collection，彻底干净
    缺点：集合数量过多影响性能（适用于 B2B 场景）
    """

    def create_user_collection(self, user_id: str):
        collection_name = f"user_{user_id}_vectors"
        self.vector_db.create_collection(collection_name)
        return collection_name

    def delete_user_collection(self, user_id: str):
        collection_name = f"user_{user_id}_vectors"
        self.vector_db.drop_collection(collection_name)
        # 物理删除，无残留
```

---

## 三、数据最小化原则

### 3.1 实践指南

```yaml
RAG 系统的数据最小化检查清单：

文档上传阶段：
  - [ ] 上传前自动扫描并脱敏 PII（姓名/邮箱/电话/身份证号）
  - [ ] 限制文档大小和数量（"只上传回答问题所需的"）
  - [ ] 设置文档保留期限（如 90 天自动归档）

索引阶段：
  - [ ] 只对正文内容做嵌入（排除页眉页脚水印）
  - [ ] 高敏感字段不做向量化（如身份证号只在元数据中加密存储）
  - [ ] 设置索引的过期时间

对话阶段：
  - [ ] 对话历史保留期限可配置（默认 30 天）
  - [ ] 会话超时自动清理
  - [ ] 用户可主动清除对话历史

监控日志：
  - [ ] 查询日志中不记录完整问题文本（只记录长度和分类）
  - [ ] 日志保留期限设置（如 90 天轮转）
  - [ ] 错误日志中自动脱敏用户输入
```

### 3.2 PII 自动脱敏示例

```python
import re

def anonymize_pii(text: str) -> tuple[str, dict]:
    """
    自动检测并脱敏个人身份信息（PII）。

    Returns:
        (脱敏后的文本, 脱敏记录)
    """
    replacements = {}

    # 邮箱
    text = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '[EMAIL]', text
    )

    # 中国手机号
    text = re.sub(r'1[3-9]\d{9}', '[PHONE]', text)

    # 身份证号
    text = re.sub(r'\d{17}[\dXx]', '[ID_NUMBER]', text)

    # IP 地址
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP_ADDR]', text)

    return text, replacements
```

---

## 四、DPIA 模板（数据保护影响评估）

### DPIA 必须包含的内容

```yaml
# 数据保护影响评估（DPIA）

项目名称: [填写项目名称]
评估日期: [YYYY-MM-DD]
评估人: [姓名/职位]
版本: [1.0]

1. 项目概述:
   描述: [简要描述 AI 系统的功能和目的]
   数据流: [数据从哪来 → 如何处理 → 存储在哪 → 谁可以访问]

2. 数据处理活动:
   收集的数据类型:
     - 用户输入文本（含潜在个人信息）
     - 上传的文档（含潜在个人信息）
     - 使用日志和元数据

   处理目的: [说明为什么需要这些数据]
   法律依据: [GDPR第6条/同意/合同履行/合法利益]
   数据最小化措施: [已采取的最小化措施]

3. 风险评估:
   风险1: 向量索引删除不彻底导致"被遗忘权"不完整
     可能性: 中 | 影响: 高
     缓解措施: 实现标记删除+定期重建机制

   风险2: LLM 可能记忆并泄露训练数据中的个人信息
     可能性: 低 | 影响: 高
     缓解措施: 使用 API 模型（不用于训练），私有化部署

   风险3: 日志中记录完整用户问题（含敏感信息）
     可能性: 高 | 影响: 中
     缓解措施: 日志脱敏、短期保留、访问控制

   风险4: 跨境数据传输到 EU 以外的 LLM 供应商
     可能性: 高 | 影响: 高
     缓解措施: 选择 EU 区域部署的 API / 本地化 LLM

4. 缓解措施汇总:
   [列出所有已实施和计划实施的缓解措施]

5. 签署:
   数据保护官 (DPO): ___________ 日期: _______
   技术负责人: ___________ 日期: _______
```

---

## 五、跨境数据传输合规

```
如果你的 RAG 系统使用以下 LLM API，需评估跨境传输：

模型提供商    服务器位置     是否涉及跨境传输    合规建议
─────────────────────────────────────────────────────
OpenAI        美国/全球      是（中国→美国）      签署 SCC + 加密传输
Anthropic     美国           是（中国→美国）      签署 SCC + 加密传输
Google        全球多区域     可配置（选 EU 区域）  选择 EU 区域部署
DeepSeek      中国           否（中国境内）       天然合规
Qwen          中国           否（中国境内）       天然合规

应对策略：
- 优先选择数据存储在本国的供应商
- 如需跨境，签署标准合同条款（SCC）
- 数据传输前进行加密和脱敏
- 在 DPIA 中明确记录跨境传输路径
```

---

## 六、合规检查清单

```
上线前自查：

□ 是否完成 DPIA？
□ 是否实现数据删除能力（"被遗忘权"）？
□ 是否实现数据导出能力（"可携带权"）？
□ 是否有隐私政策页面（告知用户数据用途）？
□ 是否有 Cookie/追踪同意机制？
□ 是否与第三方（LLM供应商）签署数据处理协议（DPA）？
□ 日志和监控数据是否脱敏？
□ 是否有数据泄露通知流程（72小时内）？
□ 是否任命了数据保护官（DPO）（如需要）？
□ 是否对员工进行了 GDPR 培训？
```
