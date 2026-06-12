# C2-03：RAGFlow 部署与 API 集成指南

> **适用读者**：需要在生产环境部署 RAGFlow 的 DevOps 工程师和后端开发者
> **前置要求**：Docker 基础知识

---

## 一、部署架构概览

```
┌────────────────────────────────────────────────────────┐
│                    RAGFlow 部署架构                     │
│                                                         │
│  用户/应用                                               │
│    │                                                    │
│    ▼                                                    │
│  ┌─────────────┐    ┌──────────────┐                   │
│  │   Nginx      │    │  API Gateway │                   │
│  │ (LB + 静态)  │    │  (Kong/Traefik)                  │
│  └─────┬───────┘    └──────┬───────┘                   │
│        │                   │                            │
│        └─────────┬─────────┘                            │
│                  │                                      │
│        ┌─────────▼─────────┐                            │
│        │   RAGFlow Core    │                            │
│        │   (Flask API)     │                            │
│        └───┬───────┬───────┘                            │
│            │       │                                    │
│     ┌──────▼──┐ ┌──▼──────┐ ┌──────────┐               │
│     │ MySQL   │ │ Redis   │ │ MinIO/S3 │               │
│     │ (元数据)│ │ (缓存)  │ │ (文件)   │               │
│     └─────────┘ └─────────┘ └──────────┘               │
│            │                                            │
│     ┌──────▼──────────┐                                 │
│     │ Elasticsearch   │                                 │
│     │ (全文索引)       │                                 │
│     └─────────────────┘                                 │
│            │                                            │
│     ┌──────▼──────────┐                                 │
│     │ 向量数据库       │                                 │
│     │ (Milvus/Qdrant) │                                 │
│     └─────────────────┘                                 │
│            │                                            │
│     ┌──────▼──────────┐                                 │
│     │ Neo4j (可选)     │                                 │
│     │ 知识图谱         │                                 │
│     └─────────────────┘                                 │
└────────────────────────────────────────────────────────┘
```

---

## 二、快速部署（Docker Compose）

### 2.1 一键部署（开发/测试环境）

```bash
# 克隆仓库
git clone https://github.com/infiniflow/ragflow.git
cd ragflow

# 启动所有服务（含 MySQL、Redis、ES、MinIO）
docker compose -f docker/docker-compose.yml up -d

# 等待服务就绪（约 2-3 分钟）
docker compose -f docker/docker-compose.yml ps

# 访问 Web UI
# http://localhost:80
# 默认账号: admin
# 默认密码: 首次启动时设置
```

### 2.2 生产环境部署

```yaml
# docker-compose-prod.yml
version: '3.8'

services:
  ragflow:
    image: infiniflow/ragflow:v0.14.0
    restart: always
    ports:
      - "9380:80"
    environment:
      - TZ=Asia/Shanghai
      - MYSQL_HOST=mysql-prod
      - MYSQL_PORT=3306
      - MYSQL_USER=ragflow
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - MYSQL_DB=ragflow
      - REDIS_HOST=redis-prod
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - ES_HOST=es-prod
      - ES_PORT=9200
      - MINIO_HOST=minio-prod
      - MINIO_PORT=9000
      - MINIO_USER=admin
      - MINIO_PASSWORD=${MINIO_PASSWORD}
      - LLM_MODEL=gpt-4o
      - EMBEDDING_MODEL=BAAI/bge-m3
    volumes:
      - ragflow_data:/ragflow/data
      - /etc/localtime:/etc/localtime:ro
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '2'
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 外部化数据库（使用已有的高可用集群）
  # MySQL、Redis、Elasticsearch、MinIO 使用外部服务
```

### 2.3 环境变量参考

| 变量名 | 说明 | 默认值 | 生产建议 |
|--------|------|--------|----------|
| `LLM_MODEL` | 默认大语言模型 | gpt-4o-mini | gpt-4o / claude-sonnet-4-20250514 |
| `EMBEDDING_MODEL` | 默认嵌入模型 | BAAI/bge-large-zh-v1.5 | BAAI/bge-m3 |
| `ASR_MODEL` | 语音识别模型 | Whisper | Whisper-large-v3 |
| `IMAGE2TEXT_MODEL` | 图像识别模型 | Qwen-VL-Chat | GPT-4o |
| `CHUNK_SIZE` | 默认分块大小 | 512 | 按文档类型设置 |
| `MAX_TOKENS` | 最大 token 数 | 8192 | >= 32000 |
| `REQUEST_TIMEOUT` | 请求超时（秒） | 300 | 600 |

---

## 三、REST API 集成指南

### 3.1 认证

```python
import requests

BASE_URL = "http://localhost:9380/api/v1"
API_KEY = "your-api-key-here"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
```

### 3.2 数据集管理

```python
# 创建数据集
def create_dataset(name: str, chunk_method: str = "naive") -> str:
    resp = requests.post(
        f"{BASE_URL}/datasets",
        headers=headers,
        json={
            "name": name,
            "chunk_method": chunk_method,
            "description": "项目知识库",
            "embedding_model": "BAAI/bge-m3",
        }
    )
    return resp.json()["data"]["id"]

# 列出数据集
def list_datasets():
    resp = requests.get(f"{BASE_URL}/datasets", headers=headers)
    return resp.json()["data"]

# 删除数据集
def delete_dataset(dataset_id: str):
    resp = requests.delete(f"{BASE_URL}/datasets/{dataset_id}", headers=headers)
    return resp.json()
```

### 3.3 文档上传与管理

```python
# 上传文档
def upload_document(dataset_id: str, file_path: str):
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/datasets/{dataset_id}/documents",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": f},
        )
    return resp.json()["data"]["id"]

# 查询文档解析状态
def get_document_status(dataset_id: str, doc_id: str):
    resp = requests.get(
        f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}",
        headers=headers,
    )
    data = resp.json()["data"]
    return data["status"]  # UNSTART / RUNNING / DONE / FAIL

# 重新解析文档
def reparse_document(dataset_id: str, doc_id: str):
    resp = requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}/chunks",
        headers=headers,
    )
    return resp.json()
```

### 3.4 检索 API

```python
# 单数据集检索
def search(dataset_ids: list, query: str, top_k: int = 5):
    resp = requests.post(
        f"{BASE_URL}/retrieval",
        headers=headers,
        json={
            "question": query,
            "dataset_ids": dataset_ids,
            "top_k": top_k,
            "similarity_threshold": 0.6,
        }
    )
    return resp.json()["data"]["chunks"]

# 对话式问答
def chat(dataset_ids: list, question: str, conversation_id: str = None):
    resp = requests.post(
        f"{BASE_URL}/chat",
        headers=headers,
        json={
            "question": question,
            "dataset_ids": dataset_ids,
            "conversation_id": conversation_id,
            "stream": False,
        }
    )
    return resp.json()["data"]

# 流式对话
def chat_stream(dataset_ids: list, question: str):
    resp = requests.post(
        f"{BASE_URL}/chat",
        headers=headers,
        json={
            "question": question,
            "dataset_ids": dataset_ids,
            "stream": True,
        },
        stream=True,
    )
    for line in resp.iter_lines():
        if line:
            yield json.loads(line.decode("utf-8").replace("data:", ""))
```

### 3.5 Python SDK（推荐）

```python
# 推荐使用官方 Python SDK，比直接调 API 更方便
from ragflow import RAGFlow

client = RAGFlow(
    api_key="your-api-key",
    base_url="http://localhost:9380",
)

# 创建数据集
dataset = client.create_dataset(
    name="项目知识库",
    chunk_method="knowledge_graph",
)

# 上传文档
dataset.upload_documents([
    "./docs/架构设计.pdf",
    "./docs/API文档.docx",
])

# 等待解析完成
import time
while True:
    status = dataset.get_status()
    if all(d["status"] == "DONE" for d in status):
        break
    time.sleep(5)

# 检索
chunks = dataset.search("系统的技术架构是什么？", top_k=5)

# 问答
response = client.chat(
    dataset_ids=[dataset.id],
    question="总结系统的技术架构",
)

print(response["answer"])
```

---

## 四、性能调优

### 4.1 分块策略选择

| 文档类型 | 推荐 chunk_method | 说明 |
|----------|-------------------|------|
| 技术文档/论文 | `paper` | 保留摘要、章节、引用 |
| 法律文本 | `laws` | 保留条款编号 |
| FAQ | `qa` | 一问一答式分块 |
| 技术手册/书籍 | `book` | 保留章节层次 |
| 含大量表格的文档 | `table` | 单独处理表格 |
| PPT | `presentation` | 按幻灯片分块 |
| 通用文本 | `knowledge_graph` | 实体关系增强 |

### 4.2 检索参数优化

```python
# 影响检索效果的关键参数
search_params = {
    "top_k": 5,                    # 返回结果数（建议 3-10）
    "similarity_threshold": 0.6,   # 相似度阈值（建议 0.5-0.8）
    "vector_similarity_weight": 0.7,  # 向量相似度权重（0.3-0.7）
    "keywords_similarity_weight": 0.3, # 关键词权重（0.3-0.7）
    "top_n": 30,                   # 粗筛阶段召回数（20-50）
}
```

### 4.3 LLM 参数配置

```python
# 在 RAGFlow Web UI 或 API 中配置
llm_config = {
    "model_name": "gpt-4o",
    "temperature": 0.1,       # 低温度减少幻觉（0-0.3）
    "top_p": 0.9,
    "max_tokens": 2048,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "system_prompt": """你是一个专业的技术助手。
请严格基于提供的文档内容回答问题。
如果文档中没有相关信息，请明确说明"根据现有文档，无法回答"。
所有引用必须标注来源文件名和段落位置。""",
}
```

---

## 五、监控与运维

### 5.1 健康检查

```bash
# 健康检查接口
curl http://localhost:9380/api/health

# 响应示例
{
  "code": 0,
  "data": {
    "mysql": "connected",
    "redis": "connected",
    "elasticsearch": "connected",
    "minio": "connected",
    "version": "v0.14.0"
  }
}
```

### 5.2 关键指标监控

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| 文档解析成功率 | DONE/(DONE+FAIL) | < 95% |
| 检索平均延迟 | P50 | > 500ms |
| 检索 P99 延迟 | 99% 请求的延迟 | > 2s |
| 问答延迟 | 端到端响应时间 | > 5s |
| API 错误率 | 5xx 响应比例 | > 1% |
| 向量数据库连接 | 连接状态 | 断开 |

### 5.3 日志查看

```bash
# Docker 环境
docker compose logs -f ragflow

# 重点关注的日志
docker compose logs ragflow | grep -E "ERROR|WARN|timeout"
```

---

## 六、安全加固清单

- [ ] 修改默认管理员密码
- [ ] 启用 HTTPS（反向代理终止 TLS）
- [ ] API Key 轮换机制
- [ ] 数据库密码使用环境变量/Secret Manager
- [ ] 网络隔离（RAGFlow 不直接暴露公网，通过 API Gateway 访问）
- [ ] 文档访问权限控制（多租户场景）
- [ ] 审计日志开启（记录所有查询和操作）
- [ ] 定期备份数据库和向量索引
- [ ] PII/敏感信息过滤（上传文档前检查）
- [ ] API 速率限制配置
