# 部署指南 (Deployment Guide)

## 1. Docker Compose 配置

```yaml
# docker-compose.yml
version: '3.8'
services:
  api-gateway:
    image: nginx:alpine
    ports: ["443:443"]
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf", "./certs:/etc/nginx/certs"]
    depends_on: [rag-agent]
    restart: unless-stopped

  rag-agent:
    build: .
    ports: ["8000:8000"]
    env_file: [".env"]
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ragdb
      - REDIS_URL=redis://redis:6379
      - MILVUS_HOST=milvus
      - RABBITMQ_URL=amqp://rabbitmq
    depends_on: [postgres, redis, milvus, rabbitmq]
    restart: unless-stopped
    deploy:
      replicas: 3

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes: ["postgres_data:/var/lib/postgresql/data"]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes: ["redis_data:/data"]
    restart: unless-stopped

  milvus:
    image: milvusdb/milvus:latest
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on: [etcd, minio]
    restart: unless-stopped

  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports: ["5672:5672", "15672:15672"]
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    volumes: ["grafana_data:/var/lib/grafana"]
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

## 2. 环境变量

```bash
# .env (不要提交到git!)
# === LLM API Keys ===
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
DEEPSEEK_API_KEY=sk-xxx

# === Database ===
DATABASE_URL=postgresql://user:password@localhost:5432/ragdb

# === Redis ===
REDIS_URL=redis://localhost:6379/0

# === Milvus ===
MILVUS_HOST=localhost
MILVUS_PORT=19530

# === RabbitMQ ===
RABBITMQ_URL=amqp://guest:guest@localhost:5672/

# === Security ===
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret
CANARY_TOKENS=token1,token2,token3

# === Application ===
ENVIRONMENT=production  # production | staging | development
LOG_LEVEL=INFO
ENABLE_LANGSMITH=true
LANGSMITH_API_KEY=ls__xxx
```

## 3. 健康检查

```bash
# /health - 基本健康
curl http://localhost:8000/health
# {"status": "healthy", "version": "1.0.0"}

# /health/readiness - 就绪探针
curl http://localhost:8000/health/readiness
# {"ready": true, "checks": {"db": "ok", "redis": "ok", "milvus": "ok"}}

# /health/liveness - 存活探针
curl http://localhost:8000/health/liveness
# {"alive": true}
```

## 4. 备份计划

| 组件 | 频率 | 保留 |
|------|------|------|
| PostgreSQL | 每日 02:00 | 30天 |
| Milvus | 每周日 03:00 | 4周 |
| Redis | 每小时 | 24小时 |
| 审计日志 | 实时同步到S3 | 365天 |

## 5. Prometheus 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-agent'
    static_configs:
      - targets: ['rag-agent:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

## 6. 部署步骤

```bash
# 1. 克隆仓库
git clone https://github.com/org/enterprise-rag.git

# 2. 配置环境
cp .env.example .env
# 编辑 .env 填入实际的密钥和配置

# 3. 初始化数据库
docker-compose run rag-agent python scripts/init_db.py

# 4. 导入向量数据
docker-compose run rag-agent python scripts/import_vectors.py

# 5. 启动服务
docker-compose up -d

# 6. 验证
curl http://localhost:8000/health
docker-compose logs -f rag-agent

# 7. 导入Grafana Dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana_dashboard.json
```
