# 🔧 环境配置指南

> 在开始任何 Phase 之前，请确保你的开发环境已正确配置。

---

## 1. 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4 核 | 8 核+ |
| 内存 | 16 GB | 32 GB+ |
| 硬盘 | 50 GB 可用 | 100 GB+ SSD |
| GPU（可选） | 无 | NVIDIA 8GB+ VRAM（用于本地模型） |
| 网络 | 稳定互联网 | 低延迟（API 调用） |

---

## 2. 软件安装

### 2.1 必装软件

```bash
# Python 3.10+
python --version  # 应显示 3.10 或更高

# Git
git --version

# pip 升级
pip install --upgrade pip
```

### 2.2 API 密钥获取

| 服务 | 注册地址 | 用途 |
|------|---------|------|
| OpenAI | https://platform.openai.com | GPT-4o API、Embedding API |
| Anthropic | https://console.anthropic.com | Claude API、Prompt Caching |
| Cohere | https://dashboard.cohere.com | Rerank API（可选） |
| Voyage | https://www.voyageai.com | Embedding API（可选） |

### 2.3 环境变量配置

创建 `.env` 文件（**切勿提交到 Git！**）：

```bash
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2.4 推荐 IDE

- **VS Code** + Python 扩展 + Jupyter 扩展
- **Cursor**（AI 辅助编码）
- **PyCharm Professional**

---

## 3. Python 虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活（Windows）
venv\Scripts\activate

# 激活（Mac/Linux）
source venv/bin/activate

# 安装核心依赖
pip install openai anthropic python-dotenv pydantic pydantic-settings
pip install tiktoken jinja2 httpx
pip install numpy pandas scikit-learn
pip install jupyter notebook ipykernel
```

---

## 4. 本地模型环境（可选但推荐）

### 4.1 Ollama 安装

```bash
# Windows/Mac: 从 https://ollama.com 下载安装包
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# 拉取常用模型
ollama pull qwen2.5:7b        # 中文推荐
ollama pull llama3:8b          # 英文通用
ollama pull nomic-embed-text   # 本地 Embedding

# 验证
ollama run qwen2.5:7b "你好，请介绍你自己"
```

### 4.2 Docker 安装（Phase 05+ 需要）

```bash
# 从 https://www.docker.com/products/docker-desktop 下载
# 验证
docker --version
docker compose version

# 启动常用服务
docker compose up -d chroma postgres redis
```

---

## 5. 项目结构初始化

```bash
# 克隆/创建项目目录
mkdir -p ~/ai-learning
cd ~/ai-learning

# 创建 .gitignore
cat > .gitignore << EOF
.env
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
*.gguf
data/
EOF

# 初始化 Git（可选）
git init
git add .
git commit -m "init: AI learning environment"
```

---

## 6. 验证安装

```python
# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

# 检查 API 密钥
assert os.getenv("OPENAI_API_KEY"), "❌ OPENAI_API_KEY 未设置！"
print("✅ OPENAI_API_KEY 已配置")

# 测试 OpenAI 调用
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=10
)
print(f"✅ OpenAI API 正常: {response.choices[0].message.content}")

# 测试 Anthropic 调用
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=10,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(f"✅ Anthropic API 正常: {response.content[0].text}")

print("\n🎉 环境配置完成！可以开始 Phase 01 了。")
```

---

## 7. 各阶段额外依赖

| Phase | 额外安装 |
|-------|---------|
| Phase 03 | `pip install chromadb pypdf2 python-docx beautifulsoup4 pytesseract` |
| Phase 04 | `pip install rank-bm25 sentence-transformers FlagEmbedding` |
| Phase 05 | `pip install langchain langchain-core langchain-openai langchain-community` |
| Phase 06-08 | `pip install langgraph langgraph-checkpoint-postgres redis` |
| Phase 09 | `pip install qdrant-client milvus pymilvus pika celery` |
| Phase 10 | `pip install ragas deepeval trulens-eval prometheus-client` |
| Track A | `pip install openai-whisper pillow clip-by-openai` |
| Track B | `pip install llama-factory datasets peft bitsandbytes` |

---

## ⚠️ 常见问题

### Q: API 调用返回 429 错误？
**A**: 速率限制。检查你的 API 使用额度，实现指数退避重试。

### Q: `CUDA out of memory`？
**A**: 使用 QLoRA 4-bit 量化，减小 batch_size，或使用 Colab 免费 GPU。

### Q: Ollama 模型下载太慢？
**A**: 使用国内镜像或手动下载 GGUF 文件，放到 Ollama models 目录。

### Q: `.env` 文件不生效？
**A**: 确保 `python-dotenv` 已安装，且在脚本开头调用 `load_dotenv()`。

---

准备好后，前往 [Phase 01：环境与基础](../phase-01-环境与基础/) 开始学习！
