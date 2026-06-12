# 坑点 #1：API Key 泄露到 Git 仓库

> "早上 10 点提交的代码，晚上 8 点收到 $5,000 账单——中间只隔了一个 git push。"

---

## 症状

你正在开发一个基于 LLM 的项目，一切进展顺利。然后某天早上醒来，你发现：

1. **安全告警邮件**：OpenAI / Anthropic 发来邮件，标题是 "[ACTION REQUIRED] Your API key has been exposed"
2. **账单飙升**：当月 API 消费从平时的 $50 暴涨到 $3,280
3. **陌生调用**：控制台里出现大量你不认识的请求——有人在用你的 Key 疯狂调用 GPT-4o
4. **Key 被吊销**：服务商自动禁用了你的 Key，导致你的线上服务直接宕机

最可怕的是，你可能根本不知道发生了什么，直到账单日。

---

## 根因

`.env` 文件没有被 `.gitignore` 排除，被 `git add .` 一锅端了。

**一张图说明问题**：

```
git add .          # 把 .env 也加进去了
git commit -m "update"  # .env 进了 commit 历史
git push origin main    # .env 跑到了 GitHub 上
       ↓
   GitHub 上的爬虫机器人 🔍
       ↓
   你的 Key 被挖矿脚本消费 💸
```

GitHub 上每秒钟都有自动化脚本在扫描公开仓库中的 API Key。从你 push 上去到 Key 被盗用，往往不超过 **5 分钟**。

注意：即使是私有仓库，如果团队成员账号被攻破，Key 同样会泄露。**永远不要把密钥提交到 Git**，不管仓库是公开还是私有。

---

## 真实场景：开发者小张的一天

```
08:30  小张到公司，开始写一个 RAG 实验脚本
09:00  在项目根目录创建 .env，写入 OPENAI_API_KEY=sk-proj-abc123...
09:15  写完代码，准备提交
09:16  $ git add .
09:16  $ git commit -m "initial commit: rag prototype"
09:17  $ git push origin main
09:20  小张的 Key 已经出现在 GitHub 的公开仓库中
09:22  一枚爬虫脚本捕获了这个 Key
09:25  攻击者开始用这个 Key 调用 GPT-4o，生成大量垃圾内容
15:00  小张收到 OpenAI 邮件："您的 API 用量异常"
18:30  小张查账单：今日消费 $4,200
19:00  小张哭着给 OpenAI 客服发邮件（通常不会退款）
```

这个故事的数据点：
- Key 暴露后的**平均捕获时间**：2-5 分钟
- 攻击者的**常见用途**：加密货币挖矿提示词生成、批量内容农场、转卖 Key 权限
- 厂商的**退款政策**：大多数情况下不退款（条款中明确写了用户有责任保护密钥安全）
- 2024 年 GitHub 统计：每天约 **10,000** 个新泄露的 API Key 被检测到

---

## 修复方案（3 层防护）

### 第 1 层：`.gitignore` 强制排除

这是最基础但最有效的一层。在项目创建之初就配置好。

```bash
# .gitignore
# === 环境变量 & 密钥 ===
.env
.env.*
*.key
*.pem
*.p12
*.pfx
secrets/
credentials/
service-account.json
*-secret.yaml

# === 本地配置 ===
.vscode/settings.json
.idea/

# === 数据 & 缓存 ===
*.bin
*.pickle
.venv/
__pycache__/
```

**关键点**：`.gitignore` 必须 **在 `git add` 之前** 创建。如果文件已经被 Git 追踪，`.gitignore` 不会生效——你需要先 `git rm --cached`。

### 第 2 层：Pre-commit Hook 自动扫描

使用 `detect-secrets` 或 `gitleaks` 在每次 commit 前自动扫描。

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        exclude: package-lock.json

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.4
    hooks:
      - id: gitleaks
```

初始化 baseline（记录当前已知的"假阳性"，防止每次 commit 都报）：

```bash
# 安装 pre-commit
pip install pre-commit

# 生成 .pre-commit-config.yaml 后安装 hooks
pre-commit install

# 创建 detect-secrets 基线（扫描现有代码，将已知的非敏感内容标记为安全）
detect-secrets scan > .secrets.baseline

# 手动审计基线（逐条检查，标记哪些是真正的密钥）
detect-secrets audit .secrets.baseline
```

**工作流**：

```
$ git commit -m "add feature"
→ pre-commit hook 触发
→ detect-secrets 扫描变更文件
→ 发现 .env 中有疑似 API Key
→ ❌ commit 被阻止！
→ 终端提示：Potential secrets found. Aborting commit.
```

### 第 3 层：`.env.example` 模板

为每个环境变量提供一个不含真实值的模板文件，方便团队成员知道需要配置什么。

```bash
# .env.example
# ============================================
# 项目名称: RAG 实验项目
# 使用方法: cp .env.example .env 然后编辑 .env 填入真实值
# 警告: .env 包含真实密钥，切勿提交到 Git！
# ============================================

# === LLM API Keys ===
# OpenAI API Key - 获取地址: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-key-here

# Anthropic API Key - 获取地址: https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=sk-ant-your-key-here

# === 向量数据库 ===
# Pinecone API Key - 获取地址: https://app.pinecone.io
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_ENVIRONMENT=us-west1-gcp

# === 应用配置 ===
LOG_LEVEL=INFO
MAX_RETRIES=3
MODEL_NAME=gpt-4o-mini

# === 可选配置 ===
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your-langsmith-key-here
```

`.env.example` 应该提交到 Git。它不包含敏感信息，却能让新成员快速上手。

---

## 补救措施（如果已经泄露）

如果你已经不小心把 Key 推到了 GitHub，不要慌，按以下步骤处理：

### 步骤 1：立即轮换密钥（最重要！）

```bash
# 去 OpenAI/Anthropic 控制台
# → API Keys → 找到泄露的 Key → Revoke（吊销）
# → Create new secret key → 复制新的 Key
# → 更新本地 .env
```

**这一步必须最先做**，因为即使你删除了 GitHub 上的提交，爬虫可能已经抓取了你的 Key。吊销旧的 Key 是唯一能阻止盗用的方法。

### 步骤 2：从 Git 历史中彻底删除

**误区**：很多人以为新建一个 commit 删除 `.env` 就够了。错！Git 历史里仍然保留着之前的版本，任何人都能通过 `git log -p` 看到。

**正确做法**：使用 `git filter-branch` 或 `git filter-repo` 从整个提交历史中抹去敏感文件。

```bash
# === 方法 A：git filter-branch（传统方法）===

# 从所有历史中删除 .env 文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送到远程（会改写历史，团队协作时需谨慎）
git push origin --force --all
git push origin --force --tags

# === 方法 B：git filter-repo（推荐，更快的替代方案）===

# 安装
pip install git-filter-repo

# 从历史中删除 .env
git filter-repo --path .env --invert-paths

# 重新添加远程并推送
git remote add origin <your-repo-url>
git push origin --force --all
```

### 步骤 3：清理 GitHub 缓存

GitHub 可能缓存了旧的 commit 内容（即使你 force push 了）。确认方法：

```bash
# 在浏览器中直接访问 commit hash 的 URL
# https://github.com/<user>/<repo>/commit/<old-commit-hash>
# 如果仍然能访问到，联系 GitHub Support 请求清除缓存
```

### 步骤 4：通知相关方

- 如果仓库有协作者，通知他们重新 clone（因为历史已被改写）
- 如果 Key 关联了其他服务（如 Pinecone、AWS），检查那些服务是否有异常使用
- 如果是公司项目，按安全事件流程上报

### 步骤 5：配置 GitHub Push 保护（事后加固）

GitHub 现在提供免费的 Secret Scanning（秘密扫描）功能：

```
Settings → Code security and analysis → Secret scanning → Enable
```

GitHub 会检测 200+ 种常见的密钥格式（包括 OpenAI、Anthropic 的 Key 格式），一旦检测到就会：
1. 自动吊销密钥（与部分厂商合作）
2. 发送邮件告警
3. 在 PR 中阻止合并

---

## 代码示例

### 安全的配置加载方式

```python
# config.py
"""安全的配置加载：永远不会让 API Key 出现在代码或版本控制中。"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 加载 .env 文件（本地开发）
# 生产环境通过环境变量注入，不需要 .env 文件
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # 生产环境：从系统环境变量读取
    load_dotenv()  # 仍然尝试加载，但不强制要求文件存在


def get_api_key(provider: str) -> Optional[str]:
    """安全地获取 API Key，绝不硬编码。

    Args:
        provider: "openai" | "anthropic" | "pinecone"

    Returns:
        API key string, or None if not found

    Raises:
        ValueError: 如果必需的 Key 未配置
    """
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "pinecone": "PINECONE_API_KEY",
    }

    env_var = key_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"未找到 {env_var}。请：\n"
            f"  1. 创建 .env 文件：cp .env.example .env\n"
            f"  2. 编辑 .env 填入你的 {env_var}\n"
            f"  3. 或设置环境变量：export {env_var}=your-key-here"
        )

    # 安全检查：防止使用示例 key
    if "your-key-here" in api_key or "example" in api_key.lower():
        raise ValueError(
            f"检测到 {env_var} 仍为示例值！"
            f" 请在 .env 中填入真实的 API Key。"
        )

    return api_key


# 使用示例
if __name__ == "__main__":
    try:
        openai_key = get_api_key("openai")
        print(f"✅ OpenAI Key 加载成功（{openai_key[:10]}...{openai_key[-4:]}）")
    except ValueError as e:
        print(f"❌ {e}")
```

### Pre-commit Hook 手工版（不依赖第三方）

```python
#!/usr/bin/env python3
# scripts/pre-commit-secret-check.py
"""
简易版 pre-commit 密钥扫描器。
在 .git/hooks/pre-commit 中调用此脚本。
"""

import re
import subprocess
import sys

# 常见的 API Key 格式模式
SECRET_PATTERNS = [
    # OpenAI: sk-proj-..., sk-...
    (r'sk-(?:proj-)?[A-Za-z0-9-_]{20,}', "OpenAI API Key"),

    # Anthropic: sk-ant-...
    (r'sk-ant-[A-Za-z0-9-_]{20,}', "Anthropic API Key"),

    # Google: AIza...
    (r'AIza[0-9A-Za-z\-_]{35}', "Google API Key"),

    # AWS Access Key: AKIA...
    (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID"),

    # AWS Secret Key
    (r'(?i)aws.?secret.?key["\']?\s*[:=]\s*["\'][A-Za-z0-9/+]{40}', "AWS Secret Key"),

    # Generic: 高熵字符串（太短的正则，仅供演示）
    (r'(?:api[_\-]?key|secret|token|password)["\']?\s*[:=]\s*["\'][A-Za-z0-9\-_\.]{16,}["\']',
     "Generic API Key/Secret"),
]


def get_staged_files() -> list[str]:
    """获取所有已暂存（staged）的文件路径。"""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def check_file(filepath: str) -> list[str]:
    """扫描单个文件中的潜在密钥。返回发现的问题列表。"""
    issues = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return issues

    for pattern, label in SECRET_PATTERNS:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            # 过滤掉 .env.example 中的占位符
            matched_text = match.group()
            if "your-key" in matched_text.lower() or "example" in matched_text.lower():
                continue
            if "your-" in matched_text.lower():
                continue

            line_num = content[:match.start()].count("\n") + 1
            # 脱敏显示：只显示前 8 个字符
            masked = matched_text[:8] + "..." + matched_text[-4:]
            issues.append(
                f"  {filepath}:{line_num} → {label}: {masked}"
            )
    return issues


def main():
    staged = get_staged_files()

    # 只检查文本文件
    text_extensions = {
        ".py", ".js", ".ts", ".json", ".yaml", ".yml",
        ".env", ".sh", ".bash", ".toml", ".ini", ".cfg",
        ".md", ".txt", ".java", ".go", ".rs", ".cpp", ".c",
        ".html", ".css", ".xml", ".rb", ".php",
    }

    all_issues = []
    for filepath in staged:
        ext = filepath[filepath.rfind("."):].lower() if "." in filepath else ""
        if ext in text_extensions:
            issues = check_file(filepath)
            all_issues.extend(issues)

    if all_issues:
        print("=" * 60)
        print("⛔ 检测到潜在的密钥泄露！Commit 已被阻止。")
        print("=" * 60)
        for issue in all_issues:
            print(issue)
        print("=" * 60)
        print("如果这是误报，请使用: git commit --no-verify")
        print("如果确实泄露了密钥，请立即吊销并轮换！")
        sys.exit(1)

    print("✅ 密钥扫描通过，未发现问题。")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

在 `.git/hooks/pre-commit` 中安装：

```bash
#!/bin/bash
# .git/hooks/pre-commit
python scripts/pre-commit-secret-check.py
```

---

## 检查清单

在你每次 `git push` 之前，确认以下 3 项：

- [ ] **`.gitignore` 已存在且包含 `.env`**：运行 `git status` 确认 `.env` 显示为 untracked（而非 staged/modified）
- [ ] **Pre-commit hook 已激活**：运行 `pre-commit run --all-files` 确认 hook 能正常执行
- [ ] **`.env.example` 已更新且不含真实密钥**：运行 `grep -E '(sk-|AKIA|AIza)' .env.example` 不应返回结果（除非是占位符 `your-key-here`）

如果以上三项有一项不满足，就不要 `git push`。

---

## 延伸阅读

- [GitHub: Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [OpenAI: API Key Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [detect-secrets 官方文档](https://github.com/Yelp/detect-secrets)
- [gitleaks 官方文档](https://github.com/gitleaks/gitleaks)

---

**一句话总结**：API Key 是你的钱袋子，`.gitignore` 是你的守卫。两个都配好，才能睡个安稳觉。
