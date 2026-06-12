# 坑点 #4：跨会话记忆泄漏

## 症状
用户 A 打开客服对话，Agent 说："张先生您好，您上次咨询的贷款问题..."  
但用户 A 姓李，贷款问题是用户 B 的——记忆跨用户泄漏了。

## 根因
1. **无用户隔离**：所有对话存在同一张 memories 表，按 `timestamp` 全局排序
2. **未过滤 tenant_id**：检索时忘记加 `WHERE user_id = ?`
3. **pickle 文件共享**：单文件存储所有用户记忆，无加密

## 解决方案

### 1. 用户级命名空间隔离
```python
class IsolatedMemoryStore:
    def __init__(self):
        self._stores: dict[str, SQLiteMemoryStore] = {}  # 每用户独立 store
    
    def get_store(self, user_id: str) -> SQLiteMemoryStore:
        if user_id not in self._stores:
            self._stores[user_id] = SQLiteMemoryStore(f"memories_{user_id}.db")
        return self._stores[user_id]
    
    def retrieve(self, user_id: str, query: str, top_k: int = 5):
        store = self.get_store(user_id)
        return store.search(query, top_k)  # 天然隔离
```

### 2. 强制 SQL WHERE 过滤
```python
def safe_retrieve(user_id: str, query: str):
    # 强制注入 user_id，不能依赖调用方传参
    cursor.execute(
        "SELECT * FROM memories WHERE user_id = ? ORDER BY importance DESC LIMIT ?",
        (user_id, 10)
    )
```

### 3. PII 存储前脱敏
```python
class PIIFilter:
    PATTERNS = {
        "phone": r'\b1[3-9]\d{9}\b',
        "id_card": r'\b\d{17}[\dXx]\b',
        "email": r'\b[\w.-]+@[\w.-]+\.\w+\b',
    }
    
    def scrub(self, text: str) -> str:
        for pii_type, pattern in self.PATTERNS.items():
            text = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', text)
        return text
```

### 4. 加密存储
```python
from cryptography.fernet import Fernet

class EncryptedStore:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)
    
    def store(self, user_id, content):
        encrypted = self.cipher.encrypt(content.encode())
        self.db.insert(user_id, encrypted)
    
    def retrieve(self, user_id, memory_id):
        encrypted = self.db.get(user_id, memory_id)
        return self.cipher.decrypt(encrypted).decode()
```

## 检查清单
- [ ] 每用户是否有独立的存储空间（独立 DB / namespace）？
- [ ] 检索时是否强制了 user_id 过滤？
- [ ] PII 是否在存储前脱敏？
- [ ] 敏感记忆是否加密存储？
- [ ] 是否有跨用户访问的审计日志？
- [ ] GDPR 被遗忘权是否支持（按 user_id 一键删除所有记忆）？
