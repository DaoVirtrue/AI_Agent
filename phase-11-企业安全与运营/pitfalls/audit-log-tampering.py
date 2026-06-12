#!/usr/bin/env python3
"""
陷阱5: 审计日志篡改 (Audit Log Tampering)
===========================================
症状: 安全事件发生后发现审计日志被修改或删除
根因: 日志存储可写 + 无完整性验证 + 无防篡改机制

解决: 哈希链 + 不可变存储 + 定期完整性验证 + 异地备份
"""

import hashlib, os

class TamperProofLogger:
    def __init__(self, log_file="audit.log"):
        self.log_file = log_file
        self.last_hash = "0" * 64
        if os.path.exists(log_file):
            self._rebuild_chain()

    def _rebuild_chain(self):
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.last_hash = line.strip().split('|')[-1]

    def log(self, event: str) -> str:
        entry_hash = hashlib.sha256(f"{event}|{self.last_hash}".encode()).hexdigest()
        with open(self.log_file, 'a') as f:
            f.write(f"{event}|{entry_hash}\n")
        self.last_hash = entry_hash
        return entry_hash

    def verify(self) -> bool:
        prev = "0" * 64
        with open(self.log_file, 'r') as f:
            for line in f:
                parts = line.strip().rsplit('|', 1)
                event, expected_hash = parts[0], parts[1] if len(parts) == 2 else ""
                actual = hashlib.sha256(f"{event}|{prev}".encode()).hexdigest()
                if actual != expected_hash:
                    print(f"篡改检测! 期望{expected_hash[:16]}... 实际{actual[:16]}...")
                    return False
                prev = actual
        return True

if __name__ == "__main__":
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "demo_audit.log")
    logger = TamperProofLogger(tmp)
    logger.log("USER_LOGIN alice")
    logger.log("ACCESS_DENIED bob")
    print(f"完整性: {logger.verify()}")
    os.remove(tmp)
