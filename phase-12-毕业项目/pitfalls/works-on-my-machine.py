#!/usr/bin/env python3
"""
陷阱3: 在我的机器上没问题 (Works on My Machine)
================================================
症状: 本地完美运行，部署到服务器后各种问题
根因: 环境差异、依赖不一致、配置不匹配

解决: Docker容器化 + CI/CD + 环境一致性 + 健康检查
"""

class EnvironmentChecker:
    def __init__(self):
        self.checks = []
    def add(self, name, fn): self.checks.append((name, fn))
    def verify(self):
        all_ok = True
        for name, fn in self.checks:
            try:
                fn()
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                all_ok = False
        return all_ok

if __name__ == "__main__":
    import sys, os
    checker = EnvironmentChecker()
    checker.add("Python版本", lambda: print(f"  {sys.version}"))
    checker.add("工作目录", lambda: print(f"  {os.getcwd()}"))
    checker.add("环境变量", lambda: print(f"  PATH={os.environ.get('PATH', '')[:50]}"))
    ok = checker.verify()
    print(f"环境检查: {'通过' if ok else '失败'}")
