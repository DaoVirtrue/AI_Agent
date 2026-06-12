#!/usr/bin/env python3
"""
A3-05: 流式安全与监控 (Streaming Security & Monitoring)
===========================================================
学习目标:
  1. 流式过程中的增量正则扫描（敏感信息检测/PII）
  2. 部分内容PII检测
  3. 流中断处理（中断+恢复+审计）
  4. P50/P95/P99延迟指标统计
"""

import re
import sys
import time
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

# ============================================================
# Section 1: 增量正则扫描（流式敏感内容检测）
# ============================================================

class IncrementalRegexScanner:
    """
    增量正则扫描器
    核心挑战: 流式tokens可能分割敏感模式
    解决: 维护一个滑动窗口缓冲区
    """

    def __init__(self):
        # 敏感信息检测规则
        # 每条规则: (名称, 正则, 严重级别, 处理策略)
        self.rules: List[Tuple[str, re.Pattern, str, str]] = [
            # (name, pattern, severity, action)
            # action: "block"(阻断), "mask"(脱敏), "warn"(仅警告)

            # 身份证号 (18位)
            ("cn_id_card", re.compile(
                r'[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]'
            ), "high", "mask"),

            # 手机号 (中国大陆)
            ("cn_phone", re.compile(
                r'(?<!\d)1[3-9]\d{9}(?!\d)'
            ), "high", "mask"),

            # 邮箱地址
            ("email", re.compile(
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            ), "medium", "mask"),

            # IP地址
            ("ip_address", re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ), "low", "warn"),

            # 银行卡号 (16-19位)
            ("bank_card", re.compile(
                r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4,7}\b'
            ), "high", "block"),

            # API Key模式 (sk-开头)
            ("openai_key", re.compile(
                r'sk-[a-zA-Z0-9]{32,}'
            ), "critical", "block"),

            # AWS Access Key
            ("aws_key", re.compile(
                r'AKIA[0-9A-Z]{16}'
            ), "critical", "block"),

            # 密码模式 (弱检测)
            ("password_pattern", re.compile(
                r'(?:password|passwd|pwd|secret)\s*[:=]\s*\S+',
                re.IGNORECASE
            ), "high", "mask"),
        ]

        self.buffer = ""      # 滑动缓冲区
        self.buffer_size = 512  # 缓冲区大小（足够容纳最长的敏感模式）
        self.scan_events: List[Dict] = []  # 扫描事件历史

    def scan(self, token: str) -> List[Dict]:
        """
        增量扫描一个token
        返回检测到的敏感信息列表

        为什么需要缓冲区？
        例如: "身份证号是44010" ... "0199901011234"
        如果不缓冲，前半段匹配不到完整模式
        """
        self.buffer += token

        # 保持缓冲区大小
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

        findings = []

        for rule_name, pattern, severity, action in self.rules:
            matches = pattern.finditer(self.buffer)
            for match in matches:
                found = match.group()
                # 检查是否可能是误报（IP地址常见误报）
                if rule_name == "ip_address":
                    # 排除版本号 (1.2.3.4 看起来像IP但不是)
                    parts = found.split(".")
                    if len(parts) == 4:
                        try:
                            nums = [int(p) for p in parts]
                            if not all(0 <= n <= 255 for n in nums):
                                continue
                        except ValueError:
                            continue

                finding = {
                    "rule": rule_name,
                    "matched": self._mask_value(found) if action == "mask" else found[:3] + "***",
                    "severity": severity,
                    "action": action,
                    "position": len(self.buffer) - len(found),
                    "timestamp": time.time(),
                }
                findings.append(finding)
                self.scan_events.append(finding)

        return findings

    def _mask_value(self, value: str) -> str:
        """脱敏处理"""
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    def get_stats(self) -> Dict:
        """获取扫描统计"""
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for event in self.scan_events:
            by_severity[event["severity"]] = by_severity.get(event["severity"], 0) + 1
        return {
            "total_scanned_tokens": len(self.scan_events),
            "blocked": sum(1 for e in self.scan_events if e["action"] == "block"),
            "masked": sum(1 for e in self.scan_events if e["action"] == "mask"),
            "warned": sum(1 for e in self.scan_events if e["action"] == "warn"),
            "by_severity": by_severity,
        }


# ============================================================
# Section 2: PII (个人身份信息) 检测器
# ============================================================

class PIIDetector:
    """
    PII检测器（在部分内容上工作）
    检测中文环境常见的PII类型
    """

    def __init__(self):
        self.pii_patterns = {
            "chinese_name": re.compile(
                r'(?:[王李张刘陈杨黄赵周吴徐孙马胡朱郭何罗高林郑梁谢唐许韩冯邓曹彭曾萧田董潘袁于蒋蔡余杜叶程苏魏吕丁任卢姚沈钟姜崔谭陆范汪廖石金韦贾夏付方邹熊]'
                r'[一-鿿]{1,2}'  # 常见姓氏+1-2字名
            ),
            "phone_number": re.compile(r'1[3-9]\d{9}'),
            "id_number": re.compile(r'\d{17}[\dXx]'),
            "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            "address": re.compile(
                r'(?:北京|上海|广州|深圳|杭州|成都|武汉|南京|天津|重庆|苏州|西安|长沙|青岛|大连|厦门|福州|合肥|郑州|济南|沈阳|昆明|贵阳|南宁|哈尔滨|长春|太原|石家庄|兰州|银川|西宁|乌鲁木齐|呼和浩特|拉萨|海口)'
                r'(?:市|省)[一-鿿0-9号路街巷楼栋单元室层]{5,30}'
            ),
            "company_name": re.compile(
                r'(?:有限公司|股份有限公司|有限责任公司|集团|科技)'
            ),
            "url": re.compile(
                r'https?://[^\s<>"{}|\\^`\[\]]+'
            ),
        }

        self.context_keywords = [
            "身份证", "手机号", "电话号码", "邮箱", "地址",
            "家庭住址", "工作单位", "银行卡", "密码", "用户名",
        ]

    def detect_on_stream(self, token: str, full_text: str) -> List[Dict]:
        """
        对流式内容进行PII检测（带上下文感知）

        参数:
            token: 新收到的token
            full_text: 到目前位置的全部文本
        """
        findings = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.finditer(full_text)
            for match in matches:
                # 上下文检查：附近是否有提示性关键词
                start = max(0, match.start() - 20)
                end = min(len(full_text), match.end() + 20)
                context = full_text[start:end]
                has_context = any(
                    kw in context for kw in self.context_keywords
                )

                findings.append({
                    "type": pii_type,
                    "value_preview": self._mask_pii(match.group(), pii_type),
                    "confidence": "high" if has_context else "medium",
                    "has_context_keyword": has_context,
                })

        return findings

    def _mask_pii(self, value: str, pii_type: str) -> str:
        """不同类型的PII脱敏策略不同"""
        if pii_type == "chinese_name":
            if len(value) == 2:
                return value[0] + "*"
            return value[0] + "*" * (len(value) - 2) + value[-1]
        elif pii_type == "phone_number":
            return value[:3] + "****" + value[-4:]
        elif pii_type == "id_number":
            return value[:3] + "***********" + value[-4:]
        elif pii_type == "email":
            parts = value.split("@")
            return parts[0][:2] + "***@" + parts[1] if len(parts) == 2 else "***@***"
        else:
            return value[:3] + "***" if len(value) > 3 else "***"


# ============================================================
# Section 3: 流中断处理
# ============================================================

@dataclass
class StreamInterruption:
    """流中断记录"""
    interruption_id: str
    timestamp: float
    reason: str                 # "user_cancelled" | "security_blocked" | "timeout" | "error"
    tokens_generated: int
    partial_text: str
    user_id: str = "anonymous"


class StreamInterruptionHandler:
    """
    流中断处理器

    场景:
    1. 用户取消: 保存已生成的部分文本
    2. 安全阻断: 敏感内容触发自动中断
    3. 超时: 生成时间超过阈值
    4. 错误: 网络/服务异常
    """

    def __init__(self, save_partial: bool = True):
        self.save_partial = save_partial
        self.interruptions: List[StreamInterruption] = []
        self.active_streams: Dict[str, Dict] = {}  # stream_id -> state

    def start_stream(self, stream_id: str, user_id: str = "anonymous"):
        """注册一个新的流"""
        self.active_streams[stream_id] = {
            "start_time": time.time(),
            "tokens": 0,
            "partial_text": "",
            "user_id": user_id,
            "status": "active",
        }

    def on_token(self, stream_id: str, token: str):
        """记录一个token"""
        if stream_id in self.active_streams:
            state = self.active_streams[stream_id]
            state["tokens"] += 1
            state["partial_text"] += token

    def interrupt(self, stream_id: str, reason: str) -> Optional[StreamInterruption]:
        """
        中断流
        返回中断记录
        """
        if stream_id not in self.active_streams:
            return None

        state = self.active_streams[stream_id]
        state["status"] = "interrupted"

        record = StreamInterruption(
            interruption_id=stream_id,
            timestamp=time.time(),
            reason=reason,
            tokens_generated=state["tokens"],
            partial_text=state["partial_text"][:500] if self.save_partial else "",
            user_id=state["user_id"],
        )
        self.interruptions.append(record)

        del self.active_streams[stream_id]
        return record

    def resume_stream(self, stream_id: str,
                      partial_text: str = "") -> str:
        """
        恢复流（重新建立连接）
        返回新的stream_id，可用于继续之前的上下文
        """
        new_stream_id = f"{stream_id}_resume_{int(time.time())}"
        self.start_stream(new_stream_id)
        return new_stream_id

    def get_interruption_stats(self) -> Dict:
        """获取中断统计"""
        by_reason = {}
        total_tokens_lost = 0
        for record in self.interruptions:
            by_reason[record.reason] = by_reason.get(record.reason, 0) + 1
            total_tokens_lost += record.tokens_generated

        return {
            "total_interruptions": len(self.interruptions),
            "by_reason": by_reason,
            "total_tokens_lost": total_tokens_lost,
            "active_streams": len(self.active_streams),
        }


# ============================================================
# Section 4: 延迟指标统计 (P50/P95/P99)
# ============================================================

class LatencyMetricsCollector:
    """
    延迟指标收集器
    使用T-Digest算法近似（简化版使用分位数插值）
    """

    def __init__(self, window_size: int = 1000):
        """
        参数:
            window_size: 滑动窗口大小（保留最近N个样本）
        """
        self.window_size = window_size
        self.ttft_samples: deque = deque(maxlen=window_size)  # Time To First Token
        self.itl_samples: deque = deque(maxlen=window_size)    # Inter-Token Latency
        self.total_latency_samples: deque = deque(maxlen=window_size)
        self.token_counts: deque = deque(maxlen=window_size)

    def record_request(self, ttft: float, itl: List[float],
                       total_latency: float, total_tokens: int):
        """记录一次请求的延迟数据"""
        self.ttft_samples.append(ttft)
        self.itl_samples.extend(itl)  # 每个inter-token延迟单独记录
        self.total_latency_samples.append(total_latency)
        self.token_counts.append(total_tokens)

    def compute_percentiles(self, data: deque) -> Dict[str, float]:
        """
        计算P50/P90/P95/P99
        使用最近邻分位数估计
        """
        if not data:
            return {}

        sorted_data = sorted(data)
        n = len(sorted_data)

        def percentile(p: float) -> float:
            """线性插值分位数"""
            idx = p * (n - 1) / 100.0
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac

        return {
            "count": n,
            "min": round(sorted_data[0], 4),
            "max": round(sorted_data[-1], 4),
            "mean": round(np.mean(sorted_data), 4),
            "std": round(np.std(sorted_data), 4),
            "p50": round(percentile(50), 4),
            "p90": round(percentile(90), 4),
            "p95": round(percentile(95), 4),
            "p99": round(percentile(99), 4),
        }

    def get_summary(self) -> Dict:
        """获取完整延迟摘要"""
        return {
            "ttft_ms": self.compute_percentiles(self.ttft_samples),
            "itl_ms": self.compute_percentiles(self.itl_samples),
            "total_latency_ms": self.compute_percentiles(self.total_latency_samples),
            "tokens_per_request": self.compute_percentiles(self.token_counts),
            "total_requests": len(self.ttft_samples),
        }

    def print_report(self) -> str:
        """打印人类可读的延迟报告"""
        summary = self.get_summary()
        lines = [
            "=" * 60,
            "延迟指标报告 (Latency Metrics Report)",
            "=" * 60,
        ]

        for metric_name, display_name in [
            ("ttft_ms", "首Token延迟 (TTFT)"),
            ("itl_ms", "Token间延迟 (ITL)"),
            ("total_latency_ms", "总请求延迟"),
        ]:
            data = summary.get(metric_name, {})
            if data:
                lines.append(f"\n{display_name}:")
                lines.append(f"  样本数: {data.get('count', 0)}")
                lines.append(f"  P50: {data.get('p50', '?'):.1f}ms")
                lines.append(f"  P95: {data.get('p95', '?'):.1f}ms")
                lines.append(f"  P99: {data.get('p99', '?'):.1f}ms")
                lines.append(f"  均值: {data.get('mean', '?'):.1f}ms")
                lines.append(f"  最大: {data.get('max', '?'):.1f}ms")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ============================================================
# Section 5: 综合安全+监控管道
# ============================================================

class StreamingSecurityMonitor:
    """
    流式安全与监控综合管道
    整合了安全扫描 + PII检测 + 中断处理 + 延迟监控
    """

    def __init__(self, block_critical: bool = True):
        self.scanner = IncrementalRegexScanner()
        self.pii_detector = PIIDetector()
        self.interruption_handler = StreamInterruptionHandler(save_partial=True)
        self.latency_collector = LatencyMetricsCollector()
        self.block_critical = block_critical

        # 安全事件计数
        self.security_events: List[Dict] = []

    async def process_stream(
        self, stream_id: str, user_id: str,
        token_generator,  # async generator yielding tokens
    ) -> Dict:
        """
        处理一个流请求的全生命周期
        包括安全扫描、PII检测、延迟统计

        返回: {final_text, blocked, security_events, latency_summary}
        """
        # 初始化
        self.interruption_handler.start_stream(stream_id, user_id)
        full_text = ""
        ttft_start = time.time()
        ttft_recorded = False
        inter_token_times: List[float] = []
        last_token_time = time.time()
        blocked = False

        try:
            async for raw_token in token_generator:
                # 记录时间
                now = time.time()
                if not ttft_recorded:
                    ttft = (now - ttft_start) * 1000  # 转为ms
                    ttft_recorded = True
                else:
                    itl = (now - last_token_time) * 1000
                    inter_token_times.append(itl)
                last_token_time = now

                # === 安全扫描 ===
                findings = self.scanner.scan(raw_token)

                should_block = False
                for finding in findings:
                    self.security_events.append(finding)
                    print(f"  [安全] {finding['severity'].upper()}: "
                          f"{finding['rule']} → {finding['matched']}")

                    if self.block_critical and finding["action"] == "block":
                        should_block = True

                if should_block:
                    self.interruption_handler.interrupt(
                        stream_id, "security_blocked"
                    )
                    blocked = True
                    break

                # === PII检测 ===
                full_text += raw_token
                pii_findings = self.pii_detector.detect_on_stream(raw_token, full_text)
                for pf in pii_findings:
                    if pf["confidence"] == "high":
                        print(f"  [PII] {pf['type']}: {pf['value_preview']}")

                # 更新状态
                self.interruption_handler.on_token(stream_id, raw_token)

            # 记录延迟
            total_latency = (time.time() - ttft_start) * 1000
            self.latency_collector.record_request(
                ttft=(time.time() - ttft_start) * 1000 if not ttft_recorded else ttft,
                itl=inter_token_times if inter_token_times else [0.0],
                total_latency=total_latency,
                total_tokens=len(full_text),
            )

        except asyncio.CancelledError:
            self.interruption_handler.interrupt(stream_id, "user_cancelled")

        except Exception as e:
            self.interruption_handler.interrupt(stream_id, "error")

        return {
            "final_text": full_text if not blocked else full_text[:200],
            "blocked": blocked,
            "security_events": len([e for e in self.security_events
                                   if time.time() - e["timestamp"] < 60]),
            "latency_summary": self.latency_collector.get_summary(),
        }


# ============================================================
# Section 6: 主流程
# ============================================================

async def simulate_token_stream(text: str):
    """模拟token流生成器"""
    for char in text:
        await asyncio.sleep(0.02)
        yield char


async def main_async():
    """异步主函数"""
    print("=" * 70)
    print("A3-05: 流式安全与监控 — 完整演示")
    print("=" * 70)

    # 1. 增量正则扫描
    print("\n[测试1] 增量正则扫描 — 敏感信息检测")
    scanner = IncrementalRegexScanner()
    test_tokens = [
        "用户信息如下：", "姓名张三，",
        "身份证号440101", "199001011234，",
        "手机号为13800138000，", "邮箱是test@example.com。",
        "API Key是sk-abc123def456ghi789jkl012mno345pqr678stu901vwx。",
    ]
    for token in test_tokens:
        findings = scanner.scan(token)
        if findings:
            for f in findings:
                print(f"  [{f['severity']}] {f['rule']}: {f['matched']} "
                      f"→ action={f['action']}")

    print(f"\n  扫描统计: {scanner.get_stats()}")

    # 2. PII检测
    print("\n[测试2] PII检测 — 个人身份信息")
    pii = PIIDetector()
    test_text = "申请人张三，身份证号440101199001011234，手机13800138000，邮箱zhangsan@company.com，居住在北京市海淀区中关村大街1号。"
    buffer = ""
    for char in test_text:
        buffer += char
        findings = pii.detect_on_stream(char, buffer)
        if findings:
            for f in findings:
                print(f"  [{f['confidence']}] {f['type']}: {f['value_preview']}")

    # 3. 流中断处理
    print("\n[测试3] 流中断处理...")
    handler = StreamInterruptionHandler(save_partial=True)
    handler.start_stream("stream_001", "user_test")
    handler.on_token("stream_001", "这是")
    handler.on_token("stream_001", "一个")
    handler.on_token("stream_001", "测试")

    record = handler.interrupt("stream_001", "user_cancelled")
    if record:
        print(f"  中断记录: {record.reason}, "
              f"tokens={record.tokens_generated}, "
              f"partial_text='{record.partial_text}'")

    stats = handler.get_interruption_stats()
    print(f"  中断统计: {stats}")

    # 4. 延迟指标
    print("\n[测试4] 延迟指标统计 (P50/P95/P99)...")
    collector = LatencyMetricsCollector(window_size=100)

    # 模拟一些请求
    np.random.seed(42)
    for i in range(30):
        ttft = np.random.gamma(2, 150)   # ~300ms average TTFT
        itl = np.random.gamma(1, 20, size=50)  # ~20ms inter-token
        total = ttft + sum(itl)
        collector.record_request(
            ttft=ttft,
            itl=itl.tolist(),
            total_latency=total,
            total_tokens=50,
        )

    print(collector.print_report())

    # 5. 综合管道
    print("\n[测试5] 综合安全+监控管道...")
    monitor = StreamingSecurityMonitor(block_critical=True)

    # 正常内容流
    safe_text = "RAG系统是一种检索增强生成技术，它结合了向量检索和大语言模型的优势。"
    print("  [正常流]")
    result = await monitor.process_stream(
        "stream_safe", "test_user",
        simulate_token_stream(safe_text),
    )
    print(f"  结果: blocked={result['blocked']}, "
          f"text_len={len(result['final_text'])}")

    # 含敏感内容流
    sensitive_text = "这是用户信息：身份证440101199001011234，手机号13800138000，密码是abcd1234。"
    print("\n  [敏感内容流]")
    result2 = await monitor.process_stream(
        "stream_sensitive", "test_user",
        simulate_token_stream(sensitive_text),
    )
    print(f"  结果: blocked={result2['blocked']}, "
          f"security_events={result2['security_events']}")

    print("\n" + "=" * 70)
    print("A3-05 演示完成！")
    print("=" * 70)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
