#!/usr/bin/env python3
"""
S1-01：AI 单元测试 —— MockLLM、MockEmbedder、工具模拟策略
=============================================================
AI 系统的单元测试面临独特挑战：LLM 行为的非确定性、API 延迟、
费用成本。解决方案是构建高质量的 Mock 层。

本文件演示：
1. MockLLM —— 固定输出 / 错误注入 / 延迟模拟 / Token 计数
2. MockEmbedder —— 确定性向量输出
3. 工具 Mock —— 模拟外部 API 调用
4. 完整测试用例

依赖：pytest（建议）
"""

import time
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ============================================================================
# MockLLM：模拟大语言模型
# ============================================================================

@dataclass
class LLMResponse:
    """模拟 LLM 响应。"""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "mock-llm"
    latency_ms: float = 0


class MockLLM:
    """
    可配置的模拟 LLM。

    特性：
    - 固定输出（可预测的测试结果）
    - 错误注入（测试错误处理逻辑）
    - 延迟模拟（测试超时处理）
    - Token 计数（测试成本计算）
    - 调用历史（验证调用参数）
    """

    def __init__(self, default_response: str = "[MockLLM] 默认回答"):
        self.default_response = default_response
        self.call_history: List[Dict] = []
        self._fail_on_next: Optional[str] = None
        self._latency_ms: float = 0
        self._response_map: Dict[str, str] = {}  # 关键词 → 回答

    def set_response_for(self, keyword: str, response: str):
        """为包含特定关键词的查询设置指定回答。"""
        self._response_map[keyword] = response

    def set_latency(self, ms: float):
        """设置模拟延迟。"""
        self._latency_ms = ms

    def inject_error(self, error_message: str):
        """注入下一次调用的错误。"""
        self._fail_on_next = error_message

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """模拟 LLM 调用。"""
        # 错误注入
        if self._fail_on_next:
            error = self._fail_on_next
            self._fail_on_next = None
            raise RuntimeError(f"[MockLLM 错误注入] {error}")

        # 延迟模拟
        if self._latency_ms:
            time.sleep(self._latency_ms / 1000)

        # 查找匹配的回答
        response_text = self.default_response
        for keyword, resp in self._response_map.items():
            if keyword in prompt:
                response_text = resp
                break

        # Token 估算
        input_tokens = len(prompt) // 3  # 粗略估算
        output_tokens = len(response_text) // 3

        # 记录历史
        response = LLMResponse(
            content=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=self._latency_ms,
        )
        self.call_history.append({"prompt": prompt[:200], "response": response})
        return response

    def assert_called_with(self, expected_keyword: str):
        """断言 LLM 被用包含特定关键词的 prompt 调用过。"""
        for record in self.call_history:
            if expected_keyword in record["prompt"]:
                return True
        raise AssertionError(
            f"LLM 未被用包含 '{expected_keyword}' 的 prompt 调用过。"
            f"历史调用: {[r['prompt'][:50] for r in self.call_history]}"
        )

    def reset(self):
        """重置状态。"""
        self.call_history.clear()
        self._fail_on_next = None
        self._latency_ms = 0


# ============================================================================
# MockEmbedder：模拟嵌入模型
# ============================================================================

class MockEmbedder:
    """
    确定性的模拟嵌入模型。

    特点：
    - 相同输入总是产生相同向量（确定性）
    - 基于文本哈希生成向量
    - 可配置维度
    - 相似文本产生"相似"向量（通过共享前缀控制）
    """

    def __init__(self, dimension: int = 64, seed: int = 42):
        self.dimension = dimension
        self.seed = seed
        self.embed_call_count = 0
        self.embedded_texts: List[str] = []

    def embed(self, text: str) -> List[float]:
        """生成确定性嵌入向量。"""
        self.embed_call_count += 1
        self.embedded_texts.append(text)

        # 基于文本哈希的确定性向量
        h = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
        # 将哈希字节扩展到指定维度
        vec = []
        for i in range(self.dimension):
            byte_val = h[i % len(h)]
            vec.append((byte_val / 255.0) * 2 - 1)  # 归一化到 [-1, 1]
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入。"""
        return [self.embed(t) for t in texts]

    def assert_embed_count(self, expected: int):
        """断言嵌入调用次数。"""
        assert self.embed_call_count == expected, (
            f"预期嵌入调用 {expected} 次，实际 {self.embed_call_count} 次"
        )


# ============================================================================
# MockTool：模拟外部工具
# ============================================================================

class MockTool:
    """
    可配置的模拟工具。

    模拟外部 API 调用（如数据库查询、HTTP API、文件系统操作）。
    """

    def __init__(self, name: str, default_result: Any = None):
        self.name = name
        self.default_result = default_result
        self.call_count = 0
        self.call_args: List[dict] = []
        self._results: Dict[str, Any] = {}  # 参数hash → 结果
        self._should_fail: bool = False
        self._fail_message: str = ""

    def set_result_for(self, params: dict, result: Any):
        """为特定参数设置返回结果。"""
        params_hash = hashlib.md5(str(params).encode()).hexdigest()[:8]
        self._results[params_hash] = result

    def inject_failure(self, message: str = "模拟工具失败"):
        """注入失败。"""
        self._should_fail = True
        self._fail_message = message

    async def __call__(self, **kwargs) -> Any:
        """模拟工具调用。"""
        self.call_count += 1
        self.call_args.append(kwargs)

        if self._should_fail:
            self._should_fail = False
            raise RuntimeError(f"[{self.name}] {self._fail_message}")

        # 查找匹配的结果
        params_hash = hashlib.md5(str(kwargs).encode()).hexdigest()[:8]
        if params_hash in self._results:
            return self._results[params_hash]

        return self.default_result

    def assert_called(self, times: int = 1):
        """断言工具被调用了指定次数。"""
        assert self.call_count == times, (
            f"工具 '{self.name}' 预期调用 {times} 次，实际 {self.call_count} 次"
        )

    def assert_called_with(self, **expected_params):
        """断言工具被以特定参数调用过。"""
        for args in self.call_args:
            if all(args.get(k) == v for k, v in expected_params.items()):
                return True
        raise AssertionError(
            f"工具 '{self.name}' 未被以参数 {expected_params} 调用过。"
        )

    def reset(self):
        """重置状态。"""
        self.call_count = 0
        self.call_args.clear()
        self._should_fail = False


# ============================================================================
# 测试用例示例
# ============================================================================

class TestRAGService:
    """
    RAG 服务的测试用例示例。

    演示如何使用 MockLLM、MockEmbedder、MockTool 进行测试。
    """

    def setup_method(self):
        """每个测试方法前的准备。"""
        self.mock_llm = MockLLM()
        self.mock_embedder = MockEmbedder()
        self.mock_search_tool = MockTool("search_kb", default_result=[
            {"content": "RAG 是检索增强生成", "score": 0.95},
            {"content": "RAG 包括文档加载和检索", "score": 0.88},
        ])

    def test_search_returns_relevant_docs(self):
        """测试：检索返回相关文档。"""
        results = self.mock_search_tool(query="什么是RAG")
        assert len(results) > 0, "检索应返回至少一个结果"
        assert results[0]["score"] >= 0.8, "第一个结果的相关度应较高"
        self.mock_search_tool.assert_called(times=1)

    def test_llm_handles_error_gracefully(self):
        """测试：LLM 错误时优雅降级。"""
        self.mock_llm.inject_error("API 超时")
        try:
            self.mock_llm.complete("测试问题")
            assert False, "应该抛出异常"
        except RuntimeError as e:
            assert "API 超时" in str(e), f"错误信息不匹配: {e}"

        # 错误恢复后应正常工作
        response = self.mock_llm.complete("另一个问题")
        assert response.content is not None

    def test_embedder_is_deterministic(self):
        """测试：嵌入是确定性的（相同输入 → 相同输出）。"""
        vec1 = self.mock_embedder.embed("测试文本")
        vec2 = self.mock_embedder.embed("测试文本")
        assert vec1 == vec2, "相同文本应产生相同向量"

    def test_llm_records_call_history(self):
        """测试：LLM 记录调用历史。"""
        self.mock_llm.complete("问题A")
        self.mock_llm.complete("问题B")
        assert len(self.mock_llm.call_history) == 2, "应有2次调用记录"
        self.mock_llm.assert_called_with("问题A")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("AI 单元测试 — Mock 策略演示")
    print("=" * 60)

    # MockLLM 演示
    print("\n【MockLLM 演示】")
    llm = MockLLM()
    llm.set_response_for("RAG", "RAG 是检索增强生成技术。它的核心组件包括...")
    llm.set_latency(50)  # 模拟 50ms 延迟

    response = llm.complete("请介绍RAG技术")
    print(f"  查询: '请介绍RAG技术'")
    print(f"  回答: {response.content[:80]}...")
    print(f"  Token: 输入{response.input_tokens} / 输出{response.output_tokens}")
    print(f"  延迟: {response.latency_ms}ms")
    print(f"  调用历史: {len(llm.call_history)} 条")

    # 错误注入演示
    print("\n【错误注入演示】")
    llm.inject_error("模拟的速率限制错误: 429 Too Many Requests")
    try:
        llm.complete("另一个查询")
    except RuntimeError as e:
        print(f"  捕获到注入的错误: {e}")
    print(f"  恢复后正常调用: {llm.complete('正常查询').content[:60]}")

    # MockEmbedder 演示
    print("\n【MockEmbedder 演示】")
    embedder = MockEmbedder(dimension=8)
    v1 = embedder.embed("苹果是一种水果")
    v2 = embedder.embed("苹果是一种水果")
    v3 = embedder.embed("香蕉是一种水果")
    print(f"  相同文本向量相同: {v1 == v2}")
    print(f"  不同文本向量不同: {v1 != v3}")
    print(f"  向量维度: {len(v1)}")
    print(f"  调用次数: {embedder.embed_call_count}")

    # MockTool 演示
    print("\n【MockTool 演示】")
    import asyncio
    tool = MockTool("database_query")
    result = asyncio.run(tool(sql="SELECT * FROM docs"))
    print(f"  工具调用结果: {result}")
    tool.assert_called(times=1)
    print(f"  断言通过: 工具被调用1次 ✅")

    print("\n[完成] Mock 策略演示结束。")
    print("  提示: 使用 pytest 运行测试: pytest s1-01-ai单元测试.py -v")
