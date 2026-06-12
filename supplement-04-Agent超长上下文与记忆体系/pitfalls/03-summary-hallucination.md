# 坑点 #3：LLM 压缩摘要引入虚假信息

## 症状
用户："我什么时候说要退款的？我从来没说过！"
Agent 声称用户在 3 天前要求退款——实际是 LLM 压缩长对话时"脑补"出来的虚假记忆。

## 根因
1. LLM 是生成式模型，压缩长文本时会"合理推测"填补细节
2. 长文本中有多个退款相关词（"退款政策"、"可以退款吗"），LLM 推断用户已申请退款
3. 摘要没有经过"事实一致性"校验

## 解决方案

### 1. 摘要事实校验
```python
class SummaryValidator:
    def validate(self, original: str, summary: str) -> dict:
        """检验摘要中每条事实是否在原文中出现"""
        original_facts = self._extract_facts(original)
        summary_facts = self._extract_facts(summary)
        
        hallucinated = []
        for fact in summary_facts:
            if not self._is_supported(fact, original_facts):
                hallucinated.append(fact)
        
        return {
            "hallucination_rate": len(hallucinated) / max(len(summary_facts), 1),
            "hallucinated_facts": hallucinated,
            "passed": len(hallucinated) == 0
        }
    
    def _extract_facts(self, text: str) -> list[str]:
        """将文本分解为原子事实陈述（按句号分割）"""
        return [s.strip() for s in text.split('。') if len(s.strip()) > 5]
    
    def _is_supported(self, fact: str, original_facts: list[str]) -> bool:
        """检查事实是否被原文中至少一条事实支持（语义相似度 > 0.8）"""
        import difflib
        for original in original_facts:
            if difflib.SequenceMatcher(None, fact, original).ratio() > 0.8:
                return True
        return False
```

### 2. 保留原文，仅压缩低重要性内容
```python
def smart_compress(messages, keep_original_threshold=0.7):
    """高重要性消息保留原文，低重要性才压缩"""
    compressed = []
    for msg in messages:
        if msg.importance >= keep_original_threshold:
            compressed.append(msg)  # 保留原文
        else:
            compressed.append(summarize(msg))  # 压缩
    return compressed
```

### 3. 置信度标记
```python
@dataclass
class MemoryEntry:
    content: str
    confidence: float  # 1.0=原文, 0.7=压缩摘要, 0.5=推测
    source: str        # "original" | "compressed" | "inferred"
    
    def to_context_string(self):
        if self.confidence < 0.8:
            return f"[低置信度记忆，请核实] {self.content}"
        return self.content
```

## 检查清单
- [ ] 压缩后是否执行了事实一致性校验？
- [ ] 高重要性原文是否保留了原始文本？
- [ ] 摘要中是否标注了置信度？
- [ ] 是否定期人工抽查摘要质量？
- [ ] 是否有"摘要幻觉"的监控指标？
