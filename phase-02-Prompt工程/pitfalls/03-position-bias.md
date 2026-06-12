# 位置偏差：Lost in the Middle 问题与缓解

## 一、问题本质

"Lost in the Middle"是LLM领域的一个核心发现：当上下文窗口较长时，LLM对**开头和结尾**的信息关注度最高，而对**中间部分**的信息关注度显著下降。这就像人类阅读长文档时，往往记住开头和结尾，中间内容容易"丢失"。

对于RAG系统来说，这意味着：如果将最相关的检索文档放在上下文窗口的中间位置，LLM可能会忽略它们，导致回答质量下降。

### 经典研究结论

来自论文 *Lost in the Middle: How Language Models Use Long Contexts* (Liu et al., 2023)：

```
模型对位置的注意力分布（示意）：

高  |  ████████                    ████████
    |  ████████                    ████████
    |  ████████    ████████        ████████
    |  ████████    ████████        ████████
中  |  ████████    ████████        ████████
    |  ████████    ████████        ████████
低  |             ████████
    |______________________________________
       开头        中间         结尾
             上下文中的位置 -->
```

---

## 二、症状与影响

| 症状 | 具体表现 | 影响 |
|------|---------|------|
| **中间文档被忽略** | 检索score最高的文档放在中间位置，但LLM未引用 | 答案质量下降，信息丢失 |
| **开头/结尾过度依赖** | LLM过度引用第一段和最后一段内容 | 答案偏差 |
| **多文档查询失败** | 需要综合多篇文档的问题回答不完整 | 多跳推理中断 |
| **长上下文退化** | 上下文越长，中间信息丢失越严重 | 可伸缩性差 |

---

## 三、解决方案：LongContextReorder

核心思想：**根据相关性分数，将最重要的文档放在开头和结尾，最不重要的放在中间**。

### 完整实现

```python
"""
LongContextReorder -- 缓解 Lost in the Middle 问题的文档重排序器

策略：
1. 将最相关的文档放在开头（primacy effect）
2. 将次相关的文档放在结尾（recency effect）
3. 将最不相关的文档放在中间（sacrificial middle）
4. 支持多种排序策略：score-based, alternating, interleaving
"""

from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import math


class ReorderStrategy(Enum):
    """重排序策略"""
    SCORE_DESCENDING = "score_desc"        # 纯分数降序（不解决Lost in Middle）
    U_SHAPED = "u_shaped"                  # U型排列：高-中-低-中-高
    INTERLEAVED = "interleaved"            # 交替排列：高-低-高-低
    ALT_CAP = "alt_cap"                    # 交替+首尾强化：最高放开头，次高放结尾


@dataclass
class Document:
    """文档数据类"""
    content: str
    score: float = 0.0           # 相关性分数（越高越相关）
    metadata: dict = field(default_factory=dict)
    index: int = 0               # 原始位置（用于追踪）

    @property
    def token_count(self) -> int:
        """粗略估算token数（中文按字符，英文按4字符/token）"""
        return len(self.content) // 2  # 简化估算


@dataclass
class ReorderResult:
    """重排序结果"""
    documents: List[Document]
    strategy: ReorderStrategy
    original_order: List[int]    # 原始索引顺序，用于调试
    position_map: dict           # {new_pos: old_pos}


class LongContextReorder:
    """
    长上下文重排序器。

    缓解 LLM 的 "Lost in the Middle" 问题：
    将最重要的文档放在上下文窗口的首尾位置。
    """

    def __init__(
        self,
        strategy: ReorderStrategy = ReorderStrategy.U_SHAPED,
        max_tokens: int = 128000,
        head_ratio: float = 0.15,     # 开头的文档占比
        tail_ratio: float = 0.15,     # 结尾的文档占比
    ):
        """
        Args:
            strategy: 重排序策略
            max_tokens: 上下文窗口最大token数
            head_ratio: 放置在开头的文档占比（0~1）
            tail_ratio: 放置在结尾的文档占比（0~1）
        """
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.head_ratio = head_ratio
        self.tail_ratio = tail_ratio

    def reorder(self, documents: List[Document]) -> ReorderResult:
        """
        对文档进行重排序。

        Args:
            documents: 待排序的文档列表（假设已按分数排序，最高分在前）

        Returns:
            ReorderResult 包含重排序后的文档和元信息
        """
        if len(documents) <= 2:
            # 文档太少，不需要重排序
            return ReorderResult(
                documents=documents,
                strategy=self.strategy,
                original_order=list(range(len(documents))),
                position_map={i: i for i in range(len(documents))},
            )

        if self.strategy == ReorderStrategy.SCORE_DESCENDING:
            return self._score_descending(documents)

        elif self.strategy == ReorderStrategy.U_SHAPED:
            return self._u_shaped(documents)

        elif self.strategy == ReorderStrategy.INTERLEAVED:
            return self._interleaved(documents)

        elif self.strategy == ReorderStrategy.ALT_CAP:
            return self._alt_cap(documents)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    # ----------------------------------------------------------
    # U-Shaped 排序（最常用）
    # ----------------------------------------------------------

    def _u_shaped(self, documents: List[Document]) -> ReorderResult:
        """
        U型排列：
        分数: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        ->
        位置: [0.9,         0.7,         0.5, 0.4, 0.3, 0.2,         0.8, 0.6]
              [最高 -> 头部]             [最低 -> 中部]                 [次高 -> 尾部]
        """
        n = len(documents)
        n_head = max(1, int(n * self.head_ratio))
        n_tail = max(1, int(n * self.tail_ratio))
        n_middle = n - n_head - n_tail

        if n_middle < 0:
            # 文档太少，头部+尾部已经覆盖所有
            return ReorderResult(
                documents=documents,
                strategy=ReorderStrategy.U_SHAPED,
                original_order=list(range(n)),
                position_map={i: i for i in range(n)},
            )

        # 排序后的文档：按分数降序
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        # 分段
        head_docs = sorted_docs[:n_head]              # 最高分 -> 头部
        tail_docs = sorted_docs[n_head:n_head + n_tail]  # 次高分 -> 尾部
        middle_docs = sorted_docs[n_head + n_tail:]   # 剩余 -> 中部

        # 中间部分按分数降序（最不重要的放在最中间）
        # 可选：将middle_docs翻转让最低分在最中间
        # middle_docs = list(reversed(middle_docs))

        # 重组
        reordered = head_docs + middle_docs + list(reversed(tail_docs))

        original_order = [d.index for d in reordered]
        position_map = {new: old for new, old in enumerate(original_order)}

        return ReorderResult(
            documents=reordered,
            strategy=ReorderStrategy.U_SHAPED,
            original_order=original_order,
            position_map=position_map,
        )

    # ----------------------------------------------------------
    # Interleaved 交替排序
    # ----------------------------------------------------------

    def _interleaved(self, documents: List[Document]) -> ReorderResult:
        """
        交替排列：高分低分交替穿插
        分数: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        ->
        位置: [0.9, 0.4, 0.8, 0.5, 0.7, 0.6]
              [高,  低,  高,  低,  高,  低]
        """
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        n = len(sorted_docs)

        reordered = []
        left, right = 0, n - 1
        take_left = True

        while left <= right:
            if take_left:
                reordered.append(sorted_docs[left])
                left += 1
            else:
                reordered.append(sorted_docs[right])
                right -= 1
            take_left = not take_left

        original_order = [d.index for d in reordered]
        position_map = {new: old for new, old in enumerate(original_order)}

        return ReorderResult(
            documents=reordered,
            strategy=ReorderStrategy.INTERLEAVED,
            original_order=original_order,
            position_map=position_map,
        )

    # ----------------------------------------------------------
    # Alt-Cap 交替+首尾强化
    # ----------------------------------------------------------

    def _alt_cap(self, documents: List[Document]) -> ReorderResult:
        """
        Alt-Cap策略：
        1. 最高分 -> 开头
        2. 次高分 -> 结尾
        3. 剩余文档交替排列在中间

        分数: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        ->
        位置: [0.9,  0.7, 0.4, 0.6, 0.5,  0.8]
              [cap] [--- interleaved ---] [cap]
        """
        if len(documents) <= 3:
            return self._u_shaped(documents)

        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        # 最高分放开头，次高分放结尾
        head_cap = [sorted_docs[0]]
        tail_cap = [sorted_docs[1]]
        middle = sorted_docs[2:]

        # 中间交替排列
        middle_reordered = []
        left, right = 0, len(middle) - 1
        take_left = True
        while left <= right:
            if take_left:
                middle_reordered.append(middle[left])
                left += 1
            else:
                middle_reordered.append(middle[right])
                right -= 1
            take_left = not take_left

        reordered = head_cap + middle_reordered + tail_cap

        original_order = [d.index for d in reordered]
        position_map = {new: old for new, old in enumerate(original_order)}

        return ReorderResult(
            documents=reordered,
            strategy=ReorderStrategy.ALT_CAP,
            original_order=original_order,
            position_map=position_map,
        )

    # ----------------------------------------------------------
    # 纯分数降序（baseline）
    # ----------------------------------------------------------

    def _score_descending(self, documents: List[Document]) -> ReorderResult:
        """纯分数降序排列（不做Lost in Middle优化）"""
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        original_order = [d.index for d in sorted_docs]

        return ReorderResult(
            documents=sorted_docs,
            strategy=ReorderStrategy.SCORE_DESCENDING,
            original_order=original_order,
            position_map={i: i for i in range(len(documents))},
        )

    # ----------------------------------------------------------
    # Token预算感知的截断
    # ----------------------------------------------------------

    def fit_to_budget(
        self,
        documents: List[Document],
        reserved_tokens: int = 2000,   # 预留给system prompt和query的token
    ) -> List[Document]:
        """
        根据token预算截断文档列表。
        在重排序之后调用，确保不超过上下文窗口限制。

        Args:
            documents: 已排序的文档列表
            reserved_tokens: 预留给非文档内容的token数

        Returns:
            截断后的文档列表
        """
        available = self.max_tokens - reserved_tokens
        kept = []
        used = 0

        for doc in documents:
            doc_tokens = doc.token_count
            if used + doc_tokens <= available:
                kept.append(doc)
                used += doc_tokens
            else:
                # 尝试截断最后一个文档
                remaining = available - used
                if remaining > 100:  # 至少保留100 tokens的内容
                    truncated = Document(
                        content=doc.content[:remaining * 2],  # 字符数约=token数*2
                        score=doc.score,
                        metadata={**doc.metadata, "truncated": True},
                        index=doc.index,
                    )
                    kept.append(truncated)
                break

        return kept


# ============================================================
# 便捷函数
# ============================================================

def reorder_for_long_context(
    documents: List[Document],
    strategy: ReorderStrategy = ReorderStrategy.U_SHAPED,
    max_tokens: int = 128000,
) -> List[Document]:
    """一行调用：对文档进行Lost in Middle优化的重排序"""
    reorderer = LongContextReorder(strategy=strategy, max_tokens=max_tokens)
    result = reorderer.reorder(documents)
    fitted = reorderer.fit_to_budget(result.documents)
    return fitted


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 模拟8篇检索文档
    docs = [
        Document(content=f"Document {i}: " + "x" * 100, score=s, index=i)
        for i, s in enumerate([0.95, 0.88, 0.76, 0.65, 0.54, 0.43, 0.32, 0.21])
    ]

    reorderer = LongContextReorder(strategy=ReorderStrategy.U_SHAPED)

    result = reorderer.reorder(docs)

    print("Original scores:  ", [d.score for d in docs])
    print("Reordered scores:", [d.score for d in result.documents])
    print("Position map:    ", result.position_map)
    print()

    # 可视化位置映射
    print("Position | Original Index | Score")
    print("-" * 38)
    for new_pos, doc in enumerate(result.documents):
        marker = ""
        if new_pos == 0:
            marker = " <-- HEAD (primacy)"
        elif new_pos == len(result.documents) - 1:
            marker = " <-- TAIL (recency)"
        print(f"   {new_pos:2d}    |      {doc.index:2d}       | {doc.score:.2f}{marker}")
```

---

## 四、策略对比

| 策略 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **U-Shaped** | 通用场景，文档数>3 | 最大化首尾信息密度 | 中间文档几乎被牺牲 |
| **Interleaved** | 需要覆盖多篇文档 | 每篇文档都有一定曝光 | 最重要的文档可能不在最佳位置 |
| **Alt-Cap** | 最高2篇文档极其重要 | 保证top-2在首尾 | 实现稍复杂 |
| **Score Descending** | 短上下文(<4篇文档) | 最简单 | 不解决Lost in Middle |

---

## 五、关键设计决策

1. **何时不重排序** -- 如果只有2-3篇文档，重排序的收益不大，直接按分数降序即可。

2. **重排序与Reranker的关系** -- Reranker（重排序器）负责重新计算相关性分数；LongContextReorder负责根据分数调整文档位置。两者互补：先Rerank后Reorder。

3. **与Token预算的配合** -- 重排序后再做token截断，确保不超过上下文限制。被牺牲的中间文档是截断的首选目标。

4. **可视化调试** -- 在生产环境中，记录position_map用于分析LLM实际引用了哪些位置的文档，持续优化head_ratio和tail_ratio参数。

---

*上一篇：[02 JSON输出格式不稳定](02-output-format-instability.md)*  
*下一篇：[04 Token成本爆炸](04-cost-explosion.md)*
