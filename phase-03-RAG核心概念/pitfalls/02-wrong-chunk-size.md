# 陷阱 02：错误的文档分块大小

> 块太小则语义碎片化，块太大则信息稀释 —— 找到那个精确的平衡点。

---

## 1. 症状（Symptoms）

| 症状 | 表现 | 根本问题 |
|------|------|----------|
| **检索到不相关文档** | 用户问"违约责任"，返回的是"合同的订立"段落 | 块太大，嵌入向量被不相关内容稀释 |
| **关键信息被截断** | 一个完整的法律条文被切在两块中，每块单独看都没有完整意义 | 块太小，语义单元被物理分割 |
| **检索排名不稳定** | 同一个查询，有时返回文档 A 排第一，有时排第五 | 块边界恰好切在关键语义点上 |
| **LLM 回答片段化** | 答案只有片段信息，缺少上下文 | 检索返回的块没有足够的上下文窗口 |
| **相似度分数整体偏低** | top-5 的余弦相似度都低于 0.65 | 块大小与嵌入模型的最优输入长度不匹配 |

---

## 2. 根因（Root Cause）

### 2.1 分块问题的核心矛盾

```python
"""
分块大小的三元悖论：

          语义完整性
          /        \
         /          \
        /            \
  检索精度 ──────── 上下文丰富度

你只能优化其中两个，第三个必然受损。

- 小块（128 tokens）：检索精度高 + 语义完整性差 + 上下文少
- 大块（1024 tokens）：检索精度低 + 语义完整性好 + 上下文多
- 中等块（512 tokens）：三者的折中，但都不最优
"""
```

### 2.2 分块大小的数学分析

```python
import numpy as np
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class ChunkingAnalysis:
    """分析不同分块策略对检索质量的影响。"""

    chunk_sizes: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    overlap_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])

    def theoretical_analysis(self) -> Dict:
        """
        从理论角度分析分块大小对检索的影响。
        """
        results = {}

        for chunk_size in self.chunk_sizes:
            analysis = {}

            # 1. 信息密度分析
            # 小块：高密度但可能不完整
            # 大块：低密度但完整
            analysis["info_density"] = 1.0 / math.log(chunk_size + 1)

            # 2. 语义完整性概率（基于经验公式）
            # 一个自然段落的平均长度约为 200-400 中文字符
            # 中文法律条文平均长度约为 150-300 字符
            avg_semantic_unit_chinese = 250  # 字符
            # 假设 token/字符 比约为 1:2（中文）
            avg_semantic_unit_tokens = avg_semantic_unit_chinese / 2

            if chunk_size >= avg_semantic_unit_tokens:
                analysis["semantic_completeness"] = min(
                    1.0,
                    1.0 - math.exp(-chunk_size / avg_semantic_unit_tokens)
                )
            else:
                # 块太小，无法装下一个完整语义单元
                analysis["semantic_completeness"] = (
                    chunk_size / avg_semantic_unit_tokens
                )

            # 3. 检索特异性
            # 小块更精确但更容易丢失上下文
            analysis["retrieval_specificity"] = 1.0 / math.sqrt(chunk_size / 128)

            # 4. 噪声敏感度（大块更容易引入噪声）
            analysis["noise_sensitivity"] = math.log(chunk_size / 64) / math.log(32)

            # 5. 综合得分（加权）
            weights = {
                "info_density": 0.15,
                "semantic_completeness": 0.40,
                "retrieval_specificity": 0.30,
                "noise_sensitivity": -0.15,  # 负向指标
            }
            composite_score = sum(
                weights[k] * analysis.get(k, 0)
                for k in weights
            )
            analysis["composite_score"] = round(composite_score, 4)

            results[chunk_size] = analysis

        return results

    def plot_theoretical_curves(self):
        """生成分块大小影响的理论曲线数据。"""
        sizes = range(64, 4096, 64)
        data = []

        for s in sizes:
            completeness = 1.0 - math.exp(-s / 250)
            specificity = 1.0 / math.sqrt(s / 128)
            noise = math.log(s / 64 + 1) / math.log(32)
            composite = (
                0.40 * completeness
                + 0.30 * specificity
                - 0.15 * noise
                + 0.15 * (1.0 / math.log(s + 1))
            )

            data.append({
                "chunk_size": s,
                "semantic_completeness": round(completeness, 4),
                "retrieval_specificity": round(specificity, 4),
                "noise_sensitivity": round(noise, 4),
                "composite_score": round(composite, 4),
            })

        return data


analysis = ChunkingAnalysis()
theoretical_results = analysis.theoretical_analysis()

print("=" * 60)
print("分块大小的理论分析")
print("=" * 60)
for size, metrics in theoretical_results.items():
    print(f"\n块大小 {size} tokens:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

curve_data = analysis.plot_theoretical_curves()
print(f"\n生成理论曲线数据: {len(curve_data)} 个数据点")
```

---

## 3. 真实场景（Real-world Scenario）

### 场景：中文法律文档检索

```python
import hashlib
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


# ============================================================
# 真实法律文本语料
# ============================================================

CHINESE_LEGAL_CORPUS = {
    "civil_code_contract": """
《中华人民共和国民法典》第三编 合同

第四百六十三条 本编调整因合同产生的民事关系。
第四百六十四条 合同是民事主体之间设立、变更、终止民事法律关系的协议。
婚姻、收养、监护等有关身份关系的协议，适用有关该身份关系的法律规定；
没有规定的，可以根据其性质参照适用本编规定。

第四百六十五条 依法成立的合同，受法律保护。
依法成立的合同，仅对当事人具有法律约束力，但是法律另有规定的除外。

第四百六十六条 当事人对合同条款的理解有争议的，应当依据本法第一百四十二条第一款的规定，
确定争议条款的含义。
合同文本采用两种以上文字订立并约定具有同等效力的，对各文本使用的词句推定具有相同含义。
各文本使用的词句不一致的，应当根据合同的相关条款、性质、目的以及诚信原则等予以解释。

第五百七十七条 当事人一方不履行合同义务或者履行合同义务不符合约定的，
应当承担继续履行、采取补救措施或者赔偿损失等违约责任。

第五百七十八条 当事人一方明确表示或者以自己的行为表明不履行合同义务的，
对方可以在履行期限届满前请求其承担违约责任。

第五百七十九条 当事人一方未支付价款、报酬、租金、利息，
或者不履行其他金钱债务的，对方可以请求其支付。

第五百八十四条 当事人一方不履行合同义务或者履行合同义务不符合约定，
造成对方损失的，损失赔偿额应当相当于因违约所造成的损失，
包括合同履行后可以获得的利益；
但是，不得超过违约方订立合同时预见到或者应当预见到的因违约可能造成的损失。

第五百八十五条 当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，
也可以约定因违约产生的损失赔偿额的计算方法。
约定的违约金低于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以增加；
约定的违约金过分高于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以适当减少。
当事人就迟延履行约定违约金的，违约方支付违约金后，还应当履行债务。
""",

    "civil_code_tort": """
《中华人民共和国民法典》第七编 侵权责任

第一千一百六十五条 行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。
依照法律规定推定行为人有过错，其不能证明自己没有过错的，应当承担侵权责任。

第一千一百六十六条 行为人造成他人民事权益损害，不论行为人有无过错，
法律规定应当承担侵权责任的，依照其规定。

第一千一百六十七条 侵权行为危及他人人身、财产安全的，
被侵权人有权请求侵权人承担停止侵害、排除妨碍、消除危险等侵权责任。

第一千一百六十八条 二人以上共同实施侵权行为，造成他人损害的，应当承担连带责任。

第一千一百七十九条 侵害他人造成人身损害的，应当赔偿医疗费、护理费、交通费、营养费、
住院伙食补助费等为治疗和康复支出的合理费用，以及因误工减少的收入。
造成残疾的，还应当赔偿辅助器具费和残疾赔偿金；
造成死亡的，还应当赔偿丧葬费和死亡赔偿金。

第一千一百八十三条 侵害自然人人身权益造成严重精神损害的，
被侵权人有权请求精神损害赔偿。
因故意或者重大过失侵害自然人具有人身意义的特定物造成严重精神损害的，
被侵权人有权请求精神损害赔偿。
""",

    "company_law_2024": """
《中华人民共和国公司法》（2024年修订）

第二十三条 设立有限责任公司，应当具备下列条件：
（一）股东符合法定人数；
（二）有符合公司章程规定的全体股东认缴的出资额；
（三）股东共同制定公司章程；
（四）有公司名称，建立符合有限责任公司要求的组织机构；
（五）有公司住所。

第四十七条 有限责任公司的注册资本为在公司登记机关登记的全体股东认缴的出资额。
全体股东认缴的出资额由股东按照公司章程的规定自公司成立之日起五年内缴足。
法律、行政法规以及国务院决定对有限责任公司注册资本实缴、注册资本最低限额、
股东出资期限另有规定的，从其规定。

第四十八条 股东可以用货币出资，也可以用实物、知识产权、
土地使用权、股权、债权等可以用货币估价并可以依法转让的非货币财产作价出资；
但是，法律、行政法规规定不得作为出资的财产除外。

对作为出资的非货币财产应当评估作价，核实财产，不得高估或者低估作价。
法律、行政法规对评估作价有规定的，从其规定。
""",
}


# ============================================================
# 分块策略实现
# ============================================================

class TextChunker:
    """
    多种分块策略的实现。

    支持：
    - 固定大小分块（Fixed-size）
    - 句子边界感知分块（Sentence-aware）
    - 递归分块（Recursive）
    - 语义分块（Semantic）
    """

    def __init__(self):
        self.chinese_sentence_endings = re.compile(r'[。；！？\n]')

    def fixed_size_chunk(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0
    ) -> List[Dict]:
        """
        固定大小分块。

        这是最基础也最常用的分块方式。
        问题：可能在句子中间切分，破坏语义。
        """
        chunks = []
        text_len = len(text)
        start = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "chunk_id": f"fixed_{len(chunks):04d}",
                "size": len(chunk_text),
            })
            start = end - overlap

        return chunks

    def sentence_aware_chunk(
        self,
        text: str,
        target_size: int,
        overlap_sentences: int = 1
    ) -> List[Dict]:
        """
        基于句子边界的分块。

        先按句号/分号/换行切分句子，
        然后尽量在每个块中放入完整的句子，
        使块大小接近 target_size。
        """
        # 按句子分割
        raw_sentences = self.chinese_sentence_endings.split(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        chunks = []
        current_chunk_sentences = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            current_chunk_sentences.append(sentence)
            current_size += sentence_len

            # 当累积大小接近或超过 target_size 时，生成一个块
            if current_size >= target_size or i == len(sentences) - 1:
                chunk_text = "。".join(current_chunk_sentences) + "。"
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": f"sent_{len(chunks):04d}",
                    "size": len(chunk_text),
                    "num_sentences": len(current_chunk_sentences),
                })

                # 重叠：保留最后几个句子
                if overlap_sentences > 0:
                    overlap_sents = current_chunk_sentences[-overlap_sentences:]
                    current_chunk_sentences = overlap_sents
                    current_size = sum(len(s) for s in overlap_sents)
                else:
                    current_chunk_sentences = []
                    current_size = 0

        return chunks

    def recursive_chunk(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int = 0
    ) -> List[Dict]:
        """
        递归分块（类似 LangChain 的 RecursiveCharacterTextSplitter）。

        使用一组分隔符，从大分隔符到小分隔符逐级尝试切分：
        1. 先按双换行（段落）切分
        2. 再按单换行切分
        3. 再按句号切分
        4. 最后按字符切分
        """
        separators = ["\n\n", "\n", "。", "；", "，", ""]

        def _split_recursive(
            text_to_split: str,
            separators_list: List[str],
            depth: int = 0
        ) -> List[str]:
            if depth >= len(separators_list):
                # 最后一级：按字符切分
                return [text_to_split[i:i+chunk_size]
                        for i in range(0, len(text_to_split), chunk_size)]

            separator = separators_list[depth]
            if separator == "":
                return [text_to_split[i:i+chunk_size]
                        for i in range(0, len(text_to_split), chunk_size)]

            splits = text_to_split.split(separator)

            # 过滤掉空字符串
            splits = [s for s in splits if s.strip()]

            result = []
            for split_text in splits:
                if len(split_text) <= chunk_size:
                    result.append(split_text)
                else:
                    # 这一段还是太大，用下一级分隔符继续切分
                    result.extend(
                        _split_recursive(split_text, separators_list, depth + 1)
                    )

            return result

        split_texts = _split_recursive(text, separators)

        # 合并过小的块 + 处理重叠
        chunks = []
        current_chunk_parts = []
        current_size = 0

        for text_part in split_texts:
            current_chunk_parts.append(text_part)
            current_size += len(text_part)

            if current_size >= chunk_size:
                chunk_text = "".join(current_chunk_parts)
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": f"recursive_{len(chunks):04d}",
                    "size": len(chunk_text),
                })
                # 重叠处理
                if chunk_overlap > 0:
                    overlap_text = chunk_text[-chunk_overlap:]
                    current_chunk_parts = [overlap_text]
                    current_size = len(overlap_text)
                else:
                    current_chunk_parts = []
                    current_size = 0

        # 处理残余
        if current_chunk_parts:
            chunk_text = "".join(current_chunk_parts)
            chunks.append({
                "text": chunk_text,
                "chunk_id": f"recursive_{len(chunks):04d}",
                "size": len(chunk_text),
            })

        return chunks


# ============================================================
# 分块质量评估
# ============================================================

class ChunkingEvaluator:
    """
    评估不同分块策略的质量。

    评估维度：
    1. 语义完整性：每个块是否包含完整的语义单元
    2. 上下文窗口：块与相邻块的信息重叠度
    3. 边界合理性：块边界是否落在自然的语义断点上
    4. 大小均匀度：各块大小的标准差
    """

    def __init__(self):
        self.chunker = TextChunker()

    def evaluate_strategy(
        self,
        text: str,
        strategy_name: str,
        chunks: List[Dict],
        query: str,
        ground_truth_passage: str,
    ) -> Dict:
        """
        对一种分块策略进行完整评估。
        """
        if not chunks:
            return {"error": "No chunks produced"}

        sizes = [c["size"] for c in chunks]
        texts = [c["text"] for c in chunks]

        # 1. 大小均匀度
        size_std = float(np.std(sizes))
        size_mean = float(np.mean(sizes))
        size_uniformity = 1.0 - min(size_std / size_mean, 1.0) if size_mean > 0 else 0

        # 2. 语义完整性（模拟：检查每个块是否包含完整句子）
        complete_sentences_count = 0
        for chunk_text in texts:
            # 检查块是否以自然断句结尾
            if chunk_text.strip().endswith(("。", "；", "）", "。\n")):
                complete_sentences_count += 1
            # 也检查开头是否以自然方式开始
            first_chars = chunk_text.strip()[:5]
            mid_sentence_start = any(
                first_chars.startswith(w) for w in ["的", "了", "和", "与", "或", "等"]
            )
            if mid_sentence_start:
                complete_sentences_count -= 0.5

        semantic_completeness = (
            complete_sentences_count / len(chunks) if chunks else 0
        )

        # 3. 目标段落命中率（关键指标）
        # 检查 ground_truth_passage 是否完整地包含在某个块中
        passage_in_one_chunk = any(
            ground_truth_passage in chunk_text for chunk_text in texts
        )
        # 也检查是否有块包含了 ground_truth_passage 的大部分（>70%）
        max_overlap_ratio = 0.0
        for chunk_text in texts:
            # 使用最长公共子序列的简化版本
            common_chars = sum(
                1 for c in ground_truth_passage if c in chunk_text
            )
            overlap_ratio = (
                common_chars / len(ground_truth_passage)
                if ground_truth_passage else 0
            )
            max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        # 4. 噪声比：块中相关内容的占比
        if passage_in_one_chunk:
            # 找到包含 ground_truth 的块
            for chunk_text in texts:
                if ground_truth_passage in chunk_text:
                    signal_ratio = len(ground_truth_passage) / len(chunk_text)
                    break
            else:
                signal_ratio = 0.0
        else:
            signal_ratio = 0.0

        # 5. 综合得分
        composite = (
            0.30 * semantic_completeness
            + 0.35 * signal_ratio
            + 0.20 * (1.0 if passage_in_one_chunk else max_overlap_ratio)
            + 0.15 * size_uniformity
        )

        return {
            "strategy": strategy_name,
            "num_chunks": len(chunks),
            "size_mean": round(size_mean, 1),
            "size_std": round(size_std, 1),
            "size_uniformity": round(size_uniformity, 4),
            "semantic_completeness": round(semantic_completeness, 4),
            "passage_in_one_chunk": passage_in_one_chunk,
            "max_overlap_ratio": round(max_overlap_ratio, 4),
            "signal_ratio": round(signal_ratio, 4),
            "composite_score": round(composite, 4),
        }


# ============================================================
# 完整的对比实验
# ============================================================

def run_chunking_comparison():
    """
    在真实法律文本上对比不同分块策略。
    """
    evaluator = ChunkingEvaluator()
    chunker = TextChunker()

    # 使用合同编文本作为测试语料
    test_text = CHINESE_LEGAL_CORPUS["civil_code_contract"]

    # 模拟一个用户查询及其标准答案段落
    query = "违约责任的承担方式有哪些？"
    ground_truth = (
        "当事人一方不履行合同义务或者履行合同义务不符合约定的，"
        "应当承担继续履行、采取补救措施或者赔偿损失等违约责任。"
    )

    print("=" * 70)
    print("分块策略对比实验")
    print(f"测试语料大小: {len(test_text)} 字符")
    print(f"查询: {query}")
    print(f"标准答案: {ground_truth}")
    print("=" * 70)

    all_results = []

    # 实验 1: 固定大小分块（不同大小）
    for chunk_size in [128, 256, 512, 1024, 2048]:
        chunks = chunker.fixed_size_chunk(
            test_text, chunk_size=chunk_size, overlap=int(chunk_size * 0.1)
        )
        result = evaluator.evaluate_strategy(
            test_text,
            f"固定大小-{chunk_size}字符",
            chunks,
            query,
            ground_truth,
        )
        all_results.append(result)

    # 实验 2: 句子边界感知分块
    for target_size in [256, 512, 1024]:
        chunks = chunker.sentence_aware_chunk(
            test_text, target_size=target_size, overlap_sentences=1
        )
        result = evaluator.evaluate_strategy(
            test_text,
            f"句子感知-{target_size}字符",
            chunks,
            query,
            ground_truth,
        )
        all_results.append(result)

    # 实验 3: 递归分块
    for chunk_size in [256, 512, 1024]:
        chunks = chunker.recursive_chunk(
            test_text, chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.1)
        )
        result = evaluator.evaluate_strategy(
            test_text,
            f"递归分块-{chunk_size}字符",
            chunks,
            query,
            ground_truth,
        )
        all_results.append(result)

    # 打印结果
    print(f"\n{'策略':<25} {'块数':<6} {'平均大小':<10} {'语义完整':<10} "
          f"{'信号比':<10} {'命中':<8} {'综合得分':<10}")
    print("-" * 85)

    for r in sorted(all_results, key=lambda x: x["composite_score"], reverse=True):
        print(
            f"{r['strategy']:<25} "
            f"{r['num_chunks']:<6} "
            f"{r['size_mean']:<10.0f} "
            f"{r['semantic_completeness']:<10.3f} "
            f"{r['signal_ratio']:<10.3f} "
            f"{str(r['passage_in_one_chunk']):<8} "
            f"{r['composite_score']:<10.4f}"
        )

    return all_results


# 运行对比实验
comparison_results = run_chunking_comparison()
```

### 分块可视化：相同查询在不同块大小下的检索表现

```python
import math

def simulate_embedding_retrieval(
    query: str,
    chunks: List[str],
    chunk_sizes: List[int],
) -> Dict[str, List[float]]:
    """
    模拟嵌入检索过程。

    在实际系统中，这里会调用真实的 Embedding 模型。
    这里使用启发式方法模拟：
    - 小块：词汇匹配度高但语义不完整
    - 大块：词汇匹配度分散但语义完整
    """
    results = {"chunk_size": [], "best_similarity": [], "noise_level": []}

    for chunk_size in chunk_sizes:
        # 模拟不同块大小下的检索表现
        # 基于经验公式：
        # - 检索精度正比于 1/sqrt(chunk_size)（小块更精确）
        # - 但小块可能不包含完整答案

        # 模拟最佳相似度：小块高但语义不完整
        specificity = math.exp(-chunk_size / 400)
        completeness_factor = 1.0 - math.exp(-chunk_size / 250)

        # 模拟余弦相似度
        base_similarity = 0.60 + 0.15 * specificity
        adjusted_similarity = base_similarity * (0.5 + 0.5 * completeness_factor)

        # 噪声水平：大块噪声更多
        noise = 0.1 + 0.3 * (1.0 - math.exp(-chunk_size / 800))

        results["chunk_size"].append(chunk_size)
        results["best_similarity"].append(round(adjusted_similarity, 4))
        results["noise_level"].append(round(noise, 4))

    return results


def visualize_chunk_impact():
    """可视化分块大小对检索的影响。"""
    query = "违约责任承担方式"
    chunks = [
        "当事人一方不履行合同义务或",
        "者履行合同义务不符合约定",
        "的，应当承担继续履行、采",
        "取补救措施或者赔偿损失等",
        "违约责任。",
    ]
    chunk_sizes = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

    sim_results = simulate_embedding_retrieval(query, chunks, chunk_sizes)

    print("\n" + "=" * 60)
    print("分块大小对检索相似度的影响（模拟）")
    print("=" * 60)
    print(f"{'块大小(字符)':<15} {'最佳相似度':<15} {'噪声水平':<15} {'质量':<10}")
    print("-" * 55)

    for i in range(len(sim_results["chunk_size"])):
        size = sim_results["chunk_size"][i]
        sim = sim_results["best_similarity"][i]
        noise = sim_results["noise_level"][i]
        quality = "优" if sim > 0.80 and noise < 0.25 else (
            "良" if sim > 0.70 and noise < 0.35 else "差"
        )
        bar = "#" * int(sim * 30)
        print(f"{size:<15} {sim:<15.4f} {noise:<15.4f} {quality:<10} {bar}")

    return sim_results

vis_results = visualize_chunk_impact()
```

---

## 4. 修复方案（Fix with Code）

### 4.1 策略一：自适应分块 —— 基于文档结构动态选择块大小

```python
import re
from typing import List, Dict, Optional
from enum import Enum


class DocumentType(Enum):
    """文档类型枚举。"""
    LEGAL_CODE = "legal_code"          # 法条：有明确的条号结构
    LEGAL_JUDGMENT = "legal_judgment"  # 判决书：半结构化
    CONTRACT = "contract"              # 合同：条款结构
    PROSE = "prose"                    # 散文/叙述
    QA_PAIR = "qa_pair"                # 问答对
    TECHNICAL = "technical"            # 技术文档


class AdaptiveChunker:
    """
    自适应分块器。

    根据文档类型和内容特征，自动选择最优的块大小和分块策略。

    设计原则：
    - 法条：按"条"分块，保持条文完整性
    - 合同：按"条款"分块
    - 判决书：按"本院认为"等标题分块
    - 普通文本：基于语义密度的自适应大小
    """

    # 不同文档类型的建议块大小（字符数）
    TYPE_CONFIG = {
        DocumentType.LEGAL_CODE: {
            "min_chunk": 200,
            "max_chunk": 800,
            "overlap": 100,
            "strategy": "structural",
        },
        DocumentType.LEGAL_JUDGMENT: {
            "min_chunk": 400,
            "max_chunk": 1200,
            "overlap": 200,
            "strategy": "heading_aware",
        },
        DocumentType.CONTRACT: {
            "min_chunk": 300,
            "max_chunk": 1000,
            "overlap": 150,
            "strategy": "clause_aware",
        },
        DocumentType.PROSE: {
            "min_chunk": 500,
            "max_chunk": 1500,
            "overlap": 200,
            "strategy": "semantic_density",
        },
        DocumentType.TECHNICAL: {
            "min_chunk": 400,
            "max_chunk": 1200,
            "overlap": 150,
            "strategy": "heading_aware",
        },
    }

    def __init__(self):
        self.type_patterns = {
            DocumentType.LEGAL_CODE: [
                r'第[一二三四五六七八九十百千\d]+条',
                r'第[一二三四五六七八九十百千\d]+节',
                r'第[一二三四五六七八九十百千\d]+章',
            ],
            DocumentType.LEGAL_JUDGMENT: [
                r'本院认为',
                r'经审理查明',
                r'判决如下',
                r'裁定如下',
            ],
            DocumentType.CONTRACT: [
                r'第[一二三四五六七八九十百千\d]+条',
                r'\d+\.\d+',
            ],
        }

    def detect_document_type(self, text: str) -> DocumentType:
        """
        自动检测文档类型。
        """
        scores = {}
        for doc_type, patterns in self.type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text))
                score += matches
            scores[doc_type] = score

        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 2:
                return best_type

        return DocumentType.PROSE

    def chunk(self, text: str, doc_type: Optional[DocumentType] = None) -> List[Dict]:
        """
        自适应分块。
        """
        if doc_type is None:
            doc_type = self.detect_document_type(text)

        config = self.TYPE_CONFIG.get(doc_type, self.TYPE_CONFIG[DocumentType.PROSE])
        strategy = config["strategy"]
        min_chunk = config["min_chunk"]
        max_chunk = config["max_chunk"]
        overlap = config["overlap"]

        if strategy == "structural":
            return self._structural_chunk(text, min_chunk, max_chunk, overlap)
        elif strategy == "heading_aware":
            return self._heading_aware_chunk(text, min_chunk, max_chunk, overlap)
        elif strategy == "clause_aware":
            return self._clause_aware_chunk(text, min_chunk, max_chunk, overlap)
        else:
            return self._semantic_density_chunk(text, min_chunk, max_chunk, overlap)

    def _structural_chunk(
        self, text: str, min_size: int, max_size: int, overlap: int
    ) -> List[Dict]:
        """
        结构感知分块：适合法条。

        按"第X条"等结构标记切分，保证每个法条独立成块。
        如果一条太长，在条款内部按句号切分。
        """
        # 匹配"第N条"作为主要分割点
        article_pattern = re.compile(
            r'(第[一二三四五六七八九十百千\d]+条[^第]*)'
        )

        articles = article_pattern.findall(text)
        if not articles:
            # 没有找到结构标记，降级为句子感知分块
            return self._sentence_aware_fallback(text, min_size, max_size, overlap)

        chunks = []
        for i, article in enumerate(articles):
            article = article.strip()
            if len(article) <= max_size:
                chunks.append({
                    "text": article,
                    "chunk_id": f"article_{i:04d}",
                    "size": len(article),
                    "strategy": "structural",
                })
            else:
                # 文章太长，按句号进一步切分
                sentences = re.split(r'[。；]', article)
                current_chunk = ""
                sub_idx = 0
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_size:
                        current_chunk += sentence + "。"
                    else:
                        if current_chunk:
                            chunks.append({
                                "text": current_chunk,
                                "chunk_id": f"article_{i:04d}_sub{sub_idx}",
                                "size": len(current_chunk),
                                "strategy": "structural_sub",
                            })
                            sub_idx += 1
                        current_chunk = sentence + "。"

                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "chunk_id": f"article_{i:04d}_sub{sub_idx}",
                        "size": len(current_chunk),
                        "strategy": "structural_sub",
                    })

        return chunks

    def _heading_aware_chunk(
        self, text: str, min_size: int, max_size: int, overlap: int
    ) -> List[Dict]:
        """
        标题感知分块：适合判决书、技术文档。

        按"本院认为""经审理查明"等标题词切分。
        """
        # 查找所有标题位置
        heading_pattern = re.compile(
            r'(?:^|\n)([^\n]{1,30}(?:：|:))'
        )
        positions = [0]
        for match in heading_pattern.finditer(text):
            positions.append(match.end())

        if len(positions) == 1:
            return self._sentence_aware_fallback(text, min_size, max_size, overlap)

        chunks = []
        positions.append(len(text))
        for i in range(len(positions) - 1):
            section_text = text[positions[i]:positions[i+1]].strip()
            if len(section_text) <= max_size:
                chunks.append({
                    "text": section_text,
                    "chunk_id": f"heading_{i:04d}",
                    "size": len(section_text),
                    "strategy": "heading_aware",
                })
            else:
                # 太大，进一步切分
                sub_chunks = self._sentence_aware_fallback(
                    section_text, min_size, max_size, overlap
                )
                for sc in sub_chunks:
                    sc["strategy"] = "heading_aware_sub"
                chunks.extend(sub_chunks)

        return chunks

    def _clause_aware_chunk(
        self, text: str, min_size: int, max_size: int, overlap: int
    ) -> List[Dict]:
        """
        条款感知分块：适合合同。
        """
        # 按"第X条"或编号"1.1"切分
        clause_pattern = re.compile(
            r'(?:第[一二三四五六七八九十百千\d]+条|\d+\.\d+\s)'
        )

        parts = clause_pattern.split(text)
        if len(parts) <= 1:
            return self._sentence_aware_fallback(text, min_size, max_size, overlap)

        matches = clause_pattern.findall(text)
        chunks = []
        for i, (match, part) in enumerate(zip(matches, parts[1:])):
            full_clause = match + part
            chunks.append({
                "text": full_clause.strip(),
                "chunk_id": f"clause_{i:04d}",
                "size": len(full_clause),
                "strategy": "clause_aware",
            })

        return chunks

    def _semantic_density_chunk(
        self, text: str, min_size: int, max_size: int, overlap: int
    ) -> List[Dict]:
        """
        语义密度分块：适合散文等非结构化文本。

        核心思路：
        1. 先按句子切分
        2. 计算相邻句子的语义相似度（简化版：词汇重叠率）
        3. 在语义断点处（低相似度）切分
        """
        sentences = re.split(r'[。！？；\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # 计算相邻句子间的"粘合度"（使用词汇重叠作为简化语义相似度）
        def sentence_cohesion(s1: str, s2: str) -> float:
            """计算两个句子之间的粘合度。"""
            s1_chars = set(s1)
            s2_chars = set(s2)
            if not s1_chars or not s2_chars:
                return 0.0
            intersection = len(s1_chars & s2_chars)
            union = len(s1_chars | s2_chars)
            return intersection / union if union > 0 else 0.0

        # 构建粘合度数组
        cohesions = []
        for i in range(len(sentences) - 1):
            coh = sentence_cohesion(sentences[i], sentences[i + 1])
            cohesions.append(coh)

        # 计算动态阈值：低于阈值的点为语义断点
        if cohesions:
            mean_cohesion = float(np.mean(cohesions))
            std_cohesion = float(np.std(cohesions)) if len(cohesions) > 1 else 0.0
            threshold = mean_cohesion - 0.5 * std_cohesion
        else:
            threshold = 0.3

        # 在语义断点处切分，同时遵守大小限制
        chunks = []
        current_chunk_sents = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            sent_len = len(sentence)
            current_chunk_sents.append(sentence)
            current_size += sent_len

            # 判断是否应该在此切分
            should_split = False
            if current_size >= max_size:
                should_split = True
            elif i < len(sentences) - 1 and current_size >= min_size:
                # 检查是否是语义断点
                if cohesions[i] < threshold:
                    should_split = True

            if should_split:
                chunk_text = "。".join(current_chunk_sents) + "。"
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": f"semantic_{len(chunks):04d}",
                    "size": len(chunk_text),
                    "strategy": "semantic_density",
                })
                # 重叠
                if overlap > 0:
                    overlap_text = chunk_text[-overlap:]
                    overlap_sents = re.split(r'[。！？；\n]', overlap_text)
                    current_chunk_sents = [s for s in overlap_sents if s.strip()]
                    current_size = sum(len(s) for s in current_chunk_sents)
                else:
                    current_chunk_sents = []
                    current_size = 0

        # 残余
        if current_chunk_sents:
            chunk_text = "。".join(current_chunk_sents) + "。"
            chunks.append({
                "text": chunk_text,
                "chunk_id": f"semantic_{len(chunks):04d}",
                "size": len(chunk_text),
                "strategy": "semantic_density",
            })

        return chunks

    def _sentence_aware_fallback(
        self, text: str, min_size: int, max_size: int, overlap: int
    ) -> List[Dict]:
        """句子感知分块的兜底实现。"""
        sentences = re.split(r'[。！？；\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) <= max_size:
                current += sentence + "。"
            else:
                if current:
                    chunks.append({
                        "text": current,
                        "chunk_id": f"sentence_{len(chunks):04d}",
                        "size": len(current),
                        "strategy": "sentence_fallback",
                    })
                current = sentence + "。"
        if current:
            chunks.append({
                "text": current,
                "chunk_id": f"sentence_{len(chunks):04d}",
                "size": len(current),
                "strategy": "sentence_fallback",
            })

        return chunks


# ============================================================
# 自适应分块演示
# ============================================================

def demo_adaptive_chunking():
    """演示自适应分块在真实法律文本上的效果。"""
    chunker = AdaptiveChunker()

    # 测试法条文本
    legal_text = CHINESE_LEGAL_CORPUS["civil_code_contract"]
    detected_type = chunker.detect_document_type(legal_text)

    print("\n" + "=" * 60)
    print("自适应分块演示")
    print("=" * 60)
    print(f"检测到的文档类型: {detected_type.value}")

    chunks = chunker.chunk(legal_text)

    print(f"\n生成 {len(chunks)} 个块:")
    for chunk in chunks:
        print(f"  [{chunk['chunk_id']}] "
              f"大小={chunk['size']} 字符 "
              f"策略={chunk.get('strategy', 'unknown')} "
              f"预览={chunk['text'][:60]}...")

    return chunks

adaptive_chunks = demo_adaptive_chunking()
```

### 4.2 策略二：基于检索评估的块大小调优

```python
from typing import List, Dict, Callable


class ChunkSizeOptimizer:
    """
    通过实验找到最优块大小。

    方法：
    1. 准备一组评估查询 + 标准答案对
    2. 对每种块大小进行分块 → 嵌入 → 检索
    3. 计算 Recall@k 和 Precision@k
    4. 选择得分最高的块大小
    """

    def __init__(
        self,
        text_corpus: str,
        eval_queries: List[Dict],
        embed_fn: Callable = None,  # 在实际使用中传入真实的嵌入函数
    ):
        self.text_corpus = text_corpus
        self.eval_queries = eval_queries  # [{"query": "", "ground_truth": ""}, ...]
        self.embed_fn = embed_fn
        self.chunker = TextChunker()

    def evaluate_chunk_size(
        self,
        chunk_size: int,
        k: int = 5,
    ) -> Dict:
        """
        评估某个块大小下的检索质量。
        """
        # 分块
        chunks = self.chunker.sentence_aware_chunk(
            self.text_corpus,
            target_size=chunk_size,
            overlap_sentences=1,
        )
        chunk_texts = [c["text"] for c in chunks]

        # 如果没有真实的嵌入函数，使用模拟评估
        if self.embed_fn is None:
            return self._simulate_evaluation(chunks, chunk_texts, chunk_size, k)
        else:
            return self._real_evaluation(chunks, chunk_texts, chunk_size, k)

    def _simulate_evaluation(
        self,
        chunks: List[Dict],
        chunk_texts: List[str],
        chunk_size: int,
        k: int,
    ) -> Dict:
        """
        模拟检索评估。

        基于启发式方法估计检索质量，适用于无法使用真实嵌入模型的场景。
        """
        total_recall = 0.0
        total_precision = 0.0
        total_queries = len(self.eval_queries)

        for eval_item in self.eval_queries:
            query = eval_item["query"]
            ground_truth = eval_item["ground_truth"]

            # 找出哪些块包含 ground_truth
            relevant_chunks = set()
            for i, chunk_text in enumerate(chunk_texts):
                # 使用 Jaccard 相似度模拟 embedding 相似度
                query_chars = set(query)
                chunk_chars = set(chunk_text)
                intersection = len(query_chars & chunk_chars)
                union = len(query_chars | chunk_chars)
                sim = intersection / union if union > 0 else 0.0

                # 检查是否实际包含 ground_truth
                if ground_truth in chunk_text:
                    relevant_chunks.add(i)

                # 这里在实际中会用 embedding + top-k
                # 模拟：大块更容易"命中"但精度低

            # 模拟 top-k 检索
            # 基于经验：小块更精确但容易遗漏
            retrieval_hit_probability = 1.0 - np.exp(-chunk_size / 300)

            # 受 chunk_size 影响的"有效 k"
            effective_k = min(k, len(chunks))
            # 模拟找到的"相关文档"数
            simulated_hits = int(effective_k * retrieval_hit_probability * 0.7)
            actual_relevant = len(relevant_chunks) if relevant_chunks else 1

            hit = 1 if simulated_hits > 0 and relevant_chunks else 0
            total_recall += hit
            # Precision: 找到的相关 / 返回的总数
            precision = hit / effective_k if effective_k > 0 else 0
            total_precision += precision

        recall_at_k = total_recall / total_queries if total_queries > 0 else 0
        precision_at_k = total_precision / total_queries if total_queries > 0 else 0
        f1 = (
            2 * recall_at_k * precision_at_k / (recall_at_k + precision_at_k)
            if (recall_at_k + precision_at_k) > 0 else 0
        )

        return {
            "chunk_size": chunk_size,
            "num_chunks": len(chunks),
            "recall_at_k": round(recall_at_k, 4),
            "precision_at_k": round(precision_at_k, 4),
            "f1_score": round(f1, 4),
        }

    def grid_search(
        self,
        chunk_sizes: List[int] = None,
        k: int = 5,
    ) -> List[Dict]:
        """
        网格搜索最优块大小。
        """
        if chunk_sizes is None:
            chunk_sizes = [128, 200, 256, 300, 384, 400, 512, 600, 768, 1024]

        results = []
        for size in chunk_sizes:
            result = self.evaluate_chunk_size(size, k=k)
            results.append(result)

        return sorted(results, key=lambda x: x["f1_score"], reverse=True)

    def _real_evaluation(
        self,
        chunks: List[Dict],
        chunk_texts: List[str],
        chunk_size: int,
        k: int,
    ) -> Dict:
        """
        使用真实嵌入模型的评估（需要嵌入函数）。
        """
        # 实际实现会使用 embedding 模型
        # 这里只是一个接口定义
        raise NotImplementedError(
            "请传入 embed_fn 参数以使用真实嵌入模型进行评估。"
        )


# ============================================================
# 块大小优化演示
# ============================================================

def demo_chunk_size_optimization():
    """演示块大小优化过程。"""

    # 准备评估集
    eval_queries = [
        {
            "query": "违约责任的承担方式",
            "ground_truth": (
                "应当承担继续履行、采取补救措施或者赔偿损失等违约责任。"
            ),
        },
        {
            "query": "合同的定义",
            "ground_truth": (
                "合同是民事主体之间设立、变更、终止民事法律关系的协议。"
            ),
        },
        {
            "query": "违约金过高怎么办",
            "ground_truth": (
                "约定的违约金过分高于造成的损失的，"
                "人民法院或者仲裁机构可以根据当事人的请求予以适当减少。"
            ),
        },
        {
            "query": "侵权行为造成人身损害要赔偿什么",
            "ground_truth": (
                "应当赔偿医疗费、护理费、交通费、营养费、"
                "住院伙食补助费等为治疗和康复支出的合理费用，以及因误工减少的收入。"
            ),
        },
    ]

    optimizer = ChunkSizeOptimizer(
        text_corpus=(
            CHINESE_LEGAL_CORPUS["civil_code_contract"]
            + CHINESE_LEGAL_CORPUS["civil_code_tort"]
        ),
        eval_queries=eval_queries,
    )

    # 网格搜索
    print("\n" + "=" * 60)
    print("块大小网格搜索结果")
    print("=" * 60)

    results = optimizer.grid_search(
        chunk_sizes=[128, 200, 256, 300, 384, 400, 512, 600, 768, 1024],
        k=5,
    )

    print(f"{'块大小':<10} {'块数':<8} {'Recall@5':<12} {'Precision@5':<14} {'F1':<10}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['chunk_size']:<10} "
            f"{r['num_chunks']:<8} "
            f"{r['recall_at_k']:<12.4f} "
            f"{r['precision_at_k']:<14.4f} "
            f"{r['f1_score']:<10.4f}"
        )

    best = results[0]
    print(f"\n最优块大小: {best['chunk_size']} 字符 (F1={best['f1_score']:.4f})")

    return results

optimization_results = demo_chunk_size_optimization()
```

---

## 5. 检查清单（Checklist）

### 分块策略设计

- [ ] 是否根据文档类型选择了合适的分块策略？（法条按条、合同按条款、散文按语义）
- [ ] 块大小是否在 256-1024 字符之间？（中文场景的经验最优区间）
- [ ] 是否有块间重叠（overlap）来避免关键信息在边界处被截断？
- [ ] 是否在句子/段落边界处切分而非在字符中间？
- [ ] 是否对分块结果做了质量检查（完整性、大小均匀度）？

### 检索质量验证

- [ ] 是否在评估集上测试了不同块大小的 Recall@k 和 Precision@k？
- [ ] top-k 检索返回的文档中，实际包含答案的比例是否 > 80%？
- [ ] 对于关键查询，是否检查了完整答案是否在**同一个块**中？
- [ ] 是否监控了"查全率突然下降"的异常？（可能是新增文档类型不适合当前分块策略）

### 运维与迭代

- [ ] 是否有自动化测试在每次分块策略变更后运行评估？
- [ ] 是否记录了每次分块配置变更和对应的检索指标变化？
- [ ] 是否定期对新增文档进行分块质量抽检？

---

> **核心教训**：块大小没有"万能值"。最优块大小取决于文档类型、嵌入模型的选择、以及用户查询的典型长度。
> 投入时间做好分块策略的评估，比事后修复检索失败的成本低得多。
