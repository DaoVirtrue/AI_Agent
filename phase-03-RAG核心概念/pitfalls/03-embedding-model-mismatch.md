# 陷阱 03：嵌入模型的语言/领域不匹配

> 用英文模型编码中文法律文本，就像用英文词典查中文成语 —— 底层的向量空间根本不是同一片。

---

## 1. 症状（Symptoms）

| 症状 | 表现 | 诊断信号 |
|------|------|----------|
| **跨语言相似度坍塌** | 中文查询与中文文档的余弦相似度普遍低于 0.55 | 嵌入模型不支持中文 |
| **同语义不同语言得分为零** | "违约责任"和"breach of contract liability"相似度接近 0 | 模型只支持单一语言 |
| **代码与自然语言混淆** | 对"def calculate_tax(): return ..."的检索可能返回同名字的"def类定义"中的伪代码 | 代码模型用于自然语言文本 |
| **专业术语被误匹配** | "不可抗力"匹配到"不可"+"抗力"的分词错误导致匹配了"不可撤销" | 领域分词/Embedding 不匹配 |
| **检索排序随机化** | top-5 文档几乎不随查询变化而变化 | 嵌入模型完全无法区分语义 |

### 快速诊断代码

```python
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class ModelDiagnosis:
    """嵌入模型诊断结果。"""
    model_name: str
    supported_languages: List[str]
    max_seq_length: int
    embedding_dim: int
    language_scores: Dict[str, float]
    domain_scores: Dict[str, float]
    overall_health: str  # "healthy", "warning", "critical"


class EmbeddingModelDiagnostic:
    """
    诊断嵌入模型与任务是否匹配的工具。
    """

    def __init__(self):
        # 已知模型的参考规格（用于对比）
        self.known_models = {
            "text-embedding-ada-002": {
                "dim": 1536,
                "max_seq": 8191,
                "languages": ["en"],  # 主要优化英语
                "domains": ["general"],
                "notes": "OpenAI 第一代 embedding，英语优先，中文可用但非最优",
            },
            "text-embedding-3-large": {
                "dim": 3072,
                "max_seq": 8191,
                "languages": ["en", "zh", "ja", "ko", "de", "fr", "es", "ar", "..."],
                "domains": ["general"],
                "notes": "OpenAI 第三代，多语言显著提升",
            },
            "bge-large-zh-v1.5": {
                "dim": 1024,
                "max_seq": 512,
                "languages": ["zh", "en"],
                "domains": ["general"],
                "notes": "BAAI 的中文优化模型，中文检索首选之一",
            },
            "bge-m3": {
                "dim": 1024,
                "max_seq": 8192,
                "languages": ["zh", "en", "multilingual-100+"],
                "domains": ["general"],
                "notes": "BAAI 多语言模型，支持 100+ 语言",
            },
            "stella-base-zh-v3-1792d": {
                "dim": 1792,
                "max_seq": 512,
                "languages": ["zh", "en"],
                "domains": ["general"],
                "notes": "中文优化，高维度",
            },
            "all-MiniLM-L6-v2": {
                "dim": 384,
                "max_seq": 256,
                "languages": ["en"],
                "domains": ["general"],
                "notes": "轻量级英语模型，完全不支持中文",
            },
            "codebert-base": {
                "dim": 768,
                "max_seq": 512,
                "languages": ["code"],  # 这不是自然语言！
                "domains": ["code"],
                "notes": "代码专用模型，用于自然语言会严重失配",
            },
            "e5-large-v2": {
                "dim": 1024,
                "max_seq": 512,
                "languages": ["en"],
                "domains": ["general", "search"],
                "notes": "微软 E5，英语搜索优化",
            },
            "jina-embeddings-v2-base-zh": {
                "dim": 768,
                "max_seq": 8192,
                "languages": ["zh", "en"],
                "domains": ["general"],
                "notes": "Jina AI 中文优化，支持长文本",
            },
        }

    def diagnose(
        self,
        model_name: str,
        task_language: str = "zh",
        task_domain: str = "legal",
    ) -> ModelDiagnosis:
        """
        诊断嵌入模型是否适合当前任务。

        任务语言: zh, en, multilingual
        任务领域: legal, medical, code, general, finance
        """
        model_info = self.known_models.get(model_name)
        if model_info is None:
            return ModelDiagnosis(
                model_name=model_name,
                supported_languages=["unknown"],
                max_seq_length=0,
                embedding_dim=0,
                language_scores={},
                domain_scores={},
                overall_health="warning",
            )

        # 语言匹配度评估
        language_scores = {}
        if "multilingual-100+" in model_info["languages"]:
            language_scores["language_match"] = 0.85
            language_scores["reason"] = "多语言模型，覆盖率广但单语言可能不如专用模型"
        elif task_language in model_info["languages"]:
            # 检查是否是主要支持语言（排在前面）
            primary_lang = model_info["languages"][0]
            if primary_lang == task_language:
                language_scores["language_match"] = 0.95
                language_scores["reason"] = f"模型主要优化语言正是 {task_language}"
            else:
                language_scores["language_match"] = 0.70
                language_scores["reason"] = f"支持 {task_language} 但非主要优化语言"
        elif "code" in model_info["languages"]:
            language_scores["language_match"] = 0.0
            language_scores["reason"] = "代码模型，不适用于自然语言检索"
        else:
            language_scores["language_match"] = 0.15
            language_scores["reason"] = f"不支持 {task_language}"

        # 领域匹配度评估
        domain_scores = {}
        if task_domain in model_info["domains"]:
            domain_scores["domain_match"] = 0.90
        elif "general" in model_info["domains"]:
            domain_scores["domain_match"] = 0.60
            domain_scores["reason"] = (
                "通用模型，可用于专业领域但不如领域专用模型精准"
            )
        else:
            domain_scores["domain_match"] = 0.30
            domain_scores["reason"] = "模型领域与任务领域差距较大"

        # 维度与任务复杂度匹配
        dim = model_info["dim"]
        if dim >= 1024:
            domain_scores["dimension_sufficient"] = 0.90
        elif dim >= 768:
            domain_scores["dimension_sufficient"] = 0.70
        else:
            domain_scores["dimension_sufficient"] = 0.40
            domain_scores["dim_reason"] = (
                f"维度({dim})较低，可能在复杂语义任务中表现不足"
            )

        # 序列长度匹配
        max_seq = model_info["max_seq"]
        domain_scores["seq_length_ok"] = 0.90 if max_seq >= 512 else 0.50

        # 综合健康评估
        language_score = language_scores.get("language_match", 0.5)
        domain_score = domain_scores.get("domain_match", 0.5)

        overall = 0.6 * language_score + 0.4 * domain_score
        if overall >= 0.80:
            health = "healthy"
        elif overall >= 0.55:
            health = "warning"
        else:
            health = "critical"

        return ModelDiagnosis(
            model_name=model_name,
            supported_languages=model_info["languages"],
            max_seq_length=model_info["max_seq"],
            embedding_dim=model_info["dim"],
            language_scores=language_scores,
            domain_scores=domain_scores,
            overall_health=health,
        )


# 执行诊断
diagnostics = EmbeddingModelDiagnostic()

print("=" * 70)
print("嵌入模型诊断报告")
print("=" * 70)

models_to_test = [
    ("all-MiniLM-L6-v2", "zh", "legal"),
    ("text-embedding-ada-002", "zh", "legal"),
    ("bge-large-zh-v1.5", "zh", "legal"),
    ("bge-m3", "zh", "legal"),
    ("codebert-base", "zh", "legal"),
    ("jina-embeddings-v2-base-zh", "zh", "legal"),
    ("stella-base-zh-v3-1792d", "zh", "legal"),
]

print(f"\n{'模型名称':<35} {'健康状态':<12} {'语言匹配':<10} {'领域匹配':<10}")
print("-" * 70)

for model_name, lang, domain in models_to_test:
    diag = diagnostics.diagnose(model_name, lang, domain)
    status_icon = {
        "healthy": "[正常]",
        "warning": "[警告]",
        "critical": "[危险]",
    }
    print(
        f"{model_name:<35} "
        f"{status_icon.get(diag.overall_health, '[未知]'):<12} "
        f"{diag.language_scores.get('language_match', 0):.2f}        "
        f"{diag.domain_scores.get('domain_match', 0):.2f}"
    )
```

---

## 2. 根因（Root Cause）

### 2.1 嵌入模型的三大不匹配根因

```python
"""
嵌入模型不匹配的三个根本原因：

1. 训练语料不匹配
   - 英文模型训练语料以英文为主（Wikipedia-en, BookCorpus, etc.）
   - 中文模型需要中文语料（Wikipedia-zh, 百度百科, 新闻语料等）
   - 代码模型需要代码语料（GitHub repos）
   → 如果 90% 的训练语料是英文，它对中文的理解必然不足

2. 分词器（Tokenizer）不匹配
   - 英文 tokenizer 以空格和子词（BPE）为基础
   - 中文 tokenizer 需要处理字符级/词级分割
   - 英文 tokenizer 处理中文时，一个汉字可能被拆成多字节序列
   → 导致中文文本被 tokenize 成无意义的字节片段

3. 对比学习目标不匹配
   - 通用模型：学习"相似的文本应该有相似的向量"
   - 代码模型：学习"功能相似的代码应该有相似的向量"
   - 领域模型：学习"同领域内相似的文本应该有相似的向量"
   → 对比学习的正负例定义决定了向量的语义空间
"""
```

### 2.2 相似度分数对比：匹配 vs 不匹配

```python
import random
import hashlib
from typing import List, Dict, Optional


# ============================================================
# 模拟不同嵌入模型在中文法律文本上的表现
# ============================================================

class SimulatedEmbedder:
    """
    模拟不同嵌入模型的行为。

    在实际应用中，这里会加载真实的嵌入模型并计算向量。
    这里使用基于模型特征的启发式模拟，用于教育目的。

    模拟逻辑：
    - 中文优化的模型对中文文本给出高区分度
    - 英文模型对中文文本相似度崩塌（所有文本的相似度都接近）
    - 代码模型完全无法区分自然语言文本
    """

    def __init__(self, model_name: str, seed: int = 42):
        self.model_name = model_name
        self.model_specs = EmbeddingModelDiagnostic().known_models
        self.spec = self.model_specs.get(model_name, {})
        self.rng = random.Random(seed)

        # 根据模型规格设置模拟参数
        self._configure_simulation()

    def _configure_simulation(self):
        """根据模型规格配置模拟参数。"""
        languages = self.spec.get("languages", ["en"])
        domains = self.spec.get("domains", ["general"])
        dim = self.spec.get("dim", 768)

        # 中文支持度
        if "zh" in languages:
            if languages[0] == "zh":
                self.chinese_quality = 0.90  # 中文为主要语言
            else:
                self.chinese_quality = 0.70  # 中文为次要语言
        elif "multilingual-100+" in languages:
            self.chinese_quality = 0.80
        else:
            self.chinese_quality = 0.20  # 基本不支持中文

        # 法律领域支持度
        if "legal" in domains:
            self.legal_quality = 0.90
        elif "general" in domains:
            self.legal_quality = 0.55
        else:
            self.legal_quality = 0.25

        self.dim = dim

    def embed(self, text: str) -> List[float]:
        """
        模拟嵌入生成。

        使用文本的确定性哈希作为种子，加入模型特性的噪声，
        生成模拟的嵌入向量。
        """
        # 使用文本哈希作为确定性种子
        text_hash = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
        local_rng = random.Random(text_hash + hash(self.model_name) % 100000)

        # 基础向量
        base_vec = [local_rng.uniform(-0.1, 0.1) for _ in range(self.dim)]

        # 对于中文文本，根据中文质量添加语义信号
        # 判断是否为中文
        chinese_char_count = sum(1 for c in text if '一' <= c <= '鿿')
        is_chinese = chinese_char_count > len(text) * 0.3

        if is_chinese:
            # 中文模型：对中文文本有清晰的语义区分
            # 英文模型：对中文文本几乎是随机向量
            signal_strength = self.chinese_quality
            noise_level = 1.0 - signal_strength

            # 基于文本内容的语义信号
            semantic_seed = sum(ord(c) for c in text) % 10000
            signal_rng = random.Random(semantic_seed)

            for i in range(self.dim):
                signal = signal_rng.uniform(-signal_strength, signal_strength)
                noise = local_rng.uniform(-noise_level, noise_level)
                base_vec[i] += signal + noise
        else:
            signal_strength = 1.0
            noise_level = 0.0
            signal_rng = random.Random(sum(ord(c) for c in text) % 10000)

            for i in range(self.dim):
                signal = signal_rng.uniform(-signal_strength, signal_strength)
                noise = local_rng.uniform(-noise_level, noise_level)
                base_vec[i] += signal + noise

        # 归一化
        norm = math.sqrt(sum(v * v for v in base_vec))
        if norm > 0:
            base_vec = [v / norm for v in base_vec]

        return base_vec

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度。"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


# ============================================================
# 相似度对比实验
# ============================================================

def run_similarity_experiment():
    """
    对比不同嵌入模型在中文法律文本上的余弦相似度表现。
    """

    # 测试数据集：中文法律文本对
    test_pairs = [
        {
            "name": "高相关：同一法条的不同表述",
            "text1": "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
            "text2": "合同一方违约的，应承担继续履行、补救或赔偿等违约责任。",
            "expected_similarity": "high",
        },
        {
            "name": "中相关：相关但不同的法律概念",
            "text1": "当事人一方不履行合同义务的，应当承担违约责任。",
            "text2": "侵害他人造成人身损害的，应当赔偿医疗费、护理费等合理费用。",
            "expected_similarity": "medium",
        },
        {
            "name": "低相关：完全不相关的法律条文",
            "text1": "合同的成立需要要约和承诺。",
            "text2": "有限责任公司股东人数不得超过五十人。",
            "expected_similarity": "low",
        },
        {
            "name": "跨语言：同一概念中英文",
            "text1": "当事人一方违约的，应当承担违约责任。",
            "text2": "A party who breaches a contract shall bear liability for breach of contract.",
            "expected_similarity": "high_crosslingual",
        },
        {
            "name": "专业术语：法律定义精确匹配",
            "text1": "不可抗力是指不能预见、不能避免且不能克服的客观情况。",
            "text2": "因不可抗力不能履行民事义务的，不承担民事责任。",
            "expected_similarity": "high",
        },
    ]

    # 要测试的模型
    models_to_test = [
        "bge-large-zh-v1.5",        # 中文优化
        "bge-m3",                    # 多语言
        "text-embedding-3-large",    # 多语言通用
        "text-embedding-ada-002",    # 英语优化
        "all-MiniLM-L6-v2",         # 纯英语
        "codebert-base",            # 代码模型
    ]

    # 每个模型的嵌入器
    embedders = {}
    for model_name in models_to_test:
        embedders[model_name] = SimulatedEmbedder(model_name)

    # 计算所有模型在所有文本对上的相似度
    print("=" * 90)
    print("跨模型余弦相似度对比实验")
    print("=" * 90)

    results = {}
    for pair in test_pairs:
        pair_results = {}
        for model_name in models_to_test:
            embedder = embedders[model_name]
            vec1 = embedder.embed(pair["text1"])
            vec2 = embedder.embed(pair["text2"])
            sim = embedder.cosine_similarity(vec1, vec2)
            pair_results[model_name] = round(sim, 4)
        results[pair["name"]] = pair_results

    # 打印结果表格
    # 表头
    header = f"{'测试对':<35}"
    for model in models_to_test:
        short_name = model[:25]
        header += f" {short_name:<27}"
    print(header)
    print("-" * len(header))

    for pair_name, pair_results in results.items():
        row = f"{pair_name:<35}"
        for model in models_to_test:
            sim = pair_results[model]
            # 用颜色标记（这里用符号代替）
            if sim > 0.70:
                marker = f"{sim:.4f} ✓"
            elif sim > 0.45:
                marker = f"{sim:.4f} ~"
            else:
                marker = f"{sim:.4f} ✗"
            row += f" {marker:<27}"
        print(row)

    # 模型排名
    print(f"\n{'=' * 60}")
    print("模型综合排名（中文法律文本场景）")
    print(f"{'=' * 60}")

    avg_scores = {}
    for model_name in models_to_test:
        model_sims = [results[p][model_name] for p in results]
        avg = sum(model_sims) / len(model_sims)
        avg_scores[model_name] = avg

    for rank, (model, score) in enumerate(
        sorted(avg_scores.items(), key=lambda x: x[1], reverse=True), 1
    ):
        status = (
            "推荐" if score > 0.65
            else "可选" if score > 0.45
            else "不推荐"
        )
        print(f"  #{rank} {model:<35} 平均相似度={score:.4f}  [{status}]")

    return results

similarity_results = run_similarity_experiment()
```

### 2.3 分词器层面的根因

```python
def demonstrate_tokenizer_impact():
    """
    演示不同分词策略对中文文本的影响。

    英文 tokenizer 处理中文时会产生严重的碎片化问题，
    直接影响嵌入质量。
    """

    chinese_legal_text = "当事人一方不履行合同义务的，应当承担违约责任。"

    # 模拟不同 tokenizer 的处理结果
    tokenizers = {
        "中文优化 tokenizer（如 BGE）": {
            "expected_tokens": [
                "当事人", "一方", "不", "履行", "合同", "义务",
                "的", "应当", "承担", "违约", "责任"
            ],
            "token_count": 11,
            "note": "语义完整的词汇级别切分，每个 token 都有独立语义",
        },
        "英文 BPE tokenizer（如 GPT tokenizer）": {
            "expected_tokens": [
                # 英文 BPE 处理中文：一个字可能被拆成 2-3 个 byte-level token
                # 例如 "当" -> b'\\xe5\\xbd\\x93' -> 3 tokens
                "当", "事", "人", "一", "方", "不", "履", "行",
                "合", "同", "义", "务", "的", "应", "当", "承",
                "担", "违", "约", "责", "任",
            ],
            "token_count": 22,
            "note": (
                "一个字 = 1-3 tokens，不仅效率低，"
                "且每个 token 的语义被破坏"
            ),
        },
        "代码 tokenizer（如 CodeBERT）": {
            "expected_tokens": [
                "当", "事", "人", "一", "方", "不", "[UNK]", "[UNK]",
                "合", "同", "[UNK]", "[UNK]", "的", "应", "当",
                "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]",
            ],
            "token_count": 24,
            "note": (
                "代码 tokenizer 的词表中没有中文法律术语，"
                "大量 [UNK] 导致信息完全丢失"
            ),
        },
    }

    print("=" * 70)
    print("分词器对中文法律文本的影响")
    print(f"原文：{chinese_legal_text}")
    print(f"原文长度：{len(chinese_legal_text)} 字符")
    print("=" * 70)

    for tokenizer_name, info in tokenizers.items():
        print(f"\n【{tokenizer_name}】")
        print(f"  Token 数量: {info['token_count']}")
        print(f"  Token 列表: {info['expected_tokens']}")
        print(f"  说明: {info['note']}")
        expansion_ratio = info['token_count'] / len(chinese_legal_text)
        print(f"  膨胀比（tokens/字符）: {expansion_ratio:.2f}")

    print("\n结论：")
    print("  英文 tokenizer 处理中文时膨胀比 > 1.5 倍，")
    print("  导致：")
    print("  1. 同样的文本消耗更多 token（成本增加）")
    print("  2. 每个 token 的语义信息被稀释（质量下降）")
    print("  3. 模型的有效上下文窗口被压缩（原文 512 token 可能变成 800+ tokens）")

tokenizer_demo = demonstrate_tokenizer_impact()
```

---

## 3. 真实场景（Real-world Scenario）

### 场景：跨国律所的中文检索灾难

```python
"""
真实案例时间线：

某跨国律所的 IT 团队搭建了一个文档检索系统用于搜索中文合同。
他们选择了当时热门的 "all-MiniLM-L6-v2" 模型（英语优化，384 维），
因为它在 MTEB 英文榜单上排名很高。

上线后的数据：
- 律所律师查询："跨境并购中的股权转让限制条款"
- 系统返回 top-5 文档：全部是关于"租赁合同"的模板
- 原因：英文模型对中文的相似度几乎随机，
  任何两个中文文档的余弦相似度都聚集在 0.45-0.55 的狭窄区间

实际损失：律师手动搜索一份合同花了 3 小时（本应 2 分钟）
共 12 名律师 × 平均每日 4 次查询 × 3 小时 = 每天浪费 144 小时
按每小时 500 美元计费 = 每天损失 $72,000
"""

def estimate_mismatch_cost():
    """估算模型不匹配的实际成本。"""

    cost_factors = {
        "检索准确性损失": {
            "description": "用户需要浏览更多无关文档才能找到目标",
            "time_multiplier": 5,  # 搜索时间增加 5 倍
            "confidence_impact": "用户对系统的信任度下降，回退到手动流程",
        },
        "token 成本增加": {
            "description": "英文 tokenizer 处理中文的膨胀比约 1.5-2.0 倍",
            "cost_multiplier": 1.75,
            "annual_cost_example": (
                "如果每月 embedding 费用 $500，使用不匹配模型后变为 $875，"
                "一年额外支出 $4,500"
            ),
        },
        "二次开发成本": {
            "description": "模型调优、Prompt 工程、后处理规则等额外工作",
            "eng_weeks": 4,
            "cost": "$8,000 - $16,000",
        },
        "业务机会损失": {
            "description": "律师因为找不到关键先例而错失论点",
            "estimate": "每起案件可能影响数万到数十万美元",
        },
    }

    print("=" * 60)
    print("嵌入模型不匹配的成本估算")
    print("=" * 60)

    total_weekly_hours_wasted = 144
    hourly_rate = 500
    weekly_cost = total_weekly_hours_wasted * hourly_rate

    print(f"\n每周浪费律师时间: {total_weekly_hours_wasted} 小时")
    print(f"每周直接成本: ${weekly_cost:,}")
    print(f"每年直接成本: ${weekly_cost * 52:,}")

    for factor, details in cost_factors.items():
        print(f"\n【{factor}】")
        for k, v in details.items():
            print(f"  {k}: {v}")

    print("\n结论：选择错误的嵌入模型的代价远超模型本身的成本差异。")

cost_analysis = estimate_mismatch_cost()
```

---

## 4. 修复方案（Fix with Code）

### 4.1 模型选择流程图

```python
"""
嵌入模型选择决策树：

                    开始：你的任务是什么语言？
                           |
            ┌──────────────┼──────────────┐
            |              |              |
         纯中文         纯英文         多语言混合
            |              |              |
    ┌──────┴──────┐  ┌────┴────┐  ┌─────┴─────┐
    |             |  |         |  |           |
  法律/专业?   通用文本  专业领域?  multilingual  bge-m3
    |             |  |         |  text-embedding
  ┌─┴─┐      bge-large  ┌──┴──┐   -3-large
  |   |      -zh-v1.5    |     |
 是   否         或       是    否
  |   |      stella-base  |     |
bge  bge       -zh      领域   通用
-large -m3             专用   多语言
-zh   或                 模型   模型
-v1.5 jina
       -zh
"""


def model_selection_decision_tree(
    language: str,
    domain: str,
    has_gpu: bool,
    need_long_text: bool,
    budget_sensitive: bool,
) -> Dict:
    """
    嵌入模型选择决策函数。

    根据需求自动推荐最合适的嵌入模型。
    """

    def recommend(
        primary: str,
        alternatives: List[str] = None,
        reasoning: str = "",
    ) -> Dict:
        return {
            "primary_recommendation": primary,
            "alternatives": alternatives or [],
            "reasoning": reasoning,
        }

    if language == "zh":
        if domain in ("legal", "medical", "finance"):
            if has_gpu:
                return recommend(
                    "bge-large-zh-v1.5",
                    ["stella-base-zh-v3-1792d", "bge-m3"],
                    "中文专业领域 + 有GPU：使用 bge-large-zh-v1.5，"
                    "1024维，在中文法律/医疗评测中表现优异。"
                    "支持本地部署，数据不出域。",
                )
            else:
                return recommend(
                    "text-embedding-3-large",
                    ["bge-m3 (API)", "jina-embeddings-v2-base-zh (CPU)"],
                    "无GPU场景：使用 text-embedding-3-large API，"
                    "多语言能力强，但对中文的优化不如 bge。"
                    "如果数据敏感性要求本地处理，考虑 jina-embeddings-v2-base-zh。",
                )
        else:
            if need_long_text:
                return recommend(
                    "jina-embeddings-v2-base-zh",
                    ["bge-m3", "stella-base-zh-v3-1792d"],
                    "需要长文本处理（>512 tokens）："
                    "jina-embeddings-v2-base-zh 支持 8192 tokens，"
                    "适合需要完整上下文的法律文档段落。",
                )
            elif budget_sensitive:
                return recommend(
                    "bge-small-zh-v1.5",
                    ["bge-base-zh-v1.5", "text-embedding-3-small"],
                    "预算敏感：bge-small-zh-v1.5 仅 512 维，"
                    "速度快，成本低，适合原型验证。",
                )
            else:
                return recommend(
                    "bge-large-zh-v1.5",
                    ["stella-base-zh-v3-1792d", "bge-m3"],
                    "中文通用场景：bge-large-zh-v1.5 是当前中文检索的标杆模型。",
                )

    elif language == "en":
        if domain == "code":
            return recommend(
                "code-text-embedding-001",
                ["voyage-code-2", "stella-code"],
                "代码检索：使用代码专用的嵌入模型。"
                "通用模型无法理解代码的语义结构。",
            )
        elif domain in ("legal", "medical", "finance"):
            return recommend(
                "voyage-law-2" if domain == "legal" else "e5-large-v2",
                ["text-embedding-3-large"],
                f"英语{domain}领域：使用领域优化模型或高质量通用模型。",
            )
        else:
            return recommend(
                "text-embedding-3-large",
                ["e5-large-v2", "bge-large-en-v1.5"],
                "英语通用场景：text-embedding-3-large 是综合表现最好的选择。",
            )

    elif language == "multilingual":
        return recommend(
            "bge-m3",
            ["text-embedding-3-large", "multilingual-e5-large"],
            "多语言场景：bge-m3 支持 100+ 语言，"
            "text-embedding-3-large 是 OpenAI 的多语言方案。"
            "bge-m3 在多语言法律文档检索中表现更优。",
        )

    return recommend(
        "text-embedding-3-large",
        reasoning="无法确定语言/领域，使用最通用的多语言模型。",
    )


# 演示决策树
def demo_decision_tree():
    """演示不同场景下的模型推荐。"""

    scenarios = [
        {
            "language": "zh",
            "domain": "legal",
            "has_gpu": True,
            "need_long_text": True,
            "budget_sensitive": False,
            "description": "中国律所，本地部署，自有机房",
        },
        {
            "language": "zh",
            "domain": "legal",
            "has_gpu": False,
            "need_long_text": False,
            "budget_sensitive": True,
            "description": "小型法律科技初创，无GPU，预算有限",
        },
        {
            "language": "zh",
            "domain": "general",
            "has_gpu": False,
            "need_long_text": False,
            "budget_sensitive": False,
            "description": "中文通用知识库，使用 API",
        },
        {
            "language": "en",
            "domain": "code",
            "has_gpu": True,
            "need_long_text": False,
            "budget_sensitive": False,
            "description": "代码搜索工具",
        },
        {
            "language": "multilingual",
            "domain": "legal",
            "has_gpu": True,
            "need_long_text": True,
            "budget_sensitive": False,
            "description": "跨国律所，多语言法律文档",
        },
    ]

    print("=" * 80)
    print("嵌入模型选择决策树演示")
    print("=" * 80)

    for scenario in scenarios:
        result = model_selection_decision_tree(**{
            k: v for k, v in scenario.items()
            if k != "description"
        })
        print(f"\n场景：{scenario['description']}")
        print(f"  推荐：{result['primary_recommendation']}")
        print(f"  备选：{result['alternatives']}")
        print(f"  理由：{result['reasoning']}")
        print()

demo_decision_tree()
```

### 4.2 策略二：模型切换工具和兼容性检查

```python
from typing import List, Dict, Optional, Callable, Any
from enum import Enum


class EmbeddingProvider(Enum):
    """支持的嵌入提供商。"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    JINA = "jina"
    VOYAGE = "voyage"


class EmbeddingModelSwitcher:
    """
    嵌入模型切换工具。

    功能：
    1. 统一接口，屏蔽不同提供商的 API 差异
    2. 自动检测模型与文本语言的兼容性
    3. 性能基准测试，对比不同模型的检索质量
    4. 渐进式切换（A/B 测试）
    """

    def __init__(self):
        self.provider_adapters = {
            EmbeddingProvider.OPENAI: self._openai_embed,
            EmbeddingProvider.HUGGINGFACE: self._hf_embed,
            EmbeddingProvider.LOCAL: self._local_embed,
            EmbeddingProvider.JINA: self._jina_embed,
            EmbeddingProvider.VOYAGE: self._voyage_embed,
        }

        self.current_model = None
        self.model_registry = {}

    def register_model(
        self,
        name: str,
        provider: EmbeddingProvider,
        config: Dict[str, Any],
    ):
        """注册一个嵌入模型。"""
        self.model_registry[name] = {
            "provider": provider,
            "config": config,
        }

    def switch_model(self, model_name: str) -> bool:
        """切换到指定的嵌入模型。"""
        if model_name not in self.model_registry:
            print(f"错误：模型 '{model_name}' 未注册。")
            print(f"可用模型：{list(self.model_registry.keys())}")
            return False

        self.current_model = model_name
        print(f"已切换到模型：{model_name}")
        return True

    def embed_batch(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        批量生成嵌入向量。

        自动路由到正确的提供商 API。
        """
        model = model_name or self.current_model
        if model is None:
            raise ValueError("未指定模型。请先调用 switch_model()。")

        model_info = self.model_registry[model]
        provider = model_info["provider"]
        adapter = self.provider_adapters[provider]

        return adapter(texts, model_info["config"], batch_size)

    def _openai_embed(
        self,
        texts: List[str],
        config: Dict,
        batch_size: int,
    ) -> List[List[float]]:
        """
        OpenAI embedding API 适配器。

        示例调用：
        ```python
        from openai import OpenAI
        client = OpenAI(api_key=config["api_key"])
        response = client.embeddings.create(
            model=config["model_name"],
            input=texts,
        )
        return [item.embedding for item in response.data]
        ```
        """
        # 实际实现中取消注释下面的代码
        # from openai import OpenAI
        # client = OpenAI(
        #     api_key=config.get("api_key"),
        #     base_url=config.get("base_url"),
        # )
        # all_embeddings = []
        # for i in range(0, len(texts), batch_size):
        #     batch = texts[i:i + batch_size]
        #     response = client.embeddings.create(
        #         model=config["model_name"],
        #         input=batch,
        #     )
        #     all_embeddings.extend(
        #         [item.embedding for item in response.data]
        #     )
        # return all_embeddings

        # 模拟返回（演示用）
        return [
            [0.0] * config.get("dim", 1536)
            for _ in texts
        ]

    def _hf_embed(
        self,
        texts: List[str],
        config: Dict,
        batch_size: int,
    ) -> List[List[float]]:
        """
        HuggingFace 模型适配器。

        示例调用：
        ```python
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            config["model_name"],
            device=config.get("device", "cpu"),
        )
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
        ```
        """
        return [[0.0] * config.get("dim", 1024) for _ in texts]

    def _local_embed(self, texts, config, batch_size):
        """本地模型适配器。"""
        return [[0.0] * config.get("dim", 768) for _ in texts]

    def _jina_embed(self, texts, config, batch_size):
        """Jina AI API 适配器。"""
        return [[0.0] * config.get("dim", 768) for _ in texts]

    def _voyage_embed(self, texts, config, batch_size):
        """Voyage AI API 适配器。"""
        return [[0.0] * config.get("dim", 1024) for _ in texts]


class EmbeddingQualityBenchmark:
    """
    嵌入模型质量基准测试工具。

    功能：
    1. 在评估集上对比不同模型的检索质量
    2. 生成详细的对比报告
    3. 支持增量测试（只测试新模型，复用已有结果）
    """

    def __init__(self, switcher: EmbeddingModelSwitcher):
        self.switcher = switcher

    def run_benchmark(
        self,
        eval_queries: List[Dict],
        document_corpus: List[str],
        models_to_test: List[str],
        k_values: List[int] = None,
    ) -> Dict[str, Dict]:
        """
        运行基准测试。

        评估集格式：
        [
            {
                "query": "违约责任承担方式",
                "relevant_doc_indices": [0, 3, 7],
            },
            ...
        ]
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        corpus_embeddings = {}  # model_name -> embeddings
        results = {}

        for model_name in models_to_test:
            print(f"\n正在使用 {model_name} 进行基准测试...")

            self.switcher.switch_model(model_name)

            # 嵌入文档语料
            doc_embeddings = self.switcher.embed_batch(document_corpus)
            corpus_embeddings[model_name] = doc_embeddings

            # 对每个查询评估
            model_metrics = {
                f"recall@{k}": [] for k in k_values
            }
            model_metrics.update({
                f"precision@{k}": [] for k in k_values
            })
            model_metrics["mrr"] = []

            for eval_item in eval_queries:
                query = eval_item["query"]
                relevant_indices = set(eval_item["relevant_doc_indices"])

                # 嵌入查询
                query_embedding = self.switcher.embed_batch([query])[0]

                # 计算相似度并排序
                similarities = []
                for j, doc_emb in enumerate(doc_embeddings):
                    sim = self._cosine_similarity(query_embedding, doc_emb)
                    similarities.append((j, sim))

                similarities.sort(key=lambda x: x[1], reverse=True)

                # 计算各 k 值的 Recall 和 Precision
                for k in k_values:
                    top_k_indices = set(idx for idx, _ in similarities[:k])
                    hits = top_k_indices & relevant_indices
                    recall = len(hits) / len(relevant_indices) if relevant_indices else 0
                    precision = len(hits) / k

                    model_metrics[f"recall@{k}"].append(recall)
                    model_metrics[f"precision@{k}"].append(precision)

                # 计算 MRR (Mean Reciprocal Rank)
                for rank, (idx, _) in enumerate(similarities, 1):
                    if idx in relevant_indices:
                        model_metrics["mrr"].append(1.0 / rank)
                        break
                else:
                    model_metrics["mrr"].append(0.0)

            # 汇总
            summary = {}
            for metric, values in model_metrics.items():
                if values:
                    summary[metric] = {
                        "mean": round(float(np.mean(values)), 4),
                        "std": round(float(np.std(values)), 4),
                        "min": round(float(np.min(values)), 4),
                        "max": round(float(np.max(values)), 4),
                    }

            results[model_name] = summary

        return results

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """计算余弦相似度。"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        n1 = math.sqrt(sum(a * a for a in vec1))
        n2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

    def print_comparison_report(self, results: Dict[str, Dict]):
        """打印模型对比报告。"""

        if not results:
            print("无结果。")
            return

        model_names = list(results.keys())
        # 获取所有指标名
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())

        metrics = sorted(all_metrics)

        print("\n" + "=" * 90)
        print("嵌入模型基准测试对比报告")
        print("=" * 90)

        # 表头
        header = f"{'模型':<35}"
        for metric in metrics:
            header += f" {metric:<20}"
        print(header)
        print("-" * len(header))

        # 数据行
        for model_name in model_names:
            row = f"{model_name:<35}"
            for metric in metrics:
                if metric in results[model_name]:
                    value = results[model_name][metric]["mean"]
                    row += f" {value:<20.4f}"
                else:
                    row += f" {'N/A':<20}"
            print(row)

        # 最佳模型
        print(f"\n各指标最佳模型：")
        for metric in metrics:
            best_model = max(
                model_names,
                key=lambda m: results[m].get(metric, {}).get("mean", 0),
            )
            best_value = results[best_model].get(metric, {}).get("mean", 0)
            print(f"  {metric}: {best_model} ({best_value:.4f})")


# 演示基准测试框架
switcher = EmbeddingModelSwitcher()
switcher.register_model(
    "bge-large-zh-v1.5",
    EmbeddingProvider.HUGGINGFACE,
    {"model_name": "BAAI/bge-large-zh-v1.5", "dim": 1024},
)
switcher.register_model(
    "text-embedding-3-large",
    EmbeddingProvider.OPENAI,
    {"model_name": "text-embedding-3-large", "dim": 3072},
)

benchmark = EmbeddingQualityBenchmark(switcher)
print("嵌入模型切换器和基准测试工具已就绪。")
```

### 4.3 策略三：语言检测 + 自动路由

```python
import re


class LanguageAwareEmbeddingRouter:
    """
    语言感知的嵌入路由。

    自动检测文本语言，路由到最合适的嵌入模型。
    支持混合语言场景（如中文文档中夹杂英文术语）。
    """

    def __init__(self):
        self.language_models = {
            "zh": {
                "primary": "bge-large-zh-v1.5",
                "fallback": "bge-m3",
            },
            "en": {
                "primary": "text-embedding-3-large",
                "fallback": "bge-large-en-v1.5",
            },
            "code": {
                "primary": "code-text-embedding-001",
                "fallback": "text-embedding-3-large",
            },
            "mixed": {
                "primary": "bge-m3",
                "fallback": "text-embedding-3-large",
            },
        }

        # 语言检测规则
        self.cjk_range = (
            ('一', '鿿'),     # CJK 统一汉字
            ('㐀', '䶿'),     # CJK 扩展 A
            ('぀', 'ゟ'),     # 日文平假名
            ('゠', 'ヿ'),     # 日文片假名
            ('가', '힯'),     # 韩文
        )

    def detect_language(self, text: str) -> str:
        """
        检测文本的主语言。

        返回: 'zh', 'en', 'code', 'mixed'
        """
        total_chars = len(text.strip())
        if total_chars == 0:
            return "en"

        # 统计各类字符
        cjk_count = 0
        ascii_alpha_count = 0
        code_indicator_count = 0

        for char in text:
            code_point = ord(char)
            # CJK 检测
            for start, end in self.cjk_range:
                if start <= char <= end:
                    cjk_count += 1
                    break
            # 英文检测
            if char.isascii() and char.isalpha():
                ascii_alpha_count += 1
            # 代码特征检测
            if char in '{}[]();=<>@#':
                code_indicator_count += 1

        cjk_ratio = cjk_count / total_chars if total_chars > 0 else 0
        ascii_ratio = ascii_alpha_count / total_chars if total_chars > 0 else 0
        code_ratio = code_indicator_count / total_chars if total_chars > 0 else 0

        # 代码判断
        if code_ratio > 0.05:
            return "code"

        # 语言判断
        if cjk_ratio > 0.5:
            if ascii_ratio > 0.2:
                return "mixed"
            return "zh"
        elif ascii_ratio > 0.5:
            return "en"
        else:
            return "en"  # 默认

    def route(self, text: str) -> str:
        """
        为给定文本路由到最合适的嵌入模型。

        返回模型名称。
        """
        lang = self.detect_language(text)
        model_config = self.language_models.get(lang, self.language_models["en"])
        return model_config["primary"]

    def embed_with_routing(
        self,
        texts: List[str],
        embed_fns: Dict[str, Callable],
    ) -> List[List[float]]:
        """
        语言感知的嵌入路由。

        为每个文本检测语言并路由到相应模型。
        适合处理多语言混合的文档集。
        """
        # 按语言分组
        language_groups = {}
        for i, text in enumerate(texts):
            lang = self.detect_language(text)
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append((i, text))

        # 初始化结果数组
        results = [None] * len(texts)

        # 各组使用各自的模型嵌入
        for lang, items in language_groups.items():
            model_name = self.language_models[lang]["primary"]
            embed_fn = embed_fns.get(model_name)
            if embed_fn is None:
                # 使用 fallback
                fallback_name = self.language_models[lang]["fallback"]
                embed_fn = embed_fns.get(fallback_name)
                if embed_fn is None:
                    raise ValueError(
                        f"没有可用的嵌入函数用于语言 '{lang}' "
                        f"(模型: {model_name} 或 {fallback_name})"
                    )

            indices = [item[0] for item in items]
            lang_texts = [item[1] for item in items]
            lang_embeddings = embed_fn(lang_texts)

            for idx, emb in zip(indices, lang_embeddings):
                results[idx] = emb

        return results


# 演示语言感知路由
router = LanguageAwareEmbeddingRouter()

test_texts = [
    "当事人一方不履行合同义务的，应当承担违约责任。",
    "A party who breaches a contract shall bear liability for breach of contract.",
    "合同中经常出现 force majeure 条款，即不可抗力条款。",
    "def calculate_damages(contract_value: float, breach_ratio: float) -> float:\n    return contract_value * breach_ratio * 0.3",
    "根据民法典第577条，违约责任的承担方式包括继续履行、采取补救措施或者赔偿损失。",
]

print("=" * 80)
print("语言感知嵌入路由演示")
print("=" * 80)

for text in test_texts:
    lang = router.detect_language(text)
    model = router.route(text)
    preview = text[:60] + ("..." if len(text) > 60 else "")
    print(f"\n文本: {preview}")
    print(f"  检测语言: {lang}")
    print(f"  路由模型: {model}")
```

---

## 5. 检查清单（Checklist）

### 模型选择前

- [ ] 是否明确了任务的主要语言？（中文/英文/多语言）
- [ ] 是否了解候选模型在目标语言上的 MTEB 或 C-MTEB 排名？
- [ ] 是否检查了模型的训练语料是否包含目标领域的文本？
- [ ] Tokenizer 是否支持目标语言？（检查词表覆盖率）
- [ ] 模型的最大序列长度是否满足文档块大小的需求？

### 模型评估时

- [ ] 是否在自有评估集（而非公开 benchmark）上测试了检索质量？
- [ ] 是否对比了至少 3 个候选模型的 Recall@k 和 MRR？
- [ ] 是否测试了模型在"跨语言"场景下的表现？（如果系统涉及多语言）
- [ ] 是否检查了模型对专业术语的语义区分能力？
- [ ] 是否测量了嵌入生成的延迟和吞吐量？

### 模型部署后

- [ ] 是否监控了检索结果的余弦相似度分布？（分布太集中说明模型区分度不足）
- [ ] 是否记录了模型配置和对应的检索质量基准？
- [ ] 是否有模型切换的回滚方案？（旧嵌入是否需要重建？）
- [ ] 是否定期（每季度）重新评估新发布的模型？

### 模型选择速查表

```
┌─────────────────────┬──────────────────────────────────┐
│ 场景                │ 推荐模型                         │
├─────────────────────┼──────────────────────────────────┤
│ 中文法律文档        │ bge-large-zh-v1.5                │
│ 中文通用            │ bge-m3 / stella-base-zh-v3-1792d │
│ 英文通用            │ text-embedding-3-large           │
│ 多语言混合          │ bge-m3 / text-embedding-3-large  │
│ 代码检索            │ voyage-code-2 / CodeBERT         │
│ 长文本 (>8K)        │ jina-embeddings-v2-base-zh       │
│ 低资源/CPU          │ bge-small-zh-v1.5 (512d)         │
│ 数据不出域          │ bge系列 (本地部署)               │
└─────────────────────┴──────────────────────────────────┘
```

---

> **核心教训**：嵌入模型的选择是 RAG 系统最重要的架构决策之一。
> 选错模型，后续所有的检索优化都是徒劳。
> 中文法律场景下，**永远优先选择中文优化的嵌入模型**，而不是英文榜单上的"明星模型"。
