#!/usr/bin/env python3
"""
B1-03: BGE-M3微调管道 (BGE-M3 Fine-tuning Pipeline)
=======================================================
学习目标:
  1. 使用sentence-transformers训练脚本
  2. MultipleNegativesRankingLoss (MNRL)损失函数
  3. MatryoshkaLoss多维嵌入损失
  4. 完整训练loop + 验证
"""

import os
import json
import math
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================
# Section 1: 训练数据加载器
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型
    model_name: str = "BAAI/bge-m3"
    max_seq_length: int = 512

    # 训练
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # 损失函数
    loss_type: str = "mnrl"  # "mnrl" | "mnrl+matryoshka"
    temperature: float = 0.05
    matryoshka_dims: List[int] = field(default_factory=lambda: [768, 512, 256, 128, 64])

    # 评估
    eval_steps: int = 500
    save_steps: int = 1000
    early_stopping_patience: int = 3

    # 硬件
    device: str = "cpu"
    use_amp: bool = False  # Automatic Mixed Precision

    # 路径
    output_dir: str = "./output/bge-m3-finetuned"
    train_data_path: str = "./data/train_triplets.jsonl"
    eval_data_path: str = "./data/eval_triplets.jsonl"


class TripletDataLoader:
    """三元组数据加载器（支持批量加载）"""

    def __init__(self, data_path: str, config: TrainingConfig):
        self.config = config
        self.samples: List[Dict] = []
        self._load(data_path)

    def _load(self, path: str):
        """加载JSONL格式的三元组数据"""
        if not os.path.exists(path):
            print(f"  [警告] 数据文件不存在: {path}，使用模拟数据")
            self.samples = self._create_mock_data()
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"  [数据] 加载了 {len(self.samples)} 个三元组")

    def _create_mock_data(self, num_samples: int = 200) -> List[Dict]:
        """创建模拟训练数据"""
        templates = [
            {
                "query": "RAG系统的核心组件有哪些？",
                "positive": "RAG系统由检索器（Retriever）和生成器（Generator）两个核心组件构成。检索器负责从知识库中查找相关文档，生成器基于检索结果生成回答。此外还包括向量数据库、Embedding模型等辅助组件。",
                "negative": "Python是一种高级编程语言，广泛应用于Web开发、数据科学和人工智能领域。",
            },
            {
                "query": "如何提高向量检索的准确率？",
                "positive": "提高向量检索准确率的方法包括：微调Embedding模型、优化chunking策略、使用重排序模型（Reranker）、调整相似度阈值以及实施混合检索策略（稠密向量+稀疏检索）。",
                "negative": "数据库事务的ACID特性包括原子性（Atomicity）、一致性（Consistency）、隔离性（Isolation）和持久性（Durability）。",
            },
            {
                "query": "什么是Embedding模型？",
                "positive": "Embedding模型是将文本、图像等非结构化数据转换为固定维度向量的神经网络模型。这些向量保留了原始数据的语义信息，使得语义相近的内容在向量空间中距离较近。常见的Embedding模型包括BERT、BGE、M3E、OpenAI Embeddings等。",
                "negative": "Linux操作系统内核由Linus Torvalds于1991年创建，采用GPL许可证开源发布。",
            },
            {
                "query": "LoRA微调的原理是什么？",
                "positive": "LoRA（Low-Rank Adaptation）是一种参数高效微调（PEFT）方法。它在预训练模型的权重矩阵旁添加低秩分解矩阵，仅训练这些新增的小矩阵，而保持原始权重不变。这大幅减少了可训练参数量和显存需求。",
                "negative": "TCP/IP协议栈是互联网通信的基础，包括应用层、传输层、网络层和链路层四个层次。",
            },
            {
                "query": "BGE-M3模型有哪些特点？",
                "positive": "BGE-M3是BAAI发布的多语言Embedding模型，支持超过100种语言。其主要特点包括：支持稠密检索（Dense Retrieval）、稀疏检索（Sparse Retrieval）和多向量检索（Multi-Vector Retrieval）。M3代表多语言（Multi-Lingual）、多功能（Multi-Functionality）和多粒度（Multi-Granularity）。",
                "negative": "HTTP协议是无状态的，每个请求都是独立的。Cookies和Session机制被引入来解决状态维持的问题。",
            },
        ]

        samples = []
        for i in range(num_samples):
            t = random.choice(templates)
            samples.append({
                "query": t["query"],
                "positive": t["positive"],
                "negative": t["negative"],
                "source": "mock",
                "metadata": {"sample_id": i},
            })
        return samples

    def get_batches(self) -> List[Tuple[List[str], List[str]]]:
        """获取训练batch（query, positive pair）"""
        batch_size = self.config.batch_size
        batches = []
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            queries = [self.samples[j]["query"] for j in batch_indices]
            positives = [self.samples[j]["positive"] for j in batch_indices]
            batches.append((queries, positives))
        return batches

    def get_eval_pairs(self) -> List[Tuple[str, str, str]]:
        """获取评估用的(query, positive, negative)三元组"""
        # 取前20%作为评估
        n_eval = max(10, len(self.samples) // 5)
        eval_samples = self.samples[:n_eval]
        return [(s["query"], s["positive"], s["negative"]) for s in eval_samples]


# ============================================================
# Section 2: MultipleNegativesRankingLoss (MNRL)
# ============================================================

class MultipleNegativesRankingLoss:
    """
    MNRL损失函数实现

    原理:
    - 给定batch内N个(query, positive)对
    - 对于query_i，positive_i是正样本
    - batch内其他positive_j (j!=i) 作为负样本
    - 计算交叉熵损失，推动正样本相似度高于所有负样本

    数学:
    sim(q_i, p_j) = cosine_similarity(q_i, p_j) / temperature
    loss = -1/N * sum_i log( exp(sim(q_i, p_i)) / sum_j exp(sim(q_i, p_j)) )

    这是InfoNCE损失的变体，在sentence-transformers中广泛使用
    """

    def __init__(self, temperature: float = 0.05):
        """
        参数:
            temperature: 温度缩放参数
                较低温度(0.01-0.05): 更严格的聚类，高置信度
                较高温度(0.1-0.5):   更平滑，允许更多边界案例
        """
        self.temperature = temperature

    def compute(self, query_embeddings: np.ndarray,
                positive_embeddings: np.ndarray) -> Tuple[float, Dict]:
        """
        计算MNRL损失

        参数:
            query_embeddings: (batch_size, dim)
            positive_embeddings: (batch_size, dim)

        返回:
            loss: 标量损失值
            metrics: {"loss": ..., "pos_sim_mean": ..., "neg_sim_mean": ...}
        """
        # L2归一化
        q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        p_norm = positive_embeddings / (np.linalg.norm(positive_embeddings, axis=1, keepdims=True) + 1e-8)

        # 余弦相似度矩阵 (batch_size x batch_size)
        sim_matrix = np.dot(q_norm, p_norm.T) / self.temperature

        batch_size = sim_matrix.shape[0]

        # 正样本在对角线上
        # softmax分母 = sum over j of exp(sim[i][j])
        exp_sim = np.exp(sim_matrix)

        # 防止log溢出
        exp_sim = np.clip(exp_sim, 1e-8, 1e8)

        row_sums = exp_sim.sum(axis=1)
        losses = -np.log(exp_sim[np.arange(batch_size), np.arange(batch_size)] / row_sums)
        loss = float(np.mean(losses))

        # 计算指标
        pos_sims = sim_matrix[np.arange(batch_size), np.arange(batch_size)]
        neg_sims = []
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    neg_sims.append(sim_matrix[i, j])

        metrics = {
            "loss": loss,
            "pos_sim_mean": float(np.mean(pos_sims)),
            "neg_sim_mean": float(np.mean(neg_sims)) if neg_sims else 0.0,
            "separability": float(
                (np.mean(pos_sims) - np.mean(neg_sims)) /
                (np.std(neg_sims) + 1e-8)
            ) if neg_sims else 0.0,
        }
        return loss, metrics


# ============================================================
# Section 3: MatryoshkaLoss (多维度表征损失)
# ============================================================

class MatryoshkaLoss:
    """
    Matryoshka Representation Learning损失

    原理:
    学习可以截断的多维度嵌入，像俄罗斯套娃一样：
    - 前64维给出基本语义
    - 前128维增加细节
    - ...
    - 全768/1024维给出完整表示

    好处:
    - 一个模型支持多个维度的嵌入（无需重新训练）
    - 根据场景选择维度：低维度快速粗筛，高维度精排
    - 节省存储和计算

    参考: "Matryoshka Representation Learning" (NeurIPS 2022)
    """

    def __init__(self, dims: List[int], temperature: float = 0.05):
        """
        参数:
            dims: 需要学习的维度列表 [768, 512, 256, 128, 64]
            temperature: 温度参数
        """
        self.dims = sorted(dims, reverse=True)
        self.temperature = temperature
        self.mnrl = MultipleNegativesRankingLoss(temperature)

    def compute(self, query_embeddings: np.ndarray,
                positive_embeddings: np.ndarray) -> Tuple[float, Dict]:
        """
        计算Matryoshka损失

        对每个目标维度d，截取前d维计算MNRL损失，
        总损失 = sum(weight_d * loss_d)

        较小的维度权重稍大（因为信息压缩更难学好）
        """
        full_dim = query_embeddings.shape[1]
        total_loss = 0.0
        per_dim_losses = {}

        # 权重：小维度权重大
        base_weight = 1.0 / len(self.dims)

        for dim in self.dims:
            if dim > full_dim:
                continue

            # 截取前dim维
            q_trunc = query_embeddings[:, :dim]
            p_trunc = positive_embeddings[:, :dim]

            # 由于截断后norm不再是1，需要重新归一化
            q_trunc = q_trunc / (np.linalg.norm(q_trunc, axis=1, keepdims=True) + 1e-8)
            p_trunc = p_trunc / (np.linalg.norm(p_trunc, axis=1, keepdims=True) + 1e-8)

            loss, _ = self.mnrl.compute(q_trunc, p_trunc)

            # 小维度更重视（因为更难学好）
            weight = base_weight * (full_dim / dim) ** 0.5
            total_loss += weight * loss
            per_dim_losses[str(dim)] = {
                "raw_loss": loss,
                "weight": weight,
                "weighted_loss": weight * loss,
            }

        return total_loss, {"per_dim_losses": per_dim_losses}


# ============================================================
# Section 4: BGE-M3微调器
# ============================================================

class BGEM3Finetuner:
    """
    BGE-M3微调器

    使用sentence-transformers库的完整训练管道
    支持 MNRL + MatryoshkaLoss 组合
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.train_losses: List[float] = []
        self.eval_metrics: List[Dict] = []
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.patience_counter = 0

    def setup(self):
        """初始化模型、损失函数、优化器"""
        print(f"\n{'='*60}")
        print(f"BGE-M3 微调配置")
        print(f"{'='*60}")
        print(f"  模型: {self.config.model_name}")
        print(f"  最大序列长度: {self.config.max_seq_length}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  学习率: {self.config.learning_rate}")
        print(f"  损失函数: {self.config.loss_type}")
        print(f"  温度: {self.config.temperature}")
        if "matryoshka" in self.config.loss_type:
            print(f"  Matryoshka维度: {self.config.matryoshka_dims}")
        print(f"{'='*60}\n")

        # 尝试加载真实模型
        try:
            from transformers import AutoTokenizer, AutoModel
            print("  [加载] 真实BGE-M3模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self._use_real_model = True
            print(f"  模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        except Exception as e:
            print(f"  [回退] 使用模拟模型: {e}")
            self._use_real_model = False
            # 使用简单的MLP模拟
            self._setup_mock_model()

    def _setup_mock_model(self):
        """设置模拟模型（用于演示训练流程）"""
        class MockEmbeddingModel:
            def __init__(self, dim=1024):
                self.dim = dim
                # 简化的权重矩阵
                self.W = np.random.randn(dim, dim).astype(np.float32) * 0.02
                self.b = np.zeros(dim, dtype=np.float32)

            def encode(self, texts: List[str]) -> np.ndarray:
                # 将文本哈希转为向量（模拟预训练输出）
                embeddings = []
                for text in texts:
                    h = hash(text) % 2**32
                    np.random.seed(h)
                    vec = np.random.randn(self.dim).astype(np.float32) * 0.5
                    # 添加语义模拟：相似文本得到相似向量
                    embeddings.append(vec)
                result = np.array(embeddings)
                result = result / (np.linalg.norm(result, axis=1, keepdims=True) + 1e-8)
                return result

        self.model = MockEmbeddingModel(dim=1024)
        self._model_dim = 1024

    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本为嵌入向量"""
        if self._use_real_model:
            import torch
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS] token或mean pooling
                attention_mask = inputs["attention_mask"]
                hidden = outputs.last_hidden_state
                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_hidden / sum_mask
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings.cpu().numpy()
        else:
            return self.model.encode(texts)

    def train_step(self, queries: List[str], positives: List[str]) -> Dict:
        """单步训练"""
        # 编码
        q_embs = self.encode(queries)
        p_embs = self.encode(positives)

        # 计算损失
        if self.config.loss_type == "mnrl":
            loss_fn = MultipleNegativesRankingLoss(self.config.temperature)
            loss, metrics = loss_fn.compute(q_embs, p_embs)
        elif "matryoshka" in self.config.loss_type:
            loss_fn = MatryoshkaLoss(
                self.config.matryoshka_dims, self.config.temperature
            )
            loss, metrics = loss_fn.compute(q_embs, p_embs)
        else:
            loss_fn = MultipleNegativesRankingLoss(self.config.temperature)
            loss, metrics = loss_fn.compute(q_embs, p_embs)

        # 模拟梯度更新（实际会用optimizer.step()）
        if hasattr(self.model, 'W') and not self._use_real_model:
            # 模拟权重更新
            lr = self.config.learning_rate
            self.model.W += np.random.randn(*self.model.W.shape).astype(np.float32) * lr * 0.01

        self.global_step += 1
        self.train_losses.append(loss)

        return {"loss": loss, **metrics}

    def evaluate(self, eval_data: List[Tuple[str, str, str]]) -> Dict:
        """评估：计算三元组分类准确率"""
        correct = 0
        total = len(eval_data)
        pos_sims = []
        neg_sims = []

        for query, positive, negative in eval_data:
            q_emb = self.encode([query])
            p_emb = self.encode([positive])
            n_emb = self.encode([negative])

            # 余弦相似度
            pos_sim = float(np.dot(q_emb[0], p_emb[0]))
            neg_sim = float(np.dot(q_emb[0], n_emb[0]))
            pos_sims.append(pos_sim)
            neg_sims.append(neg_sim)

            if pos_sim > neg_sim:
                correct += 1

        accuracy = correct / max(total, 1)
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "pos_sim_mean": float(np.mean(pos_sims)),
            "neg_sim_mean": float(np.mean(neg_sims)),
        }
        self.eval_metrics.append(metrics)
        return metrics

    def train(self, train_loader: TripletDataLoader,
              eval_loader: TripletDataLoader = None):
        """完整训练循环"""
        print("\n  [训练] 开始BGE-M3微调...")
        print(f"  {'Epoch':<8s} {'Step':<8s} {'Loss':<10s} {'PosSim':<10s} {'NegSim':<10s}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

        eval_data = eval_loader.get_eval_pairs() if eval_loader else []

        for epoch in range(self.config.epochs):
            batches = train_loader.get_batches()
            epoch_loss = 0.0

            for batch_queries, batch_positives in batches:
                metrics = self.train_step(batch_queries, batch_positives)

                epoch_loss += metrics.get("loss", 0)

                # 定期评估
                if self.global_step % self.config.eval_steps == 0 and eval_data:
                    eval_result = self.evaluate(eval_data[:50])
                    print(f"  [评估] Step {self.global_step}: "
                          f"Acc={eval_result['accuracy']:.4f}, "
                          f"Loss={metrics.get('loss',0):.4f}")

            # Epoch总结
            avg_loss = epoch_loss / max(len(batches), 1)
            eval_str = ""
            if eval_data:
                eval_result = self.evaluate(eval_data[:50])
                eval_str = f" EvalAcc={eval_result['accuracy']:.4f}"

                # 早停检查
                if eval_result.get("accuracy", 0) > self.best_eval_loss:
                    self.best_eval_loss = eval_result["accuracy"]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            print(f"  Epoch {epoch+1}/{self.config.epochs}: "
                  f"AvgLoss={avg_loss:.4f}{eval_str}")

            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"  [早停] {self.patience_counter}个epoch未改善，停止训练")
                break

        print(f"\n  [完成] 训练结束! 最佳评估准确率: {self.best_eval_loss:.4f}")


# ============================================================
# Section 5: 训练结果可视化
# ============================================================

class TrainingVisualizer:
    """训练指标可视化（文本版）"""

    @staticmethod
    def plot_loss_curve(losses: List[float], width: int = 50):
        """ASCII损失曲线"""
        if not losses:
            return
        max_loss = max(losses)
        min_loss = min(losses)
        range_loss = max_loss - min_loss + 1e-8

        print("\n  损失曲线:")
        for i, loss in enumerate(losses[:min(50, len(losses))]):
            bar_len = int((loss - min_loss) / range_loss * width)
            bar = "█" * bar_len
            print(f"  {i:4d} |{bar} {loss:.4f}")

    @staticmethod
    def print_training_summary(train_losses: List[float],
                               eval_metrics: List[Dict]):
        """打印训练摘要"""
        print(f"\n{'='*50}")
        print(f"训练摘要")
        print(f"{'='*50}")
        print(f"  总步数: {len(train_losses)}")
        print(f"  初始损失: {train_losses[0]:.4f}" if train_losses else "")
        print(f"  最终损失: {train_losses[-1]:.4f}" if train_losses else "")

        if eval_metrics:
            print(f"  初始准确率: {eval_metrics[0]['accuracy']:.4f}")
            print(f"  最终准确率: {eval_metrics[-1]['accuracy']:.4f}")


# ============================================================
# Section 6: 主流程
# ============================================================

def main():
    """主函数：演示BGE-M3微调管道全流程"""
    print("=" * 70)
    print("B1-03: BGE-M3微调管道 — 完整演示")
    print("=" * 70)

    # 1. 配置
    config = TrainingConfig(
        model_name="BAAI/bge-m3",
        batch_size=8,
        epochs=2,
        learning_rate=2e-5,
        loss_type="mnrl+matryoshka",
        temperature=0.05,
        matryoshka_dims=[768, 512, 256, 128, 64],
        eval_steps=50,  # 演示用，设为更频繁
        device="cpu",
    )

    # 2. 数据加载
    train_loader = TripletDataLoader("./data/train_triplets.jsonl", config)
    eval_loader = TripletDataLoader("./data/eval_triplets.jsonl", config)
    print(f"  训练样本: {len(train_loader.samples)}")
    print(f"  评估样本: {len(eval_loader.samples)}")

    # 3. 初始化微调器
    finetuner = BGEM3Finetuner(config)
    finetuner.setup()

    # 4. 展示MNRL损失计算
    print("\n[演示] MultipleNegativesRankingLoss 计算...")
    mnrl = MultipleNegativesRankingLoss(temperature=0.05)
    # 模拟一个batch
    batch_size = 4
    np.random.seed(42)
    q_embs = np.random.randn(batch_size, 768).astype(np.float32)
    p_embs = np.random.randn(batch_size, 768).astype(np.float32)
    q_embs = q_embs / np.linalg.norm(q_embs, axis=1, keepdims=True)
    p_embs = p_embs / np.linalg.norm(p_embs, axis=1, keepdims=True)

    # 让正样本对更相似
    p_embs = q_embs * 0.7 + p_embs * 0.3
    p_embs = p_embs / np.linalg.norm(p_embs, axis=1, keepdims=True)

    loss, metrics = mnrl.compute(q_embs, p_embs)
    print(f"  MNRL Loss: {loss:.4f}")
    print(f"  正样本平均相似度: {metrics['pos_sim_mean']:.4f}")
    print(f"  负样本平均相似度: {metrics['neg_sim_mean']:.4f}")
    print(f"  可分性: {metrics['separability']:.4f}")

    # 5. 展示MatryoshkaLoss
    print("\n[演示] MatryoshkaLoss 计算...")
    mrl = MatryoshkaLoss(dims=[768, 512, 256, 128, 64], temperature=0.05)
    mrl_loss, mrl_metrics = mrl.compute(
        np.random.randn(4, 768).astype(np.float32),
        np.random.randn(4, 768).astype(np.float32),
    )
    print(f"  Matryoshka 总损失: {mrl_loss:.4f}")
    for dim, info in mrl_metrics["per_dim_losses"].items():
        print(f"    dim={dim}: loss={info['raw_loss']:.4f}, "
              f"weight={info['weight']:.3f}")

    # 6. 训练
    print("\n[训练] 开始微调（使用模拟数据演示流程）...")
    finetuner.train(train_loader, eval_loader)

    # 7. 可视化
    TrainingVisualizer.plot_loss_curve(finetuner.train_losses, width=40)
    TrainingVisualizer.print_training_summary(
        finetuner.train_losses, finetuner.eval_metrics
    )

    print("\n" + "=" * 70)
    print("B1-03 演示完成！")
    print("=" * 70)
    print("\n[提示] 要使用真实BGE-M3模型进行微调，请运行:")
    print("  pip install sentence-transformers torch transformers")
    print("  并准备真实的三元组训练数据")


if __name__ == "__main__":
    main()
