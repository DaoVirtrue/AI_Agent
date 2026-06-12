#!/usr/bin/env python3
"""
A2-01: CLIP图片检索 (CLIP Image Retrieval)
=============================================
学习目标:
  1. CLIP-ViT-B/32 图像嵌入生成
  2. 文本→图片检索
  3. 图片→文本检索
  4. 文本+图片联合查询
  5. 相似度阈值调优
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Section 1: CLIP模型封装
# ============================================================

class CLIPRetriever:
    """CLIP跨模态检索器"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu"):
        """
        参数:
            model_name: CLIP模型变体
                - openai/clip-vit-base-patch32 (默认，512维)
                - openai/clip-vit-base-patch16 (更高精度)
                - openai/clip-vit-large-patch14 (最高精度)
            device: 'cpu' 或 'cuda'
        """
        from transformers import CLIPProcessor, CLIPModel

        self.model_name = model_name
        self.device = device

        print(f"  [加载] {model_name} ...")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embedding_dim = self.model.config.projection_dim

        # 存储索引
        self.image_paths: List[str] = []
        self.image_embeddings: Optional[np.ndarray] = None
        self.text_descriptions: List[str] = []  # 可选的图片文本描述

    # ---------- 嵌入生成 ----------

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """批量生成图像嵌入"""
        images = []
        valid_paths = []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"  [跳过] {path}: {e}")

        self.image_paths = valid_paths

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with __import__("torch").no_grad():
            embeddings = self.model.get_image_features(**inputs)

        embeddings = embeddings.cpu().numpy()
        # L2归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        self.image_embeddings = embeddings

        print(f"  [编码] {len(images)} 张图片 → {embeddings.shape[1]}维向量")
        return embeddings

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """生成文本嵌入"""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with __import__("torch").no_grad():
            embeddings = self.model.get_text_features(**inputs)

        embeddings = embeddings.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings

    # ---------- 检索 ----------

    def text_to_image_search(self, query: str, top_k: int = 5,
                             threshold: float = 0.0) -> List[Dict]:
        """
        文本→图片检索
        用自然语言查询匹配图片
        """
        query_emb = self.encode_text([query])[0]

        if self.image_embeddings is None:
            return []

        similarities = np.dot(self.image_embeddings, query_emb)

        # 按相似度排序
        indices = np.argsort(similarities)[::-1]

        results = []
        for idx in indices:
            sim = float(similarities[idx])
            if sim < threshold:
                break
            if len(results) >= top_k:
                break
            results.append({
                "rank": len(results) + 1,
                "image_path": self.image_paths[idx],
                "similarity": round(sim, 4),
            })

        return results

    def image_to_text_search(self, query_image_path: str,
                             candidate_texts: List[str],
                             top_k: int = 5) -> List[Dict]:
        """
        图片→文本检索
        用图片匹配文本描述
        """
        # 编码查询图片
        img = Image.open(query_image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with __import__("torch").no_grad():
            img_emb = self.model.get_image_features(**inputs).cpu().numpy()[0]
        img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-8)

        # 编码候选文本
        text_embs = self.encode_text(candidate_texts)

        similarities = np.dot(text_embs, img_emb)
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i, idx in enumerate(indices):
            results.append({
                "rank": i + 1,
                "text": candidate_texts[idx][:200],
                "similarity": round(float(similarities[idx]), 4),
            })

        return results

    def combined_search(self, text_query: str,
                        image_query_path: str,
                        text_weight: float = 0.6,
                        image_weight: float = 0.4,
                        top_k: int = 5) -> List[Dict]:
        """
        文本+图片联合查询
        将两种模态的相似度加权融合
        """
        # 文本嵌入
        text_emb = self.encode_text([text_query])[0]

        # 图片嵌入
        img = Image.open(image_query_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with __import__("torch").no_grad():
            img_emb = self.model.get_image_features(**inputs).cpu().numpy()[0]
        img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-8)

        # 组合查询嵌入: 加权平均
        combined_emb = text_weight * text_emb + image_weight * img_emb
        combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-8)

        if self.image_embeddings is None:
            return []

        similarities = np.dot(self.image_embeddings, combined_emb)
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i, idx in enumerate(indices):
            results.append({
                "rank": i + 1,
                "image_path": self.image_paths[idx],
                "combined_similarity": round(float(similarities[idx]), 4),
            })

        return results


# ============================================================
# Section 2: 相似度阈值调优
# ============================================================

class SimilarityThresholdTuner:
    """
    相似度阈值调优器
    问题: 如何确定检索结果的截断阈值？
    方法: 分析正负样本的相似度分布，找到最优F1阈值
    """

    @staticmethod
    def analyze_distribution(similarities: List[float],
                             labels: List[int]) -> Dict:
        """
        分析正负样本的相似度分布
        参数:
            similarities: 所有样本的相似度分数
            labels: 1=相关, 0=不相关
        """
        pos_sims = [s for s, l in zip(similarities, labels) if l == 1]
        neg_sims = [s for s, l in zip(similarities, labels) if l == 0]

        return {
            "positive_mean": float(np.mean(pos_sims)) if pos_sims else 0.0,
            "positive_std": float(np.std(pos_sims)) if pos_sims else 0.0,
            "negative_mean": float(np.mean(neg_sims)) if neg_sims else 0.0,
            "negative_std": float(np.std(neg_sims)) if neg_sims else 0.0,
            "separability": (
                (np.mean(pos_sims) - np.mean(neg_sims)) /
                (np.std(pos_sims) + np.std(neg_sims) + 1e-8)
            ) if pos_sims and neg_sims else 0.0,
            "recommended_threshold": (
                float(np.mean(pos_sims) - np.std(pos_sims))
            ) if pos_sims else 0.5,
        }

    @staticmethod
    def find_optimal_threshold(similarities: np.ndarray,
                               labels: np.ndarray,
                               n_thresholds: int = 50) -> Dict:
        """
        在[0,1]范围内搜索最优F1阈值
        """
        best_f1 = 0.0
        best_threshold = 0.5
        results = []

        for threshold in np.linspace(0.0, 1.0, n_thresholds):
            predictions = (similarities >= threshold).astype(int)
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append({
                "threshold": round(float(threshold), 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            })

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return {
            "optimal_threshold": round(float(best_threshold), 3),
            "best_f1": round(best_f1, 4),
            "curve": results,
        }


# ============================================================
# Section 3: 辅助工具
# ============================================================

def create_demo_images(output_dir: str, count: int = 6) -> List[str]:
    """创建演示用彩色图片（替代真实图片）"""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    themes = [
        ("sunset", "#FF6B35", "#F7C59F", "SUNSET"),
        ("ocean", "#0077B6", "#90E0EF", "OCEAN"),
        ("forest", "#2D6A4F", "#95D5B2", "FOREST"),
        ("city", "#6C757D", "#DEE2E6", "CITY"),
        ("desert", "#E9C46A", "#F4A261", "DESERT"),
        ("mountain", "#606C38", "#DDA15E", "MTN"),
    ]

    for i, (name, color1, color2, label) in enumerate(themes[:count]):
        img = Image.new("RGB", (200, 200), color=color1)
        draw = ImageDraw.Draw(img)
        draw.rectangle([30, 80, 170, 120], fill=color2)
        draw.text((60, 90), label, fill="white")
        path = os.path.join(output_dir, f"demo_{name}.png")
        img.save(path)
        paths.append(path)

    return paths


# ============================================================
# Section 4: 主流程
# ============================================================

def main():
    """主函数：演示CLIP跨模态检索全流程"""
    print("=" * 70)
    print("A2-01: CLIP图片检索 — 完整演示")
    print("=" * 70)

    output_dir = "demo_clip_images"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建演示图片
    print("\n[步骤1] 创建演示图片集...")
    image_paths = create_demo_images(output_dir, count=6)
    for p in image_paths:
        print(f"  {os.path.basename(p)}")

    # 2. 初始化CLIP检索器
    print("\n[步骤2] 初始化CLIP-ViT-B/32...")
    try:
        retriever = CLIPRetriever(device="cpu")
        print(f"  嵌入维度: {retriever.embedding_dim}")

        # 编码图片
        retriever.encode_images(image_paths)

        # 3. 文本→图片检索
        print("\n[步骤3] 文本→图片检索...")
        queries = [
            "a warm orange sunset scene",
            "deep blue water and waves",
            "green trees and nature",
        ]
        for query in queries:
            print(f"\n  查询: '{query}'")
            results = retriever.text_to_image_search(query, top_k=3)
            for r in results:
                fname = os.path.basename(r['image_path'])
                print(f"    #{r['rank']} {fname:<20s} sim={r['similarity']:.4f}")

        # 4. 图片→文本检索
        print("\n[步骤4] 图片→文本检索...")
        candidate_texts = [
            "a photo of a sunset over the ocean",
            "a dense green forest with tall trees",
            "a busy city street at night",
            "a sandy desert landscape",
            "snow-capped mountain peaks",
            "a tropical beach with clear water",
        ]
        query_img = image_paths[0]  # sunset
        print(f"  查询图片: sunset")
        results = retriever.image_to_text_search(query_img, candidate_texts, top_k=3)
        for r in results:
            print(f"    #{r['rank']} {r['text'][:60]}... sim={r['similarity']:.4f}")

        # 5. 联合查询
        print("\n[步骤5] 文本+图片联合查询...")
        combined_results = retriever.combined_search(
            text_query="nature landscape",
            image_query_path=image_paths[2],  # forest
            text_weight=0.5,
            image_weight=0.5,
            top_k=3,
        )
        for r in combined_results:
            fname = os.path.basename(r['image_path'])
            print(f"    #{r['rank']} {fname:<20s} combined_sim={r['combined_similarity']:.4f}")

        # 6. 阈值调优演示
        print("\n[步骤6] 相似度阈值调优...")
        # 模拟正负样本
        np.random.seed(42)
        fake_pos = np.random.normal(0.35, 0.08, 100)
        fake_neg = np.random.normal(0.15, 0.05, 100)
        all_sims = np.concatenate([fake_pos, fake_neg])
        all_labels = np.concatenate([np.ones(100), np.zeros(100)])

        tuner = SimilarityThresholdTuner()
        analysis = tuner.analyze_distribution(all_sims.tolist(), all_labels.tolist())
        print(f"  正样本均值: {analysis['positive_mean']:.4f} ± {analysis['positive_std']:.4f}")
        print(f"  负样本均值: {analysis['negative_mean']:.4f} ± {analysis['negative_std']:.4f}")
        print(f"  可分性: {analysis['separability']:.2f}")
        print(f"  推荐阈值: {analysis['recommended_threshold']:.3f}")

        optimal = tuner.find_optimal_threshold(all_sims, all_labels)
        print(f"  最优F1阈值: {optimal['optimal_threshold']:.3f} (F1={optimal['best_f1']:.4f})")

    except Exception as e:
        print(f"  [错误] CLIP模型加载失败: {e}")
        print("  请确保已安装: pip install transformers torch pillow")
        print("  首次运行会自动下载约600MB的CLIP模型")

    print("\n" + "=" * 70)
    print("A2-01 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
