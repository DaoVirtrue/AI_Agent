#!/usr/bin/env python3
"""
A2-04: 跨语言多模态检索 (Cross-Lingual Multimodal Retrieval)
================================================================
学习目标:
  1. 中文查询 → 英文图片搜索
  2. 多语言CLIP (Jina CLIP v2)的使用
  3. 翻译桥策略 (Translation Bridge)
  4. CrossLingualMultimodalRetriever类设计与实现
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Section 1: 跨语言检索策略
# ============================================================

class CrossLingualStrategy(Enum):
    """跨语言多模态检索策略"""
    TRANSLATION_BRIDGE = "translation_bridge"
    # 先翻译查询，再用单语言CLIP检索

    MULTILINGUAL_CLIP = "multilingual_clip"
    # 直接使用多语言CLIP模型 (e.g., Jina CLIP v2)

    DUAL_ENCODER = "dual_encoder"
    # 双语编码器: 中文文本编码器 + 英文图片编码器联合训练

    ZERO_SHOT_TRANSFER = "zero_shot_transfer"
    # 零样本跨语言迁移: 利用CLIP的弱跨语言能力


# ============================================================
# Section 2: 翻译桥策略
# ============================================================

class TranslationBridge:
    """翻译桥：将中文查询翻译为英文后再进行跨模态检索"""

    def __init__(self, translator_type: str = "simple",
                 api_key: str = None):
        """
        参数:
            translator_type:
                - 'simple': 内置轻量词典翻译
                - 'api':    调用翻译API
                - 'marian': 使用Helsinki-NLP MarianMT模型
        """
        self.type = translator_type
        self.api_key = api_key
        self._marian_model = None
        self._marian_tokenizer = None

        # 内置的中英专业术语词典
        self._domain_dict = {
            # 计算机视觉
            "红色": "red", "蓝色": "blue", "绿色": "green",
            "汽车": "car", "动物": "animal", "猫": "cat",
            "狗": "dog", "鸟": "bird", "花": "flower",
            "天空": "sky", "海洋": "ocean", "山": "mountain",
            "建筑": "building", "城市": "city", "街道": "street",
            "人物": "person", "人脸": "face", "手": "hand",
            "风景": "landscape", "日落": "sunset", "日出": "sunrise",
            # 文档理解
            "图表": "chart", "表格": "table", "图形": "graph",
            "柱状图": "bar chart", "折线图": "line chart",
            "饼图": "pie chart", "流程图": "flowchart",
            "截图": "screenshot", "界面": "interface",
            "按钮": "button", "菜单": "menu", "表单": "form",
            # 动作
            "跑步": "running", "游泳": "swimming", "飞行": "flying",
            "跳跃": "jumping", "坐着": "sitting", "站着": "standing",
        }

    def translate_simple(self, chinese_text: str) -> str:
        """
        简单翻译：基于词典替换 + 模板
        这是最轻量的方案，无需联网
        """
        result = chinese_text

        # 先替换已知术语
        for cn, en in sorted(self._domain_dict.items(),
                             key=lambda x: -len(x[0])):  # 长术语优先
            result = result.replace(cn, en)

        # 添加一些连接词以改善CLIP匹配
        connectors = ["的", "了", "着", "在", "有", "是", "和", "与"]
        for cc in connectors:
            result = result.replace(cc, " ")

        # 如果没有替换任何词，尝试整体翻译常见查询
        common_patterns = {
            "一个": "a", "一张": "a photo of",
            "照片": "photo", "图片": "image",
        }
        for cn, en in common_patterns.items():
            result = result.replace(cn, en)

        return result.strip()

    def translate_marian(self, text: str) -> str:
        """使用Helsinki-NLP/opus-mt-zh-en进行神经机器翻译"""
        if self._marian_model is None:
            from transformers import MarianMTModel, MarianTokenizer
            print("  [加载] Helsinki-NLP/opus-mt-zh-en ...")
            model_name = "Helsinki-NLP/opus-mt-zh-en"
            self._marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self._marian_model = MarianMTModel.from_pretrained(model_name)

        inputs = self._marian_tokenizer([text], return_tensors="pt", padding=True)
        translated = self._marian_model.generate(**inputs)
        result = self._marian_tokenizer.batch_decode(
            translated, skip_special_tokens=True
        )[0]
        return result

    def translate(self, text: str) -> str:
        """统一翻译入口"""
        if self.type == "marian":
            try:
                return self.translate_marian(text)
            except Exception as e:
                print(f"  [警告] Marian翻译失败: {e}，回退到简单翻译")
                return self.translate_simple(text)
        elif self.type == "api":
            # 使用OpenAI/DeepL API（此处省略，用simple代替）
            return self.translate_simple(text)
        else:
            return self.translate_simple(text)


# ============================================================
# Section 3: 多语言CLIP封装 (Jina CLIP v2)
# ============================================================

class MultilingualCLIP:
    """多语言CLIP封装（支持中文等多语言文本）"""

    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        支持的模型:
        - jinaai/jina-clip-v2: Jina CLIP v2 (多语言, 1024维)
        - openai/clip-vit-base-patch32: 原始CLIP (仅英文)
        """
        self.device = device

        # 尝试加载多语言模型
        self.model_name = model_name or "jinaai/jina-clip-v2"
        self._model = None
        self._processor = None
        self._loaded = False

    def _load(self):
        """加载模型"""
        if self._loaded:
            return

        try:
            from transformers import AutoModel, AutoProcessor
            print(f"  [加载] {self.model_name} ...")
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(self.device)
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._loaded = True
        except Exception as e:
            print(f"  [警告] {self.model_name} 加载失败: {e}")
            print("  [回退] 使用openai/clip-vit-base-patch32")
            from transformers import CLIPModel, CLIPProcessor
            self.model_name = "openai/clip-vit-base-patch32"
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._loaded = True

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """编码文本（支持多语言）"""
        self._load()

        if "jina" in self.model_name.lower():
            # Jina CLIP v2 API
            inputs = self._processor(
                text=texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with __import__("torch").no_grad():
                embeddings = self._model.get_text_features(**inputs)
        else:
            # 原始CLIP
            inputs = self._processor(
                text=texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with __import__("torch").no_grad():
                embeddings = self._model.get_text_features(**inputs)

        embeddings = embeddings.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """编码图片"""
        self._load()
        from PIL import Image

        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"图片不存在: {path}")
            images.append(Image.open(path).convert("RGB"))

        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with __import__("torch").no_grad():
            embeddings = self._model.get_image_features(**inputs)

        embeddings = embeddings.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)


# ============================================================
# Section 4: CrossLingualMultimodalRetriever
# ============================================================

class CrossLingualMultimodalRetriever:
    """
    跨语言多模态检索器

    核心流程:
    1. 中文查询 → (翻译桥) → 英文查询 → CLIP检索
    2. 或: 中文查询 → 多语言CLIP → 直接检索

    支持混合策略: 两种方式的结果进行RRF融合
    """

    def __init__(self, strategy: CrossLingualStrategy = CrossLingualStrategy.TRANSLATION_BRIDGE,
                 device: str = "cpu"):
        self.strategy = strategy
        self.device = device

        # 翻译桥
        self.translator = TranslationBridge(translator_type="simple")

        # CLIP模型
        if strategy == CrossLingualStrategy.MULTILINGUAL_CLIP:
            self.clip = MultilingualCLIP(model_name="jinaai/jina-clip-v2", device=device)
        else:
            self.clip = MultilingualCLIP(model_name="openai/clip-vit-base-patch32", device=device)

        # 索引
        self.image_paths: List[str] = []
        self.image_embeddings: Optional[np.ndarray] = None

    def index_images(self, image_paths: List[str]):
        """构建图片索引"""
        self.image_paths = [p for p in image_paths if os.path.exists(p)]
        if not self.image_paths:
            print("  [警告] 没有有效图片")
            return

        self.image_embeddings = self.clip.encode_images(self.image_paths)
        print(f"  [索引] {len(self.image_paths)} 张图片, "
              f"嵌入维度={self.image_embeddings.shape[1]}")

    def search_translation_bridge(self, chinese_query: str,
                                  top_k: int = 5) -> List[Dict]:
        """翻译桥策略：中文→英文→CLIP检索"""
        # 步骤1: 翻译
        english_query = self.translator.translate(chinese_query)
        print(f"  [翻译桥] '{chinese_query}' → '{english_query}'")

        # 步骤2: CLIP文本嵌入
        query_emb = self.clip.encode_text([english_query])[0]

        # 步骤3: 相似度计算
        similarities = np.dot(self.image_embeddings, query_emb)
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i, idx in enumerate(indices):
            results.append({
                "rank": i + 1,
                "image_path": self.image_paths[idx],
                "chinese_query": chinese_query,
                "english_query": english_query,
                "similarity": round(float(similarities[idx]), 4),
                "strategy": "translation_bridge",
            })

        return results

    def search_multilingual_clip(self, chinese_query: str,
                                 top_k: int = 5) -> List[Dict]:
        """多语言CLIP策略：直接用中文查询编码"""
        query_emb = self.clip.encode_text([chinese_query])[0]
        similarities = np.dot(self.image_embeddings, query_emb)
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i, idx in enumerate(indices):
            results.append({
                "rank": i + 1,
                "image_path": self.image_paths[idx],
                "chinese_query": chinese_query,
                "similarity": round(float(similarities[idx]), 4),
                "strategy": "multilingual_clip",
            })

        return results

    def search_hybrid(self, chinese_query: str,
                      top_k: int = 5) -> List[Dict]:
        """
        混合策略：翻译桥 + 多语言CLIP + RRF融合
        （两路结果取并集重新排序）
        """
        bridge_results = self.search_translation_bridge(
            chinese_query, top_k=top_k * 2
        )
        multi_results = self.search_multilingual_clip(
            chinese_query, top_k=top_k * 2
        )

        # 简单融合：取平均排名
        bridge_scores = {
            r["image_path"]: 1.0 / (r["rank"] + 60)
            for r in bridge_results
        }
        multi_scores = {
            r["image_path"]: 1.0 / (r["rank"] + 60)
            for r in multi_results
        }

        all_paths = set(bridge_scores.keys()) | set(multi_scores.keys())
        fused_scores = {}
        for path in all_paths:
            s1 = bridge_scores.get(path, 0)
            s2 = multi_scores.get(path, 0)
            fused_scores[path] = s1 + s2

        sorted_paths = sorted(fused_scores.items(),
                              key=lambda x: x[1], reverse=True)

        results = []
        for i, (path, score) in enumerate(sorted_paths[:top_k]):
            # 获取原始rank
            bridge_rank = next(
                (r["rank"] for r in bridge_results if r["image_path"] == path), "-"
            )
            multi_rank = next(
                (r["rank"] for r in multi_results if r["image_path"] == path), "-"
            )
            results.append({
                "rank": i + 1,
                "image_path": path,
                "chinese_query": chinese_query,
                "fused_score": round(score, 6),
                "bridge_rank": bridge_rank,
                "multilingual_rank": multi_rank,
                "strategy": "hybrid",
            })

        return results

    def search(self, chinese_query: str, top_k: int = 5) -> List[Dict]:
        """统一搜索入口"""
        if self.strategy == CrossLingualStrategy.TRANSLATION_BRIDGE:
            return self.search_translation_bridge(chinese_query, top_k)
        elif self.strategy == CrossLingualStrategy.MULTILINGUAL_CLIP:
            return self.search_multilingual_clip(chinese_query, top_k)
        else:
            return self.search_hybrid(chinese_query, top_k)


# ============================================================
# Section 5: 评估与对比
# ============================================================

class CrossLingualEvaluator:
    """跨语言检索评估器"""

    @staticmethod
    def compare_strategies(retriever: CrossLingualMultimodalRetriever,
                           test_queries: List[Tuple[str, str, List[str]]],
                           ) -> Dict:
        """
        对比不同策略的效果

        参数:
            test_queries: [(中文查询, 英文参照, [期望的图片路径]), ...]
        """
        results = {
            "translation_bridge": {"hits": 0, "total": 0},
            "multilingual_clip": {"hits": 0, "total": 0},
            "hybrid": {"hits": 0, "total": 0},
        }

        for cn_query, en_query, expected_images in test_queries:
            # 翻译桥
            bridge_results = retriever.search_translation_bridge(cn_query, top_k=5)
            bridge_hit = any(
                any(exp in r["image_path"] for exp in expected_images)
                for r in bridge_results
            )
            if bridge_hit:
                results["translation_bridge"]["hits"] += 1
            results["translation_bridge"]["total"] += 1

            # 多语言
            if retriever.strategy != CrossLingualStrategy.TRANSLATION_BRIDGE:
                multi_results = retriever.search_multilingual_clip(cn_query, top_k=5)
                multi_hit = any(
                    any(exp in r["image_path"] for exp in expected_images)
                    for r in multi_results
                )
                if multi_hit:
                    results["multilingual_clip"]["hits"] += 1
                results["multilingual_clip"]["total"] += 1

            # 混合
            hybrid_results = retriever.search_hybrid(cn_query, top_k=5)
            hybrid_hit = any(
                any(exp in r["image_path"] for exp in expected_images)
                for r in hybrid_results
            )
            if hybrid_hit:
                results["hybrid"]["hits"] += 1
            results["hybrid"]["total"] += 1

        # 计算Hit Rate
        for strategy, data in results.items():
            if data["total"] > 0:
                data["hit_rate"] = data["hits"] / data["total"]

        return results


# ============================================================
# Section 6: 主流程
# ============================================================

def create_demo_images(output_dir: str) -> List[str]:
    """创建演示图片集"""
    from PIL import Image, ImageDraw
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    themes = [
        ("red_car", "#D32F2F", "RED CAR"),
        ("blue_ocean", "#1976D2", "BLUE OCEAN"),
        ("green_forest", "#388E3C", "GREEN FOREST"),
        ("city_building", "#616161", "CITY"),
        ("sunset_view", "#F57C00", "SUNSET"),
        ("mountain_peak", "#455A64", "MOUNTAIN"),
    ]

    for name, color, label in themes:
        img = Image.new("RGB", (200, 200), color=color)
        draw = ImageDraw.Draw(img)
        draw.text((40, 90), label, fill="white")
        path = os.path.join(output_dir, f"{name}.png")
        img.save(path)
        paths.append(path)

    return paths


def main():
    """主函数：演示跨语言多模态检索全流程"""
    print("=" * 70)
    print("A2-04: 跨语言多模态检索 — 完整演示")
    print("=" * 70)

    output_dir = "demo_cross_lingual"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建演示图片
    print("\n[步骤1] 创建演示图片集...")
    image_paths = create_demo_images(output_dir)
    for p in image_paths:
        print(f"  {os.path.basename(p)}")

    # 2. 翻译桥演示
    print("\n[步骤2] 翻译桥策略演示...")
    translator = TranslationBridge(translator_type="simple")
    test_terms = ["红色汽车", "蓝色海洋", "城市建筑", "日落风景"]
    for term in test_terms:
        translated = translator.translate(term)
        print(f"  '{term}' → '{translated}'")

    # 3. 初始化检索器
    print("\n[步骤3] 初始化跨语言检索器...")
    try:
        retriever = CrossLingualMultimodalRetriever(
            strategy=CrossLingualStrategy.TRANSLATION_BRIDGE,
            device="cpu",
        )
        retriever.index_images(image_paths)

        # 4. 中文查询→英文图片检索
        print("\n[步骤4] 中文查询→英文图片检索...")
        cn_queries = [
            "红色汽车",
            "蓝色海洋海浪",
            "绿色森林树木",
            "日落橙色的天空",
        ]

        for query in cn_queries:
            print(f"\n  查询: '{query}'")
            results = retriever.search(query, top_k=3)
            for r in results:
                fname = os.path.basename(r["image_path"])
                print(f"    #{r['rank']} {fname:<20s} sim={r['similarity']:.4f} "
                      f"策略={r.get('strategy','?')}")

        # 5. 策略对比
        print("\n[步骤5] 策略对比...")
        test_data = [
            ("红色汽车", "red car", ["red_car"]),
            ("蓝色海洋", "blue ocean", ["blue_ocean"]),
            ("绿色自然", "green nature", ["green_forest"]),
            ("城市高楼大厦", "city tall buildings", ["city_building"]),
        ]
        evaluator = CrossLingualEvaluator()
        comparison = evaluator.compare_strategies(retriever, test_data)
        for strategy, data in comparison.items():
            if data["total"] > 0:
                print(f"  {strategy}: Hit Rate = {data.get('hit_rate', 0):.2%} "
                      f"({data['hits']}/{data['total']})")

    except Exception as e:
        print(f"  [跳过] CLIP模型加载失败: {e}")
        print("  跨语言检索架构已完整展示，核心逻辑可运行")

    print("\n" + "=" * 70)
    print("A2-04 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
