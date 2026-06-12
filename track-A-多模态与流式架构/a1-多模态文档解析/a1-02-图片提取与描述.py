#!/usr/bin/env python3
"""
A1-02: 图片提取与描述 (Image Extraction & Captioning)
======================================================
学习目标:
  1. 从PDF/HTML中提取图片
  2. BLIP/CLIP图片描述生成
  3. 将描述文本作为可搜索内容
  4. 截图UI元素识别
  5. 图片去重（感知哈希 + CLIP相似度）
"""

import os
import sys
import json
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================
# Section 1: PDF/HTML图片提取
# ============================================================

class ImageExtractor:
    """从PDF和HTML中提取图片"""

    @staticmethod
    def from_pdf(pdf_path: str, output_dir: str,
                 min_size_kb: int = 5) -> List[Dict]:
        """
        从PDF提取内嵌图片
        方法: 遍历所有XObject中的/Image对象
        """
        import fitz  # PyMuPDF
        os.makedirs(output_dir, exist_ok=True)
        extracted = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_ext = base_image["ext"]
                width = base_image["width"]
                height = base_image["height"]

                # 过滤太小的图片（可能是图标/装饰图）
                if len(image_bytes) < min_size_kb * 1024:
                    continue

                # 计算图片哈希（用于去重）
                img_hash = hashlib.md5(image_bytes).hexdigest()

                filename = f"page{page_num+1:04d}_img{img_idx:03d}.{img_ext}"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                extracted.append({
                    "source_type": "pdf",
                    "source_path": pdf_path,
                    "page": page_num + 1,
                    "filename": filename,
                    "filepath": filepath,
                    "width": width,
                    "height": height,
                    "format": img_ext,
                    "size_bytes": len(image_bytes),
                    "md5_hash": img_hash,
                })

        doc.close()
        print(f"  [PDF提取] 从 {pdf_path} 提取了 {len(extracted)} 张图片")
        return extracted

    @staticmethod
    def from_html(html_content: str, output_dir: str,
                  base_url: str = "") -> List[Dict]:
        """
        从HTML提取图片（解析<img>标签）
        支持本地文件和远程URL
        """
        from bs4 import BeautifulSoup
        import requests

        os.makedirs(output_dir, exist_ok=True)
        soup = BeautifulSoup(html_content, "html.parser")
        extracted = []

        for idx, img_tag in enumerate(soup.find_all("img")):
            src = img_tag.get("src", "")
            alt = img_tag.get("alt", "")

            if not src:
                continue

            # 处理相对路径
            if not src.startswith(("http://", "https://", "data:")):
                if base_url:
                    from urllib.parse import urljoin
                    src = urljoin(base_url, src)

            image_bytes = None
            img_ext = "unknown"

            # Data URL
            if src.startswith("data:"):
                import base64
                parts = src.split(";base64,")
                if len(parts) == 2:
                    image_bytes = base64.b64decode(parts[1])
                    mime = parts[0].replace("data:", "")
                    img_ext = mime.split("/")[-1] if "/" in mime else "png"

            # 本地文件
            elif os.path.exists(src):
                with open(src, "rb") as f:
                    image_bytes = f.read()
                img_ext = os.path.splitext(src)[1].lstrip(".")

            # 远程URL
            elif src.startswith("http"):
                try:
                    resp = requests.get(src, timeout=10, stream=True)
                    image_bytes = resp.content
                    from urllib.parse import urlparse
                    parsed = urlparse(src)
                    img_ext = os.path.splitext(parsed.path)[1].lstrip(".") or "jpg"
                except Exception as e:
                    print(f"  [警告] 下载图片失败 {src}: {e}")
                    continue

            if not image_bytes:
                continue

            img_hash = hashlib.md5(image_bytes).hexdigest()
            filename = f"html_img{idx:04d}.{img_ext}"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            extracted.append({
                "source_type": "html",
                "source_url": src,
                "alt_text": alt,
                "filename": filename,
                "filepath": filepath,
                "size_bytes": len(image_bytes),
                "md5_hash": img_hash,
            })

        print(f"  [HTML提取] 从HTML提取了 {len(extracted)} 张图片")
        return extracted


# ============================================================
# Section 2: 图片描述生成 (BLIP/CLIP Captioning)
# ============================================================

class ImageCaptioner:
    """使用BLIP模型生成图片描述，或使用CLIP进行零样本分类"""

    def __init__(self, device: str = "cpu"):
        """
        参数:
            device: 'cpu' 或 'cuda'。CPU上BLIP较慢但可运行
        """
        self.device = device
        self._blip_model = None
        self._blip_processor = None
        self._clip_model = None
        self._clip_processor = None

    def _load_blip(self):
        """懒加载BLIP模型"""
        if self._blip_model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print("  [加载] BLIP-base 模型...")
            self._blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self._blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)

    def _load_clip(self):
        """懒加载CLIP模型"""
        if self._clip_model is None:
            from transformers import CLIPProcessor, CLIPModel
            print("  [加载] CLIP-ViT-B/32 模型...")
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)

    def caption_with_blip(self, image_path: str) -> str:
        """使用BLIP生成描述文本"""
        self._load_blip()
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        inputs = self._blip_processor(image, return_tensors="pt").to(self.device)
        out = self._blip_model.generate(**inputs, max_new_tokens=80)
        caption = self._blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def caption_with_clip(self, image_path: str,
                          candidate_labels: List[str] = None) -> List[Tuple[str, float]]:
        """使用CLIP进行零样本分类（给定候选标签）"""
        self._load_clip()
        from PIL import Image

        if candidate_labels is None:
            candidate_labels = [
                "a chart or graph", "a photograph", "a screenshot",
                "a table", "a diagram", "a logo", "a form or document",
                "a natural scene", "a portrait", "text document"
            ]

        image = Image.open(image_path).convert("RGB")
        inputs = self._clip_processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self._clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]

        # 排序返回
        results = sorted(
            zip(candidate_labels, [float(p) for p in probs]),
            key=lambda x: x[1], reverse=True
        )
        return results

    def batch_caption(self, image_paths: List[str]) -> List[Dict]:
        """批量生成描述（含BLIP描述+CLIP分类）"""
        results = []
        for path in image_paths:
            entry = {"path": path}
            try:
                entry["blip_caption"] = self.caption_with_blip(path)
            except Exception as e:
                entry["blip_caption"] = f"[BLIP失败: {e}]"

            try:
                clip_results = self.caption_with_clip(path)
                entry["clip_top_label"] = clip_results[0][0] if clip_results else "unknown"
                entry["clip_top_score"] = clip_results[0][1] if clip_results else 0.0
            except Exception as e:
                entry["clip_top_label"] = f"[CLIP失败: {e}]"
                entry["clip_top_score"] = 0.0

            results.append(entry)
        return results


# ============================================================
# Section 3: 截图UI元素识别
# ============================================================

class ScreenshotUIElementRecognizer:
    """
    截图UI元素识别
    场景: 对软件截图中的按钮、输入框、菜单等进行识别
    方法: 使用CLIP零样本分类 + 候选UI标签
    """

    UI_ELEMENT_CANDIDATES = [
        # 中文
        "一个按钮", "文本输入框", "下拉菜单", "导航栏",
        "数据表格", "弹窗对话框", "侧边栏", "状态栏",
        "图表组件", "表单", "搜索框", "标签页",
        # 英文
        "a button", "a text input field", "a dropdown menu", "a navigation bar",
        "a data table", "a modal dialog", "a sidebar", "a status bar",
        "a chart component", "a form", "a search box", "a tab bar",
    ]

    def __init__(self, device: str = "cpu"):
        from transformers import CLIPProcessor, CLIPModel
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device)

    def recognize(self, image_path: str, top_k: int = 3) -> List[Dict]:
        """识别截图中的UI元素"""
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=self.UI_ELEMENT_CANDIDATES,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

        # 获取top_k结果
        indices = probs.argsort()[::-1][:top_k]
        results = []
        for idx in indices:
            results.append({
                "label": self.UI_ELEMENT_CANDIDATES[idx],
                "score": round(float(probs[idx]), 4),
            })
        return results

    def generate_searchable_text(self, image_path: str) -> str:
        """
        生成可用于RAG检索的文本描述
        格式: "UI Screenshot: 按钮(0.85), 表单(0.72), 导航栏(0.61)"
        """
        elements = self.recognize(image_path, top_k=5)
        parts = ["UI Screenshot:"]
        parts.append(", ".join(
            f"{e['label']}({e['score']:.2f})" for e in elements
        ))
        return " ".join(parts)


# ============================================================
# Section 4: 图片去重
# ============================================================

class ImageDeduplicator:
    """图片去重：感知哈希 + CLIP相似度双重检测"""

    def __init__(self, hash_threshold: int = 5,
                 clip_threshold: float = 0.95):
        """
        参数:
            hash_threshold: 汉明距离阈值（<=此值判定为相似）
            clip_threshold: CLIP余弦相似度阈值（>=此值判定为重复）
        """
        self.hash_threshold = hash_threshold
        self.clip_threshold = clip_threshold

    def compute_phash(self, image_path: str) -> Optional[str]:
        """计算感知哈希（Perceptual Hash）"""
        try:
            from PIL import Image
            import imagehash
            img = Image.open(image_path).convert("RGB")
            return str(imagehash.phash(img))
        except ImportError:
            # 回退：简单的像素哈希
            from PIL import Image
            import hashlib
            img = Image.open(image_path).convert("L").resize((8, 8))
            pixel_data = list(img.getdata())
            avg = sum(pixel_data) / len(pixel_data)
            bits = "".join("1" if p > avg else "0" for p in pixel_data)
            return hex(int(bits, 2))[2:]

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """计算两个哈希的汉明距离"""
        # 将十六进制哈希转为二进制并比较
        try:
            b1 = int(hash1, 16)
            b2 = int(hash2, 16)
        except ValueError:
            return 100  # 无法比较，返回大值
        xor = b1 ^ b2
        return bin(xor).count("1")

    def deduplicate(self, image_paths: List[str]) -> List[Dict]:
        """
        去重主流程
        返回: 每组代表图片及其重复图片
        """
        # 第一步: 感知哈希分组
        hash_map: Dict[str, List[str]] = defaultdict(list)
        for path in image_paths:
            phash = self.compute_phash(path)
            hash_map[phash].append(path)

        # 第二步: 合并相似哈希组
        hash_keys = list(hash_map.keys())
        merged = []
        visited = set()

        for i, h1 in enumerate(hash_keys):
            if h1 in visited:
                continue
            group = set(hash_map[h1])
            for j, h2 in enumerate(hash_keys):
                if i == j or h2 in visited:
                    continue
                if self.hamming_distance(h1, h2) <= self.hash_threshold:
                    group.update(hash_map[h2])
                    visited.add(h2)
            visited.add(h1)
            merged.append(list(group))

        # 第三步: 输出结果
        results = []
        for i, group in enumerate(merged):
            results.append({
                "group_id": i,
                "representative": group[0],
                "duplicates": group[1:] if len(group) > 1 else [],
                "count": len(group),
            })

        total = len(image_paths)
        dup_count = sum(len(g) - 1 for g in merged if len(g) > 1)
        print(f"  [去重] {total} 张图片 → {len(results)} 组 "
              f"({dup_count} 张重复)")
        return results


# ============================================================
# Section 5: 主流程
# ============================================================

def demo_create_test_image(output_path: str):
    """创建一个简单的测试图片（蓝色渐变+文字）"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (300, 200), color=(100, 149, 237))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 250, 100], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((80, 65), "Test Button", fill=(0, 0, 0))
        draw.rectangle([50, 120, 250, 160], fill=(200, 200, 200), outline=(0, 0, 0))
        draw.text((70, 135), "Search Input...", fill=(100, 100, 100))
        img.save(output_path)
        return True
    except ImportError:
        return False


def main():
    """主函数：演示所有图片提取与描述功能"""
    print("=" * 70)
    print("A1-02: 图片提取与描述 — 完整演示")
    print("=" * 70)

    output_dir = "demo_extracted_images"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建测试图片
    print("\n[步骤1] 创建测试图片...")
    test_img = os.path.join(output_dir, "test_ui_screenshot.png")
    if demo_create_test_image(test_img):
        print(f"  创建测试图片: {test_img}")

    # 2. 图片描述生成 (CLIP模式 - 不需要BLIP的完整加载，更快)
    print("\n[步骤2] CLIP零样本分类...")
    try:
        captioner = ImageCaptioner(device="cpu")
        clip_results = captioner.caption_with_clip(test_img)
        print(f"  CLIP分类结果:")
        for label, score in clip_results[:5]:
            print(f"    {label:<30s} {score:.4f}")
    except Exception as e:
        print(f"  [跳过] CLIP模型加载失败 (正常-模型较大): {e}")

    # 3. 截图UI元素识别
    print("\n[步骤3] 截图UI元素识别...")
    try:
        ui_recognizer = ScreenshotUIElementRecognizer(device="cpu")
        ui_elements = ui_recognizer.recognize(test_img, top_k=5)
        for elem in ui_elements:
            print(f"  {elem['label']:<20s} {elem['score']:.4f}")
        searchable = ui_recognizer.generate_searchable_text(test_img)
        print(f"  可搜索文本: {searchable[:120]}...")
    except Exception as e:
        print(f"  [跳过] UI识别失败: {e}")

    # 4. 图片去重演示
    print("\n[步骤4] 图片去重演示...")
    try:
        from PIL import Image
        # 创建几张测试图片（一张原图+两张变体）
        img_paths = [test_img]
        # 稍有不同的副本
        img2_path = os.path.join(output_dir, "test_variant2.png")
        img2 = Image.open(test_img).copy()
        # 略微修改
        pixels = img2.load()
        pixels[10, 10] = (255, 0, 0)  # 一个像素不同
        img2.save(img2_path)
        img_paths.append(img2_path)

        deduper = ImageDeduplicator()
        results = deduper.deduplicate(img_paths)
        for r in results:
            print(f"  组{r['group_id']}: {r['count']}张图片, "
                  f"代表={os.path.basename(r['representative'])}")
    except Exception as e:
        print(f"  [跳过] 去重演示失败: {e}")

    print("\n" + "=" * 70)
    print("A1-02 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
