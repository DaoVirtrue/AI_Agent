#!/usr/bin/env python3
"""
A1-01: PDF深度解析 (PDF Deep Parsing)
=========================================
学习目标:
  1. 区分扫描件PDF与数字PDF
  2. 处理双层PDF（图像层+文本层叠加）
  3. PaddleOCR vs Tesseract效果对比
  4. 表格提取（Camelot Lattice+Stream, pdfplumber fallback）
  5. 多栏布局处理
"""

import io
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# Section 1: PDF类型检测 (扫描件 vs 数字PDF)
# ============================================================

class PDFType(Enum):
    """PDF文档类型枚举"""
    DIGITAL = "digital"        # 数字PDF（文本可选择/可搜索）
    SCANNED = "scanned"        # 扫描件PDF（纯图像，需OCR）
    DUAL_LAYER = "dual_layer"  # 双层PDF（图像+文本层叠加）


class PDFTypeDetector:
    """检测PDF类型：数字/扫描件/双层"""

    def __init__(self, text_coverage_threshold: float = 0.3):
        """
        参数:
            text_coverage_threshold: 文本覆盖率阈值
                低于此值判定为扫描件
                介于阈值和阈值*3之间且图片量大判定为双层
                高于阈值*3判定为数字PDF
        """
        self.threshold = text_coverage_threshold

    def detect(self, pdf_path: str) -> PDFType:
        """
        三步检测法:
        1. 尝试提取文本 — 若无文本/文本极少，判定为扫描件
        2. 提取图片 — 若同时有文本和大量图片，判定为双层PDF
        3. 否则为数字PDF
        """
        text_pages = self._extract_text_per_page(pdf_path)
        image_pages = self._extract_images_per_page(pdf_path)
        total_pages = max(len(text_pages), len(image_pages))

        if total_pages == 0:
            return PDFType.SCANNED

        # 统计有文本的页面比例
        pages_with_text = sum(1 for t in text_pages if len(t.strip()) > 50)
        text_coverage = pages_with_text / total_pages

        # 统计有图片的页面比例
        pages_with_images = sum(1 for imgs in image_pages if len(imgs) > 0)
        image_coverage = pages_with_images / total_pages

        print(f"  [检测] 文本覆盖率: {text_coverage:.2%}, 图片覆盖率: {image_coverage:.2%}")

        if text_coverage < self.threshold:
            return PDFType.SCANNED
        elif text_coverage < self.threshold * 3 and image_coverage > 0.5:
            return PDFType.DUAL_LAYER
        else:
            return PDFType.DIGITAL

    def _extract_text_per_page(self, pdf_path: str) -> List[str]:
        """使用PyPDF2提取每页文本（轻量级检测用）"""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            return [page.extract_text() or "" for page in reader.pages]
        except ImportError:
            # 回退方案：使用pdfplumber
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                return [p.extract_text() or "" for p in pdf.pages]
        except Exception as e:
            warnings.warn(f"文本提取失败: {e}")
            return []

    def _extract_images_per_page(self, pdf_path: str) -> List[List]:
        """统计每页的内嵌图片数量"""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            result = []
            for page in reader.pages:
                if '/XObject' in page['/Resources']:
                    xobjects = page['/Resources']['/XObject'].get_object()
                    images = [k for k, v in xobjects.items()
                              if v['/Subtype'] == '/Image']
                    result.append(images)
                else:
                    result.append([])
            return result
        except Exception as e:
            warnings.warn(f"图片检测失败: {e}")
            return [[]]


# ============================================================
# Section 2: 双层PDF处理
# ============================================================

class DualLayerPDFProcessor:
    """
    双层PDF处理器
    场景：扫描件+OCR文本层叠加的PDF
    策略：图像层用于视觉分析，文本层用于检索
    """

    @staticmethod
    def extract_text_layer(pdf_path: str) -> List[Dict]:
        """提取文本层内容（已有OCR结果）"""
        import pdfplumber
        results = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                results.append({
                    "page": i + 1,
                    "text": text or "",
                    "char_count": len(text) if text else 0,
                })
        return results

    @staticmethod
    def extract_image_layer(pdf_path: str, output_dir: str) -> List[Dict]:
        """
        提取图像层（用于视觉分析/重做OCR）
        使用pdf2image将PDF页面转为图片
        """
        try:
            from pdf2image import convert_from_path
            os.makedirs(output_dir, exist_ok=True)
            images = convert_from_path(pdf_path, dpi=300)
            results = []
            for i, img in enumerate(images):
                img_path = os.path.join(output_dir, f"page_{i+1:04d}.png")
                img.save(img_path, "PNG")
                results.append({
                    "page": i + 1,
                    "image_path": img_path,
                    "size": img.size,  # (width, height)
                })
            return results
        except ImportError:
            print("[警告] pdf2image未安装，跳过图像层提取")
            return []

    @staticmethod
    def merge_layers(text_layer: List[Dict],
                     image_ocr_results: List[str]) -> List[Dict]:
        """
        合并文本层与图像OCR层
        策略：优先使用文本层；若文本层为空则使用OCR结果
        """
        merged = []
        for i, (text_entry, ocr_text) in enumerate(
            zip(text_layer, image_ocr_results)
        ):
            final_text = text_entry["text"] if text_entry["text"] and len(
                text_entry["text"].strip()) > 20 else ocr_text
            merged.append({
                "page": i + 1,
                "text": final_text,
                "source": "text_layer" if text_entry["text"] and len(
                    text_entry["text"].strip()) > 20 else "ocr_layer",
            })
        return merged


# ============================================================
# Section 3: OCR引擎对比 (PaddleOCR vs Tesseract)
# ============================================================

class OCREngineComparison:
    """PaddleOCR vs Tesseract 效果对比"""

    @staticmethod
    def run_paddleocr(image_path: str) -> Tuple[str, float]:
        """使用PaddleOCR进行识别"""
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang='ch', use_angle_cls=True, show_log=False)
            result = ocr.ocr(image_path, cls=True)
            if not result or not result[0]:
                return "", 0.0
            texts = []
            confidences = []
            for line in result[0]:
                texts.append(line[1][0])
                confidences.append(line[1][1])
            return "\n".join(texts), sum(confidences) / len(confidences)
        except ImportError:
            return "[PaddleOCR未安装]", 0.0

    @staticmethod
    def run_tesseract(image_path: str, lang: str = "chi_sim+eng") -> Tuple[str, float]:
        """使用Tesseract进行识别"""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=lang)
            # Tesseract不直接返回置信度，使用data获取
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data['conf'] if c != '-1' and int(c) > 0]
            avg_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
            return text.strip(), avg_conf
        except ImportError:
            return "[Tesseract未安装]", 0.0

    @classmethod
    def compare(cls, image_path: str, ground_truth: str = "") -> Dict:
        """
        对比两个OCR引擎
        返回:
            {
                "paddle": {"text": ..., "confidence": ..., "cer": ...},
                "tesseract": {"text": ..., "confidence": ..., "cer": ...},
                "winner": "paddle" or "tesseract"
            }
        """
        paddle_text, paddle_conf = cls.run_paddleocr(image_path)
        tess_text, tess_conf = cls.run_tesseract(image_path)

        # 简易CER计算（Character Error Rate）
        def compute_cer(pred: str, gt: str) -> float:
            if not gt:
                return -1.0
            import Levenshtein
            return Levenshtein.distance(pred, gt) / len(gt)

        paddle_cer = compute_cer(paddle_text, ground_truth)
        tess_cer = compute_cer(tess_text, ground_truth)

        # 判断胜者：优先看CER，其次看置信度
        if paddle_cer >= 0 and tess_cer >= 0:
            winner = "paddle" if paddle_cer < tess_cer else "tesseract"
        else:
            winner = "paddle" if paddle_conf > tess_conf else "tesseract"

        return {
            "paddle": {
                "text_preview": paddle_text[:500],
                "confidence": paddle_conf,
                "cer": paddle_cer,
            },
            "tesseract": {
                "text_preview": tess_text[:500],
                "confidence": tess_conf,
                "cer": tess_cer,
            },
            "winner": winner,
        }

    @classmethod
    def accuracy_table(cls) -> str:
        """
        典型场景下的准确率对比表（经验数据）
        展示常见文档类型下两个引擎的表现差异
        """
        # 基于社区基准的经验数据
        scenarios = [
            # (场景, PaddleOCR CER, Tesseract CER, Paddle速度, Tesseract速度)
            ("中文印刷体", "1.2%", "5.8%", "快", "中"),
            ("中文手写体", "8.5%", "25.3%", "快", "中"),
            ("中英混排", "2.1%", "7.2%", "快", "快"),
            ("英文印刷体", "1.8%", "1.5%", "中", "快"),
            ("表格文字", "3.5%", "12.0%", "快", "中"),
            ("竖排文字", "4.2%", "30.1%", "快", "差"),
            ("低质量扫描", "6.8%", "18.5%", "快", "中"),
        ]
        header = f"{'场景':<16} {'PaddleOCR':<12} {'Tesseract':<12} {'Paddle速度':<10} {'Tesseract速度':<12}"
        sep = "-" * 65
        lines = [header, sep]
        for s in scenarios:
            lines.append(
                f"{s[0]:<16} {s[1]:<12} {s[2]:<12} {s[3]:<10} {s[4]:<12}"
            )
        return "\n".join(lines)


# ============================================================
# Section 4: 表格提取 (Camelot + pdfplumber fallback)
# ============================================================

class TableExtractor:
    """
    表格提取器
    - 优先使用Camelot（lattice模式用于有边框表格，stream模式用于无边框表格）
    - Camelot失败时回退到pdfplumber
    """

    @staticmethod
    def extract_with_camelot(pdf_path: str,
                             pages: str = "1-end",
                             flavor: str = "lattice") -> List[Dict]:
        """
        Camelot表格提取
        flavor='lattice': 适用于有明确边框线的表格
        flavor='stream':  适用于无边框线、仅用空白分隔的表格
        """
        try:
            import camelot
            tables = camelot.read_pdf(
                pdf_path, pages=pages, flavor=flavor
            )
            results = []
            for idx, table in enumerate(tables):
                results.append({
                    "index": idx,
                    "page": table.page,
                    "accuracy": table.accuracy,
                    "whitespace": table.whitespace,
                    "data": table.df.to_dict(orient="records"),
                    "flavor": flavor,
                })
            print(f"  [Camelot/{flavor}] 提取了 {len(tables)} 个表格")
            return results
        except Exception as e:
            print(f"  [Camelot/{flavor}] 失败: {e}")
            return []

    @staticmethod
    def extract_with_pdfplumber(pdf_path: str) -> List[Dict]:
        """pdfplumber表格提取（回退方案）"""
        try:
            import pdfplumber
            results = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for idx, table in enumerate(tables):
                        if table:
                            # 将表格数据转换为dict格式
                            headers = table[0] if table else []
                            rows = []
                            for row in table[1:]:
                                row_dict = {}
                                for h, cell in zip(headers, row):
                                    row_dict[str(h) if h else "col"] = cell
                                rows.append(row_dict)
                            results.append({
                                "index": idx,
                                "page": page_num + 1,
                                "data": rows if rows else table,
                                "flavor": "pdfplumber",
                            })
            print(f"  [pdfplumber] 提取了 {len(results)} 个表格")
            return results
        except Exception as e:
            print(f"  [pdfplumber] 失败: {e}")
            return []

    @classmethod
    def extract_all(cls, pdf_path: str,
                    pages: str = "1-end") -> List[Dict]:
        """
        完整提取流程:
        1. Camelot Lattice → 2. Camelot Stream → 3. pdfplumber fallback
        """
        all_tables = []

        # 步骤1: Camelot Lattice
        lattice_tables = cls.extract_with_camelot(
            pdf_path, pages=pages, flavor="lattice"
        )
        all_tables.extend(lattice_tables)

        # 步骤2: Camelot Stream（仅在没有Lattice结果的页面尝试）
        if not lattice_tables:
            stream_tables = cls.extract_with_camelot(
                pdf_path, pages=pages, flavor="stream"
            )
            all_tables.extend(stream_tables)

            # 步骤3: pdfplumber收尾
            if not stream_tables:
                plumber_tables = cls.extract_with_pdfplumber(pdf_path)
                all_tables.extend(plumber_tables)

        return all_tables


# ============================================================
# Section 5: 多栏布局处理
# ============================================================

class MultiColumnLayoutHandler:
    """
    多栏布局处理
    问题：pdfplumber/Tesseract在多栏PDF中常将左右两栏的文字交错读取
    解决：检测栏位并分别排序文本块
    """

    @staticmethod
    def detect_columns(page) -> int:
        """检测页面栏数（基于文本块的水平位置聚类）"""
        try:
            words = page.extract_words()
            if not words:
                return 1
            x0_positions = [w['x0'] for w in words]
            page_width = page.width
            # 将x0位置归一化并聚类
            normalized = [x / page_width for x in x0_positions]
            # 简单的聚类：统计不同区间的文本块
            left_count = sum(1 for x in normalized if x < 0.48)
            right_count = sum(1 for x in normalized if x > 0.52)
            total = max(left_count + right_count, 1)
            left_ratio = left_count / total
            # 左右均衡分布 → 双栏
            if 0.3 < left_ratio < 0.7 and left_count > 3 and right_count > 3:
                return 2
            return 1
        except Exception:
            return 1

    @classmethod
    def extract_by_column(cls, page) -> Dict[int, str]:
        """
        按栏提取文本
        返回: {column_id: text}
        """
        try:
            words = page.extract_words()
            if not words:
                return {0: page.extract_text() or ""}

            num_cols = cls.detect_columns(page)
            if num_cols == 1:
                return {0: page.extract_text() or ""}

            page_width = page.width
            # 按x0位置分配到不同栏
            columns: Dict[int, List] = {0: [], 1: []}
            for word in words:
                col = 0 if word['x0'] / page_width < 0.5 else 1
                columns[col].append(word)

            # 每栏内按y0（自上而下）排序
            result = {}
            for col_id, col_words in columns.items():
                col_words.sort(key=lambda w: (w['top'], w['x0']))
                result[col_id] = " ".join(w['text'] for w in col_words)

            return result
        except Exception as e:
            return {0: f"[多栏提取失败: {e}]"}

    @classmethod
    def process_pdf(cls, pdf_path: str) -> List[Dict]:
        """处理整个PDF的多栏布局"""
        import pdfplumber
        results = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                cols = cls.extract_by_column(page)
                results.append({
                    "page": i + 1,
                    "num_columns": len(cols),
                    "columns": cols,
                })
        return results


# ============================================================
# Section 6: 主流程
# ============================================================

def demo_create_sample_pdf() -> str:
    """创建一个示例PDF用于演示（使用reportlab）"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, Frame, Table as RLTable

        output_path = "demo_sample.pdf"
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4

        # Page 1: 中文文本
        c.setFont("Helvetica", 24)
        c.drawString(100, height - 100, "RAG Multimodal Document Parsing Demo")
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 140, "This is a sample PDF for demonstrating PDF parsing capabilities.")
        c.drawString(100, height - 160, "It contains text, simulated table data, and multiple pages.")
        c.drawString(100, height - 200, "检索增强生成（RAG）系统需要处理多种文档格式。")
        c.drawString(100, height - 220, "多模态文档解析是构建高质量RAG系统的关键步骤之一。")
        c.drawString(100, height - 260, "Column A: The left column contains English text for layout testing.")
        c.drawString(100, height - 300, "Column B: The right column contains parallel content in another column.")
        c.save()
        return output_path
    except ImportError:
        # 无reportlab时创建一个最小的PDF
        output_path = "demo_sample.pdf"
        with open(output_path, "wb") as f:
            f.write(
                b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
                b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 100 700 Td (RAG Demo PDF) Tj ET\nendstream\nendobj\n"
                b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000248 00000 n \n0000000342 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n411\n%%EOF"
            )
        return output_path


def main():
    """主函数：演示所有PDF解析功能"""
    print("=" * 70)
    print("A1-01: PDF深度解析 — 完整演示")
    print("=" * 70)

    # 1. 创建示例PDF
    print("\n[步骤1] 创建示例PDF...")
    sample_pdf = "demo_sample.pdf"
    if not os.path.exists(sample_pdf):
        sample_pdf = demo_create_sample_pdf()
    print(f"  示例PDF: {os.path.abspath(sample_pdf)}")

    # 完成时清理
    try:
        # 2. PDF类型检测
        print("\n[步骤2] PDF类型检测...")
        detector = PDFTypeDetector(text_coverage_threshold=0.3)
        pdf_type = detector.detect(sample_pdf)
        print(f"  检测结果: {pdf_type.value}")

        # 3. OCR引擎对比表
        print("\n[步骤3] OCR引擎对比表 (经验数据)...")
        print(OCREngineComparison.accuracy_table())

        # 4. 表格提取
        print("\n[步骤4] 表格提取...")
        extractor = TableExtractor()
        tables = extractor.extract_all(sample_pdf)
        if tables:
            print(f"  共提取 {len(tables)} 个表格")
            for t in tables[:3]:
                print(f"    页码{t['page']}, 方式={t.get('flavor','?')}, "
                      f"行数={len(t.get('data',[]))}")
        else:
            print("  (示例PDF未包含表格，这是正常的)")

        # 5. 多栏布局
        print("\n[步骤5] 多栏布局检测...")
        handler = MultiColumnLayoutHandler()
        layout_results = handler.process_pdf(sample_pdf)
        for lr in layout_results:
            print(f"  页码{lr['page']}: {lr['num_columns']}栏")

        print("\n" + "=" * 70)
        print("A1-01 演示完成！")
        print("=" * 70)

    finally:
        # 清理临时文件
        if os.path.exists(sample_pdf) and sample_pdf == "demo_sample.pdf":
            os.remove(sample_pdf)
            print("\n[清理] 已删除示例PDF")


if __name__ == "__main__":
    main()
