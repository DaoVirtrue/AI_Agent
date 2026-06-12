#!/usr/bin/env python3
"""
A1-04: 视频帧级RAG (Video Frame-Level RAG)
=============================================
学习目标:
  1. 关键帧提取（场景变化检测）
  2. 字幕OCR + 音频转录的双通道信息提取
  3. 多模态索引（CLIP嵌入 + 文本嵌入）
  4. 帧级检索并返回视频时间戳
"""

import os
import sys
import json
import math
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# ============================================================
# Section 1: 关键帧提取 (Scene Change Detection)
# ============================================================

@dataclass
class KeyFrame:
    """关键帧数据结构"""
    frame_idx: int           # 帧序号
    timestamp: float         # 时间戳（秒）
    timestamp_formatted: str # 格式化时间戳 HH:MM:SS
    image_path: str          # 帧图片路径
    scene_score: float       # 场景变化分数
    subtitle_text: str = ""  # OCR字幕文本
    audio_text: str = ""     # 对应时间段的音频转录


class KeyFrameExtractor:
    """
    关键帧提取器
    方法1: 基于帧间差异的场景变化检测（OpenCV）
    方法2: 使用PySceneDetect库（更精确）
    """

    def __init__(self, method: str = "pyscenedetect",
                 threshold: float = 27.0,
                 min_scene_len: int = 15):
        """
        参数:
            method: 'opencv' 或 'pyscenedetect'
            threshold: 场景变化阈值（越低越敏感）
            min_scene_len: 最短场景长度（帧数），防止过度分段
        """
        self.method = method
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def extract_opencv(self, video_path: str,
                       output_dir: str,
                       max_frames: int = 100) -> List[KeyFrame]:
        """
        使用OpenCV进行场景变化检测
        算法: 计算连续帧之间的HSV直方图差异
        """
        import cv2

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"  [视频] {total_frames}帧, {fps:.2f}fps, {duration:.1f}秒")

        keyframes = []
        prev_hist = None
        frame_count = 0
        saved_count = 0

        # 计算采样间隔（避免处理每一帧）
        sample_every = max(1, fps // 2)  # 每秒处理2帧

        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 采样间隔
            if frame_count % sample_every != 0:
                continue

            # 计算HSV直方图
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                # 计算直方图差异（相关性方法）
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                # 差异大于阈值 → 场景变化
                if diff > self.threshold * 100:
                    timestamp = frame_count / fps
                    ts_formatted = self._format_timestamp(timestamp)
                    frame_path = os.path.join(
                        output_dir, f"keyframe_{saved_count:04d}_{ts_formatted.replace(':', '-')}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    keyframes.append(KeyFrame(
                        frame_idx=frame_count,
                        timestamp=timestamp,
                        timestamp_formatted=ts_formatted,
                        image_path=frame_path,
                        scene_score=float(diff),
                    ))
                    saved_count += 1

            prev_hist = hist

        cap.release()
        print(f"  [OpenCV] 提取了 {len(keyframes)} 个关键帧")
        return keyframes

    def extract_pyscenedetect(self, video_path: str,
                              output_dir: str) -> List[KeyFrame]:
        """使用PySceneDetect库进行场景变化检测"""
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
            import cv2

            os.makedirs(output_dir, exist_ok=True)

            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(threshold=self.threshold,
                                min_scene_len=self.min_scene_len)
            )
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()

            fps = video.frame_rate
            keyframes = []

            for i, (start, end) in enumerate(scene_list):
                # 取场景中间帧作为关键帧
                mid_frame = start.frame_num + (end.frame_num - start.frame_num) // 2
                video.seek(mid_frame)
                ret, frame = video.read()

                if ret and frame is not None:
                    timestamp = mid_frame / fps
                    ts_formatted = self._format_timestamp(timestamp)
                    frame_path = os.path.join(
                        output_dir, f"keyframe_{i:04d}_{ts_formatted.replace(':', '-')}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    keyframes.append(KeyFrame(
                        frame_idx=mid_frame,
                        timestamp=timestamp,
                        timestamp_formatted=ts_formatted,
                        image_path=frame_path,
                        scene_score=self.threshold,  # PySceneDetect不直接暴露分数
                    ))

            print(f"  [PySceneDetect] 提取了 {len(keyframes)} 个关键帧")
            return keyframes

        except ImportError:
            print("  [回退] PySceneDetect未安装，使用OpenCV方法")
            return self.extract_opencv(video_path, output_dir)

    def extract(self, video_path: str,
                output_dir: str,
                max_frames: int = 100) -> List[KeyFrame]:
        """提取关键帧的统一入口"""
        if self.method == "pyscenedetect":
            return self.extract_pyscenedetect(video_path, output_dir)
        return self.extract_opencv(video_path, output_dir, max_frames)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """将秒转换为 HH:MM:SS 格式"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# Section 2: 字幕OCR + 音频转录双通道
# ============================================================

class DualChannelExtractor:
    """
    双通道信息提取
    通道1: 字幕OCR（从视频帧中提取字幕文字）
    通道2: 音频转录（Whisper ASR）
    """

    @staticmethod
    def extract_subtitle_region(frame_path: str) -> Optional[str]:
        """
        从帧底部提取字幕文字
        假设字幕在画面底部20%区域
        """
        try:
            import cv2
            import pytesseract

            img = cv2.imread(frame_path)
            if img is None:
                return None

            h, w = img.shape[:2]
            # 裁剪底部20%区域
            subtitle_region = img[int(h * 0.8):h, int(w * 0.05):int(w * 0.95)]

            # 预处理: 灰度化 + 二值化
            gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # OCR
            text = pytesseract.image_to_string(
                binary, lang="chi_sim+eng",
                config="--psm 7"  # 单行文本模式
            ).strip()

            return text if len(text) > 2 else ""
        except ImportError:
            return ""

    @staticmethod
    def extract_audio_transcript(audio_path: str,
                                 start_time: float,
                                 end_time: float) -> str:
        """
        提取指定时间段的音频转录
        先裁剪音频片段，再进行转录
        """
        try:
            import whisper
            # 裁剪音频
            cropped_path = audio_path.replace(".wav", "_cropped.wav")
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c", "copy",
                "-y", cropped_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            model = whisper.load_model("base")
            result = model.transcribe(cropped_path, language="zh", verbose=False)
            os.remove(cropped_path)
            return result["text"].strip()
        except Exception:
            return ""

    @classmethod
    def process_keyframes(cls, keyframes: List[KeyFrame],
                          audio_path: str = None) -> List[KeyFrame]:
        """
        为每个关键帧补充字幕OCR和音频转录
        """
        for kf in keyframes:
            # 通道1: 字幕OCR
            subtitle = cls.extract_subtitle_region(kf.image_path)
            if subtitle:
                kf.subtitle_text = subtitle

            # 通道2: 音频转录（上下文窗口 ±5秒）
            if audio_path and os.path.exists(audio_path):
                audio_start = max(0, kf.timestamp - 5)
                audio_end = kf.timestamp + 5
                audio_text = cls.extract_audio_transcript(
                    audio_path, audio_start, audio_end
                )
                if audio_text:
                    kf.audio_text = audio_text

            # 如果两个通道都有，合并信息
            if kf.subtitle_text and kf.audio_text:
                # 字幕通常更准确，优先使用
                kf.audio_text = f"[字幕] {kf.subtitle_text} | [语音] {kf.audio_text}"

        return keyframes


# ============================================================
# Section 3: 多模态索引 (CLIP + Text Embedding)
# ============================================================

class MultiModalIndexer:
    """
    多模态索引器
    为每个关键帧建立两种向量:
    - CLIP图像嵌入（视觉语义）
    - 文本嵌入（字幕+语音转录）
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._has_clip = True
        except Exception:
            print("  [警告] CLIP模型加载失败，仅使用文本嵌入")
            self.clip_model = None
            self.clip_processor = None
            self._has_clip = False

        try:
            from sentence_transformers import SentenceTransformer
            self.text_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._has_text = True
        except Exception:
            print("  [警告] text-embedding模型加载失败")
            self.text_model = None
            self._has_text = False

    def index_keyframes(self, keyframes: List[KeyFrame]) -> Dict:
        """
        构建多模态索引
        返回: {
            "keyframes": [...],
            "image_embeddings": np.ndarray,
            "text_embeddings": np.ndarray,
        }
        """
        from PIL import Image

        image_embeddings = []
        text_embeddings = []

        for kf in keyframes:
            # CLIP图像嵌入
            if self._has_clip:
                try:
                    img = Image.open(kf.image_path).convert("RGB")
                    inputs = self.clip_processor(
                        images=img, return_tensors="pt"
                    ).to(self.device)
                    img_emb = self.clip_model.get_image_features(**inputs)
                    img_emb = img_emb.cpu().detach().numpy()[0]
                    img_emb = img_emb / np.linalg.norm(img_emb)
                except Exception:
                    img_emb = np.zeros(512)
            else:
                img_emb = np.zeros(512)

            image_embeddings.append(img_emb)

            # 文本嵌入
            text_content = (kf.subtitle_text or "") + " " + (kf.audio_text or "")
            text_content = text_content.strip()

            if self._has_text and text_content:
                text_emb = self.text_model.encode(
                    [text_content], normalize_embeddings=True
                )[0]
            else:
                text_emb = np.zeros(384)

            text_embeddings.append(text_emb)

        image_embeddings = np.array(image_embeddings)
        text_embeddings = np.array(text_embeddings)

        print(f"  [索引] 构建了 {len(keyframes)} 帧的多模态向量 "
              f"(图像维度={image_embeddings.shape[1]}, "
              f"文本维度={text_embeddings.shape[1]})")

        return {
            "keyframes": keyframes,
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }


# ============================================================
# Section 4: 帧级检索
# ============================================================

class VideoFrameRetriever:
    """视频帧级检索器"""

    def __init__(self, index: Dict,
                 image_weight: float = 0.5,
                 text_weight: float = 0.5):
        """
        参数:
            image_weight: 图像语义权重
            text_weight: 文本语义权重
        """
        self.index = index
        self.keyframes = index["keyframes"]
        self.image_embeddings = index.get("image_embeddings")
        self.text_embeddings = index.get("text_embeddings")
        self.image_weight = image_weight
        self.text_weight = text_weight

    def search(self, query: str,
               query_image_path: str = None,
               top_k: int = 5) -> List[Dict]:
        """
        多模态检索
        支持: 纯文本查询 / 纯图片查询 / 文本+图片混合查询
        """
        n = len(self.keyframes)
        scores = np.zeros(n)

        # 1. 文本维度
        if self.text_embeddings is not None and query:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                query_emb = model.encode([query], normalize_embeddings=True)[0]
                text_scores = np.dot(self.text_embeddings, query_emb)
                scores += self.text_weight * text_scores
            except Exception:
                # 回退: TF-IDF关键词匹配
                query_terms = set(query.lower().split())
                for i, kf in enumerate(self.keyframes):
                    kf_text = (kf.subtitle_text + " " + kf.audio_text).lower()
                    kf_terms = set(kf_text.split())
                    overlap = len(query_terms & kf_terms)
                    scores[i] += self.text_weight * overlap / max(len(query_terms), 1)

        # 2. 图像维度
        if self.image_embeddings is not None and query_image_path:
            try:
                from transformers import CLIPProcessor, CLIPModel
                from PIL import Image
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                img = Image.open(query_image_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")
                img_emb = model.get_image_features(**inputs).detach().numpy()[0]
                img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-8)
                img_scores = np.dot(self.image_embeddings, img_emb)
                scores += self.image_weight * img_scores
            except Exception as e:
                print(f"  [警告] 图像检索失败: {e}")

        # 获取top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            kf = self.keyframes[idx]
            results.append({
                "rank": len(results) + 1,
                "frame_idx": kf.frame_idx,
                "timestamp": kf.timestamp,
                "timestamp_formatted": kf.timestamp_formatted,
                "image_path": kf.image_path,
                "subtitle": kf.subtitle_text[:200] if kf.subtitle_text else "",
                "audio_text": kf.audio_text[:200] if kf.audio_text else "",
                "score": round(float(scores[idx]), 4),
                "deeplink": f"t={int(kf.timestamp)}",
            })

        return results


# ============================================================
# Section 5: 主流程
# ============================================================

def create_test_video(output_path: str, num_frames: int = 60):
    """使用ffmpeg创建测试视频（彩色帧序列）"""
    try:
        cmd = [
            "ffmpeg", "-f", "lavfi",
            "-i", f"testsrc=duration=3:size=640x480:rate={num_frames/3}",
            "-vf", "drawtext=text='Scene %{n}':fontsize=24:fontcolor=white:x=10:y=10",
            "-pix_fmt", "yuv420p",
            "-y", output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception:
        return False


def main():
    """主函数：演示视频帧级RAG全流程"""
    print("=" * 70)
    print("A1-04: 视频帧级RAG — 完整演示")
    print("=" * 70)

    output_dir = "demo_video_output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建测试视频
    print("\n[步骤1] 创建测试视频...")
    video_path = os.path.join(output_dir, "test_video.mp4")
    if create_test_video(video_path):
        print(f"  创建 {video_path}")
    else:
        print("  [跳过] ffmpeg不可用，使用模拟数据演示")
        video_path = None

    # 2. 关键帧提取
    print("\n[步骤2] 关键帧提取...")
    if video_path and os.path.exists(video_path):
        extractor = KeyFrameExtractor(
            method="opencv", threshold=25.0, min_scene_len=10
        )
        keyframes = extractor.extract(video_path, output_dir, max_frames=10)
    else:
        # 模拟关键帧数据
        keyframes = []
        for i in range(5):
            kf = KeyFrame(
                frame_idx=i * 100,
                timestamp=i * 5.0,
                timestamp_formatted=f"00:00:{i*5:02d}",
                image_path=os.path.join(output_dir, f"mock_frame_{i}.jpg"),
                scene_score=30.0,
                subtitle_text=f"这是第{i+1}个场景的示范字幕文本",
                audio_text=f"第{i+1}段示范音频转录内容",
            )
            keyframes.append(kf)
        print(f"  模拟 {len(keyframes)} 个关键帧")

    # 3. 多模态索引
    print("\n[步骤3] 多模态索引构建...")
    indexer = MultiModalIndexer(device="cpu")
    index_data = indexer.index_keyframes(keyframes)

    # 4. 帧级检索
    print("\n[步骤4] 帧级检索测试...")
    retriever = VideoFrameRetriever(
        index_data,
        image_weight=0.4,
        text_weight=0.6,
    )

    # 纯文本检索
    query = "示范字幕文本 场景"
    print(f"  查询: '{query}'")
    results = retriever.search(query=query, top_k=3)

    for r in results:
        print(f"  [{r['timestamp_formatted']}] 分数={r['score']:.4f}")
        print(f"    字幕: {r['subtitle'][:80] if r['subtitle'] else '(无)'}")
        print(f"    音频: {r['audio_text'][:80] if r['audio_text'] else '(无)'}")
        print(f"    跳转: {r['deeplink']}")

    print("\n" + "=" * 70)
    print("A1-04 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
