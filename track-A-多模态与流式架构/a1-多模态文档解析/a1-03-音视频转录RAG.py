#!/usr/bin/env python3
"""
A1-03: 音视频转录RAG (Audio/Video Transcription RAG)
=====================================================
学习目标:
  1. Whisper ASR 语音转录管道（本地API双模式）
  2. 说话人分离（Speaker Diarization via pyannote-audio）
  3. 按说话人轮次+时间戳进行chunk分割
  4. 时间戳对齐
  5. 带时间戳引用的内容检索
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass, field
from datetime import timedelta

# ============================================================
# Section 1: Whisper ASR 语音转录
# ============================================================

@dataclass
class TranscriptionSegment:
    """转录片段"""
    start: float          # 开始时间（秒）
    end: float            # 结束时间（秒）
    text: str             # 转录文本
    speaker: str = "UNKNOWN"  # 说话人标签
    confidence: float = 0.0   # 置信度


class WhisperTranscriber:
    """
    Whisper语音转录器
    支持两种模式:
    - local:  本地openai-whisper（离线、免费）
    - api:    OpenAI Whisper API（更快、更准）
    """

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        参数:
            model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
                精度递增，速度递减。base是中文场景的性价比之选
            device: 'cpu' 或 'cuda'
        """
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load_model(self):
        """懒加载Whisper模型"""
        if self._model is None:
            import whisper
            print(f"  [加载] Whisper {self.model_size} 模型...")
            self._model = whisper.load_model(self.model_size, device=self.device)

    def transcribe_local(self, audio_path: str,
                         language: str = "zh") -> List[TranscriptionSegment]:
        """
        本地Whisper转录
        返回带时间戳的转录片段列表
        """
        self._load_model()
        print(f"  [转录] 开始处理: {audio_path}")

        result = self._model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            verbose=False,
            word_timestamps=False,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("confidence", seg.get("avg_logprob", 0.0)),
            ))

        total_duration = segments[-1].end if segments else 0
        print(f"  [转录] 完成: {len(segments)} 个片段, "
              f"{total_duration:.1f}秒")
        return segments

    @staticmethod
    def transcribe_api(audio_path: str,
                       language: str = "zh",
                       api_key: str = None) -> List[TranscriptionSegment]:
        """
        OpenAI Whisper API转录
        需要OPENAI_API_KEY环境变量
        """
        import openai

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置OPENAI_API_KEY环境变量")

        client = openai.OpenAI(api_key=api_key)

        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                language=language,
                timestamp_granularities=["segment"],
            )

        segments = []
        for seg in transcript.segments:
            segments.append(TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=getattr(seg, "confidence", 0.95),
            ))

        print(f"  [API转录] 完成: {len(segments)} 个片段")
        return segments


# ============================================================
# Section 2: 说话人分离 (Speaker Diarization)
# ============================================================

class SpeakerDiarizer:
    """
    说话人分离器
    使用pyannote-audio模型识别不同说话人
    需要HuggingFace token（接受pyannote模型许可）
    """

    def __init__(self, hf_token: str = None,
                 device: str = "cpu"):
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        """懒加载pyannote pipeline"""
        if self._pipeline is None:
            from pyannote.audio import Pipeline
            print("  [加载] pyannote/speaker-diarization-3.1 ...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            if self.device == "cuda":
                import torch
                self._pipeline.to(torch.device("cuda"))

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        执行说话人分离
        返回: [{"start": float, "end": float, "speaker": str}, ...]
        """
        self._load_pipeline()
        print(f"  [分离] 执行说话人分离...")
        diarization = self._pipeline(audio_path)

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        unique_speakers = set(t["speaker"] for t in turns)
        print(f"  [分离] 检测到 {len(unique_speakers)} 个说话人, "
              f"{len(turns)} 个轮次")
        return turns

    @staticmethod
    def mock_diarize(duration_seconds: float) -> List[Dict]:
        """
        无pyannote时的模拟分离（用于演示）
        将音频均匀分配给两个虚假说话人
        """
        turns = []
        segment_duration = 30.0  # 每30秒切换一次说话人
        speakers = ["SPEAKER_A", "SPEAKER_B"]
        current_speaker = 0
        t = 0.0
        while t < duration_seconds:
            end = min(t + segment_duration, duration_seconds)
            turns.append({
                "start": t,
                "end": end,
                "speaker": speakers[current_speaker],
            })
            current_speaker = (current_speaker + 1) % 2
            t = end
        return turns


# ============================================================
# Section 3: 按说话人轮次+时间戳的Chunk分割
# ============================================================

class TranscriptChunker:
    """
    转录分块器
    策略:
    1. 以说话人轮次为主要边界（不切割同一轮次）
    2. 超长轮次按时间窗口分割
    3. 相邻短轮次合并
    """

    def __init__(self, max_chunk_duration: float = 120.0,
                 merge_threshold: float = 15.0,
                 max_chunk_chars: int = 2000):
        """
        参数:
            max_chunk_duration: 单个chunk最大时长（秒）
            merge_threshold: 相邻片段间隔小于此值时合并（秒）
            max_chunk_chars: 单个chunk最大字符数
        """
        self.max_duration = max_chunk_duration
        self.merge_threshold = merge_threshold
        self.max_chars = max_chunk_chars

    def chunk_by_speaker_turns(
        self, segments: List[TranscriptionSegment],
        diarization_turns: List[Dict]
    ) -> List[Dict]:
        """
        结合ASR片段和说话人分离结果进行分块
        返回: 带有说话人、时间戳、文本的chunk列表
        """
        # 第一步：将ASR片段分配给说话人轮次
        for seg in segments:
            seg_center = (seg.start + seg.end) / 2
            for turn in diarization_turns:
                if turn["start"] <= seg_center < turn["end"]:
                    seg.speaker = turn["speaker"]
                    break

        # 第二步：按说话人轮次分组
        chunks = []
        current_chunk = {
            "speaker": segments[0].speaker if segments else "UNKNOWN",
            "start": 0.0,
            "end": 0.0,
            "texts": [],
            "segments": [],
        }

        for seg in segments:
            # 切换说话人 or 超长
            speaker_changed = seg.speaker != current_chunk["speaker"]
            duration_exceeded = (seg.end - current_chunk["start"]) > self.max_duration
            chars_exceeded = len(" ".join(current_chunk["texts"])) > self.max_chars

            if current_chunk["texts"] and (speaker_changed or duration_exceeded or chars_exceeded):
                # 保存当前chunk
                current_chunk["text"] = " ".join(current_chunk["texts"])
                chunks.append(current_chunk)
                # 开始新chunk
                current_chunk = {
                    "speaker": seg.speaker,
                    "start": seg.start,
                    "end": seg.end,
                    "texts": [seg.text],
                    "segments": [seg],
                }
            else:
                if not current_chunk["texts"]:
                    current_chunk["start"] = seg.start
                current_chunk["end"] = seg.end
                current_chunk["texts"].append(seg.text)
                current_chunk["segments"].append(seg)

        # 保存最后一个chunk
        if current_chunk["texts"]:
            current_chunk["text"] = " ".join(current_chunk["texts"])
            chunks.append(current_chunk)

        # 第三步：合并相邻短chunk
        chunks = self._merge_short_chunks(chunks)
        print(f"  [分块] {len(segments)} 片段 → {len(chunks)} chunks")
        return chunks

    def _merge_short_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """合并相邻的短chunk（同一说话人，间隔<阈值）"""
        if len(chunks) <= 1:
            return chunks

        merged = []
        current = chunks[0].copy()

        for next_chunk in chunks[1:]:
            gap = next_chunk["start"] - current["end"]
            same_speaker = current["speaker"] == next_chunk["speaker"]
            total_duration = next_chunk["end"] - current["start"]

            if same_speaker and gap < self.merge_threshold and total_duration < self.max_duration:
                # 合并
                current["end"] = next_chunk["end"]
                current["texts"].extend(next_chunk["texts"])
                current["segments"].extend(next_chunk["segments"])
                current["text"] = " ".join(current["texts"])
            else:
                merged.append(current)
                current = next_chunk.copy()

        merged.append(current)
        return merged

    def simple_chunk(self, segments: List[TranscriptionSegment]) -> List[Dict]:
        """简易分块（不依赖diarization）"""
        chunks = []
        buffer_text = []
        buffer_start = 0.0
        current_duration = 0.0

        for seg in segments:
            if not buffer_text:
                buffer_start = seg.start

            buffer_text.append(seg.text)
            current_duration = seg.end - buffer_start

            if current_duration >= self.max_duration or len(" ".join(buffer_text)) > self.max_chars:
                chunks.append({
                    "start": buffer_start,
                    "end": seg.end,
                    "text": " ".join(buffer_text),
                    "speaker": "UNKNOWN",
                })
                buffer_text = []
                current_duration = 0.0

        if buffer_text:
            chunks.append({
                "start": buffer_start,
                "end": segments[-1].end,
                "text": " ".join(buffer_text),
                "speaker": "UNKNOWN",
            })

        return chunks


# ============================================================
# Section 4: 带时间戳引用的检索
# ============================================================

class TimestampedRetriever:
    """
    带时间戳引用的检索器
    功能：
    1. 对转录chunk建立向量索引
    2. 检索时返回时间戳信息
    3. 生成可跳转的时间戳链接
    """

    def __init__(self, chunks: List[Dict], embedding_model: str = None):
        self.chunks = chunks
        self._embeddings = None
        self._model = None

        # 使用轻量级句子嵌入
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                embedding_model or "paraphrase-multilingual-MiniLM-L12-v2"
            )
        except ImportError:
            self._model = None

    def build_index(self):
        """构建向量索引"""
        if self._model is None:
            print("  [警告] sentence_transformers未安装，使用关键词匹配")
            return

        texts = [c["text"] for c in self.chunks]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)
        print(f"  [索引] 为 {len(texts)} 个chunk构建了向量索引")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关chunk并返回时间戳引用
        """
        if self._embeddings is not None and self._model is not None:
            query_emb = self._model.encode([query], normalize_embeddings=True)[0]
            import numpy as np
            similarities = np.dot(self._embeddings, query_emb)
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            # 关键词匹配回退
            scores = []
            for i, chunk in enumerate(self.chunks):
                query_terms = set(query.lower().split())
                text_terms = set(chunk["text"].lower().split())
                overlap = len(query_terms & text_terms)
                scores.append((i, overlap))
            scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [s[0] for s in scores[:top_k]]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": idx,
                "start_time": chunk["start"],
                "end_time": chunk["end"],
                "start_formatted": str(timedelta(seconds=int(chunk["start"]))),
                "end_formatted": str(timedelta(seconds=int(chunk["end"]))),
                "speaker": chunk.get("speaker", "UNKNOWN"),
                "text": chunk["text"][:500],
                "timestamp_link": f"t={int(chunk['start'])}",  # 可用于视频跳转
            })

        return results


# ============================================================
# Section 5: 音频预处理工具
# ============================================================

class AudioPreprocessor:
    """音频预处理：格式转换、重采样、分割"""

    @staticmethod
    def convert_to_wav(input_path: str, target_sr: int = 16000) -> str:
        """使用ffmpeg转换音频格式为Whisper兼容的WAV"""
        output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
        cmd = [
            "ffmpeg", "-i", input_path,
            "-ar", str(target_sr),
            "-ac", "1",           # 单声道
            "-sample_fmt", "s16", # 16-bit PCM
            "-y",                 # 覆盖输出
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"  [错误] ffmpeg转换失败: {e.stderr.decode()}")
            raise
        except FileNotFoundError:
            print("  [警告] ffmpeg未找到，尝试直接使用原始文件")
            return input_path

    @staticmethod
    def extract_audio_from_video(video_path: str,
                                 target_sr: int = 16000) -> str:
        """从视频中提取音频"""
        output_path = video_path.rsplit(".", 1)[0] + "_audio.wav"
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",                # 无视频流
            "-ar", str(target_sr),
            "-ac", "1",
            "-sample_fmt", "s16",
            "-y",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path


# ============================================================
# Section 6: 主流程
# ============================================================

def create_silent_audio(output_path: str, duration: float = 10.0):
    """使用ffmpeg生成静音音频用于演示"""
    cmd = [
        "ffmpeg", "-f", "lavfi",
        "-i", f"anullsrc=r=16000:cl=mono",
        "-t", str(duration),
        "-y", output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception:
        return False


def main():
    """主函数：演示音视频转录RAG全流程"""
    print("=" * 70)
    print("A1-03: 音视频转录RAG — 完整演示")
    print("=" * 70)

    # 1. 创建演示音频
    print("\n[步骤1] 创建演示音频...")
    demo_audio = "demo_silent.wav"
    if create_silent_audio(demo_audio, duration=5.0):
        print(f"  创建 {demo_audio} (5秒静音-仅演示架构)")
    else:
        print("  [跳过] ffmpeg不可用，使用模拟数据演示")
        demo_audio = None

    # 2. Whisper转录（模拟数据演示，因为静音没有语音）
    print("\n[步骤2] 转录演示（使用模拟数据）...")
    mock_segments = [
        TranscriptionSegment(0.0, 5.2, "今天我们讨论RAG系统的多模态文档解析。"),
        TranscriptionSegment(5.5, 12.0, "多模态解析需要处理PDF图片表格和音视频。"),
        TranscriptionSegment(12.3, 18.7, "Whisper模型可以高效地进行语音转录。"),
        TranscriptionSegment(19.0, 25.5, "同时pyannote能够分离不同说话人的语音。"),
        TranscriptionSegment(26.0, 32.0, "最终我们将文本和图片一起索引到向量数据库。"),
        TranscriptionSegment(32.5, 40.0, "检索时系统返回最相关的内容和时间戳引用。"),
    ]
    print(f"  模拟 {len(mock_segments)} 个ASR片段")

    # 3. 说话人分离（模拟）
    print("\n[步骤3] 说话人分离（模拟）...")
    mock_diarization = SpeakerDiarizer.mock_diarize(40.0)
    for turn in mock_diarization:
        print(f"  {turn['speaker']}: {turn['start']:.1f}s - {turn['end']:.1f}s")

    # 4. 按说话人+时间戳分块
    print("\n[步骤4] 按说话人轮次分块...")
    chunker = TranscriptChunker(max_chunk_duration=30.0, max_chunk_chars=1500)
    chunks = chunker.chunk_by_speaker_turns(mock_segments, mock_diarization)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: [{chunk['start']:.1f}s-{chunk['end']:.1f}s] "
              f"{chunk['speaker']}: {chunk['text'][:60]}...")

    # 5. 带时间戳检索
    print("\n[步骤5] 带时间戳检索...")
    retriever = TimestampedRetriever(chunks)

    # 手动设置简单嵌入
    try:
        retriever.build_index()
    except Exception:
        pass

    results = retriever.search("语音转录 Whisper", top_k=3)
    for r in results:
        print(f"  [{r['start_formatted']}] {r['speaker']}: {r['text'][:80]}...")
        print(f"    跳转链接: {r['timestamp_link']}")

    # 清理
    if demo_audio and os.path.exists(demo_audio):
        os.remove(demo_audio)

    print("\n" + "=" * 70)
    print("A1-03 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
