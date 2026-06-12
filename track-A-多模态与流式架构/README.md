# Track A: 多模态与流式架构 (Multimodal & Streaming Architecture)

## 课程概述

本轨道聚焦于 RAG 系统的两个高级维度：**多模态文档理解**（文本+图像+音频+视频的混合检索）与**流式架构**（从后端到前端的全链路流式传输）。

**先修要求**：已完成 Phase 01-02（基础检索与向量化）

**同步阶段**：
- A1 多模态文档解析 → 对应 Phase 03 文档解析引擎
- A2 跨模态检索 → 对应 Phase 04 多模态检索
- A3 流式架构全栈 → 对应 Phase 05 生产化部署

## 学习目标

完成本轨道后，你将能够：

1. **A1 多模态文档解析**
   - 区分扫描件 PDF 与数字 PDF，选择合适的 OCR 引擎
   - 处理双层 PDF（图像层 + 文本层叠加）
   - 从 PDF 和 HTML 中提取表格数据（Camelot Lattice/Stream + pdfplumber 回退）
   - 提取文档内嵌图片并生成语义描述
   - 对音视频内容进行转录、说话人分离和时间戳对齐
   - 实现视频帧级别的多模态 RAG 检索

2. **A2 跨模态检索**
   - 使用 CLIP 模型实现文本↔图片的跨模态检索
   - 构建文本+图片+表格三路混合检索系统（RRF 融合）
   - 利用 GPT-4o/Gemini Vision 在 RAG 上下文中注入图片
   - 实现跨语言多模态检索（中文查询→英文图片检索）

3. **A3 流式架构全栈**
   - 对比 SSE 与 WebSocket 的适用场景
   - 实现 FastAPI StreamingResponse + 异步生成器
   - 掌握 LangChain LCEL 的流式模式（.astream / .astream_events）
   - 构建 React 前端流式消费组件（EventSource）
   - 实现流式安全扫描与延迟监控

## 目录结构

```
track-A-多模态与流式架构/
├── README.md                    # 本文件
├── a1-多模态文档解析/
│   ├── pitfalls/                # 常见坑点与解决方案
│   ├── a1-01-pdf深度解析.py     # PDF深度解析
│   ├── a1-02-图片提取与描述.py  # 图片提取与描述
│   ├── a1-03-音视频转录RAG.py   # 音视频转录RAG
│   └── a1-04-视频帧级RAG.py     # 视频帧级RAG
├── a2-跨模态检索/
│   ├── pitfalls/
│   ├── a2-01-clip图片检索.py    # CLIP图片检索
│   ├── a2-02-多模态混合检索.py  # 多模态混合检索
│   ├── a2-03-多模态LLM生成.py   # 多模态LLM生成
│   └── a2-04-跨语言多模态.py    # 跨语言多模态
└── a3-流式架构全栈/
    ├── pitfalls/
    ├── a3-01-sse-vs-websocket.md # SSE vs WebSocket对比
    ├── a3-02-fastapi流式端点.py  # FastAPI流式端点
    ├── a3-03-lcel流式模式.py     # LCEL流式模式
    ├── a3-04-前端流式消费.jsx    # React前端流式消费
    └── a3-05-流式安全与监控.py   # 流式安全与监控
```

## 技术栈

| 领域 | 核心技术 |
|------|---------|
| OCR | PaddleOCR, Tesseract, Surya |
| 图像理解 | CLIP, BLIP, GPT-4o Vision |
| 语音处理 | Whisper, pyannote-audio |
| 视频处理 | OpenCV, ffmpeg, scene-detect |
| 向量检索 | FAISS, ChromaDB, CLIP embeddings |
| 流式后端 | FastAPI, SSE, asyncio |
| 流式前端 | React, EventSource, ReactMarkdown |

## 快速开始

```bash
# 安装Track A依赖
pip install paddleocr paddlepaddle pytesseract camelot-py pdfplumber
pip install openai-whisper pyannote.audio opencv-python
pip install transformers torch torchvision faiss-cpu chromadb
pip install fastapi uvicorn sse-starlette sentence-transformers
```
