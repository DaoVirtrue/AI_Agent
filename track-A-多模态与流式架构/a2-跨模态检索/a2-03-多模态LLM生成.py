#!/usr/bin/env python3
"""
A2-03: 多模态LLM生成 (Multimodal LLM Generation)
===================================================
学习目标:
  1. GPT-4o / Gemini Vision API调用
  2. 在RAG提示上下文中注入图片
  3. 文本+图片联合答案生成
  4. 来源引用（含图片）
"""

import os
import sys
import json
import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Section 1: 多模态提示构建器
# ============================================================

@dataclass
class MultimodalContextItem:
    """多模态上下文项"""
    content_type: str  # "text" | "image_url" | "image_base64"
    content: str       # 文本内容 或 图片URL 或 base64
    source_id: str     # 来源文档ID
    source_type: str   # "pdf" | "docx" | "webpage" | "database"
    metadata: Dict = field(default_factory=dict)


class MultimodalPromptBuilder:
    """多模态提示构建器"""

    @staticmethod
    def image_to_base64(image_path: str,
                        max_size: int = 2048) -> str:
        """
        将图片转为base64 URL格式（兼容OpenAI Vision API）
        OpenAI要求: data:image/jpeg;base64,<base64>
        """
        from PIL import Image
        img = Image.open(image_path)

        # 缩放（保留长宽比）
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        img_format = img.format or "JPEG"
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format=img_format)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime = f"image/{img_format.lower()}"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def build_openai_messages(
        system_prompt: str,
        user_query: str,
        contexts: List[MultimodalContextItem],
        include_images: bool = True,
    ) -> List[Dict]:
        """
        构建OpenAI Vision API兼容的消息格式

        输出格式:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "..."}},
            ]},
        ]
        """
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        user_content = []

        # 文本上下文
        text_contexts = [c for c in contexts if c.content_type == "text"]
        if text_contexts:
            context_text = "\n\n---\n".join(
                f"[来源: {c.source_id} ({c.source_type})]\n{c.content}"
                for c in text_contexts
            )
            user_content.append({
                "type": "text",
                "text": f"参考以下上下文回答问题:\n\n{context_text}\n\n---\n用户问题: {user_query}\n\n请基于上下文回答，并标注信息来源。如果上下文中包含图片，也请描述图片中的相关内容。",
            })
        else:
            user_content.append({
                "type": "text",
                "text": f"用户问题: {user_query}",
            })

        # 图片上下文
        if include_images:
            image_contexts = [c for c in contexts
                            if c.content_type in ("image_url", "image_base64")]
            for img_ctx in image_contexts[:3]:  # 最多3张图片
                url = img_ctx.content
                if img_ctx.content_type == "image_base64":
                    user_content.insert(0, {  # 图片放在文本前面
                        "type": "image_url",
                        "image_url": {
                            "url": img_ctx.content,
                            "detail": "auto",
                        },
                    })
                else:
                    user_content.insert(0, {
                        "type": "image_url",
                        "image_url": {
                            "url": img_ctx.content,
                            "detail": "auto",
                        },
                    })

        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def build_gemini_contents(
        system_prompt: str,
        user_query: str,
        contexts: List[MultimodalContextItem],
    ) -> List[Dict]:
        """
        构建Gemini API兼容的内容格式

        Gemini格式支持文本和图片混合排列
        """
        contents = []
        parts = []

        # 系统提示作为第一个user turn
        parts.append({"text": f"{system_prompt}\n\n用户问题: {user_query}"})

        # 添加图片
        image_contexts = [c for c in contexts
                        if c.content_type == "image_base64"]
        for img_ctx in image_contexts[:3]:
            b64_data = img_ctx.content
            if b64_data.startswith("data:"):
                b64_data = b64_data.split(",", 1)[1]
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_data,
                }
            })

        # 添加文本上下文
        text_contexts = [c for c in contexts if c.content_type == "text"]
        if text_contexts:
            context_text = "\n\n---\n".join(
                f"[来源: {c.source_id}]\n{c.content}"
                for c in text_contexts
            )
            parts.append({
                "text": f"参考上下文:\n{context_text}"
            })

        contents.append({"role": "user", "parts": parts})
        return contents


# ============================================================
# Section 2: GPT-4o Vision API 调用
# ============================================================

class GPT4oVisionGenerator:
    """GPT-4o Vision API 调用器"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, messages: List[Dict],
                 max_tokens: int = 2048,
                 temperature: float = 0.3) -> Dict:
        """
        调用GPT-4o Vision生成答案
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "text": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "finish_reason": response.choices[0].finish_reason,
        }

    @staticmethod
    def mock_generate(query: str, contexts: List[MultimodalContextItem]) -> Dict:
        """
        模拟生成（无API Key时演示架构）
        生成带有来源引用的示例答案
        """
        context_summary = "\n".join(
            f"  - {c.source_id}: {c.content[:80]}..."
            for c in contexts if c.content_type == "text"
        )
        image_count = sum(1 for c in contexts if "image" in c.content_type)
        return {
            "text": f"""基于{len(contexts)}个上下文（含{image_count}张图片），回答如下：

# 回答

根据提供的上下文信息，相关的结论如下所示。上下文中的信息被综合处理以生成完整答案。

来源引用:
{context_summary}

注：此回答基于提供的上下文内容生成。如需更详细信息，请查阅原始文档。""",
            "model": "gpt-4o (mock)",
            "usage": {"total_tokens": len(query) + 500},
            "finish_reason": "stop",
        }


# ============================================================
# Section 3: Gemini Vision API 调用
# ============================================================

class GeminiVisionGenerator:
    """Gemini Vision API 调用器"""

    def __init__(self, api_key: str = None,
                 model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def generate(self, contents: List[Dict],
                 max_output_tokens: int = 2048) -> Dict:
        """调用Gemini Vision生成答案"""
        model = self._get_client()
        response = model.generate_content(
            contents,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": 0.3,
            },
        )
        return {
            "text": response.text,
            "model": self.model,
            "finish_reason": str(response.candidates[0].finish_reason) if response.candidates else "unknown",
        }


# ============================================================
# Section 4: 来源引用（含图片）
# ============================================================

class SourceAttributor:
    """来源归因器"""

    @staticmethod
    def extract_citations(text: str) -> List[Dict]:
        """
        从生成的文本中提取引用标记
        支持格式: [来源: doc_id], [1], (Source: xxx)
        """
        import re
        citations = []

        # 匹配 [来源: xxx]
        pattern1 = r'\[来源[：:]\s*([^\]]+)\]'
        for match in re.finditer(pattern1, text):
            citations.append({
                "type": "text_source",
                "source_ref": match.group(1).strip(),
                "position": match.start(),
            })

        # 匹配 [图N]
        pattern2 = r'\[图(\d+)\]'
        for match in re.finditer(pattern2, text):
            citations.append({
                "type": "image_reference",
                "image_number": int(match.group(1)),
                "position": match.start(),
            })

        return citations

    @staticmethod
    def format_answer_with_sources(
        answer_text: str,
        sources: List[MultimodalContextItem],
    ) -> Dict:
        """
        格式化答案及来源（含图片）
        返回结构化JSON，前端可用于渲染
        """
        text_sources = [s for s in sources if s.content_type == "text"]
        image_sources = [s for s in sources if "image" in s.content_type]

        return {
            "answer": answer_text,
            "sources": {
                "text": [
                    {
                        "source_id": s.source_id,
                        "source_type": s.source_type,
                        "preview": s.content[:200] + "...",
                        "metadata": s.metadata,
                    }
                    for s in text_sources
                ],
                "images": [
                    {
                        "source_id": s.source_id,
                        "url": s.content if s.content_type == "image_url" else "[base64]",
                        "source_type": s.source_type,
                    }
                    for s in image_sources
                ],
            },
            "total_sources": len(sources),
            "has_images": len(image_sources) > 0,
        }


# ============================================================
# Section 5: 完整RAG多模态生成管道
# ============================================================

class MultimodalRAGGenerator:
    """完整的多模态RAG生成管道"""

    def __init__(self, provider: str = "openai",
                 api_key: str = None):
        """
        参数:
            provider: "openai" (GPT-4o) | "google" (Gemini) | "mock"
        """
        self.provider = provider

        if provider == "openai":
            self.generator = GPT4oVisionGenerator(api_key=api_key)
        elif provider == "google":
            self.generator = GeminiVisionGenerator(api_key=api_key)
        else:
            self.generator = None

        self.prompt_builder = MultimodalPromptBuilder()
        self.attributor = SourceAttributor()

    def generate(self, query: str,
                 contexts: List[MultimodalContextItem],
                 system_prompt: str = None) -> Dict:
        """
        完整流程:
        1. 构建多模态提示
        2. 调用Vision LLM
        3. 解析引用
        4. 格式化输出
        """
        if system_prompt is None:
            system_prompt = """你是一个RAG（检索增强生成）助手。请基于提供的上下文（包括文本和图片）回答问题。

重要规则:
1. 回答中引用的每条信息都需标注来源，格式：[来源: 文档ID]
2. 如果上下文中包含图片信息，请描述图片中的相关内容
3. 如果上下文不足以回答问题，请明确说明"根据现有资料无法确定"
4. 使用中文回答（除非问题本身就是其他语言）"""

        # Mock模式
        if self.provider == "mock":
            result = GPT4oVisionGenerator.mock_generate(query, contexts)
        else:
            # 构建消息
            messages = self.prompt_builder.build_openai_messages(
                system_prompt, query, contexts
            )
            result = self.generator.generate(messages)

        # 解析引用
        citations = self.attributor.extract_citations(result["text"])

        # 格式化
        formatted = self.attributor.format_answer_with_sources(
            result["text"], contexts
        )

        return {
            **result,
            "citations": citations,
            "formatted": formatted,
        }


# ============================================================
# Section 6: 主流程
# ============================================================

def create_demo_image(output_path: str):
    """创建演示用的图表图片"""
    img = Image.new("RGB", (400, 250), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # 模拟柱状图
    draw.rectangle([50, 150, 90, 220], fill="#2196F3")
    draw.text((50, 225), "Q1", fill="black")
    draw.rectangle([110, 120, 150, 220], fill="#4CAF50")
    draw.text((110, 225), "Q2", fill="black")
    draw.rectangle([170, 80, 210, 220], fill="#FF9800")
    draw.text((170, 225), "Q3", fill="black")
    draw.rectangle([230, 100, 270, 220], fill="#9C27B0")
    draw.text((230, 225), "Q4", fill="black")
    draw.text((100, 20), "Quarterly Revenue Growth", fill="black")
    draw.text((100, 45), "2024 Fiscal Year", fill="gray")
    img.save(output_path)
    return output_path


def main():
    """主函数：演示多模态LLM生成全流程"""
    print("=" * 70)
    print("A2-03: 多模态LLM生成 — 完整演示")
    print("=" * 70)

    output_dir = "demo_multimodal_gen"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建演示图片
    print("\n[步骤1] 创建演示图片...")
    demo_img = os.path.join(output_dir, "chart_demo.png")
    create_demo_image(demo_img)
    print(f"  创建: {demo_img}")

    # 2. 构建多模态上下文
    print("\n[步骤2] 构建多模态上下文...")
    contexts = [
        MultimodalContextItem(
            content_type="text",
            content="2024年Q3营收达到520万元，同比增长35%。主要增长动力来自AI产品线的商业化落地。",
            source_id="report_2024q3",
            source_type="pdf",
        ),
        MultimodalContextItem(
            content_type="text",
            content="季度营收数据：Q1 280万，Q2 350万，Q3 520万，Q4预计650万。全年预期1800万。",
            source_id="finance_data",
            source_type="database",
        ),
        MultimodalContextItem(
            content_type="image_base64",
            content=MultimodalPromptBuilder.image_to_base64(demo_img),
            source_id="chart_revenue",
            source_type="pdf",
            metadata={"page": 3, "figure": "Fig 1. Quarterly Revenue"},
        ),
    ]

    # 3. 构建提示（展示消息格式）
    print("\n[步骤3] 构建多模态提示（OpenAI格式）...")
    messages = MultimodalPromptBuilder.build_openai_messages(
        system_prompt="你是财务分析助手。回答问题时请引用数据来源。",
        user_query="2024年Q3的营收表现如何？请结合图表分析增长趋势。",
        contexts=contexts,
        include_images=True,
    )
    # 展示消息结构（不展示base64数据）
    print(f"  系统提示: {messages[0]['content'][:60]}...")
    user_content = messages[1]['content']
    for item in user_content:
        if item['type'] == 'text':
            print(f"  用户文本: {item['text'][:80]}...")
        else:
            print(f"  用户图片: [base64数据, 已省略]")

    # 4. 模拟生成
    print("\n[步骤4] 多模态RAG生成（Mock模式）...")
    rag_generator = MultimodalRAGGenerator(provider="mock")
    result = rag_generator.generate(
        query="2024年Q3的营收表现如何？",
        contexts=contexts,
    )

    print(f"  模型: {result.get('model', 'mock')}")
    print(f"  答案预览:\n{result['text'][:400]}...")

    # 5. 来源引用
    print("\n[步骤5] 来源引用解析...")
    citations = result.get('citations', [])
    for c in citations:
        print(f"  引用: {c['type']} → {c.get('source_ref', c.get('image_number', '?'))}")

    formatted = result.get('formatted', {})
    sources = formatted.get('sources', {})
    print(f"  文本来源: {len(sources.get('text', []))} 项")
    print(f"  图片来源: {len(sources.get('images', []))} 项")
    print(f"  总来源数: {formatted.get('total_sources', 0)}")

    print("\n" + "=" * 70)
    print("A2-03 演示完成！")
    print("=" * 70)
    print("\n[提示] 要使用真实的GPT-4o Vision API，请设置:")
    print("  export OPENAI_API_KEY='sk-...'")
    print("  并将 provider='openai'")


if __name__ == "__main__":
    main()
