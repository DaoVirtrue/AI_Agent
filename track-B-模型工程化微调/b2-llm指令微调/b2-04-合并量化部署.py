#!/usr/bin/env python3
"""
B2-04: 合并量化部署 (Merge, Quantize & Deploy)
===================================================
学习目标:
  1. LoRA权重合并到基座模型
  2. GGUF格式导出 (q4_k_m量化)
  3. Ollama Modelfile创建
  4. Ollama部署与测试
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================
# Section 1: LoRA权重合并
# ============================================================

class LoRAMerger:
    """
    LoRA权重合并器

    原理:
    LoRA在原始权重旁边添加低秩矩阵 A*B
    合并: W_new = W_original + (alpha / rank) * A * B

    合并后的好处:
    - 不再需要加载adapter（减少依赖）
    - 可以直接导出为GGUF等格式
    - 推理速度更快（无额外计算）
    """

    def __init__(self, base_model_path: str,
                 lora_adapter_path: str,
                 output_path: str):
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.output_path = output_path

    def merge(self) -> Dict:
        """
        合并LoRA权重

        使用PEFT库的merge_and_unload方法
        """
        print(f"\n  [LoRA合并] 开始...")
        print(f"  基座模型: {self.base_model_path}")
        print(f"  LoRA适配器: {self.lora_adapter_path}")
        print(f"  输出路径: {self.output_path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # 加载基座模型
            print("  [1/3] 加载基座模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

            # 加载LoRA
            print("  [2/3] 加载LoRA适配器...")
            model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)

            # 合并
            print("  [3/3] 合并权重...")
            merged_model = model.merge_and_unload()

            # 保存
            merged_model.save_pretrained(self.output_path, safe_serialization=True)
            tokenizer.save_pretrained(self.output_path)

            # 计算参数量
            total_params = sum(p.numel() for p in merged_model.parameters())
            print(f"  [完成] 合并后模型参数量: {total_params/1e9:.2f}B")
            print(f"  输出目录: {self.output_path}")

            return {
                "status": "success",
                "output_path": self.output_path,
                "total_params": total_params,
            }

        except ImportError as e:
            print(f"  [模拟] 依赖缺失: {e}，使用模拟合并演示流程")
            return self._merge_mock()

    def _merge_mock(self) -> Dict:
        """模拟合并过程（演示流程，不真实执行）"""
        print("  [模拟] LoRA合并流程:")
        print("    1. 加载基座模型 (7B → ~14GB)")
        print("    2. 加载LoRA适配器 (rank=16 → ~40MB)")
        print("    3. 合并: W_new = W_base + (32/16) * A * B")
        print("    4. 保存合并后模型 (~14GB)")

        # 创建输出目录和占位文件
        os.makedirs(self.output_path, exist_ok=True)
        with open(os.path.join(self.output_path, "config.json"), "w") as f:
            json.dump({"model_type": "merged_rag_model", "merged_from": "qwen2.5-7b+lora"}, f)

        return {
            "status": "mock",
            "output_path": self.output_path,
            "total_params": 7_000_000_000,
            "lora_params": 40_000_000,
            "merge_formula": "W_new = W_base + (alpha/rank) * A @ B",
        }

    @staticmethod
    def verify_merge(output_path: str) -> Dict:
        """验证合并后的模型可正常推理"""
        print(f"\n  [验证] 测试合并后模型...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(
                output_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(output_path)

            test_prompt = "什么是RAG系统？"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=50)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"  输入: {test_prompt}")
            print(f"  输出: {result[:200]}...")

            return {"status": "ok", "test_output": result[:200]}
        except ImportError:
            return {"status": "mock", "test_output": "[模拟] RAG系统是..."}


# ============================================================
# Section 2: GGUF导出与量化
# ============================================================

class GGUFExporter:
    """
    GGUF导出器

    GGUF (GPT-Generated Unified Format) 是llama.cpp使用的模型格式
    支持多种量化级别，在CPU上也能高效推理

    量化级别（质量从高到低）:
    - q8_0:   8-bit量化，质量最高，文件最大
    - q6_k:   6-bit (K-quant)，高质量推荐
    - q5_k_m: 5-bit medium，质量与大小平衡
    - q4_k_m: 4-bit medium ★推荐生产使用
    - q4_0:   4-bit标准，最小
    - q3_k_m: 3-bit medium，极致压缩
    - q2_k:   2-bit，严重质量损失
    """

    QUANTIZATION_LEVELS = {
        "q8_0":   {"bits": 8,  "quality": "优秀", "size_7b": "7.0GB", "recommend": "质量优先"},
        "q6_k":   {"bits": 6,  "quality": "很好", "size_7b": "5.5GB", "recommend": "高品质"},
        "q5_k_m": {"bits": 5,  "quality": "好",   "size_7b": "4.5GB", "recommend": "均衡"},
        "q4_k_m": {"bits": 4,  "quality": "良好", "size_7b": "4.0GB", "recommend": "★推荐"},
        "q4_0":   {"bits": 4,  "quality": "尚可", "size_7b": "3.5GB", "recommend": "轻量"},
        "q3_k_m": {"bits": 3,  "quality": "降低", "size_7b": "3.0GB", "recommend": "极限压缩"},
    }

    def __init__(self, merged_model_path: str,
                 output_dir: str = "./gguf_output"):
        self.merged_model_path = merged_model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export(self, quantization: str = "q4_k_m") -> Dict:
        """
        导出为GGUF量化模型

        方法1: 使用llama.cpp的convert + quantize工具
        方法2: 使用ollama自动转换

        这里展示方法1的流程
        """
        print(f"\n  [GGUF导出] {quantization} 量化...")

        # 模拟导出过程
        level_info = self.QUANTIZATION_LEVELS.get(quantization, {})
        output_file = os.path.join(
            self.output_dir, f"rag-model-{quantization}.gguf"
        )

        print(f"  量化级别: {quantization} ({level_info.get('bits', '?')}-bit)")
        print(f"  质量: {level_info.get('quality', '?')}")
        print(f"  预期大小: {level_info.get('size_7b', '?')}")

        # 模拟导出步骤
        steps = [
            f"python llama.cpp/convert_hf_to_gguf.py {self.merged_model_path} --outtype f16",
            f"llama.cpp/quantize {output_file.replace('.gguf', '-f16.gguf')} {output_file} {quantization}",
        ]
        for i, step in enumerate(steps, 1):
            print(f"  Step {i}: {step}")

        # 创建占位文件
        Path(output_file).touch()

        return {
            "status": "complete",
            "output_file": output_file,
            "quantization": quantization,
            "bits": level_info.get("bits", "?"),
            "estimated_size": level_info.get("size_7b", "?"),
        }

    @staticmethod
    def compare_quantization_levels() -> str:
        """展示各量化级别的对比表"""
        lines = [
            "\n  GGUF量化级别对比 (7B模型为例):",
            f"  {'级别':<10s} {'Bits':<8s} {'大小':<10s} {'质量':<8s} {'推荐':<s}",
            f"  {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*20}",
        ]
        for level, info in GGUFExporter.QUANTIZATION_LEVELS.items():
            lines.append(
                f"  {level:<10s} {info['bits']:<8d} "
                f"{info['size_7b']:<10s} {info['quality']:<8s} {info['recommend']:<s}"
            )
        return "\n".join(lines)


# ============================================================
# Section 3: Ollama部署
# ============================================================

class OllamaDeployer:
    """
    Ollama部署器

    Ollama是最简单的本地LLM部署方案
    支持:
    - GGUF模型导入
    - REST API (兼容OpenAI格式)
    - 自动GPU检测和卸载
    - 模型管理和版本控制
    """

    @staticmethod
    def create_modelfile(model_name: str,
                         gguf_path: str,
                         system_prompt: str = "",
                         parameters: Dict = None) -> str:
        """创建Ollama Modelfile"""
        if parameters is None:
            parameters = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 4096,
                "repeat_penalty": 1.1,
            }

        params_lines = "\n".join(
            f"PARAMETER {k} {v}" for k, v in parameters.items()
        )

        modelfile = f"""# Ollama Modelfile for RAG Fine-tuned Model
# 生成时间: Auto-generated

FROM {gguf_path}

# 系统提示 (RAG专用)
SYSTEM \"\"\"{system_prompt or '你是基于检索增强生成（RAG）的智能助手。请基于提供的上下文回答问题并标注来源。如果信息不足，请明确说明。'}\"\"\"

# 推理参数
{params_lines}

# Chat模板 (根据基座模型调整)
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"
"""
        return modelfile

    @staticmethod
    def deploy(model_name: str, modelfile_path: str) -> Dict:
        """
        部署到Ollama

        步骤:
        1. ollama create <name> -f <modelfile>
        2. ollama list 验证
        3. ollama run <name> 测试
        """
        print(f"\n  [Ollama部署] 创建模型: {model_name}")
        print(f"  Modelfile: {modelfile_path}")

        # 模拟部署命令
        commands = [
            f"ollama create {model_name} -f {modelfile_path}",
            f"ollama list | grep {model_name}",
            f"ollama run {model_name} '什么是RAG？'",
        ]

        for cmd in commands:
            print(f"  $ {cmd}")

        return {
            "status": "deployed",
            "model_name": model_name,
            "api_endpoint": "http://localhost:11434/api/generate",
            "test_command": f"ollama run {model_name}",
        }

    @staticmethod
    def generate_api_examples(model_name: str) -> str:
        """生成Ollama API调用示例"""
        return f"""# Ollama API 调用示例

## 1. 生成 (Generate)
curl http://localhost:11434/api/generate \\
  -d '{{"model": "{model_name}", "prompt": "什么是向量数据库？"}}'

## 2. 聊天 (Chat) - 兼容OpenAI格式
curl http://localhost:11434/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{model_name}", "messages": [{{"role": "user", "content": "解释RAG原理"}}]}}'

## 3. Python客户端
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={{"model": "{model_name}", "prompt": "RAG系统的核心组件有哪些？", "stream": False}},
)
print(response.json()["response"])

## 4. OpenAI兼容接口
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(
    model="{model_name}",
    messages=[{{"role": "user", "content": "什么是RAG？"}}],
)
print(response.choices[0].message.content)
"""


# ============================================================
# Section 4: 完整部署管道
# ============================================================

class DeploymentPipeline:
    """完整部署管道: 合并 → 量化 → 部署"""

    def __init__(self,
                 base_model: str = "Qwen/Qwen2.5-7B-Instruct",
                 lora_path: str = "./saves/lora-adapter",
                 output_dir: str = "./rag-deploy",
                 model_name: str = "rag-assistant"):
        self.base_model = base_model
        self.lora_path = lora_path
        self.output_dir = output_dir
        self.model_name = model_name

        # 路径规划
        self.merged_path = os.path.join(output_dir, "merged-model")
        self.gguf_path = os.path.join(output_dir, "gguf")
        self.modelfile_path = os.path.join(output_dir, "Modelfile")

        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> Dict:
        """执行完整部署流程"""
        results = {}
        print("=" * 60)
        print("RAG模型部署管道")
        print("=" * 60)

        # 步骤1: 合并LoRA
        merger = LoRAMerger(
            self.base_model, self.lora_path, self.merged_path
        )
        results["merge"] = merger.merge()

        # 步骤2: GGUF导出
        print(f"\n{GGUFExporter.compare_quantization_levels()}")
        exporter = GGUFExporter(self.merged_path, self.gguf_path)
        results["gguf"] = exporter.export(quantization="q4_k_m")

        # 步骤3: 创建Modelfile
        print("\n[步骤3] 创建Ollama Modelfile...")
        deployer = OllamaDeployer()
        gguf_file = results["gguf"]["output_file"]
        modelfile_content = deployer.create_modelfile(
            model_name=self.model_name,
            gguf_path=gguf_file,
            system_prompt="你是RAG智能助手，擅长基于文档上下文回答专业问题并标注信息来源。",
            parameters={
                "temperature": 0.3,  # RAG场景低温度更合适
                "top_p": 0.9,
                "num_ctx": 4096,
            },
        )
        with open(self.modelfile_path, "w") as f:
            f.write(modelfile_content)
        print(f"  Modelfile: {self.modelfile_path}")

        # 步骤4: Ollama部署
        results["ollama"] = deployer.deploy(self.model_name, self.modelfile_path)

        # 步骤5: API示例
        api_examples = deployer.generate_api_examples(self.model_name)
        api_path = os.path.join(self.output_dir, "api_examples.txt")
        with open(api_path, "w") as f:
            f.write(api_examples)
        print(f"\n  API示例已保存: {api_path}")

        return results


# ============================================================
# Section 5: 主流程
# ============================================================

def main():
    """主函数：演示合并量化部署全流程"""
    print("=" * 70)
    print("B2-04: 合并量化部署 — 完整演示")
    print("=" * 70)

    output_dir = "./demo_deploy"

    # 1. LoRA合并演示
    print("\n[演示1] LoRA权重合并")
    merger = LoRAMerger(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        lora_adapter_path="./saves/lora-adapter",
        output_path=os.path.join(output_dir, "merged-model"),
    )
    merge_result = merger.merge()
    print(f"  合并状态: {merge_result['status']}")

    # 2. GGUF量化级别对比
    print(GGUFExporter.compare_quantization_levels())

    # 3. GGUF导出
    print("\n[演示2] GGUF导出 (q4_k_m)")
    exporter = GGUFExporter(
        merged_model_path=os.path.join(output_dir, "merged-model"),
        output_dir=os.path.join(output_dir, "gguf"),
    )
    gguf_result = exporter.export(quantization="q4_k_m")

    # 4. 创建Modelfile
    print("\n[演示3] 创建Ollama Modelfile...")
    deployer = OllamaDeployer()
    modelfile_content = deployer.create_modelfile(
        model_name="rag-assistant",
        gguf_path=gguf_result["output_file"],
        system_prompt="你是RAG智能助手。",
    )
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"\n  Modelfile内容预览:")
    print(f"  {'-'*40}")
    for line in modelfile_content.split("\n")[:15]:
        print(f"  {line}")
    print(f"  ...")

    # 5. API示例
    print("\n[演示4] API调用示例")
    api_examples = deployer.generate_api_examples("rag-assistant")
    print(api_examples[:500])

    # 6. 输出文件清单
    print("\n[演示5] 输出文件清单")
    print(f"  {output_dir}/")
    print(f"    ├── merged-model/    (合并后的HF模型)")
    print(f"    ├── gguf/            (GGUF量化模型)")
    print(f"    │   └── rag-model-q4_k_m.gguf (~4GB)")
    print(f"    ├── Modelfile        (Ollama配置)")
    print(f"    └── api_examples.txt (API调用示例)")

    print("\n" + "=" * 70)
    print("B2-04 演示完成！")
    print("=" * 70)
    print(f"\n所有文件位于: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
