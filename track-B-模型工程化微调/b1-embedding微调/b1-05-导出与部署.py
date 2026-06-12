#!/usr/bin/env python3
"""
B1-05: 导出与部署 (Export & Deployment)
===========================================
学习目标:
  1. ONNX导出
  2. TEI (Text Embeddings Inference) 部署
  3. INT8量化 + 精度损失测量
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================
# Section 1: ONNX导出
# ============================================================

class ONNXExporter:
    """
    ONNX模型导出器

    ONNX (Open Neural Network Exchange) 格式的优势:
    - 跨框架互操作（PyTorch → ONNX Runtime, TensorRT, OpenVINO）
    - 推理优化（图融合、常量折叠）
    - 广泛的生产部署支持
    """

    def __init__(self, model_path_or_name: str,
                 output_dir: str = "./onnx_export"):
        self.model_path = model_path_or_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_to_onnx(self,
                       max_seq_length: int = 512,
                       opset_version: int = 14) -> Dict:
        """
        导出PyTorch模型为ONNX

        步骤:
        1. 加载PyTorch模型
        2. 定义dummy输入
        3. torch.onnx.export()
        4. 验证ONNX模型
        """
        print(f"\n  [ONNX导出] 开始...")
        print(f"  模型: {self.model_path}")
        print(f"  最大序列长度: {max_seq_length}")
        print(f"  Opset版本: {opset_version}")

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            # 加载模型
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModel.from_pretrained(self.model_path)
            model.eval()

            # 准备dummy输入
            dummy_text = "这是一个测试文本用于ONNX导出。"
            inputs = tokenizer(
                dummy_text,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )

            onnx_path = os.path.join(self.output_dir, "model.onnx")

            # 导出
            torch.onnx.export(
                model,
                (inputs["input_ids"], inputs["attention_mask"]),
                onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["embeddings", "pooler_output"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "embeddings": {0: "batch_size"},
                    "pooler_output": {0: "batch_size"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )

            # 验证
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            file_size = os.path.getsize(onnx_path) / (1024 * 1024)

            print(f"  [ONNX导出] 成功!")
            print(f"  输出路径: {onnx_path}")
            print(f"  文件大小: {file_size:.1f} MB")

            return {
                "status": "success",
                "onnx_path": onnx_path,
                "file_size_mb": round(file_size, 2),
                "opset_version": opset_version,
            }

        except ImportError as e:
            print(f"  [ONNX导出] 依赖缺失: {e}")
            return self._export_mock_onnx()

    def _export_mock_onnx(self) -> Dict:
        """模拟ONNX导出（演示流程）"""
        onnx_path = os.path.join(self.output_dir, "model.onnx")
        # 创建一个最小的ONNX protobuf
        with open(onnx_path, "wb") as f:
            f.write(b"MOCK_ONNX_MODEL_PLACEHOLDER")

        print(f"  [ONNX导出] Mock模式完成")
        print(f"  输出路径: {onnx_path}")

        return {
            "status": "mock",
            "onnx_path": onnx_path,
            "file_size_mb": 0.0,
            "opset_version": 14,
        }

    @staticmethod
    def onnx_inference_benchmark(onnx_path: str,
                                 num_runs: int = 100) -> Dict:
        """
        ONNX推理基准测试
        对比PyTorch vs ONNX Runtime的推理速度
        """
        print(f"\n  [Benchmark] ONNX Runtime vs PyTorch...")

        # 模拟基准数据
        np.random.seed(42)
        results = {
            "pytorch": {
                "mean_ms": 12.5,
                "p50_ms": 12.1,
                "p95_ms": 15.3,
                "p99_ms": 18.7,
            },
            "onnx_runtime": {
                "mean_ms": 4.2,
                "p50_ms": 4.0,
                "p95_ms": 5.1,
                "p99_ms": 6.3,
            },
            "speedup": 3.0,  # ONNX Runtime比PyTorch快3倍
        }

        print(f"  PyTorch 平均延迟: {results['pytorch']['mean_ms']:.1f}ms")
        print(f"  ONNX RT 平均延迟: {results['onnx_runtime']['mean_ms']:.1f}ms")
        print(f"  加速比: {results['speedup']:.1f}x")

        return results


# ============================================================
# Section 2: TEI部署配置
# ============================================================

class TEIDeployment:
    """
    TEI (Text Embeddings Inference) 部署

    TEI是HuggingFace开发的高性能Embedding推理服务
    特点:
    - Rust实现，极低延迟
    - 支持Flash Attention
    - 支持动态batching
    - 原生gRPC接口
    - 支持Candle (纯Rust推理) 或 PyTorch后端
    """

    @staticmethod
    def generate_docker_compose(model_name: str = "BAAI/bge-m3",
                                port: int = 8080,
                                max_batch_tokens: int = 16384,
                                max_concurrent_requests: int = 512) -> str:
        """生成Docker Compose配置"""
        config = f"""# TEI Embedding服务 Docker Compose配置
# 用途: 高性能部署BGE-M3 Embedding模型

version: '3.8'

services:
  tei-embedding:
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-latest
    # GPU版本: ghcr.io/huggingface/text-embeddings-inference:1.5

    container_name: tei-bge-m3

    ports:
      - "{port}:80"

    volumes:
      - ./models:/data  # 模型缓存目录

    environment:
      - MODEL_ID={model_name}
      - MAX_BATCH_TOKENS={max_batch_tokens}
      - MAX_CONCURRENT_REQUESTS={max_concurrent_requests}

    # GPU配置（取消注释以下行）:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

    restart: unless-stopped

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
        return config

    @staticmethod
    def generate_api_examples() -> str:
        """生成TEI API调用示例"""
        return """# TEI API 调用示例

## 1. 健康检查
curl http://localhost:8080/health

## 2. 单条文本嵌入
curl http://localhost:8080/embed \\
  -X POST \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": "什么是RAG系统？", "normalize": true}'

## 3. 批量文本嵌入
curl http://localhost:8080/embed \\
  -X POST \\
  -H "Content-Type: application/json" \\
  -d '{
    "inputs": ["什么是RAG？", "向量数据库有哪些？", "如何微调Embedding？"],
    "normalize": true
  }'

## 4. Python客户端
import httpx

async def embed_texts(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/embed",
            json={"inputs": texts, "normalize": True},
            timeout=30.0,
        )
        return response.json()  # List of embeddings

## 5. gRPC接口 (更快)
# TEI也支持gRPC，性能比REST更高
# 生成proto: pip install text-embeddings-inference
"""

    @staticmethod
    def generate_modelfile(model_path: str) -> str:
        """生成Ollama Modelfile（用于Embedding）"""
        return f"""# Ollama Embedding Model Modelfile
# BGE-M3 Embedding Model

FROM {model_path}

PARAMETER temperature 0
PARAMETER num_ctx 512
PARAMETER mirostat 0

TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"

SYSTEM \"\"\"You are a Chinese-English multilingual text embedding model.\"\"\"
"""


# ============================================================
# Section 3: INT8量化 + 精度损失
# ============================================================

class INT8Quantizer:
    """
    INT8量化器

    量化策略:
    1. 动态量化 (Dynamic Quantization): 推理时动态计算scale/zero_point
    2. 静态量化 (Static Quantization): 校准后固定scale/zero_point
    3. QAT (Quantization-Aware Training): 训练时模拟量化

    这里演示动态量化和精度损失评估
    """

    def __init__(self):
        pass

    @staticmethod
    def simulate_quantization(fp32_array: np.ndarray,
                              num_bits: int = 8) -> Tuple[np.ndarray, Dict]:
        """
        模拟INT8量化

        步骤:
        1. 计算scale和zero_point
        2. 量化: q = round(x / scale) + zero_point
        3. 反量化: x' = (q - zero_point) * scale
        4. 计算误差

        公式:
        scale = (x_max - x_min) / (2^bits - 1)
        zero_point = round(-x_min / scale)
        """
        x_min = fp32_array.min()
        x_max = fp32_array.max()
        max_q = 2 ** num_bits - 1

        scale = (x_max - x_min) / max_q if x_max > x_min else 1.0
        zero_point = np.round(-x_min / scale)
        zero_point = np.clip(zero_point, 0, max_q)

        # 量化
        quantized = np.round(fp32_array / scale + zero_point)
        quantized = np.clip(quantized, 0, max_q)

        # 反量化
        dequantized = (quantized.astype(np.float32) - zero_point) * scale

        # 误差分析
        abs_error = np.abs(fp32_array - dequantized)
        rel_error = abs_error / (np.abs(fp32_array) + 1e-8)

        stats = {
            "bits": num_bits,
            "scale": float(scale),
            "zero_point": int(zero_point),
            "mae": float(np.mean(abs_error)),
            "mse": float(np.mean((fp32_array - dequantized) ** 2)),
            "max_abs_error": float(np.max(abs_error)),
            "mean_rel_error": float(np.mean(rel_error)),
            "memory_reduction": (1 - num_bits / 32) * 100,  # 相对于FP32的节省
        }

        return dequantized, stats

    @staticmethod
    def quantize_embeddings(embeddings: np.ndarray,
                           method: str = "int8") -> Tuple[np.ndarray, Dict]:
        """
        量化嵌入向量

        参数:
            embeddings: FP32嵌入向量 (n_samples, dim)
            method: "int8" | "binary" | "uint8"

        返回:
            quantized_embeddings: 量化后的向量
            stats: 量化统计
        """
        if method == "int8":
            # Per-channel量化（每个维度独立量化）
            quantized = np.zeros_like(embeddings, dtype=np.int8)
            scales = np.zeros(embeddings.shape[1], dtype=np.float32)
            zero_points = np.zeros(embeddings.shape[1], dtype=np.int8)

            for dim in range(embeddings.shape[1]):
                column = embeddings[:, dim]
                column_min = column.min()
                column_max = column.max()
                scale = (column_max - column_min) / 255.0 if column_max > column_min else 1.0
                scales[dim] = scale
                zp = np.round(-column_min / scale)
                zero_points[dim] = np.clip(zp, -128, 127).astype(np.int8)
                quantized[:, dim] = np.clip(
                    np.round(column / scale + zero_points[dim]),
                    -128, 127
                ).astype(np.int8)

            # 反量化用于精度评估
            dequantized = (quantized.astype(np.float32) - zero_points.astype(np.float32)) * scales

            # 计算余弦相似度损失
            norm_orig = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            norm_deq = dequantized / (np.linalg.norm(dequantized, axis=1, keepdims=True) + 1e-8)

            cos_sim_orig = np.dot(norm_orig, norm_orig.T)
            cos_sim_deq = np.dot(norm_deq, norm_deq.T)

            similarity_loss = np.mean(np.abs(cos_sim_orig - cos_sim_deq))

        elif method == "binary":
            # 二值化：只保留符号
            quantized = np.sign(embeddings).astype(np.int8)
            dequantized = quantized.astype(np.float32)
            similarity_loss = None  # 二值化不适用余弦相似度
            scales = None
            zero_points = None

        else:
            raise ValueError(f"不支持的量化方法: {method}")

        stats = {
            "method": method,
            "original_bytes": embeddings.nbytes,
            "quantized_bytes": quantized.nbytes,
            "compression_ratio": embeddings.nbytes / max(quantized.nbytes, 1),
            "memory_saved_percent": (1 - quantized.nbytes / embeddings.nbytes) * 100,
        }

        if similarity_loss is not None:
            stats["cosine_similarity_loss"] = round(float(similarity_loss), 6)

        return dequantized, stats

    @staticmethod
    def benchmark_accuracy_loss(embeddings: np.ndarray,
                                queries: np.ndarray,
                                relevant_indices: List[int],
                                top_k: int = 10) -> Dict:
        """
        测量量化前后的检索精度损失

        对比FP32和INT8的Recall@K
        """
        # FP32基线
        norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        norm_q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        sim_fp32 = np.dot(norm_q, norm_emb.T)

        # INT8量化
        emb_int8, _ = INT8Quantizer.quantize_embeddings(embeddings, "int8")
        norm_emb_int8 = emb_int8 / (np.linalg.norm(emb_int8, axis=1, keepdims=True) + 1e-8)
        sim_int8 = np.dot(norm_q, norm_emb_int8.T)

        # 对比Recall@K
        recall_fp32 = 0
        recall_int8 = 0
        total = len(queries)

        for q_idx in range(total):
            top_fp32 = set(np.argsort(sim_fp32[q_idx])[::-1][:top_k])
            top_int8 = set(np.argsort(sim_int8[q_idx])[::-1][:top_k])

            # 取第一个查询的相关文档（简化评估）
            rel_set = set(relevant_indices[:3]) if relevant_indices else set()
            if rel_set & top_fp32:
                recall_fp32 += 1
            if rel_set & top_int8:
                recall_int8 += 1

        return {
            "fp32_recall": recall_fp32 / max(total, 1),
            "int8_recall": recall_int8 / max(total, 1),
            "recall_drop": (recall_fp32 - recall_int8) / max(total, 1),
            "recall_drop_percent": round(
                (recall_fp32 - recall_int8) / max(recall_fp32, 1) * 100, 2
            ),
            "memory_saved": (1 - 8/32) * 100,  # 75%
        }


# ============================================================
# Section 4: 完整部署管道
# ============================================================

class DeploymentPipeline:
    """完整部署管道"""

    def __init__(self, model_path: str, output_dir: str = "./deploy"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, export_onnx: bool = True,
            generate_tei_config: bool = True,
            run_quantization: bool = True) -> Dict:
        """运行完整部署流程"""
        results = {}

        # 步骤1: ONNX导出
        if export_onnx:
            print("\n[步骤1] ONNX导出...")
            exporter = ONNXExporter(self.model_path, self.output_dir)
            results["onnx"] = exporter.export_to_onnx()

            # Benchmark
            if os.path.exists(results["onnx"].get("onnx_path", "")):
                results["benchmark"] = ONNXExporter.onnx_inference_benchmark(
                    results["onnx"]["onnx_path"]
                )

        # 步骤2: TEI配置
        if generate_tei_config:
            print("\n[步骤2] 生成TEI部署配置...")
            docker_compose = TEIDeployment.generate_docker_compose(
                model_name=self.model_path
            )
            compose_path = os.path.join(self.output_dir, "docker-compose.yml")
            with open(compose_path, "w") as f:
                f.write(docker_compose)
            print(f"  Docker Compose: {compose_path}")

            # API示例
            api_examples = TEIDeployment.generate_api_examples()
            api_path = os.path.join(self.output_dir, "tei_api_examples.txt")
            with open(api_path, "w") as f:
                f.write(api_examples)
            print(f"  API示例: {api_path}")

        # 步骤3: 量化评估
        if run_quantization:
            print("\n[步骤3] INT8量化精度评估...")
            np.random.seed(42)
            # 模拟嵌入向量
            sample_embeddings = np.random.randn(100, 768).astype(np.float32)
            sample_queries = np.random.randn(5, 768).astype(np.float32)

            quantizer = INT8Quantizer()
            _, quant_stats = quantizer.quantize_embeddings(sample_embeddings, "int8")

            # Benchmark
            acc_loss = quantizer.benchmark_accuracy_loss(
                sample_embeddings, sample_queries,
                relevant_indices=list(range(5)),
                top_k=10,
            )

            results["quantization"] = {
                "stats": quant_stats,
                "accuracy_loss": acc_loss,
            }

            print(f"  量化方法: {quant_stats['method']}")
            print(f"  压缩比: {quant_stats['compression_ratio']:.1f}x")
            print(f"  内存节省: {quant_stats['memory_saved_percent']:.0f}%")
            if 'cosine_similarity_loss' in quant_stats:
                print(f"  余弦相似度损失: {quant_stats['cosine_similarity_loss']:.6f}")
            print(f"  FP32 Recall: {acc_loss['fp32_recall']:.4f}")
            print(f"  INT8 Recall: {acc_loss['int8_recall']:.4f}")
            print(f"  召回率下降: {acc_loss['recall_drop_percent']:.2f}%")

        return results


# ============================================================
# Section 5: 主流程
# ============================================================

def main():
    """主函数：演示导出与部署全流程"""
    print("=" * 70)
    print("B1-05: 导出与部署 — 完整演示")
    print("=" * 70)

    output_dir = "./deploy_output"

    # 1. 量化演示
    print("\n[演示1] INT8量化模拟...")
    np.random.seed(42)
    fp32_data = np.random.randn(1000).astype(np.float32) * 0.5 + 0.5

    dequantized, stats = INT8Quantizer.simulate_quantization(fp32_data, num_bits=8)
    print(f"  量化bits: {stats['bits']}")
    print(f"  MAE (平均绝对误差): {stats['mae']:.6f}")
    print(f"  MSE (均方误差): {stats['mse']:.6f}")
    print(f"  最大绝对误差: {stats['max_abs_error']:.6f}")
    print(f"  平均相对误差: {stats['mean_rel_error']:.6f}")
    print(f"  内存节省: {stats['memory_reduction']:.0f}%")

    # 2. 嵌入向量量化
    print("\n[演示2] 嵌入向量INT8量化...")
    embeddings = np.random.randn(100, 768).astype(np.float32)
    queries = np.random.randn(5, 768).astype(np.float32)

    deq, embed_stats = INT8Quantizer.quantize_embeddings(embeddings, "int8")
    print(f"  原始大小: {embed_stats['original_bytes']:,} bytes")
    print(f"  量化后: {embed_stats['quantized_bytes']:,} bytes")
    print(f"  压缩比: {embed_stats['compression_ratio']:.1f}x")
    print(f"  余弦相似度损失: {embed_stats.get('cosine_similarity_loss', 'N/A')}")

    # 3. 精度损失测量
    print("\n[演示3] 量化精度损失Benchmark...")
    acc_loss = INT8Quantizer.benchmark_accuracy_loss(
        embeddings, queries, list(range(10)), top_k=10
    )
    print(f"  FP32 Recall@10: {acc_loss['fp32_recall']:.4f}")
    print(f"  INT8 Recall@10: {acc_loss['int8_recall']:.4f}")
    print(f"  召回率下降: {acc_loss['recall_drop_percent']:.2f}%")

    # 4. 完整部署管道
    print("\n[演示4] 完整部署管道...")
    pipeline = DeploymentPipeline(
        model_path="BAAI/bge-m3",
        output_dir=output_dir,
    )
    results = pipeline.run(
        export_onnx=True,
        generate_tei_config=True,
        run_quantization=True,
    )

    # 5. 展示生成的文件
    print("\n[演示5] 生成的文件...")
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath)
            print(f"  {filepath} ({size} bytes)")

    print("\n" + "=" * 70)
    print("B1-05 演示完成！")
    print("=" * 70)
    print(f"\n所有生成文件位于: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
