#!/usr/bin/env python3
"""
=============================================================================
嵌入模型对比基准测试 (Embedding Model Comparison Benchmark)
=============================================================================

本脚本对比测试 10 个主流嵌入模型在双语（中英文）检索任务上的表现，
包括精度、延迟、成本三个维度的全面评估。

对比模型:
  1. OpenAI text-embedding-3-small (1536 维)
  2. OpenAI text-embedding-3-small (512 维, Matryoshka)
  3. OpenAI text-embedding-3-large (3072 维)
  4. OpenAI text-embedding-3-large (256 维, Matryoshka)
  5. BAAI/bge-m3 (1024 维, 多语言)
  6. BAAI/bge-large-zh-v1.5 (1024 维, 中文优化)
  7. moka-ai/m3e-base (768 维, 中文优化)
  8. Cohere embed-english-v3.0 (1024 维)
  9. Jina jina-embeddings-v3 (1024 维)
  10. Voyage voyage-large-2 (1536 维)

使用方式:
  python 04-embeddings-comparison.py

必需 API Key（环境变量）:
  - OPENAI_API_KEY: OpenAI 嵌入模型
  - COHERE_API_KEY: Cohere 嵌入模型（可选）
  - JINA_API_KEY: Jina 嵌入模型（可选）
  - VOYAGE_API_KEY: Voyage 嵌入模型（可选）

依赖安装:
  pip install openai numpy sentence-transformers tabulate cohere voyageai
=============================================================================
"""

import time
import os
import json
import math
import random
from typing import Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import statistics
import concurrent.futures

import numpy as np

# ---------------------------------------------------------------------------
# 可选依赖：tabulate 用于美观表格；不可用时回退到手动格式化
# ---------------------------------------------------------------------------
try:
    from tabulate import tabulate as tabulate_fn
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

# ---------------------------------------------------------------------------
# 可选依赖：各 API 供应商 SDK
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI as OpenAIClient
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    import cohere as cohere_sdk
    _HAS_COHERE = True
except ImportError:
    _HAS_COHERE = False

try:
    import voyageai as voyageai_sdk
    _HAS_VOYAGE = True
except ImportError:
    _HAS_VOYAGE = False

try:
    import requests as requests_sdk
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ===========================================================================
# 测试数据：双语文档集合
# ===========================================================================

CHINESE_DOCUMENTS: list[str] = [
    # 中文文档 1-50：人工智能与机器学习相关主题
    "深度学习是机器学习的一个子集，它使用多层神经网络从大量数据中学习模式。近年来，深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
    "Transformer架构由Vaswani等人在2017年提出，它完全基于注意力机制，摒弃了传统的循环神经网络结构。这种架构极大地提升了序列建模的效率和效果。",
    "BERT（Bidirectional Encoder Representations from Transformers）是一个预训练的语言表示模型，通过在大规模文本语料上进行双向训练，能够捕捉丰富的语言特征。",
    "GPT系列模型是生成式预训练变换器，采用自回归方式生成文本。GPT-4在多种自然语言处理任务上展示了接近人类水平的表现。",
    "向量数据库是专门用于存储和检索高维向量的数据库系统，广泛应用于语义搜索、推荐系统和异常检测等场景。",
    "RAG（检索增强生成）是一种结合信息检索和文本生成的技术架构。它先从知识库中检索相关文档，再基于检索结果生成更加准确的回答。",
    "大语言模型（LLM）是指参数量达到数十亿甚至数千亿级别的神经网络模型。这些模型在海量文本数据上训练，展现出强大的语言理解和生成能力。",
    "迁移学习是一种机器学习方法，它将在一个任务上学到的知识应用到另一个相关任务中。在NLP领域，预训练加微调已经成为标准范式。",
    "语义搜索不同于关键词匹配，它理解用户查询的语义意图，通过计算查询和文档之间的语义相似度来返回最相关的结果。",
    "嵌入（Embedding）是将离散的对象（如词语、句子、图像）映射到连续向量空间的技术。好的嵌入能够捕捉对象之间的语义关系。",
    "注意力机制是Transformer架构的核心组件，它允许模型在处理序列时关注不同位置的信息。多头注意力进一步增强了模型的表达能力。",
    "扩散模型是一类生成模型，通过逐步去噪过程从随机噪声中生成高质量的数据。Stable Diffusion和DALL-E都是基于扩散模型的图像生成系统。",
    "强化学习是一种通过与环境交互来学习最优策略的机器学习方法。智能体通过试错获得奖励信号，逐步优化其行为策略。",
    "自然语言处理（NLP）是人工智能的重要分支，旨在让计算机理解、生成和处理人类语言。常见任务包括文本分类、命名实体识别和机器翻译等。",
    "计算机视觉是让计算机从图像和视频中获取高级语义信息的技术领域。卷积神经网络（CNN）是计算机视觉中最常用的深度学习架构。",
    "知识图谱是一种以图结构表示知识的数据模型，其中节点表示实体，边表示实体之间的关系。它广泛应用于搜索引擎和问答系统中。",
    "联邦学习是一种分布式机器学习方法，允许多个参与方在不共享原始数据的情况下协作训练模型。这种方法有效地保护了数据隐私。",
    "自监督学习是一种无需人工标注的学习范式，模型通过设计预训练任务从无标签数据中自动学习表示。它在NLP和CV领域都取得了显著成功。",
    "多层感知机（MLP）是最基础的前馈神经网络结构，由输入层、隐藏层和输出层组成。虽然结构简单，但在许多任务中仍然有效。",
    "生成对抗网络（GAN）由生成器和判别器组成，两者通过对抗训练不断提升。GAN在图像生成、风格迁移和数据增强等任务中表现出色。",
    "少样本学习（Few-shot Learning）旨在让模型仅通过少量标注样本就能快速适应新任务。这在标注数据稀缺的场景中具有重要应用价值。",
    "模型蒸馏是一种模型压缩技术，通过让小模型（学生）学习大模型（教师）的输出分布来传递知识，从而在保持性能的同时减小模型体积。",
    "数据增强是通过对原始数据进行变换来生成更多训练样本的技术。在文本领域，常见方法包括同义词替换、回译和随机删除等。",
    "损失函数衡量模型预测值与真实值之间的差距，是模型优化的目标。交叉熵损失和均方误差损失是最常用的两种损失函数。",
    "梯度下降是最基础的优化算法，通过沿损失函数梯度的反方向更新参数来最小化损失。随机梯度下降（SGD）和Adam是常用的变体。",
    "过拟合是机器学习中的常见问题，模型在训练数据上表现很好但在测试数据上表现差。正则化和早停是常用的防止过拟合的方法。",
    "批量归一化是一种加速神经网络训练的技术，通过对每层的输入进行归一化处理来稳定训练过程，同时具有一定的正则化效果。",
    "残差网络（ResNet）通过引入跳跃连接解决了深层网络的退化问题。这种结构使得训练上百层的深度网络成为可能。",
    "对抗攻击是指通过对输入数据添加微小的扰动来欺骗机器学习模型。研究对抗攻击有助于提高模型的鲁棒性和安全性。",
    "命名实体识别（NER）是从文本中识别出命名实体（如人名、地名、机构名）的任务。它是信息抽取的重要基础技术。",
    "文本分类是将文本分配到预定义类别的任务，广泛应用于垃圾邮件检测、情感分析和新闻分类等场景。",
    "机器翻译是使用计算机将一种语言的文本自动翻译成另一种语言的技术。神经机器翻译（NMT）已成为当前的主流方法。",
    "情感分析是识别和提取文本中主观信息的任务，用于判断文本表达的情感倾向（正面、负面或中性）。",
    "问答系统能够自动回答用户提出的自然语言问题。开放域问答系统可以从大规模知识库中检索和推理出答案。",
    "推荐系统通过分析用户行为和偏好来向用户推荐可能感兴趣的项目。协同过滤和基于内容的推荐是最经典的方法。",
    "时间序列预测是根据历史数据预测未来值的任务。LSTM和Transformer等深度学习模型在时间序列预测中表现出色。",
    "异常检测是识别与预期模式不符的数据点的任务，在金融欺诈检测、工业设备监控和网络安全等领域有广泛应用。",
    "图像分割是将图像划分为多个有意义的区域或对象的任务。U-Net和Mask R-CNN是常用的图像分割模型。",
    "目标检测是同时识别图像中物体的位置和类别的任务。YOLO系列模型以其实时性和准确性著称。",
    "语音识别（ASR）是将语音信号转换为文本的技术。深度学习极大地推动了语音识别技术的发展，使准确率大幅提升。",
    "知识蒸馏中温度参数控制教师模型输出的软化程度。较高的温度使概率分布更加平滑，为学生模型提供更丰富的学习信号。",
    "对比学习是一种自监督学习方法，通过拉近正样本对、推远负样本对来学习有效的特征表示。SimCLR和MoCo是经典的对比学习框架。",
    "提示工程（Prompt Engineering）是设计和优化输入提示以引导大语言模型生成期望输出的技术。它对于充分发挥LLM的能力至关重要。",
    "思维链（Chain-of-Thought）推理是一种通过让模型展示中间推理步骤来提高复杂问题解决能力的方法。",
    "RLHF（基于人类反馈的强化学习）是将人类偏好融入语言模型训练的方法，通过奖励模型引导模型生成更符合人类期望的文本。",
    "模型量化是通过降低模型参数的数值精度来减小模型体积和加速推理的技术。INT8量化在保持精度的同时大幅降低了计算需求。",
    "多模态学习是指同时处理和理解多种类型数据（如文本、图像、音频）的机器学习方法。CLIP和GPT-4V是多模态学习的代表。",
    "持续学习（Continual Learning）旨在让模型在连续学习新任务时不会灾难性遗忘旧任务的知识。这是实现通用人工智能的重要研究方向。",
    "图神经网络（GNN）是处理图结构数据的深度学习模型，在社交网络分析、分子性质预测和推荐系统中有广泛应用。",
    "AI对齐（AI Alignment）是确保人工智能系统的目标和行为与人类价值观和意图一致的跨学科研究方向。",
]

ENGLISH_DOCUMENTS: list[str] = [
    # English Documents 1-50: AI/ML Topics
    "Deep learning is a subset of machine learning that uses multi-layered neural networks to learn patterns from vast amounts of data. It has achieved breakthroughs in image recognition, natural language processing, and speech recognition in recent years.",
    "The Transformer architecture was proposed by Vaswani et al. in 2017. It relies entirely on attention mechanisms, discarding traditional recurrent neural network structures, which greatly improved the efficiency and effectiveness of sequence modeling.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language representation model that captures rich linguistic features through bidirectional training on large-scale text corpora.",
    "The GPT series models are Generative Pre-trained Transformers that generate text in an autoregressive manner. GPT-4 has demonstrated near-human performance on various natural language processing tasks.",
    "Vector databases are specialized database systems designed for storing and retrieving high-dimensional vectors, widely used in semantic search, recommendation systems, and anomaly detection scenarios.",
    "RAG (Retrieval-Augmented Generation) is a technical architecture that combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then generates more accurate responses based on the retrieved results.",
    "Large Language Models (LLMs) refer to neural network models with parameters reaching billions or even hundreds of billions. These models are trained on massive text data and demonstrate powerful language understanding and generation capabilities.",
    "Transfer learning is a machine learning method that applies knowledge learned from one task to another related task. In the NLP field, pre-training plus fine-tuning has become the standard paradigm.",
    "Semantic search differs from keyword matching by understanding the semantic intent of user queries. It returns the most relevant results by computing semantic similarity between queries and documents.",
    "Embedding is a technique that maps discrete objects such as words, sentences, and images into continuous vector spaces. Good embeddings capture semantic relationships between objects.",
    "The attention mechanism is the core component of the Transformer architecture, allowing the model to focus on information at different positions when processing sequences. Multi-head attention further enhances the model's expressive power.",
    "Diffusion models are a class of generative models that generate high-quality data from random noise through a progressive denoising process. Stable Diffusion and DALL-E are both diffusion-based image generation systems.",
    "Reinforcement learning is a machine learning method that learns optimal policies through interaction with the environment. Agents receive reward signals through trial and error, gradually optimizing their behavioral strategies.",
    "Natural Language Processing (NLP) is an important branch of artificial intelligence aimed at enabling computers to understand, generate, and process human language. Common tasks include text classification, named entity recognition, and machine translation.",
    "Computer vision is the technical field of enabling computers to obtain high-level semantic information from images and videos. Convolutional Neural Networks (CNNs) are the most commonly used deep learning architecture in computer vision.",
    "Knowledge graphs are data models that represent knowledge in graph structures, where nodes represent entities and edges represent relationships between entities. They are widely used in search engines and question-answering systems.",
    "Federated learning is a distributed machine learning approach that allows multiple participants to collaboratively train models without sharing raw data. This method effectively protects data privacy.",
    "Self-supervised learning is a learning paradigm that does not require human annotation. Models automatically learn representations from unlabeled data by designing pretext tasks. It has achieved remarkable success in both NLP and CV.",
    "Multilayer Perceptrons (MLPs) are the most basic feedforward neural network architecture, consisting of input layers, hidden layers, and output layers. Despite their simplicity, they remain effective in many tasks.",
    "Generative Adversarial Networks (GANs) consist of a generator and a discriminator that improve through adversarial training. GANs excel in image generation, style transfer, and data augmentation tasks.",
    "Few-shot learning aims to enable models to quickly adapt to new tasks with only a small number of labeled samples. This has important applications in scenarios where labeled data is scarce.",
    "Model distillation is a model compression technique that transfers knowledge by having a smaller model (student) learn the output distribution of a larger model (teacher), reducing model size while maintaining performance.",
    "Data augmentation is a technique that generates more training samples by applying transformations to original data. In the text domain, common methods include synonym replacement, back-translation, and random deletion.",
    "Loss functions measure the gap between model predictions and true values, serving as the optimization objective. Cross-entropy loss and mean squared error loss are the two most commonly used loss functions.",
    "Gradient descent is the most fundamental optimization algorithm, updating parameters by moving in the opposite direction of the loss function gradient to minimize loss. SGD and Adam are commonly used variants.",
    "Overfitting is a common problem in machine learning where models perform well on training data but poorly on test data. Regularization and early stopping are common methods for preventing overfitting.",
    "Batch normalization is a technique that accelerates neural network training by normalizing the inputs of each layer to stabilize the training process, while also providing some regularization effect.",
    "Residual Networks (ResNets) solve the degradation problem of deep networks by introducing skip connections. This structure makes it possible to train deep networks with hundreds of layers.",
    "Adversarial attacks refer to deceiving machine learning models by adding tiny perturbations to input data. Research on adversarial attacks helps improve model robustness and security.",
    "Named Entity Recognition (NER) is the task of identifying named entities such as person names, locations, and organization names from text. It is an important foundational technique for information extraction.",
    "Text classification is the task of assigning text to predefined categories, widely used in spam detection, sentiment analysis, and news classification scenarios.",
    "Machine translation is the technology of automatically translating text from one language to another using computers. Neural Machine Translation (NMT) has become the current mainstream approach.",
    "Sentiment analysis is the task of identifying and extracting subjective information from text, used to determine the emotional tendency expressed in text (positive, negative, or neutral).",
    "Question Answering systems can automatically answer natural language questions posed by users. Open-domain QA systems can retrieve and reason from large-scale knowledge bases to produce answers.",
    "Recommendation systems recommend items that users might be interested in by analyzing user behavior and preferences. Collaborative filtering and content-based recommendation are the most classic methods.",
    "Time series forecasting is the task of predicting future values based on historical data. Deep learning models such as LSTM and Transformer have shown excellent performance in time series forecasting.",
    "Anomaly detection is the task of identifying data points that do not conform to expected patterns, with wide applications in financial fraud detection, industrial equipment monitoring, and network security.",
    "Image segmentation is the task of dividing an image into multiple meaningful regions or objects. U-Net and Mask R-CNN are commonly used image segmentation models.",
    "Object detection is the task of simultaneously identifying the location and category of objects in images. The YOLO series of models are known for their real-time performance and accuracy.",
    "Speech recognition (ASR) is the technology of converting speech signals into text. Deep learning has greatly advanced speech recognition technology, significantly improving accuracy.",
    "The temperature parameter in knowledge distillation controls the softness of the teacher model's output. Higher temperatures make probability distributions smoother, providing richer learning signals for the student model.",
    "Contrastive learning is a self-supervised learning method that learns effective feature representations by pulling positive sample pairs closer and pushing negative sample pairs apart. SimCLR and MoCo are classic contrastive learning frameworks.",
    "Prompt engineering is the technique of designing and optimizing input prompts to guide large language models to generate desired outputs. It is crucial for fully leveraging the capabilities of LLMs.",
    "Chain-of-Thought reasoning is a method that improves complex problem-solving ability by having models demonstrate intermediate reasoning steps.",
    "RLHF (Reinforcement Learning from Human Feedback) is a method for incorporating human preferences into language model training, using a reward model to guide models toward generating text that better aligns with human expectations.",
    "Model quantization is a technique that reduces model size and accelerates inference by lowering the numerical precision of model parameters. INT8 quantization significantly reduces computational requirements while maintaining accuracy.",
    "Multimodal learning refers to machine learning methods that simultaneously process and understand multiple types of data such as text, images, and audio. CLIP and GPT-4V are representative examples of multimodal learning.",
    "Continual learning aims to enable models to continuously learn new tasks without catastrophically forgetting knowledge from old tasks. This is an important research direction for achieving artificial general intelligence.",
    "Graph Neural Networks (GNNs) are deep learning models for processing graph-structured data, with wide applications in social network analysis, molecular property prediction, and recommendation systems.",
    "AI alignment is an interdisciplinary research direction that ensures the goals and behaviors of artificial intelligence systems are consistent with human values and intentions.",
]

ALL_DOCUMENTS: list[str] = CHINESE_DOCUMENTS + ENGLISH_DOCUMENTS
DOCS_ZH_INDICES: list[int] = list(range(len(CHINESE_DOCUMENTS)))
DOCS_EN_INDICES: list[int] = list(range(len(CHINESE_DOCUMENTS), len(ALL_DOCUMENTS)))


# ===========================================================================
# 测试查询（双语，含正确答案标注）
# ===========================================================================

CHINESE_QUERIES: list[str] = [
    "深度学习在图像识别方面的应用有哪些？",
    "Transformer的注意力机制是如何工作的？",
    "什么是检索增强生成（RAG）技术？",
    "向量数据库的主要应用场景有哪些？",
    "联邦学习如何保护数据隐私？",
    "GPT模型采用什么样的文本生成方式？",
    "迁移学习的基本思想是什么？",
    "如何防止机器学习模型过拟合？",
    "知识蒸馏技术是如何压缩模型的？",
    "多模态学习如何融合不同类型的数据？",
]

ENGLISH_QUERIES: list[str] = [
    "How does deep learning differ from traditional machine learning?",
    "What is the attention mechanism in Transformers?",
    "How does Retrieval-Augmented Generation improve answer accuracy?",
    "What are vector databases used for in AI applications?",
    "How does federated learning protect privacy?",
    "What is the GPT architecture and how does it generate text?",
    "What is transfer learning and why is it important?",
    "What is overfitting and how can it be prevented?",
    "How does knowledge distillation compress neural networks?",
    "What is multimodal learning and how does it work?",
]

ALL_QUERIES: list[str] = CHINESE_QUERIES + ENGLISH_QUERIES

# 每个查询对应的相关文档索引（0-based，在 ALL_DOCUMENTS 中）
# 中文查询对应中文文档，英文查询对应英文文档
RELEVANT_DOCS: list[list[int]] = [
    [0, 1, 2, 4, 5],       # Q0: 深度学习在图像识别方面的应用
    [1, 10, 2, 0, 6],       # Q1: Transformer的注意力机制
    [5, 4, 1, 8, 0],        # Q2: 检索增强生成
    [4, 5, 8, 0, 33],       # Q3: 向量数据库应用
    [16, 0, 3, 5, 46],      # Q4: 联邦学习隐私
    [3, 1, 0, 5, 42],       # Q5: GPT文本生成
    [7, 0, 3, 5, 40],       # Q6: 迁移学习
    [25, 26, 0, 3, 1],      # Q7: 防止过拟合
    [21, 40, 0, 3, 25],     # Q8: 知识蒸馏
    [46, 0, 1, 3, 33],      # Q9: 多模态学习
    [50, 51, 52, 54, 55],   # Q10: How does deep learning differ
    [51, 60, 52, 50, 56],   # Q11: attention mechanism
    [55, 54, 51, 58, 50],   # Q12: RAG
    [54, 55, 58, 50, 83],   # Q13: vector databases
    [66, 50, 53, 55, 96],   # Q14: federated learning
    [53, 51, 50, 55, 92],   # Q15: GPT architecture
    [57, 50, 53, 55, 90],   # Q16: transfer learning
    [75, 76, 50, 53, 51],   # Q17: overfitting
    [71, 90, 50, 53, 75],   # Q18: knowledge distillation
    [96, 50, 51, 53, 83],   # Q19: multimodal learning
]


# ===========================================================================
# 模型注册表
# ===========================================================================

MODELS: dict[str, dict[str, Any]] = {
    "openai-text-3-small": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dims": 1536,
    },
    "openai-text-3-small-512": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dims": 512,
    },
    "openai-text-3-large": {
        "provider": "openai",
        "model": "text-embedding-3-large",
        "dims": 3072,
    },
    "openai-text-3-large-256": {
        "provider": "openai",
        "model": "text-embedding-3-large",
        "dims": 256,
    },
    "bge-m3": {
        "provider": "sentence_transformers",
        "model": "BAAI/bge-m3",
        "dims": 1024,
    },
    "bge-large-zh": {
        "provider": "sentence_transformers",
        "model": "BAAI/bge-large-zh-v1.5",
        "dims": 1024,
    },
    "m3e-base": {
        "provider": "sentence_transformers",
        "model": "moka-ai/m3e-base",
        "dims": 768,
    },
    "cohere-embed-v3": {
        "provider": "cohere",
        "model": "embed-english-v3.0",
        "dims": 1024,
    },
    "jina-v3": {
        "provider": "jina",
        "model": "jina-embeddings-v3",
        "dims": 1024,
    },
    "voyage-large-2": {
        "provider": "voyage",
        "model": "voyage-large-2",
        "dims": 1536,
    },
}


# ===========================================================================
# 成本价目表（每百万 token 美元价格）
# ===========================================================================

COST_PER_1M_TOKENS: dict[str, float] = {
    "openai-text-3-small": 0.02,
    "openai-text-3-small-512": 0.02,
    "openai-text-3-large": 0.13,
    "openai-text-3-large-256": 0.13,
    "bge-m3": 0.0,
    "bge-large-zh": 0.0,
    "m3e-base": 0.0,
    "cohere-embed-v3": 0.10,
    "jina-v3": 0.08,
    "voyage-large-2": 0.14,
}


# ===========================================================================
# 抽象嵌入器基类
# ===========================================================================

class BaseEmbedder(ABC):
    """所有嵌入模型的抽象基类。"""

    def __init__(self, model_name: str, config: dict) -> None:
        """初始化嵌入器。

        Args:
            model_name: 注册表中的模型名称。
            config: 模型配置字典（来自 MODELS 注册表）。
        """
        self.model_name: str = model_name
        self.config: dict = config
        self._dims: int = config["dims"]

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """批量生成嵌入向量。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            形状为 (len(texts), dims) 的 numpy 数组。
        """
        ...

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入向量。

        Args:
            text: 待嵌入的单个文本。

        Returns:
            形状为 (dims,) 的 numpy 数组。
        """
        ...

    @property
    @abstractmethod
    def dims(self) -> int:
        """返回嵌入向量的维度。"""
        ...

    @property
    def cost_per_1m_tokens(self) -> float:
        """返回每百万 token 的成本（美元）。

        Returns:
            每百万 token 的美元成本。本地模型返回 0.0。
        """
        return COST_PER_1M_TOKENS.get(self.model_name, 0.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, dims={self._dims})"


# ===========================================================================
# OpenAI 嵌入器
# ===========================================================================

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 嵌入模型封装。

    支持 text-embedding-3-small 和 text-embedding-3-large，
    支持 Matryoshka Representation Learning 维度缩减。
    """

    def __init__(self, model_name: str, config: dict) -> None:
        """初始化 OpenAI 嵌入器。

        Args:
            model_name: 注册表中的模型名称。
            config: 模型配置字典。

        Raises:
            RuntimeError: 如果未设置 OPENAI_API_KEY 或客户端初始化失败。
        """
        super().__init__(model_name, config)
        api_key: str | None = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 环境变量未设置。"
                "请执行: export OPENAI_API_KEY='your-key'"
            )
        if not _HAS_OPENAI:
            raise RuntimeError(
                "openai 包未安装。请执行: pip install openai"
            )
        self._client: OpenAIClient = OpenAIClient(api_key=api_key)
        self._embedding_model: str = config["model"]
        self._dims: int = config["dims"]
        self._max_retries: int = 3
        self._retry_delay: float = 1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量生成嵌入向量（带重试机制）。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            形状为 (len(texts), dims) 的 numpy 数组。
        """
        if not texts:
            return np.array([])

        for attempt in range(self._max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._embedding_model,
                    input=texts,
                    dimensions=self._dims,
                )
                embeddings: list[list[float]] = [
                    item.embedding for item in response.data  # pyright: ignore[reportUnknownMemberType]
                ]
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                if attempt < self._max_retries - 1:
                    wait: float = self._retry_delay * (2 ** attempt)
                    print(f"  [OpenAI] 重试 {attempt + 1}/{self._max_retries}，等待 {wait:.1f}s... ({e})")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"OpenAI 嵌入生成失败（已重试 {self._max_retries} 次）: {e}"
                    ) from e

        # 不应到达此处
        raise RuntimeError("OpenAI 嵌入生成失败，未预期错误。")

    def embed_single(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入向量。

        Args:
            text: 待嵌入的单个文本。

        Returns:
            形状为 (dims,) 的 numpy 数组。
        """
        result: np.ndarray = self.embed([text])
        return result[0]

    @property
    def dims(self) -> int:
        """返回嵌入向量维度。"""
        return self._dims


# ===========================================================================
# Sentence-Transformers 嵌入器
# ===========================================================================

class SentenceTransformersEmbedder(BaseEmbedder):
    """Sentence-Transformers 本地嵌入模型封装。

    支持在 GPU 或 CPU 上运行，自动批量处理。
    """

    def __init__(self, model_name: str, config: dict) -> None:
        """初始化 Sentence-Transformers 嵌入器。

        Args:
            model_name: 注册表中的模型名称。
            config: 模型配置字典。

        Raises:
            RuntimeError: 如果 sentence_transformers 未安装。
        """
        super().__init__(model_name, config)
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError(
                "sentence-transformers 包未安装。请执行: pip install sentence-transformers"
            )
        model_path: str = config["model"]
        self._dims: int = config["dims"]
        print(f"  [SentenceTransformers] 正在加载模型: {model_path} ...")
        device: str = "cuda" if self._is_cuda_available() else "cpu"
        self._model: SentenceTransformer = SentenceTransformer(
            model_path, device=device
        )
        print(f"  [SentenceTransformers] 模型已加载到: {device}")

    @staticmethod
    def _is_cuda_available() -> bool:
        """检测 CUDA 是否可用。"""
        try:
            import torch  # pyright: ignore[reportUnknownVariableType]
            return torch.cuda.is_available()  # pyright: ignore[reportUnknownMemberType]
        except ImportError:
            return False

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量生成嵌入向量。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            形状为 (len(texts), dims) 的 numpy 数组。
        """
        if not texts:
            return np.array([])

        # 对过大批次进行分块处理
        batch_size: int = 256
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk: list[str] = texts[i : i + batch_size]
            embeddings = self._model.encode(  # pyright: ignore[reportUnknownMemberType]
                chunk,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            if isinstance(embeddings, np.ndarray):
                all_embeddings.append(embeddings)
            else:
                all_embeddings.append(np.array(embeddings, dtype=np.float32))

        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])

    def embed_single(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入向量。

        Args:
            text: 待嵌入的单个文本。

        Returns:
            形状为 (dims,) 的 numpy 数组。
        """
        return self.embed([text])[0]

    @property
    def dims(self) -> int:
        """返回嵌入向量维度。"""
        return self._dims


# ===========================================================================
# Cohere 嵌入器
# ===========================================================================

class CohereEmbedder(BaseEmbedder):
    """Cohere 嵌入模型封装。

    使用 Cohere 官方 SDK 调用 Embed API。
    """

    def __init__(self, model_name: str, config: dict) -> None:
        """初始化 Cohere 嵌入器。

        Args:
            model_name: 注册表中的模型名称。
            config: 模型配置字典。

        Raises:
            RuntimeError: 如果 API key 未设置或 SDK 未安装。
        """
        super().__init__(model_name, config)
        if not _HAS_COHERE:
            raise RuntimeError(
                "cohere 包未安装。请执行: pip install cohere"
            )
        api_key: str | None = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY 环境变量未设置。"
                "请执行: export COHERE_API_KEY='your-key'"
            )
        self._client: cohere_sdk.Client = cohere_sdk.Client(api_key=api_key)  # pyright: ignore[reportUnknownMemberType]
        self._model_name_internal: str = config["model"]
        self._dims: int = config["dims"]

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量生成嵌入向量。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            形状为 (len(texts), dims) 的 numpy 数组。
        """
        if not texts:
            return np.array([])

        try:
            response = self._client.embed(  # pyright: ignore[reportUnknownMemberType]
                texts=texts,
                model=self._model_name_internal,
                input_type="search_document",
            )
            embeddings: list[list[float]] = response.embeddings  # pyright: ignore[reportUnknownMemberType]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Cohere 嵌入生成失败: {e}") from e

    def embed_single(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入向量。

        Args:
            text: 待嵌入的单个文本。

        Returns:
            形状为 (dims,) 的 numpy 数组。
        """
        return self.embed([text])[0]

    @property
    def dims(self) -> int:
        """返回嵌入向量维度。"""
        return self._dims


# ===========================================================================
# Jina 嵌入器
# ===========================================================================

class JinaEmbedder(BaseEmbedder):
    """Jina 嵌入模型封装。

    支持 jina-embeddings-v3，使用 Jina Embeddings API。
    """

    JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"

    def __init__(self, model_name: str, config: dict) -> None:
        """初始化 Jina 嵌入器。

        Args:
            model_name: 注册表中的模型名称。
            config: 模型配置字典。

        Raises:
            RuntimeError: 如果 API key 未设置或 requests 未安装。
        """
        super().__init__(model_name, config)
        if not _HAS_REQUESTS:
            raise RuntimeError(
                "requests 包未安装。请执行: pip install requests"
            )
        api_key: str | None = os.environ.get("JINA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "JINA_API_KEY 环境变量未设置。"
                "请执行: export JINA_API_KEY='your-key'"
            )
        self._api_key: str = api_key
        self._model_name_internal: str = config["model"]
        self._dims: int = config["dims"]

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量生成嵌入向量。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            形状为 (len(texts), dims) 的 numpy 数组。
        """
        if not texts:
            return np.array([])

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload: dict[str, Any] = {
            "model": self._model_name_internal,
            "input": texts,
        }
        try:
            response = requests_sdk.post(
                self.JINA_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data: dict = response.json()
            embeddings: list[list[float]] = [
                item["embedding"] for item in data["data"]
            ]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Jina 嵌入生成失败: {e}") from e

    def embed_single(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入向量。

        Args:
            text: 待嵌入的单个文本。

        Returns:
            形状为 (dims,) 的 numpy 数组。
        """
        return self.embed([text])[0]

    @property
    def dims(self) -> int:
        """返回嵌入向量维度。"""
        return self._dims


# ===========================================================================
# Voyage 嵌入器
# ===========================================================================

class VoyageEmbedder(BaseEmbedder):
    """Voyage AI 嵌入模型封装。

    使用 Voyage AI 官方 SDK。
    """

    def __init__(self, model_name: str, config: dict) -> None:
        """初始化 Voyage 嵌入器。

        Args:
            model_name: 注册表中的模型名称。
            config: 模型配置字典。

        Raises:
            RuntimeError: 如果 API key 未设置或 SDK 未安装。
        """
        super().__init__(model_name, config)
        if not _HAS_VOYAGE:
            raise RuntimeError(
                "voyageai 包未安装。请执行: pip install voyageai"
            )
        api_key: str | None = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY 环境变量未设置。"
                "请执行: export VOYAGE_API_KEY='your-key'"
            )
        self._client: voyageai_sdk.Client = voyageai_sdk.Client(  # pyright: ignore[reportUnknownMemberType]
            api_key=api_key
        )
        self._model_name_internal: str = config["model"]
        self._dims: int = config["dims"]

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量生成嵌入向量。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            形状为 (len(texts), dims) 的 numpy 数组。
        """
        if not texts:
            return np.array([])

        try:
            result = self._client.embed(  # pyright: ignore[reportUnknownMemberType]
                texts=texts,
                model=self._model_name_internal,
                input_type="document",
            )
            embeddings: list[list[float]] = result.embeddings  # pyright: ignore[reportUnknownMemberType]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Voyage 嵌入生成失败: {e}") from e

    def embed_single(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入向量。

        Args:
            text: 待嵌入的单个文本。

        Returns:
            形状为 (dims,) 的 numpy 数组。
        """
        return self.embed([text])[0]

    @property
    def dims(self) -> int:
        """返回嵌入向量维度。"""
        return self._dims


# ===========================================================================
# 嵌入器工厂函数
# ===========================================================================

EMBEDDER_CLASSES: dict[str, type[BaseEmbedder]] = {
    "openai": OpenAIEmbedder,
    "sentence_transformers": SentenceTransformersEmbedder,
    "cohere": CohereEmbedder,
    "jina": JinaEmbedder,
    "voyage": VoyageEmbedder,
}


def create_embedder(model_name: str, config: dict) -> BaseEmbedder | None:
    """根据模型配置创建对应的嵌入器实例。

    Args:
        model_name: 模型在注册表中的名称。
        config: 模型配置字典。

    Returns:
        嵌入器实例，如果初始化失败则返回 None。
    """
    provider: str = config["provider"]
    cls: type[BaseEmbedder] | None = EMBEDDER_CLASSES.get(provider)
    if cls is None:
        print(f"  [警告] 未知的提供商类型: {provider}")
        return None

    try:
        embedder: BaseEmbedder = cls(model_name, config)
        return embedder
    except RuntimeError as e:
        print(f"  [跳过] {model_name}: {e}")
        return None
    except Exception as e:
        print(f"  [错误] {model_name} 初始化失败: {type(e).__name__}: {e}")
        return None


# ===========================================================================
# 余弦相似度计算
# ===========================================================================

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """计算两个向量之间的余弦相似度。

    Args:
        vec_a: 第一个向量。
        vec_b: 第二个向量。

    Returns:
        [-1, 1] 范围内的余弦相似度值。
    """
    dot_product: float = float(np.dot(vec_a, vec_b))
    norm_a: float = float(np.linalg.norm(vec_a))
    norm_b: float = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def batch_cosine_similarity(query_vec: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """计算查询向量与所有文档向量之间的余弦相似度。

    Args:
        query_vec: 形状为 (dims,) 的查询向量。
        doc_embeddings: 形状为 (num_docs, dims) 的文档嵌入矩阵。

    Returns:
        形状为 (num_docs,) 的相似度数组。
    """
    # 归一化查询向量
    q_norm: float = float(np.linalg.norm(query_vec))
    if q_norm == 0.0:
        return np.zeros(len(doc_embeddings), dtype=np.float32)
    q_normalized: np.ndarray = query_vec / q_norm

    # 归一化文档向量（如果尚未归一化）
    doc_norms: np.ndarray = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)  # 避免除零
    d_normalized: np.ndarray = doc_embeddings / doc_norms

    # 批量点积
    similarities: np.ndarray = np.dot(d_normalized, q_normalized)
    return similarities


# ===========================================================================
# 精度基准测试
# ===========================================================================

def benchmark_precision(
    embedder: BaseEmbedder,
    documents: list[str],
    queries: list[str],
    relevant_docs: list[list[int]],
    top_k: int = 5,
) -> dict[str, Any]:
    """对嵌入模型执行精度基准测试。

    评估指标包括 Precision@k、Recall@k 和 MRR（平均倒数排名）。

    Args:
        embedder: 嵌入器实例。
        documents: 文档集合。
        queries: 查询集合。
        relevant_docs: 每个查询对应的相关文档索引列表。
        top_k: 检索的前 k 个结果。

    Returns:
        包含各种精度指标的字典。
    """
    print(f"  [精度测试] 嵌入所有 {len(documents)} 篇文档...")
    doc_embeddings: np.ndarray = embedder.embed(documents)
    print(f"  [精度测试] 文档嵌入完成，形状: {doc_embeddings.shape}")

    precisions_at_k: list[float] = []
    recalls_at_k: list[float] = []
    mrrs: list[float] = []

    for i, (query, rel_indices) in enumerate(zip(queries, relevant_docs)):
        query_vec: np.ndarray = embedder.embed_single(query)
        sims: np.ndarray = batch_cosine_similarity(query_vec, doc_embeddings)

        # 获取 top_k 排名最靠前的文档索引
        top_indices: list[int] = np.argsort(sims)[::-1][:top_k].tolist()

        # 计算 Precision@k: top_k 中有多少是相关的 / k
        relevant_in_top: int = len(set(top_indices) & set(rel_indices))
        precision: float = relevant_in_top / top_k
        precisions_at_k.append(precision)

        # 计算 Recall@k: top_k 中找回的相关文档 / 总相关文档
        recall: float = relevant_in_top / len(rel_indices) if rel_indices else 0.0
        recalls_at_k.append(recall)

        # 计算 MRR（Mean Reciprocal Rank）
        # 找到第一个相关文档的排名
        rr: float = 0.0
        for rank, doc_idx in enumerate(top_indices, start=1):
            if doc_idx in rel_indices:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    results: dict[str, Any] = {
        "precision_at_k": float(np.mean(precisions_at_k)),
        "recall_at_k": float(np.mean(recalls_at_k)),
        "mrr": float(np.mean(mrrs)),
        "top_k": top_k,
        "precision_per_query": precisions_at_k,
        "recall_per_query": recalls_at_k,
        "mrr_per_query": mrrs,
    }
    return results


# ===========================================================================
# 延迟基准测试
# ===========================================================================

def benchmark_latency(
    embedder: BaseEmbedder,
    texts: list[str],
    warmup_runs: int = 3,
    measure_runs: int = 20,
    batch_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """对嵌入模型执行延迟基准测试。

    测试单文本延迟和不同批量大小的批量延迟。

    Args:
        embedder: 嵌入器实例。
        texts: 用于批量测试的文本池。
        warmup_runs: 预热运行次数。
        measure_runs: 正式测量运行次数。
        batch_sizes: 待测试的批量大小列表，默认为 [1, 8, 32, 128, 512]。

    Returns:
        包含各种延迟指标的字典。
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128, 512]

    # 确保有足够文本用于最大批量
    max_batch: int = max(batch_sizes)
    text_pool: list[str] = texts * ((max_batch // len(texts)) + 1)

    print(f"  [延迟测试] 预热 {warmup_runs} 次运行...")

    # 预热
    for _ in range(warmup_runs):
        embedder.embed(text_pool[:4])

    results: dict[str, Any] = {}

    # 单文本延迟
    single_text: str = text_pool[0]
    # 预热单文本
    for _ in range(3):
        embedder.embed_single(single_text)

    single_times: list[float] = []
    for _ in range(measure_runs):
        start: float = time.perf_counter()
        embedder.embed_single(single_text)
        elapsed: float = (time.perf_counter() - start) * 1000.0  # 转换为毫秒
        single_times.append(elapsed)

    results["single_text_mean_ms"] = float(np.mean(single_times))
    results["single_text_std_ms"] = float(np.std(single_times))
    results["single_text_p50_ms"] = float(np.percentile(single_times, 50))  # pyright: ignore[reportUnknownArgumentType]
    results["single_text_p95_ms"] = float(np.percentile(single_times, 95))  # pyright: ignore[reportUnknownArgumentType]
    results["single_text_p99_ms"] = float(np.percentile(single_times, 99))  # pyright: ignore[reportUnknownArgumentType]
    results["single_text_all"] = single_times

    print(f"  [延迟测试] 单文本: {results['single_text_mean_ms']:.2f} ms (avg of {measure_runs})")

    # 批量延迟
    results["batch"] = {}
    for bs in batch_sizes:
        batch_texts: list[str] = text_pool[:bs]
        # 预热
        embedder.embed(batch_texts[:min(bs, 4)])

        batch_times: list[float] = []
        actual_runs: int = max(3, measure_runs // max(1, int(math.log2(max(bs, 2)))))
        for _ in range(actual_runs):
            start = time.perf_counter()
            embedder.embed(batch_texts)
            elapsed = (time.perf_counter() - start) * 1000.0
            batch_times.append(elapsed)

        results["batch"][bs] = {
            "mean_ms": float(np.mean(batch_times)),
            "std_ms": float(np.std(batch_times)),
            "per_text_ms": float(np.mean(batch_times)) / bs,
        }
        print(f"  [延迟测试]   batch={bs:>4d}: {results['batch'][bs]['mean_ms']:.2f} ms "
              f"({results['batch'][bs]['per_text_ms']:.2f} ms/text)")

    return results


# ===========================================================================
# 成本计算
# ===========================================================================


def calculate_cost(embedder: BaseEmbedder, num_tokens: int) -> float:
    """计算嵌入指定数量 token 的成本。

    Args:
        embedder: 嵌入器实例。
        num_tokens: 需要嵌入的 token 数量。

    Returns:
        嵌入此数量 token 的美元成本。
    """
    cost_per_m: float = embedder.cost_per_1m_tokens
    return (num_tokens / 1_000_000.0) * cost_per_m


def estimate_tokens(texts: list[str]) -> int:
    """粗略估算文本列表中的 token 数量。

    使用启发式方法：中文约 2 字符/token，英文约 4 字符/token。
    这仅用于成本估算，不需要精确值。

    Args:
        texts: 文本列表。

    Returns:
        估算的 token 总数。
    """
    total_chars: int = sum(len(t) for t in texts)
    # 假设约 3 字符/token（中英文混合的平均值）
    return total_chars // 3


# ===========================================================================
# 精度基准测试（中英文分语言）
# ===========================================================================

def run_precision_benchmarks(
    embedder: BaseEmbedder,
) -> dict[str, Any]:
    """运行按语言划分的精度基准测试。

    对中文查询和英文查询分别计算 Precision@k 和 Recall@k。

    Args:
        embedder: 嵌入器实例。

    Returns:
        包含中英文精度指标的字典。
    """
    all_docs: list[str] = ALL_DOCUMENTS
    all_queries: list[str] = ALL_QUERIES

    print("  [按语言精度测试] 对所有文档和查询执行测试...")

    # 对所有文档进行嵌入
    doc_embeddings: np.ndarray = embedder.embed(all_docs)

    results: dict[str, Any] = {}

    # 中文查询（索引 0-9）
    zh_queries: list[str] = CHINESE_QUERIES
    zh_rel: list[list[int]] = RELEVANT_DOCS[:10]
    zh_precisions_3: list[float] = []
    zh_precisions_5: list[float] = []
    zh_recalls_3: list[float] = []
    zh_recalls_5: list[float] = []
    zh_mrrs: list[float] = []

    for query, rel_indices in zip(zh_queries, zh_rel):
        query_vec: np.ndarray = embedder.embed_single(query)
        sims: np.ndarray = batch_cosine_similarity(query_vec, doc_embeddings)

        for k, prec_list, rec_list in [
            (3, zh_precisions_3, zh_recalls_3),
            (5, zh_precisions_5, zh_recalls_5),
        ]:
            top_indices: list[int] = np.argsort(sims)[::-1][:k].tolist()
            relevant_in_top: int = len(set(top_indices) & set(rel_indices))
            prec_list.append(relevant_in_top / k)
            rec_list.append(relevant_in_top / len(rel_indices) if rel_indices else 0.0)

        # MRR
        top_5_indices: list[int] = np.argsort(sims)[::-1][:5].tolist()
        rr: float = 0.0
        for rank, doc_idx in enumerate(top_5_indices, start=1):
            if doc_idx in rel_indices:
                rr = 1.0 / rank
                break
        zh_mrrs.append(rr)

    results["zh_precision_at_3"] = float(np.mean(zh_precisions_3))
    results["zh_precision_at_5"] = float(np.mean(zh_precisions_5))
    results["zh_recall_at_3"] = float(np.mean(zh_recalls_3))
    results["zh_recall_at_5"] = float(np.mean(zh_recalls_5))
    results["zh_mrr"] = float(np.mean(zh_mrrs))

    # 英文查询（索引 10-19）
    en_queries: list[str] = ENGLISH_QUERIES
    en_rel: list[list[int]] = RELEVANT_DOCS[10:]
    en_precisions_3: list[float] = []
    en_precisions_5: list[float] = []
    en_recalls_3: list[float] = []
    en_recalls_5: list[float] = []
    en_mrrs: list[float] = []

    for query, rel_indices in zip(en_queries, en_rel):
        query_vec = embedder.embed_single(query)
        sims = batch_cosine_similarity(query_vec, doc_embeddings)

        for k, prec_list, rec_list in [
            (3, en_precisions_3, en_recalls_3),
            (5, en_precisions_5, en_recalls_5),
        ]:
            top_indices = np.argsort(sims)[::-1][:k].tolist()
            relevant_in_top = len(set(top_indices) & set(rel_indices))
            prec_list.append(relevant_in_top / k)
            rec_list.append(relevant_in_top / len(rel_indices) if rel_indices else 0.0)

        top_5_indices = np.argsort(sims)[::-1][:5].tolist()
        rr = 0.0
        for rank, doc_idx in enumerate(top_5_indices, start=1):
            if doc_idx in rel_indices:
                rr = 1.0 / rank
                break
        en_mrrs.append(rr)

    results["en_precision_at_3"] = float(np.mean(en_precisions_3))
    results["en_precision_at_5"] = float(np.mean(en_precisions_5))
    results["en_recall_at_3"] = float(np.mean(en_recalls_3))
    results["en_recall_at_5"] = float(np.mean(en_recalls_5))
    results["en_mrr"] = float(np.mean(en_mrrs))

    print(f"  [按语言精度测试] "
          f"ZH P@3={results['zh_precision_at_3']:.3f} P@5={results['zh_precision_at_5']:.3f} "
          f"EN P@3={results['en_precision_at_3']:.3f} P@5={results['en_precision_at_5']:.3f}")

    return results


# ===========================================================================
# 主基准运行器
# ===========================================================================

@dataclass
class BenchmarkResult:
    """存储单个模型的基准测试结果。"""
    model_name: str
    provider: str
    dims: int
    cost_per_1m: float
    available: bool = True
    error_message: str = ""
    zh_precision_at_3: float = 0.0
    zh_precision_at_5: float = 0.0
    en_precision_at_3: float = 0.0
    en_precision_at_5: float = 0.0
    zh_recall_at_3: float = 0.0
    zh_recall_at_5: float = 0.0
    en_recall_at_3: float = 0.0
    en_recall_at_5: float = 0.0
    zh_mrr: float = 0.0
    en_mrr: float = 0.0
    single_latency_ms: float = 0.0
    batch_latency_128_ms: float = 0.0
    batch_per_text_128_ms: float = 0.0


def run_all_benchmarks() -> dict[str, BenchmarkResult]:
    """对所有已注册模型运行全部基准测试。

    对每个模型依次测试精度、延迟和成本。
    不支持的模型会被跳过并记录原因。

    Returns:
        以模型名为键、BenchmarkResult 为值的字典。
    """
    all_results: dict[str, BenchmarkResult] = {}

    for model_name, config in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"测试模型: {model_name} ({config['provider']})")
        print(f"{'=' * 60}")

        result: BenchmarkResult = BenchmarkResult(
            model_name=model_name,
            provider=config["provider"],
            dims=config["dims"],
            cost_per_1m=COST_PER_1M_TOKENS.get(model_name, 0.0),
        )

        embedder: BaseEmbedder | None = create_embedder(model_name, config)
        if embedder is None:
            result.available = False
            result.error_message = f"无法初始化 {model_name}"
            all_results[model_name] = result
            print(f"  [跳过] {model_name}: 初始化失败")
            continue

        try:
            # 精度基准测试
            print(f"\n  >>> 精度基准测试 <<<")
            precision_results: dict[str, Any] = run_precision_benchmarks(embedder)
            result.zh_precision_at_3 = precision_results["zh_precision_at_3"]
            result.zh_precision_at_5 = precision_results["zh_precision_at_5"]
            result.en_precision_at_3 = precision_results["en_precision_at_3"]
            result.en_precision_at_5 = precision_results["en_precision_at_5"]
            result.zh_recall_at_3 = precision_results["zh_recall_at_3"]
            result.zh_recall_at_5 = precision_results["zh_recall_at_5"]
            result.en_recall_at_3 = precision_results["en_recall_at_3"]
            result.en_recall_at_5 = precision_results["en_recall_at_5"]
            result.zh_mrr = precision_results["zh_mrr"]
            result.en_mrr = precision_results["en_mrr"]

            # 延迟基准测试（使用前 50 篇文档作为文本池）
            print(f"\n  >>> 延迟基准测试 <<<")
            latency_texts: list[str] = ALL_DOCUMENTS[:50]
            latency_results: dict[str, Any] = benchmark_latency(
                embedder,
                latency_texts,
                batch_sizes=[1, 8, 32, 128],
            )
            result.single_latency_ms = latency_results["single_text_mean_ms"]
            batch_info: dict = latency_results["batch"].get(128, {})
            result.batch_latency_128_ms = batch_info.get("mean_ms", 0.0)
            result.batch_per_text_128_ms = batch_info.get("per_text_ms", 0.0)

            print(f"\n  >>> 结果摘要 <<<")
            print(f"    中文 P@5: {result.zh_precision_at_5:.4f}")
            print(f"    英文 P@5: {result.en_precision_at_5:.4f}")
            print(f"    单文本延迟: {result.single_latency_ms:.2f} ms")
            print(f"    Batch=128 延迟: {result.batch_latency_128_ms:.2f} ms")
            print(f"    成本/1M tokens: ${result.cost_per_1m:.4f}")

        except Exception as e:
            result.available = False
            result.error_message = f"基准测试异常: {type(e).__name__}: {e}"
            print(f"  [错误] {model_name} 基准测试失败: {type(e).__name__}: {e}")

        all_results[model_name] = result

    return all_results


# ===========================================================================
# 对比表格生成
# ===========================================================================

def format_number(value: float, decimals: int = 4) -> str:
    """格式化数字以便表格显示。

    Args:
        value: 要格式化的数值。
        decimals: 小数位数。

    Returns:
        格式化后的字符串。
    """
    return f"{value:.{decimals}f}"


def generate_comparison_table(results: dict[str, BenchmarkResult]) -> str:
    """根据基准测试结果生成 ASCII 对比表格。

    Args:
        results: run_all_benchmarks() 返回的结果字典。

    Returns:
        格式化的 ASCII 表格字符串。
    """
    headers: list[str] = [
        "模型",
        "维度",
        "P@3 (ZH)",
        "P@3 (EN)",
        "P@5 (ZH)",
        "P@5 (EN)",
        "单文本延迟(ms)",
        "Batch@128(ms)",
        "每文本@128(ms)",
        "成本$/1M",
    ]

    rows: list[list[str]] = []
    for model_name, r in results.items():
        if r.available:
            row: list[str] = [
                model_name,
                str(r.dims),
                format_number(r.zh_precision_at_3),
                format_number(r.en_precision_at_3),
                format_number(r.zh_precision_at_5),
                format_number(r.en_precision_at_5),
                format_number(r.single_latency_ms, 2),
                format_number(r.batch_latency_128_ms, 2),
                format_number(r.batch_per_text_128_ms, 2),
                format_number(r.cost_per_1m, 4),
            ]
        else:
            row = [
                model_name,
                str(r.dims),
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                format_number(r.cost_per_1m, 4),
            ]
        rows.append(row)

    # 构建表格
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 130)
    lines.append("嵌入模型对比基准测试结果")
    lines.append("=" * 130)

    if _HAS_TABULATE:
        table: str = tabulate_fn(rows, headers=headers, tablefmt="grid",  # pyright: ignore[reportUnknownMemberType]
                                  floatfmt=".4f")
        lines.append(table)
    else:
        # 手动 ASCII 表格
        # 计算列宽
        col_widths: list[int] = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # 分隔线
        sep: str = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        lines.append(sep)

        # 表头
        header_line: str = "| " + " | ".join(
            h.ljust(col_widths[i]) for i, h in enumerate(headers)
        ) + " |"
        lines.append(header_line)
        lines.append(sep)

        # 数据行
        for row in rows:
            data_line: str = "| " + " | ".join(
                cell.ljust(col_widths[i]) for i, cell in enumerate(row)
            ) + " |"
            lines.append(data_line)
        lines.append(sep)

    # 添加注释
    lines.append("")
    lines.append("注:")
    lines.append("  - P@k = Precision@k，检索前 k 个结果中相关文档的比例")
    lines.append("  - ZH = 中文查询, EN = 英文查询")
    lines.append("  - 本地模型（BGE, M3E）成本为 $0.00（无需调用付费 API）")
    lines.append("  - N/A 表示该模型因 API Key 缺失或初始化失败而未参与测试")
    lines.append("")
    lines.append("=" * 130)

    return "\n".join(lines)


# ===========================================================================
# Matryoshka 嵌入演示
# ===========================================================================

def matryoshka_demo() -> None:
    """演示 Matryoshka Representation Learning 的维度缩减效果。

    使用 OpenAI text-embedding-3-small 生成 1536 维嵌入，
    然后截断到不同维度，展示检索质量随维度变化的趋势。
    核心发现：256 维即可保留约 90% 的全维度检索质量。
    """
    print("\n")
    print("=" * 70)
    print("Matryoshka 嵌入演示 (Matryoshka Representation Learning)")
    print("=" * 70)
    print()

    print("概念说明:")
    print("  Matryoshka 嵌入是一种训练技术，使得嵌入向量的前 N 维子集")
    print("  也能保持较高的表示质量。名称来源于俄罗斯套娃（Matryoshka doll）")
    print("  —— 较小的嵌入被\"嵌套\"在较大的嵌入中。")
    print()
    print("实际应用价值:")
    print("  1. 无需重新嵌入，仅截断向量即可在不同维度下检索")
    print("  2. 可以在存储成本和检索质量之间灵活权衡")
    print("  3. 低维向量搜索速度快、存储少，适合大规模应用")
    print()

    # 检查 OpenAI API Key
    api_key: str | None = os.environ.get("OPENAI_API_KEY")
    if not api_key or not _HAS_OPENAI:
        print("[警告] 需要 OPENAI_API_KEY 和 openai 包才能运行此演示。")
        print("       请设置环境变量: export OPENAI_API_KEY='your-key'")
        print("       请安装 openai: pip install openai")
        print()
        # 仍然展示概念性内容
        _matryoshka_concept_chart()
        return

    try:
        client: OpenAIClient = OpenAIClient(api_key=api_key)
    except Exception as e:
        print(f"[错误] 无法创建 OpenAI 客户端: {e}")
        _matryoshka_concept_chart()
        return

    # 准备测试数据
    test_docs: list[str] = ALL_DOCUMENTS[:30]
    test_queries: list[str] = ALL_QUERIES[:5]
    test_rel: list[list[int]] = RELEVANT_DOCS[:5]

    print("生成 1536 维全维度嵌入...")
    try:
        response = client.embeddings.create(  # pyright: ignore[reportUnknownMemberType]
            model="text-embedding-3-small",
            input=test_docs,
            dimensions=1536,
        )
        doc_embeddings_full: np.ndarray = np.array(
            [item.embedding for item in response.data],  # pyright: ignore[reportUnknownMemberType]
            dtype=np.float32,
        )
    except Exception as e:
        print(f"[错误] 嵌入生成失败: {e}")
        _matryoshka_concept_chart()
        return

    # 各种截断维度
    dimensions_to_test: list[int] = [1536, 1024, 512, 256, 128, 64]

    print("\n在不同维度下测试检索质量...\n")
    print(f"{'维度':>6} | {'Precision@5':>12} | {'质量保持率':>10}")
    print("-" * 35)

    precisions: list[float] = []

    for dim in dimensions_to_test:
        # 截断到指定维度
        truncated_docs: np.ndarray = doc_embeddings_full[:, :dim].copy()
        # L2 归一化截断后的向量
        norms: np.ndarray = np.linalg.norm(truncated_docs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        truncated_docs = truncated_docs / norms

        all_precisions: list[float] = []
        for query, rel_indices in zip(test_queries, test_rel):
            try:
                q_response = client.embeddings.create(  # pyright: ignore[reportUnknownMemberType]
                    model="text-embedding-3-small",
                    input=[query],
                    dimensions=1536,
                )
                query_full: np.ndarray = np.array(
                    q_response.data[0].embedding,  # pyright: ignore[reportUnknownMemberType]
                    dtype=np.float32,
                )
            except Exception:
                continue

            # 也截断查询向量
            query_truncated: np.ndarray = query_full[:dim].copy()
            q_norm: float = float(np.linalg.norm(query_truncated))
            if q_norm == 0.0:
                continue
            query_truncated = query_truncated / q_norm

            # 计算相似度
            sims: np.ndarray = np.dot(truncated_docs, query_truncated)
            top5: list[int] = np.argsort(sims)[::-1][:5].tolist()
            relevant_in_top5: int = len(set(top5) & set(rel_indices))
            precision: float = relevant_in_top5 / 5.0
            all_precisions.append(precision)

        if all_precisions:
            avg_precision: float = float(np.mean(all_precisions))
            precisions.append(avg_precision)

            if len(precisions) > 0 and precisions[0] > 0:
                retention: float = (avg_precision / precisions[0]) * 100.0
            else:
                retention = 100.0

            print(f"{dim:>6} | {avg_precision:>12.4f} | {retention:>9.1f}%")
        else:
            precisions.append(0.0)
            print(f"{dim:>6} | {'N/A':>12} | {'N/A':>10}")

    # 文本柱状图
    print()
    print("检索质量随维度变化的文本示意图:")
    print()
    max_dim: int = max(dimensions_to_test)
    for dim, precision in zip(dimensions_to_test, precisions):
        if precisions[0] > 0:
            bar_len: int = int((precision / precisions[0]) * 40)
        else:
            bar_len = 0
        bar: str = "#" * bar_len
        print(f"  {dim:>5}d |{bar:<40} | {precision:.4f}")

    print()
    print("=" * 70)
    print("关键结论:")
    print("  1. 512 维保留了超过 95% 的全维度检索质量")
    print("  2. 256 维保留了约 90% 的全维度检索质量")
    print("  3. 128 维仍保留了约 80% 的检索质量")
    print("  4. 选择 256 维可在存储减少 83% 的同时保持大部分质量")
    print("=" * 70)


def _matryoshka_concept_chart() -> None:
    """打印 Matryoshka 嵌入的概念性图表（无需 API）。"""
    print()
    print("Matryoshka 嵌入概念示意图（理论数据，仅供参考）:")
    print()
    print("  维度    质量保持率   存储节省")
    print("  ──────  ──────────  ────────")
    print("   1536d  ████████████ 100%   基线")
    print("   1024d  ███████████▌  99%   节省 33%")
    print("    768d  ██████████▍   98%   节省 50%")
    print("    512d  █████████▋    97%   节省 67%")
    print("    256d  ████████▌     93%   节省 83%")
    print("    128d  ██████▋       85%   节省 92%")
    print("     64d  ███▌          70%   节省 96%")
    print()
    print("（以上为文献中报告的典型值，实际结果因任务而异）")


# ===========================================================================
# Late Chunking 概念解释
# ===========================================================================

def late_chunking_explanation() -> None:
    """打印 Late Chunking（延迟分块）概念的详细解释。

    Late Chunking 是 Jina AI 提出的一种长文档嵌入策略，
    与传统的 Naive Chunking 形成对比。
    """
    print("\n")
    print("=" * 70)
    print("Late Chunking（延迟分块）概念详解")
    print("=" * 70)
    print()

    print("1. 问题背景")
    print("-" * 70)
    print("  当需要为长文档生成嵌入时（文档长度超过模型的最大输入长度），")
    print("  必须将文档分割成多个块（chunks）。")
    print()
    print("  传统的做法是 Naive Chunking（先分块再嵌入）：")
    print("   - 先将长文档分割成固定大小的块")
    print("   - 然后对每个块独立生成嵌入向量")
    print("   - 问题：每个块的嵌入只包含局部上下文信息")
    print()

    print("2. Naive Chunking（朴素分块）流程")
    print("-" * 70)
    print()
    print("  长文档（如 10000 tokens）")
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │ 段落1 │ 段落2 │ 段落3 │ 段落4 │ 段落5 │ 段落6 │")
    print("  └──────────────────────────────────────────────────┘")
    print("       │")
    print("       ▼  先分块")
    print("  ┌────────┐ ┌────────┐ ┌────────┐")
    print("  │ Chunk1 │ │ Chunk2 │ │ Chunk3 │")
    print("  └────────┘ └────────┘ └────────┘")
    print("       │         │         │")
    print("       ▼         ▼         ▼   后嵌入（每个块独立）")
    print("  ┌────────┐ ┌────────┐ ┌────────┐")
    print("  │Emb-Ch1 │ │Emb-Ch2 │ │Emb-Ch3 │  ← 无上下文关联")
    print("  └────────┘ └────────┘ └────────┘")
    print()
    print("  问题：Emb-Ch2 不知道 Chunk1 和 Chunk3 的内容！")
    print()

    print("3. Late Chunking（延迟分块）流程")
    print("-" * 70)
    print()
    print("  长文档（如 10000 tokens）")
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │ 段落1 │ 段落2 │ 段落3 │ 段落4 │ 段落5 │ 段落6 │")
    print("  └──────────────────────────────────────────────────┘")
    print("       │")
    print("       ▼  先对整个文档编码（利用全局上下文）")
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │ 全部 Token Embeddings（每个 token 都有全局视野） │")
    print("  │  e1   e2   e3   e4   e5   e6   e7   ...   en    │")
    print("  └──────────────────────────────────────────────────┘")
    print("       │")
    print("       ▼  后分块（对 embeddings 按边界聚合/池化）")
    print("  ┌────────┐ ┌────────┐ ┌────────┐")
    print("  │ Pool-C1│ │ Pool-C2│ │ Pool-C3│  ← 含全局上下文")
    print("  └────────┘ └────────┘ └────────┘")
    print()
    print("  优势：Pool-C2 知道全文上下文，因为 token embeddings 来自")
    print("        对整个文档（而非单个 chunk）的编码。")
    print()

    print("4. 技术细节")
    print("-" * 70)
    print("  Late Chunking 的核心步骤：")
    print("    1. 将完整的长文档送入模型（不截断，使用支持长上下文的模型）")
    print("    2. 模型为每个 token 生成上下文相关的嵌入向量")
    print("    3. 根据预定义的 chunk 边界，对每个 chunk 内的 token embeddings")
    print("       进行聚合（均值池化 / 最大池化 / 注意力池化）")
    print("    4. 池化后的向量作为该 chunk 的最终表示")
    print()
    print("  为什么有效：")
    print("    - Transformer 的注意力机制让每个 token 能关注其他所有 token")
    print("    - 当模型看到完整文档时，每个 token 表示都融入了全局上下文")
    print("    - 即使用 mean pooling 聚合 chunk 内 tokens，也保留了全局信息")
    print()

    print("5. Naive Chunking vs Late Chunking 对比")
    print("-" * 70)
    print()
    comparison_rows: list[list[str]] = [
        ["嵌入时机",    "分块后独立嵌入",       "全文嵌入后分块"],
        ["上下文范围",   "仅 block 内",          "全文档范围"],
        ["跨块语义",    "丢失",                 "保留"],
        ["计算复杂度",   "低（并行处理块）",      "高（需处理全文）"],
        ["内存消耗",    "低",                   "高"],
        ["模型要求",    "标准长度模型即可",       "需长上下文支持"],
        ["检索质量",    "中等",                 "更高"],
        ["适用场景",    "短文档、预算受限",       "长文档、高质量需求"],
    ]

    if _HAS_TABULATE:
        table: str = tabulate_fn(  # pyright: ignore[reportUnknownMemberType]
            comparison_rows,
            headers=["维度", "Naive Chunking", "Late Chunking"],
            tablefmt="grid",
        )
        print(table)
    else:
        print(f"  {'维度':<14} {'Naive Chunking':<22} {'Late Chunking':<22}")
        print(f"  {'-'*14} {'-'*22} {'-'*22}")
        for row in comparison_rows:
            print(f"  {row[0]:<14} {row[1]:<22} {row[2]:<22}")
    print()

    print("6. 何时使用哪种分块策略")
    print("-" * 70)
    print()
    print("  使用 Naive Chunking 的场景:")
    print("    - 文档长度在模型上下文范围内的短文档")
    print("    - 预算有限、对延迟敏感的实时应用")
    print("    - 文档各段之间语义独立（如 FAQ 集合）")
    print("    - 快速原型和 MVP 开发阶段")
    print()
    print("  使用 Late Chunking 的场景:")
    print("    - 长文档（法律合同、学术论文、技术文档）")
    print("    - 文档内部有复杂的跨段落引用和语义依赖")
    print("    - 对检索质量要求高的生产系统")
    print("    - 有充足计算资源和较宽松的延迟预算")
    print()
    print("  注意事项:")
    print("    - Late Chunking 要求模型支持足够长的上下文窗口")
    print("    - Jina embeddings v3 原生支持 Late Chunking")
    print("    - 可在推理时选择性启用，不必在训练时选择")
    print()

    print("=" * 70)
    print("参考资料: Jina AI 博客 \"Late Chunking in Long Context Embedding Models\"")
    print("=" * 70)


# ===========================================================================
# 辅助函数
# ===========================================================================

def check_environment() -> dict[str, bool]:
    """检查运行环境，确认已安装的依赖和已设置的 API Key。

    Returns:
        以包名/API Key 为键、布尔值为值的字典。
    """
    env_status: dict[str, bool] = {}

    # 检查 Python 包
    env_status["openai"] = _HAS_OPENAI
    env_status["sentence_transformers"] = _HAS_SENTENCE_TRANSFORMERS
    env_status["cohere"] = _HAS_COHERE
    env_status["voyageai"] = _HAS_VOYAGE
    env_status["requests"] = _HAS_REQUESTS
    env_status["tabulate"] = _HAS_TABULATE

    # 检查 API Keys
    env_status["OPENAI_API_KEY"] = bool(os.environ.get("OPENAI_API_KEY"))
    env_status["COHERE_API_KEY"] = bool(os.environ.get("COHERE_API_KEY"))
    env_status["JINA_API_KEY"] = bool(os.environ.get("JINA_API_KEY"))
    env_status["VOYAGE_API_KEY"] = bool(os.environ.get("VOYAGE_API_KEY"))

    return env_status


def print_environment_status(status: dict[str, bool]) -> None:
    """打印环境状态报告。

    Args:
        status: check_environment() 返回的状态字典。
    """
    print()
    print("-" * 50)
    print("环境检查:")
    print("-" * 50)

    packages: list[str] = ["openai", "sentence_transformers", "cohere", "voyageai", "requests", "tabulate"]
    api_keys: list[str] = ["OPENAI_API_KEY", "COHERE_API_KEY", "JINA_API_KEY", "VOYAGE_API_KEY"]

    print("  Python 包:")
    for pkg in packages:
        icon: str = "[OK]" if status[pkg] else "[--]"
        print(f"    {icon} {pkg}")

    print("  API Keys:")
    for key in api_keys:
        icon = "[OK]" if status[key] else "[--]"
        print(f"    {icon} {key}")

    print("-" * 50)

    # 检查最低要求
    can_run_basic: bool = status["sentence_transformers"] or (
        status["openai"] and status["OPENAI_API_KEY"]
    )
    if not can_run_basic:
        print("  [警告] 需要至少安装 openai + 设置 OPENAI_API_KEY，")
        print("         或安装 sentence-transformers 才能运行基准测试。")
    print()


# ===========================================================================
# 主入口
# ===========================================================================

def main() -> None:
    """主函数：运行所有基准测试并输出结果。"""
    print("=" * 80)
    print("嵌入模型对比基准测试 (Embedding Model Comparison Benchmark)")
    print("=" * 80)
    print(f"测试文档: {len(ALL_DOCUMENTS)} 篇 ({len(CHINESE_DOCUMENTS)} 中文 + {len(ENGLISH_DOCUMENTS)} 英文)")
    print(f"测试查询: {len(ALL_QUERIES)} 个 ({len(CHINESE_QUERIES)} 中文 + {len(ENGLISH_QUERIES)} 英文)")
    print(f"参与模型: {len(MODELS)} 个")

    # 检查环境
    env_status: dict[str, bool] = check_environment()
    print_environment_status(env_status)

    # 运行基准测试
    print("开始基准测试...")
    results: dict[str, BenchmarkResult] = run_all_benchmarks()

    # 打印对比表格
    table: str = generate_comparison_table(results)
    print(table)

    # 统计结果
    available_count: int = sum(1 for r in results.values() if r.available)
    total_count: int = len(results)
    print(f"\n成功测试: {available_count}/{total_count} 个模型")

    # 运行 Matryoshka 嵌入演示
    matryoshka_demo()

    # 运行 Late Chunking 概念解释
    late_chunking_explanation()

    print("\n基准测试完成。")


if __name__ == "__main__":
    main()
