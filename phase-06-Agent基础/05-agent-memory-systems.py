"""
Agent Memory Systems - 智能体记忆系统
======================================
实现三种记忆类型：短期记忆、长期记忆、情景记忆
"""

import time
import hashlib
import json
import math
import uuid
from collections import deque, OrderedDict
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# === ShortTermMemory: 短期记忆 ===
# ============================================================
class ShortTermMemory:
    """
    短期记忆 - 滑动窗口 + 令牌预算管理

    - 使用collections.deque实现滑动窗口
    - max_tokens限制(默认8000)
    - 达到80%容量时触发压缩
    - 压缩策略：保留最近N条 + 摘要旧消息
    """

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.messages: deque = deque()
        self.current_tokens: int = 0
        self.compression_threshold = 0.8  # 80%时触发压缩
        self._total_messages_seen: int = 0
        self._compress_count: int = 0

    def add(self, message: dict) -> None:
        """添加消息，自动检查是否需要压缩"""
        self._total_messages_seen += 1

        # 估算新消息的token数
        content = message.get("content", "")
        if isinstance(content, str):
            msg_tokens = self._count_tokens(content)
        elif isinstance(content, list):
            msg_tokens = sum(self._count_tokens(str(item)) for item in content)
        else:
            msg_tokens = self._count_tokens(str(content))

        # 添加元数据
        if "timestamp" not in message:
            message["timestamp"] = time.time()
        if "index" not in message:
            message["index"] = self._total_messages_seen

        self.messages.append(message)
        self.current_tokens += msg_tokens

        # 检查是否需要压缩
        if self.current_tokens > self.max_tokens * self.compression_threshold:
            self._compress()

    def get_context(self, max_messages: int = None) -> list:
        """获取当前上下文窗口"""
        messages_list = list(self.messages)
        if max_messages is not None:
            messages_list = messages_list[-max_messages:]
        return messages_list

    def _count_tokens(self, text: str) -> int:
        """估算token数（字符数/4的简单估算 + 中文字符加权）"""
        if not text:
            return 0

        total = 0
        for char in text:
            if '一' <= char <= '鿿' or '㐀' <= char <= '䶿':
                # 中文字符约占1.5个token
                total += 1.5
            else:
                # 英文/数字大约每4个字符1个token
                total += 0.25

        return max(1, int(total))

    def _compress(self) -> None:
        """压缩记忆：保留最近50%，摘要旧消息"""
        if len(self.messages) < 10:
            return  # 太少不压缩

        self._compress_count += 1
        total = len(self.messages)
        keep_count = max(5, total // 2)

        # 分割：旧消息和新消息
        old_messages = list(self.messages)[:total - keep_count]
        recent_messages = list(self.messages)[total - keep_count:]

        # 摘要旧消息
        summary = self._summarize_messages(old_messages)

        # 重建deque
        self.messages = deque()
        self.current_tokens = 0

        # 添加摘要消息
        summary_msg = {
            "role": "system",
            "content": f"[对话摘要 - 第{self._compress_count}次压缩] {summary}",
            "timestamp": time.time(),
            "is_summary": True,
            "summarized_count": len(old_messages),
        }
        self.add(summary_msg)

        # 添加最近消息
        for msg in recent_messages:
            self.add(msg)

    def _summarize_messages(self, messages: list) -> str:
        """摘要旧消息（简单启发式，生产环境用LLM）"""
        if not messages:
            return "无历史消息"

        # 统计主题关键词
        all_content = []
        roles_count = {}
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                all_content.append(content)
            role = msg.get("role", "unknown")
            roles_count[role] = roles_count.get(role, 0) + 1

        full_text = " ".join(all_content)

        # 简单摘要：提取关键信息
        parts = []

        # 消息统计
        parts.append(f"共{len(messages)}条消息")
        for role, count in roles_count.items():
            parts.append(f"{role}: {count}条")

        # 提取可能的主题（前50个有意义的词）
        words = []
        for word in full_text.split():
            if len(word) >= 2:
                words.append(word)
        if words:
            # 简单TF统计
            word_count = {}
            for w in words[:200]:
                word_count[w] = word_count.get(w, 0) + 1
            top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]
            keywords = ", ".join(w for w, _ in top_words)
            parts.append(f"关键词: {keywords}")

        # 提取开头和结尾
        if all_content:
            first = all_content[0][:80]
            last = all_content[-1][:80]
            parts.append(f"开头: {first}...")
            if len(all_content) > 1:
                parts.append(f"结尾: {last}...")

        return " | ".join(parts)

    def clear(self) -> None:
        """清空记忆"""
        self.messages.clear()
        self.current_tokens = 0
        self._total_messages_seen = 0
        self._compress_count = 0

    def stats(self) -> dict:
        """返回记忆统计信息"""
        return {
            "message_count": len(self.messages),
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": round(self.current_tokens / max(1, self.max_tokens) * 100, 1),
            "total_messages_seen": self._total_messages_seen,
            "compression_count": self._compress_count,
            "is_compressed": self._compress_count > 0,
        }


# ============================================================
# === LongTermMemory: 长期记忆 ===
# ============================================================
class LongTermMemory:
    """
    长期记忆 - 向量存储 + 语义检索

    - 存储关键事实、用户偏好、知识点
    - 余弦相似度检索
    - 相似度阈值0.7
    - 支持CRUD操作
    """

    def __init__(self, embedding_dim: int = 768, similarity_threshold: float = 0.7):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self._store: dict[str, dict] = {}
        self._id_counter: int = 0

    def add(self, content: str, metadata: dict = None) -> str:
        """添加记忆，返回ID"""
        self._id_counter += 1
        memory_id = f"mem_{self._id_counter:06d}_{int(time.time())}"

        embedding = self._generate_embedding(content)

        self._store[memory_id] = {
            "id": memory_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": time.time(),
            "access_count": 0,
            "last_accessed": time.time(),
        }

        return memory_id

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """语义检索最相关的记忆"""
        if not self._store:
            return []

        query_embedding = self._generate_embedding(query)
        scored = []

        for mem_id, mem_data in self._store.items():
            similarity = self._cosine_similarity(
                query_embedding, mem_data["embedding"]
            )
            if similarity >= self.similarity_threshold:
                scored.append((similarity, mem_id, mem_data))

        # 按相似度降序排序
        scored.sort(key=lambda x: x[0], reverse=True)

        # 返回top_k结果
        results = []
        for similarity, mem_id, mem_data in scored[:top_k]:
            mem_data["access_count"] += 1
            mem_data["last_accessed"] = time.time()
            results.append({
                "id": mem_id,
                "content": mem_data["content"],
                "similarity": round(similarity, 4),
                "metadata": mem_data["metadata"],
                "access_count": mem_data["access_count"],
            })

        return results

    def update(self, memory_id: str, content: str = None, metadata: dict = None):
        """更新记忆"""
        if memory_id not in self._store:
            raise KeyError(f"记忆ID '{memory_id}' 不存在")

        if content is not None:
            self._store[memory_id]["content"] = content
            self._store[memory_id]["embedding"] = self._generate_embedding(content)

        if metadata is not None:
            self._store[memory_id]["metadata"].update(metadata)

        self._store[memory_id]["last_accessed"] = time.time()

    def delete(self, memory_id: str):
        """删除记忆"""
        if memory_id not in self._store:
            raise KeyError(f"记忆ID '{memory_id}' 不存在")
        del self._store[memory_id]

    def _generate_embedding(self, text: str) -> list:
        """
        生成嵌入向量（简化实现：使用字符n-gram + 哈希）
        不依赖外部嵌入模型，可用于本地运行
        """
        if not text:
            return [0.0] * self.embedding_dim

        text = text.lower().strip()

        # 字符n-gram特征
        n_grams = []
        # unigrams
        n_grams.extend(list(text))
        # bigrams
        for i in range(len(text) - 1):
            n_grams.append(text[i:i+2])
        # trigrams
        for i in range(len(text) - 2):
            n_grams.append(text[i:i+3])
        # 单词级别bigrams
        words = text.split()
        for i in range(len(words) - 1):
            n_grams.append(f"_{words[i]}_{words[i+1]}_")

        # 使用哈希映射到embedding_dim维空间
        embedding = [0.0] * self.embedding_dim
        for gram in n_grams:
            hash_val = int(hashlib.sha256(gram.encode('utf-8')).hexdigest(), 16)
            idx = hash_val % self.embedding_dim
            # 使用哈希值的高位bit决定正负
            sign = 1 if (hash_val >> 32) % 2 == 0 else -1
            embedding[idx] += sign * 1.0

        # L2归一化
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding

    def _cosine_similarity(self, a: list, b: list) -> float:
        """计算余弦相似度"""
        if len(a) != len(b):
            raise ValueError(f"向量维度不匹配: {len(a)} vs {len(b)}")

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def stats(self) -> dict:
        """统计信息"""
        return {
            "total_memories": len(self._store),
            "embedding_dim": self.embedding_dim,
            "similarity_threshold": self.similarity_threshold,
            "avg_access_count": (
                sum(m["access_count"] for m in self._store.values()) / max(1, len(self._store))
            ),
        }


# ============================================================
# === EpisodicMemory: 情景记忆 ===
# ============================================================
@dataclass
class Episode:
    """单次任务情景"""
    episode_id: str
    task: str
    trajectory: list  # 完整状态轨迹
    result: Any
    success: bool
    timestamp: float
    lessons_learned: str = ""
    tags: list = field(default_factory=list)


class EpisodicMemory:
    """
    情景记忆 - 记录完整任务轨迹，查找相似历史情景

    - 存储完整的任务执行过程
    - 基于关键词和嵌入的相似度查找
    - 提取经验教训
    """

    def __init__(self, max_episodes: int = 100):
        self.max_episodes = max_episodes
        self.episodes: OrderedDict[str, Episode] = OrderedDict()
        self._embedding_cache: dict = {}

    def record(self, task: str, trajectory: list, result: Any,
               success: bool, lessons: str = "", tags: list = None) -> str:
        """记录一次完整情景"""
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"

        episode = Episode(
            episode_id=episode_id,
            task=task,
            trajectory=trajectory,
            result=result,
            success=success,
            timestamp=time.time(),
            lessons_learned=lessons,
            tags=tags or [],
        )

        # 添加到OrderedDict（自动按插入顺序）
        self.episodes[episode_id] = episode

        # 淘汰最旧的情景如果超过限制
        while len(self.episodes) > self.max_episodes:
            self.episodes.popitem(last=False)

        return episode_id

    def find_similar(self, task: str, top_k: int = 3,
                     success_only: bool = True) -> list:
        """查找相似的历史情景"""
        if not self.episodes:
            return []

        # 生成任务query的嵌入
        query_embedding = self._get_embedding(task)

        scored = []
        for ep_id, episode in self.episodes.items():
            if success_only and not episode.success:
                continue

            ep_embedding = self._get_embedding(episode.task)
            similarity = self._cosine_similarity(query_embedding, ep_embedding)

            # 标签匹配加分
            task_words = set(task.lower().split())
            tag_match = sum(1 for t in episode.tags if t.lower() in task_words)
            similarity += tag_match * 0.05  # 每个匹配标签加5%

            scored.append((similarity, episode))

        # 按相似度降序
        scored.sort(key=lambda x: x[0], reverse=True)

        return [ep for _, ep in scored[:top_k]]

    def get_lessons(self, task: str) -> list:
        """获取与当前任务相关的经验教训"""
        similar = self.find_similar(task, top_k=5, success_only=False)
        lessons = []

        for ep in similar:
            if ep.lessons_learned:
                lessons.append({
                    "task": ep.task,
                    "success": ep.success,
                    "lesson": ep.lessons_learned,
                    "episode_id": ep.episode_id,
                })

        return lessons

    def _get_embedding(self, text: str) -> list:
        """生成文本嵌入（简化实现）"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        text_lower = text.lower().strip()
        embedding = [0.0] * 256  # 简化维度

        # 字符n-gram哈希嵌入
        for i in range(len(text_lower)):
            # unigrams
            c = text_lower[i]
            idx = hash(f"char_{c}") % 256
            embedding[idx] += 1.0

            # bigrams
            if i < len(text_lower) - 1:
                bigram = text_lower[i:i+2]
                idx = hash(f"bi_{bigram}") % 256
                embedding[idx] += 0.5

        # L2归一化
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm > 0:
            embedding = [v / norm for v in embedding]

        self._embedding_cache[text] = embedding
        return embedding

    def _cosine_similarity(self, a: list, b: list) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def stats(self) -> dict:
        """统计信息"""
        episodes_list = list(self.episodes.values())
        total = len(episodes_list)
        success_count = sum(1 for e in episodes_list if e.success)
        return {
            "total_episodes": total,
            "max_episodes": self.max_episodes,
            "success_count": success_count,
            "success_rate": round(success_count / max(1, total) * 100, 1),
            "avg_trajectory_length": round(
                sum(len(e.trajectory) for e in episodes_list) / max(1, total), 1
            ),
            "with_lessons": sum(1 for e in episodes_list if e.lessons_learned),
        }


# ============================================================
# === Agent with Memory Integration ===
# ============================================================
class MemoryAwareAgent:
    """
    集成三种记忆的Agent示例

    1. 接收任务 -> 查询情景记忆获取历史经验
    2. 查询长期记忆获取相关知识
    3. 执行任务，短期记忆维护对话上下文
    4. 任务完成后更新所有记忆系统
    """

    def __init__(self):
        self.stm = ShortTermMemory(max_tokens=8000)
        self.ltm = LongTermMemory()
        self.em = EpisodicMemory()

    def run(self, task: str) -> str:
        """执行任务，全程使用记忆系统"""
        trajectory = []

        # Step 1: 记录任务开始到短期记忆
        self.stm.add({
            "role": "user",
            "content": f"任务: {task}",
        })
        trajectory.append({"step": "task_received", "task": task})

        # Step 2: 查询情景记忆获取历史经验
        similar_episodes = self.em.find_similar(task, top_k=3)
        lessons = self.em.get_lessons(task)

        if lessons:
            lesson_text = "\n".join(
                f"- [{l['success'] and '成功' or '失败'}] {l['task']}: {l['lesson']}"
                for l in lessons
            )
            self.stm.add({
                "role": "system",
                "content": f"历史经验:\n{lesson_text}",
            })
            trajectory.append({"step": "historical_lessons", "count": len(lessons)})

        # Step 3: 查询长期记忆获取相关知识
        relevant_memories = self.ltm.retrieve(task, top_k=5)

        if relevant_memories:
            memory_text = "\n".join(
                f"- [{m['similarity']:.2f}] {m['content']}"
                for m in relevant_memories
            )
            self.stm.add({
                "role": "system",
                "content": f"相关知识:\n{memory_text}",
            })
            trajectory.append({"step": "relevant_knowledge", "count": len(relevant_memories)})

        # Step 4: 模拟任务执行（生产环境这里调用LLM）
        task_lower = task.lower()

        # 基于历史经验决定成功概率
        if similar_episodes:
            avg_success = sum(1 for e in similar_episodes if e.success) / len(similar_episodes)
            success = avg_success >= 0.5
        else:
            success = "fail" not in task_lower

        # 执行步骤
        execution_steps = [
            f"分析任务: {task}",
            f"参考{len(similar_episodes)}个历史情景",
            f"应用{len(relevant_memories)}条相关知识",
            "执行操作..." if success else "检测到潜在问题...",
        ]

        for step_text in execution_steps:
            self.stm.add({
                "role": "assistant",
                "content": step_text,
            })
            trajectory.append({"step": "execution", "action": step_text})

        # Step 5: 生成结果
        if success:
            result = f"任务 '{task}' 执行成功。参考了{len(similar_episodes)}个历史情景和{len(relevant_memories)}条知识。"
        else:
            result = f"任务 '{task}' 执行遇到问题。需要调整策略。"

        self.stm.add({
            "role": "assistant",
            "content": f"结果: {result}",
        })
        trajectory.append({"step": "result", "result": result, "success": success})

        # Step 6: 更新长期记忆
        self.ltm.add(
            content=f"执行任务: {task}. 结果: {'成功' if success else '失败'}. {result}",
            metadata={"task": task, "success": success, "timestamp": time.time()},
        )

        # Step 7: 更新情景记忆
        lessons_text = ""
        if not success:
            lessons_text = "该类型任务需要更充分的准备和更多历史参考"
        elif len(similar_episodes) == 0:
            lessons_text = "首次执行此类任务，建议积累更多经验"

        self.em.record(
            task=task,
            trajectory=trajectory,
            result=result,
            success=success,
            lessons=lessons_text,
            tags=task_lower.split()[:5],
        )

        return result


# ============================================================
# === __main__ 演示 ===
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Agent Memory Systems - 智能体记忆系统 完整演示")
    print("=" * 70)

    # ---------------------------------------------------------
    # Demo 1: ShortTermMemory 滑动窗口和压缩
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Demo 1: 短期记忆 (ShortTermMemory)")
    print("=" * 50)

    stm = ShortTermMemory(max_tokens=8000)

    print("\n  添加消息到短期记忆...")
    conversation = [
        ("user", "你好，我想了解一下人工智能"),
        ("assistant", "你好！人工智能(AI)是计算机科学的一个分支..."),
        ("user", "什么是机器学习？"),
        ("assistant", "机器学习是AI的子领域，让计算机从数据中学习..."),
        ("user", "深度学习呢？"),
        ("assistant", "深度学习使用多层神经网络来处理复杂模式..."),
        ("user", "能给我一个简单的例子吗？"),
        ("assistant", "当然！比如图像识别，深度学习模型可以从像素数据中..."),
        ("user", "那强化学习又是什么？"),
        ("assistant", "强化学习是让智能体通过与环境的交互来学习..."),
        ("user", "这些技术有什么实际应用？"),
        ("assistant", "应用非常广泛：自动驾驶、医疗诊断、推荐系统..."),
        ("user", "学习这些需要什么数学基础？"),
        ("assistant", "主要需要线性代数、概率论、微积分和优化理论..."),
        ("user", "有推荐的入门资源吗？"),
        ("assistant", "推荐吴恩达的机器学习课程，以及《深度学习》这本书..."),
        ("user", "Python在AI中扮演什么角色？"),
        ("assistant", "Python是AI领域最主要的编程语言，拥有丰富的库..."),
        ("user", "PyTorch和TensorFlow有什么区别？"),
        ("assistant", "两者都是深度学习框架。PyTorch更灵活，适合研究..."),
        ("user", "我需要对Transformer架构了解更多"),
        ("assistant", "Transformer是2017年由Google提出的革命性架构..."),
        ("user", "注意力机制是如何工作的？"),
        ("assistant", "注意力机制允许模型关注输入序列中的相关部分..."),
        ("user", "GPT和BERT的区别？"),
        ("assistant", "GPT是自回归模型用于生成，BERT是自编码模型用于理解..."),
        ("user", "大语言模型的未来发展方向？"),
        ("assistant", "包括多模态、推理能力增强、效率优化、安全对齐等..."),
        ("user", "谢谢你的详细解释！"),
        ("assistant", "不客气！如果还有其他问题，随时问我。"),
    ]

    for role, content in conversation:
        stm.add({"role": role, "content": content})

    stats = stm.stats()
    print(f"    消息数: {stats['message_count']}")
    print(f"    当前tokens: {stats['current_tokens']}")
    print(f"    使用率: {stats['usage_percent']}%")
    print(f"    压缩次数: {stats['compression_count']}")

    # 获取上下文
    context = stm.get_context(max_messages=3)
    print(f"\n  最近3条消息:")
    for msg in context:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))[:60]
        print(f"    [{role}] {content}...")

    # 手动触发压缩测试（设置低阈值）
    print("\n  压缩测试 (设置低容量进行压力测试)...")
    stm2 = ShortTermMemory(max_tokens=500)

    long_text = "人工智能是计算机科学的一个重要分支，它研究如何创建能够模拟人类智能行为的系统。" * 5
    stm2.add({"role": "user", "content": long_text})
    stm2.add({"role": "assistant", "content": long_text})

    stats2 = stm2.stats()
    print(f"    消息数: {stats2['message_count']}")
    print(f"    压缩次数: {stats2['compression_count']}")
    print(f"    是否压缩: {stats2['is_compressed']}")

    # 查找摘要消息
    for msg in stm2.messages:
        if msg.get("is_summary"):
            print(f"    摘要: {str(msg.get('content', ''))[:120]}...")
            break

    # ---------------------------------------------------------
    # Demo 2: LongTermMemory 存储和检索
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Demo 2: 长期记忆 (LongTermMemory)")
    print("=" * 50)

    ltm = LongTermMemory(embedding_dim=256, similarity_threshold=0.3)

    # 添加多条知识
    knowledge = [
        "Python是一种解释型高级编程语言，由Guido van Rossum创建",
        "PyTorch是Facebook开发的深度学习框架，支持动态计算图",
        "TensorFlow是Google开发的深度学习框架，支持静态和动态图",
        "Transformer架构使用自注意力机制处理序列数据",
        "GPT是基于Transformer的自回归语言模型",
        "BERT是双向Transformer编码器，用于自然语言理解",
        "强化学习通过奖励信号训练智能体做出决策",
        "卷积神经网络CNN专门用于处理图像数据",
    ]

    print("\n  添加知识到长期记忆:")
    for k in knowledge:
        mem_id = ltm.add(k)
        print(f"    [+] {mem_id}: {k[:50]}...")

    print(f"\n  长期记忆统计: {ltm.stats()}")

    # 检索测试
    print("\n  检索测试:")
    queries = [
        "Python编程语言",
        "深度学习框架",
        "神经网络模型架构",
        "自然语言处理",
    ]

    for query in queries:
        print(f"\n    查询: '{query}'")
        results = ltm.retrieve(query, top_k=3)
        for r in results:
            print(f"      [{r['similarity']:.3f}] {r['content'][:60]}...")

    # CRUD测试
    print("\n  CRUD测试:")
    test_id = ltm.add("这是一条测试记忆，用于演示更新和删除功能")
    print(f"    创建: {test_id}")

    # 更新
    ltm.update(test_id, content="这是更新后的测试记忆内容")
    results = ltm.retrieve("更新后的测试", top_k=1)
    print(f"    更新后检索: {results[0]['content'] if results else '未找到'}")

    # 删除
    ltm.delete(test_id)
    results = ltm.retrieve("测试记忆", top_k=1)
    print(f"    删除后检索: {'找到' + str(len(results)) + '条' if results else '已成功删除'}")

    # ---------------------------------------------------------
    # Demo 3: EpisodicMemory 记录和相似查找
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Demo 3: 情景记忆 (EpisodicMemory)")
    print("=" * 50)

    em = EpisodicMemory(max_episodes=20)

    # 记录多个情景
    episodes_data = [
        ("查询北京天气", [{"step": 1, "action": "call_weather_api"}, {"step": 2, "action": "format_result"}],
         {"temp": 22, "condition": "晴"}, True, "天气查询简单直接，API返回结构化数据", ["weather", "api"]),
        ("计算复杂数学表达式", [{"step": 1, "action": "parse_expression"}, {"step": 2, "action": "compute"}],
         {"result": 42}, True, "使用安全eval，注意表达式注入风险", ["math", "calculation"]),
        ("分析用户文本情感", [{"step": 1, "action": "tokenize"}, {"step": 2, "action": "analyze_sentiment"}],
         {"sentiment": "positive"}, True, "情感词典方法简单但准确率有限", ["nlp", "sentiment"]),
        ("查询不存在的城市天气", [{"step": 1, "action": "call_weather_api"}, {"step": 2, "action": "error_handling"}],
         {"error": "city not found"}, False, "需要先验证城市名称的有效性", ["weather", "error"]),
        ("搜索机器学习相关论文", [{"step": 1, "action": "search_database"}, {"step": 2, "action": "rank_results"}],
         {"papers": 15}, True, "使用多关键词组合提高搜索精度", ["search", "paper"]),
    ]

    print("\n  记录情景:")
    for task, trajectory, result, success, lessons, tags in episodes_data:
        ep_id = em.record(
            task=task, trajectory=trajectory, result=result,
            success=success, lessons=lessons, tags=tags,
        )
        status = "[成功]" if success else "[失败]"
        print(f"    {status} {ep_id}: {task}")

    print(f"\n  情景记忆统计: {em.stats()}")

    # 查找相似情景
    print("\n  查找相似情景:")
    search_tasks = [
        "查询上海天气",
        "搜索深度学习文章",
        "数学公式计算",
        "分析评论情感",
    ]

    for task in search_tasks:
        print(f"\n    搜索: '{task}'")
        similar = em.find_similar(task, top_k=2)
        if similar:
            for ep in similar:
                status = "[成功]" if ep.success else "[失败]"
                print(f"      {status} {ep.task} (episodes: {ep.episode_id})")
                if ep.lessons_learned:
                    print(f"        经验: {ep.lessons_learned}")
        else:
            print(f"      未找到相似情景")

    # 获取经验教训
    print("\n  获取经验教训:")
    lessons = em.get_lessons("查询天气")
    for l in lessons:
        print(f"    任务: {l['task']}, 成功: {l['success']}")
        print(f"    经验: {l['lesson']}")

    # ---------------------------------------------------------
    # Demo 4: MemoryAwareAgent 完整流程
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Demo 4: 记忆感知Agent (MemoryAwareAgent)")
    print("=" * 50)

    agent = MemoryAwareAgent()

    # 先让agent学习一些经验
    print("\n  第一阶段: 积累经验...")
    training_tasks = [
        "查询北京天气",
        "计算数学表达式 sqrt(144)",
        "分析文本情感",
        "搜索机器学习资源",
    ]
    for task in training_tasks:
        result = agent.run(task)
        print(f"    [{task}] -> {result[:60]}...")

    print(f"\n    经验积累完成:")
    print(f"      情景数: {agent.em.stats()['total_episodes']}")
    print(f"      长期记忆数: {agent.ltm.stats()['total_memories']}")

    # 第二阶段: 利用经验执行新任务
    print("\n  第二阶段: 利用经验执行新任务...")
    new_tasks = [
        "查询上海天气",
        "搜索深度学习教程",
        "计算复杂的三角函数",
    ]
    for task in new_tasks:
        result = agent.run(task)
        print(f"    [{task}] -> {result[:60]}...")

    # 最终统计
    print(f"\n  最终统计:")
    print(f"    短期记忆: {agent.stm.stats()}")
    print(f"    长期记忆: {agent.ltm.stats()}")
    print(f"    情景记忆: {agent.em.stats()}")

    print("\n" + "=" * 70)
    print("  所有演示完成！")
    print("=" * 70)
