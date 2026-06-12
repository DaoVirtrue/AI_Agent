#!/usr/bin/env python3
"""
Phase 04 Module 08: Multi-Turn Context Completion
====================================================
- Conversation history management (last 3-5 turns)
- Coreference resolution: "它", "这个", "以上" → full entity reference
- Context stitching: merge fragmented context across turns
- Long-term memory pool integration
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: int
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = 0.0
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedQuery:
    """Result of multi-turn context completion."""
    original: str
    resolved: str
    coreferences_found: List[str] = field(default_factory=list)
    coreferences_resolved: List[Tuple[str, str]] = field(default_factory=list)
    context_summary: str = ""
    confidence: float = 1.0


@dataclass
class MemoryEntry:
    """An entry in the long-term memory pool."""
    entry_id: str
    content: str
    category: str = "general"
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    source_turn: int = 0


# ===========================================================================
# Coreference Patterns
# ===========================================================================

# Chinese coreference markers and their resolution strategies
ZH_COREFERENCE_PATTERNS: Dict[str, str] = {
    # Pronoun → entity resolution
    '它': 'entity',      # it (object)
    '他': 'entity',      # he
    '她': 'entity',      # she
    '其': 'entity',      # its
    '该': 'entity',      # the/this (formal)
    '这': 'topic',       # this
    '那': 'topic',       # that
    '此': 'topic',       # this (formal)

    # Demonstrative phrases
    '这个': 'topic',     # this one
    '那个': 'topic',     # that one
    '这些': 'topic',     # these
    '那些': 'topic',     # those
    '这种': 'topic',     # this kind
    '那种': 'topic',     # that kind

    # Reference to previous content
    '以上': 'previous',  # above
    '上述': 'previous',  # aforementioned
    '前面': 'previous',  # previous
    '之前': 'previous',  # before
    '刚刚': 'previous',  # just now

    # Comparative references
    '前者': 'former',    # the former
    '后者': 'latter',    # the latter
    '第一个': 'ordinal', # the first
    '第二个': 'ordinal', # the second
}

EN_COREFERENCE_PATTERNS: Dict[str, str] = {
    'it': 'entity',
    'he': 'entity',
    'she': 'entity',
    'they': 'entity',
    'them': 'entity',
    'its': 'entity',
    'this': 'topic',
    'that': 'topic',
    'these': 'topic',
    'those': 'topic',
    'the above': 'previous',
    'aforementioned': 'previous',
    'the former': 'former',
    'the latter': 'latter',
    'the first': 'ordinal',
    'the second': 'ordinal',
}


# ===========================================================================
# Entity Extractor
# ===========================================================================

class EntityExtractor:
    """Extract key entities and topics from text."""

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extract named entities and key noun phrases from text."""
        entities: List[str] = []

        # English acronyms (ALL_CAPS)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        entities.extend(acronyms)

        # Chinese proper nouns: 2-8 char sequences after patterns like XX模型/XX系统/XX架构
        cn_proper = re.findall(
            r'([A-Za-z]*[一-鿿]{2,6}(?:模型|系统|架构|算法|网络|框架|技术|方法|理论'
            r'|机制|数据库|引擎|协议|标准|语言|平台|工具))',
            text
        )
        entities.extend(cn_proper)

        # Quoted terms
        quoted = re.findall(r'[「「]([^」」]+)[」」]', text)
        entities.extend(quoted)

        quoted2 = re.findall(r'"([^"]{2,30})"', text)
        entities.extend(quoted2)

        return list(set(entities))

    @staticmethod
    def extract_topics(text: str) -> List[str]:
        """Extract main topics/themes from text."""
        topics: List[str] = []

        # Topic indicators in Chinese
        topic_patterns = [
            r'(?:关于|对于|至于|Regarding|About|Concerning)\s*(.+?)(?:[，,。.；;]|$)',
            r'(?:主题|话题|Topic|Subject)[：:]\s*(.+?)(?:[，,。.；;]|$)',
        ]
        for pat in topic_patterns:
            matches = re.findall(pat, text)
            topics.extend(matches)

        # First significant noun phrase (often the topic)
        first_sent = text.split('。')[0].split('.')[0]
        # Remove leading adverbs
        first_sent = re.sub(r'^(首先|然后|此外|另外|However|Therefore|Thus|Moreover)\s*,?\s*',
                           '', first_sent)
        topics.append(first_sent[:50].strip())

        return [t.strip() for t in topics if t.strip() and len(t.strip()) > 2]


# ===========================================================================
# Conversation History Manager
# ===========================================================================

class ConversationHistory:
    """Manage conversation history for multi-turn context."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._turn_counter: int = 0

    def add_turn(self, role: str, content: str,
                 entities: Optional[List[str]] = None,
                 topics: Optional[List[str]] = None) -> ConversationTurn:
        """Add a new turn to the conversation history."""
        import time
        self._turn_counter += 1

        if entities is None:
            entities = EntityExtractor.extract_entities(content)
        if topics is None:
            topics = EntityExtractor.extract_topics(content)

        turn = ConversationTurn(
            turn_id=self._turn_counter,
            role=role,
            content=content,
            timestamp=time.time(),
            entities=entities,
            topics=topics,
        )
        self.turns.append(turn)
        return turn

    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Get the most recent n turns."""
        if n is None:
            n = self.max_turns
        turns_list = list(self.turns)
        return turns_list[-n:] if n < len(turns_list) else turns_list

    def get_last_user_query(self) -> Optional[str]:
        """Get the most recent user query."""
        for turn in reversed(self.turns):
            if turn.role == 'user':
                return turn.content
        return None

    def get_last_assistant_response(self) -> Optional[str]:
        """Get the most recent assistant response."""
        for turn in reversed(self.turns):
            if turn.role == 'assistant':
                return turn.content
        return None

    def get_all_entities(self, last_n: int = 3) -> List[str]:
        """Get all entities mentioned in recent turns."""
        all_entities: List[str] = []
        recent = self.get_recent_turns(last_n)
        for turn in recent:
            all_entities.extend(turn.entities)
        return list(set(all_entities))

    def get_all_topics(self, last_n: int = 3) -> List[str]:
        """Get all topics mentioned in recent turns."""
        all_topics: List[str] = []
        recent = self.get_recent_turns(last_n)
        for turn in recent:
            all_topics.extend(turn.topics)
        return list(set(all_topics))

    def to_context_string(self, max_turns: int = 3) -> str:
        """Build a context string from recent history."""
        recent = self.get_recent_turns(max_turns)
        parts: List[str] = []
        for turn in recent:
            role_label = 'User' if turn.role == 'user' else 'Assistant'
            content_preview = turn.content[:200] + ('...' if len(turn.content) > 200 else '')
            parts.append(f"[{role_label}]: {content_preview}")
        return '\n'.join(parts)


# ===========================================================================
# Coreference Resolver
# ===========================================================================

class CoreferenceResolver:
    """Resolve coreference expressions to their full referents."""

    def __init__(self, history: ConversationHistory):
        self.history = history

    def resolve(self, query: str) -> ResolvedQuery:
        """Resolve coreferences in a query using conversation history."""
        result = ResolvedQuery(original=query)
        resolved = query

        # Step 1: Detect coreference markers
        markers = self._detect_coreferences(query)
        result.coreferences_found = markers

        if not markers:
            result.resolved = query
            result.context_summary = "No coreferences found"
            return result

        # Step 2: Resolve each coreference
        for marker, ref_type in markers:
            referent = self._find_referent(marker, ref_type)
            if referent:
                resolved = self._replace_coreference(resolved, marker, referent)
                result.coreferences_resolved.append((marker, referent))

        result.resolved = resolved
        result.context_summary = self.history.to_context_string(max_turns=3)
        result.confidence = 0.8 if result.coreferences_resolved else 0.3

        return result

    def _detect_coreferences(self, text: str) -> List[Tuple[str, str]]:
        """Detect coreference markers in text. Returns [(marker, type), ...]."""
        markers: List[Tuple[str, str]] = []

        # Chinese patterns
        for marker, ref_type in ZH_COREFERENCE_PATTERNS.items():
            if marker in text:
                markers.append((marker, ref_type))

        # English patterns (word boundary check)
        for marker, ref_type in EN_COREFERENCE_PATTERNS.items():
            pattern = r'\b' + re.escape(marker) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                markers.append((marker, ref_type))

        return markers

    def _find_referent(self, marker: str, ref_type: str) -> Optional[str]:
        """Find the referent for a coreference marker from history."""
        if ref_type == 'entity':
            return self._resolve_entity(marker)
        elif ref_type == 'topic':
            return self._resolve_topic(marker)
        elif ref_type == 'previous':
            return self._resolve_previous(marker)
        elif ref_type == 'former':
            return self._resolve_comparison('former')
        elif ref_type == 'latter':
            return self._resolve_comparison('latter')
        elif ref_type == 'ordinal':
            return self._resolve_ordinal(marker)
        return None

    def _resolve_entity(self, marker: str) -> Optional[str]:
        """Resolve pronoun to the most recent entity."""
        last_user = self.history.get_last_user_query()
        if not last_user:
            return None

        entities = EntityExtractor.extract_entities(last_user)
        if entities:
            return entities[0]  # Most prominent entity (first detected)

        # Fallback: first significant noun phrase
        words = last_user.split()
        if words:
            return words[0][:15]
        return None

    def _resolve_topic(self, marker: str) -> Optional[str]:
        """Resolve demonstrative to the main topic."""
        topics = self.history.get_all_topics(last_n=2)
        if topics:
            # Return the most recent distinct topic
            return topics[0][:60]

        # Fallback to last user query summary
        last_user = self.history.get_last_user_query()
        if last_user:
            return f"关于「{last_user[:50]}」"
        return None

    def _resolve_previous(self, marker: str) -> Optional[str]:
        """Resolve '以上/上述/the above' to the complete previous context."""
        last_assistant = self.history.get_last_assistant_response()
        if last_assistant:
            # Take the key conclusion or first 100 chars
            sentences = re.split(r'[。.；;]', last_assistant)
            if sentences:
                return f"以上关于「{sentences[0][:80]}」的讨论"
        return "以上讨论的内容"

    def _resolve_comparison(self, position: str) -> Optional[str]:
        """Resolve '前者/后者/the former/the latter'."""
        last_user = self.history.get_last_user_query()
        if not last_user:
            return None

        # Look for comparison patterns: "A 和 B", "A vs B", "A 与 B"
        comparison_patterns = [
            r'(.+?)\s*(?:和|与|及|vs\.?|versus|or|compared\s+to)\s*(.+?)(?:[，,。.]|$)',
        ]

        for pat in comparison_patterns:
            m = re.search(pat, last_user, re.IGNORECASE)
            if m:
                if position == 'former':
                    return m.group(1).strip()
                else:
                    return m.group(2).strip()

        return None

    def _resolve_ordinal(self, marker: str) -> Optional[str]:
        """Resolve ordinal references like '第一个/第二个'."""
        ordinal_map = {
            '第一个': 0, 'the first': 0,
            '第二个': 1, 'the second': 1,
            '第三个': 2, 'the third': 2,
        }

        idx = ordinal_map.get(marker.lower(), -1)
        if idx < 0:
            return None

        last_assistant = self.history.get_last_assistant_response()
        if not last_assistant:
            return None

        # Look for numbered lists
        list_items = re.findall(r'(?:\d+[\.、)]|[-•])\s*(.+?)(?:\n|$)', last_assistant)
        if len(list_items) > idx:
            return list_items[idx].strip()

        return None

    def _replace_coreference(self, text: str, marker: str, referent: str) -> str:
        """Replace a coreference marker with its referent."""
        # For Chinese markers (no word boundaries)
        if re.search(r'[一-鿿]', marker):
            replacement = f"「{referent}」"
            return text.replace(marker, replacement, 1)

        # For English markers (word boundary)
        pattern = r'\b' + re.escape(marker) + r'\b'
        replacement = f"「{referent}」"
        return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)


# ===========================================================================
# Context Stitcher
# ===========================================================================

class ContextStitcher:
    """Merge fragmented context across multiple turns into a coherent context."""

    @staticmethod
    def stitch(history: ConversationHistory, max_turns: int = 3) -> str:
        """Stitch recent conversation turns into a coherent context paragraph."""
        recent = history.get_recent_turns(max_turns)
        if not recent:
            return ""

        parts: List[str] = ["[Conversation Context]"]

        # Build a structured summary
        topics_mentioned: Set[str] = set()
        entities_mentioned: Set[str] = set()

        for turn in recent:
            topics_mentioned.update(turn.topics)
            entities_mentioned.update(turn.entities)

        if topics_mentioned:
            parts.append(f"Topics: {', '.join(list(topics_mentioned)[:5])}")
        if entities_mentioned:
            parts.append(f"Entities: {', '.join(list(entities_mentioned)[:10])}")

        # Add turn summaries
        for turn in recent:
            role = 'User' if turn.role == 'user' else 'Assistant'
            summary = turn.content[:200] + ('...' if len(turn.content) > 200 else '')
            parts.append(f"[{role} T{turn.turn_id}]: {summary}")

        return '\n'.join(parts)

    @staticmethod
    def build_query_context(query: str, history: ConversationHistory,
                           max_turns: int = 3) -> str:
        """Build a context-augmented query for retrieval.

        Combines the stitched context with the current query for
        better retrieval relevance.
        """
        context = ContextStitcher.stitch(history, max_turns)
        if not context:
            return query

        # Determine how much context to prepend (don't overwhelm the query)
        context_preview = context[:300]

        return f"{context_preview}\n\n[Current Query]: {query}"


# ===========================================================================
# Long-Term Memory Pool
# ===========================================================================

class LongTermMemoryPool:
    """A simple long-term memory pool for storing important facts across sessions."""

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.entries: Dict[str, MemoryEntry] = {}
        self._counter: int = 0

    def add(self, content: str, category: str = "general",
            importance: float = 0.5, source_turn: int = 0) -> str:
        """Add an entry to the memory pool."""
        import time
        self._counter += 1
        entry_id = f"mem_{self._counter:06d}"

        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            category=category,
            importance=importance,
            access_count=0,
            last_accessed=time.time(),
            source_turn=source_turn,
        )
        self.entries[entry_id] = entry

        # Evict least important entries if over capacity
        if len(self.entries) > self.max_entries:
            self._evict()

        return entry_id

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Simple keyword-based search in memory pool."""
        query_lower = query.lower()
        scored: List[Tuple[float, MemoryEntry]] = []

        for entry in self.entries.values():
            content_lower = entry.content.lower()
            # Simple TF scoring
            score = 0.0
            for word in query_lower.split():
                count = content_lower.count(word)
                if count > 0:
                    score += count * entry.importance
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:top_k]]

        # Update access counters
        import time
        for entry in results:
            entry.access_count += 1
            entry.last_accessed = time.time()

        return results

    def _evict(self) -> None:
        """Evict the least important entries."""
        # Sort by (importance * 0.7 + recency * 0.3), remove bottom 20%
        import time
        now = time.time()

        def score(entry: MemoryEntry) -> float:
            recency = 1.0 / (1.0 + (now - entry.last_accessed) / 86400.0)  # Days
            return entry.importance * 0.7 + recency * 0.3

        sorted_entries = sorted(self.entries.items(),
                               key=lambda x: score(x[1]))
        to_remove = int(len(sorted_entries) * 0.2)
        for entry_id, _ in sorted_entries[:max(to_remove, 1)]:
            del self.entries[entry_id]

    def to_context(self, query: str) -> str:
        """Retrieve relevant memories for context augmentation."""
        results = self.search(query, top_k=3)
        if not results:
            return ""

        parts = ["[Relevant Past Knowledge]"]
        for entry in results:
            parts.append(f"- {entry.content[:150]}")
        return '\n'.join(parts)


# ===========================================================================
# MultiTurnCompletionPipeline
# ===========================================================================

class MultiTurnCompletionPipeline:
    """Complete multi-turn conversation management with coreference resolution,
    context stitching, and long-term memory integration."""

    def __init__(
        self,
        max_history_turns: int = 5,
        memory_pool: Optional[LongTermMemoryPool] = None,
        enable_memory: bool = True,
    ):
        self.history = ConversationHistory(max_turns=max_history_turns)
        self.resolver = CoreferenceResolver(self.history)
        self.stitcher = ContextStitcher()
        self.memory_pool = memory_pool or LongTermMemoryPool()
        self.enable_memory = enable_memory

    def process(self, query: str) -> ResolvedQuery:
        """Process a query through the multi-turn pipeline.

        Steps:
        1. Add to history
        2. Resolve coreferences
        3. Optionally augment with memory context
        """
        # Resolve coreferences
        resolution = self.resolver.resolve(query)

        # Augment with long-term memory if enabled
        if self.enable_memory:
            memory_context = self.memory_pool.to_context(query)
            if memory_context:
                resolution.context_summary += f"\n\n{memory_context}"

        return resolution

    def process_and_record(self, query: str, assistant_response: str = "") -> ResolvedQuery:
        """Process query and record the turn."""
        # Process first
        resolution = self.process(query)

        # Add user turn
        self.history.add_turn('user', resolution.resolved)

        # Extract entities for memory
        entities = EntityExtractor.extract_entities(resolution.resolved)
        for entity in entities:
            self.memory_pool.add(entity, category="entity", importance=0.6)

        # Record assistant response if provided
        if assistant_response:
            self.history.add_turn('assistant', assistant_response)

        return resolution

    def get_retrieval_context(self, query: str) -> str:
        """Get the full retrieval context for augmented search."""
        # Build stitched context
        stitched = self.stitcher.build_query_context(query, self.history)

        # Add memory context
        if self.enable_memory:
            memory_ctx = self.memory_pool.to_context(query)
            if memory_ctx:
                stitched = f"{stitched}\n\n{memory_ctx}"

        return stitched


# ===========================================================================
# __main__: Demo
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Multi-Turn Context Completion - Self-Test")
    print("=" * 60)

    # --- Initialize pipeline ---
    pipeline = MultiTurnCompletionPipeline(
        max_history_turns=5,
        enable_memory=True,
    )

    # --- Simulate a conversation ---
    conversation = [
        ("什么是Transformer架构？",
         "Transformer是一种基于自注意力机制的神经网络架构，由编码器和解码器组成。"
         "它的核心创新是使用多头自注意力来捕获序列中的依赖关系。"),

        ("它的主要优点是什么？",
         "Transformer的主要优点包括：1) 并行计算能力强，训练速度快；"
         "2) 能够建模长距离依赖；3) 在NLP和CV领域都表现出色。"),

        ("这个架构和RNN有什么区别？",
         "Transformer与RNN的主要区别：1) Transformer完全基于注意力机制，"
         "不依赖循环结构；2) 并行计算 vs 序列计算；3) 更好的长距离依赖建模。"),

        ("以上提到的注意力机制具体怎么计算的？",
         "注意力机制通过Query、Key、Value三个矩阵计算。公式为："
         "Attention(Q,K,V) = softmax(QK^T/√d_k)V。"),

        ("后者在什么场景下更适用？",
         "RNN在以下场景更适用：1) 序列长度较短的任务；"
         "2) 资源受限的嵌入式设备；3) 需要流式处理的场景。"),
    ]

    for qi, (query, response) in enumerate(conversation, 1):
        print(f"\n{'─'*50}")
        print(f"Turn {qi}")
        print(f"  User Query:     {query}")

        resolution = pipeline.process_and_record(query, response)

        if resolution.coreferences_found:
            print(f"  Coreferences:   {', '.join(resolution.coreferences_found)}")
            for marker, referent in resolution.coreferences_resolved:
                print(f"    {marker} → {referent}")

        print(f"  Resolved Query: {resolution.resolved[:120]}...")

    # --- Show conversation history ---
    print(f"\n{'='*60}")
    print("Conversation History Summary")
    print(f"{'='*60}")
    print(pipeline.history.to_context_string(max_turns=5))

    # --- Show context stitching ---
    print(f"\n{'='*60}")
    print("Context Stitching for Retrieval")
    print(f"{'='*60}")
    context = pipeline.get_retrieval_context("它的计算复杂度如何？")
    print(context[:600])

    # --- Test individual components ---
    print(f"\n{'='*60}")
    print("Component Tests")
    print(f"{'='*60}")

    # Entity extraction
    test_text = "BERT模型和GPT架构是当前最流行的大型语言模型（LLM）。"
    entities = EntityExtractor.extract_entities(test_text)
    print(f"\nEntity Extraction:")
    print(f"  Text: {test_text}")
    print(f"  Entities: {entities}")

    # Coreference detection
    history2 = ConversationHistory(max_turns=3)
    history2.add_turn('user', "Transformer和CNN在图像处理中的表现如何？")
    history2.add_turn('assistant', "Transformer在图像处理中表现优异...")
    resolver = CoreferenceResolver(history2)

    test_queries = [
        "它适合哪些任务？",
        "这个架构的计算复杂度高吗？",
        "前者有什么优势？",
    ]

    print(f"\nCoreference Resolution:")
    for tq in test_queries:
        result = resolver.resolve(tq)
        print(f"  Input:   {result.original}")
        print(f"  Resolved: {result.resolved}")
        if result.coreferences_resolved:
            for marker, ref in result.coreferences_resolved:
                print(f"    {marker} → {ref}")

    # Memory pool
    print(f"\n{'='*60}")
    print("Long-Term Memory Pool")
    print(f"{'='*60}")
    mem = LongTermMemoryPool(max_entries=100)
    mem.add("Transformer架构基于自注意力机制", category="technical", importance=0.9)
    mem.add("BERT使用掩码语言模型预训练", category="technical", importance=0.8)
    mem.add("GPT采用自回归语言模型", category="technical", importance=0.8)

    search_results = mem.search("注意力机制 Transformer", top_k=3)
    print(f"\nSearch '注意力机制 Transformer':")
    for entry in search_results:
        print(f"  [{entry.entry_id}] imp={entry.importance:.1f} "
              f"accesses={entry.access_count}")
        print(f"    {entry.content}")

    print(f"\n{'='*60}")
    print("Multi-Turn Context Completion self-test completed!")
    print(f"{'='*60}")
