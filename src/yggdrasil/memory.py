"""
Memory layer for Yggdrasil.

Combines the vector store and knowledge graph to provide a unified
memory system for AI agents with importance scoring based on
access frequency and recency.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from yggdrasil.core import Vector, VectorStore
from yggdrasil.graph import Edge, KnowledgeGraph, Node


@dataclass
class MemoryEntry:
    """A single memory combining text, embedding, and graph relations.

    Attributes:
        id: Unique identifier for the memory.
        text: The text content of the memory.
        embedding: The vector representation of the text.
        metadata: Arbitrary metadata.
        created_at: Timestamp when the memory was created.
        last_accessed: Timestamp when the memory was last accessed.
        access_count: Number of times this memory has been recalled.
    """

    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


class MemoryLayer:
    """Unified memory system combining vector search and knowledge graph.

    Provides store, recall, and forget operations with importance scoring
    that factors in both access frequency and recency.

    Example::

        memory = MemoryLayer(dimension=3)
        memory.store(
            text="Odin is the Allfather",
            embedding=[0.1, 0.9, 0.3],
            relations=[("odin", "title", "Allfather")],
        )
        results = memory.recall(query_embedding=[0.1, 0.85, 0.25], k=5)
    """

    def __init__(
        self,
        dimension: int,
        decay_rate: float = 0.01,
    ) -> None:
        """Initialize the memory layer.

        Args:
            dimension: Dimensionality of the embeddings.
            decay_rate: Time-decay rate for importance scoring.
                Higher values make older memories less important faster.
        """
        self._vector_store = VectorStore(dimension=dimension)
        self._graph = KnowledgeGraph()
        self._entries: Dict[str, MemoryEntry] = {}
        self._decay_rate = decay_rate
        self._dimension = dimension

    @property
    def vector_store(self) -> VectorStore:
        """Access the underlying vector store."""
        return self._vector_store

    @property
    def graph(self) -> KnowledgeGraph:
        """Access the underlying knowledge graph."""
        return self._graph

    def __len__(self) -> int:
        return len(self._entries)

    def store(
        self,
        text: str,
        embedding: List[float],
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        relations: Optional[List[Tuple[str, str, str]]] = None,
    ) -> str:
        """Store a new memory.

        Args:
            text: The text content.
            embedding: The vector embedding of the text.
            memory_id: Optional custom ID. Auto-generated if not provided.
            metadata: Optional metadata dict.
            relations: Optional list of (source, relation, target) triples
                to add to the knowledge graph. Nodes are auto-created.

        Returns:
            The ID of the stored memory.
        """
        mid = memory_id or str(uuid.uuid4())
        now = time.time()

        entry = MemoryEntry(
            id=mid,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            created_at=now,
            last_accessed=now,
            access_count=0,
        )
        self._entries[mid] = entry

        vector = Vector(
            id=mid,
            values=embedding,
            metadata={"text": text, **(metadata or {})},
        )
        self._vector_store.upsert(vector)

        # Add graph node for this memory
        if not self._graph.has_node(mid):
            self._graph.add_node(
                Node(id=mid, label="memory", properties={"text": text})
            )

        # Add relation triples
        if relations:
            for source, relation, target in relations:
                if not self._graph.has_node(source):
                    self._graph.add_node(Node(id=source, label="entity"))
                if not self._graph.has_node(target):
                    self._graph.add_node(Node(id=target, label="entity"))
                self._graph.add_edge(
                    Edge(source=source, target=target, relation=relation)
                )

        return mid

    def recall(
        self,
        query_embedding: List[float],
        k: int = 10,
        metric: str = "cosine",
    ) -> List[Tuple[MemoryEntry, float]]:
        """Recall memories similar to a query embedding.

        Updates access statistics for returned memories and computes
        an importance-weighted score.

        Args:
            query_embedding: The query vector.
            k: Number of memories to return.
            metric: Distance metric ('cosine', 'euclidean', 'dot').

        Returns:
            List of (memory_entry, importance_score) tuples.
        """
        raw_results = self._vector_store.search(
            query_embedding, k=k, metric=metric
        )

        now = time.time()
        scored: List[Tuple[MemoryEntry, float]] = []

        for vec, similarity in raw_results:
            entry = self._entries.get(vec.id)
            if entry is None:
                continue

            # Update access statistics
            entry.last_accessed = now
            entry.access_count += 1

            # Compute importance score
            importance = self._compute_importance(entry, similarity, now)
            scored.append((entry, importance))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _compute_importance(
        self, entry: MemoryEntry, similarity: float, now: float
    ) -> float:
        """Compute importance score combining similarity, frequency, and recency.

        Score = similarity * (1 + log_frequency) * recency_factor
        where recency_factor decays exponentially with time.
        """
        import math

        # Frequency boost: logarithmic scaling of access count
        freq_boost = 1.0 + math.log1p(entry.access_count)

        # Recency factor: exponential decay based on time since last access
        time_delta = now - entry.created_at
        recency = math.exp(-self._decay_rate * time_delta)

        return similarity * freq_boost * recency

    def forget(self, memory_id: str) -> bool:
        """Remove a memory by ID.

        Removes from vector store, entries, and the memory node from
        the graph (but preserves entity nodes).

        Returns:
            True if the memory existed and was removed.
        """
        if memory_id not in self._entries:
            return False

        del self._entries[memory_id]
        self._vector_store.delete(memory_id)
        self._graph.remove_node(memory_id)
        return True

    def get_entry(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        return self._entries.get(memory_id)

    def get_related(
        self, memory_id: str, relation: Optional[str] = None
    ) -> List[Tuple[Node, Edge]]:
        """Get graph neighbors of a memory node.

        Args:
            memory_id: The memory node ID.
            relation: Optional relation type filter.

        Returns:
            List of (neighbor_node, edge) tuples.
        """
        if not self._graph.has_node(memory_id):
            return []
        return self._graph.get_neighbors(
            memory_id, direction="both", relation=relation
        )

    def clear(self) -> None:
        """Remove all memories."""
        self._entries.clear()
        self._vector_store.clear()
        self._graph.clear()
