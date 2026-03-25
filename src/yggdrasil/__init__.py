"""
Yggdrasil - High-performance vector + graph database.

Named after the Norse World Tree that connects all nine realms,
Yggdrasil bridges vector search and knowledge graphs to create
a unified memory system for AI agents.
"""

from yggdrasil.core import Vector, VectorStore, Collection
from yggdrasil.graph import Node, Edge, KnowledgeGraph
from yggdrasil.memory import MemoryEntry, MemoryLayer
from yggdrasil.config import Config

__version__ = "0.1.0"
__all__ = [
    "Vector",
    "VectorStore",
    "Collection",
    "Node",
    "Edge",
    "KnowledgeGraph",
    "MemoryEntry",
    "MemoryLayer",
    "Config",
]
