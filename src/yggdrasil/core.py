"""
Vector store implementation for Yggdrasil.

Provides an in-memory vector database with brute-force nearest neighbor
search supporting cosine similarity, euclidean distance, and dot product
metrics. Educational implementation using pure Python math.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Vector:
    """A vector with an identifier and optional metadata.

    Attributes:
        id: Unique identifier for the vector.
        values: The embedding values as a list of floats.
        metadata: Arbitrary key-value metadata associated with the vector.
    """

    id: str
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Vector values cannot be empty")
        if not all(isinstance(v, (int, float)) for v in self.values):
            raise TypeError("All vector values must be numeric")

    @property
    def dimension(self) -> int:
        """Return the dimensionality of this vector."""
        return len(self.values)

    def norm(self) -> float:
        """Compute the L2 norm (magnitude) of this vector."""
        return math.sqrt(sum(v * v for v in self.values))

    def normalize(self) -> "Vector":
        """Return a unit-length copy of this vector."""
        n = self.norm()
        if n == 0.0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector(
            id=self.id,
            values=[v / n for v in self.values],
            metadata=dict(self.metadata),
        )


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1] where 1 means identical direction,
    0 means orthogonal, and -1 means opposite direction.
    """
    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same dimension: "
            f"{len(a)} != {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean (L2) distance between two vectors.

    Returns a non-negative value where 0 means identical vectors.
    """
    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same dimension: "
            f"{len(a)} != {len(b)}"
        )
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dot_product(a: List[float], b: List[float]) -> float:
    """Compute the dot product of two vectors.

    Higher values indicate greater similarity when vectors are normalized.
    """
    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same dimension: "
            f"{len(a)} != {len(b)}"
        )
    return sum(x * y for x, y in zip(a, b))


# Mapping of metric names to (function, higher_is_better) tuples
_METRICS = {
    "cosine": (cosine_similarity, True),
    "euclidean": (euclidean_distance, False),
    "dot": (dot_product, True),
}


class VectorStore:
    """In-memory vector database with brute-force nearest neighbor search.

    Stores vectors keyed by their ID and supports similarity search using
    configurable distance metrics.

    Example::

        store = VectorStore(dimension=3)
        store.insert(Vector(id="a", values=[1.0, 0.0, 0.0]))
        results = store.search([1.0, 0.1, 0.0], k=5, metric="cosine")
    """

    def __init__(self, dimension: int) -> None:
        """Initialize a VectorStore for vectors of a given dimension.

        Args:
            dimension: The fixed dimensionality of all vectors in this store.
        """
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        self._dimension = dimension
        self._vectors: Dict[str, Vector] = {}

    @property
    def dimension(self) -> int:
        """The dimensionality of vectors in this store."""
        return self._dimension

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, vector_id: str) -> bool:
        return vector_id in self._vectors

    def insert(self, vector: Vector) -> None:
        """Insert a vector into the store.

        Args:
            vector: The vector to insert.

        Raises:
            ValueError: If the vector dimension does not match the store.
            KeyError: If a vector with the same ID already exists.
        """
        if vector.dimension != self._dimension:
            raise ValueError(
                f"Vector dimension {vector.dimension} does not match "
                f"store dimension {self._dimension}"
            )
        if vector.id in self._vectors:
            raise KeyError(f"Vector with id '{vector.id}' already exists")
        self._vectors[vector.id] = vector

    def upsert(self, vector: Vector) -> None:
        """Insert or update a vector in the store.

        If a vector with the same ID exists, it is replaced.
        """
        if vector.dimension != self._dimension:
            raise ValueError(
                f"Vector dimension {vector.dimension} does not match "
                f"store dimension {self._dimension}"
            )
        self._vectors[vector.id] = vector

    def get(self, vector_id: str) -> Optional[Vector]:
        """Retrieve a vector by its ID, or None if not found."""
        return self._vectors.get(vector_id)

    def delete(self, vector_id: str) -> bool:
        """Delete a vector by ID. Returns True if it existed."""
        if vector_id in self._vectors:
            del self._vectors[vector_id]
            return True
        return False

    def list_ids(self) -> List[str]:
        """Return all vector IDs in insertion order."""
        return list(self._vectors.keys())

    def search(
        self,
        query: List[float],
        k: int = 10,
        metric: str = "cosine",
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Vector, float]]:
        """Find the k nearest neighbors to a query vector.

        Uses brute-force search across all stored vectors.

        Args:
            query: The query vector values.
            k: Number of results to return.
            metric: One of 'cosine', 'euclidean', or 'dot'.
            filter_metadata: Optional dict of metadata key-value pairs
                that results must match.

        Returns:
            A list of (vector, score) tuples sorted by relevance.
            For cosine and dot product, higher scores are better.
            For euclidean, lower scores are better.
        """
        if len(query) != self._dimension:
            raise ValueError(
                f"Query dimension {len(query)} does not match "
                f"store dimension {self._dimension}"
            )
        if metric not in _METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Choose from: {list(_METRICS.keys())}"
            )
        if k <= 0:
            raise ValueError("k must be a positive integer")

        distance_fn, higher_is_better = _METRICS[metric]
        scored: List[Tuple[Vector, float]] = []

        for vec in self._vectors.values():
            if filter_metadata:
                if not all(
                    vec.metadata.get(key) == val
                    for key, val in filter_metadata.items()
                ):
                    continue
            score = distance_fn(query, vec.values)
            scored.append((vec, score))

        scored.sort(key=lambda x: x[1], reverse=higher_is_better)
        return scored[:k]

    def clear(self) -> None:
        """Remove all vectors from the store."""
        self._vectors.clear()


class Collection:
    """A named group of VectorStores organized by namespace.

    Collections allow logical separation of vector data while
    sharing the same dimensionality.

    Example::

        collection = Collection(name="documents", dimension=128)
        ns = collection.get_or_create_namespace("articles")
        ns.insert(Vector(id="doc1", values=[...]))
    """

    def __init__(self, name: str, dimension: int) -> None:
        """Create a named collection.

        Args:
            name: Human-readable name for this collection.
            dimension: Fixed vector dimensionality for all namespaces.
        """
        if not name:
            raise ValueError("Collection name cannot be empty")
        self._name = name
        self._dimension = dimension
        self._namespaces: Dict[str, VectorStore] = {}

    @property
    def name(self) -> str:
        """The name of this collection."""
        return self._name

    @property
    def dimension(self) -> int:
        """The dimensionality of vectors in this collection."""
        return self._dimension

    def list_namespaces(self) -> List[str]:
        """Return all namespace names."""
        return list(self._namespaces.keys())

    def get_or_create_namespace(self, namespace: str) -> VectorStore:
        """Get a namespace's VectorStore, creating it if needed.

        Args:
            namespace: The namespace name.

        Returns:
            The VectorStore for this namespace.
        """
        if namespace not in self._namespaces:
            self._namespaces[namespace] = VectorStore(
                dimension=self._dimension
            )
        return self._namespaces[namespace]

    def delete_namespace(self, namespace: str) -> bool:
        """Delete a namespace. Returns True if it existed."""
        if namespace in self._namespaces:
            del self._namespaces[namespace]
            return True
        return False

    def total_vectors(self) -> int:
        """Return the total number of vectors across all namespaces."""
        return sum(len(store) for store in self._namespaces.values())

    def search_all(
        self,
        query: List[float],
        k: int = 10,
        metric: str = "cosine",
    ) -> List[Tuple[str, Vector, float]]:
        """Search across all namespaces.

        Returns:
            List of (namespace, vector, score) tuples.
        """
        results: List[Tuple[str, Vector, float]] = []
        for ns_name, store in self._namespaces.items():
            for vec, score in store.search(query, k=k, metric=metric):
                results.append((ns_name, vec, score))

        _, higher_is_better = _METRICS[metric]
        results.sort(key=lambda x: x[2], reverse=higher_is_better)
        return results[:k]
