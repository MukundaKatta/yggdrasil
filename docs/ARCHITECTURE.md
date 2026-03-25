# Architecture

## Overview

Yggdrasil is a combined vector search and knowledge graph database designed for AI agent memory. Named after the Norse World Tree that connects all nine realms, it bridges embedding-based similarity search with structured graph relationships.

## Components

### Vector Store (`core.py`)

The vector store provides in-memory nearest neighbor search using brute-force comparison. It supports three distance metrics:

- **Cosine similarity** — measures angular distance, ideal for normalized embeddings
- **Euclidean distance** — measures geometric distance in vector space
- **Dot product** — measures projection, useful when magnitude matters

Vectors are stored in a dictionary keyed by ID with O(1) insert and lookup. Search is O(n) brute-force, which is appropriate for educational purposes and small-to-medium datasets.

**Collections** group vectors by namespace, allowing logical separation (e.g., "documents", "queries") while sharing the same dimensionality.

### Knowledge Graph (`graph.py`)

The knowledge graph stores labeled nodes and weighted directed edges with adjacency list representation for fast neighbor lookups. Key operations:

- **BFS shortest path** — finds the shortest path between two nodes
- **Subgraph extraction** — extracts a subset of the graph
- **Label and property queries** — filter nodes by type or attributes

### Memory Layer (`memory.py`)

The memory layer unifies vector and graph storage for AI agent use cases:

- **Store** — saves text with its embedding and optional graph relations
- **Recall** — retrieves memories by embedding similarity, weighted by importance
- **Forget** — removes memories and their graph connections

**Importance scoring** combines three factors:
1. Vector similarity to the query
2. Access frequency (logarithmic scaling)
3. Temporal recency (exponential decay)

## Data Flow

```
Text + Embedding + Relations
        |
        v
  +-----------+     +-----------------+
  | VectorStore| <-> |  KnowledgeGraph |
  | (search)   |     |  (relations)    |
  +-----------+     +-----------------+
        |                    |
        v                    v
    MemoryLayer (unified recall + importance scoring)
```

## Design Decisions

- **Pure Python** — no external dependencies for portability
- **In-memory** — simplicity over persistence (can be extended)
- **Brute-force search** — correct baseline before optimization
- **Dataclasses** — clean, typed data models
- **Python 3.9+** — broad compatibility
