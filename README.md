# Yggdrasil -- Vector-Graph Database

> **Norse Mythology**: The World Tree connecting all nine realms | Combined vector search and knowledge graph for AI memory

[![CI](https://github.com/MukundaKatta/yggdrasil/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/yggdrasil/actions/workflows/ci.yml)
[![GitHub Pages](https://img.shields.io/badge/Live_Demo-Visit_Site-blue?style=for-the-badge)](https://MukundaKatta.github.io/yggdrasil/)
[![GitHub](https://img.shields.io/github/license/MukundaKatta/yggdrasil?style=flat-square)](LICENSE)

## Overview

Yggdrasil is a pure-Python in-memory vector + graph database designed for AI agent memory. It combines HNSW-style vector similarity search with knowledge graph traversal and a self-learning memory layer with importance scoring.

**Tech Stack:** Python 3.9+ (zero external dependencies)

## Quick Start

```bash
git clone https://github.com/MukundaKatta/yggdrasil.git
cd yggdrasil
pip install -e ".[dev]"

# Run tests
make test
```

## Features

- **Vector Store** -- In-memory nearest neighbor search with cosine, euclidean, and dot product metrics
- **Knowledge Graph** -- Directed graph with labeled nodes, weighted edges, BFS shortest path, and subgraph extraction
- **Memory Layer** -- Unified store/recall/forget with importance scoring based on access frequency and recency
- **Collections** -- Namespace-based vector grouping with cross-namespace search
- **CLI** -- Command-line interface for insert, search, graph, and memory operations

## Usage

```python
from yggdrasil import VectorStore, Vector

# Create a vector store
store = VectorStore(dimension=3)
store.insert(Vector(id="doc1", values=[0.1, 0.9, 0.3], metadata={"type": "article"}))
store.insert(Vector(id="doc2", values=[0.8, 0.1, 0.2], metadata={"type": "query"}))

# Search for similar vectors
results = store.search([0.1, 0.85, 0.25], k=5, metric="cosine")
for vector, score in results:
    print(f"{vector.id}: {score:.4f}")
```

```python
from yggdrasil import KnowledgeGraph, Node, Edge

# Build a knowledge graph
graph = KnowledgeGraph()
graph.add_node(Node(id="odin", label="God", properties={"realm": "Asgard"}))
graph.add_node(Node(id="thor", label="God", properties={"realm": "Asgard"}))
graph.add_node(Node(id="mjolnir", label="Weapon"))
graph.add_edge(Edge(source="odin", target="thor", relation="father_of"))
graph.add_edge(Edge(source="thor", target="mjolnir", relation="wields"))

# Find shortest path
path = graph.shortest_path("odin", "mjolnir")  # ["odin", "thor", "mjolnir"]
```

```python
from yggdrasil import MemoryLayer

# Unified memory for AI agents
memory = MemoryLayer(dimension=3)
memory.store(
    text="Odin is the Allfather of the Norse gods",
    embedding=[0.1, 0.9, 0.3],
    relations=[("odin", "title", "allfather")],
)
results = memory.recall(query_embedding=[0.1, 0.85, 0.25], k=5)
```

## Project Structure

```
yggdrasil/
├── src/yggdrasil/
│   ├── __init__.py      -- Package exports
│   ├── core.py          -- Vector store and distance metrics
│   ├── graph.py         -- Knowledge graph implementation
│   ├── memory.py        -- Unified memory layer
│   ├── config.py        -- Configuration management
│   ├── cli.py           -- Command-line interface
│   └── __main__.py      -- Module entry point
├── tests/
│   ├── test_core.py     -- Vector store tests (20+)
│   ├── test_graph.py    -- Graph tests (15+)
│   └── test_memory.py   -- Memory layer tests (10+)
├── docs/
│   └── ARCHITECTURE.md  -- System architecture
├── pyproject.toml
├── Makefile
└── .github/workflows/ci.yml
```

## Live Demo

Visit the landing page: **https://MukundaKatta.github.io/yggdrasil/**

## License

MIT License -- 2026 Officethree Technologies

## Part of the Mythological Portfolio

This is project **#yggdrasil** in the [100-project Mythological Portfolio](https://github.com/MukundaKatta) by Officethree Technologies.
