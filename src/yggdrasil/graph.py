"""
Knowledge graph implementation for Yggdrasil.

Provides an in-memory directed graph database with labeled nodes,
weighted edges, BFS shortest-path search, and subgraph extraction.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Node:
    """A node in the knowledge graph.

    Attributes:
        id: Unique identifier for the node.
        label: A type label (e.g. 'Person', 'Concept').
        properties: Arbitrary key-value properties.
    """

    id: str
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A directed edge in the knowledge graph.

    Attributes:
        source: The source node ID.
        target: The target node ID.
        relation: The relationship type (e.g. 'knows', 'contains').
        weight: Numeric weight for the edge.
        properties: Arbitrary key-value properties.
    """

    source: str
    target: str
    relation: str = ""
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """In-memory directed knowledge graph.

    Supports adding nodes and edges, querying neighbors, finding
    shortest paths via BFS, extracting subgraphs, and filtering
    by label or property.

    Example::

        graph = KnowledgeGraph()
        graph.add_node(Node(id="odin", label="God", properties={"realm": "Asgard"}))
        graph.add_node(Node(id="thor", label="God", properties={"realm": "Asgard"}))
        graph.add_edge(Edge(source="odin", target="thor", relation="father_of"))
        neighbors = graph.get_neighbors("odin")
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        # Adjacency lists for fast lookup
        self._outgoing: Dict[str, List[Edge]] = {}
        self._incoming: Dict[str, List[Edge]] = {}

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self._edges)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Raises:
            KeyError: If a node with the same ID already exists.
        """
        if node.id in self._nodes:
            raise KeyError(f"Node '{node.id}' already exists")
        self._nodes[node.id] = node
        self._outgoing.setdefault(node.id, [])
        self._incoming.setdefault(node.id, [])

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID, or None if not found."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        return node_id in self._nodes

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connected edges.

        Returns True if the node existed.
        """
        if node_id not in self._nodes:
            return False
        # Remove edges involving this node
        self._edges = [
            e
            for e in self._edges
            if e.source != node_id and e.target != node_id
        ]
        # Rebuild adjacency for affected neighbors
        for edge_list in self._outgoing.values():
            edge_list[:] = [e for e in edge_list if e.target != node_id]
        for edge_list in self._incoming.values():
            edge_list[:] = [e for e in edge_list if e.source != node_id]
        del self._outgoing[node_id]
        del self._incoming[node_id]
        del self._nodes[node_id]
        return True

    def add_edge(self, edge: Edge) -> None:
        """Add a directed edge between two existing nodes.

        Raises:
            KeyError: If either the source or target node does not exist.
        """
        if edge.source not in self._nodes:
            raise KeyError(f"Source node '{edge.source}' does not exist")
        if edge.target not in self._nodes:
            raise KeyError(f"Target node '{edge.target}' does not exist")
        self._edges.append(edge)
        self._outgoing[edge.source].append(edge)
        self._incoming[edge.target].append(edge)

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "outgoing",
        relation: Optional[str] = None,
    ) -> List[Tuple[Node, Edge]]:
        """Get neighboring nodes and their connecting edges.

        Args:
            node_id: The node to find neighbors for.
            direction: 'outgoing', 'incoming', or 'both'.
            relation: Optional filter by edge relation type.

        Returns:
            List of (neighbor_node, edge) tuples.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' does not exist")

        results: List[Tuple[Node, Edge]] = []

        if direction in ("outgoing", "both"):
            for edge in self._outgoing.get(node_id, []):
                if relation is None or edge.relation == relation:
                    neighbor = self._nodes[edge.target]
                    results.append((neighbor, edge))

        if direction in ("incoming", "both"):
            for edge in self._incoming.get(node_id, []):
                if relation is None or edge.relation == relation:
                    neighbor = self._nodes[edge.source]
                    results.append((neighbor, edge))

        return results

    def shortest_path(
        self, start_id: str, end_id: str
    ) -> Optional[List[str]]:
        """Find the shortest path between two nodes using BFS.

        Returns:
            A list of node IDs from start to end, or None if no path exists.
        """
        if start_id not in self._nodes:
            raise KeyError(f"Start node '{start_id}' does not exist")
        if end_id not in self._nodes:
            raise KeyError(f"End node '{end_id}' does not exist")

        if start_id == end_id:
            return [start_id]

        visited: Set[str] = {start_id}
        queue: collections.deque = collections.deque()
        queue.append([start_id])

        while queue:
            path = queue.popleft()
            current = path[-1]

            for edge in self._outgoing.get(current, []):
                neighbor_id = edge.target
                if neighbor_id == end_id:
                    return path + [neighbor_id]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(path + [neighbor_id])

        return None

    def subgraph(self, node_ids: List[str]) -> "KnowledgeGraph":
        """Extract a subgraph containing only the specified nodes.

        Includes all edges between nodes in the set.

        Args:
            node_ids: The node IDs to include.

        Returns:
            A new KnowledgeGraph containing the subgraph.
        """
        sub = KnowledgeGraph()
        id_set = set(node_ids)

        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is not None:
                sub.add_node(
                    Node(
                        id=node.id,
                        label=node.label,
                        properties=dict(node.properties),
                    )
                )

        for edge in self._edges:
            if edge.source in id_set and edge.target in id_set:
                if sub.has_node(edge.source) and sub.has_node(edge.target):
                    sub.add_edge(
                        Edge(
                            source=edge.source,
                            target=edge.target,
                            relation=edge.relation,
                            weight=edge.weight,
                            properties=dict(edge.properties),
                        )
                    )

        return sub

    def query_by_label(self, label: str) -> List[Node]:
        """Find all nodes with a given label."""
        return [n for n in self._nodes.values() if n.label == label]

    def query_by_property(
        self, key: str, value: Any
    ) -> List[Node]:
        """Find all nodes where a property matches the given value."""
        return [
            n
            for n in self._nodes.values()
            if n.properties.get(key) == value
        ]

    def get_edges_between(
        self, source_id: str, target_id: str
    ) -> List[Edge]:
        """Get all edges from source to target."""
        return [
            e
            for e in self._outgoing.get(source_id, [])
            if e.target == target_id
        ]

    def all_nodes(self) -> List[Node]:
        """Return all nodes in the graph."""
        return list(self._nodes.values())

    def all_edges(self) -> List[Edge]:
        """Return all edges in the graph."""
        return list(self._edges)

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
        self._outgoing.clear()
        self._incoming.clear()
