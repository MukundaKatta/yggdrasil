"""
Command-line interface for Yggdrasil.

Provides commands for vector insertion, search, graph manipulation,
and memory management.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from yggdrasil.config import Config
from yggdrasil.core import Collection, Vector, VectorStore
from yggdrasil.graph import Edge, KnowledgeGraph, Node
from yggdrasil.memory import MemoryLayer


def _parse_float_list(s: str) -> List[float]:
    """Parse a comma-separated string into a list of floats."""
    return [float(x.strip()) for x in s.split(",")]


def cmd_insert(args: argparse.Namespace) -> None:
    """Handle the 'insert' command."""
    values = _parse_float_list(args.values)
    metadata = json.loads(args.metadata) if args.metadata else {}

    store = VectorStore(dimension=len(values))
    vec = Vector(id=args.id, values=values, metadata=metadata)
    store.insert(vec)
    print(f"Inserted vector '{args.id}' with dimension {len(values)}")


def cmd_search(args: argparse.Namespace) -> None:
    """Handle the 'search' command."""
    query = _parse_float_list(args.query)
    print(
        f"Search for k={args.k} neighbors using {args.metric} metric"
    )
    print(f"Query vector dimension: {len(query)}")
    print("(In-memory store is ephemeral; use the library API for persistence)")


def cmd_graph(args: argparse.Namespace) -> None:
    """Handle the 'graph' command."""
    graph = KnowledgeGraph()
    if args.action == "add-node":
        props = json.loads(args.properties) if args.properties else {}
        node = Node(id=args.id, label=args.label or "", properties=props)
        graph.add_node(node)
        print(f"Added node '{args.id}' with label '{node.label}'")
    elif args.action == "add-edge":
        edge = Edge(
            source=args.source,
            target=args.target,
            relation=args.relation or "",
        )
        print(
            f"Edge: {edge.source} --[{edge.relation}]--> {edge.target}"
        )
    elif args.action == "info":
        print(
            f"Graph: {graph.node_count} nodes, {graph.edge_count} edges"
        )


def cmd_memory(args: argparse.Namespace) -> None:
    """Handle the 'memory' command."""
    if args.action == "store":
        embedding = _parse_float_list(args.embedding)
        print(
            f"Storing memory with {len(embedding)}-dim embedding"
        )
    elif args.action == "recall":
        query = _parse_float_list(args.query)
        print(
            f"Recalling top {args.k} memories (dim={len(query)})"
        )
    elif args.action == "forget":
        print(f"Forgetting memory '{args.id}'")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="yggdrasil",
        description="Yggdrasil - Vector-Graph Database CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="yggdrasil 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Insert command
    insert_p = subparsers.add_parser("insert", help="Insert a vector")
    insert_p.add_argument("--id", required=True, help="Vector ID")
    insert_p.add_argument(
        "--values", required=True, help="Comma-separated float values"
    )
    insert_p.add_argument(
        "--metadata", default=None, help="JSON metadata string"
    )

    # Search command
    search_p = subparsers.add_parser("search", help="Search vectors")
    search_p.add_argument(
        "--query", required=True, help="Comma-separated query vector"
    )
    search_p.add_argument(
        "--k", type=int, default=10, help="Number of results"
    )
    search_p.add_argument(
        "--metric", default="cosine", help="Distance metric"
    )

    # Graph command
    graph_p = subparsers.add_parser("graph", help="Graph operations")
    graph_p.add_argument(
        "action",
        choices=["add-node", "add-edge", "info"],
        help="Graph action",
    )
    graph_p.add_argument("--id", default=None, help="Node ID")
    graph_p.add_argument("--label", default=None, help="Node label")
    graph_p.add_argument(
        "--properties", default=None, help="JSON properties"
    )
    graph_p.add_argument("--source", default=None, help="Edge source")
    graph_p.add_argument("--target", default=None, help="Edge target")
    graph_p.add_argument(
        "--relation", default=None, help="Edge relation"
    )

    # Memory command
    memory_p = subparsers.add_parser("memory", help="Memory operations")
    memory_p.add_argument(
        "action",
        choices=["store", "recall", "forget"],
        help="Memory action",
    )
    memory_p.add_argument("--id", default=None, help="Memory ID")
    memory_p.add_argument(
        "--embedding", default=None, help="Comma-separated embedding"
    )
    memory_p.add_argument(
        "--query", default=None, help="Comma-separated query embedding"
    )
    memory_p.add_argument(
        "--k", type=int, default=10, help="Number of results"
    )
    memory_p.add_argument("--text", default=None, help="Memory text")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "insert": cmd_insert,
        "search": cmd_search,
        "graph": cmd_graph,
        "memory": cmd_memory,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
