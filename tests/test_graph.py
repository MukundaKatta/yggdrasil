"""Tests for yggdrasil.graph — knowledge graph."""

import pytest

from yggdrasil.graph import Edge, KnowledgeGraph, Node


class TestKnowledgeGraph:
    def test_add_node(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="odin", label="God"))
        assert g.node_count == 1
        assert g.has_node("odin")

    def test_add_duplicate_node_raises(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="odin"))
        with pytest.raises(KeyError):
            g.add_node(Node(id="odin"))

    def test_get_node(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="thor", label="God", properties={"weapon": "Mjolnir"}))
        node = g.get_node("thor")
        assert node is not None
        assert node.label == "God"
        assert node.properties["weapon"] == "Mjolnir"
        assert g.get_node("nonexistent") is None

    def test_remove_node(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="loki"))
        g.add_node(Node(id="odin"))
        g.add_edge(Edge(source="odin", target="loki", relation="father_of"))
        assert g.remove_node("loki") is True
        assert g.node_count == 1
        assert g.edge_count == 0
        assert g.remove_node("loki") is False

    def test_add_edge(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="odin"))
        g.add_node(Node(id="thor"))
        g.add_edge(Edge(source="odin", target="thor", relation="father_of"))
        assert g.edge_count == 1

    def test_add_edge_missing_node_raises(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="odin"))
        with pytest.raises(KeyError):
            g.add_edge(Edge(source="odin", target="thor"))

    def test_get_neighbors_outgoing(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="odin"))
        g.add_node(Node(id="thor"))
        g.add_node(Node(id="loki"))
        g.add_edge(Edge(source="odin", target="thor", relation="father_of"))
        g.add_edge(Edge(source="odin", target="loki", relation="father_of"))

        neighbors = g.get_neighbors("odin", direction="outgoing")
        assert len(neighbors) == 2
        neighbor_ids = {n.id for n, _ in neighbors}
        assert neighbor_ids == {"thor", "loki"}

    def test_get_neighbors_incoming(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="odin"))
        g.add_node(Node(id="thor"))
        g.add_edge(Edge(source="odin", target="thor", relation="father_of"))

        neighbors = g.get_neighbors("thor", direction="incoming")
        assert len(neighbors) == 1
        assert neighbors[0][0].id == "odin"

    def test_get_neighbors_filtered_by_relation(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="odin"))
        g.add_node(Node(id="thor"))
        g.add_node(Node(id="frigg"))
        g.add_edge(Edge(source="odin", target="thor", relation="father_of"))
        g.add_edge(Edge(source="odin", target="frigg", relation="married_to"))

        neighbors = g.get_neighbors("odin", relation="father_of")
        assert len(neighbors) == 1
        assert neighbors[0][0].id == "thor"

    def test_get_neighbors_nonexistent_raises(self):
        g = KnowledgeGraph()
        with pytest.raises(KeyError):
            g.get_neighbors("nonexistent")

    def test_shortest_path(self):
        g = KnowledgeGraph()
        for nid in ["a", "b", "c", "d"]:
            g.add_node(Node(id=nid))
        g.add_edge(Edge(source="a", target="b"))
        g.add_edge(Edge(source="b", target="c"))
        g.add_edge(Edge(source="c", target="d"))
        g.add_edge(Edge(source="a", target="d"))  # shortcut

        path = g.shortest_path("a", "d")
        assert path == ["a", "d"]

    def test_shortest_path_no_path(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="a"))
        g.add_node(Node(id="b"))
        # No edge
        assert g.shortest_path("a", "b") is None

    def test_shortest_path_same_node(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="a"))
        assert g.shortest_path("a", "a") == ["a"]

    def test_shortest_path_nonexistent_raises(self):
        g = KnowledgeGraph()
        with pytest.raises(KeyError):
            g.shortest_path("x", "y")

    def test_subgraph(self):
        g = KnowledgeGraph()
        for nid in ["a", "b", "c"]:
            g.add_node(Node(id=nid))
        g.add_edge(Edge(source="a", target="b", relation="r1"))
        g.add_edge(Edge(source="b", target="c", relation="r2"))
        g.add_edge(Edge(source="a", target="c", relation="r3"))

        sub = g.subgraph(["a", "b"])
        assert sub.node_count == 2
        assert sub.edge_count == 1  # only a->b

    def test_query_by_label(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="odin", label="God"))
        g.add_node(Node(id="thor", label="God"))
        g.add_node(Node(id="mjolnir", label="Weapon"))

        gods = g.query_by_label("God")
        assert len(gods) == 2

    def test_query_by_property(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="odin", properties={"realm": "Asgard"}))
        g.add_node(Node(id="hel", properties={"realm": "Helheim"}))

        asgardians = g.query_by_property("realm", "Asgard")
        assert len(asgardians) == 1
        assert asgardians[0].id == "odin"

    def test_get_edges_between(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="a"))
        g.add_node(Node(id="b"))
        g.add_edge(Edge(source="a", target="b", relation="r1"))
        g.add_edge(Edge(source="a", target="b", relation="r2"))

        edges = g.get_edges_between("a", "b")
        assert len(edges) == 2

    def test_clear(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="a"))
        g.add_node(Node(id="b"))
        g.add_edge(Edge(source="a", target="b"))
        g.clear()
        assert g.node_count == 0
        assert g.edge_count == 0

    def test_all_nodes_and_edges(self):
        g = KnowledgeGraph()
        g.add_node(Node(id="a"))
        g.add_node(Node(id="b"))
        g.add_edge(Edge(source="a", target="b"))
        assert len(g.all_nodes()) == 2
        assert len(g.all_edges()) == 1
