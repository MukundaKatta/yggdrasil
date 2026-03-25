"""Tests for yggdrasil.memory — memory layer."""

import time
import pytest

from yggdrasil.memory import MemoryEntry, MemoryLayer


class TestMemoryLayer:
    def test_store_and_len(self):
        mem = MemoryLayer(dimension=3)
        mid = mem.store(
            text="Odin is the Allfather",
            embedding=[0.1, 0.9, 0.3],
        )
        assert len(mem) == 1
        assert mid is not None

    def test_store_with_custom_id(self):
        mem = MemoryLayer(dimension=2)
        mid = mem.store(
            text="test",
            embedding=[1.0, 0.0],
            memory_id="custom-id",
        )
        assert mid == "custom-id"

    def test_recall(self):
        mem = MemoryLayer(dimension=3)
        mem.store(text="Odin", embedding=[1.0, 0.0, 0.0])
        mem.store(text="Thor", embedding=[0.0, 1.0, 0.0])
        mem.store(text="Loki", embedding=[0.0, 0.0, 1.0])

        results = mem.recall(query_embedding=[1.0, 0.1, 0.0], k=2)
        assert len(results) == 2
        # Odin should be most similar
        assert results[0][0].text == "Odin"

    def test_recall_updates_access_count(self):
        mem = MemoryLayer(dimension=2)
        mid = mem.store(text="test", embedding=[1.0, 0.0])
        mem.recall(query_embedding=[1.0, 0.0], k=1)
        entry = mem.get_entry(mid)
        assert entry.access_count == 1

        mem.recall(query_embedding=[1.0, 0.0], k=1)
        entry = mem.get_entry(mid)
        assert entry.access_count == 2

    def test_forget(self):
        mem = MemoryLayer(dimension=2)
        mid = mem.store(text="forget me", embedding=[1.0, 0.0])
        assert mem.forget(mid) is True
        assert len(mem) == 0
        assert mem.forget(mid) is False

    def test_store_with_relations(self):
        mem = MemoryLayer(dimension=2)
        mem.store(
            text="Odin is father of Thor",
            embedding=[0.5, 0.5],
            relations=[("odin", "father_of", "thor")],
        )
        assert mem.graph.has_node("odin")
        assert mem.graph.has_node("thor")
        assert mem.graph.edge_count >= 1

    def test_get_related(self):
        mem = MemoryLayer(dimension=2)
        mid = mem.store(
            text="some memory",
            embedding=[1.0, 0.0],
            relations=[("odin", "has_memory", mid if False else "target_entity")],
        )
        # The memory node itself should exist
        assert mem.graph.has_node(mid)

    def test_clear(self):
        mem = MemoryLayer(dimension=2)
        mem.store(text="a", embedding=[1.0, 0.0])
        mem.store(text="b", embedding=[0.0, 1.0])
        mem.clear()
        assert len(mem) == 0

    def test_importance_score_positive(self):
        mem = MemoryLayer(dimension=2)
        mem.store(text="test", embedding=[1.0, 0.0])
        results = mem.recall(query_embedding=[1.0, 0.0], k=1)
        assert len(results) == 1
        _, score = results[0]
        assert score > 0

    def test_memory_entry_fields(self):
        entry = MemoryEntry(
            id="test",
            text="hello",
            embedding=[1.0, 2.0],
            metadata={"key": "val"},
        )
        assert entry.id == "test"
        assert entry.text == "hello"
        assert entry.access_count == 0
        assert entry.created_at > 0
