"""Tests for yggdrasil.core — vector store and distance metrics."""

import math
import pytest

from yggdrasil.core import (
    Collection,
    Vector,
    VectorStore,
    cosine_similarity,
    dot_product,
    euclidean_distance,
)


class TestVector:
    def test_create_vector(self):
        v = Vector(id="v1", values=[1.0, 2.0, 3.0])
        assert v.id == "v1"
        assert v.dimension == 3
        assert v.metadata == {}

    def test_vector_with_metadata(self):
        v = Vector(id="v2", values=[1.0], metadata={"key": "val"})
        assert v.metadata["key"] == "val"

    def test_empty_vector_raises(self):
        with pytest.raises(ValueError):
            Vector(id="v3", values=[])

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            Vector(id="v4", values=["a", "b"])

    def test_norm(self):
        v = Vector(id="v5", values=[3.0, 4.0])
        assert abs(v.norm() - 5.0) < 1e-9

    def test_normalize(self):
        v = Vector(id="v6", values=[3.0, 4.0])
        n = v.normalize()
        assert abs(n.norm() - 1.0) < 1e-9


class TestDistanceMetrics:
    def test_cosine_identical(self):
        a = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-9

    def test_cosine_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_cosine_dimension_mismatch(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_euclidean_identical(self):
        a = [1.0, 2.0, 3.0]
        assert abs(euclidean_distance(a, a)) < 1e-9

    def test_euclidean_known(self):
        a = [0.0, 0.0]
        b = [3.0, 4.0]
        assert abs(euclidean_distance(a, b) - 5.0) < 1e-9

    def test_dot_product_known(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert abs(dot_product(a, b) - 32.0) < 1e-9

    def test_dot_product_dimension_mismatch(self):
        with pytest.raises(ValueError):
            dot_product([1.0], [1.0, 2.0])


class TestVectorStore:
    def test_insert_and_len(self):
        store = VectorStore(dimension=3)
        store.insert(Vector(id="a", values=[1.0, 0.0, 0.0]))
        store.insert(Vector(id="b", values=[0.0, 1.0, 0.0]))
        assert len(store) == 2

    def test_insert_duplicate_raises(self):
        store = VectorStore(dimension=2)
        store.insert(Vector(id="a", values=[1.0, 0.0]))
        with pytest.raises(KeyError):
            store.insert(Vector(id="a", values=[0.0, 1.0]))

    def test_insert_wrong_dimension_raises(self):
        store = VectorStore(dimension=3)
        with pytest.raises(ValueError):
            store.insert(Vector(id="a", values=[1.0, 0.0]))

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            VectorStore(dimension=0)

    def test_get(self):
        store = VectorStore(dimension=2)
        store.insert(Vector(id="a", values=[1.0, 2.0]))
        v = store.get("a")
        assert v is not None
        assert v.id == "a"
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = VectorStore(dimension=2)
        store.insert(Vector(id="a", values=[1.0, 0.0]))
        assert store.delete("a") is True
        assert store.delete("a") is False
        assert len(store) == 0

    def test_upsert(self):
        store = VectorStore(dimension=2)
        store.insert(Vector(id="a", values=[1.0, 0.0]))
        store.upsert(Vector(id="a", values=[0.0, 1.0]))
        v = store.get("a")
        assert v.values == [0.0, 1.0]

    def test_contains(self):
        store = VectorStore(dimension=2)
        store.insert(Vector(id="a", values=[1.0, 0.0]))
        assert "a" in store
        assert "b" not in store

    def test_list_ids(self):
        store = VectorStore(dimension=2)
        store.insert(Vector(id="a", values=[1.0, 0.0]))
        store.insert(Vector(id="b", values=[0.0, 1.0]))
        assert set(store.list_ids()) == {"a", "b"}

    def test_search_cosine(self):
        store = VectorStore(dimension=3)
        store.insert(Vector(id="a", values=[1.0, 0.0, 0.0]))
        store.insert(Vector(id="b", values=[0.0, 1.0, 0.0]))
        store.insert(Vector(id="c", values=[1.0, 0.1, 0.0]))

        results = store.search([1.0, 0.0, 0.0], k=2, metric="cosine")
        assert len(results) == 2
        assert results[0][0].id == "a"

    def test_search_euclidean(self):
        store = VectorStore(dimension=2)
        store.insert(Vector(id="a", values=[0.0, 0.0]))
        store.insert(Vector(id="b", values=[10.0, 10.0]))

        results = store.search([0.0, 0.1], k=1, metric="euclidean")
        assert results[0][0].id == "a"

    def test_search_dot(self):
        store = VectorStore(dimension=2)
        store.insert(Vector(id="a", values=[1.0, 0.0]))
        store.insert(Vector(id="b", values=[0.0, 1.0]))

        results = store.search([1.0, 0.0], k=1, metric="dot")
        assert results[0][0].id == "a"

    def test_search_invalid_metric(self):
        store = VectorStore(dimension=2)
        with pytest.raises(ValueError):
            store.search([1.0, 0.0], metric="unknown")

    def test_search_invalid_k(self):
        store = VectorStore(dimension=2)
        with pytest.raises(ValueError):
            store.search([1.0, 0.0], k=0)

    def test_search_with_metadata_filter(self):
        store = VectorStore(dimension=2)
        store.insert(
            Vector(id="a", values=[1.0, 0.0], metadata={"type": "doc"})
        )
        store.insert(
            Vector(id="b", values=[0.9, 0.1], metadata={"type": "query"})
        )
        results = store.search(
            [1.0, 0.0], k=10, filter_metadata={"type": "doc"}
        )
        assert len(results) == 1
        assert results[0][0].id == "a"

    def test_clear(self):
        store = VectorStore(dimension=2)
        store.insert(Vector(id="a", values=[1.0, 0.0]))
        store.clear()
        assert len(store) == 0


class TestCollection:
    def test_create_collection(self):
        col = Collection(name="test", dimension=3)
        assert col.name == "test"
        assert col.dimension == 3

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            Collection(name="", dimension=3)

    def test_get_or_create_namespace(self):
        col = Collection(name="test", dimension=3)
        ns = col.get_or_create_namespace("default")
        assert isinstance(ns, VectorStore)
        assert ns.dimension == 3
        # Same namespace returned on second call
        ns2 = col.get_or_create_namespace("default")
        assert ns is ns2

    def test_list_namespaces(self):
        col = Collection(name="test", dimension=2)
        col.get_or_create_namespace("ns1")
        col.get_or_create_namespace("ns2")
        assert set(col.list_namespaces()) == {"ns1", "ns2"}

    def test_delete_namespace(self):
        col = Collection(name="test", dimension=2)
        col.get_or_create_namespace("ns1")
        assert col.delete_namespace("ns1") is True
        assert col.delete_namespace("ns1") is False

    def test_total_vectors(self):
        col = Collection(name="test", dimension=2)
        ns1 = col.get_or_create_namespace("ns1")
        ns2 = col.get_or_create_namespace("ns2")
        ns1.insert(Vector(id="a", values=[1.0, 0.0]))
        ns2.insert(Vector(id="b", values=[0.0, 1.0]))
        assert col.total_vectors() == 2

    def test_search_all(self):
        col = Collection(name="test", dimension=2)
        ns1 = col.get_or_create_namespace("ns1")
        ns2 = col.get_or_create_namespace("ns2")
        ns1.insert(Vector(id="a", values=[1.0, 0.0]))
        ns2.insert(Vector(id="b", values=[0.9, 0.1]))

        results = col.search_all([1.0, 0.0], k=2)
        assert len(results) == 2
        # First result should be exact match from ns1
        assert results[0][1].id == "a"
        assert results[0][0] == "ns1"
