"""Tests for ConjugateGraph."""

import tempfile
import os

import numpy as np
import hnswlib

from src.drift.conjugate_graph import ConjugateGraph, ConjugateEdge


def test_add_and_lookup():
    cg = ConjugateGraph()
    cg.add_edge(0, 1, 0.5, t_added=0.0, epoch_added=0, current_epoch=0)
    edges = cg.lookup(0, current_epoch=0)
    assert len(edges) == 1
    assert edges[0].neighbor_id == 1


def test_max_neighbors_enforced():
    M = 4
    cg = ConjugateGraph(M_conj=M)
    for dst in range(M + 2):
        cg.add_edge(0, dst, float(dst), t_added=0.0, epoch_added=0, current_epoch=0)
    # traversal_count bumped by lookup, so reset before checking length
    edges = cg._edges.get(0, [])
    assert len(edges) <= M


def test_stale_edge_replaced():
    M = 4
    cg = ConjugateGraph(M_conj=M)
    # Fill node 0 at epoch 0, no traversals
    for dst in range(M):
        cg.add_edge(0, dst, float(dst + 1), t_added=0.0, epoch_added=0, current_epoch=0)

    # At epoch 100, all existing edges are very stale (staleness = 100)
    # New edge at epoch 100 has staleness = 0 — much lower, should replace oldest
    added = cg.add_edge(0, M + 99, 0.1, t_added=1.0, epoch_added=100, current_epoch=100)
    assert added, "Fresh edge should replace stale edge"
    ids = {e.neighbor_id for e in cg._edges[0]}
    assert M + 99 in ids, "New neighbor should be present after replacement"


def test_traversal_count_incremented():
    cg = ConjugateGraph()
    cg.add_edge(0, 1, 0.5, t_added=0.0, epoch_added=0, current_epoch=0)
    cg.lookup(0, current_epoch=1)
    cg.lookup(0, current_epoch=2)
    assert cg._edges[0][0].traversal_count == 2


def test_enhanced_search_finds_extra_neighbor():
    rng = np.random.default_rng(42)
    dim = 4
    n_base = 50
    base = rng.random((n_base, dim), dtype=np.float32)

    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n_base, ef_construction=50, M=8)
    index.add_items(base, list(range(n_base)))
    index.set_ef(20)

    q = rng.random(dim, dtype=np.float32)
    k = 5
    labels, distances = index.knn_query(q, k=k)
    labels = labels[0]
    distances = distances[0]

    # Find a vector closer to q than the worst HNSW result, NOT already in labels.
    # Use squared L2 throughout — hnswlib space="l2" returns squared L2 distances.
    diff = base - q
    dists_all = np.sum(diff ** 2, axis=1)  # squared L2
    worst_dist = distances[-1]              # squared L2 from hnswlib
    candidates = [i for i in range(n_base) if i not in labels and dists_all[i] < worst_dist]
    if not candidates:
        # Nothing to test: HNSW already found exact top-k
        return

    target = candidates[np.argmin([dists_all[i] for i in candidates])]

    cg = ConjugateGraph()
    # Add edge from first HNSW result to target (distance in squared L2)
    cg.add_edge(int(labels[0]), target, float(dists_all[target]),
                t_added=0.0, epoch_added=0, current_epoch=0)

    ids_out, dists_out = cg.enhanced_search(labels, distances, base, q, k, current_epoch=0)
    assert target in ids_out, "enhanced_search should surface the conjugate neighbor"
    assert len(ids_out) == k


def test_eviction_reduces_edge_count():
    cg = ConjugateGraph(M_conj=10)
    for node in range(20):
        for dst in range(5):
            cg.add_edge(node, dst + node * 5, 1.0,
                        t_added=0.0, epoch_added=0, current_epoch=0)
    assert cg._total_edges == 100
    evicted = cg.evict(current_epoch=50, n_evict=30)
    assert evicted == 30
    assert cg._total_edges == 70


def test_save_load_roundtrip():
    cg = ConjugateGraph(M_conj=4)
    cg.add_edge(0, 1, 0.5, t_added=0.1, epoch_added=2, current_epoch=2)
    cg.add_edge(0, 2, 0.8, t_added=0.2, epoch_added=3, current_epoch=3)
    cg.add_edge(1, 3, 1.2, t_added=0.3, epoch_added=4, current_epoch=4)
    # bump traversal
    cg.lookup(0, current_epoch=5)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        cg.save(path)
        cg2 = ConjugateGraph.load(path)

        assert cg2.M_conj == cg.M_conj
        assert cg2._total_edges == cg._total_edges
        assert set(cg2._edges.keys()) == set(cg._edges.keys())

        for node_id in cg._edges:
            orig = {e.neighbor_id: e for e in cg._edges[node_id]}
            loaded = {e.neighbor_id: e for e in cg2._edges[node_id]}
            assert orig.keys() == loaded.keys()
            for nid in orig:
                assert orig[nid].traversal_count == loaded[nid].traversal_count
                assert orig[nid].epoch_added == loaded[nid].epoch_added
    finally:
        os.unlink(path)
