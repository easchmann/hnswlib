"""Parity test: hnswlib.ConjugateGraph (C++) vs src.drift.conjugate_graph.ConjugateGraph (Python)"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import hnswlib
from src.drift.conjugate_graph import ConjugateEdge, ConjugateGraph as PyConjugateGraph

N_NODES = 200
DIM = 16
M_CONJ = 6
# kept well above what the add_edge sequence below can ever fill (N_NODES * M_CONJ),
# so add_edge never triggers the global evict() pass — its eviction order depends on
# dict-insertion order in Python vs unordered_map order in C++, which are not
# guaranteed to match under staleness ties. Node-level eviction (age-only, exercised
# below) has no such ambiguity: it only compares within one node's own edge list.
MAX_TOTAL_EDGES = 100_000
N_ADD_CALLS = 2000


def build_seeded_graphs(seed):
    rng = np.random.default_rng(seed)
    py_g = PyConjugateGraph(M_conj=M_CONJ, max_total_edges=MAX_TOTAL_EDGES)
    cpp_g = hnswlib.ConjugateGraph(M_CONJ, MAX_TOTAL_EDGES)

    for i in range(N_ADD_CALLS):
        src = int(rng.integers(0, N_NODES))
        dst = int(rng.integers(0, N_NODES))
        distance = float(rng.random())
        t_added = float(i) / N_ADD_CALLS
        epoch_added = int(rng.integers(0, 25))
        current_epoch = epoch_added

        py_ok = py_g.add_edge(src, dst, distance, t_added, epoch_added, current_epoch)
        cpp_ok = cpp_g.add_edge(src, dst, distance, t_added, epoch_added, current_epoch)
        assert py_ok == cpp_ok, f"add_edge return mismatch at call {i}: py={py_ok} cpp={cpp_ok}"

    return rng, py_g, cpp_g


def py_edges_as_tuples(py_g):
    return [
        (node_id, e.neighbor_id, e.distance, e.t_added, e.epoch_added, e.traversal_count)
        for node_id, neighbors in py_g._edges.items()
        for e in neighbors
    ]


def group_by_src(edge_tuples):
    by_src = {}
    for src, neighbor_id, distance, t_added, epoch_added, traversal_count in edge_tuples:
        by_src.setdefault(src, []).append((neighbor_id, distance, t_added, epoch_added, traversal_count))
    return by_src


def assert_edge_graphs_equal(py_edge_tuples, cpp_edge_tuples):
    # Compare per-node, in insertion order (guaranteed identical between the two
    # backends given an identical add_edge call sequence). distance/t_added are
    # compared via a round-trip through np.float32, since pybind11 truncates the
    # incoming Python float64 to a C++ float — comparing raw rounded decimals is
    # fragile near rounding boundaries 
    py_by_src = group_by_src(py_edge_tuples)
    cpp_by_src = group_by_src(cpp_edge_tuples)
    assert py_by_src.keys() == cpp_by_src.keys()

    for src in py_by_src:
        py_list = py_by_src[src]
        cpp_list = cpp_by_src[src]
        assert len(py_list) == len(cpp_list), f"src={src}: length mismatch"
        for (p_n, p_d, p_t, p_e, p_c), (c_n, c_d, c_t, c_e, c_c) in zip(py_list, cpp_list):
            assert p_n == c_n and p_e == c_e and p_c == c_c
            assert np.float32(p_d) == np.float32(c_d)
            assert np.float32(p_t) == np.float32(c_t)


def test_add_edge_sequence_produces_identical_graph_state():
    _, py_g, cpp_g = build_seeded_graphs(seed=1234)
    assert_edge_graphs_equal(py_edges_as_tuples(py_g), cpp_g.get_all_edges())


@pytest.mark.parametrize("two_hop", [True, False])
def test_enhanced_search_matches(two_hop):
    rng, py_g, cpp_g = build_seeded_graphs(seed=5678)
    base = rng.random((N_NODES, DIM)).astype(np.float32)

    for _ in range(5):
        query_vec = rng.random(DIM).astype(np.float32)
        seed_ids = rng.choice(N_NODES, size=5, replace=False)
        seed_dists = np.array(
            [float(np.dot(query_vec - base[sid], query_vec - base[sid])) for sid in seed_ids],
            dtype=np.float32,
        )
        current_epoch = int(rng.integers(0, 25))

        py_ids, py_dists = py_g.enhanced_search(
            seed_ids, seed_dists, base, query_vec, k=10, current_epoch=current_epoch, two_hop=two_hop
        )
        cpp_ids, cpp_dists = cpp_g.enhanced_search(
            seed_ids.astype(np.int64), seed_dists, base, query_vec, k=10, current_epoch=current_epoch, two_hop=two_hop
        )

        assert list(py_ids) == list(cpp_ids)
        np.testing.assert_allclose(py_dists, cpp_dists, atol=1e-5)


def test_save_load_json_roundtrip_matches_bulk_load(tmp_path):
    _, py_g, cpp_g = build_seeded_graphs(seed=91)
    cpp_edges = cpp_g.get_all_edges()

    # Feed the C++ backend's edges through the EXISTING Python save() JSON writer.
    shadow = PyConjugateGraph(M_conj=M_CONJ, max_total_edges=MAX_TOTAL_EDGES)
    for src, neighbor_id, distance, t_added, epoch_added, traversal_count in cpp_edges:
        shadow._edges.setdefault(src, []).append(
            ConjugateEdge(neighbor_id, distance, t_added, epoch_added, traversal_count)
        )
    shadow._total_edges = len(cpp_edges)

    json_path = tmp_path / "cg_state.json"
    shadow.save(str(json_path))

    # sanity check: this is genuinely the schema used by experiments/09_latency_benchmark/cg_states/*.json
    with open(json_path) as f:
        raw = json.load(f)
    assert set(raw.keys()) == {"M_conj", "max_total_edges", "edges"}

    loaded_py = PyConjugateGraph.load(str(json_path))

    fresh_cpp = hnswlib.ConjugateGraph(M_CONJ, MAX_TOTAL_EDGES)
    fresh_cpp.bulk_load_edges(cpp_edges)

    assert_edge_graphs_equal(py_edges_as_tuples(loaded_py), fresh_cpp.get_all_edges())
    assert_edge_graphs_equal(py_edges_as_tuples(loaded_py), cpp_edges)
