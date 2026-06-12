"""Tests for adapter.py."""

import numpy as np
import hnswlib

from src.drift.adapter import (
    find_repair_candidates,
    compute_candidate_edges,
    apply_repairs,
    update_node_eh_accumulator,
)
from src.drift.conjugate_graph import ConjugateGraph


def test_find_repair_candidates_in_hot_cells():
    # 100 nodes: first 50 in cell 0, next 50 in cell 1
    cell_labels = np.array([0] * 50 + [1] * 50, dtype=np.int32)
    result = find_repair_candidates(hot_cells=[0], cell_labels=cell_labels, eh_per_node=None)
    assert len(result) == 50
    assert np.all(cell_labels[result] == 0)


def test_find_repair_candidates_respects_max():
    cell_labels = np.zeros(500, dtype=np.int32)
    result = find_repair_candidates(
        hot_cells=[0], cell_labels=cell_labels, eh_per_node=None, max_nodes=100
    )
    assert len(result) <= 100


def test_eh_accumulator_increases():
    n_base = 20
    node_eh_accumulator = np.zeros(n_base, dtype=np.float64)
    # Node 5 appears in all query results with high EH
    result_ids_batch = [np.array([5, 3, 7]) for _ in range(10)]
    eh_values = np.ones(10) * 10.0  # high EH
    update_node_eh_accumulator(result_ids_batch, eh_values, node_eh_accumulator)
    assert node_eh_accumulator[5] > 0.0
    # Nodes not visited should remain 0
    assert node_eh_accumulator[0] == 0.0


def test_apply_repairs_adds_edges():
    cg = ConjugateGraph(M_conj=8)
    candidate_edges = {0: [(1, 0.5), (2, 0.8)]}
    stats = apply_repairs(
        candidate_edges, cg,
        t_added=0.0, epoch_added=0, current_epoch=0,
        rng_relaxation=1.5,
    )
    edges = cg._edges.get(0, [])
    assert len(edges) > 0
    assert stats["edges_added"] > 0


def test_no_self_loops_in_candidates():
    rng = np.random.default_rng(7)
    dim = 8
    n = 30
    base = rng.random((n, dim), dtype=np.float32)

    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=50, M=8)
    index.add_items(base, list(range(n)))

    repair_nodes = np.arange(n, dtype=np.int64)
    candidates = compute_candidate_edges(repair_nodes, index, base, M_candidates=10, ef_search=30)

    for node_id, neighbors in candidates.items():
        for (nid, _) in neighbors:
            assert nid != node_id, f"Self-loop found at node {node_id}"
