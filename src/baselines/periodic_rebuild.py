"""Periodic full-rebuild baseline: rebuild HNSW from scratch every N epochs."""

import time

import hnswlib
import numpy as np


class PeriodicRebuildBaseline:

    def __init__(self, base, M=16, ef_construction=200,
                 rebuild_interval=5, space="l2", seed=42):
        self.base = base
        self.M = M
        self.ef_construction = ef_construction
        self.rebuild_interval = rebuild_interval
        self.space = space
        self.seed = seed
        self._rebuild_times = []
        self.index, _ = self.build_index()

    def build_index(self):
        t0 = time.perf_counter()
        index = hnswlib.Index(space=self.space, dim=self.base.shape[1])
        index.init_index(
            max_elements=len(self.base),
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=self.seed,
        )
        index.add_items(self.base, list(range(len(self.base))))
        elapsed = time.perf_counter() - t0
        return index, elapsed

    def process_epoch(self, epoch_idx, queries, groundtruth, k, ef_search):
        rebuilt_this_epoch = False
        rebuild_time = 0.0

        if epoch_idx > 0 and epoch_idx % self.rebuild_interval == 0:
            self.index, rebuild_time = self.build_index()
            self._rebuild_times.append(rebuild_time)
            rebuilt_this_epoch = True

        self.index.set_ef(ef_search)
        query_times = []
        hits = []
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            labels, _ = self.index.knn_query(q.reshape(1, -1), k=k)
            query_times.append((time.perf_counter() - t0) * 1000)
            true_set = set(groundtruth[i, :k].tolist())
            hits.append(len(set(labels[0].tolist()) & true_set) / k)

        mean_latency_ms = float(np.mean(query_times))
        return {
            "epoch_idx": epoch_idx,
            "recall_at_k": float(np.mean(hits)),
            "mean_latency_ms": mean_latency_ms,
            "queries_per_second": 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0,
            "rebuilt_this_epoch": rebuilt_this_epoch,
            "rebuild_time_seconds": rebuild_time,
        }

    def total_rebuild_time(self):
        return sum(self._rebuild_times)
