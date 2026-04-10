# Experiment: Pure Distribution Shift — Single Blob, No Insertion

## What this experiment tests

This experiment uses the same single-blob geometry as the blob insertion experiment
but **does not insert the new cluster vectors into the index**. The index stays
unchanged at 10M vectors. Ground truth is computed against the original index data
only, so the true nearest neighbours of near/far queries are the closest vectors in
the initial distribution.

This isolates navigational failure: how well does an HNSW index built on one
distribution serve queries from a shifted distribution, with no graph update?

The same two index conditions are compared, the original incremental index (unchanged
after initial construction) and a full rebuild on the **entire data**.

---

## Parameters

| Parameter | Value |
|---|---|
| `dim` | 128 |
| `n_init` | 10,000,000 |
| `n_near` / `n_far` | 100,000 each (generated, not inserted) |
| `k_init` | 1 (single blob at origin) |
| `cluster_std` | 3.0 |
| `center_spread` | 0.0 |
| `near_sigma` | 4.0 σ (centre at 12 units) |
| `far_sigma` | 20.0 σ (centre at 60 units) |
| `M` | 16 |
| `ef_construction` | 200 |
| `ef_search` (sweep) | 10, 20, 50, 100, 200, 500 |
| `k_neighbours` | 10 |
| `n_queries` | 5,000 per condition |
| GT backend | faiss GPU |
| GT pool | initial data only |

---

## Plots

### `distribution_pca.png`
Same layout as the blob insertion experiment. The initial 10M blob (blue, right),
near cluster (orange, upper right — generated but not in the index), far cluster
(red, left — generated but not in the index). The filled circles represent the
distributions queries are drawn from; none of these are in the index. The far
cluster's large visual size in PCA reflects its dominance of variance at large
distances, not its actual vector count.

### `metrics_by_condition.png`
Plots at ef=20. Solid blue = incremental (original index), dashed
green = full rebuild (on initial data only).

**Key observations:**
- **Recall@10**: incremental scores 0.040 (in-dist), 0.028 (near), **0.000** (far).
  Full rebuild scores 0.037 (in-dist), 0.029 (near), **0.163** (far). The far
  cluster recall gap is very large, incremental finds zero correct neighbours,
  full rebuild finds 16% at ef=20.
- **Entry-point distance**: ~2455 (in-dist, incremental), ~2652 (near, incremental),
  ~6086 (far, incremental). Full rebuild has substantially lower entry-point
  distances: ~2116 (in-dist), ~2330 (near), ~6007 (far). The full rebuild produces
  a different global entry point that is on average closer to all query conditions.
- **Base-layer entry distance**: 1521 (in-dist), 1637 (near), **4647** (incremental
  far) vs **3962** (full rebuild far). After upper-layer descent, the incremental
  index arrives 685 units further from far cluster queries than the full rebuild.
  This is the direct cause of zero recall, the search is started so far from the
  true neighbourhood that ef=20 cannot bridge the gap at all.
- **Layer-1 visits**: incremental far = 23.0, full rebuild far = 28.6. The full
  rebuild explores more upper-layer neighbours when descending toward the far cluster.
- **Base-layer visited nodes**: identical for in-dist and near; full rebuild far
  visits ~34 nodes vs incremental far ~27.5 — the full rebuild base-layer search
  explores further because it starts in a better location.

### `ef_sweep_recall.png`
Recall@10 vs. ef_search. The far cluster panel (right) is the most striking plot. 
The full rebuild (dashed green) rises from 0.073 at ef=10
to 0.876 at ef=500, good recall achievable with enough search budget. The
incremental index (solid blue) stays at exactly **0.000 at every ef value from 10 to 500**. 
No amount of additional search budget would improve recall for the far
cluster on the incremental index. In-distribution and near panels show overlapping
lines with no gap between index conditions.

### `ef_sweep_metrics.png`
Five different metrics sweep (different EF value).
Entry-point distance and base-layer entry distance are flat (fixed by graph structure, independent of ef). 
The incremental index far cluster lines (solid red) are at 6086 EP distance and 4647 BL entry distance
across all ef, the search always starts in the same wrong location regardless of
how much it explores from there. Base-layer visited nodes scale with ef; the far
cluster incremental and full rebuild lines diverge at high ef, with full rebuild
visiting slightly more nodes (better starting position allows more productive
exploration).

### `lower_bound_traces.png`
Mean lowerBound convergence at ef=20 (~20 iterations). The far cluster panel (right)
shows the starkest separation across all experiments. The incremental index starts
at lowerBound ~5500 and barely moves — it plateaus at ~4700 within 3 iterations
and stays there for all 20 iterations. The search has converged to a local minimum
far from the true neighbourhood and cannot escape. The full rebuild starts at ~4700
and descends to ~3200, showing genuine convergence toward the true neighbourhood.
In-distribution and near panels show both conditions overlapping.

### `recall_violin.png`
Per-query recall distributions at ef=20. The far cluster panel shows the complete
failure of the incremental index — the violin is a flat line at zero, meaning every
single one of the 5,000 far cluster queries returns recall = 0. The full rebuild
far cluster violin extends from 0 to 0.8 with a median around 0.1, showing high
variability but genuine partial recall. In-distribution and near violins are
concentrated near zero for both conditions, consistent with the low mean recall.

---

## Summary table (ef=20)

| Index | Query condition | Recall@10 | EP dist | BL entry dist | L1 visits | Base visited |
|---|---|---|---|---|---|---|
| Incremental | In-distribution | 0.0398 | 2454.8 | 1521.1 | 27.1 | 27.5 |
| Incremental | Near | 0.0276 | 2651.6 | 1637.2 | 26.7 | 27.3 |
| Incremental | **Far** | **0.0000** | **6086.2** | **4646.7** | **23.0** | **27.5** |
| Full rebuild | In-distribution | 0.0371 | 2116.1 | 1520.7 | 26.5 | 27.2 |
| Full rebuild | Near | 0.0287 | 2330.2 | 1632.5 | 26.3 | 27.4 |
| Full rebuild | **Far** | **0.1625** | **6006.6** | **3961.9** | **28.6** | **34.0** |

---

## Key findings

1. **Incremental index recall on far cluster queries is zero at all ef
   values.** This is the most extreme navigational failure observed yet.
   The graph has no navigational pathway toward the far cluster region, 
   not even with increased search budget (ef=500).

2. **The full rebuild on the same initial data achieves 87.6% recall at ef=500.**
   Both index conditions use identical data, identical parameters, and identical
   query sets. The only difference is the graph structure. The full rebuild produces
   a different global entry point and upper-layer topology that supports navigation
   to the far cluster region; the incremental index does not.

3. **The failure is caused by base-layer entry distance, not entry-point distance.**
   Entry-point distance is similar between both conditions for far cluster queries
   (~6086 vs ~6007). The base-layer entry distance differs: 4647
   (incremental) vs 3962 (full rebuild). The incremental index's upper layers
   navigate 685 units further from the true neighbourhood than the full rebuild's
   upper layers. The lowerBound then freezes around iterations 3-5 and the search
   exhausts its budget in the wrong region.

4. **In-distribution and near queries are unaffected by index condition.**
   The two index conditions perform identically for in-distribution and near cluster
   queries at all ef values. The structural difference only manifests for the more
   extreme distribution shift (far cluster at 20σ from the origin).

## Relationship to other experiments

This experiment provides the strongest evidence that HNSW upper-layer structure,
not just data coverage, determines out-of-distribution recall. The blob insertion
experiment shows that inserting far cluster vectors partially compensates (recall
reaches 0.16 at ef=20 for incremental). This experiment without insertion of the new data shows that
without insertion, the incremental index achieves zero recall for the same
query set. ->Identifying the graph structure as the bottle neck??