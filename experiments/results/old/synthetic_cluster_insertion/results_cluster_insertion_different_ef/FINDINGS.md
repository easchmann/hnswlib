# Experiment: Incremental Insertion vs. Full Rebuild

## What this experiment tests

This experiment asks: **If new data arrives and is inserted into an existing HNSW index without rebuilding, is the resulting graph worse than a fresh index built on all data?**

The index is built on 10M initial vectors across 8 tightly packed clusters (128-D Gaussian blobs). Two new clusters are then inserted (one placed close to the existing distribution (near) and one far away (far)). As a baseline we do a full rebuild on the entire data.

Three query conditions are compared across both index types:
- **In-distribution**: queries from the initial 8 clusters
- **Near cluster**: queries from the region of the newly inserted near cluster
- **Far cluster**: queries from the region of the newly inserted far cluster

Ground truth is computed against all data (initial + inserted) using faiss exact search.

---

## Parameters

| Parameter | Value |
|---|---|
| `dim` | 128 |
| `n_init` | 10,000,000 |
| `n_near` / `n_far` | 500,000 each |
| `k_init` | 8 |
| `cluster_std` | 0.3 |
| `center_spread` | 1.5 |
| `near_sigma` | 5.0 σ |
| `far_sigma` | 20.0 σ |
| `M` | 16 |
| `ef_construction` | 200 |
| `ef_search` (sweep) | 10, 20, 50, 100, 200, 500 |
| `k_neighbours` | 10 |
| `n_queries` | 5,000 per condition |
| GT backend | faiss GPU |

---

## Plots

### `distribution_pca.png`
PCA 2D projection of the full dataset. The 8 initial clusters (coloured dots) form a compact core. The near cluster (yellow/orange circles) lands adjacent to the existing distribution in PCA space. The far cluster (red circles) also appears nearby in PCA but this is only an artefact of PCA compression; in 128D the clusters are well separated. Triangles show the independently sampled query vectors for each condition.

### `metrics_by_condition.png`
The four plots show mean recall, entry-point distance, layer-1 visit count, and base-layer visited nodes across the three query conditions. Solid blue = incremental, dashed green = full rebuild.

**Key observations:**
- **Recall**: the incremental index scores 0.322 on in-distribution queries and 0.379 for the full rebuild, a small gap. For near and far queries both indices score ~0.405, essentially the same.
- **Entry-point distance**: ~553 for in-distribution queries, ~380 for near/far, for both the insertion index and the full rebuild, which shows that the same global entry point is used.
- **Layer-1 visits and base-layer nodes**: small differences, no strong trend.

### `lower_bound_traces.png`
Mean lowerBound convergence trace over 200 search iterations at ef=200.

**Key observations:**
- **In-distribution (left panel)**: the incremental index lowerBound starts at ~52, peaks briefly, then plateaus at ~42. It never converges to the true nearest distance. The full rebuild drops from ~24 to ~16 and converges cleanly. This is because the incremental index starts in the wrong region of the base-layer and the search exhausts its budget there.
- **Near cluster (centre)**: both lines overlap almost exactly, no notabe difference.
- **Far cluster (right)**: the full rebuild starts higher (~33) and converges slightly slower than incremental (~25 start), both reaching the same floor ~15. The new cluster vectors created during insertion appear to sit near the global entry point, giving the incremental index slightly better initial navigation to the far cluster.

### `ef_sweep_recall.png`
Recall@10 vs. ef_search for each query condition. Solid blue = incremental, dashed green = full rebuild.

**Key observations:**
- **In-distribution**: a persistent and widening gap between the two index conditions across all ef values. At ef=10 both score ~0.06; by ef=500 the full rebuild reaches 0.545 while incremental only reaches 0.455. This could be a structural failure: more compute cannot fix the navigational corruption caused by insertion.
- **Near and far**: lines overlap at every ef value. Incremental insertion handles these conditions as well as a full rebuild at all search budgets.

### `ef_sweep_metrics.png`
All metrics vs. ef_search. Entry-point distance and base-layer entry distance are flat across ef (they depend only on graph structure, not search budget). Base-layer visited nodes and candidates at termination scale with ef as expected. The in-distribution blue lines (incremental) sit above the near/far lines for base-layer entry distance at all ef, which shows that the navigational failure is structural.

### `recall_violin.png`
Per-query recall distributions at ef=200. In-distribution shows a lower median for incremental (0.30) vs. full rebuild (0.39), with the incremental violin having more mass near zero (many queries fail entirely). Near and far distributions are nearly identical across index conditions.

---

## Summary table (ef=200)

| Index | Query condition | Recall@10 | EP dist | BL entry dist | L1 visits | Base visited |
|---|---|---|---|---|---|---|
| Incremental | In-distribution | 0.3223 | 552.9 | 41.0 | 28.1 | 206.5 |
| Incremental | Near | 0.4054 | 385.3 | 15.7 | 27.3 | 205.5 |
| Incremental | Far | 0.4055 | 379.8 | 15.7 | 28.8 | 205.5 |
| Full rebuild | In-distribution | 0.3785 | 551.6 | 15.7 | 28.3 | 206.0 |
| Full rebuild | Near | 0.4033 | 377.6 | 15.7 | 27.9 | 205.6 |
| Full rebuild | Far | 0.4057 | 377.3 | 22.8 | 28.1 | 208.0 |

---

## Key findings

1. **Incremental insertion hurts in-distribution queries, not new-cluster queries.** The existing workload degrades by ~5.7 pp recall at ef=200 and ~9 pp at ef=500. Queries from newly inserted clusters are unaffected.

2. **The degradation is structural and ef-invariant.** The gap between incremental and full rebuild on in-distribution queries does not close as ef increases, actually it widens slightly. More search budget cannot compensate for upper-layer navigational failure.

3. **The point of falure could be entry distance.** After upper-layer descent, the incremental index arrives at a base-layer node 41 units away from the query, while the full rebuild arrives at ~16 units. The global entry point is the same node in both cases, so the difference is in upper-layer graph structure, not the starting node.

4. **The lowerBound freezes on the incremental index for in-distribution queries.** After 200 search iterations the lowerBound is still at ~42, far from convergence. The full rebuild converges to ~16. All search budget is wasted exploring the wrong neighbourhood.

5. **Newly inserted clusters benefit from insertion.** Their vectors get wired into the graph near the entry point during insertion, creating direct navigation shortcuts. This accidentally improves (or at least preserves) recall for those query conditions, contrary to what I expected.