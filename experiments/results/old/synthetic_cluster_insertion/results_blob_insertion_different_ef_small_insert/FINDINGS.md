# Experiment: Single Blob with Minority Cluster Insertion (Adapted Parameters)

## What this experiment tests

This experiment uses a **single large Gaussian blob** as the initial distribution,
with two small new clusters inserted (one nearby and one far away). The idea behind this 
design was to hopefully place the global entry point near the centre of the distribution, 
making the geometric distance to out-of-distribution queries clean and interpretable.

Two index conditions are compared (incremental insertion vs. full rebuild) across
three query conditions (in-distribution, near, far). Ground truth is computed against
all data including inserted vectors.

This is the second blob run with corrected parameters after the first attempt failed
to show a signal due to a data size imbalance and near cluster placement inside the
initial blob.

---

## Parameters

| Parameter | Value |
|---|---|
| `dim` | 128 |
| `n_init` | 10,000,000 |
| `n_near` / `n_far` | 100,000 each (1% of initial) |
| `k_init` | 1 (single blob at origin) |
| `cluster_std` | 3.0 |
| `center_spread` | 0.0 |
| `near_sigma` | 4.0 σ (centre at 12 units — just outside blob edge) |
| `far_sigma` | 20.0 σ (centre at 60 units — clearly isolated) |
| `M` | 16 |
| `ef_construction` | 200 |
| `ef_search` (sweep) | 10, 20, 50, 100, 200, 500 |
| `k_neighbours` | 10 |
| `n_queries` | 5,000 per condition |
| GT backend | faiss GPU |

---

## Plots

### `distribution_pca.png`
PCA 2D projection. The initial 10M blob (blue) occupies the right half.
The near cluster (orange) overlaps with the upper portion of the blob —
in 128D it sits just outside the 3σ edge, but PCA compression makes it appear
partially overlapping. The far cluster (red) is completely separated with
a clear gap. Triangles show query vectors. Far queries are only scattered across the left
blob, while near and in-distribution queries overlap on the right.

### `ef_sweep_metrics.png`
The plots show all metrics vs. ef_search. Solid = incremental, dashed = full rebuild.
Blue = in-distribution, orange = near, red = far.

**Key observations:**
- **Recall@10**: far cluster queries (red) achieve significantly higher recall than
  in-distribution and near at all ef values. At ef=500: far reaches 0.875 (both
  conditions), in-distribution reaches 0.287, near reaches 0.247. This is because 
  far cluster queries find their own 100k inserted vectors as nearest
  neighbours very easily since those vectors form a dense isolated subgraph. The
  ground truth for far queries is the 100k far vectors; for in-distribution queries the 
  ground truth is spread across 10M vectors with many equally close candidates.
- **Entry-point distance**: flat at ~2364 (in-dist, incremental), ~2494 (near,
  incremental), ~5909 (far, incremental). The full rebuild has ~76 lower entry-point
  distance for the far cluster (5259 vs 5909) — after rebuilding, the far cluster
  vectors integrate into the graph near a new entry point, reducing the global
  entry-point distance to those queries.
- **Base-layer entry distance**: ~1524 (in-dist), ~1642 (near), ~3972 (incremental
  far) vs ~3543 (full rebuild far). The full rebuild arrives ~430 units closer to
  far cluster queries after upper-layer descent — the same directional advantage as
  seen in the large multi-cluster experiment, but here it translates to higher recall
  for the full rebuild at low ef.
- **Base-layer visited nodes**: scale with ef as expected. Far cluster consistently
  visits more nodes probably due to the larger search radius needed to reach its isolated region.
- **Candidates at termination**: far cluster has more remaining candidates at all ef
  — consistent with a harder, more spread-out search.

### `ef_sweep_recall.png`
Recall@10 vs. ef_search per query condition. In-distribution and near lines overlap
between incremental and full rebuild at all ef values, no difference. Far cluster
shows a small but visible gap at low ef: at ef=10 incremental scores 0.073 vs full
rebuild 0.086; at ef=20 incremental 0.160 vs full rebuild 0.171. The gap closes
completely by ef=50 and above. The full rebuild has a small but real advantage for
far cluster queries at low ef, which makes sense considering its lower base-layer entry distance.

### `lower_bound_traces.png`
Mean lowerBound convergence at ef=20 (~20 iterations).
- **In-distribution**: both conditions overlap, converging from ~1900 to ~1520.
- **Near cluster**: both conditions overlap, converging from ~2100 to ~1650.
- **Far cluster**: clear separation. Incremental starts at ~4700 and ends at ~3250.
  Full rebuild starts at ~4250 and ends at ~3050. The full rebuild converges faster
  and to a lower floor since its upper-layer navigation delivers a better base-layer
  entry point.

### `recall_violin.png`
Per-query recall distributions at ef=20. In-distribution and near violins are tightly
concentrated near zero — most queries fail to find their true nearest neighbours in
just 20 search iterations. Far cluster violins have substantial mass from 0 to 0.8,
with a median around 0.1–0.15 for incremental and slightly higher for full rebuild.
Some queries land near the inserted cluster subgraph and are answered well, others do not.

---

## Summary table (ef=20)

| Index | Query condition | Recall@10 | EP dist | BL entry dist | L1 visits | Base visited |
|---|---|---|---|---|---|---|
| Incremental | In-distribution | 0.0384 | 2363.5 | 1524.4 | 27.4 | 27.4 |
| Incremental | Near | 0.0296 | 2494.3 | 1641.7 | 27.2 | 27.4 |
| Incremental | Far | 0.1596 | 5909.5 | 3972.0 | 25.5 | 33.8 |
| Full rebuild | In-distribution | 0.0395 | 2288.0 | 1520.5 | 27.5 | 27.2 |
| Full rebuild | Near | 0.0298 | 2458.6 | 1632.9 | 26.4 | 27.5 |
| Full rebuild | Far | 0.1706 | 5258.6 | 3543.2 | 23.8 | 32.1 |

---

## Key findings

1. **Far cluster queries have dramatically higher recall than in-distribution or
   near queries.** This is a GT artefact, not a real navigational advantage: the
   true nearest neighbours of far cluster queries are the 100k dense inserted vectors
   in that cluster, which form a tight easily-navigable subgraph. The GT for
   in-distribution queries is spread across 10M vectors with many equidistant candidates.

2. **Full rebuild has a small advantage for far cluster queries at low ef.** At ef=10
   the full rebuild scores 0.086 vs incremental 0.073 (17% higher). This gap closes
   by ef=50. The mechanism is base-layer entry distance — the full rebuild arrives
   429 units closer to far cluster queries after upper-layer descent (3543 vs 3972).

3. **In-distribution and near queries show no difference between index conditions.**
   Unlike the multi-cluster experiment, insertion into a single blob does not corrupt
   the navigational pathways for in-distribution queries. The blob has no multi-cluster
   upper-layer structure to corrupt.

4. **The lowerBound trace confirms the navigational advantage of the full rebuild for
   far queries.** The full rebuild converges to a lower floor (~3050 vs ~3250 at
   ef=20 termination), consistent with its better base-layer entry point.

5. **The single-blob design does not reproduce the in-distribution degradation seen
   in the multi-cluster experiment.** The mechanism observed there (insertion corrupts
   upper-layer routing toward existing clusters) requires a multi-cluster initial
   distribution. A single blob has no directional upper-layer structure to corrupt.

## Differences from first blob attempt

The first blob run used `n_near=n_far=1_000_000` (10% of n_init each) and
`near_sigma=2.0` (placing the near centre inside the blob's 3σ radius). Both
conditions resulted in a null result — incremental and full rebuild were identical.
The corrected parameters (`n_near=n_far=100_000`, `near_sigma=4.0`) give a 100:1
data ratio and place the near centre clearly outside the initial distribution,
producing the small but real gap at low ef observed here.