# Experiment: PC1-Split SIFT-10M Drift

## What this experiment tests

This experiment explores the following question: **What happens to HNSW search quality when the queries at runtime look systematically different from the vectors the index was built on?**. 

The HNSW index was built on 5M SIFT vectors drawn from the lower half of the PC1
distribution. Queries were drawn from the upper half, split into 11 bands of
increasing PC1 value. For each band I recorded recall@10, entry-point distance,
base-layer entry distance, and other internal search metrics across different ef_search values.

The shift label attached to each band is the mean PC1 value of that band. Higher
shift = queries that are further from the index distribution along the direction of
maximum variance.

Split design:
    - load all N vectors and project onto PC1 (first principal component)
    - build the index on the bottom 50% by PC1 value (low PC1 = one mode of the SIFT descriptor distribution)
    - generate query sets at increasing quantiles of the top 50%:
      level 0 samples just above the median (slight overlap with index),
      level n samples from the top quantile (far OOD)

->Goal:  hard geometric split combined with a continuous drift curve

---

## Parameters

| Parameter | Value |
|---|---|
| `Dataset` | SIFT-128 |
| `n_total` | 10,000,000 |
| `dim` | 128 |
| `n_index` | 5,000,000  (PC1 <= median)|
| `n_query_pool` | 5,000,000  (PC1 > median) |
| `shift levels` | 11 |
| `PC1 range (index)` | [−303.6, 13.5] |
| `PC1 range (query pool)` | [13.5, 292.9] |
| `M` | 16 |
| `ef_construction` | 200 |
| `ef_search` (sweep) | 10, 20, 50, 100, 200, 500 |
| `k_neighbours` | 10 |
| `n_queries` | 1,000 per shift level |
| GT backend | faiss GPU |
| GT pool | index data only |

---


## Recall degrades smoothly with shift

Recall falls monotonically as shift increases for every ef value. 

At ef=200 recall drops from **0.990 to 0.850** across the shift range. At ef=10 it
drops from **0.702 to 0.324** (roughly halved). The degradation is gradual but accumulates significantly by the time
queries are far out of distribution.

---

## Higher ef partially compensates, but never fully

Increasing ef_search recovers much of the lost recall but with two problems.

1) it is expensive. To match the recall that ef=100 achieves at low shift, a
highly shifted query set needs ef=500 ->roughly a 5× increase in search budget

2) even ef=500 does not close the gap. At shift=241.21, ef=500 gives 0.938
recall. At shift=25.91, ef=100 already gives 0.971. The shifted queries are
structurally harder regardless of how wide the search is.

---

## The entry point grows further from queries as shift increases

The distance from the (fixed) entry point to each query rises monotonically
with shift, from ~291k at shift=25.91 to ~364k at shift=241.21. This makes sense considering
the entry point was chosen during construction to sit near the centre of the index
distribution, so queries from above the PC1 median are geometrically further away by design.

---

## The upper layers amplify the problem

After descending through the upper-layer hierarchy, the search arrives at the base
layer at a distance that grows even faster than the entry-point distance. Base-layer
entry distance rises from ~77k to ~157k (roughly doubling, while entry-point
distance rises by only ~25%)

This means the upper layers are not just failing to correct for the bad starting
position, they are making it worse. The greedy traversal through the upper layers
converges to a local minimum that is progressively further from the true
neighbourhood as shift increases.

---

## lowerBound convergence slows with shift

At low shift the lowerBound drops quickly and converges. As shift increases the initial spike grows taller 
and convergence becomes slower, the search spends more iterations chasing candidates that are far
from the true nearest neighbours before settling. 

---

## Per-query recall becomes increasingly unpredictable

At low shift the per-query recall distribution is tightly concentrated near 1.0 with a thin downward tail. Most queries are answered well, poor performance is rare. As shift increases the distribution broadens substantially and the lower tail
extends toward 0.
At the highest shift levels, a non-negligible fraction of queries achieve recall near 0.

---

## Summary

| Shift | Recall (ef=200) | BL entry dist |
|-------|----------------|---------------|
| 25.91 | 0.990 | 76,997 |
| 65.67 | 0.982 | 97,023 |
| 109.76 | 0.953 | 121,457 |
| 155.73 | 0.909 | 143,617 |
| 241.21 | 0.850 | 157,209 |

HNSW recall under distribution shift is determined primarily by where the upper layers deposit the search when it reaches the base
layer. The entry point distance is a weak predictor. the base-layer entry distance is a stronger one, and it can be monitored at query time without needing the ground truth. maybe it could be used at runtime as a drift signal.