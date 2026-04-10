# Experiment:YFCC Directional Drift

## Setup

Index: 5M DINO ViT-S/16 embeddings sampled from all YFCC years (2007–2013).  
Queries: 1000 vectors shifted along the 2007-> 2013 mean direction by σ × 6.3 (magnitude of the mean vector) L2 units.  
σ=0 is in-distribution; σ=1 corresponds to one full span of real temporal drift.

---

## Main finding

Recall degrades monotonically with shift, and the degradation is **not recoverable by raising ef** at large σ, unlike the temporal year experiment, where ef=500 closed the gap almost entirely.

| σ   | ef=10 | ef=200 | ef=500 |
|-----|-------|--------|--------|
| 0   | 0.838 | 0.988  | 0.995  |
| 4   | 0.773 | 0.982  | 0.994  |
| 8   | 0.641 | 0.952  | 0.977  |
| 12  | 0.439 | 0.872  | 0.924  |
| 16  | 0.287 | 0.764  | 0.856  |

At σ=16 even ef=500 only reaches 0.856, a 14pp gap from the in-distribution ceiling.

---

## Analysis

**Entry-point distance** rises from ~15,820 at σ=0 to ~25,567 at σ=16, this is a  62% increase. 
This means the global entry point, which is fixed at a node near the centroid of the construction distribution, is progressively further from the shifted queries. Greedy descent from that entry point starts in the wrong region.

**Base-layer entry distance** rises from ~6,298 to ~15,334. a 143% increase. 
THis is a slightly stronger indicator. By the time the upper-layer traversal delivers the query to the base layer, the starting node is already far from the query's true neighborhood.
The ef-search then has to work from that bad starting point.

**Base-layer visited nodes** increases slightly (202 -> 208 at ef=200) but stays
roughly flat, giving the same conclusion as SIFT: the algorithm is doing the same amount of work, just in the wrong part of the graph.

**Candidates at termination** rises from ~311 at σ=0 to ~493 at σ=16 (ef=200). 
The candidate heap is filling with nodes that are locally consistent but globally wrong,the search terminates with a full heap of poor candidates rather than running out of nodes to check.

**Layer-1 visit count** decreases slightly (36.2 to 34.0). This is
consistent with the upper-layer traversal converging faster to the entry point of a wrong neighborhood. 
fewer corrections are made because the graph has no edges pointing toward the shifted query region. **(-> to be verified)**

---

## Lower-bound convergence
lower bound traces converge at roughly the same speed for all sigmas but they converge to increasing values

---

## Comparison with the temporal year experiment

The temporal year experiment (index: 2007–2009, queries: 2010–2013) showed only ~1% recall degradation at ef=200, recoverable almost entirely at ef=500. That experiment measured the *natural* temporal drift in YFCC DINO embeddings, which is small.

This experiment applies a *controlled* displacement along the direction of that samee drift, scaled to multiples of its magnitude. The results show that once the shift exceeds ~4 to 6 times the natural year-span displacement, HNSW's navigational structure fails and the degradation can't be compensated by increasing the search budget. This supports that the mechanism is more structural (bad entry point, wrong neighborhood) rather than a budget issue.

---

## Implications for next steps:

The entry-point distance and base-layer entry distance are proxies for the degree of structural misalignment. 
I could imagine an adaptation strategy that monitors these signals and responds by updating the upper-layer graph, e.g by moving the entry
point or adding more long-range edges toward the shifted region could adress the observed failure.