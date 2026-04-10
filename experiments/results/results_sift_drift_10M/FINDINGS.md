# Experiment: Full-Index SIFT Drift

---

## What this experiment tests

This experiment explores the following question: **What happens to HNSW search quality when the queries at runtime look systematically different from the vectors the index was built on?**. 

The index was built on all 10M SIFT vectors. Queries were drawn from shifted
distributions along a fixed direction, split into 11 bands of increasing shift
magnitude (0.0 to 585.87 -> 1 for baseline, the rest is increasing shift)


---

## Finding 1: Recall increases with shift

This result is rather counterintuitive. Across all ef values, recall rises as shift magnitude grows. At ef=200 recall goes from **0.610 at
shift=0** up to **0.809 at shift=585.87**. At ef=500 it rises from **0.773 to
0.923**.


---

## Finding 2: Per-query recall are distributions broad but improve at the median

The violin plots show a consistent pattern across shift levels: the
distribution spans almost the full range from 0 to 1 at every shift, but the median and the bulk of the distribution shift upward as shift increases. 
At shift=0 the median sits around 0.6 and the lower quartile dips close to 0.
At shift=585.87 the median is around 0.8 and the distribution is more concentrated in the upper half.

---

### Why recall improves (possible explanation)

This behaviour could be explained by the structure of SIFT descriptors. Each vector is a histogram of gradient orientations 
quantised to [0, 255]. PC1 captures the contrast axis, separating smooth patches (low gradient energy, low PC1) from textured patches (high gradient energy, high PC1). Highly textured patches are more discriminative: their k=10 nearest neighbours are more tightly clustered, making ground truth easier to satisfy at any ef. The index was built on all 10M vectors including the high-PC1 region, so HNSW had well-established navigational pathways toward it. Shifting queries toward high PC1 did not move them out of the construction distribution, it actually might have moved them toward a better-served region.
---

---

## Finding 3: Base-layer visited nodes decreases with shift

The number of base-layer nodes visited per query falls from ~216 at shift=0
to ~204 at shift=585.87, and candidates at termination drops from ~572 to
~509. The search is doing less work at high shift.
I think this can be explained the same way as finding 1. In the "denser" area the candidate
list fills up slowly because many nodes are possible candidates and the search has to
visit more of them before ef is exhausted. In the sparse periphery, the
candidate list is satisfied more quickly because far fewer nodes are within
a competitive distance. The search terminates having explored a smaller
neighbourhood, but that neighbourhood is sufficient because the ground truth
is less ambiguous.

---



## Summary

| Shift | Recall (ef=200) | EP dist | BL entry dist | Base visited |
|-------|----------------|---------|---------------|--------------|
| 0.0 | 0.610 | 747,417 | 597,022 | 216 |
| 117.17 | 0.667 | 804,192 | 590,579 | 212 |
| 234.35 | 0.712 | 887,255 | 591,957 | 208 |
| 351.52 | 0.737 | 995,864 | 608,232 | 206 |
| 468.70 | 0.770 | 1,063,142 | 630,518 | 206 |
| 585.87 | 0.809 | 1,313,884 | 731,887 | 204 |

Recall degradation under shift is not universal. 
It depends on whether shift moves queries into a harder or easier retrieval problem. 
When the index covers the query region and the peripheral neighbourhood is sparser than the dense core, 
shift actually reduces the difficulty of identifying true nearest neighbours. The graph's upper layers remain capable of routing the
search appropriately as long as the query region is represented in the index.