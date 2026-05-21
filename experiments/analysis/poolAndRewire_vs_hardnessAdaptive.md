# PoolAndRewire vs HardnessAdaptive — Detailed Comparison

## Overview

Both controllers wrap an HNSW index and adapt its structure in response to a shifting query distribution. They share the same search guarantee: a **union-merge** of a pool entry point and the original global entry point, so recall can never drop below the no-adaptation baseline.

The fundamental difference is the **model of drift** each controller assumes:

| | PoolAndRewire | HardnessAdaptive |
|---|---|---|
| Drift model | Spatial: queries move to a new region of vector space | Structural: queries become harder to navigate regardless of position |
| Primary drift indicator | `base_layer_entry_distance` (how far the fixed EP is from the query region) | `base_layer_distance_computations` (how much graph exploration the query required at the base layer) |
| Adaptation scope | Triggered once global drift is detected (sliding window); acts on a batch of recent queries | Triggered per hard query; acts immediately and locally |

---

## 1. Drift Detection

### PoolAndRewire
- Maintains a **sliding window** of the last `window_size=200` queries.
- Computes `mean(bl_entry_distance)` over the window.
- Drift is declared when this mean exceeds `baseline * 1.2` (or an optional centroid-displacement threshold).
- Has a **global cooldown** of `cooldown=50` total queries between successive adaptation events, regardless of how many hard queries arrive.

### HardnessAdaptive
- No sliding window. After every query, checks whether `base_dist_comps > hard_threshold`.
- `hard_threshold` is calibrated from warmup: the p75 of `base_dist_comps` observed on the warmup bin at `warmup_ef=50`. Queries above this percentile are considered structurally hard.
- The cooldown (`hard_rewire_cooldown=10`) counts **hard queries**, not all queries. A region that generates many hard queries in sequence will trigger rewiring much faster than PoolAndRewire's fixed cooldown would allow.

**Consequence:** PoolAndRewire can be slower to respond when the query distribution drifts gradually but immediately (bin-by-bin as in hardness drift). HardnessAdaptive responds the moment individual hard queries start appearing.

---

## 2. Pool Seeding Strategy

Both controllers maintain a Python-side set of index node IDs (`_pool`) that are promoted to `max_layer` and used as alternative entry points.

### PoolAndRewire
- At adaptation time, takes the last `k_pool = min(20, max_pool_size) = 20` queries from the recent window and runs `knn_query(q, k=1)` at `ef=500` for each. The single nearest index node per query is promoted to `max_layer` and added to the pool.
- Up to 20 nodes are promoted per adaptation event. Since `_pool` is a set, duplicates are silently skipped, so fewer may actually be added.
- Pool is seeded from a **sample of recent queries** (all hardness levels), not from hard queries specifically.

### HardnessAdaptive
- At adaptation time (after every hard query): runs `knn_query(q, k=1)` at `ef=1` (low ef because it is cheap, also ran experiment with `ef=20`: results almost identical) and promotes the single nearest index node.
- Only hard queries seed the pool. Easy queries never contribute pool entries.
- The pool therefore concentrates specifically on structurally difficult index regions, not on a query centroid in general.

**TODO:** 
Check how both methods perform on the directional YFCC drift dataset. My hypothesis is that on spatial drift, both strategies land in roughly the same region. Whereas on a query hardness based drift, the hard queries are not necessarily geometrically clustered. They share structural properties, i.e. poor graph connectivity but may be scattered in space. Since hardnessAdaptive seeds the pool from wherever the hard queries actually land, it might perform better than poolAndRewire

---

## 3. Pool Eviction Policy

### PoolAndRewire
- **`_pool_evict_by_centroid(centroid)`**: evicts the pool node farthest from the centroid of the full current query window.
- Rationale: a node far from the centroid is unlikely to be the closest pool entry for any upcoming query.
- The centroid is recomputed at each adaptation event over all `window_size` queries.

### HardnessAdaptive
- **`_pool_evict(anchor)`**: evicts the pool node farthest from the current hard query vector (the `anchor`).
- The anchor is the hard query that just triggered pool addition, not a window centroid.
- Rationale: as hard queries arrive sequentially, the pool should track the current hard region closely. Since drift might not correlate with the geometric distribution of queries, a window centroid as anchor could result in an almost arbitrary replacement.

**Consequence:** PoolAndRewire's centroid-based eviction is more stable but lags behind rapid hardness shifts. HardnessAdaptive's per-query anchor eviction is more reactive: each new hard query nudges the pool toward the current difficulty frontier.

---

## 4. Rewiring Trigger and Scope

Both controllers use `index.rewire_for_query(q, max_layer, alpha)` from the C++ extension to add improving upper-layer edges toward a query's neighborhood.

### PoolAndRewire
- Rewiring happens inside `_adapt()`, which is called only when **global drift** is detected and the **cooldown has elapsed**.
- Rewires for each of the last `queries_per_rewire=200` queries in the window — a **batch** of recent queries.
- This means each adaptation event does up to 200 `rewire_for_query` calls.

### HardnessAdaptive
- Rewiring is triggered after every `hard_rewire_cooldown=10` **hard queries**.
- Rewires for the **single current hard query** only.
- Much more frequent but much smaller scope per event: 1 `rewire_for_query` call per event vs. up to 200.

**Consequence:** Over a stream of 5000 queries with, let's say, 40% hard (2000 hard queries), HardnessAdaptive fires 200 rewire events (one per 10 hard queries), each targeting exactly where the current hard query landed. PoolAndRewire might fire 10–20 batch events, each rewiring for 200 diverse queries. 
HardnessAdaptive rewiring is more targeted but less diverse. PoolAndRewire rewiring covers more of the recent query distribution per event.

---

## 5. Adaptive ef Escalation

### PoolAndRewire
- None. ef is fixed at whatever the caller sets.

### HardnessAdaptive
- Tracks a rolling window (`escalation_window=50`) of whether each recent query was hard.
- When the fraction of hard queries in this window exceeds `escalation_trigger=0.3`, boosts ef to `ef * escalation_factor=3` for the next search.
- This is purely **retrospective**: it uses past query difficulty to predict future difficulty. Works well under monotone hardness drift (queries get progressively harder) but would over-escalate if the distribution oscillates.
- The ef boost applies to both the global-EP search and the pool search in the union-merge.

**Consequence:** At `ef=10`, HardnessAdaptive in hard bins effectively searches at `ef=30`. This is not free: it costs 3x more distance computations per hard query, but it narrows the recall gap on structurally hard regions that no graph modification alone can fix. Should the queries return to an easier average hardness, the ef is reset, s.t. computation cost is reduced.

---

## 6. Highway Edge

### PoolAndRewire
- `_build_highway()`: at each adaptation event, finds the upper-layer node closest to the current query centroid and adds a directed edge from the global entry point to it.
- Provides a long-range shortcut from the global EP toward the drifted region, improving coarse navigation at the top of the hierarchy.

### HardnessAdaptive
- No highway edge. Hard queries may not form a coherent spatial centroid, so a single long-range edge toward their centroid would often point to an arbitrary location.

---

## 7. Initialization Cost

### PoolAndRewire
- `measure_baseline()`: runs all `reference_queries` at `ef=200` to compute the baseline `bl_entry_distance`. This is used to set the drift threshold.
- Cost: `len(reference_queries)` searches at `ef=200`.

### HardnessAdaptive
- `_measure_dist_comps()`: runs all `reference_queries` at `warmup_ef=50` to compute dist_comp percentiles.
- Cost: `len(reference_queries)` searches at `ef=50`. Is cheaper.

---

## 8. Summary Table

| Dimension | PoolAndRewire | HardnessAdaptive |
|---|---|---|
| Drift signal | `bl_entry_distance` (aggregate, windowed) | `base_dist_comps` (per-query) |
| Adaptation granularity | Batch (up to 200 queries per event) | Per hard query |
| Adaptation frequency | Global cooldown (50 total queries min) | Per-hard-query cooldown (10 hard queries min) |
| Pool seeding | Recent queries (all, via centroid) | Hard queries only (nearest node to each) |
| Pool eviction anchor | Query window centroid | Current hard query vector |
| Rewire scope per event | Up to 200 recent queries | 1 current hard query |
| Highway edge | Yes (from global EP toward spatial centroid) | No |
| ef escalation | No | Yes (rolling hard-fraction trigger) |
| Initialization ef | 200 (for bl_entry baseline) | 50 (for dist_comp percentile) |
| Designed for | Spatial drift (YFCC, synthetic Gaussian) | Structural / hardness drift (SIFT hardness) |

