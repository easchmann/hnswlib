** Problem 1: Adaptation fires two sigma-levels too late **

The first adapt triggers at the sigma=6→8 boundary, when bl_entry=9607 > threshold=9604. That means sigma=0 through sigma=6 (6000 queries) run with zero pool and zero rewiring. By the time the pool is populated, most of the damage is already done. The threshold of 1.5x baseline is too loose.

Looking at the bl_entry ratios vs. baseline (6376):

sigma	bl_entry	ratio
3	    6843	    1.07
4	    7143	    1.12
6	    8104	    1.27
8	    9147	    1.43
Threshold at 1.2 = 7652 would trigger during sigma=4–5, giving the pool 2000+ queries of lead time before the heavy drift.

** Problem 2: Rewiring stops contributing early on **

After update 1 (16 edges) and update 2 (17 edges), essentially every further call only adds 0 to 2 edges.
`rewire_for_query` with alpha=1.5 only adds an edge if there is a neighbour that is 1.5 times closer to the query, but since the graph will be misaligned for larger sigmas there often would not be such a candidate for local improvement. The rewiring stops working at around shift sigma 10.


** Problem 3: Pool saturates at 20 nodes but the base layer entry distance keeps rising **
At shift sigma 12-16, the pool fills and stops growing while base layer entry distance rises from 11k to 14k. A larger pool might give better per-query coverage of the drifted distribution but at a higher search cost.
