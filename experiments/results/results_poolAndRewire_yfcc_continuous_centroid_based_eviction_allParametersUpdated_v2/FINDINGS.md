** Problem 1: Pool candidates are srearched with ef=50 **

In the `_adapt` method the knn_query that selects the pool nodes uses ef 50, which is a rather inaccurate search that might not find the truly nearest neighbours to the drifting centroid.
Since the pool quality determines how well each search starts, increasing ef could be a quick fix, but with some cost. -> next checking ef =500 for comparison.


** Problem 2: only at most 5 nodes are added in each adapt step **

We currently add at most 5 nodes per triggered update, which by design cluster near the centroid. Queries at the edges of the drifted distribution get little benefit from that. Increasing the number of added pool nodes might help as this (together with eviction to maintain the pool at a certain size) allows filling the pool with more diverse nodes.

** Additional Idea: also search from second (and third) best node and return nearest neighbours from the union of both? **