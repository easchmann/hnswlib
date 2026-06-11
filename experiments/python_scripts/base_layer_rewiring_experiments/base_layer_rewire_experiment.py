# base-layer edge addition experiment on the SIFT hardness dataset.

# For each of the top-N hardest queries (measured by base-layer distance
# computations at ef=20):

#   1. Baseline pass  – record recall@10, identify the local-optimum node
#                       (farthest beam candidate that is not a true NN — the frontier
#                       node blocking true_nn from entering the ef-set, Option A) and
#                       the true-NN node (gt[i][0]).
#   2. Edge addition  – index.add_back_edge(local_opt -> true_nn, layer=0).
#   3. Post-rewire    – re-run the same queries on the mutated index, record recall@10.
#   4. Results table  – per-query recall before/after, aggregate and quartile stats.
#   5. Density check  – scatter: recall gain vs distance to 10th true NN (density proxy).

# NOTE: the index is mutated in-place. Edges accumulate across all N queries before
# the post-rewire pass measures them collectively.

# Usage:
#   python experiments/base_layer_rewire_experiment.py --data_dir data/sift_hardness
#   python experiments/base_layer_rewire_experiment.py --data_dir data/sift_hardness --dry_run


import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

import hnswlib


# CLI
parser = argparse.ArgumentParser(description="base-layer edge addition on SIFT hardness queries")
parser.add_argument("--data_dir", required=True)
parser.add_argument("--k", type=int, default=10,)
parser.add_argument("--ef", type=int, default=20)
parser.add_argument("--n_hard", type=int, default=200)
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

N_HARD = 20 if args.dry_run else args.n_hard
K = args.k
EF = args.ef

out_dir = os.path.join(script_dir, "results")
os.makedirs(out_dir, exist_ok=True)
plot_path = os.path.join(out_dir, "base_layer_rewire_density.png")


# Data loading
def load_dataset():
    print(f"loading dataset from {args.data_dir} ...")
    queries = np.load(os.path.join(args.data_dir, "queries_by_hardness.npy"))
    gt = np.load(os.path.join(args.data_dir, "ground_truth.npy"))
    print(f"  {len(queries):,} queries  dim={queries.shape[1]}  gt shape={gt.shape}")
    return queries, gt


def load_index(dim, n_elements):
    path = os.path.join(args.data_dir, "hnsw_index.bin")
    print(f"  loading index from {path} ...")
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path, max_elements=n_elements)
    return idx



# Hardness scoring
def score_all_queries(idx, queries):
    """Run every query at EF and return base_layer_distance_computations per query."""
    print(f"  scoring {len(queries):,} queries at ef={EF} ...")
    idx.set_ef(EF)
    scores = np.empty(len(queries), dtype=np.int32)
    for i, q in enumerate(queries):
        idx.knn_query(q.reshape(1, -1), k=K)
        scores[i] = int(hnswlib.get_last_query_stats()["base_layer_distance_computations"])
    return scores


# query runner
def run_pass(idx, queries, gt_sub):
    """
    Run queries at EF.  For each query records:
      recall      – fraction of true K-NNs returned (top-K of the EF candidates)
      local_opt   – farthest returned node NOT in gt_sub[i]; None when all EF
                    candidates are true NNs.  The farthest non-true-NN is the beam
                    frontier node that blocked true_nn from entering the ef-set —
                    the correct stall-point proxy (Option A).
      true_nn     – gt_sub[i][0] (external label of the absolute nearest neighbour)
    """
    idx.set_ef(EF)
    results = []
    for i, q in enumerate(queries):
        # k=EF: retrieve all EF beam candidates, not just top-K
        labels, _ = idx.knn_query(q.reshape(1, -1), k=EF)
        labels = labels[0]   # 1-D, sorted nearest-first; length EF

        true_set = set(int(x) for x in gt_sub[i])
        recall   = len(set(int(x) for x in labels[:K]) & true_set) / K

        # farthest non-true-NN in the beam = the frontier node blocking true_nn
        local_opt = None
        for lab in reversed(labels):
            if int(lab) not in true_set:
                local_opt = int(lab)
                break

        results.append({
            "recall":    recall,
            "local_opt": local_opt,
            "true_nn":   int(gt_sub[i][0]),
        })
    return results



def main():
    mode = f"DRY RUN ({N_HARD} queries)" if args.dry_run else f"top-{N_HARD} hardest queries"
    print(f"Base-layer rewire experiment — {mode}, ef={EF}, k={K}\n")

    queries, gt = load_dataset()
    dim = queries.shape[1]

    params_path = os.path.join(args.data_dir, "params.json")
    with open(params_path) as f:
        n_index = json.load(f)["n_index"]

    idx = load_index(dim, n_index)


    # Step 1: Hardness scoring — recompute at ef=EF
    print(f"\n[1] Hardness scoring at ef={EF}")
    hardness = score_all_queries(idx, queries)
    sorted_order = np.argsort(hardness)[::-1] # descending
    hard_idx = sorted_order[:N_HARD]

    hard_queries = queries[hard_idx]
    hard_gt = gt[hard_idx]
    hard_scores = hardness[hard_idx]

    print(f"  hardness of selected queries: "
          f"min={hard_scores.min()}  max={hard_scores.max()}  "
          f"mean={hard_scores.mean():.1f}")


    # Step 2: Baseline pass (clean index)
    print(f"\n[2] Baseline pass (ef={EF}, k={K})")
    baseline = run_pass(idx, hard_queries, hard_gt)
    base_recalls = np.array([r["recall"] for r in baseline])

    n_perfect = int((base_recalls == 1.0).sum())
    n_actionable = N_HARD - n_perfect
    print(f"  mean recall = {base_recalls.mean():.4f}  "
          f"recall==1.0: {n_perfect}/{N_HARD}  "
          f"will add edges for: {n_actionable}")


    # Step 3: Edge addition — local_opt -> true_nn at base layer (layer=0)
    print(f"\n[3] Adding edges (local_opt → true_nn, layer=0)")
    edges_added = 0
    for r in baseline:
        if r["local_opt"] is None:
            continue # recall=1.0, nothing to bridge
        lo, tn = r["local_opt"], r["true_nn"]
        if lo == tn:
            continue # degenerate (shouldn't happen in practice)
        idx.add_back_edge(lo, tn, 0)
        edges_added += 1
    print(f"  edges added: {edges_added}")


    # Step 4: Post-rewire pass (mutated index)
    print(f"\n[4] Post-rewire pass (ef={EF}, k={K})")
    post = run_pass(idx, hard_queries, hard_gt)
    post_recalls = np.array([r["recall"] for r in post])
    gains = post_recalls - base_recalls
    print(f"  mean recall = {post_recalls.mean():.4f}  "
          f"delta = {gains.mean():+.4f}")


    # Step 5: Per-query results table
    print(f"\n{'─'*76}")
    print(f"{'#':>5}  {'hardness':>9}  {'before':>7}  {'after':>7}  "
          f"{'gain':>7}  {'local_opt':>10}  {'true_nn':>8}")
    print(f"{'─'*76}")
    for i in range(N_HARD):
        r = baseline[i]
        print(f"{i:>5}  {hard_scores[i]:>9}  {base_recalls[i]:>7.4f}  "
              f"{post_recalls[i]:>7.4f}  {gains[i]:>+7.4f}  "
              f"{str(r['local_opt']):>10}  {r['true_nn']:>8}")
    print(f"{'─'*76}")
    print(f"{'MEAN':>5}  {'':>9}  {base_recalls.mean():>7.4f}  "
          f"{post_recalls.mean():>7.4f}  {gains.mean():>+7.4f}")


    # Step 6: Breakdown by hardness quartile (Q1 = hardest)
    print(f"\n--- breakdown by hardness quartile (Q1 = hardest) ---")
    quartile_splits = np.array_split(np.arange(N_HARD), 4)
    labels_q = ["hardest", "hard   ", "medium ", "easiest"]
    print(f"{'':>4}  {'label':>8}  {'hardness range':>20}  {'n':>4}  "
          f"{'before':>7}  {'after':>7}  {'gain':>7}")
    for qi, qidx in enumerate(quartile_splits):
        hs_lo = hard_scores[qidx].min()
        hs_hi = hard_scores[qidx].max()
        qb    = base_recalls[qidx].mean()
        qa    = post_recalls[qidx].mean()
        qg    = gains[qidx].mean()
        print(f"  Q{qi+1}  {labels_q[qi]}  [{hs_lo:6.0f}, {hs_hi:6.0f}]  "
              f"n={len(qidx):>4}  "
              f"{qb:>7.4f}  {qa:>7.4f}  {qg:>+7.4f}")


    # Step 7: Density check scatter plot

    print(f"\n[7] Computing {K}th-NN distances for density scatter plot ...")
    tenth_nn_dists = np.empty(N_HARD, dtype=np.float32)
    for i in range(N_HARD):
        tenth_nn_id  = int(hard_gt[i][K - 1])
        tenth_nn_vec = idx.get_items([tenth_nn_id])[0]
        tenth_nn_dists[i] = float(np.sqrt(np.sum((hard_queries[i] - tenth_nn_vec) ** 2)))

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(tenth_nn_dists, gains, alpha=0.65, s=30,
                    c=hard_scores, cmap="viridis")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"hardness (base dist_comps at ef={EF})")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"L2 distance to {K}th true NN (density proxy; larger = sparser)")
    ax.set_ylabel("recall gain  (post-rewire − baseline)")
    ax.set_title(
        f"Base-layer rewire: recall gain vs local density\n"
        f"top-{N_HARD} hardest queries, ef={EF}, k={K}, edges added={edges_added}"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {plot_path}")



    # Summary
    
    n_improved = int((gains >  0).sum())
    n_unchanged = int((gains == 0).sum())
    n_hurt = int((gains <  0).sum())

    print(f"\n{'='*52}")
    print(f"SUMMARY")
    print(f"  queries evaluated:  {N_HARD}")
    print(f"  edges added:        {edges_added}")
    print(f"  recall before:      {base_recalls.mean():.4f}")
    print(f"  recall after:       {post_recalls.mean():.4f}")
    print(f"  mean gain:          {gains.mean():+.4f}")
    print(f"  queries improved:   {n_improved}")
    print(f"  queries unchanged:  {n_unchanged}")
    print(f"  queries hurt:       {n_hurt}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
