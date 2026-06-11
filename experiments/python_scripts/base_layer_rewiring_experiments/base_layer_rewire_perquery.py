# base-layer edge addition experiment — per-query evaluation.
#
# Variant of base_layer_rewire_experiment.py where each query is immediately
# re-run after its own edge is added.  This gives per-query causal attribution:
# the gain for query i reflects primarily the edge added for query i (previous
# queries' edges remain but are fixed at that point).
#
# For each of the top-N hardest queries, in order:
#   1. Run baseline search at ef=EF, record recall@K and identify local_opt
#      (farthest beam candidate not in true NNs — Option A proxy).
#   2. Add edge: local_opt → true_nn at base layer.
#   3. Immediately re-run the same query on the now-mutated index.
#   4. Record gain = recall_after - recall_before.
#
# Usage:
#   python experiments/base_layer_rewire_perquery.py --data_dir data/sift_hardness
#   python experiments/base_layer_rewire_perquery.py --data_dir data/sift_hardness --dry_run

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
parser = argparse.ArgumentParser(description="base-layer edge addition, per-query adapt-then-rerun evaluation" )
parser.add_argument("--data_dir", required=True)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--ef", type=int, default=20)
parser.add_argument("--n_hard", type=int, default=200)
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

N_HARD = 20 if args.dry_run else args.n_hard
K  = args.k
EF = args.ef

out_dir = os.path.join(script_dir, "results")
os.makedirs(out_dir, exist_ok=True)
plot_path = os.path.join(out_dir, "base_layer_rewire_perquery_density.png")



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
    print(f"  scoring {len(queries):,} queries at ef={EF} ...")
    idx.set_ef(EF)
    scores = np.empty(len(queries), dtype=np.int32)
    for i, q in enumerate(queries):
        idx.knn_query(q.reshape(1, -1), k=K)
        scores[i] = int(hnswlib.get_last_query_stats()["base_layer_distance_computations"])
    return scores



# Single-query runner
def run_single(idx, q, gt_i):
    """
    Run one query at EF, return recall@K and local_opt/true_nn.

    local_opt: farthest beam candidate (among all EF results) that is not a
               true NN — the frontier node blocking true_nn (Option A proxy).
               None if all EF candidates are true NNs.
    true_nn:   gt_i[0], the absolute nearest neighbour.
    """
    idx.set_ef(EF)
    labels, _ = idx.knn_query(q.reshape(1, -1), k=EF)
    labels = labels[0]   # sorted nearest-first, length EF

    true_set = set(int(x) for x in gt_i)
    recall = len(set(int(x) for x in labels[:K]) & true_set) / K

    local_opt = None
    for lab in reversed(labels):  # farthest-first
        if int(lab) not in true_set:
            local_opt = int(lab)
            break

    return {
        "recall":    recall,
        "local_opt": local_opt,
        "true_nn":   int(gt_i[0]),
    }




def main():
    mode = f"DRY RUN ({N_HARD} queries)" if args.dry_run else f"top-{N_HARD} hardest queries"
    print(f"Base-layer rewire (per-query eval) — {mode}, ef={EF}, k={K}\n")

    queries, gt = load_dataset()
    dim = queries.shape[1]

    params_path = os.path.join(args.data_dir, "params.json")
    with open(params_path) as f:
        n_index = json.load(f)["n_index"]

    idx = load_index(dim, n_index)


    # Step 1: Hardness scoring
    print(f"\n[1] Hardness scoring at ef={EF}")
    hardness = score_all_queries(idx, queries)
    sorted_order = np.argsort(hardness)[::-1]
    hard_idx = sorted_order[:N_HARD]

    hard_queries = queries[hard_idx]
    hard_gt = gt[hard_idx]
    hard_scores = hardness[hard_idx]

    print(f"  hardness of selected queries: "
          f"min={hard_scores.min()}  max={hard_scores.max()}  "
          f"mean={hard_scores.mean():.1f}")


    # Step 2: Per-query adapt-then-rerun loop
    print(f"\n[2] Per-query loop: baseline → add edge → immediate rerun")

    base_recalls = np.empty(N_HARD, dtype=np.float32)
    post_recalls = np.empty(N_HARD, dtype=np.float32)
    local_opts = []
    true_nns = []
    edges_added = 0

    for i in range(N_HARD):
        q = hard_queries[i]
        gt_i = hard_gt[i]

        r_before = run_single(idx, q, gt_i)
        base_recalls[i] = r_before["recall"]
        local_opts.append(r_before["local_opt"])
        true_nns.append(r_before["true_nn"])

        lo, tn = r_before["local_opt"], r_before["true_nn"]
        edge_added_this = False
        if lo is not None and lo != tn:
            idx.add_back_edge(lo, tn, 0)
            edges_added += 1
            edge_added_this = True

        r_after = run_single(idx, q, gt_i)
        post_recalls[i] = r_after["recall"]

        gain = post_recalls[i] - base_recalls[i]
        marker = "↑" if gain > 0 else ("↓" if gain < 0 else "·")
        if gain != 0 or (i % 20 == 0):
            print(f"  [{i:>3}] hardness={hard_scores[i]:>5}  "
                  f"before={base_recalls[i]:.2f}  after={post_recalls[i]:.2f}  "
                  f"gain={gain:+.2f}  edge={'yes' if edge_added_this else 'no '}  {marker}")

    gains = post_recalls - base_recalls


    # Step 3: Per-query results table
    print(f"\n{'─'*76}")
    print(f"{'#':>5}  {'hardness':>9}  {'before':>7}  {'after':>7}  "
          f"{'gain':>7}  {'local_opt':>10}  {'true_nn':>8}")
    print(f"{'─'*76}")
    for i in range(N_HARD):
        print(f"{i:>5}  {hard_scores[i]:>9}  {base_recalls[i]:>7.4f}  "
              f"{post_recalls[i]:>7.4f}  {gains[i]:>+7.4f}  "
              f"{str(local_opts[i]):>10}  {true_nns[i]:>8}")
    print(f"{'─'*76}")
    print(f"{'MEAN':>5}  {'':>9}  {base_recalls.mean():>7.4f}  "
          f"{post_recalls.mean():>7.4f}  {gains.mean():>+7.4f}")


    # Step 4: Breakdown by hardness quartile (Q1 = hardest)
    print(f"\n--- breakdown by hardness quartile (Q1 = hardest) ---")
    quartile_splits = np.array_split(np.arange(N_HARD), 4)
    labels_q = ["hardest", "hard   ", "medium ", "easiest"]
    print(f"{'':>4}  {'label':>8}  {'hardness range':>20}  {'n':>4}  "
          f"{'before':>7}  {'after':>7}  {'gain':>7}")
    for qi, qidx in enumerate(quartile_splits):
        hs_lo = hard_scores[qidx].min()
        hs_hi = hard_scores[qidx].max()
        print(f"  Q{qi+1}  {labels_q[qi]}  [{hs_lo:6.0f}, {hs_hi:6.0f}]  "
              f"n={len(qidx):>4}  "
              f"{base_recalls[qidx].mean():>7.4f}  "
              f"{post_recalls[qidx].mean():>7.4f}  "
              f"{gains[qidx].mean():>+7.4f}")

    # Step 5: Density check scatter plot
    print(f"\n[5] Computing {K}th-NN distances for density scatter plot ...")
    tenth_nn_dists = np.empty(N_HARD, dtype=np.float32)
    for i in range(N_HARD):
        tenth_nn_vec = idx.get_items([int(hard_gt[i][K - 1])])[0]
        tenth_nn_dists[i] = float(np.sqrt(np.sum((hard_queries[i] - tenth_nn_vec) ** 2)))

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(tenth_nn_dists, gains, alpha=0.65, s=30,
                    c=hard_scores, cmap="viridis")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"hardness (base dist_comps at ef={EF})")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"L2 distance to {K}th true NN (density proxy; larger = sparser)")
    ax.set_ylabel("recall gain  (immediate rerun − baseline)")
    ax.set_title(
        f"Base-layer rewire (per-query): recall gain vs local density\n"
        f"top-{N_HARD} hardest queries, ef={EF}, k={K}, edges added={edges_added}"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {plot_path}")


    # Summary
    n_improved = int((gains >  0).sum())
    n_unchanged = int((gains == 0).sum())
    n_hurt = int((gains <  0).sum())

    print(f"\n{'='*52}")
    print(f"SUMMARY  (per-query: edge added then immediately retested)")
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
