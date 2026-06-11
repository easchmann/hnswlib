# Online base-layer edge insertion: stream queries in hardness order,
# add one bridge edge after each query, measure whether accumulated edges
# improve recall for subsequent queries.
#
# Bridge strategy (Option A): farthest non-true-NN in the ef-sized beam → true_nn.
# This is the best single-edge variant from the batch experiments (+0.0075 mean gain).
#
# Evaluation design:
#   1. Baseline pass: run all queries on the unmodified index, record recalls.
#   2. Adaptive pass: reload fresh index, stream queries in hardness order,
#      add edge after each query, record recall at query time (before the edge).
#   Rolling recall (window W) shows whether accumulated edges lift future queries.
#
# Usage:
#   python experiments/base_layer_rewire_online.py --data_dir data/sift_hardness
#   python experiments/base_layer_rewire_online.py --data_dir data/sift_hardness --dry_run

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


parser = argparse.ArgumentParser(description="online base-layer edge insertion")
parser.add_argument("--data_dir", required=True)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--ef",type=int, default=20)
parser.add_argument("--n_queries", type=int, default=1000)
parser.add_argument("--window", type=int, default=50,
                    help="rolling recall window size")
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

K      = args.k
EF     = args.ef
WINDOW = args.window

out_dir   = os.path.join(script_dir, "results")
os.makedirs(out_dir, exist_ok=True)
plot_path = os.path.join(out_dir, "base_layer_rewire_online.png")


# Data loading

def load_dataset():
    print(f"loading dataset from {args.data_dir} ...")
    for fname in ("queries_by_hardness.npy", "queries.npy"):
        path = os.path.join(args.data_dir, fname)
        if os.path.exists(path):
            queries = np.load(path)
            break
    gt = np.load(os.path.join(args.data_dir, "ground_truth.npy"))
    print(f"  {len(queries):,} queries  dim={queries.shape[1]}  gt shape={gt.shape}")
    return queries, gt


def load_index(dim, n_elements):
    path = os.path.join(args.data_dir, "hnsw_index.bin")
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path, max_elements=n_elements)
    return idx



# Single query: run, compute recall, find bridge edge

def run_query(idx, q, gt_i):
    """
    Returns recall@K and bridge=(src, dst) or None.
    Uses k=EF to see full beam; recall is measured on labels[:K].
    """
    idx.set_ef(EF)
    labels, _ = idx.knn_query(q.reshape(1, -1), k=EF)
    beam = [int(x) for x in labels[0]]

    true_set = set(int(x) for x in gt_i)
    recall   = len(set(beam[:K]) & true_set) / K
    true_nn  = int(gt_i[0])

    bridge = None
    if recall < 1.0:
        non_true = [nid for nid in reversed(beam) if nid not in true_set]
        if non_true:
            bridge = (non_true[0], true_nn)

    return recall, bridge



# Rolling mean helper
def rolling_mean(arr, w):
    out = np.empty(len(arr))
    for i in range(len(arr)):
        lo = max(0, i - w + 1)
        out[i] = arr[lo:i+1].mean()
    return out




def main():
    queries, gt = load_dataset()
    dim = queries.shape[1]

    params_path = os.path.join(args.data_dir, "params.json")
    with open(params_path) as f:
        n_index = json.load(f)["n_index"]


    # Step 1: hardness ordering
    print(f"\n[1] Hardness scoring at ef={EF} ...")
    idx = load_index(dim, n_index)
    idx.set_ef(EF)
    hardness = np.empty(len(queries), dtype=np.int32)
    for i, q in enumerate(queries):
        idx.knn_query(q.reshape(1, -1), k=K)
        hardness[i] = int(hnswlib.get_last_query_stats()["base_layer_distance_computations"])

    sorted_order = np.argsort(hardness)[::-1]

    n_total = len(queries) if args.n_queries == 0 else args.n_queries
    if args.dry_run:
        n_total = min(200, n_total)
    stream_idx  = sorted_order[:n_total]
    stream_q    = queries[stream_idx]
    stream_gt   = gt[stream_idx]
    stream_hard = hardness[stream_idx]

    print(f"  streaming {n_total:,} queries  "
          f"hardness [{stream_hard.min()}, {stream_hard.max()}]")


    # Step 2: Baseline pass (no edge insertions)
    print(f"\n[2] Baseline pass (no adaptation) ...")
    # reuse idx — no modifications yet
    base_recalls = np.empty(n_total, dtype=np.float32)
    for i in range(n_total):
        base_recalls[i], _ = run_query(idx, stream_q[i], stream_gt[i])

    print(f"  baseline mean recall: {base_recalls.mean():.4f}")


    # Step 3: Adaptive pass (fresh index, add edge after each query)
    print(f"\n[3] Adaptive streaming pass ...")
    idx_adapt = load_index(dim, n_index)

    adapt_recalls = np.empty(n_total, dtype=np.float32)
    cumulative_edges = np.zeros(n_total, dtype=np.int32)
    edges_added = 0
    edges_skipped = 0

    for i in range(n_total):
        q = stream_q[i]
        gt_i = stream_gt[i]

        recall, bridge = run_query(idx_adapt, q, gt_i)
        adapt_recalls[i] = recall

        if bridge is not None:
            src, dst = bridge
            if src != dst:
                idx_adapt.add_back_edge(src, dst, 0)
                edges_added += 1
            else:
                edges_skipped += 1
        else:
            edges_skipped += 1

        cumulative_edges[i] = edges_added

        if i % 100 == 0 or i == n_total - 1:
            roll_b = base_recalls[max(0, i-WINDOW+1):i+1].mean()
            roll_a = adapt_recalls[max(0, i-WINDOW+1):i+1].mean()
            print(f"  [{i:>5}/{n_total}]  edges={edges_added:>5}  "
                  f"roll_base={roll_b:.4f}  roll_adapt={roll_a:.4f}  "
                  f"delta={roll_a-roll_b:+.4f}")

    gains = adapt_recalls - base_recalls


    # Step 4: Per-quartile breakdown
    print(f"\n--- breakdown by stream quartile (Q1 = first 25%, hardest) ---")
    splits = np.array_split(np.arange(n_total), 4)
    labels_q = ["Q1 (first)", "Q2       ", "Q3       ", "Q4 (last)"]
    print(f"{'':4}  {'label':>10}  {'hardness rng':>18}  "
          f"{'n':>5}  {'base':>7}  {'adapt':>7}  {'gain':>7}")
    for qi, qidx in enumerate(splits):
        hs_lo = stream_hard[qidx].min()
        hs_hi = stream_hard[qidx].max()
        print(f"  {labels_q[qi]}  [{hs_lo:5}, {hs_hi:5}]  "
              f"n={len(qidx):>5}  "
              f"{base_recalls[qidx].mean():>7.4f}  "
              f"{adapt_recalls[qidx].mean():>7.4f}  "
              f"{gains[qidx].mean():>+7.4f}")


    # Step 5: Plot
    roll_base = rolling_mean(base_recalls,  WINDOW)
    roll_adapt = rolling_mean(adapt_recalls, WINDOW)
    roll_gain = roll_adapt - roll_base

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # panel 1: rolling recall
    ax = axes[0]
    ax.plot(roll_base,  color="steelblue",  linewidth=1.2, label="baseline (static index)")
    ax.plot(roll_adapt, color="darkorange", linewidth=1.2, label="adaptive (online edges)")
    ax.set_ylabel(f"rolling recall@{K}  (window={WINDOW})")
    ax.set_title(f"Online base-layer edge insertion — {n_total} queries (hardness order)\n"
        f"ef={EF}, k={K}, edges added={edges_added}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # panel 2: rolling gain
    ax = axes[1]
    ax.plot(roll_gain, color="seagreen", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"rolling gain  (adapt − base)")
    ax.grid(True, alpha=0.3)

    # panel 3: cumulative edges added
    ax = axes[2]
    ax.plot(cumulative_edges, color="slategray", linewidth=1.0)
    ax.set_xlabel("query index (hardness order, hardest first)")
    ax.set_ylabel("cumulative edges added")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  saved -> {plot_path}")


    # Summary
    n_improved = int((gains >  0).sum())
    n_unchanged = int((gains == 0).sum())
    n_hurt = int((gains <  0).sum())

    print(f"\n{'='*56}")
    print(f"SUMMARY  (online edge insertion, stream of {n_total} queries)")
    print(f"  stream order:       hardness-descending (hardest first)")
    print(f"  edges added:        {edges_added}")
    print(f"  edges skipped:      {edges_skipped}")
    print(f"  recall baseline:    {base_recalls.mean():.4f}")
    print(f"  recall adaptive:    {adapt_recalls.mean():.4f}")
    print(f"  mean gain:          {gains.mean():+.4f}")
    print(f"  queries improved:   {n_improved}")
    print(f"  queries unchanged:  {n_unchanged}")
    print(f"  queries hurt:       {n_hurt}")
    late_half = n_total // 2
    print(f"  gain (first half):  {gains[:late_half].mean():+.4f}")
    print(f"  gain (second half): {gains[late_half:].mean():+.4f}")
    print(f"  (positive trend = edges accumulate benefit over stream)")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()
