# Build an "isolated cluster" dataset for testing online base-layer rewiring.
#
# Finds the node that appears most frequently across the ef-beams of the hardest queries,
# the universal bottleneck. Then keeps only queries that visit it, sorted by hardness. 
# When these queries are streamed online and we add bridge edges at the bottleneck, 
# every subsequent query in the stream also visits it and can benefit from those edges.
#
# Usage:
#   python experiments/analysis/build_isolated_cluster_dataset.py \
#       --src_dir data/sift_hardness \
#       --out_dir data/sift_isolated_cluster

import os, sys, json, argparse
from collections import Counter
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../.."))
import hnswlib

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--ef", type=int, default=20)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--n_scan", type=int, default=500)
args = parser.parse_args()

queries = np.load(os.path.join(args.src_dir, "queries_by_hardness.npy"))
gt = np.load(os.path.join(args.src_dir, "ground_truth.npy"))
with open(os.path.join(args.src_dir, "params.json")) as f:
    params = json.load(f)

dim, n_index = queries.shape[1], params["n_index"]

print("loading index ...")
idx = hnswlib.Index(space="l2", dim=dim)
idx.load_index(os.path.join(args.src_dir, "hnsw_index.bin"), max_elements=n_index)
idx.set_ef(args.ef)

# score all queries for hardness and take the top n_scan
print(f"scoring {len(queries)} queries ...")
hardness = np.empty(len(queries), dtype=np.int32)
for i, q in enumerate(queries):
    idx.knn_query(q.reshape(1, -1), k=args.k)
    hardness[i] = hnswlib.get_last_query_stats()["base_layer_distance_computations"]

top_idx = np.argsort(hardness)[::-1][:args.n_scan]

# collect ef-beams for the hardest queries and count node frequencies
print(f"collecting beams for top {args.n_scan} hardest queries ...")
node_freq = Counter()
beams = {}
for qi in top_idx:
    labels, _ = idx.knn_query(queries[qi].reshape(1, -1), k=args.ef)
    beam = [int(x) for x in labels[0]]
    beams[qi] = beam
    node_freq.update(beam)

bottleneck, count = node_freq.most_common(1)[0]
print(f"universal bottleneck node: {bottleneck}  (appears in {count}/{args.n_scan} beams)")

# keep only queries that visit the bottleneck node, sorted hardest-first
selected = [qi for qi in top_idx if bottleneck in beams[qi]]
selected = sorted(selected, key=lambda qi: hardness[qi], reverse=True)
print(f"queries visiting bottleneck: {len(selected)}")

# show how spread out the true NNs are (sanity check for geographic concentration)
true_nn_vecs = idx.get_items([int(gt[qi][0]) for qi in selected])
dists = np.linalg.norm(true_nn_vecs - true_nn_vecs.mean(axis=0), axis=1)
print(f"true-NN spread (L2 to centroid): mean={dists.mean():.1f}  max={dists.max():.1f}")

os.makedirs(args.out_dir, exist_ok=True)
np.save(os.path.join(args.out_dir, "queries.npy"), queries[selected])
np.save(os.path.join(args.out_dir, "ground_truth.npy"), gt[selected])
with open(os.path.join(args.out_dir, "params.json"), "w") as f:
    json.dump({
        "n_index": n_index,
        "n_queries": len(selected),
        "bottleneck": bottleneck,
        "ef": args.ef,
        "k": args.k,
    }, f, indent=2)

print(f"saved {len(selected)} queries -> {args.out_dir}")
