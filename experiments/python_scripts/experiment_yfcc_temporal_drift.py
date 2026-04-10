# Temporal drift experiment on YFCC DINO ViT-S/16 embeddings.
#
# Index is built on images uploaded in 2007-2009/2007. 
# Query sets are drawn from all years (2007-2013), one shift level per year.
#
# Saves summary.csv, per_query.csv, lb_traces.csv, params.csv

import os
import argparse
import numpy as np
import pandas as pd

import hnswlib

rng = np.random.default_rng(42)

INDEX_YEARS = [2007]
QUERY_YEARS = [2007, 2008, 2009, 2010, 2011, 2012, 2013]

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_path", required=True,
                    help="path to embeddings_float32.npy  (N, 384)")
parser.add_argument("--metadata_path", required=True,
                    help="path to metadata.csv  (photo_id, year, month)")
parser.add_argument("--n_index", type=int, default=None,
                    help="cap on index vectors (default: use all 2007-2009 vectors)")
parser.add_argument("--n_queries", type=int, default=1_000,
                    help="queries per year (default: 1000)")
parser.add_argument("--ef_sweep", type=int, nargs="+",
                    default=[10, 20, 50, 100, 200, 500])
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--ef_construction", type=int, default=200)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--out_dir", default="results_yfcc_temporal_drift")
args = parser.parse_args()

dim = 384
os.makedirs(args.out_dir, exist_ok=True)


# --- loading -----------------------------------------------------------------

def load_embeddings():
    print(f"loading embeddings from {args.embeddings_path}...")
    emb  = np.load(args.embeddings_path, mmap_mode='r')
    meta = pd.read_csv(args.metadata_path)
    assert len(emb) == len(meta), \
        f"embedding/metadata length mismatch: {len(emb)} vs {len(meta)}"
    print(f"  total: {len(emb):,} embeddings, dim={emb.shape[1]}")
    print(f"  year distribution:\n{meta.groupby('year').size().to_string()}")
    return emb, meta


def split_by_year(emb, meta):
    # index: 2007-2009
    index_mask = meta['year'].isin(INDEX_YEARS).values
    index_vecs = np.array(emb[index_mask], dtype=np.float32)
    if args.n_index and len(index_vecs) > args.n_index:
        chosen     = rng.choice(len(index_vecs), args.n_index, replace=False)
        index_vecs = index_vecs[chosen]
    print(f"\nindex vectors ({INDEX_YEARS}): {len(index_vecs):,}")

    # queries: one set per year from 2010-2013
    query_sets = {}
    for year in QUERY_YEARS:
        year_mask  = (meta['year'] == year).values
        year_vecs  = np.array(emb[year_mask], dtype=np.float32)

        if len(year_vecs) < args.n_queries:
            print(f"  warning: {year} has only {len(year_vecs)} vectors, "
                  f"sampling with replacement")
            chosen = rng.choice(len(year_vecs), args.n_queries, replace=True)
        else:
            chosen = rng.choice(len(year_vecs), args.n_queries, replace=False)

        query_sets[year] = year_vecs[chosen]
        print(f"  queries {year}: {len(year_vecs):,} available, using {args.n_queries}")

    return index_vecs, query_sets


# --- ground truth ------------------------------------------------------------

def compute_gt(index_vecs, queries):
    import faiss
 
    # usingCPU intentionally since faiss-gpu hits a cuBLAS workspace limit doing the full matrix multiply in one shot
    # easiest fix, still fast enough
    print(f"  faiss CPU")
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(index_vecs, dtype=np.float32))
    _, neighbours = index.search(
        np.ascontiguousarray(queries, dtype=np.float32), args.k
    )
    return neighbours.astype(np.int32)


# --- index + querying --------------------------------------------------------

def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(
        max_elements=len(data),
        ef_construction=args.ef_construction,
        M=args.M
    )
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx


def recall(predicted, ground_truth):
    hits = sum(len(set(p) & set(g)) for p, g in zip(predicted, ground_truth))
    return hits / (len(ground_truth) * ground_truth.shape[1])


def query_index(idx, queries, gt):
    results = []
    for i, q in enumerate(queries):
        pred, _ = idx.knn_query(q.reshape(1, -1), k=args.k)
        s = hnswlib.get_last_query_stats()
        results.append({
            "recall":               recall(pred, gt[i:i+1]),
            "ep_dist":              float(s["entry_point_distance"]),
            "bl_entry_dist":        float(s["base_layer_entry_distance"]),
            "ul_dist_comps":        int(s["upper_layer_distance_computations"]),
            "layer_visits":         list(s["layer_visit_counts"]),
            "base_visited":         int(s["base_layer_visited_count"]),
            "base_dist_comps":      int(s["base_layer_distance_computations"]),
            "candidates_remaining": int(s["candidates_remaining_at_termination"]),
            "lb_trace":             list(s["lowerbound_trace"]),
        })
    return results


# --- saving ------------------------------------------------------------------

def save_params(path):
    rows = [
        {"param": "dataset",          "value": "YFCC-DINO"},
        {"param": "dim",              "value": dim},
        {"param": "index_years",      "value": str(INDEX_YEARS)},
        {"param": "query_years",      "value": str(QUERY_YEARS)},
        {"param": "n_index",          "value": args.n_index or "all"},
        {"param": "n_queries",        "value": args.n_queries},
        {"param": "ef_sweep",         "value": str(args.ef_sweep)},
        {"param": "M",                "value": args.M},
        {"param": "ef_construction",  "value": args.ef_construction},
        {"param": "k",                "value": args.k},
        {"param": "split_method",     "value": "temporal_year"},
        {"param": "embeddings_path",  "value": args.embeddings_path},
        {"param": "rng_seed",         "value": 42},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


def save_summary(all_results, path):
    rows = []
    for (year, ef), results in sorted(all_results.items()):
        def avg(field):
            return float(np.nanmean([r[field] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "query_year":                   year,
            "ef_search":                    ef,
            "mean_recall":                  avg("recall"),
            "mean_ep_distance":             avg("ep_dist"),
            "mean_bl_entry_distance":       avg("bl_entry_dist"),
            "mean_ul_dist_comps":           avg("ul_dist_comps"),
            "mean_layer1_visits":           float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited":            avg("base_visited"),
            "mean_base_dist_comps":         avg("base_dist_comps"),
            "mean_candidates_remaining":    avg("candidates_remaining"),
            "n_queries":                    len(results),
        })

    df = pd.DataFrame(rows).round(4)
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    pivot = df.pivot(index="query_year", columns="ef_search",
                     values="mean_recall").round(3)
    print("\nRecall@k (rows=query year, cols=ef):")
    print(pivot.to_string())


def save_per_query(all_results, path):
    rows = []
    for (year, ef), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "query_year":           year,
                "ef_search":            ef,
                "query_id":             i,
                "recall":               r["recall"],
                "ep_dist":              r["ep_dist"],
                "bl_entry_dist":        r["bl_entry_dist"],
                "ul_dist_comps":        r["ul_dist_comps"],
                "base_visited":         r["base_visited"],
                "base_dist_comps":      r["base_dist_comps"],
                "candidates_remaining": r["candidates_remaining"],
                "lb_trace_final":       r["lb_trace"][-1] if r["lb_trace"] else float("nan"),
                "lb_trace_len":         len(r["lb_trace"]),
            }
            for layer, visits in enumerate(r["layer_visits"]):
                row[f"layer{layer}_visits"] = visits
            rows.append(row)
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def save_lb_traces(all_results, path):
    rows = []
    for (year, ef), results in sorted(all_results.items()):
        traces = [r["lb_trace"] for r in results if r["lb_trace"]]
        if not traces:
            continue
        min_len    = min(len(t) for t in traces)
        mean_trace = np.mean([t[:min_len] for t in traces], axis=0)
        for iteration, value in enumerate(mean_trace):
            rows.append({
                "query_year":      year,
                "ef_search":       ef,
                "iteration":       iteration,
                "mean_lowerbound": round(float(value), 6),
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


# --- main --------------------------------------------------------------------

def main():
    print(f"YFCC temporal drift experiment")
    print(f"index: {INDEX_YEARS}  queries: {QUERY_YEARS}")
    print(f"n_queries={args.n_queries}  k={args.k}  ef_sweep={args.ef_sweep}\n")

    emb, meta = load_embeddings()
    index_vecs, query_sets = split_by_year(emb, meta)
    del emb  # free memmap

    print(f"\nbuilding index on {len(index_vecs):,} vectors...")
    idx = build_index(index_vecs)
    print(f"  done  (M={args.M}, ef_construction={args.ef_construction})")

    print(f"\ncomputing ground truth (faiss)...")
    ground_truths = {}
    for year, queries in sorted(query_sets.items()):
        print(f"  year={year}  n={len(queries)}")
        ground_truths[year] = compute_gt(index_vecs, queries)
    print("  done")

    print(f"\nquerying: {len(query_sets)} years x {len(args.ef_sweep)} ef values")
    all_results = {}

    for ef in args.ef_sweep:
        idx.set_ef(ef)
        print(f"\nef={ef}")
        for year, queries in sorted(query_sets.items()):
            results = query_index(idx, queries, ground_truths[year])
            all_results[(year, ef)] = results
            mean_r  = np.mean([r["recall"] for r in results])
            mean_ep = np.nanmean([r["ep_dist"] for r in results])
            mean_bl = np.nanmean([r["bl_entry_dist"] for r in results])
            print(f"  year={year}  recall={mean_r:.4f}  "
                  f"ep={mean_ep:.3f}  bl_entry={mean_bl:.3f}")

    print("\nsaving results...")
    save_params(os.path.join(args.out_dir, "params.csv"))
    save_summary(all_results, os.path.join(args.out_dir, "summary.csv"))
    save_per_query(all_results, os.path.join(args.out_dir, "per_query.csv"))
    save_lb_traces(all_results, os.path.join(args.out_dir, "lb_traces.csv"))
    print(f"\ndone — results in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()