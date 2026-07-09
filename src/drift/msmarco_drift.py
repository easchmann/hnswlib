"""MS MARCO-specific data loading, embedding, qrel-based split, and drift sequence."""

import os

import numpy as np


def load_msmarco_passages(tsv_path, n_vectors):
    """Read first n_vectors lines from collection.tsv."""
    passage_ids, texts = [], []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            pid, text = line.rstrip('\n').split('\t', 1)
            passage_ids.append(int(pid))
            texts.append(text)
            if len(texts) == n_vectors:
                break
    return passage_ids, texts


def load_msmarco_queries(tsv_path):
    """Read queries.dev.small.tsv. Returns (query_ids, texts)."""
    query_ids, texts = [], []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            qid, text = line.rstrip('\n').split('\t', 1)
            query_ids.append(int(qid))
            texts.append(text)
    return query_ids, texts


def load_qrels(qrels_path):
    """Read qrels.dev.small.tsv. Returns dict mapping query_id -> passage_id."""
    qrels = {}
    with open(qrels_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            try:
                qid, pid = int(parts[0]), int(parts[2])
            except ValueError:
                continue  # skip header or malformed lines
            qrels[qid] = pid
    return qrels


def embed_texts(texts, model_name, batch_size, cache_path, show_progress=True):
    """Embed texts with sentence-transformers. Returns float32 L2-normalised array."""
    from sentence_transformers import SentenceTransformer

    sentinel = cache_path + '.done'
    if os.path.exists(sentinel) and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached) == len(texts):
            print(f"  Loading cached embeddings from {cache_path}")
            return cached
        print(f"  Cache size mismatch ({len(cached)} vs {len(texts)}) — re-embedding...")
        os.remove(sentinel)

    model = SentenceTransformer(model_name)
    model.max_seq_length = 128

    chunk_size = 50_000
    n = len(texts)
    dim = 384

    out = np.zeros((n, dim), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = model.encode(
            texts[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        out[start:end] = chunk
        if show_progress:
            print(f"  embedded {end:,}/{n:,}")

    np.save(cache_path, out)
    open(sentinel, 'w').close()
    return out


def split_queries_by_coverage(query_ids, query_vecs, qrels, base_passage_id_set):
    """Split queries into in-distribution and OOD pools based on qrels."""
    in_dist, ood = [], []
    n_no_qrel = 0
    for i, qid in enumerate(query_ids):
        if qid not in qrels:
            n_no_qrel += 1
            ood.append(query_vecs[i])
            continue
        pid = qrels[qid]
        if pid in base_passage_id_set:
            in_dist.append(query_vecs[i])
        else:
            ood.append(query_vecs[i])

    in_dist_vecs = np.array(in_dist, dtype=np.float32)
    ood_vecs = np.array(ood, dtype=np.float32)

    diagnostics = {
        "n_total_queries": len(query_ids),
        "n_in_dist": len(in_dist_vecs),
        "n_ood": len(ood_vecs),
        "n_no_qrel": n_no_qrel,
    }
    return in_dist_vecs, ood_vecs, diagnostics


def build_answerable_drift_sequence(in_dist_vecs, ood_vecs, schedule, epoch_size, base_seed):
    """Build query epochs interpolating from answerable to unanswerable."""
    epochs = []
    for epoch_idx, t in enumerate(schedule):
        rng = np.random.default_rng(base_seed + epoch_idx)
        n_ood = int(round(t * epoch_size))
        n_in = epoch_size - n_ood

        parts = []
        if n_in > 0:
            idx = rng.integers(0, len(in_dist_vecs), size=n_in)
            parts.append(in_dist_vecs[idx])
        if n_ood > 0:
            idx = rng.integers(0, len(ood_vecs), size=n_ood)
            parts.append(ood_vecs[idx])

        epoch = np.concatenate(parts, axis=0)
        rng.shuffle(epoch)
        epochs.append(epoch.astype(np.float32))
    return epochs
