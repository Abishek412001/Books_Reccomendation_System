import numpy as np
import pandas as pd

def user_eval_metrics(recs, rel, nrel, k):
    recs = recs[:k]
    hits = np.isin(recs, rel).astype(np.float64)
    n_hit = hits.sum()
    hit_rate = 1.0 if n_hit > 0 else 0.0
    disc = 1.0 / np.log2(np.arange(2, len(recs) + 2))
    ideal = (1.0 / np.log2(np.arange(2, min(nrel, k) + 2))).sum()
    precision = n_hit / k
    recall = n_hit / nrel
    ndcg = (hits * disc).sum() / ideal if ideal > 0 else 0.0
    return precision, recall, ndcg, hit_rate

def evaluate_predictions(rec_lists, rel_items, n_rel, eval_users, k):
    EMPTY = np.empty(0, dtype=np.int64)
    rows = [user_eval_metrics(rec_lists.get(u, EMPTY), rel_items.get(u, EMPTY), n_rel[u], k) for u in eval_users]
    p, r, nd, hr = np.mean(rows, axis=0)
    return {f"Precision@{k}": p, f"Recall@{k}": r, f"NDCG@{k}": nd, f"HitRate@{k}": hr}
