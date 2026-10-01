import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

class SVDClientRetriever:
    def __init__(self, n_factors=64, seed=42):
        self.n_factors = n_factors
        self.seed = seed
        self.svd = TruncatedSVD(n_components=n_factors, random_state=seed)
        self.user_factors = None
        self.item_factors = None

    def fit(self, R_centered):
        self.user_factors = self.svd.fit_transform(R_centered).astype(np.float32)
        self.item_factors = self.svd.components_.T.astype(np.float32)
        return self

    def retrieve(self, user_indices, seen_matrix, has_signal, pop_order, n_items, top_k=100, batch_size=256):
        user_indices = np.asarray(user_indices)
        k = min(top_k, n_items)
        parts = []
        
        def pop_unseen(u):
            seen = seen_matrix.indices[seen_matrix.indptr[u]:seen_matrix.indptr[u + 1]]
            head = pop_order[: k + len(seen)]
            return head[~np.isin(head, seen)][:k]

        def _topk(scores):
            n = scores.shape[1]
            k_val = min(k, n)
            if k_val < n:
                idx = np.argpartition(-scores, k_val - 1, axis=1)[:, :k_val]
            else:
                idx = np.tile(np.arange(n), (scores.shape[0], 1))
            part = np.take_along_axis(scores, idx, axis=1)
            order = np.argsort(-part, axis=1)
            return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)

        for s in range(0, len(user_indices), batch_size):
            b = user_indices[s:s + batch_size]
            scores = self.user_factors[b] @ self.item_factors.T
            coo = seen_matrix[b].tocoo()
            scores[coo.row, coo.col] = -np.inf
            idx, sc = _topk(scores)
            
            for row in np.where(~has_signal[b])[0]:
                lst = pop_unseen(b[row])
                idx[row, :] = -1
                sc[row, :] = -np.inf
                idx[row, :len(lst)] = lst
                sc[row, :len(lst)] = 0.0
                
            ok = np.isfinite(sc).ravel()
            parts.append(pd.DataFrame({
                "u": np.repeat(b, k)[ok],
                "i": idx.ravel()[ok],
                "svd_score": sc.ravel()[ok],
                "svd_rank": np.tile(np.arange(1, k + 1), len(b))[ok],
            }))
            
        out = pd.concat(parts, ignore_index=True)
        return out.astype({"u": "int64", "i": "int64", "svd_score": "float32", "svd_rank": "float32"})
