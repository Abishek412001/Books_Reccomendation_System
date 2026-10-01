import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "."
SEED = 42
REL_THRESHOLD = 7              # rating >= 7 is relevant positive
MIN_TRAIN_RATINGS = 5          # eligibility threshold
K_LIST = [5, 10, 20]
N_CANDIDATES = 100             # Top-100 candidates per user from retrieval
N_FACTORS = 64
VALID_FRAC, TEST_FRAC = 0.15, 0.20
EVAL_MAX_USERS = 500           # 500 test users evaluation sample
BATCH = 256

rng = np.random.default_rng(SEED)

print("--- Phase 4 Audit & Model Pipeline Execution ---")

# 1. Load Engineered CSVs
print("Loading data...")
inter = pd.read_csv(f"{DATA_DIR}/interaction_features.csv", usecols=["User-ID", "ISBN", "Book-Rating"])
books_cleaned = pd.read_csv(f"{DATA_DIR}/books_cleaned_engineered.csv")
users_cleaned = pd.read_csv(f"{DATA_DIR}/users_cleaned_engineered.csv")

books = books_cleaned.drop_duplicates("ISBN")[["ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication", "Image-URL-M"]].copy()
books["Book-Title"] = books["Book-Title"].fillna("")
books["Book-Author"] = books["Book-Author"].fillna("")
books["Publisher"] = books["Publisher"].fillna("")

inter = inter.dropna(subset=["User-ID", "ISBN"]).copy()
inter["User-ID"] = inter["User-ID"].astype("int64")
inter["ISBN"] = inter["ISBN"].astype(str).str.strip().str.upper()
inter["Book-Rating"] = inter["Book-Rating"].astype("float32")
inter = inter.drop_duplicates(["User-ID", "ISBN"], keep="last")

explicit = inter[inter["Book-Rating"] > 0].reset_index(drop=True)
implicit = inter[inter["Book-Rating"] == 0].reset_index(drop=True)

# 2. Per-User Train/Valid/Test Split (Preventing Data Leakage)
r = rng.random(len(explicit))
part = np.where(r < TEST_FRAC, "test",
        np.where(r < TEST_FRAC + VALID_FRAC, "valid", "train"))
train_df, valid_df, test_df = (explicit[part == p].copy() for p in ("train", "valid", "test"))

print(f"Explicit Train / Valid / Test count: {len(train_df):,} / {len(valid_df):,} / {len(test_df):,}")

# Index spaces derived FROM TRAIN ONLY
item_ids = np.sort(train_df["ISBN"].unique())
user_ids = np.union1d(train_df["User-ID"].unique(), implicit["User-ID"].unique())
item_map = pd.Series(np.arange(len(item_ids)), index=item_ids)
user_map = pd.Series(np.arange(len(user_ids)), index=user_ids)
n_items, n_users = len(item_ids), len(user_ids)

def encode(df):
    out = df.copy()
    out["u"] = out["User-ID"].map(user_map).fillna(-1).astype(np.int64)
    out["i"] = out["ISBN"].map(item_map).fillna(-1).astype(np.int64)
    return out

train, valid, test, imp = encode(train_df), encode(valid_df), encode(test_df), encode(implicit)

# User & Item Features built from TRAIN ONLY
global_mean = float(train["Book-Rating"].mean())
global_std = float(train["Book-Rating"].std())

user_feat = (train.groupby("u")["Book-Rating"]
                  .agg(User_Average_Rating="mean", User_Rating_Count="count", User_Rating_Std="std")
                  .reindex(np.arange(n_users)))
user_feat["User_Average_Rating"] = user_feat["User_Average_Rating"].fillna(global_mean)
user_feat["User_Rating_Count"] = user_feat["User_Rating_Count"].fillna(0)
user_feat["User_Rating_Std"] = user_feat["User_Rating_Std"].fillna(global_std)
user_feat["User_Implicit_Count"] = imp.groupby("u").size().reindex(user_feat.index).fillna(0)

item_feat = (train.groupby("i")["Book-Rating"]
                  .agg(Book_Average_Rating="mean", Book_Rating_Count="count")
                  .reindex(np.arange(n_items)))
item_feat["Book_Average_Rating"] = item_feat["Book_Average_Rating"].fillna(global_mean)
item_feat["Book_Rating_Count"] = item_feat["Book_Rating_Count"].fillna(0)
imp_item_cnt = imp[imp["i"] >= 0].groupby("i").size().reindex(item_feat.index).fillna(0)
item_feat["Log_Book_Popularity"] = np.log1p(item_feat["Book_Rating_Count"] + imp_item_cnt)

# 3. TF-IDF Text Content Vectorizer
print("Fitting TF-IDF Vectorizer on book text metadata...")
books_in_train = books[books["ISBN"].isin(item_ids)].set_index("ISBN").reindex(item_ids).reset_index()
books_in_train["text_content"] = (books_in_train["Book-Title"] + " " + 
                                  books_in_train["Book-Author"] + " " + 
                                  books_in_train["Publisher"])

tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(books_in_train["text_content"].fillna(""))

USER_COLS = ["User_Average_Rating", "User_Rating_Count", "User_Rating_Std", "User_Implicit_Count"]
ITEM_COLS = ["Book_Average_Rating", "Book_Rating_Count", "Log_Book_Popularity"]
FEATURES = USER_COLS + ITEM_COLS + ["svd_score", "svd_rank"]

user_arr = user_feat[USER_COLS].to_numpy(np.float32)
item_arr = item_feat[ITEM_COLS].to_numpy(np.float32)
train_cnt = user_feat["User_Rating_Count"].to_numpy()

def add_features(cand):
    mat = np.hstack([
        user_arr[cand["u"].to_numpy()],
        item_arr[cand["i"].to_numpy()],
        cand[["svd_score", "svd_rank"]].to_numpy(np.float32),
    ])
    return pd.DataFrame(mat, columns=FEATURES)

# 4. Stage 1 — TruncatedSVD Candidate Retrieval
print("Fitting TruncatedSVD model...")
train["centered"] = train["Book-Rating"] - train["u"].map(user_feat["User_Average_Rating"])
R = csr_matrix((train["centered"].to_numpy(np.float32), (train["u"], train["i"])),
               shape=(n_users, n_items))

svd = TruncatedSVD(n_components=N_FACTORS, random_state=SEED)
user_factors = svd.fit_transform(R).astype(np.float32)
item_factors = svd.components_.T.astype(np.float32)

has_signal = (train_cnt > 0) & (np.abs(user_factors).sum(axis=1) > 1e-9)
pop_order = (item_feat.sort_values(["Book_Rating_Count", "Book_Average_Rating"], ascending=False)
                      .index.to_numpy())

def build_seen(*frames):
    rows = np.concatenate([f.loc[(f["u"] >= 0) & (f["i"] >= 0), "u"].to_numpy() for f in frames])
    cols = np.concatenate([f.loc[(f["u"] >= 0) & (f["i"] >= 0), "i"].to_numpy() for f in frames])
    return csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_users, n_items))

seen_train = build_seen(train, imp)
seen_eval = build_seen(train, imp, valid)

def seen_items(seen, u):
    return seen.indices[seen.indptr[u]:seen.indptr[u + 1]]

def pop_unseen(seen_arr, k):
    head = pop_order[: k + len(seen_arr)]
    return head[~np.isin(head, seen_arr)][:k]

def _topk(scores, k):
    n = scores.shape[1]
    k = min(k, n)
    if k < n:
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    else:
        idx = np.tile(np.arange(n), (scores.shape[0], 1))
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)

def retrieve_candidates(user_idx, seen=seen_eval, top_k=N_CANDIDATES, batch=BATCH):
    user_idx = np.asarray(user_idx)
    k = min(top_k, n_items)
    parts = []
    for s in range(0, len(user_idx), batch):
        b = user_idx[s:s + batch]
        scores = user_factors[b] @ item_factors.T
        coo = seen[b].tocoo()
        scores[coo.row, coo.col] = -np.inf
        idx, sc = _topk(scores, k)
        for row in np.where(~has_signal[b])[0]:
            lst = pop_unseen(seen_items(seen, b[row]), k)
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

def label_candidates(cand, pos):
    pos = pos[pos["i"] >= 0]
    pos_keys = pos["u"].to_numpy(np.int64) * n_items + pos["i"].to_numpy(np.int64)
    keys = cand["u"].to_numpy(np.int64) * n_items + cand["i"].to_numpy(np.int64)
    cand["label"] = np.isin(keys, pos_keys).astype(np.int8)
    return cand

# 5. Stage 2 — Ranking Model Training
print("Training LightGBM Ranking Model...")
eligible = np.where(train_cnt >= MIN_TRAIN_RATINGS)[0]
valid_pos = valid[(valid["Book-Rating"] >= REL_THRESHOLD) & (valid["u"] >= 0)]

ranker_users = np.intersect1d(valid_pos["u"].unique(), eligible)
train_cand = label_candidates(retrieve_candidates(ranker_users, seen_train, top_k=N_CANDIDATES), valid_pos)

has_pos = train_cand.groupby("u")["label"].transform("max") > 0
train_cand = train_cand[has_pos]

users_with_pos = train_cand["u"].unique()
rng.shuffle(users_with_pos)
n_val = max(1, int(0.15 * len(users_with_pos)))
val_users = set(users_with_pos[:n_val].tolist())
is_val = train_cand["u"].isin(val_users).to_numpy()

def make_dataset(df, reference=None):
    df = df.sort_values(["u", "svd_rank"], kind="stable")
    groups = df.groupby("u", sort=True).size().to_numpy()
    return lgb.Dataset(add_features(df), label=df["label"].to_numpy(), group=groups, reference=reference)

dtrain = make_dataset(train_cand[~is_val])
dvalid = make_dataset(train_cand[is_val], reference=dtrain)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": SEED,
    "verbose": -1,
}
ranker = lgb.train(params, dtrain, num_boost_round=300, valid_sets=[dvalid],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
best_iter = ranker.best_iteration or None

# 6. Evaluation Framework (Precision@K, Recall@K, NDCG@K, Hit Rate@K for K = 5, 10, 20)
print(f"Evaluating models across K in {K_LIST}...")
test_pos = test[(test["Book-Rating"] >= REL_THRESHOLD) & (test["u"] >= 0)]
n_rel = test_pos.groupby("u").size().to_dict()
rel_items = {u: g["i"].to_numpy() for u, g in test_pos[test_pos["i"] >= 0].groupby("u")}

eval_users = np.intersect1d(test_pos["u"].unique(), eligible)
if EVAL_MAX_USERS is not None and len(eval_users) > EVAL_MAX_USERS:
    eval_users = np.sort(rng.choice(eval_users, EVAL_MAX_USERS, replace=False))

EMPTY = np.empty(0, dtype=np.int64)

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

def evaluate_model(rec_lists, k):
    rows = [user_eval_metrics(rec_lists.get(u, EMPTY), rel_items.get(u, EMPTY), n_rel[u], k) for u in eval_users]
    p, r, nd, hr = np.mean(rows, axis=0)
    return {f"Precision@{k}": p, f"Recall@{k}": r, f"NDCG@{k}": nd, f"HitRate@{k}": hr}

def to_lists(df):
    return {u: g["i"].to_numpy() for u, g in df.groupby("u", sort=False)}

eval_cand = retrieve_candidates(eval_users, seen_eval, top_k=N_CANDIDATES)
svd_lists = to_lists(eval_cand)

eval_cand["rank_score"] = ranker.predict(add_features(eval_cand), num_iteration=best_iter)
rank_lists = to_lists(eval_cand.sort_values(["u", "rank_score"], ascending=[True, False], kind="stable"))

results_list = []
for k in K_LIST:
    svd_res = evaluate_model(svd_lists, k)
    rank_res = evaluate_model(rank_lists, k)
    results_list.append({"Model": "Model A (Retrieval Only)", "K": k, **svd_res})
    results_list.append({"Model": "Model B (Retrieval + Ranking)", "K": k, **rank_res})

results_df = pd.DataFrame(results_list)
print("\n--- Empirical Benchmark Metrics ---")
print(results_df.to_string(index=False))
results_df.to_csv("two_stage_evaluation_results.csv", index=False)

# 7. Save Model Artifacts
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

joblib.dump(svd, "models/retrieval_svd.pkl")
joblib.dump(ranker, "models/ranking_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

artifacts = {
    "svd_model": svd,
    "user_factors": user_factors,
    "item_factors": item_factors,
    "user_map": user_map.to_dict(),
    "item_map": item_map.to_dict(),
    "user_ids": user_ids,
    "item_ids": item_ids,
    "ranker": ranker,
    "best_iter": best_iter,
    "pop_order": pop_order,
    "seen_eval": seen_eval,
    "has_signal": has_signal,
    "user_arr": user_arr,
    "item_arr": item_arr,
    "features": FEATURES,
    "global_mean": global_mean,
    "global_std": global_std,
    "books_df": books.set_index("ISBN").to_dict(orient="index"),
    "user_feat_df": user_feat.to_dict(orient="index"),
    "users_cleaned": users_cleaned.set_index("User-ID").to_dict(orient="index"),
    "tfidf_vectorizer": tfidf,
    "tfidf_matrix": tfidf_matrix
}

joblib.dump(artifacts, "model_artifacts.joblib", compress=3)
print("Artifacts saved successfully in models/ and model_artifacts.joblib!")
