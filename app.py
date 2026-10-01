import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Book Recommendation System - Project 708",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_artifacts():
    # Modular paths fallback to single artifact file
    svd_path = os.path.join("models", "retrieval_model", "retrieval_svd.pkl")
    ranker_path = os.path.join("models", "ranking_model", "ranking_model.pkl")
    meta_path = os.path.join("artifacts", "metadata", "metadata.joblib")
    encoder_path = os.path.join("artifacts", "encoders", "label_encoders.joblib")
    bundle_path = "model_artifacts.joblib"
    
    if os.path.exists(svd_path) and os.path.exists(ranker_path) and os.path.exists(meta_path) and os.path.exists(encoder_path):
        svd_model = joblib.load(svd_path)
        ranker_model = joblib.load(ranker_path)
        encoders = joblib.load(encoder_path)
        meta = joblib.load(meta_path)
        
        return {
            "svd_model": svd_model,
            "ranker": ranker_model,
            "user_map": encoders["user_map"],
            "item_map": encoders["item_map"],
            "user_factors": svd_model.fit_transform if hasattr(svd_model, 'components_') else None, # handled below
            "item_factors": svd_model.components_.T if hasattr(svd_model, 'components_') else None,
            **meta
        }
    elif os.path.exists(bundle_path):
        return joblib.load(bundle_path)
    return None

artifacts = load_all_artifacts()

st.markdown('<div class="main-title">📚 Book Recommendation System (Project 708)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Two-Stage Hybrid Architecture: TruncatedSVD Candidate Retrieval + LightGBM Lambdarank</div>', unsafe_allow_html=True)

if artifacts is None:
    st.error("⚠️ Model artifacts not found. Please run model building (`python 02_model_building_and_evaluation.py`) first!")
    st.stop()

# Unpack artifacts
user_map = artifacts["user_map"]
item_map = artifacts["item_map"]
user_ids = artifacts["user_ids"]
item_ids = artifacts["item_ids"]
user_factors = artifacts.get("user_factors")
item_factors = artifacts.get("item_factors")
ranker = artifacts["ranker"]
best_iter = artifacts.get("best_iter", None)
pop_order = artifacts["pop_order"]
seen_eval = artifacts["seen_eval"]
has_signal = artifacts["has_signal"]
user_arr = artifacts["user_arr"]
item_arr = artifacts["item_arr"]
books_dict = artifacts["books_df"]
user_feat_dict = artifacts.get("user_feat_df", {})
users_cleaned_dict = artifacts.get("users_cleaned", {})

n_items = len(item_ids)

# Sidebar UI
st.sidebar.header("⚙️ Recommendation Controls")
sample_users = [276729, 276744, 276747, 276748, 276755, 276762, 276772, 276786, 276788, 276822]
existing_users = [u for u in sample_users if u in user_map] or list(user_ids[:10])

mode = st.sidebar.radio("Input Mode:", ["Select Sample User", "Enter Custom User ID"])
if mode == "Select Sample User":
    selected_user = st.sidebar.selectbox("Select User ID:", existing_users)
else:
    selected_user = int(st.sidebar.number_input("Enter User ID:", min_value=1, value=int(existing_users[0])))

top_k = st.sidebar.slider("Top-K Recommendations:", min_value=1, max_value=20, value=10)
engine = st.sidebar.selectbox("Recommendation Engine:", ["Two-Stage Hybrid (SVD + LightGBM)", "Stage 1 Retrieval Only", "Popularity Fallback"])

def recommend_books(user_id, k=10, model_type="Two-Stage Hybrid (SVD + LightGBM)"):
    u_idx = user_map.get(user_id, -1)
    is_cold = (u_idx < 0) or (not has_signal[u_idx])
    
    if is_cold or model_type == "Popularity Fallback":
        s_items = seen_eval.indices[seen_eval.indptr[u_idx]:seen_eval.indptr[u_idx + 1]] if u_idx >= 0 else np.array([], dtype=np.int64)
        top_idx = pop_order[: k + len(s_items)]
        top_idx = top_idx[~np.isin(top_idx, s_items)][:k]
        
        recs = []
        for rank, item_idx in enumerate(top_idx, 1):
            isbn = item_ids[item_idx]
            b = books_dict.get(isbn, {})
            recs.append({
                "Rank": rank, "ISBN": isbn, "Title": b.get("Book-Title", "Unknown Title"),
                "Author": b.get("Book-Author", "Unknown Author"), "Publisher": b.get("Publisher", "Unknown"),
                "Year": b.get("Year-Of-Publication", "N/A"), "Retrieval Score": 0.0,
                "Rank Score": f"Popularity #{rank}", "Image": b.get("Image-URL-M", "")
            })
        return pd.DataFrame(recs), is_cold

    scores = user_factors[u_idx] @ item_factors.T
    s_items = seen_eval.indices[seen_eval.indptr[u_idx]:seen_eval.indptr[u_idx + 1]]
    scores[s_items] = -np.inf
    
    cand_k = min(100, n_items)
    idx = np.argpartition(-scores, cand_k - 1)[:cand_k]
    idx = idx[np.argsort(-scores[idx])]
    cand_scores = scores[idx]
    
    cand_df = pd.DataFrame({"u": u_idx, "i": idx, "svd_score": cand_scores.astype(np.float32), "svd_rank": np.arange(1, len(idx) + 1, dtype=np.float32)})
    
    if model_type == "Stage 1 Retrieval Only":
        top = cand_df.head(k)
        recs = []
        for rank, row in enumerate(top.itertuples(), 1):
            isbn = item_ids[int(row.i)]
            b = books_dict.get(isbn, {})
            recs.append({
                "Rank": rank, "ISBN": isbn, "Title": b.get("Book-Title", "Unknown Title"),
                "Author": b.get("Book-Author", "Unknown Author"), "Publisher": b.get("Publisher", "Unknown"),
                "Year": b.get("Year-Of-Publication", "N/A"), "Retrieval Score": f"{row.svd_score:.4f}",
                "Rank Score": f"{row.svd_score:.4f}", "Image": b.get("Image-URL-M", "")
            })
        return pd.DataFrame(recs), False

    mat = np.hstack([np.tile(user_arr[u_idx], (len(cand_df), 1)), item_arr[cand_df["i"].to_numpy()], cand_df[["svd_score", "svd_rank"]].to_numpy(np.float32)])
    cand_df["rank_score"] = ranker.predict(mat, num_iteration=best_iter)
    top = cand_df.nlargest(k, "rank_score")
    
    recs = []
    for rank, row in enumerate(top.itertuples(), 1):
        isbn = item_ids[int(row.i)]
        b = books_dict.get(isbn, {})
        recs.append({
            "Rank": rank, "ISBN": isbn, "Title": b.get("Book-Title", "Unknown Title"),
            "Author": b.get("Book-Author", "Unknown Author"), "Publisher": b.get("Publisher", "Unknown"),
            "Year": b.get("Year-Of-Publication", "N/A"), "Retrieval Score": f"{row.svd_score:.4f}",
            "Rank Score": f"{row.rank_score:.4f}", "Image": b.get("Image-URL-M", "")
        })
    return pd.DataFrame(recs), False

tab1, tab2, tab3 = st.tabs(["🎯 Recommendation Interface", "📊 Model Metrics & Benchmarks", "ℹ️ Pipeline Architecture"])

with tab1:
    u_idx = user_map.get(selected_user, -1)
    u_prof = users_cleaned_dict.get(selected_user, {})
    u_feat = user_feat_dict.get(u_idx, {}) if u_idx >= 0 else {}
    
    st.subheader(f"User Profile Details (User ID: #{selected_user})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age Group", u_prof.get("Age_Group", "Unknown"))
    c2.metric("Country", u_prof.get("Country", "Unknown"))
    c3.metric("Explicit Rating Count", int(u_feat.get("User_Rating_Count", 0)))
    avg_r = u_feat.get("User_Average_Rating", None)
    c4.metric("Avg Explicit Rating", f"{avg_r:.2f}" if avg_r is not None else "N/A")
    
    recs_df, is_cold = recommend_books(selected_user, k=top_k, model_type=engine)
    if is_cold:
        st.warning("⚠️ **Cold-Start User**: Minimal interaction history available. Recommending popular fallback books.")
    else:
        st.success(f"✨ Custom recommendations generated using **{engine}**")
        
    for _, row in recs_df.iterrows():
        col_img, col_txt = st.columns([1, 6])
        with col_img:
            img = str(row["Image"])
            if img and img.startswith("http"):
                st.image(img, width=85)
            else:
                st.markdown("📖")
        with col_txt:
            st.markdown(f"**#{row['Rank']}. {row['Title']}**")
            st.markdown(f"**Author:** {row['Author']} | **Publisher:** {row['Publisher']} ({row['Year']})")
            st.markdown(f"**ISBN:** `{row['ISBN']}` | **Retrieval Score:** `{row['Retrieval Score']}` | **Rank Score:** `{row['Rank Score']}`")
            st.markdown("---")

with tab2:
    st.subheader("📊 Empirical Offline Benchmark Metrics")
    eval_file = "two_stage_evaluation_results.csv"
    if os.path.exists(eval_file):
        st.dataframe(pd.read_csv(eval_file), use_container_width=True)

with tab3:
    st.markdown("### Two-Stage Pipeline Architecture Overview")
    st.info("Stage 1 retrieves 100 candidate items per user via SVD latent dot product. Stage 2 reranks candidates using LightGBM pairwise lambdarank.")
