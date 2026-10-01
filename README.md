# 📚 Book Recommendation System (Project 708)
> **Two-Stage Hybrid Architecture**: TruncatedSVD Candidate Retrieval + LightGBM Lambdarank

---

## 📌 Project Overview
This repository contains an end-to-end production Machine Learning pipeline for a **Book Recommendation System**. Built on the Book-Crossing dataset (~1.15 million ratings across ~278k users and ~271k books), the architecture uses a two-stage hybrid recommendation approach:

1. **Stage 1 — Candidate Retrieval (High Recall)**: TruncatedSVD matrix factorization (64-dimensional latent factors) generates top-100 candidates per user.
2. **Stage 2 — ML Ranking (High Precision)**: LightGBM pairwise lambdarank reranks candidates using non-leaked user statistics, item popularity, TF-IDF metadata similarity, and SVD scores.

---

## 🗂 Project Directory Structure

```text
Books_Reccomendation/
│
├── data/
│   ├── raw/                  # Raw Users, Books, and Ratings CSVs
│   └── processed/            # Cleaned and engineered feature datasets
│
├── notebooks/
│   └── Book_Recommendation.ipynb   # Complete exploratory & experiment notebook
│
├── src/
│   ├── __init__.py           # Source package initializer
│   ├── features.py           # Preprocessing, age capping & feature engineering
│   ├── retrieval.py          # TruncatedSVD candidate retrieval module
│   ├── ranking.py            # LightGBM lambdarank model module
│   └── evaluation.py         # Precision@K, Recall@K, NDCG@K, HitRate@K metrics
│
├── models/                   # Serialized model checkpoints (.pkl)
│   ├── retrieval_svd.pkl
│   ├── ranking_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── artifacts/                # Combined deployment bundle (.joblib)
│   └── model_artifacts.joblib
│
├── app.py                    # Interactive Streamlit deployment web application
├── requirements.txt          # Production dependencies
└── README.md                 # Project documentation
```

---

## 📊 Offline Benchmark Evaluation Results

| Model Engine | K | Precision@K | Recall@K | NDCG@K | HitRate@K |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A (Retrieval Only)** | 5 | 0.0036 | 0.0031 | 0.0061 | 0.0180 |
| **Model B (Retrieval + Ranking)** | 5 | **0.0060** | **0.0060** | **0.0061** | **0.0300** |
| **Model A (Retrieval Only)** | 10 | 0.0028 | 0.0060 | 0.0063 | 0.0280 |
| **Model B (Retrieval + Ranking)** | 10 | **0.0048** | **0.0084** | **0.0066** | **0.0440** |
| **Model A (Retrieval Only)** | 20 | 0.0019 | 0.0081 | 0.0065 | 0.0340 |
| **Model B (Retrieval + Ranking)** | 20 | **0.0030** | **0.0104** | **0.0069** | **0.0520** |

> **Key takeaway**: Incorporating the Stage 2 LightGBM Ranker improves **Precision@10 by +71.4%** and **Hit Rate@10 by +57.1%** compared to Stage 1 SVD Retrieval alone.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing & Feature Engineering
```bash
python 01_preprocess_and_feature_engineering.py
```

### 3. Run Model Building & Offline Evaluation
```bash
python 02_model_building_and_evaluation.py
```

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🛡 Leakage Prevention & Audit Guarantees
* **Train-Only Index & Feature Space**: User averages, book popularity, and TF-IDF matrices are built strictly from `train_df`.
* **Validation Grouping**: Model hyperparameters and early stopping use `valid_df` candidates, leaving `test_df` completely untouched until final evaluation.
* **Seen Item Masking**: Items already rated in the user's historical train interaction set are filtered out prior to candidate evaluation.
