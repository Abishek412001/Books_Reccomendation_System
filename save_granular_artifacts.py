import os
import joblib

print("Packaging granular artifacts...")
artifact_file = "model_artifacts.joblib"
if os.path.exists(artifact_file):
    artifacts = joblib.load(artifact_file)
    
    # Save models
    os.makedirs("models/retrieval_model", exist_ok=True)
    os.makedirs("models/ranking_model", exist_ok=True)
    joblib.dump(artifacts["svd_model"], "models/retrieval_model/retrieval_svd.pkl")
    joblib.dump(artifacts["ranker"], "models/ranking_model/ranking_model.pkl")
    
    # Save artifacts
    os.makedirs("artifacts/tfidf", exist_ok=True)
    os.makedirs("artifacts/encoders", exist_ok=True)
    os.makedirs("artifacts/metadata", exist_ok=True)
    
    joblib.dump(artifacts["tfidf_vectorizer"], "artifacts/tfidf/tfidf_vectorizer.pkl")
    joblib.dump({"user_map": artifacts["user_map"], "item_map": artifacts["item_map"]}, "artifacts/encoders/label_encoders.joblib")
    
    metadata = {
        "user_ids": artifacts["user_ids"],
        "item_ids": artifacts["item_ids"],
        "pop_order": artifacts["pop_order"],
        "seen_eval": artifacts["seen_eval"],
        "has_signal": artifacts["has_signal"],
        "user_arr": artifacts["user_arr"],
        "item_arr": artifacts["item_arr"],
        "features": artifacts["features"],
        "books_df": artifacts["books_df"],
        "user_feat_df": artifacts["user_feat_df"],
        "users_cleaned": artifacts["users_cleaned"]
    }
    joblib.dump(metadata, "artifacts/metadata/metadata.joblib", compress=3)
    print("Granular artifacts packaged successfully!")
