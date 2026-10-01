import lightgbm as lgb
import numpy as np
import pandas as pd

class LightGBMRanker:
    def __init__(self, seed=42):
        self.seed = seed
        self.model = None
        self.best_iteration = None

    def fit(self, dtrain, dvalid, num_boost_round=300, early_stopping_rounds=30):
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
            "seed": self.seed,
            "verbose": -1,
        }
        self.model = lgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        self.best_iteration = self.model.best_iteration or None
        return self

    def predict(self, feature_matrix):
        return self.model.predict(feature_matrix, num_iteration=self.best_iteration)
