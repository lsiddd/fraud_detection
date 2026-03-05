"""
tree_based.py — XGBoostDetector, LightGBMDetector, CatBoostDetector
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from detectors.base import BaseDetector


def _fit_xgb(model_kwargs, X_t, y_t, X_val, y_val):
    """Builds and fits an XGBClassifier; falls back to logloss if aucpr unsupported."""
    from xgboost import XGBClassifier
    try:
        clf = XGBClassifier(**model_kwargs, eval_metric="aucpr")
    except (ValueError, TypeError):
        clf = XGBClassifier(**model_kwargs, eval_metric="logloss")
    clf.fit(X_t, y_t, eval_set=[(X_val, y_val)], verbose=False)
    return clf


# ══════════════════════════════════════════════════════════════════════════════
# 1. XGBoostDetector
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostDetector(BaseDetector):
    name = "XGBoost"

    def __init__(self, n_estimators=300, seed=42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.train_time = 0.0
        self._model = None

    def fit(self, X_tr, y_tr):
        from sklearn.model_selection import train_test_split

        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        spw = n_neg / max(n_pos, 1)

        # 15% interno para early stopping
        X_t, X_val, y_t, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=self.seed
        )

        model_kwargs = dict(
            n_estimators=self.n_estimators,
            scale_pos_weight=spw,
            tree_method="hist",
            early_stopping_rounds=20,
            random_state=self.seed,
            verbosity=0,
        )

        t0 = time.time()
        self._model = _fit_xgb(model_kwargs, X_t, y_t, X_val, y_val)
        self.train_time = time.time() - t0
        return self

    def score(self, X_te):
        return self._model.predict_proba(X_te)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# 7. LightGBMDetector  (DART booster)
# ══════════════════════════════════════════════════════════════════════════════

class LightGBMDetector(BaseDetector):
    name = "LightGBM"

    def __init__(self, n_estimators=1000, seed=42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.train_time = 0.0
        self._model = None

    def fit(self, X_tr, y_tr):
        import lightgbm as lgb

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        spw = n_neg / max(n_pos, 1)

        # DART não suporta early stopping — n_estimators fixo (1000 para dar
        # margem suficiente de convergência, já que DART é mais lento que GBDT)
        print(f"    DART booster | n_estimators={self.n_estimators} | "
              f"scale_pos_weight={spw:.1f} | num_leaves=63")

        self._model = lgb.LGBMClassifier(
            boosting_type="dart",
            drop_rate=0.1,
            scale_pos_weight=spw,
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )

        t0 = time.time()
        self._model.fit(X_tr, y_tr)
        self.train_time = time.time() - t0
        print(f"    Concluído em {self.train_time:.1f}s")
        return self

    def score(self, X_te):
        return self._model.predict_proba(X_te)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# 8. CatBoostDetector
# ══════════════════════════════════════════════════════════════════════════════

class CatBoostDetector(BaseDetector):
    name = "CatBoost"

    def __init__(self, iterations=1000, depth=8, cat_features=None, seed=42):
        self.iterations = iterations
        self.depth = depth
        self.cat_features = cat_features or []
        self.seed = seed
        self.train_time = 0.0
        self._model = None

    def _to_int_cats(self, X):
        if not self.cat_features:
            return X
        # CatBoost rejeita cat_features em arrays float64 — converte para object
        # para que cada coluna possa ter dtype independente (int vs float).
        X = X.astype(object)
        for idx in self.cat_features:
            X[:, idx] = X[:, idx].astype(float).astype(int)
        return X

    def fit(self, X_tr, y_tr):
        from catboost import CatBoostClassifier
        from sklearn.model_selection import train_test_split

        X_tr = self._to_int_cats(X_tr)
        X_t, X_val, y_t, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=self.seed
        )

        print(f"    iterations={self.iterations} | depth={self.depth} | "
              f"cat_features={len(self.cat_features)} | auto_class_weights=Balanced")

        self._model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            eval_metric="AUC",
            # early_stopping_rounds=50,
            cat_features=self.cat_features,
            random_seed=self.seed,
            verbose=200,
        )

        t0 = time.time()
        self._model.fit(X_t, y_t, eval_set=(X_val, y_val))
        self.train_time = time.time() - t0
        best = self._model.get_best_iteration()
        print(f"    Melhor iteração: {best} | t={self.train_time:.1f}s")
        return self

    def score(self, X_te):
        return self._model.predict_proba(self._to_int_cats(X_te))[:, 1]
