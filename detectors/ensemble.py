"""
ensemble.py — IsolationForestDetector, StackingDetector, SuperEnsembleDetector
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import IsolationForest
from detectors.base import BaseDetector


# ══════════════════════════════════════════════════════════════════════════════
# 2. IsolationForestDetector
# ══════════════════════════════════════════════════════════════════════════════

class IsolationForestDetector(BaseDetector):
    name = "Isolation Forest"

    def __init__(self, n_estimators=100, seed=42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.train_time = 0.0
        self._model = None

    def fit(self, X_tr, y_tr=None):
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
        )
        t0 = time.time()
        self._model.fit(X_tr)
        self.train_time = time.time() - t0
        return self

    def score(self, X_te):
        # Maior = mais anômalo
        return -self._model.decision_function(X_te)


# ══════════════════════════════════════════════════════════════════════════════
# 10. StackingDetector  (XGBoost + LightGBM + CatBoost + LR  →  meta-LR)
# ══════════════════════════════════════════════════════════════════════════════

class StackingDetector(BaseDetector):
    name = "Stacking"

    def __init__(self, n_folds=5, seed=42):
        self.n_folds = n_folds
        self.seed = seed
        self.train_time = 0.0
        self._meta = None
        self._fold_models_per_base = None
        self._base_names = None

    def _make_base_learners(self, spw):
        from xgboost import XGBClassifier
        import lightgbm as lgb
        from catboost import CatBoostClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        return [
            ("XGBoost", XGBClassifier(
                n_estimators=300, scale_pos_weight=spw, tree_method="hist",
                eval_metric="aucpr", early_stopping_rounds=20,
                random_state=self.seed, verbosity=0,
            )),
            ("LightGBM", lgb.LGBMClassifier(
                boosting_type="gbdt", scale_pos_weight=spw,
                n_estimators=500, learning_rate=0.05, num_leaves=63,
                random_state=self.seed, n_jobs=-1, verbose=-1,
            )),
            ("CatBoost", CatBoostClassifier(
                iterations=500, depth=6, auto_class_weights="Balanced",
                eval_metric="AUC", early_stopping_rounds=30,
                random_seed=self.seed, verbose=0,
            )),
            ("LR", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("lr", LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=500,
                    random_state=self.seed, n_jobs=-1,
                )),
            ])),
        ]

    @staticmethod
    def _fit_base(name, model, X_tr, y_tr, X_val, y_val):
        """Treina um base learner com early stopping quando suportado."""
        if name == "XGBoost":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif name == "LightGBM":
            import lightgbm as lgb
            model.fit(
                X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )
        elif name == "CatBoost":
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        else:
            model.fit(X_tr, y_tr)

    def fit(self, X_tr, y_tr):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        spw = n_neg / max(n_pos, 1)

        base_learners = self._make_base_learners(spw)
        base_names    = [n for n, _ in base_learners]
        n_base        = len(base_names)

        oof_preds           = np.zeros((len(y_tr), n_base), dtype=np.float32)
        fold_models_per_base = [[] for _ in range(n_base)]

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.seed)

        print(f"    Base learners: {', '.join(base_names)}")
        print(f"    {self.n_folds} folds | pos_weight={spw:.1f}")
        sep = "    " + "─" * 68

        t0 = time.time()
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            X_f_tr, X_f_val = X_tr[tr_idx], X_tr[val_idx]
            y_f_tr, y_f_val = y_tr[tr_idx], y_tr[val_idx]

            print(sep)
            print(f"    Fold {fold_idx + 1}/{self.n_folds}  "
                  f"(treino={len(y_f_tr):,}  val={len(y_f_val):,})")

            aucs = []
            for b_idx, (b_name, b_model) in enumerate(base_learners):
                self._fit_base(b_name, b_model, X_f_tr, y_f_tr, X_f_val, y_f_val)
                oof_preds[val_idx, b_idx] = b_model.predict_proba(X_f_val)[:, 1]
                auc = roc_auc_score(y_f_val, oof_preds[val_idx, b_idx])
                aucs.append(f"{b_name}={auc:.4f}")
                fold_models_per_base[b_idx].append(b_model)

            print(f"      AUC: {' | '.join(aucs)}")

        print(sep)
        oof_aucs = [
            f"{b_name}={roc_auc_score(y_tr, oof_preds[:, i]):.4f}"
            for i, b_name in enumerate(base_names)
        ]
        print(f"    OOF médio: {' | '.join(oof_aucs)}")

        print("    Treinando meta-learner (Logistic Regression)…")
        self._meta = LogisticRegression(C=1.0, max_iter=1000,
                                        random_state=self.seed)
        self._meta.fit(oof_preds, y_tr)
        coef_str = " | ".join(
            f"{n}={c:.3f}" for n, c in zip(base_names, self._meta.coef_[0])
        )
        print(f"    Coeficientes: {coef_str}")

        self._fold_models_per_base = fold_models_per_base
        self._base_names = base_names
        self.train_time = time.time() - t0
        return self

    def score(self, X_te):
        n_base = len(self._base_names)
        test_preds = np.zeros((len(X_te), n_base), dtype=np.float32)
        for b_idx, fold_models in enumerate(self._fold_models_per_base):
            test_preds[:, b_idx] = np.mean(
                [m.predict_proba(X_te)[:, 1] for m in fold_models], axis=0
            )
        return self._meta.predict_proba(test_preds)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# 11. SuperEnsembleDetector  (XGBoost + LightGBM + CatBoost  →  meta-XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

class SuperEnsembleDetector(BaseDetector):
    """
    Super Ensemble: XGBoost + LightGBM + CatBoost com hiperparâmetros agressivos,
    5-fold OOF stacking e XGBoost como meta-learner otimizado por PR-AUC.

    Espera features enriquecidas de prepare_ensemble_cc / prepare_ensemble_ieee.
    """
    name = "Super Ensemble"

    def __init__(self, n_folds=5, seed=42):
        self.n_folds = n_folds
        self.seed = seed
        self.train_time = 0.0
        self._meta = None
        self._final_models = None
        self._base_names = None
        self._n_base = None

    def _make_base_learners(self, spw, n_threads=-1):
        from xgboost import XGBClassifier
        import lightgbm as lgb
        from catboost import CatBoostClassifier

        return [
            ("XGBoost", XGBClassifier(
                n_estimators=1000, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.7,
                gamma=0.1, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
                tree_method="hist", scale_pos_weight=spw,
                early_stopping_rounds=30, eval_metric="aucpr",
                random_state=self.seed, verbosity=0,
                nthread=n_threads,
            )),
            ("LightGBM", lgb.LGBMClassifier(
                boosting_type="gbdt", n_estimators=2000, num_leaves=63,
                learning_rate=0.03, min_child_samples=100,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=spw,
                random_state=self.seed, n_jobs=n_threads, verbose=-1,
            )),
            ("CatBoost", CatBoostClassifier(
                iterations=2000, depth=8, learning_rate=0.03,
                l2_leaf_reg=3.0, auto_class_weights="Balanced",
                eval_metric="AUC", early_stopping_rounds=50,
                random_seed=self.seed, verbose=0,
                thread_count=n_threads,
            )),
        ]

    @staticmethod
    def _fit_base(name, model, X_tr, y_tr, X_val, y_val):
        if name == "XGBoost":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif name == "LightGBM":
            import lightgbm as lgb
            model.fit(
                X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(-1)],
            )
        elif name == "CatBoost":
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    @staticmethod
    def _build_meta_features(preds):
        """7 meta-features: 3 raw probs + 3 pairwise products + 1 spread."""
        xgb_p, lgb_p, cat_p = preds[:, 0], preds[:, 1], preds[:, 2]
        spread = preds.max(axis=1) - preds.min(axis=1)
        return np.column_stack([
            xgb_p, lgb_p, cat_p,
            xgb_p * lgb_p, xgb_p * cat_p, lgb_p * cat_p,
            spread,
        ])

    def fit(self, X_tr, y_tr):
        import os
        from concurrent.futures import ThreadPoolExecutor
        from sklearn.model_selection import StratifiedKFold, train_test_split
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        spw = n_neg / max(n_pos, 1)

        # Divide CPU cores among base learners so they can train in parallel
        n_cpu = os.cpu_count() or 4
        n_base_count = 3  # XGBoost, LightGBM, CatBoost
        n_threads = max(1, n_cpu // n_base_count)

        base_learners = self._make_base_learners(spw, n_threads=n_threads)
        base_names    = [n for n, _ in base_learners]
        n_base        = len(base_names)

        oof_preds            = np.zeros((len(y_tr), n_base), dtype=np.float32)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.seed)

        print(f"    Base learners: {', '.join(base_names)}")
        print(f"    {self.n_folds} folds | pos_weight={spw:.1f} | features={X_tr.shape[1]}")
        print(f"    Paralelismo: {n_base_count} modelos × {n_threads} threads cada")
        sep = "    " + "─" * 68

        def _train_one(b_idx, b_name, b_model, X_f_tr, y_f_tr, X_f_val, y_f_val):
            self._fit_base(b_name, b_model, X_f_tr, y_f_tr, X_f_val, y_f_val)
            preds = b_model.predict_proba(X_f_val)[:, 1]
            return b_idx, b_name, b_model, preds

        t0 = time.time()
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            X_f_tr, X_f_val = X_tr[tr_idx], X_tr[val_idx]
            y_f_tr, y_f_val = y_tr[tr_idx], y_tr[val_idx]

            print(sep)
            print(f"    Fold {fold_idx + 1}/{self.n_folds}  "
                  f"(treino={len(y_f_tr):,}  val={len(y_f_val):,})")

            with ThreadPoolExecutor(max_workers=n_base) as pool:
                futures = [
                    pool.submit(_train_one, b_idx, b_name, b_model,
                                X_f_tr, y_f_tr, X_f_val, y_f_val)
                    for b_idx, (b_name, b_model) in enumerate(base_learners)
                ]
                results = [f.result() for f in futures]

            aucs = []
            for b_idx, b_name, b_model, preds in sorted(results):
                oof_preds[val_idx, b_idx] = preds
                auc = roc_auc_score(y_f_val, preds)
                aucs.append(f"{b_name}={auc:.4f}")

            print(f"      AUC: {' | '.join(aucs)}")

        print(sep)
        oof_aucs = [
            f"{b_name}={roc_auc_score(y_tr, oof_preds[:, i]):.4f}"
            for i, b_name in enumerate(base_names)
        ]
        print(f"    OOF médio: {' | '.join(oof_aucs)}")

        # Meta-learner: XGBoost on 7 meta-features, optimized for PR-AUC
        meta_features = self._build_meta_features(oof_preds)
        X_m_tr, X_m_val, y_m_tr, y_m_val = train_test_split(
            meta_features, y_tr, test_size=0.2, stratify=y_tr,
            random_state=self.seed,
        )
        print("    Treinando meta-learner (XGBoost, eval_metric=aucpr)…")
        self._meta = XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=1.0,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=self.seed, verbosity=0,
        )
        self._meta.fit(X_m_tr, y_m_tr, eval_set=[(X_m_val, y_m_val)], verbose=False)
        print(f"    Meta-learner: best_iteration={self._meta.best_iteration}")

        # Train one final model per base learner on 100% of training data
        # (10% stratified holdout used only for early stopping)
        print("    Treinando modelos finais em 100% dos dados de treino…")
        X_full_tr, X_full_val, y_full_tr, y_full_val = train_test_split(
            X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=self.seed,
        )
        final_learners = self._make_base_learners(spw, n_threads=n_threads)

        def _train_final(b_idx, b_name, b_model):
            self._fit_base(b_name, b_model, X_full_tr, y_full_tr,
                           X_full_val, y_full_val)
            return b_idx, b_name, b_model

        with ThreadPoolExecutor(max_workers=n_base) as pool:
            futures = [
                pool.submit(_train_final, b_idx, b_name, b_model)
                for b_idx, (b_name, b_model) in enumerate(final_learners)
            ]
            final_results = sorted([f.result() for f in futures])

        self._final_models = [b_model for _, _, b_model in final_results]
        self._base_names = base_names
        self._n_base = n_base
        self.train_time = time.time() - t0
        return self

    def score(self, X_te):
        n_base = self._n_base
        test_preds = np.zeros((len(X_te), n_base), dtype=np.float32)
        for b_idx, model in enumerate(self._final_models):
            test_preds[:, b_idx] = model.predict_proba(X_te)[:, 1]
        meta_test = self._build_meta_features(test_preds)
        return self._meta.predict_proba(meta_test)[:, 1]
