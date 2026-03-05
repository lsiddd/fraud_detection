"""
ensemble.py — IsolationForestDetector, StackingDetector
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import IsolationForest


# ══════════════════════════════════════════════════════════════════════════════
# 2. IsolationForestDetector
# ══════════════════════════════════════════════════════════════════════════════

class IsolationForestDetector:
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

class StackingDetector:
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
            ("LR", LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=500,
                random_state=self.seed, n_jobs=-1,
            )),
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
