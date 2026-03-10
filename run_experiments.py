"""
run_experiments.py — orquestração do framework de detecção de fraudes.

Treina e avalia 4 algoritmos (XGBoost, Isolation Forest, Autoencoder, GNN)
nos datasets CC e IEEE, depois gera 5 figuras comparativas.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

from preprocessing import (
    load_datasets,
    make_chronological_split,
    prepare_numeric,
    prepare_xgboost,
    prepare_xgboost_all,
    prepare_gnn,
    prepare_gat_cc,
    prepare_gat_ieee,
    prepare_ensemble_cc,
    prepare_ensemble_ieee,
    CC_NUM_FEATS,
    IEEE_NUM_FEATS,
    IEEE_CAT_FEATS,
    DatasetConfig,
    CC_CONFIG,
    IEEE_CONFIG,
)
from detectors import (
    XGBoostDetector,
    IsolationForestDetector,
    AutoencoderDetector,
    GATDetector,
    GATXGBDetector,
    GNNDetector,
    GraphSAGEXGBDetector,
    LightGBMDetector,
    CatBoostDetector,
    AutoGluonDetector,
    TabNetDetector,
    StackingDetector,
    SuperEnsembleDetector,
    build_cc_graph,
    build_cc_graph_cosine,
    build_ieee_graph,
    build_ieee_graph_edges,
    normalize_adj,
)
import reporting


# ── Config ─────────────────────────────────────────────────────────────────────
# Sem subsampling — datasets completos para avaliação justa
SEED = 42


def compute_metrics(y_test, scores):
    """Returns dict with roc_auc, pr_auc, gini, ks, f1, precision, recall, mcc, fpr_at_90rec."""
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc  = average_precision_score(y_test, scores)
    gini    = 2.0 * roc_auc - 1.0

    # KS Statistic: max separation between fraud and legit score CDFs
    ks_stat, _ = ks_2samp(scores[y_test == 1], scores[y_test == 0])

    # Best F1 threshold
    precs, recs, threshs = precision_recall_curve(y_test, scores)
    # f1s has same length as threshs (precs/recs have one extra element)
    f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = threshs[best_idx]

    y_pred = (scores >= best_thresh).astype(int)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_test, y_pred)

    # FPR at 90% Recall: how many legit transactions are flagged to catch 90% of fraud
    fpr_arr, tpr_arr, _ = roc_curve(y_test, scores)
    fpr_at_90rec = float(np.interp(0.90, tpr_arr, fpr_arr))

    return dict(
        roc_auc=roc_auc, pr_auc=pr_auc, gini=gini, ks=ks_stat,
        f1=f1, precision=prec, recall=rec, mcc=mcc,
        fpr_at_90rec=fpr_at_90rec,
    )


def _build_masks(N, idx_tr, idx_te, idx_val=None):
    """Build boolean index masks from positional index arrays."""
    train_mask = np.zeros(N, dtype=bool)
    test_mask  = np.zeros(N, dtype=bool)
    train_mask[idx_tr] = True
    test_mask[idx_te]  = True
    if idx_val is not None:
        val_mask = np.zeros(N, dtype=bool)
        val_mask[idx_val] = True
        return train_mask, val_mask, test_mask
    return train_mask, test_mask


def run_dataset(cfg: DatasetConfig, df, graph_builder, algos=None, seed=SEED,
                ag_time_limit=600, ag_presets="best_quality"):
    """
    Treina e avalia os 4 detectores em um dataset completo.
    Retorna lista de result dicts.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {cfg.name}  ({len(df):,} linhas — dataset completo)")
    print(f"{'='*60}")

    df_s = df.reset_index(drop=True)
    print(f"  Total: {len(df_s):,} linhas  "
          f"(fraude={df_s[cfg.label_col].mean()*100:.2f}%)")

    # ── Canonical chronological 70/15/15 split (same for all models) ───────────
    idx_tr, idx_val, idx_te = make_chronological_split(df_s, cfg.time_col)

    # ── 2. Preparação numérica (para IF, AE, GNN) ──────────────────────────────
    X_all, y_all, X_tr, X_te, y_tr, y_te, idx_tr, idx_te = prepare_numeric(
        df_s, cfg.num_feats, cfg.label_col, idx_tr=idx_tr, idx_te=idx_te, seed=seed
    )
    print(f"  Features numéricas: {X_all.shape[1]}  "
          f"| treino={len(y_tr):,}  teste={len(y_te):,}")

    # ── 3. Preparação para XGBoost ─────────────────────────────────────────────
    Xg_tr, Xg_te, yg_tr, yg_te = prepare_xgboost(
        df_s, cfg.num_feats, cfg.cat_feats, cfg.label_col, idx_tr=idx_tr, idx_te=idx_te, seed=seed
    )

    # ── 3b. Preparação enriquecida para Super Ensemble (lazy) ─────────────────
    super_data = None
    if "Super Ensemble" in algos:
        if cfg.name == "CC":
            Xs_tr, Xs_te, ys_tr, ys_te = prepare_ensemble_cc(df_s, idx_tr, idx_te)
        else:
            Xs_tr, Xs_te, ys_tr, ys_te = prepare_ensemble_ieee(df_s, idx_tr, idx_te)
        super_data = (Xs_tr, Xs_te, ys_tr, ys_te)
        print(f"  Features Super Ensemble: {Xs_tr.shape[1]}")

    # Índices e cardinalidades das features categóricas (CatBoost e TabNet)
    _avail_num = [c for c in cfg.num_feats if c in df_s.columns]
    _avail_cat = [c for c in cfg.cat_feats if c in df_s.columns]
    cat_idxs = list(range(len(_avail_num), len(_avail_num) + len(_avail_cat)))
    cat_dims  = [int(df_s[c].nunique()) + 1 for c in _avail_cat]

    if algos is None:
        algos = [
            "XGBoost", "Isolation Forest", "Autoencoder",
            "GNN (GATv2)", "GAT+XGB", "GNN (GCN)",
            "GraphSAGE+XGB", "LightGBM", "CatBoost", "AutoGluon", "TabNet", "Stacking",
        ]

    results = []

    # Graph adjacency (lazy — only if needed)
    adj = None
    if any(a in algos for a in ["GNN (GCN)", "GraphSAGE+XGB"]):
        adj = graph_builder(X_all, df_s)

    # GAT data (lazy — only if needed)
    gat_data = None
    if any(a in algos for a in ["GNN (GATv2)", "GAT+XGB"]):
        if cfg.name == "CC":
            X_gat, X_pca, times, y_gat, i_tr_gat, i_val_gat, i_te_gat = prepare_gat_cc(df_s)
            edge_index = build_cc_graph_cosine(X_pca, times=times, k=5, cos_dist_threshold=0.10)
        else:
            X_gat, times, y_gat, i_tr_gat, i_val_gat, i_te_gat = prepare_gat_ieee(df_s)
            edge_index = build_ieee_graph_edges(df_s, max_per_card=100)
        gat_data = (X_gat, y_gat, i_tr_gat, i_val_gat, i_te_gat, edge_index)

    # ── 4. XGBoost ─────────────────────────────────────────────────────────────
    if "XGBoost" in algos:
        print("\n  [XGBoost]")
        det = XGBoostDetector(seed=seed)
        det.fit(Xg_tr, yg_tr)
        scores = det.score(Xg_te)
        metrics = compute_metrics(yg_te, scores)
        results.append(dict(
            name="XGBoost", dataset=cfg.name,
            y_test=yg_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 5. Isolation Forest ────────────────────────────────────────────────────
    if "Isolation Forest" in algos:
        print("  [Isolation Forest]")
        det = IsolationForestDetector(seed=seed)
        det.fit(X_tr, y_tr)
        scores = det.score(X_te)
        metrics = compute_metrics(y_te, scores)
        results.append(dict(
            name="Isolation Forest", dataset=cfg.name,
            y_test=y_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 6. Autoencoder ─────────────────────────────────────────────────────────
    if "Autoencoder" in algos:
        print("  [Autoencoder]")
        det = AutoencoderDetector(seed=seed)
        det.fit(X_tr, y_tr)
        scores = det.score(X_te)
        metrics = compute_metrics(y_te, scores)
        results.append(dict(
            name="Autoencoder", dataset=cfg.name,
            y_test=y_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 7. GATv2 (SOTA — indutivo, split cronológico, grafo causal) ───────────
    if "GNN (GATv2)" in algos:
        print("\n  [GNN (GATv2)]")
        X_gat, y_gat, i_tr_gat, i_val_gat, i_te_gat, edge_index = gat_data

        N_gat = len(y_gat)
        train_mask_gat, val_mask_gat, test_mask_gat = _build_masks(
            N_gat, i_tr_gat, i_te_gat, i_val_gat
        )

        det = GATDetector(seed=seed)
        det.fit(X_gat, y_gat, train_mask_gat, test_mask_gat, edge_index,
                val_mask=val_mask_gat)

        scores      = det.score()
        y_te_gat    = y_gat[i_te_gat]
        metrics     = compute_metrics(y_te_gat, scores)
        results.append(dict(
            name="GNN (GATv2)", dataset=cfg.name,
            y_test=y_te_gat, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 7.5. GAT+XGB ───────────────────────────────────────────────────────────
    if "GAT+XGB" in algos:
        print("\n  [GAT+XGB]")
        X_gat, y_gat, i_tr_gat, i_val_gat, i_te_gat, edge_index = gat_data

        N_gat = len(y_gat)
        train_mask_gat, val_mask_gat, test_mask_gat = _build_masks(
            N_gat, i_tr_gat, i_te_gat, i_val_gat
        )

        # Features tabulares alinhadas ao split cronológico do GAT
        Xg_all_xgb, _ = prepare_xgboost_all(df_s, cfg.num_feats, cfg.cat_feats, cfg.label_col)
        Xg_gat_tr = Xg_all_xgb[i_tr_gat]
        yg_gat_tr  = y_gat[i_tr_gat]
        Xg_gat_te  = Xg_all_xgb[i_te_gat]

        det = GATXGBDetector(seed=seed)
        det.fit(X_gat, y_gat, train_mask_gat, test_mask_gat, edge_index,
                Xg_gat_tr, yg_gat_tr, val_mask=val_mask_gat)

        scores   = det.score(Xg_gat_te)
        y_te_gat = y_gat[i_te_gat]
        metrics  = compute_metrics(y_te_gat, scores)
        results.append(dict(
            name="GAT+XGB", dataset=cfg.name,
            y_test=y_te_gat, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 8. GNN (GCN) ───────────────────────────────────────────────────────────
    if "GNN (GCN)" in algos:
        print("  [GNN (GCN)]")
        adj_norm = normalize_adj(adj)

        # Features completas (numéricas + categóricas) — mesma vantagem do XGBoost
        X_gnn = prepare_gnn(df_s, cfg.num_feats, cfg.cat_feats, cfg.label_col)

        train_mask, test_mask = _build_masks(len(y_all), idx_tr, idx_te)

        det = GNNDetector(seed=seed)
        det.fit(X_gnn, y_all, train_mask, test_mask, adj_norm)
        scores = det.score()
        metrics = compute_metrics(y_te, scores)
        results.append(dict(
            name="GNN (GCN)", dataset=cfg.name,
            y_test=y_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 8. GraphSAGE+XGB ───────────────────────────────────────────────────────
    if "GraphSAGE+XGB" in algos:
        print("\n  [GraphSAGE+XGB]")
        X_gnn = prepare_gnn(df_s, cfg.num_feats, cfg.cat_feats, cfg.label_col)

        train_mask, test_mask = _build_masks(len(y_all), idx_tr, idx_te)

        det = GraphSAGEXGBDetector(seed=seed)
        det.fit(X_gnn, y_all, train_mask, test_mask, adj, Xg_tr, yg_tr)
        scores = det.score(Xg_te)
        metrics = compute_metrics(yg_te, scores)
        results.append(dict(
            name="GraphSAGE+XGB", dataset=cfg.name,
            y_test=yg_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 9. LightGBM ────────────────────────────────────────────────────────────
    if "LightGBM" in algos:
        print("  [LightGBM]")
        det = LightGBMDetector(seed=seed)
        det.fit(Xg_tr, yg_tr)
        scores = det.score(Xg_te)
        metrics = compute_metrics(yg_te, scores)
        results.append(dict(
            name="LightGBM", dataset=cfg.name,
            y_test=yg_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 10. CatBoost ───────────────────────────────────────────────────────────
    if "CatBoost" in algos:
        print("  [CatBoost]")
        det = CatBoostDetector(cat_features=cat_idxs, seed=seed)
        det.fit(Xg_tr, yg_tr)
        scores = det.score(Xg_te)
        metrics = compute_metrics(yg_te, scores)
        results.append(dict(
            name="CatBoost", dataset=cfg.name,
            y_test=yg_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 11. AutoGluon ──────────────────────────────────────────────────────────
    if "AutoGluon" in algos:
        print("  [AutoGluon]")
        det = AutoGluonDetector(time_limit=ag_time_limit, presets=ag_presets, seed=seed)
        det.fit(Xg_tr, yg_tr)
        scores = det.score(Xg_te)
        metrics = compute_metrics(yg_te, scores)
        results.append(dict(
            name="AutoGluon", dataset=cfg.name,
            y_test=yg_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 12. TabNet ─────────────────────────────────────────────────────────────
    if "TabNet" in algos:
        print("  [TabNet]")
        det = TabNetDetector(seed=seed)
        det.fit(Xg_tr, yg_tr, cat_idxs=cat_idxs, cat_dims=cat_dims)
        scores = det.score(Xg_te)
        metrics = compute_metrics(yg_te, scores)
        results.append(dict(
            name="TabNet", dataset=cfg.name,
            y_test=yg_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 13. Stacking ───────────────────────────────────────────────────────────
    if "Stacking" in algos:
        print("  [Stacking]")
        det = StackingDetector(seed=seed)
        det.fit(Xg_tr, yg_tr)
        scores = det.score(Xg_te)
        metrics = compute_metrics(yg_te, scores)
        results.append(dict(
            name="Stacking", dataset=cfg.name,
            y_test=yg_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    # ── 14. Super Ensemble ─────────────────────────────────────────────────────
    if "Super Ensemble" in algos:
        print("\n  [Super Ensemble]")
        Xs_tr, Xs_te, ys_tr, ys_te = super_data
        det = SuperEnsembleDetector(seed=seed)
        det.fit(Xs_tr, ys_tr)
        scores = det.score(Xs_te)
        metrics = compute_metrics(ys_te, scores)
        results.append(dict(
            name="Super Ensemble", dataset=cfg.name,
            y_test=ys_te, scores=scores,
            train_time=det.train_time,
            **metrics,
        ))
        print(f"    ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
              f"  F1={metrics['f1']:.4f}  t={det.train_time:.1f}s")

    return results


def print_summary(all_results):
    """Imprime tabela resumo no terminal."""
    header = (
        f"{'Algoritmo':<20} {'Dataset':<8}"
        f" {'ROC-AUC':>8} {'PR-AUC':>8} {'Gini':>7} {'KS':>7}"
        f" {'MCC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7}"
        f" {'FPR@90R':>8} {'Tempo':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in all_results:
        print(
            f"{r['name']:<20} {r['dataset']:<8}"
            f" {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f}"
            f" {r.get('gini', 0):>7.4f} {r.get('ks', 0):>7.4f}"
            f" {r.get('mcc', 0):>7.4f} {r['f1']:>7.4f}"
            f" {r['precision']:>7.4f} {r['recall']:>7.4f}"
            f" {r.get('fpr_at_90rec', 0):>8.4f} {r['train_time']:>7.1f}s"
        )
    print(sep)


TREE_ALGOS = ["XGBoost", "LightGBM", "CatBoost", "Isolation Forest", "Stacking", "Super Ensemble"]
DL_GRAPH_ALGOS = ["Autoencoder", "TabNet", "GNN (GATv2)", "GAT+XGB", "GNN (GCN)", "GraphSAGE+XGB"]

ALGO_ALIASES = {
    "xgb":       "XGBoost",
    "xgboost":   "XGBoost",
    "if":        "Isolation Forest",
    "iso":       "Isolation Forest",
    "ae":        "Autoencoder",
    "auto":      "Autoencoder",
    "gat":       "GNN (GATv2)",
    "gatv2":     "GNN (GATv2)",
    "gatxgb":    "GAT+XGB",
    "gat+xgb":   "GAT+XGB",
    "gatxgboost":"GAT+XGB",
    "gnn":       "GNN (GCN)",
    "gcn":       "GNN (GCN)",
    "sage":      "GraphSAGE+XGB",
    "sagexgb":   "GraphSAGE+XGB",
    "graphsage": "GraphSAGE+XGB",
    "lgb":       "LightGBM",
    "lgbm":      "LightGBM",
    "lightgbm":  "LightGBM",
    "cat":       "CatBoost",
    "catboost":  "CatBoost",
    "ag":        "AutoGluon",
    "autogluon": "AutoGluon",
    "tabnet":    "TabNet",
    "tab":       "TabNet",
    "stack":          "Stacking",
    "stacking":       "Stacking",
    "se":             "Super Ensemble",
    "super":          "Super Ensemble",
    "superensemble":  "Super Ensemble",
}

DATASET_ALIASES = {
    "cc":   "CC",
    "ieee": "IEEE",
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detecção de fraudes — roda algoritmos selecionados.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-a", "--algo",
        nargs="+",
        metavar="ALGO",
        help=(
            "Algoritmo(s) a executar (padrão: todos).\n"
            "Opções: xgb/xgboost | if/iso | ae/auto\n"
            "        gat/gatv2 | gatxgb/gat+xgb | gnn/gcn | sage/sagexgb/graphsage\n"
            "        lgb/lgbm/lightgbm | cat/catboost | ag/autogluon\n"
            "        tab/tabnet | stack/stacking"
        ),
    )
    group.add_argument(
        "--trees",
        action="store_true",
        help=f"Roda só métodos de árvore: {', '.join(TREE_ALGOS)}.",
    )
    group.add_argument(
        "--dl",
        action="store_true",
        help=f"Roda só métodos de DL e grafos: {', '.join(DL_GRAPH_ALGOS)}.",
    )
    parser.add_argument(
        "-d", "--dataset",
        nargs="+",
        metavar="DATASET",
        help=(
            "Dataset(s) a usar (padrão: ambos).\n"
            "Opções: cc | ieee"
        ),
    )
    parser.add_argument(
        "--ag-time-limit",
        type=int,
        default=600,
        metavar="SEGUNDOS",
        help="Time limit (s) para o AutoGluon (padrão: 600).",
    )
    parser.add_argument(
        "--ag-presets",
        default="best_quality",
        metavar="PRESET",
        help="Preset do AutoGluon: best_quality | high_quality | medium_quality (padrão: best_quality).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Não gera os gráficos ao final.",
    )
    return parser.parse_args()


def resolve_selection(raw_list, alias_map, label):
    """Converte aliases digitados para nomes canônicos; aborta se inválido."""
    if not raw_list:
        return list(dict.fromkeys(alias_map.values()))  # todos, sem duplicatas
    resolved = []
    for token in raw_list:
        key = token.lower()
        if key not in alias_map:
            valid = ", ".join(sorted(set(alias_map.keys())))
            raise SystemExit(f"[erro] {label} desconhecido: '{token}'. Válidos: {valid}")
        canonical = alias_map[key]
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def main():
    args = parse_args()

    if args.trees:
        algos = TREE_ALGOS
    elif args.dl:
        algos = DL_GRAPH_ALGOS
    else:
        algos = resolve_selection(args.algo, ALGO_ALIASES, "algoritmo")
    datasets = resolve_selection(args.dataset, DATASET_ALIASES, "dataset")

    print(f"Algoritmos : {', '.join(algos)}")
    print(f"Datasets   : {', '.join(datasets)}")

    # ── Carregar dados ─────────────────────────────────────────────────────────
    cc_df, ieee_df = load_datasets()

    all_results = []

    # ── CC ─────────────────────────────────────────────────────────────────────
    ag_kwargs = dict(ag_time_limit=args.ag_time_limit, ag_presets=args.ag_presets)

    if "CC" in datasets:
        def cc_graph(X_all, df_s):
            return build_cc_graph(X_all, k=10)
        cc_results = run_dataset(CC_CONFIG, cc_df, cc_graph, algos=algos, **ag_kwargs)
        all_results.extend(cc_results)

    # ── IEEE ───────────────────────────────────────────────────────────────────
    if "IEEE" in datasets:
        def ieee_graph(X_all, df_s):
            return build_ieee_graph(df_s, max_per_card=100)
        ieee_results = run_dataset(IEEE_CONFIG, ieee_df, ieee_graph, algos=algos, **ag_kwargs)
        all_results.extend(ieee_results)

    # ── Relatórios ─────────────────────────────────────────────────────────────
    if not args.no_plots and all_results:
        reporting.generate_all(all_results)

    # ── Tabela resumo ──────────────────────────────────────────────────────────
    print_summary(all_results)


if __name__ == "__main__":
    main()
