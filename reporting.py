"""
reporting.py — geração das 5 figuras matplotlib de comparação de algoritmos.

Cada result dict deve conter:
  name, dataset, y_test, scores, roc_auc, pr_auc, f1,
  precision, recall, train_time
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

BASE    = Path(__file__).parent
OUT_DIR = BASE / "plots" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGO_COLORS = {
    "XGBoost":          "#2ECC71",
    "Isolation Forest": "#E67E22",
    "Autoencoder":      "#9B59B6",
    "GNN (GCN)":        "#1ABC9C",
    "GraphSAGE+XGB":    "#16A085",
    "LightGBM":         "#3498DB",
    "CatBoost":         "#E74C3C",
    "TabNet":           "#F39C12",
    "Stacking":         "#2C3E50",
}

DATASETS = ["CC", "IEEE"]


def _color(name):
    return ALGO_COLORS.get(name, "#888888")


def _save(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


def _group_by_dataset(results):
    """Returns dict {dataset: [result, ...]}."""
    grouped = {ds: [] for ds in DATASETS}
    for r in results:
        grouped[r["dataset"]].append(r)
    return grouped


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(results):
    grouped = _group_by_dataset(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Curvas ROC por Dataset", fontsize=14, fontweight="bold")

    for ax, ds in zip(axes, DATASETS):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Aleatório")
        for r in grouped[ds]:
            fpr, tpr, _ = roc_curve(r["y_test"], r["scores"])
            ax.plot(fpr, tpr, color=_color(r["name"]), lw=2,
                    label=f"{r['name']} (AUC={r['roc_auc']:.3f})")
        ax.set_title(f"Dataset {ds}")
        ax.set_xlabel("Taxa de Falso Positivo")
        ax.set_ylabel("Taxa de Verdadeiro Positivo")
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "01_roc_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_pr_curves(results):
    grouped = _group_by_dataset(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Curvas Precision-Recall por Dataset", fontsize=14, fontweight="bold")

    for ax, ds in zip(axes, DATASETS):
        for r in grouped[ds]:
            prec, rec, _ = precision_recall_curve(r["y_test"], r["scores"])
            ax.plot(rec, prec, color=_color(r["name"]), lw=2,
                    label=f"{r['name']} (AP={r['pr_auc']:.3f})")
        ax.set_title(f"Dataset {ds}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "02_pr_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Metric Bars (ROC-AUC e PR-AUC)
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_bars(results):
    grouped = _group_by_dataset(results)
    # Collect all algorithm names in order
    algo_names = []
    for ds in DATASETS:
        for r in grouped[ds]:
            if r["name"] not in algo_names:
                algo_names.append(r["name"])

    n_algos = len(algo_names)
    x = np.arange(n_algos)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ROC-AUC e PR-AUC por Algoritmo × Dataset", fontsize=14, fontweight="bold")

    for ax, metric_key, metric_label in [
        (axes[0], "roc_auc", "ROC-AUC"),
        (axes[1], "pr_auc",  "PR-AUC"),
    ]:
        for di, ds in enumerate(DATASETS):
            rmap = {r["name"]: r[metric_key] for r in grouped[ds]}
            vals = [rmap.get(n, 0.0) for n in algo_names]
            offset = (di - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=ds,
                          color=[_color(n) for n in algo_names],
                          alpha=0.85 if di == 0 else 0.55,
                          edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_ylim(0, 1.1)
        ax.legend(title="Dataset", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "03_metric_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Training Times
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_times(results):
    grouped = _group_by_dataset(results)

    algo_names = []
    for ds in DATASETS:
        for r in grouped[ds]:
            if r["name"] not in algo_names:
                algo_names.append(r["name"])

    n_algos = len(algo_names)
    y = np.arange(n_algos)
    height = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Tempo de Treinamento por Algoritmo × Dataset", fontsize=14, fontweight="bold")

    for ax, ds in zip(axes, DATASETS):
        rmap = {r["name"]: r["train_time"] for r in grouped[ds]}
        vals = [rmap.get(n, 0.0) for n in algo_names]
        colors = [_color(n) for n in algo_names]
        bars = ax.barh(y, vals, height, color=colors, edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}s", va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(algo_names)
        ax.set_xlabel("Tempo (segundos)")
        ax.set_title(f"Dataset {ds}")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "04_training_times.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Score Distributions
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_distributions(results):
    grouped = _group_by_dataset(results)

    # Collect algos per dataset
    algos_cc   = [r for r in grouped["CC"]]
    algos_ieee = [r for r in grouped["IEEE"]]
    n_algos = max(len(algos_cc), len(algos_ieee))

    fig, axes = plt.subplots(2, n_algos, figsize=(5 * n_algos, 8))
    if n_algos == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Distribuição dos Scores: Fraude vs Legítima", fontsize=14, fontweight="bold")

    for row_i, (ds, algo_list) in enumerate([("CC", algos_cc), ("IEEE", algos_ieee)]):
        for col_i in range(n_algos):
            ax = axes[row_i][col_i]
            if col_i >= len(algo_list):
                ax.set_visible(False)
                continue
            r = algo_list[col_i]
            scores = r["scores"]
            y_test = r["y_test"]

            sc_legit = scores[y_test == 0]
            sc_fraud = scores[y_test == 1]

            # Clip to [1st, 99th] percentile for readability
            lo = np.percentile(scores, 1)
            hi = np.percentile(scores, 99)
            bins = np.linspace(lo, hi, 50)

            ax.hist(sc_legit, bins=bins, alpha=0.6, color="#4C72B0", label="Legítima", density=True)
            ax.hist(sc_fraud, bins=bins, alpha=0.6, color="#DD3E3E", label="Fraude",   density=True)
            ax.set_title(f"{ds} – {r['name']}", fontsize=9)
            ax.set_xlabel("Score de Anomalia")
            ax.set_ylabel("Densidade")
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "05_score_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — KS Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_ks_curves(results):
    grouped = _group_by_dataset(results)
    algos_cc   = grouped["CC"]
    algos_ieee = grouped["IEEE"]
    n_algos = max(len(algos_cc), len(algos_ieee), 1)

    fig, axes = plt.subplots(2, n_algos, figsize=(5 * n_algos, 8))
    if n_algos == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Curvas KS: Distribuições Cumulativas por Classe", fontsize=14, fontweight="bold")

    for row_i, (ds, algo_list) in enumerate([("CC", algos_cc), ("IEEE", algos_ieee)]):
        for col_i in range(n_algos):
            ax = axes[row_i][col_i]
            if col_i >= len(algo_list):
                ax.set_visible(False)
                continue
            r = algo_list[col_i]
            sc_fraud = np.sort(r["scores"][r["y_test"] == 1])
            sc_legit = np.sort(r["scores"][r["y_test"] == 0])

            cdf_fraud = np.arange(1, len(sc_fraud) + 1) / len(sc_fraud)
            cdf_legit = np.arange(1, len(sc_legit) + 1) / len(sc_legit)

            ax.plot(sc_fraud, cdf_fraud, color="#DD3E3E", lw=2, label="Fraude")
            ax.plot(sc_legit, cdf_legit, color="#4C72B0", lw=2, label="Legítima")

            # Locate and annotate KS statistic
            all_x = np.sort(np.concatenate([sc_fraud, sc_legit]))
            cdf_f = np.searchsorted(sc_fraud, all_x, side="right") / len(sc_fraud)
            cdf_l = np.searchsorted(sc_legit, all_x, side="right") / len(sc_legit)
            diff  = np.abs(cdf_f - cdf_l)
            ks_idx = np.argmax(diff)
            ks_x   = all_x[ks_idx]
            ks_y1, ks_y2 = cdf_f[ks_idx], cdf_l[ks_idx]

            ax.plot([ks_x, ks_x], [ks_y1, ks_y2], "k-", lw=2)
            ax.annotate(
                f"KS={diff[ks_idx]:.3f}",
                xy=(ks_x, (ks_y1 + ks_y2) / 2),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8, va="center",
            )

            ks_val = r.get("ks", diff[ks_idx])
            ax.set_title(f"{ds} – {r['name']}\n(KS={ks_val:.3f})", fontsize=9)
            ax.set_xlabel("Score")
            if col_i == 0:
                ax.set_ylabel("CDF")
                ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "06_ks_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Cumulative Gain Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_gain_curves(results):
    grouped = _group_by_dataset(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Curvas de Ganho Cumulativo", fontsize=14, fontweight="bold")

    for ax, ds in zip(axes, DATASETS):
        for r in grouped[ds]:
            y_test = r["y_test"]
            scores = r["scores"]

            sorted_idx  = np.argsort(scores)[::-1]
            sorted_y    = y_test[sorted_idx]
            pct_pop     = np.arange(1, len(y_test) + 1) / len(y_test)
            pct_capture = np.cumsum(sorted_y) / y_test.sum()

            ax.plot(pct_pop, pct_capture, color=_color(r["name"]), lw=2, label=r["name"])

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Aleatório")
        # Reference lines at 10% and 20% of population inspected
        for xv, alpha in [(0.10, 0.7), (0.20, 0.4)]:
            ax.axvline(xv, color="gray", linestyle=":", lw=1, alpha=alpha)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Fração de Transações Inspecionadas")
        ax.set_ylabel("Fração de Fraudes Capturadas")
        ax.set_title(f"Dataset {ds}")
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "07_gain_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Threshold Analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_threshold_analysis(results):
    grouped = _group_by_dataset(results)
    algos_cc   = grouped["CC"]
    algos_ieee = grouped["IEEE"]
    n_algos = max(len(algos_cc), len(algos_ieee), 1)

    fig, axes = plt.subplots(2, n_algos, figsize=(4.5 * n_algos, 8))
    if n_algos == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Análise de Threshold: Precision, Recall e F1", fontsize=14, fontweight="bold")

    for row_i, (ds, algo_list) in enumerate([("CC", algos_cc), ("IEEE", algos_ieee)]):
        for col_i in range(n_algos):
            ax = axes[row_i][col_i]
            if col_i >= len(algo_list):
                ax.set_visible(False)
                continue
            r = algo_list[col_i]

            precs, recs, threshs = precision_recall_curve(r["y_test"], r["scores"])
            f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-8)

            ax.plot(threshs, precs[:-1], color="#2196F3", lw=1.5, label="Precision")
            ax.plot(threshs, recs[:-1],  color="#F44336", lw=1.5, label="Recall")
            ax.plot(threshs, f1s,        color="#4CAF50", lw=2,   label="F1")

            best_t = threshs[np.argmax(f1s)]
            ax.axvline(best_t, color="gray", linestyle="--", lw=1, alpha=0.8)
            ax.text(best_t, 0.04, f"t*={best_t:.2f}", fontsize=7,
                    ha="center", color="gray")

            ax.set_title(f"{ds} – {r['name']}\n(F1*={r['f1']:.3f})", fontsize=9)
            ax.set_xlabel("Threshold")
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlim(float(threshs.min()), float(threshs.max()))
            if col_i == 0:
                ax.set_ylabel("Métrica")
                ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "08_threshold_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(results):
    grouped = _group_by_dataset(results)
    algos_cc   = grouped["CC"]
    algos_ieee = grouped["IEEE"]
    n_algos = max(len(algos_cc), len(algos_ieee), 1)

    fig, axes = plt.subplots(2, n_algos, figsize=(3.5 * n_algos, 7))
    if n_algos == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Matrizes de Confusão (threshold ótimo F1)", fontsize=14, fontweight="bold")

    for row_i, (ds, algo_list) in enumerate([("CC", algos_cc), ("IEEE", algos_ieee)]):
        for col_i in range(n_algos):
            ax = axes[row_i][col_i]
            if col_i >= len(algo_list):
                ax.set_visible(False)
                continue
            r = algo_list[col_i]

            precs, recs, threshs = precision_recall_curve(r["y_test"], r["scores"])
            f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-8)
            best_thresh = threshs[np.argmax(f1s)]
            y_pred = (r["scores"] >= best_thresh).astype(int)

            cm      = confusion_matrix(r["y_test"], y_pred)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            labels = [["TN", "FP"], ["FN", "TP"]]
            for i in range(2):
                for j in range(2):
                    text_color = "white" if cm_norm[i, j] > 0.6 else "black"
                    ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}\n({cm_norm[i, j]:.1%})",
                            ha="center", va="center", fontsize=7, color=text_color)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Legítima", "Fraude"], fontsize=8)
            ax.set_yticklabels(["Legítima", "Fraude"], fontsize=8)
            ax.set_xlabel("Predito", fontsize=8)
            if col_i == 0:
                ax.set_ylabel("Real", fontsize=8)
            mcc_val = r.get("mcc", 0)
            ax.set_title(f"{ds} – {r['name']}\n(MCC={mcc_val:.3f})", fontsize=9)

    fig.tight_layout()
    _save(fig, "09_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_all(results):
    """Generate all 9 figures from a list of result dicts."""
    print("\nGerando figuras…")
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_metric_bars(results)
    plot_training_times(results)
    plot_score_distributions(results)
    plot_ks_curves(results)
    plot_gain_curves(results)
    plot_threshold_analysis(results)
    plot_confusion_matrices(results)
    print(f"  Todos os plots salvos em: {OUT_DIR}/")
