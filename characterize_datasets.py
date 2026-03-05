"""
Caracterização dos dois datasets de detecção de fraudes:
  1. Credit Card Fraud Detection (ULB / Kaggle)
  2. IEEE-CIS Fraud Detection (Vesta / Kaggle)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from preprocessing import load_datasets, PLOTS_DIR

OUT_DIR = PLOTS_DIR
OUT_DIR.mkdir(exist_ok=True)

PALETTE = {"legit": "#4C72B0", "fraud": "#DD3E3E"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def pct_fmt(x, _):
    return f"{x:.1f}%"

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")

def missing_pct(df):
    m = df.isnull().mean() * 100
    return m[m > 0].sort_values(ascending=False)


def main():
    # ══════════════════════════════════════════════════════════════════════════════
    # 1. CARREGAMENTO
    # ══════════════════════════════════════════════════════════════════════════════
    print("Carregando dados…")
    cc, ieee = load_datasets()

    # ══════════════════════════════════════════════════════════════════════════════
    # FIG 1 – Visão Geral Comparativa (side-by-side)
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n[1/8] Visão geral comparativa…")

    cc_fraud_rate  = cc["Class"].mean() * 100
    ieee_fraud_rate = ieee["isFraud"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Visão Geral dos Datasets de Detecção de Fraude", fontsize=15, fontweight="bold")

    for ax, label, total, fraud_rate, n_feat in zip(
        axes,
        ["Credit Card\n(ULB/Kaggle)", "IEEE-CIS\n(Vesta/Kaggle)"],
        [len(cc), len(ieee)],
        [cc_fraud_rate, ieee_fraud_rate],
        [cc.shape[1] - 1, ieee.shape[1] - 1],
    ):
        legit = 100 - fraud_rate
        bars = ax.bar(["Legítima", "Fraude"], [legit, fraud_rate],
                      color=[PALETTE["legit"], PALETTE["fraud"]], width=0.5, edgecolor="white")
        for bar, val in zip(bars, [legit, fraud_rate]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, f"{val:.3f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(f"{label}\n{total:,} transações · {n_feat} features", fontsize=11)
        ax.set_ylabel("% das transações")
        ax.set_ylim(0, max(legit, fraud_rate) * 1.15)
        ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "01_overview.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # FIG 2 – Distribuição dos Valores de Transação
    # ══════════════════════════════════════════════════════════════════════════════
    print("[2/8] Distribuição dos valores de transação…")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Distribuição do Valor das Transações", fontsize=14, fontweight="bold")

    # CC – todos
    ax = axes[0, 0]
    ax.hist(cc["Amount"], bins=100, color=PALETTE["legit"], alpha=0.8, edgecolor="none")
    ax.set_title("CC – Todas as transações (Amount)")
    ax.set_xlabel("Valor (€)")
    ax.set_ylabel("Frequência")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)

    # CC – fraude vs legítima (log scale)
    ax = axes[0, 1]
    for cls, col, lbl in [(0, PALETTE["legit"], "Legítima"), (1, PALETTE["fraud"], "Fraude")]:
        sub = cc[cc["Class"] == cls]["Amount"]
        ax.hist(sub, bins=80, color=col, alpha=0.6, label=lbl, edgecolor="none")
    ax.set_title("CC – Fraude vs Legítima")
    ax.set_xlabel("Valor (€)")
    ax.set_ylabel("Frequência")
    ax.set_yscale("log")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    # IEEE – todos
    ax = axes[1, 0]
    ax.hist(ieee["TransactionAmt"], bins=100, color=PALETTE["legit"], alpha=0.8, edgecolor="none")
    ax.set_title("IEEE – Todas as transações (TransactionAmt)")
    ax.set_xlabel("Valor (USD)")
    ax.set_ylabel("Frequência")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)

    # IEEE – fraude vs legítima
    ax = axes[1, 1]
    for cls, col, lbl in [(0, PALETTE["legit"], "Legítima"), (1, PALETTE["fraud"], "Fraude")]:
        sub = ieee[ieee["isFraud"] == cls]["TransactionAmt"]
        ax.hist(sub, bins=80, color=col, alpha=0.6, label=lbl, edgecolor="none")
    ax.set_title("IEEE – Fraude vs Legítima")
    ax.set_xlabel("Valor (USD)")
    ax.set_ylabel("Frequência")
    ax.set_yscale("log")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "02_transaction_amounts.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # FIG 3 – Distribuição Temporal
    # ══════════════════════════════════════════════════════════════════════════════
    print("[3/8] Distribuição temporal…")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Distribuição Temporal das Transações", fontsize=14, fontweight="bold")

    # CC – Time em horas
    cc_hours = cc["Time"] / 3600
    ax = axes[0, 0]
    ax.hist(cc_hours, bins=96, color=PALETTE["legit"], alpha=0.8, edgecolor="none")
    ax.set_title("CC – Volume por hora (2 dias)")
    ax.set_xlabel("Tempo desde 1ª transação (horas)")
    ax.set_ylabel("Frequência")
    ax.spines[["top", "right"]].set_visible(False)

    # CC – fraude vs legítima por hora
    ax = axes[0, 1]
    for cls, col, lbl in [(0, PALETTE["legit"], "Legítima"), (1, PALETTE["fraud"], "Fraude")]:
        sub = cc_hours[cc["Class"] == cls]
        ax.hist(sub, bins=48, color=col, alpha=0.6, label=lbl, edgecolor="none", density=True)
    ax.set_title("CC – Perfil temporal normalizado")
    ax.set_xlabel("Tempo (horas)")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    # IEEE – TransactionDT em dias
    ieee_days = ieee["TransactionDT"] / 86400
    ax = axes[1, 0]
    ax.hist(ieee_days, bins=100, color=PALETTE["legit"], alpha=0.8, edgecolor="none")
    ax.set_title("IEEE – Volume por dia (TransactionDT)")
    ax.set_xlabel("Dias desde referência")
    ax.set_ylabel("Frequência")
    ax.spines[["top", "right"]].set_visible(False)

    # IEEE – fraude vs legítima por dia
    ax = axes[1, 1]
    for cls, col, lbl in [(0, PALETTE["legit"], "Legítima"), (1, PALETTE["fraud"], "Fraude")]:
        sub = ieee_days[ieee["isFraud"] == cls]
        ax.hist(sub, bins=100, color=col, alpha=0.6, label=lbl, edgecolor="none", density=True)
    ax.set_title("IEEE – Perfil temporal normalizado")
    ax.set_xlabel("Dias desde referência")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "03_temporal.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # FIG 4 – Valores Ausentes
    # ══════════════════════════════════════════════════════════════════════════════
    print("[4/8] Valores ausentes…")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Valores Ausentes por Dataset", fontsize=14, fontweight="bold")

    # CC
    cc_miss = missing_pct(cc)
    ax = axes[0]
    if len(cc_miss) == 0:
        ax.text(0.5, 0.5, "Nenhum valor ausente!", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
    else:
        ax.barh(cc_miss.index, cc_miss.values, color=PALETTE["fraud"])
    ax.set_title(f"CC – {len(cc_miss)} colunas com NaN")
    ax.set_xlabel("% de valores ausentes")
    ax.spines[["top", "right"]].set_visible(False)

    # IEEE – top 40 colunas com mais NaN
    ieee_miss = missing_pct(ieee).head(40)
    ax = axes[1]
    ax.barh(ieee_miss.index[::-1], ieee_miss.values[::-1], color=PALETTE["fraud"])
    ax.set_title(f"IEEE – Top 40 de {missing_pct(ieee).shape[0]} colunas com NaN")
    ax.set_xlabel("% de valores ausentes")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "04_missing_values.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # FIG 5 – Componentes PCA do Dataset CC (boxplot fraude vs legítima)
    # ══════════════════════════════════════════════════════════════════════════════
    print("[5/8] Componentes PCA (CC)…")

    pca_cols = [f"V{i}" for i in range(1, 29)]
    cc_legit = cc[cc["Class"] == 0][pca_cols]
    cc_fraud  = cc[cc["Class"] == 1][pca_cols]

    fig, axes = plt.subplots(4, 7, figsize=(18, 12))
    fig.suptitle("CC – Distribuição das Componentes PCA (V1–V28): Fraude vs Legítima",
                 fontsize=13, fontweight="bold")

    for i, col in enumerate(pca_cols):
        ax = axes[i // 7][i % 7]
        data_l = cc_legit[col].dropna()
        data_f  = cc_fraud[col].dropna()
        bp = ax.boxplot([data_l, data_f], patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 2},
                        flierprops={"marker": ".", "markersize": 1, "alpha": 0.3})
        bp["boxes"][0].set_facecolor(PALETTE["legit"])
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(PALETTE["fraud"])
        bp["boxes"][1].set_alpha(0.7)
        ax.set_title(col, fontsize=9)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["L", "F"], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # Remove últimos axes vazios (28 features em grade 4x7=28, exato)
    fig.tight_layout()
    save(fig, "05_cc_pca_components.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # FIG 6 – IEEE: Features Categóricas Chave
    # ══════════════════════════════════════════════════════════════════════════════
    print("[6/8] Features categóricas IEEE…")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("IEEE – Distribuição das Features Categóricas Principais", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    cat_feats = [
        ("ProductCD",      "Código do Produto"),
        ("card4",          "Bandeira do Cartão"),
        ("card6",          "Tipo do Cartão"),
        ("P_emaildomain",  "Domínio Email Pagador (top 8)"),
        ("R_emaildomain",  "Domínio Email Receptor (top 8)"),
        ("DeviceType",     "Tipo de Dispositivo"),
    ]

    for idx, (col, title) in enumerate(cat_feats):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        if col not in ieee.columns:
            ax.set_visible(False)
            continue

        top_n = 8
        vc = ieee[col].value_counts().head(top_n)
        fraud_rate_by_cat = (
            ieee.groupby(col)["isFraud"].mean() * 100
        ).reindex(vc.index)

        x = np.arange(len(vc))
        width = 0.4

        ax2 = ax.twinx()
        ax.bar(x - width / 2, vc.values, width, color=PALETTE["legit"], alpha=0.8, label="Volume")
        ax2.bar(x + width / 2, fraud_rate_by_cat.values, width,
                color=PALETTE["fraud"], alpha=0.8, label="Taxa Fraude")

        ax.set_xticks(x)
        ax.set_xticklabels(vc.index, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Contagem", fontsize=8)
        ax2.set_ylabel("% Fraude", fontsize=8, color=PALETTE["fraud"])
        ax2.tick_params(axis="y", colors=PALETTE["fraud"])
        ax.spines[["top"]].set_visible(False)
        ax2.spines[["top"]].set_visible(False)

    # Legenda global
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["legit"], alpha=0.8, label="Volume (eixo esq.)"),
        Patch(facecolor=PALETTE["fraud"], alpha=0.8, label="Taxa de Fraude % (eixo dir.)"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=False)
    save(fig, "06_ieee_categorical.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # FIG 7 – IEEE: Features C e D (contagem e distâncias)
    # ══════════════════════════════════════════════════════════════════════════════
    print("[7/8] Features C e D do IEEE…")

    c_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in ieee.columns]
    d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in ieee.columns]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("IEEE – Features C (contagens) e D (distâncias/dias)", fontsize=13, fontweight="bold")

    for ax, cols, ylabel, title in [
        (axes[0], c_cols, "Mediana", "Features C – Mediana por classe"),
        (axes[1], d_cols, "Mediana", "Features D – Mediana por classe"),
    ]:
        x = np.arange(len(cols))
        width = 0.35
        for offset, cls, col, lbl in [
            (-width / 2, 0, PALETTE["legit"], "Legítima"),
            ( width / 2, 1, PALETTE["fraud"], "Fraude"),
        ]:
            medians = [ieee[ieee["isFraud"] == cls][c].median() for c in cols]
            ax.bar(x + offset, medians, width, color=col, alpha=0.8, label=lbl, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "07_ieee_C_D_features.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # FIG 8 – Resumo Estatístico Comparativo (tabela visual)
    # ══════════════════════════════════════════════════════════════════════════════
    print("[8/8] Resumo estatístico comparativo…")

    def stats_amount(df, col, label):
        return {
            "Mínimo":   df[col].min(),
            "Mediana":  df[col].median(),
            "Média":    df[col].mean(),
            "Máximo":   df[col].max(),
            "Desvio P.": df[col].std(),
        }

    cc_amt_stats  = stats_amount(cc, "Amount", "CC")
    ieee_amt_stats = stats_amount(ieee, "TransactionAmt", "IEEE")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Resumo Comparativo dos Datasets", fontsize=14, fontweight="bold")

    # Comparação geral em tabela
    ax = axes[0]
    ax.axis("off")
    rows = [
        ["Propriedade",              "CC (ULB)",              "IEEE-CIS"],
        ["Nº Transações",            f"{len(cc):,}",          f"{len(ieee):,}"],
        ["Nº Features",              f"{cc.shape[1]-1}",      f"{ieee.shape[1]-1}"],
        ["Taxa de Fraude",           f"{cc_fraud_rate:.3f}%", f"{ieee_fraud_rate:.3f}%"],
        ["Fraudes (abs.)",           f"{cc['Class'].sum():,}", f"{ieee['isFraud'].sum():,}"],
        ["Valor Mín. Transação",     f"€{cc['Amount'].min():.2f}",    f"${ieee['TransactionAmt'].min():.2f}"],
        ["Valor Máx. Transação",     f"€{cc['Amount'].max():.2f}",    f"${ieee['TransactionAmt'].max():.2f}"],
        ["Valor Médio (Legítima)",   f"€{cc[cc['Class']==0]['Amount'].mean():.2f}",
                                      f"${ieee[ieee['isFraud']==0]['TransactionAmt'].mean():.2f}"],
        ["Valor Médio (Fraude)",     f"€{cc[cc['Class']==1]['Amount'].mean():.2f}",
                                      f"${ieee[ieee['isFraud']==1]['TransactionAmt'].mean():.2f}"],
        ["% NaN (média)",            f"{cc.isnull().mean().mean()*100:.1f}%",
                                      f"{ieee.isnull().mean().mean()*100:.1f}%"],
        ["Período",                  "2 dias (Set/2013)",     "~6 meses"],
        ["Tipo de dado",             "Cartão Crédito (POS)",  "E-commerce"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EBF5FB")
        cell.set_edgecolor("#CCCCCC")
    ax.set_title("Características Gerais", fontsize=10, pad=12)

    # Boxplot valor – CC
    ax = axes[1]
    data_cc = [
        cc[cc["Class"] == 0]["Amount"].clip(upper=500),
        cc[cc["Class"] == 1]["Amount"].clip(upper=500),
    ]
    bp = ax.boxplot(data_cc, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    bp["boxes"][0].set_facecolor(PALETTE["legit"]); bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor(PALETTE["fraud"]); bp["boxes"][1].set_alpha(0.8)
    ax.set_xticklabels(["Legítima", "Fraude"])
    ax.set_title("CC – Valor da Transação\n(recortado em €500)")
    ax.set_ylabel("Valor (€)")
    ax.spines[["top", "right"]].set_visible(False)

    # Boxplot valor – IEEE
    ax = axes[2]
    data_ieee = [
        ieee[ieee["isFraud"] == 0]["TransactionAmt"].clip(upper=1000),
        ieee[ieee["isFraud"] == 1]["TransactionAmt"].clip(upper=1000),
    ]
    bp = ax.boxplot(data_ieee, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    bp["boxes"][0].set_facecolor(PALETTE["legit"]); bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor(PALETTE["fraud"]); bp["boxes"][1].set_alpha(0.8)
    ax.set_xticklabels(["Legítima", "Fraude"])
    ax.set_title("IEEE – Valor da Transação\n(recortado em $1000)")
    ax.set_ylabel("Valor (USD)")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "08_summary.png")

    # ══════════════════════════════════════════════════════════════════════════════
    print(f"\nPronto! Todos os gráficos salvos em: {OUT_DIR}/")
    print("Arquivos gerados:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
