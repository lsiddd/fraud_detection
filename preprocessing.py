"""
preprocessing.py — carregamento, amostragem e preparação de features
para o framework de detecção de fraudes.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
CC_PATH   = BASE / "Credit Card Fraud Detection" / "creditcard.csv"
IEEE_TXN  = BASE / "ieee-fraud-detection" / "train_transaction.csv"
IEEE_ID           = BASE / "ieee-fraud-detection" / "train_identity.csv"
PLOTS_DIR         = BASE / "plots"
PLOTS_RESULTS_DIR = BASE / "plots" / "results"

# ── Feature lists ──────────────────────────────────────────────────────────────
CC_NUM_FEATS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

IEEE_NUM_FEATS = (
    ["TransactionAmt"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 6)]
    + [f"V{i}" for i in range(1, 21)]
)

IEEE_CAT_FEATS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]

# ── Dataset configurations ──────────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetConfig:
    name:      str
    label_col: str
    time_col:  str
    num_feats: list
    cat_feats: list = field(default_factory=list)

CC_CONFIG = DatasetConfig(
    name="CC",
    label_col="Class",
    time_col="Time",
    num_feats=CC_NUM_FEATS,
)

IEEE_CONFIG = DatasetConfig(
    name="IEEE",
    label_col="isFraud",
    time_col="TransactionDT",
    num_feats=IEEE_NUM_FEATS,
    cat_feats=IEEE_CAT_FEATS,
)


# ── Functions ──────────────────────────────────────────────────────────────────

def load_datasets():
    """Load CC and IEEE datasets. Returns (cc_df, ieee_df)."""
    print("Carregando datasets…")
    cc = pd.read_csv(CC_PATH)
    txn = pd.read_csv(IEEE_TXN)
    iid = pd.read_csv(IEEE_ID)
    ieee = txn.merge(iid, on="TransactionID", how="left")
    print(f"  CC   → {cc.shape[0]:,} linhas × {cc.shape[1]} colunas")
    print(f"  IEEE → {ieee.shape[0]:,} linhas × {ieee.shape[1]} colunas")
    return cc, ieee


def sample_stratified(df, label_col, n, seed=42):
    """
    Retorna um DataFrame amostrado estratificadamente com n linhas.
    Se n >= len(df), retorna df inteiro.
    """
    if n >= len(df):
        return df.reset_index(drop=True)
    _, sampled = train_test_split(
        df,
        test_size=n / len(df),
        stratify=df[label_col],
        random_state=seed,
    )
    return sampled.reset_index(drop=True)


def make_chronological_split(df, time_col):
    """
    Split cronológico 70/15/15. Retorna (idx_tr, idx_val, idx_te)
    como arrays de índices posicionais sobre df reset_index(drop=True).
    """
    times   = df[time_col].values.astype(np.float64) if time_col in df.columns \
              else np.arange(len(df), dtype=np.float64)
    chrono  = np.argsort(times, kind="stable")
    N       = len(df)
    tr_end  = int(N * 0.70)
    val_end = int(N * 0.85)
    return chrono[:tr_end], chrono[tr_end:val_end], chrono[val_end:]


def prepare_numeric(df, feat_cols, label_col, idx_tr, idx_te, seed=42):
    """
    Imputa mediana, aplica StandardScaler, faz split treino/teste.

    Returns
    -------
    X_all, y_all, X_tr, X_te, y_tr, y_te, idx_tr, idx_te
      idx_tr/idx_te são índices posicionais em X_all (para uso no GNN).
    """
    # Filtrar colunas disponíveis
    available = [c for c in feat_cols if c in df.columns]
    missing = set(feat_cols) - set(df.columns)
    if missing:
        print(f"  [aviso] {len(missing)} colunas ausentes ignoradas: {sorted(missing)[:5]}…")

    X_raw = df[available].values
    y_all = df[label_col].values.astype(int)

    # Imputação e escalonamento
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_imp)

    X_tr, X_te = X_all[idx_tr], X_all[idx_te]
    y_tr, y_te = y_all[idx_tr], y_all[idx_te]

    return X_all, y_all, X_tr, X_te, y_tr, y_te, idx_tr, idx_te


def prepare_gnn(df, num_cols, cat_cols, label_col):
    """
    Retorna X_all (N, F) com features numéricas + categóricas escaladas para o GNN.
    As categóricas são ordinal-encodadas e normalizadas com StandardScaler.
    O split treino/teste é feito externamente (usando os mesmos idx de prepare_numeric).
    """
    available_num = [c for c in num_cols if c in df.columns]
    available_cat = [c for c in cat_cols if c in df.columns]
    missing = (set(num_cols) | set(cat_cols)) - set(df.columns)
    if missing:
        print(f"  [aviso] {len(missing)} colunas ausentes ignoradas: {sorted(missing)[:5]}…")

    X_num = SimpleImputer(strategy="median").fit_transform(df[available_num].values)
    X_num = StandardScaler().fit_transform(X_num)

    if available_cat:
        X_cat = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit_transform(df[available_cat].astype(str).values).astype(np.float32)
        X_cat = StandardScaler().fit_transform(X_cat)
        return np.hstack([X_num, X_cat]).astype(np.float32)

    return X_num.astype(np.float32)


def prepare_gat_cc(df):
    """
    Preparação SOTA para GATv2 no dataset CC.

    Aplica três correções em relação ao prepare_gnn padrão:
      1. Codificação cíclica do Time (sin/cos da hora do dia) — converte variável
         monotonamente crescente em assinatura comportamental estável.
      2. RobustScaler no Amount — ignora outliers de fraude ao escalar.
      3. Split cronológico 70 / 15 / 15 — preserva causalidade temporal e
         elimina data leakage de informação futura no grafo.

    Returns
    -------
    X_gat   : (N, 31) float32 — features dos nós [V1-V28 | sin_t | cos_t | Amount]
    X_pca   : (N, 28) float32 — apenas V1-V28 escalados (para construção do grafo)
    times   : (N,)    float64 — segundos absolutos originais (filtragem causal)
    y_all   : (N,)    int
    idx_tr, idx_val, idx_te : arrays de índices posicionais no eixo N
    """
    times = df["Time"].values.astype(np.float64)
    y_all = df["Class"].values.astype(int)

    # ── Split cronológico 70 / 15 / 15 ────────────────────────────────────────
    idx_tr, idx_val, idx_te = make_chronological_split(df, "Time")

    # ── V1-V28: StandardScaler ─────────────────────────────────────────────────
    pca_cols  = [f"V{i}" for i in range(1, 29)]
    X_pca_raw = SimpleImputer(strategy="median").fit_transform(df[pca_cols].values)
    X_pca     = StandardScaler().fit_transform(X_pca_raw).astype(np.float32)

    # ── Amount: RobustScaler (imune a outliers de fraude) ─────────────────────
    X_amt = RobustScaler().fit_transform(
        df[["Amount"]].values.astype(np.float64)
    ).astype(np.float32)

    # ── Time: codificação cíclica (hora do dia) ────────────────────────────────
    hours    = (times / 3600.0) % 24.0
    time_sin = np.sin(2 * np.pi * hours / 24.0).reshape(-1, 1).astype(np.float32)
    time_cos = np.cos(2 * np.pi * hours / 24.0).reshape(-1, 1).astype(np.float32)

    # ── Features finais: [V1-V28 | sin_t | cos_t | Amount] = 31 colunas ──────
    X_gat = np.hstack([X_pca, time_sin, time_cos, X_amt])

    return X_gat, X_pca, times, y_all, idx_tr, idx_val, idx_te


def prepare_gat_ieee(df):
    """
    Preparação SOTA para GATv2 no dataset IEEE.

    Aplica as mesmas correções que prepare_gat_cc adaptadas ao IEEE:
      - TransactionDT como tempo absoluto para split cronológico e filtragem causal.
      - RobustScaler no TransactionAmt.
      - Codificação cíclica do TransactionDT (hora do dia).
      - StandardScaler nas colunas numéricas restantes.
      - OrdinalEncoder + StandardScaler nas categóricas.

    Returns
    -------
    X_gat     : (N, F+2) float32 — features dos nós com tempo cíclico
    times     : (N,)     float64 — segundos absolutos (filtragem causal)
    y_all     : (N,)     int
    idx_tr, idx_val, idx_te : arrays de índices posicionais
    """
    times = df["TransactionDT"].values.astype(np.float64) if "TransactionDT" in df.columns \
            else np.arange(len(df), dtype=np.float64)
    y_all = df["isFraud"].values.astype(int)
    N     = len(df)

    # ── Split cronológico 70 / 15 / 15 ────────────────────────────────────────
    idx_tr, idx_val, idx_te = make_chronological_split(df, "TransactionDT")

    # ── Numéricas (exceto TransactionAmt) ─────────────────────────────────────
    num_feats  = (
        [f"C{i}" for i in range(1, 15)]
        + [f"D{i}" for i in range(1, 6)]
        + [f"V{i}" for i in range(1, 21)]
    )
    avail_num = [c for c in num_feats if c in df.columns]
    X_num     = SimpleImputer(strategy="median").fit_transform(
        df[avail_num].values.astype(np.float64)
    )
    X_num = StandardScaler().fit_transform(X_num).astype(np.float32)

    # ── TransactionAmt: RobustScaler ──────────────────────────────────────────
    X_amt = RobustScaler().fit_transform(
        df[["TransactionAmt"]].values.astype(np.float64)
        if "TransactionAmt" in df.columns
        else np.zeros((N, 1))
    ).astype(np.float32)

    # ── Categóricas ────────────────────────────────────────────────────────────
    cat_feats  = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
    avail_cat  = [c for c in cat_feats if c in df.columns]
    if avail_cat:
        X_cat = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit_transform(df[avail_cat].astype(str).values).astype(np.float32)
        X_cat = StandardScaler().fit_transform(X_cat).astype(np.float32)
    else:
        X_cat = np.empty((N, 0), dtype=np.float32)

    # ── Time: codificação cíclica ──────────────────────────────────────────────
    hours    = (times / 3600.0) % 24.0
    time_sin = np.sin(2 * np.pi * hours / 24.0).reshape(-1, 1).astype(np.float32)
    time_cos = np.cos(2 * np.pi * hours / 24.0).reshape(-1, 1).astype(np.float32)

    parts = [X_num, X_amt, time_sin, time_cos]
    if X_cat.shape[1] > 0:
        parts.append(X_cat)
    X_gat = np.hstack(parts)

    return X_gat, times, y_all, idx_tr, idx_val, idx_te


def prepare_xgboost(df, num_cols, cat_cols, label_col, idx_tr, idx_te, seed=42):
    """
    Prepara features para XGBoost: numéricas imputadas + categóricas com
    OrdinalEncoder. Retorna (X_tr, X_te, y_tr, y_te).
    """
    available_num = [c for c in num_cols if c in df.columns]
    available_cat = [c for c in cat_cols if c in df.columns]

    y = df[label_col].values.astype(int)

    # Numéricas
    X_num = df[available_num].values
    imputer = SimpleImputer(strategy="median")
    X_num = imputer.fit_transform(X_num)

    # Categóricas
    if available_cat:
        X_cat_raw = df[available_cat].astype(str).values
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = enc.fit_transform(X_cat_raw)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num

    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    return X_tr, X_te, y_tr, y_te


def prepare_xgboost_all(df, num_cols, cat_cols, label_col):
    """Mesmo preprocessing de prepare_xgboost, sem train_test_split."""
    available_num = [c for c in num_cols if c in df.columns]
    available_cat = [c for c in cat_cols if c in df.columns]
    y = df[label_col].values.astype(int)
    X_num = SimpleImputer(strategy="median").fit_transform(df[available_num].values)
    if available_cat:
        X_cat = OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1).fit_transform(
            df[available_cat].astype(str).values)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num
    return X, y


def prepare_ensemble_cc(df, idx_tr, idx_te):
    """
    Feature engineering enriquecido para SuperEnsembleDetector no dataset CC.

    Adiciona sobre as 28 features PCA (V1-V28):
      - Cyclic time encoding (sin/cos da hora do dia)
      - log1p(Amount) e Amount escalado por RobustScaler
      - L2-norms dos blocos V1-V14, V15-V28, V1-V28
      - Estatísticas por transação (mean, std, max, min, neg_count) dos 28 Vs

    Returns (X_tr, X_te, y_tr, y_te)
    """
    y = df["Class"].values.astype(int)
    y_tr, y_te = y[idx_tr], y[idx_te]

    # ── V1-V28: imputar mediana (fit no treino) ─────────────────────────────
    v_cols = [f"V{i}" for i in range(1, 29)]
    V_raw = df[v_cols].values.astype(np.float64)
    v_imputer = SimpleImputer(strategy="median")
    v_imputer.fit(V_raw[idx_tr])
    V = v_imputer.transform(V_raw)

    # ── V-norms (por transação) ─────────────────────────────────────────────
    V_norm_1_14  = np.linalg.norm(V[:, :14],  axis=1)
    V_norm_15_28 = np.linalg.norm(V[:, 14:],  axis=1)
    V_norm_all   = np.linalg.norm(V,           axis=1)

    # ── V-stats (por transação) ─────────────────────────────────────────────
    V_mean      = V.mean(axis=1)
    V_std       = V.std(axis=1)
    V_max       = V.max(axis=1)
    V_min       = V.min(axis=1)
    V_neg_count = (V < 0).sum(axis=1).astype(np.float64)

    # ── Amount ──────────────────────────────────────────────────────────────
    amount = df["Amount"].values.astype(np.float64)
    amount_log = np.log1p(amount)

    robust_scaler = RobustScaler()
    robust_scaler.fit(amount[idx_tr].reshape(-1, 1))
    amount_robust = robust_scaler.transform(amount.reshape(-1, 1)).ravel()

    # ── Cyclic time ──────────────────────────────────────────────────────────
    times    = df["Time"].values.astype(np.float64)
    hours    = (times / 3600.0) % 24.0
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    # ── Assemble ─────────────────────────────────────────────────────────────
    extra = np.column_stack([
        hour_sin, hour_cos,
        amount_log, amount_robust,
        V_norm_1_14, V_norm_15_28, V_norm_all,
        V_mean, V_std, V_max, V_min, V_neg_count,
    ])
    X = np.hstack([V, extra]).astype(np.float32)

    return X[idx_tr], X[idx_te], y_tr, y_te


def prepare_ensemble_ieee(df, idx_tr, idx_te):
    """
    Feature engineering enriquecido para SuperEnsembleDetector no dataset IEEE.

    Expande sobre o prepare_xgboost padrão:
      - Numéricas: TransactionAmt + C1-C14 + D1-D15 + V1-V339
                   + card1/2/3/5 + addr1/2 + dist1/2
      - Categóricas: ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1-M9
      - Aggregate features (computadas no treino, mapeadas sem leakage):
          card1_count/mean_amt/std_amt/amt_dev, addr1_count/mean_amt,
          P_email_freq, R_email_freq, card1_target_enc
      - Time: hour_sin, hour_cos, day_of_week
      - Missing indicators: D1_isnan … D15_isnan

    Returns (X_tr, X_te, y_tr, y_te)
    """
    y = df["isFraud"].values.astype(int)
    y_tr, y_te = y[idx_tr], y[idx_te]

    df_tr = df.iloc[idx_tr]

    # ── Time features ────────────────────────────────────────────────────────
    times = (
        df["TransactionDT"].values.astype(np.float64)
        if "TransactionDT" in df.columns
        else np.zeros(len(df))
    )
    hours      = (times / 3600.0) % 24.0
    hour_sin   = np.sin(2 * np.pi * hours / 24.0)
    hour_cos   = np.cos(2 * np.pi * hours / 24.0)
    day_of_week = ((times / 86400.0) % 7)

    # ── Missing indicators for D1-D15 ────────────────────────────────────────
    d_cols_all = [f"D{i}" for i in range(1, 16)]
    d_available = [c for c in d_cols_all if c in df.columns]
    D_isnan = np.column_stack([
        df[c].isna().values.astype(np.float32) for c in d_available
    ]) if d_available else np.empty((len(df), 0), dtype=np.float32)

    # ── Aggregate features (fit on train only) ───────────────────────────────
    agg_parts = []
    global_mean_amt = float(df_tr["TransactionAmt"].mean()) if "TransactionAmt" in df.columns else 0.0
    global_fraud_rate = float(df_tr["isFraud"].mean())
    amt_col = df["TransactionAmt"].values if "TransactionAmt" in df.columns else np.zeros(len(df))

    if "card1" in df.columns:
        g = df_tr.groupby("card1")["TransactionAmt"].agg(["mean", "std", "count"])
        g["std"] = g["std"].fillna(0.0)

        card1_mean  = df["card1"].map(g["mean"]).fillna(global_mean_amt).values
        card1_std   = df["card1"].map(g["std"]).fillna(0.0).values
        card1_count = df["card1"].map(g["count"]).fillna(0.0).values
        card1_dev   = (amt_col - card1_mean) / (card1_std + 1.0)

        agg_parts.append(np.column_stack([
            card1_count, card1_mean, card1_std, card1_dev
        ]))

    if "addr1" in df.columns:
        ga = df_tr.groupby("addr1")["TransactionAmt"].agg(["mean", "count"])
        addr1_count    = df["addr1"].map(ga["count"]).fillna(0.0).values
        addr1_mean_amt = df["addr1"].map(ga["mean"]).fillna(global_mean_amt).values
        agg_parts.append(np.column_stack([addr1_count, addr1_mean_amt]))

    if "P_emaildomain" in df.columns:
        p_freq = df_tr["P_emaildomain"].value_counts() / len(df_tr)
        agg_parts.append(
            df["P_emaildomain"].map(p_freq).fillna(0.0).values.reshape(-1, 1)
        )

    if "R_emaildomain" in df.columns:
        r_freq = df_tr["R_emaildomain"].value_counts() / len(df_tr)
        agg_parts.append(
            df["R_emaildomain"].map(r_freq).fillna(0.0).values.reshape(-1, 1)
        )

    # ── Filter V columns by missing rate (fit on train only) ─────────────────
    # IEEE V1-V339 have wildly varying missingness; high-missing cols are noise
    # and multiply tree-building cost (n_features × n_leaves splits/iteration).
    v_cols_present = [f"V{i}" for i in range(1, 340) if f"V{i}" in df.columns]
    v_missing_rate = df.iloc[idx_tr][v_cols_present].isna().mean()
    v_cols_keep = v_missing_rate[v_missing_rate < 0.5].index.tolist()
    print(f"    V-cols: {len(v_cols_keep)}/{len(v_cols_present)} kept (missing < 50%)")

    # ── Numeric features (imputer fit on train) ──────────────────────────────
    num_cols = (
        ["TransactionAmt"]
        + [f"C{i}" for i in range(1, 15)]
        + [f"D{i}" for i in range(1, 16)]
        + v_cols_keep
        + ["card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]
    )
    avail_num = [c for c in num_cols if c in df.columns]
    X_num_raw = df[avail_num].values.astype(np.float64)
    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(X_num_raw[idx_tr])
    X_num = num_imputer.transform(X_num_raw)

    # ── Categorical features (encoder fit on train) ──────────────────────────
    cat_cols = (
        ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
        + [f"M{i}" for i in range(1, 10)]
    )
    avail_cat = [c for c in cat_cols if c in df.columns]
    if avail_cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df.iloc[idx_tr][avail_cat].astype(str).values)
        X_cat = enc.transform(df[avail_cat].astype(str).values)
    else:
        X_cat = np.empty((len(df), 0), dtype=np.float64)

    # ── Assemble all parts ───────────────────────────────────────────────────
    time_feats = np.column_stack([hour_sin, hour_cos, day_of_week])
    parts = [X_num, time_feats]
    if D_isnan.shape[1] > 0:
        parts.append(D_isnan)
    if X_cat.shape[1] > 0:
        parts.append(X_cat)
    for ap in agg_parts:
        parts.append(ap)

    X = np.hstack(parts).astype(np.float32)
    return X[idx_tr], X[idx_te], y_tr, y_te
