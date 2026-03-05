"""
preprocessing.py — carregamento, amostragem e preparação de features
para o framework de detecção de fraudes.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
CC_PATH   = BASE / "Credit Card Fraud Detection" / "creditcard.csv"
IEEE_TXN  = BASE / "ieee-fraud-detection" / "train_transaction.csv"
IEEE_ID   = BASE / "ieee-fraud-detection" / "train_identity.csv"

# ── Feature lists ──────────────────────────────────────────────────────────────
CC_NUM_FEATS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

IEEE_NUM_FEATS = (
    ["TransactionAmt"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 6)]
    + [f"V{i}" for i in range(1, 21)]
)

IEEE_CAT_FEATS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]


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


def prepare_numeric(df, feat_cols, label_col, seed=42):
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

    # Split
    idx = np.arange(len(y_all))
    idx_tr, idx_te = train_test_split(
        idx,
        test_size=0.2,
        stratify=y_all,
        random_state=seed,
    )

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
    N     = len(df)

    # ── Split cronológico 70 / 15 / 15 ────────────────────────────────────────
    chrono  = np.argsort(times, kind="stable")
    tr_end  = int(N * 0.70)
    val_end = int(N * 0.85)
    idx_tr  = chrono[:tr_end]
    idx_val = chrono[tr_end:val_end]
    idx_te  = chrono[val_end:]

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
    chrono  = np.argsort(times, kind="stable")
    tr_end  = int(N * 0.70)
    val_end = int(N * 0.85)
    idx_tr  = chrono[:tr_end]
    idx_val = chrono[tr_end:val_end]
    idx_te  = chrono[val_end:]

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


def prepare_xgboost(df, num_cols, cat_cols, label_col, seed=42):
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

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
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
