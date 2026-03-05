"""
detectors.py — BaseDetector (protocolo informal) + 4 implementações:
  XGBoostDetector, IsolationForestDetector, AutoencoderDetector, GNNDetector

Funções auxiliares de grafo:
  build_cc_graph, build_ieee_graph, normalize_adj
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp


# ══════════════════════════════════════════════════════════════════════════════
# 1. XGBoostDetector
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostDetector:
    name = "XGBoost"

    def __init__(self, n_estimators=300, seed=42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.train_time = 0.0
        self._model = None

    def fit(self, X_tr, y_tr):
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split

        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        spw = n_neg / max(n_pos, 1)

        # 15% interno para early stopping
        X_t, X_val, y_t, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=self.seed
        )

        try:
            self._model = XGBClassifier(
                n_estimators=self.n_estimators,
                scale_pos_weight=spw,
                tree_method="hist",
                eval_metric="aucpr",
                early_stopping_rounds=20,
                random_state=self.seed,
                verbosity=0,
            )
            t0 = time.time()
            self._model.fit(
                X_t, y_t,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        except Exception:
            self._model = XGBClassifier(
                n_estimators=self.n_estimators,
                scale_pos_weight=spw,
                tree_method="hist",
                eval_metric="logloss",
                early_stopping_rounds=20,
                random_state=self.seed,
                verbosity=0,
            )
            t0 = time.time()
            self._model.fit(
                X_t, y_t,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        self.train_time = time.time() - t0
        return self

    def score(self, X_te):
        return self._model.predict_proba(X_te)[:, 1]


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
# 3. AutoencoderDetector
# ══════════════════════════════════════════════════════════════════════════════

class AutoencoderDetector:
    name = "Autoencoder"

    def __init__(self, epochs=150, batch_size=512, lr=1e-3, patience=15, seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seed = seed
        self.train_time = 0.0
        self._model = None

    def _build_model(self, n_features):
        import torch.nn as nn

        class AE(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(16, 32), nn.ReLU(),
                    nn.Linear(32, 64), nn.ReLU(),
                    nn.Linear(64, n),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        return AE(n_features)

    def fit(self, X_tr, y_tr):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)

        # Treinar apenas em legítimas
        X_legit = X_tr[y_tr == 0].astype(np.float32)
        X_tensor = torch.from_numpy(X_legit)

        n_features = X_tr.shape[1]
        model = self._build_model(n_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        best_loss = float("inf")
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0

        t0 = time.time()
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(X_legit)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break

        model.load_state_dict(best_state)
        self.train_time = time.time() - t0
        self._model = model
        return self

    def score(self, X_te):
        import torch
        import torch.nn as nn

        self._model.eval()
        X_tensor = torch.from_numpy(X_te.astype(np.float32))
        criterion = nn.MSELoss(reduction="none")
        with torch.no_grad():
            out = self._model(X_tensor)
            mse = criterion(out, X_tensor).mean(dim=1).numpy()
        return mse


# ══════════════════════════════════════════════════════════════════════════════
# 4. GNNDetector  (transductive, 3-layer GCN)
# ══════════════════════════════════════════════════════════════════════════════

class _GCNModel:
    """3-layer GCN com dropout e batch normalization (implementado como nn.Module)."""
    pass  # definido dentro de GNNDetector.fit para evitar import top-level de torch


class GNNDetector:
    name = "GNN (GCN)"

    def __init__(self, hidden=128, epochs=500, lr=3e-3, patience=30,
                 dropout=0.5, print_every=20, seed=42):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.dropout = dropout
        self.print_every = print_every
        self.seed = seed
        self.train_time = 0.0
        self._logits = None
        self._te_mask = None

    def fit(self, X_all, y_all, train_mask, test_mask, adj_norm):
        """
        Transductive GCN com 3 camadas, dropout, batch norm e early stopping
        baseado em val_loss (split interno de 15% dos nós de treino).

        Parameters
        ----------
        X_all      : np.ndarray, shape (N, F)
        y_all      : np.ndarray, shape (N,)
        train_mask : np.ndarray bool, shape (N,)
        test_mask  : np.ndarray bool, shape (N,)
        adj_norm   : torch.sparse_coo_tensor, shape (N, N)
        """
        import torch
        import torch.nn as nn
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        N, F = X_all.shape

        # ── Split de validação a partir dos nós de treino (15%) ────────────────
        tr_indices = np.where(train_mask)[0]
        val_size = max(1, int(0.15 * len(tr_indices)))
        rng = np.random.default_rng(self.seed)
        val_indices = rng.choice(tr_indices, size=val_size, replace=False)

        val_mask_inner = np.zeros(N, dtype=bool)
        val_mask_inner[val_indices] = True
        tr_mask_inner = train_mask.copy()
        tr_mask_inner[val_indices] = False

        # ── Pesos para desequilíbrio de classes ────────────────────────────────
        y_tr_labels = y_all[tr_mask_inner]
        n_pos = int((y_tr_labels == 1).sum())
        n_neg = int((y_tr_labels == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

        print(f"    Nós: treino={tr_mask_inner.sum():,}  "
              f"val={val_mask_inner.sum():,}  teste={test_mask.sum():,}")
        print(f"    Features: {F}  |  pos_weight={pos_weight.item():.1f}  "
              f"|  hidden={self.hidden}  dropout={self.dropout}")

        X_t = torch.from_numpy(X_all.astype(np.float32))
        y_t = torch.from_numpy(y_all.astype(np.float32))
        tr_mask_t  = torch.from_numpy(tr_mask_inner)
        val_mask_t = torch.from_numpy(val_mask_inner)

        # ── Arquitetura: 3-layer GCN com BatchNorm e Dropout ──────────────────
        class _GCN(nn.Module):
            def __init__(self, in_f, hid, drop):
                super().__init__()
                self.fc1 = nn.Linear(in_f, hid, bias=False)
                self.fc2 = nn.Linear(hid, hid, bias=False)
                self.fc3 = nn.Linear(hid, 1, bias=False)
                self.bn1 = nn.BatchNorm1d(hid)
                self.bn2 = nn.BatchNorm1d(hid)
                self.drop = nn.Dropout(drop)
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
                nn.init.xavier_uniform_(self.fc3.weight)

            def forward(self, A, X):
                # Camada 1: H1 = Dropout(ReLU(BN(A @ X @ W1)))
                H = self.drop(torch.relu(self.bn1(torch.sparse.mm(A, self.fc1(X)))))
                # Camada 2: H2 = Dropout(ReLU(BN(A @ H1 @ W2)))
                H = self.drop(torch.relu(self.bn2(torch.sparse.mm(A, self.fc2(H)))))
                # Camada 3: logits = A @ H2 @ W3
                return torch.sparse.mm(A, self.fc3(H)).squeeze(1)

        model = _GCN(F, self.hidden, self.dropout)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=8, factor=0.5, min_lr=1e-5
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        A = adj_norm.coalesce()

        best_val_loss = float("inf")
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
        best_epoch     = 0
        prev_lr        = self.lr

        header = (f"    {'Época':>6} | {'Loss Treino':>11} | {'Loss Val':>9} | "
                  f"{'AUC Val':>8} | {'LR':>9} | Status")
        sep    = "    " + "-" * (len(header) - 4)
        print(header)
        print(sep)

        t0 = time.time()
        for epoch in range(self.epochs):

            # ── Passo de treino ────────────────────────────────────────────────
            model.train()
            optimizer.zero_grad()
            logits = model(A, X_t)
            tr_loss = criterion(logits[tr_mask_t], y_t[tr_mask_t])
            tr_loss.backward()
            optimizer.step()

            # ── Avaliação no conjunto de validação ─────────────────────────────
            model.eval()
            with torch.no_grad():
                logits_eval = model(A, X_t)
                val_loss = criterion(
                    logits_eval[val_mask_t], y_t[val_mask_t]
                ).item()
                val_scores = torch.sigmoid(logits_eval[val_mask_t]).numpy()
                try:
                    val_auc = roc_auc_score(y_all[val_mask_inner], val_scores)
                except Exception:
                    val_auc = float("nan")

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # ── Status da iteração ─────────────────────────────────────────────
            status = ""
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
                best_epoch     = epoch
                status = "✓ melhor"
            else:
                patience_count += 1

            if current_lr < prev_lr - 1e-10:
                status = (status + " ↓ lr").strip()
                prev_lr = current_lr

            # ── Impressão periódica ────────────────────────────────────────────
            should_print = (
                epoch % self.print_every == 0
                or epoch < 3
                or patience_count >= self.patience  # última linha antes de parar
            )
            if should_print:
                print(f"    {epoch:>6} | {tr_loss.item():>11.4f} | {val_loss:>9.4f} | "
                      f"{val_auc:>8.4f} | {current_lr:>9.2e} | {status}")

            if patience_count >= self.patience:
                print(f"    Early stopping na época {epoch}  "
                      f"(melhor: época {best_epoch}, val_loss={best_val_loss:.4f})")
                break

        else:
            print(f"    Treinamento concluído ({self.epochs} épocas).  "
                  f"Melhor: época {best_epoch}, val_loss={best_val_loss:.4f}")

        self.train_time = time.time() - t0

        # ── Inferência final com os melhores pesos ─────────────────────────────
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits_final   = model(A, X_t)
            self._logits   = logits_final.numpy()
        self._te_mask = test_mask

        return self

    def score(self):
        """Retorna scores (probabilidade de fraude) para os nós de teste."""
        import torch
        scores_all = torch.sigmoid(torch.from_numpy(self._logits)).numpy()
        return scores_all[self._te_mask]


# ══════════════════════════════════════════════════════════════════════════════
# 5. GraphSAGEXGBDetector  (GraphSAGE feature extractor → XGBoost classifier)
# ══════════════════════════════════════════════════════════════════════════════

class GraphSAGEXGBDetector:
    """
    Híbrido em dois estágios:
      1. GraphSAGE leve (2 camadas) treinado de forma supervisionada para
         extrair embeddings estruturais de cada nó (transação).
      2. Os embeddings são concatenados às features tabulares originais e
         passados para um XGBoostClassifier para a decisão final.

    A normalização da adjacência é row-wise (D⁻¹A), adequada para a
    agregação de média do GraphSAGE — diferente da normalização simétrica
    D⁻⁰·⁵AD⁻⁰·⁵ usada pelo GCN.
    """
    name = "GraphSAGE+XGB"

    def __init__(self, hidden=64, emb_dim=32, epochs=200, lr=1e-3,
                 patience=20, dropout=0.3, n_estimators=300, seed=42):
        self.hidden = hidden
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.dropout = dropout
        self.n_estimators = n_estimators
        self.seed = seed
        self.train_time = 0.0
        self._xgb = None
        self._embeddings_all = None
        self._train_mask = None
        self._test_mask = None

    @staticmethod
    def _row_normalize_adj(adj_sparse):
        """D⁻¹ A (row-normalized, sem self-loops) → torch.sparse_coo_tensor."""
        import torch

        adj_csr = adj_sparse.tocsr().astype(np.float32)
        deg = np.array(adj_csr.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        adj_coo = adj_csr.tocoo()
        norm_data = (1.0 / deg)[adj_coo.row] * adj_coo.data
        indices = torch.from_numpy(
            np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64)
        )
        values = torch.from_numpy(norm_data.astype(np.float32))
        N = adj_csr.shape[0]
        return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    def fit(self, X_all, y_all, train_mask, test_mask, adj_sparse, Xg_tr, yg_tr):
        """
        Parâmetros
        ----------
        X_all       : np.ndarray (N, F)  — features GNN (escaladas) para todos os nós
        y_all       : np.ndarray (N,)    — rótulos para todos os nós
        train_mask  : np.ndarray bool    — máscara de treino sobre N nós
        test_mask   : np.ndarray bool    — máscara de teste sobre N nós
        adj_sparse  : scipy.sparse       — adjacência bruta (SEM normalização)
        Xg_tr       : np.ndarray         — features tabulares XGBoost (treino)
        yg_tr       : np.ndarray         — rótulos XGBoost (treino)
        """
        import torch
        import torch.nn as nn
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        N, F = X_all.shape

        # ── Adjacência row-normalizada para SAGE ──────────────────────────────
        A = self._row_normalize_adj(adj_sparse)

        # ── Split de validação interno a partir dos nós de treino (15%) ───────
        tr_indices = np.where(train_mask)[0]
        val_size = max(1, int(0.15 * len(tr_indices)))
        rng = np.random.default_rng(self.seed)
        val_indices = rng.choice(tr_indices, size=val_size, replace=False)

        val_mask = np.zeros(N, dtype=bool)
        val_mask[val_indices] = True
        tr_mask = train_mask.copy()
        tr_mask[val_indices] = False

        # ── Peso para desequilíbrio de classes ────────────────────────────────
        n_pos = int((y_all[tr_mask] == 1).sum())
        n_neg = int((y_all[tr_mask] == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

        print(f"    Nós: treino={tr_mask.sum():,}  val={val_mask.sum():,}  "
              f"teste={test_mask.sum():,}")
        print(f"    Features GNN: {F}  |  hidden={self.hidden}  "
              f"emb={self.emb_dim}  dropout={self.dropout}")

        X_t = torch.from_numpy(X_all.astype(np.float32))
        y_t = torch.from_numpy(y_all.astype(np.float32))
        tr_mask_t  = torch.from_numpy(tr_mask)
        val_mask_t = torch.from_numpy(val_mask)

        # ── Arquitetura GraphSAGE (mean aggregator, pesos separados) ──────────
        class _SAGE(nn.Module):
            def __init__(self, in_f, hid, emb, drop):
                super().__init__()
                # Camada 1: h_self e h_neigh → hidden
                self.fc1_self  = nn.Linear(in_f, hid, bias=True)
                self.fc1_neigh = nn.Linear(in_f, hid, bias=False)
                self.bn1 = nn.BatchNorm1d(hid)
                # Camada 2: hidden_self e hidden_neigh → emb
                self.fc2_self  = nn.Linear(hid, emb, bias=True)
                self.fc2_neigh = nn.Linear(hid, emb, bias=False)
                self.bn2 = nn.BatchNorm1d(emb)
                # Cabeça de classificação (usada só no treino)
                self.head = nn.Linear(emb, 1, bias=True)
                self.drop = nn.Dropout(drop)
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

            def _sage_layer(self, A, H, fc_s, fc_n, bn):
                return self.drop(torch.relu(bn(
                    fc_s(H) + fc_n(torch.sparse.mm(A, H))
                )))

            def embed(self, A, X):
                H = self._sage_layer(A, X, self.fc1_self, self.fc1_neigh, self.bn1)
                E = self._sage_layer(A, H, self.fc2_self, self.fc2_neigh, self.bn2)
                return E

            def forward(self, A, X):
                return self.head(self.embed(A, X)).squeeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Mover dados e modelo (Exemplo na linha 513 em diante)
        X_t = torch.from_numpy(X_all.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y_all.astype(np.float32)).to(device)
        A = self._row_normalize_adj(adj_sparse).to(device) # A matriz esparsa também vai para GPU

        model = _SAGE(F, self.hidden, self.emb_dim, self.dropout).to(device)
        # model = _SAGE(F, self.hidden, self.emb_dim, self.dropout)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=8, factor=0.5, min_lr=1e-5
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss  = float("inf")
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
        best_epoch     = 0
        prev_lr        = self.lr

        header = (f"    {'Época':>6} | {'Loss Treino':>11} | {'Loss Val':>9} | "
                  f"{'AUC Val':>8} | {'LR':>9} | Status")
        sep_line = "    " + "-" * (len(header) - 4)
        print(header)
        print(sep_line)

        t0 = time.time()
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(A, X_t)
            tr_loss = criterion(logits[tr_mask_t], y_t[tr_mask_t])
            tr_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits_ev = model(A, X_t)
                val_loss  = criterion(logits_ev[val_mask_t], y_t[val_mask_t]).item()
                val_sc    = torch.sigmoid(logits_ev[val_mask_t]).numpy()
                try:
                    val_auc = roc_auc_score(y_all[val_mask], val_sc)
                except Exception:
                    val_auc = float("nan")

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            status = ""
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
                best_epoch     = epoch
                status = "✓ melhor"
            else:
                patience_count += 1

            if current_lr < prev_lr - 1e-10:
                status = (status + " ↓ lr").strip()
                prev_lr = current_lr

            if epoch % 20 == 0 or epoch < 3 or patience_count >= self.patience:
                print(f"    {epoch:>6} | {tr_loss.item():>11.4f} | {val_loss:>9.4f} | "
                      f"{val_auc:>8.4f} | {current_lr:>9.2e} | {status}")

            if patience_count >= self.patience:
                print(f"    Early stopping na época {epoch}  "
                      f"(melhor: época {best_epoch}, val_loss={best_val_loss:.4f})")
                break
        else:
            print(f"    SAGE concluído ({self.epochs} épocas).  "
                  f"Melhor: época {best_epoch}, val_loss={best_val_loss:.4f}")

        sage_time = time.time() - t0

        # ── Extração de embeddings para todos os nós ──────────────────────────
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            self._embeddings_all = model.embed(A, X_t).numpy()  # (N, emb_dim)

        self._train_mask = train_mask
        self._test_mask  = test_mask

        # ── Augmentação e treino do XGBoost ───────────────────────────────────
        # train_mask e prepare_xgboost usam o mesmo seed/stratify → alinhados
        tr_emb = self._embeddings_all[train_mask]  # (n_train, emb_dim)
        Xg_tr_aug = np.hstack([Xg_tr, tr_emb])

        n_cols_tab = Xg_tr.shape[1]
        print(f"\n    [Embeddings SAGE] shape={tr_emb.shape}  "
              f"SAGE time={sage_time:.1f}s")
        print(f"    Features: {n_cols_tab} tabulares + {self.emb_dim} emb "
              f"= {Xg_tr_aug.shape[1]} total")

        n_pos_xgb = int((yg_tr == 1).sum())
        n_neg_xgb = int((yg_tr == 0).sum())
        spw = n_neg_xgb / max(n_pos_xgb, 1)

        X_t_xgb, X_val_xgb, y_t_xgb, y_val_xgb = train_test_split(
            Xg_tr_aug, yg_tr, test_size=0.15, stratify=yg_tr,
            random_state=self.seed,
        )

        try:
            self._xgb = XGBClassifier(
                n_estimators=self.n_estimators,
                scale_pos_weight=spw,
                tree_method="hist",
                device="cuda", # Adicione isso para usar a GPU
                eval_metric="aucpr",
                early_stopping_rounds=20,
                random_state=self.seed,
                verbosity=0,
            )
            t1 = time.time()
            self._xgb.fit(
                X_t_xgb, y_t_xgb,
                eval_set=[(X_val_xgb, y_val_xgb)],
                verbose=False,
            )
        except Exception:
            self._xgb = XGBClassifier(
                n_estimators=self.n_estimators,
                scale_pos_weight=spw,
                tree_method="hist",
                eval_metric="logloss",
                early_stopping_rounds=20,
                random_state=self.seed,
                verbosity=0,
            )
            t1 = time.time()
            self._xgb.fit(
                X_t_xgb, y_t_xgb,
                eval_set=[(X_val_xgb, y_val_xgb)],
                verbose=False,
            )

        xgb_time = time.time() - t1
        print(f"    XGBoost time={xgb_time:.1f}s")

        self.train_time = time.time() - t0
        return self

    def score(self, Xg_te):
        """Concatena embeddings de teste com features tabulares e classifica."""
        te_emb = self._embeddings_all[self._test_mask]
        Xg_te_aug = np.hstack([Xg_te, te_emb])
        return self._xgb.predict_proba(Xg_te_aug)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# 7. LightGBMDetector  (DART booster)
# ══════════════════════════════════════════════════════════════════════════════

class LightGBMDetector:
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

class CatBoostDetector:
    name = "CatBoost"

    def __init__(self, iterations=1000, depth=8, cat_features=None, seed=42):
        self.iterations = iterations
        self.depth = depth
        self.cat_features = cat_features or []
        self.seed = seed
        self.train_time = 0.0
        self._model = None

    def _to_int_cats(self, X):
        X = X.copy()
        for idx in self.cat_features:
            X[:, idx] = X[:, idx].astype(int)
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
            early_stopping_rounds=50,
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


# ══════════════════════════════════════════════════════════════════════════════
# 9. TabNetDetector
# ══════════════════════════════════════════════════════════════════════════════

class TabNetDetector:
    name = "TabNet"

    def __init__(self, max_epochs=200, patience=20, batch_size=1024,
                 n_d=32, n_a=32, n_steps=5, gamma=1.5, seed=42):
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.seed = seed
        self.train_time = 0.0
        self._model    = None
        self._scaler   = None  # StandardScaler para features numéricas
        self._num_idxs = None

    def fit(self, X_tr, y_tr, cat_idxs=None, cat_dims=None):
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        cat_idxs = cat_idxs or []
        cat_dims  = cat_dims  or []

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        pos_weight = n_neg / max(n_pos, 1)

        print(f"    n_d={self.n_d} | n_steps={self.n_steps} | gamma={self.gamma} | "
              f"cat_features={len(cat_idxs)} | pos_weight={pos_weight:.1f}")

        X_t, X_val, y_t, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=self.seed
        )

        # TabNet é uma rede neural: normalizar features numéricas é essencial.
        # prepare_xgboost não escala (correto para árvores), então fazemos aqui.
        num_idxs = [i for i in range(X_t.shape[1]) if i not in cat_idxs]
        scaler = StandardScaler()
        X_t   = X_t.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_t[:, num_idxs]   = scaler.fit_transform(X_t[:, num_idxs])
        X_val[:, num_idxs] = scaler.transform(X_val[:, num_idxs])
        self._scaler   = scaler
        self._num_idxs = num_idxs

        for idx in cat_idxs:
            X_t[:, idx]   = X_t[:, idx].astype(int).astype(np.float32)
            X_val[:, idx] = X_val[:, idx].astype(int).astype(np.float32)

        self._model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=min(self.n_d // 2, 8),
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 2e-3, "weight_decay": 1e-5},
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_params={"T_max": self.max_epochs, "eta_min": 1e-5},
            mask_type="entmax",
            seed=self.seed,
            verbose=10,
        )

        t0 = time.time()
        self._model.fit(
            X_t, y_t,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            weights={0: 1.0, 1: float(pos_weight)},
        )
        self.train_time = time.time() - t0
        return self

    def score(self, X_te):
        X_te = X_te.astype(np.float32).copy()
        X_te[:, self._num_idxs] = self._scaler.transform(X_te[:, self._num_idxs])
        return self._model.predict_proba(X_te)[:, 1]


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


# ══════════════════════════════════════════════════════════════════════════════
# Graph helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_cc_graph(X, k=15):
    """
    Constrói grafo k-NN para o dataset CC.
    - Dá peso 3× à feature Amount (última coluna) antes do PCA para
      enfatizar diferenças de valor transacionado.
    - Usa 20 componentes PCA (vs 15 anteriores) para preservar mais variância.
    - k=15 vizinhos para grafo mais rico.
    Retorna scipy.sparse.coo_matrix (N×N).
    """
    from sklearn.decomposition import PCA

    N = X.shape[0]

    # Peso extra para Amount (última coluna)
    X_weighted = X.copy()
    X_weighted[:, -1] *= 3.0

    n_components = min(20, X_weighted.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_red = pca.fit_transform(X_weighted)

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
    nn_model.fit(X_red)
    _, indices = nn_model.kneighbors(X_red)

    rows, cols = [], []
    for i, nbrs in enumerate(indices):
        for j in nbrs[1:]:
            rows.append(i)
            cols.append(j)

    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    adj = adj + adj.T
    adj.data[:] = 1.0
    print(f"    Grafo CC: {N:,} nós | {adj.nnz // 2:,} arestas | "
          f"grau médio={adj.nnz / N:.1f}")
    return adj


def build_ieee_graph(df, max_per_card=100):
    """
    Grafo multi-relacional para o dataset IEEE:
    conecta transações que compartilham card1, addr1 ou P_emaildomain.
    Cada relação tem um limite de arestas por grupo para manter o grafo esparso.
    Retorna scipy.sparse.coo_matrix (N×N).
    """
    N = len(df)
    df_reset = df.reset_index(drop=True)

    rows, cols = [], []

    def _add_relation(col, max_per_group):
        """Adiciona arestas para todos os grupos de col."""
        if col not in df_reset.columns:
            return 0
        n_edges = 0
        groups = df_reset.dropna(subset=[col]).groupby(col, sort=False).indices
        for _, idxs in groups.items():
            idxs = list(idxs)
            if len(idxs) < 2:
                continue
            if len(idxs) > max_per_group:
                idxs = idxs[:max_per_group]
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    rows.append(idxs[a]); cols.append(idxs[b])
                    rows.append(idxs[b]); cols.append(idxs[a])
                    n_edges += 1
        return n_edges

    relacoes = [
        ("card1",         max_per_card),   # mesmo cartão → sinal mais forte
        ("addr1",         50),             # mesmo CEP de cobrança
        ("P_emaildomain", 30),             # mesmo domínio de e-mail do comprador
    ]
    for col, lim in relacoes:
        n = _add_relation(col, lim)
        print(f"      {col:<15}: {n:,} arestas")

    if not rows:
        print(f"    Grafo IEEE: {N:,} nós | sem arestas")
        return sp.coo_matrix((N, N), dtype=np.float32)

    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    # Deduplica e binariza
    adj = adj.tocsr()
    adj.data[:] = 1.0
    adj = adj.tocoo()
    print(f"    Grafo IEEE: {N:,} nós | {adj.nnz // 2:,} arestas | "
          f"grau médio={adj.nnz / N:.1f}")
    return adj


def normalize_adj(adj):
    """
    D^-0.5 (A + I) D^-0.5  →  torch.sparse_coo_tensor, coalesced.
    """
    import torch

    N = adj.shape[0]
    adj = adj.tocsr()

    # Add self-loops
    adj = adj + sp.eye(N, dtype=np.float32)
    adj = adj.tocoo()

    # Degree
    deg = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    # D^-0.5 A D^-0.5
    row, col, data = adj.row, adj.col, adj.data
    norm_data = d_inv_sqrt[row] * data * d_inv_sqrt[col]

    indices = torch.from_numpy(np.vstack([row, col]).astype(np.int64))
    values = torch.from_numpy(norm_data.astype(np.float32))
    t = torch.sparse_coo_tensor(indices, values, (N, N))
    return t.coalesce()
