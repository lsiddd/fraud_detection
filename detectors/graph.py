"""
graph.py — GNNDetector, GATDetector, GraphSAGEXGBDetector,
           build_cc_graph, build_cc_graph_cosine,
           build_ieee_graph, build_ieee_graph_edges, normalize_adj
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

from detectors.tree_based import _fit_xgb

# ══════════════════════════════════════════════════════════════════════════════
# GATDetector  (indutivo, mini-batch, 2-layer GATv2 SOTA + Focal Loss)
# ══════════════════════════════════════════════════════════════════════════════

class GATDetector:
    """
    Detector indutivo em mini-lotes baseado em GATv2Conv (torch_geometric).

    Melhorias SOTA sobre a versão transdutiva anterior:
      - SOTAGATv2Model: conv2 com heads=1/concat=False + cabeça GELU
        → comprime a representação de forma mais suave antes da classificação
      - NeighborLoader (mini-batch): amostras sub-grafos ego-cêntricos por
        lote, eliminando o risco de OOM em grafos grandes e permitindo
        generalização indutiva verdadeira
      - PR-AUC como critério de early stopping → métrica mais informativa que
        ROC-AUC em datasets com extrema assimetria de classes (0.17% fraude)
      - Split explícito treino / val / teste (val_mask opcional)
      - Focal Loss + AdamW + CosineAnnealingLR + gradient clipping

    Interface
    ---------
    fit(X_all, y_all, train_mask, test_mask, edge_index, val_mask=None)
    score() → np.ndarray de probabilidades para os nós de teste

    edge_index : np.ndarray (2, E) int64  — formato PyG
    val_mask   : np.ndarray bool (N,)     — se None, 15% de train_mask é usado
    """
    name = "GNN (GATv2)"

    def __init__(self, hidden=128, heads=4, epochs=200, lr=1e-3, patience=25,
                 dropout=0.3, batch_size=1024, num_neighbors=None, eval_every=5,
                 seed=42):
        self.hidden        = hidden
        self.heads         = heads
        self.epochs        = epochs
        self.lr            = lr
        self.patience      = patience
        self.dropout       = dropout
        self.batch_size    = batch_size
        self.num_neighbors = num_neighbors or [5, 5]
        self.eval_every    = eval_every
        self.seed          = seed
        self._scores_all   = None
        self._te_mask      = None
        self.train_time    = 0.0

    def fit(self, X_all, y_all, train_mask, test_mask, edge_index,
            val_mask=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.metrics import average_precision_score
        from torch_geometric.nn import GATv2Conv

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        N, F_dim = X_all.shape

        # ── Máscara de validação ───────────────────────────────────────────
        if val_mask is None:
            tr_indices  = np.where(train_mask)[0]
            val_size    = max(1, int(0.15 * len(tr_indices)))
            rng         = np.random.default_rng(self.seed)
            val_indices = rng.choice(tr_indices, size=val_size, replace=False)
            val_mask    = np.zeros(N, dtype=bool)
            val_mask[val_indices] = True
            train_mask  = train_mask.copy()
            train_mask[val_indices] = False

        # ── Adicionar grau do nó como feature ─────────────────────────────
        degrees = np.bincount(edge_index[0], minlength=N).astype(np.float32)
        degrees = degrees / (degrees.max() + 1e-8)
        X_aug = np.hstack([X_all, degrees.reshape(-1, 1)])
        F_dim = X_aug.shape[1]

        print(f"    Nós: treino={train_mask.sum():,}  "
              f"val={val_mask.sum():,}  teste={test_mask.sum():,}")
        print(f"    Features: {F_dim}  |  hidden={self.hidden}  "
              f"heads={self.heads}  dropout={self.dropout}  "
              f"modo=full-graph")

        # ── Tensores no device ─────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_t   = torch.from_numpy(X_aug.astype(np.float32)).to(device)
        y_t   = torch.from_numpy(y_all.astype(np.float32)).to(device)
        ei_t  = torch.from_numpy(edge_index).long().to(device)
        tr_t  = torch.from_numpy(train_mask).to(device)
        val_t = torch.from_numpy(val_mask).to(device)

        # ── Arquitetura SOTA GATv2 ─────────────────────────────────────────
        class _SOTAGATv2(nn.Module):
            """
            conv1: heads cabeças concatenadas → expande representação
            conv2: 1 cabeça sem concatenação  → comprime para classificação
            classifier: Linear → GELU → Dropout → Linear
            """
            def __init__(self, in_ch, hid_ch, heads, dropout):
                super().__init__()
                self.drop = dropout
                self.conv1 = GATv2Conv(in_ch, hid_ch, heads=heads,
                                       concat=True, dropout=dropout,
                                       add_self_loops=True)
                self.bn1   = nn.BatchNorm1d(hid_ch * heads)
                self.conv2 = GATv2Conv(hid_ch * heads, hid_ch, heads=1,
                                       concat=False, dropout=dropout,
                                       add_self_loops=True)
                self.bn2   = nn.BatchNorm1d(hid_ch)
                self.classifier = nn.Sequential(
                    nn.Linear(hid_ch, hid_ch // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_ch // 2, 1),
                )

            def forward(self, x, edge_idx):
                x = F.dropout(x, p=self.drop, training=self.training)
                x = F.elu(self.bn1(self.conv1(x, edge_idx)))
                x = F.dropout(x, p=self.drop, training=self.training)
                x = F.elu(self.bn2(self.conv2(x, edge_idx)))
                return self.classifier(x).squeeze(-1)

        # ── Funções de perda ───────────────────────────────────────────────
        class _FocalLoss(nn.Module):
            def __init__(self, alpha=0.75, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                bce  = F.binary_cross_entropy_with_logits(inputs, targets,
                                                          reduction="none")
                pt   = torch.exp(-bce)
                loss = self.alpha * (1 - pt) ** self.gamma * bce
                return loss.mean()

        class _SoftFScoreLoss(nn.Module):
            """
            Aproximação contínua do F_β Score — diferenciável via Sigmoid.

            Substitui a função degrau de Heaviside por probabilidades contínuas,
            permitindo que o gradiente flua diretamente sobre TP, FP e FN soft:
              TP_soft = Σ(y_true · y_pred)
              FP_soft = Σ((1−y_true) · y_pred)
              FN_soft = Σ(y_true · (1−y_pred))

            β > 1 (padrão: 2) → penaliza FN mais que FP → maximiza Recall,
            ideal para fraudes onde deixar passar é pior que falso alarme.
            """
            def __init__(self, beta=2.0, epsilon=1e-7):
                super().__init__()
                self.beta2   = beta ** 2
                self.epsilon = epsilon

            def forward(self, inputs, targets):
                y_pred = torch.sigmoid(inputs)
                y_true = targets.float()
                tp = (y_true * y_pred).sum()
                fp = ((1 - y_true) * y_pred).sum()
                fn = (y_true * (1 - y_pred)).sum()
                num      = (1 + self.beta2) * tp
                den      = (1 + self.beta2) * tp + self.beta2 * fn + fp + self.epsilon
                f_beta   = num / den
                return 1.0 - f_beta

        class _HybridFraudLoss(nn.Module):
            """
            Perda híbrida: Focal Loss + Soft-F_β.

            O Focal Loss fornece estabilidade nos estágios iniciais do treino
            (empurra a massa de transações normais fáceis para perto de zero).
            O Soft-F_β lapida as fronteiras exatas no final da curva,
            maximizando diretamente a métrica alvo com gradientes diferenciáveis.

            f1_weight = 0.5 → peso igual entre os dois componentes.
            """
            def __init__(self, alpha=0.75, gamma=2.0, beta=2.0, f1_weight=0.5):
                super().__init__()
                self.focal   = _FocalLoss(alpha=alpha, gamma=gamma)
                self.soft_f1 = _SoftFScoreLoss(beta=beta)
                self.w       = f1_weight

            def forward(self, inputs, targets):
                return ((1 - self.w) * self.focal(inputs, targets)
                        + self.w * self.soft_f1(inputs, targets))

        model     = _SOTAGATv2(F_dim, self.hidden, self.heads,
                                self.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)
        # beta=2 → prioriza Recall (FN mais caro que FP em detecção de fraude)
        criterion = _HybridFraudLoss(alpha=0.75, gamma=2.0, beta=2.0,
                                     f1_weight=0.5)

        best_pr_auc    = -1.0
        best_state     = None
        patience_count = 0
        best_epoch     = 0

        header = (f"    {'Época':>6} | {'Loss Treino':>11} | "
                  f"{'PR-AUC Val':>10} | Status")
        sep    = "    " + "-" * (len(header) - 4)
        print(header)
        print(sep)

        def _move_to_cpu():
            nonlocal device, model, x_t, y_t, ei_t, tr_t, val_t
            print("    [OOM] GPU sem memória — movendo para CPU e retomando...")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            model  = model.cpu()
            x_t    = x_t.cpu()
            y_t    = y_t.cpu()
            ei_t   = ei_t.cpu()
            tr_t   = tr_t.cpu()
            val_t  = val_t.cpu()

        t0 = time.time()
        val_pr = 0.0
        oom_moved = False
        epoch = 0
        while epoch < self.epochs:
            try:
                # ── Passo de treino full-graph ─────────────────────────────
                model.train()
                optimizer.zero_grad()
                out  = model(x_t, ei_t)
                loss = criterion(out[tr_t], y_t[tr_t])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                avg_loss = loss.item()

                # ── Validação full-graph (a cada eval_every épocas) ───────
                if epoch % self.eval_every == 0 or epoch == self.epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        out_v = model(x_t, ei_t)
                        y_pred_v = torch.sigmoid(out_v[val_t]).cpu().numpy()
                        y_true_v = y_t[val_t].cpu().numpy()
                    try:
                        val_pr = average_precision_score(y_true_v, y_pred_v)
                    except Exception:
                        val_pr = 0.0

                status = ""
                if val_pr > best_pr_auc:
                    best_pr_auc    = val_pr
                    best_state     = {k: v.clone()
                                      for k, v in model.state_dict().items()}
                    patience_count = 0
                    best_epoch     = epoch
                    status = "✓ melhor"
                else:
                    patience_count += 1

                should_print = (epoch % 10 == 0 or epoch < 3
                                or patience_count >= self.patience)
                if should_print:
                    print(f"    {epoch:>6} | {avg_loss:>11.4f} | "
                          f"{val_pr:>10.4f} | {status}")
                epoch += 1

            except torch.cuda.OutOfMemoryError:
                if not oom_moved and device.type == "cuda":
                    _move_to_cpu()
                    oom_moved = True
                    # retry same epoch
                else:
                    raise

        print(f"    Treinamento concluído ({self.epochs} épocas).  "
              f"Melhor: época {best_epoch}, PR-AUC val={best_pr_auc:.4f}")

        self.train_time = time.time() - t0

        # ── Inferência final full-graph ────────────────────────────────────
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out_final  = model(x_t, ei_t)
            scores_all = torch.sigmoid(out_final).cpu().numpy()

        self._scores_all = scores_all
        self._te_mask    = test_mask
        return self

    def score(self):
        return self._scores_all[self._te_mask]


# ══════════════════════════════════════════════════════════════════════════════
# GATXGBDetector  (GATv2 feature extractor → XGBoost classifier)
# ══════════════════════════════════════════════════════════════════════════════

class GATXGBDetector:
    """
    Híbrido em dois estágios análogo ao GraphSAGEXGBDetector, mas usando
    GATv2Conv (torch_geometric) como extrator de embeddings estruturais.

    Estágio 1 — GATv2 supervisionado:
      Mesma arquitetura do GATDetector (conv1 multi-head + conv2 single-head
      + cabeça classificadora GELU), treinada com HybridFraudLoss
      (Focal + Soft-F_β). O embedding de cada nó é a saída de conv2
      (dimensão hidden), antes da cabeça classificadora.

    Estágio 2 — XGBoost aumentado:
      Os embeddings são concatenados às features tabulares originais e
      passados para um XGBoostClassifier para a decisão final.

    Vantagem em relação ao GATDetector puro: o XGBoost pode capturar
    interações não lineares entre features tabulares e embeddings de grafo
    sem depender de retropropagação pela atenção.

    Interface
    ---------
    fit(X_all, y_all, train_mask, test_mask, edge_index, Xg_tr, yg_tr,
        val_mask=None)
    score(Xg_te) → np.ndarray de probabilidades para os nós de teste

    edge_index : np.ndarray (2, E) int64  — formato PyG
    Xg_tr      : np.ndarray               — features tabulares de treino (XGB)
    yg_tr      : np.ndarray               — rótulos de treino (XGB)
    Xg_te      : np.ndarray               — features tabulares de teste (XGB)
    """
    name = "GAT+XGB"

    def __init__(self, hidden=128, heads=4, epochs=200, lr=1e-3,
                 dropout=0.3, n_estimators=300, eval_every=5, seed=42):
        self.hidden       = hidden
        self.heads        = heads
        self.epochs       = epochs
        self.lr           = lr
        self.dropout      = dropout
        self.n_estimators = n_estimators
        self.eval_every   = eval_every
        self.seed         = seed
        self.train_time   = 0.0
        self._xgb             = None
        self._embeddings_all  = None
        self._train_mask      = None
        self._test_mask       = None

    def fit(self, X_all, y_all, train_mask, test_mask, edge_index,
            Xg_tr, yg_tr, val_mask=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.metrics import average_precision_score
        from sklearn.model_selection import train_test_split
        from torch_geometric.nn import GATv2Conv

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        N, F_dim = X_all.shape

        # ── Máscara de validação ───────────────────────────────────────────
        if val_mask is None:
            tr_indices  = np.where(train_mask)[0]
            val_size    = max(1, int(0.15 * len(tr_indices)))
            rng         = np.random.default_rng(self.seed)
            val_indices = rng.choice(tr_indices, size=val_size, replace=False)
            val_mask    = np.zeros(N, dtype=bool)
            val_mask[val_indices] = True
            train_mask  = train_mask.copy()
            train_mask[val_indices] = False

        # ── Adicionar grau do nó como feature ─────────────────────────────
        degrees = np.bincount(edge_index[0], minlength=N).astype(np.float32)
        degrees = degrees / (degrees.max() + 1e-8)
        X_aug = np.hstack([X_all, degrees.reshape(-1, 1)])
        F_dim = X_aug.shape[1]

        print(f"    Nós: treino={train_mask.sum():,}  "
              f"val={val_mask.sum():,}  teste={test_mask.sum():,}")
        print(f"    Features GNN: {F_dim}  |  hidden={self.hidden}  "
              f"heads={self.heads}  dropout={self.dropout}")

        # ── Tensores ──────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_t   = torch.from_numpy(X_aug.astype(np.float32)).to(device)
        y_t   = torch.from_numpy(y_all.astype(np.float32)).to(device)
        ei_t  = torch.from_numpy(edge_index).long().to(device)
        tr_t  = torch.from_numpy(train_mask).to(device)
        val_t = torch.from_numpy(val_mask).to(device)

        # ── Arquitetura GATv2 com método embed() exposto ───────────────────
        class _GATv2Extractor(nn.Module):
            def __init__(self, in_ch, hid_ch, heads, dropout):
                super().__init__()
                self.drop = dropout
                self.conv1 = GATv2Conv(in_ch, hid_ch, heads=heads,
                                       concat=True, dropout=dropout,
                                       add_self_loops=True)
                self.bn1   = nn.BatchNorm1d(hid_ch * heads)
                self.conv2 = GATv2Conv(hid_ch * heads, hid_ch, heads=1,
                                       concat=False, dropout=dropout,
                                       add_self_loops=True)
                self.bn2   = nn.BatchNorm1d(hid_ch)
                self.classifier = nn.Sequential(
                    nn.Linear(hid_ch, hid_ch // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_ch // 2, 1),
                )

            def embed(self, x, edge_idx):
                """Retorna embedding pós-conv2 (antes do classificador)."""
                x = F.dropout(x, p=self.drop, training=self.training)
                x = F.elu(self.bn1(self.conv1(x, edge_idx)))
                x = F.dropout(x, p=self.drop, training=self.training)
                x = F.elu(self.bn2(self.conv2(x, edge_idx)))
                return x

            def forward(self, x, edge_idx):
                return self.classifier(self.embed(x, edge_idx)).squeeze(-1)

        # ── Funções de perda (idênticas ao GATDetector) ───────────────────
        class _FocalLoss(nn.Module):
            def __init__(self, alpha=0.75, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                bce  = F.binary_cross_entropy_with_logits(inputs, targets,
                                                          reduction="none")
                pt   = torch.exp(-bce)
                loss = self.alpha * (1 - pt) ** self.gamma * bce
                return loss.mean()

        class _SoftFScoreLoss(nn.Module):
            def __init__(self, beta=2.0, epsilon=1e-7):
                super().__init__()
                self.beta2   = beta ** 2
                self.epsilon = epsilon

            def forward(self, inputs, targets):
                y_pred = torch.sigmoid(inputs)
                y_true = targets.float()
                tp = (y_true * y_pred).sum()
                fp = ((1 - y_true) * y_pred).sum()
                fn = (y_true * (1 - y_pred)).sum()
                num    = (1 + self.beta2) * tp
                den    = (1 + self.beta2) * tp + self.beta2 * fn + fp + self.epsilon
                return 1.0 - num / den

        class _HybridFraudLoss(nn.Module):
            def __init__(self, alpha=0.75, gamma=2.0, beta=2.0, f1_weight=0.5):
                super().__init__()
                self.focal   = _FocalLoss(alpha=alpha, gamma=gamma)
                self.soft_f1 = _SoftFScoreLoss(beta=beta)
                self.w       = f1_weight

            def forward(self, inputs, targets):
                return ((1 - self.w) * self.focal(inputs, targets)
                        + self.w * self.soft_f1(inputs, targets))

        model     = _GATv2Extractor(F_dim, self.hidden, self.heads,
                                    self.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)
        criterion = _HybridFraudLoss(alpha=0.75, gamma=2.0, beta=2.0,
                                     f1_weight=0.5)

        best_pr_auc    = -1.0
        best_state     = None
        best_epoch     = 0

        header = (f"    {'Época':>6} | {'Loss Treino':>11} | "
                  f"{'PR-AUC Val':>10} | Status")
        sep    = "    " + "-" * (len(header) - 4)
        print(header)
        print(sep)

        def _move_to_cpu():
            nonlocal device, model, x_t, y_t, ei_t, tr_t, val_t
            print("    [OOM] GPU sem memória — movendo para CPU e retomando...")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            model  = model.cpu()
            x_t    = x_t.cpu()
            y_t    = y_t.cpu()
            ei_t   = ei_t.cpu()
            tr_t   = tr_t.cpu()
            val_t  = val_t.cpu()

        t0 = time.time()
        val_pr = 0.0
        oom_moved = False
        epoch = 0
        while epoch < self.epochs:
            try:
                model.train()
                optimizer.zero_grad()
                out  = model(x_t, ei_t)
                loss = criterion(out[tr_t], y_t[tr_t])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                avg_loss = loss.item()

                # ── Validação a cada eval_every épocas ────────────────────
                if epoch % self.eval_every == 0 or epoch == self.epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        out_v    = model(x_t, ei_t)
                        y_pred_v = torch.sigmoid(out_v[val_t]).cpu().numpy()
                        y_true_v = y_t[val_t].cpu().numpy()
                    try:
                        val_pr = average_precision_score(y_true_v, y_pred_v)
                    except Exception:
                        val_pr = 0.0

                status = ""
                if val_pr > best_pr_auc:
                    best_pr_auc = val_pr
                    best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                    best_epoch  = epoch
                    status      = "✓ melhor"

                if epoch % 10 == 0 or epoch < 3:
                    print(f"    {epoch:>6} | {avg_loss:>11.4f} | "
                          f"{val_pr:>10.4f} | {status}")
                epoch += 1

            except torch.cuda.OutOfMemoryError:
                if not oom_moved and device.type == "cuda":
                    _move_to_cpu()
                    oom_moved = True
                    # retry same epoch
                else:
                    raise

        print(f"    Treinamento concluído ({self.epochs} épocas).  "
              f"Melhor: época {best_epoch}, PR-AUC val={best_pr_auc:.4f}")

        gat_time = time.time() - t0

        # ── Extração de embeddings para todos os nós ──────────────────────
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits_final         = model(x_t, ei_t)
            self._embeddings_all = model.embed(x_t, ei_t).cpu().numpy()  # (N, hidden)

        # Diagnóstico: PR-AUC do GAT sozinho no conjunto de teste
        from sklearn.metrics import average_precision_score as aps
        te_t = torch.from_numpy(test_mask).to(device)
        gat_te_scores = torch.sigmoid(logits_final[te_t]).cpu().numpy()
        try:
            gat_te_pr = aps(y_all[test_mask], gat_te_scores)
            print(f"    GAT sozinho → PR-AUC teste = {gat_te_pr:.4f}  "
                  f"(sinal estrutural antes do XGBoost)")
        except Exception:
            pass

        self._train_mask = train_mask
        self._test_mask  = test_mask

        # ── Augmentação e treino do XGBoost ──────────────────────────────
        tr_emb    = self._embeddings_all[train_mask]   # (n_train, hidden)
        Xg_tr_aug = np.hstack([Xg_tr, tr_emb])

        n_cols_tab = Xg_tr.shape[1]
        print(f"\n    [Embeddings GAT] shape={tr_emb.shape}  "
              f"GAT time={gat_time:.1f}s")
        print(f"    Features: {n_cols_tab} tabulares + {self.hidden} emb "
              f"= {Xg_tr_aug.shape[1]} total")

        n_pos_xgb = int((yg_tr == 1).sum())
        n_neg_xgb = int((yg_tr == 0).sum())
        spw = n_neg_xgb / max(n_pos_xgb, 1)

        X_t_xgb, X_val_xgb, y_t_xgb, y_val_xgb = train_test_split(
            Xg_tr_aug, yg_tr, test_size=0.15, stratify=yg_tr,
            random_state=self.seed,
        )

        model_kwargs = dict(
            n_estimators=self.n_estimators,
            scale_pos_weight=spw,
            tree_method="hist",
            early_stopping_rounds=20,
            random_state=self.seed,
            verbosity=0,
        )

        t1 = time.time()
        self._xgb = _fit_xgb(model_kwargs, X_t_xgb, y_t_xgb, X_val_xgb, y_val_xgb)
        xgb_time = time.time() - t1
        print(f"    XGBoost time={xgb_time:.1f}s")

        self.train_time = time.time() - t0
        return self

    def score(self, Xg_te):
        """Concatena embeddings de teste com features tabulares e classifica."""
        te_emb    = self._embeddings_all[self._test_mask]
        Xg_te_aug = np.hstack([Xg_te, te_emb])
        return self._xgb.predict_proba(Xg_te_aug)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# GNNDetector  (transductive, 3-layer GCN)
# ══════════════════════════════════════════════════════════════════════════════

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
                 patience=25, dropout=0.3, n_estimators=300, seed=42):
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

        model = _SAGE(F, self.hidden, self.emb_dim, self.dropout)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=8, factor=0.5, min_lr=1e-5
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_auc   = -1.0
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
            # Early stopping baseado em val_auc (otimiza diretamente a métrica)
            improved = not np.isnan(val_auc) and val_auc > best_val_auc
            if improved:
                best_val_auc   = val_auc
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

        print(f"    SAGE concluído ({self.epochs} épocas).  "
              f"Melhor: época {best_epoch}, val_auc={best_val_auc:.4f}")

        sage_time = time.time() - t0

        # ── Extração de embeddings para todos os nós ──────────────────────────
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits_final = model(A, X_t)
            self._embeddings_all = model.embed(A, X_t).numpy()  # (N, emb_dim)

        # Diagnóstico: AUC do SAGE sozinho no conjunto de teste
        te_mask_np = test_mask
        sage_te_scores = torch.sigmoid(logits_final[torch.from_numpy(te_mask_np)]).numpy()
        try:
            sage_te_auc = roc_auc_score(y_all[te_mask_np], sage_te_scores)
            print(f"    SAGE sozinho → ROC-AUC teste = {sage_te_auc:.4f}  "
                  f"(se baixo, o grafo não adiciona sinal estrutural)")
        except Exception:
            pass

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

        model_kwargs = dict(
            n_estimators=self.n_estimators,
            scale_pos_weight=spw,
            tree_method="hist",
            early_stopping_rounds=20,
            random_state=self.seed,
            verbosity=0,
        )

        t1 = time.time()
        self._xgb = _fit_xgb(model_kwargs, X_t_xgb, y_t_xgb, X_val_xgb, y_val_xgb)
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
# Graph helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_cc_graph(X, k=10):
    """
    Constrói grafo k-NN para o dataset CC.
    - Dá peso 3× à feature Amount (última coluna).
    - Usa as features diretamente (sem PCA extra — V1-V28 já são PCA da fonte).
    - k=10 vizinhos: balance entre qualidade e esparsidade do grafo.
    Retorna scipy.sparse.coo_matrix (N×N).
    """
    N = X.shape[0]

    # Peso extra para Amount (última coluna)
    X_weighted = X.copy().astype(np.float32)
    X_weighted[:, -1] *= 3.0

    print(f"    kNN: {N:,} nós, k={k}, {X_weighted.shape[1]}D — buscando vizinhos…")
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn_model.fit(X_weighted)
    _, indices = nn_model.kneighbors(X_weighted)

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
    Retorna scipy.sparse matrix (N×N).
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
        ("card2",         50),             # mesmo CVV/bin range
        ("addr1",         50),             # mesmo CEP de cobrança
        ("P_emaildomain", 30),             # mesmo domínio de e-mail do comprador
        ("R_emaildomain", 30),             # mesmo domínio de e-mail do destinatário
    ]
    for col, lim in relacoes:
        n = _add_relation(col, lim)
        print(f"      {col:<15}: {n:,} arestas")

    if not rows:
        print(f"    Grafo IEEE: {N:,} nós | sem arestas")
        return sp.coo_matrix((N, N), dtype=np.float32)

    data = np.ones(len(rows), dtype=np.float32)
    # Deduplica e binariza; keep as CSR — normalize_adj handles the rest
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    adj.data[:] = 1.0
    print(f"    Grafo IEEE: {N:,} nós | {adj.nnz // 2:,} arestas | "
          f"grau médio={adj.nnz / N:.1f}")
    return adj


def build_cc_graph_cosine(X_pca, times=None, k=5, cos_dist_threshold=0.10):
    """
    Grafo híbrido K-NN Cosseno com filtragem causal temporal para o dataset CC.

    Dois filtros aplicados em sequência:
      1. Filtro de densidade (k-NN): busca k vizinhos por distância cosseno
         (métrica nativa do sklearn), impondo grau máximo = k por nó.
         Evita a explosão de 12M+ arestas do limiar de cosseno puro e a
         saturação do Softmax da atenção (cauda longa de ruído).
      2. Filtro causal: a aresta j→i só é criada se times[j] ≤ times[i].
         Isso torna o grafo temporalmente direcionado (passado → presente),
         eliminando data leakage e obrigando a GATv2 a generalizar padrões.

    X_pca deve conter apenas as componentes PCA (V1-V28), garantindo
    ortogonalidade topológica: o grafo codifica similaridade de padrão de
    fraude, não proximidade temporal ou de valor monetário.

    Parâmetros
    ----------
    X_pca            : (N, 28) — componentes PCA escaladas
    times            : (N,)    — segundos absolutos (None desativa filtro causal)
    k                : int     — vizinhos candidatos por nó (recomendado: 3–5)
    cos_dist_threshold: float  — distância cosseno máxima (= 1 − cos_similarity)
                                 0.10 → cos_similarity > 0.90

    Retorna edge_index (2, E) int64 — compatível com PyG/GATDetector.
    """
    N = X_pca.shape[0]
    causal_tag = " causal" if times is not None else ""
    print(f"    K-NN cosine{causal_tag}: {N:,} nós, k={k}, "
          f"cos_dist<{cos_dist_threshold} — construindo grafo…")

    # ── k-NN via FAISS (inner product em vetores normalizados = cosseno) ────────
    import faiss
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_norm = (X_pca / norms).astype(np.float32)

    index = faiss.IndexFlatIP(X_norm.shape[1])  # produto interno = cosseno em vecs unitários
    index.add(X_norm)
    cos_sims, indices = index.search(X_norm, k + 1)  # (N, k+1), similaridade cosseno
    distances = 1.0 - cos_sims                        # converte para distância cosseno

    # ── filtro vetorizado (evita loop Python sobre N×k iterações) ─────────────
    j_idx = indices[:, 1:].ravel()         # nós fonte  (N*k,)
    i_idx = np.repeat(np.arange(N), k)     # nós destino (N*k,)
    dists  = distances[:, 1:].ravel()      # distâncias  (N*k,)

    mask = dists < cos_dist_threshold
    if times is not None:
        mask &= times[j_idx] <= times[i_idx]

    rows = j_idx[mask]
    cols = i_idx[mask]

    if rows.size == 0:
        print(f"    Grafo CC k-NN cosine: {N:,} nós | 0 arestas "
              f"(threshold ou filtro causal muito restritivo)")
        return np.empty((2, 0), dtype=np.int64)

    edge_index = np.unique(
        np.vstack([rows, cols]).astype(np.int64), axis=1
    )
    print(f"    Grafo CC k-NN cosine: {N:,} nós | {edge_index.shape[1]:,} arestas"
          f" | grau médio={edge_index.shape[1] / N:.1f}")
    return edge_index


def build_ieee_graph_edges(df, max_per_card=100, limit_edges=True, max_degree=25):
    """
    Grafo multi-relacional para o dataset IEEE que retorna edge_index (2, E)
    no formato PyG — compatível com GATDetector.

    Conecta transações que compartilham identificadores categóricos reais
    (card1, card2, addr1, P_emaildomain, R_emaildomain), criando ligações
    semânticas válidas e mantendo isoladas instâncias não relacionadas.

    limit_edges : bool (default True)
        Se True, aplica um cap de grau máximo por nó (max_degree) após
        construir todas as arestas, selecionando aleatoriamente quais manter.
        Isso evita OOM em GPUs ao reduzir os tensores de atenção do GATv2
        (custo ∝ E × hidden × heads). Desative para reproduzir o grafo completo.
    max_degree : int (default 10)
        Grau de saída máximo por nó quando limit_edges=True.
        Com 590 k nós e max_degree=10 o grafo terá ≤ 5,9 M arestas
        (~3 GB para os tensores conv2 do GATv2) contra ~13 GB no grafo completo.
    """
    N        = len(df)
    df_reset = df.reset_index(drop=True)
    rows, cols = [], []

    def _add_relation(col, max_per_group):
        if col not in df_reset.columns:
            return 0
        n_edges = 0
        groups  = df_reset.dropna(subset=[col]).groupby(col, sort=False).indices
        for _, idxs in groups.items():
            idxs = list(idxs)
            if len(idxs) < 2:
                continue
            idxs = idxs[:max_per_group]
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    rows.append(idxs[a]); cols.append(idxs[b])
                    rows.append(idxs[b]); cols.append(idxs[a])
                    n_edges += 1
        return n_edges

    relacoes = [
        ("card1",         max_per_card),
        ("card2",         50),
        ("addr1",         50),
        ("P_emaildomain", 30),
        ("R_emaildomain", 30),
    ]
    for col, lim in relacoes:
        n = _add_relation(col, lim)
        print(f"      {col:<15}: {n:,} arestas")

    if not rows:
        print(f"    Grafo IEEE edges: {N:,} nós | sem arestas")
        return np.empty((2, 0), dtype=np.int64)

    if limit_edges:
        # Cap de grau máximo por nó (embaralhamento aleatório para seleção imparcial)
        rng = np.random.default_rng(42)
        src = np.array(rows, dtype=np.int64)
        dst = np.array(cols, dtype=np.int64)
        perm = rng.permutation(len(src))
        src, dst = src[perm], dst[perm]
        order = np.argsort(src, kind="stable")
        src, dst = src[order], dst[order]
        group_start = np.concatenate([[0], np.where(np.diff(src))[0] + 1])
        group_size  = np.diff(np.concatenate([group_start, [len(src)]]))
        rank = np.arange(len(src)) - np.repeat(group_start, group_size)
        mask = rank < max_degree
        rows = src[mask].tolist()
        cols = dst[mask].tolist()
        print(f"    limit_edges=True  max_degree={max_degree}  "
              f"→ {len(rows):,} arestas após cap de grau")

    edge_index = np.unique(np.vstack([rows, cols]).astype(np.int64), axis=1)
    print(f"    Grafo IEEE edges: {N:,} nós | {edge_index.shape[1]:,} arestas | "
          f"grau médio={edge_index.shape[1] / N:.1f}")
    return edge_index


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
