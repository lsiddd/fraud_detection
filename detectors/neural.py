"""
neural.py — AutoencoderDetector, TabNetDetector
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np


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
