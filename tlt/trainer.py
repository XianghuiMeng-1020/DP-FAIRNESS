"""NP and DP trainers for logistic and MLP-small. Opacus 1.x accounting."""
from __future__ import annotations

import platform
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LogisticNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 2)

    def forward(self, x):
        return self.linear(x)


class MLPSmall(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class TLTTrainer:
    def __init__(self, architecture: str, seed: int, device: Optional[str] = None):
        self.architecture = architecture
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[nn.Module] = None
        set_seeds(seed)

    def _build(self, n_features: int) -> nn.Module:
        if self.architecture in {"logistic", "LR", "lr"}:
            return LogisticNet(n_features)
        if self.architecture in {"mlp_small", "MLP", "mlp"}:
            return MLPSmall(n_features, hidden=64)
        raise ValueError(f"Unknown architecture: {self.architecture}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return torch.softmax(self.model(xt), dim=1).detach().cpu().numpy()

    def compute_sample_losses(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
            logp = torch.log_softmax(self.model(xt), dim=1)
            return (-logp[range(len(y)), yt]).detach().cpu().numpy()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        mechanism: str = "none",
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        max_grad_norm: float = 1.0,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
    ) -> Dict[str, Any]:
        set_seeds(self.seed)
        n_train = len(X_train)
        self.model = self._build(X_train.shape[1]).to(self.device)
        if mechanism == "DP-SGD":
            self.model = ModuleValidator.fix(self.model).to(self.device)

        is_dp = mechanism == "DP-SGD"
        bs = batch_size or (min(256, n_train) if is_dp else 64)
        n_epochs = epochs or (30 if is_dp else 50)
        actual_delta = float(delta) if delta is not None else 1.0 / max(n_train, 1)
        sample_rate = bs / n_train

        ds = TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
        )
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        loader = DataLoader(ds, batch_size=bs, shuffle=True, generator=gen)
        opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.CrossEntropyLoss()

        privacy_engine = None
        noise_multiplier = None
        realized_eps = None
        if is_dp:
            if epsilon is None:
                raise ValueError("DP-SGD requires target epsilon")
            self.model.train()
            privacy_engine = PrivacyEngine()
            self.model, opt, loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=opt,
                data_loader=loader,
                epochs=n_epochs,
                target_epsilon=float(epsilon),
                target_delta=actual_delta,
                max_grad_norm=max_grad_norm,
            )
            noise_multiplier = getattr(opt, "noise_multiplier", None)

        use_es = bool(early_stopping and (not is_dp) and X_val is not None and y_val is not None)
        best_state = None
        best_val = float("inf")
        patience = 0
        early_stopped = False
        best_epoch = 0
        steps = 0
        epochs_completed = 0
        t0 = time.time()
        self.model.train()
        for epoch in range(n_epochs):
            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(bx), by)
                loss.backward()
                opt.step()
                steps += 1
            epochs_completed = epoch + 1
            if is_dp and privacy_engine is not None:
                try:
                    realized_eps = float(privacy_engine.get_epsilon(actual_delta))
                except Exception:
                    realized_eps = float(epsilon)
            if use_es:
                self.model.eval()
                with torch.no_grad():
                    xv = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
                    yv = torch.as_tensor(y_val, dtype=torch.long, device=self.device)
                    vloss = float(crit(self.model(xv), yv).item())
                self.model.train()
                if vloss < best_val:
                    best_val = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                    best_epoch = epochs_completed
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        early_stopped = True
                        break

        if use_es and best_state is not None:
            self.model.load_state_dict(best_state)

        proba_tr = self.predict_proba(X_train)[:, 1]
        train_auc = float(roc_auc_score(y_train, proba_tr)) if len(np.unique(y_train)) > 1 else float("nan")
        val_auc = None
        if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
            val_auc = float(roc_auc_score(y_val, self.predict_proba(X_val)[:, 1]))

        try:
            import opacus
            opacus_version = opacus.__version__
        except Exception:
            opacus_version = "unknown"

        return {
            "architecture": self.architecture,
            "mechanism": mechanism,
            "train_auc": train_auc,
            "val_auc": val_auc,
            "target_epsilon": float(epsilon) if epsilon is not None else None,
            "realized_epsilon": realized_eps if is_dp else None,
            "delta": actual_delta if is_dp else None,
            "clipping_norm": max_grad_norm if is_dp else None,
            "noise_multiplier": float(noise_multiplier) if noise_multiplier is not None else None,
            "sample_rate": sample_rate,
            "optimizer_steps": steps,
            "epoch_count": epochs_completed,
            "epochs_planned": n_epochs,
            "early_stopping": use_es,
            "early_stopped": early_stopped,
            "best_epoch": best_epoch if use_es else epochs_completed,
            "batch_size": bs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "seed": self.seed,
            "device": str(self.device),
            "opacus_version": opacus_version,
            "torch_version": torch.__version__,
            "dp_accountant": "RDP via Opacus PrivacyEngine.make_private_with_epsilon" if is_dp else "N/A",
            "dp_sampling": "poisson" if is_dp else "shuffle",
            "train_n": n_train,
            "train_wall_sec": time.time() - t0,
            "hardware": platform.processor() or platform.machine(),
        }
