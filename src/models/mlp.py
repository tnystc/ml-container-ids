"""
Simple supervised MLP classifier — the missing vanilla neural-net baseline.

No metric learning, no meta-learning, no contrastive loss, no self-supervised
pretraining. Just a feed-forward MLP trained with standard cross-entropy on
the labeled few-shot set.

Architecture: input → 256 → 128 → 64 → n_classes, with BatchNorm + ReLU +
Dropout between layers. Adam optimizer, weight decay for regularization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class _MLPNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int,
                 hidden=(256, 128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP:
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden=(256, 128, 64),
        dropout: float = 0.3,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = None,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.random_state = random_state

        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        self.net = _MLPNet(n_features, n_classes, hidden, dropout).to(self.device)
        self.classes_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = Adam(self.net.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)

        self.net.train()
        pbar = trange(self.epochs, desc=f"MLP {self.epochs} epochs", leave=False)
        for _ in pbar:
            total_loss = 0.0
            total_acc = 0.0
            n_batches = 0
            for xb, yb in loader:
                logits = self.net(xb)
                loss = F.cross_entropy(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_acc += (logits.argmax(1) == yb).float().mean().item()
                n_batches += 1
            pbar.set_postfix(
                loss=f"{total_loss/n_batches:.4f}",
                acc=f"{total_acc/n_batches:.3f}",
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            batch = 4096
            for i in range(0, len(X), batch):
                X_t = torch.tensor(X[i:i + batch], dtype=torch.float32, device=self.device)
                logits = self.net(X_t)
                out.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
