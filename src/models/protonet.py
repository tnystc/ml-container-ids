"""
Prototypical Network for network flow classification.

Encoders:
  - "cnn": 1D CNN (report baseline). Conv1d over features as if they were a signal.
  - "mlp": MLP with residual blocks. Better inductive bias for tabular flow features.

Distance metrics:
  - "euclidean": squared L2 distance (report baseline)
  - "cosine":    negative scaled cosine similarity (scale=10), scale-invariant

Training: episodic (N-way K-shot).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import trange

from src.data.preprocessing import EpisodeSampler


class CNNEncoder(nn.Module):
    """Report baseline: 1D CNN over features."""

    def __init__(self, n_features: int, embedding_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class MLPEncoder(nn.Module):
    """MLP encoder for tabular flow features. Dropout + BN for regularization."""

    def __init__(self, n_features: int, embedding_dim: int = 64, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _distances(query_emb: torch.Tensor, prototypes: torch.Tensor, metric: str) -> torch.Tensor:
    """
    Return 'logits' suitable for softmax (higher = more similar).
    For euclidean: negative squared L2 distance.
    For cosine:    scaled cosine similarity.
    """
    if metric == "euclidean":
        d = torch.cdist(query_emb, prototypes).pow(2)
        return -d
    elif metric == "cosine":
        q = F.normalize(query_emb, dim=-1)
        p = F.normalize(prototypes, dim=-1)
        return 10.0 * (q @ p.t())   # temperature=10 is standard for cosine softmax
    else:
        raise ValueError(f"Unknown metric: {metric}")


class ProtoNet:
    """Prototypical Network with sklearn-style interface."""

    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 64,
        encoder: str = "cnn",            # "cnn" | "mlp"
        distance: str = "euclidean",     # "euclidean" | "cosine"
        n_way: int = 8,
        k_shot: int = 5,
        n_query: int = 10,
        epochs: int = 4,
        epoch_size: int = 2000,
        lr: float = 1e-3,
        device: str = None,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder
        self.distance = distance
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.epochs = epochs
        self.epoch_size = epoch_size
        self.lr = lr
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.random_state = random_state

        # Deterministic weight init
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        if encoder == "cnn":
            self.encoder = CNNEncoder(n_features, embedding_dim).to(self.device)
        elif encoder == "mlp":
            self.encoder = MLPEncoder(n_features, embedding_dim).to(self.device)
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        self.prototypes: torch.Tensor = None
        self.classes_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        sampler = EpisodeSampler(X, y, self.n_way, self.k_shot, self.n_query)
        optimizer = Adam(self.encoder.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.random_state)

        self.encoder.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_acc = 0.0
            pbar = trange(self.epoch_size, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for _ in pbar:
                sx, sy, qx, qy = sampler.sample_episode(rng)
                loss, acc = self._episode_step(sx, sy, qx, qy, optimizer)
                total_loss += loss
                total_acc += acc
                pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}")

            print(
                f"Epoch {epoch+1}/{self.epochs} — "
                f"loss: {total_loss/self.epoch_size:.4f}  "
                f"acc: {total_acc/self.epoch_size:.3f}"
            )

        self._compute_prototypes(X, y)
        return self

    def _episode_step(self, sx, sy, qx, qy, optimizer):
        sx_t = torch.tensor(sx, dtype=torch.float32, device=self.device)
        qx_t = torch.tensor(qx, dtype=torch.float32, device=self.device)
        qy_t = torch.tensor(qy, dtype=torch.long, device=self.device)

        s_emb = self.encoder(sx_t)
        q_emb = self.encoder(qx_t)

        n_way = len(np.unique(sy))
        proto = s_emb.view(n_way, self.k_shot, -1).mean(dim=1)

        logits = _distances(q_emb, proto, self.distance)
        log_probs = F.log_softmax(logits, dim=1)

        loss = F.nll_loss(log_probs, qy_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (log_probs.argmax(dim=1) == qy_t).float().mean().item()
        return loss.item(), acc

    def _compute_prototypes(self, X: np.ndarray, y: np.ndarray):
        self.encoder.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            embeddings = self.encoder(X_t)

        protos = []
        for cls in self.classes_:
            mask = torch.tensor(y == cls, device=self.device)
            protos.append(embeddings[mask].mean(dim=0))
        self.prototypes = torch.stack(protos)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.prototypes is None:
            raise RuntimeError("Call fit() before predict_proba()")
        self.encoder.eval()
        # Batch inference to avoid OOM on large test sets
        batch = 4096
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                X_t = torch.tensor(X[i : i + batch], dtype=torch.float32, device=self.device)
                emb = self.encoder(X_t)
                logits = _distances(emb, self.prototypes, self.distance)
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]
