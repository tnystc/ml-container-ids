"""
Reptile: first-order meta-learning, the MAML simplification.

Reference:
    Nichol et al., "On First-Order Meta-Learning Algorithms", arXiv 2018.

Idea:
    For each meta-iteration:
        1. Sample an N-way K-shot task from the training data.
        2. Clone the encoder + a fresh linear head; run `inner_steps` of SGD
           on the task support set (cross-entropy).
        3. Move the master encoder weights toward the adapted weights:
              theta <- theta + epsilon * (theta_adapted - theta)

    The result is an encoder init that adapts quickly to new few-shot tasks.

Deployment for our 9-way 80-shot problem:
    1. Meta-train encoder via Reptile on random N-way K-shot episodes.
    2. Encode the full 720-row training set with the meta-learned encoder.
    3. Fit sklearn LogisticRegression on the embeddings as the classifier head.

Used in IDS literature, e.g. Wang et al. "Meta-IDS" (2021), Liang et al.
"Few-shot DDoS detection via MAML" (2022). Reptile (the first-order variant)
matches MAML accuracy on most few-shot benchmarks at a fraction of the cost.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.optim import SGD
from tqdm import trange

from src.data.preprocessing import EpisodeSampler


class _Encoder(nn.Module):
    def __init__(self, n_features: int, embedding_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Reptile:
    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 64,
        hidden: int = 128,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 10,
        inner_steps: int = 5,
        inner_lr: float = 1e-2,
        meta_step_size: float = 0.1,
        epochs: int = 4,
        epoch_size: int = 2000,
        head_C: float = 1.0,
        device: str = None,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.meta_step_size = meta_step_size
        self.epochs = epochs
        self.epoch_size = epoch_size
        self.head_C = head_C
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.random_state = random_state

        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        self.encoder = _Encoder(n_features, embedding_dim, hidden).to(self.device)
        self.head: LogisticRegression = None
        self.classes_: np.ndarray = None

    def _inner_adapt(self, sx: np.ndarray, sy: np.ndarray, qx: np.ndarray, qy: np.ndarray):
        """Clone encoder + fresh head, run inner_steps of SGD on the task."""
        n_way = int(np.unique(sy).size)
        encoder_clone = copy.deepcopy(self.encoder)
        head = nn.Linear(self.embedding_dim, n_way).to(self.device)
        params = list(encoder_clone.parameters()) + list(head.parameters())
        opt = SGD(params, lr=self.inner_lr)

        # Combine support+query for inner loop training (small support sets
        # under-train BatchNorm; using both is standard in Reptile-for-few-shot).
        x = np.concatenate([sx, qx], axis=0)
        y = np.concatenate([sy, qy], axis=0)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

        encoder_clone.train()
        head.train()
        for _ in range(self.inner_steps):
            logits = head(encoder_clone(x_t))
            loss = F.cross_entropy(logits, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Final loss/acc for logging
        with torch.no_grad():
            qx_t = torch.tensor(qx, dtype=torch.float32, device=self.device)
            qy_t = torch.tensor(qy, dtype=torch.long, device=self.device)
            qlogits = head(encoder_clone(qx_t))
            qloss = F.cross_entropy(qlogits, qy_t).item()
            qacc = (qlogits.argmax(dim=1) == qy_t).float().mean().item()

        return encoder_clone, qloss, qacc

    def _reptile_update(self, encoder_adapted: nn.Module):
        """theta <- theta + epsilon * (theta_adapted - theta) over encoder params."""
        with torch.no_grad():
            for p, p_a in zip(self.encoder.parameters(), encoder_adapted.parameters()):
                p.add_(self.meta_step_size * (p_a.data - p.data))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        sampler = EpisodeSampler(X, y, self.n_way, self.k_shot, self.n_query)
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_acc = 0.0
            pbar = trange(self.epoch_size,
                          desc=f"Reptile {epoch+1}/{self.epochs}",
                          leave=False)
            for _ in pbar:
                sx, sy, qx, qy = sampler.sample_episode(rng)
                adapted, qloss, qacc = self._inner_adapt(sx, sy, qx, qy)
                self._reptile_update(adapted)
                total_loss += qloss
                total_acc += qacc
                pbar.set_postfix(loss=f"{qloss:.4f}", acc=f"{qacc:.3f}")

            print(f"Reptile epoch {epoch+1}/{self.epochs} — "
                  f"query loss: {total_loss/self.epoch_size:.4f}  "
                  f"query acc: {total_acc/self.epoch_size:.3f}")

        # Deployment: encode the full training set, fit a logistic head.
        emb = self._encode(X)
        self.head = LogisticRegression(
            max_iter=1000, C=self.head_C, random_state=self.random_state,
        ).fit(emb, y)
        return self

    def _encode(self, X: np.ndarray, batch: int = 4096) -> np.ndarray:
        self.encoder.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                X_t = torch.tensor(X[i:i + batch], dtype=torch.float32, device=self.device)
                out.append(self.encoder(X_t).cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("Call fit() before predict_proba()")
        return self.head.predict_proba(self._encode(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
