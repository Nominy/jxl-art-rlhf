"""Image features (CLIP ViT-B/32 + cheap statistics) and a Bradley-Terry reward model.

The reward model is linear on standardized features, fit with L2-regularised logistic
loss on human pairwise preferences: P(a > b) = sigmoid(w . (f_a - f_b)). Ties are
encoded as half-weighted comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..compiler import ROOT

CLIP_DIR = ROOT / "models" / "clip-vit-base-patch32"
STAT_KEYS = ["std", "color_std", "edge", "compressibility"]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Featurizer:
    def __init__(self, device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or default_device()
        self.model = CLIPModel.from_pretrained(str(CLIP_DIR)).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(str(CLIP_DIR))

    @torch.no_grad()
    def embed(self, images: list[Image.Image], batch: int = 64) -> np.ndarray:
        """L2-normalised CLIP image embeddings, (n, 512)."""
        out = []
        for k in range(0, len(images), batch):
            inp = self.proc(images=images[k:k + batch], return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(**inp)
            if not isinstance(emb, torch.Tensor):  # newer transformers return a model output
                emb = emb.pooler_output
            out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def __call__(self, thumbs: list[Image.Image], stats: list[dict]) -> np.ndarray:
        if not thumbs:
            return np.zeros((0, 512 + len(STAT_KEYS) + 1), np.float32)
        emb = self.embed(thumbs)
        st = np.array([[s[k] for k in STAT_KEYS] + [np.log1p(s["unique"])] for s in stats], np.float32)
        return np.concatenate([emb, st], 1).astype(np.float32)


class BTReward:
    def __init__(self, n_models: int = 8, l2: float = 1.0):
        self.n_models, self.l2 = n_models, l2
        self.mu = self.sd = None
        self.W: np.ndarray | None = None  # (n_models, d)

    def fit(self, X: np.ndarray, pairs: list[tuple[int, int, float]], seed: int = 0) -> dict:
        """pairs: (i, j, y) with y=1 if i preferred, 0 if j preferred, 0.5 for tie."""
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        rng = np.random.default_rng(seed)
        P = np.array(pairs, dtype=np.float64)
        Ws = []
        for m in range(self.n_models):
            idx = rng.integers(0, len(P), len(P)) if m else np.arange(len(P))
            Ws.append(self._fit_one(Z, P[idx]))
        self.W = np.stack(Ws)
        acc = self._accuracy(Z, P)
        return {"pairs": len(P), "train_acc": acc}

    def _fit_one(self, Z, P) -> np.ndarray:
        D = torch.tensor(Z[P[:, 0].astype(int)] - Z[P[:, 1].astype(int)], dtype=torch.float64)
        y = torch.tensor(P[:, 2], dtype=torch.float64)
        w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logit = D @ w
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="sum")
            loss = loss + 0.5 * self.l2 * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return w.detach().numpy()

    def _accuracy(self, Z, P) -> float | None:
        strict = P[P[:, 2] != 0.5]
        if not len(strict):
            return None
        s = Z @ self.W[0]
        pred = s[strict[:, 0].astype(int)] > s[strict[:, 1].astype(int)]
        return float(np.mean(pred == (strict[:, 2] == 1)))

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of reward across the bootstrap ensemble."""
        Z = (X - self.mu) / self.sd
        S = Z @ self.W.T
        return S.mean(1), S.std(1)

    def save(self, path: Path) -> None:
        np.savez(path, mu=self.mu, sd=self.sd, W=self.W, l2=self.l2)

    @classmethod
    def load(cls, path: Path) -> "BTReward":
        d = np.load(path)
        r = cls(len(d["W"]), float(d["l2"]))
        r.mu, r.sd, r.W = d["mu"], d["sd"], d["W"]
        return r


def cross_val_accuracy(X: np.ndarray, pairs: list[tuple[int, int, float]], l2: float, folds: int = 5) -> float | None:
    strict = [p for p in pairs if p[2] != 0.5]
    if len(strict) < folds * 2:
        return None
    rng = np.random.default_rng(0)
    order = rng.permutation(len(strict))
    accs = []
    for k in range(folds):
        test = [strict[i] for i in order[k::folds]]
        train_ids = set(order[k::folds].tolist())
        train = [strict[i] for i in range(len(strict)) if i not in train_ids] + [p for p in pairs if p[2] == 0.5]
        rm = BTReward(n_models=1, l2=l2)
        rm.fit(X, train)
        s, _ = rm.score(X)
        accs += [(s[i] > s[j]) == (y == 1) for i, j, y in test]
    return float(np.mean(accs))


def load_prefs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open(encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out
