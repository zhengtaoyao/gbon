"""OOD head for G-BoN: small MLP on OFT's mean action-query hidden state.

Purpose: decide at inference time whether the observation is in-distribution
(use frozen OFT greedy) or OOD (invoke best-of-N verifier). The head is
trained independently per LIBERO suite using the existing memory bank's
`z_obs` as in-distribution samples and LIBERO-Plus-perturbed first-frame
`z_t` as OOD samples.

Design choices:
- 2-layer MLP with GELU and LayerNorm on the D=4096 input.
- BCE with logits loss.
- Returns calibrated score via sigmoid; threshold tau is selected on a
  held-out validation split by Youden's J.
- Deliberately small (<1M params); inference cost is one matmul per step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


D_HIDDEN_DEFAULT = 256


class OODHead(nn.Module):
    def __init__(self, d_in: int = 4096, d_hidden: int = D_HIDDEN_DEFAULT, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) float. returns (B,) logit; positive = OOD."""
        h = self.norm(z)
        h = self.drop(self.act(self.fc1(h)))
        h = self.drop(self.act(self.fc2(h)))
        return self.fc3(h).squeeze(-1)

    @torch.inference_mode()
    def score(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) or (D,). returns sigmoid OOD probability in [0,1]."""
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return torch.sigmoid(self.forward(z))


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    n_epochs: int = 20
    val_frac: float = 0.2
    seed: int = 7
    amp_dtype: str = "bfloat16"  # matches OFT precision


def _split(z_clean: torch.Tensor, z_ood: torch.Tensor, val_frac: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm_c = torch.randperm(len(z_clean), generator=g)
    perm_o = torch.randperm(len(z_ood), generator=g)
    nv_c = max(1, int(val_frac * len(z_clean)))
    nv_o = max(1, int(val_frac * len(z_ood)))
    val_c, tr_c = z_clean[perm_c[:nv_c]], z_clean[perm_c[nv_c:]]
    val_o, tr_o = z_ood[perm_o[:nv_o]], z_ood[perm_o[nv_o:]]
    return tr_c, tr_o, val_c, val_o


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """labels: 0 = clean, 1 = OOD. scores: higher = more OOD."""
    s = scores.detach().cpu().float()
    y = labels.detach().cpu().float()
    n_pos = int(y.sum().item())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney U formulation
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float)
    sum_ranks_pos = ranks[y == 1].sum().item()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def _youden_tau(scores: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    """Return (tau, tpr_at_tau, fpr_at_tau) that maximise Youden's J = TPR-FPR."""
    s = scores.detach().cpu().float()
    y = labels.detach().cpu().float()
    order = torch.argsort(s, descending=True)
    s_sorted = s[order]
    y_sorted = y[order]
    tp = torch.cumsum(y_sorted, dim=0)
    fp = torch.cumsum(1 - y_sorted, dim=0)
    P = float(y_sorted.sum())
    N = float(len(y_sorted) - P)
    if P == 0 or N == 0:
        return 0.5, 0.0, 0.0
    tpr = tp / P
    fpr = fp / N
    j = tpr - fpr
    k = int(torch.argmax(j))
    return float(s_sorted[k]), float(tpr[k]), float(fpr[k])


def train_ood_head(
    z_clean: torch.Tensor,
    z_ood: torch.Tensor,
    device: str = "cuda:0",
    cfg: TrainConfig = TrainConfig(),
) -> Tuple[OODHead, dict]:
    """Train an OOD head. Returns (model, metrics_dict)."""
    torch.manual_seed(cfg.seed)
    d_in = z_clean.shape[1]
    model = OODHead(d_in=d_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tr_c, tr_o, val_c, val_o = _split(z_clean, z_ood, cfg.val_frac, cfg.seed)
    n_c, n_o = len(tr_c), len(tr_o)

    tr_z = torch.cat([tr_c, tr_o], dim=0).to(device)
    tr_y = torch.cat([torch.zeros(n_c), torch.ones(n_o)]).to(device)
    val_z = torch.cat([val_c, val_o], dim=0).to(device)
    val_y = torch.cat([torch.zeros(len(val_c)), torch.ones(len(val_o))]).to(device)

    # class weights for imbalance
    pos_weight = torch.tensor([n_c / max(1, n_o)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    N_tr = len(tr_z)
    hist = {"train_loss": [], "val_auc": [], "val_tau": []}

    for epoch in range(cfg.n_epochs):
        model.train()
        perm = torch.randperm(N_tr, device=device)
        loss_sum = 0.0
        for i in range(0, N_tr, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            z_b = tr_z[idx]
            y_b = tr_y[idx]
            logits = model(z_b)
            loss = loss_fn(logits, y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(idx)
        train_loss = loss_sum / N_tr

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_z)
        auc = _roc_auc(val_logits, val_y)
        tau, tpr, fpr = _youden_tau(torch.sigmoid(val_logits), val_y)
        hist["train_loss"].append(train_loss)
        hist["val_auc"].append(auc)
        hist["val_tau"].append(tau)
        print(f"[ood] epoch {epoch+1:2d}/{cfg.n_epochs}  train_loss={train_loss:.4f}  "
              f"val_auc={auc:.3f}  tau*={tau:.3f} (TPR={tpr:.3f} FPR={fpr:.3f})",
              flush=True)

    # final calibration on val
    model.eval()
    with torch.inference_mode():
        val_logits = model(val_z)
    auc = _roc_auc(val_logits, val_y)
    tau, tpr, fpr = _youden_tau(torch.sigmoid(val_logits), val_y)
    metrics = {
        "val_auc_final": auc,
        "tau_star": tau,
        "tpr_at_tau": tpr,
        "fpr_at_tau": fpr,
        "n_clean_train": n_c,
        "n_ood_train": n_o,
        "n_clean_val": len(val_c),
        "n_ood_val": len(val_o),
        "history": hist,
        "cfg": cfg.__dict__,
    }
    return model, metrics


if __name__ == "__main__":
    # Simple self-test on synthetic data: two gaussians should be separable.
    torch.manual_seed(0)
    D = 4096
    z_c = torch.randn(1000, D)
    z_o = torch.randn(1000, D) + 0.5  # shifted mean
    model, m = train_ood_head(
        z_c, z_o, device="cuda" if torch.cuda.is_available() else "cpu",
        cfg=TrainConfig(n_epochs=5),
    )
    print("self-test AUC:", m["val_auc_final"])
