"""Best-of-N candidate verifier with retrieval-augmented scoring.

This module replaces the original Stage-3 CEM planning loop. The insight that
makes action-conditioned retrieval interesting is: for a test-time best-of-N
verifier, per-candidate retrieval provides a per-candidate signal that
observation-only retrieval cannot. The verifier learns a scalar score

    score(z_t, a_cand_i, Mem(k_i)) -> R

and the best-of-N pick is argmax_i score.

Training uses a margin-rank objective on LIBERO demos:
    score(GT) > score(policy) > score(noise)
plus a regression head trained against negative action-MSE to ground truth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalVerifier(nn.Module):
    """Lightweight verifier: cross-attention over retrieved memory + scalar head.

    Inputs per batch element:
        z_t:     [B, D_obs]
        a_cand:  [B, K, D_a]
        mem_z:   [B, TopK, D_obs]  -- retrieved memory observation embeddings
        mem_a:   [B, TopK, K, D_a] -- retrieved memory action chunks
        mem_sim: [B, TopK]         -- retrieval similarities (optional aux)
    Output:
        score:   [B]               -- larger is better
    """

    def __init__(
        self,
        d_obs: int,
        action_chunk_shape: tuple[int, int],
        d_hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        K, D_a = action_chunk_shape
        self.d_hidden = d_hidden
        self.cand_proj = nn.Linear(d_obs + K * D_a, d_hidden)
        self.mem_proj = nn.Linear(d_obs + K * D_a, d_hidden)
        self.sim_proj = nn.Linear(1, d_hidden)
        # Query-key cross-attention over retrieved memory
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(d_hidden, n_heads, batch_first=True, dropout=0.0)
                for _ in range(n_layers)
            ]
        )
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_hidden, 4 * d_hidden),
                    nn.GELU(),
                    nn.Linear(4 * d_hidden, d_hidden),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(d_hidden) for _ in range(2 * n_layers)]
        )
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, z_t, a_cand, mem_z, mem_a, mem_sim):
        B, TopK = mem_z.shape[:2]
        cand_vec = self.cand_proj(torch.cat([z_t, a_cand.flatten(1)], dim=-1))  # [B, H]
        mem_vec = self.mem_proj(
            torch.cat([mem_z, mem_a.flatten(-2)], dim=-1)
        )  # [B, TopK, H]
        mem_vec = mem_vec + self.sim_proj(mem_sim.unsqueeze(-1))

        q = cand_vec.unsqueeze(1)  # [B, 1, H]
        kv = mem_vec               # [B, TopK, H]
        for i, (attn, ffn) in enumerate(zip(self.attn_layers, self.ffn_layers)):
            q_norm = self.norm_layers[2 * i](q)
            kv_norm = self.norm_layers[2 * i](kv)
            attn_out, _ = attn(q_norm, kv_norm, kv_norm)
            q = q + attn_out
            q = q + ffn(self.norm_layers[2 * i + 1](q))

        return self.head(q.squeeze(1)).squeeze(-1)


def margin_rank_loss(scores_gt, scores_pol, scores_noise, margin: float = 0.5):
    """score(GT) > score(policy) > score(noise), both gaps margin."""
    gap1 = F.relu(margin - (scores_gt - scores_pol)).mean()
    gap2 = F.relu(margin - (scores_pol - scores_noise)).mean()
    return gap1 + gap2


def regression_loss(scores_cand, neg_mse):
    """Regress score onto -MSE-to-GT (higher score = smaller error)."""
    return F.mse_loss(scores_cand, neg_mse)
