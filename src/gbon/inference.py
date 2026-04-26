"""G-BoN: OOD-Gated Best-of-N test-time inference for VLA policies.

This file contains ONLY the G-BoN inference path. There are no baseline
branches (no frozen-only, no always-on BoN, no paraphrase BoN, no
MG-Select or RoVer reimplementations); for those, see the original
research codebase.

Design: this is a thin, reference implementation of Algorithm 1 from
the paper. It is deliberately decoupled from any specific verifier or
memory-bank class so that you can plug in your own scorer.

Public surface:
    GBoNConfig         -- N, sigma, top_k hyperparameters
    GBoN               -- stateful inference wrapper (tracks gate-fire stats)
    gbon_step          -- functional one-call helper

Notation (paper Sec. 3):
    o_t       observation at time t
    z_t       mean of policy LLM final-layer hidden states over the
              action-query token positions (shape [D])
    a_hat_t   greedy action chunk from the policy (shape [K, D_a])
    s         OOD probability sigmoid(g_phi(z_t))
    tau_star  Youden-J calibrated gating threshold
    N         best-of-N budget
    sigma     Gaussian noise scale on candidates

Verifier protocol:
    The verifier is supplied as a callable
        verifier_fn(cands_np, z_t_tensor) -> scores_np_or_tensor
    where cands_np has shape (N, K, D_a) and z_t_tensor has shape (D,).
    G-BoN calls argmax over the returned scores to pick the best
    candidate. The user is responsible for retrieval, memory lookup,
    and any verifier-internal state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from gbon.ood_head import OODHead


@dataclass
class GBoNConfig:
    """Hyperparameters for G-BoN inference."""

    n_cand: int = 8         # best-of-N budget (paper default: 8)
    sigma: float = 0.1      # Gaussian noise scale for candidate sampling
    top_k: int = 8          # passed to verifier_fn for retrieval (advisory)


class GBoN:
    """Stateful G-BoN inference wrapper.

    Construction:
        gbon = GBoN(ood_head, tau_star=0.126, verifier_fn=my_verifier)

    Per-step usage:
        chosen_chunk = gbon.act(predict_fn, obs, language_instr)

    where:
        predict_fn(obs, lang) -> (a_greedy: np.ndarray of shape [K, D_a],
                                  z_t:      torch.Tensor of shape [D])
            runs the frozen VLA forward pass.
        verifier_fn(cands, z_t) -> scores
            scores N candidates; the highest-scoring chunk is returned.
    """

    def __init__(
        self,
        ood_head: OODHead,
        tau_star: float,
        verifier_fn: Callable[[np.ndarray, torch.Tensor], np.ndarray],
        cfg: GBoNConfig = GBoNConfig(),
        device: str = "cuda:0",
    ):
        self.ood_head = ood_head.to(device).eval()
        self.tau_star = float(tau_star)
        self.verifier_fn = verifier_fn
        self.cfg = cfg
        self.device = device
        # Cumulative gating stats for diagnostics.
        self.gate_fire_count = 0
        self.gate_total_count = 0

    @torch.inference_mode()
    def _ood_score(self, z_t: torch.Tensor) -> float:
        z = z_t.to(self.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return float(self.ood_head.score(z).item())

    def _sample_candidates(self, a_hat: np.ndarray) -> np.ndarray:
        """Build N candidates: index 0 is greedy; indices 1..N-1 are noised.
        Returns array of shape (N, K, D_a)."""
        a = np.asarray(a_hat, dtype=np.float32)
        K, D_a = a.shape
        N = self.cfg.n_cand
        noise = (
            np.random.randn(N - 1, K, D_a).astype(np.float32) * self.cfg.sigma
        )
        cands = np.concatenate([a[None, :, :], a[None, :, :] + noise], axis=0)
        return cands

    def act(
        self,
        predict_fn: Callable[[dict, str], tuple],
        obs: dict,
        language_instr: str,
    ) -> np.ndarray:
        """One G-BoN inference step. Implements Algorithm 1 from the paper."""
        # Step 1: greedy forward pass (one VLA forward).
        a_greedy, z_t = predict_fn(obs, language_instr)
        # Step 2: OOD score.
        s = self._ood_score(z_t)
        self.gate_total_count += 1
        # Step 3: gate.
        if s <= self.tau_star:
            return np.asarray(a_greedy, dtype=np.float32)
        # Step 4: sample N noised candidates and score with the user verifier.
        self.gate_fire_count += 1
        cands = self._sample_candidates(a_greedy)
        scores = self.verifier_fn(cands, z_t)
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        i_star = int(np.argmax(np.asarray(scores)))
        return cands[i_star]

    @property
    def gate_fire_rate(self) -> Optional[float]:
        if self.gate_total_count == 0:
            return None
        return self.gate_fire_count / self.gate_total_count


def gbon_step(
    predict_fn: Callable[[dict, str], tuple],
    obs: dict,
    language_instr: str,
    ood_head: OODHead,
    tau_star: float,
    verifier_fn: Callable[[np.ndarray, torch.Tensor], np.ndarray],
    cfg: GBoNConfig = GBoNConfig(),
    device: str = "cuda:0",
) -> np.ndarray:
    """Functional one-call helper. For loops, prefer the GBoN class so
    cumulative gate-fire stats are tracked across calls."""
    g = GBoN(ood_head, tau_star, verifier_fn, cfg=cfg, device=device)
    return g.act(predict_fn, obs, language_instr)
