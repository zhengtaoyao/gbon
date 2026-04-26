"""Smoke test for the G-BoN package.

Validates:
1. All modules import without error.
2. OODHead trains on a synthetic, well-separable dataset and val-AUC
   converges to 1.0.
3. The end-to-end GBoN.act() path runs with a mock verifier_fn and
   the gate fires when expected.

Does NOT require LIBERO, OpenVLA-OFT, MuJoCo, or any robot env. This is
purely a code-correctness check; published results in the paper come from
the full evaluation pipeline (see scripts/eval_gbon.py).

Usage:
    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def test_imports():
    print("[smoke] importing gbon...", flush=True)
    import gbon
    from gbon import (OODHead, TrainConfig, train_ood_head,
                      GBoN, GBoNConfig, gbon_step,
                      RetrievalVerifier, MemoryBank)
    print(f"[smoke] gbon v{gbon.__version__} imported.", flush=True)


def test_ood_head_trains():
    print("[smoke] training OOD head on synthetic well-separable data...",
          flush=True)
    from gbon import OODHead, TrainConfig, train_ood_head
    torch.manual_seed(0)
    D = 32
    # Make the two classes well-separable: constant +3 shift in first 8 dims.
    z_clean = torch.randn(500, D)
    z_ood = torch.randn(500, D)
    z_ood[:, :8] += 3.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metrics = train_ood_head(
        z_clean, z_ood, device=device,
        cfg=TrainConfig(n_epochs=15, batch_size=128, lr=1e-3),
    )
    assert metrics["val_auc_final"] > 0.95, (
        f"val AUC too low: {metrics['val_auc_final']}")
    print(f"[smoke] val_auc={metrics['val_auc_final']:.3f}  "
          f"tau*={metrics['tau_star']:.3f}", flush=True)


def test_gating_path_fires_when_expected():
    print("[smoke] running GBoN.act() with mock verifier...", flush=True)
    from gbon import OODHead, GBoN, GBoNConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Trivial OOD head that always says OOD (so the gate fires).
    head = OODHead(d_in=64, d_hidden=16).to(device).eval()
    with torch.no_grad():
        head.fc3.bias.fill_(10.0)  # high logit -> sigmoid ~1.0

    K, D_a, N = 4, 7, 4

    # Mock verifier: returns a score that prefers candidate index 2.
    def mock_verifier(cands: np.ndarray, z_t: torch.Tensor) -> np.ndarray:
        scores = np.zeros(cands.shape[0], dtype=np.float32)
        scores[2] = 1.0
        return scores

    # Mock policy predict.
    def mock_predict(obs: dict, lang: str):
        return (np.random.randn(K, D_a).astype(np.float32),
                torch.randn(64))

    cfg = GBoNConfig(n_cand=N, sigma=0.1, top_k=4)
    gbon = GBoN(head, tau_star=0.5, verifier_fn=mock_verifier, cfg=cfg,
                device=device)

    chosen = gbon.act(mock_predict, obs={}, language_instr="dummy")
    assert chosen.shape == (K, D_a), f"bad chunk shape: {chosen.shape}"
    assert gbon.gate_fire_rate == 1.0, "gate should always fire on this head"
    print(f"[smoke] chosen.shape={chosen.shape}  "
          f"gate_fire_rate={gbon.gate_fire_rate:.2f}", flush=True)


def test_gating_path_idle_when_in_distribution():
    print("[smoke] verifying gate stays IDLE on in-distribution OOD score...",
          flush=True)
    from gbon import OODHead, GBoN, GBoNConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # OOD head that always says in-distribution (so the gate stays idle).
    head = OODHead(d_in=64, d_hidden=16).to(device).eval()
    with torch.no_grad():
        head.fc3.bias.fill_(-10.0)  # low logit -> sigmoid ~0.0

    K, D_a, N = 4, 7, 4

    def mock_verifier(cands: np.ndarray, z_t: torch.Tensor) -> np.ndarray:
        # If this is called, we have a bug.
        raise AssertionError("verifier must NOT be called when gate is idle")

    greedy = np.full((K, D_a), 1.234, dtype=np.float32)

    def mock_predict(obs: dict, lang: str):
        return greedy, torch.randn(64)

    cfg = GBoNConfig(n_cand=N, sigma=0.1, top_k=4)
    gbon = GBoN(head, tau_star=0.5, verifier_fn=mock_verifier, cfg=cfg,
                device=device)

    chosen = gbon.act(mock_predict, obs={}, language_instr="dummy")
    assert np.allclose(chosen, greedy), "should return greedy unmodified"
    assert gbon.gate_fire_rate == 0.0, "gate should never fire on this head"
    print(f"[smoke] gate idle confirmed; greedy returned unmodified.",
          flush=True)


def main():
    test_imports()
    test_ood_head_trains()
    test_gating_path_fires_when_expected()
    test_gating_path_idle_when_in_distribution()
    print("\n[smoke] ALL TESTS PASSED.")


if __name__ == "__main__":
    main()
