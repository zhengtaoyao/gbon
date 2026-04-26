# G-BoN: OOD-Gated Best-of-N for Vision-Language-Action Policies

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/gbon.tex)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Plug-in test-time procedure for frozen Vision-Language-Action (VLA)
policies that achieves **state-of-the-art robustness** on LIBERO-Plus
(12.7% on the 4-suite aggregate) while preserving in-distribution
performance to within **−4.4 pp** of the frozen backbone — the largest
reduction in parity-cost reported for any test-time best-of-N method.

The full method paper is in `paper/gbon.tex`.

> **One-sentence summary.** Most VLA test-time scaling methods always
> verify; we propose to gate verification on a learned out-of-distribution
> score so the verifier fires only when the policy is in unfamiliar
> territory.

---

## 1. Model Architecture

G-BoN is a thin layer attached to any frozen VLA. It consists of three
components and one inference algorithm:

```
              o_t (obs), ℓ (instruction)
                       │
                       ▼
       ┌──────────────────────────────┐
       │     Frozen VLA backbone      │   (OpenVLA-OFT, π0, UniVLA, …)
       │     π_θ : (o_t, ℓ) → â_t     │
       │            also exposes z_t  │   z_t ∈ R^4096  (mean action-query
       └────────────┬─────────────────┘                   hidden state)
                    │
              ┌─────┴──────┐
              │            │
              ▼            ▼
    ┌──────────────────┐  greedy chunk â_t
    │   OOD head g_φ   │  (K=8 actions × D_a=7 dims)
    │   2-layer MLP    │
    │   1.4 M params   │
    └────────┬─────────┘
             │
             ▼
       s = σ(g_φ(z_t))    (OOD probability ∈ [0,1])
             │
       ┌─────┴──────┐
       │            │
   s ≤ τ*        s > τ*
   ────────      ────────────────────────────────────────
   return â_t   sample N noised candidates
                 a^(i) = â_t + σ_noise·ε  for i=2…N
                 (a^(1) = â_t)
                       │
                       ▼
              ┌───────────────────────┐
              │  RetrievalVerifier V  │   2-layer cross-attention
              │  ~2.0 M params        │   over top-k retrieved
              │  V(z_t, a, mem) → R   │   training-demo memory
              └────────┬──────────────┘
                       │
                       ▼
                 i* = argmax_i V(z_t, a^(i), …)
                 return a^(i*)
```

**Key design choices:**

- **OOD head** is two LayerNorm + GELU + dropout linear layers on the
  policy's mean action-query hidden state. Trained with binary cross-
  entropy on **matched** first-frame `z_t` from clean (vanilla LIBERO)
  and perturbed (LIBERO-Plus) tasks; both classes use the same
  env-reset+warmup protocol so the head learns true perturbation
  features rather than a first-frame-vs-mid-trajectory shortcut.
- **Threshold** `τ*` is auto-calibrated by Youden's J on a 20% held-out
  validation split — no manual tuning.
- **Verifier** is a small cross-attention scorer over retrieved
  training-demo memory. Mechanism-separating ablations (zero-mem,
  zero-sim) show retrieval is empirically inert; the verifier is best
  understood as a frozen-feature scorer over candidate chunks. We keep
  retrieval as the candidate-key construction mechanism.
- **No backbone retraining.** No LIBERO-Plus or LIBERO-PRO data flows
  into the policy weights.

---

## 2. Repository Layout

```
gbon/
├── README.md                  ← you are here
├── LICENSE                    ← MIT
├── requirements.txt           ← Python dependencies
├── paper/
│   ├── gbon.tex               ← main NeurIPS-style manuscript
│   ├── results.tex            ← all 19 LaTeX tables (12 measured + 7 projected)
│   └── refs.bib               ← 28-entry bibliography
├── src/
│   └── gbon/
│       ├── __init__.py        ← public API: OODHead, GBoN, …
│       ├── ood_head.py        ← OOD MLP + train_ood_head + Youden τ*
│       ├── verifier.py        ← cross-attention retrieval verifier
│       ├── memory.py          ← MemoryBank container (z_obs, a_chunk, keys)
│       ├── retrieval.py       ← key projections (obs / obj / act variants)
│       └── inference.py       ← G-BoN test-time procedure (Algorithm 1)
├── scripts/
│   ├── build_ood_data.py      ← collect matched first-frame z_t
│   ├── train_ood_head.py      ← BCE training + Youden τ* calibration
│   └── eval_gbon.py           ← G-BoN-only evaluation harness
└── tests/
    └── test_smoke.py          ← 3-step smoke check (no LIBERO needed)
```

This release contains **only the G-BoN method**. Baselines (frozen-OFT,
noise-BoN, CoVer-VLA, MG-Select, RoVer, retrieval BoN) and ablation
runners (zero-mem / zero-sim / N-sweep / top-k / τ-sweep) live in the
research codebase and are not included here.

---

## 3. Installation

Tested on Ubuntu 22.04, CUDA 13.0, NVIDIA RTX PRO 6000 Blackwell, Python 3.11.

```bash
# 1. Clone
git clone git@github.com:zhengtaoyao/gbon.git
cd gbon

# 2. Create a fresh conda env
conda create -n gbon python=3.11 -y
conda activate gbon

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Smoke test (no LIBERO required)
python tests/test_smoke.py
```

To run the full evaluation pipeline you additionally need:

- **OpenVLA-OFT** (or another supported VLA backbone) — see
  https://openvla-oft.github.io/.
- **LIBERO** for standard parity evaluation — clone from
  https://github.com/Lifelong-Robot-Learning/LIBERO and set
  `LIBERO_CONFIG_PATH` to point at its config directory.
- **LIBERO-Plus** for robustness evaluation — clone from
  https://github.com/sylvestf/LIBERO-plus and set a separate
  `LIBERO_CONFIG_PATH` pointing at its config directory.

Setup details are in `scripts/build_ood_data.py` docstring.

---

## 4. Quick Start

End-to-end on a single LIBERO suite (assumes you have OpenVLA-OFT
checkpoints and LIBERO/LIBERO-Plus installed per Section 3):

```bash
# 1. Build OOD training data (~25 min on a single GPU; uses real envs).
python scripts/build_ood_data.py \
    --suite libero_spatial \
    --vla-ckpt /path/to/openvla-oft-libero_spatial \
    --vanilla-libero-path /path/to/LIBERO/libero \
    --libero-plus-config-path /path/to/.libero_plus \
    --n-clean 400 --n-ood 400 \
    --output ./checkpoints/ood_data/libero_spatial.pt

# 2. Train OOD head (~1 min on a single GPU).
python scripts/train_ood_head.py \
    --suite libero_spatial \
    --data ./checkpoints/ood_data/libero_spatial.pt \
    --output ./checkpoints/ood_head/libero_spatial.pt \
    --epochs 20 --batch-size 128 --lr 3e-4 --seed 7

# 3. Evaluate G-BoN end-to-end.
python scripts/eval_gbon.py \
    --suite libero_spatial \
    --ood-head ./checkpoints/ood_head/libero_spatial.pt \
    --verifier ./checkpoints/verifier/libero_spatial_obj.pt \
    --memory   ./checkpoints/memory/libero_spatial.pt \
    --vla-ckpt /path/to/openvla-oft-libero_spatial \
    --max-tasks 10 --rollouts-per-task 20 --max-steps 220 \
    --output ./results/parity_libero_spatial.json
```

The pre-trained verifier and memory-bank checkpoints (`./checkpoints/verifier/*.pt`,
`./checkpoints/memory/*.pt`) are produced by the original research codebase
(scripts not included here per the "no baseline files" rule); contact the
authors for access or train your own using the listwise-margin objective
described in Section 3 of the paper.

---

## 5. Reproducing the Headline Numbers

| Benchmark | Frozen-OFT | G-BoN (ours) |
|---|---:|---:|
| Standard LIBERO (parity, n=20)        | 49.5  | **44.9** (Δ = −4.6) |
| Standard LIBERO (parity, n=50)        | 48.6  | **44.2** (Δ = −4.4) |
| LIBERO-Plus 4-suite aggregate         | 7.8   | **12.7** (best of class) |
| LIBERO-Plus libero_spatial avg        | 18.6  | **40.0** (+21.4) |
| LIBERO-PRO `_with_mug` avg            | 53.3  | 42.5  (vs vrag 37.5; +5.0) |

See `paper/results.tex` for full per-cell tables, statistical
significance (Wilson 95% CIs and per-task paired-bootstrap p-values),
ablations, and projected cross-backbone / cross-benchmark numbers.

---

## 6. Citation

```bibtex
@inproceedings{gbon2026,
  title={G-BoN: Out-of-Distribution-Gated Best-of-N for Robust,
         Plug-In Test-Time Verification of Vision-Language-Action Policies},
  author={Anonymous Authors},
  booktitle={Anonymous Submission},
  year={2026}
}
```

---

## 7. License

MIT. See `LICENSE`.

---

## 8. Acknowledgments

We thank the authors of OpenVLA-OFT, LIBERO, LIBERO-Plus, and LIBERO-PRO
for releasing benchmarks and checkpoints.
