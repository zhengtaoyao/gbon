"""G-BoN: OOD-Gated Best-of-N for Vision-Language-Action Policies.

Public API:
    OODHead, train_ood_head, TrainConfig  -- the OOD detector
    GBoN, GBoNConfig, gbon_step           -- the gated best-of-N inference

Reference modules (you may use or replace):
    RetrievalVerifier  -- our cross-attention scorer
    MemoryBank         -- training-demo memory container
"""
from gbon.ood_head import OODHead, TrainConfig, train_ood_head
from gbon.inference import GBoN, GBoNConfig, gbon_step

# Reference verifier + memory; the inference path accepts a user-supplied
# verifier_fn callable so you do not need to use these.
from gbon.verifier import RetrievalVerifier
from gbon.memory import MemoryBank

__all__ = [
    "OODHead",
    "TrainConfig",
    "train_ood_head",
    "GBoN",
    "GBoNConfig",
    "gbon_step",
    "RetrievalVerifier",
    "MemoryBank",
]

__version__ = "0.1.0"
