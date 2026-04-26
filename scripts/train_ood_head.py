"""Train the G-BoN OOD head per LIBERO suite.

Loads {z_clean, z_ood} from the dataset built by build_ood_data.py,
trains a small two-layer MLP via BCE-with-logits, selects the gating
threshold tau_star via Youden's J on a held-out validation split, and
saves a portable checkpoint usable by the inference path.

Usage:

    python scripts/train_ood_head.py \\
        --suite libero_spatial \\
        --data ./checkpoints/ood_data/libero_spatial.pt \\
        --output ./checkpoints/ood_head/libero_spatial.pt \\
        --epochs 20 --batch-size 128 --lr 3e-4 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from gbon.ood_head import OODHead, TrainConfig, train_ood_head  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--data", required=True,
                    help="Path to {z_clean, z_ood} .pt produced by build_ood_data.py")
    ap.add_argument("--output", required=True,
                    help="Where to save the trained OOD-head checkpoint")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"[train-ood] loading {args.data}", flush=True)
    d = torch.load(args.data, weights_only=False, map_location="cpu")
    z_clean = d["z_clean"].float()
    z_ood = d["z_ood"].float()
    print(f"[train-ood] z_clean={tuple(z_clean.shape)}  "
          f"z_ood={tuple(z_ood.shape)}", flush=True)

    cfg = TrainConfig(
        lr=args.lr, batch_size=args.batch_size, n_epochs=args.epochs,
        seed=args.seed,
    )
    model, metrics = train_ood_head(z_clean, z_ood, device=args.device, cfg=cfg)

    sd = {
        "state_dict": model.state_dict(),
        "d_in": z_clean.shape[1],
        "d_hidden": model.fc1.out_features,
        "tau_star": metrics["tau_star"],
        "val_auc": metrics["val_auc_final"],
        "tpr_at_tau": metrics["tpr_at_tau"],
        "fpr_at_tau": metrics["fpr_at_tau"],
        "metrics": metrics,
        "suite": args.suite,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.output)
    print(f"[train-ood] saved -> {args.output}", flush=True)
    print(f"[train-ood] val_auc={metrics['val_auc_final']:.3f}  "
          f"tau*={metrics['tau_star']:.3f}  "
          f"TPR={metrics['tpr_at_tau']:.3f}  FPR={metrics['fpr_at_tau']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
