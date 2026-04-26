"""Collect matched-protocol first-frame z_t for OOD-head training.

Produces a (z_clean, z_ood) tensor pair per LIBERO suite:
    clean = vanilla LIBERO env reset + 5 warmup dummy steps + 1 OFT forward
    ood   = LIBERO-Plus env reset + 5 warmup dummy steps + 1 OFT forward

Critical: both classes use the SAME env-reset+warmup protocol so the
OOD head learns visual perturbation features rather than a
first-frame-vs-mid-trajectory shortcut.

Usage:

    python scripts/build_ood_data.py \\
        --suite libero_spatial \\
        --vla-ckpt /path/to/openvla-oft-libero_spatial \\
        --vanilla-libero-path /path/to/LIBERO/libero \\
        --libero-plus-config-path /path/to/.libero_plus \\
        --n-clean 400 --n-ood 400 \\
        --output ./checkpoints/ood_data/libero_spatial.pt
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    choices=["libero_spatial", "libero_object",
                             "libero_goal", "libero_10"])
    ap.add_argument("--vla-ckpt", required=True)
    ap.add_argument("--vanilla-libero-path", required=True,
                    help="Path containing libero/libero/__init__.py for vanilla LIBERO")
    ap.add_argument("--libero-plus-config-path", required=True,
                    help="LIBERO_CONFIG_PATH directory pointing at LIBERO-Plus")
    ap.add_argument("--n-clean", type=int, default=400)
    ap.add_argument("--n-ood", type=int, default=400)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--init-states-per-task", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Lazy import of the LIBERO/OFT runtime; not needed for the smoke test.
    from libero_runner import collect_first_frame_z  # type: ignore

    print(f"[build-ood-data] {args.suite}: collecting {args.n_clean} clean + "
          f"{args.n_ood} OOD first-frame z_t", flush=True)
    z_clean, names_clean = collect_first_frame_z(
        suite=args.suite,
        vla_ckpt=args.vla_ckpt,
        libero_module_path=args.vanilla_libero_path,
        libero_config_path=args.libero_plus_config_path.replace(
            "libero_plus", "libero_parity"),  # vanilla config
        is_libero_plus=False,
        n_target=args.n_clean,
        warmup_steps=args.warmup_steps,
        init_states_per_task=args.init_states_per_task,
        seed=args.seed,
        device=args.device,
    )
    z_ood, names_ood = collect_first_frame_z(
        suite=args.suite,
        vla_ckpt=args.vla_ckpt,
        libero_module_path=None,  # use the LIBERO-Plus editable install
        libero_config_path=args.libero_plus_config_path,
        is_libero_plus=True,
        n_target=args.n_ood,
        warmup_steps=args.warmup_steps,
        init_states_per_task=1,
        seed=args.seed,
        device=args.device,
    )

    print(f"[build-ood-data] saving -> {args.output}", flush=True)
    torch.save({
        "z_clean": z_clean,
        "z_ood": z_ood,
        "task_names_clean": names_clean,
        "task_names_ood": names_ood,
        "suite": args.suite,
        "n_clean": len(z_clean),
        "n_ood": len(z_ood),
    }, args.output)
    print(f"[build-ood-data] DONE: clean={len(z_clean)} ood={len(z_ood)}",
          flush=True)


if __name__ == "__main__":
    main()
