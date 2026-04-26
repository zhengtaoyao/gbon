"""G-BoN-only evaluation harness.

This script evaluates the G-BoN method on a LIBERO suite. It does NOT
contain baseline-method branches (no frozen-only, no always-on BoN,
no MG-Select / RoVer / CoVer-VLA reimplementations); for those, see
the research codebase.

Usage (single-suite eval on standard LIBERO):

    python scripts/eval_gbon.py \\
        --suite libero_spatial \\
        --ood-head ./checkpoints/ood_head/libero_spatial.pt \\
        --verifier ./checkpoints/verifier/libero_spatial_obj.pt \\
        --memory   ./checkpoints/memory/libero_spatial.pt \\
        --vla-ckpt /path/to/openvla-oft-libero_spatial \\
        --max-tasks 10 --rollouts-per-task 20 --max-steps 220 \\
        --output ./results/parity_libero_spatial.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from gbon import OODHead, RetrievalVerifier, MemoryBank, GBoN, GBoNConfig  # noqa: E402


def load_ood_head(path: str, device: str) -> tuple[OODHead, float]:
    sd = torch.load(path, map_location=device, weights_only=False)
    head = OODHead(d_in=sd["d_in"], d_hidden=sd["d_hidden"]).to(device).eval()
    head.load_state_dict(sd["state_dict"])
    return head, float(sd["tau_star"])


def load_verifier(path: str, memory: MemoryBank, device: str) -> RetrievalVerifier:
    sd = torch.load(path, map_location=device, weights_only=False)
    K, Da = memory.action_chunk_shape
    v = RetrievalVerifier(
        d_obs=memory.d_obs,
        action_chunk_shape=(K, Da),
        d_hidden=256,
    ).to(device).eval()
    v.load_state_dict(sd["verifier_state"])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    choices=["libero_spatial", "libero_object",
                             "libero_goal", "libero_10"])
    ap.add_argument("--ood-head", required=True,
                    help="Path to trained OOD head checkpoint (.pt)")
    ap.add_argument("--verifier", required=True,
                    help="Path to trained verifier checkpoint (.pt)")
    ap.add_argument("--memory", required=True,
                    help="Path to memory bank (.pt)")
    ap.add_argument("--vla-ckpt", required=True,
                    help="Path to frozen OpenVLA-OFT checkpoint dir")
    ap.add_argument("--max-tasks", type=int, default=10)
    ap.add_argument("--rollouts-per-task", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"[gbon-eval] suite={args.suite}")
    print(f"[gbon-eval] loading memory bank -> {args.memory}", flush=True)
    memory = MemoryBank.load(args.memory, device=args.device)

    print(f"[gbon-eval] loading OOD head -> {args.ood_head}", flush=True)
    ood_head, tau_star = load_ood_head(args.ood_head, device=args.device)
    print(f"[gbon-eval] tau_star = {tau_star:.3f}", flush=True)

    print(f"[gbon-eval] loading verifier -> {args.verifier}", flush=True)
    verifier = load_verifier(args.verifier, memory, device=args.device)

    print(f"[gbon-eval] loading frozen VLA -> {args.vla_ckpt}", flush=True)
    # NOTE: VLA loading depends on the user's environment (transformers,
    # OpenVLA-OFT install, LIBERO env). We import the helper lazily so
    # the smoke test can run without the heavy dependencies.
    from libero_runner import (  # type: ignore[import-not-found]
        LiberoRunner,
        load_oft_predict_fn,
    )
    predict_fn = load_oft_predict_fn(args.vla_ckpt, suite=args.suite,
                                     device=args.device)
    runner = LiberoRunner(suite=args.suite, max_steps=args.max_steps)

    cfg = GBoNConfig(n_cand=args.n_cand, sigma=args.sigma, top_k=args.top_k)
    gbon = GBoN(ood_head, tau_star, verifier, memory, cfg=cfg,
                device=args.device)

    results = {
        "suite": args.suite, "method": "gbon",
        "n_tasks_evaluated": args.max_tasks,
        "rollouts_per_task": args.rollouts_per_task,
        "max_steps": args.max_steps,
        "tau_star": tau_star,
        "per_task": {},
    }
    n_succ = n_total = 0
    t0 = time.time()
    for ti in range(args.max_tasks):
        task = runner.get_task(ti)
        succ_t = n_t = 0
        for r in range(args.rollouts_per_task):
            try:
                ok = runner.rollout(task, gbon.act, predict_fn=predict_fn,
                                    init_state_idx=r)
                succ_t += int(ok); n_t += 1
            except Exception as e:
                print(f"[warn] task{ti} rollout{r}: {e}", flush=True)
        results["per_task"][task.language] = {
            "success_rate": succ_t / max(1, n_t),
            "n_rollouts": n_t,
        }
        n_succ += succ_t; n_total += n_t
        print(f"[gbon-eval] task {ti+1}/{args.max_tasks}  "
              f"sr_so_far={n_succ/max(1, n_total):.2%}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    results["overall_success_rate"] = n_succ / max(1, n_total)
    results["wall_time_seconds"] = time.time() - t0
    if gbon.gate_fire_rate is not None:
        results["gate_fire_rate"] = gbon.gate_fire_rate
        results["gate_fire"] = gbon.gate_fire_count
        results["gate_total"] = gbon.gate_total_count
        print(f"[gbon-eval] gate fire rate = {gbon.gate_fire_rate:.3f}",
              flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gbon-eval] DONE -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
