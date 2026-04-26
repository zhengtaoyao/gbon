"""Memory bank for VRAG-LAC action-conditioned retrieval.

Stores per-chunk tuples from training demonstrations:
  (z_obs, a_chunk, episode_id, chunk_idx, task_id)

z_obs: [D_obs] visual+proprio embedding from the frozen OpenVLA-OFT vision stack
a_chunk: [K, D_a] action chunk (K=8 for OFT)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class MemoryEntry:
    z_obs: torch.Tensor          # [D_obs]
    a_chunk: torch.Tensor        # [K, D_a]
    episode_id: int
    chunk_idx: int
    task_id: int

    def to(self, device):
        return MemoryEntry(
            z_obs=self.z_obs.to(device),
            a_chunk=self.a_chunk.to(device),
            episode_id=self.episode_id,
            chunk_idx=self.chunk_idx,
            task_id=self.task_id,
        )


class MemoryBank:
    """Append-only memory with bf16 storage and torch-native cosine search."""

    def __init__(self, d_obs: int, action_chunk_shape: tuple[int, int], capacity: int = 100_000,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.d_obs = d_obs
        self.action_chunk_shape = action_chunk_shape  # (K, D_a)
        self.capacity = capacity
        self.device = device
        self.dtype = dtype

        self.z_obs = torch.zeros(capacity, d_obs, device=device, dtype=dtype)
        self.a_chunk = torch.zeros(capacity, *action_chunk_shape, device=device, dtype=dtype)
        self.episode_id = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.chunk_idx = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.task_id = torch.zeros(capacity, dtype=torch.int64, device=device)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def insert(self, z: torch.Tensor, a: torch.Tensor, episode_id: int, chunk_idx: int, task_id: int):
        if self._n >= self.capacity:
            raise RuntimeError(f"Memory bank full ({self.capacity}). Increase capacity or enable eviction.")
        i = self._n
        self.z_obs[i] = z.to(self.device, self.dtype)
        self.a_chunk[i] = a.to(self.device, self.dtype)
        self.episode_id[i] = episode_id
        self.chunk_idx[i] = chunk_idx
        self.task_id[i] = task_id
        self._n += 1

    def bulk_insert(self, z_batch: torch.Tensor, a_batch: torch.Tensor,
                    ep_ids: torch.Tensor, chunk_idxs: torch.Tensor, task_ids: torch.Tensor):
        n = z_batch.shape[0]
        if self._n + n > self.capacity:
            raise RuntimeError(f"Bulk insert of {n} would overflow memory (have {self._n}, cap {self.capacity}).")
        s, e = self._n, self._n + n
        self.z_obs[s:e] = z_batch.to(self.device, self.dtype)
        self.a_chunk[s:e] = a_batch.to(self.device, self.dtype)
        self.episode_id[s:e] = ep_ids.to(self.device)
        self.chunk_idx[s:e] = chunk_idxs.to(self.device)
        self.task_id[s:e] = task_ids.to(self.device)
        self._n = e

    def search_cosine(self, query: torch.Tensor, top_k: int = 8,
                      exclude_task_ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine similarity search over stored keys.

        query: [B, D_key] — caller is responsible for computing the key.
        Returns (topk_idx [B, K], topk_score [B, K]).
        """
        assert self._n > 0, "Memory bank is empty"
        q = torch.nn.functional.normalize(query.to(self.dtype), dim=-1)
        keys = torch.nn.functional.normalize(self.z_obs[: self._n], dim=-1)
        sim = q @ keys.T
        if exclude_task_ids is not None:
            # Mask out same-task entries (useful for held-out evaluation)
            mask = (self.task_id[: self._n].unsqueeze(0) == exclude_task_ids.unsqueeze(1))
            sim = sim.masked_fill(mask, -1e4)
        top = torch.topk(sim, k=min(top_k, self._n), dim=-1)
        return top.indices, top.values

    def gather(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather memory entries by indices. idx: [B, K]. Returns (z [B,K,D], a [B,K,K_a,D_a])."""
        z = self.z_obs[idx]
        a = self.a_chunk[idx]
        return z, a

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "z_obs": self.z_obs[: self._n].cpu(),
                "a_chunk": self.a_chunk[: self._n].cpu(),
                "episode_id": self.episode_id[: self._n].cpu(),
                "chunk_idx": self.chunk_idx[: self._n].cpu(),
                "task_id": self.task_id[: self._n].cpu(),
                "d_obs": self.d_obs,
                "action_chunk_shape": self.action_chunk_shape,
                "capacity": self.capacity,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> "MemoryBank":
        data = torch.load(path, map_location="cpu", weights_only=False)
        n = data["z_obs"].shape[0]
        bank = cls(
            d_obs=data["d_obs"],
            action_chunk_shape=tuple(data["action_chunk_shape"]),
            capacity=max(data["capacity"], n),
            device=device,
            dtype=data["z_obs"].dtype,
        )
        bank.z_obs[:n] = data["z_obs"].to(device)
        bank.a_chunk[:n] = data["a_chunk"].to(device)
        bank.episode_id[:n] = data["episode_id"].to(device)
        bank.chunk_idx[:n] = data["chunk_idx"].to(device)
        bank.task_id[:n] = data["task_id"].to(device)
        bank._n = n
        return bank
