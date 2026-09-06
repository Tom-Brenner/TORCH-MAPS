"""Dataset loader for maps_datasets/*.npz (CMUdict-39)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class MapsNpzDataset(Dataset):
    def __init__(self, npz_path: Path | str, *, max_frames: int | None = None):
        data = np.load(npz_path, allow_pickle=True)
        self.utt_id = data["utt_id"]
        self.feats = data["feats"]
        self.frame_ids = data["frame_ids"]
        self.frame_mask = data["frame_mask"]
        self.phone_ids = data["phone_ids"]
        self.b_true = data["b_true"]
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.utt_id)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feats = np.asarray(self.feats[idx], dtype=np.float32)
        frame_ids = np.asarray(self.frame_ids[idx], dtype=np.int64)
        frame_mask = np.asarray(self.frame_mask[idx], dtype=np.bool_)
        phone_ids = np.asarray(self.phone_ids[idx], dtype=np.int64)
        b_true = np.asarray(self.b_true[idx], dtype=np.float32)
        if self.max_frames is not None and feats.shape[0] > self.max_frames:
            # Skip overlong in collate by returning empty — filter in sampler later
            pass
        return {
            "utt_id": str(self.utt_id[idx]),
            "feats": torch.from_numpy(feats),
            "frame_ids": torch.from_numpy(frame_ids),
            "frame_mask": torch.from_numpy(frame_mask),
            "phone_ids": torch.from_numpy(phone_ids),
            "b_true": torch.from_numpy(b_true),
        }


def collate_maps(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Drop empty phone sequences
    batch = [b for b in batch if b["phone_ids"].numel() >= 1 and b["feats"].shape[0] >= 3]
    if not batch:
        raise RuntimeError("Empty batch after filtering")
    B = len(batch)
    T = max(b["feats"].shape[0] for b in batch)
    F = batch[0]["feats"].shape[1]
    N = max(b["phone_ids"].numel() for b in batch)

    feats = torch.full((B, T, F), -1.0, dtype=torch.float32)
    frame_ids = torch.full((B, T), -1, dtype=torch.long)
    frame_mask = torch.zeros((B, T), dtype=torch.bool)
    feat_mask = torch.zeros((B, T), dtype=torch.bool)
    phone_ids = torch.full((B, N), -1, dtype=torch.long)
    phone_mask = torch.zeros((B, N), dtype=torch.bool)
    b_true = torch.full((B, max(N - 1, 1)), -1.0, dtype=torch.float32)
    b_mask = torch.zeros((B, max(N - 1, 1)), dtype=torch.bool)
    lengths = []

    for i, b in enumerate(batch):
        t = b["feats"].shape[0]
        n = b["phone_ids"].numel()
        lengths.append(t)
        feats[i, :t] = b["feats"]
        frame_ids[i, :t] = b["frame_ids"]
        frame_mask[i, :t] = b["frame_mask"]
        feat_mask[i, :t] = True
        phone_ids[i, :n] = b["phone_ids"]
        phone_mask[i, :n] = True
        nb = b["b_true"].numel()
        if nb > 0:
            b_true[i, :nb] = b["b_true"]
            b_mask[i, :nb] = True

    return {
        "feats": feats,
        "frame_ids": frame_ids,
        "frame_mask": frame_mask,
        "feat_mask": feat_mask,
        "phone_ids": phone_ids,
        "phone_mask": phone_mask,
        "b_true": b_true,
        "b_mask": b_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "utt_id": [b["utt_id"] for b in batch],
    }
