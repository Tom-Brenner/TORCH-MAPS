"""CMUdict-39 MAPS adapter: pretrained BiLSTMs + new 39-way classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from maps_torch.model import MapsAcousticModel, load_checkpoint
from maps_torch.phones_cmudict import N_CMUDICT
from maps_torch.phones import N_FEATURES


class MapsAcousticModel39(nn.Module):
    """Adapt TORCH-MAPS: shared frontend, ``Linear(256, 39)`` instead of 61-way head."""

    def __init__(self, hidden_size: int = 128, mask_value: float = -1.0):
        super().__init__()
        self.mask_value = mask_value
        self.backbone = MapsAcousticModel(hidden_size=hidden_size, mask_value=mask_value)
        # Replace 61-way head with CMUdict-39.
        self.backbone.classifier = nn.Linear(hidden_size * 2, N_CMUDICT)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x == self.mask_value).all(dim=-1, keepdim=True)
        x = x.masked_fill(mask, 0.0)
        x = self.backbone.input_ln(x)
        x, _ = self.backbone.lstm1(x)
        x = self.backbone.ln1(x)
        x, _ = self.backbone.lstm2(x)
        x = self.backbone.ln2(x)
        x, _ = self.backbone.lstm3(x)
        x = self.backbone.ln3(x)
        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=-1)

    def freeze_backbone(self, *, freeze_classifier: bool = False) -> None:
        """Freeze all BiLSTMs / LayerNorms; optionally freeze the 39-way head too."""
        for name, p in self.backbone.named_parameters():
            if name.startswith("classifier"):
                p.requires_grad = not freeze_classifier
            else:
                p.requires_grad = False

    def freeze_last_lstm(self) -> None:
        """Freeze only the final BiLSTM (+ its LN); earlier layers and head stay trainable."""
        for p in self.backbone.lstm3.parameters():
            p.requires_grad = False
        for p in self.backbone.ln3.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def load_adapted_maps39(
    checkpoint: Path | str,
    device: torch.device | str = "cpu",
    *,
    freeze_backbone: bool = False,
) -> MapsAcousticModel39:
    """Load MAPS ``.pt``, keep BiLSTM/LN weights, re-init 39-way classifier.

    By default all parameters are trainable. Pass ``freeze_backbone=True`` to
    train only the classifier head.
    """
    base = load_checkpoint(checkpoint, device="cpu")
    model = MapsAcousticModel39()
    # Copy all non-classifier weights.
    base_sd = base.state_dict()
    target_sd = model.backbone.state_dict()
    for k, v in base_sd.items():
        if k.startswith("classifier"):
            continue
        if k in target_sd and target_sd[k].shape == v.shape:
            target_sd[k] = v
    model.backbone.load_state_dict(target_sd)
    model.unfreeze_all()
    if freeze_backbone:
        model.freeze_backbone(freeze_classifier=False)
    return model.to(device)
