"""Clipped median baseline."""

from typing import Dict

import torch

from utils.util_fusion import fusion_clipping_median


def aggregate(
    model_updates: Dict[int, torch.nn.Module],
    clipping_threshold: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return fusion_clipping_median(
        model_updates,
        clipping_threshold=clipping_threshold,
        device=device,
    )
