"""Trimmed-mean aggregation baseline."""

from typing import Dict

import torch

from utils.util_fusion import fusion_trimmed_mean


def aggregate(
    model_updates: Dict[int, torch.nn.Module],
    trimmed_ratio: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return fusion_trimmed_mean(model_updates, trimmed_ratio=trimmed_ratio, device=device)
