"""Median aggregation baseline."""

from typing import Dict

import torch

from utils.util_fusion import fusion_median


def aggregate(
    model_updates: Dict[int, torch.nn.Module],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return fusion_median(model_updates, device=device)
