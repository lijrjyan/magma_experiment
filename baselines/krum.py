"""Krum baseline wrapper."""

from typing import Dict

import torch

from utils.util_fusion import fusion_krum


def aggregate(
    model_updates: Dict[int, torch.nn.Module],
    attacker_ratio: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    max_adv = int(attacker_ratio * len(model_updates))
    return fusion_krum(model_updates, max_expected_adversaries=max_adv, device=device)
