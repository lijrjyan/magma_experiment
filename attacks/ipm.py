"""IPM attack helper."""

import torch

from utils.util_model import ipm_attack_craft_model


def craft_ipm_model(
    server_model: torch.nn.Module,
    local_model: torch.nn.Module,
    multiplier: int = 5,
) -> torch.nn.Module:
    return ipm_attack_craft_model(server_model, local_model, multiplier=multiplier)
