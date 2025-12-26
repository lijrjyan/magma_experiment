"""ALIE attack helper."""

import torch

from utils.util_model import alie_attack


def craft_alie_model(model: torch.nn.Module, epsilon: float = 0.1) -> torch.nn.Module:
    return alie_attack(model, epsilon=epsilon)
