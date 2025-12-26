"""ALIE (Gaussian-noise) model poisoning attack."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from utils.util_model import alie_attack

from .base import Attack, AttackContext


class ALIEAttack(Attack):
    name = "model_poisoning_alie"

    def __init__(self, *, epsilon: float = 0.1, start_round: int = 0) -> None:
        super().__init__(start_round=start_round)
        self.epsilon = float(epsilon)

    def poison_model(
        self,
        server_model: nn.Module,
        local_model: nn.Module,
        *,
        context: AttackContext,
    ) -> nn.Module:
        return alie_attack(local_model, epsilon=self.epsilon)

    def params(self) -> Dict[str, Any]:
        return {"alie_epsilon": self.epsilon}


def craft_alie_model(model: torch.nn.Module, epsilon: float = 0.1) -> torch.nn.Module:
    """Backwards-compatible functional wrapper."""
    return alie_attack(model, epsilon=epsilon)
