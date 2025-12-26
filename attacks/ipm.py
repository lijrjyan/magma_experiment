"""IPM (Inverse Parameter Manipulation) attack."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from utils.util_model import ipm_attack_craft_model

from .base import Attack, AttackContext


class IPMAttack(Attack):
    name = "model_poisoning_ipm"

    def __init__(self, *, multiplier: int = 5, start_round: int = 0) -> None:
        super().__init__(start_round=start_round)
        self.multiplier = int(multiplier)

    def poison_model(
        self,
        server_model: nn.Module,
        local_model: nn.Module,
        *,
        context: AttackContext,
    ) -> nn.Module:
        return ipm_attack_craft_model(server_model, local_model, multiplier=self.multiplier)

    def params(self) -> Dict[str, Any]:
        return {"ipm_multiplier": self.multiplier}


def craft_ipm_model(
    server_model: torch.nn.Module,
    local_model: torch.nn.Module,
    multiplier: int = 5,
) -> torch.nn.Module:
    """Backwards-compatible functional wrapper."""
    return ipm_attack_craft_model(server_model, local_model, multiplier=multiplier)
