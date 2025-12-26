from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AttackContext:
    dataset: str
    round_idx: int
    client_id: int


class Attack:
    """Unified attack interface with optional hooks.

    Attacks can:
    - poison labels during local training (data poisoning)
    - poison the submitted model update (model poisoning)
    - lie about local sample counts (sample-size scaling)
    """

    name: str = "none"

    def __init__(self, *, start_round: int = 0) -> None:
        self.start_round = int(start_round)

    def is_active(self, round_idx: int) -> bool:
        return round_idx >= self.start_round

    def poison_labels(self, targets: torch.Tensor, *, context: AttackContext) -> torch.Tensor:
        return targets

    def poison_model(
        self,
        server_model: nn.Module,
        local_model: nn.Module,
        *,
        context: AttackContext,
    ) -> nn.Module:
        return local_model

    def report_data_size(self, real_data_size: int, *, context: AttackContext) -> int:
        return int(real_data_size)

    def params(self) -> Dict[str, Any]:
        return {}


class NoAttack(Attack):
    name = "none"

