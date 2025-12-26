"""Sample-size scaling attack (lie about the number of local samples).

In this codebase we implement a *sample-based* scaling attack:
attackers keep their submitted model update as-is, but report a larger
local dataset size so their update receives higher weight in FedAvg-like
aggregation.

Optionally, this attack can also flip labels during local training to
make the submitted update actively harmful (controlled via
`label_flipping_ratio`).
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from .base import Attack, AttackContext
from .label_flip import flip_labels, num_classes_for_dataset


def report_fake_data_size(real_data_size: int, multiplier: float = 10.0) -> int:
    return int(real_data_size * multiplier)


class ScalingAttack(Attack):
    name = "model_poisoning_scaling"

    def __init__(
        self,
        *,
        data_size_multiplier: float = 10.0,
        label_flipping_ratio: float = 0.0,
        start_round: int = 0,
    ) -> None:
        super().__init__(start_round=start_round)
        self.data_size_multiplier = float(data_size_multiplier)
        self.label_flipping_ratio = float(label_flipping_ratio)

    def poison_labels(self, targets: torch.Tensor, *, context: AttackContext) -> torch.Tensor:
        if self.label_flipping_ratio <= 0:
            return targets
        num_classes = num_classes_for_dataset(context.dataset)
        return flip_labels(targets, self.label_flipping_ratio, num_classes)

    def report_data_size(self, real_data_size: int, *, context: AttackContext) -> int:
        return report_fake_data_size(real_data_size, multiplier=self.data_size_multiplier)

    def params(self) -> Dict[str, Any]:
        return {
            "fake_data_size_multiplier": self.data_size_multiplier,
            "label_flipping_ratio": self.label_flipping_ratio,
        }
