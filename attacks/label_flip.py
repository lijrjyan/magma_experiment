"""Label-flip data poisoning."""

from __future__ import annotations

from typing import Any, Dict

import torch

from .base import Attack, AttackContext


def num_classes_for_dataset(dataset: str) -> int:
    dataset = (dataset or "").lower()
    if dataset in {"mnist", "fmnist", "cifar10", "svhn"}:
        return 10
    if dataset == "emnist_byclass":
        return 62
    if dataset == "emnist_bymerge":
        return 47
    if dataset == "tinyimagenet":
        return 200
    return 10


def flip_labels(target: torch.Tensor, flip_ratio: float, num_classes: int) -> torch.Tensor:
    if flip_ratio <= 0:
        return target
    batch_size = target.size(0)
    flip_indices = torch.rand(batch_size, device=target.device) < flip_ratio
    if not flip_indices.any():
        return target
    flipped = torch.randint(0, num_classes - 1, size=(flip_indices.sum(),), device=target.device)
    for i, orig_label in enumerate(target[flip_indices]):
        if flipped[i] >= orig_label:
            flipped[i] += 1
    result = target.clone()
    result[flip_indices] = flipped
    return result


class LabelFlipAttack(Attack):
    name = "data_poisoning_label_flip"

    def __init__(self, *, flip_ratio: float = 0.5, start_round: int = 0) -> None:
        super().__init__(start_round=start_round)
        self.flip_ratio = float(flip_ratio)

    def poison_labels(self, targets: torch.Tensor, *, context: AttackContext) -> torch.Tensor:
        num_classes = num_classes_for_dataset(context.dataset)
        return flip_labels(targets, self.flip_ratio, num_classes)

    def params(self) -> Dict[str, Any]:
        return {"label_flipping_ratio": self.flip_ratio}
