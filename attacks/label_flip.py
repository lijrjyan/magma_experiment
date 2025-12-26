"""Label flipping helper."""

import torch


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
