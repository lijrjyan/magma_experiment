"""Cosine-similarity defense wrapper."""

from typing import Dict, Optional

import torch

from utils.util_fusion import fusion_cos_defense


def aggregate(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_sizes: Dict[int, int],
    similarity_threshold: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    return fusion_cos_defense(
        global_model,
        model_updates,
        data_sizes,
        similarity_threshold=similarity_threshold,
    )
