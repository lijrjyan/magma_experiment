"""Average aggregation baseline."""

from typing import Dict

import torch

from utils.util_fusion import fusion_avg


def aggregate(model_updates: Dict[int, torch.nn.Module]) -> Dict[str, torch.Tensor]:
    return fusion_avg(model_updates)
