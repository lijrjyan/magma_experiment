"""FedAvg aggregation baseline."""

from typing import Dict

import torch

from utils.util_fusion import fusion_fedavg


def aggregate(
    model_updates: Dict[int, torch.nn.Module],
    data_sizes: Dict[int, int],
) -> Dict[str, torch.Tensor]:
    return fusion_fedavg(model_updates, data_sizes)
