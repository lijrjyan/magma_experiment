"""Distance utilities for MAGMA (placeholder).

Future stages will add vectorization helpers for computing pairwise
client distances under different representations (last-layer delta,
feature embeddings, etc.).
"""

from typing import Dict

import torch


def last_layer_difference(models: Dict[int, torch.nn.Module], global_model: torch.nn.Module):
    """Return flattened last-layer differences for each client.

    This helper is intentionally simple for Stage 1 and mirrors the
    Dual Defense signal (last-layer direction). MAGMA will later build
    richer multi-manifold geometry on top of this representation.
    """
    diffs = {}
    global_last = list(global_model.parameters())[-2].detach().clone().view(-1)
    for cid, model in models.items():
        last_layer = list(model.parameters())[-2].detach().clone().view(-1)
        diffs[cid] = last_layer - global_last
    return diffs
