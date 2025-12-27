"""Distance utilities for MAGMA (Stage 5).

MAGMA uses per-client model deltas (typically last-layer deltas) to build
pairwise distances and perform hierarchical clustering on the server.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


def last_layer_difference(
    models: Dict[int, torch.nn.Module],
    global_model: torch.nn.Module,
) -> Dict[int, torch.Tensor]:
    """Return flattened last-layer deltas (client - global) for each client."""
    diffs: Dict[int, torch.Tensor] = {}
    global_last = list(global_model.state_dict().values())[-2].detach().clone().view(-1)
    for client_id, model in models.items():
        last_layer = list(model.state_dict().values())[-2].detach().clone().view(-1)
        diffs[client_id] = last_layer - global_last
    return diffs


def pairwise_l2_distance_matrix(
    vectors: Dict[int, torch.Tensor],
    client_ids: Sequence[int],
) -> np.ndarray:
    """Build a symmetric pairwise L2 distance matrix (n_clients x n_clients)."""
    n_clients = len(client_ids)
    distance_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
    for i in range(n_clients):
        vec_i = vectors[client_ids[i]]
        for j in range(i + 1, n_clients):
            vec_j = vectors[client_ids[j]]
            distance = torch.norm(vec_i - vec_j).item()
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


def condensed_distance_vector(distance_matrix: np.ndarray) -> List[float]:
    """Convert a square distance matrix into SciPy's condensed vector format."""
    n_clients = int(distance_matrix.shape[0])
    vector: List[float] = []
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            vector.append(float(distance_matrix[i, j]))
    return vector
