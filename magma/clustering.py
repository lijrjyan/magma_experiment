"""Clustering helpers for MAGMA (Stage 5).

MAGMA uses Ward linkage on client distances, then searches for a
"significant" jump ratio in the linkage heights:

    r_k = h_{k+1} / h_k

The dendrogram is cut at `k*` (the best jump), and the largest component
is kept for aggregation. If no significant jump exists, all clients are
kept as a conservative fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


@dataclass
class JumpRatio:
    linkage_heights: List[float]
    min_denom: float = 1e-9

    def ratios_with_indices(self) -> Tuple[List[float], List[int]]:
        ratios: List[float] = []
        indices: List[int] = []
        for i in range(len(self.linkage_heights) - 1):
            denom = self.linkage_heights[i]
            if denom < self.min_denom:
                continue
            ratios.append(float(self.linkage_heights[i + 1] / denom))
            indices.append(i)
        return ratios, indices

    def argmax(self) -> Optional[int]:
        ratios, indices = self.ratios_with_indices()
        if not ratios:
            return None
        best_pos = int(np.argmax(ratios))
        return int(indices[best_pos])

    def max_ratio(self) -> Optional[float]:
        ratios, _ = self.ratios_with_indices()
        if not ratios:
            return None
        return float(max(ratios))


def ward_linkage_from_distance_vector(distance_vector: Sequence[float]) -> np.ndarray:
    """Compute Ward linkage from a condensed distance vector."""
    return linkage(list(distance_vector), method="ward")


def select_largest_component(
    linkage_matrix: np.ndarray,
    client_ids: Sequence[int],
    *,
    jump_ratio_threshold: float = 1.5,
    min_denom: float = 1e-9,
) -> Tuple[List[int], Dict[str, Any]]:
    """Return (kept_client_ids, metrics) using jump-ratio cut + largest component."""
    n_clients = len(client_ids)
    heights = linkage_matrix[:, 2].tolist()
    jump = JumpRatio(heights, min_denom=min_denom)
    k_star = jump.argmax()
    max_ratio = jump.max_ratio()

    metrics: Dict[str, Any] = {
        "n_clients": int(n_clients),
        "linkage_heights": heights,
        "jump_ratios": jump.ratios_with_indices()[0],
        "max_jump_ratio": max_ratio,
        "k_star_merge_idx": k_star,
        "k_star_merge_idx_1based": None if k_star is None else int(k_star + 1),
        "threshold_distance": None,
        "cluster_labels": None,
        "num_clusters": None,
        "largest_cluster_label": None,
        "decision": None,
    }

    if k_star is None or max_ratio is None or max_ratio <= jump_ratio_threshold:
        metrics["decision"] = "accept_all_no_significant_jump"
        return list(client_ids), metrics

    threshold_distance = float(heights[k_star])
    cluster_labels = fcluster(linkage_matrix, t=threshold_distance, criterion="distance")
    labels, counts = np.unique(cluster_labels, return_counts=True)
    largest_label = int(labels[int(np.argmax(counts))])
    kept_indices = np.where(cluster_labels == largest_label)[0]
    kept = [int(client_ids[i]) for i in kept_indices.tolist()]

    metrics["threshold_distance"] = threshold_distance
    metrics["cluster_labels"] = cluster_labels.astype(int).tolist()
    metrics["num_clusters"] = int(len(labels))
    metrics["largest_cluster_label"] = largest_label
    metrics["decision"] = "largest_cluster_after_jump_cut"
    return kept, metrics
