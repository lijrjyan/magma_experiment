"""MAGMA aggregator wrapper (Stage 1 placeholder)."""

from typing import Dict, Iterable, Tuple, Optional

import torch

from magma.aggregator import aggregate as magma_aggregate


def aggregate(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_sizes: Dict[int, int],
    *,
    device: torch.device,
    round_idx: int,
    log_dir: str,
    attacker_list: Optional[Iterable[int]] = None,
) -> Tuple[Dict[str, torch.Tensor], Iterable[int]]:
    return magma_aggregate(
        global_model,
        model_updates,
        data_sizes,
        device=device,
        round_idx=round_idx,
        log_dir=log_dir,
        attacker_list=attacker_list,
    )
