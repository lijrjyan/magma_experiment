"""Dual Defense (DDFed) wrapper."""

from typing import Dict, Tuple, Optional

import torch

from utils.util_fusion import fusion_dual_defense


def aggregate(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_sizes: Dict[int, int],
    *,
    round_idx: int,
    log_dir: str,
    attacker_list,
    use_fhe: bool = False,
    similarity_threshold: Optional[float] = None,
    epsilon: Optional[float] = None,
) -> Tuple[Dict[str, torch.Tensor], list[int]]:
    return fusion_dual_defense(
        global_model,
        model_updates,
        data_sizes,
        _round_idx=round_idx,
        log_dir=log_dir,
        attacker_list=attacker_list,
        use_fhe=use_fhe,
        gradient_collector=None,
        similarity_threshold=similarity_threshold,
        epsilon=epsilon,
    )
