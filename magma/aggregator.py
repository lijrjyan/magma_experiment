"""MAGMA plaintext aggregation wrapper (Stage 5)."""

from typing import Dict, Tuple, Iterable, Optional

import torch


def aggregate(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    *,
    device: torch.device,
    round_idx: int,
    log_dir: str,
    attacker_list: Optional[Iterable[int]] = None,
) -> Tuple[Dict[str, torch.Tensor], Iterable[int]]:
    """Aggregate client updates using MAGMA (Ward + jump ratio)."""
    from utils.util_fusion import fusion_dendro_defense

    params, benigns, _magma_info = fusion_dendro_defense(
        global_model,
        model_updates,
        data_size,
        _round_idx=round_idx,
        log_dir=log_dir,
        attacker_list=list(attacker_list or []),
        use_fhe=False,
        gradient_collector=None,
    )
    return params, benigns
