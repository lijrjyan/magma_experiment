"""MAGMA aggregator placeholder.

Later stages will integrate the manifold-aware clustering pipeline.
For Stage 1 we merely define a placeholder that documents the expected
interface so that downstream tooling (scripts/run_exp.py, configs, etc.)
can reference MAGMA as an aggregator choice.
"""

from typing import Dict, Tuple, Any, Iterable, Optional

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
    """Placeholder MAGMA aggregator.

    The function currently falls back to the existing dendrogram-based
    defense (``fusion_dendro_defense``) to stay aligned with the Dual
    Defense baseline that ships with the repository. This keeps the
    experiment pipeline runnable in Stage 1 while we re-architect the
    codebase around the MAGMA execution plan.
    """
    from utils.util_fusion import fusion_dendro_defense

    params, benigns = fusion_dendro_defense(
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
