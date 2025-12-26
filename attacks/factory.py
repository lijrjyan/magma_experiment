from __future__ import annotations

from typing import Any, Mapping

from .alie import ALIEAttack
from .base import Attack, NoAttack
from .ipm import IPMAttack
from .label_flip import LabelFlipAttack
from .scaling import ScalingAttack


def build_attack(strategy: str | None, config: Mapping[str, Any]) -> Attack:
    strategy = (strategy or "none").lower()
    start_round = int(config.get("attack_start_round", 0))

    if strategy in {"none", "no_attack"}:
        return NoAttack(start_round=start_round)

    if strategy == IPMAttack.name:
        return IPMAttack(
            multiplier=int(config.get("ipm_multiplier", 5)),
            start_round=start_round,
        )

    if strategy == ALIEAttack.name:
        return ALIEAttack(
            epsilon=float(config.get("alie_epsilon", 0.1)),
            start_round=start_round,
        )

    if strategy == ScalingAttack.name:
        data_size_multiplier = config.get("fake_data_size_multiplier", 10.0)
        return ScalingAttack(
            data_size_multiplier=float(data_size_multiplier),
            label_flipping_ratio=float(config.get("label_flipping_ratio", 0.0)),
            start_round=start_round,
        )

    if strategy in {LabelFlipAttack.name, "label_flip", "label_flipping"}:
        return LabelFlipAttack(
            flip_ratio=float(config.get("label_flipping_ratio", 0.5)),
            start_round=start_round,
        )

    raise ValueError(f"Unknown attack strategy: {strategy}")

