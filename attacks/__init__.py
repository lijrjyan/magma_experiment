"""Poisoning attacks (Stage 3).

Exports unified attack interfaces plus small functional wrappers used by
legacy code.
"""

from .alie import ALIEAttack, craft_alie_model
from .base import Attack, AttackContext, NoAttack
from .factory import build_attack
from .ipm import IPMAttack, craft_ipm_model
from .label_flip import LabelFlipAttack, flip_labels, num_classes_for_dataset
from .scaling import ScalingAttack, report_fake_data_size

__all__ = [
    "Attack",
    "AttackContext",
    "NoAttack",
    "IPMAttack",
    "ALIEAttack",
    "ScalingAttack",
    "LabelFlipAttack",
    "build_attack",
    "craft_ipm_model",
    "craft_alie_model",
    "report_fake_data_size",
    "flip_labels",
    "num_classes_for_dataset",
]
