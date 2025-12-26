"""Baseline aggregation algorithms.

Each module re-exports the corresponding fusion function from the
original MAGMA prototype so that scripts can import stable names even
while we refactor the internals across stages.
"""

from .fedavg import aggregate as fedavg
from .average import aggregate as average
from .krum import aggregate as krum
from .median import aggregate as median
from .trimmed_mean import aggregate as trimmed_mean
from .clip_median import aggregate as clipping_median
from .cosdefense import aggregate as cos_defense
from .ddfed import aggregate as dual_defense
from .magma import aggregate as magma_defense

__all__ = [
    "fedavg",
    "average",
    "krum",
    "median",
    "trimmed_mean",
    "clipping_median",
    "cos_defense",
    "dual_defense",
    "magma_defense",
]
