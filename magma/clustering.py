"""Clustering helpers for MAGMA (placeholder).

This module will eventually expose Ward linkage utilities, jump-ratio
analysis, and largest-component selection logic. Stage 1 only provides
basic scaffolding so that unit tests and import paths remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class JumpRatio:
    linkage_heights: List[float]

    def argmax(self) -> int:
        if len(self.linkage_heights) < 2:
            return 0
        ratios = [
            (self.linkage_heights[i + 1] / max(self.linkage_heights[i], 1e-12))
            for i in range(len(self.linkage_heights) - 1)
        ]
        best_idx = max(range(len(ratios)), key=lambda idx: ratios[idx])
        return best_idx
