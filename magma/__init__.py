"""MAGMA core package.

This package will host the geometric aggregation pipeline:
- distance computations (e.g., last-layer diffs)
- clustering utilities (e.g., Ward linkage + jump ratios)
- aggregators that select benign components
- optional FHE-compatible backends

Stage 1 only wires up the repository skeleton; the concrete MAGMA
implementation will arrive in later stages of the execution plan.
"""

from . import distance, clustering, aggregator, fhe_backend  # noqa: F401

__all__ = [
    "distance",
    "clustering",
    "aggregator",
    "fhe_backend",
]
