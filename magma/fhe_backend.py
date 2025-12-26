"""FHE backend abstractions (Stage 1 placeholder).

To keep Stage 1 focused on orchestration, we only define a minimal
registry that will later wrap TenSEAL CKKS or mock backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, List


class Encryptor(Protocol):
    def encrypt_vector(self, values: List[float]):
        ...


class MockEncryptor:
    """Simple mock that stores plaintext vectors."""

    def encrypt_vector(self, values: List[float]):
        return values


@dataclass
class FHEBackend:
    name: str = "mock"

    def create_encryptor(self) -> Encryptor:
        if self.name == "mock":
            return MockEncryptor()
        raise ValueError(f"Unsupported backend: {self.name}")
