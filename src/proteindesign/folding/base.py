from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class FoldingResult:
    """Result from a structure prediction model."""

    pdb_string: str
    ca_coords: np.ndarray
    sequence: str
    plddt: float
    ptm: float


@runtime_checkable
class FoldingPredictor(Protocol):
    def predict(self, sequence: str) -> FoldingResult: ...
