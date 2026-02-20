from dataclasses import dataclass
from typing import Optional


@dataclass
class ProteinMPNNConfig:
    """Configuration for ProteinMPNN inverse folding."""

    proteinmpnn_path: str
    num_seqs: int = 8
    sampling_temp: float = 0.1
    model_name: str = "v_48_020"
    seed: int = 0
    batch_size: int = 1
    ca_only: bool = False


@dataclass
class ESMFoldConfig:
    """Configuration for ESMFold structure prediction."""

    num_recycles: int = 4
    chunk_size: Optional[int] = 128
    device: str = "cuda"


@dataclass
class AlphaFoldConfig:
    """Configuration for AlphaFold2 via ColabFold."""

    colabfold_path: str = "colabfold_batch"
    num_recycles: int = 3
    num_models: int = 1
