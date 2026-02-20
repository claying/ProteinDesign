from __future__ import annotations

import torch

from proteindesign.config import ESMFoldConfig
from proteindesign.folding.base import FoldingResult
from proteindesign.structure import pdb_string_to_ca_coords


class ESMFoldPredictor:
    """In-process ESMFold structure prediction.

    Requires ``pip install fair-esm[esmfold]`` and a CUDA GPU.
    The model is loaded once on init and reused for all predictions.
    """

    def __init__(self, config: ESMFoldConfig | None = None) -> None:
        if config is None:
            config = ESMFoldConfig()
        self.config = config

        import esm

        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval()
        if config.device != "cpu":
            self.model = self.model.to(config.device)
        if config.chunk_size is not None:
            self.model.set_chunk_size(config.chunk_size)

    @torch.no_grad()
    def predict(self, sequence: str) -> FoldingResult:
        """Predict structure from amino acid sequence.

        Args:
            sequence: Single-letter amino acid sequence.

        Returns:
            FoldingResult with predicted structure and confidence metrics.
        """
        output = self.model.infer(sequence, num_recycles=self.config.num_recycles)
        pdb_string = self.model.output_to_pdb(output)[0]
        plddt = float(output["mean_plddt"].item())
        ptm = float(output["ptm"].item())
        ca_coords = pdb_string_to_ca_coords(pdb_string)

        return FoldingResult(
            pdb_string=pdb_string,
            ca_coords=ca_coords,
            sequence=sequence,
            plddt=plddt,
            ptm=ptm,
        )
