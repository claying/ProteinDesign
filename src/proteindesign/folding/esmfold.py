from __future__ import annotations

import torch

from proteindesign.config import ESMFoldConfig
from proteindesign.folding.base import FoldingResult
from proteindesign.structure import pdb_string_to_ca_coords


class ESMFoldPredictor:
    """In-process ESMFold structure prediction via Hugging Face Transformers.

    Requires ``pip install "transformers[torch]" accelerate`` and a CUDA GPU.
    The model is loaded once on init and reused for all predictions.
    """

    def __init__(self, config: ESMFoldConfig | None = None) -> None:
        if config is None:
            config = ESMFoldConfig()
        self.config = config

        from transformers import AutoTokenizer, EsmForProteinFolding

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self.model = self.model.eval()
        if config.device != "cpu":
            self.model = self.model.to(config.device)
        if config.chunk_size is not None:
            self.model.trunk.set_chunk_size(config.chunk_size)

    @torch.no_grad()
    def predict(self, sequence: str) -> FoldingResult:
        """Predict structure from amino acid sequence.

        Args:
            sequence: Single-letter amino acid sequence.

        Returns:
            FoldingResult with predicted structure and confidence metrics.
        """
        input_ids = self.tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.model.device)
        output = self.model(input_ids, num_recycles=self.config.num_recycles)

        # Convert to PDB string
        pdb_string = self._output_to_pdb(output)[0]

        # Extract metrics
        plddt = float(output.plddt.mean().item())
        ptm = float(output.ptm.item())

        ca_coords = pdb_string_to_ca_coords(pdb_string)

        return FoldingResult(
            pdb_string=pdb_string,
            ca_coords=ca_coords,
            sequence=sequence,
            plddt=plddt,
            ptm=ptm,
        )

    @staticmethod
    def _output_to_pdb(output) -> list[str]:
        """Convert ESMFold output to PDB-format strings.

        Uses the openfold utilities bundled with Hugging Face Transformers
        to convert atom14 representation to atom37 and write PDB files.
        """
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
        from transformers.models.esm.openfold_utils.protein import (
            Protein as OFProtein,
            to_pdb,
        )

        final_atom_positions = atom14_to_atom37(output.positions[-1], output)
        output_cpu = {k: v.cpu().numpy() for k, v in output.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = output_cpu["atom37_atom_exists"]

        pdbs = []
        for i in range(output_cpu["aatype"].shape[0]):
            pred = OFProtein(
                aatype=output_cpu["aatype"][i],
                atom_positions=final_atom_positions[i],
                atom_mask=final_atom_mask[i],
                residue_index=output_cpu["residue_index"][i] + 1,
                b_factors=output_cpu["plddt"][i],
                chain_index=output_cpu.get("chain_index", [None])[i],
            )
            pdbs.append(to_pdb(pred))
        return pdbs
