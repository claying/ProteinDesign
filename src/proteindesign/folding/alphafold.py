from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from proteindesign.config import AlphaFoldConfig
from proteindesign.folding.base import FoldingResult
from proteindesign.structure import load_ca_coords

logger = logging.getLogger(__name__)


class AlphaFoldPredictor:
    """AlphaFold2 structure prediction via ColabFold subprocess.

    Requires ``colabfold_batch`` to be installed and accessible.
    See https://github.com/sokrypton/ColabFold for installation.
    """

    def __init__(self, config: AlphaFoldConfig | None = None) -> None:
        if config is None:
            config = AlphaFoldConfig()
        self.config = config

    def predict(self, sequence: str) -> FoldingResult:
        """Predict structure by running colabfold_batch.

        Args:
            sequence: Single-letter amino acid sequence.

        Returns:
            FoldingResult with predicted structure and confidence metrics.
        """
        with tempfile.TemporaryDirectory(prefix="proteindesign_af_") as tmpdir:
            tmpdir = Path(tmpdir)

            # Write input FASTA
            fasta_path = tmpdir / "input.fasta"
            fasta_path.write_text(f">query\n{sequence}\n")

            out_dir = tmpdir / "output"
            cmd = [
                self.config.colabfold_path,
                str(fasta_path),
                str(out_dir),
                "--num-recycle",
                str(self.config.num_recycles),
                "--num-models",
                str(self.config.num_models),
            ]

            logger.info("Running ColabFold: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ColabFold failed (exit code {result.returncode}):\n"
                    f"{result.stderr}"
                )

            # Find output files
            pdb_files = sorted(out_dir.glob("*rank_001*.pdb")) or sorted(
                out_dir.glob("*.pdb")
            )
            json_files = sorted(out_dir.glob("*scores_rank_001*.json")) or sorted(
                out_dir.glob("*scores*.json")
            )

            if not pdb_files:
                raise RuntimeError(
                    f"ColabFold produced no PDB output in {out_dir}.\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )

            pdb_path = pdb_files[0]
            pdb_string = pdb_path.read_text()
            ca_coords = load_ca_coords(pdb_path)

            # Parse metrics from scores JSON
            plddt = 0.0
            ptm = 0.0
            if json_files:
                with open(json_files[0]) as f:
                    scores = json.load(f)
                plddt_values = scores.get("plddt", [])
                if plddt_values:
                    plddt = float(np.mean(plddt_values))
                ptm = float(scores.get("ptm", 0.0))

            return FoldingResult(
                pdb_string=pdb_string,
                ca_coords=ca_coords,
                sequence=sequence,
                plddt=plddt,
                ptm=ptm,
            )
