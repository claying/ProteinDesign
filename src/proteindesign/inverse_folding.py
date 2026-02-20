from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from proteindesign.config import ProteinMPNNConfig

logger = logging.getLogger(__name__)


@dataclass
class DesignedSequence:
    """A single designed sequence from ProteinMPNN."""

    sequence: str
    score: float
    global_score: float
    seq_recovery: float
    sample_index: int
    temperature: float


def run_proteinmpnn(
    pdb_path: str | Path,
    output_dir: str | Path,
    config: ProteinMPNNConfig,
) -> list[DesignedSequence]:
    """Run ProteinMPNN on a single PDB file via subprocess.

    Args:
        pdb_path: Path to the input PDB file.
        output_dir: Directory for ProteinMPNN output files.
        config: ProteinMPNN configuration.

    Returns:
        List of designed sequences with metadata.
    """
    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mpnn_script = Path(config.proteinmpnn_path) / "protein_mpnn_run.py"
    if not mpnn_script.exists():
        raise FileNotFoundError(
            f"ProteinMPNN script not found at {mpnn_script}. "
            f"Clone https://github.com/dauparas/ProteinMPNN and set proteinmpnn_path."
        )

    cmd = [
        "python",
        str(mpnn_script),
        "--pdb_path",
        str(pdb_path),
        "--out_folder",
        str(output_dir),
        "--num_seq_per_target",
        str(config.num_seqs),
        "--sampling_temp",
        str(config.sampling_temp),
        "--model_name",
        config.model_name,
        "--seed",
        str(config.seed),
        "--batch_size",
        str(config.batch_size),
    ]
    if config.ca_only:
        cmd.append("--ca_only")

    logger.info("Running ProteinMPNN: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ProteinMPNN failed (exit code {result.returncode}):\n{result.stderr}"
        )

    # Parse output FASTA: <output_dir>/seqs/<pdb_stem>.fa
    fasta_path = output_dir / "seqs" / f"{pdb_path.stem}.fa"
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"ProteinMPNN FASTA output not found at {fasta_path}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    return parse_mpnn_fasta(fasta_path)


def parse_mpnn_fasta(fasta_path: str | Path) -> list[DesignedSequence]:
    """Parse ProteinMPNN output FASTA into DesignedSequence objects.

    ProteinMPNN FASTA headers have the format:
        >NAME, score=X, global_score=Y, ...        (first entry: native)
        >T=0.1, sample=1, score=X, global_score=Y, seq_recovery=Z
    """
    fasta_path = Path(fasta_path)
    sequences: list[DesignedSequence] = []

    with open(fasta_path) as f:
        lines = f.readlines()

    header: str | None = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:]
        elif header is not None:
            # Only parse designed sequences (skip the native/first entry)
            if "sample=" in header:
                fields = _parse_header_fields(header)
                # ProteinMPNN uses '/' to separate chains; normalize to ':'
                sequence = line.replace("/", ":")
                sequences.append(
                    DesignedSequence(
                        sequence=sequence,
                        score=fields.get("score", 0.0),
                        global_score=fields.get("global_score", 0.0),
                        seq_recovery=fields.get("seq_recovery", 0.0),
                        sample_index=int(fields.get("sample", 0)),
                        temperature=fields.get("T", 0.0),
                    )
                )
            header = None

    return sequences


def _parse_header_fields(header: str) -> dict[str, float]:
    """Extract numeric key=value pairs from a FASTA header string."""
    result: dict[str, float] = {}
    for part in header.split(","):
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            key = key.strip()
            try:
                result[key] = float(val.strip())
            except ValueError:
                pass
    return result
