from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from proteindesign.config import ProteinMPNNConfig
from proteindesign.folding import get_folding_predictor
from proteindesign.inverse_folding import run_proteinmpnn
from proteindesign.metrics import compute_all_metrics
from proteindesign.structure import extract_sequence, load_ca_coords

logger = logging.getLogger(__name__)


def compute_designability(
    pdb_path: str | Path,
    folding_model: str = "esmfold",
    num_seqs: int = 8,
    sampling_temp: float = 0.1,
    proteinmpnn_path: str = "",
    output_dir: Optional[str] = None,
    mpnn_model_name: str = "v_48_020",
    mpnn_seed: int = 0,
    num_recycles: int = 4,
    device: str = "cuda",
    colabfold_path: str = "colabfold_batch",
    chunk_size: Optional[int] = 128,
) -> pd.DataFrame:
    """Compute designability metrics for protein structure(s).

    Runs the self-consistency pipeline:
        input PDB → ProteinMPNN (inverse folding) → structure prediction → metrics

    Args:
        pdb_path: Path to a PDB file or directory of PDB files.
        folding_model: ``"esmfold"`` or ``"alphafold2"``.
        num_seqs: Number of sequences to design per structure.
        sampling_temp: ProteinMPNN sampling temperature.
        proteinmpnn_path: Path to cloned ProteinMPNN repository.
        output_dir: Directory for intermediate files (default: temp directory).
        mpnn_model_name: ProteinMPNN model variant.
        mpnn_seed: Random seed for ProteinMPNN.
        num_recycles: Recycling iterations for the folding model.
        device: Torch device for ESMFold (``"cuda"`` or ``"cpu"``).
        colabfold_path: Path to ``colabfold_batch`` binary.
        chunk_size: ESMFold chunk size for memory optimization.

    Returns:
        DataFrame with columns: ``pdb_name``, ``seq_index``, ``sequence``,
        ``sc_rmsd``, ``sc_tm``, ``plddt``, ``ptm``, ``seq_recovery``,
        ``mpnn_score``.
    """
    pdb_path = Path(pdb_path)

    # Discover PDB files
    if pdb_path.is_dir():
        pdb_files = sorted(pdb_path.glob("*.pdb"))
    elif pdb_path.is_file():
        pdb_files = [pdb_path]
    else:
        raise FileNotFoundError(f"PDB path not found: {pdb_path}")

    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {pdb_path}")

    # Set up output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="proteindesign_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure ProteinMPNN
    mpnn_config = ProteinMPNNConfig(
        proteinmpnn_path=proteinmpnn_path,
        num_seqs=num_seqs,
        sampling_temp=sampling_temp,
        model_name=mpnn_model_name,
        seed=mpnn_seed,
    )

    # Initialize folding model once (ESMFold loading takes ~30s)
    logger.info("Loading folding model: %s", folding_model)
    predictor = get_folding_predictor(
        folding_model,
        num_recycles=num_recycles,
        device=device,
        chunk_size=chunk_size,
        colabfold_path=colabfold_path,
    )

    # Process each PDB
    all_rows: list[dict] = []
    for pdb_file in pdb_files:
        logger.info("Processing %s", pdb_file.name)

        # Load reference structure
        ref_ca = load_ca_coords(pdb_file)
        native_seq = extract_sequence(pdb_file)

        # Step 1: Inverse folding → designed sequences
        mpnn_out = output_dir / "mpnn" / pdb_file.stem
        designed_seqs = run_proteinmpnn(pdb_file, mpnn_out, mpnn_config)
        logger.info("  Designed %d sequences", len(designed_seqs))

        # Step 2+3: Fold each sequence and compute metrics
        for des in designed_seqs:
            try:
                # For multi-chain sequences, remove chain separators for folding
                fold_seq = des.sequence.replace(":", "")
                fold_result = predictor.predict(fold_seq)

                metrics = compute_all_metrics(
                    ref_ca=ref_ca,
                    pred_ca=fold_result.ca_coords,
                    ref_seq=native_seq,
                    pred_seq=fold_seq,
                    plddt=fold_result.plddt,
                    ptm=fold_result.ptm,
                )
                all_rows.append(
                    {
                        "pdb_name": pdb_file.stem,
                        "seq_index": des.sample_index,
                        "sequence": des.sequence,
                        "sc_rmsd": metrics.sc_rmsd,
                        "sc_tm": metrics.sc_tm,
                        "plddt": metrics.plddt,
                        "ptm": metrics.ptm,
                        "seq_recovery": metrics.seq_recovery,
                        "mpnn_score": des.score,
                    }
                )
            except Exception as e:
                logger.warning(
                    "  Failed for %s seq %d: %s", pdb_file.stem, des.sample_index, e
                )
                continue

    return pd.DataFrame(all_rows)
