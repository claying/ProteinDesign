from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tmtools


@dataclass
class DesignabilityMetrics:
    """Metrics for a single designed sequence."""

    sc_rmsd: float
    sc_tm: float
    plddt: float
    ptm: float
    seq_recovery: float


def compute_sc_rmsd(ref_ca: np.ndarray, pred_ca: np.ndarray) -> float:
    """Compute CA RMSD after optimal superposition (Kabsch algorithm).

    Args:
        ref_ca: (N, 3) reference CA coordinates.
        pred_ca: (N, 3) predicted CA coordinates.

    Returns:
        RMSD in Angstroms.
    """
    assert ref_ca.shape == pred_ca.shape, (
        f"Shape mismatch: {ref_ca.shape} vs {pred_ca.shape}"
    )
    # Center both structures
    ref_center = ref_ca.mean(axis=0)
    pred_center = pred_ca.mean(axis=0)
    ref_centered = ref_ca - ref_center
    pred_centered = pred_ca - pred_center

    # Kabsch: find optimal rotation via SVD
    H = pred_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and compute RMSD
    pred_aligned = pred_centered @ R.T
    diff = ref_centered - pred_aligned
    return float(np.sqrt((diff**2).sum() / len(ref_ca)))


def compute_sc_tm(
    ref_ca: np.ndarray,
    pred_ca: np.ndarray,
    ref_seq: str,
    pred_seq: str,
) -> float:
    """Compute TM-score using TM-align via tmtools.

    Returns:
        TM-score normalized by the reference structure length (0 to 1).
    """
    result = tmtools.tm_align(ref_ca, pred_ca, ref_seq, pred_seq)
    return float(result.tm_norm_chain1)


def compute_seq_recovery(native_seq: str, designed_seq: str) -> float:
    """Compute fraction of positions where designed matches native sequence."""
    min_len = min(len(native_seq), len(designed_seq))
    if min_len == 0:
        return 0.0
    matches = sum(
        1 for a, b in zip(native_seq[:min_len], designed_seq[:min_len]) if a == b
    )
    return matches / min_len


def compute_all_metrics(
    ref_ca: np.ndarray,
    pred_ca: np.ndarray,
    ref_seq: str,
    pred_seq: str,
    plddt: float,
    ptm: float,
) -> DesignabilityMetrics:
    """Compute all self-consistency metrics for one designed sequence.

    Handles length mismatches by truncating to the shorter length.
    """
    min_len = min(len(ref_ca), len(pred_ca))
    if min_len == 0:
        return DesignabilityMetrics(
            sc_rmsd=float("inf"),
            sc_tm=0.0,
            plddt=plddt,
            ptm=ptm,
            seq_recovery=0.0,
        )

    ref_ca_t = ref_ca[:min_len]
    pred_ca_t = pred_ca[:min_len]
    ref_seq_t = ref_seq[:min_len]
    pred_seq_t = pred_seq[:min_len]

    return DesignabilityMetrics(
        sc_rmsd=compute_sc_rmsd(ref_ca_t, pred_ca_t),
        sc_tm=compute_sc_tm(ref_ca_t, pred_ca_t, ref_seq_t, pred_seq_t),
        plddt=plddt,
        ptm=ptm,
        seq_recovery=compute_seq_recovery(ref_seq, pred_seq),
    )
