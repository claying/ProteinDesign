import numpy as np

from proteindesign.metrics import (
    compute_all_metrics,
    compute_sc_rmsd,
    compute_sc_tm,
    compute_seq_recovery,
)


def test_rmsd_identical():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    assert compute_sc_rmsd(coords, coords) < 1e-6


def test_rmsd_translated():
    """RMSD should be zero for translated structures (centering removes translation)."""
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    pred = ref + np.array([10.0, 20.0, 30.0])
    assert compute_sc_rmsd(ref, pred) < 1e-6


def test_rmsd_different():
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pred = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    # After centering, these are offset by 1.0 in y (translation), so RMSD ~ 0
    # Actually: ref centered = [[-1,0,0],[0,0,0],[1,0,0]]
    #           pred centered = [[-1,0,0],[0,0,0],[1,0,0]] (same after centering y)
    # Wait -- pred centers to [[-1,1,0],[0,1,0],[1,1,0]] vs [[-1,0,0],[0,0,0],[1,0,0]]
    # After centering: pred = [[-1,0,0],[0,0,0],[1,0,0]], ref = [[-1,0,0],[0,0,0],[1,0,0]]
    # Identical after centering! RMSD = 0
    assert compute_sc_rmsd(ref, pred) < 1e-6


def test_rmsd_rotated():
    """RMSD should be near zero for structures differing only by rotation."""
    ref = np.array(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0], [11.4, 0.0, 0.0]],
        dtype=np.float64,
    )
    # Rotate 90 degrees around z axis
    pred = np.array(
        [[0.0, 0.0, 0.0], [0.0, 3.8, 0.0], [0.0, 7.6, 0.0], [0.0, 11.4, 0.0]],
        dtype=np.float64,
    )
    rmsd = compute_sc_rmsd(ref, pred)
    assert rmsd < 1e-4


def test_tm_score_identical():
    coords = np.array(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0], [11.4, 0.0, 0.0]],
        dtype=np.float64,
    )
    seq = "AGVL"
    tm = compute_sc_tm(coords, coords.copy(), seq, seq)
    assert tm > 0.99


def test_seq_recovery_identical():
    assert compute_seq_recovery("AGVLS", "AGVLS") == 1.0


def test_seq_recovery_partial():
    assert abs(compute_seq_recovery("AGVLS", "AGVMS") - 0.8) < 1e-6


def test_seq_recovery_none():
    assert compute_seq_recovery("AGVLS", "DKWMF") == 0.0


def test_compute_all_metrics():
    coords = np.array(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0], [11.4, 0.0, 0.0]],
        dtype=np.float64,
    )
    metrics = compute_all_metrics(
        ref_ca=coords,
        pred_ca=coords.copy(),
        ref_seq="AGVL",
        pred_seq="AGVL",
        plddt=85.0,
        ptm=0.92,
    )
    assert metrics.sc_rmsd < 1e-6
    assert metrics.sc_tm > 0.99
    assert metrics.plddt == 85.0
    assert metrics.ptm == 0.92
    assert metrics.seq_recovery == 1.0
