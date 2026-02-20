import numpy as np

from proteindesign.structure import extract_sequence, load_ca_coords


def test_load_ca_coords(mini_pdb_path):
    coords = load_ca_coords(mini_pdb_path)
    assert coords.shape == (5, 3)
    # First CA is at (2, 1, 1)
    np.testing.assert_allclose(coords[0], [2.0, 1.0, 1.0], atol=1e-3)
    # Second CA is at (4.5, 0, 1)
    np.testing.assert_allclose(coords[1], [4.5, 0.0, 1.0], atol=1e-3)


def test_extract_sequence(mini_pdb_path):
    seq = extract_sequence(mini_pdb_path)
    assert seq == "AGVLS"
