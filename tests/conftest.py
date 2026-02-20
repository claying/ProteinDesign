from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def mini_pdb_path():
    return DATA_DIR / "mini.pdb"


@pytest.fixture
def sample_mpnn_fasta():
    return DATA_DIR / "sample_mpnn.fa"
