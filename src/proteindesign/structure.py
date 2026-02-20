from __future__ import annotations

import io
from pathlib import Path

import biotite.structure as bs
import numpy as np
from Bio.Data.PDBData import protein_letters_3to1
from biotite.structure.io.pdb import PDBFile


def _read_atom_array(pdb_path: str | Path) -> bs.AtomArray:
    """Read a PDB file and return the first model as an AtomArray."""
    pdb_file = PDBFile.read(str(pdb_path))
    return pdb_file.get_structure(model=1)


def _filter_amino_acid_atoms(atom_array: bs.AtomArray) -> bs.AtomArray:
    """Filter to standard amino acid atoms, excluding hetero atoms."""
    return atom_array[bs.filter_amino_acids(atom_array) & ~atom_array.hetero]


def load_ca_coords(pdb_path: str | Path) -> np.ndarray:
    """Load CA atom coordinates from a PDB file.

    Returns:
        (N, 3) array of CA coordinates.
    """
    atom_array = _read_atom_array(pdb_path)
    atom_array = _filter_amino_acid_atoms(atom_array)
    ca_mask = atom_array.atom_name == "CA"
    return atom_array.coord[ca_mask].copy()


def extract_sequence(pdb_path: str | Path) -> str:
    """Extract the amino acid sequence from a PDB file.

    Uses biotite for residue iteration and BioPython's mapping for
    3-letter to 1-letter amino acid code conversion.

    Returns:
        Single-letter amino acid sequence string.
    """
    atom_array = _read_atom_array(pdb_path)
    atom_array = _filter_amino_acid_atoms(atom_array)
    residues = []
    for residue in bs.residue_iter(atom_array):
        res_name = residue[0].res_name
        one_letter = protein_letters_3to1.get(res_name, "X")
        # protein_letters_3to1 may return multi-char strings for modified residues
        if len(one_letter) != 1:
            one_letter = "X"
        residues.append(one_letter)
    return "".join(residues)


def pdb_string_to_ca_coords(pdb_string: str) -> np.ndarray:
    """Extract CA coordinates from a PDB-format string.

    Useful for parsing in-memory PDB output from folding models.

    Returns:
        (N, 3) array of CA coordinates.
    """
    pdb_file = PDBFile.read(io.StringIO(pdb_string))
    atom_array = pdb_file.get_structure(model=1)
    atom_array = _filter_amino_acid_atoms(atom_array)
    ca_mask = atom_array.atom_name == "CA"
    return atom_array.coord[ca_mask].copy()
