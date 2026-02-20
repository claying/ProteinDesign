# proteindesign

Compute the designability of protein structures using inverse folding and structure prediction.

## Overview

`proteindesign` implements the standard **self-consistency** pipeline for evaluating protein backbone designability:

1. **Inverse folding** (ProteinMPNN) designs multiple amino acid sequences compatible with an input backbone structure.
2. **Structure prediction** (ESMFold or AlphaFold2) folds each designed sequence back into a 3D structure.
3. **Self-consistency metrics** compare the predicted structures against the original input to assess designability.

The pipeline outputs a table of per-sequence metrics including scRMSD, scTM, pLDDT, pTM, and sequence recovery.

## Installation

```bash
# Core package (metrics, ProteinMPNN wrapper, AlphaFold2 via ColabFold)
pip install .

# With ESMFold support (Hugging Face Transformers, requires PyTorch + CUDA GPU)
pip install ".[esmfold]"

# Development
pip install -e ".[test]"
```

### Prerequisites

**ProteinMPNN** (required): Clone the repository and provide the path when running:

```bash
git clone https://github.com/dauparas/ProteinMPNN.git
```

**ColabFold** (optional, for AlphaFold2): Install following the [ColabFold LocalColabFold instructions](https://github.com/sokrypton/ColabFold).

## Quick start

### Python API

```python
from proteindesign import compute_designability

df = compute_designability(
    pdb_path="./my_structures/",
    folding_model="esmfold",
    proteinmpnn_path="/path/to/ProteinMPNN",
    num_seqs=8,
    sampling_temp=0.1,
    device="cuda",
)

print(df)
#   pdb_name  seq_index  sequence  sc_rmsd  sc_tm  plddt   ptm  seq_recovery  mpnn_score
# 0  1abc          1     MKLL...     1.23   0.87  82.5   0.91         0.45       0.87
# 1  1abc          2     AKVL...     1.56   0.81  78.2   0.88         0.42       0.92
# ...
```

### Command line

```bash
proteindesign \
    --pdb-dir ./my_structures/ \
    --folding-model esmfold \
    --proteinmpnn-path /path/to/ProteinMPNN \
    --num-seqs 8 \
    --sampling-temp 0.1 \
    --output-dir ./results
```

Results are saved to `./results/designability_scores.csv` and a per-structure summary is printed to stdout.

### Using AlphaFold2 instead of ESMFold

```bash
proteindesign \
    --pdb-dir ./my_structures/ \
    --folding-model alphafold2 \
    --proteinmpnn-path /path/to/ProteinMPNN \
    --colabfold-path /path/to/colabfold_batch \
    --output-dir ./results
```

## Output metrics

| Column | Description | Good design threshold |
|--------|-------------|----------------------|
| `sc_rmsd` | Self-consistency CA RMSD (Angstroms) between input and predicted structure | < 2.0 A |
| `sc_tm` | Self-consistency TM-score (0--1) between input and predicted structure | > 0.5 |
| `plddt` | Mean predicted local distance difference test (0--100) from folding model | > 70 (ESMFold) / > 80 (AlphaFold2) |
| `ptm` | Predicted TM-score (0--1) from folding model | > 0.5 |
| `seq_recovery` | Fraction of positions matching the native sequence | -- |
| `mpnn_score` | ProteinMPNN negative log-likelihood score for the designed sequence | lower is better |

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pdb-dir` | (required) | Path to a PDB file or directory of PDB files |
| `--proteinmpnn-path` | (required) | Path to cloned ProteinMPNN repository |
| `--folding-model` | `esmfold` | Structure prediction model: `esmfold` or `alphafold2` |
| `--num-seqs` | `8` | Number of sequences to design per structure |
| `--sampling-temp` | `0.1` | ProteinMPNN sampling temperature |
| `--output-dir` | `./designability_results` | Output directory |
| `--num-recycles` | `4` | Recycling iterations for folding model |
| `--device` | `cuda` | Torch device for ESMFold |
| `--colabfold-path` | `colabfold_batch` | Path to ColabFold binary |
| `--mpnn-model-name` | `v_48_020` | ProteinMPNN model variant |
| `--mpnn-seed` | `0` | Random seed for ProteinMPNN |
| `--chunk-size` | `128` | ESMFold chunk size for memory optimization |
| `-v` / `--verbose` | off | Enable debug logging |

## References

- Dauparas et al. "Robust deep learning-based protein sequence design using ProteinMPNN." *Science* 378, 49-56 (2022).
- Lin et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science* 379, 1123-1130 (2023).
- Jumper et al. "Highly accurate protein structure prediction with AlphaFold." *Nature* 596, 583-589 (2021).
- Mirdita et al. "ColabFold: making protein folding accessible to all." *Nature Methods* 19, 679-682 (2022).

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
