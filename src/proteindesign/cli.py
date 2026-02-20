from __future__ import annotations

import argparse
import logging
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="proteindesign",
        description=(
            "Compute protein designability via inverse folding (ProteinMPNN) "
            "and structure prediction (ESMFold / AlphaFold2)."
        ),
    )
    parser.add_argument(
        "--pdb-dir",
        required=True,
        help="Path to a PDB file or directory of PDB files.",
    )
    parser.add_argument(
        "--folding-model",
        choices=["esmfold", "alphafold2"],
        default="esmfold",
        help="Structure prediction model (default: esmfold).",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=8,
        help="Number of sequences to design per structure (default: 8).",
    )
    parser.add_argument(
        "--sampling-temp",
        type=float,
        default=0.1,
        help="ProteinMPNN sampling temperature (default: 0.1).",
    )
    parser.add_argument(
        "--proteinmpnn-path",
        required=True,
        help="Path to cloned ProteinMPNN repository.",
    )
    parser.add_argument(
        "--output-dir",
        default="./designability_results",
        help="Output directory for results (default: ./designability_results).",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=4,
        help="Recycling iterations for folding model (default: 4).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for ESMFold (default: cuda).",
    )
    parser.add_argument(
        "--colabfold-path",
        default="colabfold_batch",
        help="Path to colabfold_batch binary (default: colabfold_batch).",
    )
    parser.add_argument(
        "--mpnn-model-name",
        default="v_48_020",
        help="ProteinMPNN model variant (default: v_48_020).",
    )
    parser.add_argument(
        "--mpnn-seed",
        type=int,
        default=0,
        help="Random seed for ProteinMPNN (default: 0).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="ESMFold chunk size for memory optimization (default: 128).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from proteindesign.pipeline import compute_designability

    df = compute_designability(
        pdb_path=args.pdb_dir,
        folding_model=args.folding_model,
        num_seqs=args.num_seqs,
        sampling_temp=args.sampling_temp,
        proteinmpnn_path=args.proteinmpnn_path,
        output_dir=args.output_dir,
        mpnn_model_name=args.mpnn_model_name,
        mpnn_seed=args.mpnn_seed,
        num_recycles=args.num_recycles,
        device=args.device,
        colabfold_path=args.colabfold_path,
        chunk_size=args.chunk_size,
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "designability_scores.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    print(f"\nResults saved to {csv_path}")
    if not df.empty:
        n_structures = df["pdb_name"].nunique()
        print(f"\nSummary ({len(df)} sequences across {n_structures} structures):")
        summary = df.groupby("pdb_name").agg(
            {
                "sc_rmsd": "mean",
                "sc_tm": "mean",
                "plddt": "mean",
                "ptm": "mean",
                "seq_recovery": "mean",
            }
        )
        print(summary.round(3).to_string())
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
