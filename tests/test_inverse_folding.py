from proteindesign.inverse_folding import parse_mpnn_fasta


def test_parse_mpnn_fasta(sample_mpnn_fasta):
    sequences = parse_mpnn_fasta(sample_mpnn_fasta)
    # Should skip the native (first entry) and parse 3 designed sequences
    assert len(sequences) == 3

    # Check first designed sequence
    s0 = sequences[0]
    assert s0.sequence == "AKVLS"
    assert abs(s0.score - 0.8765) < 1e-4
    assert abs(s0.global_score - 1.0234) < 1e-4
    assert abs(s0.seq_recovery - 0.6) < 1e-4
    assert s0.sample_index == 1
    assert abs(s0.temperature - 0.1) < 1e-4

    # Check last designed sequence
    s2 = sequences[2]
    assert s2.sequence == "AGVMS"
    assert s2.sample_index == 3
    assert abs(s2.seq_recovery - 0.8) < 1e-4


def test_parse_mpnn_fasta_chain_separator(tmp_path):
    """ProteinMPNN uses '/' for chain separation; we normalize to ':'."""
    fasta = tmp_path / "multi_chain.fa"
    fasta.write_text(
        ">native, score=1.0, global_score=1.0\n"
        "AGVLS/DKWMF\n"
        ">T=0.1, sample=1, score=0.5, global_score=0.6, seq_recovery=0.5\n"
        "AKVLS/DGILS\n"
    )
    sequences = parse_mpnn_fasta(fasta)
    assert len(sequences) == 1
    assert sequences[0].sequence == "AKVLS:DGILS"
