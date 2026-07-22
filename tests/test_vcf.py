from collections import Counter
import gzip
from pathlib import Path

import pytest
from cyvcf2 import VCF, Writer

from selfdgs.io.vcf import (
    VCFParseError,
    dgs_from_vcf,
    read_vcf_samples,
    scan_vcf_dgs,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_vcf_header_and_default_folded_counts():
    path = FIXTURES / "basic.vcf"

    assert read_vcf_samples(path) == ["ind1", "ind2", "ind3", "ind4"]
    assert dgs_from_vcf(path, n_diploids=4) == Counter(
        {(1, 2, 1): 2, (0, 0, 4): 2}
    )


def test_compressed_vcf_and_bcf_inputs_match_plain_vcf(tmp_path):
    plain = FIXTURES / "basic.vcf"
    compressed = tmp_path / "basic.vcf.gz"
    with gzip.open(compressed, "wt") as handle:
        handle.write(plain.read_text())

    bcf = tmp_path / "basic.bcf"
    reader = VCF(str(plain))
    writer = Writer(str(bcf), reader, mode="wb")
    try:
        for record in reader:
            writer.write_record(record)
    finally:
        writer.close()
        reader.close()

    expected = dgs_from_vcf(plain, n_diploids=4)
    assert dgs_from_vcf(compressed, n_diploids=4) == expected
    assert dgs_from_vcf(bcf, n_diploids=4) == expected


def test_vcf_monomorphic_and_ref_polarization_options():
    path = FIXTURES / "basic.vcf"

    assert dgs_from_vcf(
        path, n_diploids=4, include_monomorphic=False
    ) == Counter({(1, 2, 1): 2})
    assert dgs_from_vcf(path, n_diploids=4, polarization="ref") == Counter(
        {(1, 2, 1): 2, (4, 0, 0): 1, (0, 0, 4): 1}
    )


def test_vcf_ancestral_allele_polarization():
    assert dgs_from_vcf(
        FIXTURES / "aa_polarization.vcf",
        n_diploids=2,
        polarization="aa",
    ) == Counter({(1, 1, 0): 1, (0, 1, 1): 1})


def test_vcf_outgroup_consensus_polarization():
    assert dgs_from_vcf(
        FIXTURES / "outgroup_polarization.vcf",
        n_diploids=2,
        polarization="outgroup-consensus",
        outgroup_sample_ids=["out1", "out2"],
        min_outgroup_called=2,
    ) == Counter({(1, 1, 0): 1, (0, 1, 1): 1})


def test_vcf_rejects_duplicate_outgroup_samples():
    with pytest.raises(ValueError, match="outgroup_sample_ids contains duplicate"):
        dgs_from_vcf(
            FIXTURES / "outgroup_polarization.vcf",
            n_diploids=2,
            polarization="outgroup-consensus",
            outgroup_sample_ids=["out1", "out1"],
        )


def test_vcf_missing_genotype_policies():
    path = FIXTURES / "missing.vcf"

    assert dgs_from_vcf(path, n_diploids=4) == Counter()
    assert dgs_from_vcf(
        path, n_diploids=4, missing="skip-genotype"
    ) == Counter({(1, 1, 1): 1})
    with pytest.raises(VCFParseError, match="Missing genotype"):
        dgs_from_vcf(path, n_diploids=4, missing="error")


def test_vcf_rejects_malformed_genotypes_and_sample_size_mismatch():
    with pytest.raises(VCFParseError, match="Malformed genotype"):
        dgs_from_vcf(FIXTURES / "malformed.vcf", n_diploids=4)
    with pytest.raises(VCFParseError, match="header has 4 samples"):
        dgs_from_vcf(FIXTURES / "basic.vcf", n_diploids=3)


def test_vcf_multiallelic_error_policy():
    with pytest.raises(VCFParseError, match="Multiallelic"):
        dgs_from_vcf(
            FIXTURES / "basic.vcf", n_diploids=4, multiallelic="error"
        )


def test_shared_scanner_filters_records_and_handles_multiple_sample_sets():
    path = FIXTURES / "filtering.vcf"
    scan = scan_vcf_dgs(
        path,
        sample_sets={
            ("all", 0): ["ind1", "ind2", "ind3", "ind4"],
            ("subset", 0): ["ind1", "ind2"],
        },
    )

    assert {key: sum(counts.values()) for key, counts in scan.counts_by_set.items()} == {
        ("all", 0): 1,
        ("subset", 0): 1,
    }
    assert scan.site_diagnostics["records_seen"] == 4
    assert scan.site_diagnostics["nonpass_filter"] == 1
    assert scan.site_diagnostics["non_biallelic_snp"] == 2
    assert sum(
        dgs_from_vcf(path, n_diploids=4, require_pass=False).values()
    ) == 2
