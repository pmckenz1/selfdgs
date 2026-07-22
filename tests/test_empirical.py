from collections import Counter
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from selfdgs.empirical import (
    DRAW_DIAGNOSTICS_COLUMNS,
    DRAW_FIT_SUMMARY_COLUMNS,
    DRAW_LIKELIHOOD_COLUMNS,
    DRAW_MANIFEST_COLUMNS,
    GROUP_FIT_SUMMARY_COLUMNS,
    REPRESENTATIVE_OBSERVED_VS_FITTED_COLUMNS,
    SAMPLE_AUDIT_COLUMNS,
    SITE_DIAGNOSTICS_COLUMNS,
    EmpiricalAnalysisConfig,
    aggregate_fit_summary,
    dgs_for_draws,
    fit_draws,
    make_draw_manifest,
    make_sample_groups,
    representative_observed_vs_fitted,
    run_empirical_analysis,
    write_draw_outputs,
)
from selfdgs.spectrum import read_dgs_csv

FIXTURES = Path(__file__).parent / "fixtures"


def test_make_sample_groups_supports_all_and_metadata_groups():
    metadata = pd.DataFrame(
        {
            "sample": ["ind1", "ind2", "ind3", "absent"],
            "species": ["a", "a", "b", "b"],
        }
    )

    audit, groups = make_sample_groups(
        ["ind1", "ind2", "ind3", "ind4"],
        mode="both",
        metadata=metadata,
        group_column="species",
        min_samples=2,
    )

    assert groups == {
        "all_samples": ["ind1", "ind2", "ind3", "ind4"],
        "a": ["ind1", "ind2"],
    }
    assert audit.set_index("group").loc["b", "included"] == False
    assert audit.set_index("group").loc["b", "missing_from_vcf_n"] == 1


@pytest.mark.parametrize(
    ("samples", "metadata", "message"),
    [
        (["ind1", "ind1"], None, "VCF sample IDs must be unique"),
        (
            ["ind1", "ind2"],
            pd.DataFrame({"sample": ["ind1", "ind1"], "species": ["a", "a"]}),
            "Metadata sample IDs must be unique",
        ),
        (
            ["ind1", "ind2"],
            pd.DataFrame({"sample": ["ind1", None], "species": ["a", "a"]}),
            "contain missing values",
        ),
        (
            ["ind1", "ind2"],
            pd.DataFrame(
                {"sample": ["ind1", "ind2"], "species": ["all_samples", "a"]}
            ),
            "collides with the all-samples group",
        ),
    ],
)
def test_make_sample_groups_rejects_ambiguous_inputs(samples, metadata, message):
    with pytest.raises(ValueError, match=message):
        make_sample_groups(
            samples,
            mode="all" if metadata is None else "both",
            metadata=metadata,
            group_column="species" if metadata is not None else None,
        )


def test_make_draw_manifest_is_deterministic_and_rejects_small_groups():
    groups = {"b": ["ind3", "ind4"], "a": ["ind1", "ind2", "ind3"]}

    first = make_draw_manifest(groups, n_diploids=2, n_draws=3, seed=7)
    second = make_draw_manifest(groups, n_diploids=2, n_draws=3, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first["group"].tolist() == ["a", "a", "a", "b", "b", "b"]
    assert first["draw_id"].tolist() == [0, 1, 2, 0, 1, 2]
    assert first["seed"].nunique() == len(first)
    for row in first.itertuples(index=False):
        expected = np.random.default_rng(int(row.seed)).choice(
            groups[row.group], size=2, replace=False
        )
        assert row.sample_ids.split(";") == expected.tolist()

    b_only = make_draw_manifest({"b": groups["b"]}, n_diploids=2, n_draws=3, seed=7)
    pd.testing.assert_frame_equal(
        first.loc[first["group"] == "b"].reset_index(drop=True),
        b_only,
    )

    with pytest.raises(ValueError, match="has only 2 samples"):
        make_draw_manifest({"small": ["ind1", "ind2"]}, n_diploids=3, n_draws=1, seed=1)


def test_dgs_for_draws_rejects_duplicate_draw_keys():
    manifest = pd.DataFrame(
        [
            {"group": "focal", "draw_id": 0, "sample_ids": "ind1;ind2"},
            {"group": "focal", "draw_id": 0, "sample_ids": "ind3;ind4"},
        ]
    )

    with pytest.raises(ValueError, match=r"duplicate \(group, draw_id\) keys"):
        dgs_for_draws(FIXTURES / "basic.vcf", manifest)


def test_empirical_config_uses_fixed_sample_policies_and_no_effective_size():
    config = EmpiricalAnalysisConfig(vcf_path="samples.vcf", n_diploids=4)

    assert config.missing == "skip-draw"
    assert config.malformed == "error"
    assert not hasattr(config, "N")

    invalid = EmpiricalAnalysisConfig(
        vcf_path=FIXTURES / "basic.vcf",
        n_diploids=4,
        missing="skip-genotype",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="must preserve the fixed draw size"):
        run_empirical_analysis(invalid)


def test_run_empirical_analysis_orchestrates_and_writes_outputs(tmp_path):
    result = run_empirical_analysis(
        EmpiricalAnalysisConfig(
            vcf_path=FIXTURES / "basic.vcf",
            n_diploids=4,
            n_draws=1,
            seed=7,
            mode="folded",
            s_grid=[0.0, 0.5],
            low_polymorphic_site_threshold=None,
            outdir=tmp_path,
        )
    )

    assert list(result.sample_audit["group"]) == ["all_samples"]
    assert len(result.draw_manifest) == 1
    assert (tmp_path / "group_fit_summary.csv").exists()
    assert (tmp_path / "representative_draw_observed_vs_fitted.csv").exists()


def test_dgs_for_draws_reports_outgroup_consensus_diagnostics():
    manifest = pd.DataFrame(
        [
            {
                "group": "focal",
                "draw_id": 0,
                "sample_ids": "foc1;foc2",
            }
        ]
    )

    counts_by_draw, site_diagnostics, draw_diagnostics = dgs_for_draws(
        FIXTURES / "outgroup_polarization.vcf",
        manifest,
        mode="outgroup-consensus",
        outgroup_sample_ids=["out1", "out2"],
        min_outgroup_called=2,
    )

    assert counts_by_draw[("focal", 0)] == Counter({(1, 1, 0): 1, (0, 1, 1): 1})
    site_metrics = site_diagnostics.set_index("metric")["value"].to_dict()
    assert site_metrics["outgroup_ancestral_ref"] == 2
    assert site_metrics["outgroup_ancestral_alt"] == 1
    assert site_metrics["outgroup_missing"] == 1
    assert site_metrics["outgroup_heterozygous"] == 1
    assert site_metrics["outgroup_polymorphic"] == 1
    assert site_metrics["outgroup_malformed"] == 1

    draw_metrics = draw_diagnostics.set_index(["group", "draw_id"]).loc[("focal", 0)]
    assert draw_metrics["retained_sites"] == 2
    assert draw_metrics["missing_genotype"] == 1


def test_empirical_outgroups_are_excluded_from_default_groups_and_rejected_in_explicit_groups():
    config = EmpiricalAnalysisConfig(
        vcf_path=FIXTURES / "outgroup_polarization.vcf",
        n_diploids=2,
        n_draws=1,
        mode="outgroup-consensus",
        outgroup_sample_ids=("out1", "out2"),
        min_outgroup_called=2,
        s_grid=[0.0, 0.5, 1.0],
        low_polymorphic_site_threshold=None,
    )

    result = run_empirical_analysis(config)

    assert set(result.draw_manifest.loc[0, "sample_ids"].split(";")) == {
        "foc1",
        "foc2",
    }
    with pytest.raises(ValueError, match="cannot also be focal"):
        run_empirical_analysis(
            config,
            sample_groups={"bad": ["foc1", "out1"]},
        )
    overlapping_manifest = pd.DataFrame(
        [{"group": "bad", "draw_id": 0, "sample_ids": "foc1;out1"}]
    )
    with pytest.raises(ValueError, match="Outgroups cannot also be focal"):
        dgs_for_draws(
            FIXTURES / "outgroup_polarization.vcf",
            overlapping_manifest,
            mode="outgroup-consensus",
            outgroup_sample_ids=["out1", "out2"],
        )


def test_missing_genotypes_by_default_affect_only_draws_containing_missing_samples():
    manifest = pd.DataFrame(
        [
            {"group": "ok", "draw_id": 0, "sample_ids": "ind1;ind3"},
            {"group": "missing", "draw_id": 0, "sample_ids": "ind2;ind4"},
        ]
    )

    counts_by_draw, _, draw_diagnostics = dgs_for_draws(
        FIXTURES / "missing.vcf",
        manifest,
    )

    assert counts_by_draw[("ok", 0)] == Counter({(1, 0, 1): 1})
    assert counts_by_draw[("missing", 0)] == Counter()
    diagnostics = draw_diagnostics.set_index(["group", "draw_id"])
    assert diagnostics.loc[("ok", 0), "retained_sites"] == 1
    assert diagnostics.loc[("missing", 0), "missing_genotype"] == 1


def test_fit_draws_and_aggregate_summary_on_small_counts():
    counts_by_draw = {
        ("a", 0): Counter({(0, 1, 1): 8, (1, 0, 1): 2}),
        ("a", 1): Counter({(0, 1, 1): 5, (1, 0, 1): 5}),
        ("b", 0): Counter(),
    }

    fit_summary, likelihood_curves, fit_counts_by_draw = fit_draws(
        counts_by_draw,
        n_diploids=2,
        mode="folded",
        s_grid=(s for s in [0.0, 0.5, 0.9]),
        low_polymorphic_site_threshold=0,
    )
    aggregate = aggregate_fit_summary(fit_summary)

    assert set(fit_summary["group"]) == {"a", "b"}
    assert set(fit_summary["mode"]) == {"folded"}
    assert set(fit_summary["input_polarization"]) == {"folded"}
    assert likelihood_curves[["group", "draw_id", "s", "loglik", "delta_loglik"]].shape[0] == 6
    assert fit_counts_by_draw[("b", 0)] == Counter()
    assert aggregate.loc[aggregate["group"] == "a", "n_fit_draws"].item() == 2
    assert aggregate.loc[aggregate["group"] == "b", "n_fit_draws"].item() == 0
    assert pd.isna(aggregate.loc[aggregate["group"] == "b", "median_best_s"].item())
    assert "median_best_s" in aggregate.columns


def test_fit_draws_reports_warning_flags():
    counts_by_draw = {
        ("low", 0): Counter({(0, 1, 1): 1}),
        ("empty", 0): Counter(),
    }

    fit_summary, _, _ = fit_draws(
        counts_by_draw,
        n_diploids=2,
        s_grid=[1.0],
        low_polymorphic_site_threshold=2,
    )
    warnings = fit_summary.set_index(["group", "draw_id"])["warnings"]

    assert "low polymorphic site count (<2)" in warnings.loc[("low", 0)]
    assert "all-nonfinite likelihoods" in warnings.loc[("low", 0)]
    assert pd.isna(
        fit_summary.set_index(["group", "draw_id"]).loc[("low", 0), "best_s"]
    )
    assert "unsupported DGS cells" in warnings.loc[("low", 0)]
    low = fit_summary.set_index(["group", "draw_id"]).loc[("low", 0)]
    assert low["boundary_optimum"] == False
    assert low["unsupported_cell_count"] == 1
    assert low["unsupported_site_count"] == 1
    assert "no fit sites" in warnings.loc[("empty", 0)]


def test_fit_draws_validates_fit_settings_for_empty_draws():
    empty = {("empty", 0): Counter()}
    with pytest.raises(ValueError, match="epsilon must be finite"):
        fit_draws(empty, n_diploids=2, epsilon=math.nan)
    with pytest.raises(ValueError, match="support_drop must be finite"):
        fit_draws(empty, n_diploids=2, support_drop=math.nan)


def test_fit_draws_reports_support_interval_grid_boundary():
    fit_summary, _, _ = fit_draws(
        {("a", 0): Counter({(0, 1, 1): 1})},
        n_diploids=2,
        s_grid=[0.0, 0.5, 1.0],
        low_polymorphic_site_threshold=0,
    )

    assert "support interval touches grid boundary" in fit_summary.loc[0, "warnings"]


def test_representative_observed_vs_fitted_uses_group_median_draw():
    counts_by_draw = {
        ("a", 0): Counter({(0, 1, 1): 8, (1, 0, 1): 2}),
        ("a", 1): Counter({(0, 1, 1): 5, (1, 0, 1): 5}),
    }
    fit_summary, _, fit_counts_by_draw = fit_draws(
        counts_by_draw,
        n_diploids=2,
        mode="folded",
        s_grid=[0.0, 0.5, 0.9],
        low_polymorphic_site_threshold=0,
    )

    observed_vs_fitted = representative_observed_vs_fitted(
        fit_summary,
        fit_counts_by_draw,
        n_diploids=2,
    )

    assert set(REPRESENTATIVE_OBSERVED_VS_FITTED_COLUMNS).issubset(observed_vs_fitted.columns)
    assert observed_vs_fitted["group"].unique().tolist() == ["a"]
    assert observed_vs_fitted["representative_reason"].unique().tolist() == [
        "closest_to_group_median_best_s"
    ]
    assert observed_vs_fitted["observed_count"].sum() == 10


def test_representative_observed_vs_fitted_applies_fit_regularization():
    counts = {("a", 0): Counter({(0, 1, 1): 1})}
    fit_summary, _, fit_counts = fit_draws(
        counts,
        n_diploids=2,
        s_grid=[1.0],
        epsilon=0.2,
        low_polymorphic_site_threshold=0,
    )

    table = representative_observed_vs_fitted(
        fit_summary,
        fit_counts,
        n_diploids=2,
    )

    unsupported = table.loc[(table[["n0", "n1", "n2"]] == (0, 1, 1)).all(axis=1)]
    assert unsupported["fitted_probability"].item() == pytest.approx(0.2)


def test_write_draw_outputs_creates_expected_files(tmp_path):
    counts_by_draw = {("a/b", 0): Counter({(0, 1, 1): 2})}
    fit_counts_by_draw = {("a/b", 0): Counter({(0, 1, 1): 2})}
    draw_manifest = pd.DataFrame([{"group": "a/b", "draw_id": 0, "sample_ids": "ind1;ind2"}])
    site_diagnostics = pd.DataFrame([{"metric": "records_seen", "value": 2}])
    draw_diagnostics = pd.DataFrame([{"group": "a/b", "draw_id": 0, "retained_sites": 2}])
    fit_summary = pd.DataFrame([{"group": "a/b", "draw_id": 0, "best_s": 0.5}])
    likelihood_curves = pd.DataFrame([{"group": "a/b", "draw_id": 0, "s": 0.5, "loglik": -1.0}])
    aggregate_summary = pd.DataFrame([{"group": "a/b", "median_best_s": 0.5}])
    representative = pd.DataFrame(
        [
            {
                "group": "a/b",
                "representative_draw_id": 0,
                "n0": 0,
                "n1": 1,
                "n2": 1,
                "observed_count": 2,
                "fitted_expected_count": 2.0,
                "fitted_probability": 1.0,
            }
        ]
    )
    sample_audit = pd.DataFrame([{"group": "a/b", "n_samples": 2}])

    write_draw_outputs(
        tmp_path,
        counts_by_draw=counts_by_draw,
        fit_counts_by_draw=fit_counts_by_draw,
        draw_manifest=draw_manifest,
        site_diagnostics=site_diagnostics,
        draw_diagnostics=draw_diagnostics,
        fit_summary=fit_summary,
        likelihood_curves=likelihood_curves,
        aggregate_summary=aggregate_summary,
        representative_observed_vs_fitted=representative,
        sample_audit=sample_audit,
    )

    expected = {
        "draw_manifest.csv",
        "sample_audit.csv",
        "site_diagnostics.csv",
        "draw_diagnostics.csv",
        "draw_fit_summary.csv",
        "draw_likelihood_curves.csv",
        "group_fit_summary.csv",
        "representative_draw_observed_vs_fitted.csv",
        "dgs/a%2Fb_draw_000_dgs.csv",
        "dgs/a%2Fb_draw_000_fit_dgs.csv",
    }
    assert expected == {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*.csv")}
    assert read_dgs_csv(tmp_path / "dgs" / "a%2Fb_draw_000_dgs.csv") == counts_by_draw[("a/b", 0)]

    assert set(SAMPLE_AUDIT_COLUMNS).issubset(pd.read_csv(tmp_path / "sample_audit.csv").columns)
    assert set(DRAW_MANIFEST_COLUMNS).issubset(pd.read_csv(tmp_path / "draw_manifest.csv").columns)
    assert set(SITE_DIAGNOSTICS_COLUMNS).issubset(
        pd.read_csv(tmp_path / "site_diagnostics.csv").columns
    )
    assert set(DRAW_DIAGNOSTICS_COLUMNS).issubset(
        pd.read_csv(tmp_path / "draw_diagnostics.csv").columns
    )
    assert set(DRAW_FIT_SUMMARY_COLUMNS).issubset(
        pd.read_csv(tmp_path / "draw_fit_summary.csv").columns
    )
    assert set(DRAW_LIKELIHOOD_COLUMNS).issubset(
        pd.read_csv(tmp_path / "draw_likelihood_curves.csv").columns
    )
    assert set(GROUP_FIT_SUMMARY_COLUMNS).issubset(
        pd.read_csv(tmp_path / "group_fit_summary.csv").columns
    )
    assert set(REPRESENTATIVE_OBSERVED_VS_FITTED_COLUMNS).issubset(
        pd.read_csv(tmp_path / "representative_draw_observed_vs_fitted.csv").columns
    )


def test_draw_output_labels_do_not_collide_and_old_draw_files_are_removed(tmp_path):
    common = {
        "draw_manifest": pd.DataFrame(),
        "site_diagnostics": pd.DataFrame(),
        "draw_diagnostics": pd.DataFrame(),
        "fit_summary": pd.DataFrame(),
        "likelihood_curves": pd.DataFrame(),
        "aggregate_summary": pd.DataFrame(),
    }
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("preserve me")
    write_draw_outputs(
        tmp_path,
        counts_by_draw={
            ("a/b", 0): Counter({(0, 1, 1): 1}),
            ("a_b", 0): Counter({(0, 1, 1): 2}),
        },
        fit_counts_by_draw={},
        **common,
    )

    encoded = tmp_path / "dgs" / "a%2Fb_draw_000_dgs.csv"
    literal = tmp_path / "dgs" / "a_b_draw_000_dgs.csv"
    assert read_dgs_csv(encoded) == Counter({(0, 1, 1): 1})
    assert read_dgs_csv(literal) == Counter({(0, 1, 1): 2})

    write_draw_outputs(
        tmp_path,
        counts_by_draw={("a_b", 0): Counter({(0, 1, 1): 2})},
        fit_counts_by_draw={},
        **common,
    )
    assert not encoded.exists()
    assert literal.exists()
    assert unrelated.read_text() == "preserve me"
