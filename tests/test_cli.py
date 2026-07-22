import json
from pathlib import Path

import pandas as pd
import pytest

from selfdgs.cli import build_parser, main

FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_dgs_writes_tidy_counts(tmp_path):
    out = tmp_path / "observed_dgs.csv"

    status = main(["dgs", str(FIXTURES / "basic.vcf"), "--n-diploids", "4", "--out", str(out)])

    assert status == 0
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["n0", "n1", "n2", "count"]
    assert frame["count"].sum() == 4


def test_cli_allow_heterozygous_outgroup_is_effective_by_itself(tmp_path):
    out = tmp_path / "observed_dgs.csv"

    status = main(
        [
            "dgs",
            str(FIXTURES / "outgroup_polarization.vcf"),
            "--n-diploids",
            "2",
            "--polarization",
            "outgroup-consensus",
            "--outgroup-samples",
            "out1,out2",
            "--min-outgroup-called",
            "2",
            "--allow-heterozygous-outgroup",
            "--out",
            str(out),
        ]
    )

    assert status == 0
    assert pd.read_csv(out)["count"].sum() == 3


def test_cli_fit_allow_heterozygous_outgroup_is_effective_by_itself(tmp_path):
    outdir = tmp_path / "fit"

    status = main(
        [
            "fit",
            str(FIXTURES / "outgroup_polarization.vcf"),
            "--n-diploids",
            "2",
            "--polarization",
            "outgroup-consensus",
            "--outgroup-samples",
            "out1,out2",
            "--min-outgroup-called",
            "2",
            "--allow-heterozygous-outgroup",
            "--grid",
            "0,0.5",
            "--error-rate",
            "0.01",
            "--out",
            str(outdir),
        ]
    )

    assert status == 0
    assert pd.read_csv(outdir / "observed_dgs.csv")["count"].sum() == 3


def test_cli_fit_writes_expected_outputs_from_vcf(tmp_path):
    outdir = tmp_path / "fit"
    outdir.mkdir()
    (outdir / "stale.txt").write_text("old output")

    status = main(
        [
            "fit",
            str(FIXTURES / "basic.vcf"),
            "--n-diploids",
            "4",
            "--grid",
            "0:1:3",
            "--out",
            str(outdir),
        ]
    )

    assert status == 0
    assert (outdir / "stale.txt").read_text() == "old output"
    assert (outdir / "observed_dgs.csv").exists()
    assert (outdir / "fit_dgs.csv").exists()
    assert (outdir / "fit_result.json").exists()
    assert (outdir / "fit_summary.csv").exists()
    assert (outdir / "likelihood.csv").exists()

    payload = json.loads((outdir / "fit_result.json").read_text())
    assert payload["n_diploids"] == 4
    assert payload["n_sites"] == 2
    assert payload["n_input_sites"] == 4
    assert payload["n_excluded_sites"] == 2
    assert payload["n_excluded_cells"] == 1
    assert payload["best_loglik"] is not None
    assert payload["metadata"]["site_conditioning"] == "polymorphic"
    assert payload["metadata"]["mode"] == "folded"
    assert payload["metadata"]["input_polarization"] == "folded"
    assert "n_input_sites" not in payload["metadata"]
    assert "n_fit_sites" not in payload["metadata"]
    assert len(payload["likelihood_curve"]) == 3

    likelihood = pd.read_csv(outdir / "likelihood.csv")
    assert list(likelihood.columns) == ["s", "loglik", "delta_loglik"]
    assert {path.name for path in outdir.iterdir()} == {
        "stale.txt",
        "observed_dgs.csv",
        "fit_dgs.csv",
        "fit_result.json",
        "fit_summary.csv",
        "likelihood.csv",
        "site_diagnostics.csv",
    }


def test_cli_fit_rejects_variable_sample_size_policies(tmp_path):
    parser = build_parser()

    for option in ("--missing", "--malformed"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "fit",
                    str(FIXTURES / "basic.vcf"),
                    option,
                    "skip-genotype",
                    "--out",
                    str(tmp_path / "fit"),
                ]
            )


def test_fit_commands_do_not_expose_effective_size_option(tmp_path):
    parser = build_parser()

    for command in ("fit", "empirical"):
        arguments = [command, str(FIXTURES / "basic.vcf")]
        if command == "empirical":
            arguments.extend(["--n-diploids", "4"])
        arguments.extend(["--N", "2", "--out", str(tmp_path / command)])
        with pytest.raises(SystemExit):
            parser.parse_args(arguments)


def test_cli_fit_and_empirical_share_filtering_and_counting(tmp_path):
    fit_out = tmp_path / "fit"
    empirical_out = tmp_path / "empirical"

    assert main(
        [
            "fit",
            str(FIXTURES / "filtering.vcf"),
            "--n-diploids",
            "4",
            "--grid",
            "0,0.5,0.9",
            "--out",
            str(fit_out),
        ]
    ) == 0
    assert main(
        [
            "empirical",
            str(FIXTURES / "filtering.vcf"),
            "--n-diploids",
            "4",
            "--n-draws",
            "1",
            "--grid",
            "0,0.5,0.9",
            "--out",
            str(empirical_out),
        ]
    ) == 0

    fit_counts = pd.read_csv(fit_out / "observed_dgs.csv")
    empirical_counts = pd.read_csv(
        empirical_out / "dgs" / "all_samples_draw_000_dgs.csv"
    )
    pd.testing.assert_frame_equal(fit_counts, empirical_counts)
    fit_diagnostics = pd.read_csv(fit_out / "site_diagnostics.csv")
    empirical_diagnostics = pd.read_csv(empirical_out / "site_diagnostics.csv")
    pd.testing.assert_frame_equal(fit_diagnostics, empirical_diagnostics)
    metrics = fit_diagnostics.set_index("metric")["value"]
    assert metrics["nonpass_filter"] == 1
    assert metrics["non_biallelic_snp"] == 2


def test_cli_fit_accepts_dgs_csv_input(tmp_path):
    dgs_csv = tmp_path / "observed_dgs.csv"
    outdir = tmp_path / "fit"
    main(["dgs", str(FIXTURES / "basic.vcf"), "--out", str(dgs_csv)])

    status = main(
        [
            "fit",
            str(dgs_csv),
            "--dgs-csv",
            "--grid",
            "0,0.5,1",
            "--out",
            str(outdir),
        ]
    )

    assert status == 0
    payload = json.loads((outdir / "fit_result.json").read_text())
    assert payload["n_sites"] == 2
    assert payload["n_input_sites"] == 4
    assert pd.read_csv(outdir / "observed_dgs.csv")["count"].sum() == 4
    assert pd.read_csv(outdir / "fit_dgs.csv")["count"].sum() == 2
    assert pd.read_csv(outdir / "site_diagnostics.csv").to_dict(orient="records") == [
        {"metric": "input_dgs_csv", "value": 1}
    ]
    assert {path.name for path in outdir.iterdir()} == {
        "observed_dgs.csv",
        "fit_dgs.csv",
        "fit_result.json",
        "fit_summary.csv",
        "likelihood.csv",
        "site_diagnostics.csv",
    }


def test_cli_empirical_all_sample_fixture_writes_expected_outputs(tmp_path):
    outdir = tmp_path / "empirical"

    status = main(
        [
            "empirical",
            str(FIXTURES / "basic.vcf"),
            "--n-diploids",
            "4",
            "--n-draws",
            "1",
            "--grid",
            "0:1:3",
            "--out",
            str(outdir),
        ]
    )

    assert status == 0
    expected = {
        "sample_audit.csv",
        "draw_manifest.csv",
        "site_diagnostics.csv",
        "draw_diagnostics.csv",
        "draw_fit_summary.csv",
        "draw_likelihood_curves.csv",
        "group_fit_summary.csv",
        "representative_draw_observed_vs_fitted.csv",
        "dgs/all_samples_draw_000_dgs.csv",
        "dgs/all_samples_draw_000_fit_dgs.csv",
    }
    assert expected == {str(path.relative_to(outdir)) for path in outdir.rglob("*.csv")}
    manifest = pd.read_csv(outdir / "draw_manifest.csv")
    assert set(manifest.loc[0, "sample_ids"].split(";")) == {
        "ind1",
        "ind2",
        "ind3",
        "ind4",
    }
    fit_summary = pd.read_csv(outdir / "draw_fit_summary.csv")
    assert fit_summary.loc[0, "group"] == "all_samples"
    assert fit_summary.loc[0, "mode"] == "folded"
    assert fit_summary.loc[0, "input_polarization"] == "folded"


def test_cli_empirical_metadata_grouping_fixture(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "Sample_ID": ["ind1", "ind2", "ind3", "ind4"],
            "species_v2": ["a", "a", "b", "b"],
        }
    ).to_csv(metadata, index=False)
    outdir = tmp_path / "empirical"

    status = main(
        [
            "empirical",
            str(FIXTURES / "basic.vcf"),
            "--metadata",
            str(metadata),
            "--sample-column",
            "Sample_ID",
            "--group-column",
            "species_v2",
            "--n-diploids",
            "2",
            "--n-draws",
            "1",
            "--grid",
            "0,0.5,1",
            "--out",
            str(outdir),
        ]
    )

    assert status == 0
    assert set(pd.read_csv(outdir / "draw_manifest.csv")["group"]) == {"a", "b"}
    assert set(pd.read_csv(outdir / "group_fit_summary.csv")["group"]) == {"a", "b"}


def test_cli_empirical_preserves_missing_metadata_values(tmp_path, capsys):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "sample": ["ind1", "ind2", "ind3", "ind4"],
            "group": ["a", "a", None, "b"],
        }
    ).to_csv(metadata, index=False)

    status = main(
        [
            "empirical",
            str(FIXTURES / "basic.vcf"),
            "--metadata",
            str(metadata),
            "--group-column",
            "group",
            "--n-diploids",
            "2",
            "--out",
            str(tmp_path / "empirical"),
        ]
    )

    assert status == 2
    assert "contain missing values" in capsys.readouterr().err


def test_cli_empirical_accepts_explicit_sample_list_files(tmp_path):
    sample_list = tmp_path / "focal_samples.txt"
    sample_list.write_text("ind1\nind2\n")
    outdir = tmp_path / "empirical"

    status = main(
        [
            "empirical",
            str(FIXTURES / "basic.vcf"),
            "--sample-list",
            f"focal={sample_list}",
            "--n-diploids",
            "2",
            "--n-draws",
            "1",
            "--grid",
            "0,0.5,1",
            "--out",
            str(outdir),
        ]
    )

    assert status == 0
    assert pd.read_csv(outdir / "draw_manifest.csv").loc[0, "group"] == "focal"


def test_cli_empirical_rejects_repeated_explicit_group_names(tmp_path, capsys):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("ind1\nind2\n")
    second.write_text("ind3\nind4\n")

    status = main(
        [
            "empirical",
            str(FIXTURES / "basic.vcf"),
            "--sample-list",
            f"focal={first}",
            "--sample-list",
            f"focal={second}",
            "--n-diploids",
            "2",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert status == 2
    assert "group name 'focal' is repeated" in capsys.readouterr().err


def test_cli_empirical_outgroup_consensus_fixture(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "Sample_ID": ["foc1", "foc2", "out1", "out2"],
            "species_v2": ["focal", "focal", "outgroup", "outgroup"],
            "species": ["focal", "focal", "pilo", "pilo"],
        }
    ).to_csv(metadata, index=False)
    outdir = tmp_path / "empirical"

    status = main(
        [
            "empirical",
            str(FIXTURES / "outgroup_polarization.vcf"),
            "--metadata",
            str(metadata),
            "--sample-column",
            "Sample_ID",
            "--group-column",
            "species_v2",
            "--outgroup-column",
            "species",
            "--outgroup-value",
            "pilo",
            "--mode",
            "outgroup-consensus",
            "--min-outgroup-called",
            "2",
            "--n-diploids",
            "2",
            "--n-draws",
            "1",
            "--grid",
            "0,0.5,1",
            "--out",
            str(outdir),
        ]
    )

    assert status == 0
    assert pd.read_csv(outdir / "draw_manifest.csv")["group"].tolist() == ["focal"]
    diagnostics = pd.read_csv(outdir / "site_diagnostics.csv").set_index("metric")["value"]
    assert diagnostics["outgroup_ancestral_ref"] == 2
    assert diagnostics["outgroup_ancestral_alt"] == 1
    fit_summary = pd.read_csv(outdir / "draw_fit_summary.csv")
    assert fit_summary.loc[0, "input_polarization"] == "outgroup-consensus"


def test_cli_empirical_direct_outgroups_are_not_sampled_as_focal(tmp_path):
    outdir = tmp_path / "empirical"

    status = main(
        [
            "empirical",
            str(FIXTURES / "outgroup_polarization.vcf"),
            "--mode",
            "outgroup-consensus",
            "--outgroup-samples",
            "out1,out2",
            "--min-outgroup-called",
            "2",
            "--n-diploids",
            "2",
            "--n-draws",
            "1",
            "--grid",
            "0,0.5,1",
            "--out",
            str(outdir),
        ]
    )

    assert status == 0
    manifest = pd.read_csv(outdir / "draw_manifest.csv")
    assert set(manifest.loc[0, "sample_ids"].split(";")) == {"foc1", "foc2"}


def test_cli_empirical_audit_only_preserves_existing_analysis_outputs(tmp_path):
    outdir = tmp_path / "analysis"
    dgs_dir = outdir / "dgs"
    dgs_dir.mkdir(parents=True)
    previous_summary = outdir / "draw_fit_summary.csv"
    previous_dgs = dgs_dir / "previous.csv"
    previous_summary.write_text("previous summary\n")
    previous_dgs.write_text("previous dgs\n")

    status = main(
        [
            "empirical",
            str(FIXTURES / "basic.vcf"),
            "--n-diploids",
            "4",
            "--audit-only",
            "--out",
            str(outdir),
        ]
    )

    assert status == 0
    assert (outdir / "sample_audit.csv").exists()
    assert (outdir / "draw_manifest.csv").exists()
    assert previous_summary.read_text() == "previous summary\n"
    assert previous_dgs.read_text() == "previous dgs\n"


def test_cli_empirical_reports_missing_metadata_columns_without_traceback(tmp_path, capsys):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"sample": ["ind1", "ind2"]}).to_csv(metadata, index=False)

    status = main(
        [
            "empirical",
            str(FIXTURES / "basic.vcf"),
            "--metadata",
            str(metadata),
            "--group-column",
            "species",
            "--n-diploids",
            "2",
            "--out",
            str(tmp_path / "empirical"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "Metadata is missing required columns" in captured.err
    assert "Traceback" not in captured.err


def test_failed_empirical_rerun_preserves_previous_outputs(tmp_path):
    outdir = tmp_path / "empirical"
    outdir.mkdir()
    previous = outdir / "group_fit_summary.csv"
    previous.write_text("previous\n")
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"sample": ["ind1"]}).to_csv(metadata, index=False)

    status = main(
        [
            "empirical",
            str(FIXTURES / "basic.vcf"),
            "--metadata",
            str(metadata),
            "--group-column",
            "missing_group",
            "--n-diploids",
            "2",
            "--out",
            str(outdir),
        ]
    )

    assert status == 2
    assert previous.read_text() == "previous\n"


def test_cli_probabilities_writes_model_table(tmp_path):
    out = tmp_path / "probabilities.csv"

    status = main(
        [
            "probabilities",
            "--selfing-rate",
            "0.5",
            "--n-diploids",
            "2",
            "--out",
            str(out),
        ]
    )

    assert status == 0
    frame = pd.read_csv(out)
    assert list(frame.columns) == [
        "n0",
        "n1",
        "n2",
        "probability",
        "expected_branch_length",
    ]
    assert frame["probability"].sum() == pytest.approx(1.0)


def test_cli_reports_vcf_errors_without_traceback(tmp_path, capsys):
    out = tmp_path / "bad.csv"

    status = main(
        [
            "dgs",
            str(FIXTURES / "missing.vcf"),
            "--n-diploids",
            "4",
            "--missing",
            "error",
            "--out",
            str(out),
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "Missing genotype" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()


def test_cli_reports_bad_grid_without_traceback(tmp_path, capsys):
    outdir = tmp_path / "fit"
    status = main(
        [
            "fit",
            str(FIXTURES / "basic.vcf"),
            "--n-diploids",
            "4",
            "--grid",
            "bad-grid",
            "--out",
            str(outdir),
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "grid must be" in captured.err
    assert "Traceback" not in captured.err
    assert not outdir.exists()


def test_cli_simulate_accepts_n_diploids_alias(tmp_path, monkeypatch):
    captured_configs = []

    def fake_run_validation_experiment(config):
        captured_configs.append(config)
        return ()

    monkeypatch.setattr("selfdgs.cli.run_validation_experiment", fake_run_validation_experiment)

    status = main(
        [
            "simulate",
            "--out",
            str(tmp_path / "validation"),
            "--n-diploids",
            "6",
            "--n-reps",
            "1",
            "--n-loci",
            "1",
        ]
    )

    assert status == 0
    assert captured_configs[0].n_sample == 6
    assert captured_configs[0].slim_script.name == "equilibrium_selfing.slim"
