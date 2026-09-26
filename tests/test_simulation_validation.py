from pathlib import Path
import json
import math
import shutil
import subprocess

import pandas as pd
import pytest

from selfdgs.simulation import (
    SLiMUnavailableError,
    SlimRunResult,
    SlimSimulationConfig,
    build_slim_command,
    default_slim_script_path,
    run_slim_simulation,
)
from selfdgs.simulation import slim as slim_module
from selfdgs.validation.experiments import (
    ValidationExperimentConfig,
    run_validation_experiment,
    summarize_existing_vcfs,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_slim_command_matches_expected_defines(tmp_path):
    config = SlimSimulationConfig(
        slim_script=default_slim_script_path("independent_populations"),
        vcf_path=tmp_path / "out.vcf",
        ne=100,
        selfing_rate=0.25,
        mu=1e-8,
        chrom_length=1000,
        recomb_rate=2e-8,
        n_sample=4,
        burn_mult=5,
        seed=123,
        slim_executable="slim",
    )

    command = build_slim_command(config)

    assert command[0] == "slim"
    assert "-d" in command
    assert "ne=100" in command
    assert "selfing_rate=0.25" in command
    assert "vcf_path=" + f'"{(tmp_path / "out.vcf").as_posix()}"' in command
    assert command[-1].endswith("equilibrium_selfing.slim")


def test_build_slim_command_escapes_string_defines(tmp_path):
    config = SlimSimulationConfig(
        slim_script=default_slim_script_path("independent_populations"),
        vcf_path=tmp_path / "out.vcf",
        extra_defines={"label": 'a\\b"c'},
    )

    assert 'label="a\\\\b\\"c"' in build_slim_command(config)


def test_build_slim_command_formats_boolean_defines_for_eidos(tmp_path):
    config = SlimSimulationConfig(
        slim_script=default_slim_script_path("independent_populations"),
        vcf_path=tmp_path / "out.vcf",
        extra_defines={"enabled": True, "disabled": False},
    )

    command = build_slim_command(config)

    assert "enabled=T" in command
    assert "disabled=F" in command


def test_slim_config_rejects_invalid_values(tmp_path):
    cases = [
        ({"extra_defines": {"not-valid": 1}}, "invalid Eidos identifiers"),
        ({"extra_defines": {"missing": None}}, "cannot be None"),
        ({"ne": 1.5}, "positive integers"),
        ({"burn_mult": 1.5}, "non-negative integer"),
        ({"selfing_rate": math.nan}, "selfing_rate must be finite"),
        ({"mu": math.inf}, "mu must be finite"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            SlimSimulationConfig(
                slim_script=default_slim_script_path("independent_populations"),
                vcf_path=tmp_path / "out.vcf",
                **kwargs,
            )


def test_slim_version_lookup_is_cached(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args[0], 0, stdout="SLiM 4.3\n", stderr="")

    slim_module._slim_version.cache_clear()
    monkeypatch.setattr(slim_module.subprocess, "run", fake_run)

    assert slim_module._slim_version("slim-for-cache-test") == "SLiM 4.3"
    assert slim_module._slim_version("slim-for-cache-test") == "SLiM 4.3"
    assert len(calls) == 1


def test_build_slim_command_rejects_reserved_extra_defines(tmp_path):
    for reserved in ("seed", "ne", "n_sample", "vcf_path"):
        config = SlimSimulationConfig(
            slim_script=default_slim_script_path("independent_populations"),
            vcf_path=tmp_path / "out.vcf",
            extra_defines={reserved: "contradiction"},
        )
        with pytest.raises(
            ValueError,
            match="cannot override dedicated simulation settings",
        ):
            build_slim_command(config)


def test_slim_configuration_and_script_conventions(tmp_path):
    """Lock the simulation settings and biological conventions in the Methods."""
    config = SlimSimulationConfig(
        slim_script=default_slim_script_path("independent_populations"),
        vcf_path=tmp_path / "simulation.vcf",
        ne=100,
        selfing_rate=0.7,
        mu=1e-7,
        chrom_length=1_000,
        recomb_rate=5e-8,
        n_sample=3,
        burn_mult=10,
        seed=20260704,
    )
    command = build_slim_command(config)
    joined = " ".join(command)
    for define in (
        "ne=100",
        "selfing_rate=0.7",
        "mu=1e-07",
        "chrom_length=1000",
        "recomb_rate=5e-08",
        "n_sample=3",
        "burn_mult=10",
        "seed=20260704",
    ):
        assert define in joined

    script = default_slim_script_path("independent_populations").read_text()
    for statement in (
        'initializeMutationType("m1", 0.5, "f", 0.0);',
        "defineConstant(\"burnin\", asInteger(burn_mult * ne));",
        "initializeSLiMOptions(preventIncidentalSelfing=T);",
        "p1.setSelfingRate(selfing_rate);",
        "p1.outputVCFSample(",
        "replace=F",
        "outputMultiallelics=F",
        'if (!exists("n_sample_2"))',
        "if (n_sample_2 > 0)",
        "filePath=vcf_path_2",
    ):
        assert statement in script


def test_run_slim_simulation_fails_clearly_when_slim_is_unavailable(tmp_path):
    config = SlimSimulationConfig(
        slim_script=default_slim_script_path("independent_populations"),
        vcf_path=tmp_path / "out.vcf",
        slim_executable="definitely-not-slim",
    )

    with pytest.raises(SLiMUnavailableError, match="not found on PATH"):
        run_slim_simulation(config)


@pytest.mark.slim
@pytest.mark.skipif(shutil.which("slim") is None, reason="SLiM is not installed")
def test_optional_slim_integration_smoke(tmp_path):
    config = SlimSimulationConfig(
        slim_script=default_slim_script_path("independent_populations"),
        vcf_path=tmp_path / "out.vcf",
        ne=10,
        selfing_rate=0.1,
        mu=1e-6,
        chrom_length=100,
        recomb_rate=1e-8,
        n_sample=2,
        burn_mult=1,
        seed=1,
    )

    result = run_slim_simulation(config)

    assert result.returncode == 0
    assert Path(result.vcf_path).exists()


def test_summarize_existing_vcfs_returns_pooled_fit():
    result = summarize_existing_vcfs(
        [FIXTURES / "basic.vcf", FIXTURES / "basic.vcf"],
        true_s=0.5,
        ne=500,
        n_diploids=4,
        s_grid=[0.0, 0.5, 1.0],
    )

    assert result.rep == 0
    assert result.true_s == 0.5
    assert len(result.loci) == 2
    assert sum(result.observed_counts.values()) == 8
    assert result.fit.n_diploids == 4
    assert result.fit.metadata["mode"] == "folded"
    assert result.fit.metadata["input_polarization"] == "folded"
    assert len(result.fit.likelihood_curve) == 3


def test_validation_rejects_variable_sample_size_policies():
    for policy_name in ("missing", "malformed"):
        kwargs = {f"vcf_{policy_name}": "skip-genotype"}
        with pytest.raises(ValueError, match="must preserve n_sample"):
            ValidationExperimentConfig(
                slim_script=None,
                outdir="validation",
                **kwargs,
            )

        with pytest.raises(ValueError, match="must preserve n_sample"):
            summarize_existing_vcfs(
                [FIXTURES / "missing.vcf"],
                n_diploids=4,
                s_grid=[0.0, 0.5],
                **{policy_name: "skip-genotype"},
            )


def test_validation_config_rejects_empty_experiments():
    with pytest.raises(ValueError, match="n_reps"):
        ValidationExperimentConfig(
            slim_script=None,
            outdir="validation",
            n_reps=0,
        )


def test_validation_config_rejects_non_integer_settings():
    cases = [
        ({"n_reps": 1.5}, "positive integers"),
        ({"burn_mult": 1.5}, "non-negative integer"),
        ({"seed": 1.5}, "seed must be a positive integer"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            ValidationExperimentConfig(
                slim_script=None,
                outdir="validation",
                **kwargs,
            )


def test_run_validation_experiment_with_mocked_slim(tmp_path, monkeypatch):
    source_vcf = FIXTURES / "basic.vcf"

    def fake_run_slim(config):
        Path(config.vcf_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config.vcf_path).write_text(source_vcf.read_text())
        return SlimRunResult(
            command=tuple(build_slim_command(config)),
            returncode=0,
            stdout="",
            stderr="",
            vcf_path=str(config.vcf_path),
            metadata=config.metadata(),
        )

    monkeypatch.setattr(
        "selfdgs.validation.experiments.run_slim_simulation",
        fake_run_slim,
    )

    config = ValidationExperimentConfig(
        slim_script=None,
        outdir=tmp_path / "validation",
        ne=500,
        true_s=0.5,
        n_independent_loci=2,
        sampling_design="independent_populations",
        n_reps=1,
        n_sample=4,
        s_grid=[0.0, 0.5, 1.0],
        seed=1,
    )
    stale_rep = Path(config.outdir) / "rep999_seed_old"
    stale_rep.mkdir(parents=True)
    (stale_rep / "likelihood.csv").write_text("stale")

    results = run_validation_experiment(config)

    assert len(results) == 1
    assert len(results[0].loci) == 2
    assert not stale_rep.exists()
    assert (tmp_path / "validation" / "validation_summary.csv").exists()
    config_payload = json.loads(
        (tmp_path / "validation" / "validation_config.json").read_text()
    )
    assert config_payload["census_size"] == 500
    assert config_payload["n_sample"] == 4
    assert config_payload["resolved_s_grid"] == [0.0, 0.5, 1.0]
    assert config_payload["resolved_fit_mode"] == "folded"
    rep_dirs = sorted((tmp_path / "validation").glob("rep*"))
    assert len(rep_dirs) == 1
    assert rep_dirs[0].name == "rep000"
    assert (rep_dirs[0] / "observed_dgs.csv").exists()
    assert (rep_dirs[0] / "observed_polymorphic_dgs.csv").exists()
    assert (rep_dirs[0] / "fit_result.json").exists()
    assert (rep_dirs[0] / "likelihood.csv").exists()
    assert (rep_dirs[0] / "vcfs.csv").exists()
    loci = pd.read_csv(rep_dirs[0] / "vcfs.csv")
    assert loci["slim_command"].notna().all()
    assert loci["slim_metadata"].notna().all()
    for row in loci.itertuples(index=False):
        assert json.loads(row.slim_metadata)["vcf_path"] == row.vcf_path
    assert all(Path(path).exists() for path in loci["slim_stdout_path"])
    assert all(Path(path).exists() for path in loci["slim_stderr_path"])

    summary = pd.read_csv(tmp_path / "validation" / "validation_summary.csv")
    assert summary.loc[0, "census_size"] == 500
    assert summary.loc[0, "n_diploids"] == 4
    assert summary.loc[0, "seed"] == 1
    assert summary.loc[0, "polarization"] == "folded"
    assert summary.loc[0, "fit_mode"] == "folded"


def test_failed_validation_rerun_preserves_previous_outputs(tmp_path, monkeypatch):
    outdir = tmp_path / "validation"
    outdir.mkdir()
    previous = outdir / "validation_summary.csv"
    previous.write_text("previous\n")

    def fail_slim(config):
        raise RuntimeError("simulated SLiM failure")

    monkeypatch.setattr(
        "selfdgs.validation.experiments.run_slim_simulation",
        fail_slim,
    )
    config = ValidationExperimentConfig(
        slim_script=None,
        outdir=outdir,
        n_reps=1,
        n_independent_loci=1,
    )

    with pytest.raises(RuntimeError, match="simulated SLiM failure"):
        run_validation_experiment(config)

    assert previous.read_text() == "previous\n"


@pytest.mark.parametrize("design,expected_calls", [("fixed_individuals", 2), ("independent_populations", 6)])
def test_validation_sampling_designs_preserve_provenance(tmp_path, monkeypatch, design, expected_calls):
    calls = []

    def fake_run(config):
        calls.append(config)
        path = Path(config.vcf_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((FIXTURES / "basic.vcf").read_text())
        if "history_path" in config.extra_defines:
            Path(config.extra_defines["history_path"]).write_text(
                "vcf_column,population_index,selfing_generations\n"
                "0,7,0\n1,2,1\n2,4,3\n3,8,-1\n"
            )
        return SlimRunResult(tuple(build_slim_command(config)), 0, "", "", str(path), config.metadata())

    monkeypatch.setattr("selfdgs.validation.experiments.run_slim_simulation", fake_run)
    outdir = tmp_path / design
    results = run_validation_experiment(ValidationExperimentConfig(
        slim_script=None, outdir=outdir, sampling_design=design,
        n_independent_loci=3, n_reps=2, s_grid=[0, 0.5, 0.95],
    ))
    assert len(calls) == expected_calls
    assert len({call.seed for call in calls}) == expected_calls
    for result in results:
        assert result.fit.metadata["sampling_design"] == design
        assert result.fit.metadata["population_replicate"] == result.rep
        assert result.fit.metadata["uncertainty_calibration"] == "uncalibrated_composite_likelihood"
        assert sum(locus.n_loci for locus in result.loci) == 3
        if design == "fixed_individuals":
            history_path = outdir / f"rep{result.rep:03d}" / "sampled_individuals.csv"
            manifest = pd.read_csv(history_path)
            from selfdgs.io.vcf import read_vcf_samples
            assert manifest.vcf_sample.tolist() == list(read_vcf_samples(history_path.with_name("sample.vcf")))
            assert manifest.selfing_generations.tolist() == [0, 1, 3, -1]
            assert manifest.rep.tolist() == [result.rep] * 4
            metadata = json.loads(result.loci[0].slim_metadata)
            assert metadata["extra_defines"]["history_path"] == str(history_path)
    from selfdgs.validation import discover_validation_runs, load_validation_collection
    collection = load_validation_collection(discover_validation_runs(outdir))
    assert set(collection.sampling_design) == {design}
    assert set(collection.n_loci) == {3}


@pytest.mark.parametrize("kwargs,message", [
    ({"sampling_design": "invalid"}, "sampling_design"),
    ({"n_sample": 501}, "census size"),
    ({"burn_mult": 0}, "burn_mult >= 1"),
    ({"burn_mult": None}, "non-negative integer"),
    ({"extra_defines": {"n_loci": 3}}, "managed"),
    ({"extra_defines": {"history_path": "wrong.csv"}}, "managed"),
    ({"slim_script": default_slim_script_path("independent_populations")}, "slim_script=None"),
])
def test_fixed_validation_rejects_inconsistent_config(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ValidationExperimentConfig(**{"slim_script": None, "outdir": tmp_path, **kwargs})


def test_pooling_fixed_vcfs_checks_identity_not_column_order(tmp_path):
    text = (FIXTURES / "basic.vcf").read_text()
    renamed = tmp_path / "renamed.vcf"
    lines = text.splitlines()
    header = next(line for line in lines if line.startswith("#CHROM"))
    samples = header.split("\t")[9:]
    renamed.write_text(text.replace(header, header.replace(samples[0], "different_individual")))
    with pytest.raises(ValueError, match="same focal sample IDs"):
        summarize_existing_vcfs([FIXTURES / "basic.vcf", renamed], s_grid=[0, 0.5])
    control = summarize_existing_vcfs(
        [FIXTURES / "basic.vcf", renamed], s_grid=[0, 0.5],
        sampling_design="independent_populations",
    )
    assert control.fit.n_sites > 0
    reordered = tmp_path / "reordered.vcf"
    reordered.write_text("\n".join(
        "\t".join(line.split("\t")[:9] + line.split("\t")[9:][::-1])
        if not line.startswith("##") else line for line in lines
    ) + "\n")
    result = summarize_existing_vcfs([FIXTURES / "basic.vcf", reordered], s_grid=[0, 0.5])
    assert result.observed_counts == control.observed_counts


@pytest.mark.slim
@pytest.mark.skipif(shutil.which("slim") is None, reason="SLiM is not installed")
@pytest.mark.parametrize("selfing_rate", [0.0, 0.5, 1.0])
def test_fixed_individual_slim_integration(tmp_path, selfing_rate):
    results = run_validation_experiment(ValidationExperimentConfig(
        slim_script=None, outdir=tmp_path, ne=20, true_s=selfing_rate,
        n_sample=4, n_independent_loci=3, chrom_length_each=1_000,
        mu=1e-4, burn_mult=2, n_reps=2, s_grid=[0, 0.5, 0.95, 1],
    ))
    assert len(results) == 2
    for result in results:
        assert result.fit.n_sites > 0
        manifest = pd.read_csv(tmp_path / f"rep{result.rep:03d}" / "sampled_individuals.csv")
        assert manifest.population_index.nunique() == 4
        assert manifest.vcf_column.tolist() == [0, 1, 2, 3]
        if selfing_rate == 0:
            assert manifest.selfing_generations.eq(0).all()
        elif selfing_rate == 1:
            assert manifest.selfing_generations.eq(-1).all()


@pytest.mark.parametrize("manifest", [
    "vcf_column,population_index\n0,7\n",
    "vcf_column,population_index,selfing_generations\n0,7,0\n1,7,1\n2,4,2\n3,8,0\n",
    "vcf_column,population_index,selfing_generations\n1,7,0\n0,2,1\n2,4,2\n3,8,0\n",
    "vcf_column,population_index,selfing_generations\n0,7,0\n1,2,1.5\n2,4,2\n3,8,0\n",
])
def test_invalid_sample_manifest_preserves_previous_outputs(tmp_path, monkeypatch, manifest):
    previous = tmp_path / "validation_summary.csv"
    previous.write_text("previous\n")

    def fake_run(config):
        path = Path(config.vcf_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((FIXTURES / "basic.vcf").read_text())
        Path(config.extra_defines["history_path"]).write_text(manifest)
        return SlimRunResult(tuple(build_slim_command(config)), 0, "", "", str(path), config.metadata())

    monkeypatch.setattr("selfdgs.validation.experiments.run_slim_simulation", fake_run)
    with pytest.raises(ValueError, match="Invalid fixed-individual sample manifest"):
        run_validation_experiment(ValidationExperimentConfig(slim_script=None, outdir=tmp_path))
    assert previous.read_text() == "previous\n"


def test_default_script_and_validation_config_agree(tmp_path):
    config = ValidationExperimentConfig(outdir=tmp_path)
    assert config.script_path() == default_slim_script_path()
    explicit = ValidationExperimentConfig(default_slim_script_path(), tmp_path)
    assert explicit.script_path() == config.script_path()
    assert config.census_size == 500
    assert config.n_loci == 1
    assert config.vcf_polarization == "folded"
    assert config.likelihood_mode() == "folded"


def test_simulation_parameter_aliases_and_canonical_metadata(tmp_path):
    canonical = ValidationExperimentConfig(outdir=tmp_path, census_size=100, n_loci=3)
    aliases = ValidationExperimentConfig(outdir=tmp_path, ne=100, n_independent_loci=3)
    assert canonical == aliases
    config = SlimSimulationConfig(default_slim_script_path(), tmp_path / "sample.vcf", census_size=100)
    assert config.ne == config.census_size == 100
    assert config.metadata()["census_size"] == 100
    for kwargs in ({"census_size": 100, "ne": 200}, {"n_loci": 2, "n_independent_loci": 3}):
        with pytest.raises(ValueError, match="must agree"):
            ValidationExperimentConfig(outdir=tmp_path, **kwargs)
    with pytest.raises(ValueError, match="must agree"):
        SlimSimulationConfig(default_slim_script_path(), tmp_path / "sample.vcf", census_size=100, ne=200)


def test_python_and_cli_workflow_defaults_agree(tmp_path):
    from selfdgs.cli import build_parser
    from selfdgs.empirical import EmpiricalAnalysisConfig
    parser = build_parser()
    cli = parser.parse_args(["simulate", "--out", str(tmp_path)])
    config = ValidationExperimentConfig(outdir=tmp_path)
    assert cli.census_size == config.census_size
    assert cli.n_loci == config.n_loci
    assert cli.sampling_design == config.sampling_design
    assert cli.polarization == config.vcf_polarization
    empirical = parser.parse_args(["empirical", "samples.vcf", "--n-diploids", "4", "--out", str(tmp_path)])
    assert empirical.n_draws == EmpiricalAnalysisConfig("samples.vcf", 4).n_draws == 1
    assert parser.parse_args(["simulate", "--out", str(tmp_path), "--ne", "123"]).census_size == 123
