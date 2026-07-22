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
        slim_script=default_slim_script_path(),
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
        slim_script=default_slim_script_path(),
        vcf_path=tmp_path / "out.vcf",
        extra_defines={"label": 'a\\b"c'},
    )

    assert 'label="a\\\\b\\"c"' in build_slim_command(config)


def test_build_slim_command_formats_boolean_defines_for_eidos(tmp_path):
    config = SlimSimulationConfig(
        slim_script=default_slim_script_path(),
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
                slim_script=default_slim_script_path(),
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
            slim_script=default_slim_script_path(),
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
        slim_script=default_slim_script_path(),
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

    script = default_slim_script_path().read_text()
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
        slim_script=default_slim_script_path(),
        vcf_path=tmp_path / "out.vcf",
        slim_executable="definitely-not-slim",
    )

    with pytest.raises(SLiMUnavailableError, match="not found on PATH"):
        run_slim_simulation(config)


@pytest.mark.slim
@pytest.mark.skipif(shutil.which("slim") is None, reason="SLiM is not installed")
def test_optional_slim_integration_smoke(tmp_path):
    config = SlimSimulationConfig(
        slim_script=default_slim_script_path(),
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
                slim_script=default_slim_script_path(),
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
            slim_script=default_slim_script_path(),
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
                slim_script=default_slim_script_path(),
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
        slim_script=default_slim_script_path(),
        outdir=tmp_path / "validation",
        ne=500,
        true_s=0.5,
        n_independent_loci=2,
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
    assert config_payload["ne"] == 500
    assert config_payload["n_sample"] == 4
    assert config_payload["resolved_s_grid"] == [0.0, 0.5, 1.0]
    assert config_payload["resolved_fit_mode"] == "unfolded"
    rep_dirs = sorted((tmp_path / "validation").glob("rep*"))
    assert len(rep_dirs) == 1
    assert rep_dirs[0].name == "rep000"
    assert (rep_dirs[0] / "observed_dgs.csv").exists()
    assert (rep_dirs[0] / "observed_polymorphic_dgs.csv").exists()
    assert (rep_dirs[0] / "fit_result.json").exists()
    assert (rep_dirs[0] / "likelihood.csv").exists()
    assert (rep_dirs[0] / "loci.csv").exists()
    loci = pd.read_csv(rep_dirs[0] / "loci.csv")
    assert loci["slim_command"].notna().all()
    assert loci["slim_metadata"].notna().all()
    for row in loci.itertuples(index=False):
        assert json.loads(row.slim_metadata)["vcf_path"] == row.vcf_path
    assert all(Path(path).exists() for path in loci["slim_stdout_path"])
    assert all(Path(path).exists() for path in loci["slim_stderr_path"])

    summary = pd.read_csv(tmp_path / "validation" / "validation_summary.csv")
    assert summary.loc[0, "ne"] == 500
    assert summary.loc[0, "n_diploids"] == 4
    assert summary.loc[0, "seed"] == 1
    assert summary.loc[0, "polarization"] == "ref"
    assert summary.loc[0, "fit_mode"] == "unfolded"


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
        slim_script=default_slim_script_path(),
        outdir=outdir,
        n_reps=1,
        n_independent_loci=1,
    )

    with pytest.raises(RuntimeError, match="simulated SLiM failure"):
        run_validation_experiment(config)

    assert previous.read_text() == "previous\n"
