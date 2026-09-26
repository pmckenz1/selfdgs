from pathlib import Path
import json

import pandas as pd
import pytest

from selfdgs.validation import (
    discover_validation_runs,
    load_likelihood_collection,
    load_validation_collection,
    summarize_estimator_recovery,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_discover_validation_runs_returns_typed_empty_table(tmp_path):
    runs = discover_validation_runs(tmp_path)

    assert runs.empty
    assert {"run_id", "path", "base_dir"}.issubset(runs.columns)


def test_load_validation_collection_returns_typed_empty_table():
    validation = load_validation_collection([])

    assert validation.empty
    assert {
        "run_id",
        "run_path",
        "rep",
        "best_s",
        "observed_dgs_path",
    }.issubset(validation.columns)


def _write_run(
    root: Path,
    name: str,
    *,
    ne: int,
    true_s: float,
    best_s: float,
    likelihood: bool = True,
) -> Path:
    run = root / name
    run.mkdir()
    pd.DataFrame(
        [
            {
                "ne": ne,
                "true_s": true_s,
                "best_s": best_s,
                "best_loglik": -1.0,
                "n_sites": 10,
                "n_loci": 2,
                "boundary_optimum": False,
            }
        ]
    ).to_csv(run / "validation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "best_s": best_s,
                "best_loglik": -1.0,
                "n_sites": 10,
                "n_diploids": 4,
                "method": "grid",
                "support_interval": "(0.0, 1.0)",
                "boundary_optimum": False,
                "warnings": "[]",
                "metadata": "{}",
            }
        ]
    ).to_csv(run / "fit_summary.csv", index=False)
    pd.DataFrame(
        [
            {"n0": 1, "n1": 2, "n2": 1, "count": 2},
            {"n0": 4, "n1": 0, "n2": 0, "count": 1},
        ]
    ).to_csv(run / "observed_polymorphic_dgs.csv", index=False)
    if likelihood:
        pd.DataFrame(
            [
                {"s": 0.0, "loglik": -2.0, "delta_loglik": -1.0},
                {"s": best_s, "loglik": -1.0, "delta_loglik": 0.0},
            ]
        ).to_csv(run / "likelihood.csv", index=False)
    return run


def test_discovery_parses_validation_run_directory_names(tmp_path):
    _write_run(tmp_path, "ne100_self0.500", ne=100, true_s=0.5, best_s=0.45)
    _write_run(tmp_path, "ne200_self0.250", ne=200, true_s=0.25, best_s=0.3)

    runs = discover_validation_runs(tmp_path)

    assert runs["run_id"].tolist() == ["ne100_self0.500", "ne200_self0.250"]
    assert runs["ne"].tolist() == [100, 200]
    assert runs["true_s"].tolist() == [0.5, 0.25]
    assert runs["has_likelihood"].tolist() == [True, True]


def test_loaders_handle_missing_optional_likelihood_files(tmp_path):
    _write_run(tmp_path, "ne100_self0.500", ne=100, true_s=0.5, best_s=0.45)
    _write_run(
        tmp_path,
        "ne100_self0.750",
        ne=100,
        true_s=0.75,
        best_s=0.7,
        likelihood=False,
    )
    runs = discover_validation_runs(tmp_path)

    validation = load_validation_collection(runs)
    likelihoods = load_likelihood_collection(runs)

    assert set(validation["true_s"]) == {0.5, 0.75}
    assert set(likelihoods["run_id"]) == {"ne100_self0.500"}

    with pytest.raises(FileNotFoundError):
        load_likelihood_collection(runs, require=True)


def test_likelihood_loader_handles_package_replicate_layout(tmp_path):
    root = tmp_path / "validation"
    rep_dir = root / "rep000"
    rep_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"rep": 0, "true_s": 0.5, "best_s": 0.5, "best_loglik": -1.0, "n_sites": 2}]
    ).to_csv(root / "validation_summary.csv", index=False)
    pd.DataFrame(
        [
            {"s": 0.0, "loglik": -2.0, "delta_loglik": -1.0},
            {"s": 0.5, "loglik": -1.0, "delta_loglik": 0.0},
        ]
    ).to_csv(rep_dir / "likelihood.csv", index=False)
    pd.DataFrame(
        [{"n_diploids": 4, "best_s": 0.5, "best_loglik": -1.0, "n_sites": 2}]
    ).to_csv(rep_dir / "fit_summary.csv", index=False)
    pd.DataFrame([{"n0": 1, "n1": 2, "n2": 1, "count": 2}]).to_csv(
        rep_dir / "observed_dgs.csv",
        index=False,
    )
    pd.DataFrame([{"n0": 1, "n1": 2, "n2": 1, "count": 2}]).to_csv(
        rep_dir / "observed_polymorphic_dgs.csv",
        index=False,
    )
    (root / "validation_config.json").write_text(
        json.dumps({"ne": 500, "true_s": 0.5, "n_sample": 4})
    )

    runs = discover_validation_runs(root)
    validation = load_validation_collection(runs)
    likelihoods = load_likelihood_collection(runs)

    assert runs.loc[0, "run_id"] == "validation"
    assert runs.loc[0, "has_likelihood"] == True
    assert runs.loc[0, "has_fit_summary"] == True
    assert runs.loc[0, "has_observed_dgs"] == True
    assert runs.loc[0, "has_observed_polymorphic_dgs"] == True
    assert runs.loc[0, "ne"] == 500
    assert runs.loc[0, "n_diploids"] == 4
    assert validation.loc[0, "observed_dgs_path"].endswith(
        "rep000/observed_dgs.csv"
    )
    assert validation.loc[0, "observed_polymorphic_dgs_path"].endswith(
        "rep000/observed_polymorphic_dgs.csv"
    )
    assert likelihoods["rep"].tolist() == [0, 0]
    assert likelihoods["likelihood_path"].str.endswith("rep000/likelihood.csv").all()


def test_validation_loader_resolves_legacy_replicate_directory_paths(tmp_path):
    root = tmp_path / "validation"
    rep_dir = root / "rep000_seed123"
    rep_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"rep": 0, "true_s": 0.5, "best_s": 0.4, "best_loglik": -1.0}]
    ).to_csv(root / "validation_summary.csv", index=False)
    pd.DataFrame([{"n0": 1, "n1": 2, "n2": 1, "count": 2}]).to_csv(
        rep_dir / "observed_dgs.csv",
        index=False,
    )
    pd.DataFrame([{"n0": 1, "n1": 2, "n2": 1, "count": 2}]).to_csv(
        rep_dir / "observed_polymorphic_dgs.csv",
        index=False,
    )

    validation = load_validation_collection([root])

    assert validation.loc[0, "observed_dgs_path"] == str(
        rep_dir / "observed_dgs.csv"
    )
    assert validation.loc[0, "observed_polymorphic_dgs_path"] == str(
        rep_dir / "observed_polymorphic_dgs.csv"
    )


def test_recovery_summary_metrics_are_correct_for_tiny_table():
    frame = pd.DataFrame(
        {
            "ne": [100, 100, 100],
            "true_s": [0.5, 0.5, 0.5],
            "best_s": [0.4, 0.5, 0.7],
        }
    )

    summary = summarize_estimator_recovery(frame)

    assert summary.loc[0, "n_runs"] == 3
    assert summary.loc[0, "mean_estimate"] == pytest.approx(0.5333333333)
    assert summary.loc[0, "bias"] == pytest.approx(0.0333333333)
    assert summary.loc[0, "mae"] == pytest.approx(0.1)


def test_recovery_keeps_sampling_design_size_and_locus_count_separate():
    rows = [
        {"ne": 100, "true_s": 0.5, "n_diploids": n, "n_loci": loci,
         "sampling_design": design, "best_s": 0.4}
        for n in (3, 6) for loci in (10, 30)
        for design in ("fixed_individuals", "independent_populations")
    ]
    summary = summarize_estimator_recovery(pd.DataFrame(rows))
    assert len(summary) == 8
    assert summary.n_runs.eq(1).all()


def test_discovery_reads_census_size_without_directory_naming_convention(tmp_path):
    run = tmp_path / "fixed_sample_experiment"
    run.mkdir()
    (tmp_path / "unrelated").mkdir()
    pd.DataFrame([{"census_size": 250, "true_s": 0.5, "best_s": 0.4}]).to_csv(
        run / "validation_summary.csv", index=False,
    )
    runs = discover_validation_runs(tmp_path)
    assert runs.run_id.tolist() == [run.name]
    collection = load_validation_collection(runs)
    summary = summarize_estimator_recovery(collection)
    assert summary.census_size.tolist() == [250]


def test_recovery_normalizes_mixed_population_size_columns():
    frame = pd.DataFrame([
        {"ne": 100, "true_s": 0.5, "best_s": 0.4},
        {"census_size": 100, "true_s": 0.5, "best_s": 0.6},
        {"census_size": 200, "true_s": 0.5, "best_s": 0.7},
    ])
    summary = summarize_estimator_recovery(frame).set_index("census_size")
    assert summary.loc[100, "n_runs"] == 2
    assert summary.loc[200, "n_runs"] == 1
